"""Stage 3 of the ClipForge pipeline: score the transcript for viral clips.

Given ``data/sources/{job_id}/transcript.json`` (produced by the transcribe
stage), render each segment as a timestamped line, ask the LLM to pick the most
viral-worthy 25-75s segments, then validate/snap/dedupe the result before
writing ``data/sources/{job_id}/candidates.json``.

The LLM (Gemini or Claude, per ``LLM_PROVIDER`` — see :mod:`clipforge.llm`) is
only ever reached through the injectable ``llm`` callable, so the
validation/snap/dedupe logic can be tested with a fake transcript and a fake
response — no network and no API key.

Run standalone against an already-transcribed job:

    python -m clipforge.analyze <job_id>
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# A callable that takes the fully-rendered prompt and returns the raw model text.
# Injecting this lets tests feed a canned response instead of calling the LLM.
LLMCall = Callable[[str], str]

# Videos longer than this are analyzed in chunks to keep the prompt (and the
# model's attention) manageable.
_CHUNK_S = 30 * 60.0
_CHUNK_OVERLAP_S = 2 * 60.0

# Tolerance (seconds) when checking that a candidate lies within transcript range.
_RANGE_TOL_S = 0.5

# Two candidates overlapping more than this fraction of the shorter clip are
# treated as duplicates; the higher-scoring one wins.
_DEDUPE_OVERLAP_FRAC = 0.5

SCORING_PROMPT = (
    "You are a short-form video strategist. Given a timestamped transcript of a\n"
    "long video, identify the 10-15 most viral-worthy segments of 25-75 seconds.\n"
    "Score each 0-100 using this weighted rubric: hook strength 30% (bold claims,\n"
    "curiosity gaps, pattern interrupts), emotional peak 20%, value density 20%\n"
    "(practical insight, concrete numbers), coherence 15% (self-contained, no\n"
    "missing context), payoff/quotability 15%. Segments MUST start and end at\n"
    "sentence boundaries taken from the transcript. hook_caption is a scroll-\n"
    "stopping headline (<= 60 chars, ALL CAPS) that creates a curiosity gap\n"
    "WITHOUT clickbait lies. Respond ONLY with valid JSON:\n"
    '{"candidates": [ ... ]} using the exact schema provided.'
)

_RETRY_SUFFIX = "\n\nYour previous output was invalid JSON. Output ONLY valid JSON."


class AnalyzeError(Exception):
    """Raised for any analysis failure, with a human-readable message."""


@dataclass(frozen=True)
class SubScores:
    """Per-rubric sub-scores (each 0-100)."""

    hook: int
    emotion: int
    value: int
    coherence: int
    payoff: int


@dataclass(frozen=True)
class Candidate:
    """A validated, snapped clip candidate ready for the cutting stage."""

    start: float
    end: float
    score: int
    title: str
    hook: str
    hook_caption: str
    reason: str
    sub_scores: SubScores


# --------------------------------------------------------------------------- #
# Transcript loading / rendering
# --------------------------------------------------------------------------- #


def _load_segments(job_id: str, cfg: Config) -> list[dict[str, Any]]:
    """Load transcript segments (sorted by start) for ``job_id``.

    Segments are returned as plain dicts with ``start``/``end``/``text`` so this
    module never has to import the heavy transcribe/whisper dependencies.
    """
    transcript_path = cfg.data_path / "sources" / job_id / "transcript.json"
    if not transcript_path.is_file():
        raise AnalyzeError(f"Transcript not found: {transcript_path}")

    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalyzeError(f"Could not read transcript {transcript_path}: {exc}") from exc

    raw = payload.get("segments") if isinstance(payload, dict) else None
    if not raw:
        raise AnalyzeError(f"Transcript has no segments: {transcript_path}")

    segments: list[dict[str, Any]] = []
    for seg in raw:
        try:
            segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": str(seg["text"]).strip(),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalyzeError(f"Malformed transcript segment {seg!r}: {exc}") from exc

    segments.sort(key=lambda s: s["start"])
    return segments


def _render_segments(segments: list[dict[str, Any]]) -> str:
    """Render segments as ``[123.4 - 130.2] text`` lines, one per line."""
    return "\n".join(f"[{s['start']:.1f} - {s['end']:.1f}] {s['text']}" for s in segments)


def _chunk_segments(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split segments into <=30-minute chunks with a 2-minute overlap.

    Chunking is by wall-clock time relative to the transcript start; each chunk
    keeps whole segments so sentence boundaries are preserved.
    """
    if not segments:
        return []
    total = segments[-1]["end"] - segments[0]["start"]
    if total <= _CHUNK_S:
        return [segments]

    origin = segments[0]["start"]
    chunks: list[list[dict[str, Any]]] = []
    window_start = origin
    end_time = segments[-1]["end"]
    step = _CHUNK_S - _CHUNK_OVERLAP_S
    while window_start < end_time:
        window_end = window_start + _CHUNK_S
        chunk = [s for s in segments if s["start"] < window_end and s["end"] > window_start]
        if chunk:
            chunks.append(chunk)
        window_start += step
    return chunks


# --------------------------------------------------------------------------- #
# LLM call + parsing
# --------------------------------------------------------------------------- #


def _default_call(cfg: Config) -> LLMCall:
    """Build the scoring LLM callable for the configured provider (Gemini/Claude)."""
    from . import llm

    try:
        return llm.build_llm_call(cfg, system_prompt=SCORING_PROMPT, temperature=0.2)
    except llm.LLMError as exc:
        raise AnalyzeError(str(exc)) from exc


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a ```json ... ``` fence if the model added one despite instructions.
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned.strip("`")
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON is not an object")
    return parsed


def _request_candidates(prompt: str, llm: LLMCall) -> list[dict[str, Any]]:
    """Call the LLM and return the raw candidate dicts, retrying once on bad JSON.

    On the first malformed response the prompt is re-sent with an explicit
    "output ONLY valid JSON" instruction. After two failures raises AnalyzeError.
    """
    attempts = [prompt, prompt + _RETRY_SUFFIX]
    last_error = ""
    for attempt, full_prompt in enumerate(attempts, start=1):
        raw = llm(full_prompt)
        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(
                "LLM returned invalid JSON (attempt %d/%d): %s", attempt, len(attempts), exc
            )
            continue
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list):
            last_error = "response JSON has no 'candidates' list"
            logger.warning(
                "LLM JSON missing 'candidates' list (attempt %d/%d)", attempt, len(attempts)
            )
            continue
        return candidates
    raise AnalyzeError(f"LLM did not return valid candidate JSON after 2 attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Validation / snapping / dedupe
# --------------------------------------------------------------------------- #


def _coerce_sub_scores(raw: Any) -> SubScores:
    """Build SubScores from a raw dict, defaulting missing/invalid values to 0."""

    def _score(key: str) -> int:
        if not isinstance(raw, dict):
            return 0
        try:
            return _clamp_score(int(raw.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    return SubScores(
        hook=_score("hook"),
        emotion=_score("emotion"),
        value=_score("value"),
        coherence=_score("coherence"),
        payoff=_score("payoff"),
    )


def _clamp_score(value: int) -> int:
    return max(0, min(100, value))


def _snap_candidate(
    raw: dict[str, Any], segments: list[dict[str, Any]], cfg: Config
) -> Candidate | None:
    """Validate and snap one raw candidate; return None if it must be dropped.

    Snapping pins the start to the nearest segment start and the end to the
    nearest segment end (segments == sentence boundaries). A candidate is dropped
    when it falls outside the transcript range, when its snapped span covers no
    text, or when its snapped duration is outside ``[clip_min_s, clip_max_s]``.
    """
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (KeyError, TypeError, ValueError):
        return None

    t_start = segments[0]["start"]
    t_end = segments[-1]["end"]

    # start/end must lie within the transcript range (small tolerance for rounding).
    if start < t_start - _RANGE_TOL_S or end > t_end + _RANGE_TOL_S or end <= start:
        return None

    # Snap to the nearest sentence boundaries.
    start_idx = min(range(len(segments)), key=lambda i: abs(segments[i]["start"] - start))
    end_idx = min(range(len(segments)), key=lambda i: abs(segments[i]["end"] - end))
    if end_idx < start_idx:
        return None

    snapped_start = segments[start_idx]["start"]
    snapped_end = segments[end_idx]["end"]

    # The snapped span must cover real transcript text.
    covered_text = " ".join(segments[i]["text"] for i in range(start_idx, end_idx + 1)).strip()
    if not covered_text:
        return None

    duration = snapped_end - snapped_start
    if duration < cfg.clip_min_s or duration > cfg.clip_max_s:
        return None

    try:
        score = _clamp_score(int(raw.get("score", 0)))
    except (TypeError, ValueError):
        score = 0

    title = str(raw.get("title", "")).strip()[:70]
    hook = str(raw.get("hook", "")).strip()
    hook_caption = str(raw.get("hook_caption", "")).strip().upper()[:60]
    reason = str(raw.get("reason", "")).strip()

    return Candidate(
        start=snapped_start,
        end=snapped_end,
        score=score,
        title=title,
        hook=hook,
        hook_caption=hook_caption,
        reason=reason,
        sub_scores=_coerce_sub_scores(raw.get("sub_scores")),
    )


def _validate(
    raw_candidates: list[dict[str, Any]], segments: list[dict[str, Any]], cfg: Config
) -> list[Candidate]:
    """Snap + validate every raw candidate, dropping the invalid ones."""
    validated: list[Candidate] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        candidate = _snap_candidate(raw, segments, cfg)
        if candidate is not None:
            validated.append(candidate)
    return validated


def _overlap_fraction(a: Candidate, b: Candidate) -> float:
    """Fraction of the shorter clip that overlaps the other in time (0..1)."""
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shortest = min(a.end - a.start, b.end - b.start)
    return overlap / shortest if shortest > 0 else 0.0


def _dedupe(candidates: list[Candidate], num_clips: int) -> list[Candidate]:
    """Drop >50%-overlapping duplicates (keeping higher score), sort, take top N."""
    # Process best-first so an already-kept clip always outranks the one we drop.
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for cand in ordered:
        if any(_overlap_fraction(cand, k) > _DEDUPE_OVERLAP_FRAC for k in kept):
            continue
        kept.append(cand)
    return kept[:num_clips]


def _merge_chunk_results(results: list[list[Candidate]], cfg: Config) -> list[Candidate]:
    """Flatten per-chunk candidates then dedupe across chunk overlaps."""
    flat = [c for chunk in results for c in chunk]
    return _dedupe(flat, cfg.num_clips)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def analyze(job_id: str, cfg: Config, *, llm: LLMCall | None = None) -> list[Candidate]:
    """Score ``job_id``'s transcript and write ``candidates.json``.

    Loads the transcript, renders it for the LLM (chunking videos longer than 30
    minutes with a 2-minute overlap), scores each chunk, then validates, snaps,
    and dedupes the candidates. Pass ``llm`` to inject a fake response for tests;
    otherwise the configured provider's callable is used (see
    :mod:`clipforge.llm`). Raises :class:`AnalyzeError` on any failure.
    """
    segments = _load_segments(job_id, cfg)
    call = llm if llm is not None else _default_call(cfg)

    chunks = _chunk_segments(segments)
    logger.info(
        "Analyzing %d segment(s) in %d chunk(s) for job %s", len(segments), len(chunks), job_id
    )

    chunk_results: list[list[Candidate]] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = f"Transcript:\n{_render_segments(chunk)}"
        raw_candidates = _request_candidates(prompt, call)
        # Validate against the FULL transcript so snapping can reach overlap regions.
        validated = _validate(raw_candidates, segments, cfg)
        logger.info(
            "Chunk %d/%d: %d raw -> %d valid candidate(s)",
            i,
            len(chunks),
            len(raw_candidates),
            len(validated),
        )
        chunk_results.append(validated)

    candidates = _merge_chunk_results(chunk_results, cfg)

    out_dir = cfg.data_path / "sources" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.json"
    out_path.write_text(
        json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %d candidate(s) to %s", len(candidates), out_path)
    return candidates


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m clipforge.analyze <job_id>", file=sys.stderr)
        return 2

    job_id = argv[0]
    cfg = Config.load()
    try:
        candidates = analyze(job_id, cfg)
    except AnalyzeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
