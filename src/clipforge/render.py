"""Stage 7 (final) of the ClipForge pipeline: posting-ready vertical clips.

For each candidate of a job this stage turns the raw 16:9 cut into a finished,
1080x1920 clip ready to upload:

1. **Reframe** the raw cut (``clip_NN_raw.mp4``) to vertical 1080x1920 via
   :func:`clipforge.reframe.reframe` (face-track or blur-background) -> a temp.
2. **Caption + hook overlay** are built as one ASS document
   (:func:`clipforge.captions.build_ass` + :func:`~clipforge.captions.add_hook_overlay`)
   and burned in the SAME ffmpeg pass as the final NVENC encode
   (``h264_nvenc -preset p5 -rc vbr -cq 23``, ``yuv420p``, AAC 128k) ->
   ``data/outputs/{job_id}/clip_NN.mp4`` (exactly 1080x1920).
3. **Thumbnail** at the hook moment (1s into the clip) -> ``clip_NN.jpg``.
4. **Metadata** ``clip_NN.meta.json``: ``title``, ``hook_caption``,
   ``suggested_description`` (2 sentences + a call-to-action) and ``hashtags``
   (5-8, niche+broad, lowercase). Description/hashtags for EVERY clip in the job
   come from a single cheap LLM call (batched), never one call per clip.

After a clip renders successfully its ``clip_NN_raw.mp4`` intermediate is
deleted. Rendering is idempotent: an existing ``clip_NN.mp4`` is left untouched
so an interrupted run can be resumed.

Run standalone against an already-cut job:

    python -m clipforge.render <job_id> [--no-llm]
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .captions import Word, add_hook_overlay, build_ass, burn
from .config import Config
from .reframe import reframe

logger = logging.getLogger(__name__)

# Caption / hook presentation (settings-configurable in a later phase; fixed here
# to the short-form defaults from docs/02-PRD-ADDENDUM.md).
CAPTION_STYLE = "bold"
HOOK_MODE = "3s"

# Thumbnail is grabbed at the "hook moment": 1s into the clip (clamped to length).
THUMB_OFFSET_S = 1.0

# A callable that takes the fully-rendered prompt and returns the raw model text
# (injected in tests so the metadata step needs no network / API key).
LLMCall = Callable[[str], str]

META_PROMPT = (
    "You write social-media posting metadata for short vertical video clips.\n"
    "For EACH clip you are given, produce:\n"
    "- suggested_description: exactly 2 sentences in English that tease the clip,\n"
    "  followed by a short call-to-action (e.g. 'Follow for more.').\n"
    "- hashtags: 5 to 8 hashtags mixing niche and broad reach, all lowercase, no\n"
    "  spaces, each starting with '#'.\n"
    "Respond ONLY with valid JSON of the form:\n"
    '{"clips": [{"index": 0, "suggested_description": "...", '
    '"hashtags": ["#a", "#b"]}, ...]} with one entry per input clip, the "index"\n'
    "matching the input clip number."
)

_RETRY_SUFFIX = "\n\nYour previous output was invalid JSON. Output ONLY valid JSON."

# Generic broad tags used when no LLM is available (offline / --no-llm).
_FALLBACK_HASHTAGS = ["#shorts", "#viral", "#fyp", "#clips", "#podcast"]


class RenderError(Exception):
    """Raised for any rendering failure, with a human-readable message.

    Reframe / caption / ffmpeg failures are re-raised as this exception so the
    pipeline and CLI never leak a raw traceback.
    """


@dataclass(frozen=True)
class ClipMeta:
    """Per-clip posting metadata written to ``clip_NN.meta.json``."""

    title: str
    hook_caption: str
    suggested_description: str
    hashtags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "hook_caption": self.hook_caption,
            "suggested_description": self.suggested_description,
            "hashtags": self.hashtags,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_candidates(job_id: str, cfg: Config) -> list[dict[str, Any]]:
    """Load ``candidates.json`` for ``job_id`` as a list of raw dicts."""
    path = cfg.data_path / "sources" / job_id / "candidates.json"
    if not path.is_file():
        raise RenderError(f"Candidates not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"Could not read candidates {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise RenderError(f"Candidates file is not a JSON list: {path}")
    return payload


def _load_words(job_id: str, cfg: Config) -> list[Word]:
    """Load all transcript words for ``job_id`` (tolerant; ``[]`` if absent)."""
    path = cfg.data_path / "sources" / job_id / "transcript.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        return []
    words: list[Word] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for w in seg.get("words", []):
            try:
                words.append(
                    Word(word=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
                )
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed words rather than fail the clip
    return words


# Public alias: the pipeline reuses this loader for single-clip re-renders.
load_words = _load_words


# --------------------------------------------------------------------------- #
# Metadata (one batched LLM call per job)
# --------------------------------------------------------------------------- #


def _meta_llm_call(cfg: Config) -> LLMCall:
    """Build the metadata LLM callable for the configured provider (Gemini/Claude)."""
    from . import llm

    try:
        return llm.build_llm_call(cfg, system_prompt=META_PROMPT, temperature=0.4)
    except llm.LLMError as exc:
        raise RenderError(str(exc)) from exc


def _meta_prompt(candidates: list[dict[str, Any]]) -> str:
    """Render one prompt describing every clip for the batched metadata call."""
    lines = ["Clips:"]
    for i, cand in enumerate(candidates):
        title = str(cand.get("title", "")).strip()
        hook_caption = str(cand.get("hook_caption", "")).strip()
        hook = str(cand.get("hook", "")).strip()
        lines.append(
            f"[{i}] title: {title!r} | hook_caption: {hook_caption!r} | opening_line: {hook!r}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text, tolerating a ```json fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned.strip("`")
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON is not an object")
    return parsed


def _clean_hashtags(raw: Any) -> list[str]:
    """Normalise hashtags to lowercase, space-free, ``#``-prefixed tokens (5-8)."""
    if not isinstance(raw, list):
        return []
    tags: list[str] = []
    for item in raw:
        tag = "".join(str(item).split()).lower().lstrip("#")
        if not tag:
            continue
        tag = "#" + tag
        if tag not in tags:
            tags.append(tag)
    return tags[:8]


def _fallback_meta(cand: dict[str, Any]) -> ClipMeta:
    """Deterministic metadata used offline (``--no-llm`` / missing API key)."""
    title = str(cand.get("title", "")).strip() or "Untitled clip"
    hook_caption = str(cand.get("hook_caption", "")).strip()
    hook = str(cand.get("hook", "")).strip()
    first = hook or title
    description = f"{first.rstrip('.!?')}. Watch the full moment in this clip. Follow for more."
    return ClipMeta(
        title=title[:70],
        hook_caption=hook_caption,
        suggested_description=description,
        hashtags=list(_FALLBACK_HASHTAGS),
    )


def _request_meta(prompt: str, llm: LLMCall, count: int) -> dict[int, dict[str, Any]]:
    """Call the LLM and return ``{clip_index: {description, hashtags}}``.

    Retries once on malformed JSON. Returns whatever indexes the model provided;
    missing clips are filled with fallbacks by the caller.
    """
    attempts = [prompt, prompt + _RETRY_SUFFIX]
    for attempt, full_prompt in enumerate(attempts, start=1):
        raw = llm(full_prompt)
        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Metadata LLM returned invalid JSON (attempt %d/2): %s", attempt, exc)
            continue
        clips = parsed.get("clips")
        if not isinstance(clips, list):
            logger.warning("Metadata JSON missing 'clips' list (attempt %d/2)", attempt)
            continue
        by_index: dict[int, dict[str, Any]] = {}
        for entry in clips:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < count:
                by_index[idx] = entry
        return by_index
    logger.warning("Metadata LLM failed after 2 attempts; using fallback metadata.")
    return {}


def generate_metadata(
    candidates: list[dict[str, Any]],
    cfg: Config,
    *,
    no_llm: bool = False,
    llm: LLMCall | None = None,
) -> list[ClipMeta]:
    """Build :class:`ClipMeta` for every candidate with ONE batched LLM call.

    In ``no_llm`` mode (or if the LLM call fails / omits a clip) deterministic
    fallback metadata is used so rendering never depends on the network.
    """
    if not candidates:
        return []

    if no_llm:
        return [_fallback_meta(c) for c in candidates]

    try:
        call = llm if llm is not None else _meta_llm_call(cfg)
        by_index = _request_meta(_meta_prompt(candidates), call, len(candidates))
    except RenderError as exc:
        logger.warning("%s Falling back to offline metadata.", exc)
        by_index = {}

    metas: list[ClipMeta] = []
    for i, cand in enumerate(candidates):
        entry = by_index.get(i)
        fallback = _fallback_meta(cand)
        if entry is None:
            metas.append(fallback)
            continue
        description = str(entry.get("suggested_description", "")).strip()
        hashtags = _clean_hashtags(entry.get("hashtags"))
        metas.append(
            ClipMeta(
                title=fallback.title,
                hook_caption=fallback.hook_caption,
                suggested_description=description or fallback.suggested_description,
                hashtags=hashtags if len(hashtags) >= 5 else fallback.hashtags,
            )
        )
    return metas


# --------------------------------------------------------------------------- #
# Thumbnail
# --------------------------------------------------------------------------- #


def _probe_duration(path: Path) -> float:
    """Return the media duration in seconds (0.0 if it cannot be probed)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return 0.0
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return 0.0


def _extract_thumbnail(clip_path: Path, out_path: Path) -> None:
    """Grab a single JPEG frame at the hook moment (~1s into the clip)."""
    duration = _probe_duration(clip_path)
    offset = THUMB_OFFSET_S if duration <= 0 or duration > THUMB_OFFSET_S else duration / 2.0
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{offset:.3f}",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RenderError(
            "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
        ) from exc
    if proc.returncode != 0 or not out_path.is_file():
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RenderError("ffmpeg failed to extract thumbnail:\n" + "\n".join(tail))


# --------------------------------------------------------------------------- #
# Per-clip render
# --------------------------------------------------------------------------- #


def render_clip(
    job_id: str,
    index: int,
    candidate: dict[str, Any],
    meta: ClipMeta,
    words: list[Word],
    cfg: Config,
) -> Path:
    """Render one finished 1080x1920 clip and its thumbnail + meta sidecar.

    Reframes ``clip_NN_raw.mp4``, burns captions + the hook overlay in the final
    NVENC encode, writes ``clip_NN.jpg`` and ``clip_NN.meta.json``, and deletes
    the raw + reframed intermediates. Returns the final ``clip_NN.mp4`` path.
    Idempotent: an existing final clip is not re-rendered.
    """
    out_dir = cfg.data_path / "outputs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = out_dir / f"clip_{index:02d}_raw.mp4"
    reframed = out_dir / f"clip_{index:02d}_reframed.mp4"
    final = out_dir / f"clip_{index:02d}.mp4"
    ass_path = out_dir / f"clip_{index:02d}.ass"
    thumb = out_dir / f"clip_{index:02d}.jpg"
    meta_path = out_dir / f"clip_{index:02d}.meta.json"

    # Always (re)write the cheap sidecar metadata.
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    if final.is_file():
        logger.info("clip %d already rendered, skipping: %s", index, final.name)
        raw.unlink(missing_ok=True)
        if not thumb.is_file():
            _extract_thumbnail(final, thumb)
        return final

    if not raw.is_file():
        raise RenderError(f"Raw clip not found (run the cut stage first): {raw}")

    try:
        start = float(candidate["start"])
        end = float(candidate["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError(f"Candidate {index} has invalid start/end: {candidate!r}") from exc
    duration = end - start

    logger.info("Rendering clip %d: reframe %s", index, raw.name)
    try:
        reframe(raw, reframed, cfg)

        # Build captions (word-pop) + the top hook overlay as one ASS document,
        # then burn it in the final NVENC encode.
        ass_text = build_ass(words, CAPTION_STYLE, start, end)
        hook_caption = str(candidate.get("hook_caption", "")).strip()
        ass_text = add_hook_overlay(ass_text, hook_caption, duration, mode=HOOK_MODE)
        ass_path.write_text(ass_text, encoding="utf-8")

        logger.info("Rendering clip %d: burn captions + final encode -> %s", index, final.name)
        burn(reframed, ass_path, final)

        _extract_thumbnail(final, thumb)
    except Exception as exc:  # noqa: BLE001 - normalise every render failure
        final.unlink(missing_ok=True)
        raise RenderError(f"Failed to render clip {index}: {exc}") from exc
    finally:
        reframed.unlink(missing_ok=True)

    # Success: drop the raw intermediate.
    raw.unlink(missing_ok=True)
    logger.info("Rendered clip %d -> %s", index, final)
    return final


def render_clips(
    job_id: str,
    cfg: Config,
    *,
    no_llm: bool = False,
    llm: LLMCall | None = None,
) -> list[Path]:
    """Render every candidate of ``job_id`` into a posting-ready clip.

    Generates all clips' metadata with a single batched LLM call, then reframes,
    captions, encodes, thumbnails, and writes a ``.meta.json`` per clip. Returns
    the list of final ``clip_NN.mp4`` paths in candidate order. Raises
    :class:`RenderError` on any failure.
    """
    candidates = _load_candidates(job_id, cfg)
    if not candidates:
        logger.info("No candidates to render for job %s.", job_id)
        return []

    words = _load_words(job_id, cfg)
    metas = generate_metadata(candidates, cfg, no_llm=no_llm, llm=llm)

    outputs: list[Path] = []
    for i, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            raise RenderError(f"Candidate {i} is not an object: {cand!r}")
        outputs.append(render_clip(job_id, i, cand, metas[i], words, cfg))

    logger.info("Rendered %d clip(s) for job %s", len(outputs), job_id)
    return outputs


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    no_llm = "--no-llm" in argv
    positional = [a for a in argv if a != "--no-llm"]
    if len(positional) != 1:
        print("usage: python -m clipforge.render <job_id> [--no-llm]", file=sys.stderr)
        return 2

    job_id = positional[0]
    cfg = Config.load()
    try:
        outputs = render_clips(job_id, cfg, no_llm=no_llm)
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
