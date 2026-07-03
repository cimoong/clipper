"""Pipeline orchestration for ClipForge.

Wires the four stage modules together in order:

    download -> transcribe -> analyze -> cut

Each job gets a folder under ``data/sources/{job_id}/``. After every stage a
checkpoint is written to ``state.json`` so a failed or interrupted run can be
continued with :func:`resume_job` (``python -m clipforge resume <job_id>``)
instead of starting over.

The stage modules are treated as black boxes — this module only calls their
public entry points and moves their on-disk artifacts along; it never reaches
into their internals.

The final result of a completed job is written to
``data/outputs/{job_id}/results.json``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import db, render
from .analyze import analyze
from .config import Config
from .cut import cut_clips, cut_one_clip
from .download import download
from .render import ClipMeta, generate_metadata, load_words, render_clip, render_clips
from .transcribe import transcribe

logger = logging.getLogger(__name__)

# The pipeline stages, in the order they must run.
STAGES: tuple[str, ...] = ("download", "transcribe", "analyze", "cut", "render")

# Called at the START of each stage as ``progress(stage, stage_index, total)``,
# where ``stage_index`` is the 0-based position of ``stage`` in :data:`STAGES`.
# The queue worker uses this to write live status + percent to the jobs table.
ProgressFn = Callable[[str, int, int], None]

# Transient-failure retry policy. A stage that fails with a *transient* error
# (network blip, LLM rate-limit / timeout) is retried once after a short delay
# before the failure is surfaced; permanent errors (bad URL, missing API key,
# unsupported format) are never retried. ``RETRY_DELAY_S`` is a module global so
# tests can drop it to 0.
STAGE_RETRIES = 1
RETRY_DELAY_S = 5.0

# Substrings (matched case-insensitively against the error message) that mark a
# failure as transient and therefore worth one automatic retry.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "network",
    "connection",
    "timed out",
    "timeout",
    "temporarily",
    "try again",
    "rate limit",
    "ratelimit",
    "quota",
    "deadline",
    "overloaded",
    "unavailable: the model",  # transient Gemini 503, not "video unavailable"
    "getaddrinfo",
    "reset by peer",
    "unreachable",
    " 429",
    " 500",
    " 502",
    " 503",
    " 504",
)


def _is_transient(exc: Exception) -> bool:
    """True if ``exc``'s message looks like a retryable network/LLM hiccup."""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


# --------------------------------------------------------------------------- #
# Runtime settings overlay
#
# The web UI persists tunables (LLM model, num_clips, clip durations, caption
# style, hook mode) into the SQLite ``settings`` table. Every pipeline run reads
# them at start and overlays them onto the env-loaded Config, so "re-analyze with
# current settings" (and any new job) honours the latest saved values without
# the env / .env being edited. An empty settings table leaves the Config as-is.
# --------------------------------------------------------------------------- #

VALID_CAPTION_STYLES = {"bold", "clean"}
VALID_HOOK_MODES = {"3s", "full"}

# settings-table keys that map 1:1 onto a Config field of the same name.
_SETTING_STR_FIELDS = ("llm_provider", "gemini_model", "anthropic_model", "whisper_model")
_SETTING_INT_FIELDS = ("num_clips", "clip_min_s", "clip_max_s")


def load_settings(cfg: Config) -> dict[str, Any]:
    """Read all persisted settings for ``cfg`` (typed), skipping unset keys."""
    conn = db.connect(cfg)
    try:
        out: dict[str, Any] = {}
        for key in _SETTING_STR_FIELDS + ("caption_style", "hook_mode"):
            value = db.get_setting(conn, key)
            if value is not None and value != "":
                out[key] = value
        for key in _SETTING_INT_FIELDS:
            value = db.get_setting(conn, key)
            if value is None or value == "":
                continue
            try:
                out[key] = int(value)
            except ValueError:
                logger.warning("Ignoring non-integer setting %s=%r", key, value)
        return out
    finally:
        conn.close()


def effective_config(cfg: Config, settings: dict[str, Any] | None = None) -> Config:
    """Return ``cfg`` with any saved Config-field settings overlaid on top."""
    if settings is None:
        settings = load_settings(cfg)
    overrides = {
        key: settings[key] for key in _SETTING_STR_FIELDS + _SETTING_INT_FIELDS if key in settings
    }
    return replace(cfg, **overrides) if overrides else cfg


def _apply_render_presentation(settings: dict[str, Any]) -> None:
    """Push caption-style / hook-mode settings into the render module globals.

    ``render`` reads these as module-level constants (there is no Config field for
    them); the single serial worker means mutating them here is safe. Invalid or
    unset values leave the render defaults untouched.
    """
    style = settings.get("caption_style")
    if style in VALID_CAPTION_STYLES:
        render.CAPTION_STYLE = style
    mode = settings.get("hook_mode")
    if mode in VALID_HOOK_MODES:
        render.HOOK_MODE = mode


class PipelineError(Exception):
    """Raised when a stage fails; carries the stage name and job id.

    The CLI uses these to print the failing stage, the underlying error, and the
    exact command to resume the job.
    """

    def __init__(self, stage: str, message: str, job_id: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.job_id = job_id

    @property
    def resume_command(self) -> str:
        return f"python -m clipforge resume {self.job_id}"


@dataclass
class Job:
    """Mutable state for a single pipeline run, persisted to ``state.json``.

    ``completed`` is the checkpoint list read by :func:`resume_job`. ``title``
    and ``source_url`` are carried so the final ``results.json`` can be built
    even when the download stage was completed in an earlier (resumed) run.
    """

    job_id: str
    source_url: str
    title: str = ""
    no_llm: bool = False
    completed: list[str] = field(default_factory=list)

    def to_state(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "source_url": self.source_url,
            "title": self.title,
            "no_llm": self.no_llm,
        }


# --------------------------------------------------------------------------- #
# Job / checkpoint persistence
# --------------------------------------------------------------------------- #


def new_job_id() -> str:
    """A sortable, unique job id: ``YYYYmmdd-HHMMSS-<6 hex>``."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


# Backwards-compatible private alias (this module used ``_new_job_id`` internally).
_new_job_id = new_job_id


def _source_dir(job_id: str, cfg: Config) -> Path:
    return cfg.data_path / "sources" / job_id


def _state_path(job_id: str, cfg: Config) -> Path:
    return _source_dir(job_id, cfg) / "state.json"


def _save_state(job: Job, cfg: Config) -> None:
    """Write the job's checkpoint to ``state.json`` (called after each stage)."""
    path = _state_path(job.job_id, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job.to_state(), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_job(job_id: str, cfg: Config) -> Job:
    """Reconstruct a :class:`Job` from its ``state.json`` for resuming."""
    path = _state_path(job_id, cfg)
    if not path.is_file():
        raise PipelineError(
            "resume",
            f"No job found with id {job_id!r} (missing {path}).",
            job_id,
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError("resume", f"Could not read job state {path}: {exc}", job_id) from exc

    completed = state.get("completed") if isinstance(state, dict) else None
    if not isinstance(completed, list):
        completed = []
    return Job(
        job_id=job_id,
        source_url=str(state.get("source_url", "")),
        title=str(state.get("title", "")),
        no_llm=bool(state.get("no_llm", False)),
        completed=[str(s) for s in completed],
    )


def _first_incomplete(completed: list[str]) -> int:
    """Index into :data:`STAGES` of the first stage not yet completed."""
    done = set(completed)
    for i, stage in enumerate(STAGES):
        if stage not in done:
            return i
    return len(STAGES)


# --------------------------------------------------------------------------- #
# Stage execution
# --------------------------------------------------------------------------- #


def _fake_candidate(title: str) -> dict[str, Any]:
    """A single offline candidate covering seconds 0-3 (used by ``--no-llm``)."""
    return {
        "start": 0.0,
        "end": 3.0,
        "score": 50,
        "title": (title or "Offline test clip")[:70],
        "hook": "(offline test — LLM analysis skipped)",
        "hook_caption": "OFFLINE TEST CLIP",
        "reason": "Generated by --no-llm for offline pipeline testing.",
        "sub_scores": {"hook": 0, "emotion": 0, "value": 0, "coherence": 0, "payoff": 0},
    }


def _run_analyze(job: Job, cfg: Config) -> None:
    """Run the analyze stage, or write one fake candidate in ``--no-llm`` mode.

    The fake candidate is written directly to ``candidates.json`` (bypassing the
    analyze module's LLM call and its 25-75s duration validation) so the pipeline
    can be exercised end-to-end offline.
    """
    if job.no_llm:
        out_path = _source_dir(job.job_id, cfg) / "candidates.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([_fake_candidate(job.title)], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("--no-llm: wrote 1 fake candidate to %s", out_path)
        return
    analyze(job.job_id, cfg)


def _run_stage_body(job: Job, cfg: Config, stage: str) -> None:
    """Dispatch a single stage to its module (no timing/retry/checkpointing)."""
    if stage == "download":
        result = download(job.source_url, job.job_id, cfg)
        job.title = result.title
    elif stage == "transcribe":
        audio_path = _source_dir(job.job_id, cfg) / "audio.wav"
        transcribe(audio_path, job.job_id, cfg)
    elif stage == "analyze":
        _run_analyze(job, cfg)
    elif stage == "cut":
        cut_clips(job.job_id, cfg)
    elif stage == "render":
        render_clips(job.job_id, cfg, no_llm=job.no_llm)
    else:  # pragma: no cover - guards against a typo in STAGES
        raise PipelineError(stage, f"Unknown stage {stage!r}.", job.job_id)


def _execute_stage(job: Job, cfg: Config, stage: str) -> None:
    """Run one stage (retrying once on a transient error), time it, checkpoint.

    A transient failure (see :func:`_is_transient`) is retried up to
    :data:`STAGE_RETRIES` times after a :data:`RETRY_DELAY_S`-second pause;
    anything else — or a retry that still fails — is wrapped in a
    :class:`PipelineError` naming the stage so the job can later be resumed.
    """
    logger.info("--- stage %r: start ---", stage)
    started = time.perf_counter()
    attempt = 0
    while True:
        try:
            _run_stage_body(job, cfg, stage)
            break
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize every stage failure
            if attempt < STAGE_RETRIES and _is_transient(exc):
                attempt += 1
                logger.warning(
                    "--- stage %r: transient failure (%s); retry %d/%d in %.0fs ---",
                    stage,
                    exc,
                    attempt,
                    STAGE_RETRIES,
                    RETRY_DELAY_S,
                )
                time.sleep(RETRY_DELAY_S)
                continue
            elapsed = time.perf_counter() - started
            logger.error("--- stage %r: FAILED after %.1fs ---", stage, elapsed)
            raise PipelineError(stage, str(exc), job.job_id) from exc

    elapsed = time.perf_counter() - started
    logger.info("--- stage %r: done in %.1fs ---", stage, elapsed)

    if stage not in job.completed:
        job.completed.append(stage)
    _save_state(job, cfg)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def _results_path(job_id: str, cfg: Config) -> Path:
    return cfg.data_path / "outputs" / job_id / "results.json"


def _load_clip_meta(path: Path) -> dict[str, Any]:
    """Load a clip's ``.meta.json`` sidecar; return ``{}`` if absent/unreadable."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _build_results(job: Job, cfg: Config) -> dict[str, Any]:
    """Assemble ``results.json`` from ``candidates.json`` + the rendered clips.

    Candidate ``i`` maps to the finished ``clip_{i:02d}.mp4`` (the render stage's
    naming), its ``clip_{i:02d}.jpg`` thumbnail, and its ``clip_{i:02d}.meta.json``
    sidecar. Clips are sorted by score descending.
    """
    candidates_path = _source_dir(job.job_id, cfg) / "candidates.json"
    try:
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(
            "cut", f"Could not read candidates {candidates_path}: {exc}", job.job_id
        ) from exc
    if not isinstance(candidates, list):
        raise PipelineError(
            "cut", f"Candidates file is not a JSON list: {candidates_path}", job.job_id
        )

    out_dir = cfg.data_path / "outputs" / job.job_id
    clips: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue
        meta = _load_clip_meta(out_dir / f"clip_{i:02d}.meta.json")
        clips.append(
            {
                "file": str(out_dir / f"clip_{i:02d}.mp4"),
                "thumb": str(out_dir / f"clip_{i:02d}.jpg"),
                "meta": str(out_dir / f"clip_{i:02d}.meta.json"),
                "start": cand.get("start"),
                "end": cand.get("end"),
                "score": cand.get("score", 0),
                "title": cand.get("title", ""),
                "hook": cand.get("hook", ""),
                "hook_caption": cand.get("hook_caption", ""),
                "reason": cand.get("reason", ""),
                "suggested_description": meta.get("suggested_description", ""),
                "hashtags": meta.get("hashtags", []),
            }
        )

    clips.sort(key=lambda c: c.get("score") or 0, reverse=True)

    results = {
        "job_id": job.job_id,
        "source_url": job.source_url,
        "title": job.title,
        "clips": clips,
    }

    out_path = _results_path(job.job_id, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d clip(s) to %s", len(clips), out_path)
    return results


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


def _drive(job: Job, cfg: Config, *, progress: ProgressFn | None = None) -> dict[str, Any]:
    """Run every not-yet-completed stage in order, then build results.

    ``progress`` (when given) is invoked just before each stage starts with the
    stage name, its absolute index in :data:`STAGES`, and the total stage count,
    so a caller can surface live status. It never affects control flow.

    Current UI settings are overlaid onto ``cfg`` here (see
    :func:`effective_config`), so a re-analyze or a new job always runs with the
    latest saved LLM model / num_clips / clip durations / caption presentation.
    """
    settings = load_settings(cfg)
    cfg = effective_config(cfg, settings)
    _apply_render_presentation(settings)

    start_index = _first_incomplete(job.completed)
    remaining = STAGES[start_index:]
    if remaining:
        logger.info("Job %s: running stage(s) %s", job.job_id, ", ".join(remaining))
    else:
        logger.info("Job %s: all stages already completed; rebuilding results.", job.job_id)

    overall = time.perf_counter()
    for offset, stage in enumerate(remaining):
        if progress is not None:
            progress(stage, start_index + offset, len(STAGES))
        _execute_stage(job, cfg, stage)
    logger.info("Job %s: pipeline finished in %.1fs", job.job_id, time.perf_counter() - overall)

    return _build_results(job, cfg)


def run_new(
    source_url: str,
    cfg: Config,
    *,
    no_llm: bool = False,
    job_id: str | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Start a fresh job for ``source_url`` and run it to completion.

    ``job_id`` lets a caller (e.g. the queue worker) pin the id so the on-disk
    artifacts and an external record share it; when omitted a new one is minted.
    ``progress`` is forwarded to :func:`_drive`. Returns the assembled
    ``results.json`` payload. Raises :class:`PipelineError` if any stage fails.
    """
    job = Job(job_id=job_id or _new_job_id(), source_url=source_url, no_llm=no_llm)
    _source_dir(job.job_id, cfg).mkdir(parents=True, exist_ok=True)
    _save_state(job, cfg)
    logger.info("Created job %s for source %r (no_llm=%s)", job.job_id, source_url, no_llm)
    return _drive(job, cfg, progress=progress)


def _rerender_meta(
    out_dir: Path, index: int, candidate: dict[str, Any], cfg: Config, *, no_llm: bool
) -> ClipMeta:
    """Metadata for a single re-render: reuse the existing sidecar, refresh the hook.

    A re-render only changes the on-screen hook caption, so the (possibly
    LLM-generated) description and hashtags from the clip's ``.meta.json`` are
    preserved and only ``hook_caption`` is taken from the (edited) candidate. If
    no sidecar exists, deterministic offline metadata is generated instead.
    """
    hook_caption = str(candidate.get("hook_caption", "")).strip()
    sidecar = _load_clip_meta(out_dir / f"clip_{index:02d}.meta.json")
    if sidecar:
        return ClipMeta(
            title=str(sidecar.get("title", "") or candidate.get("title", "")),
            hook_caption=hook_caption,
            suggested_description=str(sidecar.get("suggested_description", "")),
            hashtags=list(sidecar.get("hashtags") or []),
        )
    return generate_metadata([candidate], cfg, no_llm=True)[0]


def rerender_clip(
    job_id: str,
    index: int,
    cfg: Config,
    *,
    hook_caption: str | None = None,
    no_llm: bool = False,
) -> dict[str, Any]:
    """Re-render ONLY clip ``index`` of a finished job and return its results entry.

    The raw 16:9 cut is deleted after the initial render, so it is first re-cut
    from the source video (idempotent), then the final ``clip_{index:02d}.mp4`` is
    removed to force a fresh reframe + caption burn. When ``hook_caption`` is
    given the candidate's on-screen hook is updated in ``candidates.json`` before
    rendering so the new text is burned in. Finally ``results.json`` is rebuilt.
    Raises :class:`PipelineError` if the clip cannot be re-rendered.
    """
    src_dir = _source_dir(job_id, cfg)
    candidates_path = src_dir / "candidates.json"
    try:
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(
            "render", f"Could not read candidates {candidates_path}: {exc}", job_id
        ) from exc
    if not isinstance(candidates, list) or not 0 <= index < len(candidates):
        raise PipelineError("render", f"No candidate at index {index} for job {job_id}.", job_id)

    candidate = candidates[index]
    if hook_caption is not None:
        candidate["hook_caption"] = hook_caption
        candidates_path.write_text(
            json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    out_dir = cfg.data_path / "outputs" / job_id
    try:
        cut_one_clip(job_id, index, cfg)  # idempotent; re-cuts the deleted raw
        (out_dir / f"clip_{index:02d}.mp4").unlink(missing_ok=True)  # force re-render
        meta = _rerender_meta(out_dir, index, candidate, cfg, no_llm=no_llm)
        render_clip(job_id, index, candidate, meta, load_words(job_id, cfg), cfg)
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise every re-render failure
        raise PipelineError("render", str(exc), job_id) from exc

    job = _load_job(job_id, cfg)
    results = _build_results(job, cfg)
    target = f"clip_{index:02d}.mp4"
    for clip in results.get("clips", []):
        if str(clip.get("file", "")).endswith(target):
            return clip
    raise PipelineError("render", f"Re-rendered clip {index} missing from results.", job_id)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst`` (space-free), copying if linking is impossible."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def setup_reanalyze_job(
    orig_job_id: str,
    cfg: Config,
    *,
    source_url: str,
    title: str = "",
    no_llm: bool = False,
    new_id: str | None = None,
) -> str:
    """Prepare a NEW job that re-runs analyze -> cut -> render from ``orig_job_id``'s
    cached transcript, and return its id.

    The new job reuses the original's downloaded video (hard-linked, so no extra
    disk and no re-download) and its cached ``transcript.json`` (so no
    re-transcribe). Its ``state.json`` is written with the download + transcribe
    stages already marked complete, so a subsequent :func:`resume_job` (e.g. via
    the queue's ``enqueue_resume``) starts straight at the analyze stage and picks
    up the current UI settings. Raises :class:`PipelineError` if the original's
    transcript or source video is unavailable.
    """
    new_id = new_id or new_job_id()
    orig_dir = _source_dir(orig_job_id, cfg)
    transcript = orig_dir / "transcript.json"
    video = orig_dir / "video.mp4"
    if not transcript.is_file():
        raise PipelineError(
            "analyze",
            f"No cached transcript for job {orig_job_id}; cannot re-analyze.",
            new_id,
        )
    if not video.is_file():
        raise PipelineError(
            "cut",
            f"Source video for job {orig_job_id} is unavailable (was it cleaned up?).",
            new_id,
        )

    new_dir = _source_dir(new_id, cfg)
    new_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(video, new_dir / "video.mp4")
    shutil.copyfile(transcript, new_dir / "transcript.json")

    job = Job(
        job_id=new_id,
        source_url=source_url,
        title=title,
        no_llm=no_llm,
        completed=["download", "transcribe"],
    )
    _save_state(job, cfg)
    logger.info("Prepared re-analyze job %s from %s", new_id, orig_job_id)
    return new_id


def resume_job(
    job_id: str,
    cfg: Config,
    *,
    no_llm: bool | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Resume ``job_id`` from its first incomplete stage.

    ``no_llm`` overrides the value saved in the job's state when not ``None``.
    ``progress`` is forwarded to :func:`_drive`. Raises :class:`PipelineError`
    if the job is unknown or a stage fails.
    """
    job = _load_job(job_id, cfg)
    if no_llm is not None:
        job.no_llm = no_llm
    logger.info("Resuming job %s (completed: %s)", job_id, ", ".join(job.completed) or "none")
    return _drive(job, cfg, progress=progress)


# --------------------------------------------------------------------------- #
# Storage cleanup
# --------------------------------------------------------------------------- #


def _latest_mtime(path: Path) -> float | None:
    """Newest modification time (epoch seconds) across ``path`` and its tree."""
    mtimes: list[float] = []
    try:
        mtimes.append(path.stat().st_mtime)
    except OSError:
        return None
    for child in path.rglob("*"):
        try:
            mtimes.append(child.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else None


def cleanup_sources(cfg: Config, *, days: int, now: datetime | None = None) -> list[str]:
    """Delete ``data/sources/{job_id}`` folders untouched for more than ``days``.

    Only the (large) source videos / intermediate audio are removed; the finished
    clips under ``data/outputs`` and the SQLite database are left intact (PRD §9
    auto-cleanup). A folder's age is its newest file's mtime, so a job still being
    written is never collected. Returns the removed job-folder names.
    """
    if days < 0:
        raise ValueError("days must be >= 0")
    sources_root = cfg.data_path / "sources"
    if not sources_root.is_dir():
        return []

    cutoff = (now or datetime.now()).timestamp() - days * 86400.0
    removed: list[str] = []
    for child in sorted(sources_root.iterdir()):
        if not child.is_dir():
            continue
        mtime = _latest_mtime(child)
        if mtime is None or mtime >= cutoff:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed.append(child.name)
        logger.info("cleanup: removed source folder %s", child)
    return removed
