"""Stage 4 of the ClipForge pipeline: cut raw clips from the source video.

Given ``data/sources/{job_id}/candidates.json`` (produced by the analyze stage)
and the ingested ``data/sources/{job_id}/video.mp4``, cut each candidate's
[start, end] span into ``data/outputs/{job_id}/clip_{i:02d}_raw.mp4``.

The clips are still 16:9 and are NOT reframed here — reframing to 9:16 and the
final NVENC encode happen in later stages. These raw intermediates are cut with
CPU x264 (``-ss`` before ``-i`` + re-encode) for frame-accurate boundaries.

Cutting is idempotent: an existing output file is left untouched, so a partially
completed run can be resumed.

Run standalone against an already-analyzed job:

    python -m clipforge.cut <job_id>
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


class CutError(Exception):
    """Raised for any cutting failure, with a human-readable message.

    Raw ffmpeg tracebacks never escape; ffmpeg failures are re-raised as this
    exception with the last few lines of stderr attached for diagnosis.
    """


def _load_candidates(job_id: str, cfg: Config) -> list[dict[str, Any]]:
    """Load ``candidates.json`` for ``job_id`` as a list of raw dicts."""
    path = cfg.data_path / "sources" / job_id / "candidates.json"
    if not path.is_file():
        raise CutError(f"Candidates not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CutError(f"Could not read candidates {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise CutError(f"Candidates file is not a JSON list: {path}")
    return payload


def _cut_one(video_path: Path, start: float, duration: float, out_path: Path) -> None:
    """Cut a single [start, start+duration] span into ``out_path`` with x264.

    ``-ss`` is placed before ``-i`` for a fast seek, and the segment is
    re-encoded (never stream-copied) so cuts are frame-accurate.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise CutError(
            "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
        ) from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        detail = "\n".join(tail) if tail else "(no ffmpeg output)"
        raise CutError(f"ffmpeg failed to cut {out_path.name}:\n{detail}")

    if not out_path.is_file():
        raise CutError(f"ffmpeg reported success but produced no file at {out_path}.")


def cut_clips(job_id: str, cfg: Config) -> list[Path]:
    """Cut every candidate of ``job_id`` into a raw 16:9 clip.

    Reads ``candidates.json``, and for each candidate ``i`` writes
    ``data/outputs/{job_id}/clip_{i:02d}_raw.mp4`` (unless it already exists).
    Returns the list of output paths in candidate order. Raises
    :class:`CutError` on any failure.
    """
    candidates = _load_candidates(job_id, cfg)

    video_path = cfg.data_path / "sources" / job_id / "video.mp4"
    if not video_path.is_file():
        raise CutError(f"Source video not found: {video_path}")

    out_dir = cfg.data_path / "outputs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for i, cand in enumerate(candidates):
        out_path = out_dir / f"clip_{i:02d}_raw.mp4"
        outputs.append(out_path)

        if out_path.is_file():
            logger.info("clip %d already exists, skipping: %s", i, out_path.name)
            continue

        try:
            start = float(cand["start"])
            end = float(cand["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CutError(f"Candidate {i} has invalid start/end: {cand!r}") from exc

        duration = end - start
        if duration <= 0:
            raise CutError(f"Candidate {i} has non-positive duration ({start} -> {end}).")

        logger.info("Cutting clip %d: %.3f -> %.3f (%.3fs)", i, start, end, duration)
        _cut_one(video_path, start, duration, out_path)

    logger.info("Cut %d clip(s) for job %s", len(outputs), job_id)
    return outputs


def cut_one_clip(job_id: str, index: int, cfg: Config) -> Path:
    """Cut a single candidate's raw 16:9 clip (used by web single-clip re-render).

    Reads ``candidates.json``, cuts ``candidates[index]`` into
    ``clip_{index:02d}_raw.mp4``, and returns its path. Idempotent: an existing
    raw file is left untouched. Raises :class:`CutError` on any failure.
    """
    candidates = _load_candidates(job_id, cfg)
    if not 0 <= index < len(candidates):
        raise CutError(f"No candidate at index {index} for job {job_id}.")
    cand = candidates[index]

    video_path = cfg.data_path / "sources" / job_id / "video.mp4"
    if not video_path.is_file():
        raise CutError(f"Source video not found: {video_path}")

    out_dir = cfg.data_path / "outputs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clip_{index:02d}_raw.mp4"
    if out_path.is_file():
        return out_path

    try:
        start = float(cand["start"])
        end = float(cand["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CutError(f"Candidate {index} has invalid start/end: {cand!r}") from exc

    duration = end - start
    if duration <= 0:
        raise CutError(f"Candidate {index} has non-positive duration ({start} -> {end}).")

    logger.info("Re-cutting clip %d: %.3f -> %.3f (%.3fs)", index, start, end, duration)
    _cut_one(video_path, start, duration, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m clipforge.cut <job_id>", file=sys.stderr)
        return 2

    job_id = argv[0]
    cfg = Config.load()
    try:
        outputs = cut_clips(job_id, cfg)
    except CutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
