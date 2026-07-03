"""Tests for the cutting stage (clipforge.cut).

Like the download tests, these build a tiny dummy MP4 with ffmpeg's lavfi
testsrc, so they need no network and no fixture file. Candidates are written to
disk by hand (the analyze stage is not involved).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipforge.config import Config
from clipforge.cut import CutError, cut_clips

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for the cut tests"
)


def _make_dummy_mp4(path: Path, seconds: int = 3) -> None:
    """Generate a tiny test MP4 (color bars + tone) via ffmpeg lavfi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={seconds}:size=320x240:rate=15",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"ffmpeg failed to build dummy mp4:\n{proc.stderr}"
    assert path.is_file()


def _probe_duration(path: Path) -> float:
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def _setup_job(tmp_path: Path, candidates: list[dict[str, object]], seconds: int = 3) -> Config:
    """Write a dummy source video + candidates.json for a job named 'job'."""
    cfg = Config(data_dir=str(tmp_path / "data"))
    job_dir = cfg.data_path / "sources" / "job"
    _make_dummy_mp4(job_dir / "video.mp4", seconds=seconds)
    (job_dir / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    return cfg


def test_cut_produces_raw_clips(tmp_path: Path) -> None:
    cfg = _setup_job(
        tmp_path,
        [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.5},
        ],
    )

    outputs = cut_clips("job", cfg)

    out_dir = cfg.data_path / "outputs" / "job"
    assert outputs == [out_dir / "clip_00_raw.mp4", out_dir / "clip_01_raw.mp4"]
    for path in outputs:
        assert path.is_file()

    # Durations are honoured (roughly) and the clips are real, playable media.
    assert _probe_duration(outputs[0]) == pytest.approx(1.0, abs=0.3)
    assert _probe_duration(outputs[1]) == pytest.approx(1.5, abs=0.3)


def test_cut_is_idempotent(tmp_path: Path) -> None:
    cfg = _setup_job(tmp_path, [{"start": 0.0, "end": 1.0}])

    first = cut_clips("job", cfg)
    mtime = first[0].stat().st_mtime_ns

    # A second run must not re-cut the existing file.
    second = cut_clips("job", cfg)
    assert second == first
    assert second[0].stat().st_mtime_ns == mtime


def test_cut_empty_candidates_returns_empty(tmp_path: Path) -> None:
    cfg = _setup_job(tmp_path, [])
    assert cut_clips("job", cfg) == []


def test_cut_missing_candidates_raises(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    with pytest.raises(CutError):
        cut_clips("nope", cfg)


def test_cut_missing_video_raises(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    job_dir = cfg.data_path / "sources" / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "candidates.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
    )
    with pytest.raises(CutError):
        cut_clips("job", cfg)


def test_cut_invalid_duration_raises(tmp_path: Path) -> None:
    cfg = _setup_job(tmp_path, [{"start": 2.0, "end": 1.0}])
    with pytest.raises(CutError):
        cut_clips("job", cfg)
