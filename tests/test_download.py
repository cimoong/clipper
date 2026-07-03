"""Tests for the ingest stage (clipforge.download).

The local-file test generates a tiny 3-second dummy MP4 with ffmpeg's lavfi
testsrc, so it needs no network and no fixture file.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from clipforge.config import Config
from clipforge.download import DownloadError, DownloadResult, download

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for the download tests"
)


def _make_dummy_mp4(path: Path, seconds: int = 3) -> None:
    """Generate a tiny test MP4 (color bars + tone) via ffmpeg lavfi."""
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


def test_download_local_file(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    _make_dummy_mp4(source, seconds=3)

    cfg = Config(data_dir=str(tmp_path / "data"))
    result = download(str(source), job_id="job123", cfg=cfg)

    assert isinstance(result, DownloadResult)

    # Video was copied into data/sources/{job_id}/video.mp4.
    expected_dir = cfg.data_path / "sources" / "job123"
    assert result.video_path == expected_dir / "video.mp4"
    assert result.video_path.is_file()

    # Audio extracted as 16kHz mono pcm_s16le WAV.
    assert result.audio_path == expected_dir / "audio.wav"
    assert result.audio_path.is_file()
    with wave.open(str(result.audio_path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2  # pcm_s16le => 16-bit
        assert wav.getnframes() > 0

    # Title comes from the source filename; duration is ~3s.
    assert result.title == "input"
    assert result.duration_s == pytest.approx(3.0, abs=0.5)


def test_download_empty_input_raises(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    with pytest.raises(DownloadError):
        download("   ", job_id="job_empty", cfg=cfg)


def test_download_unsupported_local_format_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not a video")

    cfg = Config(data_dir=str(tmp_path / "data"))
    with pytest.raises(DownloadError):
        download(str(bogus), job_id="job_bad", cfg=cfg)
