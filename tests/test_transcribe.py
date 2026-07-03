"""Tests for the transcription stage (clipforge.transcribe).

Generates a tiny 3-second dummy WAV with ffmpeg's lavfi sine source, so the
test needs no network and no fixture file. The model runs on CPU (int8) to stay
independent of any GPU.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipforge.config import Config
from clipforge.transcribe import TranscribeError, TranscriptResult, transcribe

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for the transcribe tests"
)


def _make_dummy_wav(path: Path, seconds: int = 3) -> None:
    """Generate a 3s 16kHz mono pcm_s16le WAV (a sine tone) via ffmpeg lavfi."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"ffmpeg failed to build dummy wav:\n{proc.stderr}"
    assert path.is_file()


def test_transcribe_produces_valid_json(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _make_dummy_wav(audio, seconds=3)

    # Force CPU so the test does not depend on a GPU being present.
    cfg = Config(data_dir=str(tmp_path / "data"), whisper_device="cpu")
    result = transcribe(audio, job_id="job123", cfg=cfg)

    assert isinstance(result, TranscriptResult)

    # transcript.json written to data/sources/{job_id}/.
    expected = cfg.data_path / "sources" / "job123" / "transcript.json"
    assert result.transcript_path == expected
    assert expected.is_file()

    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert isinstance(payload["language"], str)
    assert isinstance(payload["duration_s"], float)
    assert isinstance(payload["segments"], list)

    # Schema of any produced segments (a pure sine tone may yield none).
    for seg in payload["segments"]:
        assert set(seg) == {"start", "end", "text", "words"}
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert isinstance(seg["text"], str)
        for word in seg["words"]:
            assert set(word) == {"word", "start", "end"}


def test_transcribe_missing_audio_raises(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"), whisper_device="cpu")
    with pytest.raises(TranscribeError):
        transcribe(tmp_path / "nope.wav", job_id="job_missing", cfg=cfg)
