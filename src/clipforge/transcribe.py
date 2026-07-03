"""Stage 2 of the ClipForge pipeline: transcribe audio with faster-whisper.

Given ``data/sources/{job_id}/audio.wav`` (16kHz mono PCM produced by the ingest
stage), run word-level ASR and write a checkpoint transcript to
``data/sources/{job_id}/transcript.json`` so later stages can be retried without
re-transcribing.

Hardware constraint: the target GPU is a GTX 1060 6GB. The model must never be
larger than ``cfg.whisper_model`` (max ``small``), only one Whisper model may be
resident at a time, and VRAM is freed as soon as transcription completes so the
NVENC / reframe stages can use the GPU.

Run standalone against an already-ingested job:

    python -m clipforge.transcribe <job_id>
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from .config import Config

logger = logging.getLogger(__name__)

# How much processed audio (seconds) to advance before logging a progress line.
_PROGRESS_INTERVAL_S = 60.0


class TranscribeError(Exception):
    """Raised for any transcription failure, with a human-readable message."""


@dataclass(frozen=True)
class Word:
    """A single word with frame-accurate timing (seconds)."""

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    """A transcript segment with its words."""

    start: float
    end: float
    text: str
    words: list[Word]


@dataclass(frozen=True)
class TranscriptResult:
    """Result of the transcription stage; mirrors the on-disk transcript.json."""

    language: str
    duration_s: float
    segments: list[Segment]
    transcript_path: Path


def _load_model(cfg: Config) -> tuple[WhisperModel, str, str]:
    """Load a single Whisper model, honouring the VRAM constraint.

    Returns ``(model, device, compute_type)``. When ``cfg.whisper_device`` is
    ``"auto"`` we prefer CUDA (``int8_float16``) and fall back to CPU
    (``int8``) on *any* error — a missing/insufficient GPU must not crash the
    pipeline. An explicit device is used as-is with a sensible compute type.
    """
    model_size = cfg.whisper_model

    def _make(device: str, compute_type: str) -> WhisperModel:
        # One model at a time: the caller is responsible for freeing VRAM before
        # any other GPU stage runs.
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    device = cfg.whisper_device
    if device == "auto":
        try:
            model = _make("cuda", "int8_float16")
            logger.info("Loaded Whisper %r on CUDA (int8_float16).", model_size)
            return model, "cuda", "int8_float16"
        except Exception as exc:  # noqa: BLE001 - any GPU failure must fall back
            logger.warning("CUDA unavailable for Whisper (%s); falling back to CPU (int8).", exc)
            model = _make("cpu", "int8")
            logger.info("Loaded Whisper %r on CPU (int8).", model_size)
            return model, "cpu", "int8"

    compute_type = "int8_float16" if device == "cuda" else "int8"
    model = _make(device, compute_type)
    logger.info("Loaded Whisper %r on %s (%s).", model_size, device, compute_type)
    return model, device, compute_type


def _free_vram(model: WhisperModel) -> None:
    """Release the model and reclaim VRAM for later pipeline stages."""
    del model
    gc.collect()
    try:
        import torch  # noqa: PLC0415 - optional; only present on CUDA installs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch absent or CUDA hiccup is non-fatal
        pass


def transcribe(audio_path: Path, job_id: str, cfg: Config) -> TranscriptResult:
    """Transcribe ``audio_path`` and write ``transcript.json`` for ``job_id``.

    Uses word-level timestamps and VAD filtering with auto language detection.
    The model is loaded once and freed (along with its VRAM) before returning.
    Raises :class:`TranscribeError` with a human-readable message on failure.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise TranscribeError(f"Audio file not found: {audio_path}")

    out_dir = cfg.data_path / "sources" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.json"

    try:
        model, device, _ = _load_model(cfg)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message, not a traceback
        raise TranscribeError(f"Failed to load Whisper model {cfg.whisper_model!r}: {exc}") from exc

    try:
        segment_iter, info = model.transcribe(
            str(audio_path),
            vad_filter=True,
            word_timestamps=True,
            language=None,  # auto-detect (content is mostly English)
        )

        language = info.language or "unknown"
        duration_s = float(info.duration or 0.0)
        logger.info(
            "Transcribing %s: detected language=%s, duration=%.1fs (device=%s)",
            audio_path.name,
            language,
            duration_s,
            device,
        )

        segments: list[Segment] = []
        next_progress = _PROGRESS_INTERVAL_S
        # faster-whisper yields segments lazily; iterating drives the actual work.
        for seg in segment_iter:
            words = [
                Word(word=w.word, start=float(w.start), end=float(w.end))
                for w in (seg.words or [])
                if w.start is not None and w.end is not None
            ]
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            # Log every ~60s of audio processed so long videos show progress.
            while seg.end >= next_progress:
                logger.info("  ...processed %.0fs / %.0fs audio", next_progress, duration_s)
                next_progress += _PROGRESS_INTERVAL_S
    except TranscribeError:
        raise
    except Exception as exc:  # noqa: BLE001 - never let a raw ASR traceback escape
        raise TranscribeError(f"Transcription failed for {audio_path.name}: {exc}") from exc
    finally:
        # CRITICAL: free VRAM before any later GPU stage, even on failure.
        _free_vram(model)

    if not segments:
        logger.warning(
            "No speech segments produced for %s (empty or silent audio).", audio_path.name
        )

    payload = {
        "language": language,
        "duration_s": duration_s,
        "segments": [asdict(s) for s in segments],
    }
    transcript_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d segment(s) to %s", len(segments), transcript_path)

    return TranscriptResult(
        language=language,
        duration_s=duration_s,
        segments=segments,
        transcript_path=transcript_path,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m clipforge.transcribe <job_id>", file=sys.stderr)
        return 2

    job_id = argv[0]
    cfg = Config.load()
    audio_path = cfg.data_path / "sources" / job_id / "audio.wav"
    try:
        result = transcribe(audio_path, job_id, cfg)
    except TranscribeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "job_id": job_id,
                "language": result.language,
                "duration_s": result.duration_s,
                "segments": len(result.segments),
                "transcript_path": str(result.transcript_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
