"""Stage 1 of the ClipForge pipeline: ingest a video and extract its audio.

Given a YouTube URL (or a local video file), download the best <=1080p MP4 into
``data/sources/{job_id}/video.mp4`` and extract a 16kHz mono PCM WAV alongside it
at ``audio.wav`` (the format faster-whisper expects).

Run standalone to ingest a single source (generates a random job_id):

    python -m clipforge.download <url_or_path>
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config


class DownloadError(Exception):
    """Raised for any ingest failure, with a human-readable message.

    Raw yt-dlp / ffmpeg tracebacks are never allowed to escape; they are caught
    and re-raised as this exception with a message safe to show a user.
    """


@dataclass(frozen=True)
class DownloadResult:
    """Result of the ingest stage."""

    video_path: Path
    audio_path: Path
    title: str
    duration_s: float


# Video container extensions we accept as a local file input.
_LOCAL_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def _looks_like_local_path(url_or_path: str) -> bool:
    """Return True if the input points at an existing local file."""
    try:
        return Path(url_or_path).is_file()
    except OSError:
        # e.g. a URL far longer than the OS path limit — clearly not a local file.
        return False


def _probe_duration_s(path: Path) -> float:
    """Return the duration of a media file in seconds via ffprobe (0.0 if unknown)."""
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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise DownloadError(
            "ffprobe was not found on PATH. Install FFmpeg (which bundles ffprobe) "
            "and ensure it is on your PATH."
        ) from exc
    except subprocess.CalledProcessError:
        return 0.0
    raw = proc.stdout.strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono pcm_s16le WAV from ``video_path`` into ``audio_path``."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise DownloadError(
            "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        detail = "\n".join(tail) if tail else "(no ffmpeg output)"
        raise DownloadError(f"Failed to extract audio from {video_path.name}:\n{detail}")
    if not audio_path.is_file():
        raise DownloadError(f"Audio extraction produced no output file at {audio_path}.")


def _ingest_local(source: Path, video_path: Path) -> tuple[str, float]:
    """Copy a local video file into the job folder. Returns (title, duration_s)."""
    if source.suffix.lower() not in _LOCAL_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(_LOCAL_VIDEO_SUFFIXES))
        raise DownloadError(
            f"Unsupported local video format {source.suffix!r}. Supported: {allowed}."
        )
    try:
        shutil.copyfile(source, video_path)
    except OSError as exc:
        raise DownloadError(f"Could not copy local video {source}: {exc}") from exc
    duration = _probe_duration_s(video_path)
    return source.stem, duration


def _ingest_remote(url: str, job_dir: Path, video_path: Path) -> tuple[str, float]:
    """Download a remote video with yt-dlp. Returns (title, duration_s)."""
    # Imported lazily so the module (and its local-file path) load without yt-dlp.
    import yt_dlp
    from yt_dlp.utils import DownloadError as YtDownloadError

    ydl_opts = {
        # Best <=1080p video + best audio, muxed into mp4.
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
        "best[height<=1080][ext=mp4]/best[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(job_dir / "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Swallow yt-dlp's own stderr logging; we surface a friendly message ourselves.
        "logger": _NullLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except YtDownloadError as exc:
        raise DownloadError(_humanize_ytdlp_error(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 - never let a raw traceback escape
        raise DownloadError(f"Unexpected error while downloading {url!r}: {exc}") from exc

    if info is None:
        raise DownloadError(f"Could not retrieve any video from {url!r}.")

    # yt-dlp may write the merged file under a different extension; if video.mp4
    # is missing, locate whatever it produced and normalise the name.
    if not video_path.is_file():
        produced = sorted(job_dir.glob("video.*"))
        if not produced:
            raise DownloadError(f"Download finished but no video file was produced for {url!r}.")
        produced[0].replace(video_path)

    title = info.get("title") or video_path.stem
    duration = info.get("duration")
    duration_s = float(duration) if duration else _probe_duration_s(video_path)
    return title, duration_s


class _NullLogger:
    """Silences yt-dlp's internal logging so we control all user-facing output."""

    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


def _humanize_ytdlp_error(message: str) -> str:
    """Map a raw yt-dlp error string to a friendly, actionable message."""
    lowered = message.lower()
    if "is not a valid url" in lowered or "unsupported url" in lowered:
        return "That does not look like a valid or supported video URL."
    if "video unavailable" in lowered or "private video" in lowered or "removed" in lowered:
        return "The video is unavailable (it may be private, deleted, or region-locked)."
    if (
        "unable to download" in lowered
        or "getaddrinfo" in lowered
        or "timed out" in lowered
        or "connection" in lowered
        or "network" in lowered
    ):
        return "Network error while downloading. Check your internet connection and try again."
    # Strip yt-dlp's noisy "ERROR: " prefix for anything we don't specifically map.
    cleaned = message.split("ERROR:", 1)[-1].strip() or message.strip()
    return f"Download failed: {cleaned}"


def download(url: str, job_id: str, cfg: Config) -> DownloadResult:
    """Ingest ``url`` (remote URL or local file) into ``data/sources/{job_id}/``.

    Downloads the video as ``video.mp4`` and extracts ``audio.wav`` (16kHz mono
    pcm_s16le). Raises :class:`DownloadError` with a human-readable message on any
    failure; raw yt-dlp / ffmpeg tracebacks never escape.
    """
    if not url or not url.strip():
        raise DownloadError("No URL or file path was provided.")

    job_dir = cfg.data_path / "sources" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio.wav"

    if _looks_like_local_path(url):
        title, duration_s = _ingest_local(Path(url), video_path)
    else:
        title, duration_s = _ingest_remote(url, job_dir, video_path)

    if not video_path.is_file():
        raise DownloadError(f"Expected video file was not created at {video_path}.")

    _extract_audio(video_path, audio_path)

    return DownloadResult(
        video_path=video_path,
        audio_path=audio_path,
        title=title,
        duration_s=duration_s,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m clipforge.download <url_or_path>", file=sys.stderr)
        return 2

    cfg = Config.load()
    job_id = uuid.uuid4().hex[:12]
    try:
        result = download(argv[0], job_id, cfg)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = asdict(result)
    payload["video_path"] = str(result.video_path)
    payload["audio_path"] = str(result.audio_path)
    payload["job_id"] = job_id
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
