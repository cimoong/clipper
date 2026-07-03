"""Stage 5 of the ClipForge pipeline: reframe a 16:9 clip to vertical 9:16.

Converts a landscape clip into a 1080x1920 portrait clip using one of two
strategies, chosen automatically from a cheap face-detection analysis pass:

* **track mode** — when faces are reliably present, a virtual camera follows the
  (largest) face. The crop path is linearly interpolated between sampled face
  positions, smoothed with an exponential moving average plus a dead-zone so the
  camera ignores tiny jitter, and clamped to stay inside the frame. Frames are
  streamed one at a time through OpenCV, cropped/resized, and piped as raw video
  into an ``h264_nvenc`` ffmpeg encoder; the original audio is muxed back in.

* **blur-background mode** — the fallback when faces appear in fewer than 20% of
  sampled frames. The full 16:9 frame is centered on top of a blurred, zoomed
  copy of itself filling 1080x1920. This mode is done entirely with a single
  ffmpeg ``filter_complex`` (no per-frame OpenCV loop).

Peak RAM stays low: both the analysis and render passes stream frame-by-frame
and never hold the whole clip in memory.

Run standalone:

    python -m clipforge.reframe <input.mp4> <output.mp4>
"""

from __future__ import annotations

import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

# Face detection model.
#
# The classic ``mediapipe.solutions.face_detection`` API is not present in every
# mediapipe build (recent wheels ship only the newer *Tasks* API), so we use the
# Tasks ``FaceDetector`` with the BlazeFace short-range model. The model file is
# downloaded once and cached under ``<data_dir>/models``.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_NAME = "blaze_face_short_range.tflite"
_MIN_CONFIDENCE = 0.5

# Output geometry (portrait 9:16).
OUT_W = 1080
OUT_H = 1920

# Analysis / smoothing tunables.
ANALYSIS_FPS = 5.0
EMA_ALPHA = 0.08
DEAD_ZONE_FRAC = 0.03  # ignore camera moves smaller than 3% of frame width
FACE_COVERAGE_MIN = 0.20  # below this fraction of detected samples -> blur mode

# Preferred encode: NVENC (GTX 1060, per project ffmpeg pipeline rules).
_NVENC_ARGS = [
    "-c:v",
    "h264_nvenc",
    "-preset",
    "p5",
    "-rc",
    "vbr",
    "-cq",
    "23",
    "-pix_fmt",
    "yuv420p",
]
# CPU fallback used only when NVENC cannot be opened (e.g. driver too old for the
# installed ffmpeg build). Visually equivalent output, just slower.
_X264_ARGS = [
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
]

# Resolved once per process: True if h264_nvenc can actually be opened here.
_nvenc_ok: bool | None = None


def _encoder_args() -> list[str]:
    """Return NVENC encode args if usable, else the CPU (libx264) fallback.

    Probes NVENC exactly once with a tiny throwaway encode; some environments
    list ``h264_nvenc`` yet fail to open it (driver older than the ffmpeg build
    requires).
    """
    global _nvenc_ok
    if _nvenc_ok is None:
        probe = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=256x256:d=0.1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        try:
            _nvenc_ok = subprocess.run(probe, capture_output=True).returncode == 0
        except FileNotFoundError as exc:
            raise ReframeError(
                "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
            ) from exc
        if not _nvenc_ok:
            logger.warning(
                "h264_nvenc unavailable in this environment; falling back to libx264 (CPU)."
            )
    return _NVENC_ARGS if _nvenc_ok else _X264_ARGS


class ReframeError(Exception):
    """Raised for any reframing failure, with a human-readable message.

    Raw ffmpeg / OpenCV tracebacks never escape; failures are re-raised as this
    exception with the relevant detail attached for diagnosis.
    """


def _probe(cap: "cv2.VideoCapture") -> tuple[int, int, float]:
    """Return ``(width, height, fps)`` for an opened capture."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        raise ReframeError("Could not read frame dimensions from the input clip.")
    if not fps or fps <= 0 or fps != fps:  # guard against 0 / NaN
        fps = 30.0
    return width, height, fps


def _ensure_model(cfg: Config) -> Path:
    """Return the local path to the face-detector model, downloading if absent."""
    model_dir = cfg.data_path / "models"
    model_path = model_dir / _MODEL_NAME
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face-detector model -> %s", model_path)
    tmp = model_path.with_suffix(".tflite.part")
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(model_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ReframeError(
            f"Could not download the face-detector model from {_MODEL_URL}: {exc}"
        ) from exc
    return model_path


def _make_detector(cfg: Config) -> "mp.tasks.vision.FaceDetector":
    """Build a Tasks FaceDetector in IMAGE mode using the cached model."""
    model_path = _ensure_model(cfg)
    vision = mp.tasks.vision
    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=_MIN_CONFIDENCE,
    )
    return vision.FaceDetector.create_from_options(options)


def _largest_face_center_x(detections: object) -> float | None:
    """Return the pixel x-center of the largest detected face, or ``None``.

    ``detections`` is the ``.detections`` list from a Tasks FaceDetector result;
    each ``bounding_box`` is already in pixel coordinates.
    """
    best_area = -1.0
    best_cx: float | None = None
    for det in detections:  # type: ignore[assignment]
        box = det.bounding_box
        area = float(max(box.width, 0)) * float(max(box.height, 0))
        if area > best_area:
            best_area = area
            best_cx = box.origin_x + box.width / 2.0
    return best_cx


def _analyze(
    clip_path: Path, cfg: Config
) -> tuple[int, int, float, list[int], list[float], int, int]:
    """Sample the clip at ~5 fps and detect the largest face per sampled frame.

    Returns ``(width, height, fps, sample_indices, sample_centers, total_frames,
    sampled)`` where ``sample_indices``/``sample_centers`` cover only frames in
    which a face was found (parallel lists), ``total_frames`` is the exact
    decoded count, and ``sampled`` is how many frames were run through detection.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ReframeError(f"Could not open clip for analysis: {clip_path}")

    try:
        width, height, fps = _probe(cap)
        sample_every = max(1, round(fps / ANALYSIS_FPS))

        sample_indices: list[int] = []
        sample_centers: list[float] = []
        sampled = 0
        detected = 0
        idx = 0

        detector = _make_detector(cfg)
        try:
            while True:
                if not cap.grab():
                    break
                if idx % sample_every == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    sampled += 1
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(image)
                    if result.detections:
                        cx = _largest_face_center_x(result.detections)
                        if cx is not None:
                            detected += 1
                            sample_indices.append(idx)
                            sample_centers.append(cx)
                idx += 1
        finally:
            detector.close()
    finally:
        cap.release()

    total_frames = idx
    if total_frames == 0:
        raise ReframeError(f"Clip contained no decodable frames: {clip_path}")

    coverage = (detected / sampled) if sampled else 0.0
    logger.info(
        "Analysis: %dx%d @ %.3f fps, %d sampled, faces in %d (%.0f%%)",
        width,
        height,
        fps,
        sampled,
        detected,
        coverage * 100,
    )
    return width, height, fps, sample_indices, sample_centers, total_frames, sampled


def _build_camera_path(
    sample_indices: list[int],
    sample_centers: list[float],
    total_frames: int,
    width: int,
    crop_w: int,
) -> np.ndarray:
    """Interpolate + smooth the per-frame crop-center x for track mode.

    Linear interpolation fills every frame between face samples; an EMA
    (``alpha=0.08``) with a 3%-of-width dead-zone removes jitter; the result is
    clamped so the crop window never leaves the frame.
    """
    frames = np.arange(total_frames, dtype=np.float64)
    xs = np.asarray(sample_indices, dtype=np.float64)
    ys = np.asarray(sample_centers, dtype=np.float64)
    # np.interp holds the endpoint value before the first / after the last sample.
    raw = np.interp(frames, xs, ys)

    dead_zone = DEAD_ZONE_FRAC * width
    smoothed = np.empty_like(raw)
    s = float(raw[0])
    for i, target in enumerate(raw):
        if abs(target - s) > dead_zone:
            s += EMA_ALPHA * (target - s)
        smoothed[i] = s

    half = crop_w / 2.0
    return np.clip(smoothed, half, width - half)


def _open_render_ffmpeg(out_path: Path, fps: float) -> subprocess.Popen[bytes]:
    """Start an ffmpeg process reading raw bgr24 frames from stdin.

    Video is encoded with NVENC; audio (if any) is copied from the original clip
    as a second input and muxed back in.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        # input 0: raw frames from our stdin pipe
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{OUT_W}x{OUT_H}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-map",
        "0:v:0",
        *_encoder_args(),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise ReframeError(
            "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
        ) from exc


def _mux_audio(video_only: Path, audio_src: Path, out_path: Path) -> None:
    """Mux the audio track of ``audio_src`` onto ``video_only`` -> ``out_path``.

    Audio is optional: a clip with no audio stream still succeeds.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_only),
        "-i",
        str(audio_src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise ReframeError("ffmpeg failed to mux audio:\n" + "\n".join(tail))


def _render_track(
    clip_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    centers: np.ndarray,
    crop_w: int,
) -> None:
    """Stream every frame, crop the moving window, and encode to 1080x1920.

    Frames go straight into ffmpeg's stdin; the source audio is muxed in a
    second, cheap remux pass so we never buffer decoded audio in Python.
    """
    tmp_video = out_path.with_name(out_path.stem + "_noaudio.mp4")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ReframeError(f"Could not open clip for render: {clip_path}")

    proc = _open_render_ffmpeg(tmp_video, fps)
    assert proc.stdin is not None

    n = len(centers)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            center = float(centers[min(idx, n - 1)])
            x0 = int(round(center - crop_w / 2.0))
            x0 = max(0, min(x0, width - crop_w))
            window = frame[:, x0 : x0 + crop_w]
            out_frame = cv2.resize(window, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            try:
                proc.stdin.write(out_frame.tobytes())
            except BrokenPipeError:
                break  # ffmpeg died; error surfaced below
            idx += 1
    finally:
        cap.release()
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()

    if ret != 0:
        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise ReframeError("NVENC render failed:\n" + "\n".join(tail))
    if idx == 0:
        raise ReframeError("No frames were rendered from the clip.")

    try:
        _mux_audio(tmp_video, clip_path, out_path)
    finally:
        tmp_video.unlink(missing_ok=True)


def _render_blur(clip_path: Path, out_path: Path) -> None:
    """Fallback render: 16:9 frame centered over a blurred, zoomed background.

    Implemented entirely with an ffmpeg ``filter_complex`` — no OpenCV loop.
    """
    filter_complex = (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},gblur=sigma=20[bgb];"
        f"[fg]scale={OUT_W}:-2[fgs];"
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a:0?",
        *_encoder_args(),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ReframeError(
            "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise ReframeError("Blur-background render failed:\n" + "\n".join(tail))


def reframe(clip_path: Path, out_path: Path, cfg: Config) -> None:
    """Reframe a 16:9 ``clip_path`` into a 1080x1920 clip at ``out_path``.

    Chooses face-tracking or blur-background mode automatically. ``cfg.data_path``
    is used to cache the face-detector model. Raises :class:`ReframeError` on
    failure.
    """
    clip_path = Path(clip_path)
    out_path = Path(out_path)
    if not clip_path.is_file():
        raise ReframeError(f"Input clip not found: {clip_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height, fps, sample_idx, sample_cx, total, sampled = _analyze(clip_path, cfg)

    # Crop a full-height, 9:16-wide window (clamped to the frame width).
    crop_w = min(width, int(round(height * OUT_W / OUT_H)))

    coverage = len(sample_idx) / sampled if sampled else 0.0

    if coverage < FACE_COVERAGE_MIN:
        logger.info(
            "Face coverage %.0f%% < %.0f%% -> blur-background mode",
            coverage * 100,
            FACE_COVERAGE_MIN * 100,
        )
        _render_blur(clip_path, out_path)
    else:
        logger.info("Face coverage %.0f%% -> face-tracking mode", coverage * 100)
        centers = _build_camera_path(sample_idx, sample_cx, total, width, crop_w)
        _render_track(clip_path, out_path, width, height, fps, centers, crop_w)

    logger.info("Reframed -> %s", out_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m clipforge.reframe <input.mp4> <output.mp4>", file=sys.stderr)
        return 2

    cfg = Config.load()
    try:
        reframe(Path(argv[0]), Path(argv[1]), cfg)
    except ReframeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
