"""Stage 6 of the ClipForge pipeline: word-level burned-in captions.

Given the transcript words that fall inside a clip's [start, end] span, build an
ASS subtitle file with short ALL-CAPS lines (max 4 words) centered low on the
frame, then burn it onto the clip with ffmpeg.

Two styles are supported:

* ``bold``  — white Montserrat-like bold with a thick black outline; the word
  being spoken pops to yellow (word-by-word highlight via inline colour
  overrides, one Dialogue event per word so the highlight timing matches the
  transcript exactly).
* ``clean`` — white with a thin outline and no highlight; one Dialogue event per
  line spanning the whole group.

Geometry matches the reframe stage's 1080x1920 portrait output. Lines sit at
~72% of the frame height (``MarginV`` from the bottom), clear of the speaker's
face in the upper two thirds.

This module deliberately does NOT import :mod:`clipforge.transcribe` (which pulls
in ``faster_whisper`` at import time). It defines its own light-weight
:class:`Word` with the same ``word``/``start``/``end`` fields, so it can be used
and unit-tested without any heavy ASR dependency.

Run standalone against an already-analyzed job (builds the .ass and, if a source
clip is present, burns it):

    python -m clipforge.captions <job_id> <clip_index>
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# Portrait output geometry (must match the reframe stage).
OUT_W = 1080
OUT_H = 1920

# Caption layout tunables.
MAX_WORDS_PER_LINE = 4
# Vertical anchor: lines sit at ~72% of the frame height. With bottom-centered
# alignment, MarginV is measured up from the bottom edge.
CAPTION_HEIGHT_FRAC = 0.72
MARGIN_V = round(OUT_H * (1.0 - CAPTION_HEIGHT_FRAC))  # ~538 px for 1920 tall
MARGIN_H = 60  # left/right safe margin

# ASS inline colours (\c) are &HBBGGRR&.
_YELLOW = r"&H00FFFF&"  # spoken-word highlight (yellow)
_WHITE = r"&HFFFFFF&"  # reset to the normal fill

_VALID_STYLES = ("bold", "clean")

# --- Hook overlay (F9b) ---------------------------------------------------- #
# The hook caption is a scroll-stopping headline burned at the TOP of the frame,
# clear of the speaker's face and of the karaoke captions (which sit low, at
# ``MARGIN_V`` up from the bottom). Alignment 8 = top-center, so ``HOOK_MARGIN_V``
# is measured DOWN from the top edge.
HOOK_MARGIN_V = 120  # ~120 px from the top on a 1920-tall frame
HOOK_FONT_SIZE = 92  # slightly larger than the 84 px karaoke captions
HOOK_OUTLINE = 6  # heavy black outline so it reads over any background
HOOK_MAX_LINES = 2
# Rough character budget for one line at HOOK_FONT_SIZE across a 1080 px frame;
# above this the caption is balanced across two lines.
HOOK_LINE_CHARS = 22
_VALID_HOOK_MODES = ("3s", "full")


class CaptionError(Exception):
    """Raised for any caption failure, with a human-readable message.

    Raw ffmpeg tracebacks never escape; ffmpeg failures are re-raised as this
    exception with the last few lines of stderr attached for diagnosis.
    """


@dataclass(frozen=True)
class Word:
    """A single spoken word with timing in seconds.

    Structurally identical to :class:`clipforge.transcribe.Word`; duplicated here
    only to avoid importing the heavy ASR module.
    """

    word: str
    start: float
    end: float


# --------------------------------------------------------------------------- #
# ASS generation
# --------------------------------------------------------------------------- #


def _format_ass_time(seconds: float) -> str:
    """Format ``seconds`` as an ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    hours, total_cs = divmod(total_cs, 360000)
    minutes, total_cs = divmod(total_cs, 6000)
    secs, centis = divmod(total_cs, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _prepare_words(words: list[Word], clip_start: float, clip_end: float) -> list[Word]:
    """Filter ``words`` to the clip span and shift timings so the clip starts at 0.

    A word is kept if it overlaps ``[clip_start, clip_end]`` at all; its timing is
    clamped to the clip span and shifted by ``clip_start``. Tokens are upper-cased
    (ALL-CAPS style) and surrounding whitespace stripped; empty tokens are
    dropped. The returned words are ordered by start time.
    """
    duration = clip_end - clip_start
    prepared: list[Word] = []
    for w in words:
        # Skip words entirely outside the clip span.
        if w.end <= clip_start or w.start >= clip_end:
            continue
        token = w.word.strip().upper()
        if not token:
            continue
        start = max(0.0, w.start - clip_start)
        end = min(duration, w.end - clip_start)
        if end <= start:
            continue
        prepared.append(Word(word=token, start=start, end=end))
    prepared.sort(key=lambda x: x.start)
    return prepared


def _group_words(words: list[Word], size: int) -> list[list[Word]]:
    """Split ``words`` into consecutive chunks of at most ``size`` words."""
    return [words[i : i + size] for i in range(0, len(words), size)]


def _highlight_text(group: list[Word], active_index: int) -> str:
    """Render a group's text with the ``active_index`` word popped to yellow."""
    parts: list[str] = []
    for j, w in enumerate(group):
        if j == active_index:
            parts.append(rf"{{\c{_YELLOW}}}{w.word}{{\c{_WHITE}}}")
        else:
            parts.append(w.word)
    return " ".join(parts)


def _dialogue(start: float, end: float, style: str, text: str) -> str:
    """Build one ASS ``Dialogue`` event line."""
    return f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},{style},,0,0,0,,{text}"


def _header() -> list[str]:
    """The ASS script header + both style definitions (bold and clean)."""
    fmt = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding"
    )
    # BorderStyle 1 = outline+shadow. Alignment 2 = bottom-center. Colours are
    # &HAABBGGRR (AA=00 -> fully opaque). Bold -1 = true.
    bold_style = (
        f"Style: Bold,Montserrat,84,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,4,1,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1"
    )
    clean_style = (
        f"Style: Clean,Montserrat,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,1,0,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1"
    )
    return [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {OUT_W}",
        f"PlayResY: {OUT_H}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        fmt,
        bold_style,
        clean_style,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]


def build_ass(words: list[Word], style: str, clip_start: float, clip_end: float) -> str:
    """Build the ASS subtitle document for one clip.

    ``words`` are transcript words on the *original* video timeline; they are
    filtered to ``[clip_start, clip_end]`` and shifted so the clip starts at 0.
    Words are grouped into ALL-CAPS lines of at most :data:`MAX_WORDS_PER_LINE`.

    ``style`` is ``"bold"`` (per-word yellow highlight) or ``"clean"`` (no
    highlight). Returns the full ``.ass`` file contents. Raises ``ValueError`` for
    an unknown style.
    """
    style_key = style.lower()
    if style_key not in _VALID_STYLES:
        raise ValueError(f"Unknown caption style {style!r}; expected one of {_VALID_STYLES}.")

    prepared = _prepare_words(words, clip_start, clip_end)
    groups = _group_words(prepared, MAX_WORDS_PER_LINE)

    lines = _header()
    for group in groups:
        if style_key == "clean":
            text = " ".join(w.word for w in group)
            lines.append(_dialogue(group[0].start, group[-1].end, "Clean", text))
            continue

        # Bold: one Dialogue per word so the highlight lands exactly on the word's
        # start. Each word stays lit until the next word begins (the last word of a
        # group holds until its own end), giving a continuous word-pop within the
        # line.
        for j, w in enumerate(group):
            start = w.start
            end = group[j + 1].start if j + 1 < len(group) else w.end
            if end <= start:  # guard against overlapping/zero-length timings
                end = w.end
            if end <= start:
                continue
            lines.append(_dialogue(start, end, "Bold", _highlight_text(group, j)))

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Hook overlay (F9b)
# --------------------------------------------------------------------------- #


def _wrap_hook_text(text: str) -> str:
    """Wrap ``text`` to at most :data:`HOOK_MAX_LINES` lines, joined by ``\\N``.

    A short caption stays on a single line. A longer one is split at the word
    boundary that best balances the two lines (minimising the longest line), so
    the headline never spills past two lines regardless of length.
    """
    words = text.split()
    if len(words) <= 1 or len(text) <= HOOK_LINE_CHARS:
        return text

    # Choose the split point that minimises the longer of the two lines; on ties
    # prefer the more even split (smaller absolute length difference).
    best_split = 1
    best_key = (len(text), len(text))
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        key = (max(len(left), len(right)), abs(len(left) - len(right)))
        if key < best_key:
            best_key = key
            best_split = i

    return " ".join(words[:best_split]) + r"\N" + " ".join(words[best_split:])


def _hook_style() -> str:
    """The ASS ``Hook`` style: bold white, heavy black outline, top-center."""
    # Alignment 8 = top-center; MarginV is measured from the top for that anchor.
    return (
        f"Style: Hook,Montserrat,{HOOK_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,"
        f"&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{HOOK_OUTLINE},2,8,"
        f"{MARGIN_H},{MARGIN_H},{HOOK_MARGIN_V},1"
    )


def add_hook_overlay(
    ass_content: str,
    hook_caption: str,
    duration_s: float,
    mode: str = "3s",
) -> str:
    """Add a top-center hook-caption overlay event to an existing ASS document.

    ``ass_content`` is a document as produced by :func:`build_ass`. A bold,
    white, heavily outlined headline is added at the top of the frame (clear of
    the karaoke captions, which sit low). The text is upper-cased and wrapped to
    at most two lines.

    ``mode``:

    * ``"3s"``   — visible from 0.0 s to ``min(3.0, duration_s)`` with a 0.2 s
      fade-out at the end.
    * ``"full"`` — visible for the entire clip (0.0 s .. ``duration_s``).

    Returns the augmented document. An empty ``hook_caption`` returns the input
    unchanged. Raises ``ValueError`` for an unknown ``mode``.
    """
    mode_key = mode.lower()
    if mode_key not in _VALID_HOOK_MODES:
        raise ValueError(f"Unknown hook mode {mode!r}; expected one of {_VALID_HOOK_MODES}.")

    caption = hook_caption.strip().upper()
    if not caption:
        return ass_content
    if duration_s <= 0:
        return ass_content

    end = duration_s if mode_key == "full" else min(3.0, duration_s)
    # A 0.2 s fade-out on the 3 s mode; "full" holds without fading.
    override = r"{\fad(0,200)}" if mode_key == "3s" else ""
    text = override + _wrap_hook_text(caption)

    # Layer 1 keeps the hook above the karaoke captions if they ever overlap.
    event = f"Dialogue: 1,{_format_ass_time(0.0)},{_format_ass_time(end)},Hook,,0,0,0,,{text}"

    lines = ass_content.splitlines()

    # Register the Hook style (once) inside the [V4+ Styles] section, right after
    # its Format line, so a document reused across clips isn't given duplicates.
    if "Style: Hook," not in ass_content:
        for i, ln in enumerate(lines):
            if ln.startswith("Format:") and _within_styles(lines, i):
                lines.insert(i + 1, _hook_style())
                break

    lines.append(event)
    return "\n".join(lines) + "\n"


def _within_styles(lines: list[str], index: int) -> bool:
    """True if ``lines[index]`` falls under the ``[V4+ Styles]`` section header."""
    for ln in reversed(lines[:index]):
        stripped = ln.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped == "[V4+ Styles]"
    return False


# --------------------------------------------------------------------------- #
# Burning
# --------------------------------------------------------------------------- #


def _escape_ass_path(path: Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg ``ass=`` filter arg.

    On Windows the drive colon and backslashes confuse ffmpeg's filtergraph
    parser, so backslashes become forward slashes and every colon is escaped
    (``C:\\clips\\a.ass`` -> ``C\\:/clips/a.ass``).
    """
    return str(path).replace("\\", "/").replace(":", r"\:")


# Preferred encode: NVENC (GTX 1060), with a CPU fallback for environments where
# h264_nvenc is listed but cannot actually be opened.
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
_X264_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
_nvenc_ok: bool | None = None


def _encoder_args() -> list[str]:
    """Return NVENC encode args if usable, else the libx264 (CPU) fallback."""
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
            raise CaptionError(
                "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
            ) from exc
        if not _nvenc_ok:
            logger.warning("h264_nvenc unavailable; burning captions with libx264 (CPU).")
    return _NVENC_ARGS if _nvenc_ok else _X264_ARGS


def burn(clip_path: Path, ass_path: Path, out_path: Path) -> None:
    """Burn ``ass_path`` onto ``clip_path`` and write ``out_path``.

    Video is re-encoded (NVENC when available) with the ASS overlay; the source
    audio is copied through unchanged. Raises :class:`CaptionError` on failure.
    """
    clip_path = Path(clip_path).resolve()
    ass_path = Path(ass_path).resolve()
    out_path = Path(out_path).resolve()
    if not clip_path.is_file():
        raise CaptionError(f"Source clip not found: {clip_path}")
    if not ass_path.is_file():
        raise CaptionError(f"Subtitle file not found: {ass_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The ass filter's ``filename`` option is itself parsed by ffmpeg's filtergraph
    # tokenizer, so a Windows drive colon (``C:``) has to survive two unescaping
    # passes — fragile across ffmpeg versions. Sidestep it entirely by running
    # ffmpeg from the subtitle's own directory and referencing it by its bare
    # (colon- and backslash-free) filename; the clip and output stay absolute.
    vf = f"ass={_escape_ass_path(Path(ass_path.name))}"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        vf,
        *_encoder_args(),
        "-c:a",
        "copy",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ass_path.parent))
    except FileNotFoundError as exc:
        raise CaptionError(
            "ffmpeg was not found on PATH. Install FFmpeg and ensure it is on your PATH."
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        detail = "\n".join(tail) if tail else "(no ffmpeg output)"
        raise CaptionError(f"ffmpeg failed to burn captions onto {out_path.name}:\n{detail}")
    if not out_path.is_file():
        raise CaptionError(f"ffmpeg reported success but produced no file at {out_path}.")


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #


def _load_candidate(job_id: str, clip_index: int, cfg: Config) -> dict[str, Any]:
    """Load candidate ``clip_index`` from a job's ``candidates.json``."""
    path = cfg.data_path / "sources" / job_id / "candidates.json"
    if not path.is_file():
        raise CaptionError(f"Candidates not found: {path}")
    try:
        candidates = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptionError(f"Could not read candidates {path}: {exc}") from exc
    if not isinstance(candidates, list) or not (0 <= clip_index < len(candidates)):
        raise CaptionError(
            f"Clip index {clip_index} out of range (job has "
            f"{len(candidates) if isinstance(candidates, list) else 0} candidate(s))."
        )
    cand = candidates[clip_index]
    if not isinstance(cand, dict) or "start" not in cand or "end" not in cand:
        raise CaptionError(f"Candidate {clip_index} is missing start/end: {cand!r}")
    return cand


def _load_words(job_id: str, cfg: Config) -> list[Word]:
    """Load all transcript words (across every segment) for ``job_id``."""
    path = cfg.data_path / "sources" / job_id / "transcript.json"
    if not path.is_file():
        raise CaptionError(f"Transcript not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptionError(f"Could not read transcript {path}: {exc}") from exc

    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        raise CaptionError(f"Transcript has no segments: {path}")

    words: list[Word] = []
    for seg in segments:
        for w in seg.get("words", []) if isinstance(seg, dict) else []:
            try:
                words.append(
                    Word(word=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
                )
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed words rather than fail the whole clip
    return words


def _source_clip(job_id: str, clip_index: int, cfg: Config) -> Path | None:
    """Find the best available source clip to caption, or ``None``.

    Prefers the reframed vertical clip (``clip_NN.mp4``) and falls back to the raw
    cut clip (``clip_NN_raw.mp4``).
    """
    out_dir = cfg.data_path / "outputs" / job_id
    reframed = out_dir / f"clip_{clip_index:02d}.mp4"
    raw = out_dir / f"clip_{clip_index:02d}_raw.mp4"
    if reframed.is_file():
        return reframed
    if raw.is_file():
        return raw
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m clipforge.captions <job_id> <clip_index>", file=sys.stderr)
        return 2

    job_id = argv[0]
    try:
        clip_index = int(argv[1])
    except ValueError:
        print(f"error: clip_index must be an integer, got {argv[1]!r}", file=sys.stderr)
        return 2

    cfg = Config.load()
    try:
        cand = _load_candidate(job_id, clip_index, cfg)
        words = _load_words(job_id, cfg)
        clip_start = float(cand["start"])
        clip_end = float(cand["end"])
        style = "bold"

        ass_text = build_ass(words, style, clip_start, clip_end)
        out_dir = cfg.data_path / "outputs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ass_path = out_dir / f"clip_{clip_index:02d}.ass"
        ass_path.write_text(ass_text, encoding="utf-8")
        logger.info("Wrote subtitles -> %s", ass_path)

        source = _source_clip(job_id, clip_index, cfg)
        if source is None:
            logger.warning(
                "No source clip found for clip %d; wrote the .ass only. "
                "Run the cut/reframe stages first to burn it.",
                clip_index,
            )
            print(ass_path)
            return 0

        captioned = out_dir / f"clip_{clip_index:02d}_captioned.mp4"
        burn(source, ass_path, captioned)
        logger.info("Burned captions -> %s", captioned)
        print(captioned)
    except (CaptionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
