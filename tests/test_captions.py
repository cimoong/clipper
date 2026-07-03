"""Tests for the caption stage (clipforge.captions).

These exercise ASS generation only — no ffmpeg, no video — so they run anywhere.
The document is parsed back into its Dialogue events to assert on timing, text,
grouping and per-word highlighting.
"""

from __future__ import annotations

import pytest

from clipforge.captions import (
    HOOK_FONT_SIZE,
    OUT_H,
    Word,
    _format_ass_time,
    _escape_ass_path,
    add_hook_overlay,
    build_ass,
)
from pathlib import PureWindowsPath


def _dialogue_lines(ass: str) -> list[str]:
    return [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]


def _parse_event(line: str) -> tuple[str, str, str, str]:
    """Return ``(start, end, style, text)`` from a Dialogue line.

    The text field is the 10th comma-separated field and may itself contain
    commas, so we split with a bounded maxsplit.
    """
    body = line[len("Dialogue:") :].strip()
    parts = body.split(",", 9)
    _layer, start, end, style, _name, _ml, _mr, _mv, _effect, text = parts
    return start, end, style, text


def test_time_formatting() -> None:
    assert _format_ass_time(0) == "0:00:00.00"
    assert _format_ass_time(2.0) == "0:00:02.00"
    assert _format_ass_time(65.5) == "0:01:05.50"
    assert _format_ass_time(3661.23) == "1:01:01.23"
    # Negative timings are clamped to zero, never emitted as garbage.
    assert _format_ass_time(-5.0) == "0:00:00.00"


def test_header_has_styles_and_playres() -> None:
    ass = build_ass([], "bold", 0.0, 10.0)
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Style: Bold," in ass
    assert "Style: Clean," in ass
    # No words -> no events.
    assert _dialogue_lines(ass) == []


def test_timestamps_are_shifted_to_clip_zero() -> None:
    # Clip covers 10..20s of the source; a word at 12.0-12.4 must land at 2.0-2.4.
    words = [Word("hello", 12.0, 12.4)]
    ass = build_ass(words, "bold", 10.0, 20.0)
    events = _dialogue_lines(ass)
    assert len(events) == 1
    start, end, style, text = _parse_event(events[0])
    assert start == "0:00:02.00"
    # A single word holds until its own end.
    assert end == "0:00:02.40"
    assert style == "Bold"


def test_all_caps() -> None:
    words = [Word("Hello", 0.0, 0.5), Word("world", 0.5, 1.0)]
    ass = build_ass(words, "clean", 0.0, 5.0)
    text = _parse_event(_dialogue_lines(ass)[0])[3]
    assert text == "HELLO WORLD"


def test_grouping_max_four_words_clean() -> None:
    # 9 words -> ceil(9/4) = 3 groups -> 3 Dialogue lines in clean style.
    words = [Word(f"w{i}", float(i), float(i) + 0.5) for i in range(9)]
    ass = build_ass(words, "clean", 0.0, 30.0)
    events = _dialogue_lines(ass)
    assert len(events) == 3
    # First line spans first-word start .. fourth-word end.
    start, end, _style, text = _parse_event(events[0])
    assert start == "0:00:00.00"
    assert end == "0:00:03.50"
    assert text == "W0 W1 W2 W3"


def test_bold_emits_one_event_per_word_with_highlight() -> None:
    words = [Word("a", 0.0, 0.4), Word("b", 0.5, 0.9), Word("c", 1.0, 1.4)]
    ass = build_ass(words, "bold", 0.0, 5.0)
    events = _dialogue_lines(ass)
    # One event per word (all in a single group of 3).
    assert len(events) == 3

    # Each event highlights exactly its own word in yellow.
    starts = [_parse_event(e)[0] for e in events]
    assert starts == ["0:00:00.00", "0:00:00.50", "0:00:01.00"]

    # A non-final word stays lit until the next word begins.
    first_end = _parse_event(events[0])[1]
    assert first_end == "0:00:00.50"

    # The highlighted token is wrapped in the yellow colour override.
    first_text = _parse_event(events[0])[3]
    assert r"{\c&H00FFFF&}A{\c&HFFFFFF&}" in first_text
    assert "B C" in first_text  # the other two words present, un-highlighted


def test_clean_has_no_highlight_color() -> None:
    words = [Word("a", 0.0, 0.4), Word("b", 0.5, 0.9)]
    ass = build_ass(words, "clean", 0.0, 5.0)
    assert r"\c&H00FFFF&" not in "\n".join(_dialogue_lines(ass))


def test_words_outside_clip_span_are_dropped() -> None:
    words = [
        Word("before", 1.0, 2.0),  # ends before clip_start=5 -> dropped
        Word("inside", 6.0, 6.5),  # kept, shifted to 1.0-1.5
        Word("after", 21.0, 22.0),  # starts after clip_end=20 -> dropped
    ]
    ass = build_ass(words, "clean", 5.0, 20.0)
    events = _dialogue_lines(ass)
    assert len(events) == 1
    start, _end, _style, text = _parse_event(events[0])
    assert text == "INSIDE"
    assert start == "0:00:01.00"


def test_partial_word_is_clamped_to_clip_span() -> None:
    # Word straddles clip_end; its end is clamped so it never exceeds duration.
    words = [Word("edge", 19.0, 25.0)]
    ass = build_ass(words, "bold", 10.0, 20.0)
    start, end, _style, _text = _parse_event(_dialogue_lines(ass)[0])
    assert start == "0:00:09.00"
    assert end == "0:00:10.00"  # clamped to clip duration (20-10)


def test_invalid_style_raises() -> None:
    with pytest.raises(ValueError):
        build_ass([], "neon", 0.0, 10.0)


def test_escape_ass_path_windows() -> None:
    escaped = _escape_ass_path(PureWindowsPath(r"C:\clips\job\clip_00.ass"))
    assert escaped == r"C\:/clips/job/clip_00.ass"


# --------------------------------------------------------------------------- #
# Hook overlay (F9b)
# --------------------------------------------------------------------------- #


def _hook_event(ass: str) -> tuple[str, str, str, str]:
    """Return the single Hook-style Dialogue event as (start, end, style, text)."""
    hooks = [_parse_event(ln) for ln in _dialogue_lines(ass) if _parse_event(ln)[2] == "Hook"]
    assert len(hooks) == 1
    return hooks[0]


def test_hook_adds_style_and_top_center_event() -> None:
    base = build_ass([Word("hi", 0.0, 0.5)], "bold", 0.0, 30.0)
    out = add_hook_overlay(base, "The $50k mistake", 30.0)

    # A Hook style is registered inside the styles section...
    assert "Style: Hook," in out
    assert "[V4+ Styles]" in out
    styles_block = out.split("[V4+ Styles]")[1].split("[Events]")[0]
    assert "Style: Hook," in styles_block
    # ...with top-center alignment (field 19 == 8) and a ~120px top margin.
    hook_style = next(ln for ln in out.splitlines() if ln.startswith("Style: Hook,"))
    fields = hook_style[len("Style:") :].split(",")
    assert fields[18].strip() == "8"  # Alignment: top-center
    assert fields[21].strip() == "120"  # MarginV from the top

    start, end, _style, text = _hook_event(out)
    assert start == "0:00:00.00"
    assert "MISTAKE" in text  # upper-cased


def test_hook_3s_mode_timing_and_fade() -> None:
    base = build_ass([], "bold", 0.0, 30.0)
    out = add_hook_overlay(base, "STOP SCROLLING NOW", 30.0, mode="3s")
    start, end, _style, text = _hook_event(out)
    assert start == "0:00:00.00"
    assert end == "0:00:03.00"
    # 0.2s fade-out at the end.
    assert r"{\fad(0,200)}" in text


def test_hook_full_mode_spans_whole_clip_without_fade() -> None:
    base = build_ass([], "bold", 0.0, 42.5)
    out = add_hook_overlay(base, "WATCH THIS", 42.5, mode="full")
    _start, end, _style, text = _hook_event(out)
    assert end == "0:00:42.50"
    assert r"\fad" not in text


def test_hook_3s_capped_to_short_clip_duration() -> None:
    base = build_ass([], "bold", 0.0, 2.0)
    out = add_hook_overlay(base, "SHORT", 2.0, mode="3s")
    _start, end, _style, _text = _hook_event(out)
    assert end == "0:00:02.00"


def test_hook_wraps_long_caption_to_two_lines() -> None:
    base = build_ass([], "bold", 0.0, 30.0)
    caption = "THE FIFTY THOUSAND DOLLAR MISTAKE NOBODY TALKS ABOUT"
    out = add_hook_overlay(base, caption, 30.0)
    text = _hook_event(out)[3]
    # Exactly one line break -> at most two lines.
    assert text.count(r"\N") == 1


def test_hook_short_caption_stays_one_line() -> None:
    base = build_ass([], "bold", 0.0, 30.0)
    out = add_hook_overlay(base, "ONE LINE", 30.0)
    text = _hook_event(out)[3]
    assert r"\N" not in text


def test_hook_does_not_overlap_karaoke_area() -> None:
    # Karaoke captions are bottom-anchored (alignment 2); the hook is top-anchored
    # (alignment 8). Their margins therefore measure from opposite edges and the
    # two regions cannot collide on a 1920-tall frame.
    base = build_ass([Word("spoken", 1.0, 1.5)], "bold", 0.0, 30.0)
    out = add_hook_overlay(base, "HEADLINE", 30.0)
    lines = out.splitlines()
    hook_style = next(ln for ln in lines if ln.startswith("Style: Hook,"))
    bold_style = next(ln for ln in lines if ln.startswith("Style: Bold,"))
    hook_margin = int(hook_style[len("Style:") :].split(",")[21])
    bold_margin = int(bold_style[len("Style:") :].split(",")[21])
    # Hook occupies the top band; karaoke the bottom band. No vertical overlap.
    assert hook_margin + HOOK_FONT_SIZE < OUT_H - bold_margin


def test_hook_empty_caption_is_noop() -> None:
    base = build_ass([], "bold", 0.0, 30.0)
    assert add_hook_overlay(base, "   ", 30.0) == base


def test_hook_invalid_mode_raises() -> None:
    base = build_ass([], "bold", 0.0, 30.0)
    with pytest.raises(ValueError):
        add_hook_overlay(base, "HI", 30.0, mode="blink")
