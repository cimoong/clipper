"""Tests for the analysis stage (clipforge.analyze).

These cover ONLY the deterministic validation / snap / dedupe logic. The LLM is
never called: a fake transcript is written to disk and a canned response is
injected via the ``llm`` parameter of ``analyze`` (or the helpers are exercised
directly). No network and no API key required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.analyze import (
    AnalyzeError,
    Candidate,
    _dedupe,
    _render_segments,
    _validate,
    analyze,
)
from clipforge.config import Config


def _segments() -> list[dict[str, float | str]]:
    """A fake transcript: 10 sentences, 10 seconds each, 0..100s."""
    return [
        {"start": float(i * 10), "end": float(i * 10 + 10), "text": f"Sentence number {i}."}
        for i in range(10)
    ]


def _cfg(tmp_path: Path) -> Config:
    return Config(data_dir=str(tmp_path / "data"), num_clips=3, clip_min_s=25, clip_max_s=75)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def test_render_segments_format() -> None:
    rendered = _render_segments(_segments()[:2])
    assert rendered.splitlines()[0] == "[0.0 - 10.0] Sentence number 0."
    assert rendered.splitlines()[1] == "[10.0 - 20.0] Sentence number 1."


# --------------------------------------------------------------------------- #
# snapping + validation
# --------------------------------------------------------------------------- #


def test_snap_to_sentence_boundaries() -> None:
    segs = _segments()
    # Start/end land mid-sentence; expect snapping to segment boundaries.
    raw = [{"start": 12.3, "end": 47.8, "score": 80}]
    out = _validate(raw, segs, _cfg(Path("/x")))
    assert len(out) == 1
    # 12.3 -> nearest start is 10.0; 47.8 -> nearest end is 50.0.
    assert out[0].start == 10.0
    assert out[0].end == 50.0


def test_drop_when_duration_below_min() -> None:
    segs = _segments()
    # Snaps to 0..20 = 20s, below clip_min_s (25).
    raw = [{"start": 0.0, "end": 18.0, "score": 90}]
    assert _validate(raw, segs, _cfg(Path("/x"))) == []


def test_drop_when_duration_above_max() -> None:
    segs = _segments()
    # Snaps to 0..100 = 100s, above clip_max_s (75).
    raw = [{"start": 0.0, "end": 100.0, "score": 90}]
    assert _validate(raw, segs, _cfg(Path("/x"))) == []


def test_drop_when_outside_transcript_range() -> None:
    segs = _segments()
    raw = [{"start": 200.0, "end": 260.0, "score": 90}]
    assert _validate(raw, segs, _cfg(Path("/x"))) == []


def test_drop_when_end_before_start() -> None:
    segs = _segments()
    raw = [{"start": 60.0, "end": 30.0, "score": 90}]
    assert _validate(raw, segs, _cfg(Path("/x"))) == []


def test_hook_caption_uppercased_and_truncated() -> None:
    segs = _segments()
    raw = [
        {
            "start": 0.0,
            "end": 30.0,
            "score": 70,
            "title": "t" * 100,
            "hook_caption": "the fifty thousand dollar mistake nobody ever talks about at all",
        }
    ]
    out = _validate(raw, segs, _cfg(Path("/x")))
    assert len(out) == 1
    assert len(out[0].title) == 70
    assert out[0].hook_caption == out[0].hook_caption.upper()
    assert len(out[0].hook_caption) <= 60


def test_sub_scores_default_to_zero_when_missing() -> None:
    segs = _segments()
    raw = [{"start": 0.0, "end": 30.0, "score": 70}]
    out = _validate(raw, segs, _cfg(Path("/x")))
    assert out[0].sub_scores.hook == 0
    assert out[0].sub_scores.payoff == 0


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #


def _cand(start: float, end: float, score: int) -> Candidate:
    from clipforge.analyze import SubScores

    return Candidate(
        start=start,
        end=end,
        score=score,
        title="t",
        hook="h",
        hook_caption="HC",
        reason="r",
        sub_scores=SubScores(0, 0, 0, 0, 0),
    )


def test_dedupe_keeps_higher_score_on_overlap() -> None:
    a = _cand(0.0, 40.0, 90)
    b = _cand(5.0, 45.0, 70)  # ~87% overlap with a -> dropped
    out = _dedupe([b, a], num_clips=5)
    assert out == [a]


def test_dedupe_keeps_non_overlapping() -> None:
    a = _cand(0.0, 30.0, 90)
    b = _cand(40.0, 70.0, 80)
    out = _dedupe([a, b], num_clips=5)
    assert len(out) == 2
    assert out[0] == a and out[1] == b  # sorted by score desc


def test_dedupe_sorts_and_truncates_to_num_clips() -> None:
    cands = [
        _cand(0.0, 30.0, 50),
        _cand(40.0, 70.0, 90),
        _cand(80.0, 110.0, 70),
        _cand(120.0, 150.0, 60),
    ]
    out = _dedupe(cands, num_clips=2)
    assert [c.score for c in out] == [90, 70]


# --------------------------------------------------------------------------- #
# end-to-end with an injected fake LLM
# --------------------------------------------------------------------------- #


def _write_transcript(cfg: Config, job_id: str) -> None:
    path = cfg.data_path / "sources" / job_id / "transcript.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"language": "en", "duration_s": 100.0, "segments": _segments()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_end_to_end_with_fake_llm(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_transcript(cfg, "job1")

    response = json.dumps(
        {
            "candidates": [
                {
                    "start": 2.0,
                    "end": 33.0,
                    "score": 88,
                    "title": "Good one",
                    "hook": "Here is the hook",
                    "hook_caption": "the caption",
                    "reason": "why",
                    "sub_scores": {
                        "hook": 9,
                        "emotion": 8,
                        "value": 7,
                        "coherence": 8,
                        "payoff": 9,
                    },
                },
                # near-duplicate of the first, lower score -> deduped away
                {
                    "start": 4.0,
                    "end": 31.0,
                    "score": 60,
                    "title": "Dup",
                    "hook": "x",
                    "hook_caption": "y",
                    "reason": "z",
                    "sub_scores": {},
                },
                # too short after snapping -> dropped
                {
                    "start": 0.0,
                    "end": 12.0,
                    "score": 95,
                    "title": "Too short",
                    "hook": "x",
                    "hook_caption": "y",
                    "reason": "z",
                    "sub_scores": {},
                },
                {
                    "start": 40.0,
                    "end": 71.0,
                    "score": 75,
                    "title": "Second",
                    "hook": "x",
                    "hook_caption": "y",
                    "reason": "z",
                    "sub_scores": {},
                },
            ]
        }
    )

    out = analyze("job1", cfg, llm=lambda _prompt: response)

    # Two survivors: the short one dropped, the duplicate deduped.
    assert [c.score for c in out] == [88, 75]

    # candidates.json written with the exact schema.
    written = json.loads(
        (cfg.data_path / "sources" / "job1" / "candidates.json").read_text("utf-8")
    )
    assert len(written) == 2
    assert set(written[0]) == {
        "start",
        "end",
        "score",
        "title",
        "hook",
        "hook_caption",
        "reason",
        "sub_scores",
    }
    assert set(written[0]["sub_scores"]) == {"hook", "emotion", "value", "coherence", "payoff"}


def test_analyze_retries_once_then_raises_on_bad_json(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_transcript(cfg, "job2")

    calls: list[str] = []

    def bad_llm(prompt: str) -> str:
        calls.append(prompt)
        return "not json at all"

    with pytest.raises(AnalyzeError):
        analyze("job2", cfg, llm=bad_llm)
    # One retry: two total attempts.
    assert len(calls) == 2
    assert "Output ONLY valid JSON" in calls[1]


def test_analyze_recovers_on_second_attempt(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_transcript(cfg, "job3")

    good = json.dumps(
        {
            "candidates": [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "score": 80,
                    "title": "ok",
                    "hook": "h",
                    "hook_caption": "c",
                    "reason": "r",
                    "sub_scores": {},
                }
            ]
        }
    )
    responses = iter(["```garbage", good])

    out = analyze("job3", cfg, llm=lambda _p: next(responses))
    assert len(out) == 1
    assert out[0].score == 80


def test_analyze_missing_transcript_raises(tmp_path: Path) -> None:
    with pytest.raises(AnalyzeError):
        analyze("nope", _cfg(tmp_path), llm=lambda _p: "{}")
