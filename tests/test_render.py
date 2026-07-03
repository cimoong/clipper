"""Tests for the render stage's pure logic (clipforge.render).

Only the batched metadata step is exercised here — it is the render stage's new
testable surface and needs no ffmpeg, mediapipe, or network. The heavy
reframe+burn path is covered by the end-to-end pipeline run (like reframe, it has
no standalone unit test).
"""

from __future__ import annotations

import json

from clipforge.config import Config
from clipforge.render import (
    ClipMeta,
    _clean_hashtags,
    _fallback_meta,
    generate_metadata,
)

_CANDIDATES = [
    {"title": "The $50K mistake", "hook_caption": "THE $50K MISTAKE", "hook": "Nobody talks..."},
    {"title": "Why I quit", "hook_caption": "WHY I QUIT", "hook": "So I walked in..."},
]


def test_clean_hashtags_normalises_and_dedupes() -> None:
    tags = _clean_hashtags(["#Viral", "Money Tips", "#viral", "  Startup  ", ""])
    assert tags == ["#viral", "#moneytips", "#startup"]


def test_clean_hashtags_caps_at_eight() -> None:
    raw = [f"#tag{i}" for i in range(12)]
    assert len(_clean_hashtags(raw)) == 8


def test_clean_hashtags_non_list_returns_empty() -> None:
    assert _clean_hashtags("nope") == []


def test_fallback_meta_shape() -> None:
    meta = _fallback_meta(_CANDIDATES[0])
    assert isinstance(meta, ClipMeta)
    assert meta.title == "The $50K mistake"
    assert meta.hook_caption == "THE $50K MISTAKE"
    assert meta.suggested_description.endswith("Follow for more.")
    assert 5 <= len(meta.hashtags) <= 8
    assert all(t.startswith("#") and t == t.lower() for t in meta.hashtags)


def test_generate_metadata_no_llm_uses_fallback() -> None:
    metas = generate_metadata(_CANDIDATES, Config(), no_llm=True)
    assert len(metas) == 2
    assert all(m.suggested_description.endswith("Follow for more.") for m in metas)


def test_generate_metadata_uses_injected_llm() -> None:
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            {
                "clips": [
                    {
                        "index": 0,
                        "suggested_description": "Big lesson here. It cost a fortune. Follow for more.",
                        "hashtags": [
                            "#Startup",
                            "#money tips",
                            "#business",
                            "#founder",
                            "#lessons",
                        ],
                    },
                    {
                        "index": 1,
                        "suggested_description": "He walked out. Here is why. Subscribe for more.",
                        "hashtags": ["#quit", "#career", "#work", "#life", "#story"],
                    },
                ]
            }
        )

    metas = generate_metadata(_CANDIDATES, Config(), llm=fake_llm)

    # Exactly one batched call for the whole job.
    assert len(calls) == 1
    assert metas[0].suggested_description.startswith("Big lesson")
    # Hashtags are normalised (lowercase, space-free).
    assert metas[0].hashtags == ["#startup", "#moneytips", "#business", "#founder", "#lessons"]
    # Titles/hook_captions always come from the candidate, never the LLM.
    assert metas[1].title == "Why I quit"
    assert metas[1].hook_caption == "WHY I QUIT"


def test_generate_metadata_fills_missing_clip_with_fallback() -> None:
    def partial_llm(_prompt: str) -> str:
        # Only clip 1 is returned; clip 0 must fall back.
        return json.dumps(
            {"clips": [{"index": 1, "suggested_description": "A. B. Follow.", "hashtags": []}]}
        )

    metas = generate_metadata(_CANDIDATES, Config(), llm=partial_llm)
    assert metas[0].suggested_description.endswith("Follow for more.")  # fallback
    # Clip 1 had too few hashtags (<5) -> fallback hashtags kept.
    assert 5 <= len(metas[1].hashtags) <= 8


def test_generate_metadata_bad_json_falls_back() -> None:
    metas = generate_metadata(_CANDIDATES, Config(), llm=lambda _p: "not json at all")
    assert len(metas) == 2
    assert all(m.suggested_description.endswith("Follow for more.") for m in metas)


def test_generate_metadata_empty_candidates() -> None:
    assert generate_metadata([], Config(), no_llm=True) == []
