"""Tests for the hardening pass: per-stage retry, storage cleanup, and the
yt-dlp freshness check. No network, ffmpeg, or LLM is touched — the retriable
stage is monkeypatched and cleanup works on plain temp folders.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import pytest

from clipforge import db, download, pipeline
from clipforge.config import Config


# --------------------------------------------------------------------------- #
# Transient classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        "Network error while downloading. Check your internet connection and try again.",
        "HTTPSConnectionPool: connection timed out",
        "LLM call failed: 503 Service Unavailable: the model is overloaded",
        "rate limit exceeded, please retry",
    ],
)
def test_is_transient_true(message: str) -> None:
    assert pipeline._is_transient(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Unsupported local video format '.txt'.",
        "GEMINI_API_KEY is not set; cannot call the scoring model.",
        "The video is unavailable (it may be private, deleted, or region-locked).",
    ],
)
def test_is_transient_false(message: str) -> None:
    assert pipeline._is_transient(RuntimeError(message)) is False


# --------------------------------------------------------------------------- #
# Per-stage retry (via the download stage)
# --------------------------------------------------------------------------- #


class _FakeDownloadResult:
    title = "Recovered Title"


def _job(tmp_path: Path) -> tuple[pipeline.Job, Config]:
    cfg = Config(data_dir=str(tmp_path / "data"))
    return pipeline.Job(job_id="job1", source_url="https://youtu.be/x"), cfg


def test_stage_retries_once_on_transient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "RETRY_DELAY_S", 0)
    calls = {"n": 0}

    def fake_download(url: str, job_id: str, cfg: Config) -> _FakeDownloadResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Network error while downloading. Try again.")
        return _FakeDownloadResult()

    monkeypatch.setattr(pipeline, "download", fake_download)
    job, cfg = _job(tmp_path)

    pipeline._execute_stage(job, cfg, "download")

    assert calls["n"] == 2  # failed once, retried, succeeded
    assert "download" in job.completed
    assert job.title == "Recovered Title"


def test_stage_does_not_retry_permanent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "RETRY_DELAY_S", 0)
    calls = {"n": 0}

    def fake_download(url: str, job_id: str, cfg: Config) -> _FakeDownloadResult:
        calls["n"] += 1
        raise RuntimeError("Unsupported local video format '.txt'.")

    monkeypatch.setattr(pipeline, "download", fake_download)
    job, cfg = _job(tmp_path)

    with pytest.raises(pipeline.PipelineError) as excinfo:
        pipeline._execute_stage(job, cfg, "download")

    assert calls["n"] == 1  # no retry for a permanent error
    assert excinfo.value.stage == "download"
    assert "download" not in job.completed


def test_stage_gives_up_after_one_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "RETRY_DELAY_S", 0)
    calls = {"n": 0}

    def fake_download(url: str, job_id: str, cfg: Config) -> _FakeDownloadResult:
        calls["n"] += 1
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(pipeline, "download", fake_download)
    job, cfg = _job(tmp_path)

    with pytest.raises(pipeline.PipelineError) as excinfo:
        pipeline._execute_stage(job, cfg, "download")

    assert calls["n"] == 2  # original attempt + one retry, then give up
    assert excinfo.value.stage == "download"


# --------------------------------------------------------------------------- #
# Storage cleanup
# --------------------------------------------------------------------------- #


def _age(path: Path, days: float) -> None:
    """Backdate a path's mtime (and its files') by ``days`` days."""
    ts = time.time() - days * 86400
    os.utime(path, (ts, ts))
    for child in path.rglob("*"):
        os.utime(child, (ts, ts))


def test_cleanup_removes_old_sources_only(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    sources = cfg.data_path / "sources"

    old = sources / "old_job"
    old.mkdir(parents=True)
    (old / "video.mp4").write_bytes(b"old")

    fresh = sources / "fresh_job"
    fresh.mkdir(parents=True)
    (fresh / "video.mp4").write_bytes(b"new")

    # A finished clip that must survive cleanup.
    output = cfg.data_path / "outputs" / "old_job"
    output.mkdir(parents=True)
    (output / "clip_00.mp4").write_bytes(b"keep")

    _age(old, days=45)

    removed = pipeline.cleanup_sources(cfg, days=30)

    assert removed == ["old_job"]
    assert not old.exists()  # stale source deleted
    assert fresh.exists()  # recent source kept
    assert (output / "clip_00.mp4").exists()  # outputs untouched


def test_cleanup_keeps_recent_and_handles_missing_dir(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    # No sources dir at all -> no-op, no error.
    assert pipeline.cleanup_sources(cfg, days=30) == []

    sources = cfg.data_path / "sources"
    recent = sources / "recent_job"
    recent.mkdir(parents=True)
    (recent / "video.mp4").write_bytes(b"x")
    _age(recent, days=10)

    assert pipeline.cleanup_sources(cfg, days=30) == []
    assert recent.exists()


def test_cleanup_uses_reference_now(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    folder = cfg.data_path / "sources" / "job"
    folder.mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"x")
    _age(folder, days=20)

    # From a "now" 40 days in the future, the 20-day-old folder is >30d stale.
    future = datetime.fromtimestamp(time.time() + 40 * 86400)
    assert pipeline.cleanup_sources(cfg, days=30, now=future) == ["job"]


# --------------------------------------------------------------------------- #
# yt-dlp freshness check
# --------------------------------------------------------------------------- #


def test_ytdlp_age_days_parsing() -> None:
    assert download.ytdlp_age_days("2024.01.01", today=date(2024, 1, 31)) == 30
    assert download.ytdlp_age_days("2024.03.10.123456", today=date(2024, 3, 20)) == 10
    assert download.ytdlp_age_days("not-a-version") is None
    assert download.ytdlp_age_days(None) is None


def test_freshness_warns_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "ytdlp_version", lambda: "2020.01.01")
    msg = download.check_ytdlp_freshness(today=date(2024, 1, 1))
    assert msg is not None
    assert "uv lock" in msg


def test_freshness_silent_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "ytdlp_version", lambda: "2024.01.01")
    assert download.check_ytdlp_freshness(today=date(2024, 1, 10)) is None


def test_freshness_warns_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "ytdlp_version", lambda: None)
    msg = download.check_ytdlp_freshness()
    assert msg is not None
    assert "not installed" in msg


# --------------------------------------------------------------------------- #
# Settings overlay
# --------------------------------------------------------------------------- #


def test_effective_config_overlays_saved_settings(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"), num_clips=8, gemini_model="base-model")
    conn = db.connect(cfg)
    db.set_setting(conn, "num_clips", "3")
    db.set_setting(conn, "gemini_model", "gemini-x")
    conn.close()

    eff = pipeline.effective_config(cfg)
    assert eff.num_clips == 3
    assert eff.gemini_model == "gemini-x"
    assert eff.data_dir == cfg.data_dir  # untouched fields preserved


def test_effective_config_noop_when_no_settings(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    db.connect(cfg).close()  # create an empty settings table
    assert pipeline.effective_config(cfg) is cfg


# --------------------------------------------------------------------------- #
# Re-analyze: reuse the cached transcript, honour current num_clips
# --------------------------------------------------------------------------- #


def _seed_source(cfg: Config, job_id: str) -> Path:
    """Create a source dir with a (fake) video + transcript, as after transcribe."""
    src = cfg.data_path / "sources" / job_id
    src.mkdir(parents=True)
    (src / "video.mp4").write_bytes(b"fake-video-bytes")
    (src / "transcript.json").write_text('{"segments": []}', encoding="utf-8")
    return src


def test_reanalyze_skips_download_transcribe_and_honours_num_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    orig = "orig-job"
    _seed_source(cfg, orig)

    # download / transcribe must never be invoked for a re-analyze.
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("re-analyze must not re-download or re-transcribe")

    monkeypatch.setattr(pipeline, "download", boom)
    monkeypatch.setattr(pipeline, "transcribe", boom)

    # Fake analyze writes exactly cfg.num_clips candidates -> proves the current
    # setting reached the analyze stage.
    def fake_analyze(job_id: str, cfg_arg: Config) -> None:
        cands = [
            {
                "start": float(i * 40),
                "end": float(i * 40 + 30),
                "score": 90 - i,
                "title": f"Clip {i}",
                "hook": "",
                "hook_caption": "",
                "reason": "",
            }
            for i in range(cfg_arg.num_clips)
        ]
        out = cfg_arg.data_path / "sources" / job_id / "candidates.json"
        out.write_text(json.dumps(cands), encoding="utf-8")

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(pipeline, "cut_clips", lambda job_id, cfg_arg: None)
    monkeypatch.setattr(pipeline, "render_clips", lambda job_id, cfg_arg, **kw: [])

    # num_clips = 3 -> first re-analyze produces 3 clips.
    conn = db.connect(cfg)
    db.set_setting(conn, "num_clips", "3")
    conn.close()
    new1 = pipeline.setup_reanalyze_job(orig, cfg, source_url="https://x", title="Orig")
    res1 = pipeline.resume_job(new1, cfg)
    assert new1 != orig
    assert len(res1["clips"]) == 3

    # Change the setting and re-analyze the SAME source again -> a different set.
    conn = db.connect(cfg)
    db.set_setting(conn, "num_clips", "5")
    conn.close()
    new2 = pipeline.setup_reanalyze_job(orig, cfg, source_url="https://x", title="Orig")
    res2 = pipeline.resume_job(new2, cfg)
    assert new2 != new1
    assert len(res2["clips"]) == 5

    # The new job's source reused the original video via link/copy (no re-download).
    assert (cfg.data_path / "sources" / new1 / "video.mp4").is_file()
    assert (cfg.data_path / "sources" / new1 / "transcript.json").is_file()


def test_setup_reanalyze_requires_transcript(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    src = cfg.data_path / "sources" / "orig"
    src.mkdir(parents=True)
    (src / "video.mp4").write_bytes(b"v")  # video present, transcript missing

    with pytest.raises(pipeline.PipelineError) as excinfo:
        pipeline.setup_reanalyze_job("orig", cfg, source_url="u")
    assert excinfo.value.stage == "analyze"


def test_apply_render_presentation_sets_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.render, "CAPTION_STYLE", "bold")
    monkeypatch.setattr(pipeline.render, "HOOK_MODE", "3s")

    pipeline._apply_render_presentation({"caption_style": "clean", "hook_mode": "full"})
    assert pipeline.render.CAPTION_STYLE == "clean"
    assert pipeline.render.HOOK_MODE == "full"

    # Invalid values leave the globals untouched.
    pipeline._apply_render_presentation({"caption_style": "nope", "hook_mode": "nope"})
    assert pipeline.render.CAPTION_STYLE == "clean"
    assert pipeline.render.HOOK_MODE == "full"
