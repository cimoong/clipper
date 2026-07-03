"""Tests for the FastAPI web layer (clipforge.web.app).

The heavy pipeline is mocked out: ``clipforge.queue.run_new`` is replaced with a
fake so ``POST /api/jobs`` exercises the real enqueue -> worker -> db path without
downloading, transcribing, or encoding anything. Every test uses a throwaway
``data_dir`` under ``tmp_path`` so nothing touches the real ``data/`` directory.

TestClient is httpx-based (Starlette's TestClient wraps httpx) and runs the app's
lifespan, so the queue worker really starts and drains the job.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge import db
from clipforge import queue as queue_mod
from clipforge.config import Config
from clipforge.web.app import create_app


def _fake_results(source_url: str, cfg, *, no_llm=False, job_id=None, progress=None):
    """Stand-in for pipeline.run_new: report one stage, then return two clips."""
    if progress is not None:
        progress("download", 0, 5)
    return {
        "job_id": job_id,
        "source_url": source_url,
        "title": "Mocked Title",
        "clips": [
            {
                "start": 0.0,
                "end": 30.0,
                "score": 90,
                "title": "Clip A",
                "hook": "opening line",
                "hook_caption": "CLIP A",
                "reason": "because",
            },
            {
                "start": 40.0,
                "end": 70.0,
                "score": 70,
                "title": "Clip B",
                "hook": "second line",
                "hook_caption": "CLIP B",
                "reason": "also",
            },
        ],
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(queue_mod, "run_new", _fake_results)
    cfg = Config(data_dir=str(tmp_path / "data"))
    with TestClient(create_app(cfg)) as c:
        c.app_cfg = cfg  # type: ignore[attr-defined]  # handy for direct DB seeding
        yield c


def _seed_job_with_clips(cfg: Config) -> tuple[str, list[int]]:
    """Insert a DONE job plus two clips straight into the DB (no pipeline)."""
    conn = db.connect(cfg)
    try:
        job_id = "seed0001"
        db.create_job(conn, id=job_id, source_url="https://youtu.be/seed", title="Seeded Job")
        db.update_job_status(conn, job_id, "DONE", progress=100.0)
        db.insert_clips(
            conn,
            job_id,
            [
                {
                    "start": 12.0,
                    "end": 54.0,
                    "score": 91,
                    "title": "Green clip",
                    "hook_caption": "STOP SCROLLING NOW",
                },
                {
                    "start": 70.0,
                    "end": 100.0,
                    "score": 42,
                    "title": "Gray clip",
                    "hook_caption": "MEH MOMENT",
                },
            ],
        )
        clip_ids = [c["id"] for c in db.list_clips(conn, job_id)]
    finally:
        conn.close()
    return job_id, clip_ids


def _wait_for_status(client: TestClient, job_id: str, target: str, timeout: float = 5.0) -> dict:
    """Poll the job detail endpoint until it reaches ``target`` (or time out)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == target:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {target}; last={job['status']}")


def test_post_job_returns_job_id(client: TestClient) -> None:
    res = client.post("/api/jobs", json={"url": "https://youtu.be/abc"})
    assert res.status_code == 201
    body = res.json()
    assert body["job_id"]


def test_post_job_rejects_empty_url(client: TestClient) -> None:
    res = client.post("/api/jobs", json={"url": "   "})
    assert res.status_code == 422


def test_list_jobs_includes_enqueued_job_newest_first(client: TestClient) -> None:
    first = client.post("/api/jobs", json={"url": "https://youtu.be/one"}).json()["job_id"]
    second = client.post("/api/jobs", json={"url": "https://youtu.be/two"}).json()["job_id"]

    jobs = client.get("/api/jobs").json()
    ids = [j["id"] for j in jobs]
    assert first in ids
    assert second in ids
    # newest first: the second enqueue should sort ahead of the first.
    assert ids.index(second) < ids.index(first)


def test_job_runs_through_mocked_pipeline_to_done(client: TestClient) -> None:
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/abc"}).json()["job_id"]
    job = _wait_for_status(client, job_id, "DONE")
    assert job["title"] == "Mocked Title"
    assert job["progress"] == 100.0
    # Clips from the mocked pipeline are persisted and returned, best score first.
    clips = job["clips"]
    assert [c["title"] for c in clips] == ["Clip A", "Clip B"]
    assert clips[0]["hook_caption"] == "CLIP A"


def test_get_missing_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_patch_clip_updates_hook_caption(client: TestClient) -> None:
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/abc"}).json()["job_id"]
    job = _wait_for_status(client, job_id, "DONE")
    clip_id = job["clips"][0]["id"]

    res = client.patch(f"/api/clips/{clip_id}", json={"hook_caption": "NEW HOOK"})
    assert res.status_code == 200
    assert res.json()["hook_caption"] == "NEW HOOK"


def test_health_reports_ok(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "ffmpeg" in body


# --------------------------------------------------------------------------- #
# HTML pages / htmx fragments (server-render smoke tests)
# --------------------------------------------------------------------------- #


def test_dashboard_renders_with_seeded_job(client: TestClient) -> None:
    _seed_job_with_clips(client.app_cfg)  # type: ignore[attr-defined]
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    # Seeded job appears with its status badge and a link to its results page.
    assert "Seeded Job" in res.text
    assert "/jobs/seed0001" in res.text
    assert "htmx.org" in res.text  # htmx loaded from CDN


def test_results_page_renders_clip_cards(client: TestClient) -> None:
    job_id, _ = _seed_job_with_clips(client.app_cfg)  # type: ignore[attr-defined]
    res = client.get(f"/jobs/{job_id}")
    assert res.status_code == 200
    # Both clips render, with score-colour classes (>=80 green, <60 gray).
    assert "Green clip" in res.text
    assert "Gray clip" in res.text
    assert "score high" in res.text
    assert "score low" in res.text
    assert "STOP SCROLLING NOW" in res.text


def test_results_page_missing_job_is_404(client: TestClient) -> None:
    assert client.get("/jobs/nope").status_code == 404


def test_post_jobs_form_returns_row_fragment(client: TestClient) -> None:
    res = client.post("/jobs", data={"url": "https://youtu.be/htmx"})
    assert res.status_code == 200
    # A single <tr> job-row fragment wired for live SSE updates.
    assert "<tr" in res.text
    assert "sse-connect" in res.text


def test_patch_caption_form_reveals_rerender(client: TestClient) -> None:
    _, clip_ids = _seed_job_with_clips(client.app_cfg)  # type: ignore[attr-defined]
    res = client.patch(f"/clips/{clip_ids[0]}/caption", data={"hook_caption": "NEW CAPTION"})
    assert res.status_code == 200
    assert "NEW CAPTION" in res.text
    assert 'hx-post="/clips/' in res.text  # Re-render button now present


# --------------------------------------------------------------------------- #
# Failure surface + Retry (resume from failed stage)
# --------------------------------------------------------------------------- #


def _failing_run(source_url, cfg, *, no_llm=False, job_id=None, progress=None):
    """Stand-in for run_new that fails transiently at the download stage."""
    if progress is not None:
        progress("download", 0, 5)
    from clipforge.pipeline import PipelineError

    raise PipelineError("download", "Network error while downloading.", job_id)


def _fake_resume(job_id, cfg, *, no_llm=None, progress=None):
    """Stand-in for resume_job: succeed and return the two mocked clips."""
    if progress is not None:
        progress("transcribe", 1, 5)
    return {
        "job_id": job_id,
        "title": "Recovered Title",
        "clips": [
            {"start": 0.0, "end": 30.0, "score": 90, "title": "Clip A", "hook_caption": "CLIP A"},
            {"start": 40.0, "end": 70.0, "score": 70, "title": "Clip B", "hook_caption": "CLIP B"},
        ],
    }


def test_failed_job_shows_stage_and_retry_on_dashboard(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue_mod, "run_new", _failing_run)
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    _wait_for_status(client, job_id, "FAILED")

    page = client.get("/").text
    # Error surface names the failed stage and offers a Retry button.
    assert "FAILED" in page
    assert "download" in page
    assert f'hx-post="/api/jobs/{job_id}/retry"' in page


def test_retry_resumes_failed_job_to_done(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Make the job fail.
    monkeypatch.setattr(queue_mod, "run_new", _failing_run)
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    failed = _wait_for_status(client, job_id, "FAILED")
    assert failed["error"]  # "[download] Network error..." recorded

    # 2. Retry now resumes via the (mocked) resume path and finishes.
    monkeypatch.setattr(queue_mod, "resume_job", _fake_resume)
    res = client.post(f"/api/jobs/{job_id}/retry")
    assert res.status_code == 200
    assert "sse-connect" in res.text  # row is active again (live SSE)

    done = _wait_for_status(client, job_id, "DONE")
    assert done["title"] == "Recovered Title"
    assert [c["title"] for c in done["clips"]] == ["Clip A", "Clip B"]


def test_retry_rejects_non_failed_job(client: TestClient) -> None:
    job_id, _ = _seed_job_with_clips(client.app_cfg)  # type: ignore[attr-defined]  # DONE job
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409


def test_retry_missing_job_is_404(client: TestClient) -> None:
    assert client.post("/api/jobs/nope/retry").status_code == 404


# --------------------------------------------------------------------------- #
# Settings page
# --------------------------------------------------------------------------- #


def test_settings_page_renders_with_masked_key(client: TestClient) -> None:
    res = client.get("/settings")
    assert res.status_code == 200
    assert "Settings" in res.text
    # Test config has no API key -> masked "(not set...)"; the raw key never leaks.
    assert "not set" in res.text


_VALID_SETTINGS = {
    "llm_provider": "gemini",
    "gemini_model": "gemini-x",
    "whisper_model": "small",
    "num_clips": "4",
    "clip_min_s": "20",
    "clip_max_s": "80",
    "caption_style": "clean",
    "hook_mode": "full",
}


def test_settings_post_persists(client: TestClient) -> None:
    res = client.post("/settings", data=_VALID_SETTINGS)
    assert res.status_code == 200
    assert "Settings saved" in res.text

    conn = db.connect(client.app_cfg)  # type: ignore[attr-defined]
    try:
        assert db.get_setting(conn, "num_clips") == "4"
        assert db.get_setting(conn, "caption_style") == "clean"
        assert db.get_setting(conn, "gemini_model") == "gemini-x"
    finally:
        conn.close()


def test_settings_post_rejects_invalid_durations(client: TestClient) -> None:
    bad = {**_VALID_SETTINGS, "clip_min_s": "90", "clip_max_s": "30"}
    res = client.post("/settings", data=bad)
    assert res.status_code == 200
    assert "less than" in res.text  # validation error surfaced

    conn = db.connect(client.app_cfg)  # type: ignore[attr-defined]
    try:
        assert db.get_setting(conn, "clip_min_s") is None  # nothing persisted
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Re-analyze (new job version from cached transcript)
# --------------------------------------------------------------------------- #


def _seed_transcribed_job(cfg, job_id: str = "orig0001") -> str:
    """Seed a DONE job with an on-disk video + transcript (as after transcribe)."""
    conn = db.connect(cfg)
    db.create_job(conn, id=job_id, source_url="https://youtu.be/orig", title="Orig Job")
    db.update_job_status(conn, job_id, "DONE", progress=100.0)
    conn.close()

    src = cfg.data_path / "sources" / job_id
    src.mkdir(parents=True)
    (src / "video.mp4").write_bytes(b"video-bytes")
    (src / "transcript.json").write_text('{"segments": []}', encoding="utf-8")
    return job_id


def test_reanalyze_creates_linked_new_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = client.app_cfg  # type: ignore[attr-defined]
    orig = _seed_transcribed_job(cfg)
    # The worker's resume runs the (mocked) pipeline and returns clips.
    monkeypatch.setattr(queue_mod, "resume_job", _fake_resume)

    res = client.post(f"/api/jobs/{orig}/reanalyze")
    assert res.status_code == 202
    new_id = res.json()["job_id"]
    assert new_id != orig
    assert res.headers.get("HX-Redirect") == f"/jobs/{new_id}"

    done = _wait_for_status(client, new_id, "DONE")
    assert done["source_url"] == "https://youtu.be/orig"  # linked to same source
    assert [c["title"] for c in done["clips"]] == ["Clip A", "Clip B"]


def test_reanalyze_requires_cached_transcript(client: TestClient) -> None:
    cfg = client.app_cfg  # type: ignore[attr-defined]
    conn = db.connect(cfg)
    db.create_job(conn, id="notrans", source_url="u", title="x")
    db.update_job_status(conn, "notrans", "DONE")
    conn.close()
    # No transcript on disk -> 409.
    assert client.post("/api/jobs/notrans/reanalyze").status_code == 409


def test_reanalyze_missing_job_is_404(client: TestClient) -> None:
    assert client.post("/api/jobs/ghost/reanalyze").status_code == 404
