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
        yield c


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
