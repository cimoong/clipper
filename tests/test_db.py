"""Tests for the SQLite persistence layer (clipforge.db).

Every test uses a throwaway database under ``tmp_path`` via a Config, so nothing
touches the real ``data/`` directory. No network, no external services.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clipforge import db
from clipforge.config import Config


@pytest.fixture
def conn(tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    connection = db.connect(cfg)
    yield connection
    connection.close()


# --------------------------------------------------------------------------- #
# connection / schema
# --------------------------------------------------------------------------- #


def test_connect_creates_db_file_and_wal(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    connection = db.connect(cfg)
    try:
        assert Path(db.db_path(cfg)).is_file()
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        connection.close()


def test_connect_is_idempotent(tmp_path: Path) -> None:
    cfg = Config(data_dir=str(tmp_path / "data"))
    db.connect(cfg).close()
    # Re-connecting must not error on already-existing tables.
    second = db.connect(cfg)
    second.close()


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


def test_create_and_get_job(conn: sqlite3.Connection) -> None:
    db.create_job(
        conn,
        id="job1",
        source_url="https://youtu.be/x",
        params={"num_clips": 5},
    )
    job = db.get_job(conn, "job1")
    assert job is not None
    assert job["id"] == "job1"
    assert job["source_url"] == "https://youtu.be/x"
    assert job["status"] == "QUEUED"
    assert job["progress"] == 0
    assert job["created_at"]  # stamped
    assert job["finished_at"] is None
    # params round-trip as JSON text.
    assert '"num_clips": 5' in job["params_json"]


def test_get_job_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert db.get_job(conn, "nope") is None


def test_create_job_duplicate_id_raises(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="dup", source_url="a")
    with pytest.raises(sqlite3.IntegrityError):
        db.create_job(conn, id="dup", source_url="b")


def test_update_job_status_sets_progress(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    db.update_job_status(conn, "j", "TRANSCRIBING", progress=40.0)
    job = db.get_job(conn, "j")
    assert job is not None
    assert job["status"] == "TRANSCRIBING"
    assert job["progress"] == 40.0
    # Non-terminal status must not stamp finished_at.
    assert job["finished_at"] is None


def test_update_job_status_partial_leaves_other_fields(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a", title="original")
    db.update_job_status(conn, "j", "ANALYZING", progress=60.0)
    job = db.get_job(conn, "j")
    assert job is not None
    # title was not passed, so it is untouched.
    assert job["title"] == "original"


def test_update_job_status_done_stamps_finished_at(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    db.update_job_status(conn, "j", "DONE", progress=100.0, title="Final")
    job = db.get_job(conn, "j")
    assert job is not None
    assert job["status"] == "DONE"
    assert job["finished_at"] is not None
    assert job["title"] == "Final"


def test_update_job_status_failed_stores_error(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    db.update_job_status(conn, "j", "FAILED", error="[download] boom")
    job = db.get_job(conn, "j")
    assert job is not None
    assert job["status"] == "FAILED"
    assert job["error"] == "[download] boom"
    assert job["finished_at"] is not None


def test_list_jobs_newest_first_and_limit(conn: sqlite3.Connection) -> None:
    for i in range(5):
        db.create_job(conn, id=f"job{i}", source_url=f"u{i}")
    jobs = db.list_jobs(conn, limit=3)
    assert len(jobs) == 3
    # created_at DESC, id DESC -> highest id first (same-second inserts).
    assert jobs[0]["id"] == "job4"
    assert [j["id"] for j in jobs] == ["job4", "job3", "job2"]


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #


def test_insert_clips_and_list_sorted_by_score(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    inserted = db.insert_clips(
        conn,
        "j",
        [
            {
                "start": 10.0,
                "end": 40.0,
                "score": 70,
                "title": "Second",
                "hook": "h2",
                "reason": "r2",
                "sub_scores": {"hook": 70},
                "file": "/out/clip_01.mp4",
                "thumb": "/out/clip_01.jpg",
            },
            {
                "start_s": 100.0,
                "end_s": 140.0,
                "score": 92,
                "title": "First",
                "hook": "h1",
                "reason": "r1",
                "file_path": "/out/clip_00.mp4",
                "thumb_path": "/out/clip_00.jpg",
            },
        ],
    )
    assert inserted == 2
    clips = db.list_clips(conn, "j")
    assert [c["title"] for c in clips] == ["First", "Second"]
    top = clips[0]
    assert top["score"] == 92
    assert top["start_s"] == 100.0
    assert top["end_s"] == 140.0
    assert top["file_path"] == "/out/clip_00.mp4"
    assert top["status"] == "ready"
    # sub_scores JSON-encoded (absent on the top clip -> None).
    assert top["sub_scores_json"] is None
    assert '"hook": 70' in clips[1]["sub_scores_json"]


def test_insert_clips_empty_list(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    assert db.insert_clips(conn, "j", []) == 0
    assert db.list_clips(conn, "j") == []


def test_clips_cascade_delete_with_job(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    db.insert_clips(conn, "j", [{"start": 0, "end": 30, "score": 50}])
    conn.execute("DELETE FROM jobs WHERE id = ?", ("j",))
    conn.commit()
    assert db.list_clips(conn, "j") == []


# --------------------------------------------------------------------------- #
# transcripts
# --------------------------------------------------------------------------- #


def test_insert_and_get_transcript(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    segments = [{"start": 0.0, "end": 2.0, "text": "hi"}]
    db.insert_transcript(conn, "j", segments=segments, words=None, model_used="small")
    got = db.get_transcript(conn, "j")
    assert got is not None
    assert got["model_used"] == "small"
    assert got["segments"] == segments
    assert got["words"] is None


def test_insert_transcript_replaces_existing(conn: sqlite3.Connection) -> None:
    db.create_job(conn, id="j", source_url="a")
    db.insert_transcript(conn, "j", segments=[{"text": "old"}], model_used="small")
    db.insert_transcript(conn, "j", segments=[{"text": "new"}], model_used="medium")
    got = db.get_transcript(conn, "j")
    assert got is not None
    assert got["segments"] == [{"text": "new"}]
    assert got["model_used"] == "medium"


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


def test_get_setting_default_when_missing(conn: sqlite3.Connection) -> None:
    assert db.get_setting(conn, "whisper_model") is None
    assert db.get_setting(conn, "whisper_model", "small") == "small"


def test_set_and_get_setting(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "whisper_model", "medium")
    assert db.get_setting(conn, "whisper_model") == "medium"


def test_set_setting_overwrites(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "num_clips", "8")
    db.set_setting(conn, "num_clips", "5")
    assert db.get_setting(conn, "num_clips") == "5"
