"""SQLite persistence for ClipForge (stdlib ``sqlite3``, no ORM).

The database lives at ``{DATA_DIR}/clipforge.db`` and is opened in WAL mode so a
single asyncio worker can write while the API reads job/clip state concurrently.

Tables follow PRD-ClipForge §8 (``jobs``, ``transcripts``, ``clips``,
``settings``). One deliberate addition: ``jobs.progress`` (a 0-100 REAL) stores
the pipeline percent the queue worker reports alongside ``status`` — §8 sketches
the columns but the queue contract requires a place to persist the percent.

Every helper takes an open :class:`sqlite3.Connection` (from :func:`connect`)
and commits its own write, so callers can share one connection or open a
short-lived one per operation (the pattern the queue worker uses to stay
thread-safe across executor threads).

Run standalone to list recorded jobs:

    python -m clipforge.db
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .config import Config

# Job lifecycle states. The stage-derived ones mirror the PRD user-flow
# (DOWNLOADING -> TRANSCRIBING -> ANALYZING -> RENDERING -> DONE); the queue
# worker sets them from the pipeline's per-stage progress callback.
JOB_STATUSES: frozenset[str] = frozenset(
    {
        "QUEUED",
        "DOWNLOADING",
        "TRANSCRIBING",
        "ANALYZING",
        "CUTTING",
        "RENDERING",
        "DONE",
        "FAILED",
    }
)

# Statuses at which a job has stopped running; used to auto-stamp finished_at.
_TERMINAL_STATUSES: frozenset[str] = frozenset({"DONE", "FAILED"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    source_url   TEXT,
    source_path  TEXT,
    title        TEXT,
    duration_s   REAL,
    status       TEXT NOT NULL DEFAULT 'QUEUED',
    progress     REAL NOT NULL DEFAULT 0,
    language     TEXT,
    params_json  TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS transcripts (
    job_id        TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    segments_json TEXT,
    words_json    TEXT,
    model_used    TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start_s         REAL,
    end_s           REAL,
    score           INTEGER,
    title           TEXT,
    hook            TEXT,
    hook_caption    TEXT,
    reason          TEXT,
    sub_scores_json TEXT,
    file_path       TEXT,
    thumb_path      TEXT,
    status          TEXT NOT NULL DEFAULT 'ready',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clips_job ON clips(job_id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# --------------------------------------------------------------------------- #
# Connection / schema
# --------------------------------------------------------------------------- #


def _now() -> str:
    """Current time as an ISO-8601 UTC string (stored as TEXT)."""
    return datetime.now(timezone.utc).isoformat()


def db_path(cfg: Config) -> str:
    """Absolute path to the SQLite file for ``cfg``."""
    return str(cfg.data_path / "clipforge.db")


def connect(cfg: Config) -> sqlite3.Connection:
    """Open (creating if needed) the ClipForge database in WAL mode.

    ``check_same_thread=False`` so a connection opened on the event loop can be
    handed to an executor thread; callers must still not use one connection from
    two threads at once. The schema is ensured on every connect (cheap, idempotent).
    """
    path = cfg.data_path / "clipforge.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


def create_job(
    conn: sqlite3.Connection,
    *,
    id: str,
    source_url: str = "",
    source_path: str | None = None,
    title: str = "",
    params: Mapping[str, Any] | None = None,
    status: str = "QUEUED",
) -> str:
    """Insert a new job row and return its id.

    ``params`` is JSON-encoded into ``params_json``. Raises
    ``sqlite3.IntegrityError`` if ``id`` already exists.
    """
    conn.execute(
        """
        INSERT INTO jobs (id, source_url, source_path, title, status, progress,
                          params_json, created_at)
        VALUES (?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            id,
            source_url,
            source_path,
            title,
            status,
            json.dumps(params) if params is not None else None,
            _now(),
        ),
    )
    conn.commit()
    return id


def update_job_status(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    *,
    progress: float | None = None,
    error: str | None = None,
    title: str | None = None,
    duration_s: float | None = None,
    language: str | None = None,
    finished_at: str | None = None,
) -> None:
    """Update a job's ``status`` and any provided fields.

    Only the keyword fields you pass are written (``None`` leaves a column
    untouched). When ``status`` is terminal (DONE/FAILED) and ``finished_at`` is
    not given, it is stamped automatically.
    """
    sets = ["status = ?"]
    vals: list[Any] = [status]

    if progress is not None:
        sets.append("progress = ?")
        vals.append(float(progress))
    if error is not None:
        sets.append("error = ?")
        vals.append(error)
    if title is not None:
        sets.append("title = ?")
        vals.append(title)
    if duration_s is not None:
        sets.append("duration_s = ?")
        vals.append(float(duration_s))
    if language is not None:
        sets.append("language = ?")
        vals.append(language)

    if finished_at is None and status in _TERMINAL_STATUSES:
        finished_at = _now()
    if finished_at is not None:
        sets.append("finished_at = ?")
        vals.append(finished_at)

    vals.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    """Return a job row as a dict, or ``None`` if there is no such job."""
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def list_jobs(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict[str, Any]]:
    """Return jobs newest-first (by ``created_at`` then ``id``), capped at ``limit``."""
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# transcripts
# --------------------------------------------------------------------------- #


def insert_transcript(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    segments: Any,
    words: Any = None,
    model_used: str = "",
) -> None:
    """Cache a job's transcript (replacing any existing row for that job).

    ``segments``/``words`` are JSON-encoded. Storing them lets re-analyze reuse
    the transcript without re-transcribing (PRD F12).
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO transcripts (job_id, segments_json, words_json,
                                            model_used, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            job_id,
            json.dumps(segments) if segments is not None else None,
            json.dumps(words) if words is not None else None,
            model_used,
            _now(),
        ),
    )
    conn.commit()


def get_transcript(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    """Return the cached transcript row for ``job_id`` (with JSON decoded), or None."""
    row = conn.execute("SELECT * FROM transcripts WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["segments"] = json.loads(data["segments_json"]) if data.get("segments_json") else None
    data["words"] = json.loads(data["words_json"]) if data.get("words_json") else None
    return data


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #


def _clip_value(clip: Mapping[str, Any], *keys: str) -> Any:
    """First present value among ``keys`` (supports ``start_s``/``start`` aliases)."""
    for key in keys:
        if key in clip and clip[key] is not None:
            return clip[key]
    return None


def insert_clips(
    conn: sqlite3.Connection,
    job_id: str,
    clips: Sequence[Mapping[str, Any]],
) -> int:
    """Bulk-insert clip rows for ``job_id``; return how many were inserted.

    Each clip mapping is read leniently: ``start_s``/``start`` and
    ``end_s``/``end`` aliases are both accepted, ``sub_scores`` is JSON-encoded
    into ``sub_scores_json``, and ``file``/``file_path`` (and ``thumb``/
    ``thumb_path``) are interchangeable.
    """
    rows = []
    for clip in clips:
        sub_scores = clip.get("sub_scores")
        rows.append(
            (
                job_id,
                _num(_clip_value(clip, "start_s", "start")),
                _num(_clip_value(clip, "end_s", "end")),
                _int(clip.get("score")),
                str(clip.get("title", "") or ""),
                str(clip.get("hook", "") or ""),
                str(clip.get("hook_caption", "") or ""),
                str(clip.get("reason", "") or ""),
                json.dumps(sub_scores) if sub_scores is not None else None,
                _str_or_none(_clip_value(clip, "file_path", "file")),
                _str_or_none(_clip_value(clip, "thumb_path", "thumb")),
                str(clip.get("status", "ready") or "ready"),
                _now(),
            )
        )
    conn.executemany(
        """
        INSERT INTO clips (job_id, start_s, end_s, score, title, hook, hook_caption,
                           reason, sub_scores_json, file_path, thumb_path, status,
                           created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def list_clips(conn: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
    """Return a job's clips ordered by score descending (best first)."""
    rows = conn.execute(
        "SELECT * FROM clips WHERE job_id = ? ORDER BY score DESC, id ASC",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_clip(conn: sqlite3.Connection, clip_id: int) -> dict[str, Any] | None:
    """Return a single clip row as a dict, or ``None`` if there is no such clip."""
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    return _row_to_dict(row)


def update_clip(
    conn: sqlite3.Connection,
    clip_id: int,
    *,
    hook_caption: str | None = None,
    title: str | None = None,
    file_path: str | None = None,
    thumb_path: str | None = None,
    status: str | None = None,
) -> None:
    """Update only the provided fields of a clip (``None`` leaves a column untouched).

    Used by the web API to edit a clip's on-screen hook text (``hook_caption``)
    and to flip its ``status`` around a single-clip re-render.
    """
    sets: list[str] = []
    vals: list[Any] = []
    if hook_caption is not None:
        sets.append("hook_caption = ?")
        vals.append(hook_caption)
    if title is not None:
        sets.append("title = ?")
        vals.append(title)
    if file_path is not None:
        sets.append("file_path = ?")
        vals.append(file_path)
    if thumb_path is not None:
        sets.append("thumb_path = ?")
        vals.append(thumb_path)
    if status is not None:
        sets.append("status = ?")
        vals.append(status)
    if not sets:
        return
    vals.append(clip_id)
    conn.execute(f"UPDATE clips SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()


# --------------------------------------------------------------------------- #
# settings (simple key/value store)
# --------------------------------------------------------------------------- #


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    """Return the stored value for ``key``, or ``default`` if unset."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update the ``key`` setting."""
    conn.execute(
        """
        INSERT INTO settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# small coercion helpers
# --------------------------------------------------------------------------- #


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def main() -> None:
    cfg = Config.load()
    conn = connect(cfg)
    try:
        jobs = list_jobs(conn)
        if not jobs:
            print(f"No jobs in {db_path(cfg)}")
            return
        print(f"{len(jobs)} job(s) in {db_path(cfg)}:")
        for job in jobs:
            print(
                f"  {job['id']}  {job['status']:<12} {job.get('progress', 0):>5.0f}%"
                f"  {job.get('title') or job.get('source_url') or ''}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
