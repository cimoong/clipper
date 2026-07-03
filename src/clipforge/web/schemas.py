"""Pydantic request/response models for the ClipForge web API.

Response models are built from the raw ``db`` row dicts via the ``from_row``
classmethods. On-disk file paths are never exposed to the client: a clip is
downloaded through ``/api/clips/{id}/file`` and ``/thumb`` instead, and the
models only advertise whether those artifacts exist (``has_file``/``has_thumb``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class CreateJobRequest(BaseModel):
    """Body for ``POST /api/jobs``: a YouTube URL or a local file path."""

    url: str = Field(min_length=1, description="Source YouTube URL or local file path.")


class CreateJobResponse(BaseModel):
    job_id: str


class PatchClipRequest(BaseModel):
    """Body for ``PATCH /api/clips/{id}``: the new on-screen hook caption."""

    hook_caption: str


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class ClipOut(BaseModel):
    id: int
    job_id: str
    start_s: float | None = None
    end_s: float | None = None
    score: int | None = None
    title: str | None = None
    hook: str | None = None
    hook_caption: str | None = None
    reason: str | None = None
    status: str = "ready"
    has_file: bool = False
    has_thumb: bool = False

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ClipOut":
        return cls(
            id=row["id"],
            job_id=row["job_id"],
            start_s=row.get("start_s"),
            end_s=row.get("end_s"),
            score=row.get("score"),
            title=row.get("title"),
            hook=row.get("hook"),
            hook_caption=row.get("hook_caption"),
            reason=row.get("reason"),
            status=row.get("status") or "ready",
            has_file=bool(row.get("file_path")),
            has_thumb=bool(row.get("thumb_path")),
        )


class JobOut(BaseModel):
    id: str
    source_url: str | None = None
    title: str | None = None
    status: str
    progress: float = 0.0
    error: str | None = None
    created_at: str
    finished_at: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "JobOut":
        return cls(
            id=row["id"],
            source_url=row.get("source_url"),
            title=row.get("title"),
            status=row["status"],
            progress=row.get("progress") or 0.0,
            error=row.get("error"),
            created_at=row["created_at"],
            finished_at=row.get("finished_at"),
        )


class JobDetail(JobOut):
    clips: list[ClipOut] = []

    @classmethod
    def from_rows(cls, job: dict[str, Any], clips: list[dict[str, Any]]) -> "JobDetail":
        base = JobOut.from_row(job)
        return cls(**base.model_dump(), clips=[ClipOut.from_row(c) for c in clips])
