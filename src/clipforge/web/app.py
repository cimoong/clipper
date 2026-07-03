"""FastAPI application for ClipForge: a small dashboard + JSON API.

Wires the SQLite layer (:mod:`clipforge.db`) and the single-worker job queue
(:mod:`clipforge.queue`) behind HTTP. The heavy pipeline (Whisper / NVENC) never
runs inside a request handler — ``POST /api/jobs`` only enqueues work and returns
a ``job_id``; callers poll ``GET /api/jobs/{id}`` or stream
``GET /api/jobs/{id}/events`` (SSE) for live status. The queue's single worker
keeps GPU stages serial (PRD §7.1).

Run:

    uv run uvicorn clipforge.web.app:app
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from sqlite3 import Connection

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import db
from ..config import Config
from ..queue import JobQueue
from .schemas import (
    ClipOut,
    CreateJobRequest,
    CreateJobResponse,
    JobDetail,
    JobOut,
    PatchClipRequest,
)

_WEB_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_WEB_DIR / "templates"))

# Terminal job statuses at which the SSE stream closes.
_TERMINAL = {"DONE", "FAILED"}


def create_app(cfg: Config | None = None) -> FastAPI:
    """Build the ClipForge FastAPI app.

    ``cfg`` is mainly for tests (point ``data_dir`` at a tmp path); in production
    it is loaded from the environment at startup. The job-queue worker is started
    in the lifespan handler and stopped on shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.cfg = cfg or Config.load()
        app.state.queue = JobQueue(app.state.cfg)
        await app.state.queue.start()
        try:
            yield
        finally:
            await app.state.queue.stop()

    app = FastAPI(title="ClipForge", lifespan=lifespan)

    static_dir = _WEB_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _register_routes(app)
    return app


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def get_cfg(request: Request) -> Config:
    return request.app.state.cfg


def get_queue(request: Request) -> JobQueue:
    return request.app.state.queue


def get_conn(cfg: Config = Depends(get_cfg)) -> Iterator[Connection]:
    """Yield a short-lived DB connection, closed when the request ends."""
    conn = db.connect(cfg)
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


def _register_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "index.html")

    @app.get("/health")
    def health(cfg: Config = Depends(get_cfg)) -> dict[str, object]:
        return {
            "status": "ok",
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "data_dir": str(cfg.data_path),
        }

    # -- jobs ------------------------------------------------------------- #

    @app.post("/api/jobs", response_model=CreateJobResponse, status_code=201)
    async def create_job(
        body: CreateJobRequest,
        queue: JobQueue = Depends(get_queue),
    ) -> CreateJobResponse:
        url = body.url.strip()
        if not url:
            raise HTTPException(status_code=422, detail="url must not be empty")
        # enqueue() runs on the event loop (async handler) so the asyncio.Queue
        # put is thread-safe; it returns immediately with a QUEUED job id.
        job_id = queue.enqueue(url)
        return CreateJobResponse(job_id=job_id)

    @app.get("/api/jobs", response_model=list[JobOut])
    def list_jobs(conn: Connection = Depends(get_conn)) -> list[JobOut]:
        return [JobOut.from_row(row) for row in db.list_jobs(conn)]

    @app.get("/api/jobs/{job_id}", response_model=JobDetail)
    def get_job(job_id: str, conn: Connection = Depends(get_conn)) -> JobDetail:
        job = db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobDetail.from_rows(job, db.list_clips(conn, job_id))

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        cfg: Config = Depends(get_cfg),
    ) -> EventSourceResponse:
        conn = db.connect(cfg)
        try:
            if db.get_job(conn, job_id) is None:
                raise HTTPException(status_code=404, detail="job not found")
        finally:
            conn.close()

        async def stream() -> AsyncIterator[dict[str, str]]:
            last: tuple[str, float] | None = None
            while True:
                if await request.is_disconnected():
                    break
                probe = db.connect(cfg)
                try:
                    job = db.get_job(probe, job_id)
                finally:
                    probe.close()
                if job is None:
                    yield {"event": "error", "data": json.dumps({"error": "job vanished"})}
                    break
                key = (job["status"], job["progress"] or 0.0)
                if key != last:
                    last = key
                    yield {
                        "event": "progress",
                        "data": json.dumps(
                            {
                                "status": job["status"],
                                "progress": job["progress"] or 0.0,
                                "error": job.get("error"),
                            }
                        ),
                    }
                if job["status"] in _TERMINAL:
                    break
                await asyncio.sleep(1.0)

        return EventSourceResponse(stream())

    # -- clips ------------------------------------------------------------ #

    @app.get("/api/clips/{clip_id}/file")
    def clip_file(clip_id: int, conn: Connection = Depends(get_conn)) -> FileResponse:
        return _serve_clip_artifact(conn, clip_id, "file_path", "video/mp4")

    @app.get("/api/clips/{clip_id}/thumb")
    def clip_thumb(clip_id: int, conn: Connection = Depends(get_conn)) -> FileResponse:
        return _serve_clip_artifact(conn, clip_id, "thumb_path", "image/jpeg")

    @app.patch("/api/clips/{clip_id}", response_model=ClipOut)
    def patch_clip(
        clip_id: int,
        body: PatchClipRequest,
        conn: Connection = Depends(get_conn),
    ) -> ClipOut:
        if db.get_clip(conn, clip_id) is None:
            raise HTTPException(status_code=404, detail="clip not found")
        db.update_clip(conn, clip_id, hook_caption=body.hook_caption)
        updated = db.get_clip(conn, clip_id)
        assert updated is not None  # just written above
        return ClipOut.from_row(updated)

    @app.post("/api/clips/{clip_id}/rerender", status_code=202)
    def rerender_clip(
        clip_id: int,
        conn: Connection = Depends(get_conn),
        queue: JobQueue = Depends(get_queue),
    ) -> dict[str, object]:
        clip = db.get_clip(conn, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        if not clip.get("file_path"):
            raise HTTPException(status_code=409, detail="clip has not been rendered yet")
        queue.enqueue_rerender(clip_id)
        return {"clip_id": clip_id, "status": "rendering"}


def _serve_clip_artifact(
    conn: Connection, clip_id: int, path_key: str, media_type: str
) -> FileResponse:
    """Serve a clip's on-disk mp4/jpg, raising 404 if the clip or file is absent."""
    clip = db.get_clip(conn, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    raw = clip.get(path_key)
    if not raw or not Path(raw).is_file():
        raise HTTPException(status_code=404, detail="file not available")
    return FileResponse(raw, media_type=media_type)


app = create_app()
