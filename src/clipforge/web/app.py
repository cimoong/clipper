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
import shutil
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from sqlite3 import Connection

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import db, pipeline
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


# --------------------------------------------------------------------------- #
# Template view-models / rendering
#
# The HTML pages and the htmx fragment endpoints (job rows, status cells, clip
# cards, caption blocks) all render from these plain dicts so a job/clip row
# looks the same whether it is drawn on first paint or pushed over SSE.
# --------------------------------------------------------------------------- #


def _render(name: str, **context: object) -> str:
    """Render a Jinja partial to an HTML string (for htmx fragment responses)."""
    return _TEMPLATES.get_template(name).render(**context)


def _job_vm(row: Mapping[str, object]) -> dict[str, object]:
    """Shape a raw ``jobs`` row for the dashboard templates."""
    status = str(row["status"])
    error_stage, error_message = _split_error(row.get("error"))
    return {
        "id": row["id"],
        "id_short": str(row["id"])[:8],
        "label": row.get("title") or row.get("source_url") or "",
        "status": status,
        "progress": round(float(row.get("progress") or 0.0)),
        "is_terminal": status in _TERMINAL,
        "is_done": status == "DONE",
        "is_failed": status == "FAILED",
        "error": error_message,
        "error_stage": error_stage,
    }


def _split_error(error: object) -> tuple[str, str]:
    """Split a stored ``"[stage] message"`` error into ``(stage, message)``.

    The queue records failures as ``[<stage>] <message>`` (see
    :meth:`JobQueue._process`); this recovers the stage for the UI's error card.
    Anything without that prefix yields an empty stage and the raw message.
    """
    if not error:
        return "", ""
    text = str(error)
    if text.startswith("[") and "]" in text:
        stage, _, message = text[1:].partition("]")
        return stage.strip(), message.strip()
    return "", text


def _clip_vm(row: Mapping[str, object]) -> dict[str, object]:
    """Shape a raw ``clips`` row for the results-page templates."""
    score = row.get("score")
    return {
        "id": row["id"],
        "title": row.get("title") or "Untitled clip",
        "hook_caption": row.get("hook_caption") or "",
        "score": score,
        "score_class": _score_class(score),
        "duration_str": _fmt_duration(row.get("start_s"), row.get("end_s")),
        "start_str": _mmss(row.get("start_s")),
        "has_file": bool(row.get("file_path")),
        "has_thumb": bool(row.get("thumb_path")),
        "status": row.get("status") or "ready",
    }


def _score_class(score: object) -> str:
    """Colour bucket for the score badge: >=80 green, 60-79 yellow, else gray."""
    if score is None:
        return "low"
    value = int(score)  # type: ignore[arg-type]
    if value >= 80:
        return "high"
    if value >= 60:
        return "mid"
    return "low"


def _mmss(seconds: object) -> str:
    """Format a second offset as ``M:SS`` (``—`` when unknown)."""
    if seconds is None:
        return "—"
    total = int(float(seconds))  # type: ignore[arg-type]
    return f"{total // 60}:{total % 60:02d}"


def _fmt_duration(start: object, end: object) -> str:
    """Format a clip length from its start/end offsets (``—`` when unknown)."""
    if start is None or end is None:
        return "—"
    return _mmss(float(end) - float(start))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

# Options offered by the settings form (kept in sync with the pipeline's
# validators). LLM provider is limited to what the pipeline actually supports.
_PROVIDERS = ["gemini"]
_WHISPER_MODELS = ["tiny", "base", "small", "medium"]
_CAPTION_STYLES = sorted(pipeline.VALID_CAPTION_STYLES)
_HOOK_MODES = ["3s", "full"]


def _mask_key(value: str) -> str:
    """Mask an API key for display (never render it in full)."""
    if not value:
        return "(not set — configure GEMINI_API_KEY in .env)"
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}"


def _settings_vm(conn: Connection, cfg: Config) -> dict[str, object]:
    """Current effective settings (saved value, else the Config/env default)."""

    def val(key: str, default: object) -> object:
        stored = db.get_setting(conn, key)
        return stored if stored not in (None, "") else default

    return {
        "llm_provider": val("llm_provider", cfg.llm_provider),
        "gemini_model": val("gemini_model", cfg.gemini_model),
        "whisper_model": val("whisper_model", cfg.whisper_model),
        "num_clips": int(val("num_clips", cfg.num_clips)),  # type: ignore[arg-type]
        "clip_min_s": int(val("clip_min_s", cfg.clip_min_s)),  # type: ignore[arg-type]
        "clip_max_s": int(val("clip_max_s", cfg.clip_max_s)),  # type: ignore[arg-type]
        "caption_style": val("caption_style", "bold"),
        "hook_mode": val("hook_mode", "3s"),
        "api_key_masked": _mask_key(cfg.gemini_api_key),
        "providers": _PROVIDERS,
        "whisper_models": _WHISPER_MODELS,
        "caption_styles": _CAPTION_STYLES,
        "hook_modes": _HOOK_MODES,
    }


def _validate_settings(
    *,
    num_clips: int,
    clip_min_s: int,
    clip_max_s: int,
    whisper_model: str,
    caption_style: str,
    hook_mode: str,
    llm_provider: str,
) -> str | None:
    """Return an error message for invalid settings, or ``None`` if all valid."""
    if num_clips < 1:
        return "Number of clips must be at least 1."
    if clip_min_s < 1 or clip_max_s < 1:
        return "Clip durations must be positive."
    if clip_min_s >= clip_max_s:
        return "Minimum clip duration must be less than the maximum."
    if whisper_model not in _WHISPER_MODELS:
        return f"Whisper model must be one of: {', '.join(_WHISPER_MODELS)}."
    if caption_style not in _CAPTION_STYLES:
        return f"Caption style must be one of: {', '.join(_CAPTION_STYLES)}."
    if hook_mode not in _HOOK_MODES:
        return f"Hook mode must be one of: {', '.join(_HOOK_MODES)}."
    if llm_provider not in _PROVIDERS:
        return f"LLM provider must be one of: {', '.join(_PROVIDERS)}."
    return None


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
    def index(request: Request, conn: Connection = Depends(get_conn)) -> HTMLResponse:
        jobs = [_job_vm(row) for row in db.list_jobs(conn)]
        return _TEMPLATES.TemplateResponse(request, "index.html", {"jobs": jobs})

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(
        job_id: str, request: Request, conn: Connection = Depends(get_conn)
    ) -> HTMLResponse:
        job = db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        clips = [_clip_vm(row) for row in db.list_clips(conn, job_id)]
        return _TEMPLATES.TemplateResponse(
            request, "job.html", {"job": _job_vm(job), "clips": clips}
        )

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(
        request: Request,
        conn: Connection = Depends(get_conn),
        cfg: Config = Depends(get_cfg),
    ) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "settings.html", {"s": _settings_vm(conn, cfg)})

    @app.post("/settings", response_class=HTMLResponse)
    def save_settings(
        request: Request,
        llm_provider: str = Form(...),
        gemini_model: str = Form(...),
        whisper_model: str = Form(...),
        num_clips: int = Form(...),
        clip_min_s: int = Form(...),
        clip_max_s: int = Form(...),
        caption_style: str = Form(...),
        hook_mode: str = Form(...),
        conn: Connection = Depends(get_conn),
        cfg: Config = Depends(get_cfg),
    ) -> HTMLResponse:
        """Validate + persist settings to the ``settings`` table (API key never stored)."""
        error = _validate_settings(
            num_clips=num_clips,
            clip_min_s=clip_min_s,
            clip_max_s=clip_max_s,
            whisper_model=whisper_model,
            caption_style=caption_style,
            hook_mode=hook_mode,
            llm_provider=llm_provider,
        )
        if error is None:
            values = {
                "llm_provider": llm_provider.strip(),
                "gemini_model": gemini_model.strip(),
                "whisper_model": whisper_model,
                "num_clips": str(num_clips),
                "clip_min_s": str(clip_min_s),
                "clip_max_s": str(clip_max_s),
                "caption_style": caption_style,
                "hook_mode": hook_mode,
            }
            for key, value in values.items():
                db.set_setting(conn, key, value)
        return _TEMPLATES.TemplateResponse(
            request,
            "settings.html",
            {"s": _settings_vm(conn, cfg), "saved": error is None, "error": error},
        )

    @app.post("/jobs", response_class=HTMLResponse)
    def create_job_form(
        url: str = Form(...),
        conn: Connection = Depends(get_conn),
        queue: JobQueue = Depends(get_queue),
    ) -> HTMLResponse:
        """htmx form target: enqueue a job and return its dashboard row fragment."""
        url = url.strip()
        if not url:
            raise HTTPException(status_code=422, detail="url must not be empty")
        job_id = queue.enqueue(url)
        row = db.get_job(conn, job_id)
        assert row is not None  # just inserted by enqueue
        return HTMLResponse(_render("_job_row.html", job=_job_vm(row)))

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

    @app.post("/api/jobs/{job_id}/retry", response_class=HTMLResponse)
    def retry_job(
        job_id: str,
        conn: Connection = Depends(get_conn),
        queue: JobQueue = Depends(get_queue),
    ) -> HTMLResponse:
        """Resume a FAILED job from its first incomplete stage (via state.json).

        Returns the refreshed dashboard row fragment so htmx can swap it in — now
        QUEUED, with a live SSE connection that shows the resume progressing.
        """
        job = db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "FAILED":
            raise HTTPException(status_code=409, detail="only failed jobs can be retried")
        queue.enqueue_resume(job_id)
        updated = db.get_job(conn, job_id)
        assert updated is not None
        return HTMLResponse(_render("_job_row.html", job=_job_vm(updated)))

    @app.post("/api/jobs/{job_id}/reanalyze", status_code=202)
    def reanalyze_job(
        job_id: str,
        conn: Connection = Depends(get_conn),
        queue: JobQueue = Depends(get_queue),
        cfg: Config = Depends(get_cfg),
    ) -> JSONResponse:
        """Create a NEW job version that re-runs analyze -> cut -> render from the
        original's cached transcript with the current settings (no re-download /
        re-transcribe). Returns the new job id and an ``HX-Redirect`` to it.
        """
        orig = db.get_job(conn, job_id)
        if orig is None:
            raise HTTPException(status_code=404, detail="job not found")

        source_url = orig.get("source_url") or ""
        base_title = orig.get("title") or source_url or job_id
        new_id = pipeline.new_job_id()
        try:
            pipeline.setup_reanalyze_job(
                job_id,
                cfg,
                source_url=source_url,
                title=f"{base_title} (re-analyze)",
                new_id=new_id,
            )
        except pipeline.PipelineError as exc:
            # Transcript / source video missing (e.g. old sources were cleaned up).
            raise HTTPException(status_code=409, detail=exc.message) from exc

        db.create_job(
            conn,
            id=new_id,
            source_url=source_url,
            source_path=orig.get("source_path"),
            title=f"{base_title} (re-analyze)",
            params={"reanalyze_of": job_id},
            status="QUEUED",
        )
        # Resume from the first incomplete stage (analyze) on the serial worker.
        queue.enqueue_resume(new_id)

        resp = JSONResponse({"job_id": new_id, "source_job": job_id}, status_code=202)
        resp.headers["HX-Redirect"] = f"/jobs/{new_id}"
        return resp

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

        # The `progress` event carries a rendered `_status.html` fragment so
        # htmx's sse-swap can drop it straight into the row's status cell; a
        # final `close` event tells the sse extension to stop (no reconnect).
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
                    yield {"event": "close", "data": ""}
                    break
                key = (job["status"], job["progress"] or 0.0)
                if key != last:
                    last = key
                    yield {
                        "event": "progress",
                        "data": _render("_status.html", job=_job_vm(job)),
                    }
                if job["status"] in _TERMINAL:
                    yield {"event": "close", "data": ""}
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

    # -- clips: htmx HTML fragments --------------------------------------- #

    @app.patch("/clips/{clip_id}/caption", response_class=HTMLResponse)
    def patch_caption_form(
        clip_id: int,
        hook_caption: str = Form(""),
        conn: Connection = Depends(get_conn),
    ) -> HTMLResponse:
        """htmx caption edit: save the new caption, return the block with a
        Re-render button now revealed."""
        if db.get_clip(conn, clip_id) is None:
            raise HTTPException(status_code=404, detail="clip not found")
        db.update_clip(conn, clip_id, hook_caption=hook_caption)
        updated = db.get_clip(conn, clip_id)
        assert updated is not None
        return HTMLResponse(
            _render(
                "_caption.html",
                clip=_clip_vm(updated),
                show_rerender=True,
                note="Saved.",
            )
        )

    @app.post("/clips/{clip_id}/rerender", response_class=HTMLResponse)
    def rerender_clip_form(
        clip_id: int,
        conn: Connection = Depends(get_conn),
        queue: JobQueue = Depends(get_queue),
    ) -> HTMLResponse:
        """htmx Re-render button: queue a single-clip re-render (or explain why
        it can't yet), returning the refreshed caption block."""
        clip = db.get_clip(conn, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        if not clip.get("file_path"):
            return HTMLResponse(
                _render(
                    "_caption.html",
                    clip=_clip_vm(clip),
                    show_rerender=True,
                    note="Nothing rendered yet to re-render.",
                )
            )
        queue.enqueue_rerender(clip_id)
        updated = db.get_clip(conn, clip_id)
        assert updated is not None
        return HTMLResponse(
            _render(
                "_caption.html",
                clip=_clip_vm(updated),
                show_rerender=False,
                note="Re-render queued…",
            )
        )


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
