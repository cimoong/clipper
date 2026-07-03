"""Asyncio job queue for ClipForge.

A single background task drains a FIFO queue and runs one pipeline job at a
time. Serial execution is a hard requirement, not a convenience: transcription
and NVENC encoding both need the GTX 1060's 6 GB of VRAM and must never run
concurrently (PRD §7.1).

Flow:

    enqueue(url) -> job_id           # inserts a QUEUED row, returns immediately
    worker: for each job_id FIFO ->
        run pipeline, writing status + percent to the jobs table before every
        stage (via the pipeline's progress callback); on success record clips
        and mark DONE; on any failure store the error and mark FAILED.

Each database write opens its own short-lived connection so the progress
callback — which the pipeline invokes from an executor thread — never shares a
connection across threads.

Run standalone to process a single source through the queue:

    python -m clipforge.queue <url_or_path> [--no-llm]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from .config import Config
from .download import check_ytdlp_freshness
from .pipeline import PipelineError, new_job_id, rerender_clip, resume_job, run_new

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NewJob:
    """Queue item: run the full pipeline for a freshly enqueued job."""

    job_id: str


@dataclass(frozen=True)
class _Resume:
    """Queue item: resume a failed/interrupted job from its first incomplete stage."""

    job_id: str


@dataclass(frozen=True)
class _Rerender:
    """Queue item: re-render a single already-rendered clip."""

    clip_id: int


_Task = _NewJob | _Resume | _Rerender

# Maps a pipeline stage name to the job status persisted while it runs. Mirrors
# the PRD user-flow states; ``cut`` gets its own CUTTING state for clarity.
_STAGE_STATUS: dict[str, str] = {
    "download": "DOWNLOADING",
    "transcribe": "TRANSCRIBING",
    "analyze": "ANALYZING",
    "cut": "CUTTING",
    "render": "RENDERING",
}


class JobQueue:
    """A FIFO, single-worker queue that runs pipeline jobs one at a time."""

    def __init__(self, cfg: Config, *, no_llm: bool = False) -> None:
        self._cfg = cfg
        self._no_llm = no_llm
        self._queue: asyncio.Queue[_Task] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        # Ensure the schema exists before any enqueue writes a row.
        conn = db.connect(cfg)
        conn.close()

    # -- lifecycle -------------------------------------------------------- #

    async def start(self) -> None:
        """Start the background worker if it is not already running."""
        if self._worker is None or self._worker.done():
            warning = check_ytdlp_freshness()
            if warning:
                logger.warning(warning)
            self._worker = asyncio.create_task(self._run(), name="clipforge-queue-worker")
            logger.info("Job queue worker started.")

    async def stop(self) -> None:
        """Cancel the worker and wait for it to unwind."""
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
            logger.info("Job queue worker stopped.")

    async def join(self) -> None:
        """Block until every enqueued job has been processed."""
        await self._queue.join()

    # -- producer --------------------------------------------------------- #

    def enqueue(
        self,
        source_url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Record a new QUEUED job and hand it to the worker; return its id.

        ``source_url`` may be a YouTube URL or a local file path (the download
        stage detects which). Returns immediately — processing happens on the
        worker task. Call :meth:`start` once beforehand so the job is picked up.
        """
        job_id = new_job_id()
        source_path = source_url if _is_local_path(source_url) else None

        conn = db.connect(self._cfg)
        try:
            db.create_job(
                conn,
                id=job_id,
                source_url=source_url,
                source_path=source_path,
                params=params,
                status="QUEUED",
            )
        finally:
            conn.close()

        self._queue.put_nowait(_NewJob(job_id))
        logger.info("Enqueued job %s for %r", job_id, source_url)
        return job_id

    def enqueue_resume(self, job_id: str) -> None:
        """Queue a resume of a failed/interrupted job on the same worker.

        The job is flipped back to ``QUEUED`` immediately so the UI stops showing
        the failed state; the pipeline then continues from its first incomplete
        stage using the on-disk ``state.json`` checkpoint.
        """
        self._write(lambda c: db.update_job_status(c, job_id, "QUEUED", progress=0.0))
        self._queue.put_nowait(_Resume(job_id))
        logger.info("Enqueued resume for job %s", job_id)

    def enqueue_rerender(self, clip_id: int) -> None:
        """Queue a single-clip re-render on the same worker (keeps GPU serial).

        The clip is marked ``rendering`` immediately so the API reflects the
        pending state; the actual reframe + encode happens on the worker task.
        """
        self._write(lambda c: db.update_clip(c, clip_id, status="rendering"))
        self._queue.put_nowait(_Rerender(clip_id))
        logger.info("Enqueued re-render for clip %s", clip_id)

    # -- worker ----------------------------------------------------------- #

    async def _run(self) -> None:
        """Drain the queue forever, processing one task at a time."""
        while True:
            task = await self._queue.get()
            try:
                if isinstance(task, _Rerender):
                    await self._process_rerender(task.clip_id)
                elif isinstance(task, _Resume):
                    await self._process(task.job_id, resume=True)
                else:
                    await self._process(task.job_id)
            except Exception:  # noqa: BLE001 - a single task must never kill the worker
                logger.exception("Unexpected failure processing task %r", task)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str, *, resume: bool = False) -> None:
        """Run (or resume) one job's pipeline, persisting live status + result.

        When ``resume`` is True the pipeline continues from the job's first
        incomplete stage via its ``state.json`` checkpoint instead of starting a
        fresh download; either way status/percent are written before each stage
        and the final clips are recorded on success.
        """
        job = self._read_job(job_id)
        if job is None:
            logger.error("Job %s vanished before processing; skipping.", job_id)
            return
        source_url = job["source_url"] or ""

        def on_stage(stage: str, index: int, total: int) -> None:
            # Runs in the pipeline's executor thread -> use a fresh connection.
            status = _STAGE_STATUS.get(stage, stage.upper())
            percent = round(index / total * 100, 1) if total else 0.0
            self._write(lambda c: db.update_job_status(c, job_id, status, progress=percent))

        if resume:
            # Honour the queue's offline flag when set; otherwise keep the value
            # saved in state.json (no_llm=None leaves it untouched).
            override = True if self._no_llm else None

            def runner() -> dict[str, Any]:
                return resume_job(job_id, self._cfg, no_llm=override, progress=on_stage)
        else:

            def runner() -> dict[str, Any]:
                return run_new(
                    source_url,
                    self._cfg,
                    no_llm=self._no_llm,
                    job_id=job_id,
                    progress=on_stage,
                )

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(None, runner)
        except PipelineError as exc:
            logger.error("Job %s failed at stage %r: %s", job_id, exc.stage, exc.message)
            message = f"[{exc.stage}] {exc.message}"
            self._write(lambda c: db.update_job_status(c, job_id, "FAILED", error=message))
            return
        except Exception as exc:  # noqa: BLE001 - normalize any non-pipeline failure
            logger.exception("Job %s failed", job_id)
            message = str(exc)
            self._write(lambda c: db.update_job_status(c, job_id, "FAILED", error=message))
            return

        self._finalize(job_id, source_url, results)

    async def _process_rerender(self, clip_id: int) -> None:
        """Re-render one clip, persisting its new file/thumb (or a failed status)."""
        clip = self._read_clip(clip_id)
        if clip is None:
            logger.error("Clip %s vanished before re-render; skipping.", clip_id)
            return
        index = _clip_index(clip.get("file_path"))
        if index is None:
            logger.error("Clip %s has no rendered file to re-render.", clip_id)
            self._write(lambda c: db.update_clip(c, clip_id, status="failed"))
            return

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: rerender_clip(
                    clip["job_id"],
                    index,
                    self._cfg,
                    hook_caption=clip.get("hook_caption") or "",
                    no_llm=self._no_llm,
                ),
            )
        except Exception:  # noqa: BLE001 - normalise any re-render failure
            logger.exception("Re-render of clip %s failed", clip_id)
            self._write(lambda c: db.update_clip(c, clip_id, status="failed"))
            return

        self._write(
            lambda c: db.update_clip(
                c,
                clip_id,
                status="ready",
                file_path=str(result.get("file")) if result.get("file") else None,
                thumb_path=str(result.get("thumb")) if result.get("thumb") else None,
            )
        )
        logger.info("Re-rendered clip %s (index %d).", clip_id, index)

    # -- persistence helpers --------------------------------------------- #

    def _finalize(self, job_id: str, source_url: str, results: dict[str, Any]) -> None:
        """Record clips + cached transcript and mark the job DONE."""
        clips = results.get("clips", []) if isinstance(results, dict) else []
        title = str(results.get("title", "")) if isinstance(results, dict) else ""

        def write(conn: Any) -> None:
            db.insert_clips(conn, job_id, clips)
            db.update_job_status(
                conn,
                job_id,
                "DONE",
                progress=100.0,
                title=title or None,
            )

        self._write(write)
        self._cache_transcript(job_id)
        logger.info("Job %s DONE — %d clip(s).", job_id, len(clips))

    def _cache_transcript(self, job_id: str) -> None:
        """Best-effort: copy the on-disk transcript into the transcripts table."""
        path = self._cfg.data_path / "sources" / job_id / "transcript.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return

        def write(conn: Any) -> None:
            db.insert_transcript(
                conn,
                job_id,
                segments=payload.get("segments"),
                words=payload.get("words"),
                model_used=str(payload.get("model_used", self._cfg.whisper_model)),
            )
            duration = payload.get("duration_s")
            language = payload.get("language")
            if duration is not None or language is not None:
                db.update_job_status(
                    conn,
                    job_id,
                    "DONE",
                    duration_s=float(duration) if duration is not None else None,
                    language=str(language) if language is not None else None,
                )

        try:
            self._write(write)
        except Exception:  # noqa: BLE001 - caching is optional, never fail the job
            logger.warning("Could not cache transcript for job %s", job_id, exc_info=True)

    def _read_job(self, job_id: str) -> dict[str, Any] | None:
        conn = db.connect(self._cfg)
        try:
            return db.get_job(conn, job_id)
        finally:
            conn.close()

    def _read_clip(self, clip_id: int) -> dict[str, Any] | None:
        conn = db.connect(self._cfg)
        try:
            return db.get_clip(conn, clip_id)
        finally:
            conn.close()

    def _write(self, fn: Any) -> None:
        """Run ``fn(conn)`` against a fresh, short-lived connection."""
        conn = db.connect(self._cfg)
        try:
            fn(conn)
        finally:
            conn.close()


def _clip_index(file_path: str | None) -> int | None:
    """Recover a clip's candidate index from its ``clip_NN.mp4`` file name."""
    if not file_path:
        return None
    stem = Path(file_path).stem  # e.g. "clip_03"
    _, _, tail = stem.partition("clip_")
    try:
        return int(tail)
    except ValueError:
        return None


def _is_local_path(source: str) -> bool:
    try:
        return Path(source).is_file()
    except OSError:
        return False


async def _run_one(source: str, *, no_llm: bool) -> int:
    cfg = Config.load()
    queue = JobQueue(cfg, no_llm=no_llm)
    await queue.start()
    job_id = queue.enqueue(source)
    await queue.join()
    await queue.stop()

    conn = db.connect(cfg)
    try:
        job = db.get_job(conn, job_id)
        clips = db.list_clips(conn, job_id)
    finally:
        conn.close()

    if job is None:
        print(f"error: job {job_id} not found after processing", file=sys.stderr)
        return 1
    print(f"Job {job_id}: {job['status']}")
    if job["status"] == "FAILED":
        print(f"  error: {job.get('error')}", file=sys.stderr)
        return 1
    for clip in clips:
        print(f"  [{clip['score']}] {clip['start_s']}-{clip['end_s']}s  {clip['title']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    no_llm = "--no-llm" in args
    positional = [a for a in args if not a.startswith("-")]
    if len(positional) != 1:
        print("usage: python -m clipforge.queue <url_or_path> [--no-llm]", file=sys.stderr)
        return 2
    return asyncio.run(_run_one(positional[0], no_llm=no_llm))


if __name__ == "__main__":
    raise SystemExit(main())
