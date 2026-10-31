"""Run single-book ingests in the background for POST /api/ingest.

Ingesting a book takes minutes (CPU-bound embedding), so the API starts a job
and returns its id. Jobs run one at a time in a worker thread, using the same
ingest_book() function as the CLI. Job state is in memory: it is lost on
restart, while the ingested chunks are not.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.config import Settings, get_logger
from app.errors import IngestRequestError, IngestUnavailableError
from app.models import IngestJob, IngestRequest, IngestResponse

logger = get_logger(__name__)

IngestRunner = Callable[[Path, str], dict[str, Any]]


def load_pipeline_runner() -> IngestRunner | None:
    """Return pipeline.ingest.ingest_book, or None when the pipeline is not installed."""
    try:
        from pipeline.ingest import ingest_book
    except ImportError as exc:
        logger.warning("Ingestion pipeline not importable (%s); /api/ingest is disabled.", exc)
        return None
    return ingest_book


class IngestJobManager:
    """Validate, queue, and track ingest jobs."""

    def __init__(
        self,
        runner: IngestRunner | None,
        settings: Settings,
        is_ingested: Callable[[str], Awaitable[bool]],
        on_success: Callable[[str], Awaitable[None]],
    ) -> None:
        self._runner = runner
        self._is_ingested = is_ingested
        self._books_dir = settings.ingest_books_dir.resolve()
        self._on_success = on_success
        self._jobs: dict[str, IngestJob] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()  # one ingest at a time: each saturates the CPU

    def resolve_source(self, source_path: str) -> Path:
        """Resolve *source_path* (absolute, or relative to the books folder) to a PDF inside it.

        Raises:
            IngestRequestError: if the path escapes the books folder, is missing, or is not a PDF.
        """
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = self._books_dir / candidate
        path = candidate.resolve()
        if not path.is_relative_to(self._books_dir):
            raise IngestRequestError(f"Only files inside {self._books_dir} can be ingested.")
        if path.suffix.lower() != ".pdf":
            raise IngestRequestError("Only PDF files can be ingested.")
        if not path.is_file():
            raise IngestRequestError(f"File not found: {path.name}")
        return path

    async def submit(self, request: IngestRequest, request_id: str) -> IngestJob:
        """Queue an ingest of one book and return its job.

        Raises:
            IngestUnavailableError: if the pipeline is not installed in this deployment.
            IngestRequestError: if the source path is not an ingestable PDF, or the book is
                already indexed and force_reingest is false.
        """
        if self._runner is None:
            raise IngestUnavailableError(
                "This deployment has no ingestion pipeline; run `python -m pipeline.ingest`."
            )
        path = self.resolve_source(request.source_path)
        if not request.force_reingest and await self._is_ingested(str(path)):
            raise IngestRequestError(
                f"{path.name} is already indexed; set force_reingest to replace it."
            )
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            book_title=request.book_title,
            source_path=str(path),
            state="queued",
        )
        self._jobs[job.job_id] = job
        task = asyncio.create_task(self._run(job, path, request_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info(
            "Ingest queued: job=%s book='%s'",
            job.job_id,
            job.book_title,
            extra={"request_id": request_id},
        )
        return job

    def get(self, job_id: str) -> IngestJob | None:
        """The job with this id, if this process has seen it."""
        return self._jobs.get(job_id)

    async def _run(self, job: IngestJob, path: Path, request_id: str) -> None:
        assert self._runner is not None
        log_extra = {"request_id": request_id}
        async with self._lock:
            job.state = "running"
            loop = asyncio.get_running_loop()
            try:
                summary = await loop.run_in_executor(None, self._runner, path, job.book_title)
                job.result = IngestResponse.model_validate(summary)
            except Exception as exc:
                # A background task has no caller to raise to: record and log the failure.
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                logger.error("Ingest failed: job=%s %s", job.job_id, job.error, extra=log_extra)
                return
        job.state = "succeeded" if job.result.status == "ok" else "failed"
        if job.state == "failed":
            job.error = job.result.detail
        logger.info(
            "Ingest finished: job=%s state=%s chunks=%d",
            job.job_id,
            job.state,
            job.result.chunks_indexed,
            extra=log_extra,
        )
        if job.state == "succeeded":
            await self._on_success(request_id)
