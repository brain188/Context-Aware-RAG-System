"""POST /api/ingest (start ingesting one book) and GET /api/ingest/{job_id} (job status)."""

from fastapi import APIRouter, Depends, Request

from app.dependencies import Services, get_services
from app.errors import NotFoundError
from app.models import ErrorResponse, IngestJob, IngestRequest

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post(
    "",
    response_model=IngestJob,
    status_code=202,
    responses={400: {"model": ErrorResponse}, 501: {"model": ErrorResponse}},
)
async def start_ingest(
    body: IngestRequest, request: Request, services: Services = Depends(get_services)
) -> IngestJob:
    """Queue a PDF from the books folder for ingestion; poll the returned job for progress."""
    return await services.ingest_jobs.submit(body, request.state.request_id)


@router.get("/{job_id}", response_model=IngestJob, responses={404: {"model": ErrorResponse}})
async def ingest_status(job_id: str, services: Services = Depends(get_services)) -> IngestJob:
    """Current state of an ingest job started by this process."""
    job = services.ingest_jobs.get(job_id)
    if job is None:
        raise NotFoundError(f"No ingest job {job_id}")
    return job
