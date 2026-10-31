"""GET /api/library (indexed books and chapters) and GET /api/chunks/{id} (a cited chunk's text)."""

from uuid import UUID

from fastapi import APIRouter, Depends

from app.dependencies import Services, get_services
from app.errors import NotFoundError
from app.models import ChunkSource, ErrorResponse, LibraryResponse

router = APIRouter(prefix="/api", tags=["library"])


@router.get("/library", response_model=LibraryResponse, responses={503: {"model": ErrorResponse}})
async def library(services: Services = Depends(get_services)) -> LibraryResponse:
    """Every indexed book with its chapters, for building filters."""
    return LibraryResponse(books=await services.catalog.books())


@router.get(
    "/chunks/{chunk_id}",
    response_model=ChunkSource,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def chunk(chunk_id: UUID, services: Services = Depends(get_services)) -> ChunkSource:
    """The stored text and location of one chunk, for checking a citation."""
    source = await services.catalog.chunk(chunk_id)
    if source is None:
        raise NotFoundError(f"No chunk {chunk_id}")
    return source
