"""What is indexed: books, their chapters, and individual chunks.

Feeds the UI's filters, the query router's book list, and citation checking.
Book and chapter listing use Qdrant facets, which on a server need the keyword
payload indexes that pipeline/vector_store.py creates.
"""

from uuid import UUID

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import Settings, get_logger
from app.models import BookInfo, ChunkMetadata, ChunkSource
from app.retrieval.errors import VectorStoreError

logger = get_logger(__name__)

# Facet page size; a personal library has far fewer books or chapters per book.
_FACET_LIMIT = 10_000


class LibraryCatalog:
    """Read-only views over the indexed collection."""

    def __init__(self, client: AsyncQdrantClient, settings: Settings) -> None:
        self._client = client
        self._collection = settings.qdrant_collection

    async def books(self) -> list[BookInfo]:
        """Every indexed book with its chunk count and chapters, sorted by title.

        Raises:
            VectorStoreError: if Qdrant fails (e.g. payload indexes are missing).
        """
        book_hits = await self._facet("book")
        books = []
        for hit in book_hits:
            title = str(hit.value)
            chapter_hits = await self._facet(
                "chapter",
                Filter(must=[FieldCondition(key="book", match=MatchValue(value=title))]),
            )
            books.append(
                BookInfo(
                    title=title,
                    chunk_count=hit.count,
                    chapters=sorted(str(c.value) for c in chapter_hits),
                )
            )
        return sorted(books, key=lambda b: b.title.lower())

    async def titles(self, request_id: str) -> list[str]:
        """Book titles, or [] (WARNING) if they cannot be listed; used for routing only."""
        try:
            return [str(hit.value) for hit in await self._facet("book")]
        except VectorStoreError as exc:
            logger.warning(
                "Book titles unavailable for routing: %s", exc, extra={"request_id": request_id}
            )
            return []

    async def has_source(self, source_path: str) -> bool:
        """True if any chunk was ingested from *source_path*.

        Raises:
            VectorStoreError: if Qdrant cannot be queried.
        """
        try:
            result = await self._client.count(
                self._collection,
                count_filter=Filter(
                    must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
                ),
                exact=True,
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise VectorStoreError(f"Could not check for existing chunks: {exc}") from exc
        return result.count > 0

    async def chunk(self, chunk_id: UUID) -> ChunkSource | None:
        """The stored chunk with this id, or None if it does not exist.

        Raises:
            VectorStoreError: if Qdrant fails or the stored payload is malformed.
        """
        try:
            records = await self._client.retrieve(
                self._collection, ids=[str(chunk_id)], with_payload=True
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise VectorStoreError(f"Could not fetch chunk {chunk_id}: {exc}") from exc
        if not records:
            return None
        payload = records[0].payload or {}
        try:
            return ChunkSource(
                chunk_id=str(records[0].id),
                text=payload["text"],
                metadata=ChunkMetadata.model_validate(payload),
            )
        except (KeyError, ValidationError) as exc:
            raise VectorStoreError(f"Chunk {chunk_id} has a malformed payload: {exc}") from exc

    async def _facet(self, key: str, facet_filter: Filter | None = None) -> list:
        try:
            response = await self._client.facet(
                self._collection, key=key, facet_filter=facet_filter, limit=_FACET_LIMIT
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Could not list '{key}' values ({exc}). If the collection predates payload "
                "indexes, re-run the ingest pipeline once to create them."
            ) from exc
        except ResponseHandlingException as exc:
            raise VectorStoreError(f"Could not list '{key}' values: {exc}") from exc
        return response.hits
