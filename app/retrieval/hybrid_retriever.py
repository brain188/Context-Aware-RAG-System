"""Hybrid semantic + keyword retrieval in a single Qdrant query.

One Query API call runs two filtered prefetches — dense (bge embedding) and
sparse (BM25, IDF applied by Qdrant) — and fuses them server-side with
reciprocal-rank fusion. Returns ScoredChunks; no generation or reranking here.
RETRIEVAL_MODE=dense or sparse keeps one branch, to measure each one alone.
Requires Qdrant >= 1.16 (parameterized RRF).
"""

import asyncio
import time
from typing import Any

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Filter, Prefetch, Rrf, RrfQuery

from app.config import Settings, get_logger
from app.models import Chunk, ChunkMetadata, RetrievalFilter, ScoredChunk
from app.retrieval.errors import VectorStoreError
from app.retrieval.filters import to_qdrant_filter
from app.retrieval.query_embedder import QueryEmbedder
from app.retrieval.sparse_encoder import BM25SparseEncoder

logger = get_logger(__name__)


def _payload_to_chunk(point_id: Any, payload: dict[str, Any] | None) -> Chunk:
    """Rebuild a Chunk from a Qdrant point payload written by the pipeline.

    Raises:
        VectorStoreError: if the payload is missing text or metadata fields.
    """
    payload = payload or {}
    try:
        return Chunk(
            chunk_id=str(point_id),
            text=payload["text"],
            metadata=ChunkMetadata.model_validate(payload),
        )
    except (KeyError, ValidationError) as exc:
        raise VectorStoreError(f"Point {point_id} has a malformed payload: {exc}") from exc


def _describe_qdrant_error(exc: UnexpectedResponse, collection: str) -> str:
    """Turn the Qdrant status codes we have hit in practice into actionable messages."""
    if exc.status_code == 404:
        return (
            f"Qdrant returned 404 for collection '{collection}': the collection does not "
            "exist, or the server is older than v1.10 (no Query API)"
        )
    if exc.status_code == 400:
        return (
            f"Qdrant rejected the hybrid query ({exc}). Check the server is >= v1.16 and "
            f"that '{collection}' has the dense + bm25 sparse schema; re-ingest after "
            "schema changes"
        )
    return f"Qdrant hybrid query failed: {exc}"


class HybridRetriever:
    """Retrieve top candidates by fusing dense and BM25 rankings inside Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: QueryEmbedder,
        sparse_encoder: BM25SparseEncoder,
        settings: Settings,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._collection = settings.qdrant_collection
        self._dense_name = settings.qdrant_dense_vector_name
        self._sparse_name = settings.qdrant_sparse_vector_name
        self._candidate_k = settings.retrieval_top_k
        self._rrf_k = settings.rrf_k
        self._mode = settings.retrieval_mode

    async def retrieve(
        self,
        query: str,
        request_id: str,
        filters: RetrievalFilter | None = None,
        limit: int | None = None,
        query_vector: list[float] | None = None,
    ) -> list[ScoredChunk]:
        """Return up to *limit* fused candidates (default RETRIEVAL_TOP_K), best first.

        Scores are RRF scores. Metadata filters scope both prefetches. Pass
        *query_vector* when the caller already embedded *query* (e.g. for the
        semantic cache) to skip embedding it again.

        Raises:
            ValueError: if *query* is blank.
            VectorStoreError: if Qdrant fails or returns malformed points.
            EmbeddingError: if the query cannot be embedded.
        """
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")
        limit = limit or self._candidate_k
        start = time.perf_counter()

        qdrant_filter = to_qdrant_filter(filters)
        prefetch = []
        if self._mode != "sparse":
            dense = query_vector
            if dense is None:
                loop = asyncio.get_running_loop()
                dense = await loop.run_in_executor(None, self._embedder.embed_query, query)
            prefetch.append(self._prefetch(dense, self._dense_name, qdrant_filter))
        sparse = self._sparse.encode_query(query)
        # A query with no word tokens (e.g. only symbols) has nothing to match lexically.
        if self._mode != "dense" and sparse.indices:
            prefetch.append(self._prefetch(sparse, self._sparse_name, qdrant_filter))
        if not prefetch:
            logger.info(
                "Hybrid retrieval: no keyword tokens and retrieval_mode=sparse; nothing to search",
                extra={"request_id": request_id},
            )
            return []

        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=RrfQuery(rrf=Rrf(k=self._rrf_k)),
                limit=limit,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(_describe_qdrant_error(exc, self._collection)) from exc
        except ResponseHandlingException as exc:
            raise VectorStoreError(f"Qdrant hybrid query failed: {exc}") from exc

        results = [
            ScoredChunk(chunk=_payload_to_chunk(p.id, p.payload), score=p.score)
            for p in response.points
        ]
        logger.info(
            "Hybrid retrieval: prefetch=%s per_branch_limit=%d fused=%d filtered=%s "
            "latency_ms=%.0f",
            "+".join(p.using for p in prefetch),
            self._candidate_k,
            len(results),
            qdrant_filter is not None,
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )
        return results

    def _prefetch(self, query: Any, using: str, qdrant_filter: Filter | None) -> Prefetch:
        return Prefetch(query=query, using=using, filter=qdrant_filter, limit=self._candidate_k)
