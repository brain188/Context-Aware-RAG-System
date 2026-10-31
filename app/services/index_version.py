"""Identify the state of the index, so cached answers never outlive it.

The version combines a fingerprint of every setting that shapes vectors or
answers with the collection's exact point count, which changes whenever a
book is added, removed, or re-ingested with different chunking. When
pipeline/index_versioning.py lands, the versioned collection behind the alias
replaces the point count.
"""

import hashlib
import json

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings
from app.retrieval.errors import VectorStoreError

_FINGERPRINT_FIELDS = (
    "qdrant_collection",
    "qdrant_dense_vector_name",
    "qdrant_sparse_vector_name",
    "embedding_model",
    "embedding_dimension",
    "embedding_query_instruction",
    "chunk_min_tokens",
    "chunk_max_tokens",
    "chunk_overlap_tokens",
    "bm25_k1",
    "bm25_b",
    "bm25_avg_doc_length",
    "reranker_model",
    "rerank_top_k",
    "llm_model",
    "prompt_version",
)


def settings_fingerprint(settings: Settings) -> str:
    """Short stable hash of the settings that change what an answer would be."""
    values = {field: getattr(settings, field) for field in _FINGERPRINT_FIELDS}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:12]


class IndexVersionProvider:
    """Compute the current index version from settings + live collection state."""

    def __init__(self, client: AsyncQdrantClient, settings: Settings) -> None:
        self._client = client
        self._collection = settings.qdrant_collection
        self._fingerprint = settings_fingerprint(settings)

    async def current(self) -> str:
        """Return "<fingerprint>-<point count>".

        Raises:
            VectorStoreError: if Qdrant cannot be queried.
        """
        try:
            result = await self._client.count(self._collection, exact=True)
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise VectorStoreError(f"Could not read index version: {exc}") from exc
        return f"{self._fingerprint}-{result.count}"
