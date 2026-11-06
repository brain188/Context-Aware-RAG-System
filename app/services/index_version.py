"""Identify the state of the index, so cached answers never outlive it.

The version combines a fingerprint of every setting that shapes vectors or
answers with the collection's exact point count, which changes whenever a
book is added, removed, or re-ingested with different chunking. When
pipeline/index_versioning.py lands, the versioned collection behind the alias
replaces the point count.

The pipeline also stamps the collection with the embedder that wrote it
(Qdrant collection metadata); the app refuses to start if its own query
embedder differs, because mismatched vectors fail silently with bad results.
It stamps the BM25 tokenizer and settings too; a mismatch there only degrades
keyword search, so the app warns and `python -m pipeline.reindex_sparse` fixes it.
"""

import hashlib
import json
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings, get_logger
from app.retrieval.errors import VectorStoreError
from app.retrieval.sparse_encoder import SPARSE_ENCODER_VERSION

logger = get_logger(__name__)

# Settings that must be identical on the document side (ingest) and the query side.
# The query instruction is deliberately absent: it is query-only and needs no re-ingest.
EMBEDDER_STAMP_FIELDS = ("embedding_model", "embedding_dimension")
# The document side of keyword search: the tokenizer version and the BM25 TF settings.
SPARSE_STAMP_KEY = "sparse_encoder"
SPARSE_STAMP_FIELDS = ("bm25_k1", "bm25_b", "bm25_avg_doc_length")
# Collections written before the tokenizer was stamped used the first one.
LEGACY_SPARSE_ENCODER = "bm25-v1"

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
    "retrieval_mode",
    "reranker_model",
    "rerank_top_k",
    "llm_provider",
    "llm_generation_model",
    "prompt_version",
)


class IndexMismatchError(RuntimeError):
    """The collection was embedded with a different model than the one configured."""


def embedder_stamp(settings: Settings) -> dict[str, Any]:
    """The collection metadata the pipeline records about the embedder it used."""
    return {field: getattr(settings, field) for field in EMBEDDER_STAMP_FIELDS}


def stamp_problems(recorded: dict[str, Any], settings: Settings) -> list[str]:
    """List every way the *recorded* stamp differs from the configured embedder."""
    return [
        f"index {field}={recorded[field]!r} but {field.upper()}={value!r}"
        for field, value in embedder_stamp(settings).items()
        if field in recorded and recorded[field] != value
    ]


def sparse_stamp(settings: Settings) -> dict[str, Any]:
    """The collection metadata recording how its BM25 sparse vectors were built."""
    return {
        SPARSE_STAMP_KEY: SPARSE_ENCODER_VERSION,
        **{field: getattr(settings, field) for field in SPARSE_STAMP_FIELDS},
    }


def index_stamp(settings: Settings) -> dict[str, Any]:
    """Everything the pipeline records on a collection it writes."""
    return {**embedder_stamp(settings), **sparse_stamp(settings)}


def sparse_stamp_problems(recorded: dict[str, Any], settings: Settings) -> list[str]:
    """List every way the index's sparse vectors differ from what this code would write.

    A missing tokenizer version means the first one; missing BM25 settings are not compared.
    """
    version = recorded.get(SPARSE_STAMP_KEY, LEGACY_SPARSE_ENCODER)
    problems = []
    if version != SPARSE_ENCODER_VERSION:
        problems.append(f"index tokenizer {version!r} but the code's is {SPARSE_ENCODER_VERSION!r}")
    problems += [
        f"index {field}={recorded[field]!r} but {field.upper()}={getattr(settings, field)!r}"
        for field in SPARSE_STAMP_FIELDS
        if field in recorded and recorded[field] != getattr(settings, field)
    ]
    return problems


async def check_index_embedder(client: AsyncQdrantClient, settings: Settings) -> None:
    """Refuse to serve queries against an index built with another embedder.

    A missing collection (nothing ingested yet), an unstamped one (ingested
    before stamping existed), or an unreachable Qdrant only log a WARNING:
    health checks report those, and the first ingest stamps the collection.
    Keyword vectors built by another tokenizer or BM25 settings also only warn:
    the keyword side still matches most terms, and dense search is unaffected.

    Raises:
        IndexMismatchError: if the collection's recorded embedder differs.
    """
    name = settings.qdrant_collection
    try:
        if not await client.collection_exists(name):
            logger.warning("Collection '%s' does not exist yet; ingest a book first.", name)
            return
        recorded = (await client.get_collection(name)).config.metadata or {}
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        logger.warning("Could not verify the index embedder of '%s': %s", name, exc)
        return
    if "embedding_model" not in recorded:
        logger.warning(
            "Collection '%s' does not record its embedder; re-ingest any book to stamp it.", name
        )
        return
    problems = stamp_problems(recorded, settings)
    if problems:
        raise IndexMismatchError(
            f"Collection '{name}' was built with a different embedder: {'; '.join(problems)}. "
            "Fix the embedding settings, or re-ingest every book with the new model."
        )
    sparse_problems = sparse_stamp_problems(recorded, settings)
    if sparse_problems:
        logger.warning(
            "Keyword search in '%s' is out of date: %s. Run `python -m pipeline.reindex_sparse` "
            "(no re-embedding) before serving.",
            name,
            "; ".join(sparse_problems),
        )


def settings_fingerprint(settings: Settings) -> str:
    """Short stable hash of the settings (and tokenizer) that change what an answer would be."""
    values = {field: getattr(settings, field) for field in _FINGERPRINT_FIELDS}
    values[SPARSE_STAMP_KEY] = SPARSE_ENCODER_VERSION
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
