"""Qdrant collection schema and writes for the ingestion pipeline.

Each point carries two named vectors: a dense embedding for semantic search and
a BM25 sparse vector for keyword search (Qdrant applies IDF at query time).
The app's HybridRetriever queries both in one call, so the schema here is a
contract: vector names come from Settings and must match on both sides.
"""

from app.config import Settings, get_logger
from app.models import Chunk
from app.retrieval.sparse_encoder import BM25SparseEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

logger = get_logger(__name__)


class CollectionSchemaError(RuntimeError):
    """The existing collection was created with a schema this code cannot write to."""


def get_client(settings: Settings) -> QdrantClient:
    """Build a Qdrant client from settings."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )


def collection_schema(
    settings: Settings,
) -> tuple[dict[str, VectorParams], dict[str, SparseVectorParams]]:
    """Return (dense vectors config, sparse vectors config) for the chunk collection."""
    dense = {
        settings.qdrant_dense_vector_name: VectorParams(
            size=settings.embedding_dimension, distance=Distance.COSINE
        )
    }
    sparse = {settings.qdrant_sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF)}
    return dense, sparse


def _schema_problems(info: CollectionInfo, settings: Settings) -> list[str]:
    """List every way *info* differs from collection_schema(settings)."""
    problems: list[str] = []
    dense_name = settings.qdrant_dense_vector_name
    sparse_name = settings.qdrant_sparse_vector_name

    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or dense_name not in vectors:
        problems.append(f"no dense vector named '{dense_name}'")
    elif vectors[dense_name].size != settings.embedding_dimension:
        problems.append(
            f"dense vector size {vectors[dense_name].size} != "
            f"EMBEDDING_DIMENSION {settings.embedding_dimension}"
        )

    sparse = info.config.params.sparse_vectors or {}
    if sparse_name not in sparse:
        problems.append(f"no sparse vector named '{sparse_name}'")
    elif sparse[sparse_name].modifier != Modifier.IDF:
        problems.append(f"sparse vector '{sparse_name}' lacks the IDF modifier")
    return problems


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    """Create the collection if missing; otherwise verify its schema matches.

    Raises:
        CollectionSchemaError: if the existing collection has a different schema
            (e.g. created before sparse vectors were added).
    """
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        dense, sparse = collection_schema(settings)
        client.create_collection(
            collection_name=name, vectors_config=dense, sparse_vectors_config=sparse
        )
        logger.info("Created Qdrant collection '%s' (dense + bm25 sparse).", name)
        return

    problems = _schema_problems(client.get_collection(name), settings)
    if problems:
        raise CollectionSchemaError(
            f"Collection '{name}' has an outdated schema: {'; '.join(problems)}. "
            f"Delete it (DELETE {settings.qdrant_url}/collections/{name}) and re-ingest all books."
        )


def delete_book_chunks(client: QdrantClient, source_path: str, settings: Settings) -> None:
    """Delete all points for this book (enables idempotent re-ingest)."""
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
        ),
    )
    logger.info("Cleared existing chunks for source_path='%s'.", source_path)


def build_points(
    chunks: list[Chunk], encoder: BM25SparseEncoder, settings: Settings
) -> list[PointStruct]:
    """Turn embedded chunks into points with dense + sparse vectors and full payload.

    Chunks without an embedding are skipped; a chunk with no word tokens gets
    no sparse vector (it can still be found semantically).
    """
    points: list[PointStruct] = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        vectors: dict = {settings.qdrant_dense_vector_name: chunk.embedding}
        sparse = encoder.encode_document(chunk.text)
        if sparse.indices:
            vectors[settings.qdrant_sparse_vector_name] = sparse
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=vectors,
                payload={**chunk.metadata.model_dump(), "text": chunk.text},
            )
        )
    return points


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], settings: Settings) -> int:
    """Upsert embedded chunks; return the number of points written."""
    points = build_points(chunks, BM25SparseEncoder(settings), settings)
    if not points:
        logger.warning("No points to upsert — all chunks are missing embeddings.")
        return 0
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(
        "Upserted %d points into Qdrant collection '%s'.", len(points), settings.qdrant_collection
    )
    return len(points)
