"""Qdrant collection schema and writes for the ingestion pipeline.

Each point carries two named vectors: a dense embedding for semantic search and
a BM25 sparse vector for keyword search (Qdrant applies IDF at query time).
The app's HybridRetriever queries both in one call, so the schema here is a
contract: vector names come from Settings and must match on both sides.
"""

from app.config import Settings, get_logger
from app.models import Chunk
from app.retrieval.sparse_encoder import BM25SparseEncoder
from app.services.index_version import embedder_stamp, stamp_problems
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

logger = get_logger(__name__)

# Payload fields the app filters or facets on (keys must match ChunkMetadata).
_PAYLOAD_INDEXES = {
    "book": PayloadSchemaType.KEYWORD,
    "chapter": PayloadSchemaType.KEYWORD,
    "source_path": PayloadSchemaType.KEYWORD,
    "page_start": PayloadSchemaType.INTEGER,
    "page_end": PayloadSchemaType.INTEGER,
}


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

    problems.extend(stamp_problems(info.config.metadata or {}, settings))
    return problems


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    """Create the collection if missing; otherwise verify its schema matches.

    New collections are stamped with the embedder that writes them, which the
    app checks at startup. Either way, make sure the payload indexes exist
    (adding an index to an existing collection is safe and needs no re-ingest).

    Raises:
        CollectionSchemaError: if the existing collection has a different schema
            (e.g. created before sparse vectors were added) or was written by a
            different embedding model.
    """
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        dense, sparse = collection_schema(settings)
        client.create_collection(
            collection_name=name,
            vectors_config=dense,
            sparse_vectors_config=sparse,
            metadata=embedder_stamp(settings),
        )
        logger.info("Created Qdrant collection '%s' (dense + bm25 sparse).", name)
    else:
        info = client.get_collection(name)
        problems = _schema_problems(info, settings)
        if problems:
            raise CollectionSchemaError(
                f"Collection '{name}' has an outdated schema: {'; '.join(problems)}. "
                f"Delete it (DELETE {settings.qdrant_url}/collections/{name}) and re-ingest "
                "all books."
            )
        if "embedding_model" not in (info.config.metadata or {}):
            # Created before stamping existed: its vectors passed the size check above,
            # so record the configured embedder as the one that wrote them.
            client.update_collection(name, metadata=embedder_stamp(settings))
            logger.warning(
                "Collection '%s' had no embedder stamp; recorded %s. If its existing chunks "
                "were embedded with another model, re-ingest every book.",
                name,
                embedder_stamp(settings),
            )
    _ensure_payload_indexes(client, name)


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Index the payload fields the app filters and facets on.

    Filters work without indexes (slower), but facets — used to list books and
    chapters — require a keyword index on a Qdrant server.
    """
    existing = client.get_collection(collection).payload_schema or {}
    for field, schema in _PAYLOAD_INDEXES.items():
        if field not in existing:
            client.create_payload_index(collection, field_name=field, field_schema=schema)
            logger.info("Created payload index '%s' (%s) on '%s'.", field, schema, collection)


def delete_book_chunks(
    client: QdrantClient,
    source_path: str,
    book_title: str,
    settings: Settings,
    keep_ids: list[str] | None = None,
) -> None:
    """Delete this book's points, except *keep_ids* (enables idempotent re-ingest).

    Matches the file *or* the title, so a book ingested earlier under another
    path spelling (e.g. relative before paths were made absolute) is replaced,
    not duplicated. Ingest upserts the new chunks first and passes their ids
    here, so a failed upload leaves the previous version in place.
    """
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            should=[
                FieldCondition(key="source_path", match=MatchValue(value=source_path)),
                FieldCondition(key="book", match=MatchValue(value=book_title)),
            ],
            must_not=[HasIdCondition(has_id=keep_ids)] if keep_ids else None,
        ),
    )
    logger.info(
        "Cleared existing chunks for book='%s' / source_path='%s'.", book_title, source_path
    )


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


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], settings: Settings) -> list[str]:
    """Upsert embedded chunks in batches; return the ids of the points written.

    One request for a whole book is several MB and times out on a slow uplink to
    a hosted Qdrant, so points go up in small batches, each retried on failure.
    """
    points = build_points(chunks, BM25SparseEncoder(settings), settings)
    if not points:
        logger.warning("No points to upsert — all chunks are missing embeddings.")
        return []
    client.upload_points(
        collection_name=settings.qdrant_collection,
        points=points,
        batch_size=settings.qdrant_upsert_batch_size,
        max_retries=3,
        wait=True,
    )
    logger.info(
        "Upserted %d points into Qdrant collection '%s' (batches of %d).",
        len(points),
        settings.qdrant_collection,
        settings.qdrant_upsert_batch_size,
    )
    return [str(point.id) for point in points]
