"""Rebuild every chunk's BM25 keyword vector from its stored text.

    python -m pipeline.reindex_sparse                    # re-encode all chunks, then stamp
    python -m pipeline.reindex_sparse --dry-run          # count only; writes nothing
    python -m pipeline.reindex_sparse --drop-navigation  # also delete contents/index chunks

Run it after changing app/retrieval/sparse_encoder.py (with a new
SPARSE_ENCODER_VERSION) or BM25_K1 / BM25_B / BM25_AVG_DOC_LENGTH. Until then the
API warns at startup and ingest refuses to write. Dense vectors and payloads stay
as they are, so no embedding model is loaded and no PDF is read again. Run it
before starting the API on the new code, so no answer is cached against
half-updated keyword vectors.

--drop-navigation deletes the table-of-contents and index chunks that the chunker
no longer writes (pipeline/chunker.py), so the books need no re-ingest.

Safe to re-run: the collection is stamped only after every chunk is rewritten.
Exit codes: 0 ok, 2 setup or Qdrant error.
"""

import argparse
import sys
from dataclasses import dataclass

from app.config import Settings, get_logger, get_settings
from app.retrieval.sparse_encoder import SPARSE_ENCODER_VERSION, BM25SparseEncoder
from app.services.index_version import sparse_stamp
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import PointIdsList, PointVectors

from pipeline.chunker import is_navigation_chapter
from pipeline.vector_store import CollectionSchemaError, get_client

logger = get_logger(__name__)

# Sparse vectors are small (a few hundred terms), so updates batch well beyond upsert size.
_BATCH_SIZE = 128


@dataclass
class ReindexResult:
    reencoded: int = 0  # chunks given a new keyword vector
    cleared: int = 0  # chunks with no keyword tokens: keyword vector removed
    dropped: int = 0  # table-of-contents / index chunks deleted (--drop-navigation)


def reindex_sparse(
    client: QdrantClient,
    settings: Settings,
    drop_navigation: bool = False,
    dry_run: bool = False,
    batch_size: int = _BATCH_SIZE,
) -> ReindexResult:
    """Re-encode the keyword vector of every point from its payload text, then stamp.

    Raises:
        CollectionSchemaError: if the collection is missing or has no sparse vector.
        UnexpectedResponse, ResponseHandlingException: if Qdrant fails.
    """
    name = settings.qdrant_collection
    sparse_name = settings.qdrant_sparse_vector_name
    if not client.collection_exists(name):
        raise CollectionSchemaError(f"Collection '{name}' does not exist; ingest a book first.")
    if sparse_name not in (client.get_collection(name).config.params.sparse_vectors or {}):
        raise CollectionSchemaError(
            f"Collection '{name}' has no sparse vector '{sparse_name}'; delete it and re-ingest."
        )

    encoder = BM25SparseEncoder(settings)
    result = ReindexResult()
    offset = None
    while True:
        points, offset = client.scroll(
            name,
            limit=batch_size,
            offset=offset,
            with_payload=["text", "chapter"],
            with_vectors=False,
        )
        updates: list[PointVectors] = []
        empty: list = []
        navigation: list = []
        for point in points:
            payload = point.payload or {}
            if drop_navigation and is_navigation_chapter(payload.get("chapter", "")):
                navigation.append(point.id)
                continue
            vector = encoder.encode_document(payload.get("text", ""))
            if vector.indices:
                updates.append(PointVectors(id=point.id, vector={sparse_name: vector}))
            else:
                empty.append(point.id)
        if not dry_run:
            if updates:
                client.update_vectors(name, points=updates)
            if empty:
                client.delete_vectors(name, vectors=[sparse_name], points=empty)
            if navigation:  # ids already scrolled past, so the next page is unaffected
                client.delete(name, points_selector=PointIdsList(points=navigation))
        result.reencoded += len(updates)
        result.cleared += len(empty)
        result.dropped += len(navigation)
        if offset is None:
            break

    if not dry_run:
        client.update_collection(name, metadata=sparse_stamp(settings))
    logger.info(
        "Sparse reindex of '%s'%s: reencoded=%d cleared=%d dropped_navigation=%d encoder=%s",
        name,
        " (dry run)" if dry_run else "",
        result.reencoded,
        result.cleared,
        result.dropped,
        SPARSE_ENCODER_VERSION,
    )
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild BM25 keyword vectors from stored chunk text (no re-embedding)."
    )
    parser.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    parser.add_argument(
        "--drop-navigation",
        action="store_true",
        help="also delete table-of-contents and index chunks",
    )
    args = parser.parse_args()
    settings = get_settings()
    try:
        result = reindex_sparse(get_client(settings), settings, args.drop_navigation, args.dry_run)
    except CollectionSchemaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        print(f"Error: Qdrant failed: {exc}. Re-run; nothing is stamped yet.", file=sys.stderr)
        sys.exit(2)
    verb = "Would re-encode" if args.dry_run else "Re-encoded"
    print(
        f"{verb} {result.reencoded} chunk(s) with {SPARSE_ENCODER_VERSION} "
        f"(k1={settings.bm25_k1}, b={settings.bm25_b}, avg={settings.bm25_avg_doc_length}); "
        f"{result.cleared} without keyword tokens; {result.dropped} navigation chunk(s) "
        f"{'to delete' if args.dry_run else 'deleted'}."
    )
    if not args.dry_run:
        print(f"Stamped '{settings.qdrant_collection}'. Start (or restart) the API now.")


if __name__ == "__main__":
    _cli()
