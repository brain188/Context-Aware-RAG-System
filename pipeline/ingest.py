"""Offline ingestion pipeline entrypoint.

Orchestrates the full flow for one or more PDF books:
    raw PDFs → extract → parse structure → preprocess → chunk →
    deduplicate → embed → upsert into Qdrant + BM25 keyword corpus.

Idempotent per book: re-ingesting a book deletes its existing Qdrant points
(keyed on source_path) before re-inserting, so chunks are never duplicated.

CLI usage (run from the project root):
    python -m pipeline.ingest --path /books/prml.pdf --title "PRML"
    python -m pipeline.ingest --path /books/           # ingest all PDFs in folder
"""

import argparse
import json
import sys
import time
from pathlib import Path

from app.config import get_logger, get_settings
from app.models import Chunk
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from pipeline.chunker import chunk_document
from pipeline.deduplicator import deduplicate
from pipeline.embedder import embed_chunks
from pipeline.extractors.pdf_extractor import extract_pdf
from pipeline.extractors.structure_parser import parse_structure
from pipeline.preprocessor import preprocess_document

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def _get_qdrant() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist."""
    settings = get_settings()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'.", settings.qdrant_collection)


def _delete_book_chunks(client: QdrantClient, source_path: str) -> None:
    """Delete all Qdrant points for this book (enables idempotent re-ingest)."""
    settings = get_settings()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
        ),
    )
    logger.info("Cleared existing chunks for source_path='%s'.", source_path)


def _upsert_chunks(client: QdrantClient, chunks: list[Chunk]) -> None:
    """Upsert embedded chunks with full metadata payload into Qdrant."""
    settings = get_settings()
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=chunk.embedding,
            payload={
                "book": chunk.metadata.book,
                "chapter": chunk.metadata.chapter,
                "section": chunk.metadata.section,
                "page_start": chunk.metadata.page_start,
                "page_end": chunk.metadata.page_end,
                "source_path": chunk.metadata.source_path,
                "content_hash": chunk.metadata.content_hash,
                "text": chunk.text,
            },
        )
        for chunk in chunks
        if chunk.embedding is not None
    ]
    if not points:
        logger.warning("No points to upsert — all chunks are missing embeddings.")
        return
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(
        "Upserted %d points into Qdrant collection '%s'.",
        len(points),
        settings.qdrant_collection,
    )


# ---------------------------------------------------------------------------
# BM25 corpus helpers
# ---------------------------------------------------------------------------


def _update_bm25_corpus(chunks: list[Chunk], source_path: str) -> None:
    """Update the BM25 keyword corpus file with this book's chunks.

    The corpus is a JSON-lines file keyed by chunk_id. Existing entries for
    this source_path are replaced to maintain idempotency.

    Args:
        chunks: Embedded and deduplicated chunks for this book.
        source_path: Canonical path used to identify the book's existing entries.
    """
    settings = get_settings()
    corpus_path = Path(settings.bm25_corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    existing[entry["chunk_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    pass

    # Remove stale entries for this source_path
    existing = {k: v for k, v in existing.items() if v.get("source_path") != source_path}

    # Insert new entries
    for chunk in chunks:
        existing[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "book": chunk.metadata.book,
            "source_path": chunk.metadata.source_path,
        }

    with corpus_path.open("w", encoding="utf-8") as fh:
        for entry in existing.values():
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        "BM25 corpus updated: total_entries=%d path='%s'",
        len(existing),
        corpus_path,
    )


# ---------------------------------------------------------------------------
# Public ingest function
# ---------------------------------------------------------------------------


def ingest_book(pdf_path: str | Path, book_title: str) -> dict:
    """Run the full ingestion pipeline for one PDF book.

    Args:
        pdf_path: Path to the PDF file.
        book_title: Human-readable title used as the book metadata key.

    Returns:
        Summary dict: book_title, chunks_indexed, chunks_skipped_duplicate,
        elapsed_seconds, status ("ok" | "error"), and optional detail.
    """
    start = time.perf_counter()
    pdf_path = Path(pdf_path)
    logger.info("Ingest started: book='%s' path='%s'", book_title, pdf_path)

    extracted = extract_pdf(pdf_path)
    parsed = parse_structure(extracted, book_title)
    cleaned = preprocess_document(parsed)
    chunks = chunk_document(cleaned)

    if not chunks:
        logger.warning("No chunks produced for '%s' — aborting ingest.", book_title)
        return {
            "book_title": book_title,
            "chunks_indexed": 0,
            "chunks_skipped_duplicate": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "status": "error",
            "detail": "No chunks produced",
        }

    total_before_dedup = len(chunks)
    chunks = deduplicate(chunks)
    skipped = total_before_dedup - len(chunks)

    chunks = embed_chunks(chunks)

    client = _get_qdrant()
    _ensure_collection(client)
    _delete_book_chunks(client, str(pdf_path))
    _upsert_chunks(client, chunks)
    _update_bm25_corpus(chunks, str(pdf_path))

    elapsed = time.perf_counter() - start
    logger.info(
        "Ingest complete: book='%s' chunks=%d skipped=%d elapsed_s=%.1f",
        book_title,
        len(chunks),
        skipped,
        elapsed,
    )
    return {
        "book_title": book_title,
        "chunks_indexed": len(chunks),
        "chunks_skipped_duplicate": skipped,
        "elapsed_seconds": elapsed,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF books into the RAG vector store.")
    parser.add_argument("--path", required=True, help="PDF file or folder of PDFs.")
    parser.add_argument("--title", help="Book title (required for a single file).")
    args = parser.parse_args()

    input_path = Path(args.path)
    if input_path.is_file():
        if not args.title:
            parser.error("--title is required when --path points to a single file.")
        results = [ingest_book(input_path, args.title)]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            logger.error("No PDF files found in '%s'.", input_path)
            sys.exit(1)
        results = []
        for pdf in pdfs:
            title = pdf.stem.replace("_", " ").replace("-", " ").title()
            try:
                results.append(ingest_book(pdf, title))
            except Exception as exc:
                logger.error("Failed to ingest '%s': %s", pdf.name, exc, exc_info=True)
                results.append(
                    {
                        "book_title": title,
                        "chunks_indexed": 0,
                        "chunks_skipped_duplicate": 0,
                        "elapsed_seconds": 0.0,
                        "status": "error",
                        "detail": str(exc),
                    }
                )
    else:
        parser.error(f"--path does not exist: {input_path}")

    print("\n=== Ingest Summary ===")
    total_chunks = 0
    for r in results:
        status = r.get("status", "?").upper()
        n = r.get("chunks_indexed", 0)
        total_chunks += n
        print(
            f"  {status:5s}  {r['book_title'][:50]:50s}"
            f"  chunks={n}  skipped={r.get('chunks_skipped_duplicate', 0)}"
            f"  time={r.get('elapsed_seconds', 0):.1f}s"
        )
    print(f"\nTotal: {len(results)} book(s), {total_chunks} chunk(s) indexed.")


if __name__ == "__main__":
    _cli()
