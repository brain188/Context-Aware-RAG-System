"""Offline ingestion pipeline entrypoint.

Orchestrates the full flow for one or more PDF books:
    raw PDFs → extract → parse structure → preprocess → chunk →
    deduplicate → embed → upsert into Qdrant (dense + BM25 sparse vectors).

Idempotent per book: re-ingesting a book deletes its existing Qdrant points
(keyed on source_path) before re-inserting, so chunks are never duplicated.

CLI usage (run from the project root):
    python -m pipeline.ingest --path /books/prml.pdf --title "PRML"
    python -m pipeline.ingest --path /books/           # ingest all PDFs in folder
"""

import argparse
import sys
import time
from pathlib import Path

from app.config import get_logger, get_settings

from pipeline.chunker import chunk_document
from pipeline.deduplicator import deduplicate
from pipeline.embedder import embed_chunks
from pipeline.extractors.pdf_extractor import extract_pdf
from pipeline.extractors.structure_parser import parse_structure
from pipeline.preprocessor import preprocess_document
from pipeline.vector_store import delete_book_chunks, ensure_collection, get_client, upsert_chunks

logger = get_logger(__name__)


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

    # Connect before the slow embedding step so store/auth/schema problems fail in seconds.
    settings = get_settings()
    client = get_client(settings)
    ensure_collection(client, settings)

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

    delete_book_chunks(client, str(pdf_path), settings)
    upsert_chunks(client, chunks, settings)

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
