"""Heading-aware chunker: converts ParsedDocument sections into Chunks.

Splits overlong sections into sub-chunks with overlap; accumulates small
sections until they reach the configured minimum. Every emitted Chunk carries
complete ChunkMetadata (book, chapter, section, page range, source_path, hash).

This module is the Phase-1 milestone gate (M1): chunks must carry correct
metadata validated on 3 structurally different books before Phase 2 begins.
"""

import hashlib
import time

import tiktoken
from app.config import get_logger, get_settings
from app.models import Chunk, ChunkMetadata

from pipeline.extractors.structure_parser import ParsedDocument, StructuredSection

logger = get_logger(__name__)

# Lazy singleton — initialized on first use to avoid side effects at import time
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into sub-chunks of at most max_tokens with token-level overlap.

    Overlap is applied by sliding the window back by overlap_tokens after each
    chunk so context at boundaries is preserved across adjacent chunks.

    Args:
        text: Text to split.
        max_tokens: Hard upper bound on tokens per sub-chunk.
        overlap_tokens: Number of tokens to repeat at the start of the next chunk.

    Returns:
        List of sub-chunk strings; always at least one element.
    """
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    sub_chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        sub_chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens  # slide back by overlap

    return sub_chunks


def _make_chunk(
    text: str,
    section: StructuredSection,
    doc: ParsedDocument,
) -> Chunk:
    """Build a single Chunk with complete ChunkMetadata."""
    metadata = ChunkMetadata(
        book=doc.book_title,
        chapter=section.chapter,
        section=section.section,
        page_start=section.page_start,
        page_end=section.page_end,
        source_path=doc.source_path,
        content_hash=_sha256(text),
    )
    return Chunk(text=text, metadata=metadata)


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Convert a ParsedDocument into a list of sized, metadata-rich Chunks.

    Strategy:
    - Sections within [min_tokens, max_tokens] → emitted as a single chunk.
    - Sections above max_tokens → split with overlap, each sub-chunk emitted
      separately.
    - Sections below min_tokens → accumulated into a buffer and merged with
      subsequent small sections until the buffer reaches min_tokens, then emitted.

    Args:
        doc: Preprocessed, structured document.

    Returns:
        List of Chunk objects each with complete ChunkMetadata. Empty only if the
        document contains no text.

    Raises:
        ValueError: if doc.source_path or doc.book_title is empty.
    """
    if not doc.source_path:
        raise ValueError("doc.source_path must not be empty")
    if not doc.book_title:
        raise ValueError("doc.book_title must not be empty")

    settings = get_settings()
    min_tokens = settings.chunk_min_tokens
    max_tokens = settings.chunk_max_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    start_time = time.perf_counter()
    chunks: list[Chunk] = []

    # Buffer for accumulating small sections
    buf_text: str = ""
    buf_section: StructuredSection | None = None

    def flush_buffer() -> None:
        nonlocal buf_text, buf_section
        if buf_text.strip() and buf_section is not None:
            for sub in _split_by_tokens(buf_text.strip(), max_tokens, overlap_tokens):
                sub = sub.strip()
                if sub:
                    chunks.append(_make_chunk(sub, buf_section, doc))
        buf_text = ""
        buf_section = None

    for section in doc.sections:
        text = section.text.strip()
        if not text:
            continue

        count = _count_tokens(text)

        if count > max_tokens:
            # Flush any buffered small section first
            flush_buffer()
            # Split overlong section into sub-chunks
            for sub in _split_by_tokens(text, max_tokens, overlap_tokens):
                sub = sub.strip()
                if sub and _count_tokens(sub) >= min_tokens:
                    chunks.append(_make_chunk(sub, section, doc))

        elif count >= min_tokens:
            # Section is within bounds — flush buffer then emit
            flush_buffer()
            chunks.append(_make_chunk(text, section, doc))

        else:
            # Too small: accumulate into buffer
            if buf_text:
                combined = buf_text + "\n\n" + text
                if _count_tokens(combined) >= min_tokens:
                    # Buffer now meets the minimum — flush combined
                    buf_text = combined
                    flush_buffer()
                else:
                    buf_text = combined
                    # Keep buf_section pointing to the first section for metadata
            else:
                buf_text = text
                buf_section = section

    # Final flush: emit whatever remains in the buffer
    if buf_text.strip() and buf_section is not None:
        for sub in _split_by_tokens(buf_text.strip(), max_tokens, overlap_tokens):
            sub = sub.strip()
            if sub:
                chunks.append(_make_chunk(sub, buf_section, doc))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Chunked '%s': sections=%d chunks=%d elapsed_s=%.2f",
        doc.book_title,
        len(doc.sections),
        len(chunks),
        elapsed,
    )
    return chunks
