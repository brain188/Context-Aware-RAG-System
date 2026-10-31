"""Heading-aware chunker: converts ParsedDocument sections into Chunks.

Splits overlong sections into sub-chunks with overlap; accumulates small
sections until they reach the configured minimum. Every emitted Chunk carries
complete ChunkMetadata (book, chapter, section, page range, source_path, hash).

Sizes are counted with the embedding model's own tokenizer, not a generic one:
the model silently truncates past EMBEDDING_MAX_TOKENS, and generic counts
(e.g. tiktoken) undercount code- and symbol-heavy text by up to ~30%.

This module is the Phase-1 milestone gate (M1): chunks must carry correct
metadata validated on 3 structurally different books before Phase 2 begins.
"""

import hashlib
import time
from functools import lru_cache

from app.config import get_logger, get_settings
from app.models import Chunk, ChunkMetadata
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pipeline.extractors.structure_parser import ParsedDocument, StructuredSection

logger = get_logger(__name__)


class ChunkTooLongError(ValueError):
    """A chunk exceeds what the embedding model can read without truncation."""


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Tokenizer:
    """Load *model_name*'s fast tokenizer (downloaded once, then read from the HF cache)."""
    tokenizer = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
    # Counting must see the full length; a bundled truncation config would hide overflow.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def _get_tokenizer() -> Tokenizer:
    return _load_tokenizer(get_settings().embedding_model)


def _count_tokens(text: str) -> int:
    """Content tokens in *text* as the embedding model sees them (excluding [CLS]/[SEP])."""
    return len(_get_tokenizer().encode(text, add_special_tokens=False).ids)


def _special_token_count() -> int:
    """Tokens the model adds around every input ([CLS] and [SEP] for bge)."""
    return len(_get_tokenizer().encode("", add_special_tokens=True).ids)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_start(word_ids: list[int | None], lower: int, index: int) -> int:
    """Move token *index* back to the first token of its word, staying above *lower*.

    Returns *index* unchanged if a single word fills the whole window, so the
    split cuts mid-word instead of stalling.
    """
    j = index
    while j > lower and word_ids[j] is not None and word_ids[j] == word_ids[j - 1]:
        j -= 1
    return j if j > lower else index


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into sub-chunks of at most max_tokens content tokens, with overlap.

    Windows end on word boundaries and are sliced from the original string via
    token offsets (decoding would lowercase and re-space the text). Because
    bge's pre-tokenizer splits on whitespace and punctuation, a word-aligned
    slice re-tokenizes to exactly the window's tokens.

    Args:
        text: Text to split.
        max_tokens: Hard upper bound on content tokens per sub-chunk.
        overlap_tokens: Approximate tokens to repeat at the start of the next chunk.

    Returns:
        List of sub-chunk strings; always at least one element.
    """
    encoding = _get_tokenizer().encode(text, add_special_tokens=False)
    n = len(encoding.ids)
    if n <= max_tokens:
        return [text]

    offsets, word_ids = encoding.offsets, encoding.word_ids
    sub_chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        if end < n:
            end = _word_start(word_ids, start, end)
        sub_chunks.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end >= n:
            break
        next_start = end - overlap_tokens
        start = _word_start(word_ids, start, next_start) if next_start > start else end

    return sub_chunks


def _assert_fits_model(chunks: list[Chunk], limit: int) -> None:
    """Fail loud if any chunk would be truncated by the embedding model.

    Raises:
        ChunkTooLongError: naming the first offending chunk's section and length.
    """
    encodings = _get_tokenizer().encode_batch([c.text for c in chunks], add_special_tokens=True)
    for chunk, encoding in zip(chunks, encodings):
        if len(encoding.ids) > limit:
            raise ChunkTooLongError(
                f"Chunk in '{chunk.metadata.section}' (p.{chunk.metadata.page_start}) has "
                f"{len(encoding.ids)} model tokens; limit is {limit}"
            )


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
        ChunkTooLongError: if a chunk would exceed the embedding model's input limit.
    """
    if not doc.source_path:
        raise ValueError("doc.source_path must not be empty")
    if not doc.book_title:
        raise ValueError("doc.book_title must not be empty")

    settings = get_settings()
    min_tokens = settings.chunk_min_tokens
    # Budget for content tokens: the model adds [CLS]/[SEP] to every input.
    max_tokens = settings.chunk_max_tokens - _special_token_count()
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

    _assert_fits_model(chunks, settings.chunk_max_tokens)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Chunked '%s': sections=%d chunks=%d elapsed_s=%.2f",
        doc.book_title,
        len(doc.sections),
        len(chunks),
        elapsed,
    )
    return chunks
