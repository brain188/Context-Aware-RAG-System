"""Heading-aware chunker: converts ParsedDocument sections into Chunks.

Splits overlong sections into sub-chunks with overlap; accumulates small
sections until they reach the configured minimum. Every emitted Chunk carries
complete ChunkMetadata (book, chapter, section, page range, source_path, hash).
The table of contents and the index are not chunked (see _NAVIGATION_CHAPTERS).

Sizes are counted with the embedding model's own tokenizer, not a generic one:
the model silently truncates past EMBEDDING_MAX_TOKENS, and generic counts
(e.g. tiktoken) undercount code- and symbol-heavy text by up to ~30%.

This module is the Phase-1 milestone gate (M1): chunks must carry correct
metadata validated on 3 structurally different books before Phase 2 begins.
"""

import hashlib
import time
from bisect import bisect_right
from dataclasses import replace
from functools import lru_cache

from app.config import get_logger, get_settings
from app.models import Chunk, ChunkMetadata
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pipeline.extractors.structure_parser import ParsedDocument, StructuredSection

logger = get_logger(__name__)


# Front- and back-matter chapters that only list headings and page numbers. They cannot
# answer a question, yet the table of contents holds every section number and title, so
# it tops keyword results for queries like "Sect. 11.2.1" and crowds out the real section.
_NAVIGATION_CHAPTERS = frozenset({"contents", "table of contents", "index"})


class ChunkTooLongError(ValueError):
    """A chunk exceeds what the embedding model can read without truncation."""


def is_navigation_chapter(chapter: str) -> bool:
    """True for a table of contents or an index, which are not indexed."""
    return chapter.strip().casefold() in _NAVIGATION_CHAPTERS


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


def _split_spans(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Split text into windows of at most max_tokens content tokens, with overlap.

    Windows end on word boundaries and are sliced from the original string via
    token offsets (decoding would lowercase and re-space the text). Because
    bge's pre-tokenizer splits on whitespace and punctuation, a word-aligned
    slice re-tokenizes to exactly the window's tokens.

    Args:
        text: Text to split.
        max_tokens: Hard upper bound on content tokens per sub-chunk.
        overlap_tokens: Approximate tokens to repeat at the start of the next chunk.

    Returns:
        (start, end) character spans of the windows in *text*; always at least one.
    """
    encoding = _get_tokenizer().encode(text, add_special_tokens=False)
    n = len(encoding.ids)
    if n <= max_tokens:
        return [(0, len(text))]

    offsets, word_ids = encoding.offsets, encoding.word_ids
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        if end < n:
            end = _word_start(word_ids, start, end)
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end >= n:
            break
        next_start = end - overlap_tokens
        start = _word_start(word_ids, start, next_start) if next_start > start else end

    return spans


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into sub-chunk strings; see _split_spans."""
    return [text[a:b] for a, b in _split_spans(text, max_tokens, overlap_tokens)]


def _pages_of(section: StructuredSection, start: int, end: int) -> tuple[int, int]:
    """Pages that section.text[start:end] sits on; the section's range if offsets are unknown."""
    if not section.page_offsets or end <= start:
        return section.page_start, section.page_end
    starts = [offset for offset, _ in section.page_offsets]

    def page_at(position: int) -> int:
        return section.page_offsets[max(bisect_right(starts, position) - 1, 0)][1]

    return page_at(start), page_at(end - 1)


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
    pages: tuple[int, int] | None = None,
) -> Chunk:
    """Build a single Chunk with complete ChunkMetadata (pages default to the section's)."""
    page_start, page_end = pages or (section.page_start, section.page_end)
    metadata = ChunkMetadata(
        book=doc.book_title,
        chapter=section.chapter,
        section=section.section,
        page_start=page_start,
        page_end=page_end,
        source_path=doc.source_path,
        content_hash=_sha256(text),
    )
    return Chunk(text=text, metadata=metadata)


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Convert a ParsedDocument into a list of sized, metadata-rich Chunks.

    Strategy:
    - Sections within [min_tokens, max_tokens] → emitted as a single chunk.
    - Sections above max_tokens → split with overlap, each sub-chunk emitted
      separately with the pages its own text sits on.
    - Sections below min_tokens → accumulated into a buffer and merged with
      subsequent small sections of the same chapter until the buffer reaches
      min_tokens, then emitted with the pages of every merged section.
    - Table-of-contents and index chapters → skipped.

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

    skipped_navigation = 0
    for section in doc.sections:
        text = section.text.strip()
        if not text:
            continue
        if is_navigation_chapter(section.chapter):
            skipped_navigation += 1
            continue
        lead = len(section.text) - len(section.text.lstrip())  # text[i] is section.text[lead + i]

        if buf_section is not None and buf_section.chapter != section.chapter:
            flush_buffer()  # a chunk never straddles two chapters

        count = _count_tokens(text)

        if count > max_tokens:
            # Flush any buffered small section first, then split the overlong one.
            flush_buffer()
            for start, end in _split_spans(text, max_tokens, overlap_tokens):
                window = text[start:end]
                sub = window.strip()
                if sub:  # keep a short tail: dropping it would lose the section's last words
                    first = lead + start + len(window) - len(window.lstrip())
                    pages = _pages_of(section, first, first + len(sub))
                    chunks.append(_make_chunk(sub, section, doc, pages))

        elif count >= min_tokens:
            # Section is within bounds — flush buffer then emit
            flush_buffer()
            chunks.append(
                _make_chunk(text, section, doc, _pages_of(section, lead, lead + len(text)))
            )

        else:
            # Too small: accumulate. The merged chunk keeps the first section's labels
            # but its page range covers every merged section.
            if buf_section is None:
                buf_text, buf_section = text, section
            else:
                buf_text += "\n\n" + text
                buf_section = replace(
                    buf_section,
                    page_end=max(buf_section.page_end, section.page_end),
                    page_offsets=[],  # offsets index one section's text, not the merged buffer
                )
            if _count_tokens(buf_text) >= min_tokens:
                flush_buffer()

    flush_buffer()

    _assert_fits_model(chunks, settings.chunk_max_tokens)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Chunked '%s': sections=%d chunks=%d skipped_navigation_sections=%d elapsed_s=%.2f",
        doc.book_title,
        len(doc.sections),
        len(chunks),
        skipped_navigation,
        elapsed,
    )
    return chunks
