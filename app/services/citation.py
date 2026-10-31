"""Source markers and citations.

The model sees retrieved chunks as numbered sources ([S1], [S2], …) and cites
them inline. Citations are built only from markers that point at a source we
actually provided, so a citation can never name a page that was not retrieved.
"""

import re

from app.models import ChunkMetadata, Citation, ScoredChunk

_MARKER_RE = re.compile(r"\[S(\d+)\]")


def source_marker(index: int) -> str:
    """Marker for the 1-based source *index*, e.g. 1 → "[S1]"."""
    return f"[S{index}]"


def page_range(page_start: int, page_end: int) -> str:
    """Render a page range: "p. 12" or "pp. 12–14"."""
    if page_end <= page_start:
        return f"p. {page_start}"
    return f"pp. {page_start}–{page_end}"


def citation_label(meta: ChunkMetadata) -> str:
    """Human-readable location: Book → Chapter → page range."""
    return f"{meta.book} → {meta.chapter} → {page_range(meta.page_start, meta.page_end)}"


def format_sources(chunks: list[ScoredChunk]) -> str:
    """Render chunks as numbered sources for a prompt, each headed by its marker and location."""
    blocks = []
    for i, scored in enumerate(chunks, start=1):
        meta = scored.chunk.metadata
        header = (
            f"{source_marker(i)} {meta.book} | {meta.chapter} | {meta.section} | "
            f"{page_range(meta.page_start, meta.page_end)}"
        )
        blocks.append(f"{header}\n{scored.chunk.text.strip()}")
    return "\n\n".join(blocks)


def cited_indices(text: str, source_count: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) 1-based source indices cited in *text*, each in first-seen order."""
    valid: list[int] = []
    invalid: list[int] = []
    for match in _MARKER_RE.finditer(text):
        index = int(match.group(1))
        bucket = valid if 1 <= index <= source_count else invalid
        if index not in bucket:
            bucket.append(index)
    return valid, invalid


def has_valid_marker(text: str, source_count: int) -> bool:
    """True if *text* cites at least one source that exists."""
    return bool(cited_indices(text, source_count)[0])


def _location_key(meta: ChunkMetadata) -> tuple[str, str, str, int, int]:
    return (meta.book, meta.chapter, meta.section, meta.page_start, meta.page_end)


def finalize_citations(text: str, chunks: list[ScoredChunk]) -> tuple[str, list[Citation]]:
    """Clean the answer's markers and build its citation list.

    Markers for sources that do not exist are removed. Sources at the same
    location (e.g. overlapping chunks of one page) are merged into the first
    marker that cites them. Citations are ordered by first appearance.
    """
    valid, _ = cited_indices(text, len(chunks))
    canonical: dict[int, int] = {}  # cited index -> index of the first source at its location
    first_at_location: dict[tuple[str, str, str, int, int], int] = {}
    for index in valid:
        key = _location_key(chunks[index - 1].chunk.metadata)
        canonical[index] = first_at_location.setdefault(key, index)

    def _replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return source_marker(canonical[index]) if index in canonical else ""

    cleaned = _MARKER_RE.sub(_replace, text)
    # Merging can leave a marker repeated back to back ("[S1][S1]").
    cleaned = re.sub(r"(\[S\d+\])(?:\1)+", r"\1", cleaned)

    citations = []
    for index in dict.fromkeys(canonical.values()):
        scored = chunks[index - 1]
        meta = scored.chunk.metadata
        citations.append(
            Citation(
                book=meta.book,
                chapter=meta.chapter,
                section=meta.section,
                page_start=meta.page_start,
                page_end=meta.page_end,
                chunk_id=scored.chunk.chunk_id,
                score=scored.score,
                label=citation_label(meta),
                marker=source_marker(index),
            )
        )
    return cleaned.strip(), citations
