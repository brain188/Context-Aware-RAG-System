"""Reconstruct document structure: book → chapter → section.

Headings come from the PDF outline when it has one, else from font signals
(see headings.py). Each section holds exactly the lines between its heading
and the next one, so sections never overlap and every section knows its
chapter and the pages its own text sits on.
"""

import time
from dataclasses import dataclass, field

from app.config import get_logger

from pipeline.extractors.headings import Heading, PageLine, font_headings, toc_headings
from pipeline.extractors.pdf_extractor import ExtractedDocument

logger = get_logger(__name__)


@dataclass
class StructuredSection:
    """A contiguous section of text with its hierarchical location."""

    chapter: str
    section: str
    page_start: int
    page_end: int
    text: str
    # (offset into text, page) where each page's text begins, in order. Lets the
    # chunker give each sub-chunk of a long section its own pages. Empty = unknown.
    page_offsets: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Fully structured document ready for preprocessing and chunking."""

    book_title: str
    source_path: str
    sections: list[StructuredSection] = field(default_factory=list)

    @property
    def chapter_count(self) -> int:
        return len({s.chapter for s in self.sections})

    @property
    def section_count(self) -> int:
        return len(self.sections)


def _section(chapter: str, section: str, lines: list[PageLine]) -> StructuredSection | None:
    """Join *lines* into a section (blank line between text blocks); None if no text."""
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    position = 0
    for k, (page, line) in enumerate(lines):
        if k:
            separator = "\n\n" if line.block_start else "\n"
            parts.append(separator)
            position += len(separator)
        if not page_offsets or page_offsets[-1][1] != page:
            page_offsets.append((position, page))
        parts.append(line.text)
        position += len(line.text)
    text = "".join(parts)  # line texts are stripped, so offsets need no adjusting
    if not text.strip():
        return None
    return StructuredSection(
        chapter=chapter,
        section=section,
        page_start=lines[0][0],
        page_end=lines[-1][0],
        text=text,
        page_offsets=page_offsets,
    )


def _build_sections(lines: list[PageLine], headings: list[Heading]) -> list[StructuredSection]:
    """Cut the book's lines at each heading; text before the first one is front matter."""
    bounded = [h.line_index for h in headings] + [len(lines)]
    candidates = [_section("Preamble", "Front Matter", lines[: bounded[0]])]
    chapter = "Preamble"
    for heading, end in zip(headings, bounded[1:]):
        if heading.level == 1:
            chapter = heading.title
        section = heading.title
        candidates.append(_section(chapter, section, lines[heading.line_index : end]))
    return [s for s in candidates if s is not None]


def parse_structure(doc: ExtractedDocument, book_title: str) -> ParsedDocument:
    """Build a structured section list from a raw extracted document.

    Each StructuredSection carries its chapter name, section name, page range,
    and the text of its own lines. A document with no detectable headings
    becomes one "Full Document" section.

    Args:
        doc: Raw extraction result from pdf_extractor.
        book_title: Human-readable book title used as metadata.

    Returns:
        ParsedDocument whose sections cover the entire book text, in order.
    """
    start = time.perf_counter()
    lines: list[PageLine] = [(p.page_number, line) for p in doc.pages for line in p.lines]

    source = "outline"
    headings = toc_headings(doc.toc, lines, book_title)
    if not headings:
        source = "font signals"
        headings = font_headings(lines, doc.body_font_size)

    if headings:
        sections = _build_sections(lines, headings)
    else:
        logger.warning("'%s': no headings detected — treating as flat document.", book_title)
        flat = _section("Unknown Chapter", "Full Document", lines) if lines else None
        sections = [flat] if flat else []

    parsed = ParsedDocument(book_title=book_title, source_path=doc.source_path, sections=sections)
    _log_result(book_title, parsed, source, start)
    return parsed


def _log_result(book_title: str, parsed: ParsedDocument, source: str, start: float) -> None:
    elapsed = time.perf_counter() - start
    logger.info(
        "Parsed '%s' from %s: chapters=%d sections=%d elapsed_s=%.2f",
        book_title,
        source,
        parsed.chapter_count,
        parsed.section_count,
        elapsed,
    )
    if parsed.chapter_count < 3:
        logger.warning(
            "'%s': only %d chapter(s) detected — spot-check the ToC before proceeding to Phase 2.",
            book_title,
            parsed.chapter_count,
        )
