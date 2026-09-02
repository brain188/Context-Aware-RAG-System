"""Reconstruct document structure: book → chapter → section.

Uses font-size signals from pdf_extractor and heading text patterns
(numbering like "9.2", "Chapter 9") to build a structured section list
where every section knows its chapter, page range, and full text.
"""

import re
import time
from dataclasses import dataclass, field

from app.config import get_logger

from pipeline.extractors.pdf_extractor import DocumentPage, ExtractedDocument

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heading pattern matchers — ordered most-specific to least-specific
# ---------------------------------------------------------------------------

_RE_CHAPTER_KEYWORD = re.compile(r"^chapter\s+\d+", re.IGNORECASE)
_RE_PART_KEYWORD = re.compile(r"^part\s+(i{1,3}|iv|v?i{0,3}|\d+)", re.IGNORECASE)
_RE_SUBSECTION = re.compile(r"^\d+\.\d+\.\d+[\s\.]")  # "9.2.1 Details"
_RE_SECTION = re.compile(r"^\d+\.\d+[\s\.]")  # "9.2 Mixtures"
_RE_CHAPTER_NUM = re.compile(r"^\d+\s+[A-Z]")  # "9 Introduction"
_RE_ANY_NUMBERED = re.compile(r"^\d+(\.\d+)*[\s\-\:]")  # any numbered heading


def _heading_level(text: str, font_size: float, body_font_size: float) -> int:
    """Return heading level: 1=chapter, 2=section, 3=subsection.

    Combines text patterns with font-size ratio for robustness.
    """
    if _RE_CHAPTER_KEYWORD.match(text) or _RE_PART_KEYWORD.match(text):
        return 1
    if _RE_SUBSECTION.match(text):
        return 3
    if _RE_SECTION.match(text):
        return 2
    if _RE_CHAPTER_NUM.match(text):
        return 1
    # Fall back to font-size ratio
    ratio = font_size / body_font_size if body_font_size else 1.0
    if ratio >= 1.5:
        return 1
    if ratio >= 1.2:
        return 2
    return 2


def _is_heading_candidate(
    text: str, font_size: float, is_bold: bool, body_font_size: float
) -> bool:
    """Return True if this span is a plausible heading."""
    if len(text) > 200 or len(text) < 2:
        return False
    if font_size > body_font_size * 1.2:
        return True
    if is_bold and _RE_ANY_NUMBERED.match(text):
        return True
    if _RE_CHAPTER_KEYWORD.match(text) or _RE_PART_KEYWORD.match(text):
        return True
    return False


@dataclass
class StructuredSection:
    """A contiguous section of text with its hierarchical location."""

    chapter: str
    section: str
    page_start: int
    page_end: int
    text: str


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


def _collect_headings(
    pages: list[DocumentPage], body_font_size: float
) -> list[tuple[int, int, str]]:
    """Scan all page spans and return (page_number, level, text) for headings.

    De-duplicates by (page_number, text[:80]) to avoid double-counting
    multi-span headings that repeat the same text.
    """
    headings: list[tuple[int, int, str]] = []
    seen: set[tuple[int, str]] = set()

    for page in pages:
        for span in page.spans:
            key = (page.page_number, span.text[:80])
            if key in seen:
                continue
            if _is_heading_candidate(span.text, span.font_size, span.is_bold, body_font_size):
                level = _heading_level(span.text, span.font_size, body_font_size)
                headings.append((page.page_number, level, span.text))
                seen.add(key)

    return headings


def parse_structure(doc: ExtractedDocument, book_title: str) -> ParsedDocument:
    """Build a structured section list from a raw extracted document.

    Each StructuredSection carries its chapter name, section name, page range,
    and full concatenated text. Handles flat documents (no headings detected)
    by treating the entire book as one section.

    Args:
        doc: Raw extraction result from pdf_extractor.
        book_title: Human-readable book title used as metadata.

    Returns:
        ParsedDocument whose sections cover the entire book text.
    """
    start = time.perf_counter()
    headings = _collect_headings(doc.pages, doc.body_font_size)
    page_text: dict[int, str] = {p.page_number: p.text for p in doc.pages}
    all_pages = sorted(page_text.keys())

    if not headings or not all_pages:
        logger.warning("'%s': no headings detected — treating as flat document.", book_title)
        full_text = "\n".join(page_text[p] for p in all_pages)
        parsed = ParsedDocument(
            book_title=book_title,
            source_path=doc.source_path,
            sections=[
                StructuredSection(
                    chapter="Unknown Chapter",
                    section="Full Document",
                    page_start=all_pages[0] if all_pages else 1,
                    page_end=all_pages[-1] if all_pages else 1,
                    text=full_text,
                )
            ],
        )
        _log_result(book_title, parsed, start)
        return parsed

    # Handle front matter: pages before the first heading
    sections: list[StructuredSection] = []
    first_heading_page = headings[0][0]
    if first_heading_page > all_pages[0]:
        pre_pages = [p for p in all_pages if p < first_heading_page]
        pre_text = "\n".join(page_text[p] for p in pre_pages).strip()
        if pre_text:
            sections.append(
                StructuredSection(
                    chapter="Preamble",
                    section="Front Matter",
                    page_start=all_pages[0],
                    page_end=first_heading_page - 1,
                    text=pre_text,
                )
            )

    current_chapter = "Preamble"
    current_section = "Front Matter"

    for i, (page_num, level, heading_text) in enumerate(headings):
        # Update hierarchy
        if level == 1:
            current_chapter = heading_text
            current_section = heading_text
        else:
            current_section = heading_text

        # Text spans from this heading's page up to (but not including) the next heading
        next_page = headings[i + 1][0] if i + 1 < len(headings) else all_pages[-1]
        section_pages = [p for p in all_pages if page_num <= p <= next_page]
        section_text = "\n".join(page_text[p] for p in section_pages).strip()

        if section_text:
            sections.append(
                StructuredSection(
                    chapter=current_chapter,
                    section=current_section,
                    page_start=page_num,
                    page_end=next_page,
                    text=section_text,
                )
            )

    parsed = ParsedDocument(
        book_title=book_title,
        source_path=doc.source_path,
        sections=sections,
    )
    _log_result(book_title, parsed, start)
    return parsed


def _log_result(book_title: str, parsed: ParsedDocument, start: float) -> None:
    elapsed = time.perf_counter() - start
    logger.info(
        "Parsed '%s': chapters=%d sections=%d elapsed_s=%.2f",
        book_title,
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
