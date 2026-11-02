"""PDF text extraction with page numbers, line-level layout hints, and the outline.

Primary extractor: PyMuPDF (fitz) for fast text + layout extraction.
Returns per-page lines with font metadata plus the PDF's own outline (ToC), so
the structure parser can locate headings without re-opening the file.
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from app.config import get_logger

logger = get_logger(__name__)

_BOLD_FLAG = 1 << 4
_MONO_FLAG = 1 << 3
_SPACE_GAP_RATIO = 0.15  # a gap wider than this fraction of the font size reads as a space


@dataclass
class LineInfo:
    """One visual line of text with the font signals the structure parser needs."""

    text: str
    font_size: float  # largest span size on the line
    is_bold: bool  # every span on the line is bold
    is_mono: bool = False  # any span is in a monospaced font (code)
    block_start: bool = False  # first line of a text block (a paragraph boundary)


@dataclass
class DocumentPage:
    """Extracted content of one PDF page."""

    page_number: int  # 1-indexed
    text: str  # full plain text of the page
    lines: list[LineInfo] = field(default_factory=list)


@dataclass
class TocEntry:
    """One entry of the PDF's embedded outline (bookmarks)."""

    level: int  # 1 = top level
    title: str
    page: int  # 1-indexed target page


@dataclass
class ExtractedDocument:
    """Raw extraction result for one PDF file."""

    source_path: str
    title: str  # filename stem; used as the default book key
    pages: list[DocumentPage]
    body_font_size: float  # font size carrying the most characters (body text estimate)
    toc: list[TocEntry] = field(default_factory=list)


def _join_spans(spans: list[dict]) -> str:
    """Concatenate span texts, adding a space where spans sit visibly apart.

    PDFs often position "2.1" and "Introduction" as separate spans with a gap
    but no space character between them.
    """
    parts = [spans[0]["text"]]
    for prev, span in zip(spans, spans[1:]):
        gap = span["bbox"][0] - prev["bbox"][2]
        touching_space = prev["text"][-1:].isspace() or span["text"][:1].isspace()
        if not touching_space and gap > _SPACE_GAP_RATIO * float(span.get("size", 0.0)):
            parts.append(" ")
        parts.append(span["text"])
    return "".join(parts)


def _line_info(line: dict, block_start: bool) -> LineInfo | None:
    """Build a LineInfo from a PyMuPDF line dict; None if the line has no text."""
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return None
    text = _join_spans(spans).strip()
    return LineInfo(
        text=text,
        font_size=max(float(s.get("size", 0.0)) for s in spans),
        is_bold=all(s.get("flags", 0) & _BOLD_FLAG for s in spans),
        is_mono=any(s.get("flags", 0) & _MONO_FLAG for s in spans),
        block_start=block_start,
    )


def _page_lines(page: fitz.Page, size_chars: Counter) -> list[LineInfo]:
    """Return the page's lines in reading order, tallying characters per font size."""
    lines: list[LineInfo] = []
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue
        first = True
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = round(float(span.get("size", 0.0)), 1)
                if size > 0:
                    size_chars[size] += len(span.get("text", "").strip())
            info = _line_info(line, block_start=first)
            if info is not None:
                lines.append(info)
                first = False
    return lines


def _read_toc(doc: fitz.Document) -> list[TocEntry]:
    """Return the outline entries that point at a real page."""
    entries = []
    for level, title, page in doc.get_toc(simple=True):
        title = " ".join(str(title).split())
        if title and 1 <= page <= len(doc):
            entries.append(TocEntry(level=level, title=title, page=page))
    return entries


def extract_pdf(path: str | Path) -> ExtractedDocument:
    """Extract text, line font hints, and the outline from a PDF, preserving page boundaries.

    Args:
        path: Path to the PDF file.

    Returns:
        ExtractedDocument with per-page lines and the signals needed by the structure parser.

    Raises:
        FileNotFoundError: if the file does not exist.
        fitz.FileDataError: if the file is not a valid PDF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    start = time.perf_counter()
    doc = fitz.open(str(path))
    pages: list[DocumentPage] = []
    # Weighted by characters, not spans: syntax-highlighted code splits into many tiny
    # spans and would otherwise drag the estimate below the real body size.
    size_chars: Counter = Counter()

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        lines = _page_lines(page, size_chars)
        plain_text = page.get_text("text")
        if not plain_text.strip():
            logger.warning(
                "Page %d of '%s' yielded no text — possible scan page (OCR not attempted in v1).",
                page_number,
                path.name,
            )
        pages.append(DocumentPage(page_number=page_number, text=plain_text, lines=lines))

    toc = _read_toc(doc)
    doc.close()

    body_font_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
    elapsed = time.perf_counter() - start
    logger.info(
        "Extracted '%s': pages=%d body_font_size=%.1f toc_entries=%d elapsed_s=%.2f",
        path.name,
        len(pages),
        body_font_size,
        len(toc),
        elapsed,
    )

    return ExtractedDocument(
        source_path=str(path),
        title=path.stem,
        pages=pages,
        body_font_size=body_font_size,
        toc=toc,
    )
