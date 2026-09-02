"""PDF text extraction with page numbers and font-size heading hints.

Primary extractor: PyMuPDF (fitz) for fast text + layout extraction.
Returns per-page text with span-level font metadata so the structure parser
can identify headings without re-opening the file.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import fitz  # PyMuPDF
from app.config import get_logger

logger = get_logger(__name__)


@dataclass
class SpanInfo:
    """A text span with font metadata from the PDF layer."""

    text: str
    font_size: float
    is_bold: bool


@dataclass
class DocumentPage:
    """Extracted content of one PDF page."""

    page_number: int  # 1-indexed
    text: str  # full plain text of the page
    spans: list[SpanInfo] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    """Raw extraction result for one PDF file."""

    source_path: str
    title: str  # filename stem; used as the default book key
    pages: list[DocumentPage]
    body_font_size: float  # median font size across all spans (body text estimate)


def extract_pdf(path: str | Path) -> ExtractedDocument:
    """Extract text and font-size hints from a PDF, preserving page boundaries.

    Args:
        path: Path to the PDF file.

    Returns:
        ExtractedDocument with per-page text and font-size signals needed by
        the structure parser.

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
    all_font_sizes: list[float] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        spans: list[SpanInfo] = []

        raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in raw_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    size = float(span.get("size", 0.0))
                    is_bold = bool(span.get("flags", 0) & (1 << 4))
                    spans.append(SpanInfo(text=text, font_size=size, is_bold=is_bold))
                    if size > 0:
                        all_font_sizes.append(size)

        plain_text = page.get_text("text")
        if not plain_text.strip():
            logger.warning(
                "Page %d of '%s' yielded no text — possible scan page (OCR not attempted in v1).",
                page_number,
                path.name,
            )

        pages.append(DocumentPage(page_number=page_number, text=plain_text, spans=spans))

    doc.close()

    body_font_size = median(all_font_sizes) if all_font_sizes else 10.0
    elapsed = time.perf_counter() - start
    logger.info(
        "Extracted '%s': pages=%d body_font_size=%.1f elapsed_s=%.2f",
        path.name,
        len(pages),
        body_font_size,
        elapsed,
    )

    return ExtractedDocument(
        source_path=str(path),
        title=path.stem,
        pages=pages,
        body_font_size=body_font_size,
    )
