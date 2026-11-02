"""Text preprocessing: normalize whitespace, fix hyphenation, strip artifacts.

Operates on StructuredSection text inside a ParsedDocument. Safe for technical
text: Greek letters, math symbols, and equations are not altered (NFC
normalization preserves composed characters without destroying them).
"""

import re
import unicodedata
from dataclasses import replace

from app.config import get_logger

from pipeline.extractors.structure_parser import ParsedDocument, StructuredSection

logger = get_logger(__name__)

# "estima-\ntion" → "estimation"
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

# Collapse runs of spaces/tabs to a single space (newlines are kept)
_MULTI_SPACE = re.compile(r"[ \t]+")

# Collapse 3+ consecutive newlines to exactly 2 (preserves paragraph structure)
_TRIPLE_NEWLINE = re.compile(r"\n{3,}")

# Standalone page-number lines: a line containing only 1–4 digits
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def clean_text(text: str) -> str:
    """Clean a single text string extracted from a PDF section.

    Operations applied in order:
    1. Unicode NFC normalization — safe for Greek/math symbols.
    2. Fix soft hyphenation at line breaks.
    3. Collapse runs of spaces/tabs to a single space.
    4. Strip standalone page-number lines.
    5. Collapse excess blank lines to at most one blank line.

    Args:
        text: Raw section text from the structure parser.

    Returns:
        Cleaned text stripped of leading/trailing whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _PAGE_NUMBER_LINE.sub("", text)
    text = _TRIPLE_NEWLINE.sub("\n\n", text)
    return text.strip()


def _clean_section(section: StructuredSection) -> StructuredSection:
    """Clean one section, keeping its page offsets aligned with the cleaned text.

    Each page's slice is cleaned on its own (so a word hyphenated across a page
    break stays split), then the slices are rejoined with a line break.
    """
    if not section.page_offsets:
        return replace(section, text=clean_text(section.text))
    ends = [offset for offset, _ in section.page_offsets[1:]] + [len(section.text)]
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    for (start, page), end in zip(section.page_offsets, ends):
        piece = clean_text(section.text[start:end])
        if not piece:
            continue  # e.g. a page holding only its page number
        if pieces:
            pieces.append("\n")
            position += 1
        offsets.append((position, page))
        pieces.append(piece)
        position += len(piece)
    return replace(section, text="".join(pieces), page_offsets=offsets)


def preprocess_document(doc: ParsedDocument) -> ParsedDocument:
    """Apply clean_text to every section in a ParsedDocument.

    Returns a new ParsedDocument with cleaned section texts; the input is not
    mutated.

    Args:
        doc: Structured document from the structure parser.

    Returns:
        ParsedDocument with identical structure but cleaned text in every section.
    """
    cleaned_sections = [_clean_section(s) for s in doc.sections]

    logger.info("Preprocessed '%s': %d sections cleaned.", doc.book_title, len(cleaned_sections))
    return ParsedDocument(
        book_title=doc.book_title,
        source_path=doc.source_path,
        sections=cleaned_sections,
    )
