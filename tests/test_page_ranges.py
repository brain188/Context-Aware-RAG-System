"""Page offsets flow parser → preprocessor → chunker, so sub-chunks cite their own pages."""

import os
from unittest.mock import patch

import pytest

from app.config import get_settings
from pipeline.chunker import _pages_of, chunk_document
from pipeline.extractors.pdf_extractor import DocumentPage, ExtractedDocument, LineInfo, TocEntry
from pipeline.extractors.structure_parser import ParsedDocument, StructuredSection, parse_structure
from pipeline.preprocessor import preprocess_document


@pytest.fixture(autouse=True)
def small_chunks():
    env = {
        "GEMINI_API_KEY": "test-key",
        "LLM_PROVIDER": "gemini",  # beats whatever infra/.env says
        "LLM_FALLBACK_PROVIDER": "",
        "CHUNK_MIN_TOKENS": "20",
        "CHUNK_MAX_TOKENS": "60",
        "CHUNK_OVERLAP_TOKENS": "5",
    }
    get_settings.cache_clear()
    with patch.dict(os.environ, env, clear=False):
        yield
    get_settings.cache_clear()


def _page(number: int, *texts: str) -> DocumentPage:
    lines = [LineInfo(text=t, font_size=10.0, is_bold=False, block_start=True) for t in texts]
    return DocumentPage(page_number=number, text="\n".join(texts), lines=lines)


def _one_section_book(pages: list[DocumentPage]) -> ParsedDocument:
    """A book whose outline has one chapter starting on the first page."""
    doc = ExtractedDocument(
        source_path="/books/t.pdf",
        title="t",
        pages=pages,
        body_font_size=10.0,
        toc=[TocEntry(1, "Intro", 1), TocEntry(1, "Next", 99), TocEntry(1, "Last", 99)],
    )
    return parse_structure(doc, "Test Book")


def _words(tag: str, n: int) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


class TestPageOffsets:
    def test_parser_records_where_each_page_begins(self) -> None:
        (section,) = _one_section_book([_page(1, "Intro", "alpha"), _page(2, "beta")]).sections
        assert section.page_offsets == [(0, 1), (len("Intro\n\nalpha\n\n"), 2)]
        assert section.text[section.page_offsets[1][0] :] == "beta"

    def test_preprocessing_keeps_offsets_aligned_with_cleaned_text(self) -> None:
        parsed = _one_section_book(
            [_page(1, "Intro", "spaced    out"), _page(2, "12"), _page(3, "gamma   delta")]
        )
        (section,) = preprocess_document(parsed).sections
        assert section.text == "Intro\n\nspaced out\ngamma delta"
        # Page 2 held only its page number, so it has no offset of its own.
        assert [page for _, page in section.page_offsets] == [1, 3]
        assert section.text[section.page_offsets[1][0] :] == "gamma delta"

    def test_sub_chunks_of_a_long_section_cite_their_own_pages(self) -> None:
        pages = [_page(1, "Intro", _words("a", 50)), _page(2, _words("b", 50))]
        pages.append(_page(3, _words("c", 50)))
        chunks = chunk_document(preprocess_document(_one_section_book(pages)))
        assert len(chunks) >= 3
        for chunk in chunks:
            tags = {word[0] for word in chunk.text.split() if word != "Intro"}
            cited = range(chunk.metadata.page_start, chunk.metadata.page_end + 1)
            assert {"abc".index(t) + 1 for t in tags} == set(cited), chunk.text
        assert chunks[0].metadata.page_start == 1
        assert chunks[-1].metadata.page_end == 3

    def test_unknown_offsets_fall_back_to_the_section_range(self) -> None:
        section = StructuredSection("Ch", "S", 4, 9, "some text")
        assert _pages_of(section, 0, 4) == (4, 9)

    def test_merged_small_sections_cite_every_merged_page(self) -> None:
        doc = ExtractedDocument(
            source_path="/books/t.pdf",
            title="t",
            pages=[_page(1, "A", "tiny a"), _page(2, "B", "tiny b"), _page(3, "C", "tiny c")],
            body_font_size=10.0,
            toc=[TocEntry(1, "Ch", 1), TocEntry(2, "A", 1), TocEntry(2, "B", 2)]
            + [TocEntry(2, "C", 3), TocEntry(1, "X", 99), TocEntry(1, "Y", 99)],
        )
        (chunk,) = chunk_document(preprocess_document(parse_structure(doc, "Test Book")))
        assert (chunk.metadata.page_start, chunk.metadata.page_end) == (1, 3)
