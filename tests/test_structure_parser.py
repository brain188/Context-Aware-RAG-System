"""Tests for pipeline/extractors: outline and font-signal headings, sections, extraction.

Most tests build ExtractedDocuments by hand; the extraction tests write small
PDFs with PyMuPDF into tmp_path.
"""

from pathlib import Path

import fitz
import pytest

from pipeline.extractors.headings import font_headings, toc_headings
from pipeline.extractors.pdf_extractor import (
    DocumentPage,
    ExtractedDocument,
    LineInfo,
    TocEntry,
    _join_spans,
    extract_pdf,
)
from pipeline.extractors.structure_parser import parse_structure

BODY = 10.0


def _line(text: str, size: float = BODY, bold: bool = False, start: bool = True, mono=False):
    return LineInfo(text=text, font_size=size, is_bold=bold, is_mono=mono, block_start=start)


def _doc(pages: list[list[LineInfo]], toc: tuple = ()) -> ExtractedDocument:
    """Build a document from per-page lines and (level, title, page) outline entries."""
    return ExtractedDocument(
        source_path="/books/test.pdf",
        title="test",
        pages=[
            DocumentPage(page_number=n, text="\n".join(x.text for x in lines), lines=lines)
            for n, lines in enumerate(pages, start=1)
        ],
        body_font_size=BODY,
        toc=[TocEntry(level=lv, title=t, page=p) for lv, t, p in toc],
    )


def _sections(doc: ExtractedDocument) -> dict:
    return {s.section: s for s in parse_structure(doc, "Test Book").sections}


# ---------------------------------------------------------------------------
# Outline (ToC) headings
# ---------------------------------------------------------------------------


class TestOutlineHeadings:
    def test_headings_on_one_page_split_it_instead_of_repeating_it(self) -> None:
        doc = _doc(
            [
                [
                    _line("1 Basics", 20),
                    _line("1.1 Mean", 12, bold=True),
                    _line("The mean is the average."),
                    _line("1.2 Median", 12, bold=True),
                    _line("The median is the middle value."),
                ]
            ],
            toc=[(1, "1 Basics", 1), (2, "1.1 Mean", 1), (2, "1.2 Median", 1)],
        )
        sections = _sections(doc)
        assert sections["1.1 Mean"].text == "1.1 Mean\n\nThe mean is the average."
        assert sections["1.2 Median"].text == "1.2 Median\n\nThe median is the middle value."
        assert "median" not in sections["1.1 Mean"].text.lower()

    def test_sections_carry_their_chapter_and_own_page_range(self) -> None:
        doc = _doc(
            [
                [_line("1 Basics", 20), _line("Intro text.")],
                [_line("1.1 Mean", 12, bold=True), _line("Mean text.")],
                [_line("More mean text.")],
                [_line("2 Models", 20), _line("Model text.")],
            ],
            toc=[(1, "1 Basics", 1), (2, "1.1 Mean", 2), (1, "2 Models", 4)],
        )
        sections = _sections(doc)
        mean = sections["1.1 Mean"]
        assert (mean.chapter, mean.page_start, mean.page_end) == ("1 Basics", 2, 3)
        assert sections["1 Basics"].page_end == 1
        assert sections["2 Models"].chapter == "2 Models"

    def test_title_wrapped_over_two_lines_is_found(self) -> None:
        lines = [
            (1, _line("Some earlier text.")),
            (1, _line("4. Understanding Django's MVC", 24)),
            (1, _line("(MTV) Architecture", 24, start=False)),
        ]
        toc = [TocEntry(1, "4. Understanding Django’s MVC (MTV) Architecture", 1)]
        toc += [TocEntry(1, f"{n}. Other", 2) for n in (5, 6)]
        lines.append((2, _line("5. Other", 24)))
        first, *_ = toc_headings(toc, lines, "t")
        assert first.line_index == 1

    def test_title_printed_without_its_number_is_found(self) -> None:
        doc = _doc(
            [
                [_line("Front text.")],
                [_line("3", 30), _line("Kernels", 16, start=False), _line("Kernel text.")],
            ],
            toc=[(1, "Preface", 1), (1, "Chapter 3: Kernels", 2), (1, "Index", 3)],
        )
        kernels = _sections(doc)["Chapter 3: Kernels"]
        assert kernels.text == "3\nKernels\n\nKernel text."
        assert kernels.page_start == 2

    def test_unprinted_entry_on_the_previous_headings_page_is_skipped(self) -> None:
        doc = _doc(
            [[_line("A", 20), _line("A text.")], [_line("B", 20), _line("B text.")]],
            toc=[(1, "A", 1), (1, "Ghost", 1), (1, "B", 2)],
        )
        sections = _sections(doc)
        assert "Ghost" not in sections
        assert sections["A"].text == "A\n\nA text."

    def test_outline_page_off_by_one_is_tolerated(self) -> None:
        doc = _doc(
            [[_line("Cover.")], [_line("Mixtures", 16), _line("Mixture text.")]],
            toc=[(1, "A", 1), (1, "B", 1), (1, "Mixtures", 1)],
        )
        assert _sections(doc)["Mixtures"].text == "Mixtures\n\nMixture text."

    def test_unprinted_title_starts_at_the_top_of_its_page(self) -> None:
        doc = _doc(
            [[_line("Page one.")], [_line("Page two.")], [_line("Page three.")]],
            toc=[(1, "One", 1), (1, "Two", 2), (1, "Three", 3)],
        )
        assert _sections(doc)["Two"].text == "Page two."

    def test_lone_root_entry_is_not_a_chapter(self) -> None:
        doc = _doc(
            [[_line(f"Chapter {n}", 20), _line(f"Text {n}.")] for n in (1, 2, 3)],
            toc=[(1, "My Book", 1)] + [(2, f"Chapter {n}", n) for n in (1, 2, 3)],
        )
        chapters = {s.chapter for s in parse_structure(doc, "Test Book").sections}
        assert chapters == {"Chapter 1", "Chapter 2", "Chapter 3"}

    def test_part_level_is_skipped_for_chapter_labels(self) -> None:
        toc = [(1, "Part I", 1), (2, "1 A", 1), (2, "2 B", 2), (1, "Part II", 3), (2, "3 C", 3)]
        toc += [(1, "Part III", 3)]
        doc = _doc([[_line("1 A", 20)], [_line("2 B", 20)], [_line("3 C", 20)]], toc=toc)
        chapters = [s.chapter for s in parse_structure(doc, "Test Book").sections]
        assert chapters == ["1 A", "2 B", "3 C"]

    def test_text_before_the_first_heading_is_front_matter(self) -> None:
        doc = _doc(
            [[_line("Copyright notice.")], [_line("1 A", 20)], [_line("2 B", 20)]],
            toc=[(1, "1 A", 2), (1, "2 B", 3), (1, "Index", 3)],
        )
        front = parse_structure(doc, "Test Book").sections[0]
        assert (front.chapter, front.section, front.text) == (
            "Preamble",
            "Front Matter",
            "Copyright notice.",
        )

    def test_entry_pointing_back_before_the_previous_heading_is_skipped(self) -> None:
        doc = _doc(
            [[_line("A", 20)], [_line("B", 20)], [_line("C", 20)]],
            toc=[(1, "A", 1), (1, "C", 3), (1, "Ghost", 1), (1, "B", 2)],
        )
        titles = [s.section for s in parse_structure(doc, "Test Book").sections]
        assert "Ghost" not in titles
        assert titles[:2] == ["A", "C"]

    def test_outline_without_a_chapter_level_falls_back_to_fonts(self) -> None:
        doc = _doc([[_line("Chapter 1 Basics", 20), _line("Text.")]], toc=[(1, "Only", 1)])
        assert [s.chapter for s in parse_structure(doc, "Test Book").sections] == [
            "Chapter 1 Basics"
        ]


# ---------------------------------------------------------------------------
# Font-signal headings
# ---------------------------------------------------------------------------


def _font_titles(lines: list[LineInfo], page: int = 1) -> list[tuple[int, str]]:
    return [(h.level, h.title) for h in font_headings([(page, x) for x in lines], BODY)]


class TestFontHeadings:
    def test_large_and_bold_lines_become_chapters_and_sections(self) -> None:
        titles = _font_titles(
            [_line("Mixture Models", 20), _line("Text."), _line("The EM Algorithm", 11, True)]
        )
        assert titles == [(1, "Mixture Models"), (2, "The EM Algorithm")]

    @pytest.mark.parametrize(
        "code",
        [
            "admin.site.register(MyModel)",
            ", admin.site.urls),",
            "price = models.DecimalField",
            "(MyModel)",
            "10",
            "my_view",
        ],
    )
    def test_code_and_numbers_in_large_type_are_not_headings(self, code: str) -> None:
        assert _font_titles([_line(code, 16)]) == []

    def test_monospaced_lines_are_not_headings(self) -> None:
        assert _font_titles([_line("Install Django", 16, mono=True)]) == []

    def test_mid_paragraph_lines_are_not_headings(self) -> None:
        assert _font_titles([_line("Emphasised words", 16, start=False)]) == []

    def test_wrapped_heading_is_one_title(self) -> None:
        titles = _font_titles([_line("Understanding the", 20), _line("Architecture", 20, False)])
        assert titles == [(1, "Understanding the Architecture")]

    def test_chapter_number_above_its_title_is_one_chapter(self) -> None:
        titles = _font_titles([_line("9", 36, True), _line("Mixture Models", 16, True, False)])
        assert titles == [(1, "9 Mixture Models")]

    def test_numbered_bold_line_at_body_size_is_a_section(self) -> None:
        assert _font_titles([_line("9.2 Mixtures of Gaussians", BODY, bold=True)]) == [
            (2, "9.2 Mixtures of Gaussians")
        ]

    def test_running_header_repeated_across_pages_is_dropped(self) -> None:
        lines = [(p, _line("Machine Learning", 13)) for p in range(1, 6)]
        lines.append((6, _line("Real Heading", 13)))
        assert [h.title for h in font_headings(lines, BODY)] == ["Real Heading"]

    def test_document_without_headings_is_one_flat_section(self) -> None:
        (section,) = parse_structure(_doc([[_line("a")], [_line("b")]]), "Test Book").sections
        assert (section.chapter, section.section) == ("Unknown Chapter", "Full Document")
        assert (section.page_start, section.page_end, section.text) == (1, 2, "a\n\nb")

    def test_document_without_lines_has_no_sections(self) -> None:
        assert parse_structure(_doc([[], []]), "Test Book").sections == []


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def _write_pdf(path: Path, pages: list[list[tuple[str, float, str]]], toc: list | None = None):
    """Write a PDF whose pages hold (text, font size, font name) lines, one per block."""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72.0
        for text, size, font in lines:
            page.insert_text((72, y), text, fontsize=size, fontname=font)
            y += size * 2
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return path


class TestExtractPdf:
    def test_reads_outline_lines_and_fonts(self, tmp_path: Path) -> None:
        pdf = _write_pdf(
            tmp_path / "book.pdf",
            [[("Kernels", 20, "hebo"), ("A kernel is a similarity function.", 10, "helv")]],
            toc=[[1, "Kernels", 1]],
        )
        doc = extract_pdf(pdf)
        assert doc.toc == [TocEntry(level=1, title="Kernels", page=1)]
        heading, body = doc.pages[0].lines
        assert (heading.text, heading.is_bold, heading.font_size) == ("Kernels", True, 20)
        assert (body.text, body.is_bold, body.block_start) == (
            "A kernel is a similarity function.",
            False,
            True,
        )

    def test_body_size_is_weighted_by_characters_not_spans(self, tmp_path: Path) -> None:
        many_short = [(f"x{i}", 8, "helv") for i in range(15)]
        pdf = _write_pdf(tmp_path / "b.pdf", [[("word " * 40, 11, "helv")] + many_short])
        assert extract_pdf(pdf).body_font_size == 11

    def test_parsed_real_pdf_sections_follow_the_outline(self, tmp_path: Path) -> None:
        pages = [
            [("1 Basics", 20, "hebo"), ("Basics text.", 10, "helv")],
            [("2 Models", 20, "hebo"), ("Models text.", 10, "helv")],
            [("3 Fitting", 20, "hebo"), ("Fitting text.", 10, "helv")],
        ]
        toc = [[1, "1 Basics", 1], [1, "2 Models", 2], [1, "3 Fitting", 3]]
        parsed = parse_structure(extract_pdf(_write_pdf(tmp_path / "c.pdf", pages, toc)), "B")
        assert [(s.chapter, s.page_start, s.text) for s in parsed.sections] == [
            ("1 Basics", 1, "1 Basics\n\nBasics text."),
            ("2 Models", 2, "2 Models\n\nModels text."),
            ("3 Fitting", 3, "3 Fitting\n\nFitting text."),
        ]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_pdf(tmp_path / "nope.pdf")


class TestJoinSpans:
    def _span(self, text: str, x0: float, x1: float) -> dict:
        return {"text": text, "bbox": (x0, 0, x1, 10), "size": 10.0}

    def test_visible_gap_becomes_a_space(self) -> None:
        spans = [self._span("2.1", 0, 14), self._span("Introduction", 20, 80)]
        assert _join_spans(spans) == "2.1 Introduction"

    def test_touching_spans_are_concatenated(self) -> None:
        spans = [self._span("Sci", 0, 14), self._span("ence", 14.5, 30)]
        assert _join_spans(spans) == "Science"

    def test_existing_space_is_not_doubled(self) -> None:
        spans = [self._span("of ", 0, 14), self._span("data", 20, 40)]
        assert _join_spans(spans) == "of data"
