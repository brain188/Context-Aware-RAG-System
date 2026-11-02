"""Locate chapter and section headings in an extracted document.

Two sources, best first:
1. The PDF outline (ToC bookmarks): exact titles and levels. Each entry is
   pinned to the line where its title is printed, so a page with several
   headings splits between them instead of repeating the whole page.
2. Font signals, for books without an outline: larger or bold lines that read
   like a title, never code, and not a running header repeated across pages.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.config import get_logger

from pipeline.extractors.pdf_extractor import LineInfo, TocEntry

logger = get_logger(__name__)

# A line in reading order across the whole book, tagged with its page number.
PageLine = tuple[int, LineInfo]

_RE_CHAPTER_KEYWORD = re.compile(r"^chapter\s+\d+", re.IGNORECASE)
_RE_PART_KEYWORD = re.compile(r"^part\s+([ivx]+|\d+)\b", re.IGNORECASE)
_RE_SECTION = re.compile(r"^\d+\.\d+(\.\d+)*[\s\.]")  # "9.2 Mixtures", "9.2.1 Details"
# "1 Intro", "1.2 Intro", "Chapter 3: Intro", "IV. Intro" → the title without its number.
_RE_LEADING_NUMBER = re.compile(
    r"^(chapter|part|section)?\s*([ivx]+|\d+)(\.\d+)*[.:]?\s+", re.IGNORECASE
)
# Calls, assignments, snake_case, markup: code that happens to be printed large or bold.
_RE_CODE = re.compile(r"\w\(|[=;{}<>`|\\_]")

_RE_BARE_NUMBER = re.compile(r"\d{1,2}|[IVX]{1,5}")  # a chapter number on its own line

_MAX_TITLE_LINES = 3  # a printed title may wrap onto this many lines
_RUNNING_HEADER_PAGES = 4  # a candidate on this many pages is a header/footer, not a heading


@dataclass(frozen=True)
class Heading:
    """A heading pinned to the line where its section starts."""

    line_index: int  # index into the book's PageLine list
    level: int  # 1 = chapter, 2 = section
    title: str


def _norm(text: str) -> str:
    """Casefolded alphanumerics only, so spacing, punctuation, and quote styles don't matter."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())


def _page_bounds(lines: list[PageLine]) -> dict[int, tuple[int, int]]:
    """Map each page number to its [first, last + 1) line indexes."""
    bounds: dict[int, tuple[int, int]] = {}
    for i, (page, _) in enumerate(lines):
        first, _end = bounds.get(page, (i, i))
        bounds[page] = (first, i + 1)
    return bounds


# ---------------------------------------------------------------------------
# Outline (ToC) headings
# ---------------------------------------------------------------------------


def _chapter_level(toc: list[TocEntry]) -> int | None:
    """Return the outline level that holds chapters, skipping a lone root or a level of parts."""
    by_level: dict[int, list[str]] = defaultdict(list)
    for entry in toc:
        by_level[entry.level].append(entry.title)
    for level in sorted(by_level):
        titles = by_level[level]
        parts = sum(1 for t in titles if _RE_PART_KEYWORD.match(t))
        if len(titles) >= 3 and parts < len(titles) / 2:
            return level
    return None


def _title_starts_at(lines: list[PageLine], i: int, target: str) -> bool:
    """True if *target* is printed starting at line *i* (possibly wrapped onto later lines)."""
    page = lines[i][0]
    printed = ""
    for j in range(i, min(i + _MAX_TITLE_LINES, len(lines))):
        if lines[j][0] != page:
            return False
        printed += _norm(lines[j][1].text)
        if not printed or not target.startswith(printed):
            return False
        if printed == target:
            return True
    return False


def _title_targets(title: str) -> set[str]:
    """Normalized forms a printed title may take: as listed, bare, or number + bare title.

    "Chapter 3: Kernels" may be printed as is, as "Kernels", or as "3" above "Kernels".
    """
    bare = _RE_LEADING_NUMBER.sub("", title)
    forms = {_norm(title), _norm(bare)}
    number = _RE_LEADING_NUMBER.match(title)
    if number:
        forms.add(_norm(number.group(2) + bare))
    return forms - {""}


def _locate(
    lines: list[PageLine], bounds: dict, entry: TocEntry, prev: int | None
) -> tuple[int, bool] | None:
    """Return (line index where *entry* starts, whether its title was found printed).

    Searches the outline's target page and the next one (outlines are often off by
    one), never before the previous heading *prev*. If the title isn't printed as a
    line, the section starts at the top of its page. None when it can't be placed
    without taking text from the previous section (its page is at or before
    *prev*'s) or its page lies past the last line.
    """
    targets = _title_targets(entry.title)
    floor = 0 if prev is None else prev
    for page in (entry.page, entry.page + 1):
        first, end = bounds.get(page, (0, 0))
        for i in range(max(first, floor), end):
            if any(_title_starts_at(lines, i, t) for t in targets):
                return i, True
    if prev is not None and entry.page <= lines[prev][0]:
        return None
    for i in range(floor, len(lines)):
        if lines[i][0] >= entry.page:
            return i, False
    return None


def toc_headings(toc: list[TocEntry], lines: list[PageLine], name: str) -> list[Heading]:
    """Pin outline entries to lines; [] when the outline is missing or has no chapter level."""
    chapter_level = _chapter_level(toc)
    if chapter_level is None or not lines:
        return []
    bounds = _page_bounds(lines)
    headings: list[Heading] = []
    prev: int | None = None
    printed = 0
    for entry in toc:
        if entry.level < chapter_level:
            continue  # book root or "Part" grouping: not a chapter label
        found = _locate(lines, bounds, entry, prev)
        if found is None:
            logger.debug("'%s': outline entry '%s' can't be placed; skipped.", name, entry.title)
            continue
        prev, was_printed = found
        printed += was_printed
        level = 1 if entry.level == chapter_level else 2
        headings.append(Heading(line_index=prev, level=level, title=entry.title))
    logger.info(
        "'%s': %d outline headings, %d found printed on their page.", name, len(headings), printed
    )
    return headings


# ---------------------------------------------------------------------------
# Font-signal headings (fallback)
# ---------------------------------------------------------------------------


def _reads_like_title(text: str) -> bool:
    """Short, starts with a letter or digit, mostly letters, and not code."""
    if not 2 <= len(text) <= 120 or len(text.split()) > 15:
        return False
    if not text[0].isalnum() or text[-1] in ",;([{=+":
        return False
    if _RE_CODE.search(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 3 and letters >= 0.5 * len(text.replace(" ", ""))


def _is_candidate(line: LineInfo, body: float) -> bool:
    """A paragraph-opening line set apart by size or weight that reads like a title."""
    if line.is_mono or not line.block_start or not _reads_like_title(line.text):
        return False
    if _RE_CHAPTER_KEYWORD.match(line.text) or line.font_size >= body * 1.2:
        return True
    if line.is_bold and line.font_size >= body * 1.05:
        return True
    return line.is_bold and bool(_RE_SECTION.match(line.text))


def _is_chapter_number(lines: list[PageLine], i: int, body: float) -> bool:
    """A bare, large chapter number printed on its own line directly above the title."""
    page, line = lines[i]
    if not (_RE_BARE_NUMBER.fullmatch(line.text) and line.block_start):
        return False
    if line.font_size < body * 1.5 or i + 1 >= len(lines):
        return False
    next_page, title = lines[i + 1]
    return next_page == page and not title.is_mono and _reads_like_title(title.text)


def _level(text: str, font_size: float, body: float) -> int:
    """1 for chapters (keyword or much larger type), else 2."""
    if _RE_CHAPTER_KEYWORD.match(text) or _RE_PART_KEYWORD.match(text):
        return 1
    if _RE_SECTION.match(text):
        return 2
    return 1 if font_size >= body * 1.5 else 2


def _continuation(lines: list[PageLine], i: int) -> int:
    """Return how many lines after *i* continue the same wrapped title."""
    page, head = lines[i]
    extra = 0
    while extra + 1 < _MAX_TITLE_LINES and i + extra + 1 < len(lines):
        next_page, nxt = lines[i + extra + 1]
        same_style = abs(nxt.font_size - head.font_size) < 0.3 and nxt.is_bold == head.is_bold
        if next_page != page or nxt.is_mono or not same_style or len(nxt.text) > 120:
            break
        extra += 1
    return extra


def font_headings(lines: list[PageLine], body_font_size: float) -> list[Heading]:
    """Find headings from font size/weight, dropping running headers and code."""
    found: list[tuple[int, str, LineInfo, int]] = []
    i = 0
    while i < len(lines):
        page, line = lines[i]
        if _is_chapter_number(lines, i, body_font_size):
            extra = 1 + _continuation(lines, i + 1)  # "9" above "Mixture Models": one title
        elif _is_candidate(line, body_font_size):
            extra = _continuation(lines, i)
        else:
            i += 1
            continue
        title = " ".join(text.text for _, text in lines[i : i + extra + 1])
        found.append((i, title, line, page))
        i += extra + 1

    pages_per_title = Counter()
    for title, page in {(_norm(t), p) for _, t, _, p in found}:
        pages_per_title[title] += 1
    return [
        Heading(line_index=i, level=_level(title, line.font_size, body_font_size), title=title)
        for i, title, line, _ in found
        if pages_per_title[_norm(title)] < _RUNNING_HEADER_PAGES
    ]
