"""Translate request-level retrieval constraints into Qdrant payload filters.

Pure functions only. Payload keys must match what pipeline/ingest.py writes.
"""

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from app.models import QueryRequest, RetrievalFilter

_BOOK_KEY = "book"
_CHAPTER_KEY = "chapter"
_PAGE_START_KEY = "page_start"
_PAGE_END_KEY = "page_end"


def filter_from_request(request: QueryRequest) -> RetrievalFilter:
    """Build a RetrievalFilter from the scoping fields of a QueryRequest.

    An empty book_filter list means "no book restriction", not "match nothing".
    """
    return RetrievalFilter(
        books=request.book_filter or None,
        chapter=request.chapter_filter or None,
        page_from=request.page_from,
        page_to=request.page_to,
    )


def is_unfiltered(filters: RetrievalFilter | None) -> bool:
    """Return True when *filters* imposes no constraint at all."""
    return filters is None or not filters.model_dump(exclude_none=True)


def to_qdrant_filter(filters: RetrievalFilter | None) -> Filter | None:
    """Convert a RetrievalFilter into a Qdrant Filter, or None when unconstrained.

    A page range keeps chunks that *overlap* it, so a chunk spanning pages
    41-43 matches a request for pages 42-50.
    """
    if filters is None or is_unfiltered(filters):
        return None

    conditions: list[FieldCondition] = []
    if filters.books:
        conditions.append(FieldCondition(key=_BOOK_KEY, match=MatchAny(any=filters.books)))
    if filters.chapter:
        conditions.append(FieldCondition(key=_CHAPTER_KEY, match=MatchValue(value=filters.chapter)))
    if filters.page_from is not None:
        conditions.append(FieldCondition(key=_PAGE_END_KEY, range=Range(gte=filters.page_from)))
    if filters.page_to is not None:
        conditions.append(FieldCondition(key=_PAGE_START_KEY, range=Range(lte=filters.page_to)))
    return Filter(must=conditions)
