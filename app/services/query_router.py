"""Classify a query as a simple lookup or a multi-hop / cross-book question.

v1 is a heuristic. Single-pass RAG still answers both kinds; the decision is
logged so the Phase 5 agents layer can take over multi-hop queries later.
"""

import re
from typing import Literal

from pydantic import BaseModel

RouteKind = Literal["simple", "multi_hop"]

_COMPARISON = re.compile(
    # "differ", not "different": "the different types of X" is a single lookup.
    r"\b(compare[sd]?|comparison|contrast|differences?|differs?|versus|vs\.?|"
    r"similarit(y|ies)|relationship between|trade-?offs?)\b",
    re.IGNORECASE,
)


class RouteDecision(BaseModel):
    """Which path the query should take, and why."""

    kind: RouteKind
    reason: str


def route_query(question: str, known_books: list[str]) -> RouteDecision:
    """Return multi_hop for comparisons, several questions, or several books named."""
    lowered = question.lower()
    named = [title for title in known_books if title.lower() in lowered]
    if len(named) >= 2:
        return RouteDecision(kind="multi_hop", reason=f"names {len(named)} books")
    if _COMPARISON.search(question):
        return RouteDecision(kind="multi_hop", reason="comparison wording")
    if question.count("?") >= 2:
        return RouteDecision(kind="multi_hop", reason="several questions")
    return RouteDecision(kind="simple", reason="single lookup")
