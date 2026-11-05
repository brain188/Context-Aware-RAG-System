"""Retrieval-grading prompt and the structured verdict the grader model fills.

The grade drives the retrieval stage: correct → generate, ambiguous → rewrite
the search query and retrieve again, none → "not covered". Per-excerpt
relevance is recorded for the query log and for checking the grader against
hand labels. The prompt text itself is versioned in templates/.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.prompts import load_prompt

Grade = Literal["correct", "ambiguous", "none"]
Relevance = Literal["relevant", "partial", "irrelevant"]


class SourceRelevance(BaseModel):
    """How much one numbered excerpt ([S<source>]) helps answer the question."""

    source: int = Field(description="The excerpt's marker number: 1 for [S1], 2 for [S2].")
    relevance: Relevance


class GradeReply(BaseModel):
    """The grader model's structured verdict."""

    grade: Grade
    reason: str = Field(
        default="", description="One short sentence; for ambiguous, what is missing."
    )
    sources: list[SourceRelevance] = Field(default_factory=list)


def build_grading_prompt(question: str, sources: str, version: str) -> tuple[str, str]:
    """Return (system, user) prompts for grading *sources* against *question*."""
    system = load_prompt("grading_system", version).render()
    user = load_prompt("grading_user", version).render(question=question, sources=sources)
    return system, user
