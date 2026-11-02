"""The evaluation question set: schema, loading, and selection.

One JSON object per line in eval_questions.jsonl. Expected sources name PDF
page numbers (the viewer's page counter, as stored in chunk metadata), not
the page labels printed in the book.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.models import QueryRequest

DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "eval_questions.jsonl"


class DatasetError(ValueError):
    """The question file is malformed or a selection matches nothing."""


class ExpectedSource(BaseModel):
    """Where an answer lives: a book, optionally its chapter and section, and pages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    book: str = Field(min_length=1)
    chapter: str | None = None
    section: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _pages_ordered(self) -> "ExpectedSource":
        if self.page_end < self.page_start:
            raise ValueError(f"page_end {self.page_end} < page_start {self.page_start}")
        return self


class EvalQuestion(BaseModel):
    """One question, what a correct answer must draw on, and how to tag it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1, max_length=2000)
    answerable: bool = True  # False: the library doesn't cover it; expect "not covered"
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    # Phrases a good answer contains; "a|b" accepts either. Case-insensitive.
    must_mention: list[str] = Field(default_factory=list)
    book_filter: list[str] | None = None
    chapter_filter: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _unanswerable_has_no_answer_key(self) -> "EvalQuestion":
        if not self.answerable and (self.expected_sources or self.must_mention):
            raise ValueError("an unanswerable question cannot have expected sources or phrases")
        return self

    def to_request(self) -> QueryRequest:
        """The API request this question is asked as (no session: no history)."""
        return QueryRequest(
            question=self.question,
            book_filter=self.book_filter,
            chapter_filter=self.chapter_filter,
        )


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[EvalQuestion]:
    """Parse and validate every question in *path*.

    Raises:
        DatasetError: on unreadable JSON, a schema violation, a duplicate id, or no questions.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetError(f"{path}:{number}: {exc}") from exc
        if question.id in seen:
            raise DatasetError(f"{path}:{number}: duplicate id '{question.id}'")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise DatasetError(f"{path} has no questions")
    return questions


def select(
    questions: list[EvalQuestion], ids: list[str] | None = None, tags: list[str] | None = None
) -> list[EvalQuestion]:
    """Keep the questions with any of *ids* and any of *tags* (None = no constraint).

    Raises:
        DatasetError: if an id is unknown or nothing is left.
    """
    if ids:
        unknown = set(ids) - {q.id for q in questions}
        if unknown:
            raise DatasetError(f"Unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in ids]
    if tags:
        questions = [q for q in questions if set(q.tags) & set(tags)]
    if not questions:
        raise DatasetError("No questions match the selection")
    return questions
