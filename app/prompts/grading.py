"""Retrieval-grading prompt: build it, and parse the model's verdict.

The grade drives rag_pipeline: correct → generate, ambiguous → retry wider,
none → "not covered". The prompt text itself is versioned in templates/.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.prompts import load_prompt

Grade = Literal["correct", "ambiguous", "none"]

# The reply should be bare JSON, but tolerate prose or code fences around it.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class GradeReply(BaseModel):
    """The grader model's structured verdict."""

    grade: Grade
    reason: str = ""


def build_grading_prompt(question: str, sources: str, version: str) -> tuple[str, str]:
    """Return (system, user) prompts for grading *sources* against *question*."""
    system = load_prompt("grading_system", version).render()
    user = load_prompt("grading_user", version).render(question=question, sources=sources)
    return system, user


def parse_grade_reply(text: str) -> GradeReply | None:
    """Extract the verdict from the model's reply, or None if it is not valid JSON of that shape."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        return None
    try:
        return GradeReply.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None
