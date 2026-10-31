"""Keep ungrounded answers from leaving the system, even while streaming.

GroundingGate sits between the LLM stream and the client. It holds tokens back
until the answer cites at least one real source, then streams freely. If the
answer ends without a valid citation, nothing has been sent and the caller
replaces it with the "not covered" response. The model's own refusal sentence
is recognized and passed through as "not covered".
"""

from dataclasses import dataclass
from typing import Literal

from app.services.citation import cited_indices, has_valid_marker

GroundingStatus = Literal["grounded", "not_covered", "ungrounded"]


@dataclass(frozen=True)
class GroundingVerdict:
    """Outcome of checking a finished answer."""

    status: GroundingStatus
    text: str  # the raw answer; meaningful only when status == "grounded"
    invalid_markers: list[int]  # cited indices with no matching source


class GroundingGate:
    """Buffer streamed tokens until the answer is grounded in a provided source."""

    def __init__(self, source_count: int, not_covered_message: str) -> None:
        self._source_count = source_count
        self._not_covered = not_covered_message
        self._parts: list[str] = []
        self._released = False

    @property
    def released(self) -> bool:
        """True once any text has been let through to the client."""
        return self._released

    def feed(self, token: str) -> str:
        """Add *token*; return the text that may be sent now ("" while holding back)."""
        self._parts.append(token)
        if self._released:
            return token
        text = "".join(self._parts)
        # Check the whole buffer: a marker can be split across tokens ("[S", "1]").
        if has_valid_marker(text, self._source_count):
            self._released = True
            return text
        return ""

    def finish(self) -> GroundingVerdict:
        """Judge the complete answer."""
        text = "".join(self._parts).strip()
        valid, invalid = cited_indices(text, self._source_count)
        if valid:
            return GroundingVerdict("grounded", text, invalid)
        if self._not_covered.rstrip(".").lower() in text.lower():
            return GroundingVerdict("not_covered", text, invalid)
        return GroundingVerdict("ungrounded", text, invalid)
