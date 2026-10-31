"""Generate a cited answer from reranked chunks with the versioned answer prompt."""

from collections.abc import AsyncIterator

from app.config import Settings
from app.llm import LLMClient
from app.models import ScoredChunk
from app.prompts import NOT_COVERED_MESSAGE, load_prompt
from app.services.citation import format_sources


class AnswerGenerator:
    """Stream an answer that cites the provided chunks as [S1], [S2], …"""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._max_tokens = settings.llm_max_tokens
        self._version = settings.prompt_version

    def build_prompt(self, question: str, chunks: list[ScoredChunk]) -> tuple[str, str]:
        """Return (system, user) prompts; source i is chunks[i - 1]."""
        system = load_prompt("answer_system", self._version).render(not_covered=NOT_COVERED_MESSAGE)
        user = load_prompt("answer_user", self._version).render(
            sources=format_sources(chunks), question=question
        )
        return system, user

    def generate(
        self, question: str, chunks: list[ScoredChunk], request_id: str
    ) -> AsyncIterator[str]:
        """Yield raw answer text deltas (ungated; see output_guard.GroundingGate).

        Raises:
            LLMError: if the LLM call fails.
        """
        system, prompt = self.build_prompt(question, chunks)
        return self._llm.stream(
            system=system, prompt=prompt, max_tokens=self._max_tokens, request_id=request_id
        )

    def low_confidence_note(self, widened: bool) -> str:
        """Note appended to answers built from weakly matching passages."""
        scope_note = " (searched all books, beyond your filters)" if widened else ""
        return load_prompt("low_confidence_note", self._version).render(scope_note=scope_note)
