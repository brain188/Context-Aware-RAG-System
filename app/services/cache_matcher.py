"""Decide whether a cached answer fits a new question that embeds close to its question.

Embedding similarity alone cannot tell a paraphrase from a near-identical
different question ("field-level" vs "form-level validation" embed at 0.97),
so the semantic cache treats a nearest neighbour as a *candidate* and asks
the fast model whether one answer serves both. Near-exact repeats skip the
call. Measured on evaluation/cache_pairs.jsonl by `evaluation.cache_threshold`.

Any failure (no template in this prompt version, an LLM error, an unusable
reply) is a "different" verdict: a miss only costs one pipeline run, a false
hit serves a wrong answer with valid-looking citations.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import Settings, get_logger
from app.llm import LLMClient, LLMError
from app.prompts import PromptNotFoundError, load_prompt

logger = get_logger(__name__)

Verdict = Literal["same", "different"]
# How a lookup ended, for the query log: exact (no LLM call), verified, rejected, or failed.
MatchKind = Literal["exact", "verified", "rejected", "failed"]


class CacheMatchReply(BaseModel):
    """The fast model's structured verdict."""

    verdict: Verdict
    reason: str = Field(default="", description="One short sentence.")


class CacheMatcher:
    """Confirm a cache candidate with the fast model unless it is a near-exact repeat."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._version = settings.prompt_version
        self._exact = settings.semantic_cache_exact_threshold
        self._max_tokens = settings.llm_grader_max_tokens

    async def match(
        self, question: str, cached_question: str, similarity: float, request_id: str
    ) -> MatchKind:
        """Return how *question* relates to the cached one; only exact/verified are hits."""
        if similarity >= self._exact or question.casefold() == cached_question.casefold():
            return "exact"
        try:
            system = load_prompt("cache_match_system", self._version).render()
            prompt = load_prompt("cache_match_user", self._version).render(
                question=question, cached_question=cached_question
            )
        except PromptNotFoundError:
            logger.info(
                "Prompt version %s has no cache_match template; only near-exact repeats hit.",
                self._version,
                extra={"request_id": request_id},
            )
            return "failed"
        try:
            reply = await self._llm.generate_structured(
                system=system,
                prompt=prompt,
                schema=CacheMatchReply,
                max_tokens=self._max_tokens,
                request_id=request_id,
            )
        except LLMError as exc:
            logger.warning(
                "Cache match check failed (%s); treating it as a miss.",
                exc,
                extra={"request_id": request_id},
            )
            return "failed"
        logger.info(
            "Cache match: verdict=%s similarity=%.4f reason=%s",
            reply.verdict,
            similarity,
            reply.reason,
            extra={"request_id": request_id},
        )
        return "verified" if reply.verdict == "same" else "rejected"
