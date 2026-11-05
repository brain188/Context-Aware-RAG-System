"""Grade a reranked chunk set as correct / ambiguous / none for the question.

A score floor catches clear misses without an LLM call (saves quota);
otherwise the fast model fills a structured verdict: the set's grade plus a
relevance label per excerpt.
"""

import time

from pydantic import BaseModel, Field

from app.config import Settings, get_logger
from app.llm import LLMClient, LLMOutputError
from app.models import ScoredChunk
from app.prompts.grading import Grade, GradeReply, Relevance, build_grading_prompt
from app.services.citation import format_sources

logger = get_logger(__name__)

# Better grades first; used to decide whether a retry improved on the first attempt.
GRADE_RANK: dict[Grade, int] = {"correct": 2, "ambiguous": 1, "none": 0}


class GradeResult(BaseModel):
    """The grade plus what it was based on, for routing and the query log."""

    grade: Grade
    reason: str
    top_score: float | None
    chunk_ids: list[str]
    llm_called: bool
    # chunk_id → the grader's label; only chunks the grader rated.
    relevance: dict[str, Relevance] = Field(default_factory=dict)

    def better_than(self, other: "GradeResult") -> bool:
        return GRADE_RANK[self.grade] > GRADE_RANK[other.grade]


def _relevance(reply: GradeReply, chunk_ids: list[str]) -> dict[str, Relevance]:
    """Map the reply's excerpt numbers to chunk ids, ignoring numbers out of range."""
    labels: dict[str, Relevance] = {}
    for item in reply.sources:
        if 1 <= item.source <= len(chunk_ids):
            labels.setdefault(chunk_ids[item.source - 1], item.relevance)
    return labels


class RetrievalGrader:
    """Decide whether reranked chunks can answer the question."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._min_score = settings.grade_min_rerank_score
        self._max_tokens = settings.llm_grader_max_tokens
        self._prompt_version = settings.prompt_version

    async def grade(self, question: str, chunks: list[ScoredChunk], request_id: str) -> GradeResult:
        """Grade *chunks* (reranked, best first) against *question*.

        An unusable verdict degrades to "ambiguous", which still answers but
        flags low confidence. A "correct" verdict that marks no excerpt
        relevant is downgraded to "ambiguous": the model contradicted itself.

        Raises:
            LLMError: if the grading call itself fails.
        """
        start = time.perf_counter()
        chunk_ids = [c.chunk.chunk_id for c in chunks]
        top_score = chunks[0].score if chunks else None

        if top_score is None or top_score < self._min_score:
            reason = "no candidates" if top_score is None else "top rerank score below floor"
            result = GradeResult(
                grade="none",
                reason=reason,
                top_score=top_score,
                chunk_ids=chunk_ids,
                llm_called=False,
            )
        else:
            result = await self._ask_llm(question, chunks, chunk_ids, top_score, request_id)

        logger.info(
            "Grade: grade=%s top_score=%s llm_called=%s relevant=%d/%d latency_ms=%.0f",
            result.grade,
            "-" if top_score is None else f"{top_score:.4f}",
            result.llm_called,
            sum(label == "relevant" for label in result.relevance.values()),
            len(chunk_ids),
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )
        return result

    async def _ask_llm(
        self,
        question: str,
        chunks: list[ScoredChunk],
        chunk_ids: list[str],
        top_score: float,
        request_id: str,
    ) -> GradeResult:
        system, prompt = build_grading_prompt(
            question, format_sources(chunks), self._prompt_version
        )
        try:
            reply = await self._llm.generate_structured(
                system=system,
                prompt=prompt,
                schema=GradeReply,
                max_tokens=self._max_tokens,
                request_id=request_id,
            )
        except LLMOutputError as exc:
            logger.warning(
                "Grader reply was unusable (%s); treating as ambiguous.",
                exc,
                extra={"request_id": request_id},
            )
            reply = GradeReply(grade="ambiguous", reason="unusable grader reply")

        relevance = _relevance(reply, chunk_ids)
        grade, reason = reply.grade, reply.reason
        if grade == "correct" and relevance and "relevant" not in relevance.values():
            grade, reason = "ambiguous", f"graded correct but no excerpt marked relevant: {reason}"
        return GradeResult(
            grade=grade,
            reason=reason,
            top_score=top_score,
            chunk_ids=chunk_ids,
            llm_called=True,
            relevance=relevance,
        )
