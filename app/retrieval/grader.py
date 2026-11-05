"""Grade a reranked chunk set as correct / ambiguous / none for the question.

A score floor catches clear misses without an LLM call; otherwise the LLM
judges the set with the versioned grading prompt.
"""

import time

from pydantic import BaseModel

from app.config import Settings, get_logger
from app.llm import LLMClient
from app.models import ScoredChunk
from app.prompts.grading import Grade, build_grading_prompt, parse_grade_reply
from app.services.citation import format_sources

logger = get_logger(__name__)


class GradeResult(BaseModel):
    """The grade plus what it was based on, for routing and the query log."""

    grade: Grade
    reason: str
    top_score: float | None
    chunk_ids: list[str]
    llm_called: bool


class RetrievalGrader:
    """Decide whether reranked chunks can answer the question."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._min_score = settings.grade_min_rerank_score
        self._max_tokens = settings.llm_grader_max_tokens
        self._prompt_version = settings.prompt_version

    async def grade(self, question: str, chunks: list[ScoredChunk], request_id: str) -> GradeResult:
        """Grade *chunks* (reranked, best first) against *question*.

        An unparseable LLM verdict degrades to "ambiguous", which still answers
        but flags low confidence.

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
            system, prompt = build_grading_prompt(
                question, format_sources(chunks), self._prompt_version
            )
            reply = await self._llm.generate(
                system=system, prompt=prompt, max_tokens=self._max_tokens, request_id=request_id
            )
            verdict = parse_grade_reply(reply.text)
            if verdict is None:
                logger.warning(
                    "Grader reply was not valid JSON; treating as ambiguous.",
                    extra={"request_id": request_id},
                )
                grade: Grade = "ambiguous"
                reason = "unparseable grader reply"
            else:
                grade, reason = verdict.grade, verdict.reason
            result = GradeResult(
                grade=grade,
                reason=reason,
                top_score=top_score,
                chunk_ids=chunk_ids,
                llm_called=True,
            )

        logger.info(
            "Grade: grade=%s top_score=%s llm_called=%s latency_ms=%.0f",
            result.grade,
            "-" if top_score is None else f"{top_score:.4f}",
            result.llm_called,
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )
        return result
