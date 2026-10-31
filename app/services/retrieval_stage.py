"""The retrieve → rerank → grade stage of rag_pipeline, with the ambiguous-grade retry.

An ambiguous grade on a filtered query triggers one retry across all books.
The wider set is adopted only if it grades correct; the answer is then marked
low confidence because it comes from outside the requested scope.
"""

import time
from dataclasses import dataclass

from app.config import get_logger
from app.models import RetrievalFilter, ScoredChunk
from app.retrieval.grader import GradeResult, RetrievalGrader
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.services.query_logger import QueryLogRecord, RetrievedRef

logger = get_logger(__name__)


class StageTimer:
    """Context manager factory recording per-stage latency (ms) into a dict."""

    def __init__(self, sink: dict[str, float]) -> None:
        self._sink = sink
        self._stage = ""
        self._start = 0.0

    def __call__(self, stage: str) -> "StageTimer":
        self._stage = stage
        return self

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self._sink[self._stage] = round((time.perf_counter() - self._start) * 1000, 1)


@dataclass(frozen=True)
class RetrievalOutcome:
    """Chunks to answer from, their grade, and whether filters were dropped to get them."""

    chunks: list[ScoredChunk]
    grade: GradeResult
    widened: bool


def _refs(chunks: list[ScoredChunk]) -> list[RetrievedRef]:
    return [RetrievedRef(chunk_id=c.chunk.chunk_id, score=c.score) for c in chunks]


class RetrievalStage:
    """Find, rerank, and grade the chunks for one question."""

    def __init__(
        self, retriever: HybridRetriever, reranker: Reranker, grader: RetrievalGrader
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._grader = grader

    async def run(
        self,
        question: str,
        vector: list[float],
        filters: RetrievalFilter | None,
        top_k: int | None,
        record: QueryLogRecord,
    ) -> RetrievalOutcome:
        """Return graded chunks, recording ids, scores, grade, and timings in *record*.

        Raises:
            RetrievalError, LLMError: if retrieval, reranking, or grading fails.
        """
        chunks, grade = await self._attempt(question, vector, filters, top_k, record, "")
        outcome = RetrievalOutcome(chunks, grade, widened=False)
        if grade.grade == "ambiguous" and filters is not None:
            outcome = await self._retry_unfiltered(question, vector, top_k, record, outcome)
        record.grade, record.grade_reason = outcome.grade.grade, outcome.grade.reason
        return outcome

    async def _attempt(
        self,
        question: str,
        vector: list[float],
        filters: RetrievalFilter | None,
        top_k: int | None,
        record: QueryLogRecord,
        suffix: str,
    ) -> tuple[list[ScoredChunk], GradeResult]:
        timer = StageTimer(record.latency_ms)
        rid = record.request_id
        with timer(f"retrieve{suffix}"):
            candidates = await self._retriever.retrieve(
                question, rid, filters=filters, query_vector=vector
            )
        with timer(f"rerank{suffix}"):
            reranked = await self._reranker.rerank(question, candidates, rid, top_k=top_k)
        with timer(f"grade{suffix}"):
            grade = await self._grader.grade(question, reranked, rid)
        record.retrieved, record.reranked = _refs(candidates), _refs(reranked)
        return reranked, grade

    async def _retry_unfiltered(
        self,
        question: str,
        vector: list[float],
        top_k: int | None,
        record: QueryLogRecord,
        original: RetrievalOutcome,
    ) -> RetrievalOutcome:
        record.retried_without_filters = True
        original_refs = (record.retrieved, record.reranked)
        chunks, grade = await self._attempt(question, vector, None, top_k, record, "_unfiltered")
        adopted = grade.grade == "correct"
        logger.info(
            "Unfiltered retry: grade=%s adopted=%s",
            grade.grade,
            adopted,
            extra={"request_id": record.request_id},
        )
        if adopted:
            return RetrievalOutcome(chunks, grade, widened=True)
        record.retrieved, record.reranked = original_refs
        return original
