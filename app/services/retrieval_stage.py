"""The retrieve → rerank → grade stage of rag_pipeline, with the ambiguous-grade retries.

An ambiguous grade first triggers up to GRADER_MAX_RETRIES search-query
rewrites, each retrieved within the same filters. Then, if the query is
filtered and still ambiguous, one retry runs across all books. A retry's
chunks are adopted only if they grade better; the user's question stays the
one reranked, graded and answered, only the search query changes. Answers
from the unfiltered retry are marked low confidence because they come from
outside the requested scope.
"""

import time
from dataclasses import dataclass

from app.config import get_logger
from app.llm import LLMError
from app.models import RetrievalFilter, ScoredChunk
from app.retrieval.grader import GradeResult, RetrievalGrader
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_rewriter import SearchQueryRewriter
from app.retrieval.reranker import Reranker
from app.services.query_logger import QueryLogRecord, RetrievedRef, SearchRetry

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
    """Chunks to answer from, their grade, and how they were found."""

    chunks: list[ScoredChunk]
    grade: GradeResult
    widened: bool  # filters were dropped to find them
    search_query: str  # the query the chunks were retrieved with


@dataclass(frozen=True)
class _Attempt:
    query: str
    candidates: list[ScoredChunk]
    reranked: list[ScoredChunk]
    grade: GradeResult


def _refs(chunks: list[ScoredChunk]) -> list[RetrievedRef]:
    return [RetrievedRef(chunk_id=c.chunk.chunk_id, score=c.score) for c in chunks]


class RetrievalStage:
    """Find, rerank, and grade the chunks for one question."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        grader: RetrievalGrader,
        rewriter: SearchQueryRewriter | None = None,
        max_rewrites: int = 0,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._grader = grader
        self._rewriter = rewriter
        self._max_rewrites = max_rewrites if rewriter is not None else 0

    async def run(
        self,
        question: str,
        vector: list[float],
        filters: RetrievalFilter | None,
        top_k: int | None,
        record: QueryLogRecord,
    ) -> RetrievalOutcome:
        """Return graded chunks, recording ids, scores, grade, retries, and timings in *record*.

        *vector* is the embedding of *question*. A failed grading call on a
        retry keeps the chunks found so far (WARNING); on the first attempt it raises.

        Raises:
            RetrievalError, LLMError: if retrieval, reranking, or the first grading fails.
        """
        best = await self._attempt(question, question, vector, filters, top_k, record, "")
        if best.grade.grade == "ambiguous" and self._max_rewrites:
            best = await self._rewrite_and_retry(question, filters, top_k, record, best)
        widened = False
        if best.grade.grade == "ambiguous" and filters is not None:
            best, widened = await self._retry_unfiltered(question, vector, top_k, record, best)

        record.retrieved, record.reranked = _refs(best.candidates), _refs(best.reranked)
        record.grade, record.grade_reason = best.grade.grade, best.grade.reason
        record.grade_relevance = dict(best.grade.relevance)
        record.search_query = best.query if best.query != question else None
        return RetrievalOutcome(best.reranked, best.grade, widened, best.query)

    async def _attempt(
        self,
        question: str,
        query: str,
        vector: list[float] | None,
        filters: RetrievalFilter | None,
        top_k: int | None,
        record: QueryLogRecord,
        suffix: str,
    ) -> _Attempt:
        """Retrieve with *query*; rerank and grade against *question*."""
        timer = StageTimer(record.latency_ms)
        rid = record.request_id
        with timer(f"retrieve{suffix}"):
            candidates = await self._retriever.retrieve(
                query, rid, filters=filters, query_vector=vector
            )
        with timer(f"rerank{suffix}"):
            reranked = await self._reranker.rerank(question, candidates, rid, top_k=top_k)
        with timer(f"grade{suffix}"):
            grade = await self._grader.grade(question, reranked, rid)
        return _Attempt(query, candidates, reranked, grade)

    async def _rewrite_and_retry(
        self,
        question: str,
        filters: RetrievalFilter | None,
        top_k: int | None,
        record: QueryLogRecord,
        best: _Attempt,
    ) -> _Attempt:
        assert self._rewriter is not None
        rid = record.request_id
        tried, missing = [question], best.grade.reason
        for n in range(1, self._max_rewrites + 1):
            with StageTimer(record.latency_ms)(f"search_rewrite_{n}"):
                query = await self._rewriter.rewrite(question, tried, missing, rid)
            if query is None:
                break
            tried.append(query)
            try:
                # No vector: the retriever embeds the new query itself.
                attempt = await self._attempt(
                    question, query, None, filters, top_k, record, f"_retry_{n}"
                )
            except LLMError as exc:
                logger.warning(
                    "Grading the rewritten search failed (%s); keeping the excerpts found so far.",
                    exc,
                    extra={"request_id": rid},
                )
                break
            adopted = attempt.grade.better_than(best.grade)
            record.search_retries.append(
                SearchRetry(query=query, grade=attempt.grade.grade, adopted=adopted)
            )
            logger.info(
                "Search retry %d: grade=%s adopted=%s",
                n,
                attempt.grade.grade,
                adopted,
                extra={"request_id": rid},
            )
            if adopted:
                best = attempt
            if best.grade.grade != "ambiguous":
                break
            missing = attempt.grade.reason
        return best

    async def _retry_unfiltered(
        self,
        question: str,
        vector: list[float],
        top_k: int | None,
        record: QueryLogRecord,
        best: _Attempt,
    ) -> tuple[_Attempt, bool]:
        record.retried_without_filters = True
        # Search with the best query so far; reuse the question's vector when that is the question.
        query_vector = vector if best.query == question else None
        try:
            attempt = await self._attempt(
                question, best.query, query_vector, None, top_k, record, "_unfiltered"
            )
        except LLMError as exc:
            logger.warning(
                "Grading the unfiltered retry failed (%s); keeping the filtered excerpts.",
                exc,
                extra={"request_id": record.request_id},
            )
            return best, False
        adopted = attempt.grade.grade == "correct"
        logger.info(
            "Unfiltered retry: grade=%s adopted=%s",
            attempt.grade.grade,
            adopted,
            extra={"request_id": record.request_id},
        )
        return (attempt, True) if adopted else (best, False)
