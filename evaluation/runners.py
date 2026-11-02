"""Ask the system one eval question and record what it did.

PipelineRunner runs the full RAG pipeline exactly as the API does and reads the
query-log record it writes. RetrievalRunner stops after reranking: no LLM calls,
no cost, no Redis — the fast loop for tuning retrieval.
"""

import time
import uuid
from typing import Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import get_logger
from app.guardrails.input_guard import InputRejectedError
from app.llm import LLMError
from app.models import ChunkMetadata, Citation, ScoredChunk
from app.retrieval.errors import RetrievalError
from app.retrieval.filters import filter_from_request, is_unfiltered
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.services.query_logger import QueryLogRecord, QueryLogSink
from app.services.rag_pipeline import AnswerEvent, RagPipeline
from evaluation.dataset import EvalQuestion
from evaluation.metrics import Observation, SourceRef

logger = get_logger(__name__)

# The failures a question can hit at run time; anything else is a bug and should surface.
_QUESTION_ERRORS = (RetrievalError, LLMError, InputRejectedError)
_META_FIELDS = ["book", "chapter", "section", "page_start", "page_end"]


class Runner(Protocol):
    """Runs one question through (part of) the system."""

    async def observe(self, question: EvalQuestion) -> Observation: ...


def _request_id(question: EvalQuestion) -> str:
    return f"eval-{question.id}-{uuid.uuid4().hex[:8]}"


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def ref_from_metadata(chunk_id: str, meta: ChunkMetadata | Citation) -> SourceRef:
    """A SourceRef from chunk metadata or an answer citation (same field names)."""
    return SourceRef(
        chunk_id=chunk_id,
        book=meta.book,
        chapter=meta.chapter,
        section=meta.section,
        page_start=meta.page_start,
        page_end=meta.page_end,
    )


def _ref(scored: ScoredChunk) -> SourceRef:
    return ref_from_metadata(scored.chunk.chunk_id, scored.chunk.metadata)


class ChunkLookup:
    """Fetch metadata (and optionally text) for chunk ids from Qdrant."""

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def fetch(self, ids: list[str], with_text: bool = False) -> dict[str, dict]:
        """Payloads by id; ids no longer in the collection are left out.

        Raises:
            RetrievalError: if Qdrant cannot be reached.
        """
        unique = list(dict.fromkeys(ids))
        if not unique:
            return {}
        fields = _META_FIELDS + (["text"] if with_text else [])
        try:
            points = await self._client.retrieve(
                self._collection, ids=unique, with_payload=fields, with_vectors=False
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise RetrievalError(f"Could not fetch chunk metadata: {exc}") from exc
        return {str(p.id): p.payload or {} for p in points}

    async def refs(self, ids: list[str]) -> list[SourceRef]:
        """SourceRefs for *ids*, in order."""
        payloads = await self.fetch(ids)
        return [
            SourceRef(chunk_id=i, **{f: payloads[i][f] for f in _META_FIELDS})
            for i in ids
            if i in payloads
        ]


class CapturingQueryLog:
    """A QueryLogSink that keeps each record for the runner, then forwards it."""

    def __init__(self, forward: QueryLogSink | None = None) -> None:
        self._forward = forward
        self._records: dict[str, QueryLogRecord] = {}

    def write(self, record: QueryLogRecord) -> None:
        self._records[record.request_id] = record
        if self._forward is not None:
            self._forward.write(record)

    def pop(self, request_id: str) -> QueryLogRecord | None:
        """The record written for *request_id*, removed from memory."""
        return self._records.pop(request_id, None)

    def close(self) -> None:
        if self._forward is not None:
            self._forward.close()


class PipelineRunner:
    """The whole pipeline: guard, retrieve, rerank, grade, generate, ground, cite."""

    def __init__(
        self, pipeline: RagPipeline, records: CapturingQueryLog, lookup: ChunkLookup
    ) -> None:
        self._pipeline = pipeline
        self._records = records
        self._lookup = lookup

    async def observe(self, question: EvalQuestion) -> Observation:
        """Run *question*; failures are recorded on the Observation, not raised."""
        rid = _request_id(question)
        observation = Observation()
        try:
            prepared = self._pipeline.prepare(question.to_request(), rid)
            async for event in self._pipeline.run(prepared):
                if isinstance(event, AnswerEvent):
                    answer = event.answer
                    observation.status, observation.answer = answer.status, answer.text
                    observation.cited = [ref_from_metadata(c.chunk_id, c) for c in answer.citations]
        except _QUESTION_ERRORS as exc:
            observation.error = _describe(exc)
            logger.warning("Eval question %s failed: %s", question.id, observation.error)

        record = self._records.pop(rid)
        if record is not None:  # None only when the input guard refused the question
            observation.grade = record.grade
            observation.latency_ms = record.latency_ms
            try:
                observation.candidates = await self._lookup.refs(
                    [r.chunk_id for r in record.retrieved]
                )
                observation.reranked = await self._lookup.refs(
                    [r.chunk_id for r in record.reranked]
                )
            except RetrievalError as exc:
                observation.error = observation.error or _describe(exc)
        return observation


class RetrievalRunner:
    """Hybrid retrieval + reranking only, with the API's filters; no LLM, no cache."""

    def __init__(self, retriever: HybridRetriever, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    async def observe(self, question: EvalQuestion) -> Observation:
        """Retrieve and rerank *question*; failures are recorded on the Observation."""
        rid = _request_id(question)
        filters = filter_from_request(question.to_request())
        filters = None if is_unfiltered(filters) else filters
        observation = Observation()
        try:
            start = time.perf_counter()
            candidates = await self._retriever.retrieve(question.question, rid, filters=filters)
            retrieved = time.perf_counter()
            reranked = await self._reranker.rerank(question.question, candidates, rid)
            observation.latency_ms = {
                "retrieve": round((retrieved - start) * 1000, 1),
                "rerank": round((time.perf_counter() - retrieved) * 1000, 1),
            }
        except RetrievalError as exc:
            observation.error = _describe(exc)
            logger.warning("Eval question %s failed: %s", question.id, observation.error)
            return observation
        observation.candidates = [_ref(c) for c in candidates]
        observation.reranked = [_ref(c) for c in reranked]
        return observation
