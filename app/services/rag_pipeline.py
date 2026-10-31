"""The query orchestrator: coordinates services, owns the request lifecycle.

input guard → follow-up rewrite → embed → semantic cache → route → retrieve,
rerank, grade (retrieval_stage) → generate through the grounding gate →
citations → cache store → session save.

Every query writes exactly one QueryLogRecord, including cache hits, errors,
and client disconnects.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from app.config import Settings, get_logger
from app.guardrails.input_guard import screen_question
from app.guardrails.output_guard import GroundingGate
from app.models import Answer, AnswerStatus, Citation, QueryRequest, RetrievalFilter
from app.prompts import NOT_COVERED_MESSAGE
from app.retrieval.filters import filter_from_request, is_unfiltered
from app.retrieval.query_embedder import QueryEmbedder
from app.services.citation import finalize_citations
from app.services.conversation import ConversationStore, QueryRewriter, Turn
from app.services.generator import AnswerGenerator
from app.services.index_version import IndexVersionProvider
from app.services.library import LibraryCatalog
from app.services.query_logger import QueryLogRecord, QueryLogSink, hash_session_id
from app.services.query_router import route_query
from app.services.retrieval_stage import RetrievalStage, StageTimer
from app.services.semantic_cache import CachedAnswer, SemanticCache

logger = get_logger(__name__)


class TokenEvent(BaseModel):
    """A piece of answer text, safe to show immediately."""

    kind: Literal["token"] = "token"
    text: str


class AnswerEvent(BaseModel):
    """The final, authoritative answer; always the last event."""

    kind: Literal["answer"] = "answer"
    answer: Answer


PipelineEvent = TokenEvent | AnswerEvent


@dataclass(frozen=True)
class PreparedQuery:
    """A request that passed the input guard, ready to run."""

    request_id: str
    session_id: str
    question: str
    filters: RetrievalFilter | None
    top_k: int | None


class RagPipeline:
    """Answer one question from the library, streaming grounded tokens."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: QueryEmbedder,
        retrieval: RetrievalStage,
        generator: AnswerGenerator,
        rewriter: QueryRewriter,
        conversations: ConversationStore,
        cache: SemanticCache,
        index_versions: IndexVersionProvider,
        catalog: LibraryCatalog,
        query_log: QueryLogSink,
        llm_model: str,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._retrieval = retrieval
        self._generator = generator
        self._rewriter = rewriter
        self._conversations = conversations
        self._cache = cache
        self._index_versions = index_versions
        self._catalog = catalog
        self._query_log = query_log
        self._llm_model = llm_model

    def prepare(self, request: QueryRequest, request_id: str) -> PreparedQuery:
        """Screen and normalize the request before anything slow runs.

        Raises:
            InputRejectedError: if the input guard refuses the question.
        """
        filters = filter_from_request(request)
        return PreparedQuery(
            request_id=request_id,
            session_id=request.session_id or uuid.uuid4().hex,
            question=screen_question(request.question, self._settings),
            filters=None if is_unfiltered(filters) else filters,
            top_k=request.top_k,
        )

    async def run(self, query: PreparedQuery) -> AsyncIterator[PipelineEvent]:
        """Yield TokenEvents, then one AnswerEvent.

        Raises:
            RetrievalError, LLMError: when a stage fails; logged in the query record first.
        """
        record = QueryLogRecord(
            request_id=query.request_id,
            session_hash=hash_session_id(query.session_id),
            question=query.question,
            filters=query.filters.model_dump(exclude_none=True) if query.filters else None,
            prompt_version=self._settings.prompt_version,
            model=self._llm_model,
        )
        start = time.perf_counter()
        try:
            async for event in self._run(query, record):
                yield event
        except Exception as exc:
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if record.status == "incomplete":
                record.status = "cancelled"  # the client went away mid-stream
            record.latency_ms["total"] = round((time.perf_counter() - start) * 1000, 1)
            self._write_log(record)

    async def _run(
        self, query: PreparedQuery, record: QueryLogRecord
    ) -> AsyncIterator[PipelineEvent]:
        rid = query.request_id
        timer = StageTimer(record.latency_ms)

        with timer("history"):
            history = await self._conversations.history(query.session_id, rid)
        with timer("rewrite"):
            question = await self._rewriter.rewrite(query.question, history, rid)
        record.standalone_question = question
        with timer("embed"):
            loop = asyncio.get_running_loop()
            vector = await loop.run_in_executor(None, self._embedder.embed_query, question)
        with timer("index_version"):
            index_version = await self._index_versions.current()
        record.index_version = index_version

        if self._settings.semantic_cache_enabled:
            with timer("cache_lookup"):
                lookup = await self._cache.lookup(vector, index_version, query.filters, rid)
            record.cache_similarity = lookup.similarity
            if lookup.answer is not None:
                record.cache_hit = True
                answer = self._answer(query, lookup.answer.text, lookup.answer.citations)
                answer.cache_hit = True
                yield TokenEvent(text=answer.text)
                yield await self._finish(query, answer, record)
                return

        route = route_query(question, await self._catalog.titles(rid))
        record.route, record.route_reason = route.kind, route.reason

        outcome = await self._retrieval.run(question, vector, query.filters, query.top_k, record)
        if outcome.grade.grade == "none":
            async for event in self._not_covered(query, record):
                yield event
            return

        gate = GroundingGate(len(outcome.chunks), NOT_COVERED_MESSAGE)
        with timer("generate"):
            async for delta in self._generator.generate(question, outcome.chunks, rid):
                if released := gate.feed(delta):
                    yield TokenEvent(text=released)
        verdict = gate.finish()
        record.grounding = verdict.status

        if verdict.status != "grounded":
            # Nothing was streamed: the gate only releases grounded text.
            if verdict.status == "ungrounded":
                logger.warning(
                    "Answer cited no provided source; replaced with 'not covered'.",
                    extra={"request_id": rid},
                )
            async for event in self._not_covered(query, record):
                yield event
            return

        if verdict.invalid_markers:
            logger.warning(
                "Removed citation markers with no matching source: %s",
                verdict.invalid_markers,
                extra={"request_id": rid},
            )
        text, citations = finalize_citations(verdict.text, outcome.chunks)
        low_confidence = outcome.grade.grade == "ambiguous" or outcome.widened
        if low_confidence:
            note = self._generator.low_confidence_note(outcome.widened)
            yield TokenEvent(text=f"\n\n{note}")
            text = f"{text}\n\n{note}"
        status: AnswerStatus = "low_confidence" if low_confidence else "answered"
        answer = self._answer(query, text, citations, status)

        # Only confident, grounded answers are reused; the library may grow.
        if status == "answered" and self._settings.semantic_cache_enabled:
            with timer("cache_store"):
                cached = CachedAnswer(
                    text=text, citations=citations, question=question, created_at=time.time()
                )
                await self._cache.store(vector, cached, index_version, query.filters, rid)
        yield await self._finish(query, answer, record)

    async def _not_covered(
        self, query: PreparedQuery, record: QueryLogRecord
    ) -> AsyncIterator[PipelineEvent]:
        answer = self._answer(query, NOT_COVERED_MESSAGE, [], "not_covered")
        yield TokenEvent(text=answer.text)
        yield await self._finish(query, answer, record)

    @staticmethod
    def _answer(
        query: PreparedQuery,
        text: str,
        citations: list[Citation],
        status: AnswerStatus = "answered",
    ) -> Answer:
        return Answer(
            text=text,
            citations=citations,
            request_id=query.request_id,
            session_id=query.session_id,
            status=status,
        )

    async def _finish(
        self, query: PreparedQuery, answer: Answer, record: QueryLogRecord
    ) -> AnswerEvent:
        record.status = answer.status
        record.citations = [c.chunk_id for c in answer.citations]
        turn = Turn(question=query.question, answer=answer.text)
        await self._conversations.append(query.session_id, turn, query.request_id)
        return AnswerEvent(answer=answer)

    def _write_log(self, record: QueryLogRecord) -> None:
        logger.info(
            "Query done: status=%s cache_hit=%s grade=%s citations=%d latency_ms=%s",
            record.status,
            record.cache_hit,
            record.grade or "-",
            len(record.citations),
            record.latency_ms,
            extra={"request_id": record.request_id},
        )
        try:
            self._query_log.write(record)
        except OSError as exc:
            logger.error(
                "Could not write query log record: %s", exc, extra={"request_id": record.request_id}
            )
