"""Phase 3 services against fakes: Redis via fakeredis, Qdrant in local mode, scripted LLM."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis, FakeServer

from app.config import Settings
from app.errors import IngestRequestError
from app.llm import LLMError
from app.models import Citation, IngestRequest, RetrievalFilter, ScoredChunk
from app.retrieval.grader import RetrievalGrader
from app.services.conversation import ConversationStore, QueryRewriter, Turn
from app.services.index_version import IndexVersionProvider, settings_fingerprint
from app.services.ingest_jobs import IngestJobManager
from app.services.library import LibraryCatalog
from app.services.query_logger import JsonlQueryLog, QueryLogRecord, hash_session_id
from app.services.semantic_cache import CachedAnswer, SemanticCache
from conftest import CHUNK_IDS, FakeLLM, library_chunk, make_settings, seeded_qdrant

_E1 = [1.0, 0.0, 0.0, 0.0]
_E2 = [0.0, 1.0, 0.0, 0.0]


def _cached(text: str = "KL measures divergence [S1].", age_s: float = 0.0) -> CachedAnswer:
    citation = Citation(
        book="PRML",
        chapter="1",
        section="1.1",
        page_start=10,
        page_end=10,
        chunk_id="c1",
        score=0.9,
    )
    return CachedAnswer(
        text=text, citations=[citation], question="q", created_at=time.time() - age_s
    )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class TestConversationStore:
    @pytest.mark.asyncio
    async def test_keeps_newest_turns_in_order_with_ttl(
        self, settings: Settings, redis: FakeAsyncRedis
    ) -> None:
        store = ConversationStore(redis, settings)  # conversation_max_turns = 6
        for i in range(8):
            await store.append("s1", Turn(question=f"q{i}", answer=f"a{i}"), "r")
        history = await store.history("s1", "r")
        assert [t.question for t in history] == [f"q{i}" for i in range(2, 8)]
        assert 0 < await redis.ttl("rag:session:s1") <= settings.redis_ttl_seconds

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self, settings: Settings, redis: FakeAsyncRedis) -> None:
        store = ConversationStore(redis, settings)
        await store.append("s1", Turn(question="q", answer="a"), "r")
        assert await store.history("s2", "r") == []

    @pytest.mark.asyncio
    async def test_redis_down_degrades_to_no_history(
        self, settings: Settings, redis: FakeAsyncRedis, redis_server: FakeServer
    ) -> None:
        store = ConversationStore(redis, settings)
        redis_server.connected = False
        await store.append("s1", Turn(question="q", answer="a"), "r")  # must not raise
        assert await store.history("s1", "r") == []


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_first_question_is_not_rewritten(self, settings: Settings, llm: FakeLLM) -> None:
        assert await QueryRewriter(llm, settings).rewrite("What is EM?", [], "r") == "What is EM?"
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_follow_up_uses_the_llm_rewrite(self, settings: Settings, llm: FakeLLM) -> None:
        llm.rewrites = ['"How does Murphy define the EM algorithm?"']
        history = [Turn(question="How does Bishop define EM?", answer="... [S1]")]
        rewritten = await QueryRewriter(llm, settings).rewrite("What about Murphy?", history, "r")
        assert rewritten == "How does Murphy define the EM algorithm?"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_the_question(self, settings: Settings) -> None:
        class BrokenLLM(FakeLLM):
            async def generate(self, **kwargs: Any) -> Any:
                raise LLMError("down")

        history = [Turn(question="q", answer="a")]
        rewriter = QueryRewriter(BrokenLLM(), settings)
        assert await rewriter.rewrite("What about Murphy?", history, "r") == "What about Murphy?"


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_similar_query_hits(self, settings: Settings, redis: FakeAsyncRedis) -> None:
        cache = SemanticCache(redis, settings)
        await cache.store(_E1, _cached(), "v1", None, "r")
        lookup = await cache.lookup([0.99, 0.1, 0.0, 0.0], "v1", None, "r")  # cosine ≈ 0.995
        assert lookup.answer is not None and lookup.answer.text.startswith("KL")
        assert lookup.similarity is not None and lookup.similarity > 0.98

    @pytest.mark.asyncio
    async def test_dissimilar_query_misses(self, settings: Settings, redis: FakeAsyncRedis) -> None:
        cache = SemanticCache(redis, settings)
        await cache.store(_E1, _cached(), "v1", None, "r")
        lookup = await cache.lookup(_E2, "v1", None, "r")
        assert lookup.answer is None and lookup.similarity == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_other_filters_or_index_version_never_hit(
        self, settings: Settings, redis: FakeAsyncRedis
    ) -> None:
        cache = SemanticCache(redis, settings)
        await cache.store(_E1, _cached(), "v1", None, "r")
        scoped = RetrievalFilter(books=["PRML"])
        assert (await cache.lookup(_E1, "v1", scoped, "r")).answer is None
        assert (await cache.lookup(_E1, "v2", None, "r")).answer is None

    @pytest.mark.asyncio
    async def test_expired_entry_misses(self, settings: Settings, redis: FakeAsyncRedis) -> None:
        cache = SemanticCache(redis, settings)
        await cache.store(
            _E1, _cached(age_s=settings.semantic_cache_ttl_seconds + 1), "v1", None, "r"
        )
        assert (await cache.lookup(_E1, "v1", None, "r")).answer is None

    @pytest.mark.asyncio
    async def test_evicts_oldest_beyond_cap(self, tmp_path: Path, redis: FakeAsyncRedis) -> None:
        cache = SemanticCache(redis, make_settings(tmp_path, semantic_cache_max_entries=2))
        for vector in (_E1, _E2, [0.0, 0.0, 1.0, 0.0]):
            await cache.store(vector, _cached(), "v1", None, "r")
        assert (await cache.lookup(_E1, "v1", None, "r")).answer is None
        assert (await cache.lookup(_E2, "v1", None, "r")).answer is not None
        answers_keys = [k async for k in redis.scan_iter(match="rag:cache:*:answers")]
        assert await redis.hlen(answers_keys[0]) == 2  # evicted answer deleted too

    @pytest.mark.asyncio
    async def test_redis_down_is_a_miss(
        self, settings: Settings, redis: FakeAsyncRedis, redis_server: FakeServer
    ) -> None:
        cache = SemanticCache(redis, settings)
        redis_server.connected = False
        await cache.store(_E1, _cached(), "v1", None, "r")  # must not raise
        lookup = await cache.lookup(_E1, "v1", None, "r")
        assert lookup.answer is None and lookup.similarity is None

    @pytest.mark.asyncio
    async def test_clear_removes_every_namespace(
        self, settings: Settings, redis: FakeAsyncRedis
    ) -> None:
        cache = SemanticCache(redis, settings)
        await cache.store(_E1, _cached(), "v1", None, "r")
        await cache.store(_E1, _cached(), "v2", RetrievalFilter(books=["ESL"]), "r")
        assert await cache.clear("r") == 4  # vectors + answers per namespace
        assert (await cache.lookup(_E1, "v1", None, "r")).answer is None


# ---------------------------------------------------------------------------
# Query log
# ---------------------------------------------------------------------------


def _record(**overrides: Any) -> QueryLogRecord:
    values: dict[str, Any] = {
        "request_id": "r1",
        "question": "What is EM?",
        "prompt_version": "v1",
        "model": "fake-model",
    }
    return QueryLogRecord(**{**values, **overrides})


class TestQueryLog:
    def test_appends_one_json_line_per_record(self, settings: Settings) -> None:
        log = JsonlQueryLog(settings)
        log.write(_record(request_id="r1", status="answered"))
        log.write(_record(request_id="r2", status="error", error="LLMError: down"))
        log.close()
        lines = settings.query_log_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["request_id"] for line in lines] == ["r1", "r2"]
        assert json.loads(lines[1])["error"] == "LLMError: down"

    def test_rotates_by_size(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path, query_log_max_bytes=500, query_log_backup_count=2)
        log = JsonlQueryLog(settings)
        for i in range(10):
            log.write(_record(request_id=f"r{i}"))
        log.close()
        assert settings.query_log_path.with_name("query_log.jsonl.1").exists()

    def test_session_ids_are_hashed(self) -> None:
        hashed = hash_session_id("my-session")
        assert hashed == hash_session_id("my-session") and hashed != "my-session"
        assert hash_session_id(None) is None


# ---------------------------------------------------------------------------
# Retrieval grader
# ---------------------------------------------------------------------------


def _chunks(score: float) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=library_chunk("kl"), score=score)]


class TestRetrievalGrader:
    @pytest.mark.asyncio
    async def test_low_top_score_grades_none_without_llm(
        self, settings: Settings, llm: FakeLLM
    ) -> None:
        result = await RetrievalGrader(llm, settings).grade("q", _chunks(0.01), "r")
        assert result.grade == "none" and not result.llm_called
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_no_candidates_grade_none(self, settings: Settings, llm: FakeLLM) -> None:
        result = await RetrievalGrader(llm, settings).grade("q", [], "r")
        assert result.grade == "none" and result.top_score is None

    @pytest.mark.asyncio
    async def test_llm_verdict_is_used(self, settings: Settings, llm: FakeLLM) -> None:
        result = await RetrievalGrader(llm, settings).grade("What is KL?", _chunks(0.9), "r")
        assert result.grade == "correct" and result.llm_called
        assert result.chunk_ids == [CHUNK_IDS["kl"]]

    @pytest.mark.asyncio
    async def test_unparseable_verdict_degrades_to_ambiguous(
        self, settings: Settings, llm: FakeLLM
    ) -> None:
        llm.grades = ["I think these are fine."]
        result = await RetrievalGrader(llm, settings).grade("What is KL?", _chunks(0.9), "r")
        assert result.grade == "ambiguous"


# ---------------------------------------------------------------------------
# Library catalog and index version (Qdrant local mode)
# ---------------------------------------------------------------------------


class TestLibraryCatalog:
    @pytest.mark.asyncio
    async def test_lists_books_with_chapters(self, settings: Settings) -> None:
        catalog = LibraryCatalog(await seeded_qdrant(settings), settings)
        books = await catalog.books()
        assert [b.title for b in books] == ["ESL", "PRML"]
        prml = books[1]
        assert prml.chunk_count == 3
        assert prml.chapters == ["1 Introduction", "5 Neural Networks", "7 Sparse Kernel Machines"]

    @pytest.mark.asyncio
    async def test_fetches_a_chunk_or_none(self, settings: Settings) -> None:
        catalog = LibraryCatalog(await seeded_qdrant(settings), settings)
        found = await catalog.chunk(uuid.UUID(CHUNK_IDS["em"]))
        assert found is not None and found.metadata.book == "ESL"
        assert await catalog.chunk(uuid.uuid4()) is None

    @pytest.mark.asyncio
    async def test_has_source(self, settings: Settings) -> None:
        catalog = LibraryCatalog(await seeded_qdrant(settings), settings)
        assert await catalog.has_source("/books/PRML.pdf")
        assert not await catalog.has_source("/books/missing.pdf")


class TestIndexVersion:
    @pytest.mark.asyncio
    async def test_changes_when_points_change(self, settings: Settings) -> None:
        client = await seeded_qdrant(settings)
        provider = IndexVersionProvider(client, settings)
        before = await provider.current()
        await client.delete(settings.qdrant_collection, points_selector=[CHUNK_IDS["kl"]])
        assert await provider.current() != before

    def test_fingerprint_tracks_prompt_and_model(self, tmp_path: Path) -> None:
        base = settings_fingerprint(make_settings(tmp_path))
        assert settings_fingerprint(make_settings(tmp_path, prompt_version="v2")) != base
        assert settings_fingerprint(make_settings(tmp_path, llm_generation_model="other")) != base
        assert settings_fingerprint(make_settings(tmp_path, log_level="DEBUG")) == base


# ---------------------------------------------------------------------------
# Ingest jobs
# ---------------------------------------------------------------------------


async def _never_ingested(source_path: str) -> bool:
    return False


async def _noop(request_id: str) -> None:
    return None


class TestIngestJobPaths:
    def _manager(self, settings: Settings) -> IngestJobManager:
        settings.ingest_books_dir.mkdir(parents=True, exist_ok=True)
        (settings.ingest_books_dir / "book.pdf").write_bytes(b"%PDF-1.4")
        (settings.ingest_books_dir / "notes.txt").write_text("x")
        return IngestJobManager(lambda p, t: {}, settings, _never_ingested, _noop)

    def test_resolves_relative_names_inside_the_books_folder(self, settings: Settings) -> None:
        path = self._manager(settings).resolve_source("book.pdf")
        assert path == (settings.ingest_books_dir / "book.pdf").resolve()

    @pytest.mark.parametrize(
        ("source", "message"),
        [("../outside.pdf", "inside"), ("notes.txt", "PDF"), ("missing.pdf", "not found")],
    )
    def test_rejects_paths_it_must_not_read(
        self, settings: Settings, source: str, message: str
    ) -> None:
        with pytest.raises(IngestRequestError, match=message):
            self._manager(settings).resolve_source(source)

    @pytest.mark.asyncio
    async def test_refuses_an_indexed_book_without_force(self, settings: Settings) -> None:
        async def _indexed(source_path: str) -> bool:
            return True

        manager = self._manager(settings)
        manager._is_ingested = _indexed
        with pytest.raises(IngestRequestError, match="force_reingest"):
            await manager.submit(IngestRequest(source_path="book.pdf", book_title="B"), "r")
