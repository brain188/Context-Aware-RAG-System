"""Shared fakes and fixtures for the generation, service, pipeline, and API tests.

Everything external is replaced: Qdrant runs in local in-memory mode, Redis is
fakeredis, and the embedder, cross-encoder, and LLM are deterministic fakes.
No network, GPU, model download, or API key is needed.
"""

import re
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis, FakeServer
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.dependencies import Services
from app.llm import LLMError
from app.models import Chunk, ChunkMetadata
from app.retrieval.grader import RetrievalGrader
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.sparse_encoder import BM25SparseEncoder
from app.services.conversation import ConversationStore, QueryRewriter
from app.services.generator import AnswerGenerator
from app.services.index_version import IndexVersionProvider, embedder_stamp
from app.services.ingest_jobs import IngestJobManager
from app.services.library import LibraryCatalog
from app.services.query_logger import JsonlQueryLog, QueryLogSink
from app.services.rag_pipeline import RagPipeline
from app.services.retrieval_stage import RetrievalStage
from app.services.semantic_cache import SemanticCache
from pipeline.vector_store import build_points, collection_schema

# name -> (text, book, chapter, page)
LIBRARY: dict[str, tuple[str, str, str, int]] = {
    "kl": (
        "The KL divergence measures how one probability distribution differs from another.",
        "PRML",
        "1 Introduction",
        10,
    ),
    "svm": (
        "Support vector machines maximise the margin between classes.",
        "PRML",
        "7 Sparse Kernel Machines",
        20,
    ),
    "nn": (
        "Neural networks are trained with gradient descent and backpropagation.",
        "PRML",
        "5 Neural Networks",
        30,
    ),
    "em": (
        "The EM algorithm alternates expectation and maximisation steps.",
        "ESL",
        "8 Model Inference",
        40,
    ),
}
CHUNK_IDS = {name: str(uuid.uuid5(uuid.NAMESPACE_URL, name)) for name in LIBRARY}

# Topic keyword -> one-hot direction, so paraphrases embed identically.
_TOPICS = {"kl": 0, "support vector": 1, "svm": 1, "neural": 2, "em algorithm": 3}
_CHUNK_TOPIC = {"kl": "kl", "svm": "svm", "nn": "neural", "em": "em algorithm"}
_STOPWORDS = {"what", "does", "with", "that", "this", "from", "about", "explain", "the", "how"}


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings isolated from the developer's .env, sized for the 4-d fake library."""
    values: dict[str, Any] = {
        "llm_api_key": "sk-test",
        "embedding_dimension": 4,
        "retrieval_top_k": 4,
        "rerank_top_k": 2,
        "query_log_path": tmp_path / "logs" / "query_log.jsonl",
        "ingest_books_dir": tmp_path / "books",
        "warm_models_on_startup": False,
        "health_timeout_seconds": 2.0,
    }
    return Settings(_env_file=None, **{**values, **overrides})


class FakeEmbedder:
    """Embeds by topic keyword: same topic → same unit vector (cosine 1.0)."""

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        for keyword, axis in _TOPICS.items():
            if keyword in lowered:
                return [1.0 if i == axis else 0.0 for i in range(4)]
        return [0.5, 0.5, 0.5, 0.5]


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 3 and t not in _STOPWORDS}


class FakeScorer:
    """0.95 when query and passage share a content word, else 0.01 (below the grade floor)."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.95 if _terms(q) & _terms(p) else 0.01 for q, p in pairs]


class FakeLLM:
    """Scripted LLM. Grading/rewrite replies pop from queues (the last one repeats)."""

    model = "fake-model"

    def __init__(self) -> None:
        self.answer = "The KL divergence measures how distributions differ [S1]."
        self.grades = ['{"grade": "correct", "reason": "direct match"}']
        self.rewrites = ["standalone question"]
        self.fail_stream = False
        self.calls: list[str] = []  # "grade" | "rewrite" | "answer"

    @staticmethod
    def _pop(queue: list[str]) -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def complete(self, *, system: str, prompt: str, max_tokens: int, request_id: str) -> str:
        if "You judge" in system:
            self.calls.append("grade")
            return self._pop(self.grades)
        self.calls.append("rewrite")
        return self._pop(self.rewrites)

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        self.calls.append("answer")
        if self.fail_stream:
            raise LLMError("provider down")
        # Small pieces, so markers arrive split across tokens like a real stream.
        for i in range(0, len(self.answer), 4):
            yield self.answer[i : i + 4]


def library_chunk(name: str) -> Chunk:
    text, book, chapter, page = LIBRARY[name]
    return Chunk(
        chunk_id=CHUNK_IDS[name],
        text=text,
        embedding=FakeEmbedder().embed_query(_CHUNK_TOPIC[name]),
        metadata=ChunkMetadata(
            book=book,
            chapter=chapter,
            section=f"{chapter} / overview",
            page_start=page,
            page_end=page,
            source_path=f"/books/{book}.pdf",
            content_hash=f"hash-{name}",
        ),
    )


async def seeded_qdrant(settings: Settings) -> AsyncQdrantClient:
    """Local Qdrant with the pipeline's schema and the LIBRARY chunks."""
    client = AsyncQdrantClient(location=":memory:")
    dense, sparse = collection_schema(settings)
    await client.create_collection(
        settings.qdrant_collection,
        vectors_config=dense,
        sparse_vectors_config=sparse,
        metadata=embedder_stamp(settings),
    )
    chunks = [library_chunk(n) for n in LIBRARY]
    await client.upsert(
        settings.qdrant_collection,
        points=build_points(chunks, BM25SparseEncoder(settings), settings),
    )
    return client


async def build_test_services(
    settings: Settings,
    llm: FakeLLM,
    redis: FakeAsyncRedis,
    ingest_runner: Callable[[Path, str], dict[str, Any]] | None = None,
    query_log: QueryLogSink | None = None,
) -> Services:
    """Wire the real services around the fakes, the same way build_services() does."""
    qdrant = await seeded_qdrant(settings)
    embedder, scorer = FakeEmbedder(), FakeScorer()
    catalog = LibraryCatalog(qdrant, settings)
    cache = SemanticCache(redis, settings)
    if query_log is None:
        query_log = JsonlQueryLog(settings)
    retrieval = RetrievalStage(
        HybridRetriever(qdrant, embedder, BM25SparseEncoder(settings), settings),
        Reranker(scorer, settings),
        RetrievalGrader(llm, settings),
    )
    pipeline = RagPipeline(
        settings=settings,
        embedder=embedder,
        retrieval=retrieval,
        generator=AnswerGenerator(llm, settings),
        rewriter=QueryRewriter(llm, settings),
        conversations=ConversationStore(redis, settings),
        cache=cache,
        index_versions=IndexVersionProvider(qdrant, settings),
        catalog=catalog,
        query_log=query_log,
        llm_model=llm.model,
    )

    async def _clear_cache(request_id: str) -> None:
        await cache.clear(request_id)

    ingest_jobs = IngestJobManager(
        ingest_runner, settings, is_ingested=catalog.has_source, on_success=_clear_cache
    )
    return Services(
        settings=settings,
        qdrant=qdrant,
        redis=redis,
        embedder=embedder,
        scorer=scorer,
        pipeline=pipeline,
        catalog=catalog,
        cache=cache,
        ingest_jobs=ingest_jobs,
        query_log=query_log,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def redis_server() -> FakeServer:
    """Set .connected = False to simulate Redis being down."""
    return FakeServer()


@pytest.fixture
def redis(redis_server: FakeServer) -> FakeAsyncRedis:
    return FakeAsyncRedis(server=redis_server)


@pytest_asyncio.fixture
async def services(settings: Settings, llm: FakeLLM, redis: FakeAsyncRedis) -> Services:
    built = await build_test_services(settings, llm, redis)
    yield built
    built.query_log.close()
