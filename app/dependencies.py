"""Build every runtime service once at startup and hand them to routes.

main.py's lifespan calls build_services(); routes get the container through
get_services(). Tests construct Services directly with fakes.
"""

import asyncio
from dataclasses import dataclass

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.config import Settings, get_logger
from app.llm import LLMFactory
from app.retrieval.grader import RetrievalGrader
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_embedder import QueryEmbedder, SentenceTransformerQueryEmbedder
from app.retrieval.reranker import CrossEncoderScorer, PairScorer, Reranker
from app.retrieval.sparse_encoder import BM25SparseEncoder
from app.services.conversation import ConversationStore, QueryRewriter
from app.services.generator import AnswerGenerator
from app.services.index_version import IndexVersionProvider
from app.services.ingest_jobs import IngestJobManager, load_pipeline_runner
from app.services.library import LibraryCatalog
from app.services.query_logger import JsonlQueryLog, QueryLogSink
from app.services.rag_pipeline import RagPipeline
from app.services.retrieval_stage import RetrievalStage
from app.services.semantic_cache import SemanticCache

logger = get_logger(__name__)


@dataclass
class Services:
    """Everything routes need; owned by the app for its whole lifetime."""

    settings: Settings
    qdrant: AsyncQdrantClient
    redis: Redis
    embedder: QueryEmbedder
    scorer: PairScorer
    pipeline: RagPipeline
    catalog: LibraryCatalog
    cache: SemanticCache
    ingest_jobs: IngestJobManager
    query_log: QueryLogSink
    llms: LLMFactory | None = None  # owns the LLM clients; tests inject fakes instead

    async def aclose(self) -> None:
        """Release network clients and flush the query log."""
        await self.qdrant.close()
        await self.redis.aclose()
        if self.llms is not None:
            await self.llms.aclose()
        self.query_log.close()


def build_services(settings: Settings, query_log: QueryLogSink | None = None) -> Services:
    """Wire the production services from settings (no network calls yet).

    *query_log* replaces the JSONL query log (the eval harness captures records).
    """
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
    # Bytes in, bytes out: the semantic cache stores raw float32 vectors.
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    llms = LLMFactory(settings)
    generation_llm, fast_llm = llms.get("generation"), llms.get("fast")
    embedder = SentenceTransformerQueryEmbedder(settings)
    scorer = CrossEncoderScorer(settings)

    catalog = LibraryCatalog(qdrant, settings)
    cache = SemanticCache(redis, settings)
    if query_log is None:
        query_log = JsonlQueryLog(settings)
    retrieval = RetrievalStage(
        HybridRetriever(qdrant, embedder, BM25SparseEncoder(settings), settings),
        Reranker(scorer, settings),
        RetrievalGrader(fast_llm, settings),
    )
    pipeline = RagPipeline(
        settings=settings,
        embedder=embedder,
        retrieval=retrieval,
        generator=AnswerGenerator(generation_llm, settings),
        rewriter=QueryRewriter(fast_llm, settings),
        conversations=ConversationStore(redis, settings),
        cache=cache,
        index_versions=IndexVersionProvider(qdrant, settings),
        catalog=catalog,
        query_log=query_log,
        llm_model=generation_llm.model,
    )

    async def _after_ingest(request_id: str) -> None:
        # The index version changes with the point count anyway; clearing also
        # covers a re-ingest that happens to keep the same count.
        await cache.clear(request_id)

    ingest_jobs = IngestJobManager(
        load_pipeline_runner(), settings, is_ingested=catalog.has_source, on_success=_after_ingest
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
        llms=llms,
    )


async def warm_models(services: Services) -> None:
    """Load the embedding and reranker models now, so the first query is not slow."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, services.embedder.embed_query, "warm up")
    await loop.run_in_executor(None, services.scorer.score, [("warm up", "warm up")])
    logger.info("Models warmed up.")


def get_services(request: Request) -> Services:
    """FastAPI dependency: the container created at startup."""
    return request.app.state.services
