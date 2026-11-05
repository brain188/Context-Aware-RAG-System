"""Semantic answer cache: Qdrant finds the nearest cached question, Redis holds the answer.

Each cached entry is one point in a small Qdrant collection (the standalone
question's embedding, plus its scope and creation time) and one Redis key
holding the answer with a TTL. A lookup asks Qdrant for the single nearest
point *within the request's scope*. A point at or above
SEMANTIC_CACHE_THRESHOLD whose answer is still in Redis is a candidate; it
hits when it is a near-exact repeat or the CacheMatcher's fast-model check
confirms that one answer serves both questions (similarity alone cannot tell
"field-level" from "form-level validation").

The scope is the index version plus the retrieval filters. The index version
already fingerprints the prompt version, the generation provider and model,
the embedder, and the collection's point count (services/index_version.py),
so a prompt bump, a model change, a re-ingest, or a different book filter is
always a miss.

Qdrant or Redis failures degrade to a cache miss (WARNING), never a failed query.
"""

import asyncio
import hashlib
import time
import uuid

from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings, get_logger
from app.models import Citation, RetrievalFilter
from app.services.cache_matcher import CacheMatcher, MatchKind

logger = get_logger(__name__)

_ANSWER_PREFIX = "rag:cache:answer:"
_SCOPE = "scope"
_CREATED_AT = "created_at"

# What a cache backend can raise; every one of them means "treat as a miss".
_BACKEND_ERRORS = (RedisError, UnexpectedResponse, ResponseHandlingException)


class CachedAnswer(BaseModel):
    """What a cache hit returns."""

    text: str
    citations: list[Citation]
    question: str  # the standalone question that produced it
    created_at: float


class CacheLookup(BaseModel):
    """Result of a lookup; *answer* is None on a miss."""

    answer: CachedAnswer | None
    similarity: float | None
    match: MatchKind | None = None  # set when a candidate cleared the threshold


def cache_scope(index_version: str, filters: RetrievalFilter | None) -> str:
    """The partition an entry lives in: same index version and same filters."""
    filter_json = filters.model_dump_json(exclude_none=True) if filters else "{}"
    return f"{index_version}:{hashlib.sha256(filter_json.encode()).hexdigest()[:16]}"


def _in_scope(scope: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=_SCOPE, match=models.MatchValue(value=scope))]
    )


class SemanticCache:
    """Look up and store answers by query-embedding similarity."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        redis: Redis,
        settings: Settings,
        matcher: CacheMatcher | None = None,
    ) -> None:
        """Without a *matcher*, only near-exact repeats (exact threshold) hit."""
        self._qdrant = qdrant
        self._matcher = matcher
        self._exact = settings.semantic_cache_exact_threshold
        self._redis = redis
        self._collection = settings.semantic_cache_collection
        self._dimension = settings.embedding_dimension
        self._threshold = settings.semantic_cache_threshold
        self._ttl = settings.semantic_cache_ttl_seconds
        self._max_entries = settings.semantic_cache_max_entries
        self._ready = False
        self._lock = asyncio.Lock()

    async def lookup(
        self,
        embedding: list[float],
        question: str,
        index_version: str,
        filters: RetrievalFilter | None,
        request_id: str,
    ) -> CacheLookup:
        """Return the nearest cached answer in scope if it is fresh and fits *question*."""
        start = time.perf_counter()
        candidate, similarity = None, None
        try:
            await self._ensure_collection()
            result = await self._qdrant.query_points(
                self._collection,
                query=embedding,
                query_filter=_in_scope(cache_scope(index_version, filters)),
                limit=1,
                with_payload=False,
            )
            if result.points:
                point = result.points[0]
                similarity = point.score
                if similarity >= self._threshold:
                    candidate = await self._fresh_answer(str(point.id))
        except _BACKEND_ERRORS as exc:
            logger.warning(
                "Semantic cache unavailable (%s: %s); continuing uncached.",
                type(exc).__name__,
                exc,
                extra={"request_id": request_id},
            )
            return CacheLookup(answer=None, similarity=None)

        match: MatchKind | None = None
        if candidate is not None and similarity is not None:
            match = await self._match(question, candidate.question, similarity, request_id)
        answer = candidate if match in ("exact", "verified") else None
        logger.info(
            "Semantic cache %s: best_similarity=%s threshold=%.2f match=%s latency_ms=%.0f",
            "hit" if answer else "miss",
            "-" if similarity is None else f"{similarity:.4f}",
            self._threshold,
            match or "-",
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )
        return CacheLookup(answer=answer, similarity=similarity, match=match)

    async def _match(
        self, question: str, cached_question: str, similarity: float, request_id: str
    ) -> MatchKind:
        if self._matcher is not None:
            return await self._matcher.match(question, cached_question, similarity, request_id)
        return "exact" if similarity >= self._exact else "failed"

    async def store(
        self,
        embedding: list[float],
        answer: CachedAnswer,
        index_version: str,
        filters: RetrievalFilter | None,
        request_id: str,
    ) -> None:
        """Add an answer; drop expired entries and the oldest ones beyond the scope's cap."""
        scope = cache_scope(index_version, filters)
        entry_id = str(uuid.uuid4())
        try:
            await self._ensure_collection()
            # Answer first: a point whose answer is missing is just a miss.
            await self._redis.set(_ANSWER_PREFIX + entry_id, answer.model_dump_json(), ex=self._ttl)
            point = models.PointStruct(
                id=entry_id,
                vector=embedding,
                payload={_SCOPE: scope, _CREATED_AT: answer.created_at},
            )
            await self._qdrant.upsert(self._collection, points=[point])
            await self._prune(scope)
        except _BACKEND_ERRORS as exc:
            logger.warning(
                "Could not store answer in semantic cache (%s: %s).",
                type(exc).__name__,
                exc,
                extra={"request_id": request_id},
            )
            return
        logger.info("Semantic cache store.", extra={"request_id": request_id})

    async def clear(self, request_id: str) -> int:
        """Delete every cached entry (all versions and scopes); return answers removed."""
        removed = 0
        try:
            async for key in self._redis.scan_iter(match=f"{_ANSWER_PREFIX}*"):
                removed += await self._redis.delete(key)
            if await self._qdrant.collection_exists(self._collection):
                await self._qdrant.delete_collection(self._collection)
            self._ready = False
        except _BACKEND_ERRORS as exc:
            logger.warning(
                "Could not clear semantic cache (%s: %s).",
                type(exc).__name__,
                exc,
                extra={"request_id": request_id},
            )
        logger.info("Semantic cache cleared: entries=%d", removed, extra={"request_id": request_id})
        return removed

    async def _ensure_collection(self) -> None:
        """Create the cache collection once; recreate it if the embedder width changed."""
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            if await self._qdrant.collection_exists(self._collection):
                info = await self._qdrant.get_collection(self._collection)
                size = getattr(info.config.params.vectors, "size", None)
                if size == self._dimension:
                    self._ready = True
                    return
                logger.warning(
                    "Semantic cache collection '%s' holds %s-d vectors, the embedder makes "
                    "%d-d; recreating it.",
                    self._collection,
                    size,
                    self._dimension,
                )
                await self._qdrant.delete_collection(self._collection)
            await self._qdrant.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension, distance=models.Distance.COSINE
                ),
            )
            # Scope filters every lookup; created_at orders eviction.
            await self._qdrant.create_payload_index(
                self._collection, _SCOPE, models.PayloadSchemaType.KEYWORD
            )
            await self._qdrant.create_payload_index(
                self._collection, _CREATED_AT, models.PayloadSchemaType.FLOAT
            )
            self._ready = True

    async def _fresh_answer(self, entry_id: str) -> CachedAnswer | None:
        raw = await self._redis.get(_ANSWER_PREFIX + entry_id)
        if raw is None:
            return None  # expired in Redis; _prune removes the point later
        try:
            answer = CachedAnswer.model_validate_json(raw)
        except ValidationError:
            return None
        if time.time() - answer.created_at > self._ttl:
            return None
        return answer

    async def _prune(self, scope: str) -> None:
        """Delete expired points everywhere, then the oldest points beyond the scope's cap."""
        expired = models.Filter(
            must=[
                models.FieldCondition(
                    key=_CREATED_AT, range=models.Range(lt=time.time() - self._ttl)
                )
            ]
        )
        await self._qdrant.delete(
            self._collection, points_selector=models.FilterSelector(filter=expired)
        )
        count = (
            await self._qdrant.count(self._collection, count_filter=_in_scope(scope), exact=True)
        ).count
        overflow = count - self._max_entries
        if overflow <= 0:
            return
        oldest, _ = await self._qdrant.scroll(
            self._collection,
            scroll_filter=_in_scope(scope),
            limit=overflow,
            order_by=models.OrderBy(key=_CREATED_AT, direction=models.Direction.ASC),
            with_payload=False,
        )
        ids = [str(p.id) for p in oldest]
        await self._qdrant.delete(self._collection, points_selector=models.PointIdsList(points=ids))
        await self._redis.delete(*(_ANSWER_PREFIX + i for i in ids))
