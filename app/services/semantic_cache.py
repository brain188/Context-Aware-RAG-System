"""Semantic answer cache in Redis.

Entries are grouped into namespaces keyed by index version and retrieval
scope (filters), so a re-ingest, a prompt/model change, or a different book
filter can never serve an out-of-date or out-of-scope answer. Within a
namespace, a lookup compares the query embedding with every cached query
embedding (stored as float32 bytes) and hits at cosine >= the threshold.

Redis failures degrade to a cache miss (WARNING), never a failed query.
"""

import hashlib
import time
import uuid

import numpy as np
from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings, get_logger
from app.models import Citation, RetrievalFilter

logger = get_logger(__name__)

_KEY_PREFIX = "rag:cache:"
_ID_BYTES = 16  # uuid4 bytes prefixed to each stored vector


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


def _scope_hash(filters: RetrievalFilter | None) -> str:
    scope = filters.model_dump_json(exclude_none=True) if filters else "{}"
    return hashlib.sha256(scope.encode()).hexdigest()[:16]


class SemanticCache:
    """Look up and store answers by query-embedding similarity."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._threshold = settings.semantic_cache_threshold
        self._ttl = settings.semantic_cache_ttl_seconds
        self._max_entries = settings.semantic_cache_max_entries

    @staticmethod
    def _keys(index_version: str, filters: RetrievalFilter | None) -> tuple[str, str]:
        namespace = f"{_KEY_PREFIX}{index_version}:{_scope_hash(filters)}"
        return f"{namespace}:vectors", f"{namespace}:answers"

    async def lookup(
        self,
        embedding: list[float],
        index_version: str,
        filters: RetrievalFilter | None,
        request_id: str,
    ) -> CacheLookup:
        """Return the most similar cached answer if it clears the threshold and is fresh."""
        start = time.perf_counter()
        vectors_key, answers_key = self._keys(index_version, filters)
        try:
            entries = await self._redis.lrange(vectors_key, 0, -1)
            best_id, similarity = self._best_match(entries, embedding)
            answer = None
            if best_id is not None and similarity >= self._threshold:
                raw = await self._redis.hget(answers_key, best_id)
                answer = self._parse_fresh(raw)
        except RedisError as exc:
            logger.warning(
                "Semantic cache unavailable (Redis: %s); continuing uncached.",
                exc,
                extra={"request_id": request_id},
            )
            return CacheLookup(answer=None, similarity=None)

        logger.info(
            "Semantic cache %s: entries=%d best_similarity=%s latency_ms=%.0f",
            "hit" if answer else "miss",
            len(entries),
            "-" if similarity is None else f"{similarity:.4f}",
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )
        return CacheLookup(answer=answer, similarity=similarity)

    async def store(
        self,
        embedding: list[float],
        answer: CachedAnswer,
        index_version: str,
        filters: RetrievalFilter | None,
        request_id: str,
    ) -> None:
        """Add an answer, evicting the oldest entries beyond the namespace cap."""
        vectors_key, answers_key = self._keys(index_version, filters)
        entry_id = uuid.uuid4().bytes
        vector = np.asarray(embedding, dtype=np.float32).tobytes()
        try:
            evicted = await self._redis.lrange(vectors_key, self._max_entries - 1, -1)
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.lpush(vectors_key, entry_id + vector)
                pipe.ltrim(vectors_key, 0, self._max_entries - 1)
                pipe.hset(answers_key, entry_id, answer.model_dump_json())
                if evicted:
                    pipe.hdel(answers_key, *(e[:_ID_BYTES] for e in evicted))
                pipe.expire(vectors_key, self._ttl)
                pipe.expire(answers_key, self._ttl)
                await pipe.execute()
        except RedisError as exc:
            logger.warning(
                "Could not store answer in semantic cache (Redis: %s).",
                exc,
                extra={"request_id": request_id},
            )
            return
        logger.info("Semantic cache store.", extra={"request_id": request_id})

    async def clear(self, request_id: str) -> int:
        """Delete every cache entry (all versions and scopes); return keys removed."""
        removed = 0
        try:
            async for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
                removed += await self._redis.delete(key)
        except RedisError as exc:
            logger.warning(
                "Could not clear semantic cache (Redis: %s).", exc, extra={"request_id": request_id}
            )
        logger.info("Semantic cache cleared: keys=%d", removed, extra={"request_id": request_id})
        return removed

    @staticmethod
    def _best_match(
        entries: list[bytes], embedding: list[float]
    ) -> tuple[bytes | None, float | None]:
        """Return (entry id, cosine) of the closest cached query; embeddings are L2-normalized."""
        query = np.asarray(embedding, dtype=np.float32)
        width = _ID_BYTES + query.nbytes
        usable = [e for e in entries if len(e) == width]  # skip entries from another model width
        if not usable:
            return None, None
        matrix = np.frombuffer(b"".join(e[_ID_BYTES:] for e in usable), dtype=np.float32)
        scores = matrix.reshape(len(usable), -1) @ query
        best = int(np.argmax(scores))
        return usable[best][:_ID_BYTES], float(scores[best])

    def _parse_fresh(self, raw: bytes | None) -> CachedAnswer | None:
        if raw is None:
            return None
        try:
            answer = CachedAnswer.model_validate_json(raw)
        except ValidationError:
            return None
        if time.time() - answer.created_at > self._ttl:
            return None
        return answer
