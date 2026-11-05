"""One structured record per query, appended to a rotated JSONL file.

Records hold enough to reproduce what was retrieved and why (chunk ids and
scores, grade, versions, timings) but never chunk text or secrets. Session
ids are hashed. Failed and "not covered" queries feed the eval set (Phase 4).
"""

import hashlib
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Protocol

from pydantic import BaseModel, Field

from app.config import Settings


class RetrievedRef(BaseModel):
    """A chunk id with the score a stage gave it."""

    chunk_id: str
    score: float


class SearchRetry(BaseModel):
    """One rewrite-and-retrieve attempt after an ambiguous grade."""

    query: str
    grade: str
    adopted: bool


class QueryLogRecord(BaseModel):
    """Everything about one query, written once at the end of the pipeline."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str
    session_hash: str | None = None
    question: str
    standalone_question: str | None = None
    filters: dict | None = None
    route: str | None = None
    route_reason: str | None = None
    cache_hit: bool = False
    cache_similarity: float | None = None
    retrieved: list[RetrievedRef] = Field(default_factory=list)
    reranked: list[RetrievedRef] = Field(default_factory=list)
    grade: str | None = None
    grade_reason: str | None = None
    # chunk_id → relevant | partial | irrelevant, from the grader, for the reranked chunks.
    grade_relevance: dict[str, str] = Field(default_factory=dict)
    search_retries: list[SearchRetry] = Field(default_factory=list)
    # The search query the answer's chunks came from, when a rewrite was adopted.
    search_query: str | None = None
    retried_without_filters: bool = False
    grounding: str | None = None  # output guard verdict
    status: str = "incomplete"  # answered | low_confidence | not_covered | error | cancelled
    citations: list[str] = Field(default_factory=list)  # cited chunk ids
    latency_ms: dict[str, float] = Field(default_factory=dict)
    index_version: str | None = None
    prompt_version: str
    model: str
    error: str | None = None


def hash_session_id(session_id: str | None) -> str | None:
    """Stable, non-reversible session identifier for logs."""
    if session_id is None:
        return None
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]


class QueryLogSink(Protocol):
    """Destination for query records (JSONL now, a database later)."""

    def write(self, record: QueryLogRecord) -> None: ...

    def close(self) -> None: ...


class JsonlQueryLog:
    """Append records as JSON lines, rotating by size."""

    def __init__(self, settings: Settings) -> None:
        settings.query_log_path.parent.mkdir(parents=True, exist_ok=True)
        # The handler gives us size-based rotation and a write lock; delay=True
        # defers opening the file until the first record.
        self._handler = RotatingFileHandler(
            settings.query_log_path,
            maxBytes=settings.query_log_max_bytes,
            backupCount=settings.query_log_backup_count,
            encoding="utf-8",
            delay=True,
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))

    def write(self, record: QueryLogRecord) -> None:
        """Append one record as a single JSON line."""
        self._handler.handle(logging.makeLogRecord({"msg": record.model_dump_json()}))

    def close(self) -> None:
        """Flush and close the file."""
        self._handler.close()
