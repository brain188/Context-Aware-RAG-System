"""Pydantic schemas for all HTTP request/response types and internal DTOs.

ChunkMetadata is the shared contract between the offline ingestion pipeline and
online retrieval. Freeze it before building either side — any change requires
re-ingesting the vector store.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pipeline / ingestion contract (shared with pipeline/)
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk — written by the pipeline, read by retrieval."""

    book: str
    chapter: str
    section: str
    page_start: int
    page_end: int
    source_path: str
    content_hash: str  # SHA-256 of chunk text; used for deduplication


class Chunk(BaseModel):
    """A single text chunk with its metadata and an optional embedding vector."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A verified source reference attached to an answer."""

    book: str
    chapter: str
    section: str
    page_start: int
    page_end: int
    chunk_id: str
    score: float  # rerank score; higher = more relevant


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming user query with optional retrieval constraints."""

    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    book_filter: list[str] | None = None  # restrict search to these books
    chapter_filter: str | None = None
    top_k: int | None = Field(None, ge=1, le=50)


class StreamToken(BaseModel):
    """A single token emitted during SSE streaming."""

    token: str
    request_id: str


class Answer(BaseModel):
    """Final answer payload, sent after the last stream token."""

    text: str
    citations: list[Citation]
    request_id: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    """Non-streaming query response (used in tests and sync clients)."""

    answer: Answer
    request_id: str


# ---------------------------------------------------------------------------
# Ingest API
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request to ingest a single PDF book."""

    source_path: str = Field(..., description="Absolute or relative path to the PDF")
    book_title: str = Field(..., description="Human-readable title used as the metadata key")
    force_reingest: bool = False


class IngestResponse(BaseModel):
    """Result summary of an ingestion run."""

    book_title: str
    chunks_indexed: int
    chunks_skipped_duplicate: int
    elapsed_seconds: float
    status: str  # "ok" | "error"
    detail: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class DependencyStatus(BaseModel):
    """Health status of a single external dependency."""

    healthy: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregated health of all dependencies."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    qdrant: DependencyStatus
    redis: DependencyStatus
    embedding_model: DependencyStatus


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all HTTP error responses."""

    error: str
    detail: str | None = None
    request_id: str | None = None
