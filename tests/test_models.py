"""Tests for app/models.py — Pydantic schema validation and serialization."""

import pytest
from pydantic import ValidationError

from app.models import (
    Answer,
    Chunk,
    ChunkMetadata,
    Citation,
    DependencyStatus,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

_VALID_METADATA = {
    "book": "Pattern Recognition and Machine Learning",
    "chapter": "Chapter 9: Mixture Models and EM",
    "section": "9.2 Mixtures of Gaussians",
    "page_start": 430,
    "page_end": 439,
    "source_path": "/books/bishop_prml.pdf",
    "content_hash": "abc123def456",
}

_VALID_CITATION = {
    "book": "PRML",
    "chapter": "Chapter 9",
    "section": "9.2",
    "page_start": 430,
    "page_end": 439,
    "chunk_id": "chunk-uuid-001",
    "score": 0.92,
}


class TestChunkMetadata:
    def test_valid_payload(self) -> None:
        m = ChunkMetadata(**_VALID_METADATA)
        assert m.book == "Pattern Recognition and Machine Learning"
        assert m.page_start == 430
        assert m.content_hash == "abc123def456"

    def test_round_trips_json(self) -> None:
        m = ChunkMetadata(**_VALID_METADATA)
        restored = ChunkMetadata.model_validate_json(m.model_dump_json())
        assert restored == m

    def test_missing_required_field_raises(self) -> None:
        """Every field in ChunkMetadata is required — none have defaults."""
        for field in _VALID_METADATA:
            bad = {k: v for k, v in _VALID_METADATA.items() if k != field}
            with pytest.raises(ValidationError):
                ChunkMetadata(**bad)


class TestChunk:
    def test_auto_generates_chunk_id(self) -> None:
        chunk = Chunk(text="Some text.", metadata=ChunkMetadata(**_VALID_METADATA))
        assert chunk.chunk_id  # non-empty UUID string

    def test_two_chunks_have_different_ids(self) -> None:
        a = Chunk(text="Text A.", metadata=ChunkMetadata(**_VALID_METADATA))
        b = Chunk(text="Text B.", metadata=ChunkMetadata(**_VALID_METADATA))
        assert a.chunk_id != b.chunk_id

    def test_embedding_defaults_to_none(self) -> None:
        chunk = Chunk(text="No embedding.", metadata=ChunkMetadata(**_VALID_METADATA))
        assert chunk.embedding is None

    def test_explicit_embedding(self) -> None:
        chunk = Chunk(
            text="With embedding.",
            metadata=ChunkMetadata(**_VALID_METADATA),
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_round_trips_json(self) -> None:
        chunk = Chunk(text="Round trip.", metadata=ChunkMetadata(**_VALID_METADATA))
        restored = Chunk.model_validate_json(chunk.model_dump_json())
        assert restored.text == chunk.text
        assert restored.metadata == chunk.metadata


class TestQueryRequest:
    def test_minimal_valid_request(self) -> None:
        req = QueryRequest(question="What is the EM algorithm?")
        assert req.session_id is None
        assert req.book_filter is None
        assert req.top_k is None

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_question_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 2001)

    def test_top_k_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="What?", top_k=0)

    def test_top_k_above_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="What?", top_k=51)

    def test_full_request(self) -> None:
        req = QueryRequest(
            question="Explain variational inference.",
            session_id="session-abc",
            book_filter=["bishop_prml", "murphy_ml"],
            chapter_filter="Chapter 10",
            top_k=10,
        )
        assert len(req.book_filter) == 2
        assert req.top_k == 10


class TestCitation:
    def test_valid_citation(self) -> None:
        c = Citation(**_VALID_CITATION)
        assert c.score == 0.92

    def test_missing_field_raises(self) -> None:
        bad = {k: v for k, v in _VALID_CITATION.items() if k != "chunk_id"}
        with pytest.raises(ValidationError):
            Citation(**bad)


class TestAnswerAndQueryResponse:
    def test_answer_with_citations(self) -> None:
        answer = Answer(
            text="The EM algorithm...",
            citations=[Citation(**_VALID_CITATION)],
            request_id="req-001",
        )
        assert len(answer.citations) == 1
        assert answer.session_id is None

    def test_query_response(self) -> None:
        answer = Answer(text="...", citations=[], request_id="req-002")
        resp = QueryResponse(answer=answer, request_id="req-002")
        assert resp.request_id == "req-002"


class TestIngestModels:
    def test_ingest_request_defaults(self) -> None:
        req = IngestRequest(source_path="/books/prml.pdf", book_title="PRML")
        assert not req.force_reingest

    def test_ingest_response(self) -> None:
        resp = IngestResponse(
            book_title="PRML",
            chunks_indexed=312,
            chunks_skipped_duplicate=4,
            elapsed_seconds=18.4,
            status="ok",
        )
        assert resp.status == "ok"
        assert resp.detail is None


class TestHealthResponse:
    def test_healthy_status(self) -> None:
        h = HealthResponse(
            status="healthy",
            qdrant=DependencyStatus(healthy=True),
            redis=DependencyStatus(healthy=True),
            embedding_model=DependencyStatus(healthy=True),
        )
        assert h.status == "healthy"
        assert h.qdrant.healthy

    def test_degraded_with_detail(self) -> None:
        h = HealthResponse(
            status="degraded",
            qdrant=DependencyStatus(healthy=True),
            redis=DependencyStatus(healthy=False, detail="connection refused on :6379"),
            embedding_model=DependencyStatus(healthy=True),
        )
        assert not h.redis.healthy
        assert "6379" in h.redis.detail


class TestErrorResponse:
    def test_minimal_error(self) -> None:
        e = ErrorResponse(error="Not found")
        assert e.detail is None
        assert e.request_id is None

    def test_full_error(self) -> None:
        e = ErrorResponse(
            error="Validation failed",
            detail="'question' field is required",
            request_id="req-err-001",
        )
        assert e.request_id == "req-err-001"
