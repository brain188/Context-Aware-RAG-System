"""Tests for pipeline/deduplicator.py and pipeline/vector_store.py.

Uses synthetic Chunk objects and Qdrant's in-memory local mode; no PDF,
embedding model, or Qdrant server required.
"""

import uuid
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, VectorParams

from app.config import Settings
from app.models import Chunk, ChunkMetadata
from app.retrieval.sparse_encoder import BM25SparseEncoder
from pipeline.deduplicator import deduplicate
from pipeline.vector_store import (
    CollectionSchemaError,
    build_points,
    delete_book_chunks,
    ensure_collection,
    upsert_chunks,
)


def _make_chunk(text: str, content_hash: str, page_start: int = 1) -> Chunk:
    """Build a Chunk with a caller-controlled content_hash for dedup testing."""
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            book="Test Book",
            chapter="Chapter 1",
            section="1.1 Introduction",
            page_start=page_start,
            page_end=page_start,
            source_path="/books/test.pdf",
            content_hash=content_hash,
        ),
    )


class TestDeduplicateHappyPath:
    def test_unique_chunks_all_survive(self) -> None:
        chunks = [_make_chunk("a", "h1"), _make_chunk("b", "h2"), _make_chunk("c", "h3")]
        assert len(deduplicate(chunks)) == 3

    def test_duplicate_hash_is_removed(self) -> None:
        chunks = [_make_chunk("a", "h1"), _make_chunk("a copy", "h1"), _make_chunk("b", "h2")]
        result = deduplicate(chunks)
        assert len(result) == 2
        assert [c.metadata.content_hash for c in result] == ["h1", "h2"]

    def test_keeps_first_occurrence(self) -> None:
        chunks = [_make_chunk("original", "h1"), _make_chunk("later duplicate", "h1")]
        assert deduplicate(chunks)[0].text == "original"

    def test_preserves_relative_order(self) -> None:
        chunks = [
            _make_chunk("first", "h1"),
            _make_chunk("second", "h2"),
            _make_chunk("dup of first", "h1"),
            _make_chunk("third", "h3"),
        ]
        assert [c.text for c in deduplicate(chunks)] == ["first", "second", "third"]

    def test_input_list_is_not_mutated(self) -> None:
        chunks = [_make_chunk("a", "h1"), _make_chunk("a", "h1")]
        deduplicate(chunks)
        assert len(chunks) == 2


class TestDeduplicateEdgeCases:
    def test_empty_input_returns_empty_list(self) -> None:
        assert deduplicate([]) == []

    def test_all_identical_collapses_to_one(self) -> None:
        chunks = [_make_chunk("boilerplate", "h1") for _ in range(5)]
        assert len(deduplicate(chunks)) == 1

    def test_same_text_different_hash_both_kept(self) -> None:
        """Dedup keys on content_hash only — differing hashes are never collapsed."""
        chunks = [_make_chunk("same text", "h1"), _make_chunk("same text", "h2")]
        assert len(deduplicate(chunks)) == 2


# ---------------------------------------------------------------------------
# pipeline/vector_store.py — schema contract and point building (Qdrant local mode)
# ---------------------------------------------------------------------------


def _vs_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None, llm_api_key="sk-test", **{"embedding_dimension": 3, **overrides}
    )


def _embedded_chunk(text: str, source_path: str = "/books/test.pdf", book: str = "") -> Chunk:
    chunk = _make_chunk(text, f"hash-{text}")
    chunk.metadata.source_path = source_path
    chunk.metadata.book = book or source_path
    chunk.embedding = [0.1, 0.2, 0.3]
    chunk.chunk_id = str(uuid.uuid4())
    return chunk


class TestEnsureCollection:
    def test_creates_dense_and_idf_sparse_vectors(self) -> None:
        settings = _vs_settings()
        client = QdrantClient(location=":memory:")
        ensure_collection(client, settings)
        params = client.get_collection(settings.qdrant_collection).config.params
        assert params.vectors[settings.qdrant_dense_vector_name].size == 3
        assert params.sparse_vectors[settings.qdrant_sparse_vector_name].modifier == Modifier.IDF

    def test_matching_existing_collection_is_accepted(self) -> None:
        settings = _vs_settings()
        client = QdrantClient(location=":memory:")
        ensure_collection(client, settings)
        ensure_collection(client, settings)  # no error on second run

    def test_legacy_unnamed_vector_collection_is_rejected(self) -> None:
        settings = _vs_settings()
        client = QdrantClient(location=":memory:")
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        )
        with pytest.raises(CollectionSchemaError, match="re-ingest"):
            ensure_collection(client, settings)

    def test_dimension_mismatch_is_rejected(self) -> None:
        client = QdrantClient(location=":memory:")
        ensure_collection(client, _vs_settings())
        with pytest.raises(CollectionSchemaError, match="EMBEDDING_DIMENSION"):
            ensure_collection(client, _vs_settings(embedding_dimension=4))


class TestBuildPoints:
    def test_point_has_dense_sparse_and_full_payload(self) -> None:
        settings = _vs_settings()
        chunk = _embedded_chunk("KL divergence")
        (point,) = build_points([chunk], BM25SparseEncoder(settings), settings)
        assert point.vector[settings.qdrant_dense_vector_name] == chunk.embedding
        assert point.vector[settings.qdrant_sparse_vector_name].indices
        assert point.payload == {**chunk.metadata.model_dump(), "text": chunk.text}

    def test_chunks_without_embedding_are_skipped(self) -> None:
        settings = _vs_settings()
        chunk = _embedded_chunk("KL divergence")
        chunk.embedding = None
        assert build_points([chunk], BM25SparseEncoder(settings), settings) == []

    def test_text_without_words_gets_no_sparse_vector(self) -> None:
        settings = _vs_settings()
        (point,) = build_points([_embedded_chunk("... !!!")], BM25SparseEncoder(settings), settings)
        assert settings.qdrant_sparse_vector_name not in point.vector


class TestUpsertAndDelete:
    def test_reingest_replaces_only_that_book(self) -> None:
        settings = _vs_settings()
        client = QdrantClient(location=":memory:")
        ensure_collection(client, settings)
        upsert_chunks(client, [_embedded_chunk("a", "/books/a.pdf")], settings)
        upsert_chunks(client, [_embedded_chunk("b", "/books/b.pdf")], settings)

        delete_book_chunks(client, "/books/a.pdf", "/books/a.pdf", settings)
        upsert_chunks(client, [_embedded_chunk("a v2", "/books/a.pdf")], settings)

        records, _ = client.scroll(settings.qdrant_collection, with_payload=True)
        assert sorted(r.payload["text"] for r in records) == ["a v2", "b"]

    def test_reingest_under_a_new_path_replaces_the_old_chunks(self) -> None:
        settings = _vs_settings()
        client = QdrantClient(location=":memory:")
        ensure_collection(client, settings)
        upsert_chunks(client, [_embedded_chunk("old", "books\a.pdf", book="A")], settings)
        upsert_chunks(client, [_embedded_chunk("other", "/books/b.pdf", book="B")], settings)

        delete_book_chunks(client, "C:\repo\books\a.pdf", "A", settings)

        records, _ = client.scroll(settings.qdrant_collection, with_payload=True)
        assert [r.payload["text"] for r in records] == ["other"]
