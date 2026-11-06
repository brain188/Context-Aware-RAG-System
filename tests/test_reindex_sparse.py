"""Tests for pipeline/reindex_sparse.py: keyword vectors rebuilt from stored text.

Qdrant local mode; no models, PDFs or servers.
"""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from app.config import Settings
from app.retrieval.sparse_encoder import BM25SparseEncoder
from app.services.index_version import embedder_stamp, index_stamp, sparse_stamp_problems
from pipeline.reindex_sparse import reindex_sparse
from pipeline.vector_store import CollectionSchemaError, collection_schema, ensure_collection

_STALE = SparseVector(indices=[1, 2], values=[0.5, 0.5])  # what an old tokenizer wrote


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None, gemini_api_key="test-key", **{"embedding_dimension": 3, **overrides}
    )


def _old_index(settings: Settings, chunks: dict[str, tuple[str, str]]) -> QdrantClient:
    """A collection written by the first tokenizer: embedder stamp only, stale sparse vectors.

    *chunks* maps a name to (chapter, text).
    """
    client = QdrantClient(location=":memory:")
    dense, sparse = collection_schema(settings)
    client.create_collection(
        settings.qdrant_collection,
        vectors_config=dense,
        sparse_vectors_config=sparse,
        metadata=embedder_stamp(settings),
    )
    client.upsert(
        settings.qdrant_collection,
        points=[
            PointStruct(
                id=_id(name),
                vector={
                    settings.qdrant_dense_vector_name: [0.1, 0.2, 0.3],
                    settings.qdrant_sparse_vector_name: _STALE,
                },
                payload={"text": text, "chapter": chapter},
            )
            for name, (chapter, text) in chunks.items()
        ],
    )
    return client


def _id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _vectors(client: QdrantClient, settings: Settings, name: str) -> dict:
    (point,) = client.retrieve(settings.qdrant_collection, [_id(name)], with_vectors=True)
    return point.vector


def _metadata(client: QdrantClient, settings: Settings) -> dict:
    return client.get_collection(settings.qdrant_collection).config.metadata or {}


class TestReindexSparse:
    def test_rewrites_keyword_vectors_and_keeps_dense_ones(self) -> None:
        settings = _settings()
        client = _old_index(settings, {"fig": ("8 Networks", "Fig.8.11 shows an ego network.")})
        result = reindex_sparse(client, settings)
        vectors = _vectors(client, settings, "fig")
        expected = BM25SparseEncoder(settings).encode_document("Fig.8.11 shows an ego network.")
        assert vectors[settings.qdrant_sparse_vector_name] == expected
        norm = sum(x * x for x in (0.1, 0.2, 0.3)) ** 0.5  # cosine vectors are stored normalized
        dense = [x / norm for x in (0.1, 0.2, 0.3)]
        assert vectors[settings.qdrant_dense_vector_name] == pytest.approx(dense, rel=1e-5)
        assert result.reencoded == 1

    def test_stamps_the_collection_so_ingest_and_startup_accept_it(self) -> None:
        settings = _settings()
        client = _old_index(settings, {"a": ("1 Intro", "some words")})
        reindex_sparse(client, settings)
        assert _metadata(client, settings) == index_stamp(settings)
        assert sparse_stamp_problems(_metadata(client, settings), settings) == []
        ensure_collection(client, settings)  # no longer refused

    def test_chunk_without_keyword_tokens_loses_its_stale_vector(self) -> None:
        settings = _settings()
        client = _old_index(settings, {"dots": ("1 Intro", "... !!!")})
        result = reindex_sparse(client, settings)
        assert settings.qdrant_sparse_vector_name not in _vectors(client, settings, "dots")
        assert result.cleared == 1

    def test_drop_navigation_deletes_contents_and_index_chunks(self) -> None:
        settings = _settings()
        client = _old_index(
            settings,
            {
                "toc": ("Contents", "1.1 Intro . . . 3"),
                "body": ("1 Intro", "real text"),
                "index": ("Index", "Bootstrap, 69"),
            },
        )
        result = reindex_sparse(client, settings, drop_navigation=True, batch_size=2)
        assert client.count(settings.qdrant_collection).count == 1
        assert (result.reencoded, result.dropped) == (1, 2)

    def test_navigation_chunks_stay_without_the_flag(self) -> None:
        settings = _settings()
        client = _old_index(settings, {"toc": ("Contents", "1.1 Intro . . . 3")})
        reindex_sparse(client, settings)
        assert client.count(settings.qdrant_collection).count == 1

    def test_dry_run_writes_nothing(self) -> None:
        settings = _settings()
        client = _old_index(
            settings, {"a": ("1 Intro", "some words"), "toc": ("Contents", "1.1 . . 3")}
        )
        result = reindex_sparse(client, settings, drop_navigation=True, dry_run=True)
        assert _vectors(client, settings, "a")[settings.qdrant_sparse_vector_name] == _STALE
        assert client.count(settings.qdrant_collection).count == 2
        assert "sparse_encoder" not in _metadata(client, settings)
        assert (result.reencoded, result.dropped) == (1, 1)

    def test_pages_through_more_points_than_one_batch(self) -> None:
        settings = _settings()
        client = _old_index(settings, {f"c{i}": ("1 Intro", f"chunk {i}") for i in range(5)})
        assert reindex_sparse(client, settings, batch_size=2).reencoded == 5

    def test_missing_collection_is_an_error(self) -> None:
        with pytest.raises(CollectionSchemaError, match="does not exist"):
            reindex_sparse(QdrantClient(location=":memory:"), _settings())
