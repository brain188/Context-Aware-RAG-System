"""The embedder stamp: the pipeline records which model wrote the index, and the
app (and eval) refuse to query it with a different one.

Uses Qdrant's in-memory local mode; no models or servers.
"""

import asyncio
from pathlib import Path

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.config import Settings
from app.main import create_app
from app.retrieval.sparse_encoder import SPARSE_ENCODER_VERSION
from app.services.index_version import (
    LEGACY_SPARSE_ENCODER,
    IndexMismatchError,
    check_index_embedder,
    embedder_stamp,
    index_stamp,
    sparse_stamp_problems,
    stamp_problems,
)
from conftest import FakeLLM, build_test_services, make_settings
from pipeline.vector_store import CollectionSchemaError, collection_schema, ensure_collection


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None, gemini_api_key="test-key", **{"embedding_dimension": 3, **overrides}
    )


def _bare_collection(client: QdrantClient, settings: Settings, metadata=None) -> None:
    dense, sparse = collection_schema(settings)
    client.create_collection(
        settings.qdrant_collection,
        vectors_config=dense,
        sparse_vectors_config=sparse,
        metadata=metadata,
    )


def _metadata(client: QdrantClient, settings: Settings) -> dict:
    return client.get_collection(settings.qdrant_collection).config.metadata or {}


class TestStampProblems:
    def test_matching_or_missing_fields_are_fine(self) -> None:
        settings = _settings()
        assert stamp_problems(embedder_stamp(settings), settings) == []
        assert stamp_problems({}, settings) == []

    def test_a_different_model_is_reported(self) -> None:
        settings = _settings()
        (problem,) = stamp_problems({"embedding_model": "BAAI/bge-large-en-v1.5"}, settings)
        assert "bge-large" in problem and "EMBEDDING_MODEL" in problem


class TestSparseStampProblems:
    def test_matching_stamp_is_fine(self) -> None:
        settings = _settings()
        assert sparse_stamp_problems(index_stamp(settings), settings) == []

    def test_unstamped_index_counts_as_the_first_tokenizer(self) -> None:
        (problem,) = sparse_stamp_problems({}, _settings())
        assert LEGACY_SPARSE_ENCODER in problem and SPARSE_ENCODER_VERSION in problem

    def test_changed_bm25_setting_is_reported(self) -> None:
        stamp = index_stamp(_settings())
        (problem,) = sparse_stamp_problems(stamp, _settings(bm25_k1=2.0))
        assert "BM25_K1" in problem


class TestPipelineStamp:
    def test_new_collection_records_its_embedder_and_tokenizer(self) -> None:
        settings, client = _settings(), QdrantClient(location=":memory:")
        ensure_collection(client, settings)
        assert _metadata(client, settings) == index_stamp(settings)

    def test_unstamped_collection_gets_its_embedder_then_needs_a_sparse_reindex(self) -> None:
        settings, client = _settings(), QdrantClient(location=":memory:")
        _bare_collection(client, settings)
        with pytest.raises(CollectionSchemaError, match="reindex_sparse"):
            ensure_collection(client, settings)
        assert _metadata(client, settings)["embedding_model"] == settings.embedding_model

    def test_ingest_with_another_tokenizer_is_refused(self) -> None:
        settings, client = _settings(), QdrantClient(location=":memory:")
        _bare_collection(client, settings, {**index_stamp(settings), "sparse_encoder": "bm25-v1"})
        with pytest.raises(CollectionSchemaError, match="bm25-v1"):
            ensure_collection(client, settings)

    def test_ingest_into_another_models_collection_is_refused(self) -> None:
        settings, client = _settings(), QdrantClient(location=":memory:")
        _bare_collection(client, settings, metadata={"embedding_model": "other/model"})
        with pytest.raises(CollectionSchemaError, match="other/model"):
            ensure_collection(client, settings)


async def _async_collection(settings: Settings, metadata=None) -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    dense, sparse = collection_schema(settings)
    await client.create_collection(
        settings.qdrant_collection,
        vectors_config=dense,
        sparse_vectors_config=sparse,
        metadata=metadata,
    )
    return client


class TestStartupCheck:
    @pytest.mark.asyncio
    async def test_matching_stamp_passes(self) -> None:
        settings = _settings()
        client = await _async_collection(settings, embedder_stamp(settings))
        await check_index_embedder(client, settings)

    @pytest.mark.asyncio
    async def test_outdated_keyword_vectors_only_warn(self, caplog) -> None:
        settings = _settings()
        client = await _async_collection(settings, embedder_stamp(settings))  # no tokenizer
        await check_index_embedder(client, settings)
        assert "reindex_sparse" in caplog.text and LEGACY_SPARSE_ENCODER in caplog.text

    @pytest.mark.asyncio
    async def test_matching_full_stamp_is_quiet(self, caplog) -> None:
        settings = _settings()
        client = await _async_collection(settings, index_stamp(settings))
        await check_index_embedder(client, settings)
        assert "reindex_sparse" not in caplog.text

    @pytest.mark.asyncio
    async def test_mismatched_stamp_raises(self) -> None:
        settings = _settings()
        client = await _async_collection(settings, {"embedding_model": "other/model"})
        with pytest.raises(IndexMismatchError, match="other/model"):
            await check_index_embedder(client, settings)

    @pytest.mark.asyncio
    async def test_missing_or_unstamped_collection_only_warns(self, caplog) -> None:
        settings = _settings()
        await check_index_embedder(AsyncQdrantClient(location=":memory:"), settings)
        await check_index_embedder(await _async_collection(settings), settings)
        assert "does not exist" in caplog.text and "does not record" in caplog.text

    def test_api_refuses_to_start_against_another_models_index(
        self, tmp_path: Path, llm: FakeLLM
    ) -> None:
        settings = make_settings(tmp_path)
        services = asyncio.run(build_test_services(settings, llm, FakeAsyncRedis()))
        asyncio.run(
            services.qdrant.update_collection(
                settings.qdrant_collection, metadata={"embedding_model": "other/model"}
            )
        )
        with pytest.raises(IndexMismatchError):
            with TestClient(create_app(settings=settings, services=services)):
                pass
