"""Tests for app/retrieval/ — filters, BM25 sparse encoder, hybrid retriever, reranker.

Hybrid retrieval runs against Qdrant's in-memory local mode, so the real
dense + sparse prefetch, IDF modifier, filters, and RRF fusion are exercised
with no server. Points are written with the pipeline's own build_points, which
also tests the pipeline/app payload contract. Embedding and cross-encoder
models are replaced with fakes; no network, GPU, or model download is required.
"""

import uuid
import zlib
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    SparseVector,
)

from app.config import Settings
from app.models import Chunk, ChunkMetadata, QueryRequest, RetrievalFilter, ScoredChunk
from app.retrieval.errors import RerankerError, VectorStoreError
from app.retrieval.filters import filter_from_request, is_unfiltered, to_qdrant_filter
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.sparse_encoder import BM25SparseEncoder, token_index, tokenize
from pipeline.vector_store import build_points, collection_schema

# ---------------------------------------------------------------------------
# Helpers and fakes
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    """Settings isolated from the developer's real .env files."""
    return Settings(_env_file=None, llm_api_key="sk-test", **overrides)


# name -> (text, dense embedding, book, page)
_DOCS: dict[str, tuple[str, list[float], str, int]] = {
    "kl": (
        "The KL divergence measures how one distribution differs from another.",
        [1.0, 0.0, 0.0, 0.0],
        "PRML",
        10,
    ),
    "svm": (
        "Support vector machines maximise the margin between classes.",
        [0.0, 1.0, 0.0, 0.0],
        "PRML",
        20,
    ),
    "nn": (
        "Neural networks are trained with gradient descent and backpropagation.",
        [0.0, 0.0, 1.0, 0.0],
        "PRML",
        30,
    ),
    "em": (
        "The EM algorithm alternates expectation and maximisation steps.",
        [0.0, 0.0, 0.0, 1.0],
        "ESL",
        40,
    ),
}
_IDS = {name: str(uuid.uuid5(uuid.NAMESPACE_URL, name)) for name in _DOCS}
_NAMES = {point_id: name for name, point_id in _IDS.items()}


def _chunk(name: str, with_embedding: bool = True) -> Chunk:
    text, embedding, book, page = _DOCS[name]
    return Chunk(
        chunk_id=_IDS[name],
        text=text,
        embedding=embedding if with_embedding else None,
        metadata=ChunkMetadata(
            book=book,
            chapter="1 Introduction",
            section="1.1 Example",
            page_start=page,
            page_end=page,
            source_path=f"/books/{book}.pdf",
            content_hash=f"hash-{name}",
        ),
    )


class FakeEmbedder:
    """Returns a fixed query vector so the dense ranking is fully controlled."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed_query(self, text: str) -> list[float]:
        return self.vector


async def _seeded_client(settings: Settings) -> AsyncQdrantClient:
    """In-memory Qdrant with the pipeline's schema and all _DOCS written by build_points."""
    client = AsyncQdrantClient(location=":memory:")
    dense, sparse = collection_schema(settings)
    await client.create_collection(
        settings.qdrant_collection, vectors_config=dense, sparse_vectors_config=sparse
    )
    points = build_points([_chunk(n) for n in _DOCS], BM25SparseEncoder(settings), settings)
    await client.upsert(settings.qdrant_collection, points=points)
    return client


def _settings_4d(**overrides: Any) -> Settings:
    return _settings(embedding_dimension=4, retrieval_top_k=2, **overrides)


async def _retrieve(query: str, query_vector: list[float], **kwargs: Any) -> list[ScoredChunk]:
    settings = _settings_4d()
    client = await _seeded_client(settings)
    retriever = HybridRetriever(
        client, FakeEmbedder(query_vector), BM25SparseEncoder(settings), settings
    )
    return await retriever.retrieve(query, request_id="r1", **kwargs)


def _weight(vector: SparseVector, term: str) -> float:
    return dict(zip(vector.indices, vector.values))[token_index(term)]


def _names(results: list[ScoredChunk]) -> list[str]:
    return [_NAMES[r.chunk.chunk_id] for r in results]


class FailingQdrant:
    """Stand-in client whose query_points raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def query_points(self, *args: Any, **kwargs: Any) -> None:
        raise self.exc


class FakeScorer:
    def __init__(self, scores_by_text: dict[str, float], wrong_length: bool = False) -> None:
        self.scores_by_text = scores_by_text
        self.wrong_length = wrong_length

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = [self.scores_by_text[passage] for _, passage in pairs]
        return scores[:-1] if self.wrong_length else scores


# ---------------------------------------------------------------------------
# filters.py
# ---------------------------------------------------------------------------


class TestFilters:
    def test_no_filter_maps_to_none(self) -> None:
        assert to_qdrant_filter(None) is None
        assert to_qdrant_filter(RetrievalFilter()) is None

    def test_books_use_match_any(self) -> None:
        result = to_qdrant_filter(RetrievalFilter(books=["PRML", "ESL"]))
        assert result == Filter(
            must=[FieldCondition(key="book", match=MatchAny(any=["PRML", "ESL"]))]
        )

    def test_chapter_uses_exact_match(self) -> None:
        result = to_qdrant_filter(RetrievalFilter(chapter="3 Linear Models"))
        assert result == Filter(
            must=[FieldCondition(key="chapter", match=MatchValue(value="3 Linear Models"))]
        )

    def test_page_range_keeps_overlapping_chunks(self) -> None:
        result = to_qdrant_filter(RetrievalFilter(page_from=10, page_to=20))
        assert result == Filter(
            must=[
                FieldCondition(key="page_end", range=Range(gte=10)),
                FieldCondition(key="page_start", range=Range(lte=20)),
            ]
        )

    def test_combined_constraints_are_anded(self) -> None:
        result = to_qdrant_filter(RetrievalFilter(books=["PRML"], chapter="1", page_from=5))
        assert result is not None and len(result.must) == 3

    def test_inverted_page_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalFilter(page_from=20, page_to=10)
        with pytest.raises(ValidationError):
            QueryRequest(question="q", page_from=20, page_to=10)

    def test_empty_book_list_on_filter_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalFilter(books=[])

    def test_from_request_treats_empty_book_list_as_unfiltered(self) -> None:
        request = QueryRequest(question="q", book_filter=[], chapter_filter="")
        assert is_unfiltered(filter_from_request(request))

    def test_from_request_copies_all_scoping_fields(self) -> None:
        request = QueryRequest(
            question="q", book_filter=["PRML"], chapter_filter="2", page_from=3, page_to=4
        )
        assert filter_from_request(request) == RetrievalFilter(
            books=["PRML"], chapter="2", page_from=3, page_to=4
        )


# ---------------------------------------------------------------------------
# sparse_encoder.py
# ---------------------------------------------------------------------------


class TestBM25SparseEncoder:
    def test_tokenize_lowercases_and_strips_punctuation(self) -> None:
        assert tokenize("KL-divergence, ELBO!") == ["kl", "divergence", "elbo"]

    def test_token_index_is_stable_crc32(self) -> None:
        # Pipeline and app must agree across processes and platforms; hash() would not.
        assert token_index("kl") == zlib.crc32(b"kl")

    def test_query_has_unit_weight_per_distinct_term(self) -> None:
        vector = BM25SparseEncoder(_settings()).encode_query("KL kl divergence")
        assert len(vector.indices) == 2
        assert vector.values == [1.0, 1.0]

    def test_indices_are_sorted_and_unique(self) -> None:
        vector = BM25SparseEncoder(_settings()).encode_document("b a c a b a")
        assert vector.indices == sorted(set(vector.indices))

    def test_term_frequency_saturates(self) -> None:
        encoder = BM25SparseEncoder(_settings(bm25_avg_doc_length=3.0))
        once = _weight(encoder.encode_document("kl x y"), "kl")
        thrice = _weight(encoder.encode_document("kl kl kl"), "kl")
        assert once < thrice < _settings().bm25_k1 + 1

    def test_longer_documents_weigh_a_term_less(self) -> None:
        encoder = BM25SparseEncoder(_settings())
        short = encoder.encode_document("kl divergence")
        long = encoder.encode_document("kl " + " ".join(f"w{i}" for i in range(500)))
        assert _weight(long, "kl") < _weight(short, "kl")

    def test_text_without_words_encodes_empty(self) -> None:
        encoder = BM25SparseEncoder(_settings())
        assert encoder.encode_query("?!").indices == []
        assert encoder.encode_document("...").indices == []


# ---------------------------------------------------------------------------
# hybrid_retriever.py (real Qdrant local mode)
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_keyword_only_hit_is_found(self) -> None:
        # Dense prefetch (top 2) returns nn + svm only; BM25 must surface the exact term.
        results = await _retrieve("KL divergence", [0.0, 0.5, 1.0, 0.0], limit=3)
        assert "kl" in _names(results)

    @pytest.mark.asyncio
    async def test_agreement_between_branches_ranks_first(self) -> None:
        results = await _retrieve("KL divergence", [1.0, 0.0, 0.2, 0.0])
        assert _names(results)[0] == "kl"

    @pytest.mark.asyncio
    async def test_results_carry_payload_metadata_and_rrf_scores(self) -> None:
        results = await _retrieve("KL divergence", [1.0, 0.0, 0.2, 0.0])
        top = results[0]
        assert isinstance(top, ScoredChunk)
        assert top.chunk == _chunk("kl", with_embedding=False)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True) and scores[0] > 0

    @pytest.mark.asyncio
    async def test_book_filter_scopes_both_branches(self) -> None:
        results = await _retrieve(
            "KL divergence the", [1.0, 0.0, 0.0, 0.0], filters=RetrievalFilter(books=["ESL"])
        )
        assert _names(results) == ["em"]

    @pytest.mark.asyncio
    async def test_page_range_filter_scopes_results(self) -> None:
        results = await _retrieve(
            "the margin", [1.0, 1.0, 1.0, 1.0], filters=RetrievalFilter(page_from=15, page_to=35)
        )
        assert results and set(_names(results)) <= {"svm", "nn"}

    @pytest.mark.asyncio
    async def test_symbol_only_query_falls_back_to_dense(self) -> None:
        results = await _retrieve("?!", [0.0, 0.0, 1.0, 0.0])
        assert _names(results)[0] == "nn"

    @pytest.mark.asyncio
    async def test_limit_caps_fused_results(self) -> None:
        results = await _retrieve("the", [1.0, 1.0, 1.0, 1.0], limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_blank_query_is_rejected(self) -> None:
        settings = _settings_4d()
        retriever = HybridRetriever(
            FailingQdrant(RuntimeError()),  # type: ignore[arg-type]
            FakeEmbedder([0.0] * 4),
            BM25SparseEncoder(settings),
            settings,
        )
        with pytest.raises(ValueError):
            await retriever.retrieve("   ", request_id="r1")

    @pytest.mark.parametrize(
        ("exc", "message"),
        [
            (UnexpectedResponse(404, "Not Found", b"", httpx.Headers()), "older than v1.10"),
            (UnexpectedResponse(400, "Bad Request", b"", httpx.Headers()), "re-ingest"),
            (ResponseHandlingException(ConnectionError("refused")), "hybrid query failed"),
        ],
    )
    @pytest.mark.asyncio
    async def test_qdrant_errors_map_to_vector_store_error(
        self, exc: Exception, message: str
    ) -> None:
        settings = _settings_4d()
        retriever = HybridRetriever(
            FailingQdrant(exc),  # type: ignore[arg-type]
            FakeEmbedder([0.0] * 4),
            BM25SparseEncoder(settings),
            settings,
        )
        with pytest.raises(VectorStoreError, match=message):
            await retriever.retrieve("KL", request_id="r1")

    @pytest.mark.asyncio
    async def test_malformed_payload_raises_vector_store_error(self) -> None:
        settings = _settings_4d()
        client = await _seeded_client(settings)
        await client.upsert(
            settings.qdrant_collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={settings.qdrant_dense_vector_name: [0.0, 0.0, 0.0, 9.0]},
                    payload={"text": "missing metadata"},
                )
            ],
        )
        retriever = HybridRetriever(
            client, FakeEmbedder([0.0, 0.0, 0.0, 1.0]), BM25SparseEncoder(settings), settings
        )
        with pytest.raises(VectorStoreError, match="malformed payload"):
            await retriever.retrieve("zzz", request_id="r1")


# ---------------------------------------------------------------------------
# reranker.py
# ---------------------------------------------------------------------------


def _candidates() -> list[ScoredChunk]:
    return [ScoredChunk(chunk=_chunk(name), score=0.5) for name in _DOCS]


def _text(name: str) -> str:
    return _DOCS[name][0]


class TestReranker:
    @pytest.mark.asyncio
    async def test_reorders_by_cross_encoder_score(self) -> None:
        scores = {_text("kl"): 0.1, _text("svm"): 0.9, _text("nn"): 0.5, _text("em"): 0.3}
        reranker = Reranker(FakeScorer(scores), _settings(rerank_top_k=4))
        results = await reranker.rerank("q", _candidates(), request_id="r1")
        assert _names(results) == ["svm", "nn", "em", "kl"]
        assert [r.score for r in results] == [0.9, 0.5, 0.3, 0.1]

    @pytest.mark.asyncio
    async def test_truncates_to_top_k(self) -> None:
        scores = {_text(name): float(i) for i, name in enumerate(_DOCS)}
        reranker = Reranker(FakeScorer(scores), _settings(rerank_top_k=2))
        assert len(await reranker.rerank("q", _candidates(), request_id="r1")) == 2
        assert len(await reranker.rerank("q", _candidates(), request_id="r1", top_k=3)) == 3

    @pytest.mark.asyncio
    async def test_empty_candidates_return_empty(self) -> None:
        reranker = Reranker(FakeScorer({}), _settings())
        assert await reranker.rerank("q", [], request_id="r1") == []

    @pytest.mark.asyncio
    async def test_score_count_mismatch_raises(self) -> None:
        scores = {_text(name): 1.0 for name in _DOCS}
        reranker = Reranker(FakeScorer(scores, wrong_length=True), _settings())
        with pytest.raises(RerankerError):
            await reranker.rerank("q", _candidates(), request_id="r1")

    @pytest.mark.asyncio
    async def test_scorer_exception_is_wrapped(self) -> None:
        reranker = Reranker(FakeScorer({}), _settings())  # KeyError on every passage
        with pytest.raises(RerankerError):
            await reranker.rerank("q", _candidates(), request_id="r1")
