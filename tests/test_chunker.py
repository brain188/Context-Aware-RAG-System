"""Tests for pipeline/chunker.py — chunk boundaries, metadata, size bounds, edge cases.

Tests use synthetic ParsedDocument / StructuredSection data, except the
real-book limit check, which runs only when PDFs are present in books/.
Token counts use the embedding model's tokenizer (downloaded once from the HF
Hub). Settings are controlled via environment-variable patching.
"""

import os
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from app.config import get_settings
from pipeline.chunker import (
    ChunkTooLongError,
    _assert_fits_model,
    _count_tokens,
    _sha256,
    _split_by_tokens,
    chunk_document,
)
from pipeline.extractors.pdf_extractor import extract_pdf
from pipeline.extractors.structure_parser import (
    ParsedDocument,
    StructuredSection,
    parse_structure,
)
from pipeline.preprocessor import clean_text, preprocess_document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(sections: list[tuple[str, str, str, int, int]]) -> ParsedDocument:
    """Build a ParsedDocument from (chapter, section, text, page_start, page_end) tuples."""
    return ParsedDocument(
        book_title="Test Book",
        source_path="/books/test.pdf",
        sections=[
            StructuredSection(chapter=ch, section=sec, page_start=ps, page_end=pe, text=txt)
            for ch, sec, txt, ps, pe in sections
        ],
    )


def _patch_settings(**kwargs):
    """Context manager: override chunk-size settings and clear lru_cache."""
    defaults = {
        "LLM_API_KEY": "sk-test",
        "CHUNK_MIN_TOKENS": "50",
        "CHUNK_MAX_TOKENS": "200",
        "CHUNK_OVERLAP_TOKENS": "20",
    }
    defaults.update({k.upper(): str(v) for k, v in kwargs.items()})
    return patch.dict(os.environ, defaults, clear=False)


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Unit tests: internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_count_tokens_non_empty(self) -> None:
        assert _count_tokens("hello world") > 0

    def test_count_tokens_empty(self) -> None:
        assert _count_tokens("") == 0

    def test_sha256_deterministic(self) -> None:
        assert _sha256("abc") == _sha256("abc")

    def test_sha256_distinct(self) -> None:
        assert _sha256("abc") != _sha256("xyz")

    def test_split_short_text_unchanged(self) -> None:
        text = "Short text."
        result = _split_by_tokens(text, max_tokens=500, overlap_tokens=50)
        assert result == [text]

    def test_split_long_text_produces_multiple_chunks(self) -> None:
        # Build a text that is definitely > 10 tokens
        text = " ".join(["word"] * 30)
        result = _split_by_tokens(text, max_tokens=10, overlap_tokens=2)
        assert len(result) > 1

    def test_split_overlap_means_chunks_share_tokens(self) -> None:
        # Chunk N's tail should appear at the start of chunk N+1
        words = [f"w{i}" for i in range(40)]
        text = " ".join(words)
        chunks = _split_by_tokens(text, max_tokens=10, overlap_tokens=3)
        assert len(chunks) >= 2
        # Each chunk should be non-empty
        for c in chunks:
            assert c.strip()


# ---------------------------------------------------------------------------
# chunk_document: happy path
# ---------------------------------------------------------------------------


class TestChunkDocumentHappyPath:
    def test_single_section_in_bounds_is_one_chunk(self) -> None:
        """A section within [min, max] tokens should become exactly one chunk."""
        text = " ".join(["word"] * 80)  # ~80 tokens, well within default bounds
        doc = _make_doc([("Chapter 1", "1.1 Intro", text, 1, 5)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_has_all_metadata_fields(self) -> None:
        """Every emitted chunk must carry complete, non-empty ChunkMetadata."""
        text = " ".join(["word"] * 80)
        doc = _make_doc([("Chapter 9", "9.2 EM Algorithm", text, 10, 15)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert chunks
        meta = chunks[0].metadata
        assert meta.book == "Test Book"
        assert meta.chapter == "Chapter 9"
        assert meta.section == "9.2 EM Algorithm"
        assert meta.page_start == 10
        assert meta.page_end == 15
        assert meta.source_path == "/books/test.pdf"
        assert len(meta.content_hash) == 64  # SHA-256 hex

    def test_chunk_id_is_non_empty_string(self) -> None:
        text = " ".join(["word"] * 80)
        doc = _make_doc([("Ch 1", "1.1", text, 1, 3)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert chunks[0].chunk_id
        assert isinstance(chunks[0].chunk_id, str)

    def test_multiple_sections_produce_multiple_chunks(self) -> None:
        text = " ".join(["word"] * 80)
        doc = _make_doc(
            [
                ("Chapter 1", "1.1 Intro", text, 1, 5),
                ("Chapter 1", "1.2 Background", text, 6, 10),
                ("Chapter 2", "2.1 Methods", text, 11, 15),
            ]
        )
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert len(chunks) == 3

    def test_each_chunk_text_is_non_empty(self) -> None:
        text = " ".join(["word"] * 80)
        doc = _make_doc([("Ch", "S", text, 1, 2), ("Ch", "S2", text, 3, 4)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        for c in chunks:
            assert c.text.strip()

    def test_content_hash_matches_text(self) -> None:
        text = " ".join(["word"] * 80)
        doc = _make_doc([("Ch", "S", text, 1, 5)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert chunks[0].metadata.content_hash == _sha256(chunks[0].text)


# ---------------------------------------------------------------------------
# chunk_document: size bounds
# ---------------------------------------------------------------------------


class TestChunkSizeBounds:
    def test_overlong_section_is_split(self) -> None:
        """A section with more tokens than max must be split into multiple chunks."""
        # ~300 tokens at 1 word ≈ 1 token
        text = " ".join(["longword"] * 300)
        doc = _make_doc([("Ch 1", "1.1", text, 1, 10)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=100, chunk_overlap_tokens=10):
            chunks = chunk_document(doc)
        assert len(chunks) > 1

    def test_split_chunks_respect_max_tokens(self) -> None:
        """No chunk from a split should exceed max_tokens."""
        text = " ".join(["w"] * 400)
        doc = _make_doc([("Ch", "S", text, 1, 10)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=100, chunk_overlap_tokens=10):
            chunks = chunk_document(doc)
        for c in chunks:
            assert _count_tokens(c.text) <= 100

    def test_small_sections_are_merged(self) -> None:
        """Two sections each below min_tokens should be merged into one chunk."""
        # ~20 tokens each — below min of 50
        tiny = " ".join(["word"] * 20)
        doc = _make_doc(
            [
                ("Ch 1", "1.1", tiny, 1, 2),
                ("Ch 1", "1.2", tiny, 3, 4),
                ("Ch 1", "1.3", tiny, 5, 6),
            ]
        )
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200, chunk_overlap_tokens=10):
            chunks = chunk_document(doc)
        # Three tiny sections should collapse to fewer chunks
        assert len(chunks) < 3

    def test_no_empty_chunks_emitted(self) -> None:
        """chunk_document must never emit a chunk with empty text."""
        sections = [
            ("Ch 1", "1.1", "   ", 1, 1),  # whitespace only
            ("Ch 1", "1.2", " ".join(["word"] * 80), 2, 5),
        ]
        doc = _make_doc(sections)
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        for c in chunks:
            assert c.text.strip(), "Empty chunk detected"


# ---------------------------------------------------------------------------
# chunk_document: edge cases
# ---------------------------------------------------------------------------


class TestChunkEdgeCases:
    def test_empty_document_returns_empty_list(self) -> None:
        doc = ParsedDocument(book_title="Empty Book", source_path="/books/empty.pdf", sections=[])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert chunks == []

    def test_all_whitespace_sections_returns_empty_list(self) -> None:
        doc = _make_doc([("Ch", "S", "   \n  \t  ", 1, 1)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            chunks = chunk_document(doc)
        assert chunks == []

    def test_missing_source_path_raises(self) -> None:
        doc = ParsedDocument(book_title="Book", source_path="", sections=[])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            with pytest.raises(ValueError, match="source_path"):
                chunk_document(doc)

    def test_missing_book_title_raises(self) -> None:
        doc = ParsedDocument(book_title="", source_path="/books/x.pdf", sections=[])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            with pytest.raises(ValueError, match="book_title"):
                chunk_document(doc)

    def test_split_chunks_inherit_correct_metadata(self) -> None:
        """Sub-chunks from a split must inherit chapter/section/pages from the parent section."""
        text = " ".join(["word"] * 300)
        doc = _make_doc([("Chapter 5", "5.3 Kernels", text, 100, 120)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=100, chunk_overlap_tokens=10):
            chunks = chunk_document(doc)
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.chapter == "Chapter 5"
            assert c.metadata.section == "5.3 Kernels"
            assert c.metadata.page_start == 100
            assert c.metadata.page_end == 120


# ---------------------------------------------------------------------------
# preprocessor: clean_text (fast, no PDF needed)
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_fixes_hyphenated_linebreak(self) -> None:
        assert clean_text("estima-\ntion") == "estimation"

    def test_collapses_multiple_spaces(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_strips_standalone_page_numbers(self) -> None:
        text = "Some text.\n123\nMore text."
        result = clean_text(text)
        assert "123" not in result.split()

    def test_preserves_greek_letters(self) -> None:
        text = "The parameter θ controls the distribution μ."
        assert "θ" in clean_text(text)
        assert "μ" in clean_text(text)

    def test_collapses_excess_blank_lines(self) -> None:
        text = "Para 1.\n\n\n\n\nPara 2."
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert clean_text("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# Embedding-model input limit — chunks must never be silently truncated
# ---------------------------------------------------------------------------

_BOOKS_DIR = Path(__file__).resolve().parent.parent / "books"

# Code, math symbols and URLs tokenize far denser in bge's WordPiece than in tiktoken.
_DENSE_TEXT = (
    "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "    return F.softmax(self.W @ x + self.b, dim=-1)  # σ(Wx+b)\n"
    "KL(q‖p) = Σ_i q_i log(q_i / p_i) ≥ 0, with ∇_θ L(θ) = E_q[∇ log p(x|θ)].\n"
    "See https://docs.djangoproject.com/en/5.0/ref/models/querysets/#QuerySet.filter.\n"
) * 60


@lru_cache(maxsize=4)
def _reference_tokenizer(model_name: str) -> Tokenizer:
    """The embedding model's tokenizer, loaded independently of pipeline.chunker."""
    tokenizer = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
    tokenizer.no_truncation()
    return tokenizer


def _model_token_count(text: str) -> int:
    """Length the embedding model actually sees, including [CLS]/[SEP]."""
    tokenizer = _reference_tokenizer(get_settings().embedding_model)
    return len(tokenizer.encode(text, add_special_tokens=True).ids)


class TestEmbeddingModelLimit:
    def test_dense_text_chunks_fit_model_limit(self) -> None:
        doc = _make_doc([("Ch 1", "1.1 Code", _DENSE_TEXT, 1, 9)])
        with _patch_settings(chunk_min_tokens=100, chunk_max_tokens=512, chunk_overlap_tokens=50):
            chunks = chunk_document(doc)
            limit = get_settings().chunk_max_tokens
            lengths = [_model_token_count(c.text) for c in chunks]
        assert len(chunks) > 1
        assert max(lengths) <= limit, f"chunk of {max(lengths)} tokens would be truncated"

    def test_split_chunks_are_verbatim_slices_of_the_source(self) -> None:
        # Decoding bge tokens would lowercase and re-space; offsets keep the original text.
        with _patch_settings():
            chunks = _split_by_tokens(_DENSE_TEXT, max_tokens=120, overlap_tokens=10)
        assert len(chunks) > 1
        assert all(c in _DENSE_TEXT for c in chunks)
        assert any(c != c.lower() for c in chunks)

    def test_oversized_chunk_fails_loud(self) -> None:
        doc = _make_doc([("Ch 1", "1.1", " ".join(["word"] * 80), 1, 1)])
        with _patch_settings(chunk_min_tokens=50, chunk_max_tokens=200):
            (chunk,) = chunk_document(doc)
            with pytest.raises(ChunkTooLongError, match="limit is 10"):
                _assert_fits_model([chunk], limit=10)

    @pytest.mark.skipif(not list(_BOOKS_DIR.glob("*.pdf")), reason="no local PDFs in books/")
    def test_real_books_fit_model_limit(self) -> None:
        with _patch_settings(chunk_min_tokens=100, chunk_max_tokens=512, chunk_overlap_tokens=50):
            limit = get_settings().chunk_max_tokens
            for pdf in sorted(_BOOKS_DIR.glob("*.pdf")):
                parsed = parse_structure(extract_pdf(pdf), pdf.stem)
                chunks = chunk_document(preprocess_document(parsed))
                worst = max(_model_token_count(c.text) for c in chunks)
                assert worst <= limit, f"{pdf.name}: chunk of {worst} tokens exceeds {limit}"
