"""Tests for pipeline/deduplicator.py — duplicate collapsing and order preservation.

Uses synthetic Chunk objects; no PDF, embedding model, or vector store required.
"""

from app.models import Chunk, ChunkMetadata
from pipeline.deduplicator import deduplicate


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
