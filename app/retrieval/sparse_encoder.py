"""BM25 sparse vectors for Qdrant keyword search.

BM25 = term-frequency saturation x inverse document frequency. Documents carry
the TF half (computed here, at ingest); the collection's sparse vector uses
Modifier.IDF so Qdrant applies the IDF half at query time over the live corpus.
Queries therefore only need weight 1.0 per distinct term.

Shared by pipeline (documents) and app (queries): both sides must tokenize and
hash identically, so changing anything here requires re-ingesting.
"""

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

from app.config import Settings

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase *text* and split it into word tokens."""
    return _TOKEN_RE.findall(text.lower())


def token_index(token: str) -> int:
    """Stable 32-bit sparse dimension for *token* (CRC32; same on every platform)."""
    return zlib.crc32(token.encode("utf-8"))


def _to_sparse(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[i] for i in indices])


class BM25SparseEncoder:
    """Encode chunks and queries as BM25 sparse vectors."""

    def __init__(self, settings: Settings) -> None:
        self._k1 = settings.bm25_k1
        self._b = settings.bm25_b
        self._avg_doc_length = settings.bm25_avg_doc_length

    def encode_document(self, text: str) -> SparseVector:
        """BM25 term-frequency weight per distinct term of a chunk.

        Hash collisions merge counts into one dimension; with a 32-bit space
        and a book-sized vocabulary this is rare and only slightly blurs scores.
        """
        tokens = tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])
        length_norm = 1 - self._b + self._b * len(tokens) / self._avg_doc_length
        counts = Counter(token_index(t) for t in tokens)
        return _to_sparse(
            {idx: tf * (self._k1 + 1) / (tf + self._k1 * length_norm) for idx, tf in counts.items()}
        )

    def encode_query(self, text: str) -> SparseVector:
        """Weight 1.0 per distinct query term; Qdrant supplies the IDF."""
        return _to_sparse({token_index(t): 1.0 for t in tokenize(text)})
