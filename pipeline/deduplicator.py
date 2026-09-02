"""Near-duplicate chunk removal via content hashing.

Keeps the first occurrence of any chunk with a given content_hash. SHA-256
hashes are computed by the chunker and stored in ChunkMetadata, so no
re-hashing is performed here.
"""

from app.config import get_logger
from app.models import Chunk

logger = get_logger(__name__)


def deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    """Remove duplicate chunks, keeping the first occurrence of each.

    Deduplication key: content_hash in ChunkMetadata (SHA-256 of chunk text).
    This collapses repeated boilerplate (copyright pages, running headers that
    leaked into body text) without touching unique content.

    Args:
        chunks: Chunk list from the chunker, in pipeline order.

    Returns:
        Deduplicated list preserving relative order of first occurrences.
    """
    seen: set[str] = set()
    result: list[Chunk] = []
    removed = 0

    for chunk in chunks:
        h = chunk.metadata.content_hash
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        result.append(chunk)

    logger.info(
        "Deduplicated: input=%d output=%d removed=%d",
        len(chunks),
        len(result),
        removed,
    )
    return result
