"""Batch embedding of chunks using bge-large via sentence-transformers.

Runs on GPU if available, else CPU. Returns a new list of Chunk objects with
the embedding field populated. Model name and batch size come from Settings.
"""

import time

import torch
from app.config import get_logger, get_settings
from app.models import Chunk
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# Module-level model cache — loaded once per process
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and return the cached instance."""
    global _model
    if _model is None:
        settings = get_settings()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model '%s' on %s.", settings.embedding_model, device)
        _model = SentenceTransformer(settings.embedding_model, device=device)
    return _model


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Embed each chunk and return a new list with embeddings populated.

    This function is CPU/GPU-bound. Call it from a thread executor when used
    inside an async context so the event loop is not blocked.

    Args:
        chunks: Chunks to embed; text must be non-empty.

    Returns:
        New list of Chunk objects with embedding field populated (float list,
        L2-normalized). Order matches the input list exactly.
    """
    if not chunks:
        return []

    settings = get_settings()
    model = _get_model()
    texts = [c.text for c in chunks]

    start = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - start
    throughput = len(chunks) / elapsed if elapsed > 0 else float("inf")

    logger.info(
        "Embedded %d chunks: throughput=%.0f chunks/s elapsed_s=%.2f",
        len(chunks),
        throughput,
        elapsed,
    )

    return [
        Chunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            embedding=vector.tolist(),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
