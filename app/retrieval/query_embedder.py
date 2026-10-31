"""Query-side embedding with the same model family the pipeline used for chunks.

The app does not import pipeline/ (they deploy independently), so the model is
loaded here separately. sentence-transformers is imported lazily so this module
stays cheap to import and tests can inject a fake QueryEmbedder.
"""

import threading
from typing import TYPE_CHECKING, Protocol

from app.config import Settings, get_logger
from app.retrieval.errors import EmbeddingError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class QueryEmbedder(Protocol):
    """Anything that turns a query string into a normalized embedding vector."""

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerQueryEmbedder:
    """QueryEmbedder backed by sentence-transformers; the model loads on first use."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._instruction = settings.embedding_query_instruction
        self._dimension = settings.embedding_dimension
        self._model: SentenceTransformer | None = None
        # Executor threads may race to load the model on the first concurrent queries.
        self._lock = threading.Lock()

    def _get_model(self) -> "SentenceTransformer":
        with self._lock:
            if self._model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading query embedding model '%s' on %s.", self._model_name, device)
                self._model = SentenceTransformer(self._model_name, device=device)
            return self._model

    def embed_query(self, text: str) -> list[float]:
        """Embed one query (instruction-prefixed, L2-normalized).

        CPU/GPU-bound: call from an executor inside async code.

        Raises:
            EmbeddingError: if encoding fails or the vector width mismatches the collection.
        """
        try:
            vector = self._get_model().encode(
                self._instruction + text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc
        if len(vector) != self._dimension:
            raise EmbeddingError(
                f"Model '{self._model_name}' produced {len(vector)}-d vectors but "
                f"EMBEDDING_DIMENSION is {self._dimension}"
            )
        return vector.tolist()
