"""Typed exceptions raised by the retrieval layer.

Routes map RetrievalError subclasses to HTTP responses; callers should never
need to catch qdrant-client or model-library exceptions directly.
"""


class RetrievalError(Exception):
    """Base class for all retrieval failures."""


class VectorStoreError(RetrievalError):
    """Qdrant was unreachable, rejected a request, or returned a malformed point."""


class EmbeddingError(RetrievalError):
    """The query embedder failed or produced a vector of the wrong dimension."""


class RerankerError(RetrievalError):
    """The cross-encoder failed or returned an unexpected number of scores."""
