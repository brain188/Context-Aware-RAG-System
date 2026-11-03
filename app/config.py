"""Application configuration and shared logging factory.

All runtime configuration lives in Settings. Call get_settings() to access it
(singleton via lru_cache). Call get_logger(name) at module level to obtain a
structured, request-id-aware logger for any module.
"""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, resolved from this file so config loads regardless of working directory.
_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All application configuration, loaded from environment variables.

    Required fields (no default) raise ValidationError at first get_settings()
    call when missing — startup fails loud, never mid-request.
    """

    model_config = SettingsConfigDict(
        # infra/.env is the canonical location; a root .env overrides it for local runs.
        env_file=(_ROOT / "infra" / ".env", _ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # --- Vector store (Qdrant) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"
    qdrant_api_key: str | None = None
    # Client default is 5s — too tight for collection creation on a cold server
    # and for large upsert batches.
    qdrant_timeout_seconds: int = 60
    # Points per upsert request. Each point is ~20 KB (768 floats as JSON + text), so
    # small batches keep each request well inside the timeout on a slow uplink.
    qdrant_upsert_batch_size: int = 16
    # Named vectors in the collection; pipeline writes and app queries both.
    # Changing either requires recreating the collection and re-ingesting.
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "bm25"

    # --- Embedding model ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = 32
    # Must match embedding_model's output width: bge-base=768, bge-large=1024.
    # A mismatch is silent until Qdrant rejects the upsert.
    embedding_dimension: int = 768
    # Longest input the model reads, in its own tokens including [CLS]/[SEP];
    # anything longer is silently truncated (512 for all bge v1.5 models).
    embedding_max_tokens: int = 512
    # bge v1.5 models retrieve better when queries (not passages) carry this prefix.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- Reranker model ---
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_batch_size: int = 16

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str  # REQUIRED — no default; absence raises ValidationError at startup
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2  # SDK retries 429/5xx/connection errors with backoff
    # Short auxiliary calls (follow-up rewrite, retrieval grading).
    llm_rewrite_max_tokens: int = 200
    llm_grader_max_tokens: int = 300
    # Template set under app/prompts/templates/; bump to roll out (and evaluate) new prompts.
    prompt_version: str = "v1"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600  # conversation session expiry
    conversation_max_turns: int = 6  # prior turns kept for follow-up rewriting

    # --- Retrieval ---
    retrieval_top_k: int = 20
    # Reciprocal-rank-fusion constant; 60 is the value from the original RRF paper.
    rrf_k: int = 60
    rerank_top_k: int = 5
    # Top rerank score below which retrieval grades "none" without an LLM call.
    # bge-reranker scores are sigmoid relevance in [0, 1]; tune on the eval set.
    grade_min_rerank_score: float = 0.05

    # --- Semantic cache ---
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92  # cosine similarity of query embeddings
    semantic_cache_ttl_seconds: int = 86400
    semantic_cache_max_entries: int = 500  # per index version + filter scope

    # --- Guardrails ---
    input_min_chars: int = 3

    # --- Query log (JSONL, rotated) ---
    query_log_path: Path = _ROOT / "data" / "query_log.jsonl"
    query_log_max_bytes: int = 10_000_000
    query_log_backup_count: int = 5

    # --- API runtime ---
    # POST /api/ingest only reads PDFs inside this folder.
    ingest_books_dir: Path = _ROOT / "books"
    warm_models_on_startup: bool = True
    health_timeout_seconds: float = 5.0

    # --- Chunking (counted with the embedding model's tokenizer) ---
    chunk_min_tokens: int = 100
    chunk_max_tokens: int = 512  # includes [CLS]/[SEP]; must be <= embedding_max_tokens
    chunk_overlap_tokens: int = 50

    # --- BM25 sparse vectors (term-frequency half; Qdrant applies IDF at query time) ---
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # Typical chunk length in words, for BM25 length normalization.
    bm25_avg_doc_length: float = 256.0

    @field_validator("qdrant_api_key", mode="after")
    @classmethod
    def _blank_api_key_is_none(cls, v: str | None) -> str | None:
        """Treat a blank QDRANT_API_KEY as absent.

        An empty string is truthy enough for QdrantClient to send an auth
        header and warn about insecure transport; local Qdrant wants no key.
        """
        return v or None

    @model_validator(mode="after")
    def _chunks_fit_embedding_model(self) -> "Settings":
        """Reject chunk sizes the embedding model would silently truncate."""
        if self.chunk_max_tokens > self.embedding_max_tokens:
            raise ValueError(
                f"CHUNK_MAX_TOKENS ({self.chunk_max_tokens}) exceeds EMBEDDING_MAX_TOKENS "
                f"({self.embedding_max_tokens}); longer chunks would be truncated when embedded"
            )
        return self


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s " "request_id=%(request_id)s %(message)s"


class _RequestIdFilter(logging.Filter):
    """Inject request_id into every log record, defaulting to '-' when absent."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger for *name*.

    Reads LOG_LEVEL directly from the environment so it works before
    get_settings() has been called (avoids bootstrapping issues at import time).
    Every module should call this once at module level:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler.addFilter(_RequestIdFilter())
        logger.addHandler(handler)
        logger.propagate = False
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Settings accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance, validated once from the environment.

    Raises:
        pydantic.ValidationError: if any required environment variable is missing.
    """
    return Settings()
