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

from pydantic import field_validator
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

    # --- Embedding model ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = 32
    # Must match embedding_model's output width: bge-base=768, bge-large=1024.
    # A mismatch is silent until Qdrant rejects the upsert.
    embedding_dimension: int = 768

    # --- Reranker model ---
    reranker_model: str = "BAAI/bge-reranker-large"

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str  # REQUIRED — no default; absence raises ValidationError at startup
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600

    # --- Retrieval ---
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    semantic_cache_threshold: float = 0.92

    # --- Chunking ---
    chunk_min_tokens: int = 100
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- BM25 keyword index ---
    bm25_corpus_path: str = "data/bm25_corpus.jsonl"

    @field_validator("qdrant_api_key", mode="after")
    @classmethod
    def _blank_api_key_is_none(cls, v: str | None) -> str | None:
        """Treat a blank QDRANT_API_KEY as absent.

        An empty string is truthy enough for QdrantClient to send an auth
        header and warn about insecure transport; local Qdrant wants no key.
        """
        return v or None


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
