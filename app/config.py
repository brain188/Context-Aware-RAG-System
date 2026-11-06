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

# LLM provider and reasoning-depth names; app.llm.base re-exports them as Provider / ThinkingLevel.
LLMProviderName = Literal["gemini", "groq", "anthropic", "ollama"]
ThinkingLevelName = Literal["minimal", "low", "medium", "high"]


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

    # --- LLM (roles: generation = answers, fast = rewrite/grade, judge = eval) ---
    # Primary provider for generation and fast. Its API key is required at startup.
    llm_provider: LLMProviderName = "gemini"
    llm_generation_model: str = "gemini-3.8-flash"
    llm_fast_model: str = "gemini-3.5-flash-lite"
    # Reasoning depth per role (Gemini thinking_level / Groq reasoning_effort /
    # Claude effort); None = model default.
    llm_generation_thinking: ThinkingLevelName | None = "low"
    llm_fast_thinking: ThinkingLevelName | None = "minimal"
    # Judge: a different model family than the generator reduces self-preference bias.
    # None = the primary provider. Its key is checked only when an eval uses --judge.
    llm_judge_provider: LLMProviderName | None = "groq"
    llm_judge_model: str = "openai/gpt-oss-120b"
    llm_judge_thinking: ThinkingLevelName | None = "low"
    # Used by generation and fast when the primary is still failing after retries
    # (429, timeouts, 5xx). None = no fallback.
    llm_fallback_provider: LLMProviderName | None = None
    llm_fallback_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    llm_max_tokens: int = 2048
    # None = the provider's default (recommended: Gemini 3 and Claude 5 models expect it).
    llm_temperature: float | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2  # retries per call on 429/timeouts/5xx, with backoff
    # Client-side limits per model of the primary provider (0 = off); match your quota.
    llm_rpm_limit: int = 10
    llm_tpm_limit: int = 250_000
    # The same, per model of any other provider (fallback, judge). Defaults: Groq free tier.
    llm_fallback_rpm_limit: int = 30
    llm_fallback_tpm_limit: int = 8_000
    # Short auxiliary calls (follow-up rewrite, retrieval grading).
    llm_rewrite_max_tokens: int = 200
    llm_grader_max_tokens: int = 300
    # Template set under app/prompts/templates/; bump to roll out (and evaluate) new prompts.
    # v2: grader rates each excerpt; adds the search-query rewrite used after "ambiguous".
    prompt_version: str = "v2"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600  # conversation session expiry
    conversation_max_turns: int = 6  # prior turns kept for follow-up rewriting

    # --- Retrieval ---
    # Branches fused per query. dense / sparse run one branch alone, to measure what
    # each contributes (`run_eval --retrieval-only --set retrieval_mode=sparse`).
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    retrieval_top_k: int = 20
    # Reciprocal-rank-fusion constant; 60 is the value from the original RRF paper.
    rrf_k: int = 60
    rerank_top_k: int = 5
    # Top rerank score below which retrieval grades "none" without an LLM call.
    # bge-reranker scores are sigmoid relevance in [0, 1]. Short questions score
    # low even on the right chunk ("What is a migration?" got 0.0056), so the
    # floor only skips near-certain misses; the LLM grader decides the rest.
    # 0.005 measured on 40 hand labels: no false refusals, no false accepts
    # (0.05 refused one answerable question).
    grade_min_rerank_score: float = 0.005
    # Search-query rewrites + re-retrievals after an "ambiguous" grade (0 = off).
    # Each costs two fast-model calls (rewrite, grade).
    grader_max_retries: int = 1

    # --- Semantic cache ---
    semantic_cache_enabled: bool = True
    # Qdrant collection of cached question embeddings; answers live in Redis.
    semantic_cache_collection: str = "semantic_cache"
    # Cosine similarity of standalone-question embeddings for a cache *candidate*; the fast
    # model then confirms that one answer serves both questions (services/cache_matcher.py).
    # Similarity alone cannot separate paraphrases from near-identical different questions
    # (bge-base: "field-level" vs "form-level validation" = 0.97). Tune both with
    # `python -m evaluation.cache_threshold --verify`. Measured on 50 pairs with the check:
    # 0.80 → 68% of paraphrases hit, 0 of 25 near-identical different questions did
    # (similarity alone: 44% false hits at 0.80, and 0.97 needed for none, with 4% hits).
    semantic_cache_threshold: float = 0.80
    # At or above this, a candidate is a repeat and hits without the LLM check.
    semantic_cache_exact_threshold: float = 0.99
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

    @field_validator(
        "gemini_api_key",
        "groq_api_key",
        "anthropic_api_key",
        "llm_judge_provider",
        "llm_fallback_provider",
        "llm_temperature",
        "llm_generation_thinking",
        "llm_fast_thinking",
        "llm_judge_thinking",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        """Treat blank optional values in .env (`KEY=`) as unset."""
        return None if v == "" else v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _blank_provider_is_default(cls, v: object) -> object:
        """A blank LLM_PROVIDER means the default provider."""
        return "gemini" if v == "" else v

    @model_validator(mode="after")
    def _llm_providers_have_keys(self) -> "Settings":
        """Fail at startup, not on the first query, if the primary or fallback lacks its key."""
        keys = {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "anthropic": self.anthropic_api_key,
        }
        for setting, provider in (
            ("LLM_PROVIDER", self.llm_provider),
            ("LLM_FALLBACK_PROVIDER", self.llm_fallback_provider),
        ):
            if provider in keys and not keys[provider]:
                raise ValueError(
                    f"{setting}={provider} needs {provider.upper()}_API_KEY; set it in infra/.env"
                )
        return self

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
