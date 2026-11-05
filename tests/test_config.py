"""Tests for app/config.py — Settings validation and logging factory."""

import logging
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings, get_logger, get_settings


class TestSettings:
    def setup_method(self) -> None:
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        get_settings.cache_clear()

    def test_loads_with_the_primary_providers_key(self) -> None:
        """Settings instantiate when the primary provider's key (Gemini by default) is set."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            s = Settings(_env_file=None)
        assert s.gemini_api_key == "test-key"
        assert (s.llm_provider, s.llm_fallback_provider) == ("gemini", None)
        assert s.environment == "dev"
        assert s.retrieval_top_k == 20
        assert s.rerank_top_k == 5

    @pytest.mark.parametrize(
        "values, message",
        [
            ({}, "LLM_PROVIDER=gemini needs GEMINI_API_KEY"),
            ({"gemini_api_key": ""}, "LLM_PROVIDER=gemini needs GEMINI_API_KEY"),
            (
                {"gemini_api_key": "k", "llm_fallback_provider": "groq"},
                "LLM_FALLBACK_PROVIDER=groq needs GROQ_API_KEY",
            ),
            (
                {"gemini_api_key": "k", "llm_fallback_provider": "anthropic"},
                "LLM_FALLBACK_PROVIDER=anthropic needs ANTHROPIC_API_KEY",
            ),
        ],
    )
    def test_missing_provider_key_raises(self, values: dict, message: str) -> None:
        """A provider without its key must fail at startup, not on the first query.

        _env_file=None and a cleared environment keep the developer's real keys out.
        """
        env = {k: v for k, v in os.environ.items() if not k.upper().endswith("_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError, match=message):
                Settings(_env_file=None, **values)

    def test_ollama_needs_no_key_and_blank_fallback_means_none(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.upper().endswith("_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None, llm_provider="ollama", llm_fallback_provider="")
        assert (s.llm_provider, s.llm_fallback_provider) == ("ollama", None)

    def test_chunk_max_above_model_limit_is_rejected(self) -> None:
        """Chunks longer than the embedding model reads would be silently truncated."""
        with pytest.raises(ValidationError, match="EMBEDDING_MAX_TOKENS"):
            Settings(
                _env_file=None,
                gemini_api_key="test-key",
                chunk_max_tokens=600,
                embedding_max_tokens=512,
            )

    def test_blank_qdrant_api_key_becomes_none(self) -> None:
        """A blank QDRANT_API_KEY must normalise to None.

        An empty string makes QdrantClient send an auth header and warn about
        insecure transport; local Qdrant expects no key at all.
        """
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "QDRANT_API_KEY": ""}, clear=False
        ):
            s = Settings(_env_file=None)
        assert s.qdrant_api_key is None

    def test_qdrant_api_key_preserved_when_set(self) -> None:
        """A real QDRANT_API_KEY must survive the blank-to-None validator."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "QDRANT_API_KEY": "secret"}, clear=False
        ):
            s = Settings(_env_file=None)
        assert s.qdrant_api_key == "secret"

    def test_embedding_dimension_matches_default_model(self) -> None:
        """Default dimension must match the default model's width (bge-base = 768).

        A mismatch is silent until Qdrant rejects every upsert.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            s = Settings(_env_file=None)
        assert s.embedding_model == "BAAI/bge-base-en-v1.5"
        assert s.embedding_dimension == 768

    def test_env_overrides_defaults(self) -> None:
        """Explicit environment variables should override all defaults."""
        overrides = {
            "GEMINI_API_KEY": "test-key",
            "RETRIEVAL_TOP_K": "15",
            "RERANK_TOP_K": "3",
            "LOG_LEVEL": "DEBUG",
            "ENVIRONMENT": "prod",
        }
        with patch.dict(os.environ, overrides, clear=False):
            s = Settings(_env_file=None)
        assert s.retrieval_top_k == 15
        assert s.rerank_top_k == 3
        assert s.log_level == "DEBUG"
        assert s.environment == "prod"

    def test_get_settings_is_cached(self) -> None:
        """get_settings() must return the same instance on repeated calls."""
        # Env vars beat infra/.env, so the developer's provider settings cannot interfere.
        env = {"GEMINI_API_KEY": "test-key", "LLM_PROVIDER": "gemini", "LLM_FALLBACK_PROVIDER": ""}
        with patch.dict(os.environ, env, clear=False):
            a = get_settings()
            b = get_settings()
        assert a is b

    def test_invalid_environment_value_raises(self) -> None:
        """environment must be 'dev' or 'prod'; any other value raises ValidationError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "ENVIRONMENT": "staging"}, clear=False):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)


class TestGetLogger:
    def test_returns_logger_instance(self) -> None:
        """get_logger should return a standard logging.Logger."""
        logger = get_logger("test.config.basic")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self) -> None:
        """Logger name must match the argument passed to get_logger."""
        logger = get_logger("test.config.naming")
        assert logger.name == "test.config.naming"

    def test_logger_has_handler(self) -> None:
        """Logger must have at least one handler so output is not silently dropped."""
        logger = get_logger("test.config.handler")
        assert len(logger.handlers) >= 1

    def test_logger_does_not_propagate(self) -> None:
        """Logger must not propagate to the root logger (avoids duplicate output)."""
        logger = get_logger("test.config.propagate")
        assert not logger.propagate

    def test_get_logger_idempotent(self) -> None:
        """Calling get_logger twice with the same name returns the same object."""
        a = get_logger("test.config.idempotent")
        b = get_logger("test.config.idempotent")
        assert a is b


class TestEnvExample:
    """infra/.env.example is the human-readable config contract; keep it in step with Settings."""

    _PATH = Path(__file__).resolve().parent.parent / "infra" / ".env.example"

    def _documented_keys(self) -> set[str]:
        keys = set()
        for line in self._PATH.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^#?\s*([A-Z][A-Z0-9_]*)=", line)
            if match:
                keys.add(match.group(1).lower())
        return keys

    def test_documents_every_setting(self) -> None:
        missing = set(Settings.model_fields) - self._documented_keys()
        assert not missing, f"Add to infra/.env.example: {sorted(k.upper() for k in missing)}"

    def test_documents_nothing_settings_ignores(self) -> None:
        unknown = self._documented_keys() - set(Settings.model_fields)
        assert not unknown, f"Remove from infra/.env.example: {sorted(k.upper() for k in unknown)}"

    def test_loads_as_valid_settings(self) -> None:
        Settings(_env_file=self._PATH)
