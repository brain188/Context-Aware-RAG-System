"""Tests for app/config.py — Settings validation and logging factory."""

import logging
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings, get_logger, get_settings


class TestSettings:
    def setup_method(self) -> None:
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        get_settings.cache_clear()

    def test_loads_with_required_env_present(self) -> None:
        """Settings should instantiate when LLM_API_KEY is provided."""
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-test-key"}, clear=False):
            s = Settings()
        assert s.llm_api_key == "sk-test-key"
        assert s.environment == "dev"
        assert s.retrieval_top_k == 20
        assert s.rerank_top_k == 5

    def test_missing_llm_api_key_raises(self) -> None:
        """Missing LLM_API_KEY must raise ValidationError, not silently default.

        _env_file=None disables dotenv loading so the developer's real infra/.env
        cannot mask the failure this test exists to catch.
        """
        env = {k: v for k, v in os.environ.items() if k.upper() != "LLM_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)

    def test_chunk_max_above_model_limit_is_rejected(self) -> None:
        """Chunks longer than the embedding model reads would be silently truncated."""
        with pytest.raises(ValidationError, match="EMBEDDING_MAX_TOKENS"):
            Settings(
                _env_file=None,
                llm_api_key="sk-test",
                chunk_max_tokens=600,
                embedding_max_tokens=512,
            )

    def test_blank_qdrant_api_key_becomes_none(self) -> None:
        """A blank QDRANT_API_KEY must normalise to None.

        An empty string makes QdrantClient send an auth header and warn about
        insecure transport; local Qdrant expects no key at all.
        """
        with patch.dict(
            os.environ, {"LLM_API_KEY": "sk-test-key", "QDRANT_API_KEY": ""}, clear=False
        ):
            s = Settings(_env_file=None)
        assert s.qdrant_api_key is None

    def test_qdrant_api_key_preserved_when_set(self) -> None:
        """A real QDRANT_API_KEY must survive the blank-to-None validator."""
        with patch.dict(
            os.environ, {"LLM_API_KEY": "sk-test-key", "QDRANT_API_KEY": "secret"}, clear=False
        ):
            s = Settings(_env_file=None)
        assert s.qdrant_api_key == "secret"

    def test_embedding_dimension_matches_default_model(self) -> None:
        """Default dimension must match the default model's width (bge-base = 768).

        A mismatch is silent until Qdrant rejects every upsert.
        """
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-test-key"}, clear=False):
            s = Settings(_env_file=None)
        assert s.embedding_model == "BAAI/bge-base-en-v1.5"
        assert s.embedding_dimension == 768

    def test_env_overrides_defaults(self) -> None:
        """Explicit environment variables should override all defaults."""
        overrides = {
            "LLM_API_KEY": "sk-override",
            "RETRIEVAL_TOP_K": "15",
            "RERANK_TOP_K": "3",
            "LOG_LEVEL": "DEBUG",
            "ENVIRONMENT": "prod",
        }
        with patch.dict(os.environ, overrides, clear=False):
            s = Settings()
        assert s.retrieval_top_k == 15
        assert s.rerank_top_k == 3
        assert s.log_level == "DEBUG"
        assert s.environment == "prod"

    def test_get_settings_is_cached(self) -> None:
        """get_settings() must return the same instance on repeated calls."""
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-cache-test"}, clear=False):
            a = get_settings()
            b = get_settings()
        assert a is b

    def test_invalid_environment_value_raises(self) -> None:
        """environment must be 'dev' or 'prod'; any other value raises ValidationError."""
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-x", "ENVIRONMENT": "staging"}, clear=False):
            with pytest.raises(ValidationError):
                Settings()


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
