"""Build the LLM client for each role from settings.

Roles: `generation` (cited answers), `fast` (query rewrite, retrieval grading,
routing) and `judge` (evaluation). The primary provider serves generation and
fast; the judge may use another provider (a different model family than the
generator reduces self-preference bias). With LLM_FALLBACK_PROVIDER set,
generation and fast fall back to it after their retries run out; the judge
never falls back, so a run's scores all come from one model.

Each provider+model gets one rate limiter, shared by every role that uses it:
LLM_RPM_LIMIT / LLM_TPM_LIMIT for the primary provider, LLM_FALLBACK_RPM_LIMIT /
LLM_FALLBACK_TPM_LIMIT for any other (fallback, judge).
"""

from app.config import Settings, get_logger
from app.llm.anthropic_client import AnthropicLLM
from app.llm.base import LLMClient, LLMProviderError, Provider, Role, ThinkingLevel
from app.llm.gemini_client import GeminiLLM
from app.llm.groq_client import GroqLLM
from app.llm.ollama_client import OllamaLLM
from app.llm.rate_limiter import RateLimiter
from app.llm.retry import FallbackLLM, ResilientLLM

logger = get_logger(__name__)

_KEY_SETTING: dict[Provider, str | None] = {
    "gemini": "gemini_api_key",
    "groq": "groq_api_key",
    "anthropic": "anthropic_api_key",
    "ollama": None,
}


def missing_key(provider: Provider, settings: Settings) -> str | None:
    """The env var *provider* needs but lacks, or None when it is usable."""
    key = _KEY_SETTING[provider]
    return key.upper() if key and not getattr(settings, key) else None


class LLMFactory:
    """Creates role clients and owns them (call aclose() at shutdown)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._limiters: dict[tuple[str, str], RateLimiter] = {}
        self._clients: list[LLMClient] = []

    def _limiter(self, provider: Provider, model: str) -> RateLimiter:
        """Shared limiter per provider+model: primary limits, or the fallback limits."""
        key = (provider, model)
        if key not in self._limiters:
            s = self._settings
            primary = provider == s.llm_provider
            self._limiters[key] = RateLimiter(
                f"{provider}:{model}",
                rpm=s.llm_rpm_limit if primary else s.llm_fallback_rpm_limit,
                tpm=s.llm_tpm_limit if primary else s.llm_fallback_tpm_limit,
            )
        return self._limiters[key]

    def _client(self, provider: Provider, model: str, thinking: ThinkingLevel | None) -> LLMClient:
        """A rate-limited, retrying client for one provider+model.

        Raises:
            LLMProviderError: if the provider's API key is not configured.
        """
        missing = missing_key(provider, self._settings)
        if missing:
            raise LLMProviderError(f"{missing} is not set; {provider} needs it (infra/.env)")
        classes = {
            "gemini": GeminiLLM,
            "groq": GroqLLM,
            "anthropic": AnthropicLLM,
            "ollama": OllamaLLM,
        }
        inner = classes[provider](self._settings, model, thinking)
        client = ResilientLLM(
            inner, self._limiter(provider, model), max_retries=self._settings.llm_max_retries
        )
        self._clients.append(client)
        return client

    def get(self, role: Role, model: str | None = None) -> LLMClient:
        """The client for *role*; *model* overrides the configured model for that role."""
        s = self._settings
        if role == "judge":
            provider = s.llm_judge_provider or s.llm_provider
            return self._client(provider, model or s.llm_judge_model, s.llm_judge_thinking)

        thinking = s.llm_generation_thinking if role == "generation" else s.llm_fast_thinking
        configured = s.llm_generation_model if role == "generation" else s.llm_fast_model
        primary = self._client(s.llm_provider, model or configured, thinking)
        if s.llm_fallback_provider is None:
            return primary
        fallback = self._client(s.llm_fallback_provider, s.llm_fallback_model, thinking)
        logger.info(
            "LLM role %s: %s:%s, falling back to %s:%s.",
            role,
            s.llm_provider,
            primary.model,
            s.llm_fallback_provider,
            fallback.model,
        )
        return FallbackLLM(primary, fallback)

    async def aclose(self) -> None:
        """Close every client this factory created."""
        for client in self._clients:
            await client.aclose()
        self._clients.clear()
