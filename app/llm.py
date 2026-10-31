"""LLM access behind a small interface so services can be tested with a fake.

AnthropicLLM is the only module that imports the provider SDK. It maps SDK
exceptions to LLMError so callers never handle provider-specific types.
"""

import time
from collections.abc import AsyncIterator
from typing import Protocol

import anthropic
from anthropic import AsyncAnthropic

from app.config import Settings, get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """The LLM call failed, was refused, or returned nothing usable."""


class LLMClient(Protocol):
    """Anything that completes or streams a single-turn prompt."""

    @property
    def model(self) -> str: ...

    async def complete(self, *, system: str, prompt: str, max_tokens: int, request_id: str) -> str:
        """Return the full reply text."""
        ...

    def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        """Yield reply text deltas as they are generated."""
        ...


def _describe_sdk_error(exc: anthropic.AnthropicError) -> str:
    """Turn SDK exceptions into actionable messages (most specific first)."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "LLM provider rejected LLM_API_KEY (401); check infra/.env"
    if isinstance(exc, anthropic.NotFoundError):
        return "LLM provider does not know this model (404); check LLM_MODEL"
    if isinstance(exc, anthropic.RateLimitError):
        return "LLM provider rate limit hit (429) after retries"
    if isinstance(exc, anthropic.APIStatusError):
        return f"LLM provider returned HTTP {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return f"Could not reach the LLM provider: {exc}"
    return f"LLM call failed: {exc}"


class AnthropicLLM:
    """LLMClient backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings, client: AsyncAnthropic | None = None) -> None:
        self._client = client or AsyncAnthropic(
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, *, system: str, prompt: str, max_tokens: int, request_id: str) -> str:
        """Return the reply text for one prompt.

        Raises:
            LLMError: on API failure, refusal, or an empty reply.
        """
        start = time.perf_counter()
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=self._temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(_describe_sdk_error(exc)) from exc

        self._check_stop_reason(message.stop_reason, request_id)
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if not text:
            raise LLMError("LLM returned an empty reply")
        logger.info(
            "LLM complete: model=%s in_tokens=%d out_tokens=%d latency_ms=%.0f",
            self._model,
            message.usage.input_tokens,
            message.usage.output_tokens,
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )
        return text

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        """Yield reply text deltas.

        Raises:
            LLMError: on API failure (before or mid-stream) or refusal.
        """
        start = time.perf_counter()
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=self._temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                message = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise LLMError(_describe_sdk_error(exc)) from exc

        self._check_stop_reason(message.stop_reason, request_id)
        logger.info(
            "LLM stream: model=%s in_tokens=%d out_tokens=%d stop=%s latency_ms=%.0f",
            self._model,
            message.usage.input_tokens,
            message.usage.output_tokens,
            message.stop_reason,
            (time.perf_counter() - start) * 1000,
            extra={"request_id": request_id},
        )

    @staticmethod
    def _check_stop_reason(stop_reason: str | None, request_id: str) -> None:
        if stop_reason == "refusal":
            raise LLMError("LLM declined to answer (stop_reason=refusal)")
        if stop_reason == "max_tokens":
            logger.warning(
                "LLM reply truncated at max_tokens; raise LLM_MAX_TOKENS if this recurs.",
                extra={"request_id": request_id},
            )
