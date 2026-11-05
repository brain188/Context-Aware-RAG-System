"""LLMClient over the Anthropic Messages API (Claude). The only module importing `anthropic`.

Claude 5.x models reject non-default sampling parameters and think adaptively
by default, so temperature is sent only when LLM_TEMPERATURE is set, and the
thinking level maps to `output_config.effort`. Retries live in app/llm/retry.py,
so the SDK's own retries are off.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from app.config import Settings, get_logger
from app.llm.base import (
    MIN_OUTPUT_TOKENS,
    LLMError,
    LLMOutputError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResult,
    LLMSafetyBlockError,
    LLMTimeoutError,
    T,
    ThinkingLevel,
    log_result,
    parse_structured,
    schema_instruction,
)

logger = get_logger(__name__)

_EFFORT: dict[ThinkingLevel, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None


def map_sdk_error(exc: anthropic.AnthropicError, model: str) -> LLMError:
    """Turn an SDK exception into the matching typed LLMError (most specific first)."""
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(f"Anthropic did not answer in time ({model})")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMProviderError(f"Could not reach Anthropic: {exc}", retryable=True)
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMProviderError("Anthropic rejected ANTHROPIC_API_KEY (401); check infra/.env")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMProviderError(f"Anthropic does not know the model '{model}' (404)")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError("Anthropic rate limit hit (429)", _retry_after(exc))
    if isinstance(exc, anthropic.APIStatusError):
        # 529 overloaded and other 5xx are transient; 4xx means the request is wrong.
        retryable = exc.status_code >= 500
        return LLMProviderError(
            f"Anthropic returned HTTP {exc.status_code}: {exc.message}", retryable
        )
    return LLMProviderError(f"Anthropic call failed: {exc}")


class AnthropicLLM:
    """LLMClient for one Claude model."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        thinking: ThinkingLevel | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._model = model
        self._temperature = settings.llm_temperature
        self._thinking = thinking

    @property
    def model(self) -> str:
        return self._model

    def _request(self, system: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max(max_tokens, MIN_OUTPUT_TOKENS),
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._thinking is not None:
            # Not a typed parameter in the pinned SDK; sent as-is in the request body.
            request["extra_body"] = {"output_config": {"effort": _EFFORT[self._thinking]}}
        return request

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        """Return the reply for one prompt.

        Raises:
            LLMError: on API failure, refusal, or an empty reply.
        """
        start = time.perf_counter()
        try:
            message = await self._client.messages.create(
                **self._request(system, prompt, max_tokens)
            )
        except anthropic.AnthropicError as exc:
            raise map_sdk_error(exc, self._model) from exc

        self._check_stop_reason(message.stop_reason, request_id)
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if not text:
            raise LLMOutputError("LLM returned an empty reply")
        result = LLMResult(
            text=text,
            model=self._model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=message.stop_reason,
        )
        log_result("generate", result, request_id)
        return result

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
                **self._request(system, prompt, max_tokens)
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                message = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise map_sdk_error(exc, self._model) from exc

        self._check_stop_reason(message.stop_reason, request_id)
        log_result(
            "stream",
            LLMResult(
                text="",
                model=self._model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                latency_ms=(time.perf_counter() - start) * 1000,
                finish_reason=message.stop_reason,
            ),
            request_id,
        )

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        """Return the reply parsed into *schema* (instructed JSON, then validated)."""
        result = await self.generate(
            system=schema_instruction(system, schema),
            prompt=prompt,
            max_tokens=max_tokens,
            request_id=request_id,
        )
        return parse_structured(result.text, schema)

    async def aclose(self) -> None:
        await self._client.close()

    @staticmethod
    def _check_stop_reason(stop_reason: str | None, request_id: str) -> None:
        if stop_reason == "refusal":
            raise LLMSafetyBlockError("The model declined to answer (stop_reason=refusal)")
        if stop_reason == "max_tokens":
            logger.warning(
                "LLM reply truncated at max_tokens; raise LLM_MAX_TOKENS if this recurs.",
                extra={"request_id": request_id},
            )
