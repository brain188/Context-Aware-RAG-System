"""LLMClient over the Gemini API (`google-genai`). The only module importing the SDK.

Uses the SDK's async client. Gemini 3 models choose their own sampling (Google
recommends leaving temperature at its default), so temperature is sent only
when LLM_TEMPERATURE is set; the thinking level maps to `thinking_level`.
The SDK's own retries are off (its default); app/llm/retry.py retries.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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
)

try:  # The SDK uses aiohttp for async calls when it is installed, httpx otherwise.
    from aiohttp import ClientError as _AiohttpError
except ImportError:  # pragma: no cover - depends on the environment
    _AiohttpError = httpx.TransportError  # type: ignore[assignment,misc]

logger = get_logger(__name__)

_THINKING_LEVEL: dict[ThinkingLevel, types.ThinkingLevel] = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}
# Finish reasons meaning the provider stopped the reply for policy reasons.
_BLOCKED_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


def _retry_delay(details: Any) -> float | None:
    """Find google.rpc.RetryInfo's retryDelay (e.g. "17s") anywhere in an error body."""
    if isinstance(details, dict):
        delay = details.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
        values: Any = details.values()
    elif isinstance(details, list):
        values = details
    else:
        return None
    for value in values:
        found = _retry_delay(value)
        if found is not None:
            return found
    return None


def map_sdk_error(exc: Exception, model: str) -> LLMError:
    """Turn an SDK or transport exception into the matching typed LLMError."""
    if isinstance(exc, genai_errors.APIError):
        message = exc.message or str(exc)
        if exc.code == 429:
            return LLMRateLimitError(
                f"Gemini rate limit or quota hit (429): {message}", _retry_delay(exc.details)
            )
        if exc.code in (401, 403) or "API key" in message:
            return LLMProviderError(f"Gemini rejected GEMINI_API_KEY ({exc.code}): {message}")
        if exc.code == 404:
            return LLMProviderError(
                f"Gemini does not know the model '{model}' (404); "
                "list valid ids with `python -m app.llm.list_models`"
            )
        if exc.code in (408, 504):
            return LLMTimeoutError(f"Gemini timed out ({exc.code}): {message}")
        retryable = exc.code >= 500 or exc.code == 499
        return LLMProviderError(f"Gemini returned HTTP {exc.code}: {message}", retryable)
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return LLMTimeoutError(f"Gemini did not answer in time ({model})")
    return LLMProviderError(f"Could not reach Gemini: {exc}", retryable=True)


_SDK_ERRORS = (genai_errors.APIError, TimeoutError, httpx.TransportError, _AiohttpError)


def _finish_reason(response: types.GenerateContentResponse) -> str | None:
    if response.candidates and response.candidates[0].finish_reason is not None:
        return str(response.candidates[0].finish_reason.value)
    return None


def _check_blocked(response: types.GenerateContentResponse) -> None:
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        raise LLMSafetyBlockError(f"Gemini blocked the prompt ({feedback.block_reason.value})")
    reason = _finish_reason(response)
    if reason in _BLOCKED_FINISH:
        raise LLMSafetyBlockError(f"Gemini stopped the reply (finish_reason={reason})")


def _usage(response: types.GenerateContentResponse | None) -> tuple[int | None, int | None]:
    usage = response.usage_metadata if response is not None else None
    if usage is None:
        return None, None
    # Thinking tokens are billed as output.
    output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return usage.prompt_token_count, output


class GeminiLLM:
    """LLMClient for one Gemini model."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        thinking: ThinkingLevel | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)),
        )
        self._model = model
        self._temperature = settings.llm_temperature
        self._thinking = thinking

    @property
    def model(self) -> str:
        return self._model

    def _config(self, system: str, max_tokens: int, **extra: Any) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max(max_tokens, MIN_OUTPUT_TOKENS),
            temperature=self._temperature,
            thinking_config=(
                types.ThinkingConfig(thinking_level=_THINKING_LEVEL[self._thinking])
                if self._thinking is not None
                else None
            ),
            **extra,
        )

    async def _generate(
        self, system: str, prompt: str, max_tokens: int, request_id: str, **extra: Any
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=prompt, config=self._config(system, max_tokens, **extra)
            )
        except _SDK_ERRORS as exc:
            raise map_sdk_error(exc, self._model) from exc

        _check_blocked(response)
        text = (response.text or "").strip()
        if not text:
            raise LLMOutputError(
                f"LLM returned an empty reply (finish_reason={_finish_reason(response)})"
            )
        input_tokens, output_tokens = _usage(response)
        result = LLMResult(
            text=text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=_finish_reason(response),
        )
        log_result("generate", result, request_id)
        return result

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        """Return the reply for one prompt.

        Raises:
            LLMError: on API failure, a safety block, or an empty reply.
        """
        return await self._generate(system, prompt, max_tokens, request_id)

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        """Yield reply text deltas.

        Raises:
            LLMError: on API failure (before or mid-stream) or a safety block.
        """
        start = time.perf_counter()
        last: types.GenerateContentResponse | None = None
        try:
            chunks = await self._client.aio.models.generate_content_stream(
                model=self._model, contents=prompt, config=self._config(system, max_tokens)
            )
            async for chunk in chunks:
                last = chunk
                _check_blocked(chunk)
                if chunk.text:
                    yield chunk.text
        except _SDK_ERRORS as exc:
            raise map_sdk_error(exc, self._model) from exc

        input_tokens, output_tokens = _usage(last)
        finish = _finish_reason(last) if last is not None else None
        if finish == "MAX_TOKENS":
            logger.warning(
                "LLM reply truncated at max_tokens; raise LLM_MAX_TOKENS if this recurs.",
                extra={"request_id": request_id},
            )
        latency_ms = (time.perf_counter() - start) * 1000
        log_result(
            "stream",
            LLMResult("", self._model, input_tokens, output_tokens, latency_ms, finish),
            request_id,
        )

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        """Return the reply parsed into *schema*, using Gemini's JSON-schema response mode."""
        result = await self._generate(
            system,
            prompt,
            max_tokens,
            request_id,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
        )
        return parse_structured(result.text, schema)

    async def aclose(self) -> None:
        await self._client.aio.aclose()
