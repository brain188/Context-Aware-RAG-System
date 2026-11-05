"""LLMClient over Groq's OpenAI-compatible Chat Completions API (httpx, no SDK).

Groq serves open-weight models (GPT-OSS, Qwen) fast, with a free tier, so it is
the default fallback and judge: a different model family from Gemini. Streams
are server-sent events; reasoning comes back in a separate `reasoning` field,
so only `content` is read. Structured output uses `json_schema` in best-effort
mode (strict mode needs every field required), validated with the schema.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

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

logger = get_logger(__name__)

# Groq's reasoning models take low/medium/high; "minimal" has no equivalent.
_EFFORT: dict[ThinkingLevel, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _error_detail(response: httpx.Response) -> tuple[str, str | None]:
    """Groq's `error.message` and `error.code`, or the start of the body."""
    try:
        error = response.json()["error"]
        return str(error.get("message")), error.get("code")
    except (ValueError, KeyError, TypeError, AttributeError):
        return response.text[:200], None


class GroqLLM:
    """LLMClient for one model served by Groq."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        thinking: ThinkingLevel | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=settings.groq_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            timeout=settings.llm_timeout_seconds,
        )
        self._model = model
        self._thinking = thinking
        self._temperature = settings.llm_temperature

    @property
    def model(self) -> str:
        return self._model

    def _body(self, system: str, prompt: str, max_tokens: int, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max(max_tokens, MIN_OUTPUT_TOKENS),
            "stream": stream,
        }
        if self._thinking is not None:
            body["reasoning_effort"] = _EFFORT[self._thinking]
        if self._temperature is not None:
            body["temperature"] = self._temperature
        return body

    def _status_error(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        detail, code = _error_detail(response)
        if status == 429:
            return LLMRateLimitError(
                f"Groq rate limit for {self._model}: {detail}", _retry_after(response)
            )
        if status == 401:
            return LLMProviderError("Groq rejected GROQ_API_KEY (401); check infra/.env")
        if status == 404:
            return LLMProviderError(
                f"Groq has no model '{self._model}'; see https://console.groq.com/docs/models"
            )
        if status == 413:
            return LLMProviderError(
                f"Request too large for {self._model} on your Groq tier: {detail}"
            )
        if code == "json_validate_failed":
            return LLMOutputError(f"Groq structured reply did not match the schema: {detail}")
        # 498 = Groq capacity exceeded; 5xx = server side. Both pass.
        return LLMProviderError(
            f"Groq returned HTTP {status}: {detail}", retryable=status >= 500 or status == 498
        )

    def _error(self, exc: httpx.HTTPError) -> LLMError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"Groq did not answer in time ({self._model})")
        return LLMProviderError(f"Could not reach Groq ({exc})", retryable=True)

    def _check_finish(self, finish_reason: str | None) -> None:
        if finish_reason == "content_filter":
            raise LLMSafetyBlockError(f"Groq blocked the reply ({self._model})")

    async def _complete(self, body: dict[str, Any], request_id: str) -> LLMResult:
        start = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc
        if response.is_error:
            raise self._status_error(response)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        self._check_finish(finish_reason)
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMOutputError(f"LLM returned an empty reply (stop={finish_reason})")
        usage = data.get("usage") or {}
        result = LLMResult(
            text=text,
            model=self._model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=finish_reason,
        )
        log_result("generate", result, request_id)
        return result

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        """Return the reply for one prompt.

        Raises:
            LLMError: on throttling, timeouts, HTTP errors, a block, or an empty reply.
        """
        return await self._complete(self._body(system, prompt, max_tokens, False), request_id)

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        """Yield reply text deltas from Groq's server-sent events."""
        start = time.perf_counter()
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        body = self._body(system, prompt, max_tokens, True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.is_error:
                    await response.aread()
                    raise self._status_error(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if chunk.get("error"):
                        raise LLMProviderError(
                            f"Groq stream failed: {chunk['error']}", retryable=True
                        )
                    usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        self._check_finish(finish_reason)
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc
        log_result(
            "stream",
            LLMResult(
                text="",
                model=self._model,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_ms=(time.perf_counter() - start) * 1000,
                finish_reason=finish_reason,
            ),
            request_id,
        )

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        """Return the reply parsed into *schema*, using Groq's `json_schema` response format."""
        body = self._body(system, prompt, max_tokens, False)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": False,
            },
        }
        result = await self._complete(body, request_id)
        return parse_structured(result.text, schema)

    async def aclose(self) -> None:
        await self._client.aclose()
