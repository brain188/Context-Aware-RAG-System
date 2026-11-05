"""LLMClient over a local Ollama server (`/api/chat`): the private mode.

Nothing leaves the machine. Talks plain HTTP with httpx; Ollama's JSON-schema
`format` gives structured output. Thinking levels are ignored.
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
    LLMResult,
    LLMTimeoutError,
    T,
    ThinkingLevel,
    log_result,
    parse_structured,
)

logger = get_logger(__name__)


class OllamaLLM:
    """LLMClient for one model served by Ollama."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        thinking: ThinkingLevel | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url, timeout=settings.llm_timeout_seconds
        )
        self._model = model
        self._temperature = settings.llm_temperature

    @property
    def model(self) -> str:
        return self._model

    def _body(self, system: str, prompt: str, max_tokens: int, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {"num_predict": max(max_tokens, MIN_OUTPUT_TOKENS)}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
            "options": options,
        }

    def _error(self, exc: Exception) -> LLMError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"Ollama did not answer in time ({self._model})")
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 404:
                return LLMProviderError(
                    f"Ollama has no model '{self._model}'; run `ollama pull {self._model}`"
                )
            return LLMProviderError(
                f"Ollama returned HTTP {exc.response.status_code}",
                retryable=exc.response.status_code >= 500,
            )
        return LLMProviderError(
            f"Could not reach Ollama at {self._base_url} ({exc}); is `ollama serve` running?",
            retryable=True,
        )

    async def _chat(self, body: dict[str, Any], request_id: str) -> LLMResult:
        start = time.perf_counter()
        try:
            response = await self._client.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc
        data = response.json()
        text = (data.get("message") or {}).get("content", "").strip()
        if not text:
            raise LLMOutputError("LLM returned an empty reply")
        result = LLMResult(
            text=text,
            model=self._model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=data.get("done_reason"),
        )
        log_result("generate", result, request_id)
        return result

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        """Return the reply for one prompt.

        Raises:
            LLMError: if Ollama is unreachable, lacks the model, or replies empty.
        """
        return await self._chat(self._body(system, prompt, max_tokens, stream=False), request_id)

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        """Yield reply text deltas from Ollama's newline-delimited JSON stream."""
        start = time.perf_counter()
        final: dict[str, Any] = {}
        body = self._body(system, prompt, max_tokens, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMProviderError(f"Ollama stream failed: {data['error']}")
                    delta = (data.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if data.get("done"):
                        final = data
        except httpx.HTTPError as exc:
            raise self._error(exc) from exc
        log_result(
            "stream",
            LLMResult(
                text="",
                model=self._model,
                input_tokens=final.get("prompt_eval_count"),
                output_tokens=final.get("eval_count"),
                latency_ms=(time.perf_counter() - start) * 1000,
                finish_reason=final.get("done_reason"),
            ),
            request_id,
        )

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        """Return the reply parsed into *schema*, using Ollama's JSON-schema `format`."""
        body = self._body(system, prompt, max_tokens, stream=False)
        body["format"] = schema.model_json_schema()
        result = await self._chat(body, request_id)
        return parse_structured(result.text, schema)

    async def aclose(self) -> None:
        await self._client.aclose()
