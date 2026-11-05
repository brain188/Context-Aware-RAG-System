"""The provider-agnostic LLM contract: protocol, result type, and typed errors.

Services depend only on LLMClient and these errors. Each provider module
(gemini_client, groq_client, anthropic_client, ollama_client) is the only place
its API is called, and maps that API's failures onto the errors below, so retry,
fallback and HTTP mapping work the same for every provider.
"""

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import LLMProviderName, ThinkingLevelName, get_logger

logger = get_logger("app.llm")

Role = Literal["generation", "fast", "judge"]
Provider = LLMProviderName
# How much a model may reason before answering. Mapped per provider: Gemini 3
# thinking_level, Groq reasoning_effort, Claude output_config.effort; ignored by Ollama.
ThinkingLevel = ThinkingLevelName

# On thinking models the reasoning shares the output-token budget, so a small
# cap (e.g. 200 for a rewrite) can be spent before any text appears. Providers
# raise the cap to at least this; prompts still keep replies short.
MIN_OUTPUT_TOKENS = 1024

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM call failed, was refused, or returned nothing usable."""


class LLMRateLimitError(LLMError):
    """The provider throttled the request (429). *retry_after* is in seconds, if given."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """The provider did not answer within LLM_TIMEOUT_SECONDS."""


class LLMProviderError(LLMError):
    """The provider failed. *retryable* is True for transient failures (5xx, network)."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMSafetyBlockError(LLMError):
    """The provider's safety system blocked the prompt or the reply."""


class LLMOutputError(LLMError):
    """The model answered, but not usably: an empty reply or JSON that misses the schema."""


def is_transient(exc: BaseException) -> bool:
    """True for failures worth retrying or falling back from."""
    return isinstance(exc, (LLMRateLimitError, LLMTimeoutError)) or (
        isinstance(exc, LLMProviderError) and exc.retryable
    )


@dataclass(frozen=True)
class LLMResult:
    """One completed generation, with the usage the cost accounting needs."""

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    finish_reason: str | None


def log_result(kind: str, result: LLMResult, request_id: str) -> None:
    """One INFO line per provider call: model, tokens, stop reason, latency."""
    logger.info(
        "LLM %s: model=%s in_tokens=%s out_tokens=%s stop=%s latency_ms=%.0f",
        kind,
        result.model,
        result.input_tokens,
        result.output_tokens,
        result.finish_reason,
        result.latency_ms,
        extra={"request_id": request_id},
    )


class LLMClient(Protocol):
    """Anything that completes, streams, or fills a schema from a single-turn prompt.

    Tool calling (`call_with_tools`) is added with the agent layer in TASKLIST Phase 2.
    """

    @property
    def model(self) -> str: ...

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        """Return the full reply.

        Raises:
            LLMError: or a subclass, on any failure or an empty reply.
        """
        ...

    def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        """Yield reply text deltas as they are generated.

        Raises:
            LLMError: or a subclass, before or during the stream.
        """
        ...

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        """Return the reply parsed into *schema*.

        Raises:
            LLMError: on failure, or if the reply does not match *schema*.
        """
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


# ---------------------------------------------------------------------------
# Structured output for providers without a native JSON-schema mode
# ---------------------------------------------------------------------------

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def schema_instruction(system: str, schema: type[BaseModel]) -> str:
    """*system* plus an instruction to reply with JSON matching *schema*."""
    return (
        f"{system}\n\nReply with only a JSON object that matches this JSON Schema, "
        f"with no other text:\n{json.dumps(schema.model_json_schema())}"
    )


def parse_structured(text: str, schema: type[T]) -> T:
    """Parse the first JSON object in *text* (code fences tolerated) into *schema*.

    Raises:
        LLMOutputError: if there is no JSON object or it does not match *schema*.
    """
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise LLMOutputError(f"Structured reply contained no JSON object: {text[:200]!r}")
    try:
        return schema.model_validate_json(match.group(0))
    except ValidationError as exc:
        raise LLMOutputError(f"Structured reply did not match {schema.__name__}: {exc}") from exc
