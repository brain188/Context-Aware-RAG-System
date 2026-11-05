"""Rate limiting + retry around one provider client, and fallback to another.

ResilientLLM retries transient failures (429, timeouts, 5xx, network) with
exponential backoff and jitter, honouring the provider's retry-after when it
fits within the backoff ceiling; a longer wait (e.g. a daily quota) is not
retried, so FallbackLLM can switch providers instead. Streams are retried
only until the first token arrives: after that a retry would repeat text.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_logger
from app.llm.base import LLMClient, LLMRateLimitError, LLMResult, T, is_transient
from app.llm.rate_limiter import RateLimiter, estimate_tokens

logger = get_logger(__name__)

# Longest wait between attempts; a longer retry-after is treated as "not now".
MAX_RETRY_WAIT_S = 30.0

_backoff = wait_exponential_jitter(initial=1, max=MAX_RETRY_WAIT_S, jitter=1)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, LLMRateLimitError) and (exc.retry_after or 0) > MAX_RETRY_WAIT_S:
        return False
    return is_transient(exc)


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, LLMRateLimitError) and exc.retry_after:
        return exc.retry_after
    return _backoff(state)


class ResilientLLM:
    """An LLMClient with a rate limiter and bounded retries in front of *inner*."""

    def __init__(
        self,
        inner: LLMClient,
        limiter: RateLimiter,
        max_retries: int,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._inner = inner
        self._limiter = limiter
        self._attempts = max_retries + 1
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._inner.model

    def _retrying(self, request_id: str) -> AsyncRetrying:
        def _log_retry(state: RetryCallState) -> None:
            exc = state.outcome.exception() if state.outcome else None
            logger.warning(
                "LLM call to %s failed (%s); retry %d/%d in %.1fs.",
                self.model,
                exc,
                state.attempt_number,
                self._attempts - 1,
                state.next_action.sleep if state.next_action else 0.0,
                extra={"request_id": request_id},
            )

        kwargs: dict[str, Any] = {"sleep": self._sleep} if self._sleep else {}
        return AsyncRetrying(
            stop=stop_after_attempt(self._attempts),
            wait=_wait,
            retry=retry_if_exception(_should_retry),
            before_sleep=_log_retry,
            reraise=True,
            **kwargs,
        )

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        async for attempt in self._retrying(request_id):
            with attempt:
                await self._limiter.acquire(
                    estimate_tokens(system, prompt, max_output_tokens=max_tokens)
                )
                return await self._inner.generate(
                    system=system, prompt=prompt, max_tokens=max_tokens, request_id=request_id
                )
        raise AssertionError("unreachable: tenacity re-raises the last error")

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        async for attempt in self._retrying(request_id):
            with attempt:
                await self._limiter.acquire(
                    estimate_tokens(system, prompt, max_output_tokens=max_tokens)
                )
                return await self._inner.generate_structured(
                    system=system,
                    prompt=prompt,
                    schema=schema,
                    max_tokens=max_tokens,
                    request_id=request_id,
                )
        raise AssertionError("unreachable: tenacity re-raises the last error")

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        first: str | None = None
        async for attempt in self._retrying(request_id):
            with attempt:
                await self._limiter.acquire(
                    estimate_tokens(system, prompt, max_output_tokens=max_tokens)
                )
                deltas = aiter(
                    self._inner.stream(
                        system=system, prompt=prompt, max_tokens=max_tokens, request_id=request_id
                    )
                )
                try:
                    first = await anext(deltas)
                except StopAsyncIteration:
                    return
        if first is None:
            return
        yield first
        async for delta in deltas:
            yield delta

    async def aclose(self) -> None:
        await self._inner.aclose()


class FallbackLLM:
    """Use *primary*; on a transient failure that survived its retries, use *fallback*.

    Configuration errors (bad key, unknown model) and safety blocks are not
    hidden behind the fallback: they surface so they get fixed.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def model(self) -> str:
        return self._primary.model

    def _warn(self, exc: Exception, request_id: str) -> None:
        logger.warning(
            "LLM %s unavailable (%s); falling back to %s.",
            self._primary.model,
            exc,
            self._fallback.model,
            extra={"request_id": request_id},
        )

    async def generate(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> LLMResult:
        kwargs = {"system": system, "prompt": prompt, "max_tokens": max_tokens}
        try:
            return await self._primary.generate(**kwargs, request_id=request_id)
        except Exception as exc:
            if not is_transient(exc):
                raise
            self._warn(exc, request_id)
            return await self._fallback.generate(**kwargs, request_id=request_id)

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int, request_id: str
    ) -> T:
        kwargs = {"system": system, "prompt": prompt, "schema": schema, "max_tokens": max_tokens}
        try:
            return await self._primary.generate_structured(**kwargs, request_id=request_id)
        except Exception as exc:
            if not is_transient(exc):
                raise
            self._warn(exc, request_id)
            return await self._fallback.generate_structured(**kwargs, request_id=request_id)

    async def stream(
        self, *, system: str, prompt: str, max_tokens: int, request_id: str
    ) -> AsyncIterator[str]:
        kwargs = {"system": system, "prompt": prompt, "max_tokens": max_tokens}
        started = False
        try:
            async for delta in self._primary.stream(**kwargs, request_id=request_id):
                started = True
                yield delta
            return
        except Exception as exc:
            if started or not is_transient(exc):
                raise
            self._warn(exc, request_id)
        async for delta in self._fallback.stream(**kwargs, request_id=request_id):
            yield delta

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._fallback.aclose()
