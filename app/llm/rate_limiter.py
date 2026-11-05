"""Client-side request and token limits per model, so free-tier quotas are never hit.

Callers wait (queue) rather than fail. Limits are per minute, over a rolling
60-second window: a call may start when fewer than *rpm* calls and fewer than
*tpm* estimated tokens were sent in the last minute. Token counts are estimates
made before the call (prompt characters / 4 + the output cap).
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from app.config import get_logger

logger = get_logger(__name__)

_WINDOW_S = 60.0


def estimate_tokens(*texts: str, max_output_tokens: int = 0) -> int:
    """Rough token count for rate limiting (about four characters per token)."""
    return sum(len(t) for t in texts) // 4 + max_output_tokens


class RateLimiter:
    """Rolling one-minute request (RPM) and token (TPM) limits. 0 disables a limit."""

    def __init__(
        self,
        name: str,
        rpm: int,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._name = name
        self._rpm = rpm
        self._tpm = tpm
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int]] = deque()  # (time, estimated tokens)
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._rpm > 0 or self._tpm > 0

    def _wait_needed(self, tokens: int, now: float) -> float:
        """Seconds until a call of *tokens* fits both limits (0 if it fits now)."""
        while self._sent and self._sent[0][0] <= now - _WINDOW_S:
            self._sent.popleft()
        waits = [0.0]
        if self._rpm > 0 and len(self._sent) >= self._rpm:
            waits.append(self._sent[-self._rpm][0] + _WINDOW_S - now)
        if self._tpm > 0:
            budget = self._tpm - tokens
            used = sum(t for _, t in self._sent)
            # Drop the oldest calls until the rest plus this one fit.
            for sent_at, sent_tokens in self._sent:
                if used <= budget:
                    break
                used -= sent_tokens
                waits.append(sent_at + _WINDOW_S - now)
        return max(waits)

    async def acquire(self, tokens: int) -> None:
        """Wait until a call of about *tokens* tokens may start, then record it."""
        if not self.enabled:
            return
        if self._tpm > 0:
            tokens = min(tokens, self._tpm)  # a single huge call must still get through
        async with self._lock:
            while (wait := self._wait_needed(tokens, self._clock())) > 0:
                logger.info(
                    "Rate limit (%s): waiting %.1fs before the next call.", self._name, wait
                )
                await self._sleep(wait)
            self._sent.append((self._clock(), tokens))
