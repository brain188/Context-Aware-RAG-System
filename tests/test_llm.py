"""Tests for the provider-agnostic LLM layer (app/llm/).

Provider SDKs are faked at their client boundary (no network, no keys): the
Gemini and Anthropic clients get stub SDK objects, Groq and Ollama get an
httpx MockTransport. Time is simulated for the rate limiter and retries.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.config import Settings
from app.errors import error_response
from app.llm import (
    LLMError,
    LLMFactory,
    LLMOutputError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResult,
    LLMSafetyBlockError,
    LLMTimeoutError,
)
from app.llm.anthropic_client import AnthropicLLM
from app.llm.base import MIN_OUTPUT_TOKENS, is_transient, parse_structured
from app.llm.gemini_client import GeminiLLM
from app.llm.groq_client import GroqLLM
from app.llm.ollama_client import OllamaLLM
from app.llm.rate_limiter import RateLimiter
from app.llm.retry import MAX_RETRY_WAIT_S, FallbackLLM, ResilientLLM
from conftest import make_settings

CALL = {"system": "s", "prompt": "p", "max_tokens": 100, "request_id": "r"}


class Verdict(BaseModel):
    grade: str


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------


class TestStructuredParsing:
    def test_parses_json_inside_code_fences(self) -> None:
        assert parse_structured('```json\n{"grade": "correct"}\n```', Verdict).grade == "correct"

    @pytest.mark.parametrize("text", ["no json here", '{"other": 1}'])
    def test_unusable_reply_is_an_output_error(self, text: str) -> None:
        with pytest.raises(LLMOutputError):
            parse_structured(text, Verdict)

    def test_transient_errors(self) -> None:
        assert is_transient(LLMRateLimitError("429"))
        assert is_transient(LLMTimeoutError("slow"))
        assert is_transient(LLMProviderError("503", retryable=True))
        assert not is_transient(LLMProviderError("bad key"))
        assert not is_transient(LLMSafetyBlockError("blocked"))
        assert not is_transient(LLMOutputError("empty"))

    def test_safety_block_maps_to_422_with_a_clean_message(self) -> None:
        status, body = error_response(LLMSafetyBlockError("Gemini blocked the prompt"), "r", False)
        assert (status, body.error, body.detail) == (
            422,
            "llm_safety_block",
            "Gemini blocked the prompt",
        )


# ---------------------------------------------------------------------------
# Rate limiter (simulated clock)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_requests_beyond_rpm_wait_for_the_window(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter("m", rpm=2, tpm=0, clock=clock, sleep=clock.sleep)
        for _ in range(3):
            await limiter.acquire(10)
        assert clock.now == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_tokens_beyond_tpm_wait_for_the_window(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter("m", rpm=0, tpm=1000, clock=clock, sleep=clock.sleep)
        await limiter.acquire(800)
        clock.now = 10.0
        await limiter.acquire(800)
        assert clock.now == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_disabled_limiter_never_waits(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter("m", rpm=0, tpm=0, clock=clock, sleep=clock.sleep)
        for _ in range(100):
            await limiter.acquire(10_000)
        assert clock.now == 0.0

    @pytest.mark.asyncio
    async def test_a_burst_of_30_calls_never_exceeds_the_quota(self) -> None:
        """TASKLIST 1.1: a burst of 30 eval calls completes without a single 429."""
        clock = FakeClock()
        limiter = RateLimiter("gemini:m", rpm=10, tpm=0, clock=clock, sleep=clock.sleep)
        provider = QuotaProvider(clock, rpm=10)
        llm = ResilientLLM(provider, limiter, max_retries=0, sleep=clock.sleep)
        for _ in range(30):
            await llm.generate(**CALL)
        assert provider.calls == 30 and provider.rejected == 0


class QuotaProvider:
    """A provider that answers 429 when more than *rpm* calls arrive in 60 s."""

    model = "quota-model"

    def __init__(self, clock: FakeClock, rpm: int) -> None:
        self._clock, self._rpm = clock, rpm
        self.sent: list[float] = []
        self.calls = self.rejected = 0

    async def generate(self, **kwargs: Any) -> LLMResult:
        now = self._clock()
        self.sent = [t for t in self.sent if t > now - 60] + [now]
        if len(self.sent) > self._rpm:
            self.rejected += 1
            raise LLMRateLimitError("429")
        self.calls += 1
        return LLMResult("ok", self.model, 1, 1, 0.0, "stop")


# ---------------------------------------------------------------------------
# Retry and fallback
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Raises the scripted errors in order, then answers."""

    def __init__(self, *errors: Exception, model: str = "primary", text: str = "ok") -> None:
        self.errors = list(errors)
        self.model = model
        self.text = text
        self.calls = 0

    def _next_error(self) -> Exception | None:
        self.calls += 1
        return self.errors.pop(0) if self.errors else None

    async def generate(self, **kwargs: Any) -> LLMResult:
        if error := self._next_error():
            raise error
        return LLMResult(self.text, self.model, 1, 1, 0.0, "stop")

    async def generate_structured(self, *, schema: type[BaseModel], **kwargs: Any) -> BaseModel:
        if error := self._next_error():
            raise error
        return schema(grade=self.text)

    async def stream(self, **kwargs: Any) -> AsyncIterator[str]:
        if error := self._next_error():
            raise error
        for word in self.text.split():
            yield word

    async def aclose(self) -> None:
        pass


def _resilient(inner: Any, retries: int = 2) -> tuple[ResilientLLM, list[float]]:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    limiter = RateLimiter("x", rpm=0, tpm=0)
    return ResilientLLM(inner, limiter, max_retries=retries, sleep=record), waits


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    return [delta async for delta in stream]


class TestResilientLLM:
    @pytest.mark.asyncio
    async def test_transient_errors_are_retried(self) -> None:
        inner = ScriptedLLM(LLMProviderError("503", retryable=True), LLMTimeoutError("slow"))
        llm, waits = _resilient(inner)
        assert (await llm.generate(**CALL)).text == "ok"
        assert inner.calls == 3 and len(waits) == 2

    @pytest.mark.asyncio
    async def test_configuration_errors_are_not_retried(self) -> None:
        inner = ScriptedLLM(LLMProviderError("bad key"))
        llm, _ = _resilient(inner)
        with pytest.raises(LLMProviderError, match="bad key"):
            await llm.generate(**CALL)
        assert inner.calls == 1

    @pytest.mark.asyncio
    async def test_retry_after_is_honoured(self) -> None:
        llm, waits = _resilient(ScriptedLLM(LLMRateLimitError("429", retry_after=7.0)))
        await llm.generate(**CALL)
        assert waits == [7.0]

    @pytest.mark.asyncio
    async def test_a_long_retry_after_is_not_waited_for(self) -> None:
        """A daily quota (retry in hours) should reach the fallback, not block the request."""
        inner = ScriptedLLM(LLMRateLimitError("429", retry_after=MAX_RETRY_WAIT_S * 10))
        llm, waits = _resilient(inner)
        with pytest.raises(LLMRateLimitError):
            await llm.generate(**CALL)
        assert waits == [] and inner.calls == 1

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self) -> None:
        inner = ScriptedLLM(*[LLMTimeoutError("slow")] * 5)
        llm, _ = _resilient(inner, retries=2)
        with pytest.raises(LLMTimeoutError):
            await llm.generate(**CALL)
        assert inner.calls == 3

    @pytest.mark.asyncio
    async def test_structured_calls_are_retried_too(self) -> None:
        llm, _ = _resilient(ScriptedLLM(LLMTimeoutError("slow"), text="correct"))
        assert (await llm.generate_structured(schema=Verdict, **CALL)).grade == "correct"

    @pytest.mark.asyncio
    async def test_stream_is_retried_before_the_first_token(self) -> None:
        inner = ScriptedLLM(LLMProviderError("503", retryable=True), text="cited answer")
        llm, _ = _resilient(inner)
        assert await _collect(llm.stream(**CALL)) == ["cited", "answer"]

    @pytest.mark.asyncio
    async def test_stream_failing_after_the_first_token_is_not_retried(self) -> None:
        class MidStreamFailure(ScriptedLLM):
            async def stream(self, **kwargs: Any) -> AsyncIterator[str]:
                self.calls += 1
                yield "partial"
                raise LLMProviderError("connection reset", retryable=True)

        inner = MidStreamFailure()
        llm, _ = _resilient(inner)
        received: list[str] = []
        with pytest.raises(LLMProviderError):
            async for delta in llm.stream(**CALL):
                received.append(delta)
        assert received == ["partial"] and inner.calls == 1


class TestFallbackLLM:
    @pytest.mark.asyncio
    async def test_transient_failure_falls_back(self) -> None:
        fallback = ScriptedLLM(model="claude", text="from claude")
        llm = FallbackLLM(ScriptedLLM(LLMRateLimitError("429")), fallback)
        result = await llm.generate(**CALL)
        assert (result.text, result.model) == ("from claude", "claude")

    @pytest.mark.parametrize(
        "error", [LLMProviderError("bad key"), LLMSafetyBlockError("blocked")], ids=str
    )
    @pytest.mark.asyncio
    async def test_config_errors_and_blocks_do_not_fall_back(self, error: LLMError) -> None:
        fallback = ScriptedLLM(model="claude")
        with pytest.raises(type(error)):
            await FallbackLLM(ScriptedLLM(error), fallback).generate(**CALL)
        assert fallback.calls == 0

    @pytest.mark.asyncio
    async def test_stream_falls_back_only_before_the_first_token(self) -> None:
        fallback = ScriptedLLM(model="claude", text="fallback answer")
        llm = FallbackLLM(ScriptedLLM(LLMTimeoutError("slow")), fallback)
        assert await _collect(llm.stream(**CALL)) == ["fallback", "answer"]

    @pytest.mark.asyncio
    async def test_structured_falls_back(self) -> None:
        llm = FallbackLLM(ScriptedLLM(LLMTimeoutError("slow")), ScriptedLLM(text="ambiguous"))
        assert (await llm.generate_structured(schema=Verdict, **CALL)).grade == "ambiguous"


# ---------------------------------------------------------------------------
# Factory: roles, fallback, keys
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values = {"groq_api_key": "gsk-test", "anthropic_api_key": "sk-ant-test", **overrides}
    return make_settings(tmp_path, **values)


class TestLLMFactory:
    def test_roles_resolve_to_their_configured_models(self, tmp_path: Path) -> None:
        factory = LLMFactory(_settings(tmp_path, llm_judge_model="judge-model"))
        assert factory.get("generation").model == "gemini-3.8-flash"
        assert factory.get("fast").model == "gemini-3.5-flash-lite"
        assert factory.get("judge").model == "judge-model"
        assert factory.get("judge", model="other-judge").model == "other-judge"

    def test_judge_uses_its_own_provider(self, tmp_path: Path) -> None:
        judge = LLMFactory(_settings(tmp_path)).get("judge")
        assert isinstance(judge, ResilientLLM) and isinstance(judge._inner, GroqLLM)

    def test_claude_is_still_a_provider(self, tmp_path: Path) -> None:
        judge = LLMFactory(_settings(tmp_path, llm_judge_provider="anthropic")).get("judge")
        assert isinstance(judge._inner, AnthropicLLM)

    def test_fallback_wraps_generation_and_fast_but_not_the_judge(self, tmp_path: Path) -> None:
        factory = LLMFactory(_settings(tmp_path, llm_fallback_provider="groq"))
        generation = factory.get("generation")
        assert isinstance(generation, FallbackLLM)
        assert generation._fallback.model == "openai/gpt-oss-120b"
        assert isinstance(factory.get("fast"), FallbackLLM)
        assert not isinstance(factory.get("judge"), FallbackLLM)

    def test_missing_judge_key_fails_when_the_judge_is_built(self, tmp_path: Path) -> None:
        factory = LLMFactory(make_settings(tmp_path, groq_api_key=None))
        factory.get("generation")  # the primary is fine without the judge's key
        with pytest.raises(LLMProviderError, match="GROQ_API_KEY"):
            factory.get("judge")

    def test_one_limiter_per_model_with_primary_and_fallback_limits(self, tmp_path: Path) -> None:
        settings = _settings(
            tmp_path,
            llm_fast_model="gemini-3.8-flash",
            llm_rpm_limit=10,
            llm_fallback_rpm_limit=30,
        )
        factory = LLMFactory(settings)
        generation, fast, judge = (factory.get(r) for r in ("generation", "fast", "judge"))
        assert generation._limiter is fast._limiter
        assert (generation._limiter._rpm, judge._limiter._rpm) == (10, 30)

    @pytest.mark.asyncio
    async def test_aclose_closes_every_client(self, tmp_path: Path) -> None:
        factory = LLMFactory(_settings(tmp_path))
        factory.get("generation")
        await factory.aclose()


# ---------------------------------------------------------------------------
# Anthropic client (SDK faked)
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _message(text: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _anthropic(tmp_path: Path, result: Any, **overrides: Any) -> tuple[AnthropicLLM, _FakeMessages]:
    messages = _FakeMessages(result)
    client = SimpleNamespace(messages=messages)
    settings = _settings(tmp_path, **overrides)
    llm = AnthropicLLM(settings, "claude-sonnet-5-5", "low", client=client)  # type: ignore[arg-type]
    return llm, messages


_REQUEST = httpx.Request("POST", "https://api.anthropic.test")


class TestAnthropicLLM:
    @pytest.mark.asyncio
    async def test_returns_only_the_text_blocks(self, tmp_path: Path) -> None:
        llm, _ = _anthropic(tmp_path, _message(" correct "))
        result = await llm.generate(**CALL)
        assert (result.text, result.input_tokens, result.output_tokens) == ("correct", 10, 5)

    @pytest.mark.asyncio
    async def test_request_suits_claude_5_models(self, tmp_path: Path) -> None:
        llm, messages = _anthropic(tmp_path, _message("ok"))
        await llm.generate(**CALL)
        (request,) = messages.requests
        assert "temperature" not in request
        assert request["extra_body"] == {"output_config": {"effort": "low"}}
        assert request["max_tokens"] == MIN_OUTPUT_TOKENS

    @pytest.mark.asyncio
    async def test_temperature_is_sent_only_when_configured(self, tmp_path: Path) -> None:
        llm, messages = _anthropic(tmp_path, _message("ok"), llm_temperature=0.3)
        await llm.generate(**CALL)
        assert messages.requests[0]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_refusal_is_a_safety_block(self, tmp_path: Path) -> None:
        llm, _ = _anthropic(tmp_path, _message("", stop_reason="refusal"))
        with pytest.raises(LLMSafetyBlockError, match="declined"):
            await llm.generate(**CALL)

    @pytest.mark.parametrize(
        "error, expected, transient",
        [
            (anthropic.APIConnectionError(request=_REQUEST), LLMProviderError, True),
            (anthropic.APITimeoutError(request=_REQUEST), LLMTimeoutError, True),
            (
                anthropic.RateLimitError(
                    "slow down",
                    response=httpx.Response(429, headers={"retry-after": "3"}, request=_REQUEST),
                    body=None,
                ),
                LLMRateLimitError,
                True,
            ),
            (
                anthropic.AuthenticationError(
                    "bad key", response=httpx.Response(401, request=_REQUEST), body=None
                ),
                LLMProviderError,
                False,
            ),
        ],
        ids=["connection", "timeout", "rate-limit", "auth"],
    )
    @pytest.mark.asyncio
    async def test_sdk_errors_become_typed_errors(
        self, tmp_path: Path, error: Exception, expected: type, transient: bool
    ) -> None:
        llm, _ = _anthropic(tmp_path, error)
        with pytest.raises(expected) as caught:
            await llm.generate(**CALL)
        assert is_transient(caught.value) is transient
        if expected is LLMRateLimitError:
            assert caught.value.retry_after == 3.0

    @pytest.mark.asyncio
    async def test_structured_output_is_instructed_and_validated(self, tmp_path: Path) -> None:
        llm, messages = _anthropic(tmp_path, _message('{"grade": "correct"}'))
        assert (await llm.generate_structured(schema=Verdict, **CALL)).grade == "correct"
        assert "JSON Schema" in messages.requests[0]["system"]


# ---------------------------------------------------------------------------
# Gemini client (SDK faked)
# ---------------------------------------------------------------------------


def _response(
    text: str | None = "answer", finish: str = "STOP", blocked: str | None = None
) -> types.GenerateContentResponse:
    candidates = (
        [
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason(finish),
            )
        ]
        if text is not None
        else []
    )
    return types.GenerateContentResponse(
        candidates=candidates,
        prompt_feedback=(
            types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason(blocked))
            if blocked
            else None
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=4, thoughts_token_count=3
        ),
    )


class _FakeModels:
    def __init__(self, result: Any = None, chunks: list[Any] | None = None) -> None:
        self._result = result
        self._chunks = chunks or []
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append(kwargs)

        async def chunks() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        return chunks()


def _gemini(tmp_path: Path, models: _FakeModels, **overrides: Any) -> GeminiLLM:
    async def aclose() -> None:
        pass

    client = SimpleNamespace(aio=SimpleNamespace(models=models, aclose=aclose))
    return GeminiLLM(_settings(tmp_path, **overrides), "gemini-3.8-flash", "low", client=client)  # type: ignore[arg-type]


def _quota_error() -> genai_errors.ClientError:
    body = {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}],
        }
    }
    return genai_errors.ClientError(429, body)


class TestGeminiLLM:
    @pytest.mark.asyncio
    async def test_generate_returns_text_and_usage(self, tmp_path: Path) -> None:
        result = await _gemini(tmp_path, _FakeModels(_response("the answer"))).generate(**CALL)
        assert (result.text, result.input_tokens, result.output_tokens) == ("the answer", 12, 7)
        assert result.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_config_sets_thinking_level_and_leaves_temperature_alone(
        self, tmp_path: Path
    ) -> None:
        models = _FakeModels(_response())
        await _gemini(tmp_path, models).generate(**CALL)
        config = models.calls[0]["config"]
        assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
        assert config.temperature is None
        assert config.max_output_tokens == MIN_OUTPUT_TOKENS
        assert config.system_instruction == "s"

    @pytest.mark.parametrize(
        "response", [_response(blocked="SAFETY"), _response("x", finish="RECITATION")]
    )
    @pytest.mark.asyncio
    async def test_blocks_are_safety_errors(self, tmp_path: Path, response: Any) -> None:
        with pytest.raises(LLMSafetyBlockError):
            await _gemini(tmp_path, _FakeModels(response)).generate(**CALL)

    @pytest.mark.asyncio
    async def test_empty_reply_is_an_output_error(self, tmp_path: Path) -> None:
        with pytest.raises(LLMOutputError, match="empty"):
            await _gemini(tmp_path, _FakeModels(_response(""))).generate(**CALL)

    @pytest.mark.asyncio
    async def test_quota_error_carries_the_retry_delay(self, tmp_path: Path) -> None:
        with pytest.raises(LLMRateLimitError) as caught:
            await _gemini(tmp_path, _FakeModels(_quota_error())).generate(**CALL)
        assert caught.value.retry_after == 17.0

    @pytest.mark.parametrize(
        "code, expected, transient",
        [
            (503, LLMProviderError, True),
            (404, LLMProviderError, False),
            (400, LLMProviderError, False),
            (504, LLMTimeoutError, True),
        ],
    )
    @pytest.mark.asyncio
    async def test_http_errors_are_typed(
        self, tmp_path: Path, code: int, expected: type, transient: bool
    ) -> None:
        error_cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
        error = error_cls(code, {"error": {"code": code, "message": "nope", "status": "X"}})
        with pytest.raises(expected) as caught:
            await _gemini(tmp_path, _FakeModels(error)).generate(**CALL)
        assert is_transient(caught.value) is transient

    @pytest.mark.asyncio
    async def test_network_timeout_is_typed(self, tmp_path: Path) -> None:
        with pytest.raises(LLMTimeoutError):
            await _gemini(tmp_path, _FakeModels(TimeoutError())).generate(**CALL)

    @pytest.mark.asyncio
    async def test_structured_uses_the_json_schema_mode(self, tmp_path: Path) -> None:
        models = _FakeModels(_response('{"grade": "none"}'))
        verdict = await _gemini(tmp_path, models).generate_structured(schema=Verdict, **CALL)
        config = models.calls[0]["config"]
        assert verdict.grade == "none"
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == Verdict.model_json_schema()

    @pytest.mark.asyncio
    async def test_stream_yields_text_deltas(self, tmp_path: Path) -> None:
        models = _FakeModels(chunks=[_response("Hello "), _response("world [S1].")])
        assert await _collect(_gemini(tmp_path, models).stream(**CALL)) == [
            "Hello ",
            "world [S1].",
        ]

    @pytest.mark.asyncio
    async def test_stream_errors_are_typed(self, tmp_path: Path) -> None:
        models = _FakeModels(chunks=[_response("Hello "), _quota_error()])
        with pytest.raises(LLMRateLimitError):
            await _collect(_gemini(tmp_path, models).stream(**CALL))


# ---------------------------------------------------------------------------
# Ollama client (httpx MockTransport)
# ---------------------------------------------------------------------------


def _ollama(tmp_path: Path, handler: Any) -> OllamaLLM:
    client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return OllamaLLM(_settings(tmp_path), "llama3.2", client=client)


class TestOllamaLLM:
    @pytest.mark.asyncio
    async def test_generate(self, tmp_path: Path) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "message": {"content": " local answer "},
                    "prompt_eval_count": 9,
                    "eval_count": 3,
                    "done_reason": "stop",
                },
            )

        result = await _ollama(tmp_path, handler).generate(**CALL)
        assert (result.text, result.input_tokens, result.output_tokens) == ("local answer", 9, 3)
        assert seen[0]["messages"][0] == {"role": "system", "content": "s"}
        assert seen[0]["stream"] is False and "temperature" not in seen[0]["options"]

    @pytest.mark.asyncio
    async def test_stream_reads_ndjson(self, tmp_path: Path) -> None:
        lines = [
            {"message": {"content": "Local "}, "done": False},
            {"message": {"content": "answer."}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 2},
        ]
        body = "\n".join(json.dumps(line) for line in lines)
        llm = _ollama(tmp_path, lambda request: httpx.Response(200, text=body))
        assert await _collect(llm.stream(**CALL)) == ["Local ", "answer."]

    @pytest.mark.asyncio
    async def test_missing_model_says_how_to_pull_it(self, tmp_path: Path) -> None:
        llm = _ollama(tmp_path, lambda request: httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(LLMProviderError, match="ollama pull llama3.2"):
            await llm.generate(**CALL)

    @pytest.mark.asyncio
    async def test_server_down_is_transient(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(LLMProviderError, match="ollama serve") as caught:
            await _ollama(tmp_path, handler).generate(**CALL)
        assert is_transient(caught.value)

    @pytest.mark.asyncio
    async def test_structured_sends_the_schema_as_format(self, tmp_path: Path) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"grade": "correct"}'}})

        verdict = await _ollama(tmp_path, handler).generate_structured(schema=Verdict, **CALL)
        assert verdict.grade == "correct" and seen[0]["format"] == Verdict.model_json_schema()


# ---------------------------------------------------------------------------
# Groq client (OpenAI-compatible, httpx MockTransport)
# ---------------------------------------------------------------------------


def _groq(tmp_path: Path, handler: Any, thinking: Any = "low", **overrides: Any) -> GroqLLM:
    client = httpx.AsyncClient(base_url="http://groq.test", transport=httpx.MockTransport(handler))
    return GroqLLM(_settings(tmp_path, **overrides), "openai/gpt-oss-120b", thinking, client=client)


def _completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning": "thinking..."},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


def _sse(*events: dict) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


class TestGroqLLM:
    @pytest.mark.asyncio
    async def test_generate_reads_content_not_reasoning(self, tmp_path: Path) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/chat/completions"
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=_completion(" fast answer "))

        result = await _groq(tmp_path, handler).generate(**CALL)
        assert (result.text, result.input_tokens, result.output_tokens) == ("fast answer", 12, 4)
        body = seen[0]
        assert body["messages"][0] == {"role": "system", "content": "s"}
        assert body["max_completion_tokens"] == MIN_OUTPUT_TOKENS
        assert body["reasoning_effort"] == "low" and "temperature" not in body

    @pytest.mark.asyncio
    async def test_minimal_thinking_maps_to_low_and_none_sends_nothing(
        self, tmp_path: Path
    ) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=_completion("ok"))

        await _groq(tmp_path, handler, thinking="minimal").generate(**CALL)
        await _groq(tmp_path, handler, thinking=None).generate(**CALL)
        assert seen[0]["reasoning_effort"] == "low" and "reasoning_effort" not in seen[1]

    @pytest.mark.asyncio
    async def test_stream_reads_sse_content_deltas(self, tmp_path: Path) -> None:
        body = _sse(
            {"choices": [{"delta": {"reasoning": "hmm"}}]},
            {"choices": [{"delta": {"content": "Fast "}}]},
            {"choices": [{"delta": {"content": "answer."}, "finish_reason": "stop"}]},
            {"choices": [], "x_groq": {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}},
        )
        llm = _groq(tmp_path, lambda request: httpx.Response(200, text=body))
        assert await _collect(llm.stream(**CALL)) == ["Fast ", "answer."]

    @pytest.mark.asyncio
    async def test_429_carries_retry_after(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"retry-after": "7"},
                json={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
            )

        with pytest.raises(LLMRateLimitError) as caught:
            await _groq(tmp_path, handler).generate(**CALL)
        assert caught.value.retry_after == 7.0

    @pytest.mark.asyncio
    async def test_stream_errors_before_the_first_token_are_typed(self, tmp_path: Path) -> None:
        llm = _groq(tmp_path, lambda request: httpx.Response(503, text="busy"))
        with pytest.raises(LLMProviderError) as caught:
            await _collect(llm.stream(**CALL))
        assert is_transient(caught.value)

    @pytest.mark.parametrize(
        "status, expected, transient",
        [
            (401, "GROQ_API_KEY", False),
            (404, "no model 'openai/gpt-oss-120b'", False),
            (413, "Request too large", False),
            (498, "HTTP 498", True),
            (500, "HTTP 500", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_http_errors_map_to_typed_errors(
        self, tmp_path: Path, status: int, expected: str, transient: bool
    ) -> None:
        llm = _groq(
            tmp_path, lambda request: httpx.Response(status, json={"error": {"message": "x"}})
        )
        with pytest.raises(LLMProviderError, match=expected) as caught:
            await llm.generate(**CALL)
        assert is_transient(caught.value) is transient

    @pytest.mark.asyncio
    async def test_timeout_is_transient(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(LLMTimeoutError):
            await _groq(tmp_path, handler).generate(**CALL)

    @pytest.mark.asyncio
    async def test_content_filter_is_a_safety_block(self, tmp_path: Path) -> None:
        llm = _groq(
            tmp_path, lambda request: httpx.Response(200, json=_completion("", "content_filter"))
        )
        with pytest.raises(LLMSafetyBlockError):
            await llm.generate(**CALL)

    @pytest.mark.asyncio
    async def test_structured_sends_a_json_schema_response_format(self, tmp_path: Path) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=_completion('{"grade": "correct"}'))

        verdict = await _groq(tmp_path, handler).generate_structured(schema=Verdict, **CALL)
        assert verdict.grade == "correct"
        fmt = seen[0]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["schema"] == Verdict.model_json_schema()

    @pytest.mark.asyncio
    async def test_schema_validation_failure_is_an_output_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            error = {"message": "Generated JSON does not match", "code": "json_validate_failed"}
            return httpx.Response(400, json={"error": error})

        with pytest.raises(LLMOutputError):
            await _groq(tmp_path, handler).generate_structured(schema=Verdict, **CALL)
