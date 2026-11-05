# ADR-0001: Gemini by default, Groq as fallback and judge, behind one LLM interface

- **Status:** accepted
- **Date:** 2026-11-05

## Context

Until now every LLM call went to Claude through `anthropic`, imported directly by the
LLM wrapper. That conflicts with three things in `PROJECT_BRIEF.md`:

- **Strategy 4, provider-agnostic core.** Swapping Gemini, Groq, Claude or Ollama should be a
  config change, not a rewrite.
- **NFR4, cost.** Free or low-cost by default. One query is 2–3 LLM calls (rewrite,
  grade, generate), and an eval run is hundreds.
- **NFR5, reliability.** On a 429: retry with backoff, then fall back to another provider.

Two constraints shape the design. Free-tier quotas are small (requests and tokens per
minute, plus a daily cap), so calls must queue rather than fail (R3). And the
current models changed their API surface. Claude 5.x rejects non-default `temperature`
and thinks adaptively by default. Gemini 3 recommends leaving `temperature` at its
default and controls reasoning with `thinking_level`. On both, reasoning shares the
output-token budget.

## Options

1. **Keep Claude only.** Best answer quality, but every call is paid, there is no fallback
   when the API is down, and no local mode.
   Claude as the fallback was also considered; Groq replaced it because its free tier
   (GPT-OSS 120B) keeps the fallback and the judge free as well.
2. **LangChain / LiteLLM as the abstraction.** Many providers for free, but another large
   dependency between us and each SDK's error types and features (structured output,
   thinking, retry-after), and less to show.
3. **Own small interface (`app/llm/`) with one module per provider**: the brief's plan.

## Decision

Option 3.

- `app/llm/base.py` defines the `LLMClient` protocol (`generate`, `stream`,
  `generate_structured`) and `LLMResult` (text, model, tokens, latency, finish reason).
- It also defines typed errors: `LLMRateLimitError`, `LLMTimeoutError`,
  `LLMProviderError(retryable)`, `LLMSafetyBlockError` and `LLMOutputError`. Services
  import only these.
- `gemini_client.py` (`google-genai`, async client), `anthropic_client.py`, and
  `groq_client.py` and `ollama_client.py` (plain httpx: Groq's API is OpenAI-compatible)
  are the only modules that call a provider. Each maps its failures onto the typed errors.
- `retry.py` wraps every client with a rolling one-minute RPM/TPM limiter (`rate_limiter.py`)
  and tenacity retries on transient errors, honouring retry-after.
- `FallbackLLM` switches provider when the primary still fails after retries.
- `factory.py` resolves **roles**: `generation` (answers) and `fast` (rewrite, grading) on
  the primary, and `judge` on its own provider.

Defaults:

| Role | Provider | Model |
|---|---|---|
| generation | Gemini | `gemini-3.8-flash` (thinking `low`) |
| fast | Gemini | `gemini-3.5-flash-lite` (thinking `minimal`) |
| judge | Groq | `openai/gpt-oss-120b` (reasoning `low`), a different model family from the generator, which reduces self-preference bias |
| fallback | Groq (`LLM_FALLBACK_PROVIDER=groq` in `.env`) | `openai/gpt-oss-120b` |

Claude stays available as provider `anthropic` for any role.

All four are production model ids. `python -m app.llm.list_models` lists what a key can call.

Retry and fallback rules:

- A stream is retried, or falls back, only before its first token, since text already
  sent cannot be taken back.
- Configuration errors (bad key, unknown model) and safety blocks never trigger the
  fallback; they surface so they get fixed.
- The judge never falls back, so all scores in one eval run come from one model.

### Free-tier data use (risk R2)

On Gemini's unpaid quota, Google may use prompts and responses to improve its products,
and human reviewers may read them. The EEA, Switzerland and the UK are excluded;
Cameroon is not. Every generation prompt contains up to `RERANK_TOPK` book excerpts, and
those are what Google could see.

**Accepted for personal study use on the free tier.** The excerpts are short passages
from published textbooks, the questions are study questions, and nothing personal is
sent. Mitigations if that changes:

- move the key to a paid Gemini tier, where data is not used for improvement;
- set `LLM_PROVIDER=ollama` for sensitive material (nothing leaves the machine);
- or set `LLM_PROVIDER=anthropic` (paid).

Each is a config change. Groq, which sees prompts only when Gemini fails or a judged
eval runs, does not retain customer data by default (except briefly for reliability and
abuse monitoring), per its "Your Data in GroqCloud" page.

## Consequences

- Swapping provider or model is a `.env` change. Adding a provider is one module plus a
  line in the factory. Tool calling (`call_with_tools`) comes with the LangGraph agents
  in Phase 2.
- `temperature` is sent only when `LLM_TEMPERATURE` is set. Output-token caps are raised
  to at least 1024, because reasoning shares the budget. Prompts still ask for short
  replies.
- Every model is rate-limited client-side: the primary's with `LLM_RPM_LIMIT` /
  `LLM_TPM_LIMIT`, the fallback's and judge's with `LLM_FALLBACK_RPM_LIMIT` /
  `LLM_FALLBACK_TPM_LIMIT` (Groq free tier: 30 RPM, 8K TPM), so these must match the
  account's quota. Groq's 8K TPM fits about one generation call a minute, so a long
  Gemini outage makes answers slow rather than failing. The daily cap is not
  modelled: when it is hit, the long retry-after skips retries and goes to the fallback.
- When the fallback answers, the query log and cache key still record the primary
  generation model. Per-call model and usage come with the Postgres query log (1.3) and
  cost accounting (4.8).
- The judge, grader and router can use `generate_structured`: native JSON-schema mode on
  Gemini and Ollama, best-effort `json_schema` on Groq, instructed JSON on Claude; all
  validated against the schema. The judge uses it now;
  the grader moves to it with TASKLIST 1.2.1.
- `google-genai` and `tenacity` are new. Groq needs no SDK. `anthropic` stays so Claude
  remains one `.env` change away.
