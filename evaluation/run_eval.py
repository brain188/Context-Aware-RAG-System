"""Run the eval question set through the system and write a dated report.

    python -m evaluation.run_eval --retrieval-only    # retrieve + rerank only: no LLM, free
    python -m evaluation.run_eval                     # full pipeline: ~2 LLM calls per question
    python -m evaluation.run_eval --judge             # + 1 judge call per answer
    python -m evaluation.run_eval --tag smoke --fail-on-regression
    python -m evaluation.run_eval --retrieval-only --set embedding_query_instruction=

Needs Qdrant running with the books ingested (and Redis for full runs). Eval
runs bypass the semantic cache and log queries to data/eval_query_log.jsonl,
leaving the real cache and query log untouched. Each run is compared with the
newest earlier report of the same mode.

Exit codes: 0 ok, 1 regression (with --fail-on-regression), 2 setup error.
"""

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from qdrant_client import AsyncQdrantClient

from app.config import Settings, get_logger, get_settings
from app.dependencies import build_services, warm_models
from app.llm import AnthropicLLM, LLMError
from app.retrieval.errors import RetrievalError
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_embedder import SentenceTransformerQueryEmbedder
from app.retrieval.reranker import CrossEncoderScorer, Reranker
from app.retrieval.sparse_encoder import BM25SparseEncoder
from app.services.index_version import (
    IndexMismatchError,
    IndexVersionProvider,
    check_index_embedder,
)
from app.services.query_logger import JsonlQueryLog
from evaluation.compare import DEFAULT_TOLERANCE, compare, format_comparison
from evaluation.dataset import DEFAULT_QUESTIONS, DatasetError, EvalQuestion, load_questions, select
from evaluation.judge import AnswerJudge, Excerpt
from evaluation.metrics import ANSWERED_STATUSES, Observation, score_question
from evaluation.report import (
    REPORTS_DIR,
    build_report,
    git_state,
    load_report,
    question_result,
    render_markdown,
    report_paths,
    write_report,
)
from evaluation.runners import (
    CapturingQueryLog,
    ChunkLookup,
    PipelineRunner,
    RetrievalRunner,
    Runner,
)

logger = get_logger(__name__)

_CONFIG_FIELDS = (
    "embedding_model",
    "embedding_query_instruction",
    "reranker_model",
    "llm_model",
    "llm_temperature",
    "prompt_version",
    "retrieval_top_k",
    "rerank_top_k",
    "rrf_k",
    "grade_min_rerank_score",
    "chunk_min_tokens",
    "chunk_max_tokens",
    "chunk_overlap_tokens",
)

JudgeStep = Callable[[EvalQuestion, Observation], Awaitable[dict | None]]


def eval_settings(base: Settings, overrides: dict[str, str] | None = None) -> Settings:
    """Settings for eval runs: no semantic cache, a separate query log, plus *overrides*.

    Overrides (from --set NAME=VALUE) are converted to each field's type, so
    `--set rerank_top_k=8` or `--set embedding_query_instruction=` work as expected.

    Raises:
        ValueError: if an override names an unknown setting or has an invalid value.
    """
    update: dict[str, Any] = {
        "semantic_cache_enabled": False,
        "query_log_path": base.query_log_path.with_name("eval_query_log.jsonl"),
    }
    for name, raw in (overrides or {}).items():
        field = Settings.model_fields.get(name)
        if field is None:
            raise ValueError(f"--set {name}: no such setting")
        try:
            update[name] = TypeAdapter(field.annotation).validate_python(raw)
        except ValidationError as exc:
            raise ValueError(f"--set {name}={raw!r}: {exc.errors()[0]['msg']}") from exc
    return base.model_copy(update=update)


def make_judge_step(judge: AnswerJudge, lookup: ChunkLookup) -> JudgeStep:
    """Score an answer against the excerpts it was generated from, in the order it saw them."""

    async def step(question: EvalQuestion, observation: Observation) -> dict | None:
        payloads = await lookup.fetch([r.chunk_id for r in observation.reranked], with_text=True)
        excerpts = [
            Excerpt(
                label=f"{r.book} → {r.chapter} → p. {r.page_start}",
                text=payloads[r.chunk_id]["text"],
            )
            for r in observation.reranked
            if r.chunk_id in payloads
        ]
        verdict = await judge.judge(
            question.question, observation.answer, excerpts, f"judge-{question.id}"
        )
        return None if verdict is None else verdict.model_dump()

    return step


async def evaluate(
    questions: list[EvalQuestion], runner: Runner, k: int, judge: JudgeStep | None = None
) -> list[dict]:
    """Run, score, and (optionally) judge every question; return report results in order."""
    results = []
    for n, question in enumerate(questions, start=1):
        observation = await runner.observe(question)
        metrics = score_question(question, observation, k)
        verdict = None
        if judge is not None and observation.status in ANSWERED_STATUSES and not observation.error:
            try:
                verdict = await judge(question, observation)
            except (LLMError, RetrievalError) as exc:
                logger.warning("Judge failed on %s: %s", question.id, exc)
            if verdict is not None:
                metrics["faithfulness"] = verdict["faithfulness"]
                metrics["answer_relevance"] = verdict["answer_relevance"]
        result = question_result(question, observation, metrics, verdict)
        logger.info(
            "[%d/%d] %s passed=%s status=%s hit@k=%s",
            n,
            len(questions),
            question.id,
            result["passed"],
            observation.status or "-",
            metrics.get("hit@k", "-"),
        )
        results.append(result)
    return results


def run_config(settings: Settings, index_version: str, judge_model: str | None) -> dict:
    """The settings that decide the scores, recorded so comparisons can explain changes."""
    config = {name: getattr(settings, name) for name in _CONFIG_FIELDS}
    config["index_version"] = index_version
    if judge_model:
        config["judge_model"] = judge_model
    return config


def _qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )


async def _retrieval_runner(settings: Settings, qdrant: AsyncQdrantClient) -> Runner:
    embedder, scorer = SentenceTransformerQueryEmbedder(settings), CrossEncoderScorer(settings)
    loop = asyncio.get_running_loop()  # load both models now, so q1's latency is not load time
    await loop.run_in_executor(None, embedder.embed_query, "warm up")
    await loop.run_in_executor(None, scorer.score, [("warm up", "warm up")])
    retriever = HybridRetriever(qdrant, embedder, BM25SparseEncoder(settings), settings)
    return RetrievalRunner(retriever, Reranker(scorer, settings))


async def run(args: argparse.Namespace) -> int:
    """Evaluate, report, compare; return the process exit code."""
    try:
        settings = eval_settings(get_settings(), args.set)
    except ValueError as exc:
        print(f"Invalid override: {exc}", file=sys.stderr)
        return 2
    try:
        questions = select(load_questions(args.questions), args.ids, args.tag)
    except DatasetError as exc:
        print(f"Question set error: {exc}", file=sys.stderr)
        return 2
    mode = "retrieval" if args.retrieval_only else "full"
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if mode == "full":
        calls = len(questions) * (3 if args.judge else 2)
        logger.info("Full run of %d questions: up to %d LLM calls.", len(questions), calls)

    capture = CapturingQueryLog(JsonlQueryLog(settings)) if mode == "full" else None
    services = build_services(settings, query_log=capture) if capture else None
    qdrant = services.qdrant if services else _qdrant_client(settings)
    try:
        await check_index_embedder(qdrant, settings)
        index_version = await IndexVersionProvider(qdrant, settings).current()
        lookup = ChunkLookup(qdrant, settings.qdrant_collection)
        judge = judge_model = None
        if mode == "full":
            await warm_models(services)
            runner: Runner = PipelineRunner(services.pipeline, capture, lookup)
            if args.judge:
                judge_settings = settings.model_copy(
                    update={"llm_model": args.judge_model or settings.llm_model}
                )
                answer_judge = AnswerJudge(AnthropicLLM(judge_settings))
                judge, judge_model = make_judge_step(answer_judge, lookup), answer_judge.model
        else:
            runner = await _retrieval_runner(settings, qdrant)
        results = await evaluate(questions, runner, settings.rerank_top_k, judge)
    except RetrievalError as exc:
        print(
            f"Cannot evaluate: {exc}. Is Qdrant running and the library ingested?", file=sys.stderr
        )
        return 2
    except IndexMismatchError as exc:
        print(f"Cannot evaluate: {exc}", file=sys.stderr)
        return 2
    finally:
        await (services.aclose() if services else qdrant.close())

    report = build_report(
        {
            "started_at": started_at,
            "mode": mode,
            "git": git_state(),
            "questions_file": Path(args.questions).name,
            "question_count": len(questions),
            "selection": {"ids": args.ids, "tags": args.tag, "overrides": args.set},
            "config": run_config(settings, index_version, judge_model),
        },
        results,
    )
    comparison = None
    baseline = _baseline(args, mode)
    if baseline is not None:
        comparison = compare(load_report(baseline), report, baseline.name, args.tolerance)
    comparison_md = format_comparison(comparison) if comparison else None
    print(render_markdown(report, comparison_md))
    if not args.no_save:
        print(f"Report written to {write_report(report, args.reports_dir, comparison_md)}")
    return 1 if args.fail_on_regression and comparison and comparison.regressed else 0


def _baseline(args: argparse.Namespace, mode: str) -> Path | None:
    if args.compare == "none":
        return None
    if args.compare == "latest":
        earlier = report_paths(args.reports_dir, mode)
        return earlier[-1] if earlier else None
    return Path(args.compare)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and answers on the question set."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation (no LLM)")
    parser.add_argument("--judge", action="store_true", help="score faithfulness with an LLM")
    parser.add_argument("--judge-model", help="model for --judge (default: LLM_MODEL)")
    parser.add_argument("--ids", type=lambda s: [i for i in s.split(",") if i], help="a,b,c")
    parser.add_argument("--tag", action="append", help="only questions with this tag (repeatable)")
    parser.add_argument("--compare", default="latest", help="baseline report: latest | none | PATH")
    parser.add_argument(
        "--set",
        action="append",
        type=_override,
        default=[],
        metavar="NAME=VALUE",
        help="override one setting for this run (repeatable), e.g. rerank_top_k=8",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--no-save", action="store_true", help="print the report without writing it"
    )
    args = parser.parse_args(argv)
    if args.retrieval_only and (args.judge or args.judge_model):
        parser.error("--judge needs answers; drop --retrieval-only")
    args.set = dict(args.set)
    return args


def _override(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name.strip().lower(), value


def main() -> None:
    sys.exit(asyncio.run(run(parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
