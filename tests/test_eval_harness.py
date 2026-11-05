"""Tests for the eval harness: runners, judge, reports, comparison, and the CLI.

Runners use the real pipeline wired around conftest's fakes (Qdrant local mode,
fakeredis, scripted LLM), so no models, servers, or API calls are needed.
"""

import json
from pathlib import Path

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.config import Settings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.sparse_encoder import BM25SparseEncoder
from evaluation import compare as compare_cli
from evaluation.compare import compare
from evaluation.dataset import EvalQuestion, ExpectedSource
from evaluation.judge import AnswerJudge, Excerpt, build_judge_prompt, parse_judge_reply
from evaluation.metrics import Observation
from evaluation.report import build_report, render_markdown, report_paths, write_report
from evaluation.run_eval import eval_settings, evaluate, parse_args
from evaluation.runners import CapturingQueryLog, ChunkLookup, PipelineRunner, RetrievalRunner
from tests.conftest import CHUNK_IDS, FakeEmbedder, FakeLLM, FakeScorer, build_test_services

KL = EvalQuestion(
    id="kl",
    question="What does the KL divergence measure?",
    expected_sources=[
        ExpectedSource(book="PRML", chapter="1 Introduction", page_start=10, page_end=10)
    ],
    must_mention=["distributions"],
    tags=["smoke"],
)
WRONG_KEY = EvalQuestion(
    id="kl-wrong-key",
    question="What does the KL divergence measure?",
    expected_sources=[ExpectedSource(book="ESL", page_start=99, page_end=99)],
)
OUT_OF_SCOPE = EvalQuestion(
    id="oos", question="What is the capital of France?", answerable=False, tags=["oos"]
)


@pytest_asyncio.fixture
async def eval_services(settings: Settings, llm: FakeLLM, redis: FakeAsyncRedis):
    capture = CapturingQueryLog()
    services = await build_test_services(settings, llm, redis, query_log=capture)
    yield services, capture
    await services.qdrant.close()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_answered_question_records_retrieval_and_citations(self, eval_services) -> None:
        services, capture = eval_services
        lookup = ChunkLookup(services.qdrant, services.settings.qdrant_collection)
        runner = PipelineRunner(services.pipeline, capture, lookup)
        (result,) = await evaluate([KL], runner, k=2)
        assert result["status"] == "answered" and result["passed"] is True
        assert result["reranked"][0]["chunk_id"] == CHUNK_IDS["kl"]
        assert result["cited"][0]["page_start"] == 10
        assert result["metrics"]["citation_hit"] == result["metrics"]["mention_coverage"] == 1.0
        assert result["grade"] == "correct"

    @pytest.mark.asyncio
    async def test_out_of_scope_question_is_declined(self, eval_services) -> None:
        services, capture = eval_services
        lookup = ChunkLookup(services.qdrant, services.settings.qdrant_collection)
        (result,) = await evaluate(
            [OUT_OF_SCOPE], PipelineRunner(services.pipeline, capture, lookup), 2
        )
        assert (result["status"], result["metrics"]["abstained"], result["passed"]) == (
            "not_covered",
            1.0,
            True,
        )

    @pytest.mark.asyncio
    async def test_llm_failure_is_recorded_not_raised(self, eval_services, llm: FakeLLM) -> None:
        services, capture = eval_services
        llm.fail_stream = True
        lookup = ChunkLookup(services.qdrant, services.settings.qdrant_collection)
        (result,) = await evaluate([KL], PipelineRunner(services.pipeline, capture, lookup), 2)
        assert result["error"].startswith("LLMError")
        assert result["passed"] is False
        assert result["reranked"], "retrieval ran before the failure and is still reported"


class TestRetrievalRunner:
    @pytest.mark.asyncio
    async def test_scores_retrieval_without_calling_the_llm(self, eval_services, llm) -> None:
        services, _ = eval_services
        settings = services.settings
        retriever = HybridRetriever(
            services.qdrant, FakeEmbedder(), BM25SparseEncoder(settings), settings
        )
        runner = RetrievalRunner(retriever, Reranker(FakeScorer(), settings))
        results = await evaluate([KL, WRONG_KEY], runner, k=2)
        assert [r["passed"] for r in results] == [True, False]
        assert results[0]["status"] is None and "answered" not in results[0]["metrics"]
        assert llm.calls == []


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class _JudgeLLM:
    model = "judge-model"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, *, system: str, prompt: str, max_tokens: int, request_id: str) -> str:
        self.prompt = prompt
        return self.reply


class TestJudge:
    def test_prompt_numbers_excerpts_like_the_generator(self) -> None:
        _, user = build_judge_prompt(
            "Q?", "A [S2].", [Excerpt("a", "first"), Excerpt("b", "second")]
        )
        assert "[S1] a\nfirst" in user and "[S2] b\nsecond" in user and "A [S2]." in user

    @pytest.mark.parametrize(
        "reply, ok",
        [
            ('```json\n{"faithfulness": 0.5, "answer_relevance": 1}\n```', True),
            ('{"faithfulness": 1.5, "answer_relevance": 1}', False),
            ("I think it is fine.", False),
        ],
    )
    def test_parse_reply(self, reply: str, ok: bool) -> None:
        assert (parse_judge_reply(reply) is not None) is ok

    @pytest.mark.asyncio
    async def test_judge_scores_land_in_the_metrics(self, eval_services) -> None:
        services, capture = eval_services
        lookup = ChunkLookup(services.qdrant, services.settings.qdrant_collection)
        judge_llm = _JudgeLLM('{"faithfulness": 0.8, "answer_relevance": 0.9}')

        async def judge_step(question, observation):
            texts = await lookup.fetch([r.chunk_id for r in observation.reranked], with_text=True)
            excerpts = [Excerpt("src", p["text"]) for p in texts.values()]
            verdict = await AnswerJudge(judge_llm).judge(
                question.question, observation.answer, excerpts, "r"
            )
            return verdict.model_dump()

        runner = PipelineRunner(services.pipeline, capture, lookup)
        (result,) = await evaluate([KL], runner, k=2, judge=judge_step)
        assert result["metrics"]["faithfulness"] == 0.8
        assert result["judge"]["answer_relevance"] == 0.9
        assert "KL divergence measures" in judge_llm.prompt


# ---------------------------------------------------------------------------
# Reports and comparison
# ---------------------------------------------------------------------------


def _result(qid: str, hit: float, tags=("smoke",)) -> dict:
    return {
        "id": qid, "question": f"{qid}?", "tags": list(tags), "answerable": True,
        "passed": hit == 1.0, "status": None, "grade": None, "error": None, "answer": "",
        "metrics": {"hit@k": hit, "errored": 0.0}, "judge": None,
        "reranked": [{"book": "PRML", "chapter": "1", "page_start": 3, "page_end": 4}],
        "cited": [], "candidates": [], "latency_ms": {},
    }  # fmt: skip


def _report(started_at: str, results: list[dict], **config) -> dict:
    run = {
        "started_at": started_at, "mode": "retrieval", "git": {"commit": "abc", "dirty": False},
        "questions_file": "q.jsonl", "question_count": len(results), "selection": {},
        "config": {"embedding_model": "e", "reranker_model": "r", "llm_model": "l",
                   "index_version": "v1", **config},
    }  # fmt: skip
    return build_report(run, results)


class TestReports:
    def test_written_report_is_dated_and_has_a_markdown_summary(self, tmp_path: Path) -> None:
        report = _report("2026-11-02T10:00:00Z", [_result("a", 1.0), _result("b", 0.0)])
        path = write_report(report, tmp_path, comparison_md="Baseline: x")
        assert path.name == "2026-11-02T10-00-00Z_retrieval.json"
        assert json.loads(path.read_text(encoding="utf-8"))["summary"]["hit@k"]["mean"] == 0.5
        summary = path.with_suffix(".md").read_text(encoding="utf-8")
        assert "**b**" in summary and "PRML → 1 → pp. 3–4" in summary and "Baseline: x" in summary

    def test_report_paths_sort_oldest_first_per_mode(self, tmp_path: Path) -> None:
        for stamp in ("2026-11-03T00-00-00Z", "2026-11-01T00-00-00Z"):
            (tmp_path / f"{stamp}_retrieval.json").write_text("{}")
        (tmp_path / "2026-11-02T00-00-00Z_full.json").write_text("{}")
        assert [p.name[:10] for p in report_paths(tmp_path, "retrieval")] == [
            "2026-11-01",
            "2026-11-03",
        ]

    def test_markdown_lists_failed_questions_with_what_was_retrieved(self) -> None:
        markdown = render_markdown(_report("t", [_result("a", 0.0)]))
        assert "## Failed questions (1)" in markdown and "no relevant chunk" in markdown


class TestCompare:
    def test_drop_beyond_tolerance_is_a_regression(self) -> None:
        old = _report("t1", [_result("a", 1.0), _result("b", 1.0)])
        new = _report("t2", [_result("a", 1.0), _result("b", 0.0)])
        comparison = compare(old, new, "old.json", tolerance=0.03)
        (delta,) = [d for d in comparison.deltas if d.name == "hit@k"]
        assert (delta.old, delta.new, delta.regressed) == (1.0, 0.5, True)
        assert comparison.regressed and comparison.newly_failing == ["b"]

    def test_small_drop_within_tolerance_is_not(self) -> None:
        old = _report("t1", [_result("a", 1.0)])
        new = _report("t2", [_result("a", 0.0)])
        assert not compare(old, new, "old", tolerance=1.0).regressed

    def test_only_shared_questions_are_compared(self) -> None:
        old = _report("t1", [_result("a", 1.0), _result("b", 0.0)])
        new = _report("t2", [_result("a", 1.0)])
        comparison = compare(old, new, "old", tolerance=0.0)
        assert comparison.shared_questions == 1 and not comparison.regressed

    def test_config_changes_are_reported(self) -> None:
        comparison = compare(
            _report("t1", [_result("a", 1.0)]),
            _report("t2", [_result("a", 1.0)], rerank_top_k=8),
            "o",
            0.03,
        )
        assert comparison.config_changes == {"rerank_top_k": (None, 8)}

    def test_cli_latest_exits_1_on_regression(self, tmp_path: Path, capsys) -> None:
        write_report(_report("2026-11-01T00:00:00Z", [_result("a", 1.0)]), tmp_path)
        write_report(_report("2026-11-02T00:00:00Z", [_result("a", 0.0)]), tmp_path)
        assert compare_cli.main(["--latest", "--reports-dir", str(tmp_path)]) == 1
        assert "REGRESSED" in capsys.readouterr().out

    def test_cli_latest_passes_with_fewer_than_two_reports(self, tmp_path: Path, capsys) -> None:
        assert compare_cli.main(["--latest", "--reports-dir", str(tmp_path)]) == 0
        assert "nothing to compare" in capsys.readouterr().out


class TestCli:
    def test_eval_settings_bypass_the_cache_and_production_log(self, settings: Settings) -> None:
        tuned = eval_settings(settings)
        assert tuned.semantic_cache_enabled is False
        assert tuned.query_log_path.name == "eval_query_log.jsonl"
        assert tuned.query_log_path.parent == settings.query_log_path.parent

    def test_set_overrides_are_typed_and_validated(self, settings: Settings) -> None:
        args = parse_args(["--set", "rerank_top_k=8", "--set", "EMBEDDING_QUERY_INSTRUCTION="])
        tuned = eval_settings(settings, args.set)
        assert (tuned.rerank_top_k, tuned.embedding_query_instruction) == (8, "")
        with pytest.raises(ValueError, match="no such setting"):
            eval_settings(settings, {"nope": "1"})
        with pytest.raises(ValueError, match="rerank_top_k"):
            eval_settings(settings, {"rerank_top_k": "many"})

    def test_set_needs_name_equals_value(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--set", "rerank_top_k"])

    def test_judge_needs_a_full_run(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--retrieval-only", "--judge"])

    def test_ids_and_tags_parse(self) -> None:
        args = parse_args(["--ids", "a,b", "--tag", "smoke", "--tag", "django"])
        assert (args.ids, args.tag, args.compare) == (["a", "b"], ["smoke", "django"], "latest")

    def test_observation_defaults_are_independent(self) -> None:
        first, second = Observation(), Observation()
        first.reranked.append("x")
        assert second.reranked == []
