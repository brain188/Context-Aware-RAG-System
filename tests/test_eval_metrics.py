"""Tests for evaluation/metrics.py and evaluation/dataset.py (pure, no services)."""

import json
from pathlib import Path

import pytest

from evaluation.dataset import (
    DEFAULT_QUESTIONS,
    DatasetError,
    EvalQuestion,
    ExpectedSource,
    load_questions,
    select,
)
from evaluation.metrics import (
    Observation,
    SourceRef,
    citation_hit,
    citation_precision,
    hit_at_k,
    is_relevant,
    mention_coverage,
    passed,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance,
    score_question,
    summarize,
)

KERNELS = ExpectedSource(book="PRML", chapter="6 Kernels", page_start=20, page_end=22)
MIXTURES = ExpectedSource(book="PRML", chapter="9 Mixtures", page_start=40, page_end=40)


def _ref(book="PRML", chapter="6 Kernels", start=20, end=20, section="s", cid="c") -> SourceRef:
    return SourceRef(cid, book, chapter, section, start, end)


HIT, MISS = _ref(), _ref(chapter="1 Intro", start=1, end=1)


def _question(**overrides) -> EvalQuestion:
    values = {"id": "q1", "question": "What is a kernel?", "expected_sources": [KERNELS]}
    return EvalQuestion(**{**values, **overrides})


# ---------------------------------------------------------------------------
# Relevance and ranking metrics
# ---------------------------------------------------------------------------


class TestRelevance:
    def test_same_book_chapter_and_overlapping_pages_is_relevant(self) -> None:
        assert is_relevant(_ref(start=22, end=25), KERNELS)

    @pytest.mark.parametrize(
        "ref",
        [_ref(book="ESL"), _ref(chapter="7 SVMs"), _ref(start=23, end=24), _ref(start=1, end=19)],
    )
    def test_other_book_chapter_or_pages_is_not(self, ref: SourceRef) -> None:
        assert not is_relevant(ref, KERNELS)

    def test_chapter_and_section_are_only_checked_when_given(self) -> None:
        loose = ExpectedSource(book="PRML", page_start=20, page_end=20)
        strict = ExpectedSource(book="PRML", section="6.1", page_start=20, page_end=20)
        assert is_relevant(_ref(chapter="anything"), loose)
        assert not is_relevant(_ref(section="6.2"), strict)

    def test_relevance_matches_any_expected_source(self) -> None:
        assert relevance(
            [MISS, HIT, _ref(chapter="9 Mixtures", start=40, end=40)], [KERNELS, MIXTURES]
        ) == [
            False,
            True,
            True,
        ]


class TestRankingMetrics:
    def test_precision_counts_relevant_share_of_what_was_returned(self) -> None:
        assert precision_at_k([True, False, True, False], 4) == 0.5
        assert precision_at_k([True, False], 5) == 0.5
        assert precision_at_k([], 5) == 0.0

    def test_hit_and_reciprocal_rank(self) -> None:
        assert hit_at_k([False, False, True], 2) == 0.0
        assert hit_at_k([False, False, True], 3) == 1.0
        assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
        assert reciprocal_rank([False]) == 0.0

    def test_recall_is_the_share_of_expected_sources_found(self) -> None:
        mixture = _ref(chapter="9 Mixtures", start=40, end=40)
        assert recall_at_k([HIT, MISS], [KERNELS, MIXTURES], 2) == 0.5
        assert recall_at_k([MISS, HIT, mixture], [KERNELS, MIXTURES], 2) == 0.5
        assert recall_at_k([MISS, HIT, mixture], [KERNELS, MIXTURES], 3) == 1.0

    def test_recall_without_expected_sources_raises(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k([HIT], [], 5)


class TestAnswerMetrics:
    def test_citation_precision_and_hit(self) -> None:
        assert citation_precision([HIT, MISS], [KERNELS]) == 0.5
        assert citation_hit([MISS, HIT], [KERNELS]) == 1.0
        assert citation_hit([MISS], [KERNELS]) == 0.0

    def test_no_citations_means_not_applicable(self) -> None:
        assert citation_precision([], [KERNELS]) is None
        assert citation_hit([], [KERNELS]) is None

    def test_mention_coverage_accepts_alternatives_case_insensitively(self) -> None:
        answer = "A kernel computes an Inner Product in feature space."
        assert mention_coverage(answer, ["inner product", "gram|feature space"]) == 1.0
        assert mention_coverage(answer, ["inner product", "mercer"]) == 0.5
        assert mention_coverage(answer, []) is None


# ---------------------------------------------------------------------------
# Scoring one question
# ---------------------------------------------------------------------------


class TestScoreQuestion:
    def test_retrieval_only_run_scores_retrieval_metrics_only(self) -> None:
        metrics = score_question(_question(), Observation([MISS, HIT], [MISS, HIT]), k=2)
        assert metrics["hit@1"] == 0.0 and metrics["hit@k"] == 1.0
        assert metrics["mrr"] == 0.5 and metrics["candidate_recall"] == 1.0
        assert "answered" not in metrics and "citation_hit" not in metrics
        assert passed(_question(), metrics) is True

    def test_answered_question_scores_citations_and_mentions(self) -> None:
        question = _question(must_mention=["inner product"])
        observation = Observation(
            [HIT], [HIT], status="answered", answer="An inner product [S1].", cited=[HIT]
        )
        metrics = score_question(question, observation, k=5)
        assert metrics["answered"] == 1.0
        assert metrics["citation_precision"] == metrics["citation_hit"] == 1.0
        assert metrics["mention_coverage"] == 1.0
        assert passed(question, metrics) is True

    def test_answer_citing_the_wrong_source_fails(self) -> None:
        observation = Observation([HIT], [HIT, MISS], status="answered", cited=[MISS])
        metrics = score_question(_question(), observation, k=5)
        assert metrics["citation_hit"] == 0.0
        assert passed(_question(), metrics) is False

    def test_declining_an_answerable_question_fails(self) -> None:
        metrics = score_question(_question(), Observation([HIT], [HIT], status="not_covered"), 5)
        assert metrics["answered"] == 0.0 and "citation_hit" not in metrics
        assert passed(_question(), metrics) is False

    def test_unanswerable_question_passes_only_when_declined(self) -> None:
        question = _question(answerable=False, expected_sources=[])
        declined = score_question(question, Observation(status="not_covered"), 5)
        answered = score_question(question, Observation(status="answered"), 5)
        assert (declined["abstained"], passed(question, declined)) == (1.0, True)
        assert (answered["abstained"], passed(question, answered)) == (0.0, False)
        assert "hit@k" not in declined

    def test_errored_question_fails(self) -> None:
        metrics = score_question(_question(), Observation(error="LLMError: down"), 5)
        assert metrics["errored"] == 1.0
        assert passed(_question(), metrics) is False

    def test_question_without_answer_key_has_no_verdict_in_retrieval_runs(self) -> None:
        question = _question(expected_sources=[])
        assert passed(question, score_question(question, Observation([HIT], [HIT]), 5)) is None


class TestSummarize:
    def test_means_skip_metrics_that_do_not_apply(self) -> None:
        summary = summarize([{"hit@k": 1.0, "citation_hit": None}, {"hit@k": 0.0}])
        assert summary == {"hit@k": {"mean": 0.5, "n": 2}}


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, *rows: dict | str) -> Path:
    path = tmp_path / "q.jsonl"
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestDataset:
    def test_shipped_question_set_is_valid(self) -> None:
        questions = load_questions(DEFAULT_QUESTIONS)
        assert len(questions) >= 20
        assert any(not q.answerable for q in questions)

    def test_loads_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"id": "a", "question": "Q?"}, "", {"id": "b", "question": "R?"})
        assert [q.id for q in load_questions(path)] == ["a", "b"]

    @pytest.mark.parametrize(
        "rows, message",
        [
            (["{not json"], "q.jsonl:1"),
            ([{"id": "a", "question": "Q?"}, {"id": "a", "question": "R?"}], "duplicate id"),
            ([{"id": "a", "question": "Q?", "surprise": 1}], "surprise"),
            (
                [{"id": "a", "question": "Q?", "answerable": False, "must_mention": ["x"]}],
                "unanswerable",
            ),
            (
                [
                    {
                        "id": "a",
                        "question": "Q?",
                        "expected_sources": [{"book": "B", "page_start": 5, "page_end": 2}],
                    }
                ],
                "page_end",
            ),
        ],
    )
    def test_malformed_files_name_the_problem(self, tmp_path: Path, rows, message: str) -> None:
        with pytest.raises(DatasetError, match=message):
            load_questions(_write(tmp_path, *rows))

    def test_empty_or_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="no questions"):
            load_questions(_write(tmp_path, ""))
        with pytest.raises(DatasetError, match="Cannot read"):
            load_questions(tmp_path / "missing.jsonl")

    def test_select_by_ids_and_tags(self) -> None:
        questions = [
            _question(id="a", tags=["smoke"]),
            _question(id="b", tags=["django"]),
            _question(id="c", tags=["smoke", "django"]),
        ]
        assert [q.id for q in select(questions, tags=["smoke"])] == ["a", "c"]
        assert [q.id for q in select(questions, ids=["b", "c"], tags=["smoke"])] == ["c"]
        with pytest.raises(DatasetError, match="Unknown question ids: z"):
            select(questions, ids=["z"])
        with pytest.raises(DatasetError, match="No questions"):
            select(questions, tags=["nothing"])

    def test_question_becomes_an_api_request_with_its_filters(self) -> None:
        request = _question(book_filter=["PRML"], chapter_filter="6 Kernels").to_request()
        assert (request.book_filter, request.chapter_filter, request.session_id) == (
            ["PRML"],
            "6 Kernels",
            None,
        )
