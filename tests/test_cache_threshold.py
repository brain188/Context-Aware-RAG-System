"""The semantic-cache threshold tuner: pair loading, similarity sweep, recommendation."""

from pathlib import Path

import pytest

from evaluation.cache_threshold import (
    DEFAULT_PAIRS,
    CachePair,
    ScoredPair,
    load_pairs,
    recommend,
    render,
    score_pairs,
    sweep,
    verify_pairs,
)
from evaluation.dataset import DatasetError


def _scored(same: list[float], different: list[float]) -> list[ScoredPair]:
    rows = [
        ScoredPair(CachePair(id=f"s{i}", a="a", b="b", same=True), s) for i, s in enumerate(same)
    ]
    rows += [
        ScoredPair(CachePair(id=f"d{i}", a="a", b="c", same=False), s)
        for i, s in enumerate(different)
    ]
    return rows


class TestLoadPairs:
    def test_the_committed_pairs_file_is_valid(self) -> None:
        pairs = load_pairs(DEFAULT_PAIRS)
        assert sum(p.same for p in pairs) >= 20 and sum(not p.same for p in pairs) >= 20

    def test_malformed_line_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "pairs.jsonl"
        path.write_text('{"id": "x", "a": "q", "b": "q2", "same": true}\n{"id": "y"}\n', "utf-8")
        with pytest.raises(DatasetError, match=r"pairs\.jsonl:2"):
            load_pairs(path)

    def test_needs_both_kinds_of_pair(self, tmp_path: Path) -> None:
        path = tmp_path / "pairs.jsonl"
        path.write_text('{"id": "x", "a": "q", "b": "q2", "same": true}\n', "utf-8")
        with pytest.raises(DatasetError, match="both"):
            load_pairs(path)

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        line = '{"id": "x", "a": "q", "b": "q2", "same": %s}\n'
        path = tmp_path / "pairs.jsonl"
        path.write_text(line % "true" + line % "false", "utf-8")
        with pytest.raises(DatasetError, match="duplicate"):
            load_pairs(path)


class TestScoring:
    def test_cosine_of_each_pair_embedding_each_question_once(self) -> None:
        vectors = {"a": [1.0, 0.0], "b": [2.0, 0.0], "c": [0.0, 3.0]}
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            return vectors[text]

        pairs = [
            CachePair(id="1", a="a", b="b", same=True),
            CachePair(id="2", a="a", b="c", same=False),
        ]
        scored = score_pairs(pairs, embed)
        assert [round(s.similarity, 6) for s in scored] == [1.0, 0.0]
        assert sorted(calls) == ["a", "b", "c"]

    def test_sweep_counts_hits_and_false_hits(self) -> None:
        rows = sweep(_scored(same=[0.95, 0.85], different=[0.90]), thresholds=(0.80, 0.90, 0.96))
        assert [(r.hit_rate, r.false_hit_rate) for r in rows] == [
            (1.0, 1.0),
            (0.5, 1.0),
            (0.0, 0.0),
        ]


class TestVerifiedSweep:
    def test_llm_rejections_remove_false_hits_and_count_checks(self) -> None:
        scored = [
            ScoredPair(CachePair(id="s", a="a", b="b", same=True), 0.88, "verified"),
            ScoredPair(CachePair(id="d", a="a", b="c", same=False), 0.97, "rejected"),
            ScoredPair(CachePair(id="r", a="a", b="a!", same=True), 0.995),  # repeat: no check
        ]
        [row] = sweep(scored, thresholds=(0.85,), exact=0.99)
        assert (row.hit_rate, row.false_hit_rate, row.checks) == (1.0, 0.0, 2)

    def test_failed_check_is_a_miss(self) -> None:
        scored = [
            ScoredPair(CachePair(id="s", a="a", b="b", same=True), 0.9, "failed"),
            ScoredPair(CachePair(id="d", a="a", b="c", same=False), 0.1),
        ]
        [row] = sweep(scored, thresholds=(0.85,), exact=0.99)
        assert row.hit_rate == 0.0

    @pytest.mark.asyncio
    async def test_verify_checks_only_candidate_pairs(self, tmp_path: Path) -> None:
        from app.services.cache_matcher import CacheMatcher
        from conftest import FakeLLM, make_settings

        llm = FakeLLM()
        llm.cache_matches = ['{"verdict": "different", "reason": "x"}']
        scored = [
            ScoredPair(CachePair(id="low", a="a", b="b", same=True), 0.5),
            ScoredPair(CachePair(id="mid", a="a", b="c", same=False), 0.9),
            ScoredPair(CachePair(id="top", a="a", b="d", same=True), 0.995),
        ]
        checked = await verify_pairs(scored, CacheMatcher(llm, make_settings(tmp_path)), 0.99)
        assert [s.match for s in checked] == [None, "rejected", None]
        assert llm.calls == ["cache_match"]


class TestRecommend:
    def test_lowest_threshold_with_no_false_hits(self) -> None:
        rows = sweep(_scored(same=[0.97, 0.93, 0.88], different=[0.91, 0.85]))
        best = recommend(rows)
        assert best is not None and best.threshold == 0.92
        assert best.hit_rate == pytest.approx(2 / 3) and best.false_hits == 0

    def test_tolerated_false_hit_rate_lowers_the_threshold(self) -> None:
        rows = sweep(_scored(same=[0.97, 0.93, 0.88], different=[0.91, 0.85]))
        best = recommend(rows, max_false_hit_rate=0.5)
        assert best is not None and best.threshold == 0.86

    def test_none_when_every_threshold_has_false_hits(self) -> None:
        rows = sweep(_scored(same=[0.999], different=[0.999]))
        assert recommend(rows) is None
        text = render(_scored([0.999], [0.999]), rows, None, 0.92, 0.0)
        assert "No threshold" in text

    def test_report_names_the_recommendation_and_risky_pairs(self) -> None:
        scored = _scored(same=[0.97, 0.88], different=[0.91])
        rows = sweep(scored)
        text = render(scored, rows, recommend(rows), current=0.92, max_false_hit_rate=0.0)
        assert "Recommended: SEMANTIC_CACHE_THRESHOLD=0.92" in text
        assert "Current SEMANTIC_CACHE_THRESHOLD=0.92" in text
        assert "0.9100 d0" in text
