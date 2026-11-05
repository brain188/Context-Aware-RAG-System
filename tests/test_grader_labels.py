"""The grader-vs-hand-labels tool: stored excerpts, label carry-over, agreement maths."""

from pathlib import Path

import pytest

from app.config import Settings
from app.models import ScoredChunk
from app.retrieval.grader import RetrievalGrader
from evaluation.dataset import DatasetError
from evaluation.grader_labels import (
    LabelRow,
    agreement_report,
    excerpt_rows,
    grade_rows,
    merge_labels,
    read_rows,
    render_agreement,
    write_rows,
)
from conftest import CHUNK_IDS, FakeLLM, library_chunk


def _row(
    row_id: str = "q1", label: str | None = None, names: tuple[str, ...] = ("kl",)
) -> LabelRow:
    reranked = [ScoredChunk(chunk=library_chunk(n), score=0.9) for n in names]
    return LabelRow(
        id=row_id,
        question="What is the KL divergence?",
        answerable=True,
        excerpts=excerpt_rows(reranked),
        label=label,
    )


class TestStoredExcerpts:
    def test_round_trip_rebuilds_what_the_grader_saw(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        write_rows([_row(label="correct")], path)
        [row] = read_rows(path)
        excerpt = row.excerpts[0]
        assert (excerpt.source, excerpt.chunk_id) == (1, CHUNK_IDS["kl"])
        assert excerpt.where == "PRML → 1 Introduction → p. 10"
        rebuilt = excerpt.scored_chunk()
        original = library_chunk("kl")
        assert rebuilt.chunk.text == original.text and rebuilt.score == 0.9
        assert rebuilt.chunk.metadata.section == original.metadata.section

    def test_bad_label_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.jsonl"
        write_rows([_row()], path)
        path.write_text(
            path.read_text(encoding="utf-8").replace('"label": null', '"label": "maybe"'),
            encoding="utf-8",
        )
        with pytest.raises(DatasetError, match="labels.jsonl:1"):
            read_rows(path)


class TestMergeLabels:
    def test_labels_survive_a_reexport_with_the_same_excerpts(self) -> None:
        merged = merge_labels([_row()], [_row(label="ambiguous")])
        assert merged[0].label == "ambiguous"

    def test_changed_excerpts_drop_the_old_label(self) -> None:
        merged = merge_labels([_row(names=("em",))], [_row(label="correct")])
        assert merged[0].label is None


class TestAgreement:
    def test_matrix_and_costly_errors(self) -> None:
        report = agreement_report(
            [
                ("correct", "correct"),
                ("none", "none"),
                ("none", "ambiguous"),  # false accept
                ("correct", "none"),  # false refusal
            ]
        )
        assert report["n"] == 4 and report["agreement"] == 0.5
        assert report["matrix"]["none"]["ambiguous"] == 1
        assert (report["false_accept"], report["false_refusal"]) == (1, 1)

    def test_small_sets_are_flagged(self) -> None:
        text = render_agreement(agreement_report([("correct", "correct")]), "v2")
        assert "Agreement: 1.000" in text and "label at least 30" in text

    def test_no_labels(self) -> None:
        assert "No labelled rows" in render_agreement(agreement_report([]), "v2")


@pytest.mark.asyncio
async def test_grade_rows_skips_unlabelled_rows(settings: Settings, llm: FakeLLM) -> None:
    llm.grades = ['{"grade": "ambiguous", "reason": "partial"}']
    rows = [_row("q1", label="correct"), _row("q2")]
    pairs = await grade_rows(rows, RetrievalGrader(llm, settings))
    assert pairs == [("correct", "ambiguous")]
    assert llm.calls == ["grade"]
