"""Check the retrieval grader against your own labels.

    python -m evaluation.grader_labels export            # retrieve + rerank each question (no LLM)
    # open data/grader_labels.jsonl, set "label" to correct | ambiguous | none on each line
    python -m evaluation.grader_labels score             # grade the same excerpts, compare

`export` writes each question's reranked excerpts (text, location, rerank
score) exactly as the grader would see them, without the grader's verdict, so
you label blind. `score` re-grades those stored excerpts with the current
grader settings (`--set prompt_version=v1` to compare prompt versions) and
reports agreement and a confusion matrix. Stored excerpts keep the labels valid
after a re-ingest, and scoring needs no Qdrant.

The file holds book text, so it lives in the gitignored data/ folder.

Exit codes: 0 ok, 2 setup error.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings, get_logger, get_settings
from app.llm import LLMError, LLMFactory
from app.models import Chunk, ChunkMetadata, ScoredChunk
from app.prompts.grading import Grade
from app.retrieval.errors import RetrievalError
from app.retrieval.filters import filter_from_request, is_unfiltered
from app.retrieval.grader import RetrievalGrader
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_embedder import SentenceTransformerQueryEmbedder
from app.retrieval.reranker import CrossEncoderScorer, Reranker
from app.retrieval.sparse_encoder import BM25SparseEncoder
from app.services.citation import page_range
from app.services.index_version import IndexMismatchError, check_index_embedder
from evaluation.dataset import DEFAULT_QUESTIONS, DatasetError, EvalQuestion, load_questions, select
from evaluation.run_eval import parse_override, qdrant_client_for, eval_settings

logger = get_logger(__name__)

DEFAULT_LABELS = Path(__file__).resolve().parent.parent / "data" / "grader_labels.jsonl"
GRADES: tuple[Grade, ...] = get_args(Grade)
MIN_LABELS = 30  # TASKLIST 1.2.1: agreement on at least 30 labelled examples


class LabelExcerpt(BaseModel):
    """One reranked excerpt as the grader sees it."""

    model_config = ConfigDict(extra="forbid")

    source: int
    chunk_id: str
    where: str
    score: float
    book: str
    chapter: str
    section: str
    page_start: int
    page_end: int
    text: str

    def scored_chunk(self) -> ScoredChunk:
        metadata = ChunkMetadata(
            book=self.book,
            chapter=self.chapter,
            section=self.section,
            page_start=self.page_start,
            page_end=self.page_end,
            source_path="",
            content_hash="",
        )
        chunk = Chunk(chunk_id=self.chunk_id, text=self.text, metadata=metadata)
        return ScoredChunk(chunk=chunk, score=self.score)


class LabelRow(BaseModel):
    """A question, the excerpts retrieved for it, and your label for the set."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    answerable: bool
    excerpts: list[LabelExcerpt] = Field(default_factory=list)
    label: Grade | None = None
    note: str = ""


def excerpt_rows(reranked: list[ScoredChunk]) -> list[LabelExcerpt]:
    """The reranked chunks as stored excerpts, numbered like the grading prompt."""
    rows = []
    for n, scored in enumerate(reranked, start=1):
        meta = scored.chunk.metadata
        rows.append(
            LabelExcerpt(
                source=n,
                chunk_id=scored.chunk.chunk_id,
                where=f"{meta.book} → {meta.chapter} → {page_range(meta.page_start, meta.page_end)}",
                score=round(scored.score, 4),
                book=meta.book,
                chapter=meta.chapter,
                section=meta.section,
                page_start=meta.page_start,
                page_end=meta.page_end,
                text=scored.chunk.text,
            )
        )
    return rows


def read_rows(path: Path) -> list[LabelRow]:
    """Parse the labels file.

    Raises:
        DatasetError: if the file is unreadable or a line is malformed.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc
    rows = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(LabelRow.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"{path}:{n}: {exc.errors()[0]['msg']}") from exc
    return rows


def write_rows(rows: list[LabelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def merge_labels(fresh: list[LabelRow], previous: list[LabelRow]) -> list[LabelRow]:
    """Carry labels over to re-exported rows whose excerpts did not change."""
    old = {r.id: r for r in previous}
    merged = []
    for row in fresh:
        before = old.get(row.id)
        same = before is not None and [e.chunk_id for e in before.excerpts] == [
            e.chunk_id for e in row.excerpts
        ]
        if same and before is not None:
            row = row.model_copy(update={"label": before.label, "note": before.note})
        merged.append(row)
    return merged


def agreement_report(pairs: list[tuple[Grade, Grade]]) -> dict[str, Any]:
    """Agreement, confusion matrix and the two costly error kinds, from (label, predicted)."""
    n = len(pairs)
    matrix = {label: {pred: 0 for pred in GRADES} for label in GRADES}
    for label, predicted in pairs:
        matrix[label][predicted] += 1
    agree = sum(matrix[g][g] for g in GRADES)
    return {
        "n": n,
        "agreement": agree / n if n else None,
        "matrix": matrix,
        # Labelled none but the grader let it through: risks an answer from wrong excerpts.
        "false_accept": sum(matrix["none"][g] for g in ("correct", "ambiguous")),
        # Labelled answerable but the grader refused: a needless "not covered".
        "false_refusal": sum(matrix[g]["none"] for g in ("correct", "ambiguous")),
    }


def render_agreement(report: dict[str, Any], prompt_version: str) -> str:
    lines = [f"# Grader agreement (prompt {prompt_version}, n={report['n']})", ""]
    if report["agreement"] is None:
        return "\n".join(lines + ["No labelled rows."]) + "\n"
    lines += [f"Agreement: {report['agreement']:.3f}", ""]
    lines += ["| label \\ grader | " + " | ".join(GRADES) + " |", "|---" * (len(GRADES) + 1) + "|"]
    for label in GRADES:
        cells = " | ".join(str(report["matrix"][label][g]) for g in GRADES)
        lines.append(f"| {label} | {cells} |")
    lines += [
        "",
        f"- False accepts (labelled none, grader let it through): {report['false_accept']}",
        f"- False refusals (labelled answerable, grader said none): {report['false_refusal']}",
    ]
    if report["n"] < MIN_LABELS:
        lines.append(f"- Only {report['n']} labelled rows; label at least {MIN_LABELS}.")
    return "\n".join(lines) + "\n"


async def export(questions: list[EvalQuestion], settings: Settings, out: Path) -> int:
    qdrant = qdrant_client_for(settings)
    try:
        await check_index_embedder(qdrant, settings)
        embedder, scorer = SentenceTransformerQueryEmbedder(settings), CrossEncoderScorer(settings)
        retriever = HybridRetriever(qdrant, embedder, BM25SparseEncoder(settings), settings)
        reranker = Reranker(scorer, settings)
        rows = []
        for question in questions:
            filters = filter_from_request(question.to_request())
            filters = None if is_unfiltered(filters) else filters
            rid = f"labels-{question.id}"
            candidates = await retriever.retrieve(question.question, rid, filters=filters)
            reranked = await reranker.rerank(question.question, candidates, rid)
            rows.append(
                LabelRow(
                    id=question.id,
                    question=question.question,
                    answerable=question.answerable,
                    excerpts=excerpt_rows(reranked),
                )
            )
    finally:
        await qdrant.close()
    previous = read_rows(out) if out.exists() else []
    rows = merge_labels(rows, previous)
    write_rows(rows, out)
    kept = sum(r.label is not None for r in rows)
    print(f"Wrote {len(rows)} rows to {out} ({kept} labels carried over).")
    print('Set "label" on each line to correct | ambiguous | none, then run `score`.')
    return 0


async def grade_rows(rows: list[LabelRow], grader: RetrievalGrader) -> list[tuple[Grade, Grade]]:
    """(label, grader verdict) for every labelled row; disagreements are printed."""
    pairs: list[tuple[Grade, Grade]] = []
    for row in rows:
        if row.label is None:
            continue
        chunks = [e.scored_chunk() for e in row.excerpts]
        result = await grader.grade(row.question, chunks, f"labels-{row.id}")
        pairs.append((row.label, result.grade))
        if result.grade != row.label:
            print(f"- {row.id}: label {row.label}, grader {result.grade} ({result.reason})")
    return pairs


async def score(path: Path, settings: Settings) -> int:
    rows = read_rows(path)
    llms = LLMFactory(settings)
    try:
        pairs = await grade_rows(rows, RetrievalGrader(llms.get("fast"), settings))
    finally:
        await llms.aclose()
    print(render_agreement(agreement_report(pairs), settings.prompt_version))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the retrieval grader with hand labels.")
    parser.add_argument("command", choices=["export", "score"])
    parser.add_argument("--file", type=Path, default=DEFAULT_LABELS, help="labels JSONL")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--ids", type=lambda s: [i for i in s.split(",") if i], help="a,b,c")
    parser.add_argument("--tag", action="append", help="only questions with this tag (repeatable)")
    parser.add_argument(
        "--set",
        action="append",
        type=parse_override,
        default=[],
        metavar="NAME=VALUE",
        help="override one setting (repeatable), e.g. prompt_version=v1",
    )
    args = parser.parse_args(argv)
    args.set = dict(args.set)
    return args


async def run(args: argparse.Namespace) -> int:
    try:
        settings = eval_settings(get_settings(), args.set)
        if args.command == "export":
            questions = select(load_questions(args.questions), args.ids, args.tag)
            return await export(questions, settings, args.file)
        return await score(args.file, settings)
    except ValueError as exc:  # DatasetError and bad --set values
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RetrievalError as exc:
        print(f"Cannot export: {exc}. Is Qdrant reachable?", file=sys.stderr)
        return 2
    except (IndexMismatchError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    sys.exit(asyncio.run(run(parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
