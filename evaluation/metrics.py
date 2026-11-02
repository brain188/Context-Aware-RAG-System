"""Retrieval, citation, and answer metrics for one question, and their aggregates.

Pure functions over plain data. A retrieved or cited chunk is *relevant* when it
comes from an expected source: same book (and chapter/section when the question
names one) with overlapping pages. Matching on metadata instead of chunk ids keeps
the question set valid across re-ingests.

Every per-question metric is a float in [0, 1], or None when it doesn't apply
(e.g. citation accuracy for an answer that was not given); None is left out of means.
"""

from dataclasses import dataclass, field
from statistics import mean

from evaluation.dataset import EvalQuestion, ExpectedSource

ANSWERED_STATUSES = frozenset({"answered", "low_confidence"})
# Every metric is higher-is-better except these.
LOWER_IS_BETTER = frozenset({"errored"})

Metrics = dict[str, float | None]


@dataclass(frozen=True)
class SourceRef:
    """Where a retrieved or cited chunk comes from."""

    chunk_id: str
    book: str
    chapter: str
    section: str
    page_start: int
    page_end: int


@dataclass
class Observation:
    """What the system did for one question: the input to scoring."""

    candidates: list[SourceRef] = field(default_factory=list)  # fused results, pre-rerank
    reranked: list[SourceRef] = field(default_factory=list)  # what an answer is built from
    status: str | None = None  # pipeline status; None in retrieval-only runs
    answer: str = ""
    cited: list[SourceRef] = field(default_factory=list)
    grade: str | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def is_relevant(ref: SourceRef, expected: ExpectedSource) -> bool:
    """True if *ref* comes from *expected*'s book, chapter/section (if given), and pages."""
    if ref.book != expected.book:
        return False
    if expected.chapter is not None and ref.chapter != expected.chapter:
        return False
    if expected.section is not None and ref.section != expected.section:
        return False
    return ref.page_start <= expected.page_end and expected.page_start <= ref.page_end


def relevance(refs: list[SourceRef], expected: list[ExpectedSource]) -> list[bool]:
    """Per rank: does that chunk match any expected source?"""
    return [any(is_relevant(ref, e) for e in expected) for ref in refs]


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Share of the top-k returned chunks that are relevant (0 when nothing was returned)."""
    top = relevant[:k]
    return sum(top) / len(top) if top else 0.0


def hit_at_k(relevant: list[bool], k: int) -> float:
    """1 if any of the top-k chunks is relevant."""
    return 1.0 if any(relevant[:k]) else 0.0


def recall_at_k(refs: list[SourceRef], expected: list[ExpectedSource], k: int) -> float:
    """Share of the expected sources found among the top-k chunks.

    Raises:
        ValueError: if there are no expected sources.
    """
    if not expected:
        raise ValueError("recall needs at least one expected source")
    top = refs[:k]
    return sum(any(is_relevant(r, e) for r in top) for e in expected) / len(expected)


def reciprocal_rank(relevant: list[bool]) -> float:
    """1 / rank of the first relevant chunk, 0 if none."""
    return next((1.0 / rank for rank, hit in enumerate(relevant, start=1) if hit), 0.0)


def citation_precision(cited: list[SourceRef], expected: list[ExpectedSource]) -> float | None:
    """Share of citations pointing at an expected source; None without citations."""
    if not cited:
        return None
    return sum(relevance(cited, expected)) / len(cited)


def citation_hit(cited: list[SourceRef], expected: list[ExpectedSource]) -> float | None:
    """1 if at least one citation points at an expected source; None without citations."""
    if not cited:
        return None
    return 1.0 if any(relevance(cited, expected)) else 0.0


def mention_coverage(answer: str, must_mention: list[str]) -> float | None:
    """Share of required phrases present in *answer* ("a|b" accepts either); None if none."""
    if not must_mention:
        return None
    text = answer.lower()
    found = sum(
        any(alt.strip().lower() in text for alt in phrase.split("|") if alt.strip())
        for phrase in must_mention
    )
    return found / len(must_mention)


def score_question(question: EvalQuestion, observation: Observation, k: int) -> Metrics:
    """All metrics that apply to *question* given what the system did."""
    metrics: Metrics = {"errored": 1.0 if observation.error else 0.0}
    expected = question.expected_sources
    if question.answerable and expected:
        relevant = relevance(observation.reranked, expected)
        metrics["hit@1"] = hit_at_k(relevant, 1)
        metrics["hit@k"] = hit_at_k(relevant, k)
        metrics["precision@k"] = precision_at_k(relevant, k)
        metrics["recall@k"] = recall_at_k(observation.reranked, expected, k)
        metrics["mrr"] = reciprocal_rank(relevant)
        metrics["candidate_recall"] = recall_at_k(
            observation.candidates, expected, len(observation.candidates)
        )

    if observation.status is None or observation.error:
        return metrics  # retrieval-only run, or no answer to judge
    answered = observation.status in ANSWERED_STATUSES
    if not question.answerable:
        metrics["abstained"] = 1.0 if observation.status == "not_covered" else 0.0
        return metrics
    metrics["answered"] = 1.0 if answered else 0.0
    if answered:
        if expected:
            metrics["citation_precision"] = citation_precision(observation.cited, expected)
            metrics["citation_hit"] = citation_hit(observation.cited, expected)
        metrics["mention_coverage"] = mention_coverage(observation.answer, question.must_mention)
    return metrics


def passed(question: EvalQuestion, metrics: Metrics) -> bool | None:
    """Did the system get this question right? None if there is no answer key to tell.

    Retrieval runs: a relevant chunk reached the top k. Full runs: answerable questions
    are answered citing an expected source; unanswerable ones are declined.
    """
    if metrics.get("errored"):
        return False
    if not question.answerable:
        abstained = metrics.get("abstained")
        return None if abstained is None else abstained == 1.0
    full_run = "answered" in metrics
    if not question.expected_sources:
        return metrics["answered"] == 1.0 if full_run else None
    if full_run:
        return metrics["answered"] == 1.0 and metrics.get("citation_hit") == 1.0
    return metrics["hit@k"] == 1.0


def summarize(per_question: list[Metrics]) -> dict[str, dict[str, float | int]]:
    """Mean and count of each metric over the questions where it applies."""
    names = sorted({name for metrics in per_question for name in metrics})
    summary: dict[str, dict[str, float | int]] = {}
    for name in names:
        values = [m[name] for m in per_question if m.get(name) is not None]
        if values:
            summary[name] = {"mean": round(mean(values), 4), "n": len(values)}
    return summary
