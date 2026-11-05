"""Tune SEMANTIC_CACHE_THRESHOLD on question pairs.

    python -m evaluation.cache_threshold                  # similarity only (no LLM)
    python -m evaluation.cache_threshold --verify         # + the cache-match LLM check
    python -m evaluation.cache_threshold --verify --max-false-hit-rate 0.02

Each pair in evaluation/cache_pairs.jsonl is either a paraphrase ("same":
true, a hit is wanted) or a different question on the same topic ("same":
false, a hit would serve the wrong answer). Both questions are embedded the
way the pipeline embeds a standalone question, and every threshold from 0.80
to 0.99 is scored on its hit rate (paraphrases served from cache) and its
false-hit rate (different questions served the cached answer).

With --verify, every pair at or above 0.80 and below
SEMANTIC_CACHE_EXACT_THRESHOLD also gets the fast-model check the cache runs
on a candidate (services/cache_matcher.py), and a pair hits only if it clears
the threshold and the check says "same" (or it is a near-exact repeat). That
is what the pipeline serves. The "checks" column is the number of pairs that
would cost one fast-model call at that threshold.

The recommendation is the lowest threshold whose false-hit rate is within
--max-false-hit-rate (default 0): a false hit is a wrong answer with a valid
citation, which is worse than a miss that only costs one pipeline run.

Needs the embedding model, plus the fast LLM for --verify; no Qdrant or Redis.
Exit codes: 0 ok, 2 setup error.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings, get_settings
from app.llm import LLMError, LLMFactory
from app.retrieval.errors import EmbeddingError
from app.retrieval.query_embedder import SentenceTransformerQueryEmbedder
from app.services.cache_matcher import CacheMatcher, MatchKind
from evaluation.dataset import DatasetError
from evaluation.run_eval import eval_settings, parse_override

DEFAULT_PAIRS = Path(__file__).resolve().parent / "cache_pairs.jsonl"
THRESHOLDS = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))  # 0.80 … 0.99


class CachePair(BaseModel):
    """Two questions, and whether one's cached answer is right for the other."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    a: str
    b: str
    same: bool
    note: str = ""


@dataclass(frozen=True)
class ScoredPair:
    pair: CachePair
    similarity: float
    match: MatchKind | None = None  # the LLM check's result, with --verify

    def hits(self, threshold: float, exact: float | None) -> bool:
        """Would the cache serve *a*'s answer for *b*? Without *exact*, similarity alone."""
        if self.similarity < threshold:
            return False
        if exact is None or self.similarity >= exact:
            return True
        return self.match in ("exact", "verified")


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    hit_rate: float  # paraphrases the cache would serve
    false_hit_rate: float  # different questions the cache would serve
    false_hits: int
    checks: int = 0  # pairs needing a fast-model check at this threshold (--verify)


def load_pairs(path: Path = DEFAULT_PAIRS) -> list[CachePair]:
    """Parse the pairs file.

    Raises:
        DatasetError: if the file is unreadable, a line is malformed, ids repeat,
            or it lacks either kind of pair.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc
    pairs: list[CachePair] = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            pairs.append(CachePair.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"{path}:{n}: {exc.errors()[0]['msg']}") from exc
    ids = [p.id for p in pairs]
    if len(ids) != len(set(ids)):
        raise DatasetError(f"{path}: duplicate pair ids")
    if not any(p.same for p in pairs) or all(p.same for p in pairs):
        raise DatasetError(f"{path}: needs both paraphrase and different-question pairs")
    return pairs


def score_pairs(pairs: list[CachePair], embed: Callable[[str], list[float]]) -> list[ScoredPair]:
    """Cosine similarity of each pair's embeddings (each distinct question embedded once)."""
    vectors: dict[str, np.ndarray] = {}
    for text in dict.fromkeys(q for p in pairs for q in (p.a, p.b)):
        vector = np.asarray(embed(text), dtype=np.float64)
        vectors[text] = vector / np.linalg.norm(vector)
    return [ScoredPair(p, float(vectors[p.a] @ vectors[p.b])) for p in pairs]


def sweep(
    scored: list[ScoredPair],
    thresholds: tuple[float, ...] = THRESHOLDS,
    exact: float | None = None,
) -> list[SweepRow]:
    """Hit and false-hit rates per threshold; with *exact*, after the LLM check."""
    same = [s for s in scored if s.pair.same]
    different = [s for s in scored if not s.pair.same]
    rows = []
    for t in thresholds:
        false_hits = sum(s.hits(t, exact) for s in different)
        checks = 0 if exact is None else sum(t <= s.similarity < exact for s in scored)
        rows.append(
            SweepRow(
                threshold=t,
                hit_rate=sum(s.hits(t, exact) for s in same) / len(same),
                false_hit_rate=false_hits / len(different),
                false_hits=false_hits,
                checks=checks,
            )
        )
    return rows


async def verify_pairs(
    scored: list[ScoredPair], matcher: CacheMatcher, exact: float
) -> list[ScoredPair]:
    """Run the cache-match check on every pair that some threshold would make a candidate."""
    checked = []
    for s in scored:
        if THRESHOLDS[0] <= s.similarity < exact:
            match = await matcher.match(s.pair.b, s.pair.a, s.similarity, f"cache-{s.pair.id}")
            s = ScoredPair(s.pair, s.similarity, match)
        checked.append(s)
    return checked


def recommend(rows: list[SweepRow], max_false_hit_rate: float = 0.0) -> SweepRow | None:
    """The lowest threshold (most hits) whose false-hit rate is acceptable, if any."""
    allowed = [r for r in rows if r.false_hit_rate <= max_false_hit_rate]
    return min(allowed, key=lambda r: r.threshold) if allowed else None


def render(
    scored: list[ScoredPair],
    rows: list[SweepRow],
    best: SweepRow | None,
    current: float,
    max_false_hit_rate: float,
    verified: bool = False,
) -> str:
    n_same = sum(s.pair.same for s in scored)
    mode = "similarity + LLM check" if verified else "similarity only"
    header = "| threshold | hit rate | false-hit rate | false hits |"
    lines = [
        f"# Semantic cache threshold, {mode} ({n_same} paraphrase pairs, "
        f"{len(scored) - n_same} different-question pairs)",
        "",
        header + (" checks |" if verified else ""),
        "|---|---|---|---|" + ("---|" if verified else ""),
    ]
    for r in rows:
        mark = " ←" if best is not None and r.threshold == best.threshold else ""
        lines.append(
            f"| {r.threshold:.2f}{mark} | {r.hit_rate:.3f} | {r.false_hit_rate:.3f} | "
            f"{r.false_hits} |" + (f" {r.checks} |" if verified else "")
        )
    different = sorted((s for s in scored if not s.pair.same), key=lambda s: -s.similarity)
    same = sorted((s for s in scored if s.pair.same), key=lambda s: s.similarity)
    lines += ["", "Closest different questions (false-hit risk):"]
    lines += [_pair_line(s) for s in different[:5]]
    lines += ["", "Least similar paraphrases (missed hits):"]
    lines += [_pair_line(s) for s in same[:5]]
    if verified:
        wrong = [s for s in scored if s.match and (s.match == "verified") != s.pair.same]
        lines += ["", "LLM check disagrees with the pair label:"]
        lines += [_pair_line(s) for s in wrong] or ["- none"]
    lines.append("")
    current_row = next((r for r in rows if abs(r.threshold - current) < 1e-9), None)
    if current_row is not None:
        lines.append(
            f"Current SEMANTIC_CACHE_THRESHOLD={current:.2f}: hit rate "
            f"{current_row.hit_rate:.3f}, false-hit rate {current_row.false_hit_rate:.3f}."
        )
    if best is None:
        lines.append(
            f"No threshold up to {THRESHOLDS[-1]:.2f} keeps the false-hit rate within "
            f"{max_false_hit_rate:.3f}; set a higher threshold or disable the cache."
        )
    else:
        lines.append(
            f"Recommended: SEMANTIC_CACHE_THRESHOLD={best.threshold:.2f} (hit rate "
            f"{best.hit_rate:.3f}, false-hit rate {best.false_hit_rate:.3f})."
        )
    if not verified:
        lines += [
            "",
            "These rates ignore the LLM check the cache runs on candidates below "
            "SEMANTIC_CACHE_EXACT_THRESHOLD, so the recommendation is for a cache without it. "
            "Run with --verify for the setting the pipeline uses.",
        ]
    return "\n".join(lines) + "\n"


def _pair_line(s: ScoredPair) -> str:
    check = f" [{s.match}]" if s.match else ""
    return f"- {s.similarity:.4f}{check} {s.pair.id}: {s.pair.a!r} / {s.pair.b!r}"


async def run(pairs_path: Path, settings: Settings, max_false_hit_rate: float, verify: bool) -> int:
    pairs = load_pairs(pairs_path)
    embedder = SentenceTransformerQueryEmbedder(settings)
    scored = score_pairs(pairs, embedder.embed_query)
    exact = None
    if verify:
        exact = settings.semantic_cache_exact_threshold
        llms = LLMFactory(settings)
        try:
            scored = await verify_pairs(scored, CacheMatcher(llms.get("fast"), settings), exact)
        finally:
            await llms.aclose()
    rows = sweep(scored, exact=exact)
    best = recommend(rows, max_false_hit_rate)
    current = settings.semantic_cache_threshold
    print(render(scored, rows, best, current, max_false_hit_rate, verified=verify))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the semantic cache similarity threshold.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS, help="pairs JSONL")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also run the cache-match LLM check (one fast-model call per candidate pair)",
    )
    parser.add_argument(
        "--max-false-hit-rate",
        type=float,
        default=0.0,
        help="largest acceptable share of different questions served from cache (default 0)",
    )
    parser.add_argument(
        "--set",
        action="append",
        type=parse_override,
        default=[],
        metavar="NAME=VALUE",
        help="override one setting (repeatable), e.g. embedding_query_instruction=",
    )
    args = parser.parse_args(argv)
    args.set = dict(args.set)
    return args


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        settings = eval_settings(get_settings(), args.set)
        code = asyncio.run(run(args.pairs, settings, args.max_false_hit_rate, args.verify))
    except ValueError as exc:  # DatasetError and bad --set values
        print(f"Error: {exc}", file=sys.stderr)
        code = 2
    except EmbeddingError as exc:
        print(f"Error: cannot embed the questions: {exc}", file=sys.stderr)
        code = 2
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
