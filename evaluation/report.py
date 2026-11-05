"""Build, write, and find dated eval reports.

Each run writes reports/<UTC timestamp>_<mode>.json (the full record: config,
summary, every question's metrics and retrieved sources) and a .md summary next
to it. Reports are generated, never edited by hand; file names sort by time.
"""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from evaluation.dataset import EvalQuestion
from evaluation.metrics import Metrics, Observation, passed, summarize

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODES = ("retrieval", "full")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MISSES_SHOWN = 3  # retrieved sources listed per failed question in the summary
_TAG_HEADLINE = ("citation_hit", "hit@k", "abstained")  # first present is shown per tag


def git_state() -> dict[str, str | bool | None]:
    """Current commit and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def question_result(
    question: EvalQuestion, observation: Observation, metrics: Metrics, judge: dict | None = None
) -> dict:
    """One question's entry in the report."""
    return {
        "id": question.id,
        "question": question.question,
        "tags": question.tags,
        "answerable": question.answerable,
        "passed": passed(question, metrics),
        "status": observation.status,
        "grade": observation.grade,
        "error": observation.error,
        "answer": observation.answer,
        "metrics": metrics,
        "judge": judge,
        "reranked": [asdict(r) for r in observation.reranked],
        "cited": [asdict(r) for r in observation.cited],
        "candidates": [asdict(r) for r in observation.candidates],
        "latency_ms": observation.latency_ms,
    }


def build_report(run: dict, results: list[dict]) -> dict:
    """Assemble run info, overall and per-tag summaries, and per-question results."""
    tags = sorted({t for r in results for t in r["tags"]})
    return {
        "run": run,
        "summary": summarize([r["metrics"] for r in results]),
        "by_tag": {t: summarize([r["metrics"] for r in results if t in r["tags"]]) for t in tags},
        "results": results,
    }


def report_paths(reports_dir: Path, mode: str) -> list[Path]:
    """Existing JSON reports of *mode*, oldest first."""
    return sorted(Path(reports_dir).glob(f"*_{mode}.json"))


def load_report(path: Path) -> dict:
    """Read a report written by write_report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_report(report: dict, reports_dir: Path, comparison_md: str | None = None) -> Path:
    """Write the JSON report and its Markdown summary; return the JSON path."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run']['started_at'].replace(':', '-')}_{report['run']['mode']}"
    path = reports_dir / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(render_markdown(report, comparison_md), encoding="utf-8")
    return path


def _summary_table(summary: dict) -> list[str]:
    lines = ["| Metric | Mean | n |", "|---|---|---|"]
    lines += [f"| {name} | {s['mean']:.3f} | {s['n']} |" for name, s in summary.items()]
    return lines


def _llm_label(config: dict) -> str:
    """ "provider:model" of the generator (reports before the LLM layer had `llm_model`)."""
    if "llm_generation_model" in config:
        return f"{config.get('llm_provider', '?')}:{config['llm_generation_model']}"
    return str(config.get("llm_model", "?"))


def _where(ref: dict) -> str:
    pages = (
        f"p. {ref['page_start']}"
        if ref["page_start"] == ref["page_end"]
        else (f"pp. {ref['page_start']}–{ref['page_end']}")
    )
    return f"{ref['book']} → {ref['chapter']} → {pages}"


def render_markdown(report: dict, comparison_md: str | None = None) -> str:
    """Human-readable summary: config, metrics, regressions, and what went wrong."""
    run = report["run"]
    git = run["git"]
    lines = [
        f"# Eval report — {run['started_at']} ({run['mode']})",
        "",
        f"- Commit: `{git['commit']}`{' (uncommitted changes)' if git['dirty'] else ''}",
        f"- Questions: {run['question_count']} from `{run['questions_file']}`",
        f"- Index version: `{run['config'].get('index_version')}`",
        f"- Models: embed `{run['config']['embedding_model']}`, rerank "
        f"`{run['config']['reranker_model']}`, LLM `{_llm_label(run['config'])}`",
        "",
        "## Summary",
        "",
        *_summary_table(report["summary"]),
    ]
    if comparison_md:
        lines += ["", "## Against the previous run", "", comparison_md]

    failed = [r for r in report["results"] if r["passed"] is False]
    lines += ["", f"## Failed questions ({len(failed)})", ""]
    for r in failed:
        reason = r["error"] or (f"status {r['status']}" if r["status"] else "no relevant chunk")
        lines.append(f"- **{r['id']}** — {r['question']} ({reason})")
        lines += [f"  - got: {_where(ref)}" for ref in r["reranked"][:_MISSES_SHOWN]]
    if not failed:
        lines.append("None.")
    lines += ["", "## By tag", ""]
    for tag, summary in report["by_tag"].items():
        key = next((k for k in _TAG_HEADLINE if k in summary), None)
        if key:
            lines.append(f"- {tag}: {key} {summary[key]['mean']:.3f} (n={summary[key]['n']})")
    return "\n".join(lines) + "\n"
