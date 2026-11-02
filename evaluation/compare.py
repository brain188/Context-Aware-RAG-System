"""Compare two eval reports and flag regressions.

Metrics are recomputed over the questions both runs share, so a run on a subset
(e.g. --tag smoke) compares fairly against a full one.

CLI (exit code 1 on a regression, so CI can gate on it):
    python -m evaluation.compare OLD.json NEW.json
    python -m evaluation.compare --latest            # two newest reports of each mode
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.metrics import LOWER_IS_BETTER, summarize
from evaluation.report import REPORTS_DIR, MODES, load_report, report_paths

DEFAULT_TOLERANCE = 0.03  # absolute drop in a mean that counts as a regression


@dataclass(frozen=True)
class MetricDelta:
    """One metric's mean in both runs."""

    name: str
    old: float
    new: float
    regressed: bool

    @property
    def delta(self) -> float:
        return round(self.new - self.old, 4)


@dataclass
class Comparison:
    """How a run differs from its baseline."""

    baseline: str
    shared_questions: int
    deltas: list[MetricDelta] = field(default_factory=list)
    newly_failing: list[str] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)
    config_changes: dict[str, tuple] = field(default_factory=dict)

    @property
    def regressed(self) -> bool:
        return any(d.regressed for d in self.deltas)


def _by_id(report: dict) -> dict[str, dict]:
    return {r["id"]: r for r in report["results"]}


def compare(baseline: dict, current: dict, baseline_name: str, tolerance: float) -> Comparison:
    """Diff *current* against *baseline* over their shared questions.

    A metric regresses when its mean moves the wrong way by more than *tolerance*.
    """
    old, new = _by_id(baseline), _by_id(current)
    shared = sorted(old.keys() & new.keys())
    old_summary = summarize([old[i]["metrics"] for i in shared])
    new_summary = summarize([new[i]["metrics"] for i in shared])

    deltas = []
    for name in sorted(old_summary.keys() & new_summary.keys()):
        before, after = old_summary[name]["mean"], new_summary[name]["mean"]
        worse = (after - before) if name in LOWER_IS_BETTER else (before - after)
        deltas.append(MetricDelta(name, before, after, regressed=worse > tolerance + 1e-9))

    old_config, new_config = baseline["run"]["config"], current["run"]["config"]
    changes = {
        key: (old_config.get(key), new_config.get(key))
        for key in sorted(old_config.keys() | new_config.keys())
        if old_config.get(key) != new_config.get(key)
    }
    return Comparison(
        baseline=baseline_name,
        shared_questions=len(shared),
        deltas=deltas,
        newly_failing=[i for i in shared if old[i]["passed"] is True and new[i]["passed"] is False],
        newly_passing=[i for i in shared if old[i]["passed"] is False and new[i]["passed"] is True],
        config_changes=changes,
    )


def format_comparison(comparison: Comparison) -> str:
    """Markdown tables for a comparison."""
    lines = [
        f"Baseline: `{comparison.baseline}` ({comparison.shared_questions} shared questions)",
        "",
        "| Metric | Baseline | Now | Δ | |",
        "|---|---|---|---|---|",
    ]
    for d in comparison.deltas:
        flag = "**REGRESSED**" if d.regressed else ""
        lines.append(f"| {d.name} | {d.old:.3f} | {d.new:.3f} | {d.delta:+.3f} | {flag} |")
    if comparison.newly_failing:
        lines += ["", f"Newly failing: {', '.join(comparison.newly_failing)}"]
    if comparison.newly_passing:
        lines += ["", f"Newly passing: {', '.join(comparison.newly_passing)}"]
    if comparison.config_changes:
        lines += ["", "Config changes:"]
        lines += [f"- `{k}`: {a!r} → {b!r}" for k, (a, b) in comparison.config_changes.items()]
    return "\n".join(lines)


def _compare_files(old: Path, new: Path, tolerance: float) -> bool:
    """Print the comparison of two report files; return True if it regressed."""
    comparison = compare(load_report(old), load_report(new), old.name, tolerance)
    print(f"## {new.name}\n\n{format_comparison(comparison)}\n")
    return comparison.regressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare eval reports; exit 1 on regression.")
    parser.add_argument("reports", nargs="*", type=Path, help="OLD.json NEW.json")
    parser.add_argument("--latest", action="store_true", help="compare the two newest per mode")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    if args.latest == bool(args.reports):
        parser.error("give either OLD.json NEW.json or --latest")
    if args.reports:
        if len(args.reports) != 2:
            parser.error("give exactly two report files")
        return 1 if _compare_files(*args.reports, args.tolerance) else 0

    regressed, compared = False, 0
    for mode in MODES:
        paths = report_paths(args.reports_dir, mode)
        if len(paths) >= 2:
            regressed |= _compare_files(paths[-2], paths[-1], args.tolerance)
            compared += 1
    if not compared:
        print(f"Fewer than two reports of any mode in {args.reports_dir}; nothing to compare.")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
