#!/usr/bin/env python3
"""Evaluation harness CLI for Framesleuth.

Two modes:
- ``--predicted/--expected``: bundle-vs-golden metrics (precision/recall/grounding).
- ``--behavioral``: model-free suites for classification, grounding, and citation
  integrity (the same checks gated in ``tests/test_eval_harness.py``).

The metric logic lives in :mod:`framesleuth.eval.harness`; this file is the CLI.
``evaluate_bundle`` / ``EvalMetrics`` are re-exported for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from framesleuth.eval.harness import EvalMetrics, evaluate_bundle, run_all

__all__ = ["EvalMetrics", "evaluate_bundle", "main"]


EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
THRESHOLDS_PATH = EVALS_DIR / "thresholds.json"
BASELINE_PATH = EVALS_DIR / "baseline.json"


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except (OSError, json.JSONDecodeError):
        return default


def _run_behavioral(*, update_baseline: bool = False, json_out: str | None = None) -> int:
    """Run the model-free behavioral suites and gate on per-metric thresholds.

    A single global floor could not express what these suites actually mean: an
    integrity suite that lets one fabricated citation through is a correctness
    failure (floor 1.00), while classification accuracy is a quality bar (0.85).
    A shared 0.8 also hid regressions — with 9 classification cases, one silently
    broken case scores 0.889 and passed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        results = run_all(Path(tmp))
    for result in results.values():
        print(result)

    metrics = {result.name: round(result.metric, 4) for result in results.values()}

    if update_baseline:
        EVALS_DIR.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", "utf-8")
        print(f"\nBaseline updated: {BASELINE_PATH}")
        return 0

    if json_out:
        Path(json_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", "utf-8")

    config = _load_json(THRESHOLDS_PATH, {})
    floors: dict[str, float] = config.get("floors", {})
    max_regression = float(config.get("max_regression", 1.0))
    baseline: dict[str, float] = _load_json(BASELINE_PATH, {})

    problems: list[str] = []
    for name, value in metrics.items():
        floor = floors.get(name)
        if floor is not None and value < floor:
            problems.append(f"{name}: {value:.2%} is below its floor of {floor:.2%}")
        previous = baseline.get(name)
        # Catch a slide that is still above the floor — 1.00 -> 0.89 is a real
        # regression even when the bar is 0.85.
        if previous is not None and value < previous - max_regression:
            problems.append(
                f"{name}: {value:.2%} regressed from the {previous:.2%} baseline "
                f"(max allowed drop {max_regression:.2%})"
            )

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAll suites met their thresholds.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Framesleuth eval harness")
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Run model-free classification/grounding/citation suites",
    )
    parser.add_argument("--predicted", help="Path to predicted bundle json")
    parser.add_argument("--expected", help="Path to expected golden bundle json")
    parser.add_argument("--output", default="-", help="Output JSON path or '-' for stdout")
    parser.add_argument("--k", type=int, default=5, help="Top-k for grounding hit rate")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite evals/baseline.json from this run (do this deliberately)",
    )
    parser.add_argument("--json-out", help="Also write the metric map to this path")
    args = parser.parse_args()

    if args.behavioral:
        return _run_behavioral(update_baseline=args.update_baseline, json_out=args.json_out)

    if not args.predicted or not args.expected:
        parser.error("--predicted and --expected are required (or use --behavioral)")

    predicted = json.loads(Path(args.predicted).read_text(encoding="utf-8"))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))

    metrics = evaluate_bundle(predicted, expected, k=args.k).to_dict()
    payload = json.dumps(metrics, indent=2)

    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
