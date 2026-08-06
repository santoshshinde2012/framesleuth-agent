"""Tests for eval harness metrics."""

import json
from pathlib import Path

from framesleuth.eval import (
    run_all,
    run_citation_eval,
    run_classification_eval,
    run_faithfulness_eval,
    run_grounding_eval,
)
from framesleuth.eval.harness import build_sample_workspace
from scripts.eval_harness import evaluate_bundle


def test_eval_metrics_computation() -> None:
    predicted = {
        "repro_steps": [{"action": "Click Save"}, {"action": "Wait"}],
        "error_evidence": [{"text": "TypeError"}],
        "code_candidates": [{"file": "a.py"}],
    }
    expected = {
        "repro_steps": [{"action": "Click Save"}],
        "error_evidence": [{"text": "TypeError"}],
        "code_candidates": [{"file": "a.py"}],
    }

    metrics = evaluate_bundle(predicted, expected, k=3)

    assert metrics.repro_step_precision == 0.5
    assert metrics.repro_step_recall == 1.0
    assert metrics.error_capture_rate == 1.0
    assert metrics.grounding_hit_rate_at_k == 1.0


def test_classification_accuracy_above_threshold() -> None:
    result = run_classification_eval()
    assert result.metric >= 0.8, str(result)


def test_grounding_recall_perfect(tmp_path: Path) -> None:
    build_sample_workspace(tmp_path)
    result = run_grounding_eval(tmp_path)
    assert result.metric == 1.0, str(result)


def test_citation_integrity_perfect() -> None:
    result = run_citation_eval()
    assert result.metric == 1.0, str(result)


def test_faithfulness_perfect() -> None:
    """Every emitted key moment / step must cite real, resolvable evidence."""
    result = run_faithfulness_eval()
    assert result.metric == 1.0, str(result)


def test_run_all_returns_every_suite(tmp_path: Path) -> None:
    results = run_all(tmp_path)
    assert set(results) == {
        "classification",
        "grounding",
        "citation",
        "faithfulness",
        # Controls that must FAIL on bad input, so a checker that always returned
        # True could not score 100%.
        "negative",
        # Structural injection resistance across all four untrusted channels.
        "injection",
    }
    assert all(r.total > 0 for r in results.values())
    # Per-metric floors live in evals/thresholds.json (a single global 0.8 could
    # not say "citation integrity must be perfect, accuracy need only be high").
    # This asserts the gate is honoured, rather than restating one number here.
    floors = json.loads(
        (Path(__file__).resolve().parent.parent / "evals" / "thresholds.json").read_text()
    )["floors"]
    for result in results.values():
        floor = floors.get(result.name)
        assert floor is not None, f"{result.name} has no declared floor in evals/thresholds.json"
        assert result.metric >= floor, str(result)
