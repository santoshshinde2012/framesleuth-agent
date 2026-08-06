# Evals

Model-free suites: fixed fixtures in, metrics out. Run them with

```bash
python scripts/eval_harness.py --behavioral
```

## What each suite proves

| Suite | Proves |
|---|---|
| `classification_accuracy` | The deterministic labeller picks the right label, and every label is reachable. |
| `grounding_recall` | A symbol resolves to its definition, and vendored directories stay excluded. |
| `citation_integrity` | Every emitted step/error carries a citation and the timeline is ordered. |
| `faithfulness` | Every citation resolves to evidence that actually exists — nothing fabricated. |
| `negative_controls` | The checkers **fail** on bad input. Without this, a checker that always returned `True` would score 100%. |
| `injection_resistance` | Untrusted text from all four channels (OCR, transcript, console, intent) cannot escape the fenced data block. |

## Gating

`evals/thresholds.json` holds a per-metric floor plus `max_regression`, the drop
from `evals/baseline.json` that fails a run. A single global floor could not
express "citation integrity must be perfect, classification accuracy need only
be high".

Refresh the baseline deliberately, and only when the change is understood:

```bash
python scripts/eval_harness.py --behavioral --update-baseline
```
