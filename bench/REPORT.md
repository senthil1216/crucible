<!--
  ⚠️ PLACEHOLDER — generated from SYNTHETIC data to show the report layout.
  These numbers are NOT a real benchmark result. Regenerate on a capable machine:
      python -m bench.runner --reps 5
      python -m bench.analyze bench/results/<ts>.jsonl \
          --predictions .agent_memory/predictions/predictions.jsonl --out bench/REPORT.md
-->

# Crucible — Calibration Report

Generated: 2026-05-29T09:29:24 (synthetic placeholder)

## Run overview

- Task runs: 2
- Succeeded: 2/2 (100.0%)

## Calibration by confidence bucket (headline)

Are higher-confidence self-predictions confirmed more often? Off-topic verdicts are excluded; only confirmed/falsified count.

| Confidence | Confirmed | Falsified | Tested | Confirmation rate |
| --- | --- | --- | --- | --- |
| [0.0, 0.5) | 0 | 0 | 0 | — |
| [0.5, 0.7) | 0 | 1 | 1 | 0.0% |
| [0.7, 0.9) | 0 | 0 | 0 | — |
| [0.9, 1.0] | 1 | 0 | 1 | 100.0% |

## Off-topic rate (integrity metric)

- Replays attempted: 3
- Off-topic (unapplicable): 1 (33.3%)

A high off-topic rate means the entry-point heuristic is weak and the calibration sample above is thin — read the headline in that light.

## Convergence variance (iterations-to-success per problem)

| Problem | n | mean | stdev | min | max |
| --- | --- | --- | --- | --- | --- |
| lst-01-max | 2 | 2.00 | 1.00 | 1 | 3 |

## Memory-helped (correlational, not causal)

- Successful runs with predictions surfaced: 2 (mean iters: 2)
- Successful runs without: 0 (mean iters: None)

## Surviving-predictions catalog

| Trigger | Predicted error | Conf | Tested | Confirmed | Rate |
| --- | --- | --- | --- | --- | --- |
| `[]` | IndexError | 0.9 | 2 | 2 | 100.0% |

## Retirement stats

- Total predictions: 1
- Tested at least once: 1
- Retired (≥10 tests AND <30% rate): 0
- Aggregate confirmation rate: 100.0%

Retirement is long-horizon and cross-run; within a single benchmark run it will usually be 0 by design (prediction ids are near-unique per task).
