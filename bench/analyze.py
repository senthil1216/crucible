"""
Calibration analysis (Track D phase 4).

Turns a phase-3 benchmark JSONL (from `bench.runner`) into `bench/REPORT.md` with
real numbers. Pure functions over the JSONL — runs offline, no LLM/Docker — so it
yields real numbers on a real run and harmless zeros/empties on synthetic data.

The headline is **calibration by confidence bucket**, NOT per-prediction. As the
NEXT_STEPS design note explains: a prediction id derives from the *failing* code,
which changes every rep, so each prediction is usually replayed ~once and a
per-prediction rate is just 0/1 or 1/1 — too sparse to calibrate. Instead we
bucket *all* replay verdicts by the prediction's self-reported confidence and
ask: are higher-confidence self-predictions confirmed more often?

A verdict only carries `prediction_id` + `classification`; the confidence lives in
`predictions.jsonl` (or, as a fallback, in a run's `retrieved_predictions`). The
join is by `prediction_id`.

Usage:
    python -m bench.analyze <results.jsonl> [--predictions <predictions.jsonl>] [--out bench/REPORT.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Confidence buckets (lower-inclusive, upper-exclusive; last is inclusive of 1.0).
BUCKETS: List[Tuple[float, float, str]] = [
    (0.0, 0.5, "[0.0, 0.5)"),
    (0.5, 0.7, "[0.5, 0.7)"),
    (0.7, 0.9, "[0.7, 0.9)"),
    (0.9, 1.0001, "[0.9, 1.0]"),
]
UNKNOWN_BUCKET = "unknown"


# --------------------------------------------------------------------------- #
# Loading                                                                       #
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_predictions(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Parse a PredictionMemory `predictions.jsonl` into flat dicts
    ({id, **content}). Returns [] if no path/file."""
    if not path or not Path(path).exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in load_jsonl(Path(path)):
        content = line.get("content", {})
        out.append({"id": line.get("id"), **content})
    return out


# --------------------------------------------------------------------------- #
# Confidence join + bucketing                                                   #
# --------------------------------------------------------------------------- #

def build_confidence_map(
    records: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """prediction_id → confidence. predictions.jsonl wins; a run's
    retrieved_predictions fills gaps so the analysis degrades gracefully when
    the predictions file isn't supplied."""
    conf: Dict[str, float] = {}
    for rec in records:
        for pr in rec.get("retrieved_predictions") or []:
            pid, c = pr.get("id"), pr.get("confidence")
            if pid is not None and c is not None:
                conf[pid] = c
    for p in predictions:                 # authoritative source last → overrides
        pid, c = p.get("id"), p.get("confidence")
        if pid is not None and c is not None:
            conf[pid] = c
    return conf


def bucket_for(confidence: Optional[float]) -> str:
    if confidence is None:
        return UNKNOWN_BUCKET
    for lo, hi, label in BUCKETS:
        if lo <= confidence < hi:
            return label
    return UNKNOWN_BUCKET


def calibration_by_confidence(
    records: List[Dict[str, Any]],
    conf_map: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """For each confidence bucket: confirmed / falsified counts and the
    confirmation rate. Off-topic verdicts are excluded (never calibratable)."""
    buckets: Dict[str, Dict[str, Any]] = {
        label: {"confirmed": 0, "falsified": 0} for *_, label in BUCKETS
    }
    buckets[UNKNOWN_BUCKET] = {"confirmed": 0, "falsified": 0}

    for rec in records:
        report = rec.get("replay_report") or {}
        for v in report.get("verdicts") or []:
            cls = v.get("classification")
            if cls not in ("confirmed", "falsified"):
                continue
            label = bucket_for(conf_map.get(v.get("prediction_id")))
            buckets[label]["confirmed" if cls == "confirmed" else "falsified"] += 1

    for b in buckets.values():
        tested = b["confirmed"] + b["falsified"]
        b["tested"] = tested
        b["confirmation_rate"] = (b["confirmed"] / tested) if tested else None
    return buckets


def off_topic_rate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Integrity metric — fraction of replays we couldn't apply. Reported
    prominently: a high rate means the calibration sample is thin."""
    total = off = 0
    for rec in records:
        report = rec.get("replay_report") or {}
        for v in report.get("verdicts") or []:
            total += 1
            if v.get("classification") == "off_topic":
                off += 1
    return {
        "total_replays": total,
        "off_topic": off,
        "rate": (off / total) if total else None,
    }


# --------------------------------------------------------------------------- #
# Convergence + memory-helped                                                   #
# --------------------------------------------------------------------------- #

def convergence_variance(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Iterations-to-success distribution per problem across reps."""
    by_problem: Dict[str, List[int]] = {}
    for rec in records:
        if rec.get("status") != "success" or rec.get("iterations") is None:
            continue
        by_problem.setdefault(rec["problem_id"], []).append(rec["iterations"])
    out: Dict[str, Dict[str, Any]] = {}
    for pid, iters in by_problem.items():
        out[pid] = {
            "n": len(iters),
            "iterations": sorted(iters),
            "mean": statistics.mean(iters),
            "stdev": statistics.pstdev(iters) if len(iters) > 1 else 0.0,
            "min": min(iters),
            "max": max(iters),
        }
    return out


def memory_helped(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Correlational (NOT causal): did surfacing predictions to the Planner go
    with fewer iterations / fewer off-topic replays? Splits successful runs by
    whether any prediction was surfaced."""
    with_pred_iters: List[int] = []
    without_pred_iters: List[int] = []
    for rec in records:
        if rec.get("status") != "success" or rec.get("iterations") is None:
            continue
        surfaced = bool(rec.get("retrieved_predictions"))
        (with_pred_iters if surfaced else without_pred_iters).append(rec["iterations"])

    def _mean(xs: List[int]) -> Optional[float]:
        return statistics.mean(xs) if xs else None

    return {
        "successful_runs_with_predictions_surfaced": len(with_pred_iters),
        "successful_runs_without_predictions_surfaced": len(without_pred_iters),
        "mean_iterations_with_predictions": _mean(with_pred_iters),
        "mean_iterations_without_predictions": _mean(without_pred_iters),
    }


# --------------------------------------------------------------------------- #
# Surviving-predictions catalog + retirement                                    #
# --------------------------------------------------------------------------- #

def surviving_predictions(
    predictions: List[Dict[str, Any]],
    min_rate: float = 0.5,
) -> List[Dict[str, Any]]:
    """Non-retired predictions that were tested and confirmed often — the
    concrete antipatterns worth publishing (e.g. '[] → IndexError')."""
    out: List[Dict[str, Any]] = []
    for p in predictions:
        if p.get("retired"):
            continue
        tested = int(p.get("times_tested", 0) or 0)
        confirmed = int(p.get("times_confirmed", 0) or 0)
        if tested == 0:
            continue
        rate = confirmed / tested
        if rate >= min_rate:
            out.append({
                "trigger_input": p.get("trigger_input"),
                "predicted_error_type": p.get("predicted_error_type"),
                "confidence": p.get("confidence"),
                "times_tested": tested,
                "times_confirmed": confirmed,
                "confirmation_rate": rate,
                "source_goal": p.get("source_goal"),
            })
    out.sort(key=lambda x: (x["confirmation_rate"], x["times_confirmed"]), reverse=True)
    return out


def retirement_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mirror PredictionMemory.get_stats from a parsed predictions.jsonl.
    Retirement is long-horizon/cross-run — usually 0 within a single run."""
    if not predictions:
        return {"available": False}
    total = len(predictions)
    tested = sum(1 for p in predictions if int(p.get("times_tested", 0) or 0) > 0)
    retired = sum(1 for p in predictions if p.get("retired"))
    total_tests = sum(int(p.get("times_tested", 0) or 0) for p in predictions)
    total_confirmations = sum(int(p.get("times_confirmed", 0) or 0) for p in predictions)
    return {
        "available": True,
        "total_predictions": total,
        "tested": tested,
        "retired": retired,
        "total_tests": total_tests,
        "total_confirmations": total_confirmations,
        "overall_confirmation_rate": (
            total_confirmations / total_tests if total_tests else None
        ),
    }


# --------------------------------------------------------------------------- #
# Top-level analysis + report rendering                                         #
# --------------------------------------------------------------------------- #

def analyze(
    results_path: Path,
    predictions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    records = load_jsonl(results_path)
    predictions = load_predictions(predictions_path)
    conf_map = build_confidence_map(records, predictions)

    runs = len(records)
    succeeded = sum(1 for r in records if r.get("status") == "success")
    return {
        "runs": runs,
        "succeeded": succeeded,
        "success_rate": (succeeded / runs) if runs else None,
        "calibration": calibration_by_confidence(records, conf_map),
        "off_topic": off_topic_rate(records),
        "convergence": convergence_variance(records),
        "memory_helped": memory_helped(records),
        "surviving_predictions": surviving_predictions(predictions),
        "retirement": retirement_stats(predictions),
        "predictions_available": bool(predictions),
    }


def _fmt_rate(r: Optional[float]) -> str:
    return f"{r:.1%}" if r is not None else "—"


def render_report(analysis: Dict[str, Any]) -> str:
    L: List[str] = [
        "# Crucible — Calibration Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Run overview",
        "",
        f"- Task runs: {analysis['runs']}",
        f"- Succeeded: {analysis['succeeded']}/{analysis['runs']} "
        f"({_fmt_rate(analysis['success_rate'])})",
        "",
        "## Calibration by confidence bucket (headline)",
        "",
        "Are higher-confidence self-predictions confirmed more often? Off-topic "
        "verdicts are excluded; only confirmed/falsified count.",
        "",
        "| Confidence | Confirmed | Falsified | Tested | Confirmation rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for *_, label in BUCKETS:
        b = analysis["calibration"][label]
        L.append(f"| {label} | {b['confirmed']} | {b['falsified']} | "
                 f"{b['tested']} | {_fmt_rate(b['confirmation_rate'])} |")
    unk = analysis["calibration"][UNKNOWN_BUCKET]
    if unk["tested"]:
        L.append(f"| {UNKNOWN_BUCKET} | {unk['confirmed']} | {unk['falsified']} | "
                 f"{unk['tested']} | {_fmt_rate(unk['confirmation_rate'])} |")

    ot = analysis["off_topic"]
    L += [
        "",
        "## Off-topic rate (integrity metric)",
        "",
        f"- Replays attempted: {ot['total_replays']}",
        f"- Off-topic (unapplicable): {ot['off_topic']} ({_fmt_rate(ot['rate'])})",
        "",
        "A high off-topic rate means the entry-point heuristic is weak and the "
        "calibration sample above is thin — read the headline in that light.",
        "",
        "## Convergence variance (iterations-to-success per problem)",
        "",
        "| Problem | n | mean | stdev | min | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pid in sorted(analysis["convergence"]):
        c = analysis["convergence"][pid]
        L.append(f"| {pid} | {c['n']} | {c['mean']:.2f} | {c['stdev']:.2f} | "
                 f"{c['min']} | {c['max']} |")

    mh = analysis["memory_helped"]
    L += [
        "",
        "## Memory-helped (correlational, not causal)",
        "",
        f"- Successful runs with predictions surfaced: "
        f"{mh['successful_runs_with_predictions_surfaced']} "
        f"(mean iters: {mh['mean_iterations_with_predictions']})",
        f"- Successful runs without: "
        f"{mh['successful_runs_without_predictions_surfaced']} "
        f"(mean iters: {mh['mean_iterations_without_predictions']})",
        "",
        "## Surviving-predictions catalog",
        "",
    ]
    surv = analysis["surviving_predictions"]
    if surv:
        L += [
            "| Trigger | Predicted error | Conf | Tested | Confirmed | Rate |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for s in surv:
            L.append(f"| `{s['trigger_input']}` | {s['predicted_error_type']} | "
                     f"{s['confidence']} | {s['times_tested']} | "
                     f"{s['times_confirmed']} | {_fmt_rate(s['confirmation_rate'])} |")
    else:
        L.append("_No surviving predictions "
                 "(no predictions.jsonl supplied, or none tested+confirmed)._")

    ret = analysis["retirement"]
    L += ["", "## Retirement stats", ""]
    if ret.get("available"):
        L += [
            f"- Total predictions: {ret['total_predictions']}",
            f"- Tested at least once: {ret['tested']}",
            f"- Retired (≥10 tests AND <30% rate): {ret['retired']}",
            f"- Aggregate confirmation rate: "
            f"{_fmt_rate(ret['overall_confirmation_rate'])}",
            "",
            "Retirement is long-horizon and cross-run; within a single benchmark "
            "run it will usually be 0 by design (prediction ids are near-unique "
            "per task).",
        ]
    else:
        L.append("_predictions.jsonl not supplied — pass `--predictions` for "
                 "retirement + catalog stats._")

    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bench.analyze")
    p.add_argument("results", help="Phase-3 results JSONL (from bench.runner)")
    p.add_argument("--predictions", default=None,
                   help="PredictionMemory predictions.jsonl (for confidence join, "
                        "catalog, retirement)")
    p.add_argument("--out", default="bench/REPORT.md", help="Report output path")
    args = p.parse_args(argv)

    analysis = analyze(
        Path(args.results),
        Path(args.predictions) if args.predictions else None,
    )
    report = render_report(analysis)
    Path(args.out).write_text(report)
    print(f"✅ Wrote {args.out}")
    print(f"   Runs: {analysis['runs']}, succeeded: {analysis['succeeded']}, "
          f"off-topic rate: {_fmt_rate(analysis['off_topic']['rate'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
