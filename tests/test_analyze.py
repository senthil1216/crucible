"""Tests for the phase-4 calibration analysis.

A hand-built synthetic JSONL with known verdicts/confidences — every asserted
number is computed by hand so the bucketing/rate math is pinned.
"""

import json

import pytest

from bench import analyze


def _verdict(pid, cls):
    return {
        "prediction_id": pid, "trigger_input": "[]",
        "predicted_error_type": "IndexError", "classification": cls,
        "detail": "", "actual_error_type": None,
    }


def _record(problem_id, rep, run_index, *, status="success", iterations=2,
            verdicts=None, retrieved_predictions=None):
    return {
        "problem_id": problem_id, "rep": rep, "run_index": run_index,
        "category": "list", "function_name": "f", "goal": "g",
        "status": status, "iterations": iterations,
        "retrieved_predictions": retrieved_predictions or [],
        "replay_report": {
            "entry_point": "f",
            "confirmed": sum(1 for v in (verdicts or []) if v["classification"] == "confirmed"),
            "falsified": sum(1 for v in (verdicts or []) if v["classification"] == "falsified"),
            "off_topic": sum(1 for v in (verdicts or []) if v["classification"] == "off_topic"),
            "tested": 0,
            "verdicts": verdicts or [],
        },
    }


@pytest.fixture
def synthetic(tmp_path):
    # Confidences: high-conf preds (0.9) confirm 2/2; mid-conf (0.6) confirm 1/2;
    # plus one off-topic. Confidence comes from retrieved_predictions.
    rp_high = [{"id": "h1", "confidence": 0.9}, {"id": "h2", "confidence": 0.9}]
    rp_mid = [{"id": "m1", "confidence": 0.6}, {"id": "m2", "confidence": 0.6}]
    records = [
        _record("p-00", 1, 0, iterations=3,
                verdicts=[_verdict("h1", "confirmed"), _verdict("h2", "confirmed")],
                retrieved_predictions=rp_high),
        _record("p-00", 2, 1, iterations=1,
                verdicts=[_verdict("m1", "confirmed"), _verdict("m2", "falsified")],
                retrieved_predictions=rp_mid),
        _record("p-01", 1, 2, iterations=5,
                verdicts=[_verdict("x1", "off_topic")],
                retrieved_predictions=[]),
        _record("p-01", 2, 3, status="failed", iterations=None,
                verdicts=[], retrieved_predictions=[]),
    ]
    path = tmp_path / "results.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_calibration_buckets(synthetic):
    a = analyze.analyze(synthetic)
    cal = a["calibration"]
    # High bucket [0.9,1.0]: 2 confirmed, 0 falsified → 100%.
    assert cal["[0.9, 1.0]"]["confirmed"] == 2
    assert cal["[0.9, 1.0]"]["falsified"] == 0
    assert cal["[0.9, 1.0]"]["confirmation_rate"] == pytest.approx(1.0)
    # Mid bucket [0.5,0.7): 1 confirmed, 1 falsified → 50%.
    assert cal["[0.5, 0.7)"]["tested"] == 2
    assert cal["[0.5, 0.7)"]["confirmation_rate"] == pytest.approx(0.5)
    # Empty buckets report None.
    assert cal["[0.0, 0.5)"]["confirmation_rate"] is None


def test_off_topic_rate(synthetic):
    a = analyze.analyze(synthetic)
    ot = a["off_topic"]
    # 5 verdicts total (2 + 2 + 1), 1 off-topic → 20%.
    assert ot["total_replays"] == 5
    assert ot["off_topic"] == 1
    assert ot["rate"] == pytest.approx(0.2)


def test_convergence_variance(synthetic):
    a = analyze.analyze(synthetic)
    conv = a["convergence"]
    # p-00 succeeded twice (3, 1) → mean 2.0; p-01 succeeded once (5).
    assert conv["p-00"]["n"] == 2
    assert conv["p-00"]["mean"] == pytest.approx(2.0)
    assert conv["p-00"]["min"] == 1 and conv["p-00"]["max"] == 3
    assert conv["p-01"]["n"] == 1
    # The failed run contributes no convergence sample.
    assert "p-01" in conv


def test_memory_helped_split(synthetic):
    a = analyze.analyze(synthetic)
    mh = a["memory_helped"]
    # Two successful runs surfaced predictions (p-00 r1,r2); one did not (p-01 r1).
    assert mh["successful_runs_with_predictions_surfaced"] == 2
    assert mh["successful_runs_without_predictions_surfaced"] == 1
    assert mh["mean_iterations_with_predictions"] == pytest.approx(2.0)
    assert mh["mean_iterations_without_predictions"] == pytest.approx(5.0)


def test_predictions_file_join_and_catalog(tmp_path, synthetic):
    # A predictions.jsonl in PredictionMemory serialization form.
    preds = [
        {"id": "h1", "content": {"confidence": 0.9, "times_tested": 4,
                                 "times_confirmed": 4, "retired": False,
                                 "trigger_input": "[]", "predicted_error_type": "IndexError",
                                 "source_goal": "find max"}},
        {"id": "r1", "content": {"confidence": 0.3, "times_tested": 10,
                                 "times_confirmed": 1, "retired": True,
                                 "trigger_input": "None", "predicted_error_type": "TypeError"}},
    ]
    ppath = tmp_path / "predictions.jsonl"
    with open(ppath, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    a = analyze.analyze(synthetic, ppath)
    assert a["predictions_available"] is True
    # Catalog: h1 survives (4/4 ≥ 0.5); r1 is retired → excluded.
    surv = a["surviving_predictions"]
    assert len(surv) == 1
    assert surv[0]["trigger_input"] == "[]"
    assert surv[0]["confirmation_rate"] == pytest.approx(1.0)
    # Retirement stats reflect the one retired prediction.
    assert a["retirement"]["available"] is True
    assert a["retirement"]["retired"] == 1
    assert a["retirement"]["total_predictions"] == 2


def test_render_report_is_markdown(synthetic):
    a = analyze.analyze(synthetic)
    report = analyze.render_report(a)
    assert report.startswith("# Crucible — Calibration Report")
    assert "Calibration by confidence bucket" in report
    assert "Off-topic rate" in report


def test_empty_results(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    a = analyze.analyze(path)
    assert a["runs"] == 0
    assert a["success_rate"] is None
    # Renders without crashing on empty data.
    analyze.render_report(a)
