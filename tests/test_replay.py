"""
Tests for Track D phase 2: the replay engine.

Three layers:
  - Pure helpers (entry-point selection, sentinel parsing, classification) —
    fast, no sandbox.
  - ReplayEngine with a fake executor (canned driver stdout) — verifies the
    confirmed/falsified/off-topic routing and the record_replays write-back.
  - End-to-end against the real SandboxedExecutor — proves the driver actually
    reproduces (or doesn't) a predicted error inside a real subprocess.

The replay engine is record-only: these tests assert it classifies and records,
never that it changes a task result.
"""

import asyncio
import hashlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from agent.executor.sandbox import SandboxedExecutor, ExecutionConfig
from agent.memory.embeddings import EmbeddingClient
from agent.memory.predictions import PredictionMemory
from agent.models import CodeArtifact, Prediction, TestResults
from agent.replay import (
    CONFIRMED,
    FALSIFIED,
    OFF_TOPIC,
    SENTINEL,
    ReplayEngine,
    ReplayReport,
    _build_combined_source,
    _classify,
    _parse_sentinel,
    select_entry_point,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes.
# ---------------------------------------------------------------------------


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic, model-free embedding (shared with test_predictions)."""

    DIM = 64

    def __init__(self):
        super().__init__(model_name="fake")

    def _ensure_loaded(self) -> None:
        return

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        for token in (text or "").lower().split():
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class FakeExecutor:
    """Returns canned stdout per call, so engine routing can be tested without
    a real sandbox. `outputs` is consumed in order; falls back to the last."""

    def __init__(self, outputs: List[str]):
        self.outputs = outputs
        self.calls = 0

    async def execute(self, artifact: CodeArtifact) -> TestResults:
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return TestResults(passed=True, stdout=out)


def _sentinel_line(payload_json: str) -> str:
    return f"{SENTINEL}{payload_json}"


@pytest.fixture
def tmp_pred_dir():
    d = Path(tempfile.mkdtemp(prefix="crucible_replay_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry-point selection.
# ---------------------------------------------------------------------------


class TestEntryPointSelection:
    def test_single_public_function(self):
        name, reason = select_entry_point("def solve(x):\n    return x\n")
        assert name == "solve"
        assert reason == "single_function"

    def test_no_public_function(self):
        name, reason = select_entry_point("def _helper(x):\n    return x\n")
        assert name is None
        assert reason == "no_public_function"

    def test_ambiguous_without_goal(self):
        src = "def foo(x):\n    return x\ndef bar(y):\n    return y\n"
        name, reason = select_entry_point(src, goal="do something")
        assert name is None
        assert reason == "ambiguous_entry_point"

    def test_goal_token_breaks_tie(self):
        src = "def parse_csv(x):\n    return x\ndef render_html(y):\n    return y\n"
        name, reason = select_entry_point(src, goal="parse a csv file")
        assert name == "parse_csv"
        assert reason == "goal_token_match"

    def test_unparseable_source(self):
        name, reason = select_entry_point("def broken(:\n")
        assert name is None
        assert reason == "unparseable_source"


# ---------------------------------------------------------------------------
# Sentinel parsing + classification (pure).
# ---------------------------------------------------------------------------


class TestParseSentinel:
    def test_extracts_payload_amid_noise(self):
        stdout = "hello from solution\n" + _sentinel_line('{"outcome": "no_error"}') + "\n"
        assert _parse_sentinel(stdout) == {"outcome": "no_error"}

    def test_returns_none_without_sentinel(self):
        assert _parse_sentinel("just some output") is None

    def test_takes_last_sentinel(self):
        stdout = _sentinel_line('{"outcome": "no_error"}') + "\n" + _sentinel_line(
            '{"outcome": "raised", "type": "ValueError", "mro": ["ValueError"], "in_solution": true}'
        )
        assert _parse_sentinel(stdout)["outcome"] == "raised"


class TestClassify:
    def test_confirmed_exact_type(self):
        payload = {"outcome": "raised", "type": "IndexError",
                   "mro": ["IndexError", "LookupError", "Exception"], "in_solution": True}
        cls, _, actual = _classify(payload, "IndexError")
        assert cls == CONFIRMED
        assert actual == "IndexError"

    def test_confirmed_via_superclass(self):
        # Predicted a base class; actual is a subclass raised in the solution.
        payload = {"outcome": "raised", "type": "ZeroDivisionError",
                   "mro": ["ZeroDivisionError", "ArithmeticError", "Exception"], "in_solution": True}
        cls, _, _ = _classify(payload, "ArithmeticError")
        assert cls == CONFIRMED

    def test_falsified_clean_run(self):
        cls, _, _ = _classify({"outcome": "no_error"}, "IndexError")
        assert cls == FALSIFIED

    def test_falsified_wrong_error(self):
        payload = {"outcome": "raised", "type": "KeyError",
                   "mro": ["KeyError", "LookupError", "Exception"], "in_solution": True}
        cls, _, actual = _classify(payload, "IndexError")
        assert cls == FALSIFIED
        assert actual == "KeyError"

    def test_off_topic_call_boundary(self):
        payload = {"outcome": "raised", "type": "TypeError",
                   "mro": ["TypeError"], "in_solution": False}
        cls, _, _ = _classify(payload, "TypeError")
        assert cls == OFF_TOPIC

    def test_off_topic_unparseable(self):
        cls, _, _ = _classify({"outcome": "unparseable"}, "ValueError")
        assert cls == OFF_TOPIC

    def test_off_topic_no_signal(self):
        cls, _, _ = _classify(None, "ValueError")
        assert cls == OFF_TOPIC


def test_combined_source_line_count():
    combined, sol_lines = _build_combined_source("def f(x):\n    return x[0]\n", "f", "[]")
    # Solution is two lines; the driver is appended after.
    assert sol_lines == 2
    assert combined.startswith("def f(x):")
    assert SENTINEL in combined


# ---------------------------------------------------------------------------
# ReplayEngine routing + write-back (fake executor).
# ---------------------------------------------------------------------------


class TestReplayEngineRouting:
    @pytest.mark.asyncio
    async def test_confirmed_records_tested_and_confirmed(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir, embedding_client=FakeEmbeddingClient())
        pid = await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError",
            source_failure_id="f1", source_goal="first element",
        ))
        executor = FakeExecutor([_sentinel_line(
            '{"outcome": "raised", "type": "IndexError", "mro": ["IndexError", "LookupError"], "in_solution": true}'
        )])
        engine = ReplayEngine(executor, mem)

        report = await engine.replay_for_failures(["f1"], "def first(x):\n    return x[0]\n", "first element")
        assert report.confirmed == 1
        assert report.falsified == 0
        assert report.off_topic == 0

        row = mem.find_by_failure_id("f1")[0]
        assert row["times_tested"] == 1
        assert row["times_confirmed"] == 1

    @pytest.mark.asyncio
    async def test_off_topic_not_recorded(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir, embedding_client=FakeEmbeddingClient())
        await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="TypeError",
            source_failure_id="f1", source_goal="add numbers",
        ))
        executor = FakeExecutor([_sentinel_line(
            '{"outcome": "raised", "type": "TypeError", "mro": ["TypeError"], "in_solution": false}'
        )])
        engine = ReplayEngine(executor, mem)

        report = await engine.replay_for_failures(["f1"], "def add(a, b):\n    return a + b\n", "add numbers")
        assert report.off_topic == 1
        assert report.tested == 0
        # Off-topic must NOT touch the counters — it would pollute calibration.
        row = mem.find_by_failure_id("f1")[0]
        assert row["times_tested"] == 0
        assert row["times_confirmed"] == 0

    @pytest.mark.asyncio
    async def test_ambiguous_entry_all_off_topic_no_sandbox(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir, embedding_client=FakeEmbeddingClient())
        await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError",
            source_failure_id="f1", source_goal="x",
        ))
        executor = FakeExecutor(["should not be used"])
        engine = ReplayEngine(executor, mem)

        src = "def foo(x):\n    return x\ndef bar(y):\n    return y\n"
        report = await engine.replay_for_failures(["f1"], src, "x")
        assert report.off_topic == 1
        assert report.entry_point is None
        # Engine short-circuits before spending any sandbox time.
        assert executor.calls == 0

    @pytest.mark.asyncio
    async def test_no_predictions_empty_report(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir)
        engine = ReplayEngine(FakeExecutor(["x"]), mem)
        report = await engine.replay_for_failures(["nope"], "def f(x):\n    return x\n")
        assert report.verdicts == []
        assert report.tested == 0


# ---------------------------------------------------------------------------
# PredictionMemory replay scoring + retirement.
# ---------------------------------------------------------------------------


class TestPredictionMemoryReplayScoring:
    def test_record_replays_bumps_counters(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir)
        pid = asyncio.run(mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError", source_failure_id="f1",
        )))
        assert mem.record_replays({pid: True}) == 1
        row = mem.find_by_failure_id("f1")[0]
        assert row["times_tested"] == 1
        assert row["times_confirmed"] == 1

        mem.record_replays({pid: False})
        row = mem.find_by_failure_id("f1")[0]
        assert row["times_tested"] == 2
        assert row["times_confirmed"] == 1

    def test_auto_retire_low_confirmation(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir)
        pid = asyncio.run(mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError", source_failure_id="f1",
        )))
        # 10 tests, 2 confirmations → 20% < 30% floor → retire.
        for i in range(10):
            mem.record_replays({pid: i < 2})
        # Retired predictions are excluded from find_by_failure_id by default.
        assert mem.find_by_failure_id("f1") == []
        rows = mem.find_by_failure_id("f1", include_retired=True)
        assert rows[0]["retired"] is True
        assert rows[0]["times_tested"] == 10

    def test_no_retire_above_floor(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir)
        pid = asyncio.run(mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError", source_failure_id="f1",
        )))
        # 10 tests, 5 confirmations → 50% ≥ 30% → keep.
        for i in range(10):
            mem.record_replays({pid: i < 5})
        rows = mem.find_by_failure_id("f1")
        assert len(rows) == 1
        assert not rows[0].get("retired")

    @pytest.mark.asyncio
    async def test_retired_excluded_from_find_relevant(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir, embedding_client=FakeEmbeddingClient())
        pid = await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError",
            source_failure_id="f1", source_goal="compute first element of a list",
        ))
        for i in range(10):
            mem.record_replays({pid: i < 2})  # retire it
        results = await mem.find_relevant("compute first element of list", min_similarity=0.0)
        assert results == []

    def test_get_stats_reports_replay_aggregates(self, tmp_pred_dir):
        mem = PredictionMemory(tmp_pred_dir)
        p1 = asyncio.run(mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError", source_failure_id="f1",
        )))
        p2 = asyncio.run(mem.store(Prediction(
            trigger_input="-1", predicted_error_type="ValueError", source_failure_id="f2",
        )))
        mem.record_replays({p1: True, p2: False})
        stats = mem.get_stats()
        assert stats["total_predictions"] == 2
        assert stats["total_tests"] == 2
        assert stats["total_confirmations"] == 1
        assert stats["overall_confirmation_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End-to-end against the real sandbox.
# ---------------------------------------------------------------------------


def _real_engine(tmp_pred_dir) -> ReplayEngine:
    executor = SandboxedExecutor(config=ExecutionConfig(timeout_seconds=15))
    mem = PredictionMemory(tmp_pred_dir, embedding_client=FakeEmbeddingClient())
    return ReplayEngine(executor, mem), mem


class TestReplayEngineEndToEnd:
    @pytest.mark.asyncio
    async def test_confirmed_real_indexerror(self, tmp_pred_dir):
        engine, mem = _real_engine(tmp_pred_dir)
        await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError",
            source_failure_id="f1", source_goal="first element",
        ))
        # Buggy: empty list dereferenced without a bounds check.
        report = await engine.replay_for_failures(
            ["f1"], "def first_element(items):\n    return items[0]\n", "first element"
        )
        assert report.confirmed == 1, report.to_dict()

    @pytest.mark.asyncio
    async def test_falsified_when_fixed(self, tmp_pred_dir):
        engine, mem = _real_engine(tmp_pred_dir)
        await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="IndexError",
            source_failure_id="f1", source_goal="first element",
        ))
        # Fixed: handles the empty case.
        report = await engine.replay_for_failures(
            ["f1"],
            "def first_element(items):\n    return items[0] if items else None\n",
            "first element",
        )
        assert report.falsified == 1, report.to_dict()

    @pytest.mark.asyncio
    async def test_falsified_wrong_error_real(self, tmp_pred_dir):
        engine, mem = _real_engine(tmp_pred_dir)
        await mem.store(Prediction(
            trigger_input="'abc'", predicted_error_type="IndexError",
            source_failure_id="f1", source_goal="parse int",
        ))
        # Raises ValueError (inside solution), not the predicted IndexError.
        report = await engine.replay_for_failures(
            ["f1"], "def parse_int(x):\n    return int(x)\n", "parse int"
        )
        assert report.falsified == 1, report.to_dict()
        assert report.verdicts[0].actual_error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_off_topic_arity_real(self, tmp_pred_dir):
        engine, mem = _real_engine(tmp_pred_dir)
        await mem.store(Prediction(
            trigger_input="[]", predicted_error_type="TypeError",
            source_failure_id="f1", source_goal="add numbers",
        ))
        # Two required args; feeding [] (and splatted) errors at the call
        # boundary, not inside the function → off-topic, not a false confirm.
        report = await engine.replay_for_failures(
            ["f1"], "def add_numbers(a, b):\n    return a + b\n", "add numbers"
        )
        assert report.off_topic == 1, report.to_dict()
