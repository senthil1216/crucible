"""
Tests for Track D phase 1: Prediction schema + emission + storage.

Predictions are the falsifiable-hypothesis seed of the self-improvement
loop. Phase 2 (replay engine) will write back times_tested /
times_confirmed; this file only covers the storage and emission paths.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent.memory.predictions import PredictionMemory
from agent.models import CodeArtifact, ErrorSignature, Plan, Prediction
from agent.reflector import Reflector


# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_predictions_dir():
    d = Path(tempfile.mkdtemp(prefix="crucible_pred_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class StubPredictionLLM:
    """LLM stub that returns a fixed JSON string for the prediction prompt."""

    def __init__(self, response: str):
        self.response = response
        self.last_system: str | None = None

    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        self.last_system = system
        return self.response


# ---------------------------------------------------------------------------
# Dataclass-level checks.
# ---------------------------------------------------------------------------


class TestPredictionDataclass:
    def test_round_trip(self):
        p = Prediction(
            trigger_input="-1",
            predicted_error_type="ValueError",
            predicted_explanation="Negative input not handled.",
            confidence=0.8,
            source_failure_id="fail123",
            source_task_id="task1",
            source_goal="compute square root",
            language="python",
        )
        restored = Prediction.from_dict(p.to_dict())
        assert restored.trigger_input == "-1"
        assert restored.predicted_error_type == "ValueError"
        assert restored.confidence == 0.8
        assert restored.source_failure_id == "fail123"
        assert restored.times_tested == 0
        assert restored.times_confirmed == 0

    def test_is_well_formed_requires_trigger_input(self):
        # Empty trigger_input fails the schema gate.
        p = Prediction(trigger_input="", predicted_error_type="ValueError")
        assert p.is_well_formed() is False

    def test_is_well_formed_requires_error_type(self):
        p = Prediction(trigger_input="-1", predicted_error_type="")
        assert p.is_well_formed() is False

    def test_well_formed_minimal(self):
        p = Prediction(trigger_input="-1", predicted_error_type="ValueError")
        assert p.is_well_formed() is True

    def test_confirmation_rate_smoothed(self):
        # Laplace prior: brand-new prediction sits at 0.5.
        p = Prediction(trigger_input="x", predicted_error_type="E")
        assert p.confirmation_rate() == pytest.approx(0.5)
        # 9/10 confirmed → smoothed to 10/12 ≈ 0.833, not 0.9.
        p.times_tested = 10
        p.times_confirmed = 9
        assert p.confirmation_rate() == pytest.approx(10 / 12)


# ---------------------------------------------------------------------------
# Storage.
# ---------------------------------------------------------------------------


class TestPredictionMemory:
    @pytest.mark.asyncio
    async def test_store_and_lookup_by_failure_id(self, tmp_predictions_dir):
        mem = PredictionMemory(tmp_predictions_dir)
        p = Prediction(
            trigger_input="-1",
            predicted_error_type="ValueError",
            source_failure_id="fail_abc",
        )
        pid = await mem.store(p)
        assert pid is not None

        results = mem.find_by_failure_id("fail_abc")
        assert len(results) == 1
        assert results[0]["trigger_input"] == "-1"
        assert results[0]["predicted_error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_schema_gate_rejects_malformed(self, tmp_predictions_dir):
        # No trigger_input → rejected; no row written.
        mem = PredictionMemory(tmp_predictions_dir)
        bad = Prediction(trigger_input="", predicted_error_type="ValueError")
        pid = await mem.store(bad)
        assert pid is None
        assert mem.find_by_failure_id("any") == []

    @pytest.mark.asyncio
    async def test_duplicates_dedupe(self, tmp_predictions_dir):
        # Same (failure_id, trigger_input) → same id → single entry.
        mem = PredictionMemory(tmp_predictions_dir)
        p1 = Prediction(
            trigger_input="-1",
            predicted_error_type="ValueError",
            source_failure_id="fail_x",
        )
        p2 = Prediction(
            trigger_input="-1",
            predicted_error_type="ValueError",
            source_failure_id="fail_x",
            predicted_explanation="different explanation, same trigger",
        )
        id1 = await mem.store(p1)
        id2 = await mem.store(p2)
        assert id1 == id2
        assert len(mem.find_by_failure_id("fail_x")) == 1

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_predictions_dir):
        mem1 = PredictionMemory(tmp_predictions_dir)
        await mem1.store(Prediction(
            trigger_input="None",
            predicted_error_type="TypeError",
            source_failure_id="ff",
        ))

        mem2 = PredictionMemory(tmp_predictions_dir)
        assert len(mem2.find_by_failure_id("ff")) == 1

    def test_record_tested_and_confirmed_phase2_hooks(self, tmp_predictions_dir):
        # Phase-2 will own this path; we just verify the plumbing.
        mem = PredictionMemory(tmp_predictions_dir)
        import asyncio
        pid = asyncio.run(mem.store(Prediction(
            trigger_input="[]",
            predicted_error_type="IndexError",
            source_failure_id="ff",
        )))
        assert mem.record_tested([pid]) == 1
        assert mem.record_confirmed([pid]) == 1
        results = mem.find_by_failure_id("ff")
        assert results[0]["times_tested"] == 1
        assert results[0]["times_confirmed"] == 1


# ---------------------------------------------------------------------------
# Reflector emission.
# ---------------------------------------------------------------------------


class TestReflectorExtractPredictions:
    @pytest.mark.asyncio
    async def test_emits_well_formed_predictions(self):
        llm = StubPredictionLLM(json.dumps({
            "predictions": [
                {
                    "trigger_input": "-1",
                    "predicted_error_type": "ValueError",
                    "predicted_explanation": "Negative not handled.",
                    "confidence": 0.8,
                },
                {
                    "trigger_input": "''",
                    "predicted_error_type": "IndexError",
                    "predicted_explanation": "Empty string indexed.",
                    "confidence": 0.6,
                },
            ]
        }))
        reflector = Reflector(llm=llm)
        plan = Plan(goal="compute square root", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="def sqrt(x): return x**0.5", file_path="m.py", language="python")
        err = ErrorSignature(error_type="ValueError", error_message="math domain error")

        preds = await reflector.extract_predictions(
            code=code, plan=plan, error_signature=err,
            source_failure_id="fail42", task_id="task_q",
        )
        assert len(preds) == 2
        assert preds[0].trigger_input == "-1"
        assert preds[0].source_failure_id == "fail42"
        assert preds[0].source_goal == "compute square root"

    @pytest.mark.asyncio
    async def test_drops_predictions_missing_trigger_input(self):
        # Schema gate: predictions without trigger_input are dropped.
        llm = StubPredictionLLM(json.dumps({
            "predictions": [
                {"predicted_error_type": "ValueError"},  # no trigger_input
                {"trigger_input": "  ", "predicted_error_type": "TypeError"},  # whitespace
                {"trigger_input": "-1", "predicted_error_type": "ValueError"},  # ok
            ]
        }))
        reflector = Reflector(llm=llm)
        plan = Plan(goal="x", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="pass", file_path="x.py", language="python")
        err = ErrorSignature(error_type="ValueError", error_message="x")

        preds = await reflector.extract_predictions(
            code=code, plan=plan, error_signature=err,
            source_failure_id="ff", task_id="tt",
        )
        assert len(preds) == 1
        assert preds[0].trigger_input == "-1"

    @pytest.mark.asyncio
    async def test_caps_at_three(self):
        items = [
            {"trigger_input": f"x{i}", "predicted_error_type": "ValueError"}
            for i in range(10)
        ]
        llm = StubPredictionLLM(json.dumps({"predictions": items}))
        reflector = Reflector(llm=llm)
        plan = Plan(goal="x", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="pass", file_path="x.py", language="python")
        err = ErrorSignature(error_type="ValueError", error_message="x")

        preds = await reflector.extract_predictions(
            code=code, plan=plan, error_signature=err,
            source_failure_id="ff", task_id="tt",
        )
        assert len(preds) == 3

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_empty(self):
        llm = StubPredictionLLM("this is not json at all")
        reflector = Reflector(llm=llm)
        plan = Plan(goal="x", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="pass", file_path="x.py", language="python")
        err = ErrorSignature(error_type="ValueError", error_message="x")

        preds = await reflector.extract_predictions(
            code=code, plan=plan, error_signature=err,
            source_failure_id="ff", task_id="tt",
        )
        assert preds == []

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self):
        class BrokenLLM:
            async def complete(self, prompt, system=None, temperature=0.7):
                raise RuntimeError("network down")

        reflector = Reflector(llm=BrokenLLM())
        plan = Plan(goal="x", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="pass", file_path="x.py", language="python")
        err = ErrorSignature(error_type="ValueError", error_message="x")

        preds = await reflector.extract_predictions(
            code=code, plan=plan, error_signature=err,
            source_failure_id="ff", task_id="tt",
        )
        assert preds == []
