"""
Tests for Milestone 2 (#1 + #2): only emit replayable predictions.

A prediction is worth storing only if the replay engine can actually score it —
i.e. the failure was a genuine test failure (not a harness/scaffolding error)
AND the prediction's trigger is a literal with a runtime exception type.
Storing anything else just accrues guaranteed Off-topic verdicts and skews the
calibration report.
"""

import json

import pytest

from agent.models import CodeArtifact, ErrorSignature, Plan, Prediction, TestResults
from agent.reflector import Reflector


# ---------------------------------------------------------------------------
# #2 — Prediction.is_replayable (schema-level).
# ---------------------------------------------------------------------------

class TestPredictionIsReplayable:
    def test_literal_trigger_runtime_error_is_replayable(self):
        for trigger in ("-1", "''", "[]", "None", "(None, 'a')", "0"):
            p = Prediction(trigger_input=trigger, predicted_error_type="ValueError")
            assert p.is_replayable() is True, trigger

    def test_non_literal_trigger_is_not_replayable(self):
        for trigger in ("x", "foo()", "open('f')", "1 + nope", "lambda: 1"):
            p = Prediction(trigger_input=trigger, predicted_error_type="ValueError")
            assert p.is_replayable() is False, trigger

    def test_import_or_compile_error_types_not_replayable(self):
        for etype in ("ImportError", "ModuleNotFoundError", "SyntaxError",
                      "IndentationError", "TabError"):
            p = Prediction(trigger_input="[]", predicted_error_type=etype)
            assert p.is_replayable() is False, etype

    def test_non_identifier_error_type_not_replayable(self):
        p = Prediction(trigger_input="[]", predicted_error_type="ValueError: bad")
        assert p.is_replayable() is False

    def test_malformed_is_not_replayable(self):
        assert Prediction(trigger_input="", predicted_error_type="ValueError").is_replayable() is False
        assert Prediction(trigger_input="[]", predicted_error_type="").is_replayable() is False


# ---------------------------------------------------------------------------
# #1 — Reflector._failure_is_replayable (failure-level gate).
# ---------------------------------------------------------------------------

class TestFailureIsReplayable:
    def test_genuine_pytest_failure_is_replayable(self):
        r = TestResults(passed=False, error_type="AssertionError",
                        tests_collected=2, from_pytest=True)
        assert Reflector._failure_is_replayable(r) is True

    def test_passing_is_not_replayable(self):
        r = TestResults(passed=True, tests_collected=2, from_pytest=True)
        assert Reflector._failure_is_replayable(r) is False

    @pytest.mark.parametrize("etype", [
        "NoTestsCollected", "ReportParseError", "TestGenerationError",
        "ModuleNotFoundError", "ImportError", "SyntaxError", "TimeoutError",
        "SafetyError", "UnsupportedLanguage",
    ])
    def test_harness_failures_are_not_replayable(self, etype):
        r = TestResults(passed=False, error_type=etype,
                        tests_collected=1, from_pytest=True)
        assert Reflector._failure_is_replayable(r) is False

    def test_pytest_run_with_zero_collected_is_not_replayable(self):
        r = TestResults(passed=False, error_type="AssertionError",
                        tests_collected=0, from_pytest=True)
        assert Reflector._failure_is_replayable(r) is False


# ---------------------------------------------------------------------------
# #2 — extract_predictions keeps only replayable predictions.
# ---------------------------------------------------------------------------

class _StubLLM:
    def __init__(self, response):
        self.response = response

    async def complete(self, prompt, system=None, temperature=0.7):
        return self.response


@pytest.mark.asyncio
async def test_extract_predictions_filters_non_replayable():
    llm = _StubLLM(json.dumps({"predictions": [
        {"trigger_input": "nope()", "predicted_error_type": "ValueError"},     # non-literal
        {"trigger_input": "[]", "predicted_error_type": "ModuleNotFoundError"},  # import-time
        {"trigger_input": "-1", "predicted_error_type": "ValueError"},          # ok
    ]}))
    reflector = Reflector(llm=llm)
    preds = await reflector.extract_predictions(
        code=CodeArtifact(source="def f(x): return x", file_path="solution.py", language="python"),
        plan=Plan(goal="f", steps=[], test_cases=[], language="python"),
        error_signature=ErrorSignature(error_type="ValueError", error_message="x"),
        source_failure_id="fid", task_id="tid",
    )
    assert len(preds) == 1
    assert preds[0].trigger_input == "-1"


# ---------------------------------------------------------------------------
# #1 — analyze() does not emit predictions for harness failures.
# ---------------------------------------------------------------------------

class _FakeFailureMemory:
    async def find_similar_failures(self, sig):
        return []

    async def store_failure(self, **kwargs):
        return "fid-1"


class _RecordingPredictionMemory:
    def __init__(self):
        self.stored = []

    async def store(self, pred):
        self.stored.append(pred)
        return "pid"


_REFLECTION_JSON = json.dumps({
    "success": False,
    "analysis": "wrong output",
    "root_cause": "logic",
    "suggested_fix": "fix it",
    "should_continue": True,
    "confidence": 0.5,
    "predictions": [
        {"trigger_input": "-1", "predicted_error_type": "ValueError", "confidence": 0.8},
    ],
})


def _reflector():
    return Reflector(
        llm=_StubLLM(_REFLECTION_JSON),
        failure_memory=_FakeFailureMemory(),
        prediction_memory=_RecordingPredictionMemory(),
    )


_CODE = CodeArtifact(source="def f(x):\n    return x\n", file_path="solution.py", language="python")
_PLAN = Plan(goal="f", steps=[], test_cases=[], language="python")


@pytest.mark.asyncio
async def test_analyze_skips_emission_on_harness_failure():
    reflector = _reflector()
    results = TestResults(passed=False, error_type="NoTestsCollected",
                          tests_collected=0, from_pytest=True)
    await reflector.analyze(test_results=results, code=_CODE, plan=_PLAN, iteration=1)
    assert reflector.prediction_memory.stored == []


@pytest.mark.asyncio
async def test_analyze_emits_on_genuine_failure():
    reflector = _reflector()
    results = TestResults(passed=False, error_type="AssertionError",
                          tests_collected=2, tests_passed=1, from_pytest=True)
    await reflector.analyze(test_results=results, code=_CODE, plan=_PLAN, iteration=1)
    assert len(reflector.prediction_memory.stored) == 1
    assert reflector.prediction_memory.stored[0].trigger_input == "-1"
