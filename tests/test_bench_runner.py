"""Tests for the Track D phase-3 benchmark runner.

These exercise the batch loop with a stubbed agent — no real LLM, no Docker, no
solve(). They guard the JSONL contract that the phase-4 analysis depends on:
one record per (problem, rep), monotonic run_index, the replay_report payload,
and the invariant that memory is never reset mid-run.
"""

import json

import pytest

from agent.models import Status
from bench.problems import ProblemSpec
from bench import runner as bench_runner


class FakeState:
    """Duck-typed IterationState — only the attributes run_one reads."""

    def __init__(self, *, iteration, status, replay_report):
        self.iteration = iteration
        self.status = status
        self.replay_report = replay_report


# --------------------------------------------------------------------------- #
# Stubs                                                                         #
# --------------------------------------------------------------------------- #

class FakeLongTermMemory:
    def __init__(self):
        self.cleared = 0

    async def find_similar_solutions(self, goal, k=5):
        return [{
            "id": "pat-1",
            "similarity": 0.8,
            "score_breakdown": {
                "semantic": 0.6, "project_type_bonus": 0.15,
                "deps_bonus": 0.05, "package_bonus": 0.0,
            },
            "project_type": "general",
            "goal": "prior goal",
        }]

    async def find_relevant_learnings(self, goal, k=5):
        return [{"lesson": "watch the empty list", "source_task_id": "t0",
                 "similarity": 0.7}]

    def clear(self):
        self.cleared += 1


class FakePredictionMemory:
    async def find_relevant(self, goal, k=5):
        return [{
            "id": "pred-1", "trigger_input": "[]",
            "predicted_error_type": "IndexError", "confidence": 0.7,
            "similarity": 0.65, "times_tested": 2, "times_confirmed": 1,
        }]


class FakeAgent:
    """Minimal surface run_one / run_batch touch. Records solve() calls."""

    def __init__(self):
        self.long_term_memory = FakeLongTermMemory()
        self.prediction_memory = FakePredictionMemory()
        self.solve_calls = []

    async def solve(self, goal, task_id=None):
        self.solve_calls.append((goal, task_id))
        return FakeState(
            iteration=2,
            status=Status.SUCCESS,
            replay_report={
                "entry_point": "f", "confirmed": 1, "falsified": 0,
                "off_topic": 0, "tested": 1,
                "verdicts": [{
                    "prediction_id": "pred-1", "trigger_input": "[]",
                    "predicted_error_type": "IndexError", "classification": "confirmed",
                    "detail": "raised IndexError as predicted",
                    "actual_error_type": "IndexError",
                }],
            },
        )


def _problems(n):
    return [
        ProblemSpec(
            id=f"p-{i:02d}", goal=f"do thing {i} via solve_{i}",
            category="list", function_name=f"solve_{i}", adversarial_inputs=["[]"],
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_one_record_per_problem_rep(tmp_path):
    agent = FakeAgent()
    out = tmp_path / "out.jsonl"
    results = await bench_runner.run_batch(
        agent, _problems(3), reps=4, out_path=out, verbose=False
    )
    assert len(results) == 12               # 3 problems × 4 reps
    assert len(agent.solve_calls) == 12

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 12
    for line in lines:
        json.loads(line)                    # every line is valid JSON


@pytest.mark.asyncio
async def test_run_index_is_monotonic(tmp_path):
    agent = FakeAgent()
    out = tmp_path / "out.jsonl"
    results = await bench_runner.run_batch(
        agent, _problems(2), reps=3, out_path=out, verbose=False
    )
    assert [r["run_index"] for r in results] == list(range(6))


@pytest.mark.asyncio
async def test_record_schema_and_replay_report(tmp_path):
    agent = FakeAgent()
    out = tmp_path / "out.jsonl"
    results = await bench_runner.run_batch(
        agent, _problems(1), reps=1, out_path=out, verbose=False
    )
    rec = results[0]
    for key in (
        "problem_id", "rep", "run_index", "category", "function_name",
        "status", "iterations", "retrieved_patterns", "retrieved_learnings",
        "retrieved_predictions", "replay_report",
    ):
        assert key in rec, f"missing {key}"
    assert rec["status"] == "success"
    assert rec["iterations"] == 2
    # The replay verdicts survive into the record verbatim.
    assert rec["replay_report"]["confirmed"] == 1
    assert rec["retrieved_patterns"][0]["score_breakdown"]["project_type_bonus"] == 0.15
    assert rec["retrieved_predictions"][0]["predicted_error_type"] == "IndexError"


@pytest.mark.asyncio
async def test_memory_is_not_reset_between_runs(tmp_path):
    agent = FakeAgent()
    out = tmp_path / "out.jsonl"
    await bench_runner.run_batch(
        agent, _problems(3), reps=2, out_path=out, verbose=False
    )
    # The whole point of cross-rep persistence: memory.clear() is never called.
    assert agent.long_term_memory.cleared == 0


@pytest.mark.asyncio
async def test_task_ids_are_unique_per_rep(tmp_path):
    agent = FakeAgent()
    out = tmp_path / "out.jsonl"
    await bench_runner.run_batch(
        agent, _problems(2), reps=2, out_path=out, verbose=False
    )
    task_ids = [tid for _, tid in agent.solve_calls]
    assert len(task_ids) == len(set(task_ids))


@pytest.mark.asyncio
async def test_runs_without_prediction_memory(tmp_path):
    """prediction_memory may be disabled; run_one must tolerate None."""
    agent = FakeAgent()
    agent.prediction_memory = None
    out = tmp_path / "out.jsonl"
    results = await bench_runner.run_batch(
        agent, _problems(1), reps=1, out_path=out, verbose=False
    )
    assert results[0]["retrieved_predictions"] == []
