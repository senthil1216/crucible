"""
Tests for the Planner — focused on Phase C Learning surfacing.
"""

import pytest

from agent.planner import Planner


class RecordingLLM:
    """LLM stub that records the prompt and returns a fixed plan JSON."""

    def __init__(self):
        self.last_prompt = None
        self.last_system = None

    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        self.last_prompt = prompt
        self.last_system = system
        return (
            '{"steps": ["step 1"], "test_cases": ["test 1"], '
            '"language": "python", "dependencies": [], '
            '"estimated_complexity": "low", "project_type": "general", '
            '"use_multi_file": false}'
        )


@pytest.mark.asyncio
async def test_planner_surfaces_learnings_into_prompt():
    llm = RecordingLLM()
    planner = Planner(llm=llm)

    learnings = [
        {"lesson": "For FastAPI projects, expose a /health endpoint."},
        {"lesson": "Prefer pathlib over os.path."},
    ]

    plan = await planner.create_plan(
        goal="build a service",
        relevant_learnings=learnings,
    )

    assert plan.goal == "build a service"
    assert plan.context["learnings_surfaced"] == 2
    assert "Relevant lessons" in llm.last_prompt
    assert "/health endpoint" in llm.last_prompt
    assert "pathlib" in llm.last_prompt


@pytest.mark.asyncio
async def test_planner_works_without_learnings_or_solutions():
    llm = RecordingLLM()
    planner = Planner(llm=llm)

    plan = await planner.create_plan(goal="solve x")

    assert plan.goal == "solve x"
    assert plan.context["learnings_surfaced"] == 0
    assert plan.context["similar_examples_used"] == 0
    assert "Relevant lessons" not in llm.last_prompt


@pytest.mark.asyncio
async def test_planner_caps_surfaced_learnings_at_five():
    llm = RecordingLLM()
    planner = Planner(llm=llm)

    learnings = [{"lesson": f"lesson number {i}"} for i in range(10)]
    await planner.create_plan(goal="x", relevant_learnings=learnings)

    # The first five should appear in the prompt; the rest should not.
    for i in range(5):
        assert f"lesson number {i}" in llm.last_prompt
    for i in range(5, 10):
        assert f"lesson number {i}" not in llm.last_prompt
