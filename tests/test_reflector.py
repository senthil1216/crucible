"""
Tests for the Reflector.
"""

import pytest

from agent.reflector import Reflector
from agent.models import Plan, CodeArtifact


class StubLearningLLM:
    """LLM stub that returns a fixed Learning JSON for the learning prompt."""

    def __init__(self, response: str):
        self.response = response
        self.last_system = None
        self.last_prompt = None

    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        self.last_system = system
        self.last_prompt = prompt
        return self.response


@pytest.mark.asyncio
async def test_extract_learnings_returns_structured_learnings():
    llm = StubLearningLLM(
        '{"learnings": [{"lesson": "Use ast.parse for syntax-checking Python.", "tags": ["ast", "python"]}, '
        '{"lesson": "Prefer pathlib over os.path.", "tags": []}]}'
    )
    reflector = Reflector(llm=llm)

    plan = Plan(
        goal="validate python source",
        steps=["parse with ast"],
        test_cases=[],
        language="python",
        project_type="cli_tool",
    )
    code = CodeArtifact(
        source="import ast\nast.parse(src)\n",
        file_path="check.py",
        language="python",
    )

    learnings = await reflector.extract_learnings(plan=plan, code=code, task_id="task_42")

    assert len(learnings) == 2
    assert learnings[0].lesson.startswith("Use ast.parse")
    assert learnings[0].project_type == "cli_tool"
    assert learnings[0].language == "python"
    assert learnings[0].source_task_id == "task_42"
    assert learnings[0].source_goal == "validate python source"
    assert "ast" in learnings[0].tags


@pytest.mark.asyncio
async def test_extract_learnings_handles_unparseable_response():
    llm = StubLearningLLM("this is not json at all")
    reflector = Reflector(llm=llm)

    plan = Plan(goal="x", steps=[], test_cases=[], language="python")
    code = CodeArtifact(source="pass", file_path="x.py", language="python")

    learnings = await reflector.extract_learnings(plan=plan, code=code)
    assert learnings == []


@pytest.mark.asyncio
async def test_extract_learnings_caps_at_three():
    items = ",".join('{"lesson": "lesson ' + str(i) + '"}' for i in range(10))
    llm = StubLearningLLM('{"learnings": [' + items + ']}')
    reflector = Reflector(llm=llm)

    plan = Plan(goal="x", steps=[], test_cases=[], language="python")
    code = CodeArtifact(source="pass", file_path="x.py", language="python")

    learnings = await reflector.extract_learnings(plan=plan, code=code)
    assert len(learnings) == 3
