"""
Tests for the Reflector.
"""

import pytest

from agent.reflector import Reflector
from agent.models import Plan, CodeArtifact, TestResults


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


@pytest.mark.asyncio
async def test_pytest_failure_detail_drives_reflection_prompt_and_signature():
    llm = StubLearningLLM(
        '{"success": false, "analysis": "off by one", '
        '"root_cause": "wrong boundary", "suggested_fix": "return x + 1", '
        '"should_continue": true, "confidence": 0.8}'
    )
    reflector = Reflector(llm=llm)
    results = TestResults(
        passed=False,
        stdout="",
        stderr="",
        error_type="AssertionError",
        tests_collected=1,
        tests_failed=1,
        test_failures=[{
            "nodeid": "tests/test_solution.py::test_inc",
            "outcome": "failed",
            "message": "assert 1 == 2\n + where 1 = inc(0)",
        }],
        from_pytest=True,
    )

    reflection = await reflector.analyze(
        test_results=results,
        code=CodeArtifact(
            source="def inc(x):\n    return x\n",
            file_path="solution.py",
            language="python",
        ),
        plan=Plan(goal="increment", steps=[], test_cases=[]),
        iteration=1,
    )

    assert reflection.error_signature.error_message
    assert "assert 1 == 2" in reflection.error_signature.error_message
    assert "Most Actionable Failure Detail" in llm.last_prompt
    assert "tests/test_solution.py::test_inc" in llm.last_prompt
