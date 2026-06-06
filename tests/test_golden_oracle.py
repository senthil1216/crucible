"""
Tests for caller-supplied (golden) frozen test suites.

The benchmark injects a hand-written pytest suite as the success oracle instead
of letting the agent generate its own tests. These tests pin that contract:

  - an injected suite is used verbatim; the LLM test generator is NOT called;
  - a correct implementation passes the injected (golden) suite;
  - a wrong implementation does not pass it (the oracle has teeth);
  - a structurally invalid injected suite fails fast, it is not silently ignored.
"""

import pytest

from agent.loop import ExecutionLoop
from agent.models import LoopConfig, Status, Plan, CodeArtifact
from agent.memory import ShortTermMemory
from agent.planner import Planner
from agent.code_generator import CodeGenerator
from agent.tester import Tester
from agent.reflector import Reflector
from agent.test_generator import TestGenerator, TEST_FILE_PATH
from agent.executor.sandbox import SandboxedExecutor, ExecutionConfig


GOLDEN_REVERSE = (
    "from solution import reverse_words\n\n"
    "def test_basic():\n    assert reverse_words('hello world') == 'world hello'\n\n"
    "def test_collapse_and_strip():\n"
    "    assert reverse_words('  multiple   spaces  ') == 'spaces multiple'\n"
)


class ScriptedLLM:
    """Returns a fixed implementation; counts test-generation calls (must stay 0
    when a golden suite is injected)."""

    def __init__(self, impl_source):
        self.impl_source = impl_source
        self.test_calls = 0

    async def complete(self, prompt, system=None, temperature=0.7):
        s = (system or "").lower()
        if "reusable lessons" in s:
            return '{"learnings": []}'
        if "test author" in s or "Write a pytest test suite" in prompt:
            self.test_calls += 1
            return "def test_placeholder():\n    assert True\n"
        if "Generate complete, runnable code" in prompt or "Please fix" in prompt:
            return self.impl_source
        if "debugging" in s:
            return (
                '{"success": false, "analysis": "x", "root_cause": "y", '
                '"suggested_fix": "z", "should_continue": true, "confidence": 0.5}'
            )
        return "{}"


def _executor():
    return SandboxedExecutor(config=ExecutionConfig(timeout_seconds=30, memory_limit_mb=512))


def _build_loop(llm, *, max_iterations=2):
    loop = ExecutionLoop(
        planner=Planner(llm),
        code_generator=CodeGenerator(llm),
        tester=Tester(executor=_executor()),
        reflector=Reflector(llm, failure_memory=None),
        short_term_memory=ShortTermMemory(),
        config=LoopConfig(max_iterations=max_iterations, failure_threshold=99),
        test_generator=TestGenerator(llm),
    )
    return loop


def _plan():
    return Plan(
        goal="reverse words",
        steps=["implement reverse_words"],
        test_cases=["it works"],
        language="python",
        dependencies=[],
    )


def _golden(source=GOLDEN_REVERSE):
    return CodeArtifact(source=source, file_path=TEST_FILE_PATH, language="python")


class TestGoldenOracle:
    @pytest.mark.asyncio
    async def test_correct_impl_passes_injected_suite_without_generating_tests(self):
        llm = ScriptedLLM("def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n")
        loop = _build_loop(llm)
        result = await loop.run(
            "reverse words", task_id="g-pass", plan=_plan(), frozen_tests=_golden()
        )
        assert result.status == Status.SUCCESS
        assert result.test_results.passed is True
        assert result.test_results.tests_collected == 2
        # The injected suite was used verbatim — no test generation happened.
        assert llm.test_calls == 0

    @pytest.mark.asyncio
    async def test_wrong_impl_does_not_pass_injected_suite(self):
        # Identity function: leaves whitespace/order unchanged -> fails the oracle.
        llm = ScriptedLLM("def reverse_words(s):\n    return s\n")
        loop = _build_loop(llm, max_iterations=2)
        result = await loop.run(
            "reverse words", task_id="g-fail", plan=_plan(), frozen_tests=_golden()
        )
        assert result.status != Status.SUCCESS
        assert result.test_results.passed is False
        assert llm.test_calls == 0

    @pytest.mark.asyncio
    async def test_structurally_invalid_injected_suite_fails_fast(self):
        # No import of the solution module -> static check must reject it.
        bad = _golden("def test_x():\n    assert True\n")
        llm = ScriptedLLM("def reverse_words(s):\n    return s\n")
        loop = _build_loop(llm)
        result = await loop.run(
            "reverse words", task_id="g-bad", plan=_plan(), frozen_tests=bad
        )
        assert result.status == Status.FAILED
        assert result.test_results.error_type == "TestGenerationError"
        assert llm.test_calls == 0
