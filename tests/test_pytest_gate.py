"""
Tests for the real-pytest success gate.

Covers four layers:
  1. The JSON-report parser (`build_test_results`) — the success rule.
  2. The TestGenerator vacuity helpers (static check, stub builder).
  3. The real pytest runner (`SandboxedExecutor.run_pytest`).
  4. The end-to-end loop gate: a correct impl passes; a hollow impl never
     passes; tests are frozen across fix iterations; vacuous/invalid suites are
     handled per policy.
"""

import json
import pytest

from agent.loop import ExecutionLoop
from agent.models import LoopConfig, Status
from agent.memory import ShortTermMemory
from agent.planner import Planner
from agent.code_generator import CodeGenerator
from agent.test_generator import (
    TestGenerator, static_check_test_code, build_stub_files, extract_imported_names,
)
from agent.tester import Tester
from agent.reflector import Reflector
from agent.executor.sandbox import SandboxedExecutor, ExecutionConfig
from agent.pytest_report import build_test_results


# ---------------------------------------------------------------------------
# 1. Report parser — the success rule lives here.
# ---------------------------------------------------------------------------

class TestAppLaunchHelpers:
    def test_detect_default_app_var(self):
        from agent.core import SelfImprovingAgent
        src = "from fastapi import FastAPI\napp = FastAPI()\n"
        assert SelfImprovingAgent._detect_fastapi_app_var(src) == "app"

    def test_detect_custom_app_var(self):
        from agent.core import SelfImprovingAgent
        src = "from fastapi import FastAPI\napi = FastAPI(title='x')\n"
        assert SelfImprovingAgent._detect_fastapi_app_var(src) == "api"

    def test_detect_falls_back_to_app(self):
        from agent.core import SelfImprovingAgent
        assert SelfImprovingAgent._detect_fastapi_app_var("def f(): pass") == "app"


class TestReportParser:
    def test_all_pass(self):
        report = json.dumps({
            "summary": {"passed": 3, "total": 3, "collected": 3},
            "tests": [
                {"nodeid": "t::test_a", "outcome": "passed"},
                {"nodeid": "t::test_b", "outcome": "passed"},
                {"nodeid": "t::test_c", "outcome": "passed"},
            ],
        })
        r = build_test_results(report, "", "", 0)
        assert r.passed is True
        assert r.tests_collected == 3
        assert r.tests_passed == 3
        assert r.tests_failed == 0
        assert r.from_pytest is True

    def test_failure_records_detail(self):
        report = json.dumps({
            "summary": {"passed": 1, "failed": 1, "total": 2, "collected": 2},
            "tests": [
                {"nodeid": "t::test_a", "outcome": "passed"},
                {"nodeid": "t::test_b", "outcome": "failed",
                 "call": {"longrepr": "assert 1 == 2\nAssertionError"}},
            ],
        })
        r = build_test_results(report, "", "", 1)
        assert r.passed is False
        assert r.tests_failed == 1
        assert r.failed_tests == ["t::test_b"]
        assert r.test_failures[0]["nodeid"] == "t::test_b"
        assert r.error_type == "AssertionError"

    def test_empty_suite_is_never_a_pass(self):
        report = json.dumps({"summary": {"total": 0, "collected": 0}, "tests": []})
        r = build_test_results(report, "", "", 5)
        assert r.passed is False
        assert r.tests_collected == 0
        assert r.error_type == "NoTestsCollected"

    def test_missing_report_fails_safe(self):
        r = build_test_results(None, "boom", "", 0)
        # Even with exit code 0, no report means we do NOT infer success.
        assert r.passed is False
        assert r.error_type == "ReportParseError"

    def test_collection_error(self):
        report = json.dumps({
            "summary": {"total": 0, "collected": 0},
            "collectors": [
                {"nodeid": "tests/test_solution.py", "outcome": "failed",
                 "longrepr": "ImportError: cannot import name 'add' from 'solution'"},
            ],
            "tests": [],
        })
        r = build_test_results(report, "", "", 2)
        assert r.passed is False
        assert r.tests_errors >= 1
        assert r.error_type == "ImportError"


# ---------------------------------------------------------------------------
# 2. Vacuity helpers.
# ---------------------------------------------------------------------------

class TestVacuityHelpers:
    GOOD = (
        "from solution import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n"
    )

    def test_static_check_accepts_good_suite(self):
        ok, reasons = static_check_test_code(self.GOOD)
        assert ok is True
        assert reasons == []

    def test_static_check_rejects_missing_import(self):
        ok, reasons = static_check_test_code("def test_x():\n    assert True\n")
        assert ok is False
        assert any("import" in r for r in reasons)

    def test_static_check_rejects_no_assertion(self):
        src = "from solution import add\n\ndef test_x():\n    add(1, 2)\n"
        ok, reasons = static_check_test_code(src)
        assert ok is False
        assert any("assert" in r for r in reasons)

    def test_static_check_rejects_syntax_error(self):
        ok, reasons = static_check_test_code("def test(:\n  pass")
        assert ok is False

    def test_static_check_rejects_server_launch(self):
        # The exact unsatisfiable anti-pattern Grok produced: launch the server
        # as a subprocess. Must be rejected so the suite gets regenerated.
        src = (
            "from solution import app\n"
            "import subprocess, sys\n\n"
            "def test_uvicorn_startup():\n"
            "    result = subprocess.run([sys.executable, '-m', 'uvicorn', 'solution:app'], timeout=3)\n"
            "    assert result.returncode != 1\n"
        )
        ok, reasons = static_check_test_code(src)
        assert ok is False
        assert any("uvicorn" in r or "subprocess" in r for r in reasons)

    def test_extract_imported_names(self):
        assert extract_imported_names(self.GOOD) == ["add"]
        attr = "import solution\n\ndef test_x():\n    assert solution.foo() == 1\n"
        assert extract_imported_names(attr) == ["foo"]

    def test_build_stub_files_makes_no_op_impl(self):
        stub = build_stub_files(self.GOOD)
        assert "solution.py" in stub
        assert "def add" in stub["solution.py"]
        assert "return None" in stub["solution.py"]


# ---------------------------------------------------------------------------
# 3. Real pytest runner.
# ---------------------------------------------------------------------------

def _executor():
    return SandboxedExecutor(config=ExecutionConfig(timeout_seconds=30, memory_limit_mb=512))


class TestSandboxRunPytest:
    @pytest.mark.asyncio
    async def test_passing_suite(self):
        files = {
            "solution.py": "def add(a, b):\n    return a + b\n",
            "tests/test_solution.py": (
                "from solution import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            ),
        }
        r = await _executor().run_pytest(files)
        assert r.passed is True
        assert r.tests_collected == 1
        assert r.tests_passed == 1

    @pytest.mark.asyncio
    async def test_failing_suite(self):
        files = {
            "solution.py": "def add(a, b):\n    return a - b\n",  # wrong
            "tests/test_solution.py": (
                "from solution import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            ),
        }
        r = await _executor().run_pytest(files)
        assert r.passed is False
        assert r.tests_failed == 1
        assert r.error_type == "AssertionError"

    @pytest.mark.asyncio
    async def test_missing_symbol_is_collection_error(self):
        files = {
            "solution.py": "x = 1\n",  # no `add`
            "tests/test_solution.py": (
                "from solution import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            ),
        }
        r = await _executor().run_pytest(files)
        assert r.passed is False
        # Either reported as a collection error count or zero tests collected.
        assert r.tests_collected == 0 or r.tests_errors >= 1


# ---------------------------------------------------------------------------
# 4. End-to-end loop gate.
# ---------------------------------------------------------------------------

PLAN_JSON = json.dumps({
    "steps": ["implement it"],
    "test_cases": ["it works"],
    "language": "python",
    "dependencies": [],
    "project_type": "general",
    "use_multi_file": False,
})

REFLECT_JSON = json.dumps({
    "success": False,
    "analysis": "tests failed",
    "root_cause": "logic",
    "suggested_fix": "fix the logic",
    "should_continue": True,
    "confidence": 0.5,
})


class ScriptedLLM:
    """Returns controlled plan/test/code/reflection responses and counts calls."""

    def __init__(self, *, impl_source, test_source, plan_json=PLAN_JSON):
        self.impl_source = impl_source
        self.test_source = test_source
        self.plan_json = plan_json
        self.calls = {"plan": 0, "test": 0, "code": 0, "reflect": 0}

    async def complete(self, prompt, system=None, temperature=0.7):
        s = (system or "").lower()
        if "reusable lessons" in s:
            return '{"learnings": []}'
        if "test author" in s or "Write a pytest test suite" in prompt:
            self.calls["test"] += 1
            return self.test_source
        if "Generate complete, runnable code" in prompt or "Please fix" in prompt:
            self.calls["code"] += 1
            return self.impl_source
        if "coding agent planner" in s or "Create a detailed plan" in prompt:
            self.calls["plan"] += 1
            return self.plan_json
        if "debugging" in s:
            self.calls["reflect"] += 1
            return REFLECT_JSON
        return "{}"


def _build_loop(llm, *, max_iterations=3, max_test_regenerations=2, dependency_manager=None):
    memory = ShortTermMemory()
    loop = ExecutionLoop(
        planner=Planner(llm),
        code_generator=CodeGenerator(llm),
        tester=Tester(executor=_executor()),
        reflector=Reflector(llm, failure_memory=None),
        short_term_memory=memory,
        config=LoopConfig(
            max_iterations=max_iterations,
            max_test_regenerations=max_test_regenerations,
            failure_threshold=99,  # don't let the circuit breaker interfere
        ),
        test_generator=TestGenerator(llm),
        dependency_manager=dependency_manager,
    )
    return loop, memory


class FakeDepManager:
    """Records eager-install calls; never triggers reactive recovery."""

    def __init__(self):
        self.installed = []

    def install_packages(self, deps):
        self.installed.extend(deps)
        class _R:
            success = True
            packages = list(deps)
            stderr = ""
        return _R()

    def should_attempt_recovery(self, msg):
        return False

    def extract_packages_from_error(self, msg):
        return []

    def reset_attempt_count(self):
        pass


ADD_TESTS = (
    "from solution import add\n\n"
    "def test_add_basic():\n    assert add(2, 3) == 5\n\n"
    "def test_add_zero():\n    assert add(0, 0) == 0\n"
)


class TestLoopGate:
    @pytest.mark.asyncio
    async def test_correct_impl_passes_real_pytest(self):
        llm = ScriptedLLM(
            impl_source="def add(a, b):\n    return a + b\n",
            test_source=ADD_TESTS,
        )
        loop, _ = _build_loop(llm)
        result = await loop.run("add two numbers", task_id="t-pass")

        assert result.status == Status.SUCCESS
        assert result.test_results.passed is True
        assert result.test_results.from_pytest is True
        assert result.test_results.tests_collected == 2
        # Tests were generated exactly once (frozen).
        assert llm.calls["test"] == 1

    @pytest.mark.asyncio
    async def test_hollow_impl_never_passes(self):
        # Implementation that does not satisfy the tests must NOT be a success,
        # even though the module imports and "runs" fine.
        llm = ScriptedLLM(
            impl_source="def add(a, b):\n    return 0\n",  # wrong
            test_source=ADD_TESTS,
        )
        loop, _ = _build_loop(llm, max_iterations=2)
        result = await loop.run("add two numbers", task_id="t-fail")

        assert result.status == Status.MAX_ITERATIONS
        assert result.test_results.passed is False
        assert result.test_results.tests_failed >= 1
        # Frozen: the suite was generated once and reused across the fix attempt.
        assert llm.calls["test"] == 1
        assert llm.calls["code"] == 2  # initial + one fix

    @pytest.mark.asyncio
    async def test_structurally_invalid_suite_is_fatal(self):
        # The generator only ever returns an unusable suite (no import, no assert).
        llm = ScriptedLLM(
            impl_source="def add(a, b):\n    return a + b\n",
            test_source="def test_nothing():\n    pass\n",
        )
        loop, _ = _build_loop(llm, max_test_regenerations=1)
        result = await loop.run("add two numbers", task_id="t-invalid")

        assert result.status == Status.FAILED
        assert result.test_results.error_type == "TestGenerationError"
        # initial + 1 regeneration attempt
        assert llm.calls["test"] == 2

    @pytest.mark.asyncio
    async def test_eager_install_runs_up_front(self):
        # Declared deps are installed once before iterating (not discovered
        # reactively via ImportError), so they don't burn circuit-breaker budget.
        plan_with_deps = json.dumps({
            "steps": ["implement it"],
            "test_cases": ["it works"],
            "language": "python",
            "dependencies": ["fastapi"],
            "project_type": "general",
            "use_multi_file": False,
        })
        llm = ScriptedLLM(
            impl_source="def add(a, b):\n    return a + b\n",
            test_source=ADD_TESTS,
            plan_json=plan_with_deps,
        )
        dep = FakeDepManager()
        loop, _ = _build_loop(llm, dependency_manager=dep)
        result = await loop.run("add two numbers", task_id="t-eager")

        assert dep.installed == ["fastapi"]   # installed eagerly, up front
        assert result.status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_vacuous_suite_warns_but_proceeds(self):
        # A structurally valid suite that passes against an empty stub: we can't
        # prove it has teeth, so we proceed with a warning rather than failing.
        vacuous_tests = (
            "from solution import greet\n\n"
            "def test_greet_callable():\n    assert callable(greet)\n"
        )
        llm = ScriptedLLM(
            impl_source="def greet():\n    return 'hi'\n",
            test_source=vacuous_tests,
        )
        loop, _ = _build_loop(llm, max_test_regenerations=1)
        result = await loop.run("greet", task_id="t-vacuous")

        assert result.status == Status.SUCCESS
        assert result.test_code is not None
        assert result.test_code.metadata.get("vacuity_warning") is True
