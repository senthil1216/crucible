"""
Tests for execution loop.
"""

import pytest
import asyncio
from pathlib import Path

from agent.loop import ExecutionLoop, CircuitBreaker
from agent.models import (
    LoopConfig, Plan, CodeArtifact, TestResults, 
    Reflection, IterationState, Status
)
from agent.memory import ShortTermMemory


class TestCircuitBreaker:
    """Tests for circuit breaker."""
    
    def test_circuit_closes_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record successes
        for _ in range(5):
            cb.record_result(success=True)
        
        assert cb.can_execute() is True
        assert cb.get_state() == "CLOSED"
    
    def test_circuit_opens_on_failures(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window=5)
        
        # Record failures
        for _ in range(3):
            cb.record_result(success=False)
        
        assert cb.is_open is True
        assert cb.get_state() == "OPEN"
        assert cb.can_execute() is False
    
    def test_circuit_cooldown(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_period=0)
        
        # Open circuit
        cb.record_result(success=False)
        cb.record_result(success=False)
        assert cb.is_open
        
        # Should close immediately with 0 cooldown
        assert cb.can_execute() is True


class MockPlanner:
    async def create_plan(self, goal, similar_solutions=None):
        return Plan(
            goal=goal,
            steps=["step 1", "step 2"],
            test_cases=["test 1"],
            language="python"
        )
    
    async def refine_plan(self, plan, reflection, previous_attempts):
        return plan


class MockCodeGenerator:
    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.call_count = 0
    
    async def generate(self, plan, previous_attempt=None, error_feedback=None):
        self.call_count += 1
        return CodeArtifact(
            source="print('hello')",
            file_path="test.py",
            language="python"
        )
    
    async def generate_fix(self, plan, broken_code, error_type, error_message, reflection):
        return await self.generate(plan)


class MockTester:
    def __init__(self, should_pass=True):
        self.should_pass = should_pass
    
    async def run_tests(self, code, plan):
        return TestResults(
            passed=self.should_pass,
            stdout="hello" if self.should_pass else "",
            stderr="" if self.should_pass else "error"
        )


class MockReflector:
    def __init__(self, should_continue=True):
        self.should_continue = should_continue
    
    async def analyze(self, test_results, code, plan, iteration, previous_reflections=None):
        return Reflection(
            success=test_results.passed,
            analysis="success" if test_results.passed else "failed",
            should_continue=self.should_continue if not test_results.passed else False
        )


class TestExecutionLoop:
    """Tests for execution loop."""
    
    @pytest.mark.asyncio
    async def test_successful_run(self):
        memory = ShortTermMemory()
        loop = ExecutionLoop(
            planner=MockPlanner(),
            code_generator=MockCodeGenerator(should_succeed=True),
            tester=MockTester(should_pass=True),
            reflector=MockReflector(),
            short_term_memory=memory,
            config=LoopConfig(max_iterations=5)
        )
        
        result = await loop.run("create hello world")
        
        assert result.status == Status.SUCCESS
        assert result.test_results.passed is True
        assert len(memory) == 1
    
    @pytest.mark.asyncio
    async def test_max_iterations(self):
        memory = ShortTermMemory()
        loop = ExecutionLoop(
            planner=MockPlanner(),
            code_generator=MockCodeGenerator(),
            tester=MockTester(should_pass=False),  # Always fail
            reflector=MockReflector(should_continue=True),
            short_term_memory=memory,
            config=LoopConfig(max_iterations=3)
        )
        
        result = await loop.run("impossible task")
        
        assert result.status == Status.MAX_ITERATIONS
        assert result.iteration == 3
    
    @pytest.mark.asyncio
    async def test_early_stop(self):
        memory = ShortTermMemory()
        loop = ExecutionLoop(
            planner=MockPlanner(),
            code_generator=MockCodeGenerator(),
            tester=MockTester(should_pass=False),
            reflector=MockReflector(should_continue=False),  # Stop early
            short_term_memory=memory,
            config=LoopConfig(max_iterations=10)
        )
        
        result = await loop.run("hopeless task")
        
        assert result.status == Status.FAILED
        assert result.iteration == 1  # Stopped after first iteration
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        memory = ShortTermMemory()
        loop = ExecutionLoop(
            planner=MockPlanner(),
            code_generator=MockCodeGenerator(),
            tester=MockTester(should_pass=False),
            reflector=MockReflector(should_continue=True),
            short_term_memory=memory,
            config=LoopConfig(
                max_iterations=10,
                failure_threshold=2,
                failure_window=3
            )
        )
        
        result = await loop.run("failing task")
        
        # Should stop due to circuit breaker
        assert result.status == Status.CIRCUIT_BREAKER
