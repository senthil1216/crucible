"""
Tests for memory hierarchy.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from agent.memory import ShortTermMemory, LongTermMemory, FailureMemory
from agent.models import IterationState, Plan, CodeArtifact, TestResults, Reflection, ErrorSignature, Status


class TestShortTermMemory:
    """Tests for short-term memory."""
    
    def test_add_and_retrieve(self):
        memory = ShortTermMemory(max_history=3)
        
        # Create test states
        for i in range(5):
            state = IterationState(
                iteration=i,
                plan=Plan(goal="test", steps=[], test_cases=[]),
                code=CodeArtifact(source="code", file_path="test.py", language="python"),
                test_results=TestResults(passed=True),
                reflection=Reflection(success=True, analysis="ok"),
                status=Status.SUCCESS
            )
            memory.add(state)
        
        # Should only keep last 3
        assert len(memory) == 3
        assert memory.get_recent(1)[0].iteration == 4
    
    def test_error_history(self):
        memory = ShortTermMemory()
        
        # Add failed state
        state = IterationState(
            iteration=1,
            plan=Plan(goal="test", steps=[], test_cases=[]),
            code=CodeArtifact(source="code", file_path="test.py", language="python"),
            test_results=TestResults(
                passed=False,
                error_type="SyntaxError",
                stderr="invalid syntax"
            ),
            reflection=Reflection(success=False, analysis="syntax error"),
            status=Status.FAILED
        )
        memory.add(state)
        
        errors = memory.get_error_history()
        assert len(errors) == 1
        assert errors[0]["error_type"] == "SyntaxError"
    
    def test_repeating_errors(self):
        memory = ShortTermMemory()
        
        # Add same error 3 times
        for i in range(3):
            state = IterationState(
                iteration=i,
                plan=Plan(goal="test", steps=[], test_cases=[]),
                code=CodeArtifact(source="code", file_path="test.py", language="python"),
                test_results=TestResults(
                    passed=False,
                    error_type="SyntaxError"
                ),
                reflection=Reflection(success=False, analysis="error"),
                status=Status.FAILED
            )
            memory.add(state)
        
        assert memory.is_repeating_errors()


class TestLongTermMemory:
    """Tests for long-term memory."""
    
    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, temp_dir):
        memory = LongTermMemory(temp_dir)
        
        plan = Plan(
            goal="sort a list",
            steps=["implement sort"],
            test_cases=["test sort"],
            language="python"
        )
        code = CodeArtifact(
            source="def sort(arr): return sorted(arr)",
            file_path="sort.py",
            language="python"
        )
        
        pattern_id = await memory.store_pattern(
            goal="sort a list",
            plan=plan,
            code=code
        )
        
        assert pattern_id is not None
        assert len(memory._cache) == 1
    
    @pytest.mark.asyncio
    async def test_find_similar(self, temp_dir):
        memory = LongTermMemory(temp_dir)
        
        # Store pattern
        plan = Plan(
            goal="sort a list of numbers",
            steps=["implement sort"],
            test_cases=["test sort"],
            language="python"
        )
        code = CodeArtifact(
            source="def sort(arr): return sorted(arr)",
            file_path="sort.py",
            language="python"
        )
        
        await memory.store_pattern("sort a list of numbers", plan, code)
        
        # Find similar
        results = await memory.find_similar_solutions("sort numbers in a list", k=1)
        assert len(results) == 1
        assert results[0]["similarity"] > 0


class TestFailureMemory:
    """Tests for failure memory."""
    
    @pytest.fixture
    def temp_dir(self):
        path = Path(tempfile.mkdtemp())
        yield path
        shutil.rmtree(path)
    
    @pytest.mark.asyncio
    async def test_store_failure(self, temp_dir):
        memory = FailureMemory(temp_dir)
        
        error_sig = ErrorSignature(
            error_type="SyntaxError",
            error_message="invalid syntax at line 5"
        )
        code = CodeArtifact(
            source="def foo( print('hello')",
            file_path="test.py",
            language="python"
        )
        
        failure_id = await memory.store_failure(
            error_signature=error_sig,
            attempt=code,
            root_cause="missing parenthesis",
            fix="add closing parenthesis",
            goal="create a function"
        )
        
        assert failure_id is not None
        assert len(memory._cache) == 1
    
    @pytest.mark.asyncio
    async def test_find_similar_failures(self, temp_dir):
        memory = FailureMemory(temp_dir)
        
        # Store failure
        error_sig = ErrorSignature(
            error_type="NameError",
            error_message="name 'x' is not defined"
        )
        code = CodeArtifact(
            source="print(x)",
            file_path="test.py",
            language="python"
        )
        
        await memory.store_failure(
            error_signature=error_sig,
            attempt=code,
            root_cause="undefined variable",
            fix="define x before use",
            goal="print variable"
        )
        
        # Find similar
        query_sig = ErrorSignature(
            error_type="NameError",
            error_message="name 'y' is not defined"
        )
        
        results = await memory.find_similar_failures(query_sig)
        assert len(results) >= 1
