"""
Tests for memory hierarchy.
"""

import hashlib
import math
from pathlib import Path
import shutil
import tempfile
from typing import List

import pytest

from agent.memory import ShortTermMemory, LongTermMemory, FailureMemory
from agent.memory.embeddings import EmbeddingClient, cosine_similarity
from agent.models import (
    IterationState,
    Plan,
    CodeArtifact,
    TestResults,
    Reflection,
    ErrorSignature,
    Status,
)


class FakeEmbeddingClient(EmbeddingClient):
    """
    Deterministic, model-free embedding for tests.

    Strategy: split text into tokens, hash each token into a fixed-size
    vector position. Similar texts share many tokens → higher cosine.
    Fast: no network, no torch.
    """

    DIM = 64

    def __init__(self):
        super().__init__(model_name="fake")

    def _ensure_loaded(self) -> None:  # no-op
        return

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        for token in text.lower().split():
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class TestShortTermMemory:
    """Tests for short-term memory."""

    def test_add_and_retrieve(self):
        memory = ShortTermMemory(max_history=3)

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

        assert len(memory) == 3
        assert memory.get_recent(1)[0].iteration == 4

    def test_error_history(self):
        memory = ShortTermMemory()

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


@pytest.fixture
def temp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddingClient()


class TestLongTermMemory:
    """Tests for long-term memory with semantic + structured retrieval."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

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
        # Embedding is persisted alongside the goal
        stored = memory._cache[0].content
        assert "goal_embedding" in stored
        assert len(stored["goal_embedding"]) == FakeEmbeddingClient.DIM

    @pytest.mark.asyncio
    async def test_find_similar(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

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

        results = await memory.find_similar_solutions(
            "sort a list of numbers", k=1, min_similarity=0.0
        )
        assert len(results) == 1
        assert results[0]["similarity"] > 0

    @pytest.mark.asyncio
    async def test_filter_by_project_type(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        web_plan = Plan(
            goal="build a web app",
            steps=[], test_cases=[], language="python",
            project_type="fastapi",
        )
        cli_plan = Plan(
            goal="build a web app",   # same goal text on purpose
            steps=[], test_cases=[], language="python",
            project_type="cli_tool",
        )
        code = CodeArtifact(source="x = 1", file_path="x.py", language="python")

        await memory.store_pattern("build a web app", web_plan, code)
        await memory.store_pattern("build a web app", cli_plan, code)

        only_web = await memory.find_similar_solutions(
            "build a web app", k=5, min_similarity=0.0, project_type="fastapi"
        )
        assert len(only_web) == 1
        assert only_web[0]["project_type"] == "fastapi"

        only_cli = await memory.find_similar_solutions(
            "build a web app", k=5, min_similarity=0.0, project_type="cli_tool"
        )
        assert len(only_cli) == 1
        assert only_cli[0]["project_type"] == "cli_tool"

    @pytest.mark.asyncio
    async def test_filter_by_dependencies(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        fastapi_plan = Plan(
            goal="serve an endpoint",
            steps=[], test_cases=[], language="python",
            dependencies=["fastapi", "uvicorn"],
        )
        flask_plan = Plan(
            goal="serve an endpoint",
            steps=[], test_cases=[], language="python",
            dependencies=["flask"],
        )
        code = CodeArtifact(source="x = 1", file_path="x.py", language="python")

        await memory.store_pattern("serve an endpoint", fastapi_plan, code)
        await memory.store_pattern("serve an endpoint", flask_plan, code)

        results = await memory.find_similar_solutions(
            "serve an endpoint",
            k=5,
            min_similarity=0.0,
            dependencies=["fastapi"],
        )
        assert len(results) == 1
        assert "fastapi" in results[0]["dependencies"]

    @pytest.mark.asyncio
    async def test_legacy_entry_without_embedding(self, temp_dir, fake_embeddings):
        """Entries written before embeddings should still be retrievable."""
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        # Write a legacy-shaped entry directly (no goal_embedding field).
        import json
        from datetime import datetime
        legacy = {
            "id": "legacy01",
            "content": {
                "goal": "sort a list",
                "plan": {"goal": "sort a list", "steps": [], "test_cases": [], "language": "python", "dependencies": [], "context": {}, "project_type": "general", "use_multi_file": False},
                "code": {"source": "sorted(x)", "file_path": "s.py", "language": "python", "metadata": {}},
                "keywords": ["sort", "list"],
            },
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
        }
        with open(temp_dir / "patterns.jsonl", "w") as f:
            f.write(json.dumps(legacy) + "\n")

        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)
        results = await memory.find_similar_solutions(
            "sort a list", k=1, min_similarity=0.0
        )
        assert len(results) == 1
        # Backfilled embedding now lives on the cached entry
        assert memory._cache[0].content.get("goal_embedding")


class TestFailureMemory:
    """Tests for failure memory with semantic retrieval."""

    @pytest.mark.asyncio
    async def test_store_failure(self, temp_dir, fake_embeddings):
        memory = FailureMemory(temp_dir, embedding_client=fake_embeddings)

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
        assert "error_embedding" in memory._cache[0].content

    @pytest.mark.asyncio
    async def test_find_similar_failures(self, temp_dir, fake_embeddings):
        memory = FailureMemory(temp_dir, embedding_client=fake_embeddings)

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

        # A similar NameError should match
        query_sig = ErrorSignature(
            error_type="NameError",
            error_message="name 'y' is not defined"
        )
        results = await memory.find_similar_failures(query_sig)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_legacy_failure_without_embedding(self, temp_dir, fake_embeddings):
        """A failure entry without an embedding should still be retrievable."""
        import json
        from datetime import datetime

        legacy = {
            "id": "legfail1",
            "content": {
                "error_signature": {"error_type": "NameError", "error_message": "name 'x' is not defined"},
                "error_key": "NameError:name '{var}' is not defined",
                "attempt_summary": "print(x)",
                "root_cause": "undefined variable",
                "fix": "define x",
                "goal": "print variable",
                "was_fixed": False,
                "language": "python",
            },
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
        }
        with open(temp_dir / "failures.jsonl", "w") as f:
            f.write(json.dumps(legacy) + "\n")

        memory = FailureMemory(temp_dir, embedding_client=fake_embeddings)
        query_sig = ErrorSignature(
            error_type="NameError",
            error_message="name 'y' is not defined"
        )
        results = await memory.find_similar_failures(query_sig)
        assert len(results) >= 1
        assert memory._cache[0].content.get("error_embedding")


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Empty / mismatched inputs are safe
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
