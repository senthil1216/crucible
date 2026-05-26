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
    Learning,
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
    async def test_project_type_boost_ranks_matching_first(self, temp_dir, fake_embeddings):
        """Soft-signal scoring: matching project_type boosts rank, not exclusion."""
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

        ranked = await memory.find_similar_solutions(
            "build a web app", k=5, min_similarity=0.0, project_type="fastapi"
        )
        # Both come back; fastapi ranks first because of the project_type bonus.
        assert len(ranked) == 2
        assert ranked[0]["project_type"] == "fastapi"
        assert ranked[1]["project_type"] == "cli_tool"
        assert ranked[0]["similarity"] > ranked[0]["base_similarity"]
        assert ranked[1]["similarity"] == ranked[1]["base_similarity"]

    @pytest.mark.asyncio
    async def test_dependency_overlap_boost_ranks_matching_first(self, temp_dir, fake_embeddings):
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

        ranked = await memory.find_similar_solutions(
            "serve an endpoint",
            k=5,
            min_similarity=0.0,
            dependencies=["fastapi"],
        )
        # Both come back; the fastapi entry ranks first because of overlap.
        assert len(ranked) == 2
        assert "fastapi" in ranked[0]["dependencies"]
        assert "flask" in ranked[1]["dependencies"]

    @pytest.mark.asyncio
    async def test_env_context_is_stored_and_returned(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        plan = Plan(goal="x", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="x", file_path="x.py", language="python")
        env = {
            "installed_packages": ["fastapi", "uvicorn", "pydantic"],
            "workspace_files": ["main.py", "requirements.txt"],
        }
        await memory.store_pattern("x", plan, code, environment_context=env)

        results = await memory.find_similar_solutions("x", k=1, min_similarity=0.0)
        assert results[0]["environment_context"]["installed_packages"] == [
            "fastapi", "uvicorn", "pydantic"
        ]

    @pytest.mark.asyncio
    async def test_installed_package_overlap_boosts_score(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        # Two patterns with identical goals but different env footprints.
        plan = Plan(goal="g", steps=[], test_cases=[], language="python")
        code = CodeArtifact(source="x", file_path="x.py", language="python")

        await memory.store_pattern(
            "g", plan, code,
            environment_context={"installed_packages": ["fastapi", "uvicorn"]},
        )
        await memory.store_pattern(
            "g", plan, code,
            environment_context={"installed_packages": ["flask"]},
        )

        ranked = await memory.find_similar_solutions(
            "g", k=5, min_similarity=0.0,
            installed_packages=["fastapi", "uvicorn", "pydantic"],
        )
        # The fastapi/uvicorn pattern overlaps with the query env and ranks first.
        assert "fastapi" in ranked[0]["environment_context"]["installed_packages"]
        assert ranked[0]["similarity"] > ranked[0]["base_similarity"]

    @pytest.mark.asyncio
    async def test_strict_filters_restore_exclusion(self, temp_dir, fake_embeddings):
        """strict_filters=True keeps the old hard-filter behavior."""
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        a = Plan(goal="g", steps=[], test_cases=[], language="python", project_type="fastapi")
        b = Plan(goal="g", steps=[], test_cases=[], language="python", project_type="cli_tool")
        code = CodeArtifact(source="x = 1", file_path="x.py", language="python")
        await memory.store_pattern("g", a, code)
        await memory.store_pattern("g", b, code)

        only = await memory.find_similar_solutions(
            "g", k=5, min_similarity=0.0, project_type="fastapi", strict_filters=True
        )
        assert len(only) == 1
        assert only[0]["project_type"] == "fastapi"

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


class TestLearningStorage:
    """Tests for Phase B: structured Learnings written by the Reflector."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_learning(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        learning = Learning(
            lesson="For FastAPI projects, expose a /health endpoint.",
            project_type="fastapi",
            language="python",
            tags=["fastapi", "health-check"],
            source_task_id="task_1",
            source_goal="build a web service",
        )
        learning_id = await memory.store_learning(learning)
        assert learning_id

        results = await memory.find_relevant_learnings(
            "build a web app with health checks",
            project_type="fastapi",
            min_similarity=0.0,
        )
        assert len(results) == 1
        assert "health" in results[0]["lesson"].lower()

    @pytest.mark.asyncio
    async def test_project_type_general_matches_any(self, temp_dir, fake_embeddings):
        memory = LongTermMemory(temp_dir, embedding_client=fake_embeddings)

        general_lesson = Learning(
            lesson="Always validate input before processing.",
            project_type="general",
            language="python",
        )
        fastapi_lesson = Learning(
            lesson="FastAPI dependency injection beats manual wiring.",
            project_type="fastapi",
            language="python",
        )
        await memory.store_learning(general_lesson)
        await memory.store_learning(fastapi_lesson)

        # Querying for fastapi should pull both: the fastapi lesson by
        # project_type, and the general lesson because "general" matches anything.
        results = await memory.find_relevant_learnings(
            "validate input in fastapi", project_type="fastapi", min_similarity=0.0
        )
        project_types = {r["project_type"] for r in results}
        assert "general" in project_types
        assert "fastapi" in project_types

    @pytest.mark.asyncio
    async def test_learnings_persist_across_instances(self, temp_dir, fake_embeddings):
        memory1 = LongTermMemory(temp_dir, embedding_client=fake_embeddings)
        await memory1.store_learning(Learning(
            lesson="Prefer csv.DictReader for CSV parsing.",
            project_type="general",
            language="python",
        ))

        # New instance should load the persisted learning.
        memory2 = LongTermMemory(temp_dir, embedding_client=fake_embeddings)
        results = await memory2.find_relevant_learnings(
            "parse a csv file", min_similarity=0.0
        )
        assert len(results) == 1


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Empty / mismatched inputs are safe
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
