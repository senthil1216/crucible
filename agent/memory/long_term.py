"""
Long-term memory: stores successful solution patterns with semantic + structured
retrieval.

Each stored Pattern carries the original goal/plan/code plus:
- goal_embedding: dense vector used for semantic similarity search
- project_type: filter signal (e.g. "fastapi", "cli_tool", "general")
- dependencies: filter signal for tasks that overlap on libraries

Retrieval combines semantic similarity (cosine over goal embeddings) with
optional structured filters.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from agent.models import Plan, CodeArtifact, MemoryEntry, Learning
from agent.memory.embeddings import EmbeddingClient, cosine_similarity


class LongTermMemory:
    """
    Stores and retrieves successful solution patterns.

    Storage backend is a JSONL file (`patterns.jsonl`). Entries written before
    embeddings existed are migrated transparently on load by computing and
    caching their goal embedding the first time they're read.
    """

    def __init__(
        self,
        storage_path: Path,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.patterns_file = storage_path / "patterns.jsonl"
        self.learnings_file = storage_path / "learnings.jsonl"
        self._embeddings = embedding_client or EmbeddingClient.shared()
        self._cache: List[MemoryEntry] = []
        self._learnings_cache: List[MemoryEntry] = []
        self._load_cache()
        self._load_learnings()

    def _load_cache(self) -> None:
        if not self.patterns_file.exists():
            return

        with open(self.patterns_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._cache.append(MemoryEntry(
                        id=data["id"],
                        content=data["content"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        metadata=data.get("metadata", {})
                    ))

    def _save_entry(self, entry: MemoryEntry) -> None:
        with open(self.patterns_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def _load_learnings(self) -> None:
        if not self.learnings_file.exists():
            return
        with open(self.learnings_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._learnings_cache.append(MemoryEntry(
                        id=data["id"],
                        content=data["content"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        metadata=data.get("metadata", {})
                    ))

    def _save_learning_entry(self, entry: MemoryEntry) -> None:
        with open(self.learnings_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def _embed(self, text: str) -> List[float]:
        return self._embeddings.encode(text)

    def _get_embedding(self, entry: MemoryEntry) -> List[float]:
        """Return the cached goal embedding, computing it lazily for legacy entries."""
        emb = entry.content.get("goal_embedding")
        if emb:
            return emb
        # Legacy entry: compute embedding now and cache in memory.
        # We intentionally do not rewrite the file here — backfill happens
        # the next time the entry is stored, keeping reads side-effect-free.
        goal = entry.content.get("goal", "")
        emb = self._embed(goal) if goal else []
        entry.content["goal_embedding"] = emb
        return emb

    async def store_pattern(
        self,
        goal: str,
        plan: Plan,
        code: CodeArtifact,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Store a successful solution pattern."""

        pattern_id = hashlib.sha256(
            f"{goal}:{code.source}".encode()
        ).hexdigest()[:16]

        content = {
            "goal": goal,
            "goal_embedding": self._embed(goal),
            "plan": plan.to_dict(),
            "code": code.to_dict(),
            "project_type": getattr(plan, "project_type", "general"),
            "dependencies": list(getattr(plan, "dependencies", []) or []),
            "language": getattr(plan, "language", "python"),
        }

        entry = MemoryEntry(
            id=pattern_id,
            content=content,
            metadata=metadata or {}
        )

        self._cache.append(entry)
        self._save_entry(entry)

        return pattern_id

    async def find_similar_solutions(
        self,
        goal: str,
        k: int = 3,
        min_similarity: float = 0.3,
        project_type: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar past solutions, scored by goal-embedding cosine similarity
        and optionally filtered by `project_type` and `dependencies` overlap.

        Args:
            goal: free-text goal for the new task
            k: maximum results to return
            min_similarity: cosine threshold (0..1); entries below are dropped
            project_type: if provided, only entries with the same project_type
            dependencies: if provided, only entries whose stored dependencies
                intersect this list (any overlap qualifies)
        """

        if not self._cache:
            return []

        query_emb = self._embed(goal)
        dep_filter = set(dependencies) if dependencies else None

        scored: List[tuple] = []
        for entry in self._cache:
            if project_type and entry.content.get("project_type") != project_type:
                continue

            if dep_filter:
                stored_deps = set(entry.content.get("dependencies") or [])
                if not (stored_deps & dep_filter):
                    continue

            entry_emb = self._get_embedding(entry)
            similarity = cosine_similarity(query_emb, entry_emb)
            if similarity >= min_similarity:
                scored.append((similarity, entry))

        scored.sort(reverse=True, key=lambda x: x[0])

        results: List[Dict[str, Any]] = []
        for score, entry in scored[:k]:
            results.append({
                "id": entry.id,
                "similarity": score,
                "goal": entry.content["goal"],
                "plan": entry.content["plan"],
                "code": entry.content["code"],
                "project_type": entry.content.get("project_type", "general"),
                "dependencies": entry.content.get("dependencies", []),
                "metadata": entry.metadata,
            })

        return results

    async def store_learning(self, learning: Learning) -> str:
        """Persist a reusable lesson extracted by the Reflector."""
        learning_id = hashlib.sha256(
            f"{learning.source_task_id}:{learning.lesson}".encode()
        ).hexdigest()[:16]

        content = {
            **learning.to_dict(),
            "lesson_embedding": self._embed(learning.lesson),
        }

        entry = MemoryEntry(id=learning_id, content=content)
        self._learnings_cache.append(entry)
        self._save_learning_entry(entry)
        return learning_id

    async def find_relevant_learnings(
        self,
        goal: str,
        project_type: Optional[str] = None,
        language: Optional[str] = None,
        k: int = 3,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve lessons relevant to a new task by semantic similarity, with
        optional structural filters.
        """
        if not self._learnings_cache:
            return []

        query_emb = self._embed(goal)

        scored: List[tuple] = []
        for entry in self._learnings_cache:
            if project_type and entry.content.get("project_type") not in (project_type, "general"):
                continue
            if language and entry.content.get("language") != language:
                continue

            emb = entry.content.get("lesson_embedding")
            if not emb:
                emb = self._embed(entry.content.get("lesson", ""))
                entry.content["lesson_embedding"] = emb
            similarity = cosine_similarity(query_emb, emb)
            if similarity >= min_similarity:
                scored.append((similarity, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        results: List[Dict[str, Any]] = []
        for score, entry in scored[:k]:
            results.append({
                "id": entry.id,
                "similarity": score,
                "lesson": entry.content["lesson"],
                "project_type": entry.content.get("project_type", "general"),
                "language": entry.content.get("language", "python"),
                "tags": entry.content.get("tags", []),
                "source_task_id": entry.content.get("source_task_id"),
            })
        return results

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_patterns": len(self._cache),
            "total_learnings": len(self._learnings_cache),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._learnings_cache.clear()
        if self.patterns_file.exists():
            self.patterns_file.unlink()
        if self.learnings_file.exists():
            self.learnings_file.unlink()
