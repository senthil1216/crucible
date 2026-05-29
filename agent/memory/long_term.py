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

    def _rewrite_learnings_to_disk(self) -> None:
        """Rewrite learnings.jsonl from the in-memory cache. Used for
        in-place updates to existing entries (usefulness counters). The
        cache is the source of truth; mirrors FailureMemory._rewrite_*."""
        tmp = self.learnings_file.with_suffix(".jsonl.tmp")
        with open(tmp, 'w') as f:
            for entry in self._learnings_cache:
                f.write(json.dumps(entry.to_dict()) + '\n')
        tmp.replace(self.learnings_file)

    def record_learnings_retrieved(self, learning_ids: List[str]) -> int:
        """Bump times_retrieved on the named entries. Called by the loop
        when Learnings are surfaced to the Planner. Returns the number of
        entries actually updated."""
        if not learning_ids:
            return 0
        wanted = set(learning_ids)
        updated = 0
        for entry in self._learnings_cache:
            if entry.id in wanted:
                entry.content["times_retrieved"] = int(
                    entry.content.get("times_retrieved", 0) or 0
                ) + 1
                updated += 1
        if updated:
            self._rewrite_learnings_to_disk()
        return updated

    def record_learnings_helpful(self, learning_ids: List[str]) -> int:
        """Bump times_helpful on the named entries. Called by the loop on
        task success for the Learnings that were in scope at plan time.
        Note: correlation, not causation — the LLM may have ignored the
        lesson. We accept the noise; the Laplace prior in helpfulness_rate
        prevents one-off boosts from dominating."""
        if not learning_ids:
            return 0
        wanted = set(learning_ids)
        updated = 0
        for entry in self._learnings_cache:
            if entry.id in wanted:
                entry.content["times_helpful"] = int(
                    entry.content.get("times_helpful", 0) or 0
                ) + 1
                updated += 1
        if updated:
            self._rewrite_learnings_to_disk()
        return updated

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
        environment_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a successful solution pattern.

        environment_context (Phase D): optional snapshot of the runtime env
        that produced this success, e.g.
            {"installed_packages": [...], "workspace_files": [...]}
        Captured from the executor when available. Stored verbatim and used
        as an additional retrieval signal by find_similar_solutions.
        """

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
            "environment_context": environment_context or {},
        }

        entry = MemoryEntry(
            id=pattern_id,
            content=content,
            metadata=metadata or {}
        )

        self._cache.append(entry)
        self._save_entry(entry)

        return pattern_id

    # Phase C/D: multi-signal scoring weights. Tuned so that semantic similarity
    # remains the dominant signal, with structured matches as tiebreakers /
    # boosts. Adjust if calibration data warrants.
    _PROJECT_TYPE_BONUS = 0.15
    _DEPENDENCY_BONUS_MAX = 0.10
    _ENV_PACKAGE_BONUS_MAX = 0.10

    async def find_similar_solutions(
        self,
        goal: str,
        k: int = 3,
        min_similarity: float = 0.3,
        project_type: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        installed_packages: Optional[List[str]] = None,
        strict_filters: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Find similar past solutions using multi-signal scoring.

        Final score = semantic cosine similarity + project_type bonus +
        dependency-overlap bonus. Structured matches boost ranking but do not
        exclude entries by default — pass `strict_filters=True` to restore the
        old hard-filter behavior.

        Args:
            goal: free-text goal for the new task
            k: maximum results to return
            min_similarity: lower bound applied to the *final* multi-signal
                score; entries below are dropped
            project_type: when matched, boosts score by _PROJECT_TYPE_BONUS
            dependencies: each overlapping dep adds proportional bonus, up to
                _DEPENDENCY_BONUS_MAX
            strict_filters: if True, project_type / dependencies act as hard
                filters (used by callers that explicitly want exclusion)
        """

        if not self._cache:
            return []

        query_emb = self._embed(goal)
        dep_filter = set(dependencies) if dependencies else None
        pkg_filter = {p.lower() for p in installed_packages} if installed_packages else None

        scored: List[tuple] = []
        for entry in self._cache:
            entry_project_type = entry.content.get("project_type")
            stored_deps = set(entry.content.get("dependencies") or [])
            env_ctx = entry.content.get("environment_context") or {}
            stored_pkgs = {p.lower() for p in env_ctx.get("installed_packages") or []}

            if strict_filters:
                if project_type and entry_project_type != project_type:
                    continue
                if dep_filter and not (stored_deps & dep_filter):
                    continue

            entry_emb = self._get_embedding(entry)
            base = cosine_similarity(query_emb, entry_emb)

            # Track each component separately so callers (e.g. the benchmark
            # runner) can log *why* an entry ranked where it did, not just the
            # final score. components sum exactly to `score`.
            pt_bonus = 0.0
            deps_bonus = 0.0
            pkg_bonus = 0.0
            if project_type and entry_project_type == project_type:
                pt_bonus = self._PROJECT_TYPE_BONUS
            if dep_filter and stored_deps:
                overlap = len(stored_deps & dep_filter)
                # Bonus scales with overlap fraction of the query's dependency
                # list, capped at _DEPENDENCY_BONUS_MAX.
                deps_bonus = min(
                    self._DEPENDENCY_BONUS_MAX,
                    self._DEPENDENCY_BONUS_MAX * (overlap / max(len(dep_filter), 1)),
                )
            if pkg_filter and stored_pkgs:
                overlap = len(stored_pkgs & pkg_filter)
                pkg_bonus = min(
                    self._ENV_PACKAGE_BONUS_MAX,
                    self._ENV_PACKAGE_BONUS_MAX * (overlap / max(len(pkg_filter), 1)),
                )

            score = base + pt_bonus + deps_bonus + pkg_bonus

            if score >= min_similarity:
                breakdown = {
                    "semantic": base,
                    "project_type_bonus": pt_bonus,
                    "deps_bonus": deps_bonus,
                    "package_bonus": pkg_bonus,
                }
                scored.append((score, base, breakdown, entry))

        scored.sort(reverse=True, key=lambda x: x[0])

        results: List[Dict[str, Any]] = []
        for score, base, breakdown, entry in scored[:k]:
            results.append({
                "id": entry.id,
                "similarity": score,             # multi-signal score
                "base_similarity": base,         # raw semantic cosine
                "score_breakdown": breakdown,    # per-component contributions
                "goal": entry.content["goal"],
                "plan": entry.content["plan"],
                "code": entry.content["code"],
                "project_type": entry.content.get("project_type", "general"),
                "dependencies": entry.content.get("dependencies", []),
                "environment_context": entry.content.get("environment_context") or {},
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

    # Max bonus added to a learning's similarity score based on its
    # historical helpfulness rate. Tuned to be the same order as the
    # project-type bonus used by find_similar_solutions (0.15) but
    # smaller — usefulness is a noisy correlation signal and shouldn't
    # dominate semantic match.
    _USEFULNESS_BONUS_MAX = 0.10

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
        optional structural filters and a historical-helpfulness bonus.

        Final score = cosine(goal, lesson) + helpfulness_bonus.
        Helpfulness bonus = (Laplace-smoothed rate − 0.5) × 2 × MAX, capped
        in [−MAX, +MAX]. Brand-new entries (no retrievals yet) sit at
        rate=0.5 and get a zero adjustment — neither promoted nor demoted.

        Threshold (`min_similarity`) is applied to the *raw* cosine, not
        the boosted score; we don't let a well-performing Learning sneak
        past a semantic mismatch.
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
            if similarity < min_similarity:
                continue

            times_retrieved = int(entry.content.get("times_retrieved", 0) or 0)
            times_helpful = int(entry.content.get("times_helpful", 0) or 0)
            rate = (times_helpful + 1) / (times_retrieved + 2)
            bonus = (rate - 0.5) * 2 * self._USEFULNESS_BONUS_MAX
            score = similarity + bonus
            scored.append((score, similarity, bonus, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        results: List[Dict[str, Any]] = []
        for score, similarity, bonus, entry in scored[:k]:
            results.append({
                "id": entry.id,
                "similarity": similarity,
                "score": score,
                "usefulness_bonus": bonus,
                "lesson": entry.content["lesson"],
                "project_type": entry.content.get("project_type", "general"),
                "language": entry.content.get("language", "python"),
                "tags": entry.content.get("tags", []),
                "source_task_id": entry.content.get("source_task_id"),
                "times_retrieved": int(entry.content.get("times_retrieved", 0) or 0),
                "times_helpful": int(entry.content.get("times_helpful", 0) or 0),
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
