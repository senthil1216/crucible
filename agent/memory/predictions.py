"""
Prediction memory: stores falsifiable hypotheses about code failures.

The Reflector emits predictions on failure ("input -1 will trigger
ValueError"). This module persists them. A future replay engine (Track D
phase 2) will run each prediction against new candidate code, classify it
as Confirmed / Falsified / Off-topic, and update times_tested /
times_confirmed in place.

In phase 1 we only need:
  - store(prediction) -> id
  - find_by_failure_id(failure_id) -> list of predictions linked to that failure
  - record_tested(ids) / record_confirmed(ids) — stubs the replay engine will use

Semantic-similarity retrieval is deliberately deferred until we have
real data and a defined consumption pattern. Premature retrieval API
designs tend to ossify around the wrong shape.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.models import MemoryEntry, Prediction
from agent.memory.embeddings import EmbeddingClient, cosine_similarity


class PredictionMemory:
    """
    On-disk store for Predictions. JSONL format, one Prediction per line,
    mirrors FailureMemory's structure.
    """

    # Max bonus added to a prediction's similarity score based on its
    # confirmation history. Same shape as LongTermMemory's usefulness
    # bonus; neutral (rate=0.5 → bonus=0) until phase 2's replay engine
    # produces confirmation data.
    _CONFIRMATION_BONUS_MAX = 0.10

    def __init__(
        self,
        storage_path: Path,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = storage_path / "predictions.jsonl"
        self._embeddings = embedding_client or EmbeddingClient.shared()
        self._cache: List[MemoryEntry] = []
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.predictions_file.exists():
            return
        with open(self.predictions_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self._cache.append(MemoryEntry(
                    id=data["id"],
                    content=data["content"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metadata=data.get("metadata", {}),
                ))

    def _save_entry(self, entry: MemoryEntry) -> None:
        with open(self.predictions_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _rewrite_cache_to_disk(self) -> None:
        """Used by phase-2 record_* methods to mutate existing entries."""
        tmp = self.predictions_file.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for entry in self._cache:
                f.write(json.dumps(entry.to_dict()) + "\n")
        tmp.replace(self.predictions_file)

    async def store(self, prediction: Prediction) -> Optional[str]:
        """Persist a prediction. Returns the prediction id, or None if the
        prediction fails the schema gate (no concrete trigger_input).

        ID is derived from (source_failure_id, trigger_input) so identical
        predictions emitted twice deduplicate naturally. The source_goal
        is embedded at store time and stored alongside so future tasks
        can retrieve predictions by semantic goal similarity."""
        if not prediction.is_well_formed():
            return None

        seed = f"{prediction.source_failure_id}:{prediction.trigger_input}"
        prediction_id = hashlib.sha256(seed.encode()).hexdigest()[:16]

        content = prediction.to_dict()
        # Embed the source_goal so we can retrieve by semantic similarity
        # when a new task's goal resembles a past failure context.
        goal_text = prediction.source_goal or ""
        content["goal_embedding"] = (
            self._embeddings.encode(goal_text) if goal_text else []
        )

        entry = MemoryEntry(id=prediction_id, content=content)
        # Dedupe in-memory: if we already have this id, skip the append.
        for existing in self._cache:
            if existing.id == prediction_id:
                return prediction_id
        self._cache.append(entry)
        self._save_entry(entry)
        return prediction_id

    def _get_goal_embedding(self, entry: MemoryEntry) -> List[float]:
        """Return the cached source_goal embedding, computing it lazily
        for legacy entries written before we stored embeddings."""
        emb = entry.content.get("goal_embedding")
        if emb:
            return emb
        goal_text = entry.content.get("source_goal") or ""
        emb = self._embeddings.encode(goal_text) if goal_text else []
        entry.content["goal_embedding"] = emb
        return emb

    async def find_relevant(
        self,
        goal: str,
        k: int = 3,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Return predictions whose source_goal is semantically similar to
        the given goal. Used by the Planner to surface "on a similar past
        task, this concrete input triggered this error type" context.

        Score = cosine(query_goal, stored_goal) + confirmation_bonus.
        Threshold is applied to raw cosine, not boosted score, so a
        well-confirmed prediction can't sneak past a semantic mismatch.

        Until phase 2's replay engine runs, all predictions sit at
        confirmation_rate=0.5 (Laplace prior) → bonus=0. Wired now so
        phase 2 only needs to bump counters; retrieval ranking adapts
        automatically.
        """
        if not self._cache or not goal:
            return []

        query_emb = self._embeddings.encode(goal)

        scored: List[tuple] = []
        for entry in self._cache:
            entry_emb = self._get_goal_embedding(entry)
            if not entry_emb:
                continue
            similarity = cosine_similarity(query_emb, entry_emb)
            if similarity < min_similarity:
                continue

            times_tested = int(entry.content.get("times_tested", 0) or 0)
            times_confirmed = int(entry.content.get("times_confirmed", 0) or 0)
            rate = (times_confirmed + 1) / (times_tested + 2)
            bonus = (rate - 0.5) * 2 * self._CONFIRMATION_BONUS_MAX
            score = similarity + bonus
            scored.append((score, similarity, bonus, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        results: List[Dict[str, Any]] = []
        for score, similarity, bonus, entry in scored[:k]:
            results.append({
                "id": entry.id,
                "similarity": similarity,
                "score": score,
                "confirmation_bonus": bonus,
                "trigger_input": entry.content.get("trigger_input"),
                "predicted_error_type": entry.content.get("predicted_error_type"),
                "predicted_explanation": entry.content.get("predicted_explanation", ""),
                "confidence": entry.content.get("confidence", 0.5),
                "source_failure_id": entry.content.get("source_failure_id"),
                "source_goal": entry.content.get("source_goal"),
                "times_tested": int(entry.content.get("times_tested", 0) or 0),
                "times_confirmed": int(entry.content.get("times_confirmed", 0) or 0),
            })
        return results

    def find_by_failure_id(self, failure_id: str) -> List[Dict[str, Any]]:
        """All predictions linked to a given failure. Used by phase 2's
        replay engine to fetch the falsifiable claims associated with a
        prior failure when new candidate code lands."""
        results: List[Dict[str, Any]] = []
        for entry in self._cache:
            if entry.content.get("source_failure_id") == failure_id:
                results.append({
                    "id": entry.id,
                    **entry.content,
                })
        return results

    def all_predictions(self) -> List[Dict[str, Any]]:
        """Iteration helper for phase 2 / diagnostics. Don't lean on it
        from hot paths — it's a full scan."""
        return [{"id": e.id, **e.content} for e in self._cache]

    def record_tested(self, prediction_ids: List[str]) -> int:
        """Bump times_tested. Called by the phase-2 replay engine after a
        prediction is exercised against new candidate code, regardless of
        outcome."""
        if not prediction_ids:
            return 0
        wanted = set(prediction_ids)
        updated = 0
        for entry in self._cache:
            if entry.id in wanted:
                entry.content["times_tested"] = int(
                    entry.content.get("times_tested", 0) or 0
                ) + 1
                updated += 1
        if updated:
            self._rewrite_cache_to_disk()
        return updated

    def record_confirmed(self, prediction_ids: List[str]) -> int:
        """Bump times_confirmed. Subset of `record_tested` — only the
        cases where the replay engine reproduced the predicted error."""
        if not prediction_ids:
            return 0
        wanted = set(prediction_ids)
        updated = 0
        for entry in self._cache:
            if entry.id in wanted:
                entry.content["times_confirmed"] = int(
                    entry.content.get("times_confirmed", 0) or 0
                ) + 1
                updated += 1
        if updated:
            self._rewrite_cache_to_disk()
        return updated

    def get_stats(self) -> Dict[str, int]:
        total = len(self._cache)
        tested = sum(
            1 for e in self._cache
            if int(e.content.get("times_tested", 0) or 0) > 0
        )
        confirmed = sum(
            1 for e in self._cache
            if int(e.content.get("times_confirmed", 0) or 0) > 0
        )
        return {
            "total_predictions": total,
            "tested": tested,
            "confirmed_at_least_once": confirmed,
        }
