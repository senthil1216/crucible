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


class PredictionMemory:
    """
    On-disk store for Predictions. JSONL format, one Prediction per line,
    mirrors FailureMemory's structure.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = storage_path / "predictions.jsonl"
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
        predictions emitted twice deduplicate naturally."""
        if not prediction.is_well_formed():
            return None

        seed = f"{prediction.source_failure_id}:{prediction.trigger_input}"
        prediction_id = hashlib.sha256(seed.encode()).hexdigest()[:16]

        entry = MemoryEntry(id=prediction_id, content=prediction.to_dict())
        # Dedupe in-memory: if we already have this id, skip the append.
        for existing in self._cache:
            if existing.id == prediction_id:
                return prediction_id
        self._cache.append(entry)
        self._save_entry(entry)
        return prediction_id

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
