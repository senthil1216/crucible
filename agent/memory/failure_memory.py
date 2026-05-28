"""
Failure memory: Stores error signatures with solutions.
Prevents repeating the same mistakes.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from agent.models import ErrorSignature, MemoryEntry, CodeArtifact
from agent.memory.embeddings import EmbeddingClient, cosine_similarity


class FailureMemory:
    """
    Stores failed attempts with their error signatures and fixes.
    Uses error signature similarity to avoid repeating mistakes.
    """
    
    def __init__(
        self,
        storage_path: Path,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.failures_file = storage_path / "failures.jsonl"
        self._embeddings = embedding_client or EmbeddingClient.shared()
        self._cache: List[MemoryEntry] = []
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load failures from disk."""
        if not self.failures_file.exists():
            return
        
        with open(self.failures_file, 'r') as f:
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
        """Save a single entry to disk."""
        with open(self.failures_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def _rewrite_cache_to_disk(self) -> None:
        """Rewrite the JSONL file from the current cache. Used for in-place
        updates to existing entries (e.g. marking was_fixed=True). The cache
        is the source of truth here."""
        tmp = self.failures_file.with_suffix(".jsonl.tmp")
        with open(tmp, 'w') as f:
            for entry in self._cache:
                f.write(json.dumps(entry.to_dict()) + '\n')
        tmp.replace(self.failures_file)

    def mark_fixed(self, failure_id: str, fix_diff: Optional[str] = None) -> bool:
        """Mark a stored failure as was_fixed=True.

        Called by the loop when iteration N+1 succeeds after iteration N
        failed: the prior failure was demonstrably fixable, so we boost it
        in `find_similar_failures` (fixed failures are more useful than raw
        failures because they carry the implicit lesson that the suggested
        fix actually worked).

        Returns True if the entry was found and updated.
        """
        for entry in self._cache:
            if entry.id == failure_id:
                entry.content["was_fixed"] = True
                if fix_diff is not None:
                    entry.content["fix_diff"] = fix_diff
                self._rewrite_cache_to_disk()
                return True
        return False
    
    def _extract_error_signature_key(self, sig: ErrorSignature) -> str:
        """Create a key for error matching."""
        return f"{sig.error_type}:{sig.normalize()}"
    
    async def store_failure(
        self,
        error_signature: ErrorSignature,
        attempt: CodeArtifact,
        root_cause: str,
        fix: str,
        goal: str,
        was_fixed: bool = False
    ) -> str:
        """Store a failure with its context and solution."""
        
        failure_id = hashlib.sha256(
            f"{error_signature.normalize()}:{attempt.source[:100]}".encode()
        ).hexdigest()[:16]
        
        # Embed the *raw* message (not the placeholder-heavy normalized form)
        # so the vector retains semantic content. The normalized key is still
        # used for exact-grouping via _extract_error_signature_key.
        content = {
            "error_signature": error_signature.to_dict(),
            "error_key": self._extract_error_signature_key(error_signature),
            "error_embedding": self._embeddings.encode(error_signature.error_message),
            "attempt_summary": attempt.source[:500],
            "root_cause": root_cause,
            "fix": fix,
            "goal": goal,
            "was_fixed": was_fixed,
            "language": attempt.language
        }
        
        entry = MemoryEntry(
            id=failure_id,
            content=content
        )
        
        self._cache.append(entry)
        self._save_entry(entry)
        
        return failure_id
    
    async def find_similar_failures(
        self,
        error_signature: ErrorSignature,
        k: int = 3,
        same_error_type_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Find similar past failures."""
        
        if not self._cache:
            return []
        
        target_emb = self._embeddings.encode(error_signature.error_message)

        scored = []
        for entry in self._cache:
            entry_sig = entry.content.get("error_signature", {})

            # Filter by error type if requested
            if same_error_type_only:
                if entry_sig.get("error_type") != error_signature.error_type:
                    continue

            entry_emb = self._get_embedding(entry)
            similarity = cosine_similarity(target_emb, entry_emb)

            # Boost fix-confirmed entries: they carry the implicit lesson
            # that the suggested fix actually worked next iteration.
            if entry.content.get("was_fixed", False):
                similarity += 0.05

            if similarity > 0.3:  # Threshold
                scored.append((similarity, entry))
        
        # Sort by similarity
        scored.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, entry in scored[:k]:
            results.append({
                "id": entry.id,
                "similarity": score,
                "error_type": entry.content["error_signature"]["error_type"],
                "error_message": entry.content["error_signature"]["error_message"],
                "root_cause": entry.content["root_cause"],
                "fix": entry.content["fix"],
                "was_fixed": entry.content.get("was_fixed", False)
            })
        
        return results
    
    def _get_embedding(self, entry: MemoryEntry) -> List[float]:
        """Return the cached error embedding, computing it lazily for legacy entries."""
        emb = entry.content.get("error_embedding")
        if emb:
            return emb
        # Backfill in memory. Prefer the raw error_message for semantic content;
        # fall back to error_key if the legacy entry doesn't have a signature dict.
        sig = entry.content.get("error_signature") or {}
        text = sig.get("error_message") or entry.content.get("error_key") or ""
        emb = self._embeddings.encode(text) if text else []
        entry.content["error_embedding"] = emb
        return emb
    
    def get_common_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most common types of failures."""
        from collections import Counter
        
        error_types = [e.content["error_signature"]["error_type"] 
                      for e in self._cache]
        counts = Counter(error_types)
        
        return [{"error_type": t, "count": c} for t, c in counts.most_common(limit)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get failure memory statistics."""
        total = len(self._cache)
        fixed = sum(1 for e in self._cache if e.content.get("was_fixed", False))
        
        return {
            "total_failures": total,
            "fixed_failures": fixed,
            "fix_rate": fixed / total if total > 0 else 0,
            "common_failures": self.get_common_failures(5)
        }
    
    def clear(self) -> None:
        """Clear all failure memory."""
        self._cache.clear()
        if self.failures_file.exists():
            self.failures_file.unlink()
