"""
Failure memory: Stores error signatures with solutions.
Prevents repeating the same mistakes.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from agent.models import ErrorSignature, MemoryEntry, CodeArtifact


class FailureMemory:
    """
    Stores failed attempts with their error signatures and fixes.
    Uses error signature similarity to avoid repeating mistakes.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.failures_file = storage_path / "failures.jsonl"
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
        
        content = {
            "error_signature": error_signature.to_dict(),
            "error_key": self._extract_error_signature_key(error_signature),
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
        
        target_key = self._extract_error_signature_key(error_signature)
        target_norm = error_signature.normalize()
        
        scored = []
        for entry in self._cache:
            entry_sig = entry.content.get("error_signature", {})
            
            # Filter by error type if requested
            if same_error_type_only:
                if entry_sig.get("error_type") != error_signature.error_type:
                    continue
            
            # Calculate similarity based on normalized error message
            entry_norm = entry.content.get("error_key", "")
            
            # Simple string similarity
            similarity = self._string_similarity(target_norm, entry_norm)
            
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
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings."""
        # Use Jaccard similarity on word sets
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
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
