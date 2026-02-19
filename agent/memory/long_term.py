"""
Long-term memory: Stores successful patterns indexed by problem type.
Uses simple keyword/semantic matching for retrieval.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from agent.models import Plan, CodeArtifact, MemoryEntry


class LongTermMemory:
    """
    Stores and retrieves successful solution patterns.
    Uses keyword extraction and simple similarity matching.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.patterns_file = storage_path / "patterns.jsonl"
        self._cache: List[MemoryEntry] = []
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load patterns from disk into memory."""
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
        """Save a single entry to disk."""
        with open(self.patterns_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
    
    def _extract_keywords(self, text: str) -> set[str]:
        """Extract keywords from text for matching."""
        # Simple keyword extraction
        text = text.lower()
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'and', 'but', 'or', 'yet', 'so', 'if',
                     'because', 'although', 'though', 'while', 'where', 'when',
                     'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
                     'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                     'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
                     'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text)
        return {w for w in words if w not in stopwords and len(w) > 2}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple Jaccard similarity between two texts."""
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    async def store_pattern(
        self,
        goal: str,
        plan: Plan,
        code: CodeArtifact,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a successful solution pattern."""
        
        pattern_id = hashlib.sha256(
            f"{goal}:{code.source}".encode()
        ).hexdigest()[:16]
        
        content = {
            "goal": goal,
            "plan": plan.to_dict(),
            "code": code.to_dict(),
            "keywords": list(self._extract_keywords(goal))
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
        min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Find similar past solutions based on goal similarity."""
        
        if not self._cache:
            return []
        
        # Calculate similarity scores
        scored = []
        for entry in self._cache:
            similarity = self._calculate_similarity(
                goal,
                entry.content.get("goal", "")
            )
            if similarity >= min_similarity:
                scored.append((similarity, entry))
        
        # Sort by similarity and return top k
        scored.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, entry in scored[:k]:
            result = {
                "id": entry.id,
                "similarity": score,
                "goal": entry.content["goal"],
                "plan": entry.content["plan"],
                "code": entry.content["code"],
                "metadata": entry.metadata
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "total_patterns": len(self._cache)
        }
    
    def clear(self) -> None:
        """Clear all long-term memory."""
        self._cache.clear()
        if self.patterns_file.exists():
            self.patterns_file.unlink()
