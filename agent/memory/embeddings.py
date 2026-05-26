"""
Shared embedding utility for semantic similarity in memory modules.

Used by LongTermMemory and FailureMemory to score similarity between
goals / error messages via sentence-transformer embeddings.

Default model: all-MiniLM-L6-v2 (384 dims, ~80 MB on disk, CPU-friendly).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingClient:
    """
    Wraps a sentence-transformers model with a lazy load so importing this
    module is cheap. The first call to `encode` pays the load cost.
    """

    _instance: Optional["EmbeddingClient"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # loaded lazily

    @classmethod
    def shared(cls) -> "EmbeddingClient":
        """Process-wide singleton so the model is loaded at most once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> List[float]:
        """Return a dense vector for `text` as a plain Python list."""
        self._ensure_loaded()
        # convert_to_numpy=True returns a numpy array; .tolist() is JSON-safe
        vec = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for empty/mismatched inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
