"""
Shared data models for the embedding system (Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmbeddingResult:
    """Result of embedding one or more texts with a specific model."""

    model_name: str
    vectors: np.ndarray  # shape: (n_texts, dimension)
    dimension: int
    num_texts: int
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.reshape(1, -1)
        self.dimension = self.vectors.shape[1]
        self.num_texts = self.vectors.shape[0]

    @property
    def texts_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.num_texts / self.elapsed_seconds