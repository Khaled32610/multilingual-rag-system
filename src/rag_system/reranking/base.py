"""Base interface for rerankers (Phase 7 / Step 1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_system.retrieval.models import RetrievedChunk


class BaseReranker(ABC):
    """Common interface so different cross-encoder backends are interchangeable."""

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Re-score `chunks` against `query` and return the top_n, re-ranked."""
        raise NotImplementedError