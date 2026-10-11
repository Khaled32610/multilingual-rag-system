"""
Shared data models for the retrieval system (Phase 4+).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """
    A single chunk returned by a retriever, together with its relevance
    score and rank within that retriever's result list.

    `source` records which retrieval method produced this result
    ("vector", "bm25", "hybrid", ...). It's kept generic now, rather
    than defined only inside the vector retriever, so Phase 5 (BM25)
    and Phase 6 (hybrid retrieval / RRF) can return the exact same
    shape instead of each phase inventing its own result type.
    """

    chunk_id: str
    text: str
    metadata: dict
    score: float
    rank: int
    source: str = "vector"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
        }

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievedChunk(rank={self.rank}, score={self.score:.4f}, "
            f"source={self.source!r}, id={self.chunk_id[:8]}..., text='{preview}...')"
        )
