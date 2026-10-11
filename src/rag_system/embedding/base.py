"""
Embedding model interface (Phase 3 / Step 1).

Defines the abstract contract that every embedding backend (local
sentence-transformers models now, potentially API-based models later)
must implement. The rest of the system (vector store, retrieval,
benchmarking) depends only on this interface, never on a specific
implementation — so swapping the underlying model is a one-line change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_system.embedding.models import EmbeddingResult


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding model implementations."""

    #: Human-readable identifier used in logs, benchmarks, and vector store metadata.
    name: str

    #: Output vector dimensionality. Must be set by subclasses after loading.
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Embed a batch of raw text chunks (documents to be indexed).

        Some models distinguish between "document" and "query" embeddings
        (asymmetric embedding, e.g. instruction-tuned models) — this method
        is for the document/passage side.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> EmbeddingResult:
        """
        Embed a single search query.

        Kept separate from embed_texts because some models require a
        different prefix/instruction for queries vs documents to get
        good retrieval performance.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        dim = getattr(self, "dimension", "?")
        return f"<{self.__class__.__name__} name={self.name!r} dimension={dim}>"