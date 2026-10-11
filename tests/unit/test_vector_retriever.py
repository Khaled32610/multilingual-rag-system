"""Unit tests for Phase 4 / Step 17: Vector retriever.

Uses a fake embedding model (deterministic, hash-based) instead of the
real sentence-transformers model, so these tests stay fast and don't
require downloading model weights.
"""

import numpy as np
import pytest

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.embedding.base import BaseEmbeddingModel
from rag_system.embedding.models import EmbeddingResult
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.exceptions import RetrievalError

DIM = 8


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Deterministic stand-in for a real embedding model: same text
    always maps to the same unit vector, different text to a different one."""

    def __init__(self) -> None:
        self.name = "fake-model"
        self.dimension = DIM

    def _vector_for(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=DIM).astype("float32")
        return v / np.linalg.norm(v)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        vectors = np.stack([self._vector_for(t) for t in texts])
        return EmbeddingResult(
            model_name=self.name, vectors=vectors, dimension=self.dimension, num_texts=len(texts)
        )

    def embed_query(self, query: str) -> EmbeddingResult:
        return self.embed_texts([query])


def make_chunk(chunk_index: int, text: str) -> Chunk:
    metadata = ChunkMetadata(
        source_file="test.pdf",
        document_hash="b" * 64,
        page_number=1,
        chunk_index=chunk_index,
    )
    return Chunk(text=text, metadata=metadata)


def make_retriever() -> VectorRetriever:
    return VectorRetriever(embedding_model=FakeEmbeddingModel(), vector_store=VectorStore(dimension=DIM))


class TestVectorRetrieverIndexing:
    def test_index_chunks_returns_count_added(self):
        retriever = make_retriever()
        chunks = [make_chunk(i, f"chunk {i}") for i in range(3)]

        added = retriever.index_chunks(chunks)

        assert added == 3
        assert len(retriever) == 3

    def test_index_empty_list_returns_zero(self):
        retriever = make_retriever()
        assert retriever.index_chunks([]) == 0
        assert len(retriever) == 0


class TestVectorRetrieverRetrieve:
    def test_retrieve_returns_most_relevant_chunk_first(self):
        retriever = make_retriever()
        chunks = [
            make_chunk(0, "budget law article one"),
            make_chunk(1, "tax exemption rules"),
            make_chunk(2, "customs regulations"),
        ]
        retriever.index_chunks(chunks)

        results = retriever.retrieve("budget law article one", top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == chunks[0].chunk_id
        assert results[0].rank == 1

    def test_retrieve_defaults_top_k_from_settings(self):
        retriever = make_retriever()
        chunks = [make_chunk(i, f"chunk {i}") for i in range(3)]
        retriever.index_chunks(chunks)

        results = retriever.retrieve("chunk 0")

        assert 0 < len(results) <= 3

    def test_retrieve_rejects_empty_query(self):
        retriever = make_retriever()
        retriever.index_chunks([make_chunk(0, "some text")])

        with pytest.raises(RetrievalError):
            retriever.retrieve("   ")

    def test_retrieve_on_empty_index_returns_empty_list(self):
        retriever = make_retriever()
        assert retriever.retrieve("anything") == []
