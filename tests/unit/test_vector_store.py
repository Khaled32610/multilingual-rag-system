"""Unit tests for Phase 4 / Step 16: Vector store."""

import numpy as np
import pytest

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.exceptions import VectorStoreError

DIM = 4


def make_chunk(chunk_index: int, text: str = "sample text") -> Chunk:
    metadata = ChunkMetadata(
        source_file="test.pdf",
        document_hash="a" * 64,
        page_number=1,
        chunk_index=chunk_index,
    )
    return Chunk(text=text, metadata=metadata)


def unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


class TestVectorStoreAdd:
    def test_add_chunks_increases_length(self):
        store = VectorStore(dimension=DIM)
        chunks = [make_chunk(i) for i in range(3)]
        vectors = np.stack([unit_vector(i) for i in range(3)])

        added = store.add_chunks(chunks, vectors)

        assert added == 3
        assert len(store) == 3

    def test_add_empty_list_is_a_noop(self):
        store = VectorStore(dimension=DIM)
        assert store.add_chunks([], np.zeros((0, DIM), dtype="float32")) == 0
        assert len(store) == 0

    def test_add_chunks_rejects_mismatched_lengths(self):
        store = VectorStore(dimension=DIM)
        chunks = [make_chunk(0), make_chunk(1)]
        vectors = np.stack([unit_vector(0)])

        with pytest.raises(VectorStoreError):
            store.add_chunks(chunks, vectors)

    def test_add_chunks_rejects_wrong_dimension(self):
        store = VectorStore(dimension=DIM)
        chunks = [make_chunk(0)]
        vectors = np.zeros((1, DIM + 1), dtype="float32")

        with pytest.raises(VectorStoreError):
            store.add_chunks(chunks, vectors)

    def test_duplicate_chunk_ids_are_skipped_not_duplicated(self):
        store = VectorStore(dimension=DIM)
        chunk = make_chunk(0)
        vector = unit_vector(0).reshape(1, -1)

        first = store.add_chunks([chunk], vector)
        second = store.add_chunks([chunk], vector)

        assert first == 1
        assert second == 0
        assert len(store) == 1


class TestVectorStoreSearch:
    def test_search_returns_most_similar_first(self):
        store = VectorStore(dimension=DIM)
        chunks = [make_chunk(i, text=f"chunk {i}") for i in range(3)]
        vectors = np.stack([unit_vector(i) for i in range(3)])
        store.add_chunks(chunks, vectors)

        query = vectors[1]  # identical to chunk 1's own vector
        results = store.search(query, top_k=3)

        assert results[0].chunk_id == chunks[1].chunk_id
        assert results[0].rank == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        assert results[0].source == "vector"

    def test_search_respects_top_k(self):
        store = VectorStore(dimension=DIM)
        chunks = [make_chunk(i) for i in range(5)]
        vectors = np.stack([unit_vector(i) for i in range(5)])
        store.add_chunks(chunks, vectors)

        results = store.search(unit_vector(0), top_k=2)

        assert len(results) == 2

    def test_search_on_empty_store_returns_empty_list(self):
        store = VectorStore(dimension=DIM)
        assert store.search(unit_vector(0), top_k=5) == []

    def test_search_rejects_wrong_dimension(self):
        store = VectorStore(dimension=DIM)
        store.add_chunks([make_chunk(0)], unit_vector(0).reshape(1, -1))

        with pytest.raises(VectorStoreError):
            store.search(np.zeros(DIM + 1, dtype="float32"), top_k=1)

    def test_search_rejects_non_positive_top_k(self):
        store = VectorStore(dimension=DIM)
        store.add_chunks([make_chunk(0)], unit_vector(0).reshape(1, -1))

        with pytest.raises(VectorStoreError):
            store.search(unit_vector(0), top_k=0)


class TestVectorStoreDelete:
    def test_delete_by_chunk_ids_removes_vector(self):
        store = VectorStore(dimension=DIM)
        chunk = make_chunk(0)
        store.add_chunks([chunk], unit_vector(0).reshape(1, -1))

        removed = store.delete_by_chunk_ids([chunk.chunk_id])

        assert removed == 1
        assert len(store) == 0
        assert not store.has_chunk(chunk.chunk_id)

    def test_delete_unknown_chunk_id_is_a_noop(self):
        store = VectorStore(dimension=DIM)
        store.add_chunks([make_chunk(0)], unit_vector(0).reshape(1, -1))

        removed = store.delete_by_chunk_ids(["does-not-exist"])

        assert removed == 0
        assert len(store) == 1


class TestVectorStorePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = VectorStore(dimension=DIM)
        chunks = [make_chunk(i, text=f"chunk {i}") for i in range(3)]
        vectors = np.stack([unit_vector(i) for i in range(3)])
        store.add_chunks(chunks, vectors)

        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)

        assert len(loaded) == 3
        results = loaded.search(vectors[0], top_k=1)
        assert results[0].chunk_id == chunks[0].chunk_id
        assert results[0].text == "chunk 0"

    def test_exists_false_for_empty_directory(self, tmp_path):
        assert VectorStore.exists(tmp_path) is False

    def test_load_missing_store_raises(self, tmp_path):
        with pytest.raises(VectorStoreError):
            VectorStore.load(tmp_path)
