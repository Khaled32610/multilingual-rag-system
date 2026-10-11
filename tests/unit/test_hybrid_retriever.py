"""Unit tests for HybridRetriever (Phase 6 / Step 2)."""

import pytest

from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import RetrievalError


def _rc(chunk_id: str, score: float, rank: int, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=f"text {chunk_id}", metadata={},
        score=score, rank=rank, source=source,
    )


class FakeVectorRetriever:
    def __init__(self, results=None, raise_error=False):
        self._results = results or []
        self._raise_error = raise_error
        self.indexed = []

    def index_chunks(self, chunks):
        self.indexed.extend(chunks)
        return len(chunks)

    def retrieve(self, query, top_k=None):
        if self._raise_error:
            raise RetrievalError("fake vector failure")
        return self._results[:top_k] if top_k else self._results

    def save(self):
        pass

    def __len__(self):
        return len(self.indexed)


class FakeBM25Retriever:
    def __init__(self, results=None, raise_error=False):
        self._results = results or []
        self._raise_error = raise_error
        self.indexed = []

    def index_chunks(self, chunks):
        self.indexed.extend(chunks)
        return len(chunks)

    def retrieve(self, query, top_k=None):
        if self._raise_error:
            raise RetrievalError("fake bm25 failure")
        return self._results[:top_k] if top_k else self._results

    def save(self):
        pass


class TestHybridRetriever:
    def test_index_chunks_hits_both_retrievers(self):
        vec = FakeVectorRetriever()
        bm25 = FakeBM25Retriever()
        hybrid = HybridRetriever(vector_retriever=vec, bm25_retriever=bm25)

        fake_chunks = ["chunk1", "chunk2"]  # مش لازم تكون Chunk حقيقية للتيست ده
        counts = hybrid.index_chunks(fake_chunks)

        assert counts == {"vector": 2, "bm25": 2}
        assert vec.indexed == fake_chunks
        assert bm25.indexed == fake_chunks

    def test_empty_chunks_returns_zero(self):
        hybrid = HybridRetriever(vector_retriever=FakeVectorRetriever(), bm25_retriever=FakeBM25Retriever())
        assert hybrid.index_chunks([]) == {"vector": 0, "bm25": 0}

    def test_retrieve_fuses_both_sources(self):
        vec_results = [_rc("a", 0.9, 1, "vector"), _rc("b", 0.5, 2, "vector")]
        bm25_results = [_rc("b", 5.0, 1, "bm25"), _rc("c", 1.0, 2, "bm25")]

        hybrid = HybridRetriever(
            vector_retriever=FakeVectorRetriever(vec_results),
            bm25_retriever=FakeBM25Retriever(bm25_results),
        )
        results = hybrid.retrieve("test query", top_k=5)

        ids = [r.chunk_id for r in results]
        assert set(ids) == {"a", "b", "c"}
        assert ids[0] == "b"  # ظاهر في المصدرين
        assert all(r.source == "hybrid" for r in results)

    def test_retrieve_respects_top_k(self):
        vec_results = [_rc(f"v{i}", 1.0 - i * 0.1, i + 1, "vector") for i in range(5)]
        hybrid = HybridRetriever(
            vector_retriever=FakeVectorRetriever(vec_results),
            bm25_retriever=FakeBM25Retriever([]),
        )
        results = hybrid.retrieve("query", top_k=2)
        assert len(results) == 2

    def test_empty_query_raises(self):
        hybrid = HybridRetriever(vector_retriever=FakeVectorRetriever(), bm25_retriever=FakeBM25Retriever())
        with pytest.raises(RetrievalError):
            hybrid.retrieve("   ")

    def test_vector_failure_falls_back_to_bm25_only(self):
        bm25_results = [_rc("x", 3.0, 1, "bm25")]
        hybrid = HybridRetriever(
            vector_retriever=FakeVectorRetriever(raise_error=True),
            bm25_retriever=FakeBM25Retriever(bm25_results),
        )
        results = hybrid.retrieve("query")
        assert len(results) == 1
        assert results[0].chunk_id == "x"

    def test_bm25_failure_falls_back_to_vector_only(self):
        vec_results = [_rc("y", 0.8, 1, "vector")]
        hybrid = HybridRetriever(
            vector_retriever=FakeVectorRetriever(vec_results),
            bm25_retriever=FakeBM25Retriever(raise_error=True),
        )
        results = hybrid.retrieve("query")
        assert len(results) == 1
        assert results[0].chunk_id == "y"

    def test_both_fail_raises(self):
        hybrid = HybridRetriever(
            vector_retriever=FakeVectorRetriever(raise_error=True),
            bm25_retriever=FakeBM25Retriever(raise_error=True),
        )
        with pytest.raises(RetrievalError):
            hybrid.retrieve("query")

    def test_fetch_multiplier_requests_more_than_top_k(self):
        class RecordingVector(FakeVectorRetriever):
            def retrieve(self, query, top_k=None):
                self.last_top_k = top_k
                return super().retrieve(query, top_k)

        vec = RecordingVector()
        hybrid = HybridRetriever(vector_retriever=vec, bm25_retriever=FakeBM25Retriever(), fetch_multiplier=3)
        hybrid.retrieve("query", top_k=5)
        assert vec.last_top_k == 15