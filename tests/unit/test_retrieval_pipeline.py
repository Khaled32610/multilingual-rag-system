"""Unit tests for RetrievalPipeline (Phase 7 / Step 2)."""

import pytest

from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.exceptions import RerankingError


def _rc(chunk_id: str, score: float, rank: int, source: str = "hybrid") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"text {chunk_id}", metadata={}, score=score, rank=rank, source=source)


class FakeHybridRetriever:
    def __init__(self, results=None):
        self._results = results or []
        self.last_top_k = None

    def index_chunks(self, chunks):
        return {"vector": len(chunks), "bm25": len(chunks)}

    def retrieve(self, query, top_k=None):
        self.last_top_k = top_k
        return self._results[:top_k] if top_k else self._results

    def save(self):
        pass


class FakeReranker:
    def __init__(self, raise_error=False):
        self._raise_error = raise_error
        self.last_query = None

    def rerank(self, query, chunks, top_n):
        self.last_query = query
        if self._raise_error:
            raise RerankingError("fake reranker failure")
        # عكس الترتيب عشان نتأكد إن الـ reranker فعلاً بيتنفذ
        reranked = list(reversed(chunks))[:top_n]
        for rank, r in enumerate(reranked, start=1):
            r.rank = rank
            r.source = "reranked"
        return reranked


class TestRetrievalPipeline:
    def test_retrieve_applies_reranker(self):
        hybrid_results = [_rc("a", 0.9, 1), _rc("b", 0.5, 2)]
        pipeline = RetrievalPipeline(
            hybrid_retriever=FakeHybridRetriever(hybrid_results),
            reranker=FakeReranker(),
        )
        results = pipeline.retrieve("query", top_n=2)
        assert [r.chunk_id for r in results] == ["b", "a"]  # الفيك ريرانكر بيعكس
        assert all(r.source == "reranked" for r in results)

    def test_empty_hybrid_results_returns_empty(self):
        pipeline = RetrievalPipeline(
            hybrid_retriever=FakeHybridRetriever([]), reranker=FakeReranker()
        )
        assert pipeline.retrieve("query") == []

    def test_reranker_failure_falls_back_to_hybrid_results(self):
        hybrid_results = [_rc("a", 0.9, 1), _rc("b", 0.5, 2)]
        pipeline = RetrievalPipeline(
            hybrid_retriever=FakeHybridRetriever(hybrid_results),
            reranker=FakeReranker(raise_error=True),
        )
        results = pipeline.retrieve("query", top_n=2)
        assert [r.chunk_id for r in results] == ["a", "b"]  # رجع لنتائج الـ hybrid زي ما هي

    def test_fetch_k_larger_than_top_n(self):
        hybrid = FakeHybridRetriever([_rc(f"c{i}", 1.0, i + 1) for i in range(10)])
        pipeline = RetrievalPipeline(hybrid_retriever=hybrid, reranker=FakeReranker())
        pipeline.retrieve("query", top_n=2)
        assert hybrid.last_top_k > 2  # لازم يطلب أكتر من top_n للـ reranker يشتغل عليهم

    def test_index_chunks_delegates_to_hybrid(self):
        hybrid = FakeHybridRetriever()
        pipeline = RetrievalPipeline(hybrid_retriever=hybrid, reranker=FakeReranker())
        counts = pipeline.index_chunks(["c1", "c2"])
        assert counts == {"vector": 2, "bm25": 2}

def test_arabic_query_is_translated_for_english_evidence_reranking():
    from types import SimpleNamespace

    class FakeTranslator:
        def translate_to_english(self, query):
            assert "الموازنة" in query
            return "What was the total state budget?"

    hybrid = FakeHybridRetriever([_rc("a", 0.9, 1)])
    hybrid.bm25_retriever = SimpleNamespace(query_translator=FakeTranslator())
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(hybrid_retriever=hybrid, reranker=reranker)

    pipeline.retrieve("ما إجمالي الموازنة؟", top_n=1)

    assert reranker.last_query == "What was the total state budget?"


def test_english_query_domain_synonyms_are_added_before_reranking():
    hybrid = FakeHybridRetriever([_rc("a", 0.9, 1)])
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(hybrid_retriever=hybrid, reranker=reranker)

    pipeline.retrieve("What is the average final bound tariff?", top_n=1)

    assert "simple average final bound rate" in reranker.last_query
