"""Unit tests for CrossEncoderReranker (Phase 7 / Step 1)."""

import pytest

from rag_system.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import RerankingError


def _rc(chunk_id: str, text: str, score: float = 0.5, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, metadata={}, score=score, rank=rank, source="hybrid")


class FakeCrossEncoderModel:
    """Stand-in for sentence_transformers.CrossEncoder — no real model load."""

    def __init__(self, score_map: dict[str, float]):
        self.score_map = score_map

    def predict(self, pairs):
        return [self.score_map.get(text, 0.0) for _, text in pairs]


def _make_reranker(score_map: dict[str, float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)  # skip __init__ (no real model load)
    reranker.model_name = "fake"
    reranker.model = FakeCrossEncoderModel(score_map)
    return reranker


class TestCrossEncoderReranker:
    def test_empty_chunks_returns_empty(self):
        reranker = _make_reranker({})
        assert reranker.rerank("query", [], top_n=5) == []

    def test_empty_query_raises(self):
        reranker = _make_reranker({})
        with pytest.raises(RerankingError):
            reranker.rerank("   ", [_rc("a", "text")], top_n=5)

    def test_reorders_by_cross_encoder_score(self):
        chunks = [_rc("a", "low relevance text"), _rc("b", "high relevance text")]
        reranker = _make_reranker({"low relevance text": 0.1, "high relevance text": 0.9})
        results = reranker.rerank("query", chunks, top_n=2)
        assert [r.chunk_id for r in results] == ["b", "a"]

    def test_respects_top_n(self):
        chunks = [_rc(f"c{i}", f"text {i}") for i in range(5)]
        reranker = _make_reranker({f"text {i}": float(i) for i in range(5)})
        results = reranker.rerank("query", chunks, top_n=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c4"  # أعلى score

    def test_rank_recomputed_after_rerank(self):
        chunks = [_rc("a", "t1"), _rc("b", "t2")]
        reranker = _make_reranker({"t1": 0.2, "t2": 0.8})
        results = reranker.rerank("query", chunks, top_n=2)
        assert [r.rank for r in results] == [1, 2]

    def test_source_is_reranked(self):
        chunks = [_rc("a", "t1")]
        reranker = _make_reranker({"t1": 0.5})
        results = reranker.rerank("query", chunks, top_n=1)
        assert results[0].source == "reranked"

    def test_preserves_pre_rerank_metadata(self):
        chunks = [_rc("a", "t1", score=0.77, rank=3)]
        reranker = _make_reranker({"t1": 0.5})
        results = reranker.rerank("query", chunks, top_n=1)
        assert results[0].metadata["pre_rerank_score"] == pytest.approx(0.77)
        assert results[0].metadata["pre_rerank_rank"] == 3

    def test_model_load_failure_raises_reranking_error(self, monkeypatch):
        def fail_init(*args, **kwargs):
            raise RuntimeError("no space on device")

        monkeypatch.setattr(
            "rag_system.reranking.cross_encoder_reranker.CrossEncoder", fail_init
        )
        with pytest.raises(RerankingError):
            CrossEncoderReranker(model_name="fake-model")

    def test_predict_failure_raises_reranking_error(self):
        class BrokenModel:
            def predict(self, pairs):
                raise RuntimeError("inference crashed")

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "fake"
        reranker.model = BrokenModel()
        with pytest.raises(RerankingError):
            reranker.rerank("query", [_rc("a", "text")], top_n=1)