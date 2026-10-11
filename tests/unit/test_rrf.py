"""Unit tests for Reciprocal Rank Fusion (Phase 6 / Step 1)."""

import pytest

from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.rrf import reciprocal_rank_fusion


def _rc(chunk_id: str, score: float, rank: int, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        metadata={},
        score=score,
        rank=rank,
        source=source,
    )


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        results = [
            _rc("a", 0.9, 1, "vector"),
            _rc("b", 0.5, 2, "vector"),
            _rc("c", 0.1, 3, "vector"),
        ]
        fused = reciprocal_rank_fusion([results])
        assert [r.chunk_id for r in fused] == ["a", "b", "c"]

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_agreement_boosts_rank(self):
        # "shared" رتبة 2 في القائمتين -> لازم يطلع فوق حاجة رتبة 1 في مصدر واحد بس
        vector_results = [_rc("only_vector", 0.99, 1, "vector"), _rc("shared", 0.8, 2, "vector")]
        bm25_results = [_rc("shared", 5.0, 1, "bm25"), _rc("only_bm25", 1.0, 2, "bm25")]

        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        fused_ids = [r.chunk_id for r in fused]

        assert fused_ids[0] == "shared"
        assert set(fused_ids) == {"only_vector", "shared", "only_bm25"}

    def test_no_duplicate_chunk_ids(self):
        vector_results = [_rc("x", 0.9, 1, "vector"), _rc("y", 0.5, 2, "vector")]
        bm25_results = [_rc("x", 3.0, 1, "bm25"), _rc("z", 1.0, 2, "bm25")]
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids)) == 3

    def test_rrf_score_formula(self):
        results = [_rc("a", 0.9, 1, "vector")]
        fused = reciprocal_rank_fusion([results], k=60)
        assert fused[0].score == pytest.approx(1.0 / 61.0)

    def test_rank_is_recomputed_after_fusion(self):
        vector_results = [_rc("a", 0.9, 1, "vector"), _rc("b", 0.1, 2, "vector")]
        fused = reciprocal_rank_fusion([vector_results])
        assert [r.rank for r in fused] == [1, 2]

    def test_source_is_hybrid(self):
        fused = reciprocal_rank_fusion([[_rc("a", 0.9, 1, "vector")]])
        assert fused[0].source == "hybrid"

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[]], k=0)

    def test_metadata_preserves_source_scores(self):
        vector_results = [_rc("a", 0.77, 1, "vector")]
        bm25_results = [_rc("a", 4.2, 1, "bm25")]
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        sources = fused[0].metadata["rrf_sources"]
        assert sources["vector_score"] == pytest.approx(0.77)
        assert sources["bm25_score"] == pytest.approx(4.2)

    def test_three_sources_majority_wins(self):
        a = [_rc("x", 0.9, 1, "s1"), _rc("y", 0.1, 2, "s1")]
        b = [_rc("y", 5.0, 1, "s2"), _rc("x", 1.0, 2, "s2")]
        c = [_rc("x", 0.5, 1, "s3")]
        fused = reciprocal_rank_fusion([a, b, c])
        assert len(fused) == 2
        assert fused[0].chunk_id == "x"  # ظاهر في 3 قوائم من 3