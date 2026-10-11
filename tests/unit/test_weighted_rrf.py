"""Unit tests for weighted_reciprocal_rank_fusion (Phase 8-C)."""

import pytest

from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.rrf import weighted_reciprocal_rank_fusion


def _rc(chunk_id: str, score: float, rank: int, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=f"text {chunk_id}", metadata={},
        score=score, rank=rank, source=source,
    )


class TestWeightedReciprocalRankFusion:
    def test_equal_weights_matches_uniform_ranking(self):
        vec = [_rc("a", 0.9, 1, "vector"), _rc("b", 0.5, 2, "vector")]
        bm25 = [_rc("b", 5.0, 1, "bm25"), _rc("a", 1.0, 2, "bm25")]
        fused = weighted_reciprocal_rank_fusion(
            {"vector": vec, "bm25": bm25}, weights={"vector": 1.0, "bm25": 1.0}
        )
        # بنفس الأوزان، a و b لهم نفس score (كل واحد رتبة 1 مرة و2 مرة)
        assert fused[0].score == pytest.approx(fused[1].score)

    def test_higher_bm25_weight_favors_bm25_top_result(self):
        vec = [_rc("only_vector", 0.99, 1, "vector")]
        bm25 = [_rc("only_bm25", 5.0, 1, "bm25")]
        fused = weighted_reciprocal_rank_fusion(
            {"vector": vec, "bm25": bm25}, weights={"vector": 0.2, "bm25": 0.8}
        )
        assert fused[0].chunk_id == "only_bm25"
        assert fused[0].score > fused[1].score

    def test_missing_weight_defaults_to_one(self):
        vec = [_rc("a", 0.9, 1, "vector")]
        fused = weighted_reciprocal_rank_fusion({"vector": vec}, weights={})
        assert fused[0].score == pytest.approx(1.0 / 61.0)

    def test_source_is_hybrid_weighted(self):
        fused = weighted_reciprocal_rank_fusion(
            {"vector": [_rc("a", 0.9, 1, "vector")]}, weights={"vector": 1.0}
        )
        assert fused[0].source == "hybrid_weighted"

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            weighted_reciprocal_rank_fusion({}, weights={}, k=0)

    def test_metadata_includes_weight(self):
        fused = weighted_reciprocal_rank_fusion(
            {"bm25": [_rc("a", 3.0, 1, "bm25")]}, weights={"bm25": 0.7}
        )
        assert fused[0].metadata["rrf_sources"]["bm25_weight"] == 0.7