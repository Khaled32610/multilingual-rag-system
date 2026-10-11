"""
Reciprocal Rank Fusion (RRF) for combining ranked retrieval results
(Phase 6 / Step 1).

Fuses ranked lists from different retrieval methods (vector, BM25, ...)
based on *rank* rather than raw score, since scores from different
methods live on incomparable scales (cosine similarity 0-1 vs
unbounded BM25 scores).
"""

from __future__ import annotations

from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """
    Fuse multiple ranked lists of RetrievedChunk into one ranked list.

    RRF_score(chunk) = sum over lists containing chunk of 1 / (k + rank)

    Args:
        ranked_lists: قوائم نتائج مرتبة بالفعل (الأفضل أولاً)، كل قائمة من
            مصدر استرجاع مختلف (مثلاً: [vector_results, bm25_results]).
        k: ثابت RRF (افتراضي 60). قيمة أكبر = تقليل تأثير الفروق بين أوائل الرتب.

    Returns:
        قائمة RetrievedChunk واحدة مدمجة، بدون تكرار لنفس chunk_id، مرتبة
        تنازلياً حسب rrf score. الحقل `score` يحمل الـ RRF score النهائي
        (مش الـ score الأصلي)، `rank` يُعاد حسابه بعد الدمج، `source="hybrid"`،
        والـ scores/ranks الأصلية من كل مصدر تتحفظ جوه `metadata`.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}
    source_info: dict[str, dict] = {}

    for source_idx, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list, start=1):
            cid = item.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)

            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
                source_info[cid] = {}
            source_info[cid][f"{item.source}_score"] = item.score
            source_info[cid][f"{item.source}_rank"] = rank

    fused: list[RetrievedChunk] = []
    for cid, rrf_score in fused_scores.items():
        original = chunk_lookup[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=original.chunk_id,
                text=original.text,
                metadata={**original.metadata, "rrf_sources": source_info[cid]},
                score=rrf_score,
                rank=0,  # يُحدَّث بعد الترتيب
                source="hybrid",
            )
        )

    fused.sort(key=lambda r: r.score, reverse=True)
    for rank, r in enumerate(fused, start=1):
        r.rank = rank

    logger.info(
        f"RRF fusion: {len(ranked_lists)} source list(s) -> "
        f"{len(fused)} unique chunk(s) (k={k})"
    )
    return fused

def weighted_reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]],
    weights: dict[str, float],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """
    Weighted variant of reciprocal_rank_fusion (Phase 8-C).

    Like reciprocal_rank_fusion, but each source's contribution is
    scaled by a weight before summing:
        score(chunk) = sum_source( weight[source] / (k + rank) )

    Args:
        ranked_lists: {source_name: ranked results}, e.g.
            {"vector": vector_results, "bm25": bm25_results}.
        weights: {source_name: weight}. Sources not present default to
            weight 1.0 (uniform). Weights need not sum to 1.
        k: RRF k constant (see reciprocal_rank_fusion).

    Returns:
        Fused, deduplicated, re-ranked list of RetrievedChunk.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}
    source_info: dict[str, dict] = {}

    for source_name, ranked_list in ranked_lists.items():
        w = weights.get(source_name, 1.0)
        for rank, item in enumerate(ranked_list, start=1):
            cid = item.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + w / (k + rank)

            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
                source_info[cid] = {}
            source_info[cid][f"{source_name}_score"] = item.score
            source_info[cid][f"{source_name}_rank"] = rank
            source_info[cid][f"{source_name}_weight"] = w

    fused: list[RetrievedChunk] = []
    for cid, score in fused_scores.items():
        original = chunk_lookup[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=original.chunk_id,
                text=original.text,
                metadata={**original.metadata, "rrf_sources": source_info[cid]},
                score=score,
                rank=0,
                source="hybrid_weighted",
            )
        )

    fused.sort(key=lambda r: r.score, reverse=True)
    for rank, r in enumerate(fused, start=1):
        r.rank = rank

    logger.info(
        f"Weighted RRF fusion: {len(ranked_lists)} source(s) with weights={weights} -> "
        f"{len(fused)} unique chunk(s) (k={k})"
    )
    return fused
