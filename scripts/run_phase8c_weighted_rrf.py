"""
Phase 8-C: test whether weighting RRF toward BM25 beats BM25 alone,
given Phase 8-B's finding that uniform hybrid_rrf underperforms
bm25_only on the expanded 8-document corpus.

Fetches vector + BM25 results once per query, then re-fuses with
several weight combinations without re-querying, so this is fast.
"""

from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.rrf import weighted_reciprocal_rank_fusion
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.evaluation.retrieval_benchmark_v2 import _is_match

WEIGHT_COMBOS = [
    {"vector": 1.0, "bm25": 1.0},   # uniform (Phase 8-B baseline)
    {"vector": 0.3, "bm25": 0.7},
    {"vector": 0.2, "bm25": 0.8},
    {"vector": 0.1, "bm25": 0.9},
    {"vector": 0.0, "bm25": 1.0},   # BM25 only via weighted fusion (sanity check)
]

TOP_K_FETCH = 10
EVAL_K = 5


def main() -> None:
    vector_retriever = VectorRetriever()
    bm25_retriever = BM25Retriever()

    # Step 1: نجيب نتائج كل مصدر مرة واحدة بس لكل سؤال (عربي + إنجليزي)
    cached = []  # list of (gt, lang, query, vector_results, bm25_results)
    for gt in GROUND_TRUTH_V2:
        for lang, query in (("ar", gt.question_ar), ("en", gt.question_en)):
            vec_results = vector_retriever.retrieve(query, top_k=TOP_K_FETCH)
            bm25_results = bm25_retriever.retrieve(query, top_k=TOP_K_FETCH)
            cached.append((gt, lang, query, vec_results, bm25_results))

    print(f"Cached retrieval for {len(cached)} query instances.\n")

    # Step 2: لكل توزيع أوزان، نعيد الدمج ونحسب hit@5 و MRR
    print(f"{'Weights':<30} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} {'MRR':>7}")
    print("-" * 68)

    for weights in WEIGHT_COMBOS:
        hits_at = {1: 0, 3: 0, 5: 0}
        reciprocal_ranks = []

        for gt, lang, query, vec_results, bm25_results in cached:
            fused = weighted_reciprocal_rank_fusion(
                {"vector": vec_results, "bm25": bm25_results}, weights=weights
            )[:EVAL_K]

            hit_rank = None
            for rank, chunk in enumerate(fused, start=1):
                if _is_match(gt.gold_chunk_text, chunk.text):
                    hit_rank = rank
                    break

            for k in (1, 3, 5):
                if hit_rank is not None and hit_rank <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / hit_rank if hit_rank else 0.0)

        n = len(cached)
        recall1 = hits_at[1] / n
        recall3 = hits_at[3] / n
        recall5 = hits_at[5] / n
        mrr = sum(reciprocal_ranks) / n

        weight_label = f"v={weights['vector']:.1f} b={weights['bm25']:.1f}"
        print(f"{weight_label:<30} {recall1:>9.3f} {recall3:>9.3f} {recall5:>9.3f} {mrr:>7.3f}")


if __name__ == "__main__":
    main()