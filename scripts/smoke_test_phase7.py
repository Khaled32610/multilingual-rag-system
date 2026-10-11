"""Smoke test for Phase 7 — Reranking on top of Hybrid Retrieval."""

from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

QUESTIONS = [
    "الموازنة العامة للدولة",
    "government budget revenue",
    "وزير المالية",
]


def main() -> None:
    pipeline = RetrievalPipeline()  # بيحمّل الفهارس المحفوظة من Phase 6

    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQuery: {q}\n{'=' * 70}")
        results = pipeline.retrieve(q, top_n=5)
        if not results:
            print("  (no results)")
            continue
        for r in results:
            pre_rank = r.metadata.get("pre_rerank_rank")
            pre_score = r.metadata.get("pre_rerank_score")
            print(f"  rank={r.rank}  rerank_score={r.score:.4f}  (was rank={pre_rank}, rrf_score={pre_score:.5f})")
            print(f"    text: {r.text[:100]!r}")


if __name__ == "__main__":
    main()