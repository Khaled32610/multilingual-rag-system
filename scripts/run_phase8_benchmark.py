"""Phase 8 / Step 3: run the retrieval benchmark and print a comparison table."""

from rag_system.evaluation.retrieval_benchmark import RetrievalBenchmark
from rag_system.evaluation.ground_truth import GROUND_TRUTH


def main() -> None:
    benchmark = RetrievalBenchmark(top_k=5)
    reports = benchmark.run(GROUND_TRUTH)

    print(f"\n{'Method':<16} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} {'MRR':>7} {'Latency(ms)':>12}")
    print("-" * 70)
    for report in reports.values():
        s = report.summary()
        print(
            f"{s['method']:<16} {s['recall@1']:>9} {s['recall@3']:>9} "
            f"{s['recall@5']:>9} {s['mrr']:>7} {s['avg_latency_ms']:>12}"
        )

    # تفصيل حسب اللغة لكل method — يفيدنا في قرار ترجيح RRF (Phase 6 المعلّق)
    print(f"\n{'=' * 70}\nPer-query detail (hybrid_rrf vs hybrid_rerank):\n{'=' * 70}")
    for name in ("hybrid_rrf", "hybrid_rerank"):
        print(f"\n--- {name} ---")
        for r in reports[name].query_results:
            hit = "✅" if r.hit_at_k(5) else "❌"
            print(f"  {hit} [{r.language}] rr={r.reciprocal_rank:.2f} lat={r.latency_ms:.0f}ms  {r.query!r}")


if __name__ == "__main__":
    main()