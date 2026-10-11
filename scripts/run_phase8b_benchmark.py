"""Phase 8-B / Step 2: retrieval benchmark on the expanded 8-document corpus."""

from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2
from rag_system.evaluation.retrieval_benchmark_v2 import RetrievalBenchmarkV2


def main() -> None:
    benchmark = RetrievalBenchmarkV2(top_k=5)
    reports = benchmark.run(GROUND_TRUTH_V2)

    print(f"\n{'Method':<16} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9} {'MRR':>7} {'Latency(ms)':>12}")
    print("-" * 70)
    for report in reports.values():
        s = report.summary()
        print(f"{s['method']:<16} {s['recall@1']:>9} {s['recall@3']:>9} {s['recall@5']:>9} {s['mrr']:>7} {s['avg_latency_ms']:>12}")

    # تفصيل حسب اللغة لكل method
    for name, report in reports.items():
        for lang in ("ar", "en"):
            subset = [r for r in report.query_results if r.language == lang]
            hits5 = sum(1 for r in subset if r.hit_at_k(5))
            print(f"  {name} [{lang}]: {hits5}/{len(subset)} hit@5")

    # الأسئلة اللي فشلت في hybrid_rrf (لنشوف نوعها)
    print(f"\n{'=' * 70}\nMISSES in hybrid_rrf (hit@5 failed):\n{'=' * 70}")
    for r in reports["hybrid_rrf"].query_results:
        if not r.hit_at_k(5):
            print(f"  [{r.language}] id={r.query_id}: {r.query!r}")


if __name__ == "__main__":
    main()