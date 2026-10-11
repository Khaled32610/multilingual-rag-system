"""
Phase 9-B: does better retrieval improve the actual answers?

Runs gemma3:4b and qwen2.5:3b-instruct (AR + EN) under three retrieval setups,
each saved to its own results file so resume support never mixes configs:

  bm25_k5  : BM25 only, top 5 (baseline, with the new glossary translator)
  fused_k5 : weighted RRF over bm25 + vector + vector(translated query), top 5
  fused_k8 : same fusion, top 8
  bm25_k3 / fused_k3 : top 3 (shorter prompt, for latency)

Fusion settings come from data/benchmarks/phase9b_fusion_sweep.md
(bm25 weight 3.0, k=20, +vec_tr). Differences of 1-3 questions are within
noise for 35 questions, so judge by the re-scored numbers, not by this
script's heuristic column.

Usage (project root):
    python scripts/run_phase9b_fusion.py                 # all three
    python scripts/run_phase9b_fusion.py fused_k5        # one config
    P9B_MODELS=gemma3:4b python scripts/run_phase9b_fusion.py bm25_k5   # one model
"""

from __future__ import annotations

import os
import sys

from rag_system.evaluation.llm_benchmark_v2 import LLMBenchmarkV2
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.language_utils import detect_language
from rag_system.retrieval.rrf import weighted_reciprocal_rank_fusion
from rag_system.retrieval.vector_retriever import VectorRetriever

DEPTH = 100
MODELS = os.environ.get("P9B_MODELS", "gemma3:4b,qwen2.5:3b-instruct").split(",")


class FusedRetriever:
    """Duck-typed retriever: exposes retrieve(query, top_k) like BM25Retriever."""

    def __init__(
        self,
        bm25: BM25Retriever,
        vec: VectorRetriever,
        bm25_weight: float = 3.0,
        rrf_k: int = 20,
    ) -> None:
        self.bm25 = bm25
        self.vec = vec
        self.weights = {"bm25": bm25_weight, "vec": 1.0, "vec_tr": 1.0}
        self.rrf_k = rrf_k
        self._tr_cache: dict[str, str] = {}

    def _translate(self, query: str) -> str:
        if query in self._tr_cache:
            return self._tr_cache[query]
        translated = query
        translator = self.bm25.query_translator
        if translator is not None and detect_language(query) != "en":
            try:
                translated = translator.translate_to_english(query)
            except Exception:  # noqa: BLE001 - translation is best-effort
                translated = query
        self._tr_cache[query] = translated
        return translated

    def retrieve(self, query: str, top_k: int | None = None):
        lists = {
            "bm25": self.bm25.retrieve(query, top_k=DEPTH),
            "vec": self.vec.retrieve(query, top_k=DEPTH),
        }
        if detect_language(query) != "en":
            translated = self._translate(query)
            if translated != query:
                lists["vec_tr"] = self.vec.retrieve(translated, top_k=DEPTH)
        fused = weighted_reciprocal_rank_fusion(lists, self.weights, k=self.rrf_k)
        return fused[: top_k or 5]


def main() -> None:
    bm25 = BM25Retriever()
    fused = FusedRetriever(bm25, VectorRetriever())

    configs = {
        "bm25_k5": (bm25, 5),
        "fused_k5": (fused, 5),
        "fused_k8": (fused, 8),
        "bm25_k3": (bm25, 3),
        "fused_k3": (fused, 3),
    }
    wanted = sys.argv[1:] or ["bm25_k5", "fused_k5", "fused_k8"]
    unknown = [w for w in wanted if w not in configs]
    if unknown:
        raise SystemExit(f"Unknown config(s) {unknown}. Choose from {list(configs)}")

    for tag in wanted:
        retriever, top_k = configs[tag]
        path = f"data/benchmarks/phase9b_{tag}.json"
        print(f"\n########## {tag} -> {path} ##########")
        bench = LLMBenchmarkV2(bm25_retriever=retriever, top_k=top_k, results_path=path)
        bench.run(MODELS, language="both")
        for row in bench.summary():
            print(row)


if __name__ == "__main__":
    main()