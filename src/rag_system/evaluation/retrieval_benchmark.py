"""
Retrieval benchmark (Phase 8 / Step 2).

Measures Recall@k, MRR, and latency for each retrieval method
(vector-only, BM25-only, hybrid RRF, hybrid+rerank) against a manually
labeled ground-truth query set, to make the weighting/reranking
trade-off decisions deferred from Phase 6 and Phase 7 with actual
numbers instead of intuition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    query: str
    language: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    latency_ms: float
    reciprocal_rank: float = field(init=False)

    def __post_init__(self) -> None:
        self.reciprocal_rank = 0.0
        for rank, cid in enumerate(self.retrieved_chunk_ids, start=1):
            if cid in self.relevant_chunk_ids:
                self.reciprocal_rank = 1.0 / rank
                break

    def hit_at_k(self, k: int) -> bool:
        return any(cid in self.relevant_chunk_ids for cid in self.retrieved_chunk_ids[:k])


@dataclass
class MethodReport:
    method_name: str
    query_results: list[QueryResult]

    def recall_at_k(self, k: int) -> float:
        if not self.query_results:
            return 0.0
        hits = sum(1 for r in self.query_results if r.hit_at_k(k))
        return hits / len(self.query_results)

    def mrr(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.query_results) / len(self.query_results)

    def avg_latency_ms(self) -> float:
        if not self.query_results:
            return 0.0
        return sum(r.latency_ms for r in self.query_results) / len(self.query_results)

    def summary(self) -> dict:
        return {
            "method": self.method_name,
            "recall@1": round(self.recall_at_k(1), 3),
            "recall@3": round(self.recall_at_k(3), 3),
            "recall@5": round(self.recall_at_k(5), 3),
            "mrr": round(self.mrr(), 3),
            "avg_latency_ms": round(self.avg_latency_ms(), 1),
        }


class RetrievalBenchmark:
    """
    Runs the full ground-truth query set through each retrieval method
    and produces a MethodReport per method.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        top_k: int = 5,
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            vector_retriever=self.vector_retriever, bm25_retriever=self.bm25_retriever
        )
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(
            hybrid_retriever=self.hybrid_retriever
        )
        self.top_k = top_k

    def _run_method(self, name: str, retrieve_fn, ground_truth: list) -> MethodReport:
        query_results = []
        for gt in ground_truth:
            start = time.perf_counter()
            try:
                results = retrieve_fn(gt.query)
            except Exception as e:
                logger.warning(f"[{name}] query failed: {gt.query!r} — {e}")
                results = []
            latency_ms = (time.perf_counter() - start) * 1000

            query_results.append(
                QueryResult(
                    query=gt.query,
                    language=gt.language,
                    retrieved_chunk_ids=[r.chunk_id for r in results],
                    relevant_chunk_ids=gt.relevant_chunk_ids,
                    latency_ms=latency_ms,
                )
            )
        return MethodReport(method_name=name, query_results=query_results)

    def run(self, ground_truth: list) -> dict[str, MethodReport]:
        """Run all 4 methods over the ground-truth set. Returns {method_name: MethodReport}."""
        reports = {}

        reports["vector_only"] = self._run_method(
            "vector_only",
            lambda q: self.vector_retriever.retrieve(q, top_k=self.top_k),
            ground_truth,
        )
        reports["bm25_only"] = self._run_method(
            "bm25_only",
            lambda q: self.bm25_retriever.retrieve(q, top_k=self.top_k),
            ground_truth,
        )
        reports["hybrid_rrf"] = self._run_method(
            "hybrid_rrf",
            lambda q: self.hybrid_retriever.retrieve(q, top_k=self.top_k),
            ground_truth,
        )
        reports["hybrid_rerank"] = self._run_method(
            "hybrid_rerank",
            lambda q: self.retrieval_pipeline.retrieve(q, top_n=self.top_k),
            ground_truth,
        )

        return reports