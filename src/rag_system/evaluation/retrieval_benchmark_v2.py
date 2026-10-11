"""
Retrieval benchmark v2 (Phase 8-B).

Like retrieval_benchmark.py (Phase 8), but matches on gold_chunk_text
substring presence in retrieved chunk text, rather than exact chunk_id
equality — necessary because ground_truth_v2's gold chunks were
labeled against a different chunking scheme than ours (see
ground_truth_v2.py docstring).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rag_system.evaluation.ground_truth_v2 import GroundTruthQueryV2
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for loose substring matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_match(gold_chunk_text: str, retrieved_text: str, min_overlap_words: int = 6) -> bool:
    """
    A retrieved chunk counts as a hit if a meaningful contiguous span of
    the gold chunk text appears in it. We check the gold text as a
    substring first (cheap, exact); if that fails, we fall back to
    checking whether a sliding window of `min_overlap_words` consecutive
    gold words appears verbatim in the retrieved text, since our
    chunking may have split the sentence differently than the gold
    labeling did.
    """
    gold_norm = _normalize(gold_chunk_text)
    retrieved_norm = _normalize(retrieved_text)

    if gold_norm in retrieved_norm:
        return True

    words = gold_norm.split()
    if len(words) < min_overlap_words:
        return gold_norm in retrieved_norm

    for i in range(len(words) - min_overlap_words + 1):
        window = " ".join(words[i : i + min_overlap_words])
        if window in retrieved_norm:
            return True
    return False


@dataclass
class QueryResultV2:
    query_id: int
    query: str
    language: str
    retrieved_texts: list[str]
    gold_chunk_text: str
    latency_ms: float
    hit_rank: int | None = field(init=False)

    def __post_init__(self) -> None:
        self.hit_rank = None
        for rank, text in enumerate(self.retrieved_texts, start=1):
            if _is_match(self.gold_chunk_text, text):
                self.hit_rank = rank
                break

    def hit_at_k(self, k: int) -> bool:
        return self.hit_rank is not None and self.hit_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0


@dataclass
class MethodReportV2:
    method_name: str
    query_results: list[QueryResultV2]

    def recall_at_k(self, k: int) -> float:
        if not self.query_results:
            return 0.0
        return sum(1 for r in self.query_results if r.hit_at_k(k)) / len(self.query_results)

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


class RetrievalBenchmarkV2:
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

    def _run_method(self, name: str, retrieve_fn, ground_truth: list[GroundTruthQueryV2]) -> MethodReportV2:
        results = []
        for gt in ground_truth:
            # كل سؤال عندنا نسختين (عربي وإنجليزي)؛ هنجرب الاتنين ونسجلهم منفصلين
            for lang, query in (("ar", gt.question_ar), ("en", gt.question_en)):
                start = time.perf_counter()
                try:
                    retrieved = retrieve_fn(query)
                except Exception as e:
                    logger.warning(f"[{name}] query failed: {query!r} — {e}")
                    retrieved = []
                latency_ms = (time.perf_counter() - start) * 1000

                results.append(
                    QueryResultV2(
                        query_id=gt.id,
                        query=query,
                        language=lang,
                        retrieved_texts=[r.text for r in retrieved],
                        gold_chunk_text=gt.gold_chunk_text,
                        latency_ms=latency_ms,
                    )
                )
        return MethodReportV2(method_name=name, query_results=results)

    def run(self, ground_truth: list[GroundTruthQueryV2]) -> dict[str, MethodReportV2]:
        reports = {}
        reports["vector_only"] = self._run_method(
            "vector_only", lambda q: self.vector_retriever.retrieve(q, top_k=self.top_k), ground_truth
        )
        reports["bm25_only"] = self._run_method(
            "bm25_only", lambda q: self.bm25_retriever.retrieve(q, top_k=self.top_k), ground_truth
        )
        reports["hybrid_rrf"] = self._run_method(
            "hybrid_rrf", lambda q: self.hybrid_retriever.retrieve(q, top_k=self.top_k), ground_truth
        )
        return reports