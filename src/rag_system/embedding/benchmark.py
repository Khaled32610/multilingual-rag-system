"""
Embedding model benchmark (Phase 3 / Step 3).

Compares candidate embedding models on:
1. Speed (texts/second for encoding).
2. Retrieval quality: given a small set of (query, expected_chunk) pairs,
   measures Recall@K -- whether the correct chunk appears anywhere in the
   top-K results, not just as the single top-ranked result. This matches
   how the chunk will actually be used downstream: top-K candidates go on
   to a reranker (Phase 7), so a chunk ranked 2nd or 3rd is still a win.

This produces a report used in Step 4 to select the system default model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rag_system.embedding.base import BaseEmbeddingModel
from rag_system.utils.exceptions import EmbeddingBenchmarkError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TOP_K = 3


@dataclass
class RetrievalTestCase:
    query: str
    expected_keyword: str


@dataclass
class ModelBenchmarkResult:
    display_name: str
    model_name: str
    dimension: int
    embedding_texts_per_second: float
    query_latency_ms: float
    retrieval_accuracy: float
    retrieval_hits: int
    retrieval_total: int
    top_k: int = _DEFAULT_TOP_K
    errors: list = field(default_factory=list)


class EmbeddingBenchmark:
    def __init__(self, test_cases, top_k: int = _DEFAULT_TOP_K) -> None:
        if not test_cases:
            raise EmbeddingBenchmarkError("At least one retrieval test case is required")
        self.test_cases = test_cases
        self.top_k = top_k

    def run(self, model: BaseEmbeddingModel, display_name: str, chunk_texts):
        logger.info(f"Benchmarking model: {display_name}")
        errors = []

        try:
            corpus_result = model.embed_texts(chunk_texts)
            texts_per_second = corpus_result.texts_per_second
            corpus_vectors = corpus_result.vectors
        except Exception as e:
            msg = f"Failed to embed corpus: {e}"
            errors.append(msg)
            logger.error(f"{display_name}: {msg}")
            return ModelBenchmarkResult(
                display_name=display_name,
                model_name=model.name,
                dimension=getattr(model, "dimension", 0),
                embedding_texts_per_second=0.0,
                query_latency_ms=0.0,
                retrieval_accuracy=0.0,
                retrieval_hits=0,
                retrieval_total=len(self.test_cases),
                top_k=self.top_k,
                errors=errors,
            )

        hits = 0
        query_latencies = []

        for case in self.test_cases:
            try:
                start = time.perf_counter()
                query_result = model.embed_query(case.query)
                query_latencies.append((time.perf_counter() - start) * 1000)

                query_vector = query_result.vectors[0]
                similarities = corpus_vectors @ query_vector
                top_k_indices = np.argsort(similarities)[::-1][: self.top_k]

                found_in_top_k = any(
                    case.expected_keyword.lower() in chunk_texts[idx].lower()
                    for idx in top_k_indices
                )

                if found_in_top_k:
                    hits += 1
                else:
                    logger.debug(
                        f"{display_name}: miss for query \'{case.query}\' "
                        f"(expected \'{case.expected_keyword}\') in top-{self.top_k}"
                    )
            except Exception as e:
                errors.append(f"Query failed: {case.query} - {e}")

        accuracy = hits / len(self.test_cases) if self.test_cases else 0.0
        avg_latency = float(np.mean(query_latencies)) if query_latencies else 0.0

        result = ModelBenchmarkResult(
            display_name=display_name,
            model_name=model.name,
            dimension=corpus_result.dimension,
            embedding_texts_per_second=texts_per_second,
            query_latency_ms=avg_latency,
            retrieval_accuracy=accuracy,
            retrieval_hits=hits,
            retrieval_total=len(self.test_cases),
            top_k=self.top_k,
            errors=errors,
        )

        logger.info(
            f"{display_name}: {texts_per_second:.1f} texts/sec, "
            f"{avg_latency:.1f}ms/query, "
            f"recall@{self.top_k}={accuracy:.1%} ({hits}/{len(self.test_cases)})"
        )

        return result

    def run_all(self, models, chunk_texts):
        results = [
            self.run(model, display_name, chunk_texts)
            for display_name, model in models.items()
        ]
        results.sort(key=lambda r: (r.retrieval_accuracy, r.embedding_texts_per_second), reverse=True)
        return results

    @staticmethod
    def print_report(results) -> None:
        top_k = results[0].top_k if results else _DEFAULT_TOP_K
        print()
        print("=" * 90)
        print(f"EMBEDDING MODEL BENCHMARK REPORT (Recall@{top_k})")
        print("=" * 90)
        header = f"{'Model':<28} {'Dim':>6} {'Texts/sec':>12} {'Query ms':>10} {'Recall@K':>12}"
        print(header)
        print("-" * 90)
        for r in results:
            print(
                f"{r.display_name:<28} {r.dimension:>6} "
                f"{r.embedding_texts_per_second:>12.1f} {r.query_latency_ms:>10.1f} "
                f"{r.retrieval_accuracy:>11.1%} ({r.retrieval_hits}/{r.retrieval_total})"
            )
            if r.errors:
                for err in r.errors:
                    print(f"    [!] {err}")
        print("=" * 90)
        print()
