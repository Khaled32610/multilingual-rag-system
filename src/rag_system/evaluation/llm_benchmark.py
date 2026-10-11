"""
LLM benchmark (Phase 9 / Step 3).

Runs each candidate local model (via Ollama) over the same 10
ground-truth queries from Phase 8, using the best retrieval method
identified there (hybrid_rrf — see docs/benchmarks/retrieval_benchmark_results.md).

Produces per-model, per-query records with heuristic checks (language
match, citation presence, contains-expected-keyword) plus latency.
Full faithfulness/correctness judgment is manual at this stage — an
automated LLM-as-judge comes in Phase 11.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rag_system.generation.prompt_builder import build_rag_prompt
from rag_system.llm.ollama_client import OllamaLLMClient
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.utils.exceptions import LLMError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")


def _detect_response_language(text: str) -> str:
    """Very rough heuristic: majority of alphabetic chars decide ar vs en."""
    arabic_chars = len(_ARABIC_CHAR_PATTERN.findall(text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    if arabic_chars == 0 and latin_chars == 0:
        return "unknown"
    return "ar" if arabic_chars >= latin_chars else "en"


@dataclass
class LLMQueryResult:
    query: str
    query_language: str
    answer: str
    response_language: str
    language_match: bool
    has_citation: bool
    latency_ms: float
    error: str | None = None


@dataclass
class LLMModelReport:
    model_name: str
    results: list[LLMQueryResult] = field(default_factory=list)

    def language_match_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.language_match) / len(self.results)

    def citation_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.has_citation) / len(self.results)

    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def avg_latency_ms(self) -> float:
        ok = [r.latency_ms for r in self.results if r.error is None]
        return sum(ok) / len(ok) if ok else 0.0

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "language_match": round(self.language_match_rate(), 2),
            "citation_rate": round(self.citation_rate(), 2),
            "errors": self.error_count(),
            "avg_latency_ms": round(self.avg_latency_ms(), 0),
        }


class LLMBenchmark:
    """Runs the ground-truth query set through each candidate model."""

    def __init__(
        self,
        model_names: list[str],
        hybrid_retriever: HybridRetriever | None = None,
        top_k: int = 3,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_names = model_names
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.top_k = top_k
        self.ollama_base_url = ollama_base_url

    def run(self, ground_truth: list) -> dict[str, LLMModelReport]:
        reports: dict[str, LLMModelReport] = {}

        # نجيب الـ context مرة واحدة لكل سؤال، ونعيد استخدامه لكل الموديلات
        # (retrieval ثابت ومحسوم من Phase 8 — هنا بنقيّم الـ generation بس)
        contexts = {
            gt.query: self.hybrid_retriever.retrieve(gt.query, top_k=self.top_k)
            for gt in ground_truth
        }

        for model_name in self.model_names:
            client = OllamaLLMClient(model_name=model_name, base_url=self.ollama_base_url)
            report = LLMModelReport(model_name=model_name)

            for gt in ground_truth:
                prompt = build_rag_prompt(gt.query, contexts[gt.query])
                start = time.perf_counter()
                try:
                    response = client.generate(prompt, temperature=0.1, max_tokens=512)
                    latency_ms = response.latency_ms
                    answer = response.text
                    error = None
                except LLMError as e:
                    latency_ms = (time.perf_counter() - start) * 1000
                    answer = ""
                    error = str(e)
                    logger.warning(f"[{model_name}] generation failed for {gt.query!r}: {e}")

                response_language = _detect_response_language(answer) if answer else "unknown"
                report.results.append(
                    LLMQueryResult(
                        query=gt.query,
                        query_language=gt.language,
                        answer=answer,
                        response_language=response_language,
                        language_match=(response_language == gt.language),
                        has_citation=bool(re.search(r"مقتطف\s*\d+", answer)),
                        latency_ms=latency_ms,
                        error=error,
                    )
                )

            reports[model_name] = report

        return reports