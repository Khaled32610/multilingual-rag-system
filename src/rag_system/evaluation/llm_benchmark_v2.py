"""
LLM benchmark v2 (Phase 9-B).

Runs candidate models over ground_truth_v2 (35 professionally validated
questions across 8 real documents). By default it uses BM25 retrieval only
(Phase 8-C decision: BM25 alone outperformed hybrid RRF on this ground
truth); any object exposing `retrieve(query, top_k)` can be passed instead
(see scripts/run_phase9b_fusion.py).

Results are saved incrementally to a JSON file after every single
generation, so a long-running multi-hour job can be interrupted and
resumed without losing completed work.

Phase 9-B fix: each result now records `prompt_tokens` (as reported by
Ollama) so context-window truncation can be spotted after the fact.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2, GroundTruthQueryV2
from rag_system.generation.prompt_builder import build_rag_prompt
from rag_system.llm.ollama_client import OllamaLLMClient
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.utils.exceptions import LLMError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")


def _detect_response_language(text: str) -> str:
    arabic_chars = len(_ARABIC_CHAR_PATTERN.findall(text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    if arabic_chars == 0 and latin_chars == 0:
        return "unknown"
    return "ar" if arabic_chars >= latin_chars else "en"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens (handles English and Arabic-Indic digits, %, commas)."""
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return set(re.findall(r"\d[\d,\.]*", text))


def _heuristic_correct(answer_text: str, generated: str) -> bool:
    """
    Cheap automated correctness proxy: does the generated answer contain
    the gold answer's key numbers, or a meaningful substring overlap
    with the gold answer text? Not a substitute for manual review, but
    useful for a first automated pass across 8 models. (scripts/rescore_phase9b.py
    is the punctuation-safe replacement used for real comparisons.)
    """
    gold_norm = _normalize(answer_text)
    gen_norm = _normalize(generated)

    gold_numbers = _extract_numbers(answer_text)
    if gold_numbers:
        gen_numbers = _extract_numbers(generated)
        if gold_numbers & gen_numbers:
            return True

    words = gold_norm.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if " ".join(words[i : i + 3]) in gen_norm:
                return True
    return gold_norm in gen_norm


@dataclass
class LLMResultV2:
    model: str
    query_id: int
    query: str
    language: str
    generated: str
    latency_ms: float
    response_language: str
    language_match: bool
    heuristic_correct: bool
    error: str | None = None
    prompt_tokens: int | None = None


class LLMBenchmarkV2:
    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        top_k: int = 5,
        results_path: str = "data/benchmarks/phase9b_results.json",
    ) -> None:
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.top_k = top_k
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: list[LLMResultV2] = self._load_existing()

    def _load_existing(self) -> list[LLMResultV2]:
        if not self.results_path.exists():
            return []
        with self.results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [LLMResultV2(**r) for r in data]

    def _save(self) -> None:
        with self.results_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.results], f, ensure_ascii=False, indent=2)

    def _already_done_lang(self, model: str, query_id: int, language: str) -> bool:
        return any(
            r.model == model and r.query_id == query_id and r.language == language
            for r in self.results
        )

    def run(
        self,
        model_names: list[str],
        ground_truth: list[GroundTruthQueryV2] = GROUND_TRUTH_V2,
        language: str = "ar",
    ) -> None:
        """
        Run the benchmark, skipping (model, query_id, language) triples
        already present in the results file (resume support).

        language: "ar", "en", or "both" (runs ar then en for every
        model, so results for a model are grouped together rather than
        interleaved across languages).
        """
        languages = ["ar", "en"] if language == "both" else [language]

        for model_name in model_names:
            client = OllamaLLMClient(model_name=model_name)

            for lang in languages:
                print(f"\n=== Model: {model_name} [{lang}] ===")

                for i, gt in enumerate(ground_truth, start=1):
                    if self._already_done_lang(model_name, gt.id, lang):
                        print(f"  [{i}/{len(ground_truth)}] id={gt.id} SKIP (already done)")
                        continue

                    query = gt.question_ar if lang == "ar" else gt.question_en
                    answer_text = gt.answer_ar if lang == "ar" else gt.answer_en

                    chunks = self.bm25_retriever.retrieve(query, top_k=self.top_k)
                    prompt = build_rag_prompt(query, chunks)

                    start = time.perf_counter()
                    prompt_tokens = None
                    try:
                        response = client.generate(prompt, temperature=0.1, max_tokens=512)
                        generated = response.text
                        latency_ms = response.latency_ms
                        prompt_tokens = response.prompt_tokens
                        error = None
                    except LLMError as e:
                        generated = ""
                        latency_ms = (time.perf_counter() - start) * 1000
                        error = str(e)
                        logger.warning(f"[{model_name}] failed on id={gt.id}: {e}")

                    response_language = _detect_response_language(generated) if generated else "unknown"
                    result = LLMResultV2(
                        model=model_name,
                        query_id=gt.id,
                        query=query,
                        language=lang,
                        generated=generated,
                        latency_ms=latency_ms,
                        response_language=response_language,
                        language_match=(response_language == lang),
                        heuristic_correct=_heuristic_correct(answer_text, generated) if generated else False,
                        error=error,
                        prompt_tokens=prompt_tokens,
                    )
                    self.results.append(result)
                    self._save()  # حفظ فوري بعد كل استدعاء

                    status = "ERR" if error else ("✅" if result.heuristic_correct else "❌")
                    print(
                        f"  [{i}/{len(ground_truth)}] id={gt.id} {status} "
                        f"({latency_ms:.0f}ms, tokens={prompt_tokens})"
                    )

    def summary(self) -> list[dict]:
        by_model: dict[str, list[LLMResultV2]] = {}
        for r in self.results:
            by_model.setdefault(r.model, []).append(r)

        rows = []
        for model, results in by_model.items():
            n = len(results)
            errors = sum(1 for r in results if r.error)
            correct = sum(1 for r in results if r.heuristic_correct)
            lang_match = sum(1 for r in results if r.language_match)
            ok_results = [r for r in results if not r.error]
            avg_latency = sum(r.latency_ms for r in ok_results) / len(ok_results) if ok_results else 0
            tokens = [r.prompt_tokens for r in results if r.prompt_tokens is not None]
            rows.append({
                "model": model,
                "n": n,
                "heuristic_accuracy": round(correct / n, 3) if n else 0,
                "language_match": round(lang_match / n, 3) if n else 0,
                "errors": errors,
                "avg_latency_ms": round(avg_latency, 0),
                "max_prompt_tokens": max(tokens) if tokens else None,
            })
        return rows