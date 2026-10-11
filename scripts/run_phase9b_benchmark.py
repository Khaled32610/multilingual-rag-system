"""
Phase 9-B: LLM benchmark on ground_truth_v2 (35 questions, 8 documents).

Usage:
    python scripts/run_phase9b_benchmark.py --models gemma3:4b qwen2.5:3b-instruct
    python scripts/run_phase9b_benchmark.py --models hf.co/bartowski/SILMA-9B-Instruct-v1.0-GGUF:Q4_K_S
    python scripts/run_phase9b_benchmark.py --summary   # يطبع الملخص من النتائج المحفوظة فقط
"""

import argparse

from rag_system.evaluation.llm_benchmark_v2 import LLMBenchmarkV2
from rag_system.llm.ollama_client import OllamaLLMClient

ALL_MODELS = [
    "qwen2.5:3b-instruct",
    "gemma3:4b",
    "llama3.2:3b"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--language", choices=["ar", "en", "both"], default="both")
    parser.add_argument("--summary", action="store_true", help="Print summary from saved results only")
    args = parser.parse_args()

    benchmark = LLMBenchmarkV2(top_k=5)

    if not args.summary:
        if not OllamaLLMClient.is_server_running():
            print("Ollama server not reachable. Run `ollama serve` first.")
            return
        benchmark.run(model_names=args.models, language=args.language)

    print(f"\n{'Model':<55} {'N':>4} {'Accuracy':>9} {'LangMatch':>10} {'Errors':>7} {'Latency(ms)':>12}")
    print("-" * 100)
    for row in benchmark.summary():
        print(
            f"{row['model']:<55} {row['n']:>4} {row['heuristic_accuracy']:>9} "
            f"{row['language_match']:>10} {row['errors']:>7} {row['avg_latency_ms']:>12}"
        )


if __name__ == "__main__":
    main()