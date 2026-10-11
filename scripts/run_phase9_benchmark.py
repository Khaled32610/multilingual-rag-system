"""Phase 9 / Step 3: run the LLM benchmark and print results + full transcripts."""

from rag_system.evaluation.ground_truth import GROUND_TRUTH
from rag_system.evaluation.llm_benchmark import LLMBenchmark
from rag_system.llm.ollama_client import OllamaLLMClient

MODELS = ["aya:latest", "qwen2.5:3b-instruct", "gemma3:4b", "phi4-mini"]


def main() -> None:
    if not OllamaLLMClient.is_server_running():
        print("Ollama server not reachable at localhost:11434. Run `ollama serve` first.")
        return

    benchmark = LLMBenchmark(model_names=MODELS, top_k=3)
    reports = benchmark.run(GROUND_TRUTH)

    print(f"\n{'Model':<22} {'Lang Match':>10} {'Citation':>10} {'Errors':>8} {'Latency(ms)':>12}")
    print("-" * 68)
    for report in reports.values():
        s = report.summary()
        print(
            f"{s['model']:<22} {s['language_match']:>10} {s['citation_rate']:>10} "
            f"{s['errors']:>8} {s['avg_latency_ms']:>12}"
        )

    # Transcripts كاملة عشان تقرأها وتحكم بنفسك على الصحة والالتزام بالسياق
    for model_name, report in reports.items():
        print(f"\n{'=' * 70}\nMODEL: {model_name}\n{'=' * 70}")
        for r in report.results:
            print(f"\n[{r.query_language}->{r.response_language}] {r.query}")
            if r.error:
                print(f"  ERROR: {r.error}")
            else:
                print(f"  ({r.latency_ms:.0f}ms) {r.answer}")


if __name__ == "__main__":
    main()