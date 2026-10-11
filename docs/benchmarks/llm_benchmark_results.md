# Phase 9-B — Local LLM benchmark and model selection

## Setup

The benchmark used 70 questions (35 Arabic and 35 English) over the repaired 772-chunk corpus. Eight local Ollama models were compared for answer accuracy, language match, errors, and latency. The Phase 9-B detailed results and model disagreements are retained under `data/benchmarks/`.

## Overall results

| Model | Accuracy | Language match | Errors | Mean latency |
|---|---:|---:|---:|---:|
| **gemma3:4b** | **0.800** | 0.900 | 0 | 28.6 s |
| qwen2.5:3b-instruct | 0.771 | 0.971 | 0 | 21.8 s |
| SILMA-9B (Q4_K_S) | 0.671 | 0.657 | 0 | 66.6 s |
| phi4-mini | 0.557 | 0.729 | 0 | 22.2 s |
| ALLaM-7B (Q4_K_M) | 0.514 | 0.757 | 0 | 55.3 s |
| aya:latest | 0.271 | 0.529 | 0 | 58.8 s |
| AceGPT-7B (Q4_K_M) | 0.243 | 0.786 | 0 | 111.4 s |
| jais-family-6p7b (Q4_K_M) | 0.143 | 0.414 | 0 | 23.4 s |

Arabic-only accuracy was 0.686 for Gemma and 0.629 for Qwen; English accuracy was 0.914 for Gemma and 0.914 for Qwen. The full result tables are in the progress report and `data/benchmarks/phase9b_results.json`.

## Punctuation-safe re-score

The saved 70-question outputs were also re-scored with citation markers removed before numeric matching, Arabic-Indic digits normalized, refusals excluded from correct answers, and question 94 excluded because its gold answer is absent from extracted text. This leaves 68 evaluated answers per model (34 per language).

| Model | Primary, all | Strict, all | Primary, Arabic | Primary, English |
|---|---:|---:|---:|---:|
| **gemma3:4b** | **0.779** | **0.618** | **0.647** | **0.912** |
| qwen2.5:3b-instruct | 0.721 | 0.588 | 0.559 | 0.882 |

`Primary` accepts the first gold number or a content-word overlap threshold; `strict` requires all gold numbers. These are automated comparison signals, not a substitute for manual evaluation. The full scorer and per-question audit are `scripts/rescore_phase9b.py` and `data/benchmarks/phase9b_rescore_audit.md`.

## Selection

**Selected default: `gemma3:4b`.** It has the highest measured overall accuracy in this benchmark and ties Qwen on English accuracy while scoring higher on Arabic. Qwen remains a reasonable alternative when latency or language-match rate is more important: it was about 24% faster and had a 0.971 language-match rate versus Gemma's 0.900.

The comparison is based on this project's 70-question corpus and the benchmark's heuristic/manual rescoring; it does not establish that Gemma is more accurate on every workload. No other model in the tested set exceeded its measured overall accuracy.

## Configuration

The selected model is the default in both `configs/config.yaml` and the `LLMConfig` fallback. Individual clients can still be constructed with another Ollama model name for experiments.
