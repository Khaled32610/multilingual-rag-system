# Phase 15 — Full evaluation system

Adds a repeatable end-to-end evaluator over labeled questions. Each case records verified output, citation validity, whether labeled evidence was retrieved, strict normalized answer match, lexical token F1, latency, and runtime errors. Aggregate rates are calculated over successful cases while failure rate remains explicit.

The evaluator exports JSON for further analysis and Markdown for review. Run `python scripts/evaluate_rag.py`; use `--ids 1,2` or `--langs ar` for smaller runs. It uses the configured local Gemma model and the same model as the faithfulness judge, so Ollama and local indexes must be ready. A lexical token score is only a baseline and does not claim semantic correctness.
