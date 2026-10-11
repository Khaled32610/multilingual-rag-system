# Phase 16 — Experiments

`ExperimentRunner` executes named pipeline variants over the same labeled cases, stores JSON and Markdown reports for every variant, and writes a machine-readable experiment index with parameters and run timestamps. `compare()` reports metric deltas against a selected baseline; negative latency deltas are faster, while positive quality-rate deltas are higher. Experiment and variant IDs are validated before creating paths.

Example:

```python
runner = ExperimentRunner("data/experiments")
results = runner.run("retrieval-ab", cases, {
    "baseline": (bm25_pipeline, {"retriever": "bm25", "top_k": 5}),
    "candidate": (hybrid_pipeline, {"retriever": "hybrid", "top_k": 5}),
})
print(runner.compare(results, baseline="baseline"))
```
