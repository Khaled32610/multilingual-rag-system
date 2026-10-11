"""Phase 9-B analysis: fair comparison across all 7 working models,
filtered to Arabic-only results (some models were only run in Arabic)."""

import json
from collections import defaultdict

with open("data/benchmarks/phase9b_results.json", encoding="utf-8") as f:
    data = json.load(f)

ar_results = [r for r in data if r["language"] == "ar"]

by_model = defaultdict(list)
for r in ar_results:
    by_model[r["model"]].append(r)

print(f"{'Model':<55} {'N':>4} {'Accuracy':>9} {'LangMatch':>10} {'Errors':>7} {'Latency(s)':>11}")
print("-" * 100)

rows = []
for model, results in by_model.items():
    n = len(results)
    correct = sum(1 for r in results if r["heuristic_correct"])
    lang_match = sum(1 for r in results if r["language_match"])
    errors = sum(1 for r in results if r["error"])
    ok = [r for r in results if not r["error"]]
    avg_latency = sum(r["latency_ms"] for r in ok) / len(ok) / 1000 if ok else 0
    rows.append((model, n, correct / n, lang_match / n, errors, avg_latency))

for model, n, acc, lm, err, lat in sorted(rows, key=lambda x: -x[2]):
    print(f"{model:<55} {n:>4} {acc:>9.3f} {lm:>10.3f} {err:>7} {lat:>11.1f}")
