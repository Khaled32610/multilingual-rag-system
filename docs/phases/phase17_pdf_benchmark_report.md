# Phase 17 — PDF benchmark report

The PDF export converts Phase 15 evaluation JSON and an optional Phase 16 experiment index into a printable report with aggregate metrics, per-case outcomes, experiment parameters, and variant metrics. Install the optional renderer using `pip install -e ".[reports]"`; the core library does not require ReportLab.

```bash
python scripts/build_benchmark_pdf.py data/evaluation/latest.json \
  --experiment-json data/experiments/retrieval-ab/experiment.json \
  --output data/reports/rag_benchmark.pdf
```

The PDF includes no generated answer text; it reports numeric evaluation results and IDs to keep exports concise.
