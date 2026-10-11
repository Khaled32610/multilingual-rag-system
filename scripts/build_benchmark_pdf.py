"""Build a PDF from Phase 15 evaluation JSON and optional Phase 16 experiment JSON."""

from __future__ import annotations

import argparse

from rag_system.evaluation.pdf_report import build_pdf_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", help="Phase 15 evaluation JSON")
    parser.add_argument("--experiment-json", help="Optional Phase 16 experiment index JSON")
    parser.add_argument("--output", default="data/reports/rag_benchmark.pdf")
    parser.add_argument("--title", default="RAG Benchmark Report")
    args = parser.parse_args()
    path = build_pdf_report(args.evaluation_json, args.output, args.experiment_json, args.title)
    print(f"PDF report saved to {path}")


if __name__ == "__main__":
    main()
