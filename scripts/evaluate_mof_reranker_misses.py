"""Rerank top-15 hybrid candidates for cases missing gold in a benchmark's top five."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.evaluation.benchmark_pdf import render_benchmark_pdf
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.reranking.default import get_default_reranker
from rag_system.utils.config import get_settings
from rag_system.retrieval.retrieval_pipeline import _prepare_rerank_query
from scripts.evaluate_mof_workbook import _gold_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/evaluation/mof_semantic_paraphrases_2026-09-29.json",
    )
    parser.add_argument("--output", default="data/evaluation/mof_reranker_miss_diagnostics_2026-09-29")
    parser.add_argument("--pdf-output", default="data/benchmarks/mof_reranker_miss_diagnostics_2026-09-29.pdf")
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = [item for item in source["results"] if item["hit_rank_any_source"] is None]
    if not cases:
        parser.error("The input benchmark has no top-five gold misses")

    settings = get_settings()
    hybrid = HybridRetriever()
    reranker = get_default_reranker()
    final_results = []
    candidate_recall = 0

    for ordinal, case in enumerate(cases, start=1):
        query = case["query"]
        requested = 5
        candidate_count = max(requested * 3, settings.retrieval.top_k)
        candidates = hybrid.retrieve(query, top_k=candidate_count)
        candidate_matches = [
            rank for rank, chunk in enumerate(candidates, start=1)
            if _gold_match(case["gold_chunk"], chunk.text)
        ]
        expected_matches = [
            rank for rank, chunk in enumerate(candidates, start=1)
            if _gold_match(case["gold_chunk"], chunk.text)
            and chunk.metadata.get("source_file") == case["source_file"]
        ]
        if candidate_matches:
            candidate_recall += 1

        translator = getattr(hybrid.bm25_retriever, "query_translator", None)
        rerank_query = _prepare_rerank_query(query, candidates, translator)

        reranked = reranker.rerank(rerank_query, candidates, top_n=requested)
        any_hits = [
            rank for rank, chunk in enumerate(reranked, start=1)
            if _gold_match(case["gold_chunk"], chunk.text)
        ]
        expected_hits = [
            rank for rank, chunk in enumerate(reranked, start=1)
            if _gold_match(case["gold_chunk"], chunk.text)
            and chunk.metadata.get("source_file") == case["source_file"]
        ]
        item = {
            key: case[key]
            for key in ("id", "category", "source_file", "language", "variant", "query", "answer", "gold_chunk")
        }
        item.update({
            "candidate_count": len(candidates),
            "candidate_rank_before_rerank": candidate_matches[0] if candidate_matches else None,
            "candidate_expected_source_rank_before_rerank": expected_matches[0] if expected_matches else None,
            "hit_rank_any_source": any_hits[0] if any_hits else None,
            "hit_rank_expected_source": expected_hits[0] if expected_hits else None,
            "retrieved_chunks": [
                {
                    "rank": rank,
                    "source_file": chunk.metadata.get("source_file", ""),
                    "chunk_id": chunk.chunk_id,
                    "score": round(float(chunk.score), 6),
                    "gold_match": _gold_match(case["gold_chunk"], chunk.text),
                    "expected_source_match": (
                        _gold_match(case["gold_chunk"], chunk.text)
                        and chunk.metadata.get("source_file") == case["source_file"]
                    ),
                    "text": chunk.text,
                }
                for rank, chunk in enumerate(reranked, start=1)
            ],
        })
        final_results.append(item)
        print(
            f"[{ordinal}/{len(cases)}] id={case['id']} {case['language']} {case['variant']} "
            f"candidate_rank={item['candidate_rank_before_rerank']} reranked_rank={item['hit_rank_any_source']}",
            flush=True,
        )

    count = len(final_results)
    summary = {
        "mode": "retrieval",
        "retriever": "bge-reranker-v2-m3 on top-15 hybrid candidates",
        "top_k": 5,
        "candidate_count": 15,
        "test_cases_run": count,
        "paraphrase_test_cases_run": sum(item["variant"] == "paraphrase" for item in final_results),
        "gold_candidate_hit_at_15_any_source": round(candidate_recall / count, 4),
        "gold_chunk_hit_at_1_any_source": round(sum(item["hit_rank_any_source"] == 1 for item in final_results) / count, 4),
        "gold_chunk_hit_at_5_any_source": round(sum(item["hit_rank_any_source"] is not None for item in final_results) / count, 4),
        "gold_chunk_hit_at_1_expected_source": round(sum(item["hit_rank_expected_source"] == 1 for item in final_results) / count, 4),
        "gold_chunk_hit_at_5_expected_source": round(sum(item["hit_rank_expected_source"] is not None for item in final_results) / count, 4),
        "variant_metrics": {},
    }
    for variant in ("original", "paraphrase"):
        subset = [item for item in final_results if item["variant"] == variant]
        if subset:
            summary["variant_metrics"][variant] = {
                "count": len(subset),
                "gold_chunk_hit_at_1_any_source": round(sum(item["hit_rank_any_source"] == 1 for item in subset) / len(subset), 4),
                "gold_chunk_hit_at_5_any_source": round(sum(item["hit_rank_any_source"] is not None for item in subset) / len(subset), 4),
                "gold_chunk_hit_at_1_expected_source": round(sum(item["hit_rank_expected_source"] == 1 for item in subset) / len(subset), 4),
                "gold_chunk_hit_at_5_expected_source": round(sum(item["hit_rank_expected_source"] is not None for item in subset) / len(subset), 4),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps({"summary": summary, "results": final_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = Path(args.pdf_output)
    render_benchmark_pdf(json_path, pdf_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
