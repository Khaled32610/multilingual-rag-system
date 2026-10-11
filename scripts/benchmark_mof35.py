"""Re-run the fixed 35-question bilingual retrieval benchmark and write 36 PDFs.

Run from the repository root: venv/bin/python scripts/benchmark_mof35.py
No LLM generation is scored. Retrieval never receives the gold passage or answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
IDS = tuple(str(i) for i in [*range(1, 25), *range(42, 47), *range(90, 96)])
WORKBOOK = ROOT / "WhatsApp Unknown 2026-09-24 at 18.57.26/MoF_RAG_Benchmark_GroundTruth.xlsx"


def metrics(results: list[dict], top_k: int = 5) -> dict:
    count = len(results)
    hit1 = sum(r["hit_rank_expected_source"] == 1 for r in results)
    hits = sum(
        r["hit_rank_expected_source"] is not None and r["hit_rank_expected_source"] <= top_k
        for r in results
    )
    return {
        "cases": count,
        "hits_at_1": hit1,
        "hits_at_5": hits,
        "hit_at_1_percent": round(100 * hit1 / count, 2) if count else 0,
        "hit_at_5_percent": round(100 * hits / count, 2) if count else 0,
        "exceeds_90_percent": bool(count and hits / count > 0.90),
    }


def validate_results(results: list[dict], rows: list[dict]) -> None:
    expected = {(row["id"], lang): row for row in rows for lang in ("ar", "en")}
    pairs = [(r["id"], r["language"]) for r in results]
    if len(pairs) != 70 or len(set(pairs)) != 70 or set(pairs) != set(expected):
        raise ValueError(
            "Expected exactly 35 fixed questions x 2 languages; refusing partial results"
        )
    for result in results:
        row = expected[(result["id"], result["language"])]
        for field, value in {
            "query": row[f'question_{result["language"]}'],
            "answer": row[f'answer_{result["language"]}'],
            "gold_chunk": row["gold_chunk"],
            "source_file": Path(row["original_path"]).name,
        }.items():
            if result[field] != value:
                raise ValueError(f"Reference mismatch: {result['id']} {result['language']} {field}")
        chunks = result["retrieved_chunks"]
        expected_ranks = [c["rank"] for c in chunks if c["expected_source_match"]]
        if result["hit_rank_expected_source"] != (min(expected_ranks) if expected_ranks else None):
            raise ValueError("Stored hit rank does not agree with retrieved chunks")


def fingerprints(paths: list[Path]) -> dict:
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_reports(payload: dict, rows: list[dict], index: list[dict], output: Path) -> dict:
    from rag_system.evaluation.benchmark_pdf import render_benchmark_pdf
    from scripts.evaluate_mof_workbook import _gold_match, _normalize

    results = payload["results"]
    validate_results(results, rows)
    row_map = {r["id"]: r for r in rows}
    chunk_map = {r["chunk_id"]: r for r in index}
    for result in results:
        result["workbook_record"] = row_map[result["id"]]
        for chunk in result["retrieved_chunks"]:
            record = chunk_map.get(chunk["chunk_id"])
            if record is None or record["text"] != chunk["text"]:
                raise ValueError("Results and current corpus differ; run a fresh benchmark")
            chunk["metadata"] = record.get("metadata", {})
            if record.get("metadata", {}).get("source_file") != chunk["source_file"]:
                raise ValueError("Retrieved source does not match indexed chunk metadata")
            actual_match = _gold_match(result["gold_chunk"], chunk["text"])
            expected_match = actual_match and chunk["source_file"] == result["source_file"]
            if (
                chunk["gold_match"] != actual_match
                or chunk["expected_source_match"] != expected_match
            ):
                raise ValueError("Stored match flags disagree with current gold matcher")
            chunk["normalized_exact_gold_match"] = bool(
                _normalize(result["gold_chunk"])
            ) and _normalize(result["gold_chunk"]) in _normalize(chunk["text"])
        exact_ranks = [
            c["rank"]
            for c in result["retrieved_chunks"]
            if c["source_file"] == result["source_file"] and c["normalized_exact_gold_match"]
        ]
        result["normalized_exact_gold_rank_expected_source"] = (
            min(exact_ranks) if exact_ranks else None
        )
    overall = metrics(results)
    by_language = {
        lang: metrics([r for r in results if r["language"] == lang]) for lang in ("ar", "en")
    }
    both = sum(
        all(r["hit_rank_expected_source"] is not None for r in results if r["id"] == qid)
        for qid in IDS
    )
    audit = {
        "scope": "Gold passage retrieval from the expected source; NOT generated-answer accuracy",
        "primary_metric": "Expected-source Hit@5, 70 cases (35 questions in Arabic and English)",
        "matching_rule": "Existing _gold_match: normalized containment OR >=85% unique gold-word overlap (>=4 words) OR any consecutive 6 gold words. Approximate passage match, not exact chunk-ID equality.",
        "overall": overall,
        "by_language": by_language,
        "normalized_exact_gold_hit_at_5": {
            lang: {
                "hits": sum(
                    r["normalized_exact_gold_rank_expected_source"] is not None
                    for r in results
                    if lang == "all" or r["language"] == lang
                ),
                "total": 70 if lang == "all" else 35,
            }
            for lang in ("all", "ar", "en")
        },
        "questions_successful_in_both_languages": {
            "hits": both,
            "total": 35,
            "percent": round(100 * both / 35, 2),
        },
        "indexed_documents": dict(
            Counter(r.get("metadata", {}).get("source_file", "") for r in index)
        ),
        "benchmark_source_documents": len({Path(r["original_path"]).name for r in rows}),
        "fixed_question_ids": list(IDS),
        "excluded_workbook_questions": 65,
        "note": "The 35-question subset is fixed. Missing required sources fail the run, never shrink the denominator. Existing retrieval code has previously been tuned on this workbook; this is not an unseen test set.",
    }
    summary = payload["summary"]
    summary.update(
        workbook_questions_selected=35,
        selected_question_language_pairs=70,
        supported_question_language_pairs_run=70,
        selected_question_pairs_with_missing_source=0,
        missing_source_files=[],
        audit=audit,
    )
    payload["full_text"] = True
    output.mkdir(parents=True, exist_ok=True)
    combined = output / "results.json"
    combined.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_benchmark_pdf(combined, output / "summary_and_all_questions.pdf")
    question_dir = output / "questions"
    question_dir.mkdir(exist_ok=True)
    for qid in IDS:
        subset = [r for r in results if r["id"] == qid]
        per_summary = {
            "mode": "retrieval",
            "retriever": summary["retriever"],
            "top_k": 5,
            "test_cases_run": 2,
            "supported_question_language_pairs_run": 2,
            "audit": {
                "question_id": qid,
                "metrics": metrics(subset),
                "reference": row_map[qid],
                "matching_rule": audit["matching_rule"],
            },
        }
        for source in ("any_source", "expected_source"):
            for k in (1, 5):
                per_summary[f"gold_chunk_hit_at_{k}_{source}"] = (
                    sum(
                        r[f"hit_rank_{source}"] is not None and r[f"hit_rank_{source}"] <= k
                        for r in subset
                    )
                    / 2
                )
        path = question_dir / f"question_{int(qid):03d}.json"
        path.write_text(
            json.dumps(
                {"summary": per_summary, "full_text": True, "results": subset},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        render_benchmark_pdf(path, path.with_suffix(".pdf"))
        print(f"PDF question {qid}: {path.with_suffix('.pdf')}", flush=True)
    (output / "summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmarks/mof35_latest")
    parser.add_argument(
        "--report-from",
        type=Path,
        help="Explicitly rebuild reports from existing results; does NOT run retrieval",
    )
    args = parser.parse_args()
    if not any(shutil.which(name) for name in ("google-chrome", "chromium", "chromium-browser")):
        parser.error("Google Chrome or Chromium is required for PDF reports")
    from rag_system.utils.config import get_settings
    from scripts.evaluate_mof_workbook import read_workbook

    settings = get_settings()
    rows = [r for r in read_workbook(WORKBOOK) if r["id"] in IDS]
    if len(rows) != 35 or {r["id"] for r in rows} != set(IDS):
        parser.error("The workbook must contain all 35 fixed question IDs")
    index_path = Path(settings.paths.bm25_index_dir) / "records.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {r.get("metadata", {}).get("source_file") for r in index}
    missing = {Path(r["original_path"]).name for r in rows} - indexed
    if missing:
        parser.error(f"Missing required indexed sources: {sorted(missing)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        WORKBOOK,
        ROOT / "configs/config.yaml",
        index_path,
        Path(settings.paths.vector_store_dir) / "faiss.index",
        Path(settings.paths.vector_store_dir) / "metadata.json",
    ]
    paths.append(ROOT / "scripts/evaluate_mof_workbook.py")
    for directory in ("retrieval", "reranking", "embedding"):
        paths.extend(sorted((ROOT / "src/rag_system" / directory).glob("*.py")))
    before = fingerprints(paths)
    started = datetime.now(UTC).isoformat()
    if args.report_from:
        source = args.report_from.resolve()
    else:
        source = output / "raw_results.json"
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_mof_workbook.py"),
            "--workbook",
            str(WORKBOOK),
            "--ids",
            ",".join(IDS),
            "--langs",
            "ar,en",
            "--mode",
            "retrieval",
            "--retriever",
            "reranked",
            "--top-k",
            "5",
            "--output",
            str(source.with_suffix("")),
            "--no-pdf",
        ]
        log_path = output / "run.log"
        print(f"Running 70 cases. Progress log: {log_path}", flush=True)
        with (
            log_path.open("w") as log,
            subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process,
        ):
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("["):
                    print(line.rstrip(), flush=True)
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f"Retrieval failed (exit {returncode}); inspect {log_path}")
    if fingerprints(paths) != before:
        raise RuntimeError(
            "Corpus/configuration changed during this run; refusing to publish mixed results"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["summary"].get("top_k") != 5 or payload["summary"].get("retriever") != "reranked":
        parser.error("This benchmark requires reranked retrieval with top_k=5")
    payload["provenance"] = {
        "report_created_at": datetime.now(UTC).isoformat(),
        "retrieval_started_at": None if args.report_from else started,
        "reused_existing_results": bool(args.report_from),
        "source_results": str(source),
        "sha256_at_report_time": before,
        "configuration": settings.yaml.model_dump(),
    }
    audit = write_reports(payload, rows, index, output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Reports: {output}")
    passed = audit["overall"]["exceeds_90_percent"] and all(
        m["exceeds_90_percent"] for m in audit["by_language"].values()
    )
    print(
        "PASS: Hit@5 exceeds 90% overall AND in each language"
        if passed
        else "FAIL: Hit@5 must exceed 90% overall AND in each language"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
