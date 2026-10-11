"""Evaluate retrieval or end-to-end RAG against the supplied MoF Excel workbook."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from rag_system.evaluation.full_evaluation import answer_token_f1
from rag_system.generation.grounded_generator import GroundedGenerator
from rag_system.llm.default import get_default_llm_client
from rag_system.pipeline.rag_pipeline import RAGPipeline
from rag_system.reranking.default import get_default_reranker
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.verification.faithfulness_checker import FaithfulnessChecker
from rag_system.verification.pipeline import VerificationPipeline

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _column_index(cell_reference: str) -> int:
    value = 0
    for letter in re.findall(r"[A-Za-z]+", cell_reference)[0].upper():
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def read_workbook(path: str | Path) -> list[dict[str, str]]:
    """Read the visible values from the workbook's Benchmark sheet (stdlib only)."""
    try:
        with zipfile.ZipFile(path) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                strings = [
                    "".join(text.text or "" for text in item.findall(".//main:t", _NS))
                    for item in shared.findall("main:si", _NS)
                ]
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {item.attrib["Id"]: item.attrib["Target"] for item in relations}
            sheet = next(
                item for item in workbook.findall("main:sheets/main:sheet", _NS)
                if item.attrib["name"] == "Benchmark"
            )
            target = targets[sheet.attrib[f"{{{_NS['rel']}}}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            worksheet = ET.fromstring(archive.read(target))
    except (OSError, KeyError, StopIteration, zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError(f"Could not read Benchmark sheet from {path}: {exc}") from exc

    rows: list[dict[int, str]] = []
    for row in worksheet.findall(".//main:sheetData/main:row", _NS):
        values: dict[int, str] = {}
        for cell in row.findall("main:c", _NS):
            kind = cell.attrib.get("t")
            value = cell.find("main:v", _NS)
            text = value.text if value is not None and value.text else ""
            if kind == "s" and text:
                text = strings[int(text)]
            elif kind == "inlineStr":
                text = "".join(node.text or "" for node in cell.findall(".//main:t", _NS))
            values[_column_index(cell.attrib["r"])] = text
        rows.append(values)
    if not rows:
        raise ValueError("Benchmark worksheet is empty")
    headers = rows[0]
    records = [
        {headers[column]: row.get(column, "") for column in headers}
        for row in rows[1:]
        if row.get(0)
    ]
    return records


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text)
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _gold_match(gold: str, content: str) -> bool:
    expected, actual = _normalize(gold), _normalize(content)
    if not expected:
        return False
    if expected in actual:
        return True
    words = expected.split()
    actual_words = set(actual.split())
    if len(set(words)) >= 4 and len(set(words) & actual_words) / len(set(words)) >= 0.85:
        return True
    return len(words) >= 6 and any(
        " ".join(words[start : start + 6]) in actual
        for start in range(len(words) - 5)
    )


def _numeric_tokens(text: str) -> set[str]:
    digits = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    values = re.findall(r"\d[\d,]*(?:\.\d+)?", digits)
    return {value.replace(",", "") for value in values}


def _query_cases(
    records: list[dict[str, str]],
    languages: set[str],
    paraphrases: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    cases = []
    for row in records:
        filename = row.get("original_path", "").replace("\\", "/").split("/")[-1]
        for language, question_key, answer_key in (
            ("en", "question_en", "answer_en"),
            ("ar", "question_ar", "answer_ar"),
        ):
            if language in languages and row.get(question_key):
                case = {
                    "id": row.get("id", ""),
                    "category": row.get("category", ""),
                    "source_file": filename,
                    "language": language,
                    "query": row[question_key],
                    "answer": row.get(answer_key, ""),
                    "gold_chunk": row.get("gold_chunk", ""),
                    "variant": "original",
                }
                cases.append(case)
                paraphrase = (paraphrases or {}).get(str(case["id"]), {}).get(language)
                if paraphrases is not None and paraphrase:
                    cases.append({**case, "query": paraphrase, "variant": "paraphrase"})
    return cases


class _FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int | None = None):
        return self.chunks[:top_k] if top_k else self.chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        default="WhatsApp Unknown 2026-09-24 at 18.57.26/MoF_RAG_Benchmark_GroundTruth.xlsx",
    )
    parser.add_argument("--mode", choices=("retrieval", "rag"), default="retrieval")
    parser.add_argument("--retriever", choices=("hybrid", "reranked"), default="reranked")
    parser.add_argument("--ids", default="", help="Comma-separated question IDs")
    parser.add_argument("--langs", default="ar,en", help="Comma-separated languages")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N supported question/language pairs")
    parser.add_argument("--output", default="data/evaluation/mof_ground_truth")
    parser.add_argument(
        "--include-paraphrases",
        action="store_true",
        help="Also run the curated meaning-preserving variants in data/benchmarks/mof_semantic_paraphrases.json",
    )
    parser.add_argument("--paraphrase-file", default="data/benchmarks/mof_semantic_paraphrases.json")
    parser.add_argument("--pdf-output", default="", help="PDF report path (defaults to --output with .pdf)")
    parser.add_argument("--no-pdf", action="store_true", help="Save JSON only")
    args = parser.parse_args()
    if args.top_k < 1 or args.limit < 0:
        parser.error("--top-k must be positive and --limit non-negative")
    records = read_workbook(args.workbook)
    wanted = {int(item) for item in args.ids.split(",") if item.strip()}
    if wanted:
        records = [row for row in records if int(row["id"]) in wanted]
    languages = {item.strip() for item in args.langs.split(",") if item.strip()}
    if not languages or not languages <= {"ar", "en"}:
        parser.error("--langs must contain ar and/or en")
    paraphrases = None
    if args.include_paraphrases:
        try:
            paraphrases = json.loads(Path(args.paraphrase_file).read_text(encoding="utf-8"))["paraphrases"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(f"Could not read paraphrases from {args.paraphrase_file}: {exc}")
    cases = _query_cases(records, languages, paraphrases)

    hybrid = HybridRetriever()
    indexed_documents = {
        record.get("metadata", {}).get("source_file", "")
        for record in hybrid.bm25_retriever.bm25_index._records
    }
    missing = sorted({case["source_file"] for case in cases if case["source_file"] not in indexed_documents})
    supported = [case for case in cases if case["source_file"] in indexed_documents]
    if args.limit:
        selected_pairs = list(dict.fromkeys((case["id"], case["language"]) for case in supported))[: args.limit]
        selected_pairs = set(selected_pairs)
        supported = [case for case in supported if (case["id"], case["language"]) in selected_pairs]
    if not supported:
        parser.error("No selected questions have a source document in the current index")
    if args.include_paraphrases:
        available_pairs = {(case["id"], case["language"]) for case in supported}
        paraphrased_pairs = {
            (case["id"], case["language"])
            for case in supported
            if case["variant"] == "paraphrase"
        }
        missing_variants = sorted(available_pairs - paraphrased_pairs)
        if missing_variants:
            parser.error(f"Missing paraphrases for indexed workbook cases: {missing_variants}")

    pipeline = None
    if args.retriever == "reranked":
        pipeline = RetrievalPipeline(hybrid_retriever=hybrid, reranker=get_default_reranker())
    llm = get_default_llm_client() if args.mode == "rag" else None

    results = []
    for ordinal, case in enumerate(supported, start=1):
        print(
            f"[{ordinal}/{len(supported)}] id={case['id']} lang={case['language']} variant={case['variant']} source={case['source_file']}",
            flush=True,
        )
        if args.retriever == "hybrid":
            retrieved = hybrid.retrieve(case["query"], top_k=args.top_k)
        else:
            retrieved = pipeline.retrieve(case["query"], top_n=args.top_k)
        matching = [
            index for index, chunk in enumerate(retrieved, start=1)
            if _gold_match(case["gold_chunk"], chunk.text)
        ]
        source_matching = [
            index for index, chunk in enumerate(retrieved, start=1)
            if _gold_match(case["gold_chunk"], chunk.text)
            and chunk.metadata.get("source_file") == case["source_file"]
        ]
        item = {
            key: case[key]
            for key in ("id", "category", "source_file", "language", "variant", "query", "answer", "gold_chunk")
        }
        item.update({
            "hit_rank_any_source": matching[0] if matching else None,
            "hit_rank_expected_source": source_matching[0] if source_matching else None,
            "top_sources": [chunk.metadata.get("source_file", "") for chunk in retrieved],
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
                for rank, chunk in enumerate(retrieved, start=1)
            ],
        })
        if llm is not None:
            llm_pipeline = RAGPipeline(
                GroundedGenerator(_FixedRetriever(retrieved), llm),
                VerificationPipeline(FaithfulnessChecker(llm)),
            )
            response = llm_pipeline.run(case["query"])
            answer_numbers = _numeric_tokens(response.answer)
            gold_numbers = _numeric_tokens(case["answer"])
            item.update({
                "generated_answer": response.answer,
                "verified": response.verified,
                "citations_valid": response.verification.citations.valid,
                "answer_exact_match": bool(_normalize(case["answer"])) and _normalize(case["answer"]) == _normalize(response.answer),
                "answer_token_f1": answer_token_f1(case["answer"], response.answer),
                "gold_numbers": sorted(gold_numbers),
                "matched_gold_numbers": sorted(gold_numbers & answer_numbers),
                "numeric_recall": len(gold_numbers & answer_numbers) / len(gold_numbers) if gold_numbers else None,
                "gold_numbers_all_present": bool(gold_numbers) and gold_numbers <= answer_numbers,
            })
        results.append(item)

    count = len(results)
    def rate(predicate):
        return round(sum(bool(predicate(item)) for item in results) / count, 4) if count else 0.0

    summary = {
        "mode": args.mode,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "workbook_questions_selected": len(records),
        "selected_question_language_pairs": len({(case["id"], case["language"]) for case in cases}),
        "supported_question_language_pairs_run": len({(case["id"], case["language"]) for case in supported}),
        "test_cases_run": count,
        "paraphrase_test_cases_run": sum(item["variant"] == "paraphrase" for item in results),
        "indexed_source_documents": len(indexed_documents),
        "selected_question_pairs_with_missing_source": len({(case["id"], case["language"]) for case in cases if case["source_file"] not in indexed_documents}),
        "missing_source_files": missing,
        "gold_chunk_hit_at_1_any_source": rate(lambda item: item["hit_rank_any_source"] == 1),
        f"gold_chunk_hit_at_{args.top_k}_any_source": rate(lambda item: item["hit_rank_any_source"] is not None),
        "gold_chunk_hit_at_1_expected_source": rate(lambda item: item["hit_rank_expected_source"] == 1),
        f"gold_chunk_hit_at_{args.top_k}_expected_source": rate(lambda item: item["hit_rank_expected_source"] is not None),
    }
    summary["variant_metrics"] = {}
    for variant in ("original", "paraphrase"):
        variant_results = [item for item in results if item["variant"] == variant]
        variant_count = len(variant_results)
        summary["variant_metrics"][variant] = {
            "count": variant_count,
            "gold_chunk_hit_at_1_any_source": round(
                sum(item["hit_rank_any_source"] == 1 for item in variant_results) / variant_count, 4
            ) if variant_count else 0.0,
            f"gold_chunk_hit_at_{args.top_k}_any_source": round(
                sum(item["hit_rank_any_source"] is not None for item in variant_results) / variant_count, 4
            ) if variant_count else 0.0,
            "gold_chunk_hit_at_1_expected_source": round(
                sum(item["hit_rank_expected_source"] == 1 for item in variant_results) / variant_count, 4
            ) if variant_count else 0.0,
            f"gold_chunk_hit_at_{args.top_k}_expected_source": round(
                sum(item["hit_rank_expected_source"] is not None for item in variant_results) / variant_count, 4
            ) if variant_count else 0.0,
        }
    if llm is not None:
        summary.update({
            "verified_rate": rate(lambda item: item["verified"]),
            "citation_valid_rate": rate(lambda item: item["citations_valid"]),
            "strict_answer_match_rate": rate(lambda item: item["answer_exact_match"]),
            "all_gold_numbers_present_rate": rate(lambda item: item["gold_numbers_all_present"]),
            "mean_numeric_recall": round(sum(item["numeric_recall"] or 0 for item in results) / count, 4),
            "mean_answer_token_f1": round(sum(item["answer_token_f1"] for item in results) / count, 4),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output.with_suffix('.json')}")
    if not args.no_pdf:
        from rag_system.evaluation.benchmark_pdf import render_benchmark_pdf

        pdf_path = Path(args.pdf_output) if args.pdf_output else output.with_suffix(".pdf")
        render_benchmark_pdf(output.with_suffix(".json"), pdf_path)
        print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
