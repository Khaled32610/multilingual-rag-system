"""Run the end-to-end RAG evaluation set through the configured local stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.evaluation.full_evaluation import EvaluationCase, RAGEvaluator
from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2
from rag_system.generation.grounded_generator import GroundedGenerator
from rag_system.llm.default import get_default_llm_client
from rag_system.pipeline.rag_pipeline import RAGPipeline
from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.verification.faithfulness_checker import FaithfulnessChecker
from rag_system.verification.pipeline import VerificationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="", help="Comma-separated query IDs; default runs all")
    parser.add_argument("--langs", default="ar,en", help="Comma-separated languages: ar,en")
    parser.add_argument("--output", default="data/evaluation/latest", help="Output path prefix")
    args = parser.parse_args()
    ids = {int(value) for value in args.ids.split(",") if value.strip()}
    languages = {value.strip() for value in args.langs.split(",") if value.strip()}
    if not languages or not languages <= {"ar", "en"}:
        parser.error("--langs must contain ar and/or en")
    cases = []
    for item in GROUND_TRUTH_V2:
        if ids and item.id not in ids:
            continue
        for lang in ("ar", "en"):
            if lang in languages:
                cases.append(EvaluationCase(
                    case_id=f"{item.id}-{lang}",
                    query=item.question_ar if lang == "ar" else item.question_en,
                    reference_answer=item.answer_ar if lang == "ar" else item.answer_en,
                    language=lang,
                    gold_evidence=item.gold_chunk_text,
                ))
    if not cases:
        parser.error("No evaluation cases selected")

    llm = get_default_llm_client()
    pipeline = RAGPipeline(
        GroundedGenerator(RetrievalPipeline(), llm),
        VerificationPipeline(FaithfulnessChecker(llm)),
    )
    report = RAGEvaluator(pipeline).evaluate(cases)
    output = Path(args.output)
    report.save_json(output.with_suffix(".json"))
    report.save_markdown(output.with_suffix(".md"))
    print(report.to_markdown())
    print(f"Saved: {output.with_suffix('.json')} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
