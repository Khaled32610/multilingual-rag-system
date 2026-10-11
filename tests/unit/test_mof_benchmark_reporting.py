import json
from pathlib import Path

from scripts.evaluate_mof_workbook import _query_cases
from rag_system.evaluation.benchmark_pdf import _html_report


def test_query_cases_adds_original_and_fixed_paraphrase():
    records = [{
        "id": "1",
        "category": "history",
        "original_path": "docs/source.pdf",
        "question_en": "When did Egypt join the WTO?",
        "question_ar": "متى انضمت مصر؟",
        "answer_en": "1995",
        "answer_ar": "1995",
        "gold_chunk": "Egypt joined in 1995.",
    }]
    paraphrases = {"1": {"en": "In what year did Egypt become a WTO member?", "ar": "في أي عام انضمت مصر؟"}}

    cases = _query_cases(records, {"en", "ar"}, paraphrases)

    assert len(cases) == 4
    assert [(case["language"], case["variant"]) for case in cases] == [
        ("en", "original"), ("en", "paraphrase"),
        ("ar", "original"), ("ar", "paraphrase"),
    ]
    assert cases[1]["query"] == paraphrases["1"]["en"]


def test_pdf_html_contains_every_case_query_and_retrieved_excerpt():
    payload = {
        "summary": {"top_k": 1, "test_cases_run": 1, "gold_chunk_hit_at_1_any_source": 1.0},
        "results": [{
            "id": "12",
            "language": "ar",
            "variant": "paraphrase",
            "query": "ما قراءة المؤشر؟",
            "source_file": "source.pdf",
            "category": "economy",
            "answer": "50.7",
            "gold_chunk": "The PMI reached 50.7 points.",
            "hit_rank_any_source": 1,
            "hit_rank_expected_source": 1,
            "retrieved_chunks": [{
                "rank": 1,
                "source_file": "source.pdf",
                "score": 0.9,
                "gold_match": True,
                "expected_source_match": True,
                "text": "The PMI reached 50.7 points.",
            }],
        }],
    }

    report = _html_report(payload)

    assert "ما قراءة المؤشر؟" in report
    assert "The PMI reached 50.7 points." in report
    assert "Gold at #1" in report


def test_original_only_pdf_does_not_advertise_paraphrase_metrics():
    report = _html_report({
        "summary": {
            "mode": "retrieval",
            "retriever": "reranked",
            "top_k": 5,
            "test_cases_run": 1,
            "paraphrase_test_cases_run": 0,
            "variant_metrics": {"original": {"count": 1, "gold_chunk_hit_at_1_any_source": 1.0}},
        },
        "results": [{"id": "1", "language": "en", "variant": "original", "query": "When?"}],
    })

    assert "Original Arabic and English workbook queries" in report
    assert "Original-query scores" not in report
    assert "Paraphrase cases" not in report
    assert "Original and meaning-preserving paraphrase queries" not in report
    assert "<th>Paraphrase</th>" not in report
