from copy import deepcopy

import pytest

from rag_system.evaluation.benchmark_pdf import _case_html, _html_report
from scripts.benchmark_mof35 import IDS, metrics, validate_results


def fixture_data():
    rows = [
        {
            "id": qid,
            "question_ar": "سؤال",
            "question_en": "Question",
            "answer_ar": "جواب",
            "answer_en": "Answer",
            "gold_chunk": "gold text",
            "original_path": "docs/source.pdf",
        }
        for qid in IDS
    ]
    results = [
        {
            "id": r["id"],
            "language": lang,
            "query": r[f"question_{lang}"],
            "answer": r[f"answer_{lang}"],
            "gold_chunk": r["gold_chunk"],
            "source_file": "source.pdf",
            "hit_rank_expected_source": 1,
            "retrieved_chunks": [{"rank": 1, "expected_source_match": True}],
        }
        for r in rows
        for lang in ("ar", "en")
    ]
    return rows, results


def test_missing_or_duplicated_case_cannot_reduce_denominator():
    rows, results = fixture_data()
    validate_results(results, rows)
    with pytest.raises(ValueError, match="exactly 35"):
        validate_results(results[:-1], rows)
    with pytest.raises(ValueError, match="exactly 35"):
        validate_results(results[:-1] + [results[0]], rows)


def test_changed_gold_or_false_hit_is_rejected():
    rows, results = fixture_data()
    changed = deepcopy(results)
    changed[0]["gold_chunk"] = "different gold"
    with pytest.raises(ValueError, match="Reference mismatch"):
        validate_results(changed, rows)
    results[0]["hit_rank_expected_source"] = None
    with pytest.raises(ValueError, match="hit rank"):
        validate_results(results, rows)


def test_threshold_strictly_exceeds_90_and_top1_separate():
    results = [{"hit_rank_expected_source": 5}] * 9 + [{"hit_rank_expected_source": None}]
    assert metrics(results)["hit_at_5_percent"] == 90
    assert metrics(results)["hit_at_1_percent"] == 0
    assert not metrics(results)["exceeds_90_percent"]
    assert metrics(results + [{"hit_rank_expected_source": 1}])["exceeds_90_percent"]


def test_wrong_source_is_not_displayed_as_success():
    html = _case_html(
        {"language": "en", "hit_rank_any_source": 1, "hit_rank_expected_source": None}
    )
    assert "Gold not in top results" in html
    assert 'class="miss"' in html


def test_full_report_keeps_end_of_gold_and_retrieved_text_and_zero_score():
    text = "word " * 300 + "TAIL_MARKER"
    html = _html_report(
        {
            "full_text": True,
            "summary": {},
            "results": [
                {
                    "gold_chunk": text,
                    "retrieved_chunks": [
                        {"rank": 1, "text": text, "score": 0, "chunk_id": "chunk-123"}
                    ],
                }
            ],
        }
    )
    assert html.count("TAIL_MARKER") == 2
    assert "chunk-123" in html
    assert "score 0" in html
