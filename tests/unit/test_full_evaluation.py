import json
from types import SimpleNamespace

import pytest

from rag_system.evaluation.full_evaluation import (
    EvaluationCase,
    EvaluationItem,
    EvaluationReport,
    RAGEvaluator,
    answer_token_f1,
    evidence_match,
)
from rag_system.retrieval.models import RetrievedChunk


def _chunk(text):
    return RetrievedChunk(chunk_id="1", text=text, metadata={}, score=1.0, rank=1, source="test")


def test_token_f1_normalizes_arabic_diacritics_and_digits():
    assert answer_token_f1("١٠٪ النمو", "10% النَّمُو") == 1.0


def test_evidence_match_uses_chunk_content():
    assert evidence_match("revenue reached 100 million pounds", (_chunk("Revenue reached 100 million pounds in 2024."),))
    assert not evidence_match("specific gold statement", (_chunk("unrelated material"),))


def test_report_summary_and_exports(tmp_path):
    report = EvaluationReport(items=[EvaluationItem(
        case_id="1-ar", query="سؤال", language="ar", answer="الإجابة", latency_ms=10,
        verified=True, citation_valid=True, evidence_hit=True,
        answer_exact_match=True, answer_token_f1=1.0,
    )])
    assert report.summary()["verified_rate"] == 1.0
    report.save_json(tmp_path / "reports" / "run.json")
    report.save_markdown(tmp_path / "reports" / "run.md")
    assert json.loads((tmp_path / "reports" / "run.json").read_text())["summary"]["cases"] == 1
    assert "answer_token_f1" in (tmp_path / "reports" / "run.md").read_text()


def test_evaluator_keeps_case_errors_in_report():
    class Broken:
        def run(self, query):
            raise RuntimeError("offline")

    report = RAGEvaluator(Broken()).evaluate([
        EvaluationCase("x", "question", "answer", "en")
    ])
    assert report.summary()["errors"] == 1
    assert report.items[0].error == "RuntimeError: offline"


def test_empty_case_list_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        RAGEvaluator(SimpleNamespace()).evaluate([])
