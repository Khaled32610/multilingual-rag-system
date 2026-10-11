import json

import pytest

from rag_system.evaluation.pdf_report import PDFReportError, load_report_data


def test_loads_evaluation_and_experiment_inputs(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    experiment = tmp_path / "experiment.json"
    evaluation.write_text(json.dumps({"summary": {"verified_rate": 0.9}, "items": []}))
    experiment.write_text(json.dumps({"experiment_id": "ab", "variants": []}))

    payload = load_report_data(evaluation, experiment)

    assert payload["evaluation"]["summary"]["verified_rate"] == 0.9
    assert payload["experiment"]["experiment_id"] == "ab"


def test_rejects_invalid_report_shape(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"items": []}))
    with pytest.raises(PDFReportError, match="summary object"):
        load_report_data(path)


def test_reports_missing_input(tmp_path):
    with pytest.raises(PDFReportError, match="Could not read"):
        load_report_data(tmp_path / "missing.json")
