import json
from types import SimpleNamespace

import pytest

from rag_system.evaluation.experiments import ExperimentRunner


class FakePipeline:
    def __init__(self, answer, fail=False):
        self.answer = answer
        self.fail = fail

    def run(self, query):
        if self.fail:
            raise RuntimeError(self.answer)
        return SimpleNamespace(
            answer=self.answer, verified=True,
            draft=SimpleNamespace(chunks=()),
            verification=SimpleNamespace(citations=SimpleNamespace(valid=True)),
        )


def test_runner_persists_one_comparable_report_per_variant(tmp_path):
    from rag_system.evaluation.full_evaluation import EvaluationCase

    cases = [EvaluationCase("a", "q", "ref", "en")]
    runner = ExperimentRunner(tmp_path)
    results = runner.run("retrieval-ab", cases, {
        "baseline": (FakePipeline("reference response"), {"retrieval": "bm25"}),
        "candidate": (FakePipeline("candidate failure", fail=True), {"retrieval": "hybrid"}),
    })
    comparison = runner.compare(results, "baseline")

    assert len(results) == 2
    assert comparison[0]["delta"]["errors"] == 0
    assert comparison[1]["delta"]["errors"] == 1
    index = json.loads((tmp_path / "retrieval-ab" / "experiment.json").read_text())
    assert [variant["name"] for variant in index["variants"]] == ["baseline", "candidate"]
    assert (tmp_path / "retrieval-ab" / "candidate.md").exists()


def test_runner_rejects_unsafe_paths_and_empty_variants(tmp_path):
    runner = ExperimentRunner(tmp_path)
    with pytest.raises(ValueError, match="experiment_id"):
        runner.run("../escape", [], {"x": (FakePipeline("unused"), {})})
    with pytest.raises(ValueError, match="at least one"):
        runner.run("valid", [], {})


def test_compare_requires_named_baseline():
    with pytest.raises(ValueError, match="baseline variant"):
        ExperimentRunner.compare([], "missing")
