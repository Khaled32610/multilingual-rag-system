"""Repeatable, side-by-side experiment runs over a fixed evaluation set."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_system.evaluation.full_evaluation import EvaluationCase, EvaluationReport, RAGEvaluator


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    parameters: dict[str, Any]
    metrics: dict[str, int | float]
    report_path: str
    run_at: str


class ExperimentRunner:
    """Evaluate named pipeline variants against identical cases and persist artifacts."""

    def __init__(self, output_dir: str | Path = "data/experiments") -> None:
        self.output_dir = Path(output_dir)

    def run(
        self,
        experiment_id: str,
        cases: list[EvaluationCase],
        variants: dict[str, tuple[object, dict[str, Any]]],
    ) -> list[ExperimentResult]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", experiment_id):
            raise ValueError("experiment_id must be 1-80 safe filename characters")
        if not variants:
            raise ValueError("at least one experiment variant is required")
        run_dir = self.output_dir / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results: list[ExperimentResult] = []
        for name, (pipeline, parameters) in variants.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
                raise ValueError(f"unsafe variant name: {name!r}")
            report: EvaluationReport = RAGEvaluator(pipeline).evaluate(cases)  # type: ignore[arg-type]
            report.metadata.update({"experiment_id": experiment_id, "variant": name})
            report.metadata["parameters"] = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
            report_path = run_dir / f"{name}.json"
            report.save_json(report_path)
            report.save_markdown(run_dir / f"{name}.md")
            results.append(ExperimentResult(
                name=name,
                parameters=parameters,
                metrics=report.summary(),
                report_path=str(report_path),
                run_at=datetime.now(UTC).isoformat(),
            ))
        self._write_index(run_dir / "experiment.json", experiment_id, results)
        return results

    @staticmethod
    def compare(results: list[ExperimentResult], baseline: str) -> list[dict[str, Any]]:
        by_name = {result.name: result for result in results}
        if baseline not in by_name:
            raise ValueError(f"baseline variant not found: {baseline}")
        base = by_name[baseline]
        comparisons = []
        for result in results:
            delta = {
                key: round(float(value) - float(base.metrics.get(key, 0)), 4)
                for key, value in result.metrics.items()
                if isinstance(value, (int, float)) and key in base.metrics
            }
            comparisons.append({
                "variant": result.name,
                "baseline": baseline,
                "delta": delta,
                "parameters": result.parameters,
                "report_path": result.report_path,
            })
        return comparisons

    @staticmethod
    def _write_index(path: Path, experiment_id: str, results: list[ExperimentResult]) -> None:
        payload = {"experiment_id": experiment_id, "variants": [asdict(item) for item in results]}
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)


__all__ = ["ExperimentResult", "ExperimentRunner"]
