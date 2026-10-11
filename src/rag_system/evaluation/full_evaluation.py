"""End-to-end evaluation for retrieval, grounded answers, and verification."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True)
class EvaluationCase:
    """One human-labeled query and reference answer."""

    case_id: str
    query: str
    reference_answer: str
    language: str
    gold_evidence: str = ""


@dataclass(frozen=True)
class EvaluationItem:
    """Measured output for one evaluation case."""

    case_id: str
    query: str
    language: str
    answer: str
    latency_ms: float
    verified: bool
    citation_valid: bool
    evidence_hit: bool
    answer_exact_match: bool
    answer_token_f1: float
    error: str | None = None


@dataclass
class EvaluationReport:
    """Run metadata, per-case outcomes, and aggregate metrics."""

    items: list[EvaluationItem] = field(default_factory=list)
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def summary(self) -> dict[str, int | float]:
        count = len(self.items)
        successful = [item for item in self.items if item.error is None]
        denom = max(count, 1)
        return {
            "cases": count,
            "success_rate": round(len(successful) / denom, 4),
            "verified_rate": round(sum(item.verified for item in successful) / max(len(successful), 1), 4),
            "citation_valid_rate": round(sum(item.citation_valid for item in successful) / max(len(successful), 1), 4),
            "evidence_hit_rate": round(sum(item.evidence_hit for item in successful) / max(len(successful), 1), 4),
            "answer_exact_match": round(sum(item.answer_exact_match for item in successful) / max(len(successful), 1), 4),
            "answer_token_f1": round(sum(item.answer_token_f1 for item in successful) / max(len(successful), 1), 4),
            "avg_latency_ms": round(sum(item.latency_ms for item in successful) / max(len(successful), 1), 2),
            "errors": count - len(successful),
        }

    def to_dict(self) -> dict:
        return {"metadata": self.metadata, "summary": self.summary(), "items": [asdict(item) for item in self.items]}

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = ["# RAG evaluation report", "", "## Summary", "", "| Metric | Value |", "|---|---:|"]
        lines.extend(f"| {key} | {value} |" for key, value in summary.items())
        lines.extend(["", "## Cases", "", "| ID | Lang | Verified | Evidence | Exact | Token F1 | Latency ms | Error |", "|---|---|---:|---:|---:|---:|---:|---|"])
        for item in self.items:
            error = (item.error or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {item.case_id} | {item.language} | {item.verified} | {item.evidence_hit} | {item.answer_exact_match} | {item.answer_token_f1:.3f} | {item.latency_ms:.1f} | {error} |")
        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")


def _normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text)
    return re.findall(r"[\w]+", text, flags=re.UNICODE)


def answer_token_f1(reference: str, prediction: str) -> float:
    """Multiset token F1; a transparent lexical baseline, not a judge of meaning."""
    expected, actual = _normalize(reference), _normalize(prediction)
    if not expected or not actual:
        return float(expected == actual)
    counts: dict[str, int] = {}
    for token in expected:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in actual:
        if counts.get(token, 0):
            overlap += 1
            counts[token] -= 1
    precision, recall = overlap / len(actual), overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evidence_match(gold_evidence: str, chunks: tuple) -> bool:
    """Match labeled evidence against retrieved chunk text with a word-overlap floor."""
    gold = _normalize(gold_evidence)
    if not gold:
        return False
    for chunk in chunks:
        candidate = _normalize(getattr(chunk, "text", ""))
        if gold == candidate or " ".join(gold) in " ".join(candidate):
            return True
        gold_set = set(gold)
        if len(gold_set) >= 4 and len(gold_set & set(candidate)) / len(gold_set) >= 0.85:
            return True
    return False


class PipelineLike(Protocol):
    def run(self, query: str) -> object: ...


class RAGEvaluator:
    """Run fixed cases through the production pipeline and measure its outputs."""

    def __init__(self, pipeline: PipelineLike) -> None:
        self.pipeline = pipeline

    def evaluate(self, cases: list[EvaluationCase]) -> EvaluationReport:
        if not cases:
            raise ValueError("Evaluation requires at least one case")
        report = EvaluationReport(metadata={"case_count": len(cases)})
        for case in cases:
            started = perf_counter()
            try:
                response = self.pipeline.run(case.query)
                draft = response.draft
                report.items.append(EvaluationItem(
                    case_id=case.case_id,
                    query=case.query,
                    language=case.language,
                    answer=response.answer,
                    latency_ms=(perf_counter() - started) * 1000,
                    verified=response.verified,
                    citation_valid=response.verification.citations.valid,
                    evidence_hit=evidence_match(case.gold_evidence, draft.chunks),
                    answer_exact_match=bool(_normalize(case.reference_answer)) and _normalize(case.reference_answer) == _normalize(response.answer),
                    answer_token_f1=answer_token_f1(case.reference_answer, response.answer),
                ))
            except Exception as exc:  # noqa: BLE001 - preserve failures in the evaluation artifact
                report.items.append(EvaluationItem(
                    case_id=case.case_id, query=case.query, language=case.language,
                    answer="", latency_ms=(perf_counter() - started) * 1000,
                    verified=False, citation_valid=False, evidence_hit=False,
                    answer_exact_match=False, answer_token_f1=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                ))
        return report


__all__ = ["EvaluationCase", "EvaluationItem", "EvaluationReport", "RAGEvaluator", "answer_token_f1", "evidence_match"]
