"""Create a printable PDF report from Phase 15 evaluation and Phase 16 experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PDFReportError(RuntimeError):
    """Raised when report input is invalid or the optional PDF extra is unavailable."""


def load_report_data(evaluation_path: str | Path, experiment_path: str | Path | None = None) -> dict[str, Any]:
    try:
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
        experiment = (
            json.loads(Path(experiment_path).read_text(encoding="utf-8"))
            if experiment_path else None
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PDFReportError(f"Could not read report input: {exc}") from exc
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("summary"), dict):
        raise PDFReportError("Evaluation JSON must contain a summary object")
    if experiment is not None and not isinstance(experiment.get("variants"), list):
        raise PDFReportError("Experiment JSON must contain a variants list")
    return {"evaluation": evaluation, "experiment": experiment}


def build_pdf_report(
    evaluation_path: str | Path,
    output_path: str | Path,
    experiment_path: str | Path | None = None,
    title: str = "RAG Benchmark Report",
) -> Path:
    """Render a concise English-labeled PDF; install `rag_system[reports]` first."""
    payload = load_report_data(evaluation_path, experiment_path)
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise PDFReportError(
            "PDF export requires the optional dependency. Install with: pip install -e '.[reports]'"
        ) from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [Paragraph(title, styles["ReportTitle"]), Paragraph("Evaluation summary", styles["Heading2"])]
    evaluation = payload["evaluation"]
    summary = evaluation["summary"]
    rows = [["Metric", "Value"]] + [[str(key), str(value)] for key, value in summary.items()]
    table = Table(rows, colWidths=[75 * mm, 55 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23415c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa8b5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 8 * mm), Paragraph("Per-case results", styles["Heading2"])])
    case_rows = [["Case", "Lang", "Verified", "Evidence", "Exact", "Token F1", "Latency ms"]]
    for item in evaluation.get("items", []):
        case_rows.append([
            str(item.get("case_id", "")), str(item.get("language", "")),
            str(bool(item.get("verified"))), str(bool(item.get("evidence_hit"))),
            str(bool(item.get("answer_exact_match"))),
            f"{float(item.get('answer_token_f1', 0)):.3f}",
            f"{float(item.get('latency_ms', 0)):.1f}",
        ])
    cases = Table(case_rows, repeatRows=1, colWidths=[23*mm, 13*mm, 17*mm, 17*mm, 14*mm, 20*mm, 22*mm])
    cases.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23415c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aab5be")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f8")]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cases)

    experiment = payload["experiment"]
    if experiment:
        story.extend([PageBreak(), Paragraph(f"Experiment: {experiment.get('experiment_id', '')}", styles["Heading1"])])
        experiment_rows = [["Variant", "Parameters", "Metrics"]]
        for variant in experiment["variants"]:
            experiment_rows.append([
                Paragraph(str(variant.get("name", "")), styles["Small"]),
                Paragraph(json.dumps(variant.get("parameters", {}), ensure_ascii=False), styles["Small"]),
                Paragraph(json.dumps(variant.get("metrics", {}), ensure_ascii=False), styles["Small"]),
            ])
        exp_table = Table(experiment_rows, colWidths=[35*mm, 60*mm, 75*mm], repeatRows=1)
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23415c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa8b5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(exp_table)

    doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=16*mm, bottomMargin=16*mm)
    doc.build(story)
    return target


__all__ = ["PDFReportError", "build_pdf_report", "load_report_data"]
