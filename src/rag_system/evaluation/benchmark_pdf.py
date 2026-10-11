"""Render retrieval benchmark JSON as a searchable, bilingual PDF report."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _excerpt(value: str, limit: int = 300) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _case_html(item: dict[str, Any], full_text: bool = False) -> str:
    lang = item.get("language", "en")
    direction = "rtl" if lang == "ar" else "ltr"
    rank = item.get("hit_rank_expected_source")
    status = f"Gold at #{rank}" if rank is not None else "Gold not in top results"
    status_class = "hit" if rank is not None else "miss"
    variant = "Paraphrase" if item.get("variant") == "paraphrase" else "Original"
    gold = item.get("gold_chunk", "")
    answer = item.get("answer", "")
    chunks = item.get("retrieved_chunks", [])
    rows = []
    for chunk in chunks:
        is_match = chunk.get("gold_match")
        expected = chunk.get("expected_source_match")
        label = "GOLD • expected source" if expected else "GOLD • other source" if is_match else ""
        chip = f'<span class="chip">{_safe(label)}</span>' if label else ""
        rows.append(
            "<li>"
            f'<div class="result-head"><b>#{_safe(chunk.get("rank"))}</b> '
            f'<span>{_safe(chunk.get("source_file"))}</span> '
            f'<small>score {_safe(chunk.get("score"))}</small>{chip}</div>'
            f'<div class="meta">Chunk: {_safe(chunk.get("chunk_id"))} · Page: {_safe(chunk.get("metadata", {}).get("page_number"))}</div>'
            f'<div class="excerpt" dir="auto">{_safe(chunk.get("text", "") if full_text else _excerpt(chunk.get("text", "")))}</div>'
            "</li>"
        )
    if not rows:
        rows = [f"<li>{_safe(source)}</li>" for source in item.get("top_sources", [])]
    candidate_detail = ""
    if "candidate_rank_before_rerank" in item:
        candidate_rank = item.get("candidate_rank_before_rerank")
        candidate_text = (
            f"Gold was candidate #{candidate_rank} of {item.get('candidate_count')} before reranking."
            if candidate_rank is not None
            else f"Gold was absent from the {item.get('candidate_count')} hybrid candidates."
        )
        candidate_detail = (
            f'<div class="field"><b>Pre-rerank candidate check</b>'
            f"<div>{_safe(candidate_text)}</div></div>"
        )
    answer_block = (
        f'<div class="field"><b>Reference answer</b><div dir="auto">{_safe(answer)}</div></div>'
        if answer
        else ""
    )
    generated = item.get("generated_answer", "")
    if generated:
        verified = "verified" if item.get("verified") else "not verified"
        citations = "valid citation" if item.get("citations_valid") else "invalid citation"
        answer_block += (
            f'<div class="field"><b>Generated answer · {verified} · {citations}</b>'
            f'<div dir="auto">{_safe(generated)}</div></div>'
        )
    return (
        f'<section class="case" lang="{_safe(lang)}">'
        f'<div class="case-head"><b>ID {_safe(item.get("id"))} · {_safe(lang.upper())} · {variant}</b>'
        f'<span class="{status_class}">{_safe(status)}</span></div>'
        f'<div class="meta">Expected source: {_safe(item.get("source_file"))} · Category: {_safe(item.get("category"))}</div>'
        f'<div class="field"><b>Query</b><div dir="{direction}" class="query">{_safe(item.get("query"))}</div></div>'
        f"{candidate_detail}"
        f"{answer_block}"
        f'<div class="field"><b>Gold chunk</b><div dir="auto">{_safe(gold if full_text else _excerpt(gold, 700))}</div></div>'
        f'<div class="field"><b>Retrieved top {len(chunks)}</b><ol>{"".join(rows)}</ol></div>'
        "</section>"
    )


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    rows = payload.get("results", [])
    has_paraphrases = any(item.get("variant") == "paraphrase" for item in rows)
    metrics = [
        ("Cases tested", summary.get("test_cases_run", len(rows))),
        (
            "Question/language pairs with source files",
            summary.get("supported_question_language_pairs_run", "—"),
        ),
        ("Gold in top 1 · any source", summary.get("gold_chunk_hit_at_1_any_source", "—")),
        (
            f"Gold in top {summary.get('top_k', 5)} · any source",
            summary.get(f"gold_chunk_hit_at_{summary.get('top_k', 5)}_any_source", "—"),
        ),
        (
            "Gold in top 1 · expected source",
            summary.get("gold_chunk_hit_at_1_expected_source", "—"),
        ),
        (
            f"Gold in top {summary.get('top_k', 5)} · expected source",
            summary.get(f"gold_chunk_hit_at_{summary.get('top_k', 5)}_expected_source", "—"),
        ),
        (
            "Question pairs missing source",
            summary.get("selected_question_pairs_with_missing_source", "—"),
        ),
    ]
    if "gold_candidate_hit_at_15_any_source" in summary:
        metrics.insert(
            4, ("Gold in candidate set", summary.get("gold_candidate_hit_at_15_any_source"))
        )
    if has_paraphrases:
        metrics.insert(2, ("Paraphrase cases", summary.get("paraphrase_test_cases_run", 0)))
    metric_html = "".join(
        f"<div class='metric'><span>{_safe(label)}</span><b>{_safe(format(value * 100, '.2f') + '%' if label.startswith('Gold') and isinstance(value, (int, float)) else value)}</b></div>"
        for label, value in metrics
    )
    missing = summary.get("missing_source_files", [])
    missing_html = ""
    if missing:
        missing_html = (
            "<h2>Source files not present in the index</h2><ul class='missing'>"
            + "".join(f"<li>{_safe(name)}</li>" for name in missing)
            + "</ul>"
        )
    variant_metrics = summary.get("variant_metrics", {})
    hit_k_any = f"gold_chunk_hit_at_{summary.get('top_k', 5)}_any_source"
    hit_k_expected = f"gold_chunk_hit_at_{summary.get('top_k', 5)}_expected_source"
    variant_rows = []
    for label, key in (("Original", "original"), ("Paraphrase", "paraphrase")):
        data = variant_metrics.get(key)
        if data and data.get("count", 0) > 0:
            variant_rows.append(
                "<tr>"
                f"<th>{label} ({_safe(data.get('count'))})</th>"
                f"<td>{_safe(data.get('gold_chunk_hit_at_1_any_source'))}</td>"
                f"<td>{_safe(data.get(hit_k_any))}</td>"
                f"<td>{_safe(data.get('gold_chunk_hit_at_1_expected_source'))}</td>"
                f"<td>{_safe(data.get(hit_k_expected))}</td>"
                "</tr>"
            )
    variant_table = (
        f"<h2>{'Original vs paraphrased query scores' if has_paraphrases else 'Original-query scores'}</h2>"
        "<table class='variant-table'><thead><tr><th>Variant</th><th>Hit@1 any</th>"
        f"<th>Hit@{_safe(summary.get('top_k', 5))} any</th><th>Hit@1 expected</th>"
        f"<th>Hit@{_safe(summary.get('top_k', 5))} expected</th></tr></thead>"
        f"<tbody>{''.join(variant_rows)}</tbody></table>"
        if variant_rows and has_paraphrases
        else ""
    )
    run_info = (
        f"Mode: {_safe(summary.get('mode'))} · Retriever: {_safe(summary.get('retriever'))} · "
        f"Generated cases shown below: {len(rows)}"
    )
    subtitle = (
        "Original and meaning-preserving paraphrase queries"
        if has_paraphrases
        else "Original Arabic and English workbook queries"
    )
    audit = summary.get("audit", {})
    audit_html = (
        (
            "<h2>Measurement details</h2><pre>"
            + _safe(json.dumps(audit, ensure_ascii=False, indent=2))
            + "</pre>"
        )
        if audit
        else ""
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 11mm 12mm 13mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Noto Sans Arabic", "Noto Naskh Arabic", "Arial", sans-serif; color:#17212b; font-size:9pt; line-height:1.35; }}
h1 {{ font-size:22pt; margin:0 0 4pt; color:#173b57; }}
.subtitle {{ color:#536575; font-size:9pt; margin-bottom:14pt; }}
h2 {{ font-size:12pt; margin:15pt 0 6pt; color:#173b57; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin:8pt 0 12pt; }}
.metric {{ border:1px solid #ccd7df; border-radius:5px; padding:7px; background:#f5f8fa; min-height:43px; }}
.metric span {{ display:block; color:#536575; font-size:7.5pt; }} .metric b {{ font-size:11pt; }}
.legend {{ color:#4b5d6d; font-size:8pt; margin:8pt 0 12pt; }}
.case {{ border:1px solid #b9c7d1; border-radius:5px; padding:8px 9px; margin:0 0 8px; break-inside:avoid-page; }}
.case-head {{ display:flex; justify-content:space-between; align-items:center; font-size:9pt; background:#edf3f6; padding:4px 6px; border-radius:3px; }}
.meta {{ color:#586b79; font-size:7.4pt; margin:4px 0; overflow-wrap:anywhere; }}
.hit {{ color:#176b42; font-weight:700; }} .miss {{ color:#a23832; font-weight:700; }}
.field {{ margin:5px 0 0; }} .field>b {{ display:block; color:#485d6c; font-size:7.5pt; text-transform:uppercase; letter-spacing:.2px; margin-bottom:2px; }}
.query {{ font-weight:600; }}
[dir=rtl] {{ text-align:right; unicode-bidi:plaintext; }} [dir=auto] {{ unicode-bidi:plaintext; }}
ol {{ margin:2px 0 0; padding-left:20px; }} li {{ margin:3px 0; }}
.result-head {{ display:flex; gap:5px; align-items:baseline; flex-wrap:wrap; font-size:7.3pt; color:#394e5d; }}
.result-head small {{ color:#627583; }} .excerpt {{ font-size:7.5pt; color:#243542; margin:1px 0 4px; }}
.chip {{ font-size:6.5pt; font-weight:700; color:#176b42; border:1px solid #9bc8ae; border-radius:8px; padding:0 4px; }}
.missing {{ columns:2; font-size:8pt; }}
.cover {{ break-after:page; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font-size:8pt; }}
{'.case { break-inside:auto; break-before:page; } .excerpt { white-space:pre-wrap; }' if payload.get("full_text") else ""}
</style></head><body>
<div class="cover"><h1>MoF RAG · Retrieval Benchmark</h1>
<div class="subtitle">استرجاع المقاطع المرجعية · {_safe(subtitle)}<br>{run_info}</div>
<div class="metrics">{metric_html}</div>
{variant_table}
<div class="legend">Expected source = the reference workbook file; Any source = any indexed file with a matching gold passage. Green marks a gold passage found in the expected file. “Any source” hit metrics may include passages from another indexed PDF. Every tested query below includes its question, reference answer, gold text, and retrieved top results.</div>
<div class="legend" dir="rtl">Expected source: الملف المحدد في المرجع. Any source: أي ملف مفهرس يحتوي نصًا مطابقًا للمقطع المرجعي. هذه مقاييس لاسترجاع الدليل.</div>
{missing_html}{audit_html}</div>
{"".join(_case_html(item, full_text=payload.get("full_text", False)) for item in rows)}
</body></html>"""


def render_benchmark_pdf(json_path: str | Path, pdf_path: str | Path) -> Path:
    """Render a benchmark JSON file into a searchable PDF using headless Chrome."""
    source = Path(json_path)
    destination = Path(pdf_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    browser = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if browser is None:
        raise RuntimeError("A Chromium-based browser is required to render benchmark PDF")
    with tempfile.TemporaryDirectory(prefix="mof-benchmark-") as temp_dir:
        html_path = Path(temp_dir) / "report.html"
        html_path.write_text(_html_report(payload), encoding="utf-8")
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--user-data-dir={temp_dir}/browser",
            "--no-pdf-header-footer",
            f"--print-to-pdf={destination.resolve()}",
            html_path.resolve().as_uri(),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=180, check=False
        )
        if completed.returncode != 0 or not destination.exists():
            raise RuntimeError(
                "Chrome PDF rendering failed: "
                + (completed.stderr[-2000:] or completed.stdout[-2000:])
            )
    return destination
