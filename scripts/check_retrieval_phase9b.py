"""
Phase 9-B: did BM25 actually retrieve the gold passage?

For every ground-truth question, in Arabic and in English, runs the same
BM25Retriever the LLM benchmark used and checks whether gold_chunk_text shows
up in the retrieved chunks (rank of the first hit). The LLM only saw the top 5,
so hit@5 is what matters; hit@10 shows how close the misses were.

A chunk counts as a hit if the normalized gold span is a substring of it, or if
at least 85% of the gold span's words appear in that single chunk (chunking and
table extraction change spacing/punctuation, so exact matching alone is too
strict).

The interesting output is the list of questions that hit in English but miss in
Arabic: those are losses caused by the Arabic query path (translation + BM25),
not by the LLM.

Run from project root:
    python scripts/check_retrieval_phase9b.py
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2
from rag_system.retrieval.bm25_retriever import BM25Retriever

TOP_K = 10
SEEN_BY_LLM = 5
WORD_COVERAGE = 0.85
OUT_PATH = Path("data/benchmarks/phase9b_retrieval_check.md")


def _norm(text: str) -> str:
    text = (text or "").replace("’", "'").replace("‘", "'").lower()
    return re.sub(r"[^\w']+", " ", text).strip()


def _chunk_text(chunk) -> str:
    """The retrieval result type was not shown, so probe the usual shapes."""
    for path in (("text",), ("chunk", "text"), ("content",), ("chunk", "content")):
        obj = chunk
        try:
            for attr in path:
                obj = obj[attr] if isinstance(obj, dict) else getattr(obj, attr)
        except (AttributeError, KeyError):
            continue
        if isinstance(obj, str):
            return obj
    raise SystemExit(
        f"Could not find the text of a retrieved chunk. Type={type(chunk).__name__}, "
        f"attributes={[a for a in dir(chunk) if not a.startswith('_')]}. "
        "Send this line so the script can be adjusted."
    )


def _is_hit(gold: str, text: str) -> bool:
    g, t = _norm(gold), _norm(text)
    if not g:
        return False
    if g in t:
        return True
    gold_words = g.split()
    if len(gold_words) < 3:
        return False
    text_words = set(t.split())
    return sum(w in text_words for w in gold_words) / len(gold_words) >= WORD_COVERAGE


def _first_hit_rank(gold: str, chunks) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if _is_hit(gold, _chunk_text(chunk)):
            return rank
    return None


def main() -> None:
    retriever = BM25Retriever()
    rows = []
    for gt in GROUND_TRUTH_V2:
        row = {"id": gt.id, "file": gt.source_file, "gold": gt.gold_chunk_text}
        for lang, query in (("ar", gt.question_ar), ("en", gt.question_en)):
            chunks = retriever.retrieve(query, top_k=TOP_K)
            row[lang] = _first_hit_rank(gt.gold_chunk_text, chunks)
            row[f"{lang}_top1"] = _chunk_text(chunks[0])[:160].replace("\n", " ") if chunks else ""
        rows.append(row)

    def show(rank):
        return "-" if rank is None else str(rank)

    lines = ["# BM25 retrieval check (rank of first chunk containing the gold span)", ""]
    lines.append("| id | source file | rank AR | rank EN |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['id']} | {r['file']} | {show(r['ar'])} | {show(r['en'])} |")

    summary = []
    for lang in ("ar", "en"):
        at5 = sum(1 for r in rows if r[lang] is not None and r[lang] <= SEEN_BY_LLM)
        at10 = sum(1 for r in rows if r[lang] is not None)
        summary.append(f"[{lang}] hit@{SEEN_BY_LLM}={at5}/{len(rows)}  hit@{TOP_K}={at10}/{len(rows)}")

    ar_only_loss = [
        r for r in rows
        if r["en"] is not None and r["en"] <= SEEN_BY_LLM and (r["ar"] is None or r["ar"] > SEEN_BY_LLM)
    ]
    both_miss = [
        r for r in rows
        if (r["en"] is None or r["en"] > SEEN_BY_LLM) and (r["ar"] is None or r["ar"] > SEEN_BY_LLM)
    ]

    lines += ["", "## Summary", ""] + [f"- {s}" for s in summary]
    lines += ["", f"## Hit in EN@{SEEN_BY_LLM} but missed in AR@{SEEN_BY_LLM} (Arabic query path loss): {len(ar_only_loss)}", ""]
    for r in ar_only_loss:
        lines.append(f"- id={r['id']} ({r['file']}) AR rank={show(r['ar'])} | gold: {r['gold']} | AR top-1: {r['ar_top1']}")
    lines += ["", f"## Missed in both languages@{SEEN_BY_LLM}: {len(both_miss)}", ""]
    for r in both_miss:
        lines.append(f"- id={r['id']} ({r['file']}) AR={show(r['ar'])} EN={show(r['en'])} | gold: {r['gold']} | EN top-1: {r['en_top1']}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"AR-only losses: {[r['id'] for r in ar_only_loss]}")
    print(f"Missed in both: {[r['id'] for r in both_miss]}")
    print(f"Details saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
