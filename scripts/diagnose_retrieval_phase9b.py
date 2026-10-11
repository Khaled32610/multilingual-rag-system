"""
Phase 9-B diagnosis: which retrieval path loses the Arabic questions?

For every ground-truth question, in Arabic and English, measures the rank of the
first chunk containing the gold span under four methods:

  bm25 : BM25Retriever alone (what check_retrieval_phase9b.py measured)
  vec  : VectorRetriever alone (multilingual-e5, raw query)
  hyb  : HybridRetriever (vector + BM25 via RRF) = the production default
  hyb2 : Arabic only. RRF over [vector(AR query), vector(translated query), bm25(AR query)]

Ranks are searched up to DEPTH. Hit@5 is what the LLM sees.

Run from project root:
    python scripts/diagnose_retrieval_phase9b.py 2>&1 | grep -vE "INFO|Warning|Loading weights"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_retrieval_phase9b import _first_hit_rank, _is_hit  # noqa: E402
from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2  # noqa: E402
from rag_system.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from rag_system.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from rag_system.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402
from rag_system.retrieval.vector_retriever import VectorRetriever  # noqa: E402

DEPTH = 50
SEEN = 5
METHODS = ["bm25", "vec", "hyb", "hyb2"]
OUT_PATH = Path("data/benchmarks/phase9b_retrieval_diagnosis.md")


def show(rank):
    return "-" if rank is None else str(rank)


def main() -> None:
    bm25 = BM25Retriever()
    vec = VectorRetriever()
    hyb = HybridRetriever(vector_retriever=vec, bm25_retriever=bm25)
    translator = bm25.query_translator

    rows = []
    for gt in GROUND_TRUTH_V2:
        gold = gt.gold_chunk_text
        row = {"id": gt.id, "file": gt.source_file, "gold": gold, "gold_chunk": ""}
        for lang, q in (("ar", gt.question_ar), ("en", gt.question_en)):
            lists = {
                "bm25": bm25.retrieve(q, top_k=DEPTH),
                "vec": vec.retrieve(q, top_k=DEPTH),
                "hyb": hyb.retrieve(q, top_k=DEPTH),
            }
            if lang == "ar":
                translated = q
                if translator is not None:
                    try:
                        translated = translator.translate_to_english(q)
                    except Exception as e:  # noqa: BLE001
                        print(f"translation failed for id={gt.id}: {e}")
                lists["hyb2"] = reciprocal_rank_fusion(
                    [lists["vec"], vec.retrieve(translated, top_k=DEPTH), lists["bm25"]]
                )[:DEPTH]
            else:
                lists["hyb2"] = lists["hyb"]

            for name in METHODS:
                row[(lang, name)] = _first_hit_rank(gold, lists[name])

            if lang == "en" and not row["gold_chunk"]:
                for name in ("bm25", "vec"):
                    for c in lists[name]:
                        if _is_hit(gold, c.text):
                            row["gold_chunk"] = c.text
                            break
                    if row["gold_chunk"]:
                        break
        rows.append(row)

    def hits(lang, method, limit):
        return sum(1 for r in rows if r[(lang, method)] is not None and r[(lang, method)] <= limit)

    summary = []
    for lang in ("ar", "en"):
        for m in METHODS:
            if lang == "en" and m == "hyb2":
                continue
            summary.append(
                f"[{lang}] {m:<5} hit@5={hits(lang, m, 5)}/{len(rows)}  hit@10={hits(lang, m, 10)}/{len(rows)}"
                f"  hit@{DEPTH}={hits(lang, m, DEPTH)}/{len(rows)}"
            )

    def is_problem(r):
        return any(r[("ar", m)] is None or r[("ar", m)] > SEEN for m in ("bm25", "hyb", "hyb2"))

    lines = ["# Retrieval diagnosis (rank of first chunk containing the gold span)", ""]
    lines += ["## Summary", ""] + [f"- {s}" for s in summary]
    lines += ["", "## Per question (AR ranks | EN ranks)", ""]
    lines.append("| id | file | AR bm25 | AR vec | AR hyb | AR hyb2 | EN bm25 | EN vec | EN hyb |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['file'][:22]} | "
            + " | ".join(show(r[("ar", m)]) for m in METHODS)
            + " | "
            + " | ".join(show(r[("en", m)]) for m in ("bm25", "vec", "hyb"))
            + " |"
        )
    lines += ["", "## Problem questions: gold span and the chunk that contains it", ""]
    for r in rows:
        if is_problem(r):
            lines.append(f"### id={r['id']} ({r['file']})")
            lines.append(f"- gold: {r['gold']}")
            lines.append(f"- chunk: {r['gold_chunk'][:400].replace(chr(10), ' ') or '(not found in top-50 of EN bm25/vec)'}")
            lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print()
    print("id | AR bm25 vec hyb hyb2 | EN bm25 vec hyb   (only questions with an AR miss@5)")
    for r in rows:
        if is_problem(r):
            print(
                f"{r['id']:>3} | "
                + " ".join(f"{show(r[('ar', m)]):>4}" for m in METHODS)
                + " | "
                + " ".join(f"{show(r[('en', m)]):>4}" for m in ("bm25", "vec", "hyb"))
            )
    print(f"\nDetails saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
