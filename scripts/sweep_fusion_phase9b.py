"""
Phase 9-B: sweep fusion settings offline.

Retrieves candidate lists ONCE per question (depth 100), then evaluates many
weighted-RRF configurations on the cached lists. No LLM calls.

Candidate lists:
  AR question: bm25(AR), vec(AR), vec(translated AR->EN)
  EN question: bm25(EN), vec(EN)

Stricter gold matching than check_retrieval_phase9b.py: a gold span shorter than
6 words must appear as an exact (normalized) substring. Longer spans keep the 85%
word-coverage rule. This fixes id=1, whose 4-word gold ("WTO member since 1995")
was fuzzily matched to the wrong (COMESA) chunk.

Run from project root:
    python scripts/sweep_fusion_phase9b.py 2>&1 | grep -vE "INFO|Warning|Loading weights"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_retrieval_phase9b import _norm  # noqa: E402
from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2  # noqa: E402
from rag_system.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from rag_system.retrieval.rrf import weighted_reciprocal_rank_fusion  # noqa: E402
from rag_system.retrieval.vector_retriever import VectorRetriever  # noqa: E402

DEPTH = 100
COVERAGE = 0.85
MIN_WORDS_FUZZY = 6
OUT_PATH = Path("data/benchmarks/phase9b_fusion_sweep.md")


def is_hit(gold: str, text: str) -> bool:
    g, t = _norm(gold), _norm(text)
    if not g:
        return False
    if g in t:
        return True
    words = g.split()
    if len(words) < MIN_WORDS_FUZZY:
        return False
    tw = set(t.split())
    return sum(w in tw for w in words) / len(words) >= COVERAGE


def first_rank(gold: str, chunks) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if is_hit(gold, c.text):
            return i
    return None


def fuse(lists: dict, weights: dict, k: int):
    return weighted_reciprocal_rank_fusion(lists, weights, k=k)


def main() -> None:
    bm25 = BM25Retriever()
    vec = VectorRetriever()
    translator = bm25.query_translator

    cache = []
    for gt in GROUND_TRUTH_V2:
        translated = gt.question_ar
        if translator is not None:
            try:
                translated = translator.translate_to_english(gt.question_ar)
            except Exception as e:  # noqa: BLE001
                print(f"translation failed id={gt.id}: {e}")
        cache.append(
            {
                "id": gt.id,
                "gold": gt.gold_chunk_text,
                "ar": {
                    "bm25": bm25.retrieve(gt.question_ar, top_k=DEPTH),
                    "vec": vec.retrieve(gt.question_ar, top_k=DEPTH),
                    "vec_tr": vec.retrieve(translated, top_k=DEPTH),
                },
                "en": {
                    "bm25": bm25.retrieve(gt.question_en, top_k=DEPTH),
                    "vec": vec.retrieve(gt.question_en, top_k=DEPTH),
                },
            }
        )

    def score(lang: str, builder):
        ranks = {}
        for item in cache:
            ranks[item["id"]] = first_rank(item["gold"], builder(item[lang]))
        h5 = sum(1 for r in ranks.values() if r is not None and r <= 5)
        h8 = sum(1 for r in ranks.values() if r is not None and r <= 8)
        misses = [i for i, r in ranks.items() if r is None or r > 5]
        return h5, h8, misses

    results = []

    # single-source baselines (strict matching)
    for name in ("bm25", "vec"):
        ar = score("ar", lambda d, n=name: d[n])
        en = score("en", lambda d, n=name: d[n])
        results.append((f"only {name}", ar, en))

    for use_tr in (False, True):
        for k in (60, 20):
            for w_bm in (1.0, 1.5, 2.0, 3.0, 5.0):
                weights = {"bm25": w_bm, "vec": 1.0, "vec_tr": 1.0}

                def build(d, weights=weights, k=k, use_tr=use_tr):
                    lists = {"bm25": d["bm25"], "vec": d["vec"]}
                    if use_tr and "vec_tr" in d:
                        lists["vec_tr"] = d["vec_tr"]
                    return fuse(lists, weights, k)

                ar = score("ar", build)
                en = score("en", build)
                label = f"bm25_w={w_bm:<3} k={k:<2} {'+vec_tr' if use_tr else '       '}"
                results.append((label, ar, en))

    results.sort(key=lambda r: -(r[1][0] + r[2][0]))

    header = f"{'config':<32} {'AR@5':>5} {'AR@8':>5} {'EN@5':>5} {'EN@8':>5}  AR misses@5"
    lines = [header, "-" * len(header)]
    for label, ar, en in results:
        lines.append(f"{label:<32} {ar[0]:>5} {ar[1]:>5} {en[0]:>5} {en[1]:>5}  {ar[2]}")
    text = "\n".join(lines)

    OUT_PATH.write_text("# Fusion sweep (strict gold matching)\n\n```\n" + text + "\n```\n", encoding="utf-8")
    print(text)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
