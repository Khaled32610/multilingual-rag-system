"""
Phase 9-B: union candidates + cross-encoder rerank vs. the current fused retriever.

Question answered: BM25 and the vector retriever fail on *different* questions
(BM25 loses 1/4/16/22/91 in Arabic, vector loses 3/8). If we pool the top-N of
both (plus the vector run on the translated query) and let bge-reranker-v2-m3
pick the final top-5, how many gold chunks land in the top 5?

Configs compared, per language:
  fused_w3_k20   : current FusedRetriever (weighted RRF, bm25 weight 3, k=20)
  rr_ar / rr_en  : rerank the union pool with the original question
  <rr>+fused     : RRF (k=20, equal weights) of the fused order and the rerank order
  rr_tr          : (Arabic only) rerank the union pool with the translated query
                   (note: the translation includes the glossary terms appended)
Each reranked config is evaluated at candidate depth 10 and 20 per source.
Reranker scores are computed once on the deepest pool and reused for the
shallower one, so depth 10 costs nothing extra.

Metric: same fuzzy gold matching as check_retrieval_phase9b.py (imported, so the
numbers are comparable). Question 94 (cover page) is unfixable by retrieval and
is reported separately in the "excl. 94" column.

Run from project root (needs the e5 + bge models cached locally). Defaults are
the FAST mode (Arabic only, depth 10, no translated-query rerank):
    python scripts/check_union_rerank_phase9b.py 2>&1 | grep -vE "INFO|Warning|Loading weights"

Options:
    --ids 1,3,4,8,16,22,91,95   only these questions (fastest sanity check)
    --langs ar,en               add English (default: ar)
    --depths 10,20              candidate depth per source (default: 10)
    --tr                        also rerank with the translated query (Arabic)
    --max-len 384               truncate (query, chunk) pairs to N tokens (default 384)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_retrieval_phase9b import _first_hit_rank  # noqa: E402

from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2  # noqa: E402
from rag_system.reranking.default import get_default_reranker  # noqa: E402
from rag_system.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from rag_system.retrieval.language_utils import detect_language  # noqa: E402
from rag_system.retrieval.rrf import weighted_reciprocal_rank_fusion  # noqa: E402
from rag_system.retrieval.vector_retriever import VectorRetriever  # noqa: E402

FUSE_DEPTH = 100          # same as run_phase9b_fusion.py
BM25_WEIGHT, RRF_K = 3.0, 20
CUTOFFS = (3, 5, 8)
EXCLUDE = {94}
OUT_PATH = Path("data/benchmarks/phase9b_union_rerank.md")


def build_pool(lists: dict[str, list], max_depth: int) -> dict[str, list]:
    """chunk_id -> [chunk, best rank across sources], limited to top max_depth per source."""
    pool: dict[str, list] = {}
    for ranked in lists.values():
        for rank, c in enumerate(ranked[:max_depth], start=1):
            entry = pool.setdefault(c.chunk_id, [c, 10**9])
            entry[1] = min(entry[1], rank)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--langs", default="ar")
    ap.add_argument("--depths", default="10")
    ap.add_argument("--tr", action="store_true")
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()
    DEPTHS = tuple(int(x) for x in args.depths.split(","))
    langs = set(args.langs.split(","))
    only_ids = {int(x) for x in args.ids.split(",") if x.strip()}
    questions = [g for g in GROUND_TRUTH_V2 if not only_ids or g.id in only_ids]

    bm25 = BM25Retriever()
    vec = VectorRetriever()
    reranker = get_default_reranker()
    if args.max_len:
        reranker.model.max_length = args.max_len
    translator = bm25.query_translator
    max_depth = max(DEPTHS)

    # results[(lang, config)][qid] = rank of first gold hit (or None)
    results: dict[tuple[str, str], dict[int, int | None]] = {}
    rerank_secs: list[float] = []

    def record(lang: str, config: str, qid: int, rank: int | None) -> None:
        results.setdefault((lang, config), {})[qid] = rank

    for n, gt in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] id={gt.id}", file=sys.stderr, flush=True)
        for lang, q in (("ar", gt.question_ar), ("en", gt.question_en)):
            if lang not in langs:
                continue
            lists = {
                "bm25": bm25.retrieve(q, top_k=FUSE_DEPTH),
                "vec": vec.retrieve(q, top_k=FUSE_DEPTH),
            }
            translated = None
            if lang == "ar" and translator is not None and detect_language(q) != "en":
                try:
                    translated = translator.translate_to_english(q)
                except Exception as e:  # noqa: BLE001 - best effort, same as the runner
                    print(f"translation failed for id={gt.id}: {e}", file=sys.stderr)
                if translated and translated != q:
                    lists["vec_tr"] = vec.retrieve(translated, top_k=FUSE_DEPTH)

            fused = weighted_reciprocal_rank_fusion(
                lists, {"bm25": BM25_WEIGHT, "vec": 1.0, "vec_tr": 1.0}, k=RRF_K
            )
            record(lang, "fused_w3_k20", gt.id, _first_hit_rank(gt.gold_chunk_text, fused))

            pool = build_pool(lists, max_depth)
            chunks = [entry[0] for entry in pool.values()]
            best_rank = {cid: entry[1] for cid, entry in pool.items()}

            rerank_queries = {f"rr_{lang}": q}
            if args.tr and lang == "ar" and translated:
                rerank_queries["rr_tr"] = translated

            for name, rq in rerank_queries.items():
                t0 = time.perf_counter()
                reranked = reranker.rerank(rq, chunks, top_n=len(chunks))
                rerank_secs.append(time.perf_counter() - t0)
                print(f"    {name}: {len(chunks)} pairs in {rerank_secs[-1]:.1f}s",
                      file=sys.stderr, flush=True)
                for depth in DEPTHS:
                    subset = [c for c in reranked if best_rank[c.chunk_id] <= depth]
                    record(lang, f"{name}_d{depth}", gt.id,
                           _first_hit_rank(gt.gold_chunk_text, subset))
                    both = weighted_reciprocal_rank_fusion(
                        {"fused": fused, "rerank": subset},
                        {"fused": 1.0, "rerank": 1.0}, k=RRF_K,
                    )
                    record(lang, f"{name}+fused_d{depth}", gt.id,
                           _first_hit_rank(gt.gold_chunk_text, both))

    def hits(lang: str, config: str, cutoff: int, skip: set[int]) -> int:
        return sum(
            1 for qid, r in results[(lang, config)].items()
            if qid not in skip and r is not None and r <= cutoff
        )

    total = {lang: len(questions) for lang in ("ar", "en")}
    skip_ids = EXCLUDE & {g.id for g in questions}
    lines = ["# Union + rerank vs fused (rank of first chunk containing the gold span)", ""]
    lines.append("| lang | config | " + " | ".join(f"hit@{c}" for c in CUTOFFS) + " | hit@5 excl. 94 | misses@5 |")
    lines.append("|---|---|" + "---|" * (len(CUTOFFS) + 2))
    for (lang, config) in results:
        row = " | ".join(f"{hits(lang, config, c, set())}/{total[lang]}" for c in CUTOFFS)
        ex = hits(lang, config, 5, skip_ids)
        misses = [qid for qid, r in results[(lang, config)].items() if r is None or r > 5]
        lines.append(
            f"| {lang} | {config} | {row} | {ex}/{total[lang] - len(skip_ids)} | {misses} |"
        )

    lines += ["", "## Rank of first gold hit per question ('-' = not found)", ""]
    keys = list(results)
    lines.append("| id | " + " | ".join(f"{l}:{c}" for l, c in keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for g in questions:
        cells = []
        for k in keys:
            r = results[k].get(g.id)
            cells.append("-" if r is None else str(r))
        lines.append(f"| {g.id} | " + " | ".join(cells) + " |")

    avg = sum(rerank_secs) / len(rerank_secs) if rerank_secs else 0.0
    lines += ["", f"Average rerank time per query (pool up to {max_depth} per source): {avg:.2f}s "
                  f"over {len(rerank_secs)} calls."]

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()