"""
Phase 9-B: compare any number of result files with the punctuation-safe scorer
from rescore_phase9b.py. No LLM calls.

By default question 94 is excluded: its gold answer ("No 370 AUGUST 2025") is not
in the extracted text, so no model can answer it (extraction problem, not a
generation problem). Pass --exclude with no ids to keep everything.

Usage (project root):
    python scripts/rescore_compare.py \
        data/benchmarks/old_ctx/phase9b_bm25_k5.json \
        data/benchmarks/phase9b_bm25_k5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rescore_phase9b import score  # noqa: E402
from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exclude", nargs="*", type=int, default=[94])
    args = ap.parse_args()

    gt_by_id = {g.id: g for g in GROUND_TRUTH_V2}
    excluded = set(args.exclude)

    header = (
        f"{'file':<34} {'model':<22} {'lang':<4} {'N':>3} {'primary':>8} "
        f"{'strict':>7} {'refuse':>7} {'wrong':>6} {'maxTok':>7}"
    )
    print(f"(excluded ids: {sorted(excluded) or 'none'})")
    print(header)
    print("-" * len(header))

    for path in args.files:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = defaultdict(lambda: defaultdict(int))
        for r in records:
            if r["query_id"] in excluded or r["query_id"] not in gt_by_id:
                continue
            s = score(gt_by_id[r["query_id"]], r.get("generated", ""))
            for lang in (r["language"], "all"):
                st = stats[(r["model"], lang)]
                st["n"] += 1
                st["primary"] += s["primary"]
                st["strict"] += s["strict"]
                st["refused"] += s["refused"]
                st["wrong"] += (not s["primary"]) and (not s["refused"])
                tok = r.get("prompt_tokens")
                if tok:
                    st["max_tok"] = max(st["max_tok"], tok)

        label = f"{Path(path).parent.name}/{Path(path).stem}"[-34:]
        for (model, lang), st in sorted(stats.items()):
            n = st["n"]
            max_tok = st["max_tok"] or "-"
            print(
                f"{label:<34} {model[:22]:<22} {lang:<4} {n:>3} {st['primary'] / n:>8.3f} "
                f"{st['strict'] / n:>7.3f} {st['refused'] / n:>7.3f} {st['wrong'] / n:>6.3f} {max_tok!s:>7}"
            )
        print()


if __name__ == "__main__":
    main()
