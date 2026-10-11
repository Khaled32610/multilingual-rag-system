"""
Phase 9-B: re-score the saved generations with a punctuation-safe scorer.

No LLM calls. Reads data/benchmarks/phase9b_results.json plus GROUND_TRUTH_V2
and compares the original `heuristic_correct` with a stricter scorer that:
  * strips citation markers like "[مقتطف 3]" before looking at numbers (their
    digits used to match gold numbers such as "1" or "4"/"5" by accident),
  * ignores trailing punctuation on numbers ("1952." now equals "1952"),
    thousands separators and Arabic-Indic digits,
  * normalizes curly quotes/apostrophes and Arabic diacritics,
  * matches non-numeric gold answers by content-word overlap against BOTH the
    Arabic and English gold (models often answer names in Latin script),
  * flags refusals ("المعلومة غير متوفرة", "does not state", ...) so a refusal
    that merely mentions a number is not counted as correct.

Two verdicts are reported:
  primary = the FIRST gold number is present (or overlap >= threshold)
  strict  = ALL gold numbers are present
Neither is a substitute for manual review; the audit file lists every question
whose verdict changed so it can be checked by eye.

Run from project root:
    python scripts/rescore_phase9b.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from rag_system.evaluation.ground_truth_v2 import GROUND_TRUTH_V2

RESULTS_PATH = Path("data/benchmarks/phase9b_results.json")
AUDIT_PATH = Path("data/benchmarks/phase9b_rescore_audit.md")

OVERLAP_THRESHOLD = 0.6
REFUSAL_WINDOW = 150  # refusal phrase must appear in the first N characters
CLIP = 200

_TRANS = {ord(a): b for a, b in zip("٠١٢٣٤٥٦٧٨٩", "0123456789")}
_TRANS.update({ord("٫"): ".", ord("٬"): ",", ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"'})

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
_CITATION = re.compile(r"\[[^\]]*\d+[^\]]*\]")
_ENUM = re.compile(r"\(\d\)")

_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are",
    "into", "its", "per", "about", "which", "has", "have", "had", "been",
    "من", "في", "على", "إلى", "عن", "التي", "الذي", "هذا", "هذه", "كان", "كانت",
}

REFUSAL_PHRASES = [
    "غير متوفر", "غير متاح", "لم يتم تقديم", "لم يتم توفير", "غير مذكور", "لا يوجد",
    "لا توجد", "لم يرد", "not available", "not provided", "does not state",
    "does not provide", "does not mention", "not mentioned", "no information",
    "cannot be found", "not specified", "not stated",
]


def _norm(text: str) -> str:
    return _DIACRITICS.sub("", (text or "").translate(_TRANS)).lower()


def _numbers(text: str) -> list[str]:
    """Ordered, de-duplicated numeric tokens (citations and '(1)' markers removed)."""
    t = _ENUM.sub(" ", _CITATION.sub(" ", _norm(text)))
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    seen: list[str] = []
    for n in re.findall(r"\d+(?:\.\d+)?", t):
        if n not in seen:
            seen.append(n)
    return seen


def _words(text: str) -> set[str]:
    t = _CITATION.sub(" ", _norm(text))
    return {w for w in re.findall(r"[\w']+", t) if len(w) >= 3 and w not in _STOP and not w.isdigit()}


def _ratio(gold: set[str], gen: set[str]) -> float:
    return len(gold & gen) / len(gold) if gold else 0.0


def score(gt, generated: str) -> dict:
    gen = generated or ""
    refused = any(p in _norm(gen)[:REFUSAL_WINDOW] for p in REFUSAL_PHRASES)

    gold_nums = _numbers(gt.answer_en) or _numbers(gt.answer_ar)
    if gold_nums:
        gen_nums = set(_numbers(gen))
        primary = gold_nums[0] in gen_nums
        strict = all(n in gen_nums for n in gold_nums)
    else:
        gen_words = _words(gen)
        best = max(_ratio(_words(gt.answer_en), gen_words), _ratio(_words(gt.answer_ar), gen_words))
        primary = strict = best >= OVERLAP_THRESHOLD

    return {
        "refused": refused,
        "primary": primary and not refused,
        "strict": strict and not refused,
    }


def _clip(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= CLIP else text[:CLIP] + "..."


def main() -> None:
    records = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    gt_by_id = {g.id: g for g in GROUND_TRUTH_V2}

    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flips: list[str] = []

    for r in records:
        gt = gt_by_id.get(r["query_id"])
        if gt is None:
            continue
        s = score(gt, r.get("generated", ""))
        old = bool(r["heuristic_correct"])
        for key in ((r["model"], r["language"]), (r["model"], "all")):
            st = stats[key]
            st["n"] += 1
            st["old"] += old
            st["primary"] += s["primary"]
            st["strict"] += s["strict"]
            st["refused"] += s["refused"]
            st["wrong"] += (not s["primary"]) and (not s["refused"])
        if old != s["primary"]:
            arrow = f"{'OK' if old else 'X'} -> {'OK' if s['primary'] else 'X'}"
            flips.append(
                f"- [{r['model']}][{r['language']}] id={r['query_id']} {arrow}"
                f"{' (refusal)' if s['refused'] else ''} | gold: {gt.answer_en} | got: {_clip(r.get('generated', ''))}"
            )

    header = f"{'Model':<55} {'Lang':<4} {'N':>3} {'old':>6} {'primary':>8} {'strict':>7} {'refuse':>7} {'wrong':>6}"
    print(header)
    print("-" * len(header))
    models = sorted(
        {m for m, _ in stats},
        key=lambda m: -stats[(m, "all")]["primary"] / stats[(m, "all")]["n"],
    )
    for model in models:
        for lang in ("ar", "en", "all"):
            st = stats.get((model, lang))
            if not st:
                continue
            n = st["n"]
            print(
                f"{model:<55} {lang:<4} {n:>3} {st['old'] / n:>6.3f} {st['primary'] / n:>8.3f} "
                f"{st['strict'] / n:>7.3f} {st['refused'] / n:>7.3f} {st['wrong'] / n:>6.3f}"
            )

    AUDIT_PATH.write_text(
        "# Verdict changes: original heuristic vs re-score (primary)\n\n" + "\n".join(flips) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(flips)} verdict changes saved to {AUDIT_PATH}")


if __name__ == "__main__":
    main()