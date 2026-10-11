"""
Phase 9-B follow-up: where do two models disagree?

Reads data/benchmarks/phase9b_results.json and, for each language, lists the
questions that only MODEL_A got right, only MODEL_B got right, and the ones
both missed. Prints a summary and saves the details to a markdown file.

The JSON layout of phase9b_results.json was never shown to Claude, so field
names are auto-detected from a list of common candidates. If detection fails
the script prints the fields it found and stops, so nothing is guessed.

Run from project root:
    python scripts/compare_phase9b_models.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("data/benchmarks/phase9b_results.json")
OUT_PATH = Path("data/benchmarks/phase9b_model_disagreements.md")

MODEL_A = "gemma3:4b"
MODEL_B = "qwen2.5:3b-instruct"
LANGS = ("ar", "en")
MAX_CHARS = 240

CANDIDATES = {
    "model": ["model", "model_name"],
    "lang": ["lang", "language"],
    "id": ["id", "question_id", "qid", "query_id"],
    "correct": ["heuristic_correct", "correct", "is_correct", "hit", "match", "accuracy", "score"],
    "question": ["question", "query"],
    "answer": ["answer", "response", "generated_answer", "generated", "prediction"],
    "gold": ["expected", "gold", "ground_truth", "reference", "expected_answer"],
}


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, list):
                for r in value:
                    if isinstance(r, dict):
                        r = dict(r)
                        r.setdefault("model", key)
                        records.append(r)
        if records:
            return records
    raise SystemExit(
        f"Unrecognized JSON layout in {path}. Top-level type: {type(data).__name__}. "
        "Send the first ~30 lines of the file so the script can be adjusted."
    )


def find_key(record: dict, field: str) -> str | None:
    for name in CANDIDATES[field]:
        if name in record:
            return name
    return None


def is_correct(value) -> bool:
    if isinstance(value, (int, float, bool)):
        return float(value) >= 0.5
    return bool(value)


def clip(value) -> str:
    text = str(value if value is not None else "").replace("\n", " ").strip()
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "..."


def main() -> None:
    records = load_records(RESULTS_PATH)
    sample = records[0]

    keys = {field: find_key(sample, field) for field in CANDIDATES}
    missing = [f for f in ("model", "id", "correct") if keys[f] is None]
    if missing:
        print(f"Could not detect fields: {missing}")
        print(f"Fields found in the first record: {sorted(sample.keys())}")
        sys.exit(1)
    print(f"Detected fields: {keys}")

    # (lang, id) -> {model: record}
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        model = str(r.get(keys["model"], ""))
        if model not in (MODEL_A, MODEL_B):
            continue
        lang = str(r.get(keys["lang"], "?")) if keys["lang"] else "?"
        grouped[(lang, r[keys["id"]])][model] = r

    lines = [f"# {MODEL_A} vs {MODEL_B}: disagreements", ""]
    for lang in LANGS:
        only_a, only_b, both_wrong, both_right = [], [], [], []
        for (l, qid), by_model in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            if l != lang or len(by_model) < 2:
                continue
            a_ok = is_correct(by_model[MODEL_A][keys["correct"]])
            b_ok = is_correct(by_model[MODEL_B][keys["correct"]])
            bucket = (
                both_right if a_ok and b_ok
                else only_a if a_ok
                else only_b if b_ok
                else both_wrong
            )
            bucket.append((qid, by_model))

        total = len(only_a) + len(only_b) + len(both_wrong) + len(both_right)
        summary = (
            f"[{lang}] compared={total} | both right={len(both_right)} | "
            f"only {MODEL_A}={len(only_a)} | only {MODEL_B}={len(only_b)} | "
            f"both wrong={len(both_wrong)}"
        )
        print(summary)
        lines += [f"## {lang}", "", summary, ""]

        for title, bucket in (
            (f"Only {MODEL_A} correct", only_a),
            (f"Only {MODEL_B} correct", only_b),
            ("Both wrong", both_wrong),
        ):
            lines += [f"### {title} ({len(bucket)})", ""]
            for qid, by_model in bucket:
                first = next(iter(by_model.values()))
                q = first.get(keys["question"]) if keys["question"] else None
                g = first.get(keys["gold"]) if keys["gold"] else None
                lines.append(f"- **id={qid}** Q: {clip(q)}")
                if g is not None:
                    lines.append(f"  - gold: {clip(g)}")
                if keys["answer"]:
                    for m in (MODEL_A, MODEL_B):
                        lines.append(f"  - {m}: {clip(by_model[m].get(keys['answer']))}")
            lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Details saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
