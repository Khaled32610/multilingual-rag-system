"""
Arabic/English-aware tokenizer helper for BM25 (Phase 5 / Step 18).

BM25 is exact lexical matching, so tokenization quality controls what
it can and can't match. This project's documents mix Arabic and
English, so:

- English tokens are lowercased by `bm25s.tokenize()` itself, no extra
  handling needed.
- Arabic tokens go through *light normalization* here — NOT stemming.
  Normalization collapses superficial spelling variants that should
  always count as the same word: different hamza-carrying alef forms
  (أ/إ/آ/ٱ -> ا), optional diacritics (tashkeel) and tatweel, and a
  couple of common alternate spellings (alef maksura "ى" vs yeh "ي",
  taa marbuta "ة" vs haa "ه"). A query typed one way should still
  match a document typed another way.

  Stemming goes further and strips prefixes/suffixes (definite article
  "ال", conjunctions "و"/"ف", possessive suffixes, etc.) to collapse
  different *words* down to a shared root — e.g. "الكتاب" and "كتابها"
  would become "the same word". That trades precision for recall, and
  this project's stated priority is accuracy, so this function
  deliberately stops at normalization and does not stem.

`normalize_tokens` is passed as the `stemmer=` argument of
`bm25s.tokenize()` in bm25_index.py. Despite the parameter's name,
bm25s accepts any `list[str] -> list[str]` callable there — applied
once per document/query to that document's already-split, lowercased
word list — which is exactly the shape normalization needs.
"""

from __future__ import annotations

import re

# Arabic diacritics (tashkeel/harakat), Quranic annotation marks, and
# tatweel (kashida, the elongation character) — all purely cosmetic for
# matching purposes.
_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0640]"
)

# Hamza-carrying / variant alef forms -> bare alef (ا).
_ALEF_VARIANTS_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")

_CHAR_NORMALIZATION_TABLE = {
    ord("ى"): "ي",  # alef maksura -> yeh
    ord("ة"): "ه",  # taa marbuta -> haa
    ord("ؤ"): "ء",  # waw with hamza above -> bare hamza
    ord("ئ"): "ء",  # yeh with hamza above -> bare hamza
}


def normalize_arabic_word(word: str) -> str:
    """Apply light Arabic normalization to a single token. Non-Arabic
    characters (e.g. English words, digits) pass through unchanged."""
    word = _DIACRITICS_RE.sub("", word)
    word = _ALEF_VARIANTS_RE.sub("ا", word)
    word = word.translate(_CHAR_NORMALIZATION_TABLE)
    return word


def normalize_tokens(tokens: list[str]) -> list[str]:
    """
    Normalize a list of already-split, lowercased tokens.

    This is the function handed to `bm25s.tokenize(..., stemmer=...)`.
    Empty strings (which can appear after normalization strips a
    token down to nothing, e.g. a token that was pure diacritics) are
    dropped rather than kept as empty terms.
    """
    normalized = (normalize_arabic_word(t) for t in tokens)
    return [t for t in normalized if t]
