"""
Lightweight language detection (Phase 5 / Step 2 support).

A full language-detection library is overkill for what this project
actually needs — telling Arabic queries apart from English ones on
short text (a search query, not a whole document) — so this uses a
simple Unicode-range heuristic instead: whichever script (Arabic block
vs Latin letters) has more characters in the string wins.
"""

from __future__ import annotations

import re

_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """
    Return "ar" or "en" for `text`, based on which script most of its
    letters belong to.

    Defaults to "en" when there are no letters of either script (pure
    numbers/punctuation, or an empty string) since English is this
    project's majority document language.
    """
    arabic_count = len(_ARABIC_LETTER_RE.findall(text))
    latin_count = len(_LATIN_LETTER_RE.findall(text))

    if arabic_count == 0 and latin_count == 0:
        return "en"
    return "ar" if arabic_count >= latin_count else "en"
