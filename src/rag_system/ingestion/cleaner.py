"""
Text cleaning (Phase 1 / Step 4).

Normalizes raw text extracted from PDFs before it moves on to chunking.
Handles common PDF extraction artifacts:
- Excessive whitespace / blank lines
- Repeated headers/footers/page numbers
- Broken hyphenation across line breaks
- Control characters and non-printable junk
- Mixed Arabic/English spacing issues

Designed to be safe for both Arabic (RTL) and English (LTR) text.
"""

from __future__ import annotations

import re
import unicodedata

from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

# Matches 3+ consecutive newlines (collapse to max 2, i.e. one blank line)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Matches runs of horizontal whitespace (spaces/tabs), not newlines
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Matches a hyphen at end-of-line followed by a lowercase letter continuation
# e.g. "govern-\nment" -> "government"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# Standalone numeric lines can be meaningful cover text (e.g. report No. 370
# and year 2025), so bare numbers are removed only at page edges. Explicit
# labels such as "Page 12" can be removed anywhere on the page.
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_PAGE_LABEL_LINE_RE = re.compile(r"^\s*page\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE)

# Non-printable / control characters (excluding normal whitespace \n \t)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TextCleaner:
    """Cleans raw extracted PDF text for downstream chunking."""

    def __init__(self, remove_page_number_lines: bool = True) -> None:
        self.remove_page_number_lines = remove_page_number_lines

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on a block of text."""
        if not text or not text.strip():
            return ""

        original_len = len(text)

        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)
        text = self._fix_hyphenation(text)

        if self.remove_page_number_lines:
            text = self._strip_page_number_lines(text)

        text = self._collapse_whitespace(text)
        text = text.strip()

        cleaned_len = len(text)
        logger.debug(f"Cleaned text: {original_len} -> {cleaned_len} chars")

        return text

    def clean_pages(self, pages: list[str]) -> list[str]:
        """Clean a list of page texts (order preserved)."""
        return [self.clean(p) for p in pages]

    # --- internal steps -----------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize unicode to NFC form (important for Arabic diacritics/ligatures)."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        return _CONTROL_CHAR_RE.sub("", text)

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """Rejoin words that were hyphenated across a line break."""
        return _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    @staticmethod
    def _strip_page_number_lines(text: str) -> str:
        """Remove labeled page numbers anywhere and bare numbers at page edges."""
        lines = text.split("\n")
        content_positions = [index for index, line in enumerate(lines) if line.strip()]
        if not content_positions:
            return text
        edge_positions = {content_positions[0], content_positions[-1]}
        kept = []
        for index, line in enumerate(lines):
            if _PAGE_LABEL_LINE_RE.match(line):
                continue
            if index in edge_positions and _PAGE_NUMBER_LINE_RE.match(line):
                continue
            kept.append(line)
        return "\n".join(kept)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse repeated spaces/tabs and excessive blank lines."""
        text = _MULTI_SPACE_RE.sub(" ", text)
        # Strip trailing whitespace on each line before collapsing newlines
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        return text