"""
Text chunking / splitting (Phase 2 / Step 2).

Splits cleaned page text into overlapping chunks suitable for embedding.
Uses a recursive strategy: tries to split on paragraph boundaries first,
then sentences, then words, only falling back to a hard character cut
if nothing else works. Chunk size and overlap are measured in tokens
(via tiktoken) for accuracy with embedding models, not raw characters.
"""

from __future__ import annotations

import re

import tiktoken

from rag_system.utils.exceptions import ChunkingError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

# Separators tried in order, from "biggest" structural break to smallest.
_SEPARATORS = [
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # sentence end (English)
    "، ",     # sentence/clause separator (Arabic comma)
    "۔ ",     # Arabic full stop (used in Urdu/Farsi-influenced texts)
    "؛ ",     # Arabic semicolon
    " ",      # word boundary
    "",       # hard character split (last resort)
]

_DEFAULT_ENCODING = "cl100k_base"  # used by many modern embedding/LLM models


class TokenCounter:
    """Counts tokens using tiktoken, cached per encoding."""

    def __init__(self, encoding_name: str = _DEFAULT_ENCODING) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


class RecursiveTextSplitter:
    """
    Splits text into token-bounded chunks with overlap, preferring to
    break on natural boundaries (paragraphs > lines > sentences > words).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                "chunk_overlap must be smaller than chunk_size",
                details={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.counter = token_counter or TokenCounter()

    def split(self, text: str) -> list[str]:
        """Split a single block of text into a list of chunk strings."""
        if not text or not text.strip():
            return []

        chunks = self._recursive_split(text, _SEPARATORS)
        chunks = self._merge_small_pieces(chunks)
        chunks = self._add_overlap(chunks)

        logger.debug(f"Split text ({self.counter.count(text)} tokens) into {len(chunks)} chunk(s)")
        return chunks

    # --- internal steps -----------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the first separator that actually
        helps, falling back to the next one if a piece is still too big.
        """
        if self.counter.count(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Absolute fallback: hard split by token count
            return self._hard_split_by_tokens(text)

        sep = separators[0]
        remaining_separators = separators[1:]

        if sep == "":
            return self._hard_split_by_tokens(text)

        pieces = text.split(sep)
        if len(pieces) == 1:
            # This separator did not occur in the text; try the next one
            return self._recursive_split(text, remaining_separators)

        results: list[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if self.counter.count(piece) <= self.chunk_size:
                results.append(piece)
            else:
                results.extend(self._recursive_split(piece, remaining_separators))

        return results

    def _hard_split_by_tokens(self, text: str) -> list[str]:
        """Last-resort split: cut strictly by token count, no natural boundary."""
        encoding = self.counter._encoding
        tokens = encoding.encode(text, disallowed_special=())
        pieces = []
        for i in range(0, len(tokens), self.chunk_size):
            token_slice = tokens[i : i + self.chunk_size]
            pieces.append(encoding.decode(token_slice))
        return pieces

    def _merge_small_pieces(self, pieces: list[str]) -> list[str]:
        """
        Greedily merge consecutive small pieces together so we don't end up
        with many tiny chunks (e.g. one sentence each) when they could fit
        together under chunk_size.
        """
        if not pieces:
            return []

        merged: list[str] = []
        buffer = ""

        for piece in pieces:
            candidate = f"{buffer} {piece}".strip() if buffer else piece
            if self.counter.count(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = piece

        if buffer:
            merged.append(buffer)

        return merged

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        Prepend a token-based overlap from the end of the previous chunk
        onto the start of each subsequent chunk, so context isn't lost
        at chunk boundaries.
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        encoding = self.counter._encoding
        overlapped: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_tokens = encoding.encode(chunks[i - 1], disallowed_special=())
            overlap_tokens = prev_tokens[-self.chunk_overlap :]
            overlap_text = encoding.decode(overlap_tokens)
            overlapped.append(f"{overlap_text} {chunks[i]}".strip())

        return overlapped