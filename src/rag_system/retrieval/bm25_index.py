"""
BM25 lexical index (Phase 5 / Step 18), backed by the `bm25s` library.

Complements the dense vector store (Phase 4): vector search finds
chunks that are *semantically* similar to a query, while BM25 finds
chunks that share *exact* vocabulary with it — useful for queries that
contain specific numbers, article/law references, or proper nouns that
an embedding model might blur together. Phase 6 (hybrid retrieval)
merges both signals via RRF.

Design notes:

- **No stopword removal.** Removing English stopwords ("the", "is", ...)
  but not Arabic ones (or vice versa, without a curated Arabic list)
  would treat the two languages asymmetrically. Since this project
  prioritizes accuracy and its documents/queries are domain-specific
  (legal/budget text, not everyday prose), we index every token.
- **Normalization, not stemming**, for Arabic text — see
  `bm25_tokenizer.normalize_tokens` for the full rationale.
- **Rebuilt, not updated in place.** Unlike the FAISS vector store,
  bm25s computes its term-frequency statistics over the whole corpus
  at once and has no supported way to add/remove individual documents
  afterwards. `build()` therefore replaces the entire index. This is
  acceptable here because building a BM25 index at this project's
  corpus sizes is fast (milliseconds to low seconds) — cheap enough to
  redo whenever the chunk set changes (Phase 14), unlike re-embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s

from rag_system.chunking.models import Chunk
from rag_system.retrieval.bm25_tokenizer import normalize_tokens
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import BM25Error
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_RECORDS_FILE_NAME = "records.json"


class BM25Index:
    """Lexical (keyword) index over a fixed set of chunks, built with bm25s."""

    def __init__(self) -> None:
        self._retriever: bm25s.BM25 | None = None
        # Position in the indexed corpus (0..n-1) -> chunk record. bm25s
        # returns result *positions* into the corpus it was built from
        # (we don't pass corpus= to bm25s itself), so this list doubles
        # as that position -> metadata map.
        self._records: list[dict] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def is_built(self) -> bool:
        return self._retriever is not None

    def clear(self) -> None:
        """Reset the index to an empty state after the final document is removed."""
        self._retriever = None
        self._records = []

    # --- Build (full rebuild; see module docstring) -------------------------

    def build(self, chunks: list[Chunk]) -> int:
        """Build the index from scratch over the given chunks. Returns count indexed."""
        if not chunks:
            raise BM25Error("Cannot build a BM25 index from an empty chunk list")

        texts = [c.text for c in chunks]
        try:
            corpus_tokens = bm25s.tokenize(
                texts, stopwords=[], stemmer=normalize_tokens, show_progress=False
            )
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens, show_progress=False)
        except Exception as e:
            raise BM25Error(
                f"Failed to build BM25 index over {len(chunks)} chunk(s)",
                details={"error": str(e)},
            ) from e

        self._retriever = retriever
        self._records = [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata.to_dict()}
            for c in chunks
        ]
        logger.info(f"Built BM25 index over {len(chunks)} chunk(s)")
        return len(chunks)

    # --- Search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise BM25Error("Cannot search a BM25 index with an empty query")
        if top_k <= 0:
            raise BM25Error(f"top_k must be positive, got {top_k}")

        if not self.is_built or not self._records:
            logger.warning("BM25 search called on an unbuilt/empty index")
            return []

        k = min(top_k, len(self._records))
        try:
            query_tokens = bm25s.tokenize(
                [query], stopwords=[], stemmer=normalize_tokens, show_progress=False
            )
            positions, scores = self._retriever.retrieve(query_tokens, k=k, show_progress=False)
        except Exception as e:
            raise BM25Error(
                f"BM25 search failed for query: {query!r}", details={"error": str(e)}
            ) from e

        results: list[RetrievedChunk] = []
        for rank, (pos, score) in enumerate(zip(positions[0], scores[0]), start=1):
            record = self._records[int(pos)]
            results.append(
                RetrievedChunk(
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    metadata=record["metadata"],
                    score=float(score),
                    rank=rank,
                    source="bm25",
                )
            )
        return results

    # --- Persistence -----------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        if not self.is_built:
            raise BM25Error("Cannot save an unbuilt BM25 index")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        try:
            self._retriever.save(str(directory))
        except Exception as e:
            raise BM25Error(
                f"Failed to save BM25 index to {directory}", details={"error": str(e)}
            ) from e

        with (directory / _RECORDS_FILE_NAME).open("w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved BM25 index ({len(self._records)} chunk(s)) to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        directory = Path(directory)
        records_path = directory / _RECORDS_FILE_NAME
        if not records_path.exists():
            raise BM25Error(f"No BM25 index found at {directory}")

        index = cls()
        try:
            index._retriever = bm25s.BM25.load(str(directory), load_corpus=False)
        except Exception as e:
            raise BM25Error(
                f"Failed to load BM25 index from {directory}", details={"error": str(e)}
            ) from e

        with records_path.open("r", encoding="utf-8") as f:
            index._records = json.load(f)

        logger.info(f"Loaded BM25 index ({len(index._records)} chunk(s)) from {directory}")
        return index

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / _RECORDS_FILE_NAME).exists()
