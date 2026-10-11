"""
BM25 retriever (Phase 5 / Step 1, extended in Step 2).

Thin wrapper around BM25Index giving it the same shape as
VectorRetriever (Phase 4 / Step 17) — `index_chunks()` / `retrieve()` /
`save()` — so Phase 6 (hybrid retrieval) can hold one of each behind an
identical interface and merge their `RetrievedChunk` results with RRF.

Step 2 adds cross-lingual query translation (see query_translator.py
for the full rationale): when the query's detected language differs
from English, `retrieve()` also runs BM25 with an English translation
of the query and merges the two result sets, keeping each chunk's best
score. Translation is a best-effort enhancement, not a hard
requirement — if it fails (model not downloaded yet, no network, etc.)
`retrieve()` logs a warning and falls back to the original-language
results only, rather than failing the whole search.
"""

from __future__ import annotations

from pathlib import Path

from rag_system.chunking.models import Chunk
from rag_system.retrieval.bm25_index import BM25Index
from rag_system.retrieval.language_utils import detect_language
from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.query_translator import QueryTranslator, expand_english_query
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import RetrievalError, TranslationError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


def _merge_bm25_results(
    primary: list[RetrievedChunk], secondary: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """
    Merge two BM25 result lists for the same underlying query (original
    language + translated language), keeping each chunk's best score
    across the two searches, then re-ranking and truncating to top_k.
    """
    best: dict[str, RetrievedChunk] = {}
    for r in primary + secondary:
        existing = best.get(r.chunk_id)
        if existing is None or r.score > existing.score:
            best[r.chunk_id] = r

    merged = sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]
    for rank, r in enumerate(merged, start=1):
        r.rank = rank
    return merged


class BM25Retriever:
    """High-level entry point for lexical (BM25) retrieval."""

    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        query_translator: QueryTranslator | None = None,
    ) -> None:
        self.settings = get_settings()

        if bm25_index is not None:
            self.bm25_index = bm25_index
        else:
            index_dir = Path(self.settings.paths.bm25_index_dir)
            if BM25Index.exists(index_dir):
                self.bm25_index = BM25Index.load(index_dir)
            else:
                self.bm25_index = BM25Index()

        if query_translator is not None:
            self.query_translator = query_translator
        elif self.settings.retrieval.enable_query_translation:
            self.query_translator = QueryTranslator(
                model_name=self.settings.retrieval.translation_model
            )
        else:
            self.query_translator = None

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """
        (Re)build the BM25 index over the given chunks.

        Unlike VectorRetriever.index_chunks, this replaces any existing
        index rather than adding to it — see BM25Index's module
        docstring for why BM25 is rebuilt rather than updated in place.
        """
        if not chunks:
            return 0
        try:
            return self.bm25_index.build(chunks)
        except Exception as e:
            raise RetrievalError(
                f"Failed to build BM25 index over {len(chunks)} chunk(s)",
                details={"error": str(e)},
            ) from e

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Return the top-k chunks with the highest BM25 score for `query`.

        If query translation is enabled and `query` isn't already
        English, also searches with an English translation of it and
        merges both result sets — see module docstring.
        """
        if not query or not query.strip():
            raise RetrievalError("Cannot retrieve for an empty query")

        k = top_k or self.settings.retrieval.top_k
        results = self.bm25_index.search(query, top_k=k)

        if detect_language(query) == "en":
            expanded_query = expand_english_query(query)
            if expanded_query != query:
                expanded_results = self.bm25_index.search(expanded_query, top_k=k)
                results = _merge_bm25_results(results, expanded_results, top_k=k)
                logger.info(f"BM25 retrieval: expanded English query={expanded_query!r}")

        if self.query_translator is not None and detect_language(query) != "en":
            try:
                translated = self.query_translator.translate_to_english(query)
            except TranslationError as e:
                logger.warning(
                    f"Query translation failed, continuing with original-language "
                    f"BM25 results only: {e}"
                )
                translated = None

            if translated and translated.strip().lower() != query.strip().lower():
                translated_results = self.bm25_index.search(translated, top_k=k)
                results = _merge_bm25_results(results, translated_results, top_k=k)
                logger.info(
                    f"BM25 retrieval: translated query={query!r} -> {translated!r}"
                )

        logger.info(f"BM25 retrieval: query={query!r} -> {len(results)} result(s)")
        return results

    def save(self) -> None:
        """Persist the underlying BM25 index, or remove stale files if empty."""
        index_dir = Path(self.settings.paths.bm25_index_dir)
        if self.bm25_index.is_built:
            self.bm25_index.save(index_dir)
            return
        if index_dir.exists():
            for path in index_dir.iterdir():
                if path.is_file():
                    path.unlink()

    def __len__(self) -> int:
        return len(self.bm25_index)