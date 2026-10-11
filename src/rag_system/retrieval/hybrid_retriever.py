"""
Hybrid retriever (Phase 6 / Step 2).

Combines VectorRetriever (Phase 4, semantic/dense) and BM25Retriever
(Phase 5, lexical, with cross-lingual query translation) behind one
interface, merging their ranked results via Reciprocal Rank Fusion
(Phase 6 / Step 1).

Design note: each underlying retriever is asked for `fetch_k` results
(more than the final `top_k` the caller wants), so RRF has enough
candidates from each source to fuse meaningfully before truncating to
the final top_k.
"""

from __future__ import annotations

from rag_system.chunking.models import Chunk
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import RetrievalError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    High-level entry point for hybrid (vector + BM25) retrieval via RRF.

    Owns one VectorRetriever and one BM25Retriever. `index_chunks()`
    indexes into both. `retrieve()` queries both, then fuses with RRF.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        fetch_multiplier: int = 2,
    ) -> None:
        self.settings = get_settings()
        self.vector_retriever = vector_retriever if vector_retriever is not None else VectorRetriever()
        self.bm25_retriever = bm25_retriever if bm25_retriever is not None else BM25Retriever()
        self.rrf_k = rrf_k
        # كل مصدر يجيب fetch_multiplier * top_k مرشحين قبل الدمج
        self.fetch_multiplier = fetch_multiplier

    def index_chunks(self, chunks: list[Chunk]) -> dict[str, int]:
        """Index the given chunks into both the vector store and BM25 index."""
        if not chunks:
            return {"vector": 0, "bm25": 0}

        vector_count = self.vector_retriever.index_chunks(chunks)
        bm25_count = self.bm25_retriever.index_chunks(chunks)

        logger.info(
            f"Hybrid indexing: {vector_count} chunk(s) -> vector store, "
            f"{bm25_count} chunk(s) -> BM25 index"
        )
        return {"vector": vector_count, "bm25": bm25_count}

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Return the top-k chunks after fusing vector + BM25 results via RRF.

        If one retriever fails (raises RetrievalError), the search
        continues with the surviving retriever's results alone rather
        than failing the whole hybrid search — a single backend outage
        shouldn't take down retrieval entirely. If *both* fail, the
        original RetrievalError from the vector retriever is raised.
        """
        if not query or not query.strip():
            raise RetrievalError("Cannot retrieve for an empty query")

        k = top_k or self.settings.retrieval.top_k
        fetch_k = k * self.fetch_multiplier

        vector_results: list[RetrievedChunk] = []
        bm25_results: list[RetrievedChunk] = []
        vector_error: Exception | None = None

        try:
            vector_results = self.vector_retriever.retrieve(query, top_k=fetch_k)
        except RetrievalError as e:
            vector_error = e
            logger.warning(f"Vector retrieval failed in hybrid search, continuing with BM25 only: {e}")

        try:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)
        except RetrievalError as e:
            logger.warning(f"BM25 retrieval failed in hybrid search, continuing with vector only: {e}")
            if vector_error is not None:
                # الاتنين فشلوا -> مفيش نتيجة ممكنة، نرفع الخطأ الأصلي
                raise vector_error

        fused = reciprocal_rank_fusion([vector_results, bm25_results], k=self.rrf_k)
        final = fused[:k]

        logger.info(
            f"Hybrid retrieval: query={query!r} -> "
            f"{len(vector_results)} vector + {len(bm25_results)} bm25 "
            f"-> {len(final)} fused result(s)"
        )
        return final

    def save(self) -> None:
        """Persist both underlying retrievers to their configured storage dirs."""
        self.vector_retriever.save()
        self.bm25_retriever.save()

    def __len__(self) -> int:
        return len(self.vector_retriever)