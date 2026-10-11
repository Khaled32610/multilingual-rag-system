"""
Full retrieval pipeline (Phase 7 / Step 2): Hybrid Retrieval + Reranking.

Flow: query -> HybridRetriever.retrieve() (broad candidate set via RRF)
-> Reranker.rerank() (precise cross-encoder re-scoring) -> final top_n.

Reranking is applied only to the hybrid stage's output (small candidate
set), never to the full corpus, since cross-encoders are too slow for
that.
"""

from __future__ import annotations

from rag_system.reranking.base import BaseReranker
from rag_system.reranking.default import get_default_reranker
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.language_utils import detect_language
from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.query_translator import expand_english_query
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import RerankingError, RetrievalError, TranslationError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


def _prepare_rerank_query(query: str, candidates: list[RetrievedChunk], translator) -> str:
    """Translate Arabic evidence queries when needed, then add domain synonyms."""
    rerank_query = query
    excerpt_sample = " ".join(chunk.text for chunk in candidates[: min(5, len(candidates))])
    if (
        translator is not None
        and detect_language(query) == "ar"
        and detect_language(excerpt_sample) == "en"
    ):
        try:
            translated = translator.translate_to_english(query)
            if translated and translated.strip():
                rerank_query = translated
        except TranslationError as exc:
            logger.warning(f"Could not translate Arabic query for reranking; using original: {exc}")
    return expand_english_query(rerank_query)


class RetrievalPipeline:
    """
    High-level entry point combining hybrid retrieval and reranking.

    If the reranker fails (model error, etc.), retrieve() falls back to
    the hybrid (pre-rerank) results rather than failing the whole
    search — reranking is a quality refinement, not a hard requirement.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.settings = get_settings()
        self.hybrid_retriever = (
            hybrid_retriever if hybrid_retriever is not None else HybridRetriever()
        )
        self.reranker = reranker if reranker is not None else get_default_reranker()

    def index_chunks(self, chunks) -> dict[str, int]:
        return self.hybrid_retriever.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_n: int | None = None,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve via hybrid search, then rerank to the requested result count.

        ``top_k`` matches the retriever interface used by ``GroundedGenerator``;
        ``top_n`` remains supported for existing pipeline callers.
        """
        if top_n is not None and top_k is not None:
            raise ValueError("Pass either top_n or top_k, not both")
        requested_count = top_k if top_k is not None else top_n
        n = requested_count or self.settings.reranking.top_n

        # هنجيب مرشحين أكتر من top_n من الـ hybrid عشان الـ reranker
        # يكون عنده مجال كافي يعيد الترتيب فيه (مش مجرد يقص).
        fetch_k = max(n * 3, self.settings.retrieval.top_k)

        candidates = self.hybrid_retriever.retrieve(query, top_k=fetch_k)
        if not candidates:
            return []

        bm25_retriever = getattr(self.hybrid_retriever, "bm25_retriever", None)
        translator = getattr(bm25_retriever, "query_translator", None)
        rerank_query = _prepare_rerank_query(query, candidates, translator)

        try:
            return self.reranker.rerank(rerank_query, candidates, top_n=n)
        except RerankingError as e:
            logger.warning(
                f"Reranking failed, falling back to hybrid (pre-rerank) results: {e}"
            )
            return candidates[:n]

    def save(self) -> None:
        self.hybrid_retriever.save()