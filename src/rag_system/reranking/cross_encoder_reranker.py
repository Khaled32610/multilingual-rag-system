"""
Cross-encoder reranker (Phase 7 / Step 1).

Wraps sentence-transformers' CrossEncoder to re-score (query, chunk)
pairs jointly, which captures finer semantic interactions than the
bi-encoder used for initial retrieval (Phase 3/4). Applied only to the
small top-k set returned by HybridRetriever (Phase 6), since
cross-encoders are too slow to run over a full corpus.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_system.reranking.base import BaseReranker
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import RerankingError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker backed by a sentence-transformers CrossEncoder model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        try:
            logger.info(f"Loading reranker model: {model_name} (device={device})")
            self.model = CrossEncoder(model_name, device=device)
            logger.info(f"Loaded reranker: {model_name}")
        except Exception as e:
            raise RerankingError(
                f"Failed to load reranker model: {model_name}",
                details={"error": str(e)},
            ) from e

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not query or not query.strip():
            raise RerankingError("Cannot rerank for an empty query")

        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            raw_scores = self.model.predict(pairs)
        except Exception as e:
            raise RerankingError(
                f"Reranking failed for {len(chunks)} chunk(s)",
                details={"error": str(e)},
            ) from e

        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, raw_scores):
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={**chunk.metadata, "pre_rerank_score": chunk.score, "pre_rerank_rank": chunk.rank},
                    score=float(score),
                    rank=0,
                    source="reranked",
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        final = reranked[:top_n]
        for rank, r in enumerate(final, start=1):
            r.rank = rank

        logger.info(f"Reranked {len(chunks)} chunk(s) -> top {len(final)} for query={query!r}")
        return final