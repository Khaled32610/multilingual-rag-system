"""
Vector-based retriever (Phase 4 / Step 17).

Combines the default embedding model (Phase 3) with the FAISS vector
store (Phase 4 / Step 16) into a single object exposing two operations:
indexing chunks and retrieving the top-k most similar chunks for a
query. This is the "vector" half of hybrid retrieval — Phase 5 (BM25)
provides the lexical half, and Phase 6 (hybrid + RRF) will merge both
retrievers' `RetrievedChunk` lists behind this same shape.
"""

from __future__ import annotations

from pathlib import Path

from rag_system.chunking.models import Chunk
from rag_system.embedding.base import BaseEmbeddingModel
from rag_system.embedding.default import get_default_embedding_model
from rag_system.retrieval.models import RetrievedChunk
from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import RetrievalError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """
    High-level entry point for semantic (dense vector) retrieval.

    Owns an embedding model and a VectorStore. If no store is passed in
    and a persisted one already exists on disk (at
    `settings.paths.vector_store_dir`), it's loaded automatically;
    otherwise a fresh, empty store is created.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_model = embedding_model or get_default_embedding_model()

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            store_dir = Path(self.settings.paths.vector_store_dir)
            if VectorStore.exists(store_dir):
                self.vector_store = VectorStore.load(store_dir)
            else:
                self.vector_store = VectorStore(dimension=self.embedding_model.dimension)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and add a batch of chunks to the vector store. Returns count added."""
        if not chunks:
            return 0

        try:
            result = self.embedding_model.embed_texts(
                [c.text for c in chunks],
                batch_size=self.settings.embedding.batch_size,
            )
        except Exception as e:
            raise RetrievalError(
                f"Failed to embed {len(chunks)} chunk(s) for indexing",
                details={"error": str(e)},
            ) from e

        return self.vector_store.add_chunks(chunks, result.vectors)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks most semantically similar to `query`."""
        if not query or not query.strip():
            raise RetrievalError("Cannot retrieve for an empty query")

        k = top_k or self.settings.retrieval.top_k

        try:
            query_result = self.embedding_model.embed_query(query)
        except Exception as e:
            raise RetrievalError(
                f"Failed to embed query: {query!r}", details={"error": str(e)}
            ) from e

        results = self.vector_store.search(query_result.vectors[0], top_k=k)
        logger.info(f"Vector retrieval: query={query!r} -> {len(results)} result(s)")
        return results

    def save(self) -> None:
        """Persist the underlying vector store to `settings.paths.vector_store_dir`."""
        self.vector_store.save(self.settings.paths.vector_store_dir)

    def __len__(self) -> int:
        return len(self.vector_store)
