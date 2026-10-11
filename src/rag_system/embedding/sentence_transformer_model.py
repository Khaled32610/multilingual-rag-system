"""
Sentence-Transformers-based embedding model implementation (Phase 3 / Step 2).

Wraps any model available via the `sentence-transformers` library behind
our BaseEmbeddingModel interface. Handles the E5-family's requirement for
"query:" / "passage:" prefixes automatically based on model name, since
that family needs it for good retrieval performance.
"""

from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.embedding.base import BaseEmbeddingModel
from rag_system.embedding.models import EmbeddingResult
from rag_system.utils.exceptions import EmbeddingError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

# Model families that require specific prefixes on input text for
# asymmetric (query vs. passage) embedding quality.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """Embedding model backed by a local sentence-transformers model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.name = model_name
        self._is_e5 = _is_e5_model(model_name)

        logger.info(f"Loading embedding model: {model_name} (device={device})")
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model: {model_name}",
                details={"error": str(e)},
            ) from e

        self.dimension = self._model.get_embedding_dimension()
        logger.info(f"Loaded {model_name}: dimension={self.dimension}")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts")

        prepared = self._prepare_texts(texts, is_query=False)

        start = time.perf_counter()
        try:
            vectors = self._model.encode(
                prepared,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed {len(texts)} text(s) with {self.name}",
                details={"error": str(e)},
            ) from e
        elapsed = time.perf_counter() - start

        return EmbeddingResult(
            model_name=self.name,
            vectors=np.asarray(vectors),
            dimension=self.dimension,
            num_texts=len(texts),
            elapsed_seconds=elapsed,
        )

    def embed_query(self, query: str) -> EmbeddingResult:
        if not query or not query.strip():
            raise EmbeddingError("Cannot embed an empty query")

        prepared = self._prepare_texts([query], is_query=True)

        start = time.perf_counter()
        try:
            vector = self._model.encode(
                prepared,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed query with {self.name}",
                details={"error": str(e)},
            ) from e
        elapsed = time.perf_counter() - start

        return EmbeddingResult(
            model_name=self.name,
            vectors=np.asarray(vector),
            dimension=self.dimension,
            num_texts=1,
            elapsed_seconds=elapsed,
        )

    def _prepare_texts(self, texts: list[str], is_query: bool) -> list[str]:
        """Apply model-family-specific prefixes (e.g. E5's query:/passage:)."""
        if not self._is_e5:
            return texts

        prefix = _E5_QUERY_PREFIX if is_query else _E5_PASSAGE_PREFIX
        return [f"{prefix}{t}" for t in texts]