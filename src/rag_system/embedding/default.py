"""
Default embedding model factory (Phase 3 / Step 4).

Provides a single entry point, `get_default_embedding_model()`, that
returns the system's selected embedding model (chosen via benchmarking
in Phase 3 / Step 3). All later phases (vector store, retrieval, chat)
should use this instead of instantiating a specific model directly, so
switching the default model later means changing config, not code.
"""

from __future__ import annotations

import functools

from rag_system.embedding.base import BaseEmbeddingModel
from rag_system.embedding.sentence_transformer_model import (
    SentenceTransformerEmbeddingModel,
)
from rag_system.utils.config import get_settings
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def get_default_embedding_model() -> BaseEmbeddingModel:
    """
    Return the system default embedding model, loaded once and cached.

    Selected via Phase 3 benchmarking: intfloat/multilingual-e5-base
    scored Recall@3 = 100% on the retrieval test set, versus 75% and 25%
    for the other candidates, making it the clear choice for AR/EN
    document retrieval despite being the slowest to encode.
    """
    settings = get_settings()
    model_name = settings.embedding.default_model
    logger.info(f"Loading default embedding model: {model_name}")
    return SentenceTransformerEmbeddingModel(model_name=model_name)
