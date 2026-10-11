"""Factory for the default reranker (Phase 7 / Step 1)."""

from __future__ import annotations

from functools import lru_cache

from rag_system.reranking.base import BaseReranker
from rag_system.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_system.utils.config import get_settings


@lru_cache(maxsize=1)
def get_default_reranker() -> BaseReranker:
    settings = get_settings()
    return CrossEncoderReranker(model_name=settings.reranking.model_name)