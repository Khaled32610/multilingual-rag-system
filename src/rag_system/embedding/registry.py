"""
Candidate embedding model registry (Phase 3 / Step 2).

Central list of embedding models to be benchmarked against each other
(Phase 3 / Step 3) before one is selected as the system default
(Phase 3 / Step 4). Keeping this as a simple registry makes it trivial
to add/remove candidates without touching benchmarking logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_system.embedding.base import BaseEmbeddingModel
from rag_system.embedding.sentence_transformer_model import (
    SentenceTransformerEmbeddingModel,
)
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateModelSpec:
    """Specification for a candidate embedding model."""

    model_name: str
    display_name: str
    description: str


CANDIDATE_MODELS: list[CandidateModelSpec] = [
    CandidateModelSpec(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        display_name="MiniLM-L12-multilingual",
        description="Small & fast, 384-dim, good baseline for AR+EN.",
    ),
    CandidateModelSpec(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        display_name="mpnet-base-multilingual",
        description="Medium size, 768-dim, higher accuracy than MiniLM.",
    ),
    CandidateModelSpec(
        model_name="intfloat/multilingual-e5-base",
        display_name="multilingual-e5-base",
        description="Strong multilingual retrieval model, 768-dim, uses query/passage prefixes.",
    ),
]


def load_candidate_model(spec: CandidateModelSpec, device: str = "cpu") -> BaseEmbeddingModel:
    """Instantiate (download-if-needed and load) a single candidate model."""
    logger.info(f"Loading candidate model: {spec.display_name} ({spec.model_name})")
    return SentenceTransformerEmbeddingModel(model_name=spec.model_name, device=device)


def load_all_candidates(device: str = "cpu") -> dict[str, BaseEmbeddingModel]:
    """Load all registered candidate models. Returns dict of display_name -> model."""
    models: dict[str, BaseEmbeddingModel] = {}
    for spec in CANDIDATE_MODELS:
        try:
            models[spec.display_name] = load_candidate_model(spec, device=device)
        except Exception as e:
            logger.error(f"Failed to load candidate {spec.display_name}: {e}")
    return models