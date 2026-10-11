"""
Custom exception hierarchy for rag_system.

All project-specific errors should inherit from RAGSystemError so callers
can catch `except RAGSystemError` to handle any known failure mode, while
still being able to catch specific subtypes when needed.
"""

from __future__ import annotations


class RAGSystemError(Exception):
    """Base exception for all rag_system errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


# --- Configuration ---------------------------------------------------------

class ConfigError(RAGSystemError):
    """Raised when configuration loading or validation fails."""


# --- PDF Ingestion (Phase 1) ------------------------------------------------

class PDFError(RAGSystemError):
    """Base class for all PDF-related errors."""


class PDFValidationError(PDFError):
    """Raised when a PDF fails validation (bad extension, size, corrupted)."""


class PDFHashingError(PDFError):
    """Raised when computing a PDF's hash fails."""


class PDFExtractionError(PDFError):
    """Raised when text/metadata extraction from a PDF fails."""


# --- Chunking (Phase 2) -----------------------------------------------------

class ChunkingError(RAGSystemError):
    """Raised when splitting text into chunks fails."""


# --- Embedding (Phase 3) ----------------------------------------------------

class EmbeddingError(RAGSystemError):
    """Raised when generating embeddings fails."""


class EmbeddingBenchmarkError(RAGSystemError):
    """Raised when the embedding benchmarking process fails."""


# --- Retrieval (Phases 4-8) -------------------------------------------------

class VectorStoreError(RAGSystemError):
    """Raised for vector store read/write/query failures."""


class BM25Error(RAGSystemError):
    """Raised for BM25 index build/query failures."""


class RetrievalError(RAGSystemError):
    """Raised for general/hybrid retrieval failures."""


class TranslationError(RAGSystemError):
    """Raised when cross-lingual query translation fails (used by BM25 to
    bridge Arabic queries against English documents and vice versa)."""


class RerankingError(RAGSystemError):
    """Raised when reranking retrieved candidates fails."""


# --- LLM & Generation (Phases 9-10) -----------------------------------------

class LLMError(RAGSystemError):
    """Raised when an LLM call fails."""


class PromptBuildError(RAGSystemError):
    """Raised when building a prompt fails."""


class GenerationError(RAGSystemError):
    """Raised when answer generation fails."""


# --- Verification (Phase 11) ------------------------------------------------

class EvidenceGateError(RAGSystemError):
    """Raised when the evidence gate check fails to run."""


class CitationValidationError(RAGSystemError):
    """Raised when citation validation fails to run."""


class FaithfulnessCheckError(RAGSystemError):
    """Raised when the faithfulness checker fails to run."""


# --- Pipeline & Chat (Phases 12-13) -----------------------------------------

class PipelineError(RAGSystemError):
    """Raised for failures in the end-to-end RAG pipeline."""


class ChatHistoryError(RAGSystemError):
    """Raised for chat/session history failures."""


# --- Document Management (Phase 14) -----------------------------------------

class IndexingError(RAGSystemError):
    """Raised when incremental indexing fails."""


class DocumentManagementError(RAGSystemError):
    """Raised for document add/remove/update failures."""


# --- Evaluation (Phases 15-17) ----------------------------------------------

class EvaluationError(RAGSystemError):
    """Raised when evaluation or experiment runs fail."""

