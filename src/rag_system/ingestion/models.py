"""
Shared data models for the PDF ingestion pipeline (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PDFValidationResult:
    """Result of validating a single PDF file before ingestion."""

    file_path: Path
    is_valid: bool
    file_size_mb: float
    num_pages: int | None = None
    errors: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        from rag_system.utils.exceptions import PDFValidationError

        if not self.is_valid:
            raise PDFValidationError(
                f"PDF validation failed for {self.file_path.name}",
                details={"errors": self.errors, "path": str(self.file_path)},
            )

@dataclass
class PageContent:
    """Extracted content of a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    char_count: int = 0
    has_tables: bool = False
    tables_discarded: int = 0  # جديد: عدد الجداول اللي اتجاهلت كمشكوك فيها (رسم بياني/فاضية)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ExtractionResult:
    """Result of extracting text from an entire PDF document."""

    file_path: Path
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""
    total_pages: int = 0
    total_chars: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return len(self.errors) == 0 and self.total_chars > 0

    def raise_if_failed(self) -> None:
        from rag_system.utils.exceptions import PDFExtractionError

        if not self.is_successful:
            raise PDFExtractionError(
                f"PDF extraction failed for {self.file_path.name}",
                details={"errors": self.errors, "path": str(self.file_path)},
            )
@dataclass
class DocumentMetadata:
    """
    Unified metadata for a single ingested PDF document.

    This is the final output of Phase 1 (PDF Ingestion) and the input
    that Phase 2 (Chunking) will consume.
    """

    file_path: Path
    file_name: str
    content_hash: str
    file_size_mb: float
    total_pages: int
    total_chars: int
    is_valid: bool
    is_extracted: bool
    validation_errors: list[str] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)
    cleaned_pages: list[str] = field(default_factory=list)

    @property
    def is_ready_for_chunking(self) -> bool:
        return (
            self.is_valid
            and self.is_extracted
            and self.total_chars > 0
            and len(self.cleaned_pages) > 0
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for saving as JSON, DB row, etc)."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "content_hash": self.content_hash,
            "file_size_mb": self.file_size_mb,
            "total_pages": self.total_pages,
            "total_chars": self.total_chars,
            "is_valid": self.is_valid,
            "is_extracted": self.is_extracted,
            "validation_errors": self.validation_errors,
            "extraction_errors": self.extraction_errors,
            "is_ready_for_chunking": self.is_ready_for_chunking,
        }
