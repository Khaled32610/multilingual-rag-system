"""
PDF ingestion pipeline orchestrator (Phase 1 / Step 5).

Ties together validation, hashing, extraction, and cleaning into a single
`ingest()` call that produces a DocumentMetadata object ready for Phase 2
(Chunking).
"""

from __future__ import annotations

from pathlib import Path

from rag_system.ingestion.cleaner import TextCleaner
from rag_system.ingestion.extractor import PDFExtractor
from rag_system.ingestion.hasher import PDFHasher
from rag_system.ingestion.models import DocumentMetadata
from rag_system.ingestion.validator import PDFValidator
from rag_system.utils.config import get_settings
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full Phase 1 flow for a single PDF:
    validate -> hash -> extract -> clean -> package as DocumentMetadata.
    """

    def __init__(self) -> None:
        self.validator = PDFValidator()
        self.hasher = PDFHasher()
        pdf_settings = get_settings().pdf
        self.extractor = PDFExtractor(
            enable_table_extraction=pdf_settings.enable_table_extraction,
            max_table_text_ratio=pdf_settings.max_table_text_ratio,
            min_table_density=pdf_settings.min_table_density,
        )
        self.cleaner = TextCleaner()

    def ingest(self, file_path: str | Path) -> DocumentMetadata:
        """
        Run the full ingestion flow on a single PDF file.

        Never raises on expected failures (invalid PDF, extraction issues) —
        instead returns a DocumentMetadata with is_valid/is_extracted flags
        set accordingly, so callers can decide how to handle failures
        (e.g. skip and log, vs. hard stop).
        """
        path = Path(file_path)
        logger.info(f"=== Ingesting: {path.name} ===")

        # Step 1: Validate
        validation = self.validator.validate(path)

        if not validation.is_valid:
            logger.warning(f"Ingestion stopped early — invalid PDF: {path.name}")
            return DocumentMetadata(
                file_path=path,
                file_name=path.name,
                content_hash="",
                file_size_mb=validation.file_size_mb,
                total_pages=0,
                total_chars=0,
                is_valid=False,
                is_extracted=False,
                validation_errors=validation.errors,
            )

        # Step 2: Hash
        content_hash = self.hasher.compute_hash(path)

        # Step 3: Extract
        extraction = self.extractor.extract(path)

        # Step 4: Clean
        cleaned_pages = self.cleaner.clean_pages([p.text for p in extraction.pages])

        metadata = DocumentMetadata(
            file_path=path,
            file_name=path.name,
            content_hash=content_hash,
            file_size_mb=validation.file_size_mb,
            total_pages=extraction.total_pages,
            total_chars=sum(len(t) for t in cleaned_pages),
            is_valid=validation.is_valid,
            is_extracted=extraction.is_successful,
            validation_errors=validation.errors,
            extraction_errors=extraction.errors,
            cleaned_pages=cleaned_pages,
        )

        if metadata.is_ready_for_chunking:
            logger.info(
                f"=== Ingestion complete: {path.name} — "
                f"{metadata.total_pages} pages, {metadata.total_chars} chars, "
                f"hash={content_hash[:12]}... ==="
            )
        else:
            logger.warning(f"=== Ingestion finished with issues: {path.name} ===")

        return metadata

    def ingest_batch(self, file_paths: list[str | Path]) -> list[DocumentMetadata]:
        """Run ingestion on multiple PDF files."""
        logger.info(f"Starting batch ingestion of {len(file_paths)} file(s)")
        results = [self.ingest(fp) for fp in file_paths]

        ready = sum(1 for r in results if r.is_ready_for_chunking)
        logger.info(f"Batch ingestion complete: {ready}/{len(results)} ready for chunking")

        return results