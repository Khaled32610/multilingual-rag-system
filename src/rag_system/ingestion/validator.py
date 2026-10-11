"""
PDF validation (Phase 1 / Step 04).

Checks, before any expensive extraction happens:
1. The file exists and is readable.
2. The extension is an allowed one (.pdf).
3. The file size is within the configured limit.
4. The file is a structurally valid, openable PDF (not corrupted/encrypted-without-access).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_system.ingestion.models import PDFValidationResult
from rag_system.utils.config import get_settings
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class PDFValidator:
    """Validates PDF files against configured rules before ingestion."""

    def __init__(self) -> None:
        settings = get_settings()
        self.max_file_size_mb: int = settings.pdf.max_file_size_mb
        self.allowed_extensions: list[str] = settings.pdf.allowed_extensions

    def validate(self, file_path: str | Path) -> PDFValidationResult:
        """Run all validation checks on a single PDF file and return the result."""
        path = Path(file_path)
        errors: list[str] = []

        logger.info(f"Validating PDF: {path}")

        # 1. Existence check
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            logger.warning(errors[-1])
            return PDFValidationResult(
                file_path=path, is_valid=False, file_size_mb=0.0, errors=errors
            )

        if not path.is_file():
            errors.append(f"Path is not a file: {path}")
            logger.warning(errors[-1])
            return PDFValidationResult(
                file_path=path, is_valid=False, file_size_mb=0.0, errors=errors
            )

        # 2. Extension check
        if path.suffix.lower() not in self.allowed_extensions:
            errors.append(
                f"Invalid extension '{path.suffix}'. Allowed: {self.allowed_extensions}"
            )
            logger.warning(errors[-1])

        # 3. Size check
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            errors.append(
                f"File size {file_size_mb:.2f}MB exceeds max of {self.max_file_size_mb}MB"
            )
            logger.warning(errors[-1])

        if file_size_mb == 0:
            errors.append("File is empty (0 bytes)")
            logger.warning(errors[-1])

        # 4. Structural validity check (can pypdf open and read it?)
        num_pages: int | None = None
        if not errors:  # no point opening a file that already failed basic checks
            try:
                reader = PdfReader(str(path))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")  # try empty password
                    except Exception:
                        pass
                    if reader.is_encrypted:
                        errors.append("PDF is encrypted and cannot be read without a password")
                        logger.warning(errors[-1])

                if not errors:
                    num_pages = len(reader.pages)
                    if num_pages == 0:
                        errors.append("PDF has zero pages")
                        logger.warning(errors[-1])

            except PdfReadError as e:
                errors.append(f"Corrupted or unreadable PDF structure: {e}")
                logger.warning(errors[-1])
            except Exception as e:
                errors.append(f"Unexpected error opening PDF: {e}")
                logger.error(errors[-1])

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(
                f"PDF valid: {path.name} ({file_size_mb:.2f}MB, {num_pages} pages)"
            )
        else:
            logger.warning(f"PDF invalid: {path.name} — {len(errors)} error(s)")

        return PDFValidationResult(
            file_path=path,
            is_valid=is_valid,
            file_size_mb=round(file_size_mb, 3),
            num_pages=num_pages,
            errors=errors,
        )

    def validate_batch(self, file_paths: list[str | Path]) -> list[PDFValidationResult]:
        """Validate multiple PDF files and return their results."""
        logger.info(f"Validating batch of {len(file_paths)} PDF file(s)")
        return [self.validate(fp) for fp in file_paths]