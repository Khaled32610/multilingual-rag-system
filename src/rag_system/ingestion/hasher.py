"""
PDF hashing (Phase 1 / Step 05).

Computes a stable content-based hash (SHA-256) for a PDF file. This hash
is used later for:
- Deduplication (skip re-processing an identical file).
- Incremental indexing (Phase 14): detect when a previously indexed file
  has changed on disk and needs re-processing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_system.utils.exceptions import PDFHashingError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CHUNK_SIZE = 8192  # bytes read per iteration, keeps memory usage low


class PDFHasher:
    """Computes content-based hashes for PDF files."""

    def __init__(self, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> None:
        self.chunk_size = chunk_size

    def compute_hash(self, file_path: str | Path) -> str:
        """
        Compute the SHA-256 hash of a file's contents.

        Reads the file in chunks to avoid loading large PDFs entirely
        into memory. Returns the hash as a hex string.
        """
        path = Path(file_path)

        if not path.exists():
            raise PDFHashingError(f"Cannot hash: file does not exist: {path}")

        logger.info(f"Computing hash for: {path.name}")

        sha256 = hashlib.sha256()
        try:
            with path.open("rb") as f:
                while chunk := f.read(self.chunk_size):
                    sha256.update(chunk)
        except OSError as e:
            raise PDFHashingError(
                f"Failed to read file while hashing: {path}", details={"error": str(e)}
            ) from e

        digest = sha256.hexdigest()
        logger.info(f"Hash computed for {path.name}: {digest[:12]}...")
        return digest

    def compute_hash_batch(self, file_paths: list[str | Path]) -> dict[str, str]:
        """
        Compute hashes for multiple files.

        Returns a dict mapping file path (as string) -> hash.
        """
        logger.info(f"Computing hashes for batch of {len(file_paths)} file(s)")
        return {str(fp): self.compute_hash(fp) for fp in file_paths}

    @staticmethod
    def find_duplicates(hash_map: dict[str, str]) -> dict[str, list[str]]:
        """
        Given a mapping of file_path -> hash, return only the hashes that
        are shared by more than one file path (i.e. actual duplicates).

        Returns: dict of hash -> list of file paths sharing that hash.
        """
        reverse: dict[str, list[str]] = {}
        for path, digest in hash_map.items():
            reverse.setdefault(digest, []).append(path)

        duplicates = {h: paths for h, paths in reverse.items() if len(paths) > 1}
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate hash group(s)")
        return duplicates