"""
Chunk data model (Phase 2 / Step 1).

Defines the core unit that flows through the rest of the RAG system:
embedding, vector storage, retrieval, reranking, and citation in the
final generated answer all operate on Chunk objects.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ChunkMetadata:
    """Metadata attached to a chunk, tracing it back to its source document."""

    source_file: str
    document_hash: str
    page_number: int
    chunk_index: int  # position of this chunk within the document (0-indexed)
    total_chunks_in_doc: int = 0
    char_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "document_hash": self.document_hash,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "total_chunks_in_doc": self.total_chunks_in_doc,
            "char_count": self.char_count,
            "created_at": self.created_at,
        }


@dataclass
class Chunk:
    """
    A single retrievable unit of text.

    `chunk_id` is deterministic (derived from document hash + chunk index),
    so re-chunking the same document produces the same IDs — important for
    incremental indexing (Phase 14) to detect unchanged chunks.
    """

    text: str
    metadata: ChunkMetadata
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
        self.metadata.char_count = len(self.text)

    def _generate_id(self) -> str:
        raw = f"{self.metadata.document_hash}:{self.metadata.chunk_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id[:8]}..., "
            f"page={self.metadata.page_number}, "
            f"idx={self.metadata.chunk_index}, "
            f"text='{preview}...')"
        )