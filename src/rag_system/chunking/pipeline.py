"""
Chunking pipeline orchestrator (Phase 2 / Step 3).

Combines Phase 1's DocumentMetadata (cleaned page text) with the
RecursiveTextSplitter to produce a final list of Chunk objects,
each carrying full traceability metadata back to its source page.
"""

from __future__ import annotations

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.chunking.splitter import RecursiveTextSplitter, TokenCounter
from rag_system.ingestion.models import DocumentMetadata
from rag_system.ingestion.pipeline import IngestionPipeline
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import ChunkingError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingPipeline:
    """
    End-to-end: PDF file path -> validated/cleaned pages -> list[Chunk].

    Wraps IngestionPipeline (Phase 1) + RecursiveTextSplitter (Phase 2)
    so the rest of the system can go straight from "here's a PDF" to
    "here are its chunks" in one call.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap

        self.ingestion_pipeline = IngestionPipeline()
        self.counter = TokenCounter()
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            token_counter=self.counter,
        )

    def chunk_document(self, doc_metadata: DocumentMetadata) -> list[Chunk]:
        """
        Split an already-ingested document (from Phase 1) into Chunk objects.
        Use this when you already have a DocumentMetadata (e.g. from a batch
        ingestion run) and don't want to re-run validation/extraction.
        """
        if not doc_metadata.is_ready_for_chunking:
            raise ChunkingError(
                f"Document is not ready for chunking: {doc_metadata.file_name}",
                details={
                    "validation_errors": doc_metadata.validation_errors,
                    "extraction_errors": doc_metadata.extraction_errors,
                },
            )

        chunks: list[Chunk] = []
        chunk_index = 0

        for page_number, page_text in enumerate(doc_metadata.cleaned_pages, start=1):
            page_pieces = self.splitter.split(page_text)

            for piece in page_pieces:
                metadata = ChunkMetadata(
                    source_file=doc_metadata.file_name,
                    document_hash=doc_metadata.content_hash,
                    page_number=page_number,
                    chunk_index=chunk_index,
                )
                chunks.append(Chunk(text=piece, metadata=metadata))
                chunk_index += 1

        # Now that we know the final count, backfill total_chunks_in_doc
        for chunk in chunks:
            chunk.metadata.total_chunks_in_doc = len(chunks)

        logger.info(
            f"Chunked document: {doc_metadata.file_name} — "
            f"{len(doc_metadata.cleaned_pages)} pages -> {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return chunks

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """
        Full pipeline: ingest a raw PDF file path and return its Chunk objects.
        Convenience method combining Phase 1 + Phase 2 in a single call.
        """
        doc_metadata = self.ingestion_pipeline.ingest(file_path)
        return self.chunk_document(doc_metadata)

    def chunk_files(self, file_paths: list[str]) -> dict[str, list[Chunk]]:
        """Chunk multiple PDF files. Returns dict of file_path -> list[Chunk]."""
        results: dict[str, list[Chunk]] = {}
        for fp in file_paths:
            try:
                results[str(fp)] = self.chunk_file(fp)
            except ChunkingError as e:
                logger.warning(f"Skipping file due to chunking error: {fp} — {e}")
                results[str(fp)] = []
        return results