"""Unit tests for Phase 2 / Step 3: ChunkingPipeline."""

import pytest

from rag_system.chunking.pipeline import ChunkingPipeline
from rag_system.ingestion.models import DocumentMetadata
from rag_system.utils.exceptions import ChunkingError
from pathlib import Path


def make_ready_document(pages: list[str]) -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("fake.pdf"),
        file_name="fake.pdf",
        content_hash="f" * 64,
        file_size_mb=0.1,
        total_pages=len(pages),
        total_chars=sum(len(p) for p in pages),
        is_valid=True,
        is_extracted=True,
        cleaned_pages=pages,
    )


def make_invalid_document() -> DocumentMetadata:
    return DocumentMetadata(
        file_path=Path("bad.pdf"),
        file_name="bad.pdf",
        content_hash="",
        file_size_mb=0.0,
        total_pages=0,
        total_chars=0,
        is_valid=False,
        is_extracted=False,
        validation_errors=["invalid extension"],
    )


@pytest.fixture(scope="module")
def pipeline():
    return ChunkingPipeline(chunk_size=50, chunk_overlap=10)


class TestChunkDocument:
    def test_raises_on_document_not_ready(self, pipeline):
        doc = make_invalid_document()
        with pytest.raises(ChunkingError):
            pipeline.chunk_document(doc)

    def test_produces_chunks_for_valid_document(self, pipeline):
        doc = make_ready_document(["This is page one text. " * 10])
        chunks = pipeline.chunk_document(doc)
        assert len(chunks) > 0

    def test_chunk_metadata_traces_back_to_source(self, pipeline):
        doc = make_ready_document(["Page one content here. " * 5])
        chunks = pipeline.chunk_document(doc)

        for chunk in chunks:
            assert chunk.metadata.source_file == "fake.pdf"
            assert chunk.metadata.document_hash == "f" * 64

    def test_page_numbers_are_correct_across_multiple_pages(self, pipeline):
        doc = make_ready_document(
            ["First page content. " * 5, "Second page content. " * 5]
        )
        chunks = pipeline.chunk_document(doc)

        page_numbers = {c.metadata.page_number for c in chunks}
        assert page_numbers == {1, 2}

    def test_chunk_index_is_sequential(self, pipeline):
        doc = make_ready_document(
            ["Alpha content here. " * 5, "Beta content here. " * 5]
        )
        chunks = pipeline.chunk_document(doc)

        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_in_doc_is_backfilled_correctly(self, pipeline):
        doc = make_ready_document(["Some content. " * 20])
        chunks = pipeline.chunk_document(doc)

        for chunk in chunks:
            assert chunk.metadata.total_chunks_in_doc == len(chunks)

    def test_empty_pages_produce_no_chunks(self, pipeline):
        doc = make_ready_document(["", "   ", ""])
        chunks = pipeline.chunk_document(doc)
        assert chunks == []



class TestChunkFile:
    def test_chunk_real_pdf_file(self, pipeline, tmp_path):
        """Test chunking a generated PDF without external files."""
        import fitz

        pdf_path = tmp_path / "test_document.pdf"

        # Create a temporary PDF with extractable text
        with fitz.open() as document:
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "This is a sample PDF document for testing. "
                "It contains enough text to produce chunks. "
                "Retrieval augmented generation processes documents.",
                fontsize=11,
            )
            document.save(str(pdf_path))

        # Run the actual ingestion and chunking pipeline
        chunks = pipeline.chunk_file(str(pdf_path))

        assert len(chunks) > 0
        assert all(
            chunk.metadata.source_file == pdf_path.name
            for chunk in chunks
        )
        assert all(
            chunk.metadata.document_hash
            for chunk in chunks
        )

    def test_chunk_files_handles_invalid_gracefully(self, pipeline):
        results = pipeline.chunk_files(["README.md"])
        assert results["README.md"] == []
