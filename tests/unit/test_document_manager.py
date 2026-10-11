"""Unit tests for Phase 14 local document management."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.ingestion.hasher import PDFHasher
from rag_system.utils.document_manager import DocumentManager
from rag_system.utils.exceptions import DocumentManagementError, IndexingError


class FakeChunker:
    def chunk_file(self, path):
        path = Path(path)
        raw = path.read_bytes().decode("utf-8")
        digest = PDFHasher().compute_hash(path)
        text = raw.split(":", 1)[1]
        return [
            Chunk(
                text=text,
                metadata=ChunkMetadata(
                    source_file=path.name,
                    document_hash=digest,
                    page_number=1,
                    chunk_index=0,
                ),
            )
        ]


class FakeVectorStore:
    def __init__(self):
        self.chunks = {}

    def delete_by_chunk_ids(self, chunk_ids):
        removed = 0
        for chunk_id in chunk_ids:
            if self.chunks.pop(chunk_id, None) is not None:
                removed += 1
        return removed


class FakeVectorRetriever:
    def __init__(self):
        self.vector_store = FakeVectorStore()

    def index_chunks(self, chunks):
        for chunk in chunks:
            self.vector_store.chunks[chunk.chunk_id] = chunk
        return len(chunks)


class FakeBM25Index:
    def clear(self):
        self.records = []


class FakeBM25Retriever:
    def __init__(self):
        self.bm25_index = FakeBM25Index()
        self.bm25_index.records = []
        self.fail_next = False

    def index_chunks(self, chunks):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated BM25 failure")
        self.bm25_index.records = list(chunks)
        return len(chunks)


class FakeHybridRetriever:
    def __init__(self):
        self.vector_retriever = FakeVectorRetriever()
        self.bm25_retriever = FakeBM25Retriever()

    def save(self):
        pass


def _manager(tmp_path):
    hybrid = FakeHybridRetriever()
    manager = DocumentManager(
        chunker=FakeChunker(),
        retriever=hybrid,
        raw_dir=tmp_path / "raw",
        catalog_path=tmp_path / "catalog.json",
    )
    return manager, hybrid


def _pdf(path, marker, content):
    # FakeChunker consumes plain fixtures; manager focuses on indexing orchestration.
    path.write_text(f"{marker}:{content}", encoding="utf-8")
    return path


def test_add_indexes_chunks_and_survives_manager_restart(tmp_path):
    manager, hybrid = _manager(tmp_path)
    source = _pdf(tmp_path / "first.pdf", "digest-a", "budget revenues")

    result = manager.add_pdf(source)

    assert result["status"] == "indexed"
    assert result["chunk_count"] == 1
    assert len(hybrid.vector_retriever.vector_store.chunks) == 1
    assert len(hybrid.bm25_retriever.bm25_index.records) == 1
    restarted = DocumentManager(
        chunker=FakeChunker(),
        retriever=hybrid,
        raw_dir=tmp_path / "raw",
        catalog_path=tmp_path / "catalog.json",
    )
    assert restarted.list_documents()[0]["file_name"] == "first.pdf"


def test_duplicate_content_is_skipped_even_with_a_different_name(tmp_path):
    manager, hybrid = _manager(tmp_path)
    manager.add_pdf(_pdf(tmp_path / "first.pdf", "same-digest", "same content"))

    result = manager.add_pdf(_pdf(tmp_path / "copy.pdf", "different-pdf-bytes", "same content"))

    assert result["status"] == "already_indexed"
    assert result["file_name"] == "first.pdf"
    assert len(hybrid.vector_retriever.vector_store.chunks) == 1
    assert not (tmp_path / "raw" / "copy.pdf").exists()


def test_replacing_same_name_replaces_old_chunks(tmp_path):
    manager, hybrid = _manager(tmp_path)
    source = _pdf(tmp_path / "doc.pdf", "old-hash", "old contents")
    manager.add_pdf(source)
    _pdf(source, "new-hash", "new contents")

    result = manager.add_pdf(source)

    assert result["status"] == "replaced"
    assert result["document_hash"] != manager.hasher.compute_hash(source)
    assert [c.text for c in hybrid.vector_retriever.vector_store.chunks.values()] == ["new contents"]
    assert [c.text for c in hybrid.bm25_retriever.bm25_index.records] == ["new contents"]


def test_remove_document_clears_both_indexes_and_catalog(tmp_path):
    manager, hybrid = _manager(tmp_path)
    manager.add_pdf(_pdf(tmp_path / "doc.pdf", "hash", "content"))

    assert manager.remove_document("doc.pdf")
    assert not manager.remove_document("doc.pdf")
    assert manager.list_documents() == []
    assert hybrid.vector_retriever.vector_store.chunks == {}
    assert hybrid.bm25_retriever.bm25_index.records == []
    assert not (tmp_path / "raw" / "doc.pdf").exists()


def test_index_failure_rolls_back_previous_document_state(tmp_path):
    manager, hybrid = _manager(tmp_path)
    source = _pdf(tmp_path / "doc.pdf", "stable-hash", "old contents")
    manager.add_pdf(source)
    original_digest = manager.list_documents()[0]["document_hash"]
    _pdf(source, "replacement-hash", "new contents")
    hybrid.bm25_retriever.fail_next = True

    with pytest.raises(IndexingError, match="previous index restored"):
        manager.add_pdf(source)

    assert manager.list_documents()[0]["document_hash"] == original_digest
    assert [c.text for c in hybrid.vector_retriever.vector_store.chunks.values()] == ["old contents"]
    assert [c.text for c in hybrid.bm25_retriever.bm25_index.records] == ["old contents"]
    assert (tmp_path / "raw" / "doc.pdf").read_text().startswith("stable-hash:")


def test_unsupported_extension_and_missing_file_are_rejected(tmp_path, monkeypatch):
    manager, _ = _manager(tmp_path)
    monkeypatch.setattr(
        "rag_system.utils.document_manager.get_settings",
        lambda: SimpleNamespace(pdf=SimpleNamespace(allowed_extensions=[".pdf"], max_file_size_mb=5)),
    )
    with pytest.raises(DocumentManagementError, match="does not exist"):
        manager.add_pdf(tmp_path / "missing.pdf")
    invalid = tmp_path / "notes.txt"
    invalid.write_text("not a pdf")
    with pytest.raises(DocumentManagementError, match="Unsupported document type"):
        manager.add_pdf(invalid)


def test_bootstraps_catalog_from_existing_bm25_records(tmp_path):
    hybrid = FakeHybridRetriever()
    existing = _pdf(tmp_path / "legacy.pdf", "legacy", "already indexed")
    legacy_chunk = FakeChunker().chunk_file(str(existing))[0]
    hybrid.bm25_retriever.bm25_index.records = [legacy_chunk.to_dict()]

    manager = DocumentManager(
        chunker=FakeChunker(),
        retriever=hybrid,
        raw_dir=tmp_path / "raw",
        catalog_path=tmp_path / "catalog.json",
    )

    assert manager.list_documents()[0]["file_name"] == "legacy.pdf"
    assert manager.list_documents()[0]["chunk_count"] == 1
    manager.add_pdf(_pdf(tmp_path / "new.pdf", "new", "new content"))
    assert len(hybrid.bm25_retriever.bm25_index.records) == 2


def test_remove_rejects_path_traversal_even_when_name_is_not_indexed(tmp_path):
    manager, _ = _manager(tmp_path)

    with pytest.raises(DocumentManagementError, match="plain file name"):
        manager.remove_document("../outside.pdf")
