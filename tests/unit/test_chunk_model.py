"""Unit tests for Phase 2 / Step 1: Chunk data model."""

from rag_system.chunking.models import Chunk, ChunkMetadata


def make_metadata(**overrides) -> ChunkMetadata:
    defaults = dict(
        source_file="test.pdf",
        document_hash="a" * 64,
        page_number=1,
        chunk_index=0,
    )
    defaults.update(overrides)
    return ChunkMetadata(**defaults)


class TestChunkMetadata:
    def test_to_dict_contains_all_fields(self):
        meta = make_metadata()
        d = meta.to_dict()
        assert d["source_file"] == "test.pdf"
        assert d["document_hash"] == "a" * 64
        assert d["page_number"] == 1
        assert d["chunk_index"] == 0
        assert "created_at" in d


class TestChunk:
    def test_chunk_id_is_generated_automatically(self):
        chunk = Chunk(text="hello world", metadata=make_metadata())
        assert chunk.chunk_id != ""
        assert len(chunk.chunk_id) == 24

    def test_chunk_id_is_deterministic(self):
        meta1 = make_metadata(document_hash="x" * 64, chunk_index=5)
        meta2 = make_metadata(document_hash="x" * 64, chunk_index=5, source_file="other.pdf")

        chunk1 = Chunk(text="text A", metadata=meta1)
        chunk2 = Chunk(text="text B", metadata=meta2)

        assert chunk1.chunk_id == chunk2.chunk_id

    def test_different_chunk_index_gives_different_id(self):
        meta1 = make_metadata(document_hash="y" * 64, chunk_index=0)
        meta2 = make_metadata(document_hash="y" * 64, chunk_index=1)

        chunk1 = Chunk(text="text", metadata=meta1)
        chunk2 = Chunk(text="text", metadata=meta2)

        assert chunk1.chunk_id != chunk2.chunk_id

    def test_char_count_is_set_automatically(self):
        chunk = Chunk(text="12345", metadata=make_metadata())
        assert chunk.metadata.char_count == 5

    def test_to_dict_structure(self):
        chunk = Chunk(text="sample text", metadata=make_metadata())
        d = chunk.to_dict()
        assert set(d.keys()) == {"chunk_id", "text", "metadata"}
        assert d["text"] == "sample text"