"""Unit tests for Phase 5 / Step 18: BM25 index."""

import pytest

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.retrieval.bm25_index import BM25Index
from rag_system.utils.exceptions import BM25Error


def make_chunk(chunk_index: int, text: str) -> Chunk:
    metadata = ChunkMetadata(
        source_file="test.pdf",
        document_hash="c" * 64,
        page_number=1,
        chunk_index=chunk_index,
    )
    return Chunk(text=text, metadata=metadata)


CHUNKS = [
    make_chunk(0, "the state general budget for the fiscal year"),
    make_chunk(1, "tax exemptions apply to research grants"),
    make_chunk(2, "customs regulations for imported goods"),
    make_chunk(3, "الموازنة العامة للدولة للسنة المالية"),
]


class TestBM25IndexBuild:
    def test_build_returns_count_indexed(self):
        index = BM25Index()
        count = index.build(CHUNKS)

        assert count == len(CHUNKS)
        assert len(index) == len(CHUNKS)
        assert index.is_built

    def test_build_on_empty_list_raises(self):
        index = BM25Index()
        with pytest.raises(BM25Error):
            index.build([])

    def test_build_replaces_previous_index(self):
        index = BM25Index()
        index.build(CHUNKS)
        smaller = CHUNKS[:2]

        index.build(smaller)

        assert len(index) == 2

    def test_clear_resets_index_and_search_results(self):
        index = BM25Index()
        index.build(CHUNKS)

        index.clear()

        assert len(index) == 0
        assert not index.is_built
        assert index.search("budget", top_k=5) == []


class TestBM25IndexSearch:
    def test_search_on_unbuilt_index_returns_empty_list(self):
        index = BM25Index()
        assert index.search("budget", top_k=5) == []

    def test_search_rejects_empty_query(self):
        index = BM25Index()
        index.build(CHUNKS)
        with pytest.raises(BM25Error):
            index.search("   ", top_k=5)

    def test_search_rejects_non_positive_top_k(self):
        index = BM25Index()
        index.build(CHUNKS)
        with pytest.raises(BM25Error):
            index.search("budget", top_k=0)

    def test_exact_keyword_match_ranks_first(self):
        index = BM25Index()
        index.build(CHUNKS)

        results = index.search("customs regulations imported goods", top_k=2)

        assert results[0].chunk_id == CHUNKS[2].chunk_id
        assert results[0].rank == 1
        assert results[0].source == "bm25"

    def test_search_respects_top_k(self):
        index = BM25Index()
        index.build(CHUNKS)

        results = index.search("budget fiscal year tax customs", top_k=2)

        assert len(results) == 2

    def test_arabic_query_matches_arabic_chunk(self):
        index = BM25Index()
        index.build(CHUNKS)

        results = index.search("الموازنة العامة", top_k=1)

        assert results[0].chunk_id == CHUNKS[3].chunk_id

    def test_arabic_alef_variant_still_matches(self):
        """A query using a hamza-carrying alef should still match text
        indexed with a bare alef, thanks to normalization."""
        index = BM25Index()
        index.build(CHUNKS)

        # "إلموازنة" is not real Arabic, so use a controlled pair instead:
        variant_chunks = [
            make_chunk(0, "أحمد ذهب الى المدرسة"),
            make_chunk(1, "unrelated english sentence about weather"),
        ]
        index.build(variant_chunks)

        results = index.search("احمد", top_k=1)  # bare alef query

        assert results[0].chunk_id == variant_chunks[0].chunk_id


class TestBM25IndexPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        index = BM25Index()
        index.build(CHUNKS)
        index.save(tmp_path)

        loaded = BM25Index.load(tmp_path)

        assert len(loaded) == len(CHUNKS)
        results = loaded.search("customs regulations imported goods", top_k=1)
        assert results[0].chunk_id == CHUNKS[2].chunk_id

    def test_save_unbuilt_index_raises(self, tmp_path):
        index = BM25Index()
        with pytest.raises(BM25Error):
            index.save(tmp_path)

    def test_exists_false_for_empty_directory(self, tmp_path):
        assert BM25Index.exists(tmp_path) is False

    def test_load_missing_index_raises(self, tmp_path):
        with pytest.raises(BM25Error):
            BM25Index.load(tmp_path)
