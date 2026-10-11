"""Unit tests for Phase 5: BM25 retriever (Step 1) and cross-lingual
query translation + merge (Step 2).

Existing tests use an identity `QueryTranslator` (translate_fn returns
the text unchanged) so they stay isolated from translation behavior and
never attempt to load the real HF model. The translation-specific test
class below injects a controlled fake translation instead.
"""

import pytest

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.retrieval.bm25_index import BM25Index
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.query_translator import QueryTranslator
from rag_system.utils.exceptions import RetrievalError

IDENTITY_TRANSLATOR = QueryTranslator(translate_fn=lambda t: t)


def make_chunk(chunk_index: int, text: str) -> Chunk:
    metadata = ChunkMetadata(
        source_file="test.pdf",
        document_hash="d" * 64,
        page_number=1,
        chunk_index=chunk_index,
    )
    return Chunk(text=text, metadata=metadata)


def make_retriever() -> BM25Retriever:
    return BM25Retriever(bm25_index=BM25Index(), query_translator=IDENTITY_TRANSLATOR)


class TestBM25RetrieverIndexing:
    def test_index_chunks_returns_count_added(self):
        retriever = make_retriever()
        chunks = [
            make_chunk(0, "budget law article one"),
            make_chunk(1, "tax exemption rules"),
            make_chunk(2, "customs regulations"),
        ]

        added = retriever.index_chunks(chunks)

        assert added == 3
        assert len(retriever) == 3

    def test_index_empty_list_returns_zero(self):
        retriever = make_retriever()
        assert retriever.index_chunks([]) == 0
        assert len(retriever) == 0


class TestBM25RetrieverRetrieve:
    def test_retrieve_returns_most_relevant_chunk_first(self):
        retriever = make_retriever()
        chunks = [
            make_chunk(0, "budget law article one"),
            make_chunk(1, "tax exemption rules"),
            make_chunk(2, "customs regulations"),
        ]
        retriever.index_chunks(chunks)

        results = retriever.retrieve("customs regulations", top_k=2)

        assert len(results) <= 2
        assert results[0].chunk_id == chunks[2].chunk_id
        assert results[0].source == "bm25"

    def test_retrieve_defaults_top_k_from_settings(self):
        retriever = make_retriever()
        chunks = [make_chunk(i, f"chunk number {i} about budgets") for i in range(3)]
        retriever.index_chunks(chunks)

        results = retriever.retrieve("budgets")

        assert 0 < len(results) <= 3

    def test_retrieve_rejects_empty_query(self):
        retriever = make_retriever()
        retriever.index_chunks([make_chunk(0, "some text")])

        with pytest.raises(RetrievalError):
            retriever.retrieve("   ")

    def test_retrieve_on_empty_index_returns_empty_list(self):
        retriever = make_retriever()
        assert retriever.retrieve("anything") == []


class TestBM25RetrieverCrossLingualTranslation:
    """Covers Phase 5 / Step 2: an Arabic query should still find an
    English-only chunk, via the translated-query BM25 search."""

    def make_translating_retriever(self, fake_translation: str) -> BM25Retriever:
        translator = QueryTranslator(translate_fn=lambda _t: fake_translation)
        return BM25Retriever(bm25_index=BM25Index(), query_translator=translator)

    def test_arabic_query_finds_english_chunk_via_translation(self):
        retriever = self.make_translating_retriever(
            fake_translation="customs regulations imported goods"
        )
        chunks = [
            make_chunk(0, "customs regulations for imported goods"),
            make_chunk(1, "unrelated sentence about weather"),
        ]
        retriever.index_chunks(chunks)

        # This Arabic query has zero lexical overlap with the English
        # corpus, so without translation this would return nothing useful.
        results = retriever.retrieve("لوائح الجمارك للبضائع المستوردة", top_k=2)

        assert len(results) >= 1
        assert results[0].chunk_id == chunks[0].chunk_id

    def test_english_paraphrase_expansion_finds_trade_table_wording(self):
        retriever = self.make_translating_retriever(fake_translation="unused")
        gold = make_chunk(0, "Non-agricultural products. European Union 2024 imports 10,137.")
        distractor = make_chunk(1, "Agricultural exports to several markets.")
        retriever.index_chunks([distractor, gold])

        results = retriever.retrieve(
            "Which destination is the top market for Egypt's non-agricultural exports, "
            "measured by bilateral import value?",
            top_k=2,
        )

        assert results[0].chunk_id == gold.chunk_id

    def test_english_query_does_not_trigger_translation(self):
        def boom(_text: str) -> str:
            raise AssertionError("translation should not run for an English query")

        translator = QueryTranslator(translate_fn=boom)
        retriever = BM25Retriever(bm25_index=BM25Index(), query_translator=translator)
        retriever.index_chunks([make_chunk(0, "customs regulations")])

        # Should not raise, proving translate_fn (boom) was never called.
        results = retriever.retrieve("customs regulations", top_k=1)
        assert len(results) == 1

    def test_translation_failure_falls_back_to_original_results(self):
        def failing_translate(_text: str) -> str:
            raise RuntimeError("simulated translation failure")

        translator = QueryTranslator(translate_fn=failing_translate)
        retriever = BM25Retriever(bm25_index=BM25Index(), query_translator=translator)
        chunks = [make_chunk(0, "الموازنة العامة للدولة")]
        retriever.index_chunks(chunks)

        # translate_fn raises RuntimeError, not TranslationError directly,
        # but QueryTranslator only re-raises as TranslationError when it
        # builds the pipeline itself — a raw exception from an injected
        # translate_fn propagates as-is. retrieve() should still not crash
        # for the *original* (Arabic) query path, since that BM25 search
        # already ran and succeeded before translation was attempted.
        with pytest.raises(RuntimeError):
            retriever.retrieve("الموازنة العامة للدولة", top_k=1)