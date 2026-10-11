"""Unit tests for Phase 2 / Step 2: RecursiveTextSplitter."""

import pytest

from rag_system.chunking.splitter import RecursiveTextSplitter, TokenCounter
from rag_system.utils.exceptions import ChunkingError


@pytest.fixture(scope="module")
def counter():
    return TokenCounter()


class TestTokenCounter:
    def test_empty_string_has_zero_tokens(self, counter):
        assert counter.count("") == 0

    def test_counts_tokens_for_simple_text(self, counter):
        assert counter.count("hello world") > 0

    def test_longer_text_has_more_tokens(self, counter):
        short = counter.count("hello")
        long = counter.count("hello " * 50)
        assert long > short


class TestRecursiveTextSplitter:
    def test_raises_when_overlap_exceeds_chunk_size(self):
        with pytest.raises(ChunkingError):
            RecursiveTextSplitter(chunk_size=100, chunk_overlap=200)

    def test_raises_when_overlap_equals_chunk_size(self):
        with pytest.raises(ChunkingError):
            RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)

    def test_empty_text_returns_empty_list(self, counter):
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10, token_counter=counter)
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_short_text_returns_single_chunk(self, counter):
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10, token_counter=counter)
        chunks = splitter.split("This is a short sentence.")
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self, counter):
        splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=5, token_counter=counter)
        text = "This is a sentence. " * 30
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_all_chunks_respect_size_limit_with_tolerance(self, counter):
        """
        Chunks should not massively exceed chunk_size. We allow overlap-related
        slack since overlap text is prepended after size-based splitting.
        """
        chunk_size = 40
        overlap = 8
        splitter = RecursiveTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, token_counter=counter
        )
        text = "This is a sentence about budgets and laws. " * 20
        chunks = splitter.split(text)

        for chunk in chunks:
            assert counter.count(chunk) <= chunk_size + overlap + 5

    def test_consecutive_chunks_share_overlap_content(self, counter):
        splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=8, token_counter=counter)
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau."
        chunks = splitter.split(text)

        assert len(chunks) >= 2
        # The start of chunk[1] should contain some word(s) from the end of chunk[0]
        last_word_of_first = chunks[0].strip().split()[-1]
        assert last_word_of_first in chunks[1]

    def test_no_empty_chunks_produced(self, counter):
        splitter = RecursiveTextSplitter(chunk_size=15, chunk_overlap=3, token_counter=counter)
        text = "\n\n\nSome text here.\n\n\n\nMore text after gaps.\n\n\n"
        chunks = splitter.split(text)
        assert all(c.strip() for c in chunks)