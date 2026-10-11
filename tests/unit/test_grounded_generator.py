"""Unit tests for the Phase 10 grounded generation flow."""

import pytest

from rag_system.generation.grounded_generator import GroundedGenerator
from rag_system.llm.base import BaseLLM, LLMResponse
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import PromptBuildError


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def retrieve(self, query, top_k=None):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class FakeLLM(BaseLLM):
    def __init__(self, text="grounded answer"):
        self.text = text
        self.calls = []

    def generate(self, prompt, temperature=None, max_tokens=None):
        self.calls.append((prompt, temperature, max_tokens))
        return LLMResponse(
            text=self.text,
            model="fake-model",
            latency_ms=12.5,
            prompt_tokens=100,
            completion_tokens=8,
        )


def _chunk(chunk_id="c1", text="evidence"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"source": "report.pdf"},
        score=0.9,
        rank=1,
        source="bm25",
    )


def test_generate_returns_answer_with_exact_retrieved_evidence():
    chunk = _chunk()
    retriever = FakeRetriever([chunk])
    llm = FakeLLM("  supported answer [مقتطف 1]  ")
    generator = GroundedGenerator(retriever, llm, top_k=3, temperature=0.2, max_tokens=64)

    result = generator.generate("What does the report say?")

    assert result.answer == "supported answer [مقتطف 1]"
    assert result.chunks == (chunk,)
    assert result.model == "fake-model"
    assert result.latency_ms == 12.5
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 8
    assert retriever.calls == [("What does the report say?", 3)]
    prompt, temperature, max_tokens = llm.calls[0]
    assert "evidence" in prompt
    assert temperature == 0.2
    assert max_tokens == 64


def test_generate_with_no_retrieved_chunks_still_asks_llm_to_abstain():
    retriever = FakeRetriever()
    llm = FakeLLM()
    result = GroundedGenerator(retriever, llm).generate("unknown question")

    assert result.chunks == ()
    assert "No excerpts available" in llm.calls[0][0]


def test_empty_query_fails_before_retrieval():
    retriever = FakeRetriever()
    generator = GroundedGenerator(retriever, FakeLLM())

    with pytest.raises(PromptBuildError):
        generator.generate("  ")

    assert retriever.calls == []


def test_non_positive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        GroundedGenerator(FakeRetriever(), FakeLLM(), top_k=0)



def test_follow_up_query_uses_last_user_turn_for_retrieval_but_keeps_current_prompt():
    retriever = FakeRetriever([_chunk()])
    llm = FakeLLM()
    generator = GroundedGenerator(retriever, llm)
    history = [
        {"role": "user", "content": "What was Egypt's GDP growth?"},
        {"role": "assistant", "content": "It grew by 4.3%."},
    ]

    result = generator.generate("And what about 2024?", conversation_history=history)

    expected_search = "What was Egypt's GDP growth?\nAnd what about 2024?"
    assert retriever.calls == [(expected_search, 5)]
    assert result.retrieval_query == expected_search
    assert "### Question:\nAnd what about 2024?" in llm.calls[0][0]
    assert "It grew by 4.3%." in llm.calls[0][0]


def test_standalone_query_does_not_mix_previous_turn_into_retrieval():
    retriever = FakeRetriever()
    generator = GroundedGenerator(retriever, FakeLLM())

    result = generator.generate(
        "What was GDP growth in 2024?",
        conversation_history=[{"role": "user", "content": "Who is the finance minister?"}],
    )

    assert retriever.calls == [("What was GDP growth in 2024?", 5)]
    assert result.retrieval_query == "What was GDP growth in 2024?"


def test_numeric_answer_gets_arabic_unit_only_from_matching_cited_evidence():
    chunk = _chunk(text="The budget uses were 5,541,394,196,000 pounds.")
    llm = FakeLLM("5,541,394,196,000 [مقتطف 1]")
    result = GroundedGenerator(FakeRetriever([chunk]), llm).generate("كم بلغت الاستخدامات؟")

    assert result.answer == "5,541,394,196,000 جنيه [مقتطف 1]"


def test_numeric_answer_gets_english_unit_only_from_matching_cited_evidence():
    chunk = _chunk(text="The budget uses were 5,541,394,196,000 pounds.")
    llm = FakeLLM("5,541,394,196,000 [Excerpt 1]")
    result = GroundedGenerator(FakeRetriever([chunk]), llm).generate("How much were the budget uses?")

    assert result.answer == "5,541,394,196,000 pounds [Excerpt 1]"


def test_existing_arabic_unit_is_not_duplicated_when_source_is_english():
    chunk = _chunk(text="The PMI rose to 50.7 point in January 2025.")
    llm = FakeLLM("50.7 نقطة [مقتطف 1]")
    result = GroundedGenerator(FakeRetriever([chunk]), llm).generate("ما مستوى PMI؟")

    assert result.answer == "50.7 نقطة [مقتطف 1]"


def test_existing_arabic_scale_unit_is_not_duplicated():
    chunk = _chunk(text="Net International Reserves rose to $48.14 billion in April 2025.")
    llm = FakeLLM("48.14 مليار دولار [مقتطف 1]")
    result = GroundedGenerator(FakeRetriever([chunk]), llm).generate("ما مستوى الاحتياطيات؟")

    assert result.answer == "48.14 مليار دولار [مقتطف 1]"
