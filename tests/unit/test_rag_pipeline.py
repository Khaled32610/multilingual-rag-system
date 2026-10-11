"""Unit tests for the Phase 12 end-to-end RAG orchestration."""

from dataclasses import dataclass

import pytest

from rag_system.generation.grounded_generator import GeneratedAnswer
from rag_system.pipeline.rag_pipeline import RAGPipeline
from rag_system.utils.exceptions import GenerationError, PipelineError


@dataclass
class FakeVerification:
    passed: bool


class FakeGenerator:
    def __init__(self, answer="verified answer", error=None):
        self.answer = answer
        self.error = error
        self.history = None

    def generate(self, query, conversation_history=()):
        self.history = list(conversation_history)
        if self.error:
            raise self.error
        return GeneratedAnswer(
            query=query,
            answer=self.answer,
            chunks=(),
            model="fake-model",
            latency_ms=1.0,
        )


class FakeVerifier:
    def __init__(self, passed=True):
        self.passed = passed
        self.received = None

    def verify(self, draft):
        self.received = draft
        return FakeVerification(self.passed)


def test_verified_answer_is_returned_as_user_facing_answer():
    generator = FakeGenerator()
    verifier = FakeVerifier(passed=True)

    response = RAGPipeline(generator, verifier).run("What is the answer?")

    assert response.answer == "verified answer"
    assert response.verified
    assert response.draft is verifier.received


def test_unverified_draft_is_not_exposed_as_final_answer_and_arabic_fallback_is_used():
    response = RAGPipeline(FakeGenerator("unsupported claim"), FakeVerifier(False)).run(
        "ما الإجابة؟"
    )

    assert response.answer == "تعذر التحقق من الإجابة من الأدلة المتاحة."
    assert response.draft.answer == "unsupported claim"
    assert not response.verified


def test_unverified_english_answer_uses_english_fallback():
    response = RAGPipeline(FakeGenerator("unsupported claim"), FakeVerifier(False)).run(
        "What is the answer?"
    )

    assert response.answer == "I couldn't verify the answer from the available evidence."


def test_generation_errors_are_wrapped_as_pipeline_errors():
    pipeline = RAGPipeline(
        FakeGenerator(error=GenerationError("model unavailable")), FakeVerifier()
    )

    with pytest.raises(PipelineError, match="RAG pipeline failed") as exc_info:
        pipeline.run("question")

    assert isinstance(exc_info.value.__cause__, GenerationError)


def test_pipeline_forwards_chat_history_to_generator():
    generator = FakeGenerator()
    pipeline = RAGPipeline(generator, FakeVerifier())
    history = [{"role": "user", "content": "previous question"}]

    pipeline.run("follow-up", conversation_history=history)

    assert generator.history == history
