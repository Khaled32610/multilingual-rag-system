"""Unit tests for Phase 11 evidence and answer verification."""

import pytest

from rag_system.generation.grounded_generator import GeneratedAnswer
from rag_system.llm.base import BaseLLM, LLMResponse
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import FaithfulnessCheckError
from rag_system.verification.citation_validator import CitationValidator
from rag_system.verification.evidence_gate import EvidenceGate
from rag_system.verification.faithfulness_checker import FaithfulnessChecker
from rag_system.verification.pipeline import VerificationPipeline


def _chunk(chunk_id="c1", text="Egypt's GDP grew by 4.3%."):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_file": "report.pdf"},
        score=0.9,
        rank=1,
        source="bm25",
    )


def _answer(text, chunks=None):
    return GeneratedAnswer(
        query="What was GDP growth?",
        answer=text,
        chunks=tuple(chunks or ()),
        model="fake-model",
        latency_ms=1.0,
    )


class FakeLLM(BaseLLM):
    def __init__(self, text='{"faithful": true, "unsupported_claims": []}'):
        self.text = text
        self.calls = []

    def generate(self, prompt, temperature=None, max_tokens=None):
        self.calls.append((prompt, temperature, max_tokens))
        return LLMResponse(self.text, "fake-judge", 1.0)


class BrokenLLM(BaseLLM):
    def generate(self, prompt, temperature=None, max_tokens=None):
        from rag_system.utils.exceptions import LLMError

        raise LLMError("offline")


def test_evidence_gate_accepts_empty_context_only_for_complete_refusal():
    gate = EvidenceGate()

    accepted = gate.check(_answer("المعلومة غير متوفرة في المستندات المتاحة."))
    rejected = gate.check(_answer("المعلومة غير متوفرة، لكن أظن أنها 4.3%."))

    assert accepted.passed and accepted.is_abstention
    assert not rejected.passed and not rejected.is_abstention


def test_evidence_gate_rejects_substantive_answer_without_context():
    result = EvidenceGate().check(_answer("GDP grew by 4.3%."))

    assert not result.passed
    assert "without retrieved evidence" in result.reason


def test_citation_validator_accepts_arabic_digits_and_english_label():
    chunks = [_chunk("a"), _chunk("b", "More evidence.")]
    result = CitationValidator().validate(_answer("4.3% [مقتطف ١] and [Excerpt 2].", chunks))

    assert result.valid
    assert result.citations == (1, 2)


def test_citation_validator_rejects_missing_or_out_of_range_citations():
    validator = CitationValidator()
    chunks = [_chunk()]

    assert not validator.validate(_answer("GDP grew by 4.3%.", chunks)).valid
    invalid = validator.validate(_answer("4.3% [مقتطف 3]", chunks))
    assert not invalid.valid
    assert invalid.invalid_citations == (3,)


def test_pipeline_checks_only_cited_evidence_and_passes_supported_answer():
    chunks = [_chunk("a"), _chunk("b", "The policy started in 2019.")]
    llm = FakeLLM()
    pipeline = VerificationPipeline(FaithfulnessChecker(llm))

    result = pipeline.verify(_answer("Policy started in 2019 [مقتطف 2].", chunks))

    assert result.passed
    assert result.faithfulness.faithful
    prompt, temperature, max_tokens = llm.calls[0]
    assert "[Excerpt 2]" in prompt
    assert "policy started in 2019" in prompt
    assert "Egypt's GDP grew" not in prompt
    assert temperature == 0.0
    assert max_tokens == 256


def test_pipeline_fails_closed_before_judge_for_no_evidence_or_bad_citation():
    llm = FakeLLM()
    pipeline = VerificationPipeline(FaithfulnessChecker(llm))

    refusal = pipeline.verify(_answer("المعلومة غير متوفرة في المستندات المتاحة."))
    bad_citation = pipeline.verify(_answer("4.3% [مقتطف 2]", [_chunk()]))

    assert refusal.passed
    assert refusal.faithfulness is not None and refusal.faithfulness.faithful
    assert not bad_citation.passed
    assert bad_citation.faithfulness is None
    assert llm.calls == []


def test_pipeline_rejects_unsupported_claim_from_judge():
    llm = FakeLLM('{"faithful": false, "unsupported_claims": ["GDP doubled"]}')
    pipeline = VerificationPipeline(FaithfulnessChecker(llm))

    result = pipeline.verify(_answer("GDP doubled [مقتطف 1].", [_chunk()]))

    assert not result.passed
    assert result.faithfulness.unsupported_claims == ("GDP doubled",)


def test_faithfulness_checker_rejects_malformed_output_and_llm_failure():
    chunk = _chunk()
    answer = _answer("GDP grew [مقتطف 1]", [chunk])

    with pytest.raises(FaithfulnessCheckError, match="no JSON"):
        FaithfulnessChecker(FakeLLM("not a verdict")).check(answer, [chunk])
    with pytest.raises(FaithfulnessCheckError, match="LLM faithfulness check failed"):
        FaithfulnessChecker(BrokenLLM()).check(answer, [chunk])


def test_pipeline_rejects_numbers_not_present_in_cited_excerpt():
    llm = FakeLLM()
    pipeline = VerificationPipeline(FaithfulnessChecker(llm))
    result = pipeline.verify(_answer("GDP grew by 4.8% [مقتطف 1].", [_chunk(text="GDP grew by 4.3%. ")]))

    assert not result.passed
    assert result.faithfulness.unsupported_claims == ("Unsupported number: 4.8",)
    assert llm.calls == []


def test_pipeline_accepts_arabic_number_and_separator_when_supported():
    llm = FakeLLM()
    pipeline = VerificationPipeline(FaithfulnessChecker(llm))
    result = pipeline.verify(
        _answer("بلغت القيمة ٤٬٣٠٠ جنيه [مقتطف 1].", [_chunk(text="The amount was EGP 4,300.")])
    )

    assert result.passed
    assert llm.calls
