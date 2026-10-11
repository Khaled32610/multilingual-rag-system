"""Run evidence, citation, and faithfulness checks for a generated answer."""

from __future__ import annotations

from dataclasses import dataclass

from rag_system.generation.grounded_generator import GeneratedAnswer
from rag_system.verification.citation_validator import CitationValidationResult, CitationValidator
from rag_system.verification.evidence_gate import EvidenceGate, EvidenceGateResult
from rag_system.verification.faithfulness_checker import FaithfulnessChecker, FaithfulnessResult
from rag_system.generation.numeric_grounding import unsupported_answer_numbers


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    evidence: EvidenceGateResult
    citations: CitationValidationResult
    faithfulness: FaithfulnessResult | None


class VerificationPipeline:
    """Fail closed unless the answer has evidence, valid citations, and support."""

    def __init__(
        self,
        faithfulness_checker: FaithfulnessChecker,
        evidence_gate: EvidenceGate | None = None,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self.evidence_gate = evidence_gate or EvidenceGate()
        self.citation_validator = citation_validator or CitationValidator()
        self.faithfulness_checker = faithfulness_checker

    def verify(self, result: GeneratedAnswer) -> VerificationResult:
        evidence = self.evidence_gate.check(result)
        citations = self.citation_validator.validate(result)

        if not evidence.passed or not citations.valid:
            return VerificationResult(False, evidence, citations, None)

        if evidence.is_abstention:
            faithfulness = FaithfulnessResult(faithful=True, unsupported_claims=())
            return VerificationResult(True, evidence, citations, faithfulness)

        unsupported_numbers = unsupported_answer_numbers(result, citations.citations)
        if unsupported_numbers:
            faithfulness = FaithfulnessResult(
                faithful=False,
                unsupported_claims=tuple(f"Unsupported number: {number}" for number in unsupported_numbers),
                explanation="Every answer number must appear in its cited evidence.",
            )
            return VerificationResult(False, evidence, citations, faithfulness)

        cited_chunks = [result.chunks[index - 1] for index in citations.citations]
        faithfulness = self.faithfulness_checker.check(
            result, cited_chunks, citation_numbers=list(citations.citations)
        )
        return VerificationResult(
            passed=faithfulness.faithful,
            evidence=evidence,
            citations=citations,
            faithfulness=faithfulness,
        )
