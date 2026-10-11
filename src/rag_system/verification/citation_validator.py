"""Validate that answer citations refer to retrieved excerpts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_system.generation.grounded_generator import GeneratedAnswer
from rag_system.utils.exceptions import CitationValidationError
from rag_system.verification.evidence_gate import EvidenceGate

_CITATION = re.compile(r"\[\s*(?:مقتطف|excerpt)\s+([0-9٠-٩]+)\s*\]", re.IGNORECASE)
_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass(frozen=True)
class CitationValidationResult:
    valid: bool
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...] = ()
    reason: str | None = None


class CitationValidator:
    """Check citation presence and range against the retrieved chunk list."""

    def validate(self, result: GeneratedAnswer) -> CitationValidationResult:
        if not isinstance(result.answer, str):
            raise CitationValidationError("Generated answer must be text")

        references = tuple(
            dict.fromkeys(int(match.translate(_DIGIT_TRANSLATION)) for match in _CITATION.findall(result.answer))
        )
        invalid = tuple(ref for ref in references if ref < 1 or ref > len(result.chunks))
        if invalid:
            return CitationValidationResult(
                valid=False,
                citations=references,
                invalid_citations=invalid,
                reason=f"Citation number(s) out of range: {list(invalid)}.",
            )

        if not references and not EvidenceGate().check(result).is_abstention:
            return CitationValidationResult(
                valid=False,
                citations=(),
                reason="A non-abstaining answer must cite at least one retrieved excerpt.",
            )

        return CitationValidationResult(valid=True, citations=references)
