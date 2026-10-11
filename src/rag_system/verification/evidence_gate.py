"""Deterministic gate for answers that have no retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_system.generation.grounded_generator import GeneratedAnswer
from rag_system.utils.exceptions import EvidenceGateError

_REFUSAL_PATTERNS = (
    re.compile(r"المعلومة غير متوفرة(?: في المستندات المتاحة)?", re.IGNORECASE),
    re.compile(r"المعلومات غير متوفرة(?: في المستندات المتاحة)?", re.IGNORECASE),
    re.compile(r"(?:المعلومة|المعلومات) غير متاحة(?: في المستندات المتاحة)?", re.IGNORECASE),
    re.compile(r"(?:the )?information(?: provided)?(?: in the (?:documents|excerpts))? is not available", re.IGNORECASE),
    re.compile(r"(?:the )?information(?: provided)?(?: in the (?:documents|excerpts))? does not (?:state|provide|mention) [^.!?]+", re.IGNORECASE),
    re.compile(r"(?:the )?information (?:is )?not available in the documents", re.IGNORECASE),
    re.compile(r"(?:the )?information (?:is )?not provided in the documents", re.IGNORECASE),
    re.compile(r"cannot be found in the documents", re.IGNORECASE),
    re.compile(r"not specified in the documents", re.IGNORECASE),
)
_CITATIONS = re.compile(r"\[\s*(?:مقتطف|excerpt)\s+[0-9٠-٩]+\s*\]", re.IGNORECASE)


def _is_abstention(answer: str) -> bool:
    """Only accept a complete refusal sentence, never a refusal plus a claim."""
    text = _CITATIONS.sub("", answer).strip().rstrip(" .。!؟")
    return any(pattern.fullmatch(text) for pattern in _REFUSAL_PATTERNS)



@dataclass(frozen=True)
class EvidenceGateResult:
    passed: bool
    has_evidence: bool
    is_abstention: bool
    reason: str | None = None


class EvidenceGate:
    """Reject substantive answers when retrieval returned no evidence."""

    def check(self, result: GeneratedAnswer) -> EvidenceGateResult:
        if not isinstance(result.answer, str):
            raise EvidenceGateError("Generated answer must be text")

        has_evidence = bool(result.chunks)
        answer = result.answer.strip()
        is_abstention = _is_abstention(answer)

        if has_evidence:
            passed = bool(answer)
            reason = None if passed else "The model returned an empty answer."
        elif is_abstention:
            passed = True
            reason = None
        else:
            passed = False
            reason = "The model answered without retrieved evidence instead of abstaining."

        return EvidenceGateResult(
            passed=passed,
            has_evidence=has_evidence,
            is_abstention=is_abstention,
            reason=reason,
        )
