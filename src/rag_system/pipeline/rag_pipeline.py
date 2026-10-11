"""End-to-end RAG orchestration for retrieval, generation, and verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_system.generation.grounded_generator import GeneratedAnswer, GroundedGenerator
from rag_system.retrieval.language_utils import detect_language
from rag_system.utils.exceptions import PipelineError, RAGSystemError
from rag_system.verification.pipeline import VerificationPipeline, VerificationResult


@dataclass(frozen=True)
class RAGResponse:
    """Safe user-facing answer plus its draft and verification trace."""

    query: str
    answer: str
    draft: GeneratedAnswer
    verification: VerificationResult

    @property
    def verified(self) -> bool:
        return self.verification.passed


class RAGPipeline:
    """Compose grounded generation with evidence verification.

    The final `answer` is the generated text only when every verification
    check passes. Otherwise it is a language-matched refusal; the original
    draft remains available as `draft` for evaluation and debugging.
    """

    def __init__(
        self,
        generator: GroundedGenerator,
        verifier: VerificationPipeline,
    ) -> None:
        self.generator = generator
        self.verifier = verifier

    def run(
        self,
        query: str,
        conversation_history: Sequence[Mapping[str, str]] = (),
    ) -> RAGResponse:
        """Generate and verify one answer, failing closed on unverifiable output."""
        try:
            if conversation_history:
                draft = self.generator.generate(
                    query, conversation_history=conversation_history
                )
            else:
                draft = self.generator.generate(query)
            verification = self.verifier.verify(draft)
        except RAGSystemError as exc:
            raise PipelineError(
                "The RAG pipeline failed.", details={"error": str(exc)}
            ) from exc
        except Exception as exc:
            raise PipelineError(
                "The RAG pipeline failed unexpectedly.",
                details={"error": str(exc), "type": type(exc).__name__},
            ) from exc

        answer = draft.answer if verification.passed else self._verification_fallback(query)
        return RAGResponse(
            query=query,
            answer=answer,
            draft=draft,
            verification=verification,
        )

    @staticmethod
    def _verification_fallback(query: str) -> str:
        if detect_language(query) == "ar":
            return "تعذر التحقق من الإجابة من الأدلة المتاحة."
        return "I couldn't verify the answer from the available evidence."
