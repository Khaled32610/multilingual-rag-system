"""Grounded answer generation (Phase 10).

This module composes retrieval, prompt construction, and an LLM call. It
returns the exact retrieved evidence alongside the answer so later phases
can validate citations and faithfulness without losing provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Protocol

from rag_system.generation.prompt_builder import build_rag_prompt
from rag_system.llm.base import BaseLLM
from rag_system.retrieval.models import RetrievedChunk
from rag_system.generation.numeric_grounding import add_source_unit_if_missing
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import PromptBuildError

_FOLLOW_UP_QUERY = re.compile(
    r"^(?:and\b|also\b|what about\b|how about\b|what was it\b|"
    r"what did it\b|where was it\b|when was it\b|"
    r"وماذا(?: عن)?(?:\s|$)|وماذا كان(?:\s|$)|ماذا عن(?:\s|$)|"
    r"وما(?:\s|$)|وكم(?:\s|$)|وهل(?:\s|$)|طيب(?:\s|$))",
    re.IGNORECASE,
)


class ChunkRetriever(Protocol):
    """Minimum retrieval interface required by the generation stage."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        ...


@dataclass(frozen=True)
class GeneratedAnswer:
    """Generated text with its input evidence and basic execution metadata."""

    query: str
    answer: str
    chunks: tuple[RetrievedChunk, ...]
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retrieval_query: str | None = None


class GroundedGenerator:
    """Generate answers using only evidence returned by the supplied retriever.

    The retriever and model are injected deliberately: callers can select a
    benchmarked retrieval strategy and an Ollama model without coupling this
    phase to a particular backend. Empty retrieval results are still sent to
    the prompt builder, which instructs the model to abstain.
    """

    def __init__(
        self,
        retriever: ChunkRetriever,
        llm: BaseLLM,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        top_k = 5 if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.temperature = settings.llm.temperature if temperature is None else temperature
        self.max_tokens = settings.llm.max_tokens if max_tokens is None else max_tokens

    def generate(
        self,
        query: str,
        conversation_history: Sequence[Mapping[str, str]] = (),
    ) -> GeneratedAnswer:
        """Retrieve evidence, build a grounded prompt, and generate an answer."""
        if not query or not query.strip():
            raise PromptBuildError("Cannot generate an answer for an empty query")
        history = tuple(conversation_history)
        retrieval_query = self._build_retrieval_query(query, history)
        chunks = self.retriever.retrieve(retrieval_query, top_k=self.top_k)
        prompt = build_rag_prompt(query, chunks, history)
        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = add_source_unit_if_missing(query, response.text.strip(), chunks)
        return GeneratedAnswer(
            query=query,
            answer=answer,
            chunks=tuple(chunks),
            model=response.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            retrieval_query=retrieval_query,
        )

    @staticmethod
    def _build_retrieval_query(
        query: str, history: Sequence[Mapping[str, str]]
    ) -> str:
        """Add the last user question to a follow-up query for retrieval context."""
        previous_user_query = next(
            (
                item.get("content", "").strip()
                for item in reversed(history)
                if item.get("role") == "user" and item.get("content", "").strip()
            ),
            "",
        )
        if not previous_user_query or not _FOLLOW_UP_QUERY.search(query.strip()):
            return query
        return f"{previous_user_query}\n{query}"
