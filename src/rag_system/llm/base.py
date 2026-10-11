"""Base interface for LLM clients (Phase 9 / Step 1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BaseLLM(ABC):
    """Common interface so different local LLM backends are interchangeable."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for `prompt`."""
        raise NotImplementedError