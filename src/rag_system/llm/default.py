"""Factory for the project's configured local LLM client."""

from __future__ import annotations

from rag_system.llm.ollama_client import OllamaLLMClient
from rag_system.utils.config import get_settings


def get_default_llm_client() -> OllamaLLMClient:
    """Create an Ollama client using the selected model and configured URL."""
    llm_settings = get_settings().llm
    return OllamaLLMClient(
        model_name=llm_settings.default_model,
        base_url=llm_settings.ollama_base_url,
    )
