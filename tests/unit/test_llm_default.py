"""Unit tests for the default configured LLM client factory."""

from types import SimpleNamespace

from rag_system.llm.default import get_default_llm_client


def test_factory_uses_configured_model_and_ollama_url(monkeypatch):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            default_model="gemma3:4b",
            ollama_base_url="http://ollama.local:11434",
        )
    )
    monkeypatch.setattr("rag_system.llm.default.get_settings", lambda: settings)

    client = get_default_llm_client()

    assert client.model_name == "gemma3:4b"
    assert client.base_url == "http://ollama.local:11434"
