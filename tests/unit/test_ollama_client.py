"""Unit tests for OllamaLLMClient (Phase 9 / Step 1). No real server needed — requests is mocked."""

from unittest.mock import Mock, patch

import pytest
import requests

from rag_system.llm.ollama_client import OllamaLLMClient
from rag_system.utils.exceptions import LLMError


def _mock_response(json_data: dict, status_ok: bool = True) -> Mock:
    resp = Mock()
    resp.json.return_value = json_data
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("bad status")
    return resp


class TestOllamaLLMClient:
    def test_empty_prompt_raises(self):
        client = OllamaLLMClient(model_name="fake-model")
        with pytest.raises(LLMError):
            client.generate("   ")

    @patch("rag_system.llm.ollama_client.requests.post")
    def test_successful_generate(self, mock_post):
        mock_post.return_value = _mock_response(
            {"response": "hello world", "prompt_eval_count": 5, "eval_count": 2}
        )
        client = OllamaLLMClient(model_name="fake-model")
        result = client.generate("hi")

        assert result.text == "hello world"
        assert result.model == "fake-model"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2
        assert result.latency_ms >= 0

    @patch("rag_system.llm.ollama_client.requests.post")
    def test_connection_error_raises_llm_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        client = OllamaLLMClient(model_name="fake-model")
        with pytest.raises(LLMError, match="Is `ollama serve` running"):
            client.generate("hi")

    @patch("rag_system.llm.ollama_client.requests.post")
    def test_timeout_raises_llm_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("too slow")
        client = OllamaLLMClient(model_name="fake-model")
        with pytest.raises(LLMError, match="timed out"):
            client.generate("hi")

    @patch("rag_system.llm.ollama_client.requests.post")
    def test_ollama_error_field_raises(self, mock_post):
        mock_post.return_value = _mock_response({"error": "model not found"})
        client = OllamaLLMClient(model_name="fake-model")
        with pytest.raises(LLMError, match="returned an error"):
            client.generate("hi")

    @patch("rag_system.llm.ollama_client.requests.post")
    def test_passes_temperature_and_max_tokens(self, mock_post):
        mock_post.return_value = _mock_response({"response": "ok"})
        client = OllamaLLMClient(model_name="fake-model")
        client.generate("hi", temperature=0.7, max_tokens=100)

        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["options"]["temperature"] == 0.7
        assert sent_payload["options"]["num_predict"] == 100

    @patch("rag_system.llm.ollama_client.requests.get")
    def test_is_server_running_true(self, mock_get):
        mock_get.return_value = Mock()
        assert OllamaLLMClient.is_server_running() is True

    @patch("rag_system.llm.ollama_client.requests.get")
    def test_is_server_running_false(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        assert OllamaLLMClient.is_server_running() is False