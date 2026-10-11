"""
Ollama-backed LLM client (Phase 9 / Step 1).

Talks to a local Ollama server (default http://localhost:11434) over
its REST API. This keeps the whole system offline and local, as
required — no external API keys, no network calls beyond localhost.

Phase 9-B fix: `num_ctx` is now set explicitly on every request. Without
it Ollama uses its own default context window (often 2048-4096 tokens),
and a RAG prompt with 5-8 retrieved chunks is 3-6k tokens, so the START
of the prompt (the instructions and the first chunk) was silently
truncated. `prompt_tokens` from the response is checked against the
window and a warning is logged when the prompt is close to filling it.
"""

from __future__ import annotations

import time

import requests

from rag_system.llm.base import BaseLLM, LLMResponse
from rag_system.utils.exceptions import LLMError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_GENERATE_ENDPOINT = "/api/generate"
_DEFAULT_TIMEOUT_S = 300  # نماذج على CPU ممكن تاخد وقت طويل نسبياً
_DEFAULT_NUM_CTX = 8192
_TRUNCATION_WARN_RATIO = 0.95


class OllamaLLMClient(BaseLLM):
    """LLM client backed by a locally running Ollama server."""

    def __init__(
        self,
        model_name: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: int = _DEFAULT_TIMEOUT_S,
        num_ctx: int = _DEFAULT_NUM_CTX,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise LLMError("Cannot generate for an empty prompt")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}{_GENERATE_ENDPOINT}",
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise LLMError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Is `ollama serve` running?",
                details={"error": str(e)},
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMError(
                f"Ollama request timed out after {self.timeout_s}s "
                f"(model={self.model_name!r})",
                details={"error": str(e)},
            ) from e
        except requests.exceptions.RequestException as e:
            raise LLMError(
                f"Ollama request failed (model={self.model_name!r})",
                details={"error": str(e)},
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000

        try:
            data = response.json()
        except ValueError as e:
            raise LLMError(
                "Ollama returned a non-JSON response", details={"error": str(e)}
            ) from e

        if "error" in data:
            raise LLMError(
                f"Ollama returned an error for model={self.model_name!r}",
                details={"error": data["error"]},
            )

        text = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count")
        logger.info(
            f"Ollama generate: model={self.model_name!r} "
            f"prompt_len={len(prompt)} prompt_tokens={prompt_tokens} "
            f"-> {len(text)} chars ({latency_ms:.0f}ms)"
        )
        if prompt_tokens is not None and prompt_tokens >= self.num_ctx * _TRUNCATION_WARN_RATIO:
            logger.warning(
                f"Prompt used {prompt_tokens} tokens of num_ctx={self.num_ctx}: "
                f"it may have been truncated. Raise num_ctx or send fewer chunks."
            )

        return LLMResponse(
            text=text,
            model=self.model_name,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=data.get("eval_count"),
        )

    @staticmethod
    def is_server_running(base_url: str = _DEFAULT_BASE_URL) -> bool:
        """Quick check that a local Ollama server is reachable."""
        try:
            requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
            return True
        except requests.exceptions.RequestException:
            return False