"""
ollama_cloud.py
================
`BaseLLM` implementation targeting the Ollama Cloud API
(`/api/generate` endpoint, standard Ollama request/response shape).

The HTTP session is an injectable dependency (`session` constructor
param) rather than a hardcoded `requests` call inline — this is what
makes `generate()`/`stream()` unit-testable with a fake HTTP session
that returns canned responses, with zero real network calls, while
production code transparently uses a real `requests.Session`.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, Protocol

from config import settings
from src.llm.base import BaseLLM, LLMResponse
from src.utils.exceptions import LLMAuthenticationError, LLMResponseError, LLMTimeoutError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HTTPResponse(Protocol):
    """Structural type for whatever `session.post()` returns."""

    status_code: int
    text: str

    def json(self) -> dict[str, Any]: ...
    def iter_lines(self) -> Any: ...


class HTTPSession(Protocol):
    """
    Structural type describing the subset of `requests.Session` this
    class depends on. A real `requests.Session` satisfies this
    automatically; test doubles only need to implement `.post()`.
    """

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
        stream: bool = False,
    ) -> HTTPResponse: ...


class OllamaCloudLLM(BaseLLM):
    """LLM provider implementation calling the Ollama Cloud `/generate` API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        session: HTTPSession | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.llm.api_key
        self.base_url = (base_url or settings.llm.api_base_url).rstrip("/")
        self.model_name = model_name or settings.llm.model_name
        self.temperature = settings.llm.temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm.max_tokens
        self.timeout = timeout or settings.llm.request_timeout_seconds

        # Injected session bypasses lazy `requests` loading entirely —
        # used in tests and to keep the `requests` import out of the
        # module's top level (consistent with the lazy-import pattern
        # used for `sentence_transformers` in embedding_service.py).
        self._session: HTTPSession | None = session

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _get_session(self) -> HTTPSession:
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self._require_api_key()
        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, model, stream=False)

        start = time.monotonic()
        response = self._post(payload, stream=False)
        latency_ms = int((time.monotonic() - start) * 1000)

        self._raise_for_status(response)

        try:
            data = response.json()
            content = data["response"]
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("Malformed Ollama Cloud response: %s", exc)
            raise LLMResponseError(f"Malformed response from Ollama Cloud: {exc}") from exc

        logger.info(
            "LLM generate() completed in %dms (model=%s)", latency_ms, payload["model"]
        )
        return LLMResponse(
            content=content,
            model=payload["model"],
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason="stop" if data.get("done") else None,
        )

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        self._require_api_key()
        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, model, stream=True)

        response = self._post(payload, stream=True)
        self._raise_for_status(response)

        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                text = chunk.get("response", "")
                if text:
                    yield text
                if chunk.get("done"):
                    break
        except json.JSONDecodeError as exc:
            logger.error("Malformed streaming chunk from Ollama Cloud: %s", exc)
            raise LLMResponseError(f"Malformed streaming response: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_api_key(self) -> None:
        if not self.api_key:
            raise LLMAuthenticationError(
                "OLLAMA_CLOUD_API_KEY is not configured. Set it in your .env file."
            )

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        model: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _post(self, payload: dict[str, Any], stream: bool) -> HTTPResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            return self._get_session().post(
                f"{self.base_url}/generate",
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
        except TimeoutError as exc:
            logger.error("Ollama Cloud request timed out after %ds", self.timeout)
            raise LLMTimeoutError(
                f"Ollama Cloud did not respond within {self.timeout}s."
            ) from exc
        except Exception as exc:
            logger.error("Ollama Cloud request failed: %s", exc, exc_info=True)
            raise LLMResponseError(f"Failed to reach Ollama Cloud: {exc}") from exc

    @staticmethod
    def _raise_for_status(response: HTTPResponse) -> None:
        if response.status_code in (401, 403):
            raise LLMAuthenticationError(
                f"Ollama Cloud rejected the configured API key (status {response.status_code})."
            )
        if response.status_code == 429:
            raise LLMResponseError("Ollama Cloud rate limit exceeded. Please retry shortly.")
        if response.status_code >= 500:
            raise LLMResponseError(
                f"Ollama Cloud service error (status {response.status_code})."
            )
        if response.status_code != 200:
            raise LLMResponseError(
                f"Ollama Cloud returned unexpected status {response.status_code}: "
                f"{getattr(response, 'text', '')[:200]}"
            )
