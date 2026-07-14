"""
ollama_cloud.py
================
`BaseLLM` implementation targeting Ollama Cloud, via the official
`ollama` Python SDK's chat client rather than hand-rolled REST calls.

Why the SDK instead of raw `requests`
--------------------------------------
An earlier version of this module POSTed directly to
`{base_url}/generate` with `requests`. That endpoint shape matches a
*local* Ollama server's legacy completion API, but Ollama Cloud only
accepts authenticated chat-style requests through `/api/chat`, with
the exact header/body handling the official SDK already implements.
Calling the wrong path/shape by hand surfaced as an opaque
`405 Method Not Allowed` with no useful detail. Delegating to
`ollama.Client(...).chat(...)` (the same call proven to work in
standalone scripts against Ollama Cloud) removes an entire class of
"reimplemented the wire protocol slightly wrong" bugs.

The SDK client is still an injectable dependency (`client` constructor
param), exactly like the `requests.Session` it replaces — this is
what keeps `generate()`/`stream()` unit-testable with a fake client
that returns canned responses, with zero real network calls, while
production code transparently builds a real `ollama.Client`.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Protocol

from config import settings
from src.llm.base import BaseLLM, LLMResponse
from src.utils.exceptions import LLMAuthenticationError, LLMResponseError, LLMTimeoutError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatClient(Protocol):
    """
    Structural type describing the subset of `ollama.Client` this class
    depends on. A real `ollama.Client(host=..., headers=...)` satisfies
    this automatically; test doubles only need to implement `.chat()`.
    """

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        stream: bool = False,
    ) -> Any: ...


class OllamaCloudLLM(BaseLLM):
    """LLM provider implementation calling Ollama Cloud via the `ollama` SDK's chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        client: ChatClient | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.llm.api_key
        # Host for the SDK is just the origin — the SDK itself appends
        # `/api/chat`. Strip an accidental trailing `/api` left over
        # from the old REST-based base URL so existing `.env` files
        # (`OLLAMA_CLOUD_BASE_URL=https://ollama.com/api`) keep working.
        raw_base_url = (base_url or settings.llm.api_base_url).rstrip("/")
        self.base_url = raw_base_url[: -len("/api")] if raw_base_url.endswith("/api") else raw_base_url
        self.model_name = model_name or settings.llm.model_name
        self.temperature = settings.llm.temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm.max_tokens
        self.timeout = timeout or settings.llm.request_timeout_seconds

        # Injected client bypasses lazy `ollama` loading entirely —
        # used in tests and to keep the `ollama` import out of the
        # module's top level (consistent with the lazy-import pattern
        # used for `sentence_transformers` in embedding_service.py).
        self._client: ChatClient | None = client

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------
    def _get_client(self) -> ChatClient:
        if self._client is None:
            import ollama

            self._client = ollama.Client(
                host=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

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
        messages = self._build_messages(prompt, system_prompt)
        model_name = model or self.model_name
        options = self._build_options(temperature, max_tokens)

        start = time.monotonic()
        response = self._chat(model_name, messages, options, stream=False)
        latency_ms = int((time.monotonic() - start) * 1000)

        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            logger.error("Malformed Ollama Cloud response: %s", exc)
            raise LLMResponseError(f"Malformed response from Ollama Cloud: {exc}") from exc

        logger.info("LLM generate() completed in %dms (model=%s)", latency_ms, model_name)
        return LLMResponse(
            content=content,
            model=model_name,
            latency_ms=latency_ms,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            finish_reason="stop" if response.get("done") else None,
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
        messages = self._build_messages(prompt, system_prompt)
        model_name = model or self.model_name
        options = self._build_options(temperature, max_tokens)

        # `ollama.Client.chat(..., stream=True)` returns a lazy generator:
        # the initial call rarely raises anything itself, and the real
        # HTTP request only fires once iteration begins. So unlike
        # `generate()`, error mapping has to wrap the *iteration*, not
        # just the call that produced the generator.
        chunks = self._chat(model_name, messages, options, stream=True)
        try:
            for chunk in chunks:
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
                if chunk.get("done"):
                    break
        except (LLMAuthenticationError, LLMResponseError, LLMTimeoutError):
            raise
        except Exception as exc:
            self._raise_mapped(exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_api_key(self) -> None:
        if not self.api_key:
            raise LLMAuthenticationError(
                "OLLAMA_CLOUD_API_KEY is not configured. Set it in your .env file."
            )

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_options(self, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": max_tokens or self.max_tokens,
        }

    def _chat(
        self, model: str, messages: list[dict[str, str]], options: dict[str, Any], stream: bool
    ) -> Any:
        try:
            return self._get_client().chat(
                model=model, messages=messages, options=options, stream=stream
            )
        except LLMAuthenticationError:
            raise
        except Exception as exc:
            self._raise_mapped(exc)

    @staticmethod
    def _raise_mapped(exc: Exception) -> None:
        """
        Map SDK-level exceptions onto this project's `LLMError` hierarchy.

        `ollama.ResponseError` carries a real HTTP status code from the
        server; anything else (connection refusal, DNS failure, or a
        timeout bubbling up from the underlying `httpx` client) is
        treated as a timeout if it names itself one, otherwise a
        generic response error.
        """
        status_code = getattr(exc, "status_code", None)
        if status_code in (401, 403):
            raise LLMAuthenticationError(
                f"Ollama Cloud rejected the configured API key (status {status_code})."
            ) from exc
        if status_code == 429:
            raise LLMResponseError("Ollama Cloud rate limit exceeded. Please retry shortly.") from exc
        if status_code is not None and status_code >= 500:
            raise LLMResponseError(f"Ollama Cloud service error (status {status_code}).") from exc
        if status_code is not None:
            raise LLMResponseError(
                f"Ollama Cloud returned unexpected status {status_code}: {str(exc)[:200]}"
            ) from exc

        exc_name = type(exc).__name__
        if "Timeout" in exc_name or isinstance(exc, TimeoutError):
            logger.error("Ollama Cloud request timed out")
            raise LLMTimeoutError("Ollama Cloud did not respond in time.") from exc

        logger.error("Ollama Cloud request failed: %s", exc, exc_info=True)
        raise LLMResponseError(f"Failed to reach Ollama Cloud: {exc}") from exc
