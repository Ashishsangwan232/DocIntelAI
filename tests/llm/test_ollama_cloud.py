"""
Unit tests for src/llm/ollama_cloud.py

Uses `FakeOllamaChatClient` (see tests/conftest.py) instead of real
network calls — this environment cannot reach ollama.com, and even in
production, hitting a real LLM API on every test run would be slow,
flaky, and cost money. The fake preserves the exact `ChatClient`
structural contract the real `ollama.Client` satisfies, so all
message-building and response-parsing logic is genuinely exercised.
"""

from __future__ import annotations

import pytest

from src.llm.ollama_cloud import OllamaCloudLLM
from src.utils.exceptions import LLMAuthenticationError, LLMResponseError, LLMTimeoutError
from tests.conftest import FakeOllamaChatClient, FakeOllamaResponseError, make_stream_chunks


class TestGenerateHappyPath:
    def test_returns_parsed_response(self) -> None:
        client = FakeOllamaChatClient(
            response={
                "message": {"role": "assistant", "content": "The answer is 42."},
                "done": True,
                "prompt_eval_count": 100,
                "eval_count": 12,
            }
        )
        llm = OllamaCloudLLM(api_key="test-key", client=client)
        result = llm.generate("What is the answer?")

        assert result.content == "The answer is 42."
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 12
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0

    def test_sends_correct_payload(self) -> None:
        client = FakeOllamaChatClient(response={"message": {"content": "ok"}, "done": True})
        llm = OllamaCloudLLM(
            api_key="test-key", model_name="gpt-oss:120b-cloud", temperature=0.5, max_tokens=500,
            client=client,
        )
        llm.generate("hello", system_prompt="be concise")

        assert client.last_call["model"] == "gpt-oss:120b-cloud"
        assert client.last_call["messages"] == [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ]
        assert client.last_call["options"]["temperature"] == 0.5
        assert client.last_call["options"]["num_predict"] == 500
        assert client.last_call["stream"] is False

    def test_no_system_message_when_not_provided(self) -> None:
        client = FakeOllamaChatClient(response={"message": {"content": "ok"}, "done": True})
        llm = OllamaCloudLLM(api_key="k", client=client)
        llm.generate("hello")

        assert client.last_call["messages"] == [{"role": "user", "content": "hello"}]

    def test_per_call_overrides_take_precedence(self) -> None:
        client = FakeOllamaChatClient(response={"message": {"content": "ok"}, "done": True})
        llm = OllamaCloudLLM(api_key="k", temperature=0.3, max_tokens=100, client=client)
        llm.generate("hello", temperature=0.9, max_tokens=999)

        assert client.last_call["options"]["temperature"] == 0.9
        assert client.last_call["options"]["num_predict"] == 999

    def test_model_override_takes_precedence(self) -> None:
        client = FakeOllamaChatClient(response={"message": {"content": "ok"}, "done": True})
        llm = OllamaCloudLLM(api_key="k", model_name="default-model", client=client)
        llm.generate("hello", model="override-model")

        assert client.last_call["model"] == "override-model"


class TestGenerateErrors:
    def test_missing_api_key_raises_before_any_request(self) -> None:
        client = FakeOllamaChatClient(response={"message": {"content": "ok"}, "done": True})
        llm = OllamaCloudLLM(api_key="", client=client)
        with pytest.raises(LLMAuthenticationError):
            llm.generate("hello")
        assert client.call_count == 0

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_error_statuses_raise_authentication_error(self, status: int) -> None:
        client = FakeOllamaChatClient(raise_exception=FakeOllamaResponseError("denied", status))
        llm = OllamaCloudLLM(api_key="bad-key", client=client)
        with pytest.raises(LLMAuthenticationError):
            llm.generate("hello")

    def test_rate_limit_raises_response_error(self) -> None:
        client = FakeOllamaChatClient(raise_exception=FakeOllamaResponseError("slow down", 429))
        llm = OllamaCloudLLM(api_key="k", client=client)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")

    def test_server_error_raises_response_error(self) -> None:
        client = FakeOllamaChatClient(raise_exception=FakeOllamaResponseError("unavailable", 503))
        llm = OllamaCloudLLM(api_key="k", client=client)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")

    def test_unexpected_status_raises_response_error(self) -> None:
        """Reproduces the original bug report: a wrong endpoint/method
        shape returning 405 must surface as a clean LLMResponseError,
        not an unhandled exception."""
        client = FakeOllamaChatClient(raise_exception=FakeOllamaResponseError("Method Not Allowed", 405))
        llm = OllamaCloudLLM(api_key="k", client=client)
        with pytest.raises(LLMResponseError, match="405"):
            llm.generate("hello")

    def test_timeout_raises_llm_timeout_error(self) -> None:
        client = FakeOllamaChatClient(raise_exception=TimeoutError("timed out"))
        llm = OllamaCloudLLM(api_key="k", client=client)
        with pytest.raises(LLMTimeoutError):
            llm.generate("hello")

    def test_connection_failure_raises_response_error(self) -> None:
        client = FakeOllamaChatClient(raise_exception=ConnectionError("network unreachable"))
        llm = OllamaCloudLLM(api_key="k", client=client)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")

    def test_missing_message_key_raises_response_error(self) -> None:
        client = FakeOllamaChatClient(response={"done": True})
        llm = OllamaCloudLLM(api_key="k", client=client)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")


class TestStreaming:
    def test_yields_text_chunks_in_order(self) -> None:
        client = FakeOllamaChatClient(stream_chunks=make_stream_chunks("Hello", " world", "!"))
        llm = OllamaCloudLLM(api_key="k", client=client)
        chunks = list(llm.stream("hi"))
        assert chunks == ["Hello", " world", "!"]

    def test_stream_request_sets_stream_flag(self) -> None:
        client = FakeOllamaChatClient(stream_chunks=make_stream_chunks("a"))
        llm = OllamaCloudLLM(api_key="k", client=client)
        list(llm.stream("hi"))
        assert client.last_call["stream"] is True

    def test_stops_at_done_true(self) -> None:
        chunks_in = [
            {"message": {"content": "first"}, "done": False},
            {"message": {"content": "last"}, "done": True},
            {"message": {"content": "should_not_appear"}, "done": False},
        ]
        client = FakeOllamaChatClient(stream_chunks=chunks_in)
        llm = OllamaCloudLLM(api_key="k", client=client)
        chunks = list(llm.stream("hi"))
        assert chunks == ["first", "last"]

    def test_auth_error_before_streaming_begins(self) -> None:
        client = FakeOllamaChatClient(raise_exception=FakeOllamaResponseError("denied", 401))
        llm = OllamaCloudLLM(api_key="bad", client=client)
        with pytest.raises(LLMAuthenticationError):
            list(llm.stream("hi"))

    def test_skips_empty_content_chunks(self) -> None:
        chunks_in = [
            {"message": {"content": ""}, "done": False},
            {"message": {"content": "text"}, "done": True},
        ]
        client = FakeOllamaChatClient(stream_chunks=chunks_in)
        llm = OllamaCloudLLM(api_key="k", client=client)
        chunks = list(llm.stream("hi"))
        assert chunks == ["text"]


class TestConfiguration:
    def test_defaults_from_settings(self) -> None:
        llm = OllamaCloudLLM(client=FakeOllamaChatClient())
        assert llm.model_name
        assert llm.base_url

    def test_explicit_overrides(self) -> None:
        llm = OllamaCloudLLM(
            api_key="k", base_url="https://custom.example.com/",
            model_name="custom-model", client=FakeOllamaChatClient(),
        )
        assert llm.base_url == "https://custom.example.com"  # trailing slash stripped
        assert llm.model_name == "custom-model"

    def test_trailing_api_suffix_is_stripped(self) -> None:
        """Old `.env` files may still have `.../api` from the previous
        REST-based implementation — the SDK wants the bare origin."""
        llm = OllamaCloudLLM(
            api_key="k", base_url="https://ollama.com/api", client=FakeOllamaChatClient(),
        )
        assert llm.base_url == "https://ollama.com"
