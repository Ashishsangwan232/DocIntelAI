"""
Unit tests for src/llm/ollama_cloud.py

Uses `FakeHTTPSession`/`FakeHTTPResponse` (see tests/conftest.py)
instead of real network calls — this environment cannot reach
ollama.com, and even in production, hitting a real LLM API on every
test run would be slow, flaky, and cost money. The fakes preserve the
exact `HTTPSession`/`HTTPResponse` structural contract the real
`requests.Session` satisfies, so all request-building and
response-parsing logic is genuinely exercised.
"""

from __future__ import annotations

import pytest

from src.llm.ollama_cloud import OllamaCloudLLM
from src.utils.exceptions import LLMAuthenticationError, LLMResponseError, LLMTimeoutError
from tests.conftest import FakeHTTPResponse, FakeHTTPSession, make_stream_lines


class TestGenerateHappyPath:
    def test_returns_parsed_response(self) -> None:
        session = FakeHTTPSession(
            response=FakeHTTPResponse(
                json_data={
                    "response": "The answer is 42.",
                    "done": True,
                    "prompt_eval_count": 100,
                    "eval_count": 12,
                }
            )
        )
        llm = OllamaCloudLLM(api_key="test-key", session=session)
        result = llm.generate("What is the answer?")

        assert result.content == "The answer is 42."
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 12
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0

    def test_sends_correct_payload(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(json_data={"response": "ok", "done": True}))
        llm = OllamaCloudLLM(
            api_key="test-key", model_name="gpt-oss:120b", temperature=0.5, max_tokens=500,
            session=session,
        )
        llm.generate("hello", system_prompt="be concise")

        assert session.last_call["json"]["model"] == "gpt-oss:120b"
        assert session.last_call["json"]["prompt"] == "hello"
        assert session.last_call["json"]["system"] == "be concise"
        assert session.last_call["json"]["options"]["temperature"] == 0.5
        assert session.last_call["json"]["options"]["num_predict"] == 500
        assert session.last_call["json"]["stream"] is False
        assert session.last_call["headers"]["Authorization"] == "Bearer test-key"

    def test_per_call_overrides_take_precedence(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(json_data={"response": "ok", "done": True}))
        llm = OllamaCloudLLM(api_key="k", temperature=0.3, max_tokens=100, session=session)
        llm.generate("hello", temperature=0.9, max_tokens=999)

        assert session.last_call["json"]["options"]["temperature"] == 0.9
        assert session.last_call["json"]["options"]["num_predict"] == 999


class TestGenerateErrors:
    def test_missing_api_key_raises_before_any_request(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(json_data={"response": "ok", "done": True}))
        llm = OllamaCloudLLM(api_key="", session=session)
        with pytest.raises(LLMAuthenticationError):
            llm.generate("hello")
        assert session.call_count == 0

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_error_statuses_raise_authentication_error(self, status: int) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(status_code=status, text="denied"))
        llm = OllamaCloudLLM(api_key="bad-key", session=session)
        with pytest.raises(LLMAuthenticationError):
            llm.generate("hello")

    def test_rate_limit_raises_response_error(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(status_code=429, text="slow down"))
        llm = OllamaCloudLLM(api_key="k", session=session)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")

    def test_server_error_raises_response_error(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(status_code=503, text="unavailable"))
        llm = OllamaCloudLLM(api_key="k", session=session)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")

    def test_timeout_raises_llm_timeout_error(self) -> None:
        session = FakeHTTPSession(raise_exception=TimeoutError("timed out"))
        llm = OllamaCloudLLM(api_key="k", session=session)
        with pytest.raises(LLMTimeoutError):
            llm.generate("hello")

    def test_connection_failure_raises_response_error(self) -> None:
        session = FakeHTTPSession(raise_exception=ConnectionError("network unreachable"))
        llm = OllamaCloudLLM(api_key="k", session=session)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")

    def test_missing_response_key_raises_response_error(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(json_data={"done": True}))
        llm = OllamaCloudLLM(api_key="k", session=session)
        with pytest.raises(LLMResponseError):
            llm.generate("hello")


class TestStreaming:
    def test_yields_text_chunks_in_order(self) -> None:
        session = FakeHTTPSession(
            response=FakeHTTPResponse(lines=make_stream_lines("Hello", " world", "!"))
        )
        llm = OllamaCloudLLM(api_key="k", session=session)
        chunks = list(llm.stream("hi"))
        assert chunks == ["Hello", " world", "!"]

    def test_stream_request_sets_stream_flag(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(lines=make_stream_lines("a")))
        llm = OllamaCloudLLM(api_key="k", session=session)
        list(llm.stream("hi"))
        assert session.last_call["stream"] is True
        assert session.last_call["json"]["stream"] is True

    def test_stops_at_done_true(self) -> None:
        import json as json_module

        lines = [
            json_module.dumps({"response": "first", "done": False}).encode(),
            json_module.dumps({"response": "last", "done": True}).encode(),
            json_module.dumps({"response": "should_not_appear", "done": False}).encode(),
        ]
        session = FakeHTTPSession(response=FakeHTTPResponse(lines=lines))
        llm = OllamaCloudLLM(api_key="k", session=session)
        chunks = list(llm.stream("hi"))
        assert chunks == ["first", "last"]

    def test_auth_error_before_streaming_begins(self) -> None:
        session = FakeHTTPSession(response=FakeHTTPResponse(status_code=401))
        llm = OllamaCloudLLM(api_key="bad", session=session)
        with pytest.raises(LLMAuthenticationError):
            list(llm.stream("hi"))

    def test_skips_blank_lines(self) -> None:
        import json as json_module

        lines = [
            b"",
            json_module.dumps({"response": "text", "done": True}).encode(),
        ]
        session = FakeHTTPSession(response=FakeHTTPResponse(lines=lines))
        llm = OllamaCloudLLM(api_key="k", session=session)
        chunks = list(llm.stream("hi"))
        assert chunks == ["text"]


class TestConfiguration:
    def test_defaults_from_settings(self) -> None:
        llm = OllamaCloudLLM(session=FakeHTTPSession())
        assert llm.model_name
        assert llm.base_url

    def test_explicit_overrides(self) -> None:
        llm = OllamaCloudLLM(
            api_key="k", base_url="https://custom.example.com/api/",
            model_name="custom-model", session=FakeHTTPSession(),
        )
        assert llm.base_url == "https://custom.example.com/api"  # trailing slash stripped
        assert llm.model_name == "custom-model"
