"""
Tests for Phase A — the FastAPI foundation.

Uses FastAPI's `TestClient` (built on httpx), the same style of tool
`AppTest` plays for the Streamlit UI tests: a real app, real routing,
zero network calls, fast.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import register_exception_handlers, status_code_for
from api.main import app as real_app
from src.utils.exceptions import (
    DatabaseError,
    DocIntelError,
    DuplicateDocumentError,
    LLMAuthenticationError,
    LLMResponseError,
    LLMTimeoutError,
    RecordNotFoundError,
    UnsupportedFileTypeError,
    ValidationError,
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(real_app)


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "app" in body
        assert "environment" in body

    def test_health_does_not_require_any_service(self, client: TestClient) -> None:
        """Regression guard: the health route must never depend on
        EmbeddingService/ChromaManager/etc. — it should answer even if
        those are broken or unconfigured."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_config_check_reports_issues_list(self, client: TestClient) -> None:
        response = client.get("/api/v1/health/config")
        assert response.status_code == 200
        body = response.json()
        assert "ok" in body
        assert isinstance(body["issues"], list)

    def test_config_check_never_leaks_the_api_key_value(self, client: TestClient) -> None:
        from config import settings

        response = client.get("/api/v1/health/config")
        body = response.json()
        if settings.llm.api_key:
            for issue in body["issues"]:
                assert settings.llm.api_key not in issue


class TestOpenAPIDocs:
    def test_docs_page_is_served(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_is_valid_json(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"]
        assert "/api/v1/health" in schema["paths"]


class TestStatusCodeMapping:
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (RecordNotFoundError("not found"), 404),
            (DuplicateDocumentError("dup"), 409),
            (UnsupportedFileTypeError("bad type"), 400),
            (ValidationError("bad input"), 400),
            (LLMTimeoutError("timed out"), 504),
            (LLMAuthenticationError("bad key"), 502),
            (LLMResponseError("malformed"), 502),
            (DatabaseError("sqlite exploded"), 500),
            (DocIntelError("generic"), 500),
        ],
    )
    def test_maps_to_expected_status(self, exc: DocIntelError, expected_status: int) -> None:
        assert status_code_for(exc) == expected_status


class TestExceptionHandlerIntegration:
    """
    Builds a throwaway app with one route per exception type so the
    *handler wiring itself* (not just the pure mapping function above)
    is exercised end-to-end through real HTTP responses.
    """

    @pytest.fixture()
    def error_app_client(self) -> TestClient:
        test_app = FastAPI()
        register_exception_handlers(test_app)

        @test_app.get("/raise/not-found")
        def _raise_not_found():
            raise RecordNotFoundError("Document xyz not found")

        @test_app.get("/raise/validation")
        def _raise_validation():
            raise ValidationError("Bad file type")

        @test_app.get("/raise/llm-timeout")
        def _raise_llm_timeout():
            raise LLMTimeoutError("Ollama Cloud did not respond in time.")

        @test_app.get("/raise/unexpected")
        def _raise_unexpected():
            raise RuntimeError("something truly unplanned")

        return TestClient(test_app, raise_server_exceptions=False)

    def test_not_found_returns_404_with_error_envelope(self, error_app_client: TestClient) -> None:
        response = error_app_client.get("/raise/not-found")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["type"] == "RecordNotFoundError"
        assert "not found" in body["error"]["message"].lower()

    def test_validation_error_returns_400(self, error_app_client: TestClient) -> None:
        response = error_app_client.get("/raise/validation")
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "ValidationError"

    def test_llm_timeout_returns_504(self, error_app_client: TestClient) -> None:
        response = error_app_client.get("/raise/llm-timeout")
        assert response.status_code == 504

    def test_unexpected_exception_returns_500_without_leaking_details(
        self, error_app_client: TestClient
    ) -> None:
        response = error_app_client.get("/raise/unexpected")
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["type"] == "InternalServerError"
        # Never leak the real exception message/class for unexpected errors.
        assert "RuntimeError" not in body["error"]["message"]
        assert "unplanned" not in body["error"]["message"]


class TestCORS:
    def test_configured_dev_origin_is_allowed(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/health", headers={"Origin": "http://localhost:5173"}
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
