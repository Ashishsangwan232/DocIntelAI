"""Tests for Phase D — the Settings API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_settings_service
from api.main import app
from src.database.sqlite_manager import SQLiteManager
from src.services.settings_service import RuntimeSettingsService


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def settings_service(db: SQLiteManager) -> RuntimeSettingsService:
    return RuntimeSettingsService(db=db)


@pytest.fixture()
def client(settings_service: RuntimeSettingsService) -> TestClient:
    app.dependency_overrides[get_settings_service] = lambda: settings_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestGetSettings:
    def test_returns_config_defaults_when_unset(self, client: TestClient) -> None:
        from config import settings as app_settings

        response = client.get("/api/v1/settings")
        assert response.status_code == 200
        body = response.json()
        assert body["chunk_size"] == app_settings.rag.chunk_size
        assert body["model"] == app_settings.llm.model_name


class TestUpdateSettings:
    def test_partial_update_only_changes_given_fields(self, client: TestClient) -> None:
        first = client.get("/api/v1/settings").json()

        response = client.put("/api/v1/settings", json={"top_k": 10})
        assert response.status_code == 200
        updated = response.json()
        assert updated["top_k"] == 10
        assert updated["chunk_size"] == first["chunk_size"]  # untouched

    def test_persists_across_requests(self, client: TestClient) -> None:
        client.put("/api/v1/settings", json={"temperature": 0.9})
        response = client.get("/api/v1/settings")
        assert response.json()["temperature"] == 0.9

    def test_invalid_chunk_overlap_returns_400(self, client: TestClient) -> None:
        # chunk_overlap >= chunk_size is invalid — validated dynamically
        # by the service (relative constraint, not a static Field bound).
        response = client.put("/api/v1/settings", json={"chunk_size": 200, "chunk_overlap": 200})
        assert response.status_code == 400

    def test_out_of_range_top_k_returns_422(self, client: TestClient) -> None:
        response = client.put("/api/v1/settings", json={"top_k": 500})
        assert response.status_code == 422


class TestResetSettings:
    def test_reset_reverts_to_config_defaults(self, client: TestClient) -> None:
        from config import settings as app_settings

        client.put("/api/v1/settings", json={"top_k": 15})
        response = client.post("/api/v1/settings/reset")
        assert response.status_code == 200
        assert response.json()["top_k"] == app_settings.rag.top_k

    def test_reset_restores_default_model_not_empty_string(self, client: TestClient) -> None:
        from config import settings as app_settings

        client.put("/api/v1/settings", json={"top_k": 15})
        response = client.post("/api/v1/settings/reset")
        assert response.json()["model"] == app_settings.llm.model_name


class TestAvailableModels:
    def test_lists_available_models(self, client: TestClient) -> None:
        response = client.get("/api/v1/settings/models")
        assert response.status_code == 200
        assert "gpt-oss:120b-cloud" in response.json()["models"]
