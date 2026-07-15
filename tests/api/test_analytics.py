"""Tests for Phase D — the Analytics API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_analytics_service
from api.main import app
from src.database.models import Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.services.analytics_service import AnalyticsService


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def analytics_service(db: SQLiteManager) -> AnalyticsService:
    return AnalyticsService(db=db)  # chroma_manager omitted -> falls back to chunk count


@pytest.fixture()
def client(analytics_service: AnalyticsService) -> TestClient:
    app.dependency_overrides[get_analytics_service] = lambda: analytics_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestAnalyticsSummary:
    def test_empty_database_returns_zeroed_summary(self, client: TestClient) -> None:
        response = client.get("/api/v1/analytics/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["total_documents"] == 0
        assert body["average_response_time_ms"] is None
        assert body["average_response_time_display"] == "N/A"
        assert body["most_queried_documents"] == []

    def test_reflects_created_documents(self, client: TestClient, db: SQLiteManager) -> None:
        doc = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1000, file_hash="h1")
        db.create_document(doc)
        db.update_document_status(doc.id, DocumentStatus.READY)

        response = client.get("/api/v1/analytics/summary")
        body = response.json()
        assert body["total_documents"] == 1
        assert body["total_storage_bytes"] == 1000
        assert "MB" in body["total_storage_display"] or "KB" in body["total_storage_display"] or "B" in body["total_storage_display"]

    def test_respects_most_queried_limit_param(self, client: TestClient) -> None:
        response = client.get("/api/v1/analytics/summary", params={"most_queried_limit": 2})
        assert response.status_code == 200

    def test_out_of_range_limit_returns_422(self, client: TestClient) -> None:
        response = client.get("/api/v1/analytics/summary", params={"most_queried_limit": 500})
        assert response.status_code == 422
