"""
Tests for Phase K — serving the built frontend from FastAPI.

Uses a fake, minimal `dist/` (not a real `npm run build` output) so
this test suite doesn't depend on Node/npm being available wherever
tests run — it only needs to verify `register_frontend()`'s routing
logic (SPA fallback, asset mounting, the api/ 404 guard), not that
Vite's build output looks a particular way.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.static as static_module
from api.static import register_frontend


@pytest.fixture()
def fake_dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>DocIntel AI</body></html>")
    (dist / "favicon.svg").write_text("<svg></svg>")
    (dist / "assets" / "index-abc123.js").write_text("console.log('app');")

    monkeypatch.setattr(static_module, "FRONTEND_DIST", dist)
    return dist


@pytest.fixture()
def app_with_frontend(fake_dist: Path) -> FastAPI:
    app = FastAPI()

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ok"}

    register_frontend(app)
    return app


class TestFrontendServing:
    def test_root_serves_index_html(self, app_with_frontend: FastAPI) -> None:
        client = TestClient(app_with_frontend)
        response = client.get("/")
        assert response.status_code == 200
        assert "DocIntel AI" in response.text
        assert response.headers["content-type"].startswith("text/html")

    def test_client_side_route_falls_back_to_index_html(self, app_with_frontend: FastAPI) -> None:
        """A hard refresh on /documents must still work — the SPA
        router, not the server, owns that path."""
        client = TestClient(app_with_frontend)
        response = client.get("/documents")
        assert response.status_code == 200
        assert "DocIntel AI" in response.text

    def test_deeply_nested_unknown_path_also_falls_back(self, app_with_frontend: FastAPI) -> None:
        client = TestClient(app_with_frontend)
        response = client.get("/chat/some/nested/thing")
        assert response.status_code == 200
        assert "DocIntel AI" in response.text

    def test_asset_is_served_directly(self, app_with_frontend: FastAPI) -> None:
        client = TestClient(app_with_frontend)
        response = client.get("/assets/index-abc123.js")
        assert response.status_code == 200
        assert "console.log" in response.text

    def test_favicon_is_served(self, app_with_frontend: FastAPI) -> None:
        client = TestClient(app_with_frontend)
        response = client.get("/favicon.svg")
        assert response.status_code == 200

    def test_real_api_route_is_not_shadowed_by_fallback(self, app_with_frontend: FastAPI) -> None:
        client = TestClient(app_with_frontend)
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_unmatched_api_path_404s_instead_of_returning_html(self, app_with_frontend: FastAPI) -> None:
        client = TestClient(app_with_frontend)
        response = client.get("/api/v1/this-route-does-not-exist")
        assert response.status_code == 404
        assert "text/html" not in response.headers["content-type"]


class TestNoFrontendBuilt:
    def test_register_frontend_is_a_no_op_without_dist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local API-only development (no `npm run build` yet) must not break."""
        monkeypatch.setattr(static_module, "FRONTEND_DIST", tmp_path / "nonexistent-dist")

        app = FastAPI()

        @app.get("/api/v1/health")
        def health() -> dict:
            return {"status": "ok"}

        register_frontend(app)  # should not raise

        client = TestClient(app)
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/").status_code == 404  # no SPA mounted, and that's correct here
