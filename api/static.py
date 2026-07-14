"""
static.py
=========
Serves the built React frontend (`frontend/dist/`) from this same
FastAPI process in production, so the deployed app is one Render
service and one origin — no CORS in production at all, matching
`frontend/vite.config.js`'s dev-time proxy already doing the
equivalent job locally.

This is a no-op in local development if `frontend/dist/` hasn't been
built: `uvicorn api.main:app` still serves `/api/v1/*` and `/docs`
exactly as before, since a developer running the API standalone is
almost always pointing a separate `npm run dev` Vite server at it
instead.

SPA fallback: any path that isn't `/api/v1/*`, `/docs`, `/openapi.json`,
or a real static asset returns `index.html` — this is what lets a
hand-rolled client-side router (see `frontend/src/router.js`) handle
a hard refresh or a direct link to e.g. `/documents` correctly,
rather than the server 404ing a path only the frontend router knows
about.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.utils.logger import get_logger

logger = get_logger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def register_frontend(app: FastAPI) -> None:
    """Mounts the built frontend, if present. Safe to call even when it isn't (local API-only dev)."""
    if not (FRONTEND_DIST / "index.html").exists():
        logger.info(
            "frontend/dist not found — skipping static frontend mount. "
            "Run `npm run build` in frontend/, or use `npm run dev` separately for local development."
        )
        return

    # Vite's asset hashes (dist/assets/*.js, *.css) are safe to mount
    # directly; dist/'s other top-level files (favicon.svg, index.html)
    # are served individually below rather than via a directory mount,
    # so the catch-all route stays the single source of truth for
    # "anything not an asset falls back to index.html".
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "favicon.svg")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        # A genuinely unmatched /api/v1/* path (a typo, a removed
        # endpoint) should 404 like any other REST API, not silently
        # return an HTML page — this guard is what makes that
        # distinction instead of every unknown path becoming the SPA.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(FRONTEND_DIST / "index.html")

    logger.info("Serving built frontend from %s", FRONTEND_DIST)
