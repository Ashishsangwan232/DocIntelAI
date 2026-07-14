"""
main.py
=======
FastAPI application entrypoint — the API-layer counterpart to `app.py`.

Run locally with:
    uvicorn api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs        (interactive Swagger UI)
    http://localhost:8000/api/v1/health

Every route this app will ever expose lives under `/api/v1`, added one
resource at a time in later migration phases (Documents, Chat, Search,
Analytics/Settings, Export). This file itself should almost never
change once those routers exist — it only wires cross-cutting concerns
(CORS, error handling, router registration), never business logic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.errors import register_exception_handlers
from api.routers import analytics, chat, documents, export, health, search, summary
from api.routers import settings as settings_router
from api.static import register_frontend
from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    for issue in settings.validate():
        logger.warning("Startup config warning: %s", issue)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app.app_name,
        description="AI-Powered Intelligent Document Analysis & Retrieval Platform — REST API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Dev-only in practice: in production this same FastAPI process
    # also serves the built React `dist/` as static files, so browser
    # requests are same-origin and CORS is never exercised there. Kept
    # active always (rather than env-gated) so pointing a local Vite
    # dev server at a prod-mode API "just works" too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(summary.router, prefix=API_PREFIX)
    app.include_router(chat.router, prefix=API_PREFIX)
    app.include_router(export.router, prefix=API_PREFIX)
    app.include_router(search.router, prefix=API_PREFIX)
    app.include_router(analytics.router, prefix=API_PREFIX)
    app.include_router(settings_router.router, prefix=API_PREFIX)

    # Registered last and deliberately: its catch-all route must never
    # get a chance to match before a real /api/v1/* route does.
    register_frontend(app)

    return app


app = create_app()
