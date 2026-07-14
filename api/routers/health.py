"""
routers/health.py
==================
Liveness/readiness endpoints. No service dependencies are touched
here on purpose — this route must answer even if the embedding model
hasn't loaded yet or Ollama Cloud is unreachable, so uptime checks
(and a human hitting `/api/v1/health` first) get a fast, honest
signal about the process itself versus its dependencies.
"""

from __future__ import annotations

from fastapi import APIRouter

from config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> dict:
    """Process liveness — always 200 if the API process is up and routing requests."""
    return {
        "status": "ok",
        "app": settings.app.app_name,
        "environment": settings.app.app_env,
    }


@router.get("/config")
def config_check() -> dict:
    """
    Surfaces the same non-secret configuration warnings the Streamlit
    sidebar shows today (`Settings.validate()`) — e.g. a missing
    `OLLAMA_CLOUD_API_KEY` — without ever exposing the key itself.
    """
    issues = settings.validate()
    return {"ok": not issues, "issues": issues}
