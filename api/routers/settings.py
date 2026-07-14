"""
routers/settings.py
====================
Persisted runtime preferences (chunk size/overlap, top-K, temperature,
max tokens, model) — thin wrapper around `RuntimeSettingsService`.

`PUT /settings` behaves like a partial update (`PATCH` semantics):
every field in `UpdatePreferencesRequest` defaults to `None`, meaning
"leave this preference unchanged", matching
`RuntimeSettingsService.update_preferences` exactly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_settings_service
from api.schemas.settings import (
    AvailableModelsResponse,
    RuntimePreferencesResponse,
    UpdatePreferencesRequest,
)
from src.services.settings_service import RuntimeSettingsService

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=RuntimePreferencesResponse)
def get_settings(
    settings_service: RuntimeSettingsService = Depends(get_settings_service),
) -> RuntimePreferencesResponse:
    return RuntimePreferencesResponse.from_preferences(settings_service.get_preferences())


@router.put("", response_model=RuntimePreferencesResponse)
def update_settings(
    body: UpdatePreferencesRequest,
    settings_service: RuntimeSettingsService = Depends(get_settings_service),
) -> RuntimePreferencesResponse:
    prefs = settings_service.update_preferences(**body.model_dump())
    return RuntimePreferencesResponse.from_preferences(prefs)


@router.post("/reset", response_model=RuntimePreferencesResponse)
def reset_settings(
    settings_service: RuntimeSettingsService = Depends(get_settings_service),
) -> RuntimePreferencesResponse:
    return RuntimePreferencesResponse.from_preferences(settings_service.reset_to_defaults())


@router.get("/models", response_model=AvailableModelsResponse)
def list_available_models() -> AvailableModelsResponse:
    return AvailableModelsResponse(models=RuntimeSettingsService.available_models())
