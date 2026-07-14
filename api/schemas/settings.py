"""schemas/settings.py — request/response models for runtime preferences."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.services.settings_service import RuntimePreferences


class RuntimePreferencesResponse(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_tokens: int
    model: str

    @classmethod
    def from_preferences(cls, prefs: RuntimePreferences) -> "RuntimePreferencesResponse":
        return cls(
            chunk_size=prefs.chunk_size,
            chunk_overlap=prefs.chunk_overlap,
            top_k=prefs.top_k,
            temperature=prefs.temperature,
            max_tokens=prefs.max_tokens,
            model=prefs.model,
        )


class UpdatePreferencesRequest(BaseModel):
    """
    Every field is optional and defaults to `None`, meaning "leave
    unchanged" — matches `RuntimeSettingsService.update_preferences`'s
    partial-update semantics exactly, so `PUT` here behaves like a
    `PATCH` (only the fields sent are touched).
    """

    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=64, le=8192)
    model: str | None = None


class AvailableModelsResponse(BaseModel):
    models: list[str]
