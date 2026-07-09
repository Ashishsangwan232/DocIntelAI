"""
settings_service.py
=====================
Manages persisted, user-configurable runtime preferences: chunk size,
chunk overlap, retrieval top-K, LLM temperature, max tokens, and model
selection.

These are genuinely functional, not cosmetic: `ui/chat.py` reads them
and passes them as per-call overrides to `ChatService.send_message*`,
and `ui/upload.py` passes chunk_size/chunk_overlap to
`DocumentService.process_upload` — the exact override parameters those
methods already expose. Persisted via the `settings` key-value table
(`sqlite_manager.py`), with `config.settings` as the fallback default
for any preference the user hasn't overridden yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings as app_settings
from src.database.sqlite_manager import SQLiteManager
from src.utils.exceptions import ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Key prefix keeps these entries clearly namespaced within the shared
# key-value `settings` table (which could later hold unrelated
# preferences from other features).
_KEY_PREFIX = "runtime_pref."

_AVAILABLE_MODELS = ["gpt-oss:120b"]


@dataclass
class RuntimePreferences:
    """The current effective set of user-configurable preferences."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_tokens: int
    model: str


class RuntimeSettingsService:
    """Reads/writes persisted runtime preferences, with config defaults as fallback."""

    def __init__(self, db: SQLiteManager | None = None) -> None:
        self.db = db or SQLiteManager()

    def get_preferences(self) -> RuntimePreferences:
        """Return the current effective preferences (persisted override, or config default)."""
        return RuntimePreferences(
            chunk_size=self._get_int("chunk_size", app_settings.rag.chunk_size),
            chunk_overlap=self._get_int("chunk_overlap", app_settings.rag.chunk_overlap),
            top_k=self._get_int("top_k", app_settings.rag.top_k),
            temperature=self._get_float("temperature", app_settings.llm.temperature),
            max_tokens=self._get_int("max_tokens", app_settings.llm.max_tokens),
            model=self.db.get_setting(f"{_KEY_PREFIX}model", app_settings.llm.model_name),
        )

    def update_preferences(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> RuntimePreferences:
        """
        Validate and persist any provided preferences (fields left as
        `None` are left unchanged). Returns the full updated preference set.

        Raises:
            ValidationError: If any provided value is out of range.
        """
        current = self.get_preferences()

        new_chunk_size = current.chunk_size if chunk_size is None else chunk_size
        new_chunk_overlap = current.chunk_overlap if chunk_overlap is None else chunk_overlap
        new_top_k = current.top_k if top_k is None else top_k
        new_temperature = current.temperature if temperature is None else temperature
        new_max_tokens = current.max_tokens if max_tokens is None else max_tokens
        new_model = current.model if model is None else model

        self._validate(new_chunk_size, new_chunk_overlap, new_top_k, new_temperature, new_max_tokens, new_model)

        if chunk_size is not None:
            self.db.set_setting(f"{_KEY_PREFIX}chunk_size", str(chunk_size))
        if chunk_overlap is not None:
            self.db.set_setting(f"{_KEY_PREFIX}chunk_overlap", str(chunk_overlap))
        if top_k is not None:
            self.db.set_setting(f"{_KEY_PREFIX}top_k", str(top_k))
        if temperature is not None:
            self.db.set_setting(f"{_KEY_PREFIX}temperature", str(temperature))
        if max_tokens is not None:
            self.db.set_setting(f"{_KEY_PREFIX}max_tokens", str(max_tokens))
        if model is not None:
            self.db.set_setting(f"{_KEY_PREFIX}model", model)

        logger.info("Updated runtime preferences")
        return self.get_preferences()

    def reset_to_defaults(self) -> RuntimePreferences:
        """Clear all persisted overrides, reverting to config.py defaults."""
        for key in ("chunk_size", "chunk_overlap", "top_k", "temperature", "max_tokens", "model"):
            self.db.set_setting(f"{_KEY_PREFIX}{key}", "")
        # Empty string sentinel is treated as "unset" by the getters below.
        return self.get_preferences()

    @staticmethod
    def available_models() -> list[str]:
        """Models selectable in the Settings panel."""
        return list(_AVAILABLE_MODELS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_int(self, key: str, default: int) -> int:
        raw = self.db.get_setting(f"{_KEY_PREFIX}{key}")
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _get_float(self, key: str, default: float) -> float:
        raw = self.db.get_setting(f"{_KEY_PREFIX}{key}")
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _validate(
        self,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> None:
        if chunk_size < 100 or chunk_size > 4000:
            raise ValidationError("Chunk size must be between 100 and 4000 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValidationError("Chunk overlap must be non-negative and smaller than chunk size.")
        if top_k < 1 or top_k > 20:
            raise ValidationError("Top-K must be between 1 and 20.")
        if temperature < 0.0 or temperature > 2.0:
            raise ValidationError("Temperature must be between 0.0 and 2.0.")
        if max_tokens < 64 or max_tokens > 8192:
            raise ValidationError("Max tokens must be between 64 and 8192.")
        if not model or not model.strip():
            raise ValidationError("Model must not be empty.")
