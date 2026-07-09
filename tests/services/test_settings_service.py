"""Unit tests for src/services/settings_service.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database.sqlite_manager import SQLiteManager
from src.services.settings_service import RuntimeSettingsService
from src.utils.exceptions import ValidationError


@pytest.fixture()
def service(tmp_path: Path) -> RuntimeSettingsService:
    db = SQLiteManager(db_path=tmp_path / "test.db")
    return RuntimeSettingsService(db=db)


class TestGetPreferences:
    def test_returns_config_defaults_when_unset(self, service: RuntimeSettingsService) -> None:
        prefs = service.get_preferences()
        assert prefs.chunk_size == 800
        assert prefs.chunk_overlap == 120
        assert prefs.top_k == 5
        assert prefs.temperature == 0.3
        assert prefs.max_tokens == 1024
        assert prefs.model == "gpt-oss:120b"


class TestUpdatePreferences:
    def test_updates_only_provided_fields(self, service: RuntimeSettingsService) -> None:
        updated = service.update_preferences(top_k=10)
        assert updated.top_k == 10
        assert updated.chunk_size == 800  # unchanged

    def test_updates_multiple_fields_at_once(self, service: RuntimeSettingsService) -> None:
        updated = service.update_preferences(temperature=0.9, max_tokens=2048)
        assert updated.temperature == 0.9
        assert updated.max_tokens == 2048

    def test_persists_across_new_service_instance(
        self, service: RuntimeSettingsService, tmp_path: Path
    ) -> None:
        service.update_preferences(top_k=12)
        new_service = RuntimeSettingsService(db=service.db)
        assert new_service.get_preferences().top_k == 12

    def test_updates_model(self, service: RuntimeSettingsService) -> None:
        updated = service.update_preferences(model="gpt-oss:120b")
        assert updated.model == "gpt-oss:120b"


class TestValidation:
    def test_chunk_size_too_small_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(chunk_size=50)

    def test_chunk_size_too_large_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(chunk_size=10000)

    def test_overlap_greater_than_chunk_size_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(chunk_size=500, chunk_overlap=600)

    def test_negative_overlap_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(chunk_overlap=-10)

    def test_top_k_out_of_range_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(top_k=0)
        with pytest.raises(ValidationError):
            service.update_preferences(top_k=100)

    def test_temperature_out_of_range_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(temperature=-0.1)
        with pytest.raises(ValidationError):
            service.update_preferences(temperature=3.0)

    def test_max_tokens_out_of_range_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(max_tokens=10)
        with pytest.raises(ValidationError):
            service.update_preferences(max_tokens=100000)

    def test_empty_model_rejected(self, service: RuntimeSettingsService) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(model="   ")

    def test_invalid_update_does_not_persist_partial_changes(
        self, service: RuntimeSettingsService
    ) -> None:
        with pytest.raises(ValidationError):
            service.update_preferences(top_k=8, temperature=99.0)
        # top_k must NOT have been saved despite being valid on its own —
        # validation happens against the full resulting state before any writes.
        assert service.get_preferences().top_k == 5


class TestResetToDefaults:
    def test_reset_clears_overrides(self, service: RuntimeSettingsService) -> None:
        service.update_preferences(top_k=15, temperature=1.5)
        service.reset_to_defaults()
        prefs = service.get_preferences()
        assert prefs.top_k == 5
        assert prefs.temperature == 0.3


class TestAvailableModels:
    def test_returns_non_empty_list(self) -> None:
        models = RuntimeSettingsService.available_models()
        assert len(models) >= 1
        assert "gpt-oss:120b" in models
