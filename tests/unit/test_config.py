"""Settings defaults + env overrides."""

from __future__ import annotations

import pytest

from src.config import Settings, clear_settings_cache, get_settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.api.port == 6317
    assert settings.pipeline.frame_interval_seconds == 3.0
    assert settings.pipeline.max_frames == 200
    assert settings.vision.default_provider == "wm"
    assert settings.transcription_client.url == "http://localhost:6318"
    assert settings.storage.upload_dir == "/app/data/uploads"
    assert settings.storage.max_upload_size_bytes == 4 * 1024 * 1024 * 1024


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("PIPELINE_FRAME_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("PIPELINE_MAX_FRAMES", "100")
    monkeypatch.setenv("VISION_DEFAULT_PROVIDER", "stub")
    monkeypatch.setenv("TRANSCRIPTION_CLIENT_URL", "http://worker:6318")
    clear_settings_cache()
    settings = Settings()
    assert settings.api.port == 9000
    assert settings.pipeline.frame_interval_seconds == 5.0
    assert settings.pipeline.max_frames == 100
    assert settings.vision.default_provider == "stub"
    assert settings.transcription_client.url == "http://worker:6318"


def test_get_settings_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b
