"""Tests for environment-backed settings."""

import pytest
from pydantic import ValidationError

from system_one.core.config import Settings

JEV_ENV_NAMES = (
    "JEV_ENABLED",
    "JEV_MODEL",
    "JEV_BASE_URL",
    "JEV_TIMEOUT_SECONDS",
    "SYSTEM_ONE_JEV_ENABLED",
    "SYSTEM_ONE_JEV_MODEL",
    "SYSTEM_ONE_JEV_BASE_URL",
    "SYSTEM_ONE_JEV_TIMEOUT_SECONDS",
)
POLICY_ENV_NAMES = ("POLICY_FILE", "SYSTEM_ONE_POLICY_FILE")
MODELS_ENV_NAMES = ("MODELS_FILE", "SYSTEM_ONE_MODELS_FILE")


def test_jev_settings_have_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in JEV_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    settings = Settings()

    assert settings.jev_enabled is False
    assert settings.jev_model == "~typesafe/jev-latest"
    assert settings.jev_base_url == "https://openrouter.ai/api/alpha"
    assert settings.jev_timeout_seconds == 10


def test_jev_settings_read_short_env_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("JEV_MODEL", "~typesafe/jev-test")
    monkeypatch.setenv("JEV_BASE_URL", "https://jev.test/api/alpha")
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", "7.5")

    settings = Settings()

    assert settings.jev_enabled is True
    assert settings.jev_model == "~typesafe/jev-test"
    assert settings.jev_base_url == "https://jev.test/api/alpha"
    assert settings.jev_timeout_seconds == 7.5


def test_jev_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(jev_timeout_seconds=0)


def test_policy_file_setting_defaults_to_balanced_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in POLICY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    assert Settings().policy_file == "policies/balanced.yaml"


@pytest.mark.parametrize("env_name", POLICY_ENV_NAMES)
def test_policy_file_setting_reads_environment_aliases(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    monkeypatch.setenv(env_name, "custom/policy.yaml")

    assert Settings().policy_file == "custom/policy.yaml"


def test_models_file_setting_defaults_to_default_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in MODELS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    assert Settings().models_file == "registry/models.yaml"


@pytest.mark.parametrize("env_name", MODELS_ENV_NAMES)
def test_models_file_setting_reads_environment_aliases(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    monkeypatch.setenv(env_name, "custom/models.yaml")

    assert Settings().models_file == "custom/models.yaml"
