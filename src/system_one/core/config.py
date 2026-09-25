"""Environment-backed application settings."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Phase 0 service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SYSTEM_ONE_",
        extra="ignore",
        populate_by_name=True,
    )

    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "SYSTEM_ONE_OPENROUTER_API_KEY"),
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "SYSTEM_ONE_OPENROUTER_BASE_URL"),
    )
    openrouter_timeout_seconds: float = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices(
            "OPENROUTER_TIMEOUT_SECONDS",
            "SYSTEM_ONE_OPENROUTER_TIMEOUT_SECONDS",
        ),
    )
    openrouter_http_referer: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENROUTER_HTTP_REFERER",
            "SYSTEM_ONE_OPENROUTER_HTTP_REFERER",
        ),
    )
    openrouter_app_title: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_APP_TITLE", "SYSTEM_ONE_OPENROUTER_APP_TITLE"),
    )
    jev_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("JEV_ENABLED", "SYSTEM_ONE_JEV_ENABLED"),
    )
    jev_model: str = Field(
        default="~typesafe/jev-latest",
        validation_alias=AliasChoices("JEV_MODEL", "SYSTEM_ONE_JEV_MODEL"),
    )
    jev_base_url: str = Field(
        default="https://openrouter.ai/api/alpha",
        validation_alias=AliasChoices("JEV_BASE_URL", "SYSTEM_ONE_JEV_BASE_URL"),
    )
    jev_timeout_seconds: float = Field(
        default=10,
        gt=0,
        validation_alias=AliasChoices(
            "JEV_TIMEOUT_SECONDS",
            "SYSTEM_ONE_JEV_TIMEOUT_SECONDS",
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
