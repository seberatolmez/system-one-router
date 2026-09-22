"""Environment-backed application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Phase 0 service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SYSTEM_ONE_",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
