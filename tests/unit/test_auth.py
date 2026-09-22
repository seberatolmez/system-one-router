"""Tests for gateway API-key authentication."""

import pytest
from fastapi import HTTPException

from system_one.api.auth import require_api_key
from system_one.core.config import get_settings


@pytest.mark.asyncio
async def test_authentication_is_disabled_without_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYSTEM_ONE_API_KEY", raising=False)
    get_settings.cache_clear()

    await require_api_key()

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_authentication_accepts_matching_bearer_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "local-dev-key")
    get_settings.cache_clear()

    await require_api_key("Bearer local-dev-key")

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_authentication_rejects_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "local-dev-key")
    get_settings.cache_clear()

    with pytest.raises(HTTPException) as error:
        await require_api_key("Bearer wrong-key")

    assert error.value.status_code == 401
    assert error.value.headers == {"WWW-Authenticate": "Bearer"}
    get_settings.cache_clear()
