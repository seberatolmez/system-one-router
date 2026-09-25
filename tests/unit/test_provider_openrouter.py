"""Tests for the OpenRouter provider adapter."""

import httpx
import pytest

from system_one.core.config import Settings
from system_one.domain.completions import CompletionMessage, CompletionRequest
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)
from system_one.providers.openrouter import OpenRouterProvider


def make_provider(handler: httpx.MockTransport) -> OpenRouterProvider:
    settings = Settings(
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_http_referer="https://system-one.test",
        openrouter_app_title="System One Router",
    )
    return OpenRouterProvider(settings, transport=handler)


async def test_chat_returns_upstream_status_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer openrouter-test-key"
        assert request.headers["HTTP-Referer"] == "https://system-one.test"
        assert request.headers["X-Title"] == "System One Router"
        return httpx.Response(200, json={"id": "chatcmpl_test"})

    provider = make_provider(httpx.MockTransport(handler))
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        options={"temperature": 0.2},
    )

    response = await provider.chat(request)

    assert response.status_code == 200
    assert response.body == {"id": "chatcmpl_test"}


async def test_chat_maps_network_failures_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = make_provider(httpx.MockTransport(handler))
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(ProviderUnavailableError, match="OpenRouter request failed"):
        await provider.chat(request)


async def test_chat_maps_invalid_json_to_unavailable() -> None:
    provider = make_provider(
        httpx.MockTransport(lambda _: httpx.Response(200, content=b"<html>"))
    )
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(ProviderUnavailableError, match="invalid JSON"):
        await provider.chat(request)


async def test_requests_without_api_key_raise_configuration_error() -> None:
    settings = Settings(openrouter_api_key=None)
    provider = OpenRouterProvider(settings)
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(
        ProviderConfigurationError,
        match="OpenRouter API key is not configured",
    ):
        await provider.chat(request)


async def test_list_models_returns_upstream_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/models"
        return httpx.Response(200, json={"object": "list", "data": []})

    provider = make_provider(httpx.MockTransport(handler))

    response = await provider.list_models()

    assert response.status_code == 200
    assert response.body == {"object": "list", "data": []}
