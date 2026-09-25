"""Integration tests for the gateway using a fake provider implementation."""

import pytest
from fastapi.testclient import TestClient

from system_one.api.app import app
from system_one.core.config import get_settings
from system_one.domain.completions import CompletionRequest, CompletionResponse
from system_one.providers.base import LLMProvider
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)


class FakeProvider:
    """Fake provider recording calls and replaying scripted outcomes."""

    def __init__(self) -> None:
        self.chats: list[CompletionRequest] = []
        self.models_called = False

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        self.chats.append(request)
        return CompletionResponse(
            status_code=200,
            body={"id": "chatcmpl_fake", "model": request.model},
        )

    async def list_models(self) -> CompletionResponse:
        self.models_called = True
        return CompletionResponse(status_code=200, body={"object": "list", "data": []})


class UnavailableProvider:
    """Fake provider failing with a transport error."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        raise ProviderUnavailableError("OpenRouter request failed")

    async def list_models(self) -> CompletionResponse:
        raise ProviderUnavailableError("OpenRouter request failed")


class ConfigurationErrorProvider:
    """Fake provider failing before any HTTP call."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        raise ProviderConfigurationError("OpenRouter API key is not configured")

    async def list_models(self) -> CompletionResponse:
        raise ProviderConfigurationError("OpenRouter API key is not configured")


@pytest.fixture(autouse=True)
def configured_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "local-dev-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_client(provider: LLMProvider) -> TestClient:
    app.state.provider = provider
    return TestClient(app)


AUTH = {"Authorization": "Bearer local-dev-key"}


def test_gateway_calls_fake_provider_for_chat_completions() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.4,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl_fake", "model": "openai/gpt-4o-mini"}
    assert len(provider.chats) == 1
    assert provider.chats[0].model == "openai/gpt-4o-mini"
    assert provider.chats[0].messages[0].content == "Hello"
    assert provider.chats[0].options == {"temperature": 0.4}


def test_gateway_calls_fake_provider_for_models() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        response = client.get("/api/v1/models", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}
    assert provider.models_called


def test_gateway_maps_transport_failure_to_upstream_error() -> None:
    with make_client(UnavailableProvider()) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_gateway_maps_configuration_failure() -> None:
    with make_client(ConfigurationErrorProvider()) as client:
        response = client.get("/api/v1/models", headers=AUTH)

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "configuration_error"


def test_gateway_requires_authentication() -> None:
    with make_client(FakeProvider()) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_gateway_blocks_explicit_routing_models() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "system-one/auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Automatic routing models are not enabled yet"
    assert provider.chats == []


def test_gateway_blocks_streaming_requests() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 501
    assert provider.chats == []
