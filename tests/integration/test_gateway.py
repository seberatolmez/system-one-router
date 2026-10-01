"""Integration tests for the gateway flow through the provider abstraction."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from system_one.api.app import app
from system_one.core.config import Settings, get_settings
from system_one.providers.openrouter import OpenRouterProvider
from system_one.routing.decision import InternalDecisionEngine
from system_one.routing.orchestrator import RoutingOrchestrator
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry import load_model_registry


@pytest.fixture(autouse=True)
def configured_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "local-dev-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_client(handler: httpx.MockTransport) -> TestClient:
    settings = Settings(
        api_key="local-dev-key",
        openrouter_api_key="openrouter-test-key",
    )
    provider = OpenRouterProvider(settings, transport=handler)
    app.state.provider = provider
    app.state.orchestrator = RoutingOrchestrator(
        decision_engine=InternalDecisionEngine(),
        policy_engine=DeterministicPolicyEngine(load_policy_config("policies/balanced.yaml")),
        model_registry=load_model_registry("registry/models.yaml"),
        provider=provider,
    )
    return TestClient(app)


def test_models_forward_upstream_response_with_virtual_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/models"
        assert request.headers["Authorization"] == "Bearer openrouter-test-key"
        return httpx.Response(200, json={"object": "list", "data": []})

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer local-dev-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "system-one-router"}
            for name in (
                "system-one/auto",
                "system-one/fast",
                "system-one/balanced",
                "system-one/reasoning",
            )
        ],
    }


def test_chat_completion_forwards_request_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/chat/completions"
        assert json.loads(request.content) == {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "choices": [],
            },
        )

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json=payload,
        )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl_test"


def test_gateway_rejects_missing_api_key() -> None:
    with make_client(httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


@pytest.mark.parametrize(
    ("model", "expected_message"),
    [
        ("system-one/unknown-model", "Unsupported virtual model: 'system-one/unknown-model'"),
    ],
)
def test_gateway_rejects_unknown_virtual_models(model: str, expected_message: str) -> None:
    with make_client(httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json={"model": model, "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == expected_message


def test_gateway_routes_auto_through_default_wiring_with_fallback_engine() -> None:
    """With Jev disabled, auto still resolves through policy + registry."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "anthropic/claude-3.5-haiku"
        return httpx.Response(200, json={"id": "chatcmpl_fallback", "model": "routed"})

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json={"model": "system-one/auto", "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl_fallback"


def test_gateway_rejects_streaming_until_streaming_milestone() -> None:
    with make_client(httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 501
    assert response.json()["error"]["type"] == "invalid_request_error"
