"""Integration tests for the gateway using fake provider implementations."""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from system_one.api.app import app
from system_one.core.config import get_settings
from system_one.domain.completions import (
    CompletionRequest,
    CompletionResponse,
    StreamingCompletionResponse,
)
from system_one.domain.decision import Decision, RoutingRequest
from system_one.providers.base import LLMProvider
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)
from system_one.routing.decision.base import DecisionEngine
from system_one.routing.orchestrator import RoutingOrchestrator
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry.models import ModelCost, ModelProfile
from system_one.routing.registry.registry import ModelRegistry


class FakeProvider:
    """Fake provider recording calls and replaying scripted outcomes."""

    def __init__(self) -> None:
        self.chats: list[CompletionRequest] = []
        self.streams: list[CompletionRequest] = []
        self.closed_streams: list[CompletionRequest] = []
        self.models_called = False

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        self.chats.append(request)
        return CompletionResponse(
            status_code=200,
            body={"id": "chatcmpl_fake", "model": request.model},
        )

    async def stream(
        self, request: CompletionRequest
    ) -> StreamingCompletionResponse:
        self.streams.append(request)

        async def chunks() -> AsyncIterator[bytes]:
            yield b'data: {"id":"chatcmpl_fake","choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield b'data: {"id":"chatcmpl_fake","choices":[],"usage":{"completion_tokens":1}}\n\n'
            yield b"data: [DONE]\n\n"

        async def close() -> None:
            self.closed_streams.append(request)

        return StreamingCompletionResponse(200, chunks(), close)

    async def list_models(self) -> CompletionResponse:
        self.models_called = True
        return CompletionResponse(status_code=200, body={"object": "list", "data": []})


class UnavailableProvider:
    """Fake provider failing with a transport error."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        raise ProviderUnavailableError("OpenRouter request failed")

    async def stream(
        self, request: CompletionRequest
    ) -> CompletionResponse | StreamingCompletionResponse:
        raise ProviderUnavailableError("OpenRouter request failed")

    async def list_models(self) -> CompletionResponse:
        raise ProviderUnavailableError("OpenRouter request failed")


class ConfigurationErrorProvider:
    """Fake provider failing before any HTTP call."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        raise ProviderConfigurationError("OpenRouter API key is not configured")

    async def stream(
        self, request: CompletionRequest
    ) -> CompletionResponse | StreamingCompletionResponse:
        raise ProviderConfigurationError("OpenRouter API key is not configured")

    async def list_models(self) -> CompletionResponse:
        raise ProviderConfigurationError("OpenRouter API key is not configured")


class FakeDecisionEngine:
    """Fake decision engine returning a scripted decision."""

    def __init__(self, decision: Decision) -> None:
        self.requests: list[RoutingRequest] = []
        self._decision = decision

    async def decide(self, request: RoutingRequest) -> Decision:
        self.requests.append(request)
        return self._decision


CONFIDENT_DECISION = Decision(
    task_type="coding",
    complexity=3.0,
    quality_requirement=3.0,
    latency_requirement=3.0,
    confidence=0.95,
    decision_source="jev",
)

LOW_CONFIDENCE_DECISION = Decision(
    task_type="coding",
    complexity=5.0,
    quality_requirement=3.0,
    latency_requirement=3.0,
    confidence=0.6,
    decision_source="jev",
)

AUTH = {"Authorization": "Bearer local-dev-key"}


@pytest.fixture(autouse=True)
def configured_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEM_ONE_API_KEY", "local-dev-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_client(provider: LLMProvider) -> TestClient:
    """Wire the app with a fake provider for the full routing chain."""
    app.state.provider = provider
    app.state.orchestrator = RoutingOrchestrator(
        decision_engine=FakeDecisionEngine(CONFIDENT_DECISION),
        policy_engine=DeterministicPolicyEngine(load_policy_config("policies/balanced.yaml")),
        model_registry=ModelRegistry(make_fallback_profiles()),
        provider=provider,
    )
    return TestClient(app)


def make_fallback_profiles() -> tuple[ModelProfile, ModelProfile, ModelProfile]:
    """Profiles with distinct vendor ids used to verify tier rewrites."""
    tiers: dict[str, str] = {
        "fast": "vendor/fast-primary",
        "balanced": "vendor/balanced-primary",
        "reasoning": "vendor/reasoning-primary",
    }
    return tuple(
        ModelProfile(
            id=model_id,
            provider="openrouter",
            tier=tier,  # type: ignore[arg-type]
            capabilities={"coding": 3},
            cost=ModelCost(input_per_million=0.1, output_per_million=0.4),
        )
        for tier, model_id in tiers.items()
    )


def make_routing_client(
    decision_engine: DecisionEngine,
) -> tuple[TestClient, FakeProvider]:
    """Wire the app with an explicit decision engine and a recording provider."""
    provider = FakeProvider()
    app.state.provider = provider
    app.state.orchestrator = RoutingOrchestrator(
        decision_engine=decision_engine,
        policy_engine=DeterministicPolicyEngine(load_policy_config("policies/balanced.yaml")),
        model_registry=ModelRegistry(make_fallback_profiles()),
        provider=provider,
    )
    return TestClient(app), provider


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


def test_gateway_lists_virtual_models_beside_the_upstream_catalog() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        response = client.get("/api/v1/models", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"][:4]] == [
        "system-one/auto",
        "system-one/fast",
        "system-one/balanced",
        "system-one/reasoning",
    ]
    assert [model["object"] for model in body["data"][:4]] == ["model"] * 4
    assert provider.models_called


def test_gateway_routes_system_one_auto_through_the_full_chain() -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    client, provider = make_routing_client(engine)
    with client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "system-one/auto",
                "messages": [{"role": "user", "content": "Route this for me."}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl_fake", "model": "vendor/balanced-primary"}
    assert len(engine.requests) == 1
    assert engine.requests[0].model == "system-one/auto"
    assert engine.requests[0].messages[0].content == "Route this for me."
    assert provider.chats[0].model == "vendor/balanced-primary"


@pytest.mark.parametrize(
    ("virtual_model", "expected_model"),
    [
        ("system-one/fast", "vendor/fast-primary"),
        ("system-one/balanced", "vendor/balanced-primary"),
        ("system-one/reasoning", "vendor/reasoning-primary"),
    ],
)
def test_gateway_resolves_tier_virtual_models_without_decisions(
    virtual_model: str, expected_model: str
) -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    client, provider = make_routing_client(engine)
    with client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": virtual_model,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["model"] == expected_model
    assert engine.requests == []
    assert provider.chats[0].model == expected_model


def test_gateway_rejects_unknown_virtual_models() -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    client, provider = make_routing_client(engine)
    with client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "system-one/unknown-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 400
    assert "Unsupported virtual model" in response.json()["error"]["message"]
    assert engine.requests == []
    assert provider.chats == []


def test_gateway_uses_policy_fallback_for_low_confidence_decisions() -> None:
    engine = FakeDecisionEngine(LOW_CONFIDENCE_DECISION)
    client, provider = make_routing_client(engine)
    with client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "system-one/auto",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["model"] == "vendor/balanced-primary"
    assert provider.chats[0].model == "vendor/balanced-primary"


@pytest.mark.parametrize(
    ("model", "expected_model", "decision_count"),
    [
        ("openai/gpt-4o-mini", "openai/gpt-4o-mini", 0),
        ("system-one/auto", "vendor/balanced-primary", 1),
        ("system-one/fast", "vendor/fast-primary", 0),
    ],
)
def test_gateway_streams_explicit_and_virtual_models(
    model: str, expected_model: str, decision_count: int
) -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    client, provider = make_routing_client(engine)
    with client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert b'"usage":{"completion_tokens":1}' in response.content
    assert len(engine.requests) == decision_count
    assert provider.chats == []
    assert len(provider.streams) == 1
    assert provider.streams[0].model == expected_model
    assert provider.streams[0].stream is True
    assert provider.closed_streams == provider.streams


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


def test_gateway_maps_stream_start_failure_to_upstream_error() -> None:
    with make_client(UnavailableProvider()) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
