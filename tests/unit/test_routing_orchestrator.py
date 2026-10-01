"""Tests for the routing orchestrator lifecycle."""

import pytest

from system_one.domain.completions import (
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
)
from system_one.domain.decision import Decision, RoutingRequest
from system_one.routing.orchestrator import (
    VIRTUAL_MODEL_NAMES,
    RoutingOrchestrator,
    UnknownVirtualModelError,
)
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry.models import ModelCost, ModelProfile
from system_one.routing.registry.registry import ModelRegistry


class RecordingProvider:
    """Fake provider recording the exact requests it receives."""

    def __init__(self) -> None:
        self.chats: list[CompletionRequest] = []

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        self.chats.append(request)
        return CompletionResponse(
            status_code=200,
            body={"id": "chatcmpl_fake", "model": request.model},
        )

    async def list_models(self) -> CompletionResponse:
        raise AssertionError("routing must not call list_models")


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


def make_a_profile(tier: str, model_id: str) -> ModelProfile:
    """Return a minimal validated profile for one tier."""
    return ModelProfile(
        id=model_id,
        provider="openrouter",
        tier=tier,  # type: ignore[arg-type]
        capabilities={"coding": 3},
        cost=ModelCost(input_per_million=0.1, output_per_million=0.4),
    )


def make_profiles() -> tuple[ModelProfile, ModelProfile, ModelProfile]:
    """Return one profile per tier with distinct model ids."""
    tiers = (
        make_a_profile("fast", "vendor/fast-primary"),
        make_a_profile("balanced", "vendor/balanced-primary"),
        make_a_profile("reasoning", "vendor/reasoning-primary"),
    )
    return tiers


def make_orchestrator(
    decision_engine: FakeDecisionEngine,
) -> tuple[RoutingOrchestrator, RecordingProvider]:
    """Wire the full routing chain with test doubles."""
    provider = RecordingProvider()
    orchestrator = RoutingOrchestrator(
        decision_engine=decision_engine,
        policy_engine=DeterministicPolicyEngine(load_policy_config("policies/balanced.yaml")),
        model_registry=ModelRegistry(make_profiles()),
        provider=provider,
    )
    return orchestrator, provider


def make_request(model: str) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=(CompletionMessage(role="user", content="Route this for me."),),
        options={"temperature": 0.2},
    )


@pytest.mark.asyncio
async def test_auto_reroutes_full_decision_policy_registry_chain() -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    orchestrator, provider = make_orchestrator(engine)

    response = await orchestrator.route(make_request("system-one/auto"))

    assert response.status_code == 200
    assert response.body == {"id": "chatcmpl_fake", "model": "vendor/balanced-primary"}
    assert len(engine.requests) == 1
    assert engine.requests[0].model == "system-one/auto"
    assert engine.requests[0].messages[0].content == "Route this for me."
    assert len(provider.chats) == 1
    assert provider.chats[0].model == "vendor/balanced-primary"
    assert provider.chats[0].options == {"temperature": 0.2}


@pytest.mark.parametrize(
    ("virtual_model", "expected_model"),
    [
        ("system-one/fast", "vendor/fast-primary"),
        ("system-one/balanced", "vendor/balanced-primary"),
        ("system-one/reasoning", "vendor/reasoning-primary"),
    ],
)
@pytest.mark.asyncio
async def test_tier_virtual_models_bypass_the_decision_engine(
    virtual_model: str, expected_model: str
) -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    orchestrator, provider = make_orchestrator(engine)

    response = await orchestrator.route(make_request(virtual_model))

    assert response.status_code == 200
    assert engine.requests == []
    assert provider.chats[0].model == expected_model


@pytest.mark.asyncio
async def test_low_confidence_decision_falls_back_to_policy_tier() -> None:
    engine = FakeDecisionEngine(LOW_CONFIDENCE_DECISION)
    orchestrator, provider = make_orchestrator(engine)

    await orchestrator.route(make_request("system-one/auto"))

    assert engine.requests != []
    assert provider.chats[0].model == "vendor/balanced-primary"


@pytest.mark.asyncio
async def test_explicit_models_pass_through_without_routing() -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    orchestrator, provider = make_orchestrator(engine)

    response = await orchestrator.route(make_request("openai/gpt-4o-mini"))

    assert response.status_code == 200
    assert engine.requests == []
    assert provider.chats[0].model == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_unknown_virtual_model_is_rejected_before_execution() -> None:
    engine = FakeDecisionEngine(CONFIDENT_DECISION)
    orchestrator, provider = make_orchestrator(engine)

    with pytest.raises(UnknownVirtualModelError, match="system-one/unknown-model"):
        await orchestrator.route(make_request("system-one/unknown-model"))

    assert engine.requests == []
    assert provider.chats == []


def test_virtual_model_names_cover_all_routing_entry_points() -> None:
    assert VIRTUAL_MODEL_NAMES == (
        "system-one/auto",
        "system-one/fast",
        "system-one/balanced",
        "system-one/reasoning",
    )
