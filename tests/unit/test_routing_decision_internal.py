"""Tests for the internal deterministic decision engine."""

import pytest

from system_one.domain.completions import CompletionMessage
from system_one.domain.decision import RoutingRequest
from system_one.routing.decision.internal import InternalDecisionEngine
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config


def make_request() -> RoutingRequest:
    return RoutingRequest(
        model="system-one/auto",
        messages=(
            CompletionMessage(role="system", content="You are helpful."),
            CompletionMessage(role="user", content="Route this for me."),
        ),
    )


@pytest.mark.asyncio
async def test_internal_engine_emits_sentinel_fallback_decision() -> None:
    decision = await InternalDecisionEngine().decide(make_request())

    assert decision.task_type == "general"
    assert decision.complexity == 3.0
    assert decision.quality_requirement == 3.0
    assert decision.latency_requirement == 3.0
    assert decision.confidence == 0.0
    assert decision.decision_source == "internal"
    assert decision.fallback_reason == "decision_engine_disabled"


@pytest.mark.asyncio
async def test_internal_engine_decision_reroutes_to_policy_fallback_tier() -> None:
    """Zero confidence must resolve to the deterministic policy fallback tier."""
    decision = await InternalDecisionEngine().decide(make_request())
    policy = DeterministicPolicyEngine(load_policy_config("policies/balanced.yaml"))

    assert policy.evaluate(decision).tier == "balanced"
