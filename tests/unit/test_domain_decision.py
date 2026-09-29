"""Tests for the typed routing decision schema."""

import dataclasses

import pytest

from system_one.domain.completions import CompletionMessage
from system_one.domain.decision import Decision, RoutingRequest


def make_decision(
    task_type: str = "coding",
    complexity: float = 3.0,
    quality_requirement: float = 4.0,
    latency_requirement: float = 2.0,
    confidence: float = 0.9,
    decision_source: str = "internal",
    fallback_reason: str | None = None,
) -> Decision:
    return Decision(
        task_type=task_type,
        complexity=complexity,
        quality_requirement=quality_requirement,
        latency_requirement=latency_requirement,
        confidence=confidence,
        decision_source=decision_source,
        fallback_reason=fallback_reason,
    )


def test_routing_request_holds_minimal_fields() -> None:
    request = RoutingRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    assert request.model == "openai/gpt-4o-mini"
    assert request.messages == (CompletionMessage(role="user", content="Hello"),)


def test_routing_request_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        RoutingRequest(model="  ", messages=())


def test_decision_holds_validated_fields_with_defaults() -> None:
    decision = make_decision()

    assert decision.task_type == "coding"
    assert decision.decision_source == "internal"
    assert decision.fallback_reason is None


def test_decision_accepts_boundary_values() -> None:
    decision = make_decision(
        task_type="general",
        complexity=1.0,
        quality_requirement=5.0,
        latency_requirement=1.0,
        confidence=1.0,
        decision_source="fallback",
        fallback_reason="decision_engine_timeout",
    )

    assert decision.complexity == 1.0
    assert decision.quality_requirement == 5.0
    assert decision.latency_requirement == 1.0
    assert decision.confidence == 1.0


def test_decision_rejects_empty_task_type() -> None:
    with pytest.raises(ValueError, match="task_type must not be empty"):
        make_decision(task_type="")


def test_decision_rejects_complexity_below_range() -> None:
    with pytest.raises(ValueError, match="complexity must be within 1.0 and 5.0"):
        make_decision(complexity=0.9)


def test_decision_rejects_quality_requirement_above_range() -> None:
    with pytest.raises(
        ValueError,
        match="quality_requirement must be within 1.0 and 5.0",
    ):
        make_decision(quality_requirement=5.1)


def test_decision_rejects_latency_requirement_below_range() -> None:
    with pytest.raises(
        ValueError,
        match="latency_requirement must be within 1.0 and 5.0",
    ):
        make_decision(latency_requirement=0.0)


def test_decision_rejects_confidence_above_range() -> None:
    with pytest.raises(ValueError, match="confidence must be within 0.0 and 1.0"):
        make_decision(confidence=1.5)


def test_decision_rejects_empty_decision_source() -> None:
    with pytest.raises(ValueError, match="decision_source must not be empty"):
        make_decision(decision_source=" ")


def test_fallback_decision_construction_is_explicit() -> None:
    decision = make_decision(
        task_type="general",
        complexity=3.0,
        quality_requirement=3.0,
        latency_requirement=3.0,
        confidence=0.0,
        decision_source="fallback",
        fallback_reason="decision_engine_unavailable",
    )

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_unavailable"


def test_decision_is_frozen() -> None:
    decision = make_decision()

    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'task_type'",
    ):
        decision.task_type = "general"  # type: ignore[misc]
