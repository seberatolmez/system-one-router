"""Tests for routing receipts, usage extraction, and cost estimates."""

import json
import logging

import pytest

from system_one.domain.decision import Decision
from system_one.routing.registry.models import ModelCost
from system_one.telemetry import (
    SSEUsageExtractor,
    TokenUsage,
    build_routing_receipt,
    estimate_cost,
    extract_token_usage,
    log_routing_receipt,
)


def test_extract_token_usage_from_openai_completion_body() -> None:
    assert extract_token_usage(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 25}}
    ) == TokenUsage(input_tokens=100, output_tokens=25)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"usage": {"prompt_tokens": -1, "completion_tokens": 2}},
            TokenUsage(input_tokens=None, output_tokens=2),
        ),
        ({"usage": {"prompt_tokens": True, "completion_tokens": False}}, None),
        ({"usage": None}, None),
        ({}, None),
    ],
)
def test_invalid_or_missing_token_usage_is_not_fabricated(
    value: dict[str, object], expected: TokenUsage | None
) -> None:
    assert extract_token_usage(value) == expected


def test_estimate_cost_uses_per_million_registry_prices() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=20)
    cost = ModelCost(input_per_million=0.8, output_per_million=4.0)

    assert estimate_cost(usage, cost) == pytest.approx(0.000088)
    assert estimate_cost(TokenUsage(input_tokens=10, output_tokens=None), cost) is None


def test_sse_usage_extractor_handles_chunk_boundaries() -> None:
    stream = (
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\r\n\r\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":13,"completion_tokens":8}}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    extractor = SSEUsageExtractor()

    for start, end in ((0, 9), (9, 47), (47, 81), (81, len(stream))):
        extractor.feed(stream[start:end])
    extractor.finish()

    assert extractor.usage == TokenUsage(input_tokens=13, output_tokens=8)


def test_routing_receipt_is_emitted_as_json_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="system_one.telemetry")
    receipt = build_routing_receipt(
        request_id="req_test",
        model_requested="system-one/auto",
        model_selected="vendor/balanced",
        route_type="auto",
        tier="balanced",
        decision=Decision(
            task_type="coding",
            complexity=3,
            quality_requirement=4,
            latency_requirement=2,
            confidence=0.95,
            decision_source="jev",
        ),
        policy_name="balanced",
        decision_latency_ms=4.0,
        routing_latency_ms=5.0,
        provider_latency_ms=20.0,
        total_latency_ms=25.0,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        cost=ModelCost(input_per_million=0.8, output_per_million=4.0),
        status_code=200,
        error_type=None,
    )

    log_routing_receipt(receipt)

    event = json.loads(caplog.records[-1].getMessage())
    assert event["event"] == "routing_receipt"
    assert event["receipt"]["request_id"] == "req_test"
    assert event["receipt"]["decision"]["decision_source"] == "jev"
    assert event["receipt"]["estimated_cost"] == pytest.approx(0.000088)
