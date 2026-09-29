"""Tests for the Jev decision engine."""

import json
from typing import cast

import httpx
import pytest

from system_one.core.config import Settings
from system_one.domain.completions import CompletionMessage
from system_one.domain.decision import Decision, RoutingRequest
from system_one.providers.errors import ProviderConfigurationError
from system_one.routing.decision.jev import JevDecisionEngine


def make_engine(handler: httpx.MockTransport) -> JevDecisionEngine:
    settings = Settings(
        openrouter_api_key="openrouter-test-key",
        openrouter_http_referer="https://system-one.test",
        openrouter_app_title="System One Router",
        jev_model="~typesafe/jev-latest",
        jev_base_url="https://openrouter.test/api/alpha",
        jev_timeout_seconds=10,
    )
    return JevDecisionEngine(settings, transport=handler)


def make_request() -> RoutingRequest:
    return RoutingRequest(
        model="openai/gpt-4o-mini",
        messages=(
            CompletionMessage(role="system", content="You are helpful."),
            CompletionMessage(role="user", content="Route this for me."),
        ),
    )


def valid_body(**answer_overrides: object) -> dict[str, object]:
    """Return the documented Jev decision response shape."""
    answers: dict[str, object] = {
        "task_type": {
            "type": "choice",
            "choice": "coding",
            "confidence": 0.67,
            "probabilities": {"coding": 0.78, "analysis": 0.22},
        },
        "complexity": {
            "type": "score",
            "score": 3.4,
            "confidence": 0.91,
            "probabilities": {"0": 0, "1": 0, "2": 1},
            "legend": {"2": "Moderate", "3": "Advanced"},
        },
        "quality_requirement": {
            "type": "score",
            "score": 4.2,
            "confidence": 0.93,
        },
        "latency_requirement": {
            "type": "score",
            "score": 1.99,
            "confidence": 0.99,
            "probabilities": {"0": 0, "1": 0, "2": 1},
            "legend": {"1": "Instant", "2": "Tight"},
        },
    }
    answers.update(answer_overrides)
    return {
        "id": "gen-dec-abc123",
        "model": "~typesafe/jev-latest",
        "provider": "TypeSafe",
        "answers": answers,
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
    }


def fallback_decision(reason: str) -> Decision:
    return Decision(
        task_type="general",
        complexity=3.0,
        quality_requirement=3.0,
        latency_requirement=3.0,
        confidence=0.0,
        decision_source="fallback",
        fallback_reason=reason,
    )


async def test_valid_response_maps_to_typed_decision() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=valid_body())

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.task_type == "coding"
    assert decision.complexity == 3.4
    assert decision.quality_requirement == 4.2
    assert decision.latency_requirement == 1.99
    assert decision.confidence == 0.67
    assert decision.decision_source == "jev"
    assert decision.fallback_reason is None


async def test_confidence_is_minimum_across_answers() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        body = valid_body(style={"type": "noul", "noul": 0.42})
        return httpx.Response(200, json=body)

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.confidence == 0.42


async def test_answer_without_confidence_is_skipped_for_minimum() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        body = valid_body(complexity={"type": "score", "score": 2.0})
        return httpx.Response(200, json=body)

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.complexity == 2.0
    assert decision.confidence == 0.67


async def test_out_of_range_confidence_is_invalid_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        body = valid_body(style={"type": "noul", "noul": 1.3})
        return httpx.Response(200, json=body)

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_invalid_response"


@pytest.mark.parametrize("score", [6.0, 0.5, "abc", None, True, [2.0]])
async def test_unusable_score_triggers_invalid_response_fallback(score: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        answer = {"type": "score", "score": score, "confidence": 0.9}
        return httpx.Response(200, json=valid_body(complexity=answer))

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_invalid_response"


async def test_string_encoded_score_is_coerced() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        answer = {"type": "score", "score": "2", "confidence": 0.9}
        return httpx.Response(200, json=valid_body(complexity=answer))

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.complexity == 2.0
    assert decision.decision_source == "jev"


async def test_missing_answers_object_triggers_invalid_response_fallback() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        body = valid_body()
        del body["answers"]
        return httpx.Response(200, json=body)

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_invalid_response"


async def test_missing_single_answer_triggers_invalid_response_fallback() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=valid_body(quality_requirement=None))

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_invalid_response"


async def test_non_object_json_triggers_invalid_response_fallback() -> None:
    engine = make_engine(httpx.MockTransport(lambda _: httpx.Response(200, json=["nope"])))

    decision = await engine.decide(make_request())

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_invalid_response"


async def test_malformed_json_triggers_invalid_response_fallback() -> None:
    engine = make_engine(
        httpx.MockTransport(lambda _: httpx.Response(200, content=b"<html>"))
    )

    decision = await engine.decide(make_request())

    assert decision.decision_source == "fallback"
    assert decision.fallback_reason == "decision_engine_invalid_response"


async def test_http_500_returns_explicit_fallback() -> None:
    engine = make_engine(httpx.MockTransport(lambda _: httpx.Response(500, json={})))

    decision = await engine.decide(make_request())

    assert decision == fallback_decision("decision_engine_unavailable")


async def test_network_error_returns_unavailable_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision == fallback_decision("decision_engine_unavailable")


async def test_timeout_returns_timeout_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision == fallback_decision("decision_engine_timeout")


async def test_missing_api_key_fails_loud_without_request() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=valid_body())

    settings = Settings(openrouter_api_key=None)
    engine = JevDecisionEngine(settings, transport=httpx.MockTransport(handler))

    with pytest.raises(ProviderConfigurationError, match="OpenRouter API key"):
        await engine.decide(make_request())

    assert called is False


async def test_request_payload_matches_jev_contract() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["authorization"] = request.headers["Authorization"]
        captured["content_type"] = request.headers["Content-Type"]
        captured["http_referer"] = request.headers["HTTP-Referer"]
        captured["x_title"] = request.headers["X-Title"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=valid_body())

    engine = make_engine(httpx.MockTransport(handler))

    decision = await engine.decide(make_request())

    assert decision.decision_source == "jev"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/alpha/decisions"
    assert captured["authorization"] == "Bearer openrouter-test-key"
    assert captured["content_type"] == "application/json"
    assert captured["http_referer"] == "https://system-one.test"
    assert captured["x_title"] == "System One Router"

    body = cast(dict[str, object], captured["body"])
    assert body["model"] == "~typesafe/jev-latest"
    assert body["state"] == "system: You are helpful.\nuser: Route this for me."

    questions = cast(dict[str, dict[str, object]], body["questions"])
    assert set(questions) == {
        "task_type",
        "complexity",
        "quality_requirement",
        "latency_requirement",
    }
    assert questions["task_type"]["type"] == "choice"
    task_type_criteria = cast(dict[str, str], questions["task_type"]["criteria"])
    assert set(task_type_criteria) == {
        "coding",
        "summarization",
        "question_answering",
        "analysis",
        "translation",
    }
    for name in ("complexity", "quality_requirement", "latency_requirement"):
        assert questions[name]["type"] == "score"
        criteria = cast(list[str], questions[name]["criteria"])
        assert len(criteria) == 5
