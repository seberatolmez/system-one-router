"""Tests for the framework-independent completion contracts."""

import dataclasses

import pytest

from system_one.domain.completions import (
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
)


def test_completion_request_holds_minimal_contract_fields() -> None:
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    assert request.model == "openai/gpt-4o-mini"
    assert request.messages[0] == CompletionMessage(role="user", content="Hello")
    assert request.stream is False
    assert request.options == {}


def test_completion_response_is_a_passthrough_envelope() -> None:
    response = CompletionResponse(status_code=200, body={"id": "chatcmpl_test"})

    assert response.status_code == 200
    assert response.body == {"id": "chatcmpl_test"}


def test_completion_request_is_frozen() -> None:
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field 'model'"):
        request.model = "other"  # type: ignore[misc]
