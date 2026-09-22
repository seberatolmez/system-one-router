"""Tests for the initial OpenAI-compatible request contract."""

import pytest
from pydantic import ValidationError

from system_one.api.schemas import ChatCompletionRequest


def test_chat_completion_request_accepts_common_fields() -> None:
    request = ChatCompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2,
        max_tokens=100,
    )

    assert request.model == "openai/gpt-4o-mini"
    assert request.messages[0].content == "Hello"
    assert request.stream is False


def test_chat_completion_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="openai/gpt-4o-mini", messages=[])


def test_chat_completion_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            unsupported_field=True,
        )
