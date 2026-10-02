"""Tests for the OpenRouter provider adapter."""

from collections.abc import AsyncIterator

import httpx
import pytest

from system_one.core.config import Settings
from system_one.domain.completions import (
    CompletionMessage,
    CompletionRequest,
    StreamingCompletionResponse,
)
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)
from system_one.providers.openrouter import OpenRouterProvider


class TrackingStream(httpx.AsyncByteStream):
    """Small upstream body stream recording when HTTPX closes it."""

    def __init__(self, chunks: tuple[bytes, ...], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True


def make_provider(handler: httpx.MockTransport) -> OpenRouterProvider:
    settings = Settings(
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_http_referer="https://system-one.test",
        openrouter_app_title="System One Router",
    )
    return OpenRouterProvider(settings, transport=handler)


async def test_chat_returns_upstream_status_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer openrouter-test-key"
        assert request.headers["HTTP-Referer"] == "https://system-one.test"
        assert request.headers["X-Title"] == "System One Router"
        return httpx.Response(200, json={"id": "chatcmpl_test"})

    provider = make_provider(httpx.MockTransport(handler))
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        options={"temperature": 0.2},
    )

    response = await provider.chat(request)

    assert response.status_code == 200
    assert response.body == {"id": "chatcmpl_test"}


async def test_stream_forwards_sse_bytes_and_closes_upstream_on_completion() -> None:
    stream = TrackingStream(
        (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/chat/completions"
        assert b'"stream":true' in request.content
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    provider = make_provider(httpx.MockTransport(handler))
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        stream=True,
    )

    result = await provider.stream(request)

    assert isinstance(result, StreamingCompletionResponse)
    body = b"".join([chunk async for chunk in result])
    assert body == (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert stream.closed


async def test_stream_closes_upstream_when_consumer_cancels() -> None:
    stream = TrackingStream((b"data: first\n\n", b"data: second\n\n"))
    provider = make_provider(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        )
    )
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        stream=True,
    )

    result = await provider.stream(request)
    assert isinstance(result, StreamingCompletionResponse)
    chunks = result.__aiter__()
    assert await anext(chunks) == b"data: first\n\n"
    await chunks.aclose()

    assert stream.closed


async def test_stream_failure_before_first_chunk_is_a_normal_provider_error() -> None:
    stream = TrackingStream((), error=httpx.ReadError("upstream read failed"))
    provider = make_provider(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        )
    )
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        stream=True,
    )

    with pytest.raises(ProviderUnavailableError, match="OpenRouter request failed"):
        await provider.stream(request)

    assert stream.closed


async def test_stream_rejects_non_sse_success_and_closes_upstream() -> None:
    stream = TrackingStream((b'{"unexpected":"json"}',))
    provider = make_provider(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=stream,
            )
        )
    )
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        stream=True,
    )

    with pytest.raises(ProviderUnavailableError, match="non-streaming response"):
        await provider.stream(request)

    assert stream.closed


async def test_stream_failure_after_first_chunk_closes_upstream() -> None:
    stream = TrackingStream(
        (b"data: first\n\n",), error=httpx.ReadError("upstream read failed")
    )
    provider = make_provider(
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        )
    )
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
        stream=True,
    )
    result = await provider.stream(request)
    assert isinstance(result, StreamingCompletionResponse)

    chunks = result.__aiter__()
    assert await anext(chunks) == b"data: first\n\n"
    with pytest.raises(ProviderUnavailableError, match="OpenRouter stream failed"):
        await anext(chunks)

    assert stream.closed


async def test_chat_maps_network_failures_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = make_provider(httpx.MockTransport(handler))
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(ProviderUnavailableError, match="OpenRouter request failed"):
        await provider.chat(request)


async def test_chat_maps_invalid_json_to_unavailable() -> None:
    provider = make_provider(
        httpx.MockTransport(lambda _: httpx.Response(200, content=b"<html>"))
    )
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(ProviderUnavailableError, match="invalid JSON"):
        await provider.chat(request)


async def test_requests_without_api_key_raise_configuration_error() -> None:
    settings = Settings(openrouter_api_key=None)
    provider = OpenRouterProvider(settings)
    request = CompletionRequest(
        model="openai/gpt-4o-mini",
        messages=(CompletionMessage(role="user", content="Hello"),),
    )

    with pytest.raises(
        ProviderConfigurationError,
        match="OpenRouter API key is not configured",
    ):
        await provider.chat(request)


async def test_list_models_returns_upstream_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/models"
        return httpx.Response(200, json={"object": "list", "data": []})

    provider = make_provider(httpx.MockTransport(handler))

    response = await provider.list_models()

    assert response.status_code == 200
    assert response.body == {"object": "list", "data": []}
