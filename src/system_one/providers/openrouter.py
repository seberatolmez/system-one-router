"""OpenRouter adapter implementing the generic provider contract."""

from collections.abc import AsyncIterator
from typing import cast

import httpx

from system_one.core.config import Settings
from system_one.domain.completions import (
    CompletionRequest,
    CompletionResponse,
    StreamingCompletionResponse,
)
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)


class OpenRouterProvider:
    """Make isolated HTTP requests to the OpenRouter-compatible API."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        """Forward a completion request to OpenRouter."""
        return await self._request("POST", "/chat/completions", self._payload(request))

    async def stream(
        self, request: CompletionRequest
    ) -> CompletionResponse | StreamingCompletionResponse:
        """Start streaming and keep the HTTP client alive until it is finalized."""
        headers = self._headers()
        client = httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url.rstrip("/"),
            headers=headers,
            timeout=self._settings.openrouter_timeout_seconds,
            transport=self._transport,
        )

        try:
            upstream_request = client.build_request(
                "POST", "/chat/completions", json=self._payload(request)
            )
            response = await client.send(upstream_request, stream=True)
        except httpx.RequestError as error:
            await client.aclose()
            raise ProviderUnavailableError("OpenRouter request failed") from error
        except BaseException:
            await client.aclose()
            raise

        async def close_upstream() -> None:
            try:
                await response.aclose()
            finally:
                await client.aclose()

        if not response.is_success:
            try:
                await response.aread()
                body = cast(dict[str, object], response.json())
            except httpx.RequestError as error:
                raise ProviderUnavailableError("OpenRouter request failed") from error
            except ValueError as error:
                raise ProviderUnavailableError("OpenRouter returned invalid JSON") from error
            finally:
                await close_upstream()
            return CompletionResponse(status_code=response.status_code, body=body)

        content_type = response.headers.get("content-type", "")
        if content_type.partition(";")[0].strip().lower() != "text/event-stream":
            await close_upstream()
            raise ProviderUnavailableError("OpenRouter returned a non-streaming response")

        chunks: AsyncIterator[bytes] = response.aiter_bytes()
        try:
            first_chunk = await anext(chunks)
        except StopAsyncIteration:
            first_chunk = None
        except httpx.RequestError as error:
            await close_upstream()
            raise ProviderUnavailableError("OpenRouter request failed") from error
        except BaseException:
            await close_upstream()
            raise

        async def pass_through() -> AsyncIterator[bytes]:
            if first_chunk is not None:
                yield first_chunk
            try:
                async for chunk in chunks:
                    yield chunk
            except httpx.RequestError as error:
                raise ProviderUnavailableError("OpenRouter stream failed") from error

        return StreamingCompletionResponse(
            status_code=response.status_code,
            chunks=pass_through(),
            finalize=close_upstream,
        )

    async def list_models(self) -> CompletionResponse:
        """Fetch the upstream model catalog."""
        return await self._request("GET", "/models")

    def _payload(self, request: CompletionRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": request.stream,
        }
        payload.update(request.options)
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> CompletionResponse:
        headers = self._headers()
        base_url = self._settings.openrouter_base_url.rstrip("/")

        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=self._settings.openrouter_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.request(method, path, json=payload)
        except httpx.RequestError as error:
            raise ProviderUnavailableError("OpenRouter request failed") from error

        try:
            body = cast(dict[str, object], response.json())
        except ValueError as error:
            raise ProviderUnavailableError("OpenRouter returned invalid JSON") from error

        return CompletionResponse(status_code=response.status_code, body=body)

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise ProviderConfigurationError("OpenRouter API key is not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_http_referer:
            headers["HTTP-Referer"] = self._settings.openrouter_http_referer
        if self._settings.openrouter_app_title:
            headers["X-Title"] = self._settings.openrouter_app_title
        return headers
