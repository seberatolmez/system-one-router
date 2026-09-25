"""OpenRouter adapter implementing the generic provider contract."""

from typing import cast

import httpx

from system_one.core.config import Settings
from system_one.domain.completions import CompletionRequest, CompletionResponse
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
