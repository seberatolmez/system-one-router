"""Minimal OpenRouter transport used by the initial passthrough gateway."""

from dataclasses import dataclass
from typing import cast

import httpx

from system_one.api.schemas import ChatCompletionRequest
from system_one.core.config import Settings


class OpenRouterConfigurationError(RuntimeError):
    """Raised when the provider cannot be called with the current settings."""


class OpenRouterUnavailableError(RuntimeError):
    """Raised when the provider cannot be reached or returns invalid JSON."""


@dataclass(frozen=True)
class OpenRouterResponse:
    """The status and JSON object returned by OpenRouter."""

    status_code: int
    body: dict[str, object]


class OpenRouterClient:
    """Make isolated HTTP requests to the OpenRouter-compatible API."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def list_models(self) -> OpenRouterResponse:
        """Fetch the upstream model catalog."""
        return await self._request("GET", "/models")

    async def chat(self, request: ChatCompletionRequest) -> OpenRouterResponse:
        """Forward a validated completion request to OpenRouter."""
        return await self._request(
            "POST",
            "/chat/completions",
            request.model_dump(exclude_none=True),
        )

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> OpenRouterResponse:
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
            raise OpenRouterUnavailableError("OpenRouter request failed") from error

        try:
            body = cast(dict[str, object], response.json())
        except ValueError as error:
            raise OpenRouterUnavailableError("OpenRouter returned invalid JSON") from error

        return OpenRouterResponse(status_code=response.status_code, body=body)

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise OpenRouterConfigurationError("OpenRouter API key is not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_http_referer:
            headers["HTTP-Referer"] = self._settings.openrouter_http_referer
        if self._settings.openrouter_app_title:
            headers["X-Title"] = self._settings.openrouter_app_title
        return headers
