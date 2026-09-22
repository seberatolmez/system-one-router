"""OpenAI-compatible gateway routes."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from system_one.api.auth import require_api_key
from system_one.api.openrouter import (
    OpenRouterClient,
    OpenRouterConfigurationError,
    OpenRouterUnavailableError,
)
from system_one.api.schemas import ChatCompletionRequest

router = APIRouter(prefix="/api/v1")
Authenticated = Annotated[None, Depends(require_api_key)]


def get_openrouter_client(request: Request) -> OpenRouterClient:
    """Resolve the application-scoped passthrough client."""
    return cast(OpenRouterClient, request.app.state.openrouter_client)


def error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    """Return the standard error envelope used by OpenAI-compatible APIs."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models(
    client: Annotated[OpenRouterClient, Depends(get_openrouter_client)],
) -> JSONResponse:
    """Return the upstream OpenRouter model catalog."""
    try:
        response = await client.list_models()
    except OpenRouterConfigurationError as error:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(error),
            "configuration_error",
        )
    except OpenRouterUnavailableError as error:
        return error_response(status.HTTP_502_BAD_GATEWAY, str(error), "upstream_error")

    return JSONResponse(status_code=response.status_code, content=response.body)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: Authenticated,
    client: Annotated[OpenRouterClient, Depends(get_openrouter_client)],
) -> JSONResponse:
    """Forward an explicit model completion request to OpenRouter."""
    if request.model.startswith("system-one/"):
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "Automatic routing models are not enabled yet",
        )
    if request.stream:
        return error_response(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Streaming is not enabled yet",
        )

    try:
        response = await client.chat(request)
    except OpenRouterConfigurationError as error:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(error),
            "configuration_error",
        )
    except OpenRouterUnavailableError as error:
        return error_response(status.HTTP_502_BAD_GATEWAY, str(error), "upstream_error")

    return JSONResponse(status_code=response.status_code, content=response.body)
