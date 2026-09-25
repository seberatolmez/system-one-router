"""OpenAI-compatible gateway routes on top of the provider contract."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from system_one.api.auth import require_api_key
from system_one.api.schemas import ChatCompletionRequest
from system_one.domain.completions import (
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
)
from system_one.providers.base import LLMProvider
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)

router = APIRouter(prefix="/api/v1")
Authenticated = Annotated[None, Depends(require_api_key)]

OPTION_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "user",
)


def get_provider(request: Request) -> LLMProvider:
    """Resolve the application-scoped provider."""
    return cast(LLMProvider, request.app.state.provider)


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


def passthrough_response(response: CompletionResponse) -> JSONResponse:
    """Return the upstream body with the upstream status code."""
    return JSONResponse(status_code=response.status_code, content=response.body)


def to_completion_request(request: ChatCompletionRequest) -> CompletionRequest:
    """Map the validated API schema to the domain completion contract."""
    options = {
        field: value
        for field in OPTION_FIELDS
        if (value := getattr(request, field)) is not None
    }
    return CompletionRequest(
        model=request.model,
        messages=tuple(
            CompletionMessage(role=message.role, content=message.content)
            for message in request.messages
        ),
        stream=request.stream,
        options=options,
    )


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models(
    provider: Annotated[LLMProvider, Depends(get_provider)],
) -> JSONResponse:
    """Return the upstream model catalog."""
    try:
        response = await provider.list_models()
    except ProviderConfigurationError as error:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(error),
            "configuration_error",
        )
    except ProviderUnavailableError as error:
        return error_response(status.HTTP_502_BAD_GATEWAY, str(error), "upstream_error")

    return passthrough_response(response)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: Authenticated,
    provider: Annotated[LLMProvider, Depends(get_provider)],
) -> JSONResponse:
    """Forward an explicit model completion request through the provider."""
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
        response = await provider.chat(to_completion_request(request))
    except ProviderConfigurationError as error:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(error),
            "configuration_error",
        )
    except ProviderUnavailableError as error:
        return error_response(status.HTTP_502_BAD_GATEWAY, str(error), "upstream_error")

    return passthrough_response(response)
