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
from system_one.routing.orchestrator import (
    VIRTUAL_MODEL_NAMES,
    RoutingOrchestrator,
    UnknownVirtualModelError,
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


def get_orchestrator(request: Request) -> RoutingOrchestrator:
    """Resolve the application-scoped routing orchestrator."""
    return cast(RoutingOrchestrator, request.app.state.orchestrator)


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


def merged_models_response(response: CompletionResponse) -> JSONResponse:
    """Return the upstream model catalog with virtual routing models added."""
    body = dict(response.body)
    virtual_entries: list[dict[str, object]] = [
        {"id": model_name, "object": "model", "owned_by": "system-one-router"}
        for model_name in VIRTUAL_MODEL_NAMES
    ]
    data = body.get("data")
    if response.status_code == 200 and isinstance(data, list):
        body["data"] = virtual_entries + list(data)
    return JSONResponse(status_code=response.status_code, content=body)


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
    return merged_models_response(response)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: Authenticated,
    orchestrator: Annotated[RoutingOrchestrator, Depends(get_orchestrator)],
) -> JSONResponse:
    """Execute the routing lifecycle; explicit models pass through unchanged."""
    if request.stream:
        return error_response(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Streaming is not enabled yet",
        )

    domain_request = to_completion_request(request)
    try:
        response = await orchestrator.route(domain_request)
    except UnknownVirtualModelError as error:
        return error_response(status.HTTP_400_BAD_REQUEST, str(error))
    except ProviderConfigurationError as error:
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(error),
            "configuration_error",
        )
    except ProviderUnavailableError as error:
        return error_response(status.HTTP_502_BAD_GATEWAY, str(error), "upstream_error")

    return passthrough_response(response)
