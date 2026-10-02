"""OpenAI-compatible gateway routes on top of the provider contract."""

import asyncio
from collections.abc import AsyncIterator, Callable
from time import perf_counter
from typing import Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from system_one.api.auth import require_api_key
from system_one.api.schemas import ChatCompletionRequest
from system_one.domain.completions import (
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    StreamingCompletionResponse,
)
from system_one.providers.base import LLMProvider
from system_one.providers.errors import (
    ProviderConfigurationError,
    ProviderUnavailableError,
)
from system_one.routing.orchestrator import (
    VIRTUAL_MODEL_NAMES,
    RoutingExecutionError,
    RoutingOrchestrator,
    RoutingOutcome,
    UnknownVirtualModelError,
)
from system_one.telemetry import (
    SSEUsageExtractor,
    TokenUsage,
    build_routing_receipt,
    emit_telemetry_event,
    extract_token_usage,
    log_routing_receipt,
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


def passthrough_response(
    response: CompletionResponse, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Return the upstream body with the upstream status code."""
    return JSONResponse(
        status_code=response.status_code,
        content=response.body,
        headers=headers,
    )


def streaming_passthrough_response(
    response: StreamingCompletionResponse,
    *,
    request_id: str,
    on_complete: Callable[[TokenUsage | None, str | None], None],
) -> StreamingResponse:
    """Observe final usage while passing SSE bytes through unchanged."""
    usage_extractor = SSEUsageExtractor()

    async def body() -> AsyncIterator[bytes]:
        stream_error: str | None = None
        try:
            async for chunk in response:
                usage_extractor.feed(chunk)
                yield chunk
        except ProviderUnavailableError:
            stream_error = "upstream_error"
            raise
        except asyncio.CancelledError:
            stream_error = "client_disconnected"
            raise
        except Exception:
            stream_error = "stream_error"
            raise
        finally:
            usage_extractor.finish()
            try:
                await response.aclose()
            finally:
                on_complete(usage_extractor.usage, stream_error)

    return StreamingResponse(
        body(),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
        background=BackgroundTask(response.aclose),
    )


def with_request_id(response: Response, request_id: str) -> Response:
    """Attach the generated correlation ID to every completion response."""
    response.headers["X-Request-ID"] = request_id
    return response


def log_request_outcome(
    *,
    request_id: str,
    model_requested: str,
    model_selected: str,
    route_type: str,
    status_code: int,
    total_latency_ms: float,
    error_type: str | None,
) -> None:
    """Log a compact outcome for explicit-model pass-through requests."""
    emit_telemetry_event(
        {
            "event": "completion_request",
            "request_id": request_id,
            "model_requested": model_requested,
            "model_selected": model_selected,
            "route_type": route_type,
            "status_code": status_code,
            "total_latency_ms": total_latency_ms,
            "error_type": error_type,
        }
    )


def log_unrouted_failure(
    *,
    request_id: str,
    model_requested: str,
    status_code: int,
    error_type: str,
    total_latency_ms: float,
) -> None:
    """Log a failure that occurred before a routing outcome was produced."""
    emit_telemetry_event(
        {
            "event": "completion_request",
            "request_id": request_id,
            "model_requested": model_requested,
            "status_code": status_code,
            "total_latency_ms": total_latency_ms,
            "error_type": error_type,
        }
    )


def response_error_type(response: CompletionResponse) -> str | None:
    """Extract an upstream error type without logging its message or body."""
    if response.status_code < 400:
        return None
    error = response.body.get("error")
    if isinstance(error, dict):
        error_type = error.get("type")
        if isinstance(error_type, str):
            return error_type
    return "upstream_error"


def log_routed_outcome(
    *,
    outcome: RoutingOutcome,
    request_id: str,
    total_latency_ms: float,
    usage: TokenUsage | None,
    status_code: int,
    error_type: str | None,
) -> None:
    """Emit a full receipt for virtual routes or a compact explicit-model log."""
    if outcome.route_type == "explicit":
        log_request_outcome(
            request_id=request_id,
            model_requested=outcome.model_requested,
            model_selected=outcome.model_selected,
            route_type=outcome.route_type,
            status_code=status_code,
            total_latency_ms=total_latency_ms,
            error_type=error_type,
        )
        return

    if outcome.tier is None or outcome.model_cost is None:
        raise RuntimeError("virtual routing outcome is missing registry metadata")

    receipt = build_routing_receipt(
        request_id=request_id,
        model_requested=outcome.model_requested,
        model_selected=outcome.model_selected,
        route_type=outcome.route_type,
        tier=outcome.tier,
        decision=outcome.decision,
        policy_name=outcome.policy_name,
        decision_latency_ms=outcome.decision_latency_ms,
        routing_latency_ms=outcome.routing_latency_ms,
        provider_latency_ms=max(0.0, total_latency_ms - outcome.routing_latency_ms),
        total_latency_ms=total_latency_ms,
        usage=usage,
        cost=outcome.model_cost,
        status_code=status_code,
        error_type=error_type,
    )
    log_routing_receipt(receipt)


def completion_error_response(
    *,
    request_id: str,
    model_requested: str,
    started_at: float,
    status_code: int,
    message: str,
    error_type: str,
) -> Response:
    """Log and return a completion error with its correlation ID."""
    log_unrouted_failure(
        request_id=request_id,
        model_requested=model_requested,
        status_code=status_code,
        error_type=error_type,
        total_latency_ms=(perf_counter() - started_at) * 1000,
    )
    return with_request_id(
        error_response(status_code, message, error_type),
        request_id,
    )


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
) -> Response:
    """Execute the routing lifecycle; explicit models pass through unchanged."""
    request_id = f"req_{uuid4().hex}"
    started_at = perf_counter()
    domain_request = to_completion_request(request)
    try:
        outcome = await orchestrator.route(domain_request)
    except UnknownVirtualModelError as error:
        return completion_error_response(
            request_id=request_id,
            model_requested=request.model,
            started_at=started_at,
            status_code=status.HTTP_400_BAD_REQUEST,
            message=str(error),
            error_type="invalid_request_error",
        )
    except RoutingExecutionError as error:
        if isinstance(error.provider_error, ProviderConfigurationError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error_type = "configuration_error"
        else:
            status_code = status.HTTP_502_BAD_GATEWAY
            error_type = "upstream_error"
        log_routed_outcome(
            outcome=error.outcome,
            request_id=request_id,
            total_latency_ms=(perf_counter() - started_at) * 1000,
            usage=None,
            status_code=status_code,
            error_type=error_type,
        )
        return with_request_id(
            error_response(status_code, str(error.provider_error), error_type),
            request_id,
        )
    except ProviderConfigurationError as error:
        return completion_error_response(
            request_id=request_id,
            model_requested=request.model,
            started_at=started_at,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=str(error),
            error_type="configuration_error",
        )
    except ProviderUnavailableError as error:
        return completion_error_response(
            request_id=request_id,
            model_requested=request.model,
            started_at=started_at,
            status_code=status.HTTP_502_BAD_GATEWAY,
            message=str(error),
            error_type="upstream_error",
        )

    response = outcome.response
    if response is None:
        raise RuntimeError("successful routing outcome is missing its provider response")
    if isinstance(response, StreamingCompletionResponse):

        def finish_stream(usage: TokenUsage | None, error_type: str | None) -> None:
            log_routed_outcome(
                outcome=outcome,
                request_id=request_id,
                total_latency_ms=(perf_counter() - started_at) * 1000,
                usage=usage,
                status_code=response.status_code,
                error_type=error_type,
            )

        return streaming_passthrough_response(
            response,
            request_id=request_id,
            on_complete=finish_stream,
        )

    log_routed_outcome(
        outcome=outcome,
        request_id=request_id,
        total_latency_ms=(perf_counter() - started_at) * 1000,
        usage=extract_token_usage(response.body),
        status_code=response.status_code,
        error_type=response_error_type(response),
    )
    return with_request_id(passthrough_response(response), request_id)
