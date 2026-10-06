"""Strategy contract and adapter for the existing routing orchestrator."""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from system_one.domain.completions import CompletionRequest, StreamingCompletionResponse
from system_one.domain.decision import RoutingTier
from system_one.evaluation.dataset import EvaluationItem
from system_one.providers.errors import ProviderConfigurationError
from system_one.routing.orchestrator import (
    RoutingExecutionError,
    RoutingOrchestrator,
    RoutingOutcome,
)
from system_one.telemetry import estimate_cost, extract_token_usage


@dataclass(frozen=True)
class StrategyOutcome:
    """Measured routing and provider result for one strategy/item pair."""

    item_id: str
    strategy_name: str
    model_requested: str | None
    model_selected: str | None
    tier: RoutingTier | None
    latency_ms: float
    routing_latency_ms: float
    provider_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None
    decision_confidence: float | None
    status_code: int | None
    error_type: str | None


class BenchmarkStrategy(Protocol):
    """An executable strategy that can be applied to one evaluation item."""

    @property
    def name(self) -> str:
        """Return a stable identifier used in benchmark output."""
        ...

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        """Execute the item and return measured routing/provider results."""
        ...


@dataclass(frozen=True)
class OrchestratorStrategy:
    """Run a virtual System One model through the production orchestrator."""

    name: str
    model: str
    orchestrator: RoutingOrchestrator

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        """Execute one non-streaming request and normalize its measurements."""
        started_at = perf_counter()
        request = CompletionRequest(model=self.model, messages=item.messages)
        try:
            routed_outcome = await self.orchestrator.route(request)
        except RoutingExecutionError as error:
            elapsed_ms = (perf_counter() - started_at) * 1000
            status_code = 503 if isinstance(
                error.provider_error, ProviderConfigurationError
            ) else 502
            return _from_routing_outcome(
                item_id=item.id,
                strategy_name=self.name,
                outcome=error.outcome,
                elapsed_ms=elapsed_ms,
                status_code=status_code,
                error_type=type(error.provider_error).__name__,
            )
        except Exception as error:
            elapsed_ms = (perf_counter() - started_at) * 1000
            return StrategyOutcome(
                item_id=item.id,
                strategy_name=self.name,
                model_requested=self.model,
                model_selected=None,
                tier=None,
                latency_ms=elapsed_ms,
                routing_latency_ms=0.0,
                provider_latency_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                estimated_cost=None,
                decision_confidence=None,
                status_code=None,
                error_type=type(error).__name__,
            )

        elapsed_ms = (perf_counter() - started_at) * 1000
        response = routed_outcome.response
        if response is None:
            return _from_routing_outcome(
                item_id=item.id,
                strategy_name=self.name,
                outcome=routed_outcome,
                elapsed_ms=elapsed_ms,
                status_code=None,
                error_type="MissingProviderResponse",
            )
        if isinstance(response, StreamingCompletionResponse):
            await response.aclose()
            return _from_routing_outcome(
                item_id=item.id,
                strategy_name=self.name,
                outcome=routed_outcome,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                status_code=response.status_code,
                error_type="UnexpectedStreamingResponse",
            )

        usage = extract_token_usage(response.body)
        error_type = _response_error_type(response.status_code, response.body)
        estimated_cost = (
            estimate_cost(usage, routed_outcome.model_cost)
            if routed_outcome.model_cost is not None
            else None
        )
        return StrategyOutcome(
            item_id=item.id,
            strategy_name=self.name,
            model_requested=routed_outcome.model_requested,
            model_selected=routed_outcome.model_selected,
            tier=routed_outcome.tier,
            latency_ms=elapsed_ms,
            routing_latency_ms=routed_outcome.routing_latency_ms,
            provider_latency_ms=max(0.0, elapsed_ms - routed_outcome.routing_latency_ms),
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            estimated_cost=estimated_cost,
            decision_confidence=(
                routed_outcome.decision.confidence
                if routed_outcome.decision is not None
                else None
            ),
            status_code=response.status_code,
            error_type=error_type,
        )


def _from_routing_outcome(
    *,
    item_id: str,
    strategy_name: str,
    outcome: RoutingOutcome,
    elapsed_ms: float,
    status_code: int | None,
    error_type: str | None,
) -> StrategyOutcome:
    return StrategyOutcome(
        item_id=item_id,
        strategy_name=strategy_name,
        model_requested=outcome.model_requested,
        model_selected=outcome.model_selected,
        tier=outcome.tier,
        latency_ms=elapsed_ms,
        routing_latency_ms=outcome.routing_latency_ms,
        provider_latency_ms=max(0.0, elapsed_ms - outcome.routing_latency_ms),
        input_tokens=None,
        output_tokens=None,
        estimated_cost=None,
        decision_confidence=(
            outcome.decision.confidence if outcome.decision is not None else None
        ),
        status_code=status_code,
        error_type=error_type,
    )


def _response_error_type(status_code: int, body: dict[str, object]) -> str | None:
    if status_code < 400:
        return None
    raw_error = body.get("error")
    if isinstance(raw_error, dict):
        raw_type = raw_error.get("type")
        if isinstance(raw_type, str) and raw_type:
            return raw_type
    return "provider_error"
