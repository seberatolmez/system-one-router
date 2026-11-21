"""Strategy contract and adapter for the existing routing orchestrator."""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from system_one.domain.completions import CompletionRequest, StreamingCompletionResponse
from system_one.domain.decision import Decision, RoutingRequest, RoutingTier
from system_one.evaluation.dataset import EvaluationItem
from system_one.providers.errors import ProviderConfigurationError
from system_one.routing.decision.base import DecisionEngine
from system_one.routing.orchestrator import (
    RoutingExecutionError,
    RoutingOrchestrator,
    RoutingOutcome,
)
from system_one.routing.policy.engine import DeterministicPolicyEngine
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
    decision_source: str | None = None
    decision_cost: float | None = None
    response_text: str | None = None
    quality_passed: bool | None = None
    decision_complexity: float | None = None


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
            decision_source=_decision_source(routed_outcome.decision),
            decision_cost=_decision_cost(routed_outcome.decision),
            response_text=extract_response_text(response.body) if error_type is None else None,
        )


@dataclass(frozen=True)
class DecisionStrategy:
    """Measure only a decision engine and the policy; no completion is requested.

    A fallback decision is reported as an error with no tier, so it is excluded
    from accuracy and calibration and visible in the error rate instead.
    """

    name: str
    engine: DecisionEngine
    policy_engine: DeterministicPolicyEngine

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        started_at = perf_counter()
        decision = await self.engine.decide(
            RoutingRequest(model=self.name, messages=item.messages)
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        failed = decision.fallback_reason is not None
        return StrategyOutcome(
            item_id=item.id,
            strategy_name=self.name,
            model_requested=self.name,
            model_selected=None,
            tier=None if failed else self.policy_engine.evaluate(decision).tier,
            latency_ms=elapsed_ms,
            routing_latency_ms=elapsed_ms,
            provider_latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            estimated_cost=None,
            decision_confidence=None if failed else decision.confidence,
            status_code=None,
            error_type=decision.fallback_reason,
            decision_source=decision.decision_source,
            decision_cost=decision.cost,
            decision_complexity=None if failed else decision.complexity,
        )


def extract_response_text(body: dict[str, object]) -> str | None:
    """Return the first choice's message text, or None when the shape is unexpected."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else None


def _decision_source(decision: Decision | None) -> str | None:
    return decision.decision_source if decision is not None else None


def _decision_cost(decision: Decision | None) -> float | None:
    return decision.cost if decision is not None else None


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
        decision_source=_decision_source(outcome.decision),
        decision_cost=_decision_cost(outcome.decision),
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
