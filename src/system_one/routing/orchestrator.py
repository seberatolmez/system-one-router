"""Orchestration of the full automatic routing request lifecycle.

The orchestrator composes the decision engine, the deterministic policy
engine, the model registry, and the provider abstraction. It owns virtual
model dispatch (``system-one/auto`` and the tier entry points) and rewrites
the requested model to the concrete registry profile before the provider
call. Explicit third-party models keep the existing direct provider path.

OpenRouter-specific HTTP details remain behind the provider boundary: the
orchestrator itself is provider-agnostic.
"""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from system_one.domain.completions import (
    CompletionRequest,
    CompletionResponse,
    StreamingCompletionResponse,
)
from system_one.domain.decision import Decision, RoutingRequest, RoutingTier
from system_one.providers.base import LLMProvider
from system_one.providers.errors import ProviderConfigurationError, ProviderUnavailableError
from system_one.routing.decision.base import DecisionEngine
from system_one.routing.policy.engine import DeterministicPolicyEngine
from system_one.routing.registry.models import ModelCost
from system_one.routing.registry.registry import ModelRegistry

VIRTUAL_MODEL_PREFIX = "system-one/"

VIRTUAL_MODEL_NAMES: tuple[str, ...] = (
    "system-one/auto",
    "system-one/fast",
    "system-one/balanced",
    "system-one/reasoning",
)

VIRTUAL_TIER_MODELS: dict[str, RoutingTier] = {
    "fast": "fast",
    "balanced": "balanced",
    "reasoning": "reasoning",
}

_VIRTUAL_AUTO_MODEL = "auto"
RouteType = Literal["auto", "tier", "explicit"]


@dataclass(frozen=True)
class RoutingOutcome:
    """Routing provenance; provider failures retain this before any response exists."""

    response: CompletionResponse | StreamingCompletionResponse | None
    model_requested: str
    model_selected: str
    route_type: RouteType
    tier: RoutingTier | None
    decision: Decision | None
    policy_name: str | None
    model_cost: ModelCost | None
    decision_latency_ms: float | None
    routing_latency_ms: float


class UnknownVirtualModelError(ValueError):
    """Raised when a ``system-one/`` virtual model name is not supported."""


class RoutingExecutionError(RuntimeError):
    """A provider failure paired with the route metadata selected before it."""

    def __init__(
        self,
        outcome: RoutingOutcome,
        provider_error: ProviderConfigurationError | ProviderUnavailableError,
    ) -> None:
        super().__init__(str(provider_error))
        self.outcome = outcome
        self.provider_error = provider_error


class RoutingOrchestrator:
    """Route a completion request through the decision-policy-registry chain."""

    def __init__(
        self,
        decision_engine: DecisionEngine,
        policy_engine: DeterministicPolicyEngine,
        model_registry: ModelRegistry,
        provider: LLMProvider,
        policy_name: str | None = None,
    ) -> None:
        self._decision_engine = decision_engine
        self._policy_engine = policy_engine
        self._model_registry = model_registry
        self._provider = provider
        self._policy_name = policy_name

    async def route(
        self, request: CompletionRequest
    ) -> RoutingOutcome:
        """Execute the routing lifecycle for the request."""
        route_started = perf_counter()
        if not request.model.startswith(VIRTUAL_MODEL_PREFIX):
            routing_latency_ms = (perf_counter() - route_started) * 1000
            outcome = RoutingOutcome(
                response=None,
                model_requested=request.model,
                model_selected=request.model,
                route_type="explicit",
                tier=None,
                decision=None,
                policy_name=None,
                model_cost=None,
                decision_latency_ms=None,
                routing_latency_ms=routing_latency_ms,
            )
            return await self._execute(request, outcome)

        virtual_name = request.model[len(VIRTUAL_MODEL_PREFIX) :]
        if virtual_name == _VIRTUAL_AUTO_MODEL:
            decision_started = perf_counter()
            decision = await self._decision_engine.decide(
                RoutingRequest(model=request.model, messages=request.messages)
            )
            decision_latency_ms = (perf_counter() - decision_started) * 1000
            result = self._policy_engine.evaluate(decision)
            return await self._chat_for_tier(
                request,
                result.tier,
                route_started=route_started,
                route_type="auto",
                decision=decision,
                decision_latency_ms=decision_latency_ms,
                policy_name=self._policy_name,
            )
        if virtual_name in VIRTUAL_TIER_MODELS:
            return await self._chat_for_tier(
                request,
                VIRTUAL_TIER_MODELS[virtual_name],
                route_started=route_started,
                route_type="tier",
            )
        raise UnknownVirtualModelError(
            f"Unsupported virtual model: {request.model!r}"
        )

    async def _chat_for_tier(
        self,
        request: CompletionRequest,
        tier: RoutingTier,
        *,
        route_started: float,
        route_type: Literal["auto", "tier"],
        decision: Decision | None = None,
        decision_latency_ms: float | None = None,
        policy_name: str | None = None,
    ) -> RoutingOutcome:
        """Resolve the tier model and execute the provider call."""
        profile = self._model_registry.select(tier)
        routed_request = replace(request, model=profile.id)
        routing_latency_ms = (perf_counter() - route_started) * 1000
        outcome = RoutingOutcome(
            response=None,
            model_requested=request.model,
            model_selected=profile.id,
            route_type=route_type,
            tier=tier,
            decision=decision,
            policy_name=policy_name,
            model_cost=profile.cost,
            decision_latency_ms=decision_latency_ms,
            routing_latency_ms=routing_latency_ms,
        )
        return await self._execute(routed_request, outcome)

    async def _execute(
        self, request: CompletionRequest, outcome: RoutingOutcome
    ) -> RoutingOutcome:
        """Select the provider operation without changing model routing policy."""
        try:
            response: CompletionResponse | StreamingCompletionResponse
            if request.stream:
                response = await self._provider.stream(request)
            else:
                response = await self._provider.chat(request)
        except (ProviderConfigurationError, ProviderUnavailableError) as error:
            raise RoutingExecutionError(outcome, error) from error
        return replace(outcome, response=response)
