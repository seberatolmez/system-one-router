"""Orchestration of the full automatic routing request lifecycle.

The orchestrator composes the decision engine, the deterministic policy
engine, the model registry, and the provider abstraction. It owns virtual
model dispatch (``system-one/auto`` and the tier entry points) and rewrites
the requested model to the concrete registry profile before the provider
call. Explicit third-party models keep the existing direct provider path.

OpenRouter-specific HTTP details remain behind the provider boundary: the
orchestrator itself is provider-agnostic.
"""

from dataclasses import replace

from system_one.domain.completions import (
    CompletionRequest,
    CompletionResponse,
    StreamingCompletionResponse,
)
from system_one.domain.decision import RoutingRequest, RoutingTier
from system_one.providers.base import LLMProvider
from system_one.routing.decision.base import DecisionEngine
from system_one.routing.policy.engine import DeterministicPolicyEngine
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


class UnknownVirtualModelError(ValueError):
    """Raised when a ``system-one/`` virtual model name is not supported."""


class RoutingOrchestrator:
    """Route a completion request through the decision-policy-registry chain."""

    def __init__(
        self,
        decision_engine: DecisionEngine,
        policy_engine: DeterministicPolicyEngine,
        model_registry: ModelRegistry,
        provider: LLMProvider,
    ) -> None:
        self._decision_engine = decision_engine
        self._policy_engine = policy_engine
        self._model_registry = model_registry
        self._provider = provider

    async def route(
        self, request: CompletionRequest
    ) -> CompletionResponse | StreamingCompletionResponse:
        """Execute the routing lifecycle for the request."""
        if not request.model.startswith(VIRTUAL_MODEL_PREFIX):
            return await self._execute(request)

        virtual_name = request.model[len(VIRTUAL_MODEL_PREFIX) :]
        if virtual_name == _VIRTUAL_AUTO_MODEL:
            decision = await self._decision_engine.decide(
                RoutingRequest(model=request.model, messages=request.messages)
            )
            result = self._policy_engine.evaluate(decision)
            return await self._chat_for_tier(request, result.tier)
        if virtual_name in VIRTUAL_TIER_MODELS:
            return await self._chat_for_tier(request, VIRTUAL_TIER_MODELS[virtual_name])
        raise UnknownVirtualModelError(
            f"Unsupported virtual model: {request.model!r}"
        )

    async def _chat_for_tier(
        self, request: CompletionRequest, tier: RoutingTier
    ) -> CompletionResponse | StreamingCompletionResponse:
        """Resolve the tier model and execute the provider call."""
        profile = self._model_registry.select(tier)
        routed_request = replace(request, model=profile.id)
        return await self._execute(routed_request)

    async def _execute(
        self, request: CompletionRequest
    ) -> CompletionResponse | StreamingCompletionResponse:
        """Select the provider operation without changing model routing policy."""
        if request.stream:
            return await self._provider.stream(request)
        return await self._provider.chat(request)
