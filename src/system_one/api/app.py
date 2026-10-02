"""FastAPI application and system endpoints."""

from pathlib import Path

from fastapi import FastAPI

from system_one.api.routes import router
from system_one.core.config import get_settings
from system_one.providers.openrouter import OpenRouterProvider
from system_one.routing.decision import DecisionEngine, InternalDecisionEngine, JevDecisionEngine
from system_one.routing.orchestrator import RoutingOrchestrator
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry import load_model_registry
from system_one.telemetry import configure_telemetry_logging


def create_app() -> FastAPI:
    """Create the System One Router HTTP application.

    Configuration is validated eagerly: an invalid policy or model registry
    fails fast at startup instead of producing per-request failures.
    """
    settings = get_settings()
    configure_telemetry_logging(settings.log_level)
    application = FastAPI(
        title="System One Router",
        version="0.1.0",
        description="An OpenRouter-compatible LLM routing gateway.",
    )
    provider = OpenRouterProvider(settings)
    policy_engine = DeterministicPolicyEngine(load_policy_config(settings.policy_file))
    model_registry = load_model_registry(settings.models_file)
    decision_engine: DecisionEngine = (
        JevDecisionEngine(settings)
        if settings.jev_enabled
        else InternalDecisionEngine()
    )
    application.state.provider = provider
    application.state.orchestrator = RoutingOrchestrator(
        decision_engine=decision_engine,
        policy_engine=policy_engine,
        model_registry=model_registry,
        provider=provider,
        policy_name=Path(settings.policy_file).stem,
    )
    application.include_router(router)

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Report process liveness."""
        return {"status": "ok"}

    @application.get("/ready", tags=["system"])
    async def ready() -> dict[str, str]:
        """Report whether the application is ready to receive requests."""
        return {"status": "ready", "environment": settings.environment}

    return application


app = create_app()
