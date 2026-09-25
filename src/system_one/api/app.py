"""FastAPI application and system endpoints."""

from fastapi import FastAPI

from system_one.api.routes import router
from system_one.core.config import get_settings
from system_one.providers.openrouter import OpenRouterProvider


def create_app() -> FastAPI:
    """Create the System One Router HTTP application."""
    settings = get_settings()
    application = FastAPI(
        title="System One Router",
        version="0.1.0",
        description="An OpenRouter-compatible LLM routing gateway.",
    )
    application.state.provider = OpenRouterProvider(settings)
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
