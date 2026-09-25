"""Generic LLM provider contract."""

from typing import Protocol

from system_one.domain.completions import CompletionRequest, CompletionResponse


class LLMProvider(Protocol):
    """The contract every inference provider must satisfy."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a chat completion request against the provider."""
        ...

    async def list_models(self) -> CompletionResponse:
        """Fetch the provider model catalog."""
        ...
