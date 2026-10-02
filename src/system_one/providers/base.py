"""Generic LLM provider contract."""

from typing import Protocol

from system_one.domain.completions import (
    CompletionRequest,
    CompletionResponse,
    StreamingCompletionResponse,
)


class LLMProvider(Protocol):
    """The contract every inference provider must satisfy."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a non-streaming chat completion against the provider."""
        ...

    async def stream(
        self, request: CompletionRequest
    ) -> CompletionResponse | StreamingCompletionResponse:
        """Start a stream, or return a normal response for upstream errors."""
        ...

    async def list_models(self) -> CompletionResponse:
        """Fetch the provider model catalog."""
        ...
