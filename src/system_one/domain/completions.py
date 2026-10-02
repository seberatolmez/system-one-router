"""Framework-independent chat completion contracts shared across layers."""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompletionMessage:
    """A single chat message in the provider-agnostic completion contract."""

    role: str
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    """A chat completion request decoupled from the HTTP API schema."""

    model: str
    messages: tuple[CompletionMessage, ...]
    stream: bool = False
    options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResponse:
    """A tolerant pass-through of the upstream provider response body."""

    status_code: int
    body: dict[str, object]


@dataclass
class StreamingCompletionResponse:
    """An SSE response whose owner controls upstream stream finalization."""

    status_code: int
    chunks: AsyncIterator[bytes]
    finalize: Callable[[], Awaitable[None]] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Iterate response bytes and finalize on completion or cancellation."""
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.chunks:
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Finalize the stream once, including when the consumer stops early."""
        if self._closed:
            return
        self._closed = True
        await self.finalize()
