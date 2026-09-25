"""Framework-independent chat completion contracts shared across layers."""

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
