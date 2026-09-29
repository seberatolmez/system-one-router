"""Typed routing decision contracts shared across routing layers."""

from dataclasses import dataclass

from system_one.domain.completions import CompletionMessage


@dataclass(frozen=True)
class RoutingRequest:
    """Provider-agnostic input consumed by the decision engine."""

    model: str
    messages: tuple[CompletionMessage, ...]

    def __post_init__(self) -> None:
        """Validate invariants so routing never runs on unvalidated state."""
        if not self.model or not self.model.strip():
            raise ValueError("model must not be empty")


@dataclass(frozen=True)
class Decision:
    """A validated structured routing decision."""

    task_type: str
    complexity: float
    quality_requirement: float
    latency_requirement: float
    confidence: float
    decision_source: str = "internal"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate value ranges on construction."""
        if not self.task_type or not self.task_type.strip():
            raise ValueError("task_type must not be empty")
        if not self.decision_source or not self.decision_source.strip():
            raise ValueError("decision_source must not be empty")
        if self.fallback_reason is not None and not self.fallback_reason.strip():
            raise ValueError("fallback_reason must not be empty when set")
        if not 1.0 <= self.complexity <= 5.0:
            raise ValueError("complexity must be within 1.0 and 5.0")
        if not 1.0 <= self.quality_requirement <= 5.0:
            raise ValueError("quality_requirement must be within 1.0 and 5.0")
        if not 1.0 <= self.latency_requirement <= 5.0:
            raise ValueError("latency_requirement must be within 1.0 and 5.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within 0.0 and 1.0")
