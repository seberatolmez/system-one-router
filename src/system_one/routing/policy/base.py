"""Policy-engine contract for turning a decision into a routing tier."""

from typing import Protocol

from system_one.domain.decision import Decision, RoutingResult


class PolicyEngine(Protocol):
    """The contract every deterministic routing policy must satisfy."""

    def evaluate(self, decision: Decision) -> RoutingResult:
        """Return the tier selected for a validated decision."""
        ...
