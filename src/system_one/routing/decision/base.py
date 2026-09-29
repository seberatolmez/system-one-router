"""Decision-engine contracts for routing."""

from typing import Protocol

from system_one.domain.decision import Decision, RoutingRequest


class DecisionEngine(Protocol):
    """The contract every decision engine must satisfy."""

    async def decide(self, request: RoutingRequest) -> Decision:
        """Return a validated routing decision for the request."""
        ...
