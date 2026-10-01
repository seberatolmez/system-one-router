"""Deterministic internal decision engine used when Jev is disabled.

It emits a single standalone fallback decision with zero confidence, which
the policy engine resolves to its configured fallback tier. Routing therefore
never runs on unvalidated state and the failure mode stays explicit:
``decision_source="internal"`` with ``fallback_reason="decision_engine_disabled"``.
"""

from system_one.domain.decision import Decision, RoutingRequest

_FALLBACK_TASK_TYPE = "general"
_FALLBACK_SCORE = 3.0
_FALLBACK_CONFIDENCE = 0.0
_FALLBACK_REASON_DISABLED = "decision_engine_disabled"


class InternalDecisionEngine:
    """Provide the deterministic fallback decision without an external engine."""

    async def decide(self, request: RoutingRequest) -> Decision:
        """Return the standalone fallback decision for any request."""
        return Decision(
            task_type=_FALLBACK_TASK_TYPE,
            complexity=_FALLBACK_SCORE,
            quality_requirement=_FALLBACK_SCORE,
            latency_requirement=_FALLBACK_SCORE,
            confidence=_FALLBACK_CONFIDENCE,
            decision_source="internal",
            fallback_reason=_FALLBACK_REASON_DISABLED,
        )
