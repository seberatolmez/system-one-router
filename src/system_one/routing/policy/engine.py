"""Deterministic policy evaluation for typed Jev decisions."""

from system_one.domain.decision import Decision, RoutingResult, RoutingTier
from system_one.routing.policy.config import ComplexityRange, PolicyConfig

_TIER_ORDER: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")


class DeterministicPolicyEngine:
    """Select a configured tier based only on confidence and complexity."""

    def __init__(self, config: PolicyConfig) -> None:
        self._confidence_threshold = config.confidence_threshold
        self._fallback_tier = config.fallback_tier
        self._tiers: tuple[tuple[RoutingTier, ComplexityRange], ...] = tuple(
            (tier, config.tiers[tier]) for tier in _TIER_ORDER
        )

    def evaluate(self, decision: Decision) -> RoutingResult:
        """Return the fallback below threshold, otherwise the matching tier.

        Complexity ranges include their minimum and exclude their maximum, so a
        shared boundary belongs to exactly one tier. The final range includes 5.0.
        """
        if decision.confidence < self._confidence_threshold:
            return RoutingResult(tier=self._fallback_tier)

        for tier, complexity_range in self._tiers:
            if complexity_range.min_complexity <= decision.complexity < (
                complexity_range.max_complexity
            ):
                return RoutingResult(tier=tier)
            if (
                complexity_range.max_complexity == 5.0
                and decision.complexity == 5.0
            ):
                return RoutingResult(tier=tier)

        raise RuntimeError(
            f"validated policy has no tier for complexity {decision.complexity}"
        )
