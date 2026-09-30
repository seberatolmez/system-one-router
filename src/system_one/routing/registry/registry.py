"""Validated model catalog with deterministic tier-based selection."""

from collections.abc import Iterable

from system_one.domain.decision import RoutingTier
from system_one.routing.registry.models import ModelProfile

_TIER_ORDER: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")


class ModelRegistryError(ValueError):
    """Raised when model registry contents cannot support routing safely."""


class ModelRegistry:
    """Index model profiles and select the first registered profile for a tier."""

    def __init__(self, profiles: Iterable[ModelProfile]) -> None:
        """Validate unique model identities and complete tier coverage."""
        profiles_by_tier: dict[RoutingTier, list[ModelProfile]] = {
            tier: [] for tier in _TIER_ORDER
        }
        model_keys: set[tuple[str, str]] = set()
        profile_count = 0

        for profile in profiles:
            profile_count += 1
            model_key = (profile.provider, profile.id)
            if model_key in model_keys:
                raise ModelRegistryError(
                    f"duplicate model profile for provider '{profile.provider}' "
                    f"and id '{profile.id}'"
                )
            model_keys.add(model_key)
            profiles_by_tier[profile.tier].append(profile)

        if profile_count == 0:
            raise ModelRegistryError("model registry must not be empty")

        missing_tiers = [tier for tier in _TIER_ORDER if not profiles_by_tier[tier]]
        if missing_tiers:
            raise ModelRegistryError(
                f"missing model tiers: {', '.join(missing_tiers)}"
            )

        self._profiles_by_tier = {
            tier: tuple(tier_profiles)
            for tier, tier_profiles in profiles_by_tier.items()
        }

    def select(self, tier: RoutingTier) -> ModelProfile:
        """Return the first registered profile for ``tier``.

        Registration order is the explicit deterministic tie-break when a tier
        contains multiple profiles.
        """
        if tier not in _TIER_ORDER:
            raise ModelRegistryError(f"unsupported model tier: {tier!r}")
        return self._profiles_by_tier[tier][0]
