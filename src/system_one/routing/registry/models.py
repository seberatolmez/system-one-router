"""Validated model metadata used by routing and registry selection."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from system_one.domain.decision import RoutingTier

type CapabilityValue = int | float | bool


@dataclass(frozen=True)
class ModelCost:
    """Per-million-token prices for one model."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        """Reject negative, non-finite, or non-numeric prices."""
        for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class ModelProfile:
    """Provider-neutral metadata for a model available to a routing tier."""

    id: str
    provider: str
    tier: RoutingTier
    capabilities: Mapping[str, CapabilityValue]
    cost: ModelCost

    def __post_init__(self) -> None:
        """Validate profile identifiers and freeze capability metadata."""
        if not self.id.strip():
            raise ValueError("model id must not be empty")
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if self.tier not in ("fast", "balanced", "reasoning"):
            raise ValueError(f"unsupported model tier: {self.tier!r}")

        capabilities: dict[str, CapabilityValue] = {}
        for name, value in self.capabilities.items():
            if not name.strip():
                raise ValueError("capability names must not be empty")
            if isinstance(value, bool):
                capabilities[name] = value
                continue
            if (
                not isinstance(value, (int, float))
                or not isfinite(value)
                or not 1.0 <= value <= 5.0
            ):
                raise ValueError(
                    f"capability '{name}' must be a boolean or a finite score from 1 to 5"
                )
            capabilities[name] = value

        object.__setattr__(self, "capabilities", MappingProxyType(capabilities))
