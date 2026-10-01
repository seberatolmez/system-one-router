"""Validated policy configuration and safe YAML loading."""

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from system_one.domain.decision import RoutingTier
from system_one.routing.yaml_safe import load_unique_keys_yaml

_TIER_ORDER: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")


class PolicyConfigurationError(ValueError):
    """Raised when the policy file cannot be loaded or validated."""


class ComplexityRange(BaseModel):
    """Half-open complexity interval; the final interval includes 5.0."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    min_complexity: float = Field(ge=1.0, le=5.0, allow_inf_nan=False)
    max_complexity: float = Field(ge=1.0, le=5.0, allow_inf_nan=False)


class PolicyConfig(BaseModel):
    """Validated settings inside a YAML document's top-level ``policy`` key."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    confidence_threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fallback_tier: RoutingTier = "balanced"
    tiers: dict[RoutingTier, ComplexityRange]

    @model_validator(mode="after")
    def validate_tiers(self) -> "PolicyConfig":
        """Require all supported tiers to partition the full complexity scale."""
        configured_tiers = set(self.tiers)
        expected_tiers = set(_TIER_ORDER)
        if configured_tiers != expected_tiers:
            missing = sorted(expected_tiers - configured_tiers)
            unsupported = sorted(configured_tiers - expected_tiers)
            details = []
            if missing:
                details.append(f"missing tiers: {', '.join(missing)}")
            if unsupported:
                details.append(f"unsupported tiers: {', '.join(unsupported)}")
            raise ValueError("; ".join(details))

        expected_minimum = 1.0
        for tier in _TIER_ORDER:
            complexity_range = self.tiers[tier]
            if complexity_range.min_complexity != expected_minimum:
                raise ValueError(
                    f"tier '{tier}' must start at {expected_minimum}; ranges must be "
                    "contiguous and non-overlapping"
                )
            if complexity_range.min_complexity >= complexity_range.max_complexity:
                raise ValueError(
                    f"tier '{tier}' min_complexity must be less than max_complexity"
                )
            expected_minimum = complexity_range.max_complexity

        if expected_minimum != 5.0:
            raise ValueError("reasoning tier must end at complexity 5.0")
        return self


class _PolicyDocument(BaseModel):
    """Strict YAML document envelope."""

    model_config = ConfigDict(extra="forbid", strict=True)

    policy: PolicyConfig


def load_policy_config(path: str | Path) -> PolicyConfig:
    """Load and validate a policy YAML file, raising a clear error on failure."""
    policy_path = Path(path)
    try:
        with policy_path.open("r", encoding="utf-8") as policy_stream:
            document = load_unique_keys_yaml(policy_stream)
    except FileNotFoundError as error:
        raise PolicyConfigurationError(f"Policy file not found: {policy_path}") from error
    except (OSError, UnicodeError) as error:
        raise PolicyConfigurationError(
            f"Unable to read policy file '{policy_path}': {error}"
        ) from error
    except yaml.YAMLError as error:
        raise PolicyConfigurationError(
            f"Invalid YAML in policy file '{policy_path}': {error}"
        ) from error

    try:
        return _PolicyDocument.model_validate(document).policy
    except ValidationError as error:
        raise PolicyConfigurationError(
            f"Invalid policy configuration in '{policy_path}': {error}"
        ) from error
