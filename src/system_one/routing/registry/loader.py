"""Validated loading of the model registry from a YAML file.

The file structure mirrors the policy file conventions: a single top-level
``models`` key, strict validation, and duplicate-key rejection. It wraps
registry and validation failures into ``ModelsConfigurationError`` so callers
handle one coherent configuration-error type at startup.
"""

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from system_one.domain.decision import RoutingTier
from system_one.routing.registry.models import ModelCost, ModelProfile
from system_one.routing.registry.registry import ModelRegistry, ModelRegistryError
from system_one.routing.yaml_safe import load_unique_keys_yaml


class ModelsConfigurationError(ValueError):
    """Raised when the model registry file cannot be loaded or validated."""


class _ModelCostSpec(BaseModel):
    """Per-million-token prices for one model profile entry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_per_million: float
    output_per_million: float


class _ModelProfileSpec(BaseModel):
    """A strictly validated model profile entry from the registry document."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    provider: str
    capabilities: dict[str, int | float | bool] = Field(default_factory=dict)
    cost: _ModelCostSpec


class _RegistryDocument(BaseModel):
    """Strict YAML document envelope for the model registry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    models: dict[RoutingTier, list[_ModelProfileSpec]]


def _profile_from_spec(tier: RoutingTier, spec: _ModelProfileSpec) -> ModelProfile:
    """Build the validated domain profile for one registry entry."""
    return ModelProfile(
        id=spec.id,
        provider=spec.provider,
        tier=tier,
        capabilities=dict(spec.capabilities),
        cost=ModelCost(
            input_per_million=spec.cost.input_per_million,
            output_per_million=spec.cost.output_per_million,
        ),
    )


def load_model_registry(path: str | Path) -> ModelRegistry:
    """Load and validate a model registry YAML file, raising a clear error."""
    registry_path = Path(path)
    try:
        with registry_path.open("r", encoding="utf-8") as registry_stream:
            document = load_unique_keys_yaml(registry_stream)
    except FileNotFoundError as error:
        raise ModelsConfigurationError(
            f"Model registry file not found: {registry_path}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise ModelsConfigurationError(
            f"Unable to read model registry file '{registry_path}': {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ModelsConfigurationError(
            f"Invalid YAML in model registry file '{registry_path}': {error}"
        ) from error

    try:
        parsed = _RegistryDocument.model_validate(document).models
    except (ValidationError, TypeError) as error:
        raise ModelsConfigurationError(
            f"Invalid model registry configuration in '{registry_path}': {error}"
        ) from error

    profiles = [
        _profile_from_spec(tier, spec)
        for tier, entries in parsed.items()
        for spec in entries
    ]
    try:
        return ModelRegistry(profiles)
    except (ModelRegistryError, ValueError) as error:
        raise ModelsConfigurationError(
            f"Invalid model registry configuration in '{registry_path}': {error}"
        ) from error
