"""Tests for validated model tier resolution."""

from pathlib import Path

import pytest

from system_one.domain.decision import Decision
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry import ModelCost, ModelProfile, ModelRegistry, ModelRegistryError


def make_profile(
    tier: str,
    model_id: str,
    provider: str = "openrouter",
) -> ModelProfile:
    return ModelProfile(
        id=model_id,
        provider=provider,
        tier=tier,  # type: ignore[arg-type]
        capabilities={"coding": 3},
        cost=ModelCost(input_per_million=0.1, output_per_million=0.4),
    )


def complete_profiles() -> tuple[ModelProfile, ...]:
    return (
        make_profile("fast", "vendor/fast-primary"),
        make_profile("fast", "vendor/fast-secondary"),
        make_profile("balanced", "vendor/balanced"),
        make_profile("reasoning", "vendor/reasoning"),
    )


def test_registry_selects_first_registered_profile_for_each_tier() -> None:
    profiles = complete_profiles()
    registry = ModelRegistry(profiles)

    assert registry.select("fast") is profiles[0]
    assert registry.select("balanced") is profiles[2]
    assert registry.select("reasoning") is profiles[3]


def test_registry_requires_profiles_for_all_supported_tiers() -> None:
    with pytest.raises(ModelRegistryError, match="missing model tiers: reasoning"):
        ModelRegistry(
            (
                make_profile("fast", "vendor/fast"),
                make_profile("balanced", "vendor/balanced"),
            )
        )


def test_registry_rejects_an_empty_catalog() -> None:
    with pytest.raises(ModelRegistryError, match="must not be empty"):
        ModelRegistry(())


def test_registry_rejects_duplicate_provider_model_pairs() -> None:
    duplicate = make_profile("fast", "vendor/fast-primary")
    with pytest.raises(ModelRegistryError, match="duplicate model profile"):
        ModelRegistry((*complete_profiles(), duplicate))


def test_registry_allows_same_model_id_from_different_providers() -> None:
    registry = ModelRegistry(
        (
            make_profile("fast", "shared/model", "provider-a"),
            make_profile("fast", "shared/model", "provider-b"),
            make_profile("balanced", "vendor/balanced"),
            make_profile("reasoning", "vendor/reasoning"),
        )
    )

    assert registry.select("fast").provider == "provider-a"


def test_registry_rejects_unsupported_selection_tier() -> None:
    registry = ModelRegistry(complete_profiles())

    with pytest.raises(ModelRegistryError, match="unsupported model tier"):
        registry.select("premium")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("complexity", "expected_tier"),
    [(1.5, "fast"), (3.5, "balanced"), (4.8, "reasoning")],
)
def test_policy_tier_resolves_to_a_registered_model(
    complexity: float,
    expected_tier: str,
) -> None:
    policy_path = Path(__file__).parents[2] / "policies" / "balanced.yaml"
    policy = DeterministicPolicyEngine(load_policy_config(policy_path))
    registry = ModelRegistry(complete_profiles())
    decision = Decision(
        task_type="coding",
        complexity=complexity,
        quality_requirement=3.0,
        latency_requirement=3.0,
        confidence=0.95,
    )

    result = policy.evaluate(decision)
    selected = registry.select(result.tier)

    assert result.tier == expected_tier
    assert selected.tier == result.tier
    assert selected.provider == "openrouter"
