"""Tests for provider-neutral model profile metadata."""

from dataclasses import FrozenInstanceError

import pytest

from system_one.routing.registry import ModelCost, ModelProfile


def make_profile(
    *,
    model_id: str = "vendor/model-fast",
    provider: str = "openrouter",
    tier: str = "fast",
    capabilities: dict[str, int | float | bool] | None = None,
    cost: ModelCost | None = None,
) -> ModelProfile:
    return ModelProfile(
        id=model_id,
        provider=provider,
        tier=tier,  # type: ignore[arg-type]
        capabilities=(
            capabilities if capabilities is not None else {"coding": 3, "vision": False}
        ),
        cost=cost or ModelCost(input_per_million=0.1, output_per_million=0.4),
    )


def test_model_profile_retains_valid_provider_neutral_metadata() -> None:
    profile = make_profile()

    assert profile.id == "vendor/model-fast"
    assert profile.provider == "openrouter"
    assert profile.tier == "fast"
    assert profile.capabilities == {"coding": 3, "vision": False}
    assert profile.cost == ModelCost(input_per_million=0.1, output_per_million=0.4)


def test_model_profile_freezes_attributes_and_capabilities() -> None:
    profile = make_profile()

    with pytest.raises(FrozenInstanceError):
        profile.id = "vendor/other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.capabilities["coding"] = 5  # type: ignore[index]


@pytest.mark.parametrize("model_id", ["", "  "])
def test_model_profile_rejects_empty_model_id(model_id: str) -> None:
    with pytest.raises(ValueError, match="model id must not be empty"):
        make_profile(model_id=model_id)


@pytest.mark.parametrize("provider", ["", "  "])
def test_model_profile_rejects_empty_provider(provider: str) -> None:
    with pytest.raises(ValueError, match="provider must not be empty"):
        make_profile(provider=provider)


def test_model_profile_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError, match="unsupported model tier"):
        make_profile(tier="premium")


@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf"), True])
def test_model_cost_rejects_invalid_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative number"):
        ModelCost(input_per_million=rate, output_per_million=0.0)


@pytest.mark.parametrize("score", [0, 5.1, float("nan"), float("inf"), "high"])
def test_model_profile_rejects_invalid_capability_scores(score: object) -> None:
    with pytest.raises(ValueError, match="finite score from 1 to 5"):
        make_profile(capabilities={"coding": score})  # type: ignore[dict-item]


def test_model_profile_rejects_empty_capability_name() -> None:
    with pytest.raises(ValueError, match="capability names must not be empty"):
        make_profile(capabilities={" ": 3})


def test_model_cost_accepts_zero_and_boundary_capability_scores() -> None:
    profile = make_profile(
        capabilities={"reasoning": 1, "coding": 5.0, "vision": True},
        cost=ModelCost(input_per_million=0, output_per_million=0.0),
    )

    assert profile.capabilities == {"reasoning": 1, "coding": 5.0, "vision": True}
    assert profile.cost.input_per_million == 0
