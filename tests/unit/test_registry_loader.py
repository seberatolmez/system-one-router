"""Tests for validated model registry loading."""

import pytest

from system_one.routing.registry import (
    ModelRegistry,
    ModelsConfigurationError,
    load_model_registry,
)


def test_default_registry_file_loads_all_tiers() -> None:
    registry = load_model_registry("registry/models.yaml")

    assert registry.select("fast").id == "google/gemini-2.5-flash"
    assert registry.select("balanced").id == "anthropic/claude-3.5-haiku"
    assert registry.select("reasoning").id == "anthropic/claude-sonnet-4"
    assert registry.select("fast").provider == "openrouter"


def test_missing_registry_file_raises_configuration_error(tmp_path) -> None:
    with pytest.raises(
        ModelsConfigurationError, match="Model registry file not found"
    ):
        load_model_registry(tmp_path / "missing.yaml")


def test_invalid_yaml_raises_configuration_error(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("models: [unclosed", encoding="utf-8")

    with pytest.raises(ModelsConfigurationError, match="Invalid YAML"):
        load_model_registry(path)


def test_unknown_tier_key_raises_configuration_error(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  absurd:\n    - id: vendor/x\n      provider: openrouter\n"
        "      cost: {input_per_million: 0.1, output_per_million: 0.4}\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelsConfigurationError, match="Invalid model registry"):
        load_model_registry(path)


def test_missing_tier_raises_configuration_error(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  fast:\n    - id: vendor/x\n      provider: openrouter\n"
        "      cost: {input_per_million: 0.1, output_per_million: 0.4}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ModelsConfigurationError, match="missing model tiers: balanced"
    ):
        load_model_registry(path)


def test_duplicate_model_id_raises_configuration_error(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    entry = (
        "    - id: vendor/x\n"
        "      provider: openrouter\n"
        "      cost: {input_per_million: 0.1, output_per_million: 0.4}\n"
    )
    path.write_text(
        "models:\n"
        "  fast:\n" + entry +
        "  balanced:\n" + entry +
        "  reasoning:\n" + entry,
        encoding="utf-8",
    )
    # Same (provider, id) in two tiers is a duplicate model identity.
    with pytest.raises(ModelsConfigurationError, match="duplicate model profile"):
        load_model_registry(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  fast:\n"
        "    - id: vendor/x\n"
        "    - id: vendor/y\n"
        "  fast:\n"
        "    - id: vendor/z\n"
        "  balanced: []\n"
        "  reasoning: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelsConfigurationError, match="Invalid YAML"):
        load_model_registry(path)


def test_registration_order_is_the_tier_tie_break(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    entry = (
        "    - id: {mid}\n"
        "      provider: openrouter\n"
        "      cost: {{input_per_million: 0.1, output_per_million: 0.4}}\n"
    )
    path.write_text(
        "models:\n"
        "  fast:\n"
        + entry.format(mid="vendor/fast-a")
        + entry.format(mid="vendor/fast-b")
        + "  balanced:\n"
        + entry.format(mid="vendor/balanced")
        + "  reasoning:\n"
        + entry.format(mid="vendor/reasoning"),
        encoding="utf-8",
    )

    registry: ModelRegistry = load_model_registry(path)

    assert registry.select("fast").id == "vendor/fast-a"
