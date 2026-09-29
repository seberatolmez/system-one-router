"""Behavioral tests for deterministic policy loading and tier selection."""

from pathlib import Path

import pytest

from system_one.domain.decision import Decision
from system_one.routing.policy import (
    DeterministicPolicyEngine,
    PolicyConfigurationError,
    load_policy_config,
)

_BALANCED_POLICY = """\
policy:
  confidence_threshold: 0.85
  tiers:
    fast:
      min_complexity: 1.0
      max_complexity: 2.5
    balanced:
      min_complexity: 2.5
      max_complexity: 4.5
    reasoning:
      min_complexity: 4.5
      max_complexity: 5.0
"""


def _decision(complexity: float, confidence: float = 0.9) -> Decision:
    return Decision(
        task_type="coding",
        complexity=complexity,
        quality_requirement=3.0,
        latency_requirement=3.0,
        confidence=confidence,
    )


def _load_text(tmp_path: Path, contents: str) -> DeterministicPolicyEngine:
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(contents, encoding="utf-8")
    return DeterministicPolicyEngine(load_policy_config(policy_file))


@pytest.mark.parametrize(
    ("complexity", "expected_tier"),
    [
        (1.0, "fast"),
        (2.0, "fast"),
        (2.499, "fast"),
        (2.5, "balanced"),
        (3.25, "balanced"),
        (4.499, "balanced"),
        (4.5, "reasoning"),
        (4.999, "reasoning"),
        (5.0, "reasoning"),
    ],
)
def test_complexity_ranges_route_fractional_values_without_gaps_or_overlaps(
    tmp_path: Path, complexity: float, expected_tier: str
) -> None:
    engine = _load_text(tmp_path, _BALANCED_POLICY)

    assert engine.evaluate(_decision(complexity)).tier == expected_tier


def test_confidence_below_threshold_uses_configured_default_fallback(
    tmp_path: Path,
) -> None:
    engine = _load_text(tmp_path, _BALANCED_POLICY)

    assert engine.evaluate(_decision(complexity=1.5, confidence=0.849)).tier == "balanced"


def test_confidence_at_threshold_uses_complexity_rule(tmp_path: Path) -> None:
    engine = _load_text(tmp_path, _BALANCED_POLICY)

    assert engine.evaluate(_decision(complexity=1.5, confidence=0.85)).tier == "fast"


def test_configured_fallback_tier_is_used_below_threshold(tmp_path: Path) -> None:
    policy = _BALANCED_POLICY.replace(
        "  confidence_threshold: 0.85", "  confidence_threshold: 0.85\n  fallback_tier: fast"
    )
    engine = _load_text(tmp_path, policy)

    assert engine.evaluate(_decision(complexity=5.0, confidence=0.2)).tier == "fast"


def test_same_decision_and_policy_produce_same_result(tmp_path: Path) -> None:
    engine = _load_text(tmp_path, _BALANCED_POLICY)
    decision = _decision(complexity=3.75)

    assert engine.evaluate(decision) == engine.evaluate(decision)


def test_repository_balanced_policy_loads() -> None:
    policy_file = Path(__file__).parents[2] / "policies" / "balanced.yaml"

    policy = load_policy_config(policy_file)

    assert policy.confidence_threshold == 0.85
    assert policy.fallback_tier == "balanced"


def test_missing_policy_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(PolicyConfigurationError, match="Policy file not found"):
        load_policy_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("policy: [", "Invalid YAML"),
        ("", "Invalid policy configuration"),
        ("[]", "Invalid policy configuration"),
        ("policy:\n  confidence_threshold: 0.85\n", "tiers"),
        (
            _BALANCED_POLICY.replace("confidence_threshold: 0.85", "confidence_threshold: 1.1"),
            "confidence_threshold",
        ),
        (
            _BALANCED_POLICY.replace(
                "  tiers:", "  fallback_tier: unsupported\n  tiers:"
            ),
            "fallback_tier",
        ),
        (
            _BALANCED_POLICY.replace(
                "    reasoning:\n",
                "    unsupported:\n"
                "      min_complexity: 4.5\n"
                "      max_complexity: 5.0\n"
                "    reasoning:\n",
            ),
            "unsupported",
        ),
        (
            _BALANCED_POLICY.replace("max_complexity: 2.5", "max_complexity: 2.4"),
            "contiguous and non-overlapping",
        ),
        (
            _BALANCED_POLICY.replace("min_complexity: 2.5", "min_complexity: 2.6"),
            "contiguous and non-overlapping",
        ),
        (
            _BALANCED_POLICY.replace("min_complexity: 2.5", "min_complexity: 2.4"),
            "contiguous and non-overlapping",
        ),
        (
            _BALANCED_POLICY.replace("max_complexity: 4.5", "max_complexity: 4.4"),
            "contiguous and non-overlapping",
        ),
        (
            _BALANCED_POLICY.replace(
                "max_complexity: 2.5", "max_complexity: 2.5\n      extra: true", 1
            ),
            "extra",
        ),
        (
            _BALANCED_POLICY.replace(
                "confidence_threshold: 0.85",
                "confidence_threshold: 0.85\n  confidence_threshold: 0.7",
            ),
            "Invalid YAML",
        ),
    ],
)
def test_invalid_policy_config_fails_clearly(
    tmp_path: Path, contents: str, message: str
) -> None:
    policy_file = tmp_path / "invalid.yaml"
    policy_file.write_text(contents, encoding="utf-8")

    with pytest.raises(PolicyConfigurationError, match=message):
        load_policy_config(policy_file)


def test_missing_tier_is_rejected(tmp_path: Path) -> None:
    policy = _BALANCED_POLICY.replace(
        "    balanced:\n      min_complexity: 2.5\n      max_complexity: 4.5\n", ""
    )

    with pytest.raises(PolicyConfigurationError, match="missing tiers: balanced"):
        _load_text(tmp_path, policy)
