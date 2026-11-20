"""Tests for offline router replay over measured tier answers."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from system_one.domain.completions import CompletionMessage
from system_one.domain.decision import RoutingTier
from system_one.evaluation.dataset import EvaluationDataset, EvaluationItem
from system_one.evaluation.oracle import cheapest_passing_tier
from system_one.evaluation.replay import load_outcomes, replay
from system_one.evaluation.strategy import StrategyOutcome
from system_one.routing.policy.config import ComplexityRange, PolicyConfig
from system_one.routing.policy.engine import DeterministicPolicyEngine

TIERS: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")
COST = {"fast": 0.001, "balanced": 0.01, "reasoning": 0.1}
LATENCY = {"fast": 100.0, "balanced": 200.0, "reasoning": 400.0}
# item -> pass verdict per tier (fast, balanced, reasoning)
VERDICTS = {
    "easy": (True, True, True),
    "mid": (False, True, True),
    "hard": (False, False, True),
}


def _policy() -> DeterministicPolicyEngine:
    return DeterministicPolicyEngine(
        PolicyConfig(
            confidence_threshold=0.85,
            fallback_tier="balanced",
            tiers={
                "fast": ComplexityRange(min_complexity=1.0, max_complexity=2.5),
                "balanced": ComplexityRange(min_complexity=2.5, max_complexity=4.5),
                "reasoning": ComplexityRange(min_complexity=4.5, max_complexity=5.0),
            },
        )
    )


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        "d",
        tuple(
            EvaluationItem(item_id, (CompletionMessage("user", "Hi"),), None)
            for item_id in VERDICTS
        ),
    )


def _outcome(strategy: str, item_id: str, **overrides: object) -> StrategyOutcome:
    values: dict[str, object] = {
        "item_id": item_id,
        "strategy_name": strategy,
        "model_requested": strategy,
        "model_selected": None,
        "tier": None,
        "latency_ms": 0.0,
        "routing_latency_ms": 0.0,
        "provider_latency_ms": 0.0,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost": None,
        "decision_confidence": None,
        "status_code": None,
        "error_type": None,
    }
    values.update(overrides)
    return StrategyOutcome(**values)  # type: ignore[arg-type]


def _tiers() -> dict[str, dict[str, StrategyOutcome]]:
    return {
        f"system-one/{tier}": {
            item_id: _outcome(
                f"system-one/{tier}",
                item_id,
                tier=tier,
                latency_ms=LATENCY[tier],
                estimated_cost=COST[tier],
                status_code=200,
                quality_passed=verdicts[index],
            )
            for item_id, verdicts in VERDICTS.items()
        }
        for index, tier in enumerate(TIERS)
    }


def _metrics(result: object, name: str) -> dict[str, float | None]:
    strategy = next(s for s in result.strategies if s.name == name)  # type: ignore[attr-defined]
    return {metric.name: metric.value for metric in strategy.metrics}


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ((True, True, True), "fast"),
        ((False, True, False), "balanced"),
        ((False, False, True), "reasoning"),
        ((False, False, False), None),
        ((None, True, True), None),
        ((False, None, True), None),
    ],
)
def test_cheapest_passing_tier(
    verdicts: tuple[bool | None, ...], expected: RoutingTier | None
) -> None:
    assert cheapest_passing_tier(verdicts) == expected


async def test_replay_scores_baselines_and_decision_models_on_identical_answers() -> None:
    decisions = {
        # perfect router: picks the cheapest passing tier
        "perfect/model": {
            "easy": _outcome(
                "perfect/model",
                "easy",
                tier="fast",
                latency_ms=10.0,
                decision_confidence=0.9,
                decision_cost=0.0001,
            ),
            "mid": _outcome(
                "perfect/model",
                "mid",
                tier="balanced",
                latency_ms=10.0,
                decision_confidence=0.9,
                decision_cost=0.0001,
            ),
            "hard": _outcome(
                "perfect/model",
                "hard",
                tier="reasoning",
                latency_ms=10.0,
                decision_confidence=0.9,
                decision_cost=0.0001,
            ),
        },
        # always picks fast; one decision failed -> fallback tier (balanced)
        "cheap/model": {
            "easy": _outcome("cheap/model", "easy", tier="fast", latency_ms=5.0),
            "mid": _outcome("cheap/model", "mid", tier="fast", latency_ms=5.0),
            "hard": _outcome(
                "cheap/model",
                "hard",
                tier=None,
                latency_ms=5.0,
                error_type="decision_engine_timeout",
            ),
        },
    }

    result = await replay(_dataset(), _tiers(), decisions, _policy(), "balanced")

    assert [s.name for s in result.strategies] == [
        "always-fast",
        "always-strong",
        "heuristic",
        "perfect/model",
        "cheap/model",
    ]
    assert result.dataset_name == "d-measured"

    perfect = _metrics(result, "perfect/model")
    assert perfect["quality_pass_rate"] == 1.0
    assert perfect["routing_accuracy"] == 1.0
    assert perfect["under_routing_rate"] == 0.0
    assert perfect["cost_per_successful_task"] == pytest.approx((0.111 + 0.0003) / 3)
    assert perfect["p50_latency_ms"] == pytest.approx(210.0)

    strong = _metrics(result, "always-strong")
    assert strong["quality_pass_rate"] == 1.0
    assert strong["cost_per_successful_task"] == pytest.approx(0.1)
    assert strong["escalation_rate"] == pytest.approx(2 / 3)

    fast = _metrics(result, "always-fast")
    assert fast["quality_pass_rate"] == pytest.approx(1 / 3)
    assert fast["under_routing_rate"] == pytest.approx(2 / 3)

    cheap = _metrics(result, "cheap/model")
    cheap_outcomes = next(s for s in result.strategies if s.name == "cheap/model").outcomes
    assert [o.tier for o in cheap_outcomes] == ["fast", "fast", "balanced"]
    assert cheap["quality_pass_rate"] == pytest.approx(1 / 3)
    assert cheap["error_rate"] == pytest.approx(1 / 3)

    heuristic = next(s for s in result.strategies if s.name == "heuristic").outcomes
    assert [o.tier for o in heuristic] == ["fast", "fast", "fast"]


async def test_replay_rejects_incomplete_reports() -> None:
    tiers = _tiers()
    del tiers["system-one/reasoning"]["hard"]
    with pytest.raises(ValueError, match="tier report lacks"):
        await replay(_dataset(), tiers, {}, _policy(), "balanced")

    partial = {"m": {"easy": _outcome("m", "easy", tier="fast")}}
    with pytest.raises(ValueError, match="lacks item"):
        await replay(_dataset(), _tiers(), partial, _policy(), "balanced")


def test_load_outcomes_reads_a_written_report(tmp_path: Path) -> None:
    tiers = _tiers()
    document = {
        "strategies": [
            {"name": name, "outcomes": [asdict(o) for o in by_item.values()], "metrics": []}
            for name, by_item in tiers.items()
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert load_outcomes(path) == tiers
