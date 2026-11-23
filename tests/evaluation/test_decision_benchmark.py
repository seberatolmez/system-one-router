"""Tests for the decision-model benchmark: strategy, metrics, factory, and CLI guards."""

import json

import httpx
import pytest

from system_one.core.config import Settings
from system_one.domain.decision import Decision, RoutingRequest, RoutingTier
from system_one.evaluation import cli
from system_one.evaluation.cli import build_decision_strategies
from system_one.evaluation.dataset import EvaluationDataset, EvaluationItem
from system_one.evaluation.metrics import (
    BrierScoreMetric,
    DecisionCostPer1000Metric,
    UnderRoutingRateMetric,
)
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import DecisionStrategy, StrategyOutcome
from system_one.routing.policy.config import ComplexityRange, PolicyConfig
from system_one.routing.policy.engine import DeterministicPolicyEngine


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


class StubEngine:
    def __init__(self, decision: Decision) -> None:
        self._decision = decision

    async def decide(self, request: RoutingRequest) -> Decision:
        return self._decision


def _decision(**overrides: object) -> Decision:
    fields: dict[str, object] = {
        "task_type": "coding",
        "complexity": 1.0,
        "quality_requirement": 3.0,
        "latency_requirement": 3.0,
        "confidence": 0.95,
        "decision_source": "jev",
        "cost": 0.00002,
    }
    fields.update(overrides)
    return Decision(**fields)  # type: ignore[arg-type]


def _outcome(
    item_id: str,
    tier: RoutingTier | None,
    confidence: float | None = None,
    decision_cost: float | None = None,
) -> StrategyOutcome:
    return StrategyOutcome(
        item_id=item_id,
        strategy_name="s",
        model_requested="s",
        model_selected=None,
        tier=tier,
        latency_ms=1.0,
        routing_latency_ms=1.0,
        provider_latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        estimated_cost=None,
        decision_confidence=confidence,
        status_code=None,
        error_type=None,
        decision_cost=decision_cost,
    )


def _labeled(*tiers: RoutingTier | None) -> EvaluationDataset:
    return EvaluationDataset(
        "d", tuple(EvaluationItem(f"i{n}", (), tier) for n, tier in enumerate(tiers))
    )


async def test_decision_strategy_reports_tier_confidence_and_cost() -> None:
    strategy = DecisionStrategy("m/one", StubEngine(_decision()), _policy())

    outcome = await strategy.run(EvaluationItem("a", (), "fast"))

    assert outcome.tier == "fast"
    assert outcome.decision_confidence == 0.95
    assert outcome.decision_cost == 0.00002
    assert outcome.decision_source == "jev"
    assert outcome.error_type is None
    assert outcome.model_selected is None


async def test_decision_strategy_counts_fallback_as_error_without_tier() -> None:
    fallback = _decision(
        confidence=0.0,
        decision_source="fallback",
        fallback_reason="decision_engine_timeout",
        cost=None,
    )
    strategy = DecisionStrategy("m/one", StubEngine(fallback), _policy())

    outcome = await strategy.run(EvaluationItem("a", (), "fast"))

    assert outcome.tier is None
    assert outcome.decision_confidence is None
    assert outcome.error_type == "decision_engine_timeout"


def test_under_routing_rate_counts_only_lower_than_expected_tiers() -> None:
    dataset = _labeled("balanced", "reasoning", "fast", None)
    outcomes = [
        _outcome("i0", "fast"),
        _outcome("i1", "reasoning"),
        _outcome("i2", "reasoning"),
        _outcome("i3", "fast"),
    ]

    result = UnderRoutingRateMetric().calculate(dataset, outcomes)

    assert result.value == pytest.approx(1 / 3)
    assert result.samples == 3


def test_brier_score_compares_confidence_to_correctness() -> None:
    dataset = _labeled("fast", "fast", "fast")
    outcomes = [
        _outcome("i0", "fast", confidence=0.9),
        _outcome("i1", "balanced", confidence=0.8),
        _outcome("i2", "fast", confidence=None),
    ]

    result = BrierScoreMetric().calculate(dataset, outcomes)

    assert result.value == pytest.approx((0.1**2 + 0.8**2) / 2)
    assert result.samples == 2


def test_decision_cost_is_scaled_to_1000_requests_and_skips_unmeasured() -> None:
    outcomes = [
        _outcome("i0", "fast", decision_cost=0.00002),
        _outcome("i1", "fast", decision_cost=0.00004),
        _outcome("i2", None),
    ]

    result = DecisionCostPer1000Metric().calculate(_labeled(None, None, None), outcomes)

    assert result.value == pytest.approx(0.03)
    assert result.samples == 2


def test_new_metrics_report_none_without_observations() -> None:
    dataset = _labeled(None)
    outcomes = [_outcome("i0", None)]

    assert UnderRoutingRateMetric().calculate(dataset, outcomes).value is None
    assert BrierScoreMetric().calculate(dataset, outcomes).value is None
    assert DecisionCostPer1000Metric().calculate(dataset, outcomes).value is None


async def test_decision_models_are_benchmarked_through_the_jev_contract() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        requested.append(model)
        if model == "broken/model":
            return httpx.Response(500, json={})
        answer = {"type": "score", "score": 0, "confidence": 0.95}
        answers = {
            "task_type": {"type": "choice", "choice": "coding", "confidence": 0.95},
            "complexity": answer,
            "quality_requirement": answer,
            "latency_requirement": answer,
        }
        return httpx.Response(200, json={"answers": answers, "usage": {"cost": 0.00002}})

    strategies = build_decision_strategies(
        ("good/model", "broken/model"),
        Settings(
            openrouter_api_key="key",
            jev_base_url="https://openrouter.test/api/alpha",
            policy_file="policies/balanced.yaml",
        ),
        httpx.MockTransport(handler),
    )
    dataset = EvaluationDataset(
        "d",
        (
            EvaluationItem("a", (), "fast"),
            EvaluationItem("b", (), "fast"),
        ),
    )

    result = await run_benchmark(
        dataset,
        strategies,
    )

    metrics = {
        strategy.name: {metric.name: metric.value for metric in strategy.metrics}
        for strategy in result.strategies
    }
    assert set(requested) == {"good/model", "broken/model"}
    assert metrics["good/model"]["routing_accuracy"] == 1.0
    assert metrics["good/model"]["error_rate"] == 0.0
    assert metrics["broken/model"]["routing_accuracy"] is None
    assert metrics["broken/model"]["error_rate"] == 1.0


@pytest.mark.parametrize(
    "argv",
    [
        ["--decision-model", "a/one", "--decision-model", "b/two", "--strategy", "system-one/auto"],
        ["--decision-model", "a/one"],
        ["--decision-model", "a/one", "--decision-model", "a/one"],
    ],
)
def test_cli_rejects_invalid_decision_model_combinations(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_cli_requires_api_key_for_decision_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(openrouter_api_key=None))

    with pytest.raises(SystemExit):
        cli.main(["--decision-model", "a/one", "--decision-model", "b/two"])
