"""Extensible, deterministic metrics over collected strategy outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Protocol

from system_one.domain.decision import RoutingTier
from system_one.evaluation.dataset import EvaluationDataset
from system_one.evaluation.strategy import StrategyOutcome

_TIER_RANK: dict[RoutingTier, int] = {"fast": 0, "balanced": 1, "reasoning": 2}


class EvaluationMetric(Protocol):
    """A metric calculated from one dataset and one strategy's results."""

    @property
    def name(self) -> str:
        """Return a stable metric identifier."""
        ...

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> "MetricResult":
        """Compute the metric and report how many observations contributed."""
        ...


@dataclass(frozen=True)
class MetricResult:
    """A numeric measurement or ``None`` when no observations are available."""

    name: str
    value: float | None
    unit: str
    samples: int


@dataclass(frozen=True)
class RoutingAccuracyMetric:
    """Fraction of labeled items routed to their expected tier."""

    name: str = "routing_accuracy"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        expected = {item.id: item.expected_tier for item in dataset.items}
        comparable = [
            outcome
            for outcome in outcomes
            if expected.get(outcome.item_id) is not None and outcome.tier is not None
        ]
        value = (
            sum(outcome.tier == expected[outcome.item_id] for outcome in comparable)
            / len(comparable)
            if comparable
            else None
        )
        return MetricResult(self.name, value, "fraction", len(comparable))


@dataclass(frozen=True)
class AverageCostPerRequestMetric:
    """Mean measured cost across requests with complete usage and registry prices."""

    name: str = "average_cost_per_request"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        del dataset
        measured_costs = [
            outcome.estimated_cost
            for outcome in outcomes
            if outcome.estimated_cost is not None
        ]
        value = sum(measured_costs) / len(measured_costs) if measured_costs else None
        return MetricResult(self.name, value, "cost units/request", len(measured_costs))


@dataclass(frozen=True)
class LatencyPercentileMetric:
    """Nearest-rank wall-clock latency percentile in milliseconds."""

    percentile: int

    def __post_init__(self) -> None:
        if not 1 <= self.percentile <= 100:
            raise ValueError("percentile must be between 1 and 100")

    @property
    def name(self) -> str:
        return f"p{self.percentile}_latency_ms"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        del dataset
        latencies = sorted(outcome.latency_ms for outcome in outcomes)
        if not latencies:
            value = None
        else:
            rank = max(1, ceil(self.percentile / 100 * len(latencies)))
            value = latencies[rank - 1]
        return MetricResult(self.name, value, "milliseconds", len(latencies))


@dataclass(frozen=True)
class ErrorRateMetric:
    """Fraction of requests with a provider or strategy error."""

    name: str = "error_rate"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        del dataset
        if not outcomes:
            value = None
        else:
            errors = sum(
                outcome.error_type is not None
                or (outcome.status_code is not None and outcome.status_code >= 400)
                for outcome in outcomes
            )
            value = errors / len(outcomes)
        return MetricResult(self.name, value, "fraction", len(outcomes))


@dataclass(frozen=True)
class EscalationRateMetric:
    """Fraction of labeled requests routed above the expected tier."""

    name: str = "escalation_rate"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        expected = {item.id: item.expected_tier for item in dataset.items}
        comparable = [
            outcome
            for outcome in outcomes
            if expected.get(outcome.item_id) is not None and outcome.tier is not None
        ]
        escalations = sum(_is_escalation(outcome, expected) for outcome in comparable)
        value = escalations / len(comparable) if comparable else None
        return MetricResult(self.name, value, "fraction", len(comparable))


def _is_escalation(
    outcome: StrategyOutcome,
    expected: dict[str, RoutingTier | None],
) -> bool:
    selected_tier = outcome.tier
    expected_tier = expected[outcome.item_id]
    return (
        selected_tier is not None
        and expected_tier is not None
        and _TIER_RANK[selected_tier] > _TIER_RANK[expected_tier]
    )


@dataclass(frozen=True)
class UnderRoutingRateMetric:
    """Fraction of labeled requests routed below the expected tier (a quality risk)."""

    name: str = "under_routing_rate"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        expected = {item.id: item.expected_tier for item in dataset.items}
        comparable = _labeled_routed(expected, outcomes)
        under = sum(_TIER_RANK[selected] < _TIER_RANK[wanted] for selected, wanted, _ in comparable)
        value = under / len(comparable) if comparable else None
        return MetricResult(self.name, value, "fraction", len(comparable))


@dataclass(frozen=True)
class BrierScoreMetric:
    """Mean squared gap between decision confidence and routing correctness (lower is better).

    Confidence is the decision's minimum answer confidence, the value the policy
    engine thresholds on; correctness is an exact match with the expected tier.
    """

    name: str = "brier_score"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        expected = {item.id: item.expected_tier for item in dataset.items}
        pairs = [
            (outcome.decision_confidence, float(selected == wanted))
            for selected, wanted, outcome in _labeled_routed(expected, outcomes)
            if outcome.decision_confidence is not None
        ]
        value = sum((conf - hit) ** 2 for conf, hit in pairs) / len(pairs) if pairs else None
        return MetricResult(self.name, value, "squared error", len(pairs))


@dataclass(frozen=True)
class DecisionCostPer1000Metric:
    """Mean measured decision-engine cost, scaled to 1,000 requests."""

    name: str = "decision_cost_per_1000_requests"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        del dataset
        costs = [o.decision_cost for o in outcomes if o.decision_cost is not None]
        value = sum(costs) / len(costs) * 1000 if costs else None
        return MetricResult(self.name, value, "cost units/1000 requests", len(costs))


@dataclass(frozen=True)
class QualityPassRateMetric:
    """Fraction of graded requests that passed; errored requests count as failed."""

    name: str = "quality_pass_rate"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        del dataset
        graded = [o.quality_passed for o in outcomes if o.quality_passed is not None]
        value = sum(graded) / len(graded) if graded else None
        return MetricResult(self.name, value, "fraction", len(graded))


@dataclass(frozen=True)
class CostPerSuccessfulTaskMetric:
    """Measured spend divided by passed tasks, over requests with both cost and a verdict."""

    name: str = "cost_per_successful_task"

    def calculate(
        self,
        dataset: EvaluationDataset,
        outcomes: Sequence[StrategyOutcome],
    ) -> MetricResult:
        del dataset
        measured = [
            (o.estimated_cost, o.quality_passed)
            for o in outcomes
            if o.estimated_cost is not None and o.quality_passed is not None
        ]
        passes = sum(passed for _, passed in measured)
        value = sum(cost for cost, _ in measured) / passes if passes else None
        return MetricResult(self.name, value, "cost units/passed task", len(measured))


def _labeled_routed(
    expected: dict[str, RoutingTier | None],
    outcomes: Sequence[StrategyOutcome],
) -> list[tuple[RoutingTier, RoutingTier, StrategyOutcome]]:
    """Return (selected, expected, outcome), skipping unlabeled or unrouted items."""
    pairs: list[tuple[RoutingTier, RoutingTier, StrategyOutcome]] = []
    for outcome in outcomes:
        expected_tier = expected.get(outcome.item_id)
        if expected_tier is not None and outcome.tier is not None:
            pairs.append((outcome.tier, expected_tier, outcome))
    return pairs


DEFAULT_METRICS: tuple[EvaluationMetric, ...] = (
    RoutingAccuracyMetric(),
    AverageCostPerRequestMetric(),
    LatencyPercentileMetric(50),
    LatencyPercentileMetric(95),
    ErrorRateMetric(),
    EscalationRateMetric(),
    QualityPassRateMetric(),
    CostPerSuccessfulTaskMetric(),
)

DECISION_METRICS: tuple[EvaluationMetric, ...] = (
    RoutingAccuracyMetric(),
    UnderRoutingRateMetric(),
    EscalationRateMetric(),
    BrierScoreMetric(),
    DecisionCostPer1000Metric(),
    LatencyPercentileMetric(50),
    LatencyPercentileMetric(95),
    ErrorRateMetric(),
)
