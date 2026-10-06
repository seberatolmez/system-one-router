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


DEFAULT_METRICS: tuple[EvaluationMetric, ...] = (
    RoutingAccuracyMetric(),
    AverageCostPerRequestMetric(),
    LatencyPercentileMetric(50),
    LatencyPercentileMetric(95),
    ErrorRateMetric(),
    EscalationRateMetric(),
)
