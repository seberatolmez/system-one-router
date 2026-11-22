"""Run a dataset through the same set of strategies and collect comparable results."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter

from system_one.evaluation.dataset import EvaluationDataset, EvaluationItem
from system_one.evaluation.grading import QualityGrader
from system_one.evaluation.metrics import DEFAULT_METRICS, EvaluationMetric, MetricResult
from system_one.evaluation.strategy import BenchmarkStrategy, StrategyOutcome

_logger = logging.getLogger("system_one.evaluation")


@dataclass(frozen=True)
class StrategyResult:
    """All item outcomes and aggregate metrics for one strategy."""

    name: str
    outcomes: tuple[StrategyOutcome, ...]
    metrics: tuple[MetricResult, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Serializable output of one dataset-vs-strategies benchmark run."""

    run_id: str
    created_at: str
    dataset_name: str
    item_count: int
    strategies: tuple[StrategyResult, ...]


async def _grade(
    grader: QualityGrader, item: EvaluationItem, outcome: StrategyOutcome
) -> StrategyOutcome:
    if outcome.response_text is None:
        return replace(outcome, quality_passed=False) if outcome.error_type is not None else outcome
    try:
        passed = await grader.grade(item, outcome.response_text)
    except Exception as error:
        _logger.warning("grading %s failed, leaving it unmeasured: %r", item.id, error)
        passed = None
    return replace(outcome, quality_passed=passed)


async def run_benchmark(
    dataset: EvaluationDataset,
    strategies: Sequence[BenchmarkStrategy],
    metrics: Sequence[EvaluationMetric] = DEFAULT_METRICS,
    grader: QualityGrader | None = None,
) -> BenchmarkResult:
    """Run all dataset items sequentially for each strategy.

    Sequential execution avoids one strategy's concurrency pattern influencing
    the provider latency observed by another. A failing item is captured and
    does not stop the remaining items or strategies.

    With a ``grader``, each response is graded and then its text is dropped, so
    reports hold verdicts rather than model output. A request that errored has no
    answer and is recorded as failed; a grader that cannot decide or raises leaves
    the verdict unmeasured.
    """
    if not dataset.items:
        raise ValueError("benchmark dataset must contain at least one item")
    if len(strategies) < 2:
        raise ValueError("a benchmark must compare at least two strategies")
    strategy_names = [strategy.name for strategy in strategies]
    if any(not name.strip() for name in strategy_names):
        raise ValueError("strategy names must not be empty")
    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError("strategy names must be unique")
    metric_names = [metric.name for metric in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metric names must be unique")

    strategy_results: list[StrategyResult] = []
    for strategy in strategies:
        outcomes: list[StrategyOutcome] = []
        for item in dataset.items:
            started_at = perf_counter()
            try:
                outcome = await strategy.run(item)
                if outcome.item_id != item.id or outcome.strategy_name != strategy.name:
                    raise ValueError("strategy returned an outcome for a different item or name")
            except Exception as error:
                elapsed_ms = (perf_counter() - started_at) * 1000
                outcome = StrategyOutcome(
                    item_id=item.id,
                    strategy_name=strategy.name,
                    model_requested=None,
                    model_selected=None,
                    tier=None,
                    latency_ms=elapsed_ms,
                    routing_latency_ms=0.0,
                    provider_latency_ms=elapsed_ms,
                    input_tokens=None,
                    output_tokens=None,
                    estimated_cost=None,
                    decision_confidence=None,
                    status_code=None,
                    error_type=type(error).__name__,
                )
            if grader is not None:
                outcome = await _grade(grader, item, outcome)
            outcomes.append(replace(outcome, response_text=None))
            _logger.info(
                "%s %d/%d %s tier=%s %.0fms error=%s passed=%s",
                strategy.name,
                len(outcomes),
                len(dataset.items),
                item.id,
                outcome.tier,
                (perf_counter() - started_at) * 1000,
                outcome.error_type,
                outcome.quality_passed,
            )

        immutable_outcomes = tuple(outcomes)
        metric_results = tuple(
            metric.calculate(dataset, immutable_outcomes) for metric in metrics
        )
        strategy_results.append(
            StrategyResult(
                name=strategy.name,
                outcomes=immutable_outcomes,
                metrics=metric_results,
            )
        )

    now = datetime.now(UTC)
    return BenchmarkResult(
        run_id=now.strftime("%Y%m%dT%H%M%S%fZ"),
        created_at=now.isoformat(),
        dataset_name=dataset.name,
        item_count=len(dataset.items),
        strategies=tuple(strategy_results),
    )
