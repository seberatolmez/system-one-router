"""Score routers offline by replaying measured tier answers (RouterBench-style).

Inputs are two reports written by ``system_one.evaluation.cli``:

* a tier report: ``system-one/fast``, ``system-one/balanced`` and
  ``system-one/reasoning`` run on the dataset with grading enabled, and
* optionally a decision report: ``--decision-model`` runs on the same items.

Each router picks a tier per item; its outcome is the measured outcome of that
tier on that item, with the decision's latency and cost added. Labels are the
measured ones: the cheapest tier whose answer passed. So every router is
compared on identical answers and only the routing choice differs.
"""

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

from system_one.core.config import get_settings
from system_one.domain.decision import Decision, RoutingRequest, RoutingTier
from system_one.evaluation.dataset import EvaluationDataset, load_dataset
from system_one.evaluation.metrics import (
    BrierScoreMetric,
    CostPerSuccessfulTaskMetric,
    ErrorRateMetric,
    EscalationRateMetric,
    EvaluationMetric,
    LatencyPercentileMetric,
    QualityPassRateMetric,
    RoutingAccuracyMetric,
    UnderRoutingRateMetric,
)
from system_one.evaluation.oracle import cheapest_passing_tier
from system_one.evaluation.reporting import write_reports
from system_one.evaluation.runner import BenchmarkResult, StrategyResult
from system_one.evaluation.strategy import StrategyOutcome
from system_one.routing.decision.heuristic import HeuristicDecisionEngine
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config

TIER_STRATEGIES: dict[RoutingTier, str] = {
    "fast": "system-one/fast",
    "balanced": "system-one/balanced",
    "reasoning": "system-one/reasoning",
}

REPLAY_METRICS: tuple[EvaluationMetric, ...] = (
    QualityPassRateMetric(),
    CostPerSuccessfulTaskMetric(),
    RoutingAccuracyMetric(),
    UnderRoutingRateMetric(),
    EscalationRateMetric(),
    BrierScoreMetric(),
    LatencyPercentileMetric(50),
    LatencyPercentileMetric(95),
    ErrorRateMetric(),
)

Outcomes = dict[str, dict[str, StrategyOutcome]]


def load_outcomes(path: str | Path) -> Outcomes:
    """Read a report JSON into ``{strategy: {item_id: outcome}}``."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {field.name for field in fields(StrategyOutcome)}
    return {
        strategy["name"]: {
            outcome["item_id"]: StrategyOutcome(
                **{key: value for key, value in outcome.items() if key in names}
            )
            for outcome in strategy["outcomes"]
        }
        for strategy in document["strategies"]
    }


def measured_labels(tiers: Outcomes, dataset: EvaluationDataset) -> EvaluationDataset:
    """Replace each item's expected tier with the cheapest tier that passed."""
    items = tuple(
        replace(
            item,
            expected_tier=cheapest_passing_tier(
                [tiers[TIER_STRATEGIES[tier]][item.id].quality_passed for tier in TIER_STRATEGIES]
            ),
        )
        for item in dataset.items
    )
    return EvaluationDataset(name=f"{dataset.name}-measured", items=items)


def replay_router(
    name: str,
    choices: Mapping[str, tuple[RoutingTier, StrategyOutcome | None]],
    tiers: Outcomes,
    dataset: EvaluationDataset,
) -> StrategyResult:
    """Build a router's outcomes from its per-item tier choice and the measured tier runs.

    ``choices`` maps item id to (chosen tier, decision outcome or None). Latency
    and cost add the decision's; confidence and decision source come from the
    decision. A decision error stays visible as the outcome's ``error_type``.
    """
    outcomes = []
    for item in dataset.items:
        tier, decision = choices[item.id]
        answer = tiers[TIER_STRATEGIES[tier]][item.id]
        decision_ms = decision.latency_ms if decision is not None else 0.0
        decision_cost = decision.decision_cost if decision is not None else None
        cost = answer.estimated_cost
        outcomes.append(
            replace(
                answer,
                strategy_name=name,
                tier=tier,
                latency_ms=answer.latency_ms + decision_ms,
                routing_latency_ms=decision_ms,
                estimated_cost=cost + (decision_cost or 0.0) if cost is not None else None,
                decision_confidence=decision.decision_confidence if decision else None,
                decision_source=decision.decision_source if decision else None,
                decision_cost=decision_cost,
                error_type=(decision.error_type if decision else None) or answer.error_type,
            )
        )
    immutable = tuple(outcomes)
    metrics = tuple(metric.calculate(dataset, immutable) for metric in REPLAY_METRICS)
    return StrategyResult(name=name, outcomes=immutable, metrics=metrics)


async def heuristic_tiers(
    dataset: EvaluationDataset, policy: DeterministicPolicyEngine
) -> dict[str, RoutingTier]:
    """Route every item with the heuristic baseline; it makes no network call."""
    engine = HeuristicDecisionEngine()
    tiers: dict[str, RoutingTier] = {}
    for item in dataset.items:
        decision: Decision = await engine.decide(
            RoutingRequest(model="system-one/auto", messages=item.messages)
        )
        tiers[item.id] = policy.evaluate(decision).tier
    return tiers


def _replayed_tier(
    decision: StrategyOutcome, policy: DeterministicPolicyEngine, fallback_tier: RoutingTier
) -> RoutingTier:
    """Re-run the replay policy on the recorded decision; a failed one gets the fallback tier."""
    if decision.decision_complexity is None or decision.decision_confidence is None:
        return decision.tier or fallback_tier
    recorded = Decision(
        task_type="recorded",
        complexity=decision.decision_complexity,
        quality_requirement=3.0,
        latency_requirement=3.0,
        confidence=decision.decision_confidence,
    )
    return policy.evaluate(recorded).tier


async def replay(
    dataset: EvaluationDataset,
    tiers: Outcomes,
    decisions: Outcomes,
    policy: DeterministicPolicyEngine,
    fallback_tier: RoutingTier,
) -> BenchmarkResult:
    """Score always-fast, always-strong, heuristic and every decision model."""
    missing = [
        f"{strategy}/{item.id}"
        for strategy in TIER_STRATEGIES.values()
        for item in dataset.items
        if item.id not in tiers.get(strategy, {})
    ]
    if missing:
        raise ValueError(f"tier report lacks outcomes for: {', '.join(missing[:5])}")
    labeled = measured_labels(tiers, dataset)

    results = [
        replay_router("always-fast", {i.id: ("fast", None) for i in labeled.items}, tiers, labeled),
        replay_router(
            "always-balanced", {i.id: ("balanced", None) for i in labeled.items}, tiers, labeled
        ),
        replay_router(
            "always-strong", {i.id: ("reasoning", None) for i in labeled.items}, tiers, labeled
        ),
    ]
    heuristic = await heuristic_tiers(labeled, policy)
    results.append(
        replay_router(
            "heuristic", {i.id: (heuristic[i.id], None) for i in labeled.items}, tiers, labeled
        )
    )
    for model, by_item in decisions.items():
        choices: dict[str, tuple[RoutingTier, StrategyOutcome | None]] = {}
        for item in labeled.items:
            decision = by_item.get(item.id)
            if decision is None:
                raise ValueError(f"decision report for {model!r} lacks item {item.id!r}")
            choices[item.id] = (_replayed_tier(decision, policy, fallback_tier), decision)
        results.append(replay_router(model, choices, tiers, labeled))

    now = datetime.now(UTC)
    return BenchmarkResult(
        run_id=now.strftime("%Y%m%dT%H%M%S%fZ"),
        created_at=now.isoformat(),
        dataset_name=labeled.name,
        item_count=len(labeled.items),
        strategies=tuple(results),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Replay routers against measured tier answers and write JSON plus Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tier-report", type=Path, required=True)
    parser.add_argument("--decision-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/reports"))
    parser.add_argument(
        "--policy",
        type=Path,
        help="policy file to route recorded decisions with (default: the configured policy)",
    )
    args = parser.parse_args(argv)

    config = load_policy_config(args.policy or get_settings().policy_file)
    dataset = load_dataset(args.dataset)
    decisions = load_outcomes(args.decision_report) if args.decision_report else {}
    result = asyncio.run(
        replay(
            dataset,
            load_outcomes(args.tier_report),
            decisions,
            DeterministicPolicyEngine(config),
            config.fallback_tier,
        )
    )
    json_path, markdown_path = write_reports(result, args.output_dir)
    print(f"Wrote JSON results: {json_path}")
    print(f"Wrote Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
