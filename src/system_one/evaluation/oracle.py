"""Relabel a dataset with measured tiers: the cheapest tier whose answer passes grading."""

import argparse
import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from system_one.core.config import get_settings
from system_one.domain.decision import RoutingTier
from system_one.evaluation.cli import build_grader, build_live_strategies
from system_one.evaluation.dataset import (
    EvaluationDataset,
    load_dataset,
    write_dataset,
)
from system_one.evaluation.grading import QualityGrader
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import BenchmarkStrategy

_TIERS: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")


async def measure_expected_tiers(
    dataset: EvaluationDataset,
    strategies: Sequence[BenchmarkStrategy],
    grader: QualityGrader,
) -> EvaluationDataset:
    """Label each item with the cheapest tier that passed, running tiers cheapest first.

    ``strategies`` are the fast, balanced and reasoning strategies in that order.
    An item is left unlabeled when a cheaper tier could not be graded, or when no
    tier passed, rather than guessing.
    """
    if len(strategies) != len(_TIERS):
        raise ValueError("oracle labeling needs exactly the fast, balanced, reasoning strategies")
    result = await run_benchmark(dataset, strategies, metrics=(), grader=grader)
    verdicts = [
        {outcome.item_id: outcome.quality_passed for outcome in strategy.outcomes}
        for strategy in result.strategies
    ]
    relabeled = []
    for item in dataset.items:
        label: RoutingTier | None = None
        for tier, tier_verdicts in zip(_TIERS, verdicts, strict=True):
            verdict = tier_verdicts[item.id]
            if verdict is None:
                break
            if verdict:
                label = tier
                break
        relabeled.append(replace(item, expected_tier=label))
    return EvaluationDataset(name=dataset.name, items=tuple(relabeled))


def main(argv: Sequence[str] | None = None) -> None:
    """Run all three tiers over a dataset and write a copy with measured labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="relabeled JSONL to write")
    parser.add_argument(
        "--judge-model",
        help="chat model that grades items without a reference answer (else they stay unlabeled)",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    if not settings.openrouter_api_key:
        parser.error("OPENROUTER_API_KEY is required for oracle labeling")

    strategies = build_live_strategies([f"system-one/{tier}" for tier in _TIERS], settings)
    grader = build_grader(settings, args.judge_model)
    relabeled = asyncio.run(measure_expected_tiers(load_dataset(args.dataset), strategies, grader))
    write_dataset(relabeled, args.output)
    counts = Counter(item.expected_tier or "unlabeled" for item in relabeled.items)
    print(f"Wrote {args.output}: {dict(counts)}")


if __name__ == "__main__":
    main()
