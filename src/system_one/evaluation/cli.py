"""Command-line entry point for live System One routing evaluations."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx

from system_one.core.config import Settings, get_settings
from system_one.evaluation.dataset import load_dataset
from system_one.evaluation.grading import ReferenceOrJudgeGrader
from system_one.evaluation.metrics import DECISION_METRICS, DEFAULT_METRICS
from system_one.evaluation.reporting import write_reports
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import (
    BenchmarkStrategy,
    DecisionStrategy,
    OrchestratorStrategy,
)
from system_one.providers.openrouter import OpenRouterProvider
from system_one.routing.decision import (
    DecisionEngine,
    HeuristicDecisionEngine,
    InternalDecisionEngine,
    JevDecisionEngine,
)
from system_one.routing.orchestrator import VIRTUAL_MODEL_NAMES, RoutingOrchestrator
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry import load_model_registry

_DEFAULT_STRATEGIES = ("system-one/auto", "system-one/fast")
HEURISTIC_STRATEGY = "heuristic"
STRATEGY_NAMES = (*VIRTUAL_MODEL_NAMES, HEURISTIC_STRATEGY)


def build_live_strategies(
    model_names: Sequence[str],
    settings: Settings,
) -> tuple[OrchestratorStrategy, ...]:
    """Build configured live strategies sharing one provider and routing stack.

    ``heuristic`` routes ``system-one/auto`` through the keyword/length baseline
    instead of the configured decision engine.
    """
    unsupported = set(model_names) - set(STRATEGY_NAMES)
    if unsupported:
        raise ValueError(f"unsupported strategy model(s): {', '.join(sorted(unsupported))}")

    provider = OpenRouterProvider(settings)
    policy_engine = DeterministicPolicyEngine(load_policy_config(settings.policy_file))
    model_registry = load_model_registry(settings.models_file)

    def orchestrator(decision_engine: DecisionEngine) -> RoutingOrchestrator:
        return RoutingOrchestrator(
            decision_engine=decision_engine,
            policy_engine=policy_engine,
            model_registry=model_registry,
            provider=provider,
            policy_name=Path(settings.policy_file).stem,
        )

    primary = orchestrator(
        JevDecisionEngine(settings) if settings.jev_enabled else InternalDecisionEngine()
    )
    heuristic = orchestrator(HeuristicDecisionEngine())
    return tuple(
        OrchestratorStrategy(
            name=HEURISTIC_STRATEGY, model="system-one/auto", orchestrator=heuristic
        )
        if model_name == HEURISTIC_STRATEGY
        else OrchestratorStrategy(name=model_name, model=model_name, orchestrator=primary)
        for model_name in model_names
    )


def build_grader(settings: Settings, judge_model: str | None) -> ReferenceOrJudgeGrader:
    """Grade by exact match where a reference exists, and with the judge model otherwise."""
    return ReferenceOrJudgeGrader(OpenRouterProvider(settings), judge_model)


def build_decision_strategies(
    model_slugs: Sequence[str],
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[DecisionStrategy, ...]:
    """Build one decision-only strategy per decision-model slug, sharing one policy."""
    policy_engine = DeterministicPolicyEngine(load_policy_config(settings.policy_file))
    return tuple(
        DecisionStrategy(
            name=slug,
            engine=JevDecisionEngine(settings.model_copy(update={"jev_model": slug}), transport),
            policy_engine=policy_engine,
        )
        for slug in model_slugs
    )


async def _run(args: argparse.Namespace) -> tuple[Path, Path]:
    settings = get_settings()
    dataset = load_dataset(args.dataset)
    strategies: Sequence[BenchmarkStrategy]
    if args.decision_model:
        strategies = build_decision_strategies(args.decision_model, settings)
        result = await run_benchmark(dataset, strategies, DECISION_METRICS)
    else:
        strategies = build_live_strategies(args.strategy, settings)
        grader = build_grader(settings, args.judge_model)
        result = await run_benchmark(dataset, strategies, DEFAULT_METRICS, grader)
    return write_reports(result, args.output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments, run the benchmark, and print report locations."""
    parser = argparse.ArgumentParser(
        description="Run one evaluation dataset through multiple System One strategies."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluations/datasets/sample.jsonl"),
        help="JSONL evaluation dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGY_NAMES,
        help=(
            "strategy to compare; repeat this option. 'heuristic' is the keyword/length "
            "baseline. Defaults to system-one/auto and system-one/fast."
        ),
    )
    parser.add_argument(
        "--judge-model",
        help=(
            "chat model that grades answers for items without a reference answer. "
            "Without it, only items with a reference are graded."
        ),
    )
    parser.add_argument(
        "--decision-model",
        action="append",
        help=(
            "decision-model slug to benchmark on routing decisions only, e.g. "
            "typesafe/jev-1.14; repeat this option. Cannot be combined with --strategy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/reports"),
        help="directory for JSON runs and summary.md (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.decision_model:
        if args.strategy or args.judge_model:
            parser.error("--decision-model cannot be combined with --strategy or --judge-model")
        if len(args.decision_model) < 2:
            parser.error("at least two --decision-model options are required for comparison")
        if len(args.decision_model) != len(set(args.decision_model)):
            parser.error("--decision-model values must be unique")
        if not get_settings().openrouter_api_key:
            parser.error("OPENROUTER_API_KEY is required to benchmark decision models")
    else:
        args.strategy = args.strategy or list(_DEFAULT_STRATEGIES)
        if len(args.strategy) < 2:
            parser.error("at least two --strategy options are required for comparison")
        if len(args.strategy) != len(set(args.strategy)):
            parser.error("--strategy values must be unique")

    json_path, markdown_path = asyncio.run(_run(args))
    print(f"Wrote JSON results: {json_path}")
    print(f"Wrote Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
