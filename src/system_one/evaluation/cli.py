"""Command-line entry point for live System One routing evaluations."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from system_one.core.config import Settings, get_settings
from system_one.evaluation.dataset import load_dataset
from system_one.evaluation.reporting import write_reports
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import OrchestratorStrategy
from system_one.providers.openrouter import OpenRouterProvider
from system_one.routing.decision import InternalDecisionEngine, JevDecisionEngine
from system_one.routing.orchestrator import VIRTUAL_MODEL_NAMES, RoutingOrchestrator
from system_one.routing.policy import DeterministicPolicyEngine, load_policy_config
from system_one.routing.registry import load_model_registry

_DEFAULT_STRATEGIES = ("system-one/auto", "system-one/fast")


def build_live_strategies(
    model_names: Sequence[str],
    settings: Settings,
) -> tuple[OrchestratorStrategy, ...]:
    """Build configured live strategies sharing one provider and routing stack."""
    unsupported = set(model_names) - set(VIRTUAL_MODEL_NAMES)
    if unsupported:
        raise ValueError(f"unsupported strategy model(s): {', '.join(sorted(unsupported))}")

    provider = OpenRouterProvider(settings)
    policy_engine = DeterministicPolicyEngine(load_policy_config(settings.policy_file))
    model_registry = load_model_registry(settings.models_file)
    decision_engine = (
        JevDecisionEngine(settings) if settings.jev_enabled else InternalDecisionEngine()
    )
    orchestrator = RoutingOrchestrator(
        decision_engine=decision_engine,
        policy_engine=policy_engine,
        model_registry=model_registry,
        provider=provider,
        policy_name=Path(settings.policy_file).stem,
    )
    return tuple(
        OrchestratorStrategy(name=model_name, model=model_name, orchestrator=orchestrator)
        for model_name in model_names
    )


async def _run(args: argparse.Namespace) -> tuple[Path, Path]:
    settings = get_settings()
    dataset = load_dataset(args.dataset)
    strategies = build_live_strategies(args.strategy, settings)
    result = await run_benchmark(dataset, strategies)
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
        choices=VIRTUAL_MODEL_NAMES,
        help=(
            "virtual model strategy to compare; repeat this option. "
            "Defaults to system-one/auto and system-one/fast."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/reports"),
        help="directory for JSON runs and summary.md (default: %(default)s)",
    )
    args = parser.parse_args(argv)
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
