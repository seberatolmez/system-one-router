"""Tests for evaluation data, strategy wiring, metrics, and generated reports."""

import json
from pathlib import Path

import pytest

from system_one.core.config import Settings
from system_one.domain.completions import (
    CompletionRequest,
    CompletionResponse,
)
from system_one.domain.decision import RoutingTier
from system_one.evaluation.cli import build_live_strategies
from system_one.evaluation.dataset import (
    DatasetFormatError,
    EvaluationDataset,
    EvaluationItem,
    load_dataset,
)
from system_one.evaluation.reporting import write_reports
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import OrchestratorStrategy, StrategyOutcome
from system_one.providers.errors import ProviderUnavailableError
from system_one.routing.decision.internal import InternalDecisionEngine
from system_one.routing.orchestrator import VIRTUAL_MODEL_NAMES, RoutingOrchestrator
from system_one.routing.policy.config import ComplexityRange, PolicyConfig
from system_one.routing.policy.engine import DeterministicPolicyEngine
from system_one.routing.registry.models import ModelCost, ModelProfile
from system_one.routing.registry.registry import ModelRegistry


def test_loader_validates_jsonl_items_and_preserves_optional_expected_tier(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    dataset_path.write_text(
        '\n{"id":"case-1","messages":[{"role":"user","content":"Hi"}],'
        '"expected_tier":"fast"}\n'
        '{"id":"case-2","messages":[{"role":"user","content":"Hello"}]}\n',
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_path)

    assert dataset.name == "examples"
    assert [item.id for item in dataset.items] == ["case-1", "case-2"]
    assert dataset.items[0].expected_tier == "fast"
    assert dataset.items[1].expected_tier is None


def test_included_sample_dataset_is_loadable() -> None:
    dataset = load_dataset("evaluations/datasets/sample.jsonl")

    assert len(dataset.items) == 6
    assert {item.expected_tier for item in dataset.items} == {
        "fast",
        "balanced",
        "reasoning",
    }


@pytest.mark.parametrize(
    "content",
    [
        '{"id":"same","messages":[{"role":"user","content":"Hi"}]}\n'
        '{"id":"same","messages":[{"role":"user","content":"Again"}]}\n',
        '{"id":"bad","messages":[],"expected_tier":"fast"}\n',
        '{"id":"bad","messages":[{"role":"user","content":"Hi"}],'
        '"expected_tier":"unknown"}\n',
        '{"id":"bad","messages":[{"role":"user","content":2}]}\n',
        '{not json}\n',
    ],
)
def test_loader_rejects_malformed_or_ambiguous_records(
    tmp_path: Path,
    content: str,
) -> None:
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetFormatError):
        load_dataset(dataset_path)


@pytest.mark.asyncio
async def test_runner_compares_same_items_and_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    dataset = EvaluationDataset(
        name="sample",
        items=(
            EvaluationItem(
                id="expected-balanced",
                messages=(),
                expected_tier="balanced",
            ),
            EvaluationItem(
                id="expected-fast",
                messages=(),
                expected_tier="fast",
            ),
        ),
    )
    provider = FakeProvider()
    orchestrator = RoutingOrchestrator(
        decision_engine=InternalDecisionEngine(),
        policy_engine=DeterministicPolicyEngine(_policy_config()),
        model_registry=_registry(),
        provider=provider,
        policy_name="test-policy",
    )
    strategies = (
        OrchestratorStrategy("system-one/auto", "system-one/auto", orchestrator),
        OrchestratorStrategy("system-one/fast", "system-one/fast", orchestrator),
    )

    result = await run_benchmark(dataset, strategies)
    json_path, markdown_path = write_reports(result, tmp_path / "reports")

    assert [strategy.name for strategy in result.strategies] == [
        "system-one/auto",
        "system-one/fast",
    ]
    assert [
        [outcome.item_id for outcome in strategy.outcomes]
        for strategy in result.strategies
    ] == [["expected-balanced", "expected-fast"], ["expected-balanced", "expected-fast"]]
    auto_outcomes = result.strategies[0].outcomes
    assert [outcome.tier for outcome in auto_outcomes] == ["balanced", "balanced"]
    assert all(outcome.estimated_cost == pytest.approx(0.000088) for outcome in auto_outcomes)
    fast_outcomes = result.strategies[1].outcomes
    assert all(outcome.error_type == "ProviderUnavailableError" for outcome in fast_outcomes)
    metrics = {metric.name: metric for metric in result.strategies[0].metrics}
    assert metrics["routing_accuracy"].value == pytest.approx(0.5)
    assert metrics["error_rate"].value == 0.0
    assert metrics["escalation_rate"].value == pytest.approx(0.5)

    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    assert serialized["dataset_name"] == "sample"
    assert len(serialized["strategies"]) == 2
    summary = markdown_path.read_text(encoding="utf-8")
    assert "| Strategy | routing_accuracy |" in summary
    assert "system-one/auto" in summary
    assert "sample report is an evaluation artifact" in summary


@pytest.mark.asyncio
async def test_runner_records_strategy_failure_and_continues() -> None:
    dataset = EvaluationDataset(
        name="sample",
        items=(
            EvaluationItem("one", (), "balanced"),
            EvaluationItem("two", (), "fast"),
        ),
    )
    strategy = FailingStrategy()
    second_strategy = WorkingStrategy()

    result = await run_benchmark(dataset, (strategy, second_strategy))

    assert [outcome.error_type for outcome in result.strategies[0].outcomes] == [
        "RuntimeError",
        "RuntimeError",
    ]
    assert [outcome.item_id for outcome in result.strategies[1].outcomes] == ["one", "two"]
    error_rate = result.strategies[0].metrics[4]
    assert error_rate.name == "error_rate"
    assert error_rate.value == 1.0


@pytest.mark.asyncio
async def test_runner_requires_multiple_unique_strategies() -> None:
    dataset = EvaluationDataset("sample", (EvaluationItem("one", (), None),))

    with pytest.raises(ValueError, match="at least two"):
        await run_benchmark(dataset, (WorkingStrategy(),))
    with pytest.raises(ValueError, match="unique"):
        await run_benchmark(dataset, (WorkingStrategy(), WorkingStrategy()))


def test_live_strategy_factory_reuses_configured_routing_stack() -> None:
    assert VIRTUAL_MODEL_NAMES == (
        "system-one/auto",
        "system-one/fast",
        "system-one/balanced",
        "system-one/reasoning",
    )
    settings = Settings(
        policy_file="policies/balanced.yaml",
        models_file="registry/models.yaml",
    )

    strategies = build_live_strategies(VIRTUAL_MODEL_NAMES[:2], settings)

    assert [(strategy.name, strategy.model) for strategy in strategies] == [
        ("system-one/auto", "system-one/auto"),
        ("system-one/fast", "system-one/fast"),
    ]
    with pytest.raises(ValueError, match="unsupported strategy"):
        build_live_strategies(("external/model",), settings)


class FakeProvider:
    """Return repeatable token usage without any external provider calls."""

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        if request.model.endswith("/fast"):
            raise ProviderUnavailableError("synthetic provider failure")
        return CompletionResponse(
            status_code=200,
            body={
                "id": "completion-test",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    async def stream(self, request: CompletionRequest) -> CompletionResponse:
        return await self.chat(request)

    async def list_models(self) -> CompletionResponse:
        return CompletionResponse(status_code=200, body={"data": []})


class FailingStrategy:
    name = "fails"

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        raise RuntimeError(f"synthetic failure for {item.id}")


class WorkingStrategy:
    name = "works"

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        tier: RoutingTier = "balanced"
        return StrategyOutcome(
            item_id=item.id,
            strategy_name=self.name,
            model_requested="system-one/balanced",
            model_selected="test/balanced",
            tier=tier,
            latency_ms=2.0,
            routing_latency_ms=0.5,
            provider_latency_ms=1.5,
            input_tokens=1,
            output_tokens=1,
            estimated_cost=0.000001,
            decision_confidence=None,
            status_code=200,
            error_type=None,
        )


def _policy_config() -> PolicyConfig:
    return PolicyConfig(
        confidence_threshold=0.85,
        fallback_tier="balanced",
        tiers={
            "fast": ComplexityRange(min_complexity=1.0, max_complexity=2.5),
            "balanced": ComplexityRange(min_complexity=2.5, max_complexity=4.5),
            "reasoning": ComplexityRange(min_complexity=4.5, max_complexity=5.0),
        },
    )


def _registry() -> ModelRegistry:
    cost = ModelCost(input_per_million=0.8, output_per_million=4.0)
    tiers: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")
    return ModelRegistry(
        ModelProfile(
            id=f"test/{tier}",
            provider="test",
            tier=tier,
            capabilities={},
            cost=cost,
        )
        for tier in tiers
    )
