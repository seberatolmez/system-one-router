"""Tests for resumable benchmark runs."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from system_one.evaluation import cli
from system_one.evaluation.dataset import EvaluationDataset, EvaluationItem
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import StrategyOutcome


def _outcome(strategy: str, item_id: str, error_type: str | None = None) -> StrategyOutcome:
    return StrategyOutcome(
        item_id=item_id,
        strategy_name=strategy,
        model_requested=strategy,
        model_selected=None,
        tier="fast",
        latency_ms=1.0,
        routing_latency_ms=0.0,
        provider_latency_ms=1.0,
        input_tokens=None,
        output_tokens=None,
        estimated_cost=0.001,
        decision_confidence=None,
        status_code=200,
        error_type=error_type,
    )


class CountingStrategy:
    def __init__(self, name: str, crash_on: str | None = None, fail_once: set[str] | None = None):
        self.name = name
        self.calls: list[str] = []
        self._crash_on = crash_on
        self._fail_once = set(fail_once or ())

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        if item.id == self._crash_on:
            raise SystemExit("simulated kill")  # not an Exception: the runner must not swallow it
        self.calls.append(item.id)
        if item.id in self._fail_once:
            self._fail_once.discard(item.id)
            return _outcome(self.name, item.id, error_type="ProviderUnavailableError")
        return _outcome(self.name, item.id)


def _write(name: str, text: str) -> None:
    Path(name).write_text(text, encoding="utf-8")


def _dataset(n: int = 4) -> EvaluationDataset:
    return EvaluationDataset("d", tuple(EvaluationItem(f"i{k}", (), None) for k in range(n)))


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


async def test_interrupted_run_resumes_without_repeating_finished_items() -> None:
    crashing = (CountingStrategy("a", crash_on="i2"), CountingStrategy("b"))
    with pytest.raises(SystemExit):
        await run_benchmark(_dataset(), crashing, metrics=(), checkpoint="run.jsonl")
    assert crashing[0].calls == ["i0", "i1"]

    resumed = (CountingStrategy("a"), CountingStrategy("b"))
    result = await run_benchmark(_dataset(), resumed, metrics=(), checkpoint="run.jsonl")

    assert resumed[0].calls == ["i2", "i3"]
    assert resumed[1].calls == ["i0", "i1", "i2", "i3"]
    assert [o.item_id for o in result.strategies[0].outcomes] == ["i0", "i1", "i2", "i3"]
    assert all(o.error_type is None for s in result.strategies for o in s.outcomes)


async def test_resume_retries_errored_items_only() -> None:
    first = (CountingStrategy("a", fail_once={"i1"}), CountingStrategy("b"))
    await run_benchmark(_dataset(3), first, metrics=(), checkpoint="run.jsonl")

    second = (CountingStrategy("a"), CountingStrategy("b"))
    result = await run_benchmark(_dataset(3), second, metrics=(), checkpoint="run.jsonl")

    assert second[0].calls == ["i1"]
    assert second[1].calls == []
    assert next(o for o in result.strategies[0].outcomes if o.item_id == "i1").error_type is None


async def test_unreadable_checkpoint_lines_are_skipped() -> None:
    good = {"strategy": "a", "outcome": asdict(_outcome("a", "i0"))}
    _write("run.jsonl", json.dumps(good) + '\n{"strategy": "a", "outc')

    strategies = (CountingStrategy("a"), CountingStrategy("b"))
    await run_benchmark(_dataset(2), strategies, metrics=(), checkpoint="run.jsonl")

    assert strategies[0].calls == ["i1"]


async def test_checkpoint_from_a_different_run_is_rejected() -> None:
    other = {"strategy": "zzz", "outcome": asdict(_outcome("zzz", "i0"))}
    _write("run.jsonl", json.dumps(other) + "\n")

    with pytest.raises(ValueError, match="does not belong to this run"):
        await run_benchmark(
            _dataset(2),
            (CountingStrategy("a"), CountingStrategy("b")),
            metrics=(),
            checkpoint="run.jsonl",
        )


async def test_checkpoint_outside_the_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    os.chdir(workspace)

    with pytest.raises(ValueError, match="outside the working directory"):
        await run_benchmark(
            _dataset(2),
            (CountingStrategy("a"), CountingStrategy("b")),
            metrics=(),
            checkpoint="../escaped.jsonl",
        )


async def test_without_a_checkpoint_nothing_is_written() -> None:
    await run_benchmark(_dataset(2), (CountingStrategy("a"), CountingStrategy("b")), metrics=())

    assert os.listdir(".") == []


def test_cli_checkpoint_flag_is_parsed_and_rejects_decision_conflicts() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--checkpoint", "x.jsonl", "--decision-model", "a/one"])  # needs two models
