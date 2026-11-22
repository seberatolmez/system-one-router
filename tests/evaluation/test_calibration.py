"""Tests for policy calibration: split, search, held-out comparison and the CLI."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from system_one.domain.completions import CompletionMessage
from system_one.evaluation import calibration
from system_one.evaluation.calibration import (
    ALWAYS_BALANCED,
    CalibrationItem,
    Policy,
    build_items,
    candidate_policies,
    choose_policy,
    compare,
    policy_yaml,
    route,
    score,
    split_ids,
)
from system_one.evaluation.dataset import EvaluationDataset, EvaluationItem, write_dataset
from system_one.evaluation.strategy import StrategyOutcome
from system_one.routing.policy import load_policy_config


def _item(
    item_id: str = "a",
    *,
    complexity: float | None = 2.0,
    confidence: float | None = 0.9,
    fast_passed: bool = True,
    balanced_passed: bool = True,
    fast_cost: float = 0.001,
    balanced_cost: float = 0.003,
    decision_cost: float = 0.00001,
) -> CalibrationItem:
    return CalibrationItem(
        item_id=item_id,
        fast_passed=fast_passed,
        balanced_passed=balanced_passed,
        fast_cost=fast_cost,
        balanced_cost=balanced_cost,
        decision_cost=decision_cost,
        complexity=complexity,
        confidence=confidence,
    )


def _dataset(per_cell: int = 20) -> EvaluationDataset:
    items = [
        EvaluationItem(
            f"{task}-{diff}-{n:02d}",
            (CompletionMessage("user", "q"),),
            None,
            task_type=task,
            difficulty=diff,
        )
        for task in ("coding", "analysis")
        for diff in ("easy", "hard")
        for n in range(per_cell)
    ]
    return EvaluationDataset("d", tuple(items))


def test_split_is_stratified_disjoint_complete_and_deterministic() -> None:
    dataset = _dataset()

    train, test = split_ids(dataset, 0.35, seed=1)

    assert train.isdisjoint(test)
    assert train | test == {item.id for item in dataset.items}
    for task in ("coding", "analysis"):
        for diff in ("easy", "hard"):
            assert len({i for i in test if i.startswith(f"{task}-{diff}-")}) == 7
    assert split_ids(dataset, 0.35, seed=1) == (train, test)
    assert split_ids(dataset, 0.35, seed=2) != (train, test)


def test_route_applies_threshold_gate_and_failed_decision_fallback() -> None:
    policy = Policy(fast_below=2.5, confidence_gate=0.6)

    assert route(policy, _item(complexity=2.0, confidence=0.9)) == "fast"
    assert route(policy, _item(complexity=2.5, confidence=0.9)) == "balanced"
    assert route(policy, _item(complexity=2.0, confidence=0.5)) == "balanced"
    assert route(policy, _item(complexity=None, confidence=None)) == "balanced"
    assert route(ALWAYS_BALANCED, _item(complexity=1.0, confidence=1.0)) == "balanced"


def test_score_counts_decision_cost_only_for_routers() -> None:
    items = [_item("a", complexity=1.5), _item("b", complexity=4.0, fast_passed=False)]
    policy = Policy(fast_below=2.5, confidence_gate=0.0)

    router = score(items, policy, with_decision=True)
    baseline = score(items, ALWAYS_BALANCED, with_decision=False)

    assert router.quality == 1.0 and router.fast_share == 0.5
    assert router.cost_per_success == pytest.approx((0.001 + 0.003 + 2 * 0.00001) / 2)
    assert baseline.cost_per_success == pytest.approx(0.003)
    assert (
        score(
            [_item(fast_passed=False, balanced_passed=False)], policy, with_decision=True
        ).cost_per_success
        is None
    )


def test_candidates_are_midpoints_inside_the_scale_with_a_zero_gate() -> None:
    train = [_item(str(k), complexity=1.0 + k * 0.1, confidence=0.5 + k * 0.01) for k in range(30)]

    candidates = candidate_policies(train)

    thresholds = {p.fast_below for p in candidates}
    assert all(1.0 < t < calibration.REASONING_FLOOR for t in thresholds)
    assert 1.05 in thresholds and len(thresholds) <= 40
    assert 0.0 in {p.confidence_gate for p in candidates}
    assert len({p.confidence_gate for p in candidates}) == 4


def _separable(n: int = 40, *, fast_cheap: bool = True) -> list[CalibrationItem]:
    easy = [
        _item(f"e{k}", complexity=1.5, fast_cost=0.001 if fast_cheap else 0.006)
        for k in range(n // 2)
    ]
    hard = [
        _item(f"h{k}", complexity=4.0, fast_passed=False, fast_cost=0.004) for k in range(n // 2)
    ]
    return easy + hard


def test_choose_policy_finds_the_threshold_that_separates_easy_from_hard() -> None:
    policy = choose_policy(_separable(), train_margin=0.03)

    assert policy.fast_below == 2.75


def test_choose_policy_keeps_balanced_when_fast_is_never_cheaper() -> None:
    assert choose_policy(_separable(fast_cheap=False), train_margin=0.03) == ALWAYS_BALANCED


def test_choose_policy_refuses_savings_that_cost_too_much_quality() -> None:
    # fast is cheaper for everyone but fails on every hard item: routing all to fast is
    # cheapest per call, yet drops quality by 50 points
    items = _separable()

    chosen = choose_policy(items, train_margin=0.03)

    assert score(items, chosen, with_decision=True).quality >= 1.0 - 0.03


def test_compare_declares_a_win_only_for_a_real_saving_without_a_quality_loss() -> None:
    policy = Policy(fast_below=2.75, confidence_gate=0.0)
    test = _separable(100)

    result = compare(test, policy, seed=7, margin=0.05, min_saving=0.10)

    assert result.wins
    assert result.cost_change == pytest.approx(-0.33, abs=1e-9)
    assert result.quality_difference == 0.0
    assert result.cost_change_ci is not None
    assert result.cost_change_ci[1] < -0.10
    assert compare(test, policy, seed=7, margin=0.05, min_saving=0.10) == result


def test_compare_rejects_the_baseline_a_quality_loss_and_a_small_saving() -> None:
    test = _separable(100)

    assert not compare(test, ALWAYS_BALANCED, seed=1, margin=0.05, min_saving=0.10).wins
    sloppy = Policy(fast_below=4.5, confidence_gate=0.0)  # sends hard items to fast, they fail
    lossy = compare(test, sloppy, seed=1, margin=0.05, min_saving=0.10)
    assert not lossy.wins and lossy.quality_difference_ci[1] < -0.05
    tiny_saving = _separable(100, fast_cheap=True)
    for index, item in enumerate(tiny_saving):
        tiny_saving[index] = _item(
            item.item_id, complexity=item.complexity, fast_passed=item.fast_passed, fast_cost=0.0029
        )
    assert not compare(tiny_saving, Policy(2.75, 0.0), seed=1, margin=0.05, min_saving=0.10).wins


def test_policy_yaml_is_a_valid_policy_file(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(policy_yaml(Policy(2.25, 0.6), "m/one"), encoding="utf-8")

    config = load_policy_config(path)

    assert config.confidence_threshold == 0.6
    assert config.tiers["fast"].max_complexity == 2.25
    assert config.tiers["balanced"].max_complexity == calibration.REASONING_FLOOR
    assert config.tiers["reasoning"].max_complexity == 5.0


def _outcome(strategy: str, item_id: str, **fields: object) -> StrategyOutcome:
    values: dict[str, object] = {
        "item_id": item_id,
        "strategy_name": strategy,
        "model_requested": strategy,
        "model_selected": None,
        "tier": None,
        "latency_ms": 1.0,
        "routing_latency_ms": 0.0,
        "provider_latency_ms": 1.0,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost": 0.001,
        "decision_confidence": None,
        "status_code": 200,
        "error_type": None,
    }
    values.update(fields)
    return StrategyOutcome(**values)  # type: ignore[arg-type]


def test_build_items_reports_why_items_were_dropped() -> None:
    dataset = EvaluationDataset("d", tuple(EvaluationItem(i, (), None) for i in "abcde"))
    ok = {"quality_passed": True, "estimated_cost": 0.002}
    tiers = {
        "system-one/fast": {
            "a": _outcome("f", "a", **ok),
            "b": _outcome("f", "b", **ok),
            "c": _outcome("f", "c", quality_passed=None, estimated_cost=0.002),
            "d": _outcome("f", "d", quality_passed=True, estimated_cost=None),
            "e": _outcome("f", "e", **ok),
        },
        "system-one/balanced": {i: _outcome("b", i, **ok) for i in "abcde"},
    }
    decision = {"decision_complexity": 2.0, "decision_confidence": 0.8, "decision_cost": 0.00002}
    decisions = {
        "a": _outcome("m", "a", **decision),
        "c": _outcome("m", "c", **decision),
        "d": _outcome("m", "d", **decision),
        "e": _outcome("m", "e", **{**decision, "decision_complexity": 4.95}),
    }

    items, dropped = build_items(dataset, tiers, decisions)

    assert [i.item_id for i in items] == ["a"]
    assert sorted(dropped) == [
        "b: missing outcome",
        "c: ungraded answer",
        "d: answer cost unmeasured",
        "e: decision reaches the unmeasured reasoning band",
    ]


def test_cli_end_to_end_on_synthetic_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _dataset(per_cell=10)
    write_dataset(dataset, "data.jsonl")
    fast, balanced, decisions = {}, {}, {}
    for item in dataset.items:
        easy = item.difficulty == "easy"
        fast[item.id] = _outcome("f", item.id, quality_passed=easy, estimated_cost=0.001)
        balanced[item.id] = _outcome("b", item.id, quality_passed=True, estimated_cost=0.003)
        decisions[item.id] = _outcome(
            "m/good",
            item.id,
            tier="fast",
            decision_complexity=1.5 if easy else 4.0,
            decision_confidence=0.9,
            decision_cost=0.00001,
        )

    def report(name: str, outcomes: dict[str, dict[str, StrategyOutcome]]) -> None:
        document = {
            "strategies": [
                {"name": n, "outcomes": [asdict(o) for o in by.values()], "metrics": []}
                for n, by in outcomes.items()
            ]
        }
        Path(name).write_text(json.dumps(document), encoding="utf-8")

    report("tiers.json", {"system-one/fast": fast, "system-one/balanced": balanced})
    report("decisions.json", {"m/good": decisions})

    calibration.main(
        [
            "--dataset",
            "data.jsonl",
            "--tier-report",
            "tiers.json",
            "--decision-report",
            "decisions.json",
            "--output-dir",
            "out",
        ]
    )

    markdown = Path("out/calibration.md").read_text(encoding="utf-8")
    assert "`m/good`" in markdown and "**yes**" in markdown
    saved = json.loads(Path("out/calibration.json").read_text(encoding="utf-8"))
    assert saved["results"][0]["comparison"]["wins"] is True
    config = load_policy_config("out/m-good.yaml")
    assert config.tiers["fast"].max_complexity == 2.75


def test_cli_refuses_an_output_directory_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    write_dataset(_dataset(per_cell=2), "data.jsonl")
    Path("r.json").write_text(json.dumps({"strategies": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        calibration.main(
            [
                "--dataset",
                "data.jsonl",
                "--tier-report",
                "r.json",
                "--decision-report",
                "r.json",
                "--output-dir",
                "../escaped",
            ]
        )
