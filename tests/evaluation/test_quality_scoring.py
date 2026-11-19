"""Tests for quality grading, quality metrics, the heuristic baseline, and oracle labels."""

from pathlib import Path

import pytest

from system_one.core.config import Settings
from system_one.domain.completions import (
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
)
from system_one.domain.decision import RoutingRequest, RoutingTier
from system_one.evaluation.cli import build_live_strategies
from system_one.evaluation.dataset import (
    EvaluationDataset,
    EvaluationItem,
    load_dataset,
    write_dataset,
)
from system_one.evaluation.grading import ReferenceOrJudgeGrader, matches_reference
from system_one.evaluation.metrics import CostPerSuccessfulTaskMetric, QualityPassRateMetric
from system_one.evaluation.oracle import measure_expected_tiers
from system_one.evaluation.runner import run_benchmark
from system_one.evaluation.strategy import StrategyOutcome
from system_one.routing.decision.heuristic import HeuristicDecisionEngine

_TIERS: tuple[RoutingTier, ...] = ("fast", "balanced", "reasoning")


@pytest.mark.parametrize(
    ("reference", "response", "expected"),
    [
        ("18", "She makes $18 every day.", True),
        ("18", "9 eggs, so 9 * 2 = 18.", True),
        ("18", "18 is wrong, the answer is 19.", False),
        ("70000", "Profit: 70,000 dollars.", True),
        ("2.5", "2.50", True),
        ("18", "no number here", False),
        ("C", "C", True),
        ("C", " (C). ", True),
        ("C", "The answer is C", True),
        ("C", "A is tempting but the answer is: C", True),
        ("C", "B", False),
        ("C", "A cat sat on a mat", False),
        ("Paris", " paris ", True),
        ("Paris", "London", False),
    ],
)
def test_matches_reference(reference: str, response: str, expected: bool) -> None:
    assert matches_reference(reference, response) is expected


class JudgeProvider:
    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.requests: list[CompletionRequest] = []

    async def chat(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self.reply is None:
            return CompletionResponse(status_code=500, body={"error": {"message": "boom"}})
        message = {"content": self.reply}
        return CompletionResponse(status_code=200, body={"choices": [{"message": message}]})

    async def stream(self, request: CompletionRequest) -> CompletionResponse:
        raise NotImplementedError

    async def list_models(self) -> CompletionResponse:
        raise NotImplementedError


def _item(reference: str | None = None) -> EvaluationItem:
    return EvaluationItem(
        "a", (CompletionMessage("user", "Explain X."),), None, reference=reference
    )


async def test_reference_items_are_graded_without_calling_the_judge() -> None:
    provider = JudgeProvider("PASS")
    grader = ReferenceOrJudgeGrader(provider, "judge/model")

    assert await grader.grade(_item("4"), "It is 4.") is True
    assert await grader.grade(_item("4"), "It is 5.") is False
    assert provider.requests == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("PASS", True),
        (" pass.", True),
        ("FAIL", False),
        ("Fail - wrong", False),
        ("maybe", None),
        (None, None),
    ],
)
async def test_judge_verdict_is_parsed_and_unparseable_is_unmeasured(
    reply: str | None, expected: bool | None
) -> None:
    provider = JudgeProvider(reply)
    grader = ReferenceOrJudgeGrader(provider, "judge/model")

    assert await grader.grade(_item(), "An answer.") is expected

    request = provider.requests[0]
    assert request.model == "judge/model"
    assert request.options == {"temperature": 0}
    assert "Explain X." in request.messages[1].content
    assert "An answer." in request.messages[1].content


async def test_items_without_reference_stay_ungraded_without_a_judge_model() -> None:
    provider = JudgeProvider("PASS")

    assert await ReferenceOrJudgeGrader(provider).grade(_item(), "answer") is None
    assert provider.requests == []


def _outcome(
    item_id: str = "a",
    *,
    text: str | None = "answer",
    cost: float | None = 0.01,
    error_type: str | None = None,
    passed: bool | None = None,
) -> StrategyOutcome:
    return StrategyOutcome(
        item_id=item_id,
        strategy_name="s",
        model_requested="s",
        model_selected="m",
        tier="fast",
        latency_ms=1.0,
        routing_latency_ms=0.0,
        provider_latency_ms=1.0,
        input_tokens=1,
        output_tokens=1,
        estimated_cost=cost,
        decision_confidence=None,
        status_code=200,
        error_type=error_type,
        response_text=text,
        quality_passed=passed,
    )


class ScriptedStrategy:
    def __init__(self, name: str, scripts: dict[str, StrategyOutcome]) -> None:
        self.name = name
        self._scripts = scripts

    async def run(self, item: EvaluationItem) -> StrategyOutcome:
        outcome = self._scripts[item.id]
        return StrategyOutcome(**{**outcome.__dict__, "strategy_name": self.name})


class TextGrader:
    """Grade by response text: 'pass' -> True, 'fail' -> False, 'none' -> unmeasured."""

    async def grade(self, item: EvaluationItem, response_text: str) -> bool | None:
        if response_text == "boom":
            raise RuntimeError("grader failure")
        return {"pass": True, "fail": False}.get(response_text)


async def test_runner_grades_marks_errors_failed_and_drops_response_text() -> None:
    dataset = EvaluationDataset(
        "d", tuple(EvaluationItem(i, (), None) for i in ("ok", "bad", "err", "skip", "boom"))
    )
    scripts = {
        "ok": _outcome("ok", text="pass"),
        "bad": _outcome("bad", text="fail"),
        "err": _outcome("err", text=None, error_type="provider_error"),
        "skip": _outcome("skip", text="none"),
        "boom": _outcome("boom", text="boom"),
    }
    strategies = (ScriptedStrategy("one", scripts), ScriptedStrategy("two", scripts))

    result = await run_benchmark(dataset, strategies, metrics=(), grader=TextGrader())

    verdicts = {o.item_id: o.quality_passed for o in result.strategies[0].outcomes}
    assert verdicts == {"ok": True, "bad": False, "err": False, "skip": None, "boom": None}
    assert all(o.response_text is None for s in result.strategies for o in s.outcomes)


async def test_runner_without_grader_leaves_verdicts_unmeasured() -> None:
    dataset = EvaluationDataset("d", (EvaluationItem("ok", (), None),))
    scripts = {"ok": _outcome("ok", text="pass")}

    result = await run_benchmark(
        dataset, (ScriptedStrategy("one", scripts), ScriptedStrategy("two", scripts))
    )

    assert result.strategies[0].outcomes[0].quality_passed is None


def test_quality_pass_rate_and_cost_per_successful_task() -> None:
    dataset = EvaluationDataset("d", ())
    outcomes = [
        _outcome(passed=True, cost=0.01),
        _outcome(passed=True, cost=0.03),
        _outcome(passed=False, cost=0.02),
        _outcome(passed=None, cost=0.50),
        _outcome(passed=True, cost=None),
    ]

    rate = QualityPassRateMetric().calculate(dataset, outcomes)
    cost = CostPerSuccessfulTaskMetric().calculate(dataset, outcomes)

    assert rate.value == pytest.approx(3 / 4)
    assert rate.samples == 4
    assert cost.value == pytest.approx(0.06 / 2)
    assert cost.samples == 3


def test_quality_metrics_report_none_without_verdicts_or_passes() -> None:
    dataset = EvaluationDataset("d", ())

    assert QualityPassRateMetric().calculate(dataset, [_outcome()]).value is None
    assert CostPerSuccessfulTaskMetric().calculate(dataset, [_outcome(passed=False)]).value is None


@pytest.mark.parametrize(
    ("prompt", "expected_complexity"),
    [
        ("What is a database index?", 1.0),
        ("Please " + "word " * 45, 2.0),
        ("Design a cache.", 2.0),
        ("Implement, debug and refactor this " + "word " * 45, 4.0),
        ("Design, implement and debug a secure architecture " + "word " * 160, 5.0),
    ],
)
async def test_heuristic_engine_scores_length_and_keywords(
    prompt: str, expected_complexity: float
) -> None:
    request = RoutingRequest(
        model="system-one/auto", messages=(CompletionMessage("user", prompt),)
    )

    decision = await HeuristicDecisionEngine().decide(request)

    assert decision.complexity == expected_complexity
    assert decision.confidence == 1.0
    assert decision.decision_source == "heuristic"
    assert decision.fallback_reason is None


def test_heuristic_strategy_routes_auto_through_the_baseline_engine() -> None:
    settings = Settings(policy_file="policies/balanced.yaml", models_file="registry/models.yaml")

    strategies = build_live_strategies(("heuristic", "system-one/fast"), settings)

    assert [(s.name, s.model) for s in strategies] == [
        ("heuristic", "system-one/auto"),
        ("system-one/fast", "system-one/fast"),
    ]
    assert strategies[0].orchestrator is not strategies[1].orchestrator


async def test_oracle_labels_cheapest_passing_tier_and_leaves_doubtful_items_unlabeled() -> None:
    names = ("fast", "balanced", "reasoning")
    #            fast    balanced reasoning
    plan = {
        "easy": ("pass", "pass", "pass"),
        "mid": ("fail", "pass", "pass"),
        "hard": ("fail", "fail", "pass"),
        "never": ("fail", "fail", "fail"),
        "doubt": ("none", "pass", "pass"),
    }
    strategies = tuple(
        ScriptedStrategy(
            name, {item: _outcome(item, text=texts[i]) for item, texts in plan.items()}
        )
        for i, name in enumerate(names)
    )
    dataset = EvaluationDataset("d", tuple(EvaluationItem(i, (), "reasoning") for i in plan))

    relabeled = await measure_expected_tiers(dataset, strategies, TextGrader())

    assert {item.id: item.expected_tier for item in relabeled.items} == {
        "easy": "fast",
        "mid": "balanced",
        "hard": "reasoning",
        "never": None,
        "doubt": None,
    }
    with pytest.raises(ValueError, match="exactly"):
        await measure_expected_tiers(dataset, strategies[:2], TextGrader())


def test_write_dataset_round_trips_through_the_loader(tmp_path: Path) -> None:
    original = load_dataset("evaluations/datasets/routing-benchmark.jsonl")
    path = tmp_path / "copy.jsonl"

    write_dataset(original, path)

    assert load_dataset(path).items == original.items
