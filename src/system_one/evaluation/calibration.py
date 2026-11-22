"""Calibrate a fast/balanced routing policy for one decision model, with a held-out test.

Inputs are reports written by ``system_one.evaluation.cli``: graded answers from
the fast and balanced tiers, and a decision report for the same items. Nothing
here calls a model.

Protocol, fixed before any held-out number is looked at:

1. A deterministic stratified split (by task type and difficulty) gives a train
   and a test set. Every decision model uses the same split.
2. On the train set only, search policies "complexity below ``t`` goes to fast,
   otherwise balanced; confidence below ``g`` always goes to balanced". The best
   policy has the lowest cost per successful task among those whose quality is at
   most ``train_margin`` below always-balanced. Always-balanced is a candidate, so
   "routing does not pay" is a possible outcome.
3. The chosen policy is scored once on the test set against always-balanced with
   a paired bootstrap. It wins only if its cost per success is at least
   ``min_saving`` lower and the lower 95% bound of its quality difference is
   within ``margin`` of always-balanced.

Only the fast and balanced tiers are measured on every item, so the reasoning
tier is reserved for a sliver at the very top of the complexity scale. Items whose
decision would reach it are excluded and counted, never substituted.
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from system_one.evaluation.dataset import EvaluationDataset, load_dataset, resolve_in_workspace
from system_one.evaluation.replay import Outcomes, load_outcomes
from system_one.evaluation.strategy import StrategyOutcome

Tier = Literal["fast", "balanced"]

REASONING_FLOOR = 4.9
_MAX_THRESHOLDS = 40
_CONFIDENCE_QUANTILES = (0.1, 0.25, 0.5)


@dataclass(frozen=True)
class CalibrationItem:
    """What one routing decision and the two measured tier answers say about one item."""

    item_id: str
    fast_passed: bool
    balanced_passed: bool
    fast_cost: float
    balanced_cost: float
    decision_cost: float
    complexity: float | None
    confidence: float | None


@dataclass(frozen=True)
class Policy:
    """Complexity below ``fast_below`` goes to fast; low confidence goes to balanced."""

    fast_below: float
    confidence_gate: float


ALWAYS_BALANCED = Policy(fast_below=1.0, confidence_gate=0.0)


@dataclass(frozen=True)
class Score:
    n: int
    quality: float
    cost_per_success: float | None
    fast_share: float


def route(policy: Policy, item: CalibrationItem) -> Tier:
    """Pick a tier; a failed decision (no complexity or confidence) goes to balanced."""
    if item.complexity is None or item.confidence is None:
        return "balanced"
    if item.confidence < policy.confidence_gate:
        return "balanced"
    return "fast" if item.complexity < policy.fast_below else "balanced"


def _outcome(item: CalibrationItem, tier: Tier) -> tuple[bool, float]:
    if tier == "fast":
        return item.fast_passed, item.fast_cost
    return item.balanced_passed, item.balanced_cost


def score(items: Sequence[CalibrationItem], policy: Policy, *, with_decision: bool) -> Score:
    """Quality and cost per successful task; decision cost counts only for routers."""
    passes = fast = 0
    cost = 0.0
    for item in items:
        tier = route(policy, item)
        passed, answer_cost = _outcome(item, tier)
        passes += passed
        fast += tier == "fast"
        cost += answer_cost + (item.decision_cost if with_decision else 0.0)
    return Score(
        n=len(items),
        quality=passes / len(items),
        cost_per_success=cost / passes if passes else None,
        fast_share=fast / len(items),
    )


def split_ids(
    dataset: EvaluationDataset, test_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Stratified, deterministic train/test split of dataset item ids."""
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in dataset.items:
        cells[(item.task_type or "unknown", item.difficulty or "unknown")].append(item.id)
    rng = random.Random(seed)
    train: set[str] = set()
    test: set[str] = set()
    for cell in sorted(cells):
        ids = sorted(cells[cell])
        rng.shuffle(ids)
        n_test = round(len(ids) * test_fraction)
        test.update(ids[:n_test])
        train.update(ids[n_test:])
    return train, test


def build_items(
    dataset: EvaluationDataset,
    tiers: Outcomes,
    decisions: dict[str, StrategyOutcome],
) -> tuple[list[CalibrationItem], list[str]]:
    """Join tier answers and decisions; return usable items and why others were dropped."""
    fast = tiers.get("system-one/fast", {})
    balanced = tiers.get("system-one/balanced", {})
    items: list[CalibrationItem] = []
    dropped: list[str] = []
    for source in dataset.items:
        f, b, d = fast.get(source.id), balanced.get(source.id), decisions.get(source.id)
        if f is None or b is None or d is None:
            dropped.append(f"{source.id}: missing outcome")
        elif f.quality_passed is None or b.quality_passed is None:
            dropped.append(f"{source.id}: ungraded answer")
        elif d.decision_complexity is not None and d.decision_complexity >= REASONING_FLOOR:
            dropped.append(f"{source.id}: decision reaches the unmeasured reasoning band")
        elif f.estimated_cost is None or b.estimated_cost is None:
            dropped.append(f"{source.id}: answer cost unmeasured")
        else:
            items.append(
                CalibrationItem(
                    item_id=source.id,
                    fast_passed=f.quality_passed,
                    balanced_passed=b.quality_passed,
                    fast_cost=f.estimated_cost,
                    balanced_cost=b.estimated_cost,
                    decision_cost=d.decision_cost or 0.0,
                    complexity=d.decision_complexity,
                    confidence=d.decision_confidence,
                )
            )
    return items, dropped


def candidate_policies(train: Sequence[CalibrationItem]) -> list[Policy]:
    """Thresholds between observed train complexities, crossed with confidence gates."""
    values = sorted({round(i.complexity, 2) for i in train if i.complexity is not None})
    midpoints = [round((a + b) / 2, 2) for a, b in zip(values, values[1:], strict=False)]
    midpoints = [m for m in midpoints if 1.0 < m < REASONING_FLOOR]
    if len(midpoints) > _MAX_THRESHOLDS:
        stride = len(midpoints) / _MAX_THRESHOLDS
        midpoints = [midpoints[int(k * stride)] for k in range(_MAX_THRESHOLDS)]
    confidences = sorted(i.confidence for i in train if i.confidence is not None)
    gates = {0.0}
    if confidences:
        gates |= {round(confidences[int(q * len(confidences))], 2) for q in _CONFIDENCE_QUANTILES}
    return [Policy(t, g) for t in midpoints for g in sorted(gates)]


def choose_policy(train: Sequence[CalibrationItem], train_margin: float) -> Policy:
    """Cheapest policy per success within ``train_margin`` of balanced; else always-balanced."""
    baseline = score(train, ALWAYS_BALANCED, with_decision=False)
    best, best_cost = ALWAYS_BALANCED, baseline.cost_per_success
    for policy in candidate_policies(train):
        s = score(train, policy, with_decision=True)
        if s.cost_per_success is None or s.quality < baseline.quality - train_margin:
            continue
        if best_cost is None or s.cost_per_success < best_cost:
            best, best_cost = policy, s.cost_per_success
    return best


@dataclass(frozen=True)
class Comparison:
    """Held-out router vs always-balanced, with paired-bootstrap 95% intervals."""

    router: Score
    baseline: Score
    quality_difference: float
    quality_difference_ci: tuple[float, float]
    cost_change: float | None
    cost_change_ci: tuple[float, float] | None
    wins: bool


def compare(
    test: Sequence[CalibrationItem],
    policy: Policy,
    *,
    seed: int,
    margin: float,
    min_saving: float,
    resamples: int = 2000,
) -> Comparison:
    router = score(test, policy, with_decision=True)
    baseline = score(test, ALWAYS_BALANCED, with_decision=False)
    router_rows = [_outcome(i, route(policy, i)) for i in test]
    router_costs = [c + i.decision_cost for (_, c), i in zip(router_rows, test, strict=True)]
    base_rows = [_outcome(i, "balanced") for i in test]

    rng = random.Random(seed)
    n = len(test)
    quality_diffs: list[float] = []
    cost_changes: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        r_pass = sum(router_rows[k][0] for k in picks)
        b_pass = sum(base_rows[k][0] for k in picks)
        quality_diffs.append((r_pass - b_pass) / n)
        if r_pass and b_pass:
            r_cost = sum(router_costs[k] for k in picks) / r_pass
            b_cost = sum(base_rows[k][1] for k in picks) / b_pass
            cost_changes.append(r_cost / b_cost - 1)
    quality_ci = _interval(quality_diffs)
    cost_ci = _interval(cost_changes) if cost_changes else None
    cost_change = (
        router.cost_per_success / baseline.cost_per_success - 1
        if router.cost_per_success is not None and baseline.cost_per_success
        else None
    )
    wins = (
        policy != ALWAYS_BALANCED
        and cost_change is not None
        and cost_change <= -min_saving
        and quality_ci[0] >= -margin
    )
    return Comparison(
        router=router,
        baseline=baseline,
        quality_difference=router.quality - baseline.quality,
        quality_difference_ci=quality_ci,
        cost_change=cost_change,
        cost_change_ci=cost_ci,
        wins=wins,
    )


def _interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def policy_yaml(policy: Policy, model: str) -> str:
    """Render the policy in the project's policy-file format, reasoning kept as a top sliver."""
    return (
        f"# Calibrated for {model} by system_one.evaluation.calibrate.\n"
        "# Two-tier: fast below the threshold, balanced above. The reasoning band is a\n"
        f"# sliver at the top of the scale ({REASONING_FLOOR}+), unreached in calibration data.\n"
        "policy:\n"
        f"  confidence_threshold: {policy.confidence_gate}\n"
        "  tiers:\n"
        "    fast:\n"
        "      min_complexity: 1.0\n"
        f"      max_complexity: {policy.fast_below}\n"
        "    balanced:\n"
        f"      min_complexity: {policy.fast_below}\n"
        f"      max_complexity: {REASONING_FLOOR}\n"
        "    reasoning:\n"
        f"      min_complexity: {REASONING_FLOOR}\n"
        "      max_complexity: 5.0\n"
    )


@dataclass(frozen=True)
class ModelCalibration:
    model: str
    policy: Policy
    train_baseline: Score
    train_router: Score
    comparison: Comparison
    excluded: int
    train_items: int
    test_items: int


def calibrate_model(
    model: str,
    dataset: EvaluationDataset,
    tiers: Outcomes,
    decisions: dict[str, StrategyOutcome],
    split: tuple[set[str], set[str]],
    *,
    seed: int,
    train_margin: float,
    margin: float,
    min_saving: float,
) -> ModelCalibration:
    items, dropped = build_items(dataset, tiers, decisions)
    train_ids, test_ids = split
    train = [i for i in items if i.item_id in train_ids]
    test = [i for i in items if i.item_id in test_ids]
    if not train or not test:
        raise ValueError(
            f"{model}: need usable train and test items (have {len(train)}/{len(test)})"
        )
    policy = choose_policy(train, train_margin)
    return ModelCalibration(
        model=model,
        policy=policy,
        train_baseline=score(train, ALWAYS_BALANCED, with_decision=False),
        train_router=score(train, policy, with_decision=True),
        comparison=compare(test, policy, seed=seed, margin=margin, min_saving=min_saving),
        excluded=len(dropped),
        train_items=len(train),
        test_items=len(test),
    )


def render_report(results: Sequence[ModelCalibration], settings: dict[str, object]) -> str:
    lines = [
        "# Policy calibration",
        "",
        "Calibrated on a train split and scored once on a held-out test split. "
        "A policy wins only if it cuts cost per successful task by at least "
        f"{settings['min_saving']:.0%} and its quality difference to always-balanced has a "
        f"95% lower bound of at least -{settings['margin']}.",
        "",
        f"Settings: `{json.dumps(settings, sort_keys=True)}`",
        "",
        "| Decision model | Policy (fast below / confidence gate) | Test fast share "
        "| Quality diff (95% CI) | Cost per success change (95% CI) | Wins |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        c = r.comparison
        policy = (
            "always-balanced (nothing beat it on train)"
            if r.policy == ALWAYS_BALANCED
            else (f"{r.policy.fast_below} / {r.policy.confidence_gate}")
        )
        cost = (
            f"{c.cost_change:+.1%} ({c.cost_change_ci[0]:+.1%} to {c.cost_change_ci[1]:+.1%})"
            if c.cost_change is not None and c.cost_change_ci is not None
            else "n/a"
        )
        lines.append(
            f"| `{r.model}` | {policy} | {c.router.fast_share:.0%} | "
            f"{c.quality_difference:+.3f} ({c.quality_difference_ci[0]:+.3f} to "
            f"{c.quality_difference_ci[1]:+.3f}) | {cost} | {'**yes**' if c.wins else 'no'} |"
        )
    lines.extend(["", "Items used per model (train / test / excluded):", ""])
    lines.extend(f"- `{r.model}`: {r.train_items} / {r.test_items} / {r.excluded}" for r in results)
    return "\n".join(lines) + "\n"


def _merge(paths: Sequence[Path]) -> Outcomes:
    merged: Outcomes = {}
    for path in paths:
        for strategy, by_item in load_outcomes(path).items():
            merged.setdefault(strategy, {}).update(by_item)
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    """Calibrate every decision model in a decision report and write reports and policies."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tier-report", type=Path, action="append", required=True)
    parser.add_argument("--decision-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/reports/calibration"))
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--train-margin", type=float, default=0.03)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-saving", type=float, default=0.10)
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    tiers = _merge(args.tier_report)
    decisions = load_outcomes(args.decision_report)
    split = split_ids(dataset, args.test_fraction, args.seed)
    settings: dict[str, object] = {
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "train_margin": args.train_margin,
        "margin": args.margin,
        "min_saving": args.min_saving,
        "train_size": len(split[0]),
        "test_size": len(split[1]),
    }
    results = [
        calibrate_model(
            model,
            dataset,
            tiers,
            by_item,
            split,
            seed=args.seed,
            train_margin=args.train_margin,
            margin=args.margin,
            min_saving=args.min_saving,
        )
        for model, by_item in decisions.items()
    ]

    out = resolve_in_workspace(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.md").write_text(render_report(results, settings), encoding="utf-8")
    (out / "calibration.json").write_text(
        json.dumps({"settings": settings, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    for r in results:
        if r.policy != ALWAYS_BALANCED:
            name = r.model.replace("/", "-").replace(":", "-")
            (out / f"{name}.yaml").write_text(policy_yaml(r.policy, r.model), encoding="utf-8")
    print(f"Wrote {out / 'calibration.md'}")


if __name__ == "__main__":
    main()
