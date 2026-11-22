"""Crude baseline decision engine: prompt length and keywords, no model call.

It exists only as a benchmark baseline for the Jev-backed engine. Its
thresholds are fixed a priori and deliberately simple; they are not tuned
against any dataset.
"""

from system_one.domain.decision import Decision, RoutingRequest

_LONG_WORDS = 40
_VERY_LONG_WORDS = 150
_KEYWORDS = (
    "implement",
    "design",
    "debug",
    "refactor",
    "optimi",
    "architecture",
    "concurren",
    "race condition",
    "security",
    "trade-off",
    "tradeoff",
    "analy",
    "prove",
    "step by step",
)


class HeuristicDecisionEngine:
    """Score complexity 1..5 from word count and reasoning-keyword hits."""

    async def decide(self, request: RoutingRequest) -> Decision:
        text = " ".join(message.content for message in request.messages).lower()
        words = len(text.split())
        hits = sum(keyword in text for keyword in _KEYWORDS)
        points = (
            (words >= _LONG_WORDS)
            + (words >= _VERY_LONG_WORDS)
            + (hits >= 1)
            + (hits >= 3)
        )
        return Decision(
            task_type="general",
            complexity=1.0 + points,
            quality_requirement=3.0,
            latency_requirement=3.0,
            confidence=1.0,
            decision_source="heuristic",
        )
