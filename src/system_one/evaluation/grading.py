"""Answer-quality graders: exact match against a reference, else an optional LLM judge.

A grader returns ``True``/``False`` for a verdict and ``None`` when it cannot
judge, so ungradable items stay unmeasured instead of counting as failures.
"""

import re
from typing import Protocol

from system_one.domain.completions import CompletionMessage, CompletionRequest
from system_one.evaluation.dataset import EvaluationItem
from system_one.evaluation.strategy import extract_response_text
from system_one.providers.base import LLMProvider

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LETTER_ONLY = re.compile(r"\W*([A-J])\W*")
_LETTER_AFTER_ANSWER = re.compile(r"(?i:answer)(?:\s+is)?\W{0,5}([A-J])\b")

_JUDGE_INSTRUCTIONS = (
    "You are a strict grader. Decide whether the RESPONSE correctly and completely "
    "fulfils the TASK. Fail responses that are wrong, incomplete, ignore stated "
    "constraints, or invent facts. Reply with exactly one word: PASS or FAIL."
)


class QualityGrader(Protocol):
    """Judge one response; ``None`` means the grader could not decide."""

    async def grade(self, item: EvaluationItem, response_text: str) -> bool | None: ...


def matches_reference(reference: str, response_text: str) -> bool:
    """Compare a response with a short reference: option letter, number, or text."""
    text = response_text.strip()
    if re.fullmatch(r"[A-J]", reference):
        match = _LETTER_ONLY.fullmatch(text) or _LETTER_AFTER_ANSWER.search(text)
        return match is not None and match.group(1) == reference
    try:
        expected = float(reference)
    except ValueError:
        return text.casefold() == reference.strip().casefold()
    numbers = _NUMBER.findall(text)
    return bool(numbers) and abs(float(numbers[-1].replace(",", "")) - expected) < 1e-6


class ReferenceOrJudgeGrader:
    """Exact-match items that have a reference; send the rest to a judge model if set."""

    def __init__(self, provider: LLMProvider, judge_model: str | None = None) -> None:
        self._provider = provider
        self._judge_model = judge_model

    async def grade(self, item: EvaluationItem, response_text: str) -> bool | None:
        if item.reference is not None:
            return matches_reference(item.reference, response_text)
        if self._judge_model is None:
            return None
        task = "\n\n".join(f"{message.role}: {message.content}" for message in item.messages)
        request = CompletionRequest(
            model=self._judge_model,
            messages=(
                CompletionMessage(role="system", content=_JUDGE_INSTRUCTIONS),
                CompletionMessage(
                    role="user", content=f"TASK:\n{task}\n\nRESPONSE:\n{response_text}"
                ),
            ),
            options={"temperature": 0},
        )
        judged = extract_response_text((await self._provider.chat(request)).body)
        verdict = judged.strip().upper() if judged is not None else ""
        if verdict.startswith("PASS"):
            return True
        if verdict.startswith("FAIL"):
            return False
        return None
