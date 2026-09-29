"""Jev decision engine backed by the OpenRouter decisions API.

The engine posts a deterministic, code-owned decision request to
``POST <jev_base_url>/decisions`` and maps the answers onto a validated
``Decision``. It never raises on engine failure: timeouts, network errors,
unexpected HTTP statuses, and schema-invalid bodies all produce an explicit
fallback decision with ``decision_source="fallback"`` and a ``fallback_reason``
from ``decision_engine_timeout`` / ``decision_engine_unavailable`` /
``decision_engine_invalid_response``. A missing or unconfigured OpenRouter API
key raises ``ProviderConfigurationError`` instead of falling back.

The fallback decision uses the sentinel ``task_type="general"``, which is
deliberately not one of the Jev ``choice`` options so that only the fallback
path can produce it. Jev's response ``id`` and ``usage`` stay out of the
domain ``Decision`` (receipt telemetry is a later concern).
"""

from typing import cast

import httpx

from system_one.core.config import Settings
from system_one.domain.decision import Decision, RoutingRequest
from system_one.providers.errors import ProviderConfigurationError

_FALLBACK_REASON_TIMEOUT = "decision_engine_timeout"
_FALLBACK_REASON_UNAVAILABLE = "decision_engine_unavailable"
_FALLBACK_REASON_INVALID = "decision_engine_invalid_response"

_FALLBACK_TASK_TYPE = "general"
_FALLBACK_SCORE = 3.0
_FALLBACK_CONFIDENCE = 0.0

_MIN_SCORE = 1.0
_MAX_SCORE = 5.0

_TASK_TYPE_CRITERIA: dict[str, str] = {
    "coding": "Writing or debugging code.",
    "summarization": "Condensing existing content.",
    "question_answering": "Answering a direct question.",
    "analysis": "Reasoning in depth about provided material.",
    "translation": "Translating text between languages.",
}
_COMPLEXITY_CRITERIA = (
    "Trivial single-turn request",
    "Simple request with light reasoning",
    "Moderate multi-part request",
    "Advanced request requiring careful reasoning",
    "Extremely complex multi-step problem",
)
_QUALITY_CRITERIA = (
    "Best-effort acceptable",
    "Light quality needs",
    "Balanced quality needs",
    "High quality expected",
    "Correctness is critical",
)
_LATENCY_CRITERIA = (
    "Instant response required",
    "Tight latency budget",
    "Moderate latency budget",
    "Relaxed latency budget",
    "Latency is irrelevant",
)


class _DecisionFailure(RuntimeError):
    """Raised when Jev cannot be trusted; carries the fallback reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class JevDecisionEngine:
    """Decide routing inputs with Jev and fall back explicitly on failure."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def decide(self, request: RoutingRequest) -> Decision:
        """Return a decision for ``request``; return the fallback on engine failure."""
        try:
            body = await self._request_body(request)
            return self._map_decision(body)
        except _DecisionFailure as error:
            return self._fallback(error.reason)

    def _fallback(self, reason: str) -> Decision:
        return Decision(
            task_type=_FALLBACK_TASK_TYPE,
            complexity=_FALLBACK_SCORE,
            quality_requirement=_FALLBACK_SCORE,
            latency_requirement=_FALLBACK_SCORE,
            confidence=_FALLBACK_CONFIDENCE,
            decision_source="fallback",
            fallback_reason=reason,
        )

    async def _request_body(self, request: RoutingRequest) -> dict[str, object]:
        """Post the decision request and return the parsed JSON object body."""
        headers = self._headers()
        base_url = self._settings.jev_base_url.rstrip("/")

        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=self._settings.jev_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post("/decisions", json=self._payload(request))
        except httpx.TimeoutException as error:
            raise _DecisionFailure(_FALLBACK_REASON_TIMEOUT) from error
        except httpx.RequestError as error:
            raise _DecisionFailure(_FALLBACK_REASON_UNAVAILABLE) from error

        if response.status_code != 200:
            raise _DecisionFailure(_FALLBACK_REASON_UNAVAILABLE)

        try:
            body = cast(object, response.json())
        except ValueError as error:
            raise _DecisionFailure(_FALLBACK_REASON_INVALID) from error
        if not isinstance(body, dict):
            raise _DecisionFailure(_FALLBACK_REASON_INVALID)
        return cast(dict[str, object], body)

    def _map_decision(self, body: dict[str, object]) -> Decision:
        answers = self._answers(body)
        task_type = _choice_value(_answer(answers, "task_type"))
        complexity = _score_value(_answer(answers, "complexity"))
        quality_requirement = _score_value(_answer(answers, "quality_requirement"))
        latency_requirement = _score_value(_answer(answers, "latency_requirement"))
        confidences = _confidence_values(answers)
        if not confidences:
            raise _DecisionFailure(_FALLBACK_REASON_INVALID)

        return Decision(
            task_type=task_type,
            complexity=complexity,
            quality_requirement=quality_requirement,
            latency_requirement=latency_requirement,
            confidence=min(confidences),
            decision_source="jev",
        )

    def _answers(self, body: dict[str, object]) -> dict[str, object]:
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise _DecisionFailure(_FALLBACK_REASON_INVALID)
        return {str(key): value for key, value in answers.items()}

    def _payload(self, request: RoutingRequest) -> dict[str, object]:
        """Build the deterministic decision request; instructions are code-owned."""
        return {
            "model": self._settings.jev_model,
            "state": _state(request),
            "questions": {
                "task_type": {
                    "type": "choice",
                    "instructions": "Classify the primary task type of the request.",
                    "criteria": _TASK_TYPE_CRITERIA,
                },
                "complexity": {
                    "type": "score",
                    "instructions": "Rate the complexity of the request from 1 to 5.",
                    "criteria": _COMPLEXITY_CRITERIA,
                },
                "quality_requirement": {
                    "type": "score",
                    "instructions": "Rate the quality requirement of the request from 1 to 5.",
                    "criteria": _QUALITY_CRITERIA,
                },
                "latency_requirement": {
                    "type": "score",
                    "instructions": "Rate the latency requirement of the request from 1 to 5.",
                    "criteria": _LATENCY_CRITERIA,
                },
            },
        }

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise ProviderConfigurationError(
                "OpenRouter API key is not configured for the Jev decision engine"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_http_referer:
            headers["HTTP-Referer"] = self._settings.openrouter_http_referer
        if self._settings.openrouter_app_title:
            headers["X-Title"] = self._settings.openrouter_app_title
        return headers


def _state(request: RoutingRequest) -> str:
    """Render the messages as a plain-text conversation transcript."""
    return "\n".join(
        f"{message.role}: {message.content}" for message in request.messages
    )


def _answer(answers: dict[str, object], name: str) -> dict[str, object]:
    answer = answers.get(name)
    if not isinstance(answer, dict):
        raise _DecisionFailure(_FALLBACK_REASON_INVALID)
    return {str(key): value for key, value in answer.items()}


def _choice_value(answer: dict[str, object]) -> str:
    choice = answer.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        raise _DecisionFailure(_FALLBACK_REASON_INVALID)
    return choice


def _score_value(answer: dict[str, object]) -> float:
    """Read the score answer, accepting numbers and string-encoded numbers."""
    score = answer.get("score")
    value = _plain_number(score)
    if value is None and isinstance(score, str):
        try:
            value = float(score)
        except ValueError:
            value = None
    if value is None or not _MIN_SCORE <= value <= _MAX_SCORE:
        raise _DecisionFailure(_FALLBACK_REASON_INVALID)
    return value


def _confidence_values(answers: dict[str, object]) -> list[float]:
    """Collect trust values: explicit confidence, else the noul probability."""
    values: list[float] = []
    for answer in answers.values():
        if not isinstance(answer, dict):
            continue
        value = _plain_number(answer.get("confidence"))
        if value is None and answer.get("type") == "noul":
            value = _plain_number(answer.get("noul"))
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise _DecisionFailure(_FALLBACK_REASON_INVALID)
            values.append(value)
    return values


def _plain_number(value: object) -> float | None:
    """Return the value as a float when it is a plain JSON number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
