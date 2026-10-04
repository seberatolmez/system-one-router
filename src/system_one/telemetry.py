"""Routing receipts, provider usage extraction, and structured telemetry logs."""

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from system_one.domain.decision import Decision, RoutingTier
from system_one.routing.registry.models import ModelCost

_logger = logging.getLogger("system_one.telemetry")


def configure_telemetry_logging(level: str) -> None:
    """Ensure the standalone service emits one-line JSON logs at its configured level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format="%(message)s")
    _logger.setLevel(numeric_level)


@dataclass(frozen=True)
class TokenUsage:
    """Normalized input/output token counts; missing values remain unknown."""

    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class RoutingReceipt:
    """Internal provenance and measured outcome for one virtual-model request."""

    request_id: str
    timestamp: str
    model_requested: str
    model_selected: str
    route_type: str
    tier: RoutingTier
    decision: Decision | None
    policy_name: str | None
    decision_latency_ms: float | None
    routing_latency_ms: float
    provider_latency_ms: float
    total_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None
    status_code: int
    error_type: str | None


def extract_token_usage(body: Mapping[str, object]) -> TokenUsage | None:
    """Read OpenAI-compatible prompt/completion token counts from a response."""
    raw_usage = body.get("usage")
    if not isinstance(raw_usage, Mapping):
        return None

    input_tokens = _token_count(raw_usage.get("prompt_tokens"))
    output_tokens = _token_count(raw_usage.get("completion_tokens"))
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def estimate_cost(usage: TokenUsage | None, cost: ModelCost) -> float | None:
    """Estimate cost from known usage and per-million-token registry prices."""
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    return (
        usage.input_tokens * cost.input_per_million
        + usage.output_tokens * cost.output_per_million
    ) / 1_000_000


class SSEUsageExtractor:
    """Observe usage in streamed SSE events without changing forwarded bytes."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: list[bytes] = []
        self.usage: TokenUsage | None = None

    def feed(self, chunk: bytes) -> None:
        """Accept arbitrary byte chunks, including partial SSE lines/events."""
        self._buffer.extend(chunk)
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            self._consume_line(bytes(line).rstrip(b"\r"))

    def finish(self) -> None:
        """Process a final event that was not terminated by a blank line."""
        if self._buffer:
            self._consume_line(bytes(self._buffer).rstrip(b"\r"))
            self._buffer.clear()
        self._consume_event()

    def _consume_line(self, line: bytes) -> None:
        if not line:
            self._consume_event()
        elif line.startswith(b"data:"):
            value = line[5:]
            self._data_lines.append(value[1:] if value.startswith(b" ") else value)

    def _consume_event(self) -> None:
        if not self._data_lines:
            return
        payload = b"\n".join(self._data_lines)
        self._data_lines.clear()
        if payload == b"[DONE]":
            return

        try:
            event = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if isinstance(event, Mapping):
            usage = extract_token_usage(event)
            if usage is not None:
                self.usage = usage


def emit_telemetry_event(event: Mapping[str, object]) -> None:
    """Write one parseable JSON object as a single structured log record."""
    _logger.info(json.dumps(event, separators=(",", ":"), sort_keys=True))


def build_routing_receipt(
    *,
    request_id: str,
    model_requested: str,
    model_selected: str,
    route_type: str,
    tier: RoutingTier,
    decision: Decision | None,
    policy_name: str | None,
    decision_latency_ms: float | None,
    routing_latency_ms: float,
    provider_latency_ms: float,
    total_latency_ms: float,
    usage: TokenUsage | None,
    cost: ModelCost,
    status_code: int,
    error_type: str | None,
) -> RoutingReceipt:
    """Construct a receipt with one timestamp and a cost only for complete usage."""
    return RoutingReceipt(
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        model_requested=model_requested,
        model_selected=model_selected,
        route_type=route_type,
        tier=tier,
        decision=decision,
        policy_name=policy_name,
        decision_latency_ms=decision_latency_ms,
        routing_latency_ms=routing_latency_ms,
        provider_latency_ms=provider_latency_ms,
        total_latency_ms=total_latency_ms,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        estimated_cost=estimate_cost(usage, cost),
        status_code=status_code,
        error_type=error_type,
    )


def log_routing_receipt(receipt: RoutingReceipt) -> None:
    """Emit a routing receipt as a structured, machine-readable log event."""
    emit_telemetry_event({"event": "routing_receipt", "receipt": asdict(receipt)})


def _token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
