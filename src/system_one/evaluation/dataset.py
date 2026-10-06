"""Strict JSONL dataset loading for repeatable router evaluations."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from system_one.domain.completions import CompletionMessage
from system_one.domain.decision import RoutingTier

_ALLOWED_ITEM_KEYS = {"id", "messages", "expected_tier"}
_ALLOWED_MESSAGE_KEYS = {"role", "content"}
_VALID_TIERS = {"fast", "balanced", "reasoning"}


class DatasetFormatError(ValueError):
    """Raised when an evaluation dataset cannot be read or validated."""


@dataclass(frozen=True)
class EvaluationItem:
    """One request and its optional evaluation-only expected tier label."""

    id: str
    messages: tuple[CompletionMessage, ...]
    expected_tier: RoutingTier | None


@dataclass(frozen=True)
class EvaluationDataset:
    """A named, validated set of requests to run against every strategy."""

    name: str
    items: tuple[EvaluationItem, ...]


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate a JSON Lines dataset with line-specific errors."""
    dataset_path = Path(path)
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetFormatError(
            f"Unable to read evaluation dataset '{dataset_path}': {error}"
        ) from error

    items: list[EvaluationItem] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            document: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetFormatError(
                f"Invalid JSON in '{dataset_path}' on line {line_number}: {error.msg}"
            ) from error

        item = _parse_item(document, dataset_path, line_number)
        if item.id in seen_ids:
            raise DatasetFormatError(
                f"Duplicate item id {item.id!r} in '{dataset_path}' on line {line_number}"
            )
        seen_ids.add(item.id)
        items.append(item)

    if not items:
        raise DatasetFormatError(f"Evaluation dataset '{dataset_path}' contains no items")
    return EvaluationDataset(name=dataset_path.stem, items=tuple(items))


def _parse_item(document: object, path: Path, line_number: int) -> EvaluationItem:
    if not isinstance(document, dict):
        raise _invalid_item(path, line_number, "each JSONL record must be an object")
    record = cast(Mapping[str, object], document)
    unknown_keys = set(record) - _ALLOWED_ITEM_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(str(key) for key in unknown_keys))
        raise _invalid_item(path, line_number, f"unknown fields: {unknown}")

    item_id = record.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise _invalid_item(path, line_number, "'id' must be a non-empty string")

    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _invalid_item(path, line_number, "'messages' must be a non-empty array")
    messages = tuple(
        _parse_message(message, path, line_number, message_index)
        for message_index, message in enumerate(raw_messages, start=1)
    )

    raw_tier = record.get("expected_tier")
    if raw_tier is not None and (
        not isinstance(raw_tier, str) or raw_tier not in _VALID_TIERS
    ):
        raise _invalid_item(
            path,
            line_number,
            "'expected_tier' must be 'fast', 'balanced', 'reasoning', or null",
        )
    expected_tier = cast(RoutingTier, raw_tier) if raw_tier is not None else None

    return EvaluationItem(
        id=item_id,
        messages=messages,
        expected_tier=expected_tier,
    )


def _parse_message(
    value: object,
    path: Path,
    line_number: int,
    message_index: int,
) -> CompletionMessage:
    if not isinstance(value, dict):
        raise _invalid_item(
            path,
            line_number,
            f"message {message_index} must be an object",
        )
    message_value = cast(Mapping[str, object], value)
    unknown_keys = set(message_value) - _ALLOWED_MESSAGE_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(str(key) for key in unknown_keys))
        raise _invalid_item(
            path,
            line_number,
            f"message {message_index} has unknown fields: {unknown}",
        )
    role = message_value.get("role")
    content = message_value.get("content")
    if not isinstance(role, str) or not role.strip():
        raise _invalid_item(
            path,
            line_number,
            f"message {message_index} 'role' must be a non-empty string",
        )
    if not isinstance(content, str):
        raise _invalid_item(
            path,
            line_number,
            f"message {message_index} 'content' must be a string",
        )
    return CompletionMessage(role=role, content=content)


def _invalid_item(path: Path, line_number: int, reason: str) -> DatasetFormatError:
    return DatasetFormatError(f"Invalid item in '{path}' on line {line_number}: {reason}")
