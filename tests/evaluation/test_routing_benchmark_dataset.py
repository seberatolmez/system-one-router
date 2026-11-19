"""Tests for dataset metadata parsing and the shipped routing benchmark dataset."""

from collections import Counter
from pathlib import Path

import pytest

from system_one.evaluation.dataset import DatasetFormatError, load_dataset

_TASK_TYPES = {"coding", "summarization", "question_answering", "analysis", "translation"}


def test_loader_reads_optional_provenance_metadata(tmp_path: Path) -> None:
    path = tmp_path / "meta.jsonl"
    path.write_text(
        '{"id":"a","messages":[{"role":"user","content":"2+2?"}],"task_type":"coding",'
        '"difficulty":"easy","source":"hand-written","license":"MIT","reference":"4"}\n',
        encoding="utf-8",
    )

    item = load_dataset(path).items[0]

    assert (item.task_type, item.difficulty, item.source, item.license, item.reference) == (
        "coding",
        "easy",
        "hand-written",
        "MIT",
        "4",
    )


@pytest.mark.parametrize(
    "fields",
    ['"difficulty":"extreme"', '"reference":""', '"source":3', '"license":"  "'],
)
def test_loader_rejects_invalid_metadata(tmp_path: Path, fields: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        f'{{"id":"a","messages":[{{"role":"user","content":"Hi"}}],{fields}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError):
        load_dataset(path)


def test_routing_benchmark_dataset_is_balanced_and_fully_attributed() -> None:
    dataset = load_dataset("evaluations/datasets/routing-benchmark.jsonl")

    assert len(dataset.items) == 300
    cells = Counter((item.task_type, item.difficulty) for item in dataset.items)
    assert {task for task, _ in cells} == _TASK_TYPES
    assert set(cells.values()) == {20}
    for item in dataset.items:
        assert item.source and item.license
        assert item.expected_tier == {"easy": "fast", "medium": "balanced", "hard": "reasoning"}[
            item.difficulty or ""
        ]
        assert all(message.content.strip() for message in item.messages)


def test_routing_benchmark_references_are_checkable_answers() -> None:
    dataset = load_dataset("evaluations/datasets/routing-benchmark.jsonl")
    referenced = [item for item in dataset.items if item.reference is not None]

    assert len(referenced) == 40
    for item in referenced:
        if item.source == "TIGER-Lab/MMLU-Pro":
            assert item.reference in set("ABCDEFGHIJ")
        else:
            float(item.reference or "")
