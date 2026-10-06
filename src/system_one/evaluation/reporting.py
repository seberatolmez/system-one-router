"""Machine-readable JSON and compact Markdown benchmark reports."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from system_one.evaluation.runner import BenchmarkResult


def write_reports(
    result: BenchmarkResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write one immutable JSON run and the latest human-readable summary."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset_name = re.sub(r"[^A-Za-z0-9._-]+", "-", result.dataset_name).strip("-")
    filename = f"run-{safe_dataset_name or 'dataset'}-{result.run_id}"
    json_path = report_dir / f"{filename}.json"
    markdown_path = report_dir / "summary.md"

    json_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_summary(result), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_summary(result: BenchmarkResult) -> str:
    """Render comparable strategy-level metrics without embedding prompts."""
    metric_columns = result.strategies[0].metrics if result.strategies else ()
    headers = ["Strategy", *(metric.name for metric in metric_columns)]
    lines = [
        f"# Benchmark summary: {_escape_cell(result.dataset_name)}",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Created at: `{result.created_at}`",
        f"- Dataset items: {result.item_count}",
        "- Metrics use measured provider responses and configured registry prices;",
        "  missing measurements are excluded and reported through sample counts in JSON.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for strategy in result.strategies:
        values = {metric.name: metric for metric in strategy.metrics}
        cells = [_escape_cell(strategy.name)]
        for column in metric_columns:
            metric = values[column.name]
            if metric.value is None:
                cells.append("n/a")
            else:
                cells.append(f"{metric.value:.6g} {metric.unit}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "This sample report is an evaluation artifact, not a claim that any "
            "strategy is better.",
            "",
        ]
    )
    return "\n".join(lines)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
