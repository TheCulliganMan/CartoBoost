"""Aggregate benchmark JSONL result rows into report-ready JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.runners.significance import normal_mean_ci  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str | None, str, str | None, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("track"),
            row["task_id"],
            row.get("split_id"),
            row["model_family"],
            row["metric"],
        )
        grouped[key].append(float(row["value"]))

    metrics = []
    for (track, task_id, split_id, model_family, metric), values in sorted(grouped.items()):
        center, low, high = normal_mean_ci(values)
        metric_row = {
            "task_id": task_id,
            "model_family": model_family,
            "metric": metric,
            "n": len(values),
            "mean": center,
            "ci95_low": low,
            "ci95_high": high,
        }
        if track is not None:
            metric_row["track"] = track
        if split_id is not None:
            metric_row["split_id"] = split_id
        metrics.append(metric_row)
    return {"metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL result file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON summary path")
    args = parser.parse_args(argv)

    summary = aggregate(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
