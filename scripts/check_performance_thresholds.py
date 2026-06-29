#!/usr/bin/env python3
"""Fail release checks when maintained benchmark timings exceed thresholds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "benches" / "benchmark_summary.json"

REQUIRED_GROUPS = {"data_loading", "prediction", "serialize", "training"}


def main() -> int:
    report = check_performance_thresholds()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        failed = ", ".join(report["failed_benchmarks"] + report["missing_groups"])
        raise SystemExit(f"performance threshold checks failed: {failed}")
    return 0


def check_performance_thresholds(path: Path = SUMMARY) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("benchmark summary must be a JSON list")

    checked: list[dict[str, Any]] = []
    failed: list[str] = []
    groups: set[str] = set()
    schema_errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            schema_errors.append(f"row {index} is not an object")
            continue
        benchmark = str(row.get("benchmark", ""))
        group = benchmark.split("/", 1)[0] if "/" in benchmark else ""
        if group:
            groups.add(group)
        try:
            mean_ns = float(row["mean_ns"])
            max_mean_ns = float(row["max_mean_ns"])
        except (KeyError, TypeError, ValueError):
            schema_errors.append(f"{benchmark or f'row {index}'}: missing numeric timing fields")
            continue
        finite = math.isfinite(mean_ns) and math.isfinite(max_mean_ns)
        passed = finite and mean_ns > 0.0 and max_mean_ns > 0.0 and mean_ns <= max_mean_ns
        if not passed:
            failed.append(benchmark)
        checked.append(
            {
                "benchmark": benchmark,
                "mean_ns": mean_ns,
                "max_mean_ns": max_mean_ns,
                "headroom_ratio": max_mean_ns / mean_ns if mean_ns > 0.0 else None,
                "passed": passed,
            }
        )

    missing_groups = sorted(REQUIRED_GROUPS - groups)
    passed = not failed and not missing_groups and not schema_errors
    return {
        "artifact_type": "cartoboost.performance_threshold_audit",
        "artifact_version": 1,
        "path": str(path.relative_to(ROOT)),
        "passed": passed,
        "required_groups": sorted(REQUIRED_GROUPS),
        "missing_groups": missing_groups,
        "failed_benchmarks": failed,
        "schema_errors": schema_errors,
        "checked": checked,
    }


if __name__ == "__main__":
    raise SystemExit(main())
