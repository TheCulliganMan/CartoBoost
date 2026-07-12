#!/usr/bin/env python3
"""Check the four leakage-safe NYC row-model comparisons for release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_COMPARISONS = (
    ("duration", "spatial_holdout"),
    ("duration", "out_of_time"),
    ("fare", "spatial_holdout"),
    ("fare", "out_of_time"),
)


def check_nyc_row_quality_gate(
    artifact: Path,
    *,
    maximum_relative_rmse: float = 1.02,
    minimum_wins: int = 3,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    dataset = payload.get("dataset") or {}
    comparisons: list[dict[str, Any]] = []
    for task_name, split_name in REQUIRED_COMPARISONS:
        split = (
            ((payload.get("tasks") or {}).get(task_name) or {}).get("splits", {}).get(split_name)
        )
        models = split.get("models") if isinstance(split, dict) else None
        if not isinstance(models, dict):
            comparisons.append(
                {
                    "task": task_name,
                    "split": split_name,
                    "passed": False,
                    "reason": "required split or model table is missing",
                }
            )
            continue
        cartoboost = models.get("cartoboost") or {}
        carto_rmse = (cartoboost.get("metrics") or {}).get("rmse")
        baselines = {
            name: ((row.get("metrics") or {}).get("rmse"))
            for name, row in models.items()
            if name != "cartoboost" and row.get("status") == "ok"
        }
        baselines = {name: value for name, value in baselines.items() if value is not None}
        best_name, best_rmse = (
            min(baselines.items(), key=lambda item: float(item[1])) if baselines else (None, None)
        )
        ratio = (
            float(carto_rmse) / float(best_rmse)
            if carto_rmse is not None and best_rmse is not None and float(best_rmse) > 0
            else float("inf")
        )
        comparisons.append(
            {
                "task": task_name,
                "split": split_name,
                "cartoboost_rmse": carto_rmse,
                "best_external_baseline": best_name,
                "best_external_rmse": best_rmse,
                "relative_rmse": ratio,
                "passed": ratio <= maximum_relative_rmse,
            }
        )
    wins = sum(bool(item.get("passed")) for item in comparisons)
    passed = (
        dataset.get("source") == "nyc_tlc_trip_records"
        and len(comparisons) == len(REQUIRED_COMPARISONS)
        and wins >= minimum_wins
    )
    return {
        "artifact_type": "cartoboost.nyc_row_quality_gate",
        "artifact_version": 1,
        "artifact": str(artifact),
        "dataset_source": dataset.get("source"),
        "required_comparisons": len(REQUIRED_COMPARISONS),
        "minimum_wins": minimum_wins,
        "maximum_relative_rmse": maximum_relative_rmse,
        "wins": wins,
        "comparisons": comparisons,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    report = check_nyc_row_quality_gate(args.artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("NYC row quality gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
