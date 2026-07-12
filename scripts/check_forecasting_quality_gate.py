#!/usr/bin/env python3
"""Enforce the real lane-demand acceptance gate from a benchmark artifact.

The gate is deliberately artifact-based: it never recomputes metrics or
silently substitutes a different baseline. The artifact must contain at least
three leakage-safe rolling origins and an explicit aggregate comparison against
the strongest completed external learned baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_forecasting_quality_gate(
    artifact: Path | str,
    *,
    minimum_origins: int = 3,
    maximum_external_rmse_ratio: float = 1.05,
) -> dict[str, Any]:
    path = Path(artifact)
    report: dict[str, Any] = {
        "artifact_type": "cartoboost.forecasting_quality_gate",
        "artifact_version": 1,
        "artifact": str(path),
        "minimum_origins": int(minimum_origins),
        "maximum_external_rmse_ratio": float(maximum_external_rmse_ratio),
        "checks": {},
        "passed": False,
    }
    if not path.exists():
        report["checks"]["artifact_exists"] = False
        report["reason"] = "artifact does not exist"
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["checks"]["artifact_exists"] = False
        report["reason"] = f"artifact could not be parsed: {exc}"
        return report
    if not isinstance(payload, dict):
        report["checks"]["artifact_exists"] = False
        report["reason"] = "artifact root must be an object"
        return report

    rolling_origin = payload.get("rolling_origin")
    quality = payload.get("quality")
    comparability = payload.get("comparability_audit")
    if not isinstance(rolling_origin, dict) or not isinstance(quality, dict):
        report["reason"] = "artifact must contain rolling_origin and quality objects"
        return report
    if not isinstance(comparability, dict):
        report["reason"] = "artifact must contain a comparability_audit object"
        return report

    folds = int(rolling_origin.get("folds", 0) or 0)
    ratio = quality.get("rmse_ratio_vs_best_external_baseline")
    gate_passed = quality.get("external_baseline_rmse_gate_passed") is True
    ratio_valid = isinstance(ratio, int | float) and float(ratio) == float(ratio)
    ratio_within_limit = ratio_valid and float(ratio) <= float(maximum_external_rmse_ratio)
    split_rows = rolling_origin.get("splits")
    split_count = len(split_rows) if isinstance(split_rows, dict) else 0
    checks = report["checks"]
    checks.update(
        {
            "artifact_exists": True,
            "minimum_origin_count": folds >= minimum_origins,
            "recorded_split_count": split_count,
            "same_forecast_rows": comparability.get("same_forecast_rows") is True,
            "selection_uses_outer_test_labels": comparability.get(
                "selection_uses_outer_test_labels"
            )
            is False,
            "external_baseline_present": isinstance(quality.get("best_external_baseline"), str),
            "external_rmse_ratio": ratio,
            "external_rmse_ratio_within_limit": ratio_within_limit,
            "recorded_gate_passed": gate_passed,
        }
    )
    report["passed"] = all(
        [
            checks["artifact_exists"],
            checks["minimum_origin_count"],
            checks["recorded_split_count"] >= minimum_origins,
            checks["same_forecast_rows"],
            checks["selection_uses_outer_test_labels"],
            checks["external_baseline_present"],
            checks["external_rmse_ratio_within_limit"],
            checks["recorded_gate_passed"],
        ]
    )
    if not report["passed"]:
        report["reason"] = "real forecasting quality acceptance gate failed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--minimum-origins", type=int, default=3)
    parser.add_argument("--maximum-external-rmse-ratio", type=float, default=1.05)
    args = parser.parse_args()
    report = check_forecasting_quality_gate(
        args.artifact,
        minimum_origins=args.minimum_origins,
        maximum_external_rmse_ratio=args.maximum_external_rmse_ratio,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("forecasting quality gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
