#!/usr/bin/env python3
"""Validate the v0.3 scale-performance artifact against release budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check(
    path: Path,
    *,
    minimum_rows: int = 1_000_000,
    max_fit_seconds: float = 3_600.0,
    max_peak_rss_mb: float = 24_576.0,
    minimum_predict_rows_per_second: float = 1_000_000.0,
    minimum_thread_speedup: float = 3.0,
    require_thread_speedup: bool = True,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("scale-performance artifact is missing workload")
    checks = {
        "rows": int(workload.get("rows", 0)) >= minimum_rows,
        "fit_seconds": float(workload.get("fit_seconds", float("inf"))) <= max_fit_seconds,
        "peak_rss_mb": float(workload.get("peak_rss_mb", float("inf"))) <= max_peak_rss_mb,
        "predict_rows_per_second": float(workload.get("predict_rows_per_second", 0.0))
        >= minimum_predict_rows_per_second,
    }
    if "thread_speedup" in workload:
        checks["thread_speedup"] = float(workload["thread_speedup"]) >= minimum_thread_speedup
    else:
        checks["thread_speedup"] = not require_thread_speedup
    return {
        "artifact_type": "cartoboost.scale_performance_audit",
        "artifact_version": 1,
        "path": str(path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--minimum-rows", type=int, default=1_000_000)
    parser.add_argument("--max-fit-seconds", type=float, default=3_600.0)
    parser.add_argument("--max-peak-rss-mb", type=float, default=24_576.0)
    parser.add_argument("--minimum-predict-rows-per-second", type=float, default=1_000_000.0)
    parser.add_argument("--minimum-thread-speedup", type=float, default=3.0)
    parser.add_argument(
        "--allow-missing-thread-speedup",
        action="store_true",
        help="Permit smoke artifacts without a 1-thread baseline; release checks require it.",
    )
    args = parser.parse_args()
    report = check(
        args.artifact,
        minimum_rows=args.minimum_rows,
        max_fit_seconds=args.max_fit_seconds,
        max_peak_rss_mb=args.max_peak_rss_mb,
        minimum_predict_rows_per_second=args.minimum_predict_rows_per_second,
        minimum_thread_speedup=args.minimum_thread_speedup,
        require_thread_speedup=not args.allow_missing_thread_speedup,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("scale performance gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
