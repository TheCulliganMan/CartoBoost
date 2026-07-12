#!/usr/bin/env python3
"""Measure the maintained structured fit on a bounded, reproducible CPU workload.

The default workload is intentionally large enough to expose conversion and
allocation regressions.  CI uses a smaller smoke size; release qualification
uses the 1M/10M-row settings from the v0.3 beta plan on the fixed host.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from cartoboost import CartoBoostRegressor
from cartoboost.config import SplitPolicy
from cartoboost.schema import (
    FeatureSchema,
    NumericSpec,
    PeriodicSpec,
    SparseSetSpec,
    SpatialPairSpec,
)

ROOT = Path(__file__).resolve().parents[1]


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return float(value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024)


def _workload(
    rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray, FeatureSchema, dict[str, list[list[int]]]]:
    rng = np.random.default_rng(seed)
    hour = np.arange(rows, dtype=np.float64) % 24.0
    spatial_x = rng.normal(size=rows)
    spatial_y = rng.normal(size=rows)
    dense = np.column_stack(
        [
            rng.normal(size=(rows, 17)),
            hour,
            spatial_x,
            spatial_y,
        ]
    )
    target = (
        dense[:, 0] * 0.7
        + np.sin(hour * 2.0 * np.pi / 24.0)
        + 0.2 * spatial_x
        - 0.15 * spatial_y
        + rng.normal(scale=0.05, size=rows)
    )
    schema = FeatureSchema.from_specs(
        [NumericSpec(f"feature_{idx}") for idx in range(17)]
        + [
            PeriodicSpec("hour", 24),
            SpatialPairSpec("x", "y"),
            SpatialPairSpec("y", "x"),
        ],
        [SparseSetSpec("membership")],
    )
    sparse_sets = {"membership": [[int(row % 97)] for row in range(rows)]}
    return dense, target, schema, sparse_sets


def run(rows: int, threads: int, estimators: int, seed: int) -> dict[str, Any]:
    x, y, schema, sparse_sets = _workload(rows, seed)
    model = CartoBoostRegressor(
        n_estimators=estimators,
        max_depth=5,
        min_samples_leaf=20,
        split_policy=SplitPolicy.STRUCTURED,
        n_threads=threads,
    )
    started = time.perf_counter()
    model.fit(x, y, feature_schema=schema, sparse_sets=sparse_sets)
    fit_seconds = time.perf_counter() - started
    predict_rows = min(rows, 1_000_000)
    started = time.perf_counter()
    predictions = model.predict(
        x[:predict_rows], sparse_sets={"membership": sparse_sets["membership"][:predict_rows]}
    )
    predict_seconds = time.perf_counter() - started
    return {
        "rows": rows,
        "features": int(x.shape[1]),
        "estimators": estimators,
        "threads": threads,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "predict_rows_per_second": predict_rows / max(predict_seconds, 1e-12),
        "peak_rss_mb": _peak_rss_mb(),
        "prediction_checksum": float(np.sum(predictions, dtype=np.float64)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--baseline-threads", type=int)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 100 or args.threads <= 0 or args.estimators <= 0:
        raise SystemExit("rows must be >=100 and threads/estimators must be positive")
    workload = run(args.rows, args.threads, args.estimators, args.seed)
    baseline = None
    if args.baseline_threads is not None:
        if args.baseline_threads <= 0:
            raise SystemExit("baseline-threads must be positive")
        baseline = run(args.rows, args.baseline_threads, args.estimators, args.seed)
        workload["thread_speedup"] = baseline["fit_seconds"] / max(workload["fit_seconds"], 1e-12)
    result = {
        "artifact_type": "cartoboost.scale_performance_gate",
        "artifact_version": 1,
        "git_commit": _commit(),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "workload": workload,
    }
    if baseline is not None:
        result["baseline"] = baseline
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
