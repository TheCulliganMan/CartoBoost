#!/usr/bin/env python3
"""Validate a sanitized, non-public v0.3 private shadow-gate report.

The evaluator that owns confidential data must write the input JSON locally.
This checker intentionally emits only aggregate counts and pass/fail state: it
never copies row data, field names, segment identifiers, or raw private
results into a report. Keep both the input and output outside the repository.

Input contract::

    {
      "schema_version": 1,
      "metric_name": "rmse",
      "candidate": {
        "primary": 1.0,
        "segments": [{"metric": 1.0, "important": true}],
        "artifact_roundtrip": true,
        "fit_seconds": 10.0,
        "peak_rss_mb": 100.0,
        "predict_rows_per_second": 1000000.0
      },
      "incumbent": {
        "primary": 1.0,
        "segments": [{"metric": 1.0}]
      }
    }

All segment arrays are positional and must have the same length. The private
evaluator may keep identifiers in its input, but this checker ignores them and
does not echo them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_MAX_SEGMENT_REGRESSION = 0.05
DEFAULT_MAX_FIT_SECONDS = 3_600.0
DEFAULT_MAX_PEAK_RSS_MB = 24_576.0
DEFAULT_MIN_PREDICT_ROWS_PER_SECOND = 1_000_000.0
DEFAULT_MIN_THREAD_SPEEDUP = 3.0
ALLOWED_METRICS = {"mae", "mape", "primary", "r2", "rmse", "wape"}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _segment_metrics(value: Any, name: str) -> list[tuple[float, bool]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    segments: list[tuple[float, bool]] = []
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        metric = _finite(segment.get("metric"), f"{name}[{index}].metric")
        important = segment.get("important", True)
        if not isinstance(important, bool):
            raise ValueError(f"{name}[{index}].important must be boolean")
        segments.append((metric, important))
    return segments


def check_private_shadow(
    payload: dict[str, Any],
    *,
    max_segment_regression: float = DEFAULT_MAX_SEGMENT_REGRESSION,
    max_fit_seconds: float = DEFAULT_MAX_FIT_SECONDS,
    max_peak_rss_mb: float = DEFAULT_MAX_PEAK_RSS_MB,
    min_predict_rows_per_second: float = DEFAULT_MIN_PREDICT_ROWS_PER_SECOND,
    min_thread_speedup: float = DEFAULT_MIN_THREAD_SPEEDUP,
) -> dict[str, Any]:
    """Return a sanitized gate report without copying confidential values."""

    if payload.get("schema_version") != 1:
        raise ValueError("private shadow input schema_version must be 1")
    metric_name = payload.get("metric_name")
    if not isinstance(metric_name, str) or metric_name.strip().lower() not in ALLOWED_METRICS:
        raise ValueError("private shadow metric_name must be a generic supported metric")
    metric_name = metric_name.strip().lower()
    candidate = payload.get("candidate")
    incumbent = payload.get("incumbent")
    if not isinstance(candidate, dict) or not isinstance(incumbent, dict):
        raise ValueError("private shadow candidate and incumbent must be objects")

    candidate_primary = _finite(candidate.get("primary"), "candidate.primary")
    incumbent_primary = _finite(incumbent.get("primary"), "incumbent.primary")
    if incumbent_primary <= 0.0:
        raise ValueError("incumbent.primary must be positive for relative comparison")
    candidate_segments = _segment_metrics(candidate.get("segments"), "candidate.segments")
    incumbent_segments = _segment_metrics(incumbent.get("segments"), "incumbent.segments")
    if len(candidate_segments) != len(incumbent_segments):
        raise ValueError("candidate and incumbent segment counts must match")

    primary_relative_delta = (candidate_primary - incumbent_primary) / incumbent_primary
    important_regressions = [
        (candidate_metric - incumbent_metric) / max(abs(incumbent_metric), 1.0e-12)
        for (candidate_metric, important), (incumbent_metric, _) in zip(
            candidate_segments, incumbent_segments, strict=True
        )
        if important
    ]
    max_important_regression = max(important_regressions, default=0.0)
    artifact_roundtrip = candidate.get("artifact_roundtrip") is True
    fit_seconds = _finite(candidate.get("fit_seconds"), "candidate.fit_seconds")
    peak_rss_mb = _finite(candidate.get("peak_rss_mb"), "candidate.peak_rss_mb")
    predict_rows_per_second = _finite(
        candidate.get("predict_rows_per_second"), "candidate.predict_rows_per_second"
    )
    thread_speedup = candidate.get("thread_speedup")
    thread_speedup_value = (
        None if thread_speedup is None else _finite(thread_speedup, "candidate.thread_speedup")
    )

    checks = {
        "primary_not_worse": primary_relative_delta <= 0.0,
        "important_segments_within_budget": max_important_regression <= max_segment_regression,
        "artifact_roundtrip": artifact_roundtrip,
        "fit_budget": fit_seconds <= max_fit_seconds,
        "memory_budget": peak_rss_mb <= max_peak_rss_mb,
        "prediction_budget": predict_rows_per_second >= min_predict_rows_per_second,
        "thread_speedup": thread_speedup_value is not None
        and thread_speedup_value >= min_thread_speedup,
    }
    return {
        "artifact_type": "cartoboost.private_shadow_gate",
        "artifact_version": 1,
        "metric_name": metric_name,
        "candidate_beats_or_ties_incumbent": checks["primary_not_worse"],
        "segment_count": len(candidate_segments),
        "important_segment_count": sum(important for _, important in candidate_segments),
        "max_important_segment_regression": max_important_regression,
        "artifact_roundtrip": artifact_roundtrip,
        "thread_speedup_recorded": thread_speedup_value is not None,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="private evaluator JSON; keep it outside the repo")
    parser.add_argument("--output", type=Path, required=True, help="private sanitized report path")
    parser.add_argument(
        "--max-segment-regression", type=float, default=DEFAULT_MAX_SEGMENT_REGRESSION
    )
    parser.add_argument("--max-fit-seconds", type=float, default=DEFAULT_MAX_FIT_SECONDS)
    parser.add_argument("--max-peak-rss-mb", type=float, default=DEFAULT_MAX_PEAK_RSS_MB)
    parser.add_argument(
        "--min-predict-rows-per-second",
        type=float,
        default=DEFAULT_MIN_PREDICT_ROWS_PER_SECOND,
    )
    parser.add_argument("--min-thread-speedup", type=float, default=DEFAULT_MIN_THREAD_SPEEDUP)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("private shadow input must be a JSON object")
    report = check_private_shadow(
        payload,
        max_segment_regression=args.max_segment_regression,
        max_fit_seconds=args.max_fit_seconds,
        max_peak_rss_mb=args.max_peak_rss_mb,
        min_predict_rows_per_second=args.min_predict_rows_per_second,
        min_thread_speedup=args.min_thread_speedup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("private shadow gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
