#!/usr/bin/env python3
"""Measure input conversion and public calls across formats and batch sizes.

Each case runs in a fresh process. Fixtures and model fitting are outside timed
inference; public-call timings include input conversion and output construction.
These deterministic taxi-shaped fixtures are performance checks, not real-data
quality comparisons. Optional formats fail clearly when their package is absent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

try:
    import resource
except ImportError:  # Windows does not expose process RSS through resource.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ("numpy", "float32", "fortran", "strided", "pandas", "polars", "duckdb", "list")
OPERATIONS = (
    "parse_fit",
    "parse_predict",
    "reg_predict",
    "reg_additive",
    "reg_contributions",
    "reg_score",
    "classifier_predict",
    "classifier_proba",
    "classifier_margins",
    "multiclass_proba",
    "multiclass_margins",
    "classifier_score",
    "rank_predict",
    "rank_metrics",
    "fit_regressor",
    "fit_classifier",
    "fit_ranker",
    "forecast_frame",
    "forecast_metrics",
    "pinball",
)


def _rss_mb() -> float | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if sys.platform == "darwin" else value / 1024)


def _array(rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, features))
    target = 12.0 + 2.0 * values[:, 0] - values[:, 1] + 0.2 * np.sin(values[:, 2])
    return values, target


def _table(values: np.ndarray, kind: str) -> Any:
    columns = [f"taxi_feature_{i}" for i in range(values.shape[1])]
    if kind == "numpy":
        return values
    if kind == "float32":
        return values.astype(np.float32)
    if kind == "fortran":
        return np.asfortranarray(values)
    if kind == "strided":
        padded = np.empty((len(values), values.shape[1] * 2))
        padded[:, ::2] = values
        return padded[:, ::2]
    if kind == "pandas":
        import pandas as pd

        return pd.DataFrame(values, columns=columns)
    if kind == "polars":
        import polars as pl

        return pl.DataFrame(values, schema=columns, orient="row")
    if kind == "duckdb":
        import duckdb
        import pandas as pd

        return duckdb.from_df(pd.DataFrame(values, columns=columns))
    if kind == "list":
        return values.tolist()
    raise ValueError(f"unknown format {kind!r}")


def _groups(rows: int) -> list[int]:
    groups = [32] * (rows // 32)
    if rows % 32:
        groups.append(rows % 32)
    return groups


def _fit_model(
    cls: Any, config: dict[str, Any], table: Any, labels: Any, kwargs: dict[str, Any]
) -> Any:
    return cls(**config).fit(table, labels, **kwargs)


def _summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _summary(item) for key, item in sorted(value.items())}
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("benchmark operation returned nonfinite values")
    flat = array.ravel()
    sample = flat[np.linspace(0, max(0, len(flat) - 1), min(16, len(flat)), dtype=int)]
    return {
        "shape": list(array.shape),
        "sum": float(array.sum()),
        "sample": sample.tolist(),
        "rounded_sha256": hashlib.sha256(np.round(array, 9).tobytes()).hexdigest(),
    }


def _assert_equivalent(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_equivalent(actual[key], value)
        return
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    if actual_array.dtype.kind in "biufc" and expected_array.dtype.kind in "biufc":
        np.testing.assert_allclose(actual_array, expected_array, rtol=1e-10, atol=1e-10)
    else:
        # Classifier labels deliberately preserve object/string label types.
        np.testing.assert_array_equal(actual_array, expected_array)


def compare_outputs(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Check shapes and numeric summaries, also reporting rounded digest equality."""
    if before.keys() != after.keys():
        raise ValueError("before/after output fields differ")
    same_digest = True
    for key, value in before.items():
        other = after[key]
        if isinstance(value, dict):
            same_digest = compare_outputs(value, other) and same_digest
        elif key == "rounded_sha256":
            same_digest = value == other and same_digest
        elif key == "shape":
            if value != other:
                raise ValueError("before/after output shapes differ")
        else:
            np.testing.assert_allclose(value, other, rtol=1e-9, atol=1e-8)
    return same_digest


def run_case(case: dict[str, Any], package_root: Path, repeats: int) -> dict[str, Any]:
    sys.path.insert(0, str(package_root))
    import cartoboost
    from cartoboost import CartoBoostClassifier, CartoBoostRanker, CartoBoostRegressor
    from cartoboost.regressor import (
        _fit_transform_categorical_features,
        _transform_categorical_features,
    )

    if not Path(cartoboost.__file__).resolve().is_relative_to(package_root.resolve()):
        raise RuntimeError("benchmark imported a different package than --package-root")
    operation = case["operation"]
    rows, width = case["rows"], case["features"]
    values, targets = _array(rows, width, 20261124)
    table = _table(values, case["format"])
    # float32 inputs must be compared against their actual representable values.
    reference = (
        values.astype(np.float32).astype(np.float64) if case["format"] == "float32" else values
    )
    config = dict(
        n_estimators=case["estimators"],
        max_depth=3,
        min_samples_leaf=4,
        split_policy="structured",
        n_threads=case["threads"],
    )
    train, train_targets = _array(512, width, 20261125)
    probes = train[:32]
    predictor = None
    reference_call = None
    if operation == "parse_fit":
        call = partial(_fit_transform_categorical_features, table, targets, None)
        expected = reference
    elif operation == "parse_predict":
        call = partial(_transform_categorical_features, table, None)
        expected = reference
    elif operation.startswith("reg_"):
        model = CartoBoostRegressor(**config).fit(train, train_targets)
        methods = {
            "reg_predict": "predict",
            "reg_additive": "predict_additive_values",
            "reg_contributions": "predict_feature_contributions",
            "reg_score": "score",
        }
        method = getattr(model, methods[operation])
        args = (targets,) if operation == "reg_score" else ()
        call = partial(method, table, *args)
        reference_call = partial(method, reference, *args)
        expected = None
        if not args:
            predictor = method
    elif operation.startswith(("classifier_", "multiclass_")):
        multi = operation.startswith("multiclass_")
        classes = np.digitize(train[:, 0], [-0.4, 0.4]) if multi else (train[:, 0] > 0).astype(int)
        model = CartoBoostClassifier(**config).fit(train, classes)
        methods = {
            "predict": "predict",
            "proba": "predict_proba",
            "margins": "decision_function",
            "score": "score",
        }
        method = getattr(model, methods[operation.split("_", 1)[1]])
        args = ((values[:, 0] > 0).astype(int),) if operation.endswith("score") else ()
        call = partial(method, table, *args)
        reference_call = partial(method, reference, *args)
        expected = None
        if not args:
            predictor = method
    elif operation.startswith("rank_"):
        model = CartoBoostRanker(**config).fit(
            train, np.digitize(train[:, 0], [-0.5, 0.5]), groups=_groups(len(train))
        )
        if operation == "rank_predict":
            call = partial(model.predict, table)
            reference_call = partial(model.predict, reference)
            expected = None
            predictor = model.predict
        else:
            labels = np.digitize(values[:, 0], [-0.5, 0.5])
            call = partial(model.score_groups, table, labels, groups=_groups(rows))
            reference_call = partial(model.score_groups, reference, labels, groups=_groups(rows))
            expected = None
    elif operation.startswith("fit_"):
        if rows < 32:
            raise ValueError("fit benchmarks require at least 32 rows")
        kind = operation.removeprefix("fit_")
        cls = {
            "regressor": CartoBoostRegressor,
            "classifier": CartoBoostClassifier,
            "ranker": CartoBoostRanker,
        }[kind]
        labels = targets if kind == "regressor" else np.digitize(values[:, 0], [-0.5, 0.5])
        kwargs = {"groups": _groups(rows)} if kind == "ranker" else {}
        call = partial(_fit_model, cls, config, table, labels, kwargs)
        expected = None
    elif operation == "forecast_frame":
        import pandas as pd
        from cartoboost.forecasting import ForecastFrame

        frame = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"), "fare": targets}
        )
        call = partial(
            ForecastFrame.from_pandas, frame, timestamp_col="timestamp", target_col="fare", freq="h"
        )
        expected = None
    elif operation == "forecast_metrics":
        from cartoboost.forecasting.metrics import ForecastMetricSet

        metric = ForecastMetricSet()
        call = partial(metric.evaluate, targets, targets + 0.1, y_train=train_targets)
        expected = None
    elif operation == "pinball":
        from cartoboost.forecasting.metrics import pinball_loss

        call = partial(pinball_loss, targets, targets + 0.1, 0.5)
        expected = 0.05
    else:
        raise ValueError(operation)

    rss_before = _rss_mb()
    first_started = time.perf_counter_ns()
    warm = call()
    first_seconds = (time.perf_counter_ns() - first_started) / 1e9
    # Capture this before parity checks and output hashing allocate more arrays.
    rss_after_first_call = _rss_mb()
    if reference_call is not None:
        expected = reference_call()
    if operation == "parse_fit":
        warm = warm[0]
    fit_input_parity = False
    if operation.startswith("fit_"):
        reference_model = _fit_model(cls, config, reference, labels, kwargs)
        _assert_equivalent(warm.predict(probes), reference_model.predict(probes))
        fit_input_parity = True
    if expected is not None:
        _assert_equivalent(warm, expected)
    parity_rows = min(rows, 32)
    if predictor is not None:
        individual = np.concatenate(
            [np.atleast_1d(predictor(reference[i : i + 1])) for i in range(parity_rows)], axis=0
        )
        _assert_equivalent(np.asarray(warm)[:parity_rows], individual)
    # Amortize timer resolution for short calls without concealing first-call cost.
    calibration_started = time.perf_counter_ns()
    call()
    calibration_seconds = (time.perf_counter_ns() - calibration_started) / 1e9
    iterations = min(1000, max(1, int(0.02 / max(calibration_seconds, 1e-9))))
    if operation.startswith("fit_") or rows >= 65536:
        iterations = 1
    elapsed = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter_ns()
        for _ in range(iterations):
            result = call()
        elapsed.append((time.perf_counter_ns() - started) / 1e9 / iterations)
    if operation.startswith("fit_"):
        output = _summary(result.predict(probes))
    elif operation == "forecast_frame":
        output = {"rows": result.n_rows, "columns": 2}
    else:
        output = _summary(result[0] if operation == "parse_fit" else result)
    seconds = statistics.median(elapsed)
    return {
        **case,
        "package": cartoboost.__file__,
        "first_call_ms": first_seconds * 1000,
        "median_ms": seconds * 1000,
        "rows_per_second": rows / seconds,
        "microseconds_per_row": seconds * 1e6 / rows,
        "samples_seconds": elapsed,
        "iterations_per_sample": iterations,
        "peak_process_rss_mb": rss_after_first_call,
        "operation_peak_growth_mb": (
            None
            if rss_after_first_call is None or rss_before is None
            else max(0.0, rss_after_first_call - rss_before)
        ),
        "numpy_input_parity": expected is not None or fit_input_parity,
        "single_row_parity_rows": parity_rows if predictor is not None else 0,
        "output": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT / "python")
    parser.add_argument(
        "--compare-root",
        type=Path,
        help="Frozen baseline package; interleave isolated baseline/candidate cases",
    )
    parser.add_argument("--formats", default="numpy,pandas,polars,duckdb")
    parser.add_argument(
        "--operations", default="parse_fit,parse_predict,reg_predict,classifier_proba,rank_predict"
    )
    parser.add_argument("--sizes", default="1,32,1024,65536")
    parser.add_argument("--threads", default="1")
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--estimators", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.repeats <= 0 or args.features < 3 or args.estimators <= 0:
        parser.error("repeats/estimators must be positive and features must be >=3")
    if args.worker:
        print(json.dumps(run_case(json.loads(args.worker), args.package_root, args.repeats)))
        return 0
    formats, operations = args.formats.split(","), args.operations.split(",")
    sizes, threads = (
        [int(n) for n in args.sizes.split(",")],
        [int(n) for n in args.threads.split(",")],
    )
    if set(formats) - set(FORMATS) or set(operations) - set(OPERATIONS):
        parser.error("unknown operation or format")
    if any(n < 1 for n in sizes) or any(n < 0 for n in threads):
        parser.error("sizes must be positive; threads must be >=0 (0 means default)")
    cases = [
        dict(
            operation=op,
            format=fmt,
            rows=n,
            features=args.features,
            estimators=args.estimators,
            threads=t or None,
        )
        for op in operations
        for fmt in formats
        for n in sizes
        for t in threads
        if not (op.startswith("fit_") and n < 32)
        if not (op == "reg_score" and n < 2)
        if not (
            op in {"forecast_frame", "forecast_metrics", "pinball"}
            and (fmt != "numpy" or t != threads[0])
        )
    ]
    if not cases:
        parser.error("no valid benchmark cases selected")
    report = {
        "artifact_type": "cartoboost.input_batch_performance",
        "artifact_version": 2,
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "package_root": str(args.package_root.resolve()),
        "cases": [],
        "comparisons": [],
        "notes": (
            "Synthetic fixtures. RSS through the first public call includes setup/imports; "
            "peak growth is a high-water increment, not allocation volume. Memory is read "
            "before parity checks or output hashing. First call follows model fitting."
        ),
    }
    for index, case in enumerate(cases, 1):
        variants = [("candidate", args.package_root)]
        if args.compare_root is not None:
            variants.insert(0, ("baseline", args.compare_root))
            if index % 2 == 0:
                variants.reverse()
        results = {}
        for label, package_root in variants:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--package-root",
                str(package_root.resolve()),
                "--repeats",
                str(args.repeats),
                "--worker",
                json.dumps(case),
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False, timeout=300
            )
            if completed.returncode:
                raise RuntimeError(
                    f"{label} case {case} failed:\n{completed.stderr}\n{completed.stdout}"
                )
            result = json.loads(completed.stdout)
            result["variant"] = label
            results[label] = result
            report["cases"].append(result)
        if args.compare_root is not None:
            before, after = results["baseline"], results["candidate"]
            exact = compare_outputs(before["output"], after["output"])
            report["comparisons"].append(
                {
                    **case,
                    "baseline_ms": before["median_ms"],
                    "candidate_ms": after["median_ms"],
                    "speedup": before["median_ms"] / after["median_ms"],
                    "output_summary_parity": True,
                    "rounded_output_digest_equal": exact,
                }
            )
        print(
            f"{index}/{len(cases)} {case['operation']} {case['format']} "
            f"n={case['rows']} threads={case['threads']}",
            flush=True,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.output:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
