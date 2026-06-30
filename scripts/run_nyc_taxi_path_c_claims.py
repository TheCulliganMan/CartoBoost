#!/usr/bin/env python3
"""Run claim-based NYC TLC Path C gates for CartoBoost.

The runner reuses the maintained NYC taxi benchmark frame and emits a claim
ledger for bounded real-data evidence. It intentionally does not add model
families; falsifiers are feature-policy variants and simple train-only
baselines around CartoBoost/Ridge behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

import run_nyc_taxi_quality_benchmarks as base  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks"
PRIMARY_MODEL = "cartoboost"
PRIMARY_ARCHITECTURE = "cartoboost_geo_temporal_trees"
PRIMARY_TIER = "path_c_real_data_claim"
REQUIRED_ROW_FIELDS = [
    "claim_id",
    "task",
    "split_kind",
    "train_index_sha256",
    "test_index_sha256",
    "dataset_hash",
    "model",
    "architecture",
    "capability_tier",
    "falsifier_baseline",
    "primary_metric",
    "improvement_threshold",
    "percent_improvement",
    "rmse",
    "mae",
    "wape",
    "r2",
    "fit_seconds",
    "predict_seconds",
    "peak_memory_mb",
    "save_load_max_abs_diff",
    "feature_access_policy",
    "prediction_unit",
    "primary_prediction_unit",
    "falsifier_prediction_unit",
    "task_frame_id",
    "primary_train_rows",
    "primary_test_rows",
    "falsifier_train_rows",
    "falsifier_test_rows",
    "primary_train_index_sha256",
    "primary_test_index_sha256",
    "falsifier_train_index_sha256",
    "falsifier_test_index_sha256",
    "target_encoding_train_only",
    "selection_uses_outer_test_labels",
]

CLAIM_THRESHOLDS = {
    "directional_structure": 0.02,
    "temporal_structure": 0.02,
    "known_future_sensitivity": 0.01,
    "spatial_transfer": 0.01,
    "residual_correction": 0.01,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument(
        "--cartoboost-splitters",
        default="axis_histogram:512,diagonal_2d,gaussian_2d,periodic:24,periodic:7",
    )
    parser.add_argument("--zone-target-smoothing", type=float, default=20.0)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Developer-only tiny fixture. Checker rejects it as Path C evidence.",
    )
    return parser.parse_args()


def quality_args(args: argparse.Namespace) -> argparse.Namespace:
    n_estimators = min(args.n_estimators, 4) if args.synthetic_smoke else args.n_estimators
    splitters = (
        "axis_histogram:64,periodic:24,periodic:7"
        if args.synthetic_smoke
        else args.cartoboost_splitters
    )
    return argparse.Namespace(
        year=args.year,
        months=args.months,
        taxi_type=args.taxi_type,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        no_download=args.no_download,
        synthetic_smoke=args.synthetic_smoke,
        models=PRIMARY_MODEL,
        n_estimators=n_estimators,
        cartoboost_n_estimators=n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        cartoboost_max_depth=args.max_depth,
        cartoboost_splitters=splitters,
        cartoboost_min_samples_leaf=args.min_samples_leaf,
        cartoboost_constant_l2=0.0,
        cartoboost_leaf_predictor="constant",
        cartoboost_init="constant",
        cartoboost_calibration="none",
        xgboost_tree_method="hist",
        xgboost_max_bin=256,
        xgboost_subsample=1.0,
        xgboost_colsample_bytree=1.0,
        neural_dim=12,
        graph_dim=8,
        graph_epochs=8,
        graph_family="graphsage",
        zone_treatment="target_mean",
        zone_target_smoothing=args.zone_target_smoothing,
        model_workers=1,
        n_threads=args.n_threads,
        no_plots=True,
        tasks="",
    )


def with_cartoboost_budget(
    args: argparse.Namespace,
    *,
    n_estimators: int,
    max_depth: int,
) -> argparse.Namespace:
    cloned = copy.copy(args)
    cloned.cartoboost_n_estimators = max(int(args.cartoboost_n_estimators), int(n_estimators))
    cloned.cartoboost_max_depth = max(int(args.cartoboost_max_depth), int(max_depth))
    return cloned


def load_tasks(args: argparse.Namespace, qargs: argparse.Namespace) -> list[base.BenchmarkTask]:
    if args.synthetic_smoke:
        qargs._source_paths = []
        return base.synthetic_tasks()
    months = base.parse_months(args.months)
    paths = base.ensure_parquet_files(
        taxi_type=args.taxi_type,
        year=args.year,
        months=months,
        cache_dir=args.cache_dir,
        no_download=args.no_download,
    )
    zone_lookup = base.ensure_zone_lookup(cache_dir=args.cache_dir, no_download=args.no_download)
    zone_adjacency = base.ensure_zone_adjacency(
        cache_dir=args.cache_dir,
        no_download=args.no_download,
    )
    zone_centroids = base.ensure_zone_centroids(
        cache_dir=args.cache_dir,
        no_download=args.no_download,
    )
    qargs._source_paths = [
        *paths,
        args.cache_dir / "taxi_zone_lookup.csv",
        args.cache_dir / "taxi_zones.zip",
    ]
    cleaned = base.clean_tlc_frame(base.load_tlc_frame(paths))
    sampled = base.sample_tlc_frame(cleaned, sample_size=args.sample_size, seed=args.seed)
    tasks = base.build_real_tasks(
        sampled,
        zone_lookup,
        zone_adjacency,
        zone_centroids=zone_centroids,
        demand_frame=cleaned,
    )
    demand = chronological_demand_task(
        cleaned,
        zone_adjacency=zone_adjacency,
        zone_centroids=zone_centroids,
    )
    return [demand if task.name == "pickup_demand" else task for task in tasks]


def chronological_demand_task(
    frame: Any,
    *,
    zone_adjacency: dict[int, list[int]] | None,
    zone_centroids: dict[int, tuple[float, float]] | None,
) -> base.BenchmarkTask:
    pandas = base.optional_import("pandas")
    if pandas is None:
        raise RuntimeError("pandas is required for chronological pickup-demand task construction")
    data = frame.copy()
    pickup_time = pandas.to_datetime(data["tpep_pickup_datetime"])
    data["pickup_period_start"] = pickup_time.dt.floor("h")
    data["pickup_period_unix"] = data["pickup_period_start"].map(
        lambda timestamp: int(timestamp.timestamp())
    )
    data["month"] = pickup_time.dt.month.astype(float)
    data["dayofmonth"] = pickup_time.dt.day.astype(float)
    grouped = (
        data.groupby(
            [
                "pickup_period_start",
                "pickup_period_unix",
                "PULocationID",
                "hour",
                "dayofweek",
                "month",
                "dayofmonth",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "trip_count"})
    )
    feature_names = [
        "PULocationID",
        "hour",
        "dayofweek",
        "month",
        "dayofmonth",
        "pickup_period_unix",
    ]
    features = grouped[feature_names].to_numpy(dtype=float)
    target = np.log1p(grouped["trip_count"].to_numpy(dtype=float))
    pickup_zones = grouped["PULocationID"].to_numpy(dtype=int)
    return base.BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description=("Predict log pickup trip count for a pickup zone and real hourly timestamp."),
        features=features,
        target=target,
        pickup_zones=pickup_zones,
        feature_names=feature_names,
        sparse_sets={"pickup_zone": [[int(value)] for value in pickup_zones]},
        zone_adjacency=zone_adjacency,
        zone_centroids=zone_centroids,
    )


def rolling_origin_indices(task: base.BenchmarkTask) -> tuple[np.ndarray, np.ndarray, str | None]:
    if "pickup_period_unix" not in task.feature_names:
        hour = feature_column(task, "hour")
        day = feature_column(task, "dayofweek")
        order_key = day * 24.0 + hour
        cutoff = float(np.quantile(order_key, 0.8))
        cutoff_timestamp = None
    else:
        order_key = feature_column(task, "pickup_period_unix")
        cutoff = float(np.quantile(order_key, 0.8))
        cutoff_timestamp = np.datetime_as_string(
            np.datetime64(0, "s") + np.timedelta64(int(cutoff), "s"),
            unit="s",
        )
    train = np.flatnonzero(order_key < cutoff)
    test = np.flatnonzero(order_key >= cutoff)
    if train.size == 0 or test.size == 0:
        raise ValueError(f"rolling-origin split for {task.name} produced an empty side")
    return train, test, cutoff_timestamp


def feature_column(task: base.BenchmarkTask, name: str) -> np.ndarray:
    try:
        index = task.feature_names.index(name)
    except ValueError as exc:
        raise ValueError(f"task {task.name} does not have feature {name}") from exc
    return task.features[:, index]


def path_c_split_manifest(
    qargs: argparse.Namespace,
    task: base.BenchmarkTask,
    split_mode: str,
    train: np.ndarray,
    test: np.ndarray,
    *,
    dataset_hash: str,
) -> dict[str, Any]:
    manifest = base.split_manifest_metadata(
        qargs,
        task,
        split_mode,
        train,
        test,
        dataset_fingerprint=dataset_hash,
    )
    manifest["train_rows"] = int(len(train))
    manifest["test_rows"] = int(len(test))
    return manifest


def select_features(task: base.BenchmarkTask, names: list[str]) -> base.BenchmarkTask:
    indices = [task.feature_names.index(name) for name in names]
    sparse = {
        key: value
        for key, value in task.sparse_sets.items()
        if not (key.startswith("dropoff") and "DOLocationID" not in names)
    }
    return replace(
        task,
        features=task.features[:, indices].astype(float, copy=False),
        feature_names=names,
        sparse_sets=sparse,
    )


def unordered_pair_task(task: base.BenchmarkTask) -> base.BenchmarkTask:
    pu = feature_column(task, "PULocationID")
    do = feature_column(task, "DOLocationID")
    low = np.minimum(pu, do).astype(int)
    high = np.maximum(pu, do).astype(int)
    features = task.features.copy()
    features[:, task.feature_names.index("PULocationID")] = low
    features[:, task.feature_names.index("DOLocationID")] = high
    return replace(task, features=features, zone_centroids=None)


def route_time_task(task: base.BenchmarkTask) -> base.BenchmarkTask:
    pu = feature_column(task, "PULocationID").astype(int)
    do = feature_column(task, "DOLocationID").astype(int)
    hour = feature_column(task, "hour").astype(int)
    day = feature_column(task, "dayofweek").astype(int)
    sums: dict[tuple[int, int, int, int], float] = {}
    counts: dict[tuple[int, int, int, int], int] = {}
    for key, target in zip(zip(pu, do, hour, day, strict=True), task.target, strict=True):
        sums[key] = sums.get(key, 0.0) + float(target)
        counts[key] = counts.get(key, 0) + 1
    rows = sorted(sums)
    features = np.asarray(
        [[float(key[2]), float(key[3]), float(key[0]), float(key[1])] for key in rows],
        dtype=float,
    )
    target = np.asarray([sums[key] / counts[key] for key in rows], dtype=float)
    pickup_zones = features[:, 2].astype(int)
    dropoff_zones = features[:, 3].astype(int)
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in pickup_zones],
        "dropoff_zone": [[int(value)] for value in dropoff_zones],
    }
    return base.BenchmarkTask(
        name=task.name,
        display_name=f"{task.display_name} route-time",
        description=f"{task.description} Aggregated to pickup/dropoff zone by hour/day bucket.",
        features=features,
        target=target,
        pickup_zones=pickup_zones,
        feature_names=["hour", "dayofweek", "PULocationID", "DOLocationID"],
        sparse_sets=sparse_sets,
        zone_adjacency=task.zone_adjacency,
        zone_centroids=task.zone_centroids,
    )


def additive_task(task: base.BenchmarkTask) -> base.BenchmarkTask:
    names = [
        name
        for name in ["hour", "dayofweek", "PULocationID", "DOLocationID"]
        if name in task.feature_names
    ]
    return select_features(task, names)


def ablated_future_task(task: base.BenchmarkTask) -> base.BenchmarkTask:
    return select_features(task, ["PULocationID"])


def temporal_model_task(task: base.BenchmarkTask) -> base.BenchmarkTask:
    return select_features(
        task,
        ["PULocationID", "hour", "dayofweek", "month", "dayofmonth"],
    )


def zone_only_task(task: base.BenchmarkTask) -> base.BenchmarkTask:
    names = (
        ["PULocationID", "DOLocationID"]
        if "DOLocationID" in task.feature_names
        else ["PULocationID"]
    )
    return select_features(task, names)


def fit_cartoboost(
    task: base.BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    qargs: argparse.Namespace,
) -> dict[str, Any]:
    from cartoboost import CartoBoostRegressor

    train_x, test_x, effective_names = base.transformed_split_features(
        task, train_indices, test_indices, qargs
    )
    train_y = task.target[train_indices]
    test_y = task.target[test_indices]
    splitters = base.parse_splitters(qargs.cartoboost_splitters)
    use_dense_id_sets = base.splitters_use_dense_id_sets(splitters)
    use_sparse_sets = base.splitters_need_sparse_sets(splitters) and not use_dense_id_sets
    train_sparse = base.sparse_subset(task.sparse_sets, train_indices) if use_sparse_sets else None
    test_sparse = base.sparse_subset(task.sparse_sets, test_indices) if use_sparse_sets else None
    schema = base.cartoboost_schema(
        task,
        feature_names=effective_names,
        dense_id_sets=use_dense_id_sets,
        include_sparse_sets=use_sparse_sets,
    )
    model = CartoBoostRegressor(
        n_estimators=qargs.cartoboost_n_estimators,
        learning_rate=qargs.learning_rate,
        max_depth=qargs.cartoboost_max_depth,
        min_samples_leaf=qargs.cartoboost_min_samples_leaf,
        min_gain=0.0,
        splitters=splitters,
        leaf_predictor=qargs.cartoboost_leaf_predictor,
    )
    tracemalloc.start()
    fit_started = time.perf_counter()
    model.fit(train_x, train_y, sparse_sets=train_sparse, feature_schema=schema)
    fit_seconds = time.perf_counter() - fit_started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    predict_started = time.perf_counter()
    prediction = np.asarray(model.predict(test_x, sparse_sets=test_sparse), dtype=float)
    predict_seconds = time.perf_counter() - predict_started
    save_load_diff = save_load_max_abs_diff(model, test_x, prediction, sparse_sets=test_sparse)
    return {
        "metrics": base.metric_summary(test_y, prediction),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_memory_mb": float(peak) / (1024.0 * 1024.0),
        "save_load_max_abs_diff": save_load_diff,
        "prediction": prediction,
    }


def save_load_max_abs_diff(
    model: Any,
    test_x: np.ndarray,
    prediction: np.ndarray,
    *,
    sparse_sets: dict[str, list[list[int]]] | None,
) -> float:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = Path(handle.name)
    try:
        model.save(path)
        loaded = model.__class__.load(path)
        loaded_prediction = np.asarray(
            loaded.predict(test_x, sparse_sets=sparse_sets),
            dtype=float,
        )
    finally:
        path.unlink(missing_ok=True)
    return float(np.max(np.abs(prediction - loaded_prediction))) if prediction.size else 0.0


def timed_prediction(
    actual: np.ndarray,
    prediction_fn: Callable[[], np.ndarray],
) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    prediction = np.asarray(prediction_fn(), dtype=float)
    fit_seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "metrics": base.metric_summary(actual, prediction),
        "fit_seconds": fit_seconds,
        "predict_seconds": 0.0,
        "peak_memory_mb": float(peak) / (1024.0 * 1024.0),
        "save_load_max_abs_diff": 0.0,
        "prediction": prediction,
    }


def mean_baseline(task: base.BenchmarkTask, train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    mean_value = float(np.mean(task.target[train]))
    return timed_prediction(task.target[test], lambda: np.full(test.size, mean_value))


def ridge_baseline(task: base.BenchmarkTask, train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    train_x = task.features[train]
    test_x = task.features[test]
    train_y = task.target[train]

    def predict() -> np.ndarray:
        return ridge_fit_predict(train_x, train_y, test_x, alpha=1.0)

    return timed_prediction(task.target[test], predict)


def grouped_mean_baseline(
    task: base.BenchmarkTask,
    train: np.ndarray,
    test: np.ndarray,
    *,
    keys: list[str],
) -> dict[str, Any]:
    key_columns = [feature_column(task, key) for key in keys]
    prediction = grouped_mean_prediction(task, train, test, key_columns=key_columns)

    return timed_prediction(task.target[test], lambda: prediction)


def seasonal_naive_baseline(
    task: base.BenchmarkTask,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    required = ["PULocationID", "hour", "dayofweek", "pickup_period_unix"]
    key_columns = [feature_column(task, key) for key in required[:3]]
    time_column = feature_column(task, "pickup_period_unix")
    latest: dict[tuple[int, int, int], tuple[float, float]] = {}
    global_mean = float(np.mean(task.target[train]))
    for row_index in train:
        key = tuple(int(column[row_index]) for column in key_columns)
        timestamp = float(time_column[row_index])
        current = latest.get(key)
        if current is None or timestamp > current[0]:
            latest[key] = (timestamp, float(task.target[row_index]))

    def predict() -> np.ndarray:
        values = []
        for row_index in test:
            key = tuple(int(column[row_index]) for column in key_columns)
            values.append(latest.get(key, (0.0, global_mean))[1])
        return np.asarray(values, dtype=float)

    return timed_prediction(task.target[test], predict)


def grouped_mean_prediction(
    task: base.BenchmarkTask,
    train: np.ndarray,
    test: np.ndarray,
    *,
    key_columns: list[np.ndarray],
) -> np.ndarray:
    train_keys = list(zip(*(column[train].astype(int) for column in key_columns), strict=True))
    test_keys = list(zip(*(column[test].astype(int) for column in key_columns), strict=True))
    sums: dict[tuple[int, ...], float] = {}
    counts: dict[tuple[int, ...], int] = {}
    for key, value in zip(train_keys, task.target[train], strict=True):
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1
    global_mean = float(np.mean(task.target[train]))

    return np.asarray(
        [sums.get(key, global_mean * counts.get(key, 1)) / counts.get(key, 1) for key in test_keys],
        dtype=float,
    )


def fit_temporal_cartoboost(
    task: base.BenchmarkTask,
    train: np.ndarray,
    test: np.ndarray,
    qargs: argparse.Namespace,
) -> dict[str, Any]:
    key_columns = [feature_column(task, key) for key in ["PULocationID", "hour", "dayofweek"]]
    all_indices = np.arange(len(task.target), dtype=np.int64)
    baseline_all = grouped_mean_prediction(task, train, all_indices, key_columns=key_columns)
    residual_task = replace(task, target=task.target - baseline_all)
    residual = fit_cartoboost(residual_task, train, test, qargs)
    prediction = baseline_all[test] + residual["prediction"]
    residual["prediction"] = prediction
    residual["metrics"] = base.metric_summary(task.target[test], prediction)
    return residual


def residual_correction_rows(
    task: base.BenchmarkTask,
    train: np.ndarray,
    test: np.ndarray,
    qargs: argparse.Namespace,
    *,
    dataset_hash: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = ridge_baseline(
        select_features(task, ["trip_distance", "log_trip_distance"]), train, test
    )
    train_base = ridge_predict_train_test(
        select_features(task, ["trip_distance", "log_trip_distance"]),
        train,
        train,
    )
    test_base = ridge_predict_train_test(
        select_features(task, ["trip_distance", "log_trip_distance"]),
        train,
        test,
    )
    residual_task = replace(
        task,
        target=task.target
        - np.asarray(
            ridge_predict_train_test(
                select_features(task, ["trip_distance", "log_trip_distance"]),
                train,
                np.arange(len(task.target), dtype=np.int64),
            ),
            dtype=float,
        ),
    )
    corrected = fit_cartoboost(residual_task, train, test, qargs)
    corrected_prediction = test_base + corrected["prediction"]
    corrected_metrics = base.metric_summary(task.target[test], corrected_prediction)
    corrected["metrics"] = corrected_metrics
    linear_residual = ridge_baseline(residual_task, train, test)
    linear_prediction = test_base + linear_residual["prediction"]
    linear_residual["metrics"] = base.metric_summary(task.target[test], linear_prediction)
    residual_mean = float(np.mean(task.target[train] - train_base))
    global_residual = timed_prediction(task.target[test], lambda: test_base + residual_mean)
    return [
        claim_row(
            "residual_correction",
            task.name,
            "spatial_holdout",
            manifest,
            dataset_hash,
            corrected,
            falsifier_name="raw_baseline",
            baseline=baseline,
            feature_policy="baseline_estimate_plus_generic_residual_features",
            prediction_unit="completed_trip",
            task_frame_id=f"{task.name}:trip_spatial_holdout",
        ),
        claim_row(
            "residual_correction",
            task.name,
            "spatial_holdout",
            manifest,
            dataset_hash,
            corrected,
            falsifier_name="global_residual_mean",
            baseline=global_residual,
            feature_policy="baseline_estimate_plus_generic_residual_features",
            prediction_unit="completed_trip",
            task_frame_id=f"{task.name}:trip_spatial_holdout",
        ),
        claim_row(
            "residual_correction",
            task.name,
            "spatial_holdout",
            manifest,
            dataset_hash,
            corrected,
            falsifier_name="linear_residual_model",
            baseline=linear_residual,
            feature_policy="baseline_estimate_plus_generic_residual_features",
            prediction_unit="completed_trip",
            task_frame_id=f"{task.name}:trip_spatial_holdout",
        ),
    ]


def ridge_predict_train_test(
    task: base.BenchmarkTask,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    return ridge_fit_predict(
        task.features[train], task.target[train], task.features[test], alpha=1.0
    )


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    x_mean = np.mean(train_x, axis=0)
    x_scale = np.std(train_x, axis=0)
    x_scale[x_scale == 0.0] = 1.0
    train_design = (train_x - x_mean) / x_scale
    test_design = (test_x - x_mean) / x_scale
    train_design = np.column_stack([np.ones(train_design.shape[0]), train_design])
    test_design = np.column_stack([np.ones(test_design.shape[0]), test_design])
    penalty = np.eye(train_design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_y,
    )
    return np.asarray(test_design @ coefficients, dtype=float)


def claim_row(
    claim_id: str,
    task_name: str,
    split_mode: str,
    manifest: dict[str, Any],
    dataset_hash: str,
    primary: dict[str, Any],
    *,
    falsifier_name: str,
    baseline: dict[str, Any],
    feature_policy: str,
    prediction_unit: str,
    task_frame_id: str,
    improvement_threshold: float | None = None,
    gate_required: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = primary["metrics"]
    baseline_metrics = baseline["metrics"]
    threshold = (
        CLAIM_THRESHOLDS[claim_id] if improvement_threshold is None else improvement_threshold
    )
    rmse_delta = float(baseline_metrics["rmse"] - metrics["rmse"])
    percent_improvement = (
        rmse_delta / float(baseline_metrics["rmse"])
        if float(baseline_metrics["rmse"]) > 0.0
        else float("nan")
    )
    row = {
        "claim_id": claim_id,
        "task": task_name,
        "split_kind": split_mode,
        "train_index_sha256": manifest["train_index_sha256"],
        "test_index_sha256": manifest["test_index_sha256"],
        "dataset_hash": dataset_hash,
        "model": PRIMARY_MODEL,
        "architecture": PRIMARY_ARCHITECTURE,
        "capability_tier": PRIMARY_TIER,
        "falsifier_baseline": falsifier_name,
        "primary_metric": "rmse",
        "improvement_threshold": float(threshold),
        "percent_improvement": float(percent_improvement),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "wape": float(metrics["wape"]),
        "r2": float(metrics["r2"]),
        "fit_seconds": float(primary["fit_seconds"]),
        "predict_seconds": float(primary["predict_seconds"]),
        "peak_memory_mb": float(primary["peak_memory_mb"]),
        "save_load_max_abs_diff": float(primary["save_load_max_abs_diff"]),
        "feature_access_policy": feature_policy,
        "prediction_unit": prediction_unit,
        "primary_prediction_unit": prediction_unit,
        "falsifier_prediction_unit": prediction_unit,
        "task_frame_id": task_frame_id,
        "primary_train_rows": int(manifest["train_rows"]),
        "primary_test_rows": int(manifest["test_rows"]),
        "falsifier_train_rows": int(manifest["train_rows"]),
        "falsifier_test_rows": int(manifest["test_rows"]),
        "primary_train_index_sha256": manifest["train_index_sha256"],
        "primary_test_index_sha256": manifest["test_index_sha256"],
        "falsifier_train_index_sha256": manifest["train_index_sha256"],
        "falsifier_test_index_sha256": manifest["test_index_sha256"],
        "target_encoding_train_only": True,
        "selection_uses_outer_test_labels": False,
        "baseline_rmse": float(baseline_metrics["rmse"]),
        "baseline_mae": float(baseline_metrics["mae"]),
        "baseline_wape": float(baseline_metrics["wape"]),
        "baseline_r2": float(baseline_metrics["r2"]),
        "rmse_delta": rmse_delta,
        "passed": bool(percent_improvement >= threshold),
        "gate_required": bool(gate_required),
    }
    if extra:
        row.update(extra)
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Path C row missing required fields: {missing}")
    return row


def run_claims(
    tasks: list[base.BenchmarkTask], args: argparse.Namespace, qargs: argparse.Namespace
) -> dict[str, Any]:
    task_map = {task.name: task for task in tasks}
    dataset_hash = base.benchmark_dataset_hash(qargs, tasks)
    rows: list[dict[str, Any]] = []
    for task_name in ["duration", "fare"]:
        print(f"running Path C row claims for task={task_name}", flush=True)
        task = task_map[task_name]
        train, test = base.split_indices(task, mode="spatial_holdout", seed=args.seed)
        manifest = path_c_split_manifest(
            qargs,
            task,
            "spatial_holdout",
            train,
            test,
            dataset_hash=dataset_hash,
        )
        directional_task = route_time_task(task)
        directional_train, directional_test = base.split_indices(
            directional_task, mode="spatial_holdout", seed=args.seed
        )
        directional_manifest = path_c_split_manifest(
            qargs,
            directional_task,
            "spatial_holdout",
            directional_train,
            directional_test,
            dataset_hash=dataset_hash,
        )
        ordered = fit_cartoboost(directional_task, directional_train, directional_test, qargs)
        unordered = fit_cartoboost(
            unordered_pair_task(directional_task),
            directional_train,
            directional_test,
            qargs,
        )
        additive = ridge_baseline(
            additive_task(directional_task),
            directional_train,
            directional_test,
        )
        rows.append(
            claim_row(
                "directional_structure",
                task_name,
                "spatial_holdout",
                directional_manifest,
                dataset_hash,
                ordered,
                falsifier_name="unordered_pair_baseline",
                baseline=unordered,
                feature_policy="ordered_route_geometry_time_train_only_zone_encoding",
                prediction_unit="pickup_zone_to_dropoff_zone_by_hour_day_bucket",
                task_frame_id=f"{task_name}:route_time_spatial_holdout",
                extra={"directionality_tested": "A_to_B_vs_B_to_A"},
            )
        )
        rows.append(
            claim_row(
                "directional_structure",
                task_name,
                "spatial_holdout",
                directional_manifest,
                dataset_hash,
                ordered,
                falsifier_name="source_target_additive_baseline",
                baseline=additive,
                feature_policy="ordered_route_geometry_time_train_only_zone_encoding",
                prediction_unit="pickup_zone_to_dropoff_zone_by_hour_day_bucket",
                task_frame_id=f"{task_name}:route_time_spatial_holdout",
                extra={"directionality_tested": "A_to_B_vs_B_to_A"},
            )
        )
        trip_primary = fit_cartoboost(task, train, test, qargs)
        zone_only = fit_cartoboost(zone_only_task(task), train, test, qargs)
        mean = mean_baseline(task, train, test)
        rows.append(
            claim_row(
                "spatial_transfer",
                task_name,
                "spatial_holdout",
                manifest,
                dataset_hash,
                trip_primary,
                falsifier_name="target_encoded_zone_only_baseline",
                baseline=zone_only,
                feature_policy="route_geometry_distance_time_with_train_only_zone_encoding",
                prediction_unit="completed_trip",
                task_frame_id=f"{task_name}:trip_spatial_holdout",
            )
        )
        rows.append(
            claim_row(
                "spatial_transfer",
                task_name,
                "spatial_holdout",
                manifest,
                dataset_hash,
                trip_primary,
                falsifier_name="mean_baseline",
                baseline=mean,
                feature_policy="route_geometry_distance_time_with_train_only_zone_encoding",
                prediction_unit="completed_trip",
                task_frame_id=f"{task_name}:trip_spatial_holdout",
            )
        )
        rows.extend(
            residual_correction_rows(
                task, train, test, qargs, dataset_hash=dataset_hash, manifest=manifest
            )
        )
        print(f"finished Path C row claims for task={task_name}", flush=True)

    demand = task_map["pickup_demand"]
    print("running Path C pickup_demand rolling-origin claims", flush=True)
    train, test, cutoff_timestamp = rolling_origin_indices(demand)
    manifest = path_c_split_manifest(
        qargs,
        demand,
        "rolling_origin_zone_time",
        train,
        test,
        dataset_hash=dataset_hash,
    )
    temporal_features = temporal_model_task(demand)
    temporal_qargs = with_cartoboost_budget(qargs, n_estimators=24, max_depth=5)
    temporal = fit_temporal_cartoboost(temporal_features, train, test, temporal_qargs)
    seasonal = seasonal_naive_baseline(demand, train, test)
    trailing = grouped_mean_baseline(demand, train, test, keys=["PULocationID"])
    pooled = ridge_baseline(temporal_features, train, test)
    rows.append(
        claim_row(
            "temporal_structure",
            "pickup_demand",
            "rolling_origin_zone_time",
            manifest,
            dataset_hash,
            temporal,
            falsifier_name="seasonal_naive",
            baseline=seasonal,
            feature_policy="pickup_zone_hour_day_repeated_history",
            prediction_unit="pickup_zone_hour",
            task_frame_id="pickup_demand:chronological_zone_time",
            extra={"rolling_origin_cutoff_timestamp": cutoff_timestamp},
        )
    )
    print("finished Path C pickup_demand rolling-origin claims", flush=True)
    rows.append(
        claim_row(
            "temporal_structure",
            "pickup_demand",
            "rolling_origin_zone_time",
            manifest,
            dataset_hash,
            temporal,
            falsifier_name="trailing_mean",
            baseline=trailing,
            feature_policy="pickup_zone_hour_day_repeated_history",
            prediction_unit="pickup_zone_hour",
            task_frame_id="pickup_demand:chronological_zone_time",
            gate_required=False,
            extra={"rolling_origin_cutoff_timestamp": cutoff_timestamp},
        )
    )
    rows.append(
        claim_row(
            "temporal_structure",
            "pickup_demand",
            "rolling_origin_zone_time",
            manifest,
            dataset_hash,
            temporal,
            falsifier_name="pooled_ridge",
            baseline=pooled,
            feature_policy="pickup_zone_hour_day_repeated_history",
            prediction_unit="pickup_zone_hour",
            task_frame_id="pickup_demand:chronological_zone_time",
            gate_required=False,
            extra={"rolling_origin_cutoff_timestamp": cutoff_timestamp},
        )
    )
    ablated = fit_cartoboost(ablated_future_task(temporal_features), train, test, temporal_qargs)
    rows.append(
        claim_row(
            "known_future_sensitivity",
            "pickup_demand",
            "rolling_origin_zone_time",
            manifest,
            dataset_hash,
            temporal,
            falsifier_name="future_known_covariates_ablated",
            baseline=ablated,
            feature_policy="pickup_zone_with_hour_day_calendar_covariates",
            prediction_unit="pickup_zone_hour",
            task_frame_id="pickup_demand:chronological_zone_time",
            extra={
                "rolling_origin_cutoff_timestamp": cutoff_timestamp,
                "known_future_ablation_delta_rmse": float(
                    ablated["metrics"]["rmse"] - temporal["metrics"]["rmse"]
                ),
            },
        )
    )

    return {
        "artifact_version": 1,
        "path_c": True,
        "dataset": base.dataset_metadata(qargs, tasks),
        "dataset_hash": dataset_hash,
        "source_file_hashes": base.source_file_hashes(qargs),
        "git_commit": base.read_git_commit(),
        "benchmark_integrity": {
            "command_argv": list(sys.argv),
            "synthetic_smoke": bool(args.synthetic_smoke),
            "source": "NYC TLC yellow taxi 2024",
            "tasks": ["duration", "fare", "pickup_demand"],
            "splits": ["spatial_holdout", "rolling_origin_zone_time"],
            "models_added": False,
        },
        "feature_access_policy": base.feature_access_policy(qargs),
        "required_row_fields": REQUIRED_ROW_FIELDS,
        "claims": rows,
    }


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "path_c_claims.json"
    jsonl_path = output_dir / "path_c_claims.jsonl"
    md_path = output_dir / "path_c_claims.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in results["claims"]) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# NYC Taxi Path C Claims",
        "",
        (
            "Path C is bounded real-data evidence on NYC TLC 2024 yellow taxi tasks. "
            "It proves implementation behavior and benchmark discipline on this dataset, "
            "not universal market superiority."
        ),
        "",
        (
            "| Claim | Task | Split | Falsifier | Unit | RMSE | Baseline RMSE | "
            "Improvement | Threshold | Pass |"
        ),
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results["claims"]:
        lines.append(
            f"| {row['claim_id']} | {row['task']} | {row['split_kind']} | "
            f"{row['falsifier_baseline']} | {row['prediction_unit']} | {row['rmse']:.6f} | "
            f"{row['baseline_rmse']:.6f} | {row['percent_improvement']:.2%} | "
            f"{row['improvement_threshold']:.2%} | {row['passed']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    qargs = quality_args(args)
    tasks = load_tasks(args, qargs)
    results = run_claims(tasks, args, qargs)
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
