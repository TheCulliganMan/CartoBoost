#!/usr/bin/env python3
"""Run v0.2 spatial boosting smoke benchmarks.

The workloads are deterministic taxi-shaped synthetic checks for the v0.2
release gates. They are not a public real-world superiority claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

CartoBoostClassifier: Any = None
CartoBoostRanker: Any = None
CartoBoostRegressor: Any = None
FeatureKind: Any = None
brier_score: Any = None
ece_calibration_error: Any = None
logloss: Any = None
mean_average_precision: Any = None
mean_reciprocal_rank: Any = None
ndcg_at_k: Any = None
pr_auc: Any = None
roc_auc: Any = None
spatial_buffered_cv: Any = None
spatial_cv_gap: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "target" / "v02-benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=240)
    parser.add_argument("--n-estimators", type=int, default=24)
    parser.add_argument("--regression-slowdown-threshold", type=float, default=0.05)
    parser.add_argument(
        "--regression-baseline-json",
        type=Path,
        default=None,
        help=(
            "Optional prior v02_modeling_benchmark.json to use as the regression "
            "fit-speed baseline. Without it, the guard records current-code "
            "repeatability evidence only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_cartoboost()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = [
        binary_spatial_classification(args, rng),
        grouped_ranking(args, rng),
        categorical_vs_one_hot(args, rng),
        spatial_leakage(args, rng),
        regression_speed(args, rng),
        unsupported_export(args),
    ]
    payload = {
        "artifact_type": "cartoboost.v02_modeling_benchmark",
        "artifact_version": 1,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "n_estimators": args.n_estimators,
        "data_kind": "deterministic synthetic taxi-shaped smoke checks",
        "results": results,
        "acceptance": {
            "passed": all(result["passed"] for result in results),
            "gates": {result["name"]: result["passed"] for result in results},
        },
    }
    (args.output_dir / "v02_modeling_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "v02_modeling_benchmark.jsonl").open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, sort_keys=True) + "\n")
    (args.output_dir / "v02_modeling_benchmark.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    return 0 if payload["acceptance"]["passed"] else 1


def ensure_cartoboost() -> None:
    global CartoBoostClassifier
    global CartoBoostRanker
    global CartoBoostRegressor
    global FeatureKind
    global brier_score
    global ece_calibration_error
    global logloss
    global mean_average_precision
    global mean_reciprocal_rank
    global ndcg_at_k
    global pr_auc
    global roc_auc
    global spatial_buffered_cv
    global spatial_cv_gap
    if CartoBoostRegressor is not None:
        return
    from cartoboost import (
        CartoBoostClassifier as _CartoBoostClassifier,
    )
    from cartoboost import (
        CartoBoostRanker as _CartoBoostRanker,
    )
    from cartoboost import (
        CartoBoostRegressor as _CartoBoostRegressor,
    )
    from cartoboost import (
        FeatureKind as _FeatureKind,
    )
    from cartoboost import (
        brier_score as _brier_score,
    )
    from cartoboost import (
        ece_calibration_error as _ece_calibration_error,
    )
    from cartoboost import (
        logloss as _logloss,
    )
    from cartoboost import (
        mean_average_precision as _mean_average_precision,
    )
    from cartoboost import (
        mean_reciprocal_rank as _mean_reciprocal_rank,
    )
    from cartoboost import (
        ndcg_at_k as _ndcg_at_k,
    )
    from cartoboost import (
        pr_auc as _pr_auc,
    )
    from cartoboost import (
        roc_auc as _roc_auc,
    )
    from cartoboost import (
        spatial_buffered_cv as _spatial_buffered_cv,
    )
    from cartoboost import (
        spatial_cv_gap as _spatial_cv_gap,
    )

    CartoBoostClassifier = _CartoBoostClassifier
    CartoBoostRanker = _CartoBoostRanker
    CartoBoostRegressor = _CartoBoostRegressor
    FeatureKind = _FeatureKind
    brier_score = _brier_score
    ece_calibration_error = _ece_calibration_error
    logloss = _logloss
    mean_average_precision = _mean_average_precision
    mean_reciprocal_rank = _mean_reciprocal_rank
    ndcg_at_k = _ndcg_at_k
    pr_auc = _pr_auc
    roc_auc = _roc_auc
    spatial_buffered_cv = _spatial_buffered_cv
    spatial_cv_gap = _spatial_cv_gap


def binary_spatial_classification(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    coords = rng.normal(size=(args.sample_size, 2))
    hour = rng.integers(0, 24, size=args.sample_size)
    score = coords[:, 0] + 0.8 * coords[:, 1] + 0.35 * np.sin(2.0 * np.pi * hour / 24.0)
    y = (score > np.median(score)).astype(int)
    x = np.column_stack([coords, hour.astype(float)])
    train, test = deterministic_split(args.sample_size)

    model = CartoBoostClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.2,
        max_depth=3,
        min_samples_leaf=5,
        min_gain=0.0,
        split_policy="structured",
    )
    fit_s, pred_s, _ = timed_fit_predict(model, x[train], y[train], x[test])
    proba = model.predict_proba(x[test])[:, list(model.classes_).index(1)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "classifier.json"
        model.save(path)
        loaded = CartoBoostClassifier.load(path)
        loaded_proba = loaded.predict_proba(x[test])[:, list(loaded.classes_).index(1)]
    dummy_rate = float(np.mean(y[train]))
    dummy_proba = np.full(y[test].shape[0], dummy_rate)
    metrics = {
        "cartoboost_logloss": logloss(y[test], proba),
        "dummy_logloss": logloss(y[test], dummy_proba),
        "cartoboost_roc_auc": roc_auc(y[test], proba),
        "cartoboost_pr_auc": pr_auc(y[test], proba),
        "dummy_roc_auc": roc_auc(y[test], dummy_proba),
        "dummy_pr_auc": pr_auc(y[test], dummy_proba),
        "cartoboost_brier": brier_score(y[test], proba),
        "cartoboost_ece": ece_calibration_error(y[test], proba, n_bins=10),
        "roundtrip_max_abs_diff": float(np.max(np.abs(proba - loaded_proba))),
        "fit_seconds": fit_s,
        "predict_seconds": pred_s,
    }
    return gate(
        "binary_spatial_classification",
        metrics,
        metrics["cartoboost_logloss"] < metrics["dummy_logloss"]
        and metrics["cartoboost_roc_auc"] >= 0.8
        and metrics["roundtrip_max_abs_diff"] <= 1e-12,
    )


def grouped_ranking(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    group_count = max(12, args.sample_size // 12)
    group_size = 6
    rows = group_count * group_size
    route_quality = np.tile(np.arange(group_size, dtype=float), group_count)
    group_effect = np.repeat(rng.normal(size=group_count), group_size)
    distance = rng.normal(size=rows)
    relevance = route_quality + 0.2 * distance
    x = np.column_stack([route_quality, distance, group_effect])
    groups = [group_size] * group_count
    train_groups = group_count * 2 // 3
    split = train_groups * group_size

    ranker = CartoBoostRanker(
        n_estimators=args.n_estimators,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=2,
        min_gain=0.0,
        split_policy="axis_only",
    )
    start = time.perf_counter()
    ranker.fit(x[:split], relevance[:split], groups=groups[:train_groups])
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    scores = ranker.predict(x[split:])
    pred_s = time.perf_counter() - start
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ranker.json"
        ranker.save(path)
        loaded = CartoBoostRanker.load(path)
        loaded_scores = loaded.predict(x[split:])
    test_groups = groups[train_groups:]
    baseline_scores = -route_quality[split:]
    metrics = {
        "cartoboost_ndcg": ndcg_at_k(relevance[split:], scores, groups=test_groups),
        "baseline_ndcg": ndcg_at_k(relevance[split:], baseline_scores, groups=test_groups),
        "cartoboost_map": mean_average_precision(relevance[split:], scores, groups=test_groups),
        "cartoboost_mrr": mean_reciprocal_rank(relevance[split:], scores, groups=test_groups),
        "roundtrip_max_abs_diff": float(np.max(np.abs(scores - loaded_scores))),
        "fit_seconds": fit_s,
        "predict_seconds": pred_s,
    }
    return gate(
        "grouped_ranking",
        metrics,
        metrics["cartoboost_ndcg"] > metrics["baseline_ndcg"]
        and metrics["roundtrip_max_abs_diff"] <= 1e-12,
    )


def categorical_vs_one_hot(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    zones = np.asarray(["airport", "midtown", "outer", "downtown"])
    zone_idx = rng.integers(0, zones.size, size=args.sample_size)
    distance = rng.uniform(0.5, 8.0, size=args.sample_size)
    y = 4.0 + distance * 1.5 + np.asarray([8.0, 3.0, -1.0, 1.5])[zone_idx]
    y += rng.normal(scale=0.1, size=args.sample_size)
    raw_x = np.column_stack([zones[zone_idx], distance.astype(object)])
    one_hot = np.column_stack([(zone_idx == idx).astype(float) for idx in range(zones.size)])
    one_hot_x = np.column_stack([one_hot, distance])
    schema = {
        "dense": [
            {"name": "PULocationID", "kind": FeatureKind.CATEGORICAL},
            {"name": "trip_distance", "kind": FeatureKind.NUMERIC},
        ]
    }
    train, test = deterministic_split(args.sample_size)
    native = CartoBoostRegressor(
        n_estimators=args.n_estimators,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=3,
        min_gain=0.0,
        split_policy="axis_only",
    )
    one_hot_model = CartoBoostRegressor(
        n_estimators=args.n_estimators,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=3,
        min_gain=0.0,
        split_policy="axis_only",
    )
    native_fit, native_pred_s, native_pred = timed_fit_predict(
        native,
        raw_x[train],
        y[train],
        raw_x[test],
        feature_schema=schema,
    )
    one_hot_fit, one_hot_pred_s, one_hot_pred = timed_fit_predict(
        one_hot_model,
        one_hot_x[train],
        y[train],
        one_hot_x[test],
    )
    native_rmse = rmse(y[test], native_pred)
    one_hot_rmse = rmse(y[test], one_hot_pred)
    categorical_column = native.categorical_encoder_["columns"][0]
    train_categories = set(categorical_column.get("categories", []))
    test_tokens = [f"str:{value}" for value in zones[zone_idx[test]].tolist()]
    unknown_category_rate = float(np.mean([token not in train_categories for token in test_tokens]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "categorical.json"
        native.save(path)
        loaded = CartoBoostRegressor.load(path)
        loaded_pred = loaded.predict(raw_x[test])
    metrics = {
        "native_rmse": native_rmse,
        "one_hot_rmse": one_hot_rmse,
        "category_count": len(categorical_column.get("categories", [])),
        "encoding_strategy": str(categorical_column.get("strategy")),
        "unknown_category_rate": unknown_category_rate,
        "native_fit_seconds": native_fit,
        "one_hot_fit_seconds": one_hot_fit,
        "native_predict_seconds": native_pred_s,
        "one_hot_predict_seconds": one_hot_pred_s,
        "roundtrip_max_abs_diff": float(np.max(np.abs(native_pred - loaded_pred))),
    }
    return gate(
        "categorical_native_vs_one_hot",
        metrics,
        native_rmse <= one_hot_rmse * 1.05 and metrics["roundtrip_max_abs_diff"] <= 1e-12,
    )


def spatial_leakage(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    coords = rng.uniform(0.0, 10.0, size=(args.sample_size, 2))
    y = np.sin(coords[:, 0] * 1.5) + np.cos(coords[:, 1] * 1.5)
    x = coords.copy()
    random_train, random_test = deterministic_split(args.sample_size)
    buffered_train, buffered_test = next(
        spatial_buffered_cv(
            coords,
            n_splits=4,
            buffer_radius=1.0,
            coordinate_units="projected",
        )
    )
    random_rmse = fit_rmse(args, x, y, random_train, random_test)
    buffered_rmse = fit_rmse(args, x, y, buffered_train, buffered_test)
    metrics = {
        "random_cv_rmse": random_rmse,
        "buffered_cv_rmse": buffered_rmse,
        "rmse_gap_random_minus_buffered": spatial_cv_gap(random_rmse, buffered_rmse),
        "rmse_gap_buffered_minus_random": buffered_rmse - random_rmse,
    }
    return gate("spatial_leakage_random_vs_buffered", metrics, buffered_rmse > random_rmse)


def regression_speed(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    x = rng.normal(size=(args.sample_size, 4))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.5 * np.sin(x[:, 2])
    train, test = deterministic_split(args.sample_size)
    repeat_fit, repeat_pred_s, repeat_pred = best_regression_timing(args, x, y, train, test)
    current_fit, current_pred_s, current_pred = best_regression_timing(args, x, y, train, test)
    baseline_fit = regression_baseline_fit_seconds(args.regression_baseline_json)
    evidence_kind = (
        "external_baseline_comparison" if baseline_fit is not None else "current_code_repeatability"
    )
    if baseline_fit is None:
        baseline_fit = repeat_fit
    slowdown = current_fit / max(baseline_fit, 1e-12) - 1.0
    metrics = {
        "evidence_kind": evidence_kind,
        "external_baseline_path": None
        if args.regression_baseline_json is None
        else str(args.regression_baseline_json),
        "current_fit_seconds": current_fit,
        "baseline_fit_seconds": baseline_fit,
        "fit_slowdown": slowdown,
        "current_predict_seconds": current_pred_s,
        "repeat_predict_seconds": repeat_pred_s,
        "prediction_max_abs_diff": float(np.max(np.abs(current_pred - repeat_pred))),
    }
    return gate(
        "regression_speed_guard",
        metrics,
        slowdown <= args.regression_slowdown_threshold
        and metrics["prediction_max_abs_diff"] <= 1e-12,
    )


def regression_baseline_fit_seconds(path: Path | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for result in payload.get("results", []):
        if result.get("name") == "regression_speed_guard":
            metrics = result.get("metrics", {})
            value = metrics.get("current_fit_seconds") or metrics.get("baseline_fit_seconds")
            if value is None:
                raise ValueError(f"{path} regression_speed_guard lacks current_fit_seconds")
            baseline = float(value)
            if not np.isfinite(baseline) or baseline <= 0.0:
                raise ValueError(f"{path} regression baseline fit time must be positive")
            return baseline
    raise ValueError(f"{path} does not contain a regression_speed_guard result")


def best_regression_timing(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    best: tuple[float, float, np.ndarray] | None = None
    for _ in range(3):
        model = CartoBoostRegressor(
            n_estimators=args.n_estimators,
            learning_rate=0.2,
            max_depth=3,
            min_samples_leaf=5,
            min_gain=0.0,
            split_policy="axis_only",
        )
        result = timed_fit_predict(model, x[train], y[train], x[test])
        if best is None or result[0] < best[0]:
            best = result
    assert best is not None
    return best


def unsupported_export(args: argparse.Namespace) -> dict[str, Any]:
    x = np.asarray([["airport"], ["midtown"], ["airport"], ["midtown"]], dtype=object)
    y = np.asarray([10.0, 2.0, 10.0, 2.0])
    schema = {"dense": [{"name": "PULocationID", "kind": FeatureKind.CATEGORICAL}]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(x, y, feature_schema=schema)
    categorical_raised = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            model.save_weights(Path(tmp) / "categorical.onnx", format="onnx")
        except NotImplementedError:
            categorical_raised = True
        classifier_raised = False
        classifier = CartoBoostClassifier(n_estimators=1, max_depth=0).fit(
            [[0.0], [1.0]],
            [0, 1],
        )
        try:
            classifier.save_weights(Path(tmp) / "classifier.onnx", format="onnx")
        except NotImplementedError:
            classifier_raised = True
        ranker_raised = False
        ranker = CartoBoostRanker(n_estimators=1, max_depth=0, min_samples_leaf=1).fit(
            [[0.0], [1.0], [0.0], [1.0]],
            [0.0, 1.0, 0.0, 2.0],
            groups=[2, 2],
        )
        try:
            ranker.save_weights(Path(tmp) / "ranker.onnx", format="onnx")
        except NotImplementedError:
            ranker_raised = True
    return gate(
        "unsupported_export_fails_loudly",
        {
            "categorical_onnx_raised": categorical_raised,
            "classifier_onnx_raised": classifier_raised,
            "ranker_onnx_raised": ranker_raised,
        },
        categorical_raised and classifier_raised and ranker_raised,
    )


def deterministic_split(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    split = n_rows * 2 // 3
    return np.arange(split), np.arange(split, n_rows)


def timed_fit_predict(
    model: Any,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    **fit_kwargs: Any,
) -> tuple[float, float, np.ndarray]:
    start = time.perf_counter()
    model.fit(x_train, y_train, **fit_kwargs)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred = np.asarray(model.predict(x_test), dtype=float)
    pred_s = time.perf_counter() - start
    return fit_s, pred_s, pred


def fit_rmse(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    model = CartoBoostRegressor(
        n_estimators=args.n_estimators,
        learning_rate=0.2,
        max_depth=3,
        min_samples_leaf=5,
        min_gain=0.0,
        split_policy="structured",
    )
    model.fit(x[train_idx], y[train_idx])
    return rmse(y[test_idx], model.predict(x[test_idx]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def gate(name: str, metrics: dict[str, Any], passed: bool) -> dict[str, Any]:
    return {"name": name, "metrics": metrics, "passed": bool(passed)}


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# v0.2 Modeling Benchmark Smoke Checks",
        "",
        "These deterministic synthetic checks exercise the v0.2 classifier, ranker, "
        "categorical, spatial validation, regression-speed, and export gates. They "
        "are release smoke evidence, not a real NYC TLC quality claim.",
        "",
        f"- seed: `{payload['seed']}`",
        f"- sample size: `{payload['sample_size']}`",
        f"- n estimators: `{payload['n_estimators']}`",
        f"- overall passed: `{payload['acceptance']['passed']}`",
        "",
        "| Gate | Passed | Key metrics |",
        "| --- | --- | --- |",
    ]
    for result in payload["results"]:
        key_metrics = ", ".join(
            f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in result["metrics"].items()
        )
        lines.append(f"| `{result['name']}` | `{result['passed']}` | {key_metrics} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
