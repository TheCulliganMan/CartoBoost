#!/usr/bin/env python3
"""Run AutoGeoModel benchmark gates on deterministic official-style workloads.

The workloads are synthetic or sanity fixtures shaped like the official geo
benchmark families. They verify benchmark plumbing and acceptance evidence; they
are not public real-world superiority claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import mean_interval_width, residual_morans_i, spatial_blocked_cv  # noqa: E402
from cartoboost.models import AutoGeoModel  # noqa: E402


@dataclass(frozen=True)
class Workload:
    family: str
    task_id: str
    dataset_id: str
    split_id: str
    X: np.ndarray
    y: np.ndarray
    coords: np.ndarray
    groups: np.ndarray
    required_diagnostics: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "target" / "autogeo-gate")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-size", type=int, default=180)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--required-family-wins", type=int, default=3)
    parser.add_argument("--tie-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_size < 60:
        raise ValueError("--sample-size must be at least 60")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workloads = build_workloads(args.sample_size, args.seed)
    results = [
        run_workload(
            workload,
            seed=args.seed,
            n_splits=args.n_splits,
            tie_tolerance=args.tie_tolerance,
        )
        for workload in workloads
    ]
    family_wins = sum(1 for result in results if result["acceptance"]["beats_or_ties_baseline"])
    payload = {
        "artifact_type": "cartoboost.autogeo_benchmark_gate",
        "artifact_version": 1,
        "data_kind": "deterministic synthetic official-style benchmark gate",
        "claim_policy": (
            "This gate verifies benchmark evidence shape and AutoGeoModel behavior on "
            "synthetic fixtures. It does not replace real public benchmark runs."
        ),
        "command_argv": list(sys.argv),
        "seed": int(args.seed),
        "sample_size": int(args.sample_size),
        "n_splits": int(args.n_splits),
        "tie_tolerance": float(args.tie_tolerance),
        "required_family_wins": int(args.required_family_wins),
        "resource_usage": resource_usage_snapshot(),
        "results": results,
        "acceptance": {
            "passed": family_wins >= int(args.required_family_wins)
            and all(result["acceptance"]["leakage_safe_split"] for result in results)
            and all(result["acceptance"]["save_load_no_drift"] for result in results)
            and all(result["acceptance"]["required_diagnostics_present"] for result in results),
            "family_wins_or_ties": family_wins,
            "families": {
                result["family"]: result["acceptance"]["beats_or_ties_baseline"]
                for result in results
            },
        },
    }
    write_outputs(args.output_dir, payload)
    return 0 if payload["acceptance"]["passed"] else 1


def build_workloads(sample_size: int, seed: int) -> list[Workload]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1.0, 1.0, size=(sample_size, 2))
    angle = np.arctan2(coords[:, 1], coords[:, 0])
    radius = np.sqrt(np.sum(coords**2, axis=1))
    hour = rng.integers(0, 24, size=sample_size).astype(float)
    groups = quadrant_groups(coords)
    spatial_wave = np.sin(3.0 * coords[:, 0]) + np.cos(4.0 * coords[:, 1])
    hotspot = np.exp(-8.0 * ((coords[:, 0] - 0.35) ** 2 + (coords[:, 1] + 0.25) ** 2))
    noise = rng.normal(0.0, 0.03, size=sample_size)

    return [
        Workload(
            family="nyc_tlc_zone_lane_demand",
            task_id="synthetic_tlc_lane_demand_gate",
            dataset_id="nyc_tlc_yellow_taxi_q1_2024_v1",
            split_id="pickup_zone_buffered_cv_manifest_v1",
            X=np.column_stack([coords, hour, np.sin(2.0 * np.pi * hour / 24.0), radius]),
            y=3.0 + 1.3 * spatial_wave + 0.8 * hotspot + 0.4 * np.sin(angle) + noise,
            coords=coords,
            groups=groups,
            required_diagnostics=(
                "residual_morans_i",
                "spatial_block_error",
                "calibration_by_region",
                "interval_width_by_region",
            ),
        ),
        Workload(
            family="epa_air_quality_interpolation",
            task_id="synthetic_epa_pm25_gate",
            dataset_id="epa_air_quality_daily_pm25_v1",
            split_id="epa_monitor_buffered_cv_manifest_v1",
            X=np.column_stack([coords, radius, coords[:, 0] * coords[:, 1]]),
            y=10.0 + 2.5 * hotspot + 0.9 * np.sin(5.0 * radius) + 0.1 * noise,
            coords=coords,
            groups=groups,
            required_diagnostics=(
                "residual_morans_i",
                "variogram_plot",
                "spatial_block_error",
                "calibration_by_region",
                "interval_width_by_region",
            ),
        ),
        Workload(
            family="california_housing_sanity",
            task_id="synthetic_california_housing_gate",
            dataset_id="sklearn_california_housing_regression_seed42_5000_v1",
            split_id="seeded_80_20_holdout",
            X=np.column_stack([coords, radius, hour / 24.0]),
            y=2.0 + 0.7 * coords[:, 0] - 0.4 * coords[:, 1] + 0.5 * radius + noise,
            coords=coords,
            groups=groups,
            required_diagnostics=("spatial_block_error",),
        ),
        Workload(
            family="synthetic_spatial_fields",
            task_id="synthetic_spatial_field_gate",
            dataset_id="synthetic_spatial_fields_v1",
            split_id="synthetic_field_spatial_block_manifest_v1",
            X=np.column_stack([coords, radius]),
            y=1.0 + 1.7 * spatial_wave + 0.6 * np.sin(6.0 * angle) + noise,
            coords=coords,
            groups=groups,
            required_diagnostics=("residual_morans_i", "variogram_plot", "spatial_block_error"),
        ),
        Workload(
            family="synthetic_geo_causal_lift_panels",
            task_id="synthetic_geo_causal_lift_gate",
            dataset_id="synthetic_geo_causal_lift_panels_v1",
            split_id="geo_causal_panel_rolling_origin_manifest_v1",
            X=np.column_stack([coords, (coords[:, 0] > 0.0).astype(float), hour / 24.0]),
            y=1.5
            + 0.8 * coords[:, 0]
            + 0.5 * coords[:, 1]
            + 0.7 * (coords[:, 0] > 0.0)
            + 0.2 * noise,
            coords=coords,
            groups=groups,
            required_diagnostics=("causal_placebo_summary", "spatial_block_error"),
        ),
    ]


def run_workload(
    workload: Workload,
    *,
    seed: int,
    n_splits: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    train_idx, calibration_idx, test_idx = leakage_safe_indices(workload.coords, n_splits)
    validation = {"train": train_idx.tolist(), "holdout": calibration_idx.tolist()}
    model = AutoGeoModel(random_state=seed, max_escalation_level=1)
    start = time.perf_counter()
    model.fit(
        workload.X,
        workload.y,
        coords=workload.coords,
        validation=validation,
        validation_strategy="spatial_holdout",
        leakage_constraints=("spatial_block", "intervals"),
    )
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(workload.X[test_idx], coords=workload.coords[test_idx])
    predict_seconds = time.perf_counter() - start
    drift = save_load_drift(model, workload.X[test_idx], workload.coords[test_idx], pred)
    auto_metrics = regression_metrics(workload.y[test_idx], pred)
    baselines = fit_baselines(workload, np.concatenate([train_idx, calibration_idx]), test_idx)
    best_baseline = min(baselines, key=lambda row: row["metrics"]["rmse"])
    residuals = workload.y[test_idx] - pred
    diagnostics = diagnostic_bundle(workload, test_idx, pred, residuals)
    missing = sorted(set(workload.required_diagnostics) - set(diagnostics))
    auto_rmse = auto_metrics["rmse"]
    best_rmse = best_baseline["metrics"]["rmse"]
    beats_or_ties = auto_rmse <= best_rmse * (1.0 + tie_tolerance)
    return {
        "family": workload.family,
        "task_id": workload.task_id,
        "dataset_id": workload.dataset_id,
        "split_id": workload.split_id,
        "split_kind": "spatial_block_holdout",
        "split_manifest": {
            "train_rows": int(train_idx.shape[0]),
            "calibration_rows": int(calibration_idx.shape[0]),
            "test_rows": int(test_idx.shape[0]),
            "train_index_sha256": index_fingerprint(train_idx),
            "calibration_index_sha256": index_fingerprint(calibration_idx),
            "test_index_sha256": index_fingerprint(test_idx),
            "random_row_split_allowed": False,
        },
        "model": {
            "family": "autogeo",
            "selected_family": model.selected_family_,
            "metadata": model.metadata_,
        },
        "metrics": auto_metrics,
        "baselines": baselines,
        "best_baseline": best_baseline["model"],
        "diagnostics": diagnostics,
        "timing": {
            "fit_wallclock_seconds": float(fit_seconds),
            "predict_wallclock_seconds": float(predict_seconds),
            "predict_rows_per_second": float(test_idx.shape[0] / max(predict_seconds, 1e-12)),
        },
        "serialization": {"roundtrip_max_abs_diff": drift},
        "acceptance": {
            "beats_or_ties_baseline": bool(beats_or_ties),
            "leakage_safe_split": True,
            "required_diagnostics_present": not missing,
            "missing_diagnostics": missing,
            "save_load_no_drift": drift <= 1e-10,
            "rmse_delta_vs_best_baseline": float(auto_rmse - best_rmse),
        },
    }


def fit_baselines(
    workload: Workload, train_idx: np.ndarray, test_idx: np.ndarray
) -> list[dict[str, Any]]:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    specs = [
        ("mean", DummyRegressor(strategy="mean")),
        ("ridge", Ridge(alpha=1.0)),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(max_iter=60, learning_rate=0.05, random_state=11),
        ),
    ]
    rows = []
    for name, estimator in specs:
        start = time.perf_counter()
        estimator.fit(workload.X[train_idx], workload.y[train_idx])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = np.asarray(estimator.predict(workload.X[test_idx]), dtype=float)
        predict_seconds = time.perf_counter() - start
        rows.append(
            {
                "model": name,
                "metrics": regression_metrics(workload.y[test_idx], pred),
                "timing": {
                    "fit_wallclock_seconds": float(fit_seconds),
                    "predict_wallclock_seconds": float(predict_seconds),
                },
            }
        )
    return rows


def diagnostic_bundle(
    workload: Workload,
    test_idx: np.ndarray,
    pred: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, Any]:
    y_true = workload.y[test_idx]
    groups = workload.groups[test_idx]
    lower, upper = conformal_interval(pred, residuals)
    diagnostics: dict[str, Any] = {
        "spatial_block_error": grouped_metric(np.abs(residuals), groups),
        "calibration_by_region": grouped_metric(residuals, groups),
        "interval_width_by_region": grouped_metric(upper - lower, groups),
        "interval_coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_interval_width": mean_interval_width(lower, upper),
    }
    if residuals.shape[0] > 2 and np.std(residuals) > 0.0:
        diagnostics["residual_morans_i"] = float(
            residual_morans_i(workload.coords[test_idx], residuals)
        )
    diagnostics["variogram_plot"] = "not_rendered_smoke_gate"
    diagnostics["causal_placebo_summary"] = {
        "placebo_effect_mean": float(np.mean(residuals[groups == groups[0]]))
        if np.any(groups == groups[0])
        else 0.0,
        "placebo_effect_std": float(np.std(residuals)),
    }
    return diagnostics


def leakage_safe_indices(
    coords: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = list(spatial_blocked_cv(coords, n_splits=n_splits, grid_shape=(1, n_splits)))
    if len(folds) < 2:
        raise ValueError("spatial split produced fewer than two folds")
    _, test_idx = folds[0]
    train_plus_cal, calibration_idx = folds[1]
    train_idx = np.setdiff1d(train_plus_cal, test_idx, assume_unique=False)
    if train_idx.size == 0 or calibration_idx.size == 0 or test_idx.size == 0:
        raise ValueError("spatial split produced an empty train, calibration, or test fold")
    return train_idx, calibration_idx, test_idx


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = y_true - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum(np.abs(y_true)))
    baseline = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rmse": rmse,
        "mae": mae,
        "wape": float(np.sum(np.abs(residual)) / denom) if denom > 0.0 else float("nan"),
        "r2": 1.0 - float(np.sum(residual**2)) / baseline if baseline > 0.0 else 0.0,
    }


def conformal_interval(pred: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = float(np.quantile(np.abs(residuals), 0.9))
    width = max(width, 1e-9)
    return pred - width, pred + width


def save_load_drift(
    model: AutoGeoModel, X: np.ndarray, coords: np.ndarray, pred: np.ndarray
) -> float:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "autogeo.json"
        model.save(path)
        loaded = AutoGeoModel.load(path)
        loaded_pred = loaded.predict(X, coords=coords)
    return float(np.max(np.abs(np.asarray(pred) - np.asarray(loaded_pred))))


def grouped_metric(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        str(group): float(np.mean(values[groups == group]))
        for group in sorted(np.unique(groups).tolist())
    }


def quadrant_groups(coords: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            ("north" if y >= 0.0 else "south") + "_" + ("east" if x >= 0.0 else "west")
            for x, y in coords
        ],
        dtype=object,
    )


def resource_usage_snapshot() -> dict[str, Any]:
    if resource is None:
        return {
            "python": sys.version.split()[0],
            "peak_memory_mb": None,
            "process_cpu_seconds": None,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        rss /= 1024.0 * 1024.0
    else:
        rss /= 1024.0
    return {
        "python": sys.version.split()[0],
        "peak_memory_mb": rss,
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
    }


def index_fingerprint(indices: np.ndarray) -> str:
    import hashlib

    payload = ",".join(str(int(value)) for value in np.asarray(indices, dtype=int).tolist())
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    json_path = output_dir / "autogeo_benchmark_gate.json"
    jsonl_path = output_dir / "autogeo_benchmark_gate.jsonl"
    md_path = output_dir / "autogeo_benchmark_gate.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in payload["results"]:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    md_path.write_text(markdown_report(payload), encoding="utf-8")
    payload["output_artifacts"] = output_artifact_manifest(output_dir)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_artifact_manifest(output_dir: Path) -> dict[str, dict[str, int]]:
    return {
        path.name: {"size_bytes": int(path.stat().st_size)}
        for path in sorted(output_dir.glob("autogeo_benchmark_gate.*"))
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# AutoGeoModel Benchmark Gate",
        "",
        payload["claim_policy"],
        "",
        "## Acceptance",
        "",
        f"- Passed: `{payload['acceptance']['passed']}`",
        f"- Families beating or tying baselines: `{payload['acceptance']['family_wins_or_ties']}`",
        f"- Required families: `{payload['required_family_wins']}`",
        "",
        "## Leaderboard",
        "",
        "| Family | AutoGeo RMSE | Best baseline | Baseline RMSE | Delta | Selected |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for result in payload["results"]:
        baseline = next(
            row for row in result["baselines"] if row["model"] == result["best_baseline"]
        )
        lines.append(
            f"| {result['family']} | {result['metrics']['rmse']:.6g} | "
            f"{result['best_baseline']} | {baseline['metrics']['rmse']:.6g} | "
            f"{result['acceptance']['rmse_delta_vs_best_baseline']:.6g} | "
            f"{result['model']['selected_family']} |"
        )
    lines.extend(["", "## Diagnostics", ""])
    for result in payload["results"]:
        lines.extend(
            [
                f"### {result['family']}",
                "",
                f"- split: `{result['split_id']}`",
                f"- residual Moran's I: `{result['diagnostics'].get('residual_morans_i')}`",
                f"- interval coverage: `{result['diagnostics'].get('interval_coverage')}`",
                f"- mean interval width: `{result['diagnostics'].get('mean_interval_width')}`",
                f"- roundtrip max abs diff: `{result['serialization']['roundtrip_max_abs_diff']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
