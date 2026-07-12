#!/usr/bin/env python3
"""Run deterministic embedding-table benchmarks across split protocols.

Usage:
  python scripts/run_neural_embedding_benchmark.py --output target/validation/neural_benchmark.json
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import (
    CartoBoostRegressor,
    NeuralEmbeddingRegressor,
    grouped_blocked_cv,
    out_of_time_split,
    spatial_blocked_cv,
)

DEFAULT_OUTPUT = ROOT / "target" / "validation" / "neural_embedding_benchmark.json"

SPLIT_MODES = {
    "tail",
    "random",
    "temporal_blocked",
    "geo_blocked",
    "cold_origin",
    "cold_destination",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=2_000)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--n-cells", type=int, default=128)
    parser.add_argument("--n-neural-dim", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-sklearn", action="store_true")
    parser.add_argument(
        "--split-mode",
        choices=sorted(SPLIT_MODES | {"all"}),
        default="all",
        help="Split protocol for train/validation partitioning",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-fold", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def ms(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_absolute_error(actual, predicted))


def synthetic_dataset(
    *,
    n_rows: int,
    n_features: int,
    n_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_rows, n_features))

    times = np.arange(n_rows, dtype=np.int64) + rng.integers(-12, 12, size=n_rows)
    times = np.clip(times, 0, None).astype(np.float64)

    origin_ids = rng.integers(1, n_cells + 1, size=n_rows, dtype=np.uint64)
    destination_ids = rng.integers(1, n_cells + 1, size=n_rows, dtype=np.uint64)

    # Build a synthetic geography-like signal that is stronger when the lane is in-sample.
    origin_effect = np.sin(origin_ids.astype(np.float64) / 7.0) + 0.08 * (origin_ids % 13)
    destination_effect = 0.6 * np.cos(destination_ids.astype(np.float64) / 9.0) + 0.05 * (
        destination_ids % 11
    )

    hour = (times.astype(np.int64) % 24).astype(float)
    hourly_effect = np.where((hour >= 17) & (hour <= 19), 1.3, -0.2)

    cell_lane_interaction = ((origin_ids % destination_ids) % 11).astype(np.float64) * 0.015

    y = (
        2.5 * x[:, 0]
        - 1.1 * x[:, 1]
        + 0.42 * x[:, 2] ** 2
        + 0.25 * destination_effect
        + 0.8 * origin_effect
        + 0.15 * hourly_effect
        + 0.5 * cell_lane_interaction
        + rng.normal(0.0, 0.35, size=n_rows)
    )

    grid = int(np.ceil(np.sqrt(n_cells)))
    lon = (origin_ids % grid) / float(grid) + rng.normal(0.0, 0.02, size=n_rows)
    lat = ((origin_ids // grid) % grid) / float(grid) + rng.normal(0.0, 0.02, size=n_rows)
    lon = np.clip(lon, 0.0, 1.0)
    lat = np.clip(lat, 0.0, 1.0)
    coordinates = np.column_stack([lon, lat])

    return (
        x,
        origin_ids.astype(np.float64),
        y,
        {
            "times": times,
            "origin_ids": origin_ids,
            "destination_ids": destination_ids,
            "coordinates": coordinates,
        },
    )


def _pick_fold(
    splits: list[tuple[np.ndarray, np.ndarray]], fold: int
) -> tuple[np.ndarray, np.ndarray]:
    if not splits:
        raise ValueError("split generation returned no folds")
    fold = int(fold)
    if fold < 0:
        fold = len(splits) + fold
    fold = min(max(fold, 0), len(splits) - 1)
    return splits[fold]


def _tail_split(n_rows: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    split = int(n_rows * train_fraction)
    if split <= 0 or split >= n_rows:
        raise ValueError("train_frac must keep both train and validation sets")

    idx = np.arange(n_rows)
    return idx[:split], idx[split:]


def _random_split(
    n_rows: int, train_fraction: float, *, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_rows)
    split = int(n_rows * train_fraction)
    if split <= 0 or split >= n_rows:
        raise ValueError("train_frac must keep both train and validation sets")

    return perm[:split], perm[split:]


def _temporal_block_split(
    *,
    times: np.ndarray,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    return out_of_time_split(
        times,
        validation_fraction=1.0 - train_fraction,
        gap=0,
    )


def _spatial_block_split(
    *,
    coordinates: np.ndarray,
    n_splits: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = list(spatial_blocked_cv(coordinates, n_splits=n_splits))
    return _pick_fold(splits, fold)


def _group_block_split(
    *,
    groups: np.ndarray,
    n_splits: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = list(grouped_blocked_cv(groups, n_splits=n_splits))
    return _pick_fold(splits, fold)


def _run_model_comparison(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    include_sklearn: bool,
    neural_dim: int,
) -> dict[str, Any]:
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    id_train = ids[train_idx]
    id_test = ids[test_idx]

    model = CartoBoostRegressor(
        n_estimators=50,
        learning_rate=0.07,
        max_depth=5,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    carto_fit_ms = ms(start)

    start = time.perf_counter()
    pred = model.predict(x_test)
    carto_predict_ms = ms(start)
    carto_mae = mae(y_test, pred)

    hybrid = NeuralEmbeddingRegressor(
        dim=neural_dim,
        id_column=None,
        drop_id_column=False,
        random_state=42,
    )

    start = time.perf_counter()
    hybrid.fit(x_train, y_train, ids=id_train)
    hybrid_fit_ms = ms(start)

    start = time.perf_counter()
    hybrid_pred = hybrid.predict(x_test, ids=id_test)
    hybrid_predict_ms = ms(start)
    hybrid_mae = mae(y_test, hybrid_pred)

    result = {
        "cartoboost": {
            "mae": carto_mae,
            "fit_ms": carto_fit_ms,
            "predict_ms": carto_predict_ms,
        },
        "neural_embedding_hybrid": {
            "mae": hybrid_mae,
            "fit_ms": hybrid_fit_ms,
            "predict_ms": hybrid_predict_ms,
        },
        "hybrid_vs_baseline_improvement_mae": carto_mae - hybrid_mae,
    }

    if include_sklearn:
        sklearn_model = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=4,
            random_state=42,
        )

        start = time.perf_counter()
        sklearn_model.fit(x_train, y_train)
        sklearn_fit_ms = ms(start)

        start = time.perf_counter()
        sklearn_pred = sklearn_model.predict(x_test)
        sklearn_predict_ms = ms(start)

        result["sklearn_gbr"] = {
            "mae": mae(y_test, sklearn_pred),
            "fit_ms": sklearn_fit_ms,
            "predict_ms": sklearn_predict_ms,
        }

    return result


def _build_scenarios(
    mode: str,
    *,
    n_rows: int,
    train_frac: float,
    metadata: dict[str, Any],
    n_splits: int,
    fold: int,
    random_state: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, str, np.ndarray]]:
    times = metadata["times"]
    origin_ids = metadata["origin_ids"]
    destination_ids = metadata["destination_ids"]
    coordinates = metadata["coordinates"]

    if mode == "all":
        requested = sorted(SPLIT_MODES)
    else:
        requested = [mode]

    scenarios: dict[str, tuple[np.ndarray, np.ndarray, str, np.ndarray]] = {}

    for split_mode in requested:
        if split_mode == "tail":
            train_idx, test_idx = _tail_split(n_rows, train_frac)
            scenarios["tail"] = (train_idx, test_idx, "origin", origin_ids)

        elif split_mode == "random":
            train_idx, test_idx = _random_split(n_rows, train_frac, random_state=random_state)
            scenarios["random"] = (train_idx, test_idx, "origin", origin_ids)

        elif split_mode == "temporal_blocked":
            train_idx, test_idx = _temporal_block_split(times=times, train_fraction=train_frac)
            scenarios["temporal_blocked"] = (train_idx, test_idx, "origin", origin_ids)

        elif split_mode == "geo_blocked":
            train_idx, test_idx = _spatial_block_split(
                coordinates=coordinates,
                n_splits=max(2, n_splits),
                fold=fold,
            )
            scenarios["geo_blocked"] = (train_idx, test_idx, "origin", origin_ids)

        elif split_mode == "cold_origin":
            train_idx, test_idx = _group_block_split(
                groups=origin_ids,
                n_splits=max(2, n_splits),
                fold=fold,
            )
            scenarios["cold_origin"] = (train_idx, test_idx, "origin", origin_ids)

        elif split_mode == "cold_destination":
            train_idx, test_idx = _group_block_split(
                groups=destination_ids,
                n_splits=max(2, n_splits),
                fold=fold,
            )
            scenarios["cold_destination"] = (train_idx, test_idx, "destination", destination_ids)

        else:
            raise ValueError(f"unknown split mode {split_mode}")

    return scenarios


def _format_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name:22} | MAE {metrics['mae']:.4f} "
        f"| fit {metrics['fit_ms']:.1f} ms "
        f"| predict {metrics['predict_ms']:.1f} ms"
    )


def main() -> None:
    args = parse_args()

    x, _, y, metadata = synthetic_dataset(
        n_rows=args.n_rows,
        n_features=args.n_features,
        n_cells=args.n_cells,
        seed=args.seed,
    )

    scenarios = _build_scenarios(
        args.split_mode,
        n_rows=args.n_rows,
        train_frac=args.train_frac,
        metadata=metadata,
        n_splits=args.n_splits,
        fold=args.block_fold,
        random_state=args.random_state,
    )

    scenario_reports: dict[str, dict[str, Any]] = {}

    for mode_name, (train_idx, test_idx, _id_name, scenario_ids) in scenarios.items():
        results = _run_model_comparison(
            x,
            y,
            scenario_ids,
            train_idx,
            test_idx,
            include_sklearn=args.include_sklearn,
            neural_dim=args.n_neural_dim,
        )

        scenario_reports[mode_name] = {
            "split_size": {
                "train_rows": int(train_idx.size),
                "test_rows": int(test_idx.size),
            },
            "ids": _id_name,
            "results": results,
        }

    payload = {
        "seed": args.seed,
        "n_rows": args.n_rows,
        "n_features": args.n_features,
        "n_cells": args.n_cells,
        "n_neural_dim": args.n_neural_dim,
        "train_frac": args.train_frac,
        "split_mode": args.split_mode,
        "scenarios": scenario_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Benchmarks written to {args.output}")
    for mode_name, report in scenario_reports.items():
        print(f"\nScenario: {mode_name} (ids={report['ids']})")
        for name, metrics in report["results"].items():
            if name == "hybrid_vs_baseline_improvement_mae":
                print(f"{name:22} | {metrics:+.4f}")
            else:
                print(_format_row(name, metrics))


if __name__ == "__main__":
    main()
