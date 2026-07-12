#!/usr/bin/env python3
"""Run synthetic splitter acceptance checks and write JSON/Markdown artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import CartoBoostRegressor  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "target" / "validation" / "splitter_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def total_variation(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(values))))


def gate(
    name: str,
    passed: bool,
    actual: float,
    threshold: float,
    comparator: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": float(actual),
        "threshold": float(threshold),
        "comparator": comparator,
    }


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    splitters: list[str],
    n_estimators: int = 1,
    learning_rate: float = 1.0,
    max_depth: int = 1,
    min_samples_leaf: int = 1,
    leaf_predictor: str = "constant",
    linear_leaf_features: list[str] | None = None,
    fuzzy: bool = False,
    fuzzy_bandwidth: float = 0.0,
) -> CartoBoostRegressor:
    requested = {str(value).split(":", 1)[0].lower() for value in splitters}
    feature_schema: dict[str, list[dict[str, object]]] | None = None
    sparse_sets: dict[str, list[list[int]]] | None = None
    if "diagonal_2d" in requested or "gaussian_2d" in requested:
        if x.shape[1] < 2:
            raise ValueError("spatial acceptance fixtures require two dense columns")
        feature_schema = {
            "dense": [
                {"name": "x0", "kind": "spatial", "other": "x1"},
                {"name": "x1", "kind": "spatial", "other": "x0"},
            ],
            "sparse_sets": [],
        }
    elif "periodic_time" in requested:
        feature_schema = {
            "dense": [{"name": "hour", "kind": "periodic", "period": 24}],
            "sparse_sets": [],
        }
    elif "sparse_set" in requested:
        sparse_sets = {"acceptance_ids": [[int(value)] for value in x[:, 0]]}
        feature_schema = {
            "dense": [{"name": "id", "kind": "numeric"}],
            "sparse_sets": [{"name": "acceptance_ids", "kind": "sparse_set"}],
        }
    policy = "axis_only" if requested <= {"axis"} else "structured"
    model = CartoBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_gain=0.0,
        split_policy=policy,
        leaf_predictor=leaf_predictor,
        linear_leaf_features=linear_leaf_features,
        fuzzy=fuzzy,
        fuzzy_bandwidth=fuzzy_bandwidth,
        l2_regularization=0.0,
    )
    model.fit(x, y, feature_schema=feature_schema, sparse_sets=sparse_sets)
    model._acceptance_sparse_sets = sparse_sets
    return model


def grid_2d(size: int = 31, low: float = -3.0, high: float = 3.0) -> np.ndarray:
    values = np.linspace(low, high, size)
    xx, yy = np.meshgrid(values, values)
    return np.column_stack([xx.ravel(), yy.ravel()])


def axis_metrics() -> dict[str, Any]:
    x = np.linspace(-3.0, 3.0, 49).reshape(-1, 1)
    y = np.where(x[:, 0] > 0.0, 10.0, -10.0)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=2)
    pred = axis.predict(x)
    return {
        "models": {"axis": {"train_rmse": rmse(y, pred)}},
        "inspection_metrics": {
            "left_prediction": float(axis.predict([[-2.0]])[0]),
            "right_prediction": float(axis.predict([[2.0]])[0]),
        },
        "acceptance_gates": [
            gate("axis_train_rmse_lt_1e_12", rmse(y, pred) < 1e-12, rmse(y, pred), 1e-12, "<"),
            gate(
                "axis_left_right_margin_gt_15",
                float(axis.predict([[2.0]])[0] - axis.predict([[-2.0]])[0]) > 15.0,
                float(axis.predict([[2.0]])[0] - axis.predict([[-2.0]])[0]),
                15.0,
                ">",
            ),
        ],
        "future_checks": [
            "A depth-1 axis stump must solve a clean threshold fixture.",
            "Left and right inspection predictions should remain well separated.",
        ],
    }


def diagonal_metrics() -> dict[str, Any]:
    x = grid_2d()
    y = np.where(x[:, 0] + x[:, 1] > 0.0, 10.0, -10.0)
    noisy_y = y + np.random.default_rng(10).normal(0.0, 0.5, size=y.shape)
    holdout = grid_2d(size=29, low=-2.9, high=2.9)
    holdout_y = np.where(holdout[:, 0] + holdout[:, 1] > 0.0, 10.0, -10.0)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=4)
    diagonal = fit_model(x, y, splitters=["diagonal_2d"], min_samples_leaf=4)
    noisy_axis = fit_model(x, noisy_y, splitters=["axis"], min_samples_leaf=4)
    noisy_diagonal = fit_model(x, noisy_y, splitters=["diagonal_2d"], min_samples_leaf=4)
    axis_pred = axis.predict(x)
    diagonal_pred = diagonal.predict(x)
    axis_holdout = axis.predict(holdout)
    diagonal_holdout = diagonal.predict(holdout)
    noisy_axis_holdout = noisy_axis.predict(holdout)
    noisy_diagonal_holdout = noisy_diagonal.predict(holdout)
    holdout_ratio = rmse(holdout_y, diagonal_holdout) / rmse(holdout_y, axis_holdout)
    noisy_holdout_ratio = rmse(holdout_y, noisy_diagonal_holdout) / rmse(
        holdout_y, noisy_axis_holdout
    )
    return {
        "models": {
            "axis": {
                "train_rmse": rmse(y, axis_pred),
                "holdout_rmse": rmse(holdout_y, axis_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_axis_holdout),
            },
            "diagonal_2d": {
                "train_rmse": rmse(y, diagonal_pred),
                "holdout_rmse": rmse(holdout_y, diagonal_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_diagonal_holdout),
            },
        },
        "inspection_metrics": {
            "axis_boundary_mae": mae(y, axis_pred),
            "diagonal_boundary_mae": mae(y, diagonal_pred),
            "holdout_rmse_ratio": holdout_ratio,
            "noisy_holdout_rmse_ratio": noisy_holdout_ratio,
            "negative_corner_prediction": float(diagonal.predict([[-2.0, -2.0]])[0]),
            "positive_corner_prediction": float(diagonal.predict([[2.0, 2.0]])[0]),
        },
        "acceptance_gates": [
            gate("diagonal_holdout_ratio_lt_0_25", holdout_ratio < 0.25, holdout_ratio, 0.25, "<"),
            gate(
                "diagonal_noisy_holdout_ratio_lt_0_05",
                noisy_holdout_ratio < 0.05,
                noisy_holdout_ratio,
                0.05,
                "<",
            ),
            gate(
                "diagonal_holdout_rmse_lt_1e_10",
                rmse(holdout_y, diagonal_holdout) < 1e-10,
                rmse(holdout_y, diagonal_holdout),
                1e-10,
                "<",
            ),
        ],
        "future_checks": [
            "Diagonal splitters should beat a one-stump axis baseline on x + y boundaries.",
            "Corner probes catch swapped branch polarity.",
        ],
    }


def gaussian_metrics() -> dict[str, Any]:
    x = grid_2d(size=35)
    y = np.where(np.hypot(x[:, 0], x[:, 1]) <= 1.5, 10.0, -10.0)
    noisy_y = y + np.random.default_rng(20).normal(0.0, 0.5, size=y.shape)
    holdout = grid_2d(size=31, low=-2.85, high=2.85)
    holdout_y = np.where(np.hypot(holdout[:, 0], holdout[:, 1]) <= 1.5, 10.0, -10.0)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=4)
    gaussian = fit_model(x, y, splitters=["gaussian_2d"], min_samples_leaf=4)
    noisy_axis = fit_model(x, noisy_y, splitters=["axis"], min_samples_leaf=4)
    noisy_gaussian = fit_model(x, noisy_y, splitters=["gaussian_2d"], min_samples_leaf=4)
    axis_pred = axis.predict(x)
    gaussian_pred = gaussian.predict(x)
    axis_holdout = axis.predict(holdout)
    gaussian_holdout = gaussian.predict(holdout)
    noisy_axis_holdout = noisy_axis.predict(holdout)
    noisy_gaussian_holdout = noisy_gaussian.predict(holdout)
    holdout_ratio = rmse(holdout_y, gaussian_holdout) / rmse(holdout_y, axis_holdout)
    noisy_holdout_ratio = rmse(holdout_y, noisy_gaussian_holdout) / rmse(
        holdout_y, noisy_axis_holdout
    )
    return {
        "models": {
            "axis": {
                "train_rmse": rmse(y, axis_pred),
                "holdout_rmse": rmse(holdout_y, axis_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_axis_holdout),
            },
            "gaussian_2d": {
                "train_rmse": rmse(y, gaussian_pred),
                "holdout_rmse": rmse(holdout_y, gaussian_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_gaussian_holdout),
            },
        },
        "inspection_metrics": {
            "axis_radial_mae": mae(y, axis_pred),
            "gaussian_radial_mae": mae(y, gaussian_pred),
            "holdout_rmse_ratio": holdout_ratio,
            "noisy_holdout_rmse_ratio": noisy_holdout_ratio,
            "center_prediction": float(gaussian.predict([[0.0, 0.0]])[0]),
            "outer_prediction": float(gaussian.predict([[3.0, 3.0]])[0]),
        },
        "acceptance_gates": [
            gate("gaussian_holdout_ratio_lt_0_30", holdout_ratio < 0.30, holdout_ratio, 0.30, "<"),
            gate(
                "gaussian_noisy_holdout_ratio_lt_0_05",
                noisy_holdout_ratio < 0.05,
                noisy_holdout_ratio,
                0.05,
                "<",
            ),
            gate(
                "gaussian_holdout_rmse_lt_1e_10",
                rmse(holdout_y, gaussian_holdout) < 1e-10,
                rmse(holdout_y, gaussian_holdout),
                1e-10,
                "<",
            ),
        ],
        "future_checks": [
            "Gaussian splitters should beat axis stumps on radial fixtures.",
            "Center and outer probes verify inside/outside routing.",
        ],
    }


def periodic_metrics() -> dict[str, Any]:
    hours = np.array([[float(hour)] for hour in range(24) for _ in range(3)])
    y = np.array([15.0 if hour[0] >= 22.0 or hour[0] <= 2.0 else -5.0 for hour in hours])
    noisy_y = y + np.random.default_rng(30).normal(0.0, 0.5, size=y.shape)
    holdout = np.arange(0.25, 24.0, 0.5).reshape(-1, 1)
    holdout_y = np.where((holdout[:, 0] >= 22.0) | (holdout[:, 0] <= 2.0), 15.0, -5.0)
    axis = fit_model(hours, y, splitters=["axis"], min_samples_leaf=3)
    periodic = fit_model(hours, y, splitters=["periodic_time"], min_samples_leaf=3)
    noisy_axis = fit_model(hours, noisy_y, splitters=["axis"], min_samples_leaf=3)
    noisy_periodic = fit_model(hours, noisy_y, splitters=["periodic_time"], min_samples_leaf=3)
    axis_pred = axis.predict(hours)
    periodic_pred = periodic.predict(hours)
    axis_holdout = axis.predict(holdout)
    periodic_holdout = periodic.predict(holdout)
    noisy_axis_holdout = noisy_axis.predict(holdout)
    noisy_periodic_holdout = noisy_periodic.predict(holdout)
    holdout_ratio = rmse(holdout_y, periodic_holdout) / rmse(holdout_y, axis_holdout)
    noisy_holdout_ratio = rmse(holdout_y, noisy_periodic_holdout) / rmse(
        holdout_y, noisy_axis_holdout
    )
    return {
        "models": {
            "axis": {
                "train_rmse": rmse(y, axis_pred),
                "holdout_rmse": rmse(holdout_y, axis_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_axis_holdout),
            },
            "periodic_time": {
                "train_rmse": rmse(y, periodic_pred),
                "holdout_rmse": rmse(holdout_y, periodic_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_periodic_holdout),
            },
        },
        "inspection_metrics": {
            "axis_wrap_gap": abs(float(axis.predict([[23.0]])[0] - axis.predict([[1.0]])[0])),
            "periodic_wrap_gap": abs(
                float(periodic.predict([[23.0]])[0] - periodic.predict([[1.0]])[0])
            ),
            "periodic_edge_mid_gap": float(
                np.mean(periodic.predict([[23.0], [1.0]])) - periodic.predict([[12.0]])[0]
            ),
            "holdout_rmse_ratio": holdout_ratio,
            "noisy_holdout_rmse_ratio": noisy_holdout_ratio,
        },
        "acceptance_gates": [
            gate("periodic_holdout_ratio_lt_0_50", holdout_ratio < 0.50, holdout_ratio, 0.50, "<"),
            gate(
                "periodic_noisy_holdout_ratio_lt_0_08",
                noisy_holdout_ratio < 0.08,
                noisy_holdout_ratio,
                0.08,
                "<",
            ),
            gate(
                "periodic_wrap_gap_lt_1e_12",
                abs(float(periodic.predict([[23.0]])[0] - periodic.predict([[1.0]])[0])) < 1e-12,
                abs(float(periodic.predict([[23.0]])[0] - periodic.predict([[1.0]])[0])),
                1e-12,
                "<",
            ),
        ],
        "future_checks": [
            "Periodic splitters must handle wraparound intervals such as 22..2.",
            "Equivalent edge-hour predictions should remain close across the period boundary.",
        ],
    }


def fuzzy_metrics() -> dict[str, Any]:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    hard = fit_model(x, y, splitters=["axis"])
    fuzzy = fit_model(x, y, splitters=["axis"], fuzzy=True, fuzzy_bandwidth=1.0)
    probes = np.linspace(1.0, 2.0, 101).reshape(-1, 1)
    hard_pred = hard.predict(probes)
    fuzzy_pred = fuzzy.predict(probes)
    hard_tv = total_variation(hard_pred)
    fuzzy_tv = total_variation(fuzzy_pred)
    jump_ratio = abs(float(fuzzy.predict([[1.49]])[0] - fuzzy.predict([[1.51]])[0])) / abs(
        float(hard.predict([[1.49]])[0] - hard.predict([[1.51]])[0])
    )
    return {
        "models": {
            "hard_axis": {"probe_rmse": rmse(np.linspace(0.0, 10.0, 101), hard_pred)},
            "fuzzy_axis": {"probe_rmse": rmse(np.linspace(0.0, 10.0, 101), fuzzy_pred)},
        },
        "inspection_metrics": {
            "hard_boundary_jump": abs(float(hard.predict([[1.49]])[0] - hard.predict([[1.51]])[0])),
            "fuzzy_boundary_jump": abs(
                float(fuzzy.predict([[1.49]])[0] - fuzzy.predict([[1.51]])[0])
            ),
            "boundary_jump_ratio": jump_ratio,
            "hard_total_variation": hard_tv,
            "fuzzy_total_variation": fuzzy_tv,
            "fuzzy_midpoint_prediction": float(fuzzy.predict([[1.5]])[0]),
        },
        "acceptance_gates": [
            gate("fuzzy_jump_ratio_lt_0_05", jump_ratio < 0.05, jump_ratio, 0.05, "<"),
            gate(
                "fuzzy_total_variation_lte_hard", fuzzy_tv <= hard_tv, fuzzy_tv - hard_tv, 0.0, "<="
            ),
            gate(
                "fuzzy_midpoint_exact",
                abs(float(fuzzy.predict([[1.5]])[0]) - 5.0) < 1e-12,
                abs(float(fuzzy.predict([[1.5]])[0]) - 5.0),
                1e-12,
                "<",
            ),
        ],
        "future_checks": [
            "Fuzzy routing should smooth local boundary jumps.",
            "The exact threshold midpoint should blend both child leaves.",
        ],
    }


def fuzzy_periodic_metrics() -> dict[str, Any]:
    hours = np.array([[float(hour)] for hour in range(24) for _ in range(4)])
    y = np.array([20.0 if hour[0] >= 22.0 or hour[0] <= 2.0 else 0.0 for hour in hours])
    hard = fit_model(hours, y, splitters=["periodic_time"], min_samples_leaf=4)
    fuzzy = fit_model(
        hours,
        y,
        splitters=["periodic_time"],
        min_samples_leaf=4,
        fuzzy=True,
        fuzzy_bandwidth=2.0,
    )
    probes = np.linspace(20.0, 24.0, 161).reshape(-1, 1)
    hard_pred = hard.predict(probes)
    fuzzy_pred = fuzzy.predict(probes)
    hard_diffs = np.abs(np.diff(hard_pred))
    fuzzy_diffs = np.abs(np.diff(fuzzy_pred))
    boundary_index = int(np.argmax(hard_diffs))
    hard_jump = float(hard_diffs[boundary_index])
    fuzzy_jump = float(fuzzy_diffs[boundary_index])
    jump_ratio = fuzzy_jump / hard_jump if hard_jump > 0.0 else 0.0
    return {
        "models": {
            "hard_periodic": {"probe_total_variation": total_variation(hard_pred)},
            "fuzzy_periodic": {"probe_total_variation": total_variation(fuzzy_pred)},
        },
        "inspection_metrics": {
            "hard_boundary_jump": hard_jump,
            "fuzzy_boundary_jump": fuzzy_jump,
            "boundary_jump_ratio": jump_ratio,
            "outside_near_boundary_prediction": float(fuzzy.predict([[21.0]])[0]),
            "inside_near_boundary_prediction": float(fuzzy.predict([[23.0]])[0]),
            "far_outside_prediction": float(fuzzy.predict([[12.0]])[0]),
        },
        "acceptance_gates": [
            gate("fuzzy_periodic_jump_ratio_lt_0_05", jump_ratio < 0.05, jump_ratio, 0.05, "<"),
            gate(
                "fuzzy_periodic_inside_gt_outside",
                float(fuzzy.predict([[23.0]])[0] - fuzzy.predict([[21.0]])[0]) > 5.0,
                float(fuzzy.predict([[23.0]])[0] - fuzzy.predict([[21.0]])[0]),
                5.0,
                ">",
            ),
            gate(
                "fuzzy_periodic_far_outside_lt_1",
                abs(float(fuzzy.predict([[12.0]])[0])) < 1.0,
                abs(float(fuzzy.predict([[12.0]])[0])),
                1.0,
                "<",
            ),
        ],
        "future_checks": [
            "Fuzzy periodic routing should smooth jumps at wraparound interval boundaries.",
            "Fuzzy smoothing should keep far-outside leakage bounded on this one-stump fixture.",
        ],
    }


def linear_metrics() -> dict[str, Any]:
    x = np.linspace(0.0, 5.0, 30).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 3.0
    noisy_y = y + np.random.default_rng(40).normal(0.0, 0.25, size=y.shape)
    holdout = np.linspace(0.05, 4.95, 37).reshape(-1, 1)
    holdout_y = 2.0 * holdout[:, 0] + 3.0
    constant = fit_model(x, y, splitters=["axis"], max_depth=1, min_samples_leaf=16)
    linear = fit_model(
        x,
        y,
        splitters=["axis"],
        max_depth=1,
        min_samples_leaf=16,
        leaf_predictor="linear",
        linear_leaf_features=["0"],
    )
    noisy_constant = fit_model(x, noisy_y, splitters=["axis"], max_depth=1, min_samples_leaf=16)
    noisy_linear = fit_model(
        x,
        noisy_y,
        splitters=["axis"],
        max_depth=1,
        min_samples_leaf=16,
        leaf_predictor="linear",
        linear_leaf_features=["0"],
    )
    constant_pred = constant.predict(x)
    linear_pred = linear.predict(x)
    constant_holdout = constant.predict(holdout)
    linear_holdout = linear.predict(holdout)
    noisy_constant_holdout = noisy_constant.predict(holdout)
    noisy_linear_holdout = noisy_linear.predict(holdout)
    holdout_ratio = rmse(holdout_y, linear_holdout) / rmse(holdout_y, constant_holdout)
    noisy_holdout_ratio = rmse(holdout_y, noisy_linear_holdout) / rmse(
        holdout_y, noisy_constant_holdout
    )
    return {
        "models": {
            "constant_leaf": {
                "train_rmse": rmse(y, constant_pred),
                "holdout_rmse": rmse(holdout_y, constant_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_constant_holdout),
            },
            "linear_leaf": {
                "train_rmse": rmse(y, linear_pred),
                "holdout_rmse": rmse(holdout_y, linear_holdout),
                "noisy_train_noiseless_holdout_rmse": rmse(holdout_y, noisy_linear_holdout),
            },
        },
        "inspection_metrics": {
            "linear_low_prediction": float(linear.predict([[0.0]])[0]),
            "linear_high_prediction": float(linear.predict([[5.0]])[0]),
            "holdout_rmse_ratio": holdout_ratio,
            "noisy_holdout_rmse_ratio": noisy_holdout_ratio,
        },
        "acceptance_gates": [
            gate("linear_holdout_ratio_lt_0_05", holdout_ratio < 0.05, holdout_ratio, 0.05, "<"),
            gate(
                "linear_noisy_holdout_ratio_lt_0_08",
                noisy_holdout_ratio < 0.08,
                noisy_holdout_ratio,
                0.08,
                "<",
            ),
            gate(
                "linear_holdout_rmse_lt_1e_10",
                rmse(holdout_y, linear_holdout) < 1e-10,
                rmse(holdout_y, linear_holdout),
                1e-10,
                "<",
            ),
        ],
        "future_checks": [
            "A linear leaf should solve a noiseless linear residual fixture.",
            "Endpoint predictions should preserve the learned local slope.",
        ],
    }


def sparse_metrics() -> dict[str, Any]:
    x = np.array([[7.0], [7.0], [3.0], [4.0], [9.0], [9.0], [8.0], [8.0]])
    y = np.array([30.0, 30.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    probe = np.array([[3.0], [4.0], [7.0], [8.0], [9.0], [11.0]])
    probe_expected = np.array([-5.0, -5.0, 30.0, -5.0, -5.0, -5.0])
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=2)
    sparse = fit_model(x, y, splitters=["sparse_set"], min_samples_leaf=2)
    axis_pred = axis.predict(x)
    sparse_sets = sparse._acceptance_sparse_sets
    sparse_pred = sparse.predict(x, sparse_sets=sparse_sets)
    probe_sparse_sets = {"acceptance_ids": [[int(value)] for value in probe[:, 0]]}
    sparse_probe = sparse.predict(probe, sparse_sets=probe_sparse_sets)
    return {
        "models": {
            "axis": {"train_rmse": rmse(y, axis_pred)},
            "sparse_set": {
                "train_rmse": rmse(y, sparse_pred),
                "probe_rmse": rmse(probe_expected, sparse_probe),
            },
        },
        "inspection_metrics": {
            "toll_cell_prediction": float(
                sparse.predict([[7.0]], sparse_sets={"acceptance_ids": [[7]]})[0]
            ),
            "cold_cell_prediction": float(
                sparse.predict([[3.0]], sparse_sets={"acceptance_ids": [[3]]})[0]
            ),
            "unseen_cell_prediction": float(
                sparse.predict([[11.0]], sparse_sets={"acceptance_ids": [[11]]})[0]
            ),
            "observed_cell_count": float(len(set(x[:, 0]))),
            "dense_one_hot_columns_avoided": float(len(set(x[:, 0])) - x.shape[1]),
        },
        "acceptance_gates": [
            gate(
                "sparse_probe_rmse_lt_1e_12",
                rmse(probe_expected, sparse_probe) < 1e-12,
                rmse(probe_expected, sparse_probe),
                1e-12,
                "<",
            ),
            gate(
                "sparse_toll_margin_gt_20",
                float(
                    sparse.predict([[7.0]], sparse_sets={"acceptance_ids": [[7]]})[0]
                    - sparse.predict([[3.0]], sparse_sets={"acceptance_ids": [[3]]})[0]
                )
                > 20.0,
                float(
                    sparse.predict([[7.0]], sparse_sets={"acceptance_ids": [[7]]})[0]
                    - sparse.predict([[3.0]], sparse_sets={"acceptance_ids": [[3]]})[0]
                ),
                20.0,
                ">",
            ),
        ],
        "future_checks": [
            "Sparse set routing should isolate high-cardinality membership effects.",
            "The fixture stores IDs directly and avoids dense one-hot expansion.",
        ],
    }


def learning_rate_metrics() -> dict[str, Any]:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    model = fit_model(x, y, splitters=["axis"], learning_rate=0.25)
    pred = model.predict(x)
    expected = np.array([3.75, 3.75, 6.25, 6.25])
    return {
        "models": {"axis_shrinkage": {"train_rmse": rmse(expected, pred)}},
        "inspection_metrics": {
            "expected_left_prediction": float(expected[0]),
            "actual_left_prediction": float(pred[0]),
            "expected_right_prediction": float(expected[-1]),
            "actual_right_prediction": float(pred[-1]),
        },
        "acceptance_gates": [
            gate(
                "shrinkage_predictions_exact",
                rmse(expected, pred) < 1e-12,
                rmse(expected, pred),
                1e-12,
                "<",
            )
        ],
        "future_checks": [
            "The fitted tree must target the L2 negative gradient and apply learning_rate once.",
            "This catches accidentally fitting raw targets or double-applying shrinkage.",
        ],
    }


def collect_metrics() -> dict[str, Any]:
    return {
        "axis_threshold": axis_metrics(),
        "diagonal_2d": diagonal_metrics(),
        "gaussian_2d": gaussian_metrics(),
        "periodic_wraparound": periodic_metrics(),
        "fuzzy_axis": fuzzy_metrics(),
        "fuzzy_periodic_wraparound": fuzzy_periodic_metrics(),
        "linear_leaf": linear_metrics(),
        "sparse_set": sparse_metrics(),
        "learning_rate_gradient_shrinkage": learning_rate_metrics(),
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# CartoBoost splitter acceptance metrics",
        "",
        "Synthetic checks are generated from deterministic fixtures. They prove exact behavior for",
        "fixed examples and qualitative splitter superiority on designed structures; they do not",
        "claim universal production superiority.",
        "",
        "| phase | model | metric | value |",
        "| --- | --- | --- | ---: |",
    ]
    for phase, phase_metrics in metrics.items():
        for model_name, model_metrics in phase_metrics["models"].items():
            for metric, value in model_metrics.items():
                lines.append(f"| {phase} | {model_name} | {metric} | {value:.8f} |")
    lines.extend(["", "## Inspection Metrics", ""])
    for phase, phase_metrics in metrics.items():
        lines.append(f"### {phase}")
        for metric, value in phase_metrics["inspection_metrics"].items():
            lines.append(f"- `{metric}`: {value:.8f}")
        if "acceptance_gates" in phase_metrics:
            for check in phase_metrics["acceptance_gates"]:
                status = "PASS" if check["passed"] else "FAIL"
                lines.append(
                    f"- `{check['name']}`: {status} "
                    f"({check['actual']:.8f} {check['comparator']} {check['threshold']:.8f})"
                )
        lines.append("")
    return "\n".join(lines)


def assert_finite(metrics: dict[str, Any]) -> None:
    for phase_metrics in metrics.values():
        for group in ("models", "inspection_metrics"):
            values = phase_metrics[group]
            if group == "models":
                values = {
                    f"{model}.{metric}": value
                    for model, model_metrics in values.items()
                    for metric, value in model_metrics.items()
                }
            for name, value in values.items():
                if not math.isfinite(float(value)):
                    raise ValueError(f"non-finite metric {name}: {value}")
        for check in phase_metrics.get("acceptance_gates", []):
            for field in ("actual", "threshold"):
                if not math.isfinite(float(check[field])):
                    raise ValueError(f"non-finite gate {check['name']} {field}: {check[field]}")


def assert_acceptance_gates(metrics: dict[str, Any]) -> None:
    failures = [
        f"{phase}.{check['name']}"
        for phase, phase_metrics in metrics.items()
        for check in phase_metrics.get("acceptance_gates", [])
        if not check["passed"]
    ]
    if failures:
        raise AssertionError("acceptance gates failed: " + ", ".join(failures))


def main() -> None:
    args = parse_args()
    metrics = collect_metrics()
    assert_finite(metrics)
    assert_acceptance_gates(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acceptance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "acceptance_metrics.md").write_text(
        render_markdown(metrics) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
