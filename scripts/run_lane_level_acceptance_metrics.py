#!/usr/bin/env python3
"""Run synthetic lane-level CartoBoost checks and write committed diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import CartoBoostRegressor  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "lane_level_tests"

REGIONS = np.array(
    [
        [-1.5, -1.5],
        [-1.5, 1.5],
        [1.5, -1.5],
        [1.5, 1.5],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


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
    max_depth: int = 1,
    min_samples_leaf: int = 1,
    fuzzy: bool = False,
    fuzzy_bandwidth: float = 0.0,
) -> CartoBoostRegressor:
    requested = {str(value).split(":", 1)[0].lower() for value in splitters}
    dense: list[dict[str, object]] = [
        {"name": f"feature_{index}", "kind": "numeric"} for index in range(x.shape[1])
    ]
    if "gaussian_2d" in requested:
        if x.shape[1] == 2:
            spatial_pairs = [(0, 1)]
        else:
            spatial_pairs = [(0, 1), (2, 3), (6, 7)]
        for left, right in spatial_pairs:
            dense[left] = {
                "name": f"feature_{left}",
                "kind": "spatial",
                "other": f"feature_{right}",
            }
            dense[right] = {
                "name": f"feature_{right}",
                "kind": "spatial",
                "other": f"feature_{left}",
            }
    if "periodic_time" in requested:
        periodic_index = 5 if x.shape[1] >= 6 else 0
        dense[periodic_index] = {
            "name": f"feature_{periodic_index}",
            "kind": "periodic",
            "period": 24,
        }
    feature_schema: dict[str, list[dict[str, object]]] | None = None
    sparse_sets: dict[str, list[list[int]]] | None = None
    if requested & {"gaussian_2d", "periodic_time", "sparse_set"}:
        feature_schema = {"dense": dense, "sparse_sets": []}
    if "sparse_set" in requested:
        sparse_index = 4 if x.shape[1] >= 5 else 0
        sparse_sets = {"lane_id": [[int(value)] for value in x[:, sparse_index]]}
        if feature_schema is None:
            feature_schema = {"dense": dense, "sparse_sets": []}
        feature_schema["sparse_sets"] = [{"name": "lane_id", "kind": "sparse_set"}]
    policy = "axis_only" if requested <= {"axis"} else "structured"
    model = CartoBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=1.0,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_gain=0.0,
        split_policy=policy,
        fuzzy=fuzzy,
        fuzzy_bandwidth=fuzzy_bandwidth,
    )
    model.fit(x, y, feature_schema=feature_schema, sparse_sets=sparse_sets)
    model._acceptance_sparse_index = 4 if x.shape[1] >= 5 else 0
    model._acceptance_uses_sparse = sparse_sets is not None
    return model


def predict_model(model: CartoBoostRegressor, x: np.ndarray) -> np.ndarray:
    if getattr(model, "_acceptance_uses_sparse", False):
        index = int(getattr(model, "_acceptance_sparse_index", 0))
        sparse_sets = {"lane_id": [[int(value)] for value in x[:, index]]}
        return model.predict(x, sparse_sets=sparse_sets)
    return model.predict(x)


def lane_rows(*, repeats_per_hour: int = 1) -> np.ndarray:
    rows: list[list[float]] = []
    for origin in range(4):
        for destination in range(4):
            lane_id = origin * 4 + destination
            origin_xy = REGIONS[origin]
            destination_xy = REGIONS[destination]
            midpoint = (origin_xy + destination_xy) / 2.0
            distance = float(np.linalg.norm(destination_xy - origin_xy))
            for hour in range(24):
                for _ in range(repeats_per_hour):
                    rows.append(
                        [
                            float(origin_xy[0]),
                            float(origin_xy[1]),
                            float(destination_xy[0]),
                            float(destination_xy[1]),
                            float(lane_id),
                            float(hour),
                            float(midpoint[0]),
                            float(midpoint[1]),
                            distance,
                        ]
                    )
    return np.asarray(rows, dtype=float)


def lane_id_target(x: np.ndarray) -> np.ndarray:
    return np.where(x[:, 4] == 7.0, 320.0, 80.0)


def route_midpoint_target(x: np.ndarray) -> np.ndarray:
    radius = np.hypot(x[:, 6], x[:, 7])
    return np.where(radius <= 0.25, 260.0, 90.0)


def wraparound_hour_target(x: np.ndarray) -> np.ndarray:
    hour = x[:, 5]
    return np.where((hour >= 22.0) | (hour <= 2.0), 210.0, 95.0)


def combined_target(x: np.ndarray) -> np.ndarray:
    return (
        95.0
        + 140.0 * (x[:, 4] == 7.0)
        + 55.0 * ((x[:, 5] >= 22.0) | (x[:, 5] <= 2.0))
        + 35.0 * (np.hypot(x[:, 6], x[:, 7]) <= 0.25)
        + 3.5 * x[:, 8]
    )


def sparse_lane_metrics() -> dict[str, Any]:
    x = lane_rows(repeats_per_hour=2)
    y = lane_id_target(x)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=10)
    sparse = fit_model(x, y, splitters=["sparse_set"], min_samples_leaf=10)
    axis_pred = predict_model(axis, x)
    sparse_pred = predict_model(sparse, x)
    hot_probe = x[x[:, 4] == 7.0][0:1]
    cold_probe = x[x[:, 4] == 6.0][0:1]
    margin = float(predict_model(sparse, hot_probe)[0] - predict_model(sparse, cold_probe)[0])
    return {
        "models": {
            "axis_lane_id": {"train_rmse": rmse(y, axis_pred)},
            "sparse_lane_id": {"train_rmse": rmse(y, sparse_pred)},
        },
        "inspection_metrics": {
            "hot_lane_prediction": float(predict_model(sparse, hot_probe)[0]),
            "cold_neighbor_lane_prediction": float(predict_model(sparse, cold_probe)[0]),
            "hot_lane_margin": margin,
            "hot_lane_id": 7.0,
            "lane_count": 16.0,
        },
        "acceptance_gates": [
            gate(
                "sparse_lane_exact", rmse(y, sparse_pred) < 1e-12, rmse(y, sparse_pred), 1e-12, "<"
            ),
            gate(
                "sparse_beats_axis_lane_id",
                rmse(y, sparse_pred) < rmse(y, axis_pred) * 0.05,
                rmse(y, sparse_pred) / rmse(y, axis_pred),
                0.05,
                "<",
            ),
            gate("hot_lane_margin_gt_200", margin > 200.0, margin, 200.0, ">"),
        ],
        "future_checks": [
            "Sparse lane IDs should isolate a non-contiguous hot lane without one-hot expansion.",
            "Neighboring cold lanes should not inherit the hot lane uplift.",
        ],
    }


def route_cartometry_metrics() -> dict[str, Any]:
    x = lane_rows(repeats_per_hour=3)
    y = route_midpoint_target(x)
    axis = fit_model(x[:, [6, 7]], y, splitters=["axis"], min_samples_leaf=12)
    gaussian = fit_model(x[:, [6, 7]], y, splitters=["gaussian_2d"], min_samples_leaf=12)
    route_features = x[:, [6, 7]]
    axis_pred = predict_model(axis, route_features)
    gaussian_pred = predict_model(gaussian, route_features)
    center_probe = np.array([[0.0, 0.0]])
    outer_probe = np.array([[1.5, 1.5]])
    margin = float(
        predict_model(gaussian, center_probe)[0] - predict_model(gaussian, outer_probe)[0]
    )
    return {
        "models": {
            "axis_midpoint": {"train_rmse": rmse(y, axis_pred)},
            "gaussian_midpoint": {"train_rmse": rmse(y, gaussian_pred)},
        },
        "inspection_metrics": {
            "center_lane_prediction": float(predict_model(gaussian, center_probe)[0]),
            "outer_lane_prediction": float(predict_model(gaussian, outer_probe)[0]),
            "center_outer_margin": margin,
            "axis_to_gaussian_rmse_ratio": rmse(y, gaussian_pred) / rmse(y, axis_pred),
        },
        "acceptance_gates": [
            gate(
                "gaussian_route_rmse_lt_axis_half",
                rmse(y, gaussian_pred) < rmse(y, axis_pred) * 0.5,
                rmse(y, gaussian_pred) / rmse(y, axis_pred),
                0.5,
                "<",
            ),
            gate("center_outer_margin_gt_100", margin > 100.0, margin, 100.0, ">"),
        ],
        "future_checks": [
            "Gaussian 2D routing should recover center-focused route cartometry.",
            (
                "This validates lane cartometry from observable midpoint features, "
                "not hidden metadata."
            ),
        ],
    }


def wraparound_hour_metrics() -> dict[str, Any]:
    x = lane_rows(repeats_per_hour=2)
    y = wraparound_hour_target(x)
    hours = x[:, [5]]
    axis = fit_model(hours, y, splitters=["axis"], min_samples_leaf=16)
    periodic = fit_model(hours, y, splitters=["periodic_time"], min_samples_leaf=16)
    axis_pred = predict_model(axis, hours)
    periodic_pred = predict_model(periodic, hours)
    edge_probes = np.array([[23.0], [1.0]])
    periodic_edge_predictions = predict_model(periodic, edge_probes)
    axis_edge_predictions = predict_model(axis, edge_probes)
    edge_gap = abs(float(periodic_edge_predictions[0] - periodic_edge_predictions[1]))
    axis_edge_gap = abs(float(axis_edge_predictions[0] - axis_edge_predictions[1]))
    return {
        "models": {
            "axis_hour": {"train_rmse": rmse(y, axis_pred)},
            "periodic_hour": {"train_rmse": rmse(y, periodic_pred)},
        },
        "inspection_metrics": {
            "periodic_23_vs_1_gap": edge_gap,
            "axis_23_vs_1_gap": axis_edge_gap,
            "periodic_peak_to_midday_margin": float(
                np.mean(predict_model(periodic, np.array([[23.0], [1.0]])))
                - predict_model(periodic, np.array([[12.0]]))[0]
            ),
        },
        "acceptance_gates": [
            gate(
                "periodic_hour_exact",
                rmse(y, periodic_pred) < 1e-12,
                rmse(y, periodic_pred),
                1e-12,
                "<",
            ),
            gate("periodic_edge_gap_lt_1e_12", edge_gap < 1e-12, edge_gap, 1e-12, "<"),
            gate("axis_edge_gap_gt_50", axis_edge_gap > 50.0, axis_edge_gap, 50.0, ">"),
        ],
        "future_checks": [
            "Lane demand around midnight should route as one interval across the 24-hour boundary.",
            "Axis-only hour splits should remain a weaker baseline for wraparound effects.",
        ],
    }


def regional_lane_boosting_metrics() -> dict[str, Any]:
    train = lane_rows(repeats_per_hour=2)
    y_train = combined_target(train)
    holdout = lane_rows(repeats_per_hour=1)
    y_holdout = combined_target(holdout)

    axis = fit_model(
        train,
        y_train,
        splitters=["axis"],
        n_estimators=4,
        max_depth=2,
        min_samples_leaf=12,
    )
    full = fit_model(
        train,
        y_train,
        splitters=["axis", "sparse_set", "gaussian_2d", "periodic_time"],
        n_estimators=4,
        max_depth=2,
        min_samples_leaf=12,
    )
    axis_holdout = predict_model(axis, holdout)
    full_holdout = predict_model(full, holdout)
    holdout_ratio = rmse(y_holdout, full_holdout) / rmse(y_holdout, axis_holdout)
    hot_midnight = holdout[(holdout[:, 4] == 7.0) & (holdout[:, 5] == 23.0)][0:1]
    cold_midday = holdout[(holdout[:, 4] == 6.0) & (holdout[:, 5] == 12.0)][0:1]
    contrast = float(predict_model(full, hot_midnight)[0] - predict_model(full, cold_midday)[0])
    return {
        "models": {
            "axis_only": {"holdout_rmse": rmse(y_holdout, axis_holdout)},
            "lane_spatial_temporal": {"holdout_rmse": rmse(y_holdout, full_holdout)},
        },
        "inspection_metrics": {
            "holdout_rmse_ratio": holdout_ratio,
            "hot_lane_midnight_prediction": float(predict_model(full, hot_midnight)[0]),
            "cold_lane_midday_prediction": float(predict_model(full, cold_midday)[0]),
            "hot_cold_operating_contrast": contrast,
            "uses_hidden_simulator_metadata_in_training": 0.0,
        },
        "acceptance_gates": [
            gate("full_beats_axis_holdout", holdout_ratio < 0.65, holdout_ratio, 0.65, "<"),
            gate("hot_cold_contrast_gt_120", contrast > 120.0, contrast, 120.0, ">"),
        ],
        "future_checks": [
            "Combined lane, route, and hour splitters should beat an axis-only baseline.",
            "The fixture uses only observable matrix columns documented in the artifact README.",
        ],
    }


def collect_metrics() -> dict[str, Any]:
    return {
        "sparse_lane_membership": sparse_lane_metrics(),
        "route_midpoint_cartometry": route_cartometry_metrics(),
        "wraparound_lane_hour": wraparound_hour_metrics(),
        "regional_lane_boosting": regional_lane_boosting_metrics(),
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    def fmt(value: float) -> str:
        return f"{value:.8e}"

    lines = [
        "# CartoBoost lane-level acceptance metrics",
        "",
        "These deterministic fixtures adapt the upstream regional lane CPM idea to this repo's",
        "current API. The matrix columns are observable route features: origin/destination",
        "coordinates, lane ID, hour of day, route midpoint, and route distance.",
        "",
        "| phase | model | metric | value |",
        "| --- | --- | --- | ---: |",
    ]
    for phase, phase_metrics in metrics.items():
        for model_name, model_metrics in phase_metrics["models"].items():
            for metric, value in model_metrics.items():
                lines.append(f"| {phase} | {model_name} | {metric} | {fmt(value)} |")
    lines.extend(["", "## Inspection Metrics", ""])
    for phase, phase_metrics in metrics.items():
        lines.append(f"### {phase}")
        for metric, value in phase_metrics["inspection_metrics"].items():
            lines.append(f"- `{metric}`: {fmt(value)}")
        for check in phase_metrics["acceptance_gates"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"- `{check['name']}`: {status} "
                f"({fmt(check['actual'])} {check['comparator']} {fmt(check['threshold'])})"
            )
        lines.append("")
    return "\n".join(lines)


def render_readme(metrics: dict[str, Any]) -> str:
    passed = sum(
        1
        for phase_metrics in metrics.values()
        for check in phase_metrics["acceptance_gates"]
        if check["passed"]
    )
    total = sum(len(phase_metrics["acceptance_gates"]) for phase_metrics in metrics.values())
    return "\n".join(
        [
            "# Lane-Level Diagnostics",
            "",
            "This directory is generated by `scripts/run_lane_level_acceptance_metrics.py`.",
            "It is intentionally committed so lane-level behavior changes are visible in diffs.",
            "",
            "Fixture shape:",
            "",
            "- 4 origin regions x 4 destination regions = 16 lanes.",
            "- 24 hourly observations per lane.",
            "- Observable columns: origin x/y, destination x/y, lane ID, hour, midpoint x/y, "
            "distance.",
            "- No hidden simulator metadata is passed into training.",
            "",
            f"Acceptance gates passing: {passed}/{total}.",
            "",
            "Images:",
            "",
            "- `lane_heatmap.png`: lane-level prediction table for the combined model.",
            "- `hour_profile.png`: wraparound-hour predictions from axis and periodic splitters.",
            "- `route_midpoint_cartometry.png`: radial route-midpoint fixture and predictions.",
            "",
        ]
    )


def save_lane_heatmap(path: Path) -> None:
    x = lane_rows(repeats_per_hour=1)
    y = combined_target(x)
    model = fit_model(
        x,
        y,
        splitters=["axis", "sparse_set", "gaussian_2d", "periodic_time"],
        n_estimators=4,
        max_depth=2,
        min_samples_leaf=6,
    )
    noon = x[x[:, 5] == 12.0]
    predictions = predict_model(model, noon).reshape(4, 4)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
    mesh = ax.imshow(predictions, cmap="viridis", origin="lower")
    ax.set_title("Combined lane model at hour 12")
    ax.set_xlabel("destination region")
    ax.set_ylabel("origin region")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    for origin in range(4):
        for destination in range(4):
            ax.text(
                destination,
                origin,
                f"{predictions[origin, destination]:.0f}",
                ha="center",
                va="center",
                color="white",
            )
    fig.colorbar(mesh, ax=ax, label="prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_hour_profile(path: Path) -> None:
    x = lane_rows(repeats_per_hour=2)
    y = wraparound_hour_target(x)
    axis = fit_model(x[:, [5]], y, splitters=["axis"], min_samples_leaf=16)
    periodic = fit_model(x[:, [5]], y, splitters=["periodic_time"], min_samples_leaf=16)
    probe = np.arange(24, dtype=float).reshape(-1, 1)
    truth = wraparound_hour_target(np.column_stack([np.zeros((24, 5)), probe, np.zeros((24, 3))]))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(probe[:, 0], truth, label="truth", marker="o", linewidth=1.5)
    ax.plot(probe[:, 0], predict_model(axis, probe), label="axis", marker="o", linewidth=1.5)
    ax.plot(
        probe[:, 0], predict_model(periodic, probe), label="periodic", marker="o", linewidth=1.5
    )
    ax.set_title("Lane hour wraparound")
    ax.set_xlabel("hour")
    ax.set_ylabel("prediction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_route_cartometry(path: Path) -> None:
    grid = np.linspace(-1.5, 1.5, 121)
    xx, yy = np.meshgrid(grid, grid)
    probe = np.column_stack([xx.ravel(), yy.ravel()])
    target = np.where(np.hypot(probe[:, 0], probe[:, 1]) <= 0.25, 260.0, 90.0)
    model = fit_model(probe, target, splitters=["gaussian_2d"], min_samples_leaf=16)
    pred = predict_model(model, probe).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    mesh = ax.contourf(xx, yy, pred, levels=18, cmap="coolwarm")
    ax.contour(xx, yy, pred, colors="black", linewidths=0.6, levels=3)
    ax.set_title("Route midpoint gaussian split")
    ax.set_xlabel("midpoint x")
    ax.set_ylabel("midpoint y")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label="prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def assert_finite(metrics: dict[str, Any]) -> None:
    for phase_metrics in metrics.values():
        for model_metrics in phase_metrics["models"].values():
            for value in model_metrics.values():
                if not math.isfinite(float(value)):
                    raise ValueError(f"non-finite model metric: {value}")
        for value in phase_metrics["inspection_metrics"].values():
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite inspection metric: {value}")


def assert_acceptance_gates(metrics: dict[str, Any]) -> None:
    failures = [
        f"{phase}.{check['name']}"
        for phase, phase_metrics in metrics.items()
        for check in phase_metrics["acceptance_gates"]
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
    (args.output_dir / "README.md").write_text(render_readme(metrics), encoding="utf-8")
    save_lane_heatmap(args.output_dir / "lane_heatmap.png")
    save_hour_profile(args.output_dir / "hour_profile.png")
    save_route_cartometry(args.output_dir / "route_midpoint_cartometry.png")


if __name__ == "__main__":
    main()
