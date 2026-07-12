#!/usr/bin/env python3
"""Generate deterministic PNG proof images for CartoBoost splitter behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import CartoBoostRegressor  # noqa: E402

OUT = ROOT / "docs" / "assets"
PHASE_OUT = OUT / "splitter_tests"


def train_diagonal() -> CartoBoostRegressor:
    points = []
    target = []
    for x in np.linspace(-3.0, 3.0, 25):
        for y in np.linspace(-3.0, 3.0, 25):
            points.append([float(x), float(y)])
            target.append(10.0 if x + y > 0.0 else -10.0)
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=2,
        split_policy="structured",
    )
    model.fit(points, target)
    return model


def train_radial() -> CartoBoostRegressor:
    points = []
    target = []
    for x in np.linspace(-3.0, 3.0, 31):
        for y in np.linspace(-3.0, 3.0, 31):
            points.append([float(x), float(y)])
            target.append(10.0 if np.hypot(x, y) <= 1.5 else -10.0)
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=2,
        split_policy="structured",
    )
    model.fit(points, target)
    return model


def plot_segmentation(model: CartoBoostRegressor, path: Path, title: str) -> None:
    grid = np.linspace(-3.0, 3.0, 180)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    pred = model.predict(points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    mesh = ax.contourf(xx, yy, pred, levels=24, cmap="coolwarm", alpha=0.92)
    ax.contour(xx, yy, pred, colors="black", linewidths=0.7, levels=3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label="prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    splitters: list[str],
    learning_rate: float = 1.0,
    max_depth: int = 1,
    min_samples_leaf: int = 1,
    leaf_predictor: str = "constant",
    linear_leaf_features: list[str] | None = None,
    fuzzy: bool = False,
    fuzzy_bandwidth: float = 0.0,
) -> CartoBoostRegressor:
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_gain=0.0,
        split_policy="structured",
        leaf_predictor=leaf_predictor,
        linear_leaf_features=linear_leaf_features,
        fuzzy=fuzzy,
        fuzzy_bandwidth=fuzzy_bandwidth,
        l2_regularization=0.0,
    )
    model.fit(x, y)
    return model


def save_line_plot(
    path: Path,
    title: str,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    xlabel: str,
    ylabel: str = "prediction",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    for label, values in series:
        ax.plot(x, values, marker="o", markersize=2.5, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bar_plot(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    colors = ["#4c78a8" if value < 0 else "#f58518" for value in values]
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("prediction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(
    path: Path,
    title: str,
    points: np.ndarray,
    values: np.ndarray,
    *,
    grid_size: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    image = values.reshape(grid_size, grid_size)
    mesh = ax.imshow(
        image,
        origin="lower",
        extent=[points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()],
        cmap="coolwarm",
        aspect="equal",
    )
    ax.contour(
        np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size),
        np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size),
        image,
        colors="black",
        linewidths=0.6,
        levels=3,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(mesh, ax=ax, label="prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def phase_axis_threshold() -> None:
    x = np.linspace(-3.0, 3.0, 49).reshape(-1, 1)
    y = np.where(x[:, 0] > 0.0, 10.0, -10.0)
    model = fit_model(x, y, splitters=["axis"], min_samples_leaf=2)
    probe = np.linspace(-3.0, 3.0, 121).reshape(-1, 1)
    save_line_plot(
        PHASE_OUT / "phase_1_axis_threshold.png",
        "Axis threshold stump",
        probe[:, 0],
        [("truth", np.where(probe[:, 0] > 0.0, 10.0, -10.0)), ("prediction", model.predict(probe))],
        xlabel="x",
    )


def phase_diagonal() -> None:
    values = np.linspace(-3.0, 3.0, 91)
    xx, yy = np.meshgrid(values, values)
    train = np.column_stack([xx.ravel(), yy.ravel()])
    target = np.where(train[:, 0] + train[:, 1] > 0.0, 10.0, -10.0)
    model = fit_model(train, target, splitters=["diagonal_2d"], min_samples_leaf=4)
    save_heatmap(
        PHASE_OUT / "phase_2_diagonal_2d.png",
        "Diagonal 2D segmentation",
        train,
        model.predict(train),
        grid_size=len(values),
    )


def phase_gaussian() -> None:
    values = np.linspace(-3.0, 3.0, 91)
    xx, yy = np.meshgrid(values, values)
    train = np.column_stack([xx.ravel(), yy.ravel()])
    target = np.where(np.hypot(train[:, 0], train[:, 1]) <= 1.5, 10.0, -10.0)
    model = fit_model(train, target, splitters=["gaussian_2d"], min_samples_leaf=4)
    save_heatmap(
        PHASE_OUT / "phase_3_gaussian_2d.png",
        "Gaussian radial segmentation",
        train,
        model.predict(train),
        grid_size=len(values),
    )


def phase_periodic() -> None:
    hours = np.array([[float(hour)] for hour in range(24) for _ in range(3)])
    target = np.array([15.0 if hour[0] >= 22.0 or hour[0] <= 2.0 else -5.0 for hour in hours])
    axis = fit_model(hours, target, splitters=["axis"], min_samples_leaf=3)
    periodic = fit_model(hours, target, splitters=["periodic_time"], min_samples_leaf=3)
    probe = np.linspace(0.0, 24.0, 145).reshape(-1, 1)
    truth = np.where((probe[:, 0] >= 22.0) | (probe[:, 0] <= 2.0), 15.0, -5.0)
    save_line_plot(
        PHASE_OUT / "phase_4_periodic_wraparound.png",
        "Periodic wraparound split",
        probe[:, 0],
        [("truth", truth), ("axis", axis.predict(probe)), ("periodic", periodic.predict(probe))],
        xlabel="hour of day",
    )


def phase_fuzzy() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    hard = fit_model(x, y, splitters=["axis"])
    fuzzy = fit_model(x, y, splitters=["axis"], fuzzy=True, fuzzy_bandwidth=1.0)
    probe = np.linspace(0.5, 2.5, 161).reshape(-1, 1)
    save_line_plot(
        PHASE_OUT / "phase_5_fuzzy_boundary.png",
        "Fuzzy split boundary smoothing",
        probe[:, 0],
        [("hard", hard.predict(probe)), ("fuzzy", fuzzy.predict(probe))],
        xlabel="x",
    )


def phase_linear_leaf() -> None:
    x = np.linspace(0.0, 5.0, 30).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 3.0
    constant = fit_model(x, y, splitters=["axis"], min_samples_leaf=16)
    linear = fit_model(
        x,
        y,
        splitters=["axis"],
        min_samples_leaf=16,
        leaf_predictor="linear",
        linear_leaf_features=["0"],
    )
    probe = np.linspace(0.0, 5.0, 121).reshape(-1, 1)
    save_line_plot(
        PHASE_OUT / "phase_6_linear_leaf.png",
        "Linear leaf predictor",
        probe[:, 0],
        [
            ("truth", 2.0 * probe[:, 0] + 3.0),
            ("constant leaf", constant.predict(probe)),
            ("linear leaf", linear.predict(probe)),
        ],
        xlabel="distance",
    )


def phase_sparse_set() -> None:
    x = np.array([[7.0], [7.0], [3.0], [4.0], [9.0], [9.0], [8.0], [8.0]])
    y = np.array([30.0, 30.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    model = fit_model(x, y, splitters=["sparse_set"], min_samples_leaf=2)
    labels = ["cell 3", "cell 4", "cell 7", "cell 8", "cell 9", "cell 11"]
    probe = np.array([[3.0], [4.0], [7.0], [8.0], [9.0], [11.0]])
    save_bar_plot(
        PHASE_OUT / "phase_7_sparse_set.png",
        "Sparse set contains-any routing",
        labels,
        [float(value) for value in model.predict(probe)],
    )


def phase_learning_rate() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    full = fit_model(x, y, splitters=["axis"], learning_rate=1.0)
    shrink = fit_model(x, y, splitters=["axis"], learning_rate=0.25)
    probe = np.linspace(0.0, 3.0, 121).reshape(-1, 1)
    save_line_plot(
        PHASE_OUT / "phase_8_learning_rate_shrinkage.png",
        "Gradient update scaled by learning_rate",
        probe[:, 0],
        [("learning_rate=1.0", full.predict(probe)), ("learning_rate=0.25", shrink.predict(probe))],
        xlabel="x",
    )


def phase_fuzzy_periodic() -> None:
    hours = np.array([[float(hour)] for hour in range(24) for _ in range(4)])
    target = np.array([20.0 if hour[0] >= 22.0 or hour[0] <= 2.0 else 0.0 for hour in hours])
    hard = fit_model(hours, target, splitters=["periodic_time"], min_samples_leaf=4)
    fuzzy = fit_model(
        hours,
        target,
        splitters=["periodic_time"],
        min_samples_leaf=4,
        fuzzy=True,
        fuzzy_bandwidth=2.0,
    )
    probe = np.linspace(18.0, 24.0, 181).reshape(-1, 1)
    truth = np.where((probe[:, 0] >= 22.0) | (probe[:, 0] <= 2.0), 20.0, 0.0)
    save_line_plot(
        PHASE_OUT / "phase_9_fuzzy_periodic_wraparound.png",
        "Fuzzy periodic wraparound smoothing",
        probe[:, 0],
        [
            ("truth", truth),
            ("hard periodic", hard.predict(probe)),
            ("fuzzy periodic", fuzzy.predict(probe)),
        ],
        xlabel="hour of day",
    )


def generate_phase_proofs() -> None:
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    phase_axis_threshold()
    phase_diagonal()
    phase_gaussian()
    phase_periodic()
    phase_fuzzy()
    phase_linear_leaf()
    phase_sparse_set()
    phase_learning_rate()
    phase_fuzzy_periodic()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_segmentation(
        train_diagonal(),
        OUT / "segmentation_diagonal_2d.png",
        "CartoBoost diagonal_2d learned segmentation",
    )
    plot_segmentation(
        train_radial(),
        OUT / "segmentation_gaussian_2d.png",
        "CartoBoost gaussian_2d learned segmentation",
    )
    generate_phase_proofs()


if __name__ == "__main__":
    main()
