from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cartoboost as cb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

Zone = tuple[str, float, float, float]


def example_pickup_zones() -> list[Zone]:
    """Small deterministic taxi-zone geometry for docs and smoke tests."""

    return [
        ("PULocationID_132", 0.0, 0.2, 126.0),
        ("PULocationID_138", 0.7, 0.0, 113.0),
        ("PULocationID_161", 2.2, 1.5, 86.0),
        ("PULocationID_236", 2.8, 2.4, 72.0),
        ("PULocationID_239", 1.4, 2.8, 68.0),
        ("PULocationID_230", 1.7, 1.8, 91.0),
        ("PULocationID_79", 2.4, 0.9, 97.0),
        ("PULocationID_261", 0.6, 2.1, 58.0),
    ]


def observations(zones: list[Zone]) -> list[tuple[float, float, float]]:
    return [(x, y, pickup_count) for _, x, y, pickup_count in zones]


def grid_targets(step: float = 0.08) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    xs = [round(value * step, 4) for value in range(0, int(3.0 / step) + 1)]
    ys = [round(value * step, 4) for value in range(0, int(3.0 / step) + 1)]
    return [(x, y) for y in ys for x in xs], xs, ys


def fit_config(obs: list[tuple[float, float, float]]) -> dict[str, Any]:
    fit = cb.fit_ordinary_kriging_variogram(
        obs,
        variogram_models=["exponential", "spherical", "gaussian"],
        range_candidates=[0.5, 0.9, 1.3, 1.8, 2.4, 3.0],
        nugget_candidates=[0.0, 4.0, 9.0],
        sill_candidates=[120.0, 240.0, 360.0, 520.0],
        bin_count=6,
    )
    return dict(fit["config"])


def predict_grid(
    obs: list[tuple[float, float, float]], config: dict[str, Any]
) -> tuple[list[tuple[float, float]], list[float], list[float], list[float]]:
    targets, xs, ys = grid_targets()
    predictions = cb.ordinary_kriging_predict(
        obs,
        targets,
        range=config["range"],
        nugget=config["nugget"],
        sill=config["sill"],
        variogram_model=config["variogram_model"],
        detailed=True,
    )
    return targets, xs, ys, [row["mean"] for row in predictions]


def write_surface_plot(
    zones: list[Zone],
    xs: list[float],
    ys: list[float],
    means: list[float],
    output: Path,
) -> None:
    rows = [means[row_idx * len(xs) : (row_idx + 1) * len(xs)] for row_idx in range(len(ys))]
    fig, axis = plt.subplots(figsize=(8.4, 6.4))
    image = axis.imshow(
        rows,
        origin="lower",
        extent=[min(xs), max(xs), min(ys), max(ys)],
        cmap="viridis",
        aspect="equal",
    )
    for zone_id, x, y, pickup_count in zones:
        axis.scatter(x, y, s=90, color="#f8fafc", edgecolor="#111827", linewidth=1.1)
        axis.text(x + 0.04, y + 0.04, zone_id.removeprefix("PULocationID_"), fontsize=8)
        axis.text(x + 0.04, y - 0.09, f"{pickup_count:.0f}", fontsize=7, color="#111827")
    axis.set_title("Example NYC Taxi Pickup Demand Kriging Surface")
    axis.set_xlabel("example east-west zone coordinate")
    axis.set_ylabel("example north-south zone coordinate")
    axis.set_xlim(-0.16, max(xs))
    axis.set_ylim(-0.12, max(ys))
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("estimated pickups")
    axis.grid(color="white", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_variogram_plot(
    obs: list[tuple[float, float, float]],
    config: dict[str, Any],
    output: Path,
) -> None:
    bins = cb.empirical_variogram(obs, bin_count=6)
    distances = [bin_row["mean_distance"] for bin_row in bins]
    semivariances = [bin_row["semivariance"] for bin_row in bins]
    counts = [bin_row["pair_count"] for bin_row in bins]
    max_distance = max(distances) if distances else 1.0
    fitted_x = [max_distance * idx / 120.0 for idx in range(121)]
    fitted_y = [
        theoretical_semivariogram(
            distance,
            range_=config["range"],
            nugget=config["nugget"],
            sill=config["sill"],
            variogram_model=config["variogram_model"],
        )
        for distance in fitted_x
    ]

    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    sizes = [35 + 18 * count for count in counts]
    axis.scatter(distances, semivariances, s=sizes, color="#2563eb", alpha=0.82)
    axis.plot(fitted_x, fitted_y, color="#dc2626", linewidth=2.2)
    for distance, semivariance, count in zip(distances, semivariances, counts, strict=True):
        axis.text(distance, semivariance, f" n={count}", fontsize=8, va="center")
    axis.set_title(f"Empirical Variogram With Fitted {config['variogram_model'].title()} Model")
    axis.set_xlabel("lag distance")
    axis.set_ylabel("semivariance")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_loo_plot(
    obs: list[tuple[float, float, float]],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    diagnostics = cb.ordinary_kriging_leave_one_out_diagnostics(
        obs,
        range=config["range"],
        nugget=config["nugget"],
        sill=config["sill"],
        variogram_model=config["variogram_model"],
    )
    predictions = diagnostics["predictions"]
    actual = [value for _, _, value in obs]
    predicted = [row["mean"] for row in predictions]
    errors = [
        actual_value - predicted_value
        for actual_value, predicted_value in zip(actual, predicted, strict=True)
    ]
    std_errors = [
        error / math.sqrt(max(row["variance"], 1.0e-12))
        for error, row in zip(errors, predictions, strict=True)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    axes[0].scatter(actual, predicted, color="#2563eb", s=70)
    lo = min(min(actual), min(predicted)) - 4
    hi = max(max(actual), max(predicted)) + 4
    axes[0].plot([lo, hi], [lo, hi], color="#111827", linestyle="--", linewidth=1)
    axes[0].set_title("Leave-One-Out: Actual vs Predicted")
    axes[0].set_xlabel("actual pickups")
    axes[0].set_ylabel("predicted pickups")
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0.0, color="#111827", linewidth=1)
    axes[1].axhline(1.96, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].axhline(-1.96, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].bar(range(len(std_errors)), std_errors, color="#0f766e")
    axes[1].set_title("Standardized Leave-One-Out Errors")
    axes[1].set_xlabel("held-out zone index")
    axes[1].set_ylabel("standardized error")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return diagnostics


def theoretical_semivariogram(
    distance: float, *, range_: float, nugget: float, sill: float, variogram_model: str
) -> float:
    ratio = distance / range_
    if variogram_model == "gaussian":
        correlation = math.exp(-(ratio * ratio))
    elif variogram_model == "spherical":
        correlation = 0.0 if ratio >= 1.0 else 1.0 - 1.5 * ratio + 0.5 * ratio**3
    elif variogram_model == "linear":
        correlation = max(1.0 - ratio, 0.0)
    else:
        correlation = math.exp(-ratio)
    return nugget + sill * (1.0 - correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/kriging_examples"),
        help="Directory for generated PNG and JSON artifacts.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zones = example_pickup_zones()
    obs = observations(zones)
    config = fit_config(obs)
    _, xs, ys, means = predict_grid(obs, config)

    surface_path = args.output_dir / "kriging_surface.png"
    variogram_path = args.output_dir / "kriging_variogram_fit.png"
    loo_path = args.output_dir / "kriging_leave_one_out.png"
    summary_path = args.output_dir / "kriging_example_summary.json"

    write_surface_plot(zones, xs, ys, means, surface_path)
    write_variogram_plot(obs, config, variogram_path)
    diagnostics = write_loo_plot(obs, config, loo_path)

    payload = {
        "task": "example_taxi_zone_kriging",
        "zones": len(zones),
        "selected_config": config,
        "diagnostics": diagnostics["diagnostics"],
        "assets": {
            "surface": str(surface_path),
            "variogram": str(variogram_path),
            "leave_one_out": str(loo_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
