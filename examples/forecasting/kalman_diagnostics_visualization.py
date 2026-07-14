from __future__ import annotations

import json
from pathlib import Path

import cartoboost as cb
import matplotlib

matplotlib.use("Agg")


def main() -> None:
    import matplotlib.pyplot as plt

    pickup_hours = list(range(18))
    airport_pickups = [
        74.0,
        76.0,
        79.0,
        78.0,
        83.0,
        86.0,
        84.0,
        90.0,
        92.0,
        91.0,
        97.0,
        101.0,
        99.0,
        104.0,
        108.0,
        109.0,
        114.0,
        117.0,
    ]

    state = cb.kalman_filter(
        airport_pickups,
        level_process_variance=0.08,
        trend_process_variance=0.01,
        observation_variance=2.0,
        horizon=6,
    )

    estimate_hours = [row["step"] for row in state["estimates"]]
    filtered = [row["level"] for row in state["estimates"]]
    fitted = [row["fitted"] for row in state["estimates"]]
    smoothed_hours = [row["step"] for row in state["smoothed_states"]]
    smoothed = [row["level"] for row in state["smoothed_states"]]
    future_hours = [pickup_hours[-1] + row["step"] for row in state["forecast_distribution"]]
    forecast_mean = [row["mean"] for row in state["forecast_distribution"]]
    forecast_lower = [row["lower"] for row in state["forecast_distribution"]]
    forecast_upper = [row["upper"] for row in state["forecast_distribution"]]
    standardized = [row["standardized_innovation"] for row in state["estimates"]]

    output_dir = Path("target/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "kalman_diagnostics_visualization.png"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    axes[0].plot(pickup_hours, airport_pickups, marker="o", label="Observed pickups")
    axes[0].plot(estimate_hours, fitted, linestyle="--", label="One-step fitted")
    axes[0].plot(estimate_hours, filtered, label="Filtered level")
    axes[0].plot(smoothed_hours, smoothed, label="Smoothed level")
    axes[0].plot(future_hours, forecast_mean, marker="o", label="Forecast mean")
    axes[0].fill_between(
        future_hours, forecast_lower, forecast_upper, alpha=0.18, label="95% interval"
    )
    axes[0].set_title("Airport Pickup Demand: Kalman State And Forecast")
    axes[0].set_ylabel("pickup count")
    axes[0].legend(loc="upper left")

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(1.96, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(-1.96, color="gray", linestyle="--", linewidth=1)
    axes[1].bar(estimate_hours, standardized)
    axes[1].set_title("Standardized Innovations")
    axes[1].set_xlabel("hour index")
    axes[1].set_ylabel("z-score")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(
        json.dumps(
            {
                "plot": str(plot_path),
                "final_level": state["final_state"]["level"],
                "final_trend": state["final_state"]["trend"],
                "rmse": state["diagnostics"]["rmse"],
                "mae": state["diagnostics"]["mae"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
