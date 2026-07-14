from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from cartoboost.forecasting import ForecastFrame, OptimizedThetaForecaster, ThetaForecaster


def synthetic_zone_pickups(hours: int) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    zones = [
        ("PU132_JFK", 96.0, 0.33, 18.0),
        ("PU236_UES", 72.0, 0.10, 7.0),
    ]
    rows: list[dict[str, Any]] = []
    for zone_id, baseline, trend, airport_ramp in zones:
        for hour in range(hours):
            pickup_hour = start + timedelta(hours=hour)
            hour_of_day = pickup_hour.hour
            morning = 9.0 if 6 <= hour_of_day <= 9 else 0.0
            evening = 11.0 if 16 <= hour_of_day <= 20 else 0.0
            overnight = -15.0 if hour_of_day <= 4 else 0.0
            daily_wave = 4.5 * math.sin((hour_of_day - 7) / 24.0 * math.tau)
            local_variation = 2.5 * math.sin(hour / 5.0)
            pickup_count = (
                baseline
                + trend * hour
                + airport_ramp * hour / max(hours - 1, 1)
                + morning
                + evening
                + overnight
                + daily_wave
                + local_variation
            )
            rows.append(
                {
                    "PULocationID": zone_id,
                    "pickup_hour": pickup_hour,
                    "pickup_count": pickup_count,
                }
            )
    return pd.DataFrame(rows)


def forecast_panel(
    table: pd.DataFrame,
    model: ThetaForecaster | OptimizedThetaForecaster,
    train_hours: int,
    horizon: int,
) -> pd.DataFrame:
    train = table.groupby("PULocationID", sort=False).head(train_hours).reset_index(drop=True)
    frame = ForecastFrame.from_pandas(
        train,
        timestamp_col="pickup_hour",
        target_col="pickup_count",
        series_id_col="PULocationID",
        freq="h",
    )
    model.fit(frame)
    return pd.DataFrame(
        [
            {
                "PULocationID": series_id,
                "pickup_hour": pd.Timestamp(timestamp),
                "horizon": horizon_step,
                "model": model_name,
                "prediction": prediction,
            }
            for series_id, timestamp, horizon_step, model_name, prediction in model.predict(
                horizon
            ).predictions()
        ]
    )


def metric_summary(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    joined = actual.merge(
        forecast,
        on=["PULocationID", "pickup_hour"],
        validate="one_to_one",
    )
    errors = [row.prediction - row.pickup_count for row in joined.itertuples()]
    return {
        "mae": mean(abs(error) for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
    }


def grid_holdout_scores(
    table: pd.DataFrame,
    theta_grid: tuple[float, ...],
    alpha_grid: tuple[float, ...],
    train_hours: int,
    horizon: int,
) -> list[dict[str, float]]:
    actual = holdout_actual(table, train_hours, horizon)
    scores: list[dict[str, float]] = []
    for theta in theta_grid:
        for alpha in alpha_grid:
            forecast = forecast_panel(
                table,
                ThetaForecaster(
                    theta=theta,
                    alpha=alpha,
                    seasonality="additive",
                    season_length=24,
                ),
                train_hours=train_hours,
                horizon=horizon,
            )
            metrics = metric_summary(actual, forecast)
            scores.append(
                {
                    "theta": theta,
                    "alpha": alpha,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                }
            )
    return sorted(scores, key=lambda row: (row["rmse"], row["theta"], row["alpha"]))


def holdout_actual(table: pd.DataFrame, train_hours: int, horizon: int) -> pd.DataFrame:
    actual = table.groupby("PULocationID", sort=False).nth(
        list(range(train_hours, train_hours + horizon))
    )
    return actual.reset_index()


def write_plot(
    table: pd.DataFrame,
    manual_forecast: pd.DataFrame,
    optimized_forecast: pd.DataFrame,
    train_hours: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zone_id = "PU132_JFK"
    observed = table[table["PULocationID"] == zone_id]
    manual = manual_forecast[manual_forecast["PULocationID"] == zone_id]
    optimized = optimized_forecast[optimized_forecast["PULocationID"] == zone_id]

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        observed["pickup_hour"],
        observed["pickup_count"],
        label="observed pickups",
        color="#172554",
    )
    axis.plot(
        manual["pickup_hour"],
        manual["prediction"],
        label="Theta theta=2.0 alpha=0.25",
        color="#2563eb",
        linewidth=2,
    )
    axis.plot(
        optimized["pickup_hour"],
        optimized["prediction"],
        label="OptimizedTheta grid",
        color="#b91c1c",
        linewidth=2,
    )
    split_time = observed["pickup_hour"].iloc[train_hours - 1]
    axis.axvline(split_time, color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("JFK Pickup Demand: Theta Forecast Comparison")
    axis.set_xlabel("pickup hour")
    axis.set_ylabel("pickup count")
    axis.legend()
    axis.grid(alpha=0.22)
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=96)
    parser.add_argument("--train-hours", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("target/examples/theta_optimized_visualization.png"),
    )
    args = parser.parse_args()

    if args.train_hours < 48:
        raise ValueError("--train-hours must be at least 48 for daily additive theta seasonality")
    if args.hours <= args.train_hours:
        raise ValueError("--hours must be greater than --train-hours")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.train_hours + args.horizon > args.hours:
        raise ValueError("--train-hours + --horizon must be <= --hours")

    table = synthetic_zone_pickups(args.hours)
    actual = holdout_actual(table, args.train_hours, args.horizon)
    theta_grid = (1.0, 1.5, 2.0, 2.5, 3.0)
    alpha_grid = (0.15, 0.25, 0.45, 0.65)

    manual_forecast = forecast_panel(
        table,
        ThetaForecaster(theta=2.0, alpha=0.25, seasonality="additive", season_length=24),
        train_hours=args.train_hours,
        horizon=args.horizon,
    )
    optimized_forecast = forecast_panel(
        table,
        OptimizedThetaForecaster(
            theta_grid=theta_grid,
            alpha_grid=alpha_grid,
            seasonality="additive",
            season_length=24,
        ),
        train_hours=args.train_hours,
        horizon=args.horizon,
    )
    scores = grid_holdout_scores(table, theta_grid, alpha_grid, args.train_hours, args.horizon)

    write_plot(table, manual_forecast, optimized_forecast, args.train_hours, args.output)

    payload = {
        "task": "example_taxi_zone_theta_forecast",
        "rows": int(len(table)),
        "zones": int(table["PULocationID"].nunique()),
        "train_hours": int(args.train_hours),
        "horizon": int(args.horizon),
        "manual_theta": metric_summary(actual, manual_forecast),
        "optimized_theta": metric_summary(actual, optimized_forecast),
        "best_holdout_grid_candidate": scores[0],
        "plot": str(args.output),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
