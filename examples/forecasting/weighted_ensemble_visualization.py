from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from cartoboost.preview.forecasting import (
    ForecastFrame,
    KalmanForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)

SEASON_LENGTH = 24


def synthetic_lane_demand(hours: int) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    lanes = [
        ("PU132->DO138", 96.0, 20.0, 0.18),
        ("PU79->DO230", 66.0, 9.0, 0.07),
    ]
    rows: list[dict[str, Any]] = []
    for lane_id, baseline, rush_amp, trend in lanes:
        for hour in range(hours):
            pickup_hour = start + timedelta(hours=hour)
            hour_of_day = pickup_hour.hour
            morning = rush_amp if 6 <= hour_of_day <= 9 else 0.0
            evening = 0.55 * rush_amp if 16 <= hour_of_day <= 19 else 0.0
            overnight = -8.0 if hour_of_day <= 4 else 0.0
            local_wave = 3.5 * math.sin(hour / 7.0)
            pickup_count = baseline + morning + evening + overnight + local_wave + trend * hour
            rows.append(
                {
                    "lane_id": lane_id,
                    "pickup_hour": pickup_hour,
                    "pickup_count": pickup_count,
                }
            )
    return pd.DataFrame(rows)


def fit_forecast(
    table: pd.DataFrame,
    model: Any,
    train_hours: int,
    horizon: int,
) -> pd.DataFrame:
    train = table.groupby("lane_id", sort=False).head(train_hours).reset_index(drop=True)
    frame = ForecastFrame.from_pandas(
        train,
        timestamp_col="pickup_hour",
        target_col="pickup_count",
        series_id_col="lane_id",
        freq="h",
    )
    model.fit(frame)
    forecast = pd.DataFrame(
        model.predict(horizon).predictions(),
        columns=["lane_id", "pickup_hour", "horizon", "model", "prediction"],
    )
    forecast["pickup_hour"] = pd.to_datetime(forecast["pickup_hour"])
    return forecast


def actual_window(table: pd.DataFrame, train_hours: int, horizon: int) -> pd.DataFrame:
    actual = table.groupby("lane_id", sort=False).nth(
        list(range(train_hours, train_hours + horizon))
    )
    return actual.reset_index()


def metric_summary(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    joined = actual.merge(forecast, on=["lane_id", "pickup_hour"], validate="one_to_one")
    errors = [row.prediction - row.pickup_count for row in joined.itertuples()]
    return {
        "mae": mean(abs(error) for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
    }


def write_plot(
    table: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    train_hours: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lane_id = "PU132->DO138"
    observed = table[table["lane_id"] == lane_id]

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        observed["pickup_hour"],
        observed["pickup_count"],
        color="#1f2937",
        label="observed pickups",
    )
    colors = {
        "seasonal": "#059669",
        "theta": "#7c3aed",
        "kalman": "#ea580c",
        "ensemble": "#2563eb",
    }
    for name, forecast in forecasts.items():
        lane_forecast = forecast[forecast["lane_id"] == lane_id]
        axis.plot(
            lane_forecast["pickup_hour"],
            lane_forecast["prediction"],
            marker="o",
            linewidth=2 if name == "ensemble" else 1.5,
            color=colors[name],
            label=name,
        )
    axis.axvline(
        observed["pickup_hour"].iloc[train_hours - 1],
        color="#6b7280",
        linestyle="--",
        linewidth=1,
    )
    axis.set_title("Airport Lane Pickup Demand: Weighted Forecast Ensemble")
    axis.set_xlabel("pickup hour")
    axis.set_ylabel("pickup count")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=96)
    parser.add_argument("--train-hours", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("target/examples/weighted_ensemble.png"),
    )
    args = parser.parse_args()

    if args.hours <= args.train_hours:
        raise ValueError("--hours must be greater than --train-hours")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.train_hours + args.horizon > args.hours:
        raise ValueError("--train-hours + --horizon must be <= --hours")
    if args.train_hours <= SEASON_LENGTH:
        raise ValueError(f"--train-hours must be greater than {SEASON_LENGTH}")

    table = synthetic_lane_demand(args.hours)
    actual = actual_window(table, args.train_hours, args.horizon)
    weights = {"seasonal": 0.55, "theta": 0.30, "kalman": 0.15}

    forecasts = {
        "seasonal": fit_forecast(
            table,
            SeasonalNaiveForecaster(season_length=SEASON_LENGTH),
            args.train_hours,
            args.horizon,
        ),
        "theta": fit_forecast(
            table,
            ThetaForecaster(theta=2.0, alpha=0.25),
            args.train_hours,
            args.horizon,
        ),
        "kalman": fit_forecast(
            table,
            KalmanForecaster(
                level_process_variance=0.08,
                trend_process_variance=0.01,
                observation_variance=2.0,
            ),
            args.train_hours,
            args.horizon,
        ),
        "ensemble": fit_forecast(
            table,
            WeightedEnsembleForecaster(
                models={
                    "seasonal": SeasonalNaiveForecaster(season_length=SEASON_LENGTH),
                    "theta": ThetaForecaster(theta=2.0, alpha=0.25),
                    "kalman": KalmanForecaster(
                        level_process_variance=0.08,
                        trend_process_variance=0.01,
                        observation_variance=2.0,
                    ),
                },
                weights=weights,
                metadata={"purpose": "airport lane pickup demand comparison"},
            ),
            args.train_hours,
            args.horizon,
        ),
    }

    if args.output is not None:
        write_plot(table, forecasts, args.train_hours, args.output)

    payload = {
        "task": "example_taxi_lane_weighted_ensemble_forecast",
        "rows": int(len(table)),
        "lanes": int(table["lane_id"].nunique()),
        "train_hours": int(args.train_hours),
        "horizon": int(args.horizon),
        "weights": weights,
        "seasonal": metric_summary(actual, forecasts["seasonal"]),
        "theta": metric_summary(actual, forecasts["theta"]),
        "kalman": metric_summary(actual, forecasts["kalman"]),
        "ensemble": metric_summary(actual, forecasts["ensemble"]),
        "plot": str(args.output) if args.output is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
