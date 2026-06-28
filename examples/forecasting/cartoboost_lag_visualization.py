from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from cartoboost.forecasting import CartoBoostLagForecaster

LAGS = [1, 2, 24]
ROLLING_WINDOWS = [6, 24]


def synthetic_zone_demand(hours: int) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    zones = [
        ("132", 92.0, 18.0, 0.20),
        ("161", 74.0, 11.0, 0.12),
        ("236", 68.0, 8.0, 0.08),
    ]
    rows: list[dict[str, Any]] = []
    for zone_id, baseline, rush_amp, trend in zones:
        for hour in range(hours):
            pickup_hour = start + timedelta(hours=hour)
            hour_of_day = pickup_hour.hour
            morning = rush_amp if 6 <= hour_of_day <= 9 else 0.0
            evening = 0.75 * rush_amp if 16 <= hour_of_day <= 19 else 0.0
            overnight = -9.0 if hour_of_day <= 4 else 0.0
            airport_wave = 5.0 * math.sin(hour / 8.0) if zone_id == "132" else 0.0
            pickup_count = baseline + morning + evening + overnight + airport_wave + trend * hour
            rows.append(
                {
                    "PULocationID": zone_id,
                    "pickup_hour": pickup_hour,
                    "pickup_count": pickup_count,
                }
            )
    return pd.DataFrame(rows)


def forecast_lag_model(table: pd.DataFrame, train_hours: int, horizon: int) -> pd.DataFrame:
    train = table.groupby("PULocationID", sort=False).head(train_hours).reset_index(drop=True)
    model = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_count",
        panel_cols=["PULocationID"],
        frequency="h",
        lags=LAGS,
        rolling_windows=ROLLING_WINDOWS,
        calendar_features=True,
        trend_features=True,
        recursive=True,
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=8,
        splitters=["axis", "periodic:24"],
    )
    model.fit(train)
    forecast = pd.DataFrame(
        model.predict(horizon).predictions(),
        columns=["PULocationID", "pickup_hour", "horizon", "model", "prediction"],
    )
    forecast["pickup_hour"] = pd.to_datetime(forecast["pickup_hour"])
    return forecast


def actual_window(table: pd.DataFrame, train_hours: int, horizon: int) -> pd.DataFrame:
    actual = table.groupby("PULocationID", sort=False).nth(
        list(range(train_hours, train_hours + horizon))
    )
    return actual.reset_index()


def metric_summary(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    joined = actual.merge(forecast, on=["PULocationID", "pickup_hour"], validate="one_to_one")
    errors = [row.prediction - row.pickup_count for row in joined.itertuples()]
    return {
        "mae": mean(abs(error) for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
    }


def write_plot(
    table: pd.DataFrame,
    forecast: pd.DataFrame,
    train_hours: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zone_id = "132"
    observed = table[table["PULocationID"] == zone_id]
    predicted = forecast[forecast["PULocationID"] == zone_id]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    axes[0].plot(
        observed["pickup_hour"],
        observed["pickup_count"],
        color="#1f2937",
        label="observed pickups",
    )
    axes[0].plot(
        predicted["pickup_hour"],
        predicted["prediction"],
        marker="o",
        color="#2563eb",
        label="CartoBoost lag forecast",
    )
    axes[0].axvline(
        observed["pickup_hour"].iloc[train_hours - 1],
        color="#6b7280",
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_title("Zone 132 Pickup Demand: Recursive Lag Forecast")
    axes[0].set_ylabel("pickup count")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    joined = observed.merge(predicted, on=["PULocationID", "pickup_hour"], how="inner")
    errors = joined["prediction"] - joined["pickup_count"]
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].bar(joined["pickup_hour"], errors, width=0.03, color="#dc2626")
    axes[1].set_title("Holdout Residuals")
    axes[1].set_xlabel("pickup hour")
    axes[1].set_ylabel("forecast error")
    axes[1].grid(alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=96)
    parser.add_argument("--train-hours", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("target/examples/cartoboost_lag.png"))
    args = parser.parse_args()

    if args.hours <= args.train_hours:
        raise ValueError("--hours must be greater than --train-hours")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.train_hours + args.horizon > args.hours:
        raise ValueError("--train-hours + --horizon must be <= --hours")
    min_train_hours = max([*LAGS, *ROLLING_WINDOWS]) + 2
    if args.train_hours < min_train_hours:
        raise ValueError(f"--train-hours must be at least {min_train_hours}")

    table = synthetic_zone_demand(args.hours)
    forecast = forecast_lag_model(table, train_hours=args.train_hours, horizon=args.horizon)
    actual = actual_window(table, train_hours=args.train_hours, horizon=args.horizon)
    metrics = metric_summary(actual, forecast)

    if args.output is not None:
        write_plot(table, forecast, args.train_hours, args.output)

    payload = {
        "task": "example_taxi_zone_cartoboost_lag_forecast",
        "rows": int(len(table)),
        "zones": int(table["PULocationID"].nunique()),
        "train_hours": int(args.train_hours),
        "horizon": int(args.horizon),
        "lag_model": metrics,
        "plot": str(args.output) if args.output is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
