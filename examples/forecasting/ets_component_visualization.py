from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from cartoboost.forecasting import ETSForecaster, ForecastFrame


def synthetic_airport_pickups(hours: int) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    rows: list[dict[str, Any]] = []
    for hour in range(hours):
        pickup_hour = start + timedelta(hours=hour)
        hour_of_day = pickup_hour.hour
        overnight = -18.0 if hour_of_day <= 4 else 0.0
        morning = 20.0 if 6 <= hour_of_day <= 9 else 0.0
        evening = 14.0 if 16 <= hour_of_day <= 20 else 0.0
        daily_wave = 8.5 * math.sin((hour_of_day - 7) / 24.0 * math.tau)
        terminal_ramp = 0.16 * hour
        deterministic_noise = 2.8 * math.sin(hour / 3.4) + 1.6 * math.cos(hour / 5.1)
        pickup_count = 82.0 + overnight + morning + evening + daily_wave + terminal_ramp
        pickup_count += deterministic_noise
        rows.append(
            {
                "PULocationID": "132",
                "pickup_hour": pickup_hour,
                "pickup_count": pickup_count,
            }
        )
    return pd.DataFrame(rows)


def forecast_frame(table: pd.DataFrame, train_hours: int) -> ForecastFrame:
    train = table.iloc[:train_hours].reset_index(drop=True)
    return ForecastFrame.from_pandas(
        train,
        timestamp_col="pickup_hour",
        target_col="pickup_count",
        series_id_col="PULocationID",
        freq="h",
    )


def rust_ets_forecast(
    table: pd.DataFrame,
    *,
    train_hours: int,
    horizon: int,
    alpha: float,
    beta: float,
    gamma: float,
    season_length: int,
) -> tuple[ETSForecaster, pd.DataFrame]:
    model = ETSForecaster(
        trend="additive",
        seasonal="additive",
        seasonal_periods=season_length,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    model.fit(forecast_frame(table, train_hours))
    rows = [
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
    return model, pd.DataFrame(rows)


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
    model: ETSForecaster,
    *,
    train_hours: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = table.iloc[:train_hours]
    actual_future = table.iloc[train_hours : train_hours + len(forecast)]
    component_hours = train["pickup_hour"]
    series_id = "132"

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.6), sharex=False)
    axes[0].plot(table["pickup_hour"], table["pickup_count"], color="#111827", label="Observed")
    axes[0].plot(
        train["pickup_hour"],
        model.fitted_values(series_id),
        color="#2563eb",
        linestyle="--",
        label="One-step fitted",
    )
    axes[0].plot(
        forecast["pickup_hour"],
        forecast["prediction"],
        color="#dc2626",
        marker="o",
        label="ETS forecast",
    )
    axes[0].axvline(train["pickup_hour"].iloc[-1], color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title("JFK Pickup Demand: Additive ETS Forecast")
    axes[0].set_ylabel("pickup count")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.22)

    axes[1].plot(component_hours, model.levels(series_id), color="#0f766e", label="Smoothed level")
    axes[1].plot(component_hours, model.trends(series_id), color="#7c3aed", label="Smoothed trend")
    axes[1].set_title("Level And Trend Components")
    axes[1].set_ylabel("component value")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.22)

    axes[2].bar(component_hours, model.seasonal_components(series_id), color="#f59e0b", alpha=0.82)
    axes[2].axhline(0.0, color="#111827", linewidth=1)
    axes[2].set_title("Additive Hour-Of-Day Seasonal Component")
    axes[2].set_xlabel("pickup hour")
    axes[2].set_ylabel("seasonal lift")
    axes[2].grid(axis="y", alpha=0.22)

    if not actual_future.empty:
        axes[0].scatter(
            actual_future["pickup_hour"],
            actual_future["pickup_count"],
            color="#111827",
            s=28,
            zorder=3,
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=96)
    parser.add_argument("--train-hours", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--season-length", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.46)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("target/examples/ets_component_visualization.png"),
    )
    args = parser.parse_args()

    if args.season_length <= 1:
        raise ValueError("--season-length must be greater than 1")
    if args.train_hours < args.season_length * 2:
        raise ValueError("--train-hours must include at least two full seasonal cycles")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.train_hours + args.horizon > args.hours:
        raise ValueError("--train-hours + --horizon must be <= --hours")

    table = synthetic_airport_pickups(args.hours)
    model, forecast = rust_ets_forecast(
        table,
        train_hours=args.train_hours,
        horizon=args.horizon,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        season_length=args.season_length,
    )
    actual = table.iloc[args.train_hours : args.train_hours + args.horizon].reset_index(drop=True)
    metrics = metric_summary(actual, forecast)
    write_plot(
        table,
        forecast,
        model,
        train_hours=args.train_hours,
        output=args.output,
    )
    levels = model.levels("132")
    trends = model.trends("132")
    seasonals = model.seasonal_components("132")

    payload = {
        "task": "example_taxi_zone_ets_components",
        "rows": int(len(table)),
        "PULocationID": "132",
        "train_hours": int(args.train_hours),
        "horizon": int(args.horizon),
        "season_length": int(args.season_length),
        "metrics": metrics,
        "metadata": model.get_metadata(),
        "final_level": levels[-1],
        "final_trend": trends[-1],
        "seasonal_range": max(seasonals) - min(seasonals),
        "plot": str(args.output),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
