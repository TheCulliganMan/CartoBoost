from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from cartoboost.forecasting import ArimaForecaster, AutoARIMAForecaster, ForecastFrame

LANE_TO_PLOT = "PU132->DO138"


def synthetic_lane_demand(hours: int) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    rows: list[dict[str, Any]] = []
    lanes = [
        ("PU132->DO138", 92.0, 18.0, 0.18),
        ("PU79->DO230", 64.0, 7.0, 0.08),
    ]
    for lane_id, baseline, airport_ramp, trend in lanes:
        for hour in range(hours):
            pickup_hour = start + timedelta(hours=hour)
            hour_of_day = pickup_hour.hour
            morning = 13.0 if 6 <= hour_of_day <= 9 else -4.0
            evening = 6.0 if 16 <= hour_of_day <= 19 else 0.0
            weekly = 4.0 * math.sin((hour % 168) / 168.0 * math.tau)
            local_wave = 2.5 * math.sin(hour / 5.0)
            pickup_count = baseline + morning + evening + weekly + local_wave
            pickup_count += trend * hour + airport_ramp * (hour / max(hours - 1, 1))
            rows.append(
                {
                    "lane_id": lane_id,
                    "pickup_hour": pickup_hour,
                    "pickup_count": pickup_count,
                }
            )
    return pd.DataFrame(rows)


def order_label(order: dict[str, int]) -> str:
    return f"ARIMA({order['p']},{order['d']},{order['q']})"


def forecast_panel(
    table: pd.DataFrame,
    model: ArimaForecaster | AutoARIMAForecaster,
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
    rows = [
        {
            "lane_id": series_id,
            "pickup_hour": pd.Timestamp(timestamp),
            "horizon": horizon_step,
            "model": model_name,
            "prediction": mean_value,
        }
        for series_id, timestamp, horizon_step, model_name, mean_value in model.predict(
            horizon
        ).predictions()
    ]
    return pd.DataFrame(rows)


def evaluated_forecast(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    joined = actual.merge(
        forecast,
        on=["lane_id", "pickup_hour"],
        validate="one_to_one",
    )
    joined["residual"] = joined["prediction"] - joined["pickup_count"]
    joined["absolute_error"] = joined["residual"].abs()
    joined["squared_error"] = joined["residual"] * joined["residual"]
    return joined


def metric_summary(evaluation: pd.DataFrame) -> dict[str, float]:
    errors = [float(row.residual) for row in evaluation.itertuples()]
    mae = mean(abs(error) for error in errors)
    rmse = math.sqrt(mean(error * error for error in errors))
    bias = mean(errors)
    max_abs_error = max(abs(error) for error in errors)
    return {"mae": mae, "rmse": rmse, "bias": bias, "max_abs_error": max_abs_error}


def lane_residuals(evaluation: pd.DataFrame, lane_id: str) -> list[dict[str, float | int | str]]:
    lane = evaluation[evaluation["lane_id"] == lane_id].sort_values("horizon")
    return [
        {
            "horizon": int(row.horizon),
            "pickup_hour": str(row.pickup_hour),
            "actual": float(row.pickup_count),
            "prediction": float(row.prediction),
            "residual": float(row.residual),
        }
        for row in lane.itertuples()
    ]


def top_auto_candidates(metadata: dict[str, Any], limit: int = 5) -> list[dict[str, float | int]]:
    scores = metadata.get("validation_scores", [])
    ranked = sorted(
        scores, key=lambda score: (float(score["mse"]), score["d"], score["p"], score["q"])
    )
    return [
        {
            "p": int(score["p"]),
            "d": int(score["d"]),
            "q": int(score["q"]),
            "mse": float(score["mse"]),
        }
        for score in ranked[:limit]
    ]


def write_plot(
    table: pd.DataFrame,
    fixed_evaluation: pd.DataFrame,
    auto_evaluation: pd.DataFrame,
    train_hours: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lane_id = LANE_TO_PLOT
    observed = table[table["lane_id"] == lane_id]
    fixed = fixed_evaluation[fixed_evaluation["lane_id"] == lane_id].sort_values("horizon")
    auto = auto_evaluation[auto_evaluation["lane_id"] == lane_id].sort_values("horizon")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    axes[0].plot(
        observed["pickup_hour"],
        observed["pickup_count"],
        label="Observed pickups",
        color="#1f2937",
    )
    axes[0].plot(
        fixed["pickup_hour"],
        fixed["prediction"],
        label="ARIMA(2,1,1)",
        color="#2563eb",
        linewidth=2,
    )
    axes[0].plot(
        auto["pickup_hour"],
        auto["prediction"],
        label="AutoARIMA",
        color="#dc2626",
        linewidth=2,
    )
    split_time = observed["pickup_hour"].iloc[train_hours - 1]
    axes[0].axvline(split_time, color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title(f"{lane_id} hourly pickup forecast")
    axes[0].set_ylabel("pickup count")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    width = 0.35
    horizons = [int(value) for value in fixed["horizon"]]
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].bar(
        [horizon - width / 2 for horizon in horizons],
        fixed["residual"],
        width=width,
        label="ARIMA(2,1,1)",
        color="#2563eb",
    )
    axes[1].bar(
        [horizon + width / 2 for horizon in horizons],
        auto["residual"],
        width=width,
        label="AutoARIMA",
        color="#dc2626",
    )
    axes[1].set_title("Held-out residuals by forecast horizon")
    axes[1].set_xlabel("forecast horizon")
    axes[1].set_ylabel("prediction - actual")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", alpha=0.2)

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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.hours <= args.train_hours:
        raise ValueError("--hours must be greater than --train-hours")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.train_hours + args.horizon > args.hours:
        raise ValueError("--train-hours + --horizon must be <= --hours")

    table = synthetic_lane_demand(args.hours)
    actual = table.groupby("lane_id", sort=False).nth(
        list(range(args.train_hours, args.train_hours + args.horizon))
    )
    actual = actual.reset_index()

    fixed_model = ArimaForecaster(p=2, d=1, q=1)
    fixed_forecast = forecast_panel(
        table,
        fixed_model,
        train_hours=args.train_hours,
        horizon=args.horizon,
    )
    auto_model = AutoARIMAForecaster(max_p=3, max_d=1, max_q=2)
    auto_forecast = forecast_panel(
        table,
        auto_model,
        train_hours=args.train_hours,
        horizon=args.horizon,
    )
    fixed_evaluation = evaluated_forecast(actual, fixed_forecast)
    auto_evaluation = evaluated_forecast(actual, auto_forecast)
    fixed_metrics = metric_summary(fixed_evaluation)
    auto_metrics = metric_summary(auto_evaluation)
    auto_metadata = auto_model.get_metadata()
    selected_order = auto_metadata["selected_order"]
    winner = "arima_2_1_1" if fixed_metrics["rmse"] <= auto_metrics["rmse"] else "auto_arima"

    payload = {
        "task": "example_taxi_lane_arima_forecast",
        "rows": int(len(table)),
        "lanes": int(table["lane_id"].nunique()),
        "train_hours": int(args.train_hours),
        "horizon": int(args.horizon),
        "heldout_lane": LANE_TO_PLOT,
        "arima_2_1_1": fixed_metrics,
        "auto_arima": auto_metrics,
        "auto_arima_selected_order": selected_order,
        "auto_arima_selected_label": order_label(selected_order),
        "auto_arima_top_candidates": top_auto_candidates(auto_metadata),
        "auto_arima_metadata": auto_metadata,
        "heldout_winner_by_rmse": winner,
        "residuals": {
            "arima_2_1_1": lane_residuals(fixed_evaluation, LANE_TO_PLOT),
            "auto_arima": lane_residuals(auto_evaluation, LANE_TO_PLOT),
        },
    }

    if args.output is not None:
        write_plot(table, fixed_evaluation, auto_evaluation, args.train_hours, args.output)
        payload["plot"] = str(args.output)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
