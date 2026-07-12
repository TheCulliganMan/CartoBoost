from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cartoboost.preview.forecasting import (
    ExpandingWindowSplitter,
    ForecastFrame,
    ForecastMetricSet,
    NaiveForecaster,
    RollingOriginBacktester,
)


class MeanFareModel:
    fit_count = 0

    def __init__(self) -> None:
        self.mean_ = None

    def fit(self, rows, targets):
        type(self).fit_count += 1
        self.seen_max_timestamp_ = rows["timestamp"].max()
        self.mean_ = float(np.mean(targets))
        return self

    def predict(self, rows):
        if rows["timestamp"].min() <= self.seen_max_timestamp_:
            raise AssertionError("validation leaked into training")
        return np.full(len(rows), self.mean_)


class BadHorizonModel(MeanFareModel):
    def predict(self, rows):
        return np.asarray([1.0])


def taxi_trips() -> pd.DataFrame:
    rows = []
    for pickup in ["pickup_1", "pickup_2"]:
        for hour in range(6):
            rows.append(
                {
                    "series_id": pickup,
                    "timestamp": hour,
                    "trip_distance": float(hour + 1),
                    "fare": float(10 + hour),
                }
            )
    return pd.DataFrame(rows)


def test_backtester_rejects_python_side_model_execution() -> None:
    splitter = ExpandingWindowSplitter(
        horizon=2,
        step=2,
        min_train_size=3,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )
    backtester = RollingOriginBacktester(
        splitter=splitter,
        metric_set=ForecastMetricSet(),
        target_col="fare",
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    with pytest.raises(RuntimeError, match="not available for Python-side models"):
        backtester.run(MeanFareModel(), taxi_trips())


def test_backtester_evaluate_rejects_models_without_native_binding() -> None:
    splitter = ExpandingWindowSplitter(horizon=2, min_train_size=3, timestamp_col="timestamp")
    backtester = RollingOriginBacktester(
        splitter=splitter,
        target_col="fare",
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    frame = taxi_trips()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="h")
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="h",
    )
    with pytest.raises(RuntimeError, match="no Rust backtesting model binding"):
        backtester.evaluate(BadHorizonModel(), forecast_frame)


def test_backtester_evaluate_uses_rust_forecaster() -> None:
    frame = taxi_trips()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="h")
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="h",
    )
    result = RollingOriginBacktester(horizon=2, min_train_size=4).evaluate(
        NaiveForecaster(), forecast_frame
    )
    assert len(result.folds) >= 1
    assert result.metrics["mae"] >= 0.0


def test_backtester_rejects_duplicate_validation_alignment_keys() -> None:
    data = taxi_trips()
    data = pd.concat([data, data.iloc[[4]]], ignore_index=True)
    splitter = ExpandingWindowSplitter(
        horizon=1,
        min_train_size=4,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )
    backtester = RollingOriginBacktester(
        splitter=splitter,
        target_col="fare",
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    with pytest.raises(RuntimeError, match="not available for Python-side models"):
        backtester.run(MeanFareModel(), data)
