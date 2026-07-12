from __future__ import annotations

import math

import cartoboost.metrics as metrics_module
import pandas as pd
import pytest
from cartoboost.forecasting import ForecastMetricSet
from cartoboost.forecasting.metrics import pinball_loss
from cartoboost.metrics import competition_forecast_metrics


def test_point_metrics_zero_safe_and_grouped() -> None:
    metrics = ForecastMetricSet(seasonal_period=1).evaluate(
        [0.0, 10.0, 20.0],
        [0.0, 12.0, 18.0],
        horizon=[1, 1, 2],
        series_id=["pickup_1", "pickup_1", "pickup_2"],
        y_train=[5.0, 7.0, 9.0, 11.0],
    )

    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(8.0 / 3.0))
    assert metrics["mape"] == pytest.approx(0.15)
    assert metrics["wape"] == pytest.approx(4.0 / 30.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["mase"] == pytest.approx((4.0 / 3.0) / 2.0)
    assert set(metrics["per_horizon"]) == {"1", "2"}
    assert set(metrics["per_series"]) == {"pickup_1", "pickup_2"}


def test_pinball_coverage_and_interval_width() -> None:
    metric_set = ForecastMetricSet(quantiles=(0.5, 0.9))
    metrics = metric_set.evaluate(
        [10.0, 20.0],
        [11.0, 18.0],
        quantile_predictions={0.5: [11.0, 18.0], 0.9: [13.0, 23.0]},
        lower=[11.0, 19.0],
        upper=[12.0, 24.0],
    )

    assert metrics["pinball"]["0.5"] == pytest.approx(pinball_loss([10.0, 20.0], [11.0, 18.0], 0.5))
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["interval_width"] == pytest.approx(3.0)


def test_competition_forecast_metrics_delegate_to_native(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeNative:
        def competition_forecast_metrics_value(
            self,
            training_series,
            actuals,
            forecasts,
            seasonality,
            baseline_smape,
            baseline_mase,
        ):
            calls["payload"] = (
                training_series,
                actuals,
                forecasts,
                seasonality,
                baseline_smape,
                baseline_mase,
            )
            return (
                '{"smape":0.1,"mase":0.2,'
                '"smape_ratio_to_baseline":0.5,'
                '"mase_ratio_to_baseline":0.25,"owa":0.375}'
            )

    monkeypatch.setattr(metrics_module, "_native", FakeNative())

    result = competition_forecast_metrics(
        [[1.0, 2.0, 3.0]],
        [4.0],
        [3.5],
        seasonality=1,
        baseline_smape=0.2,
        baseline_mase=0.8,
    )

    assert result["owa"] == pytest.approx(0.375)
    assert calls["payload"] == (
        [[1.0, 2.0, 3.0]],
        [4.0],
        [3.5],
        1,
        0.2,
        0.8,
    )


def test_frame_evaluation_aligns_series_timestamp_horizon_columns() -> None:
    frame = pd.DataFrame(
        {
            "series_id": ["pickup_1", "pickup_1"],
            "timestamp": [1, 2],
            "horizon": [1, 2],
            "actual": [10.0, 20.0],
            "prediction": [9.0, 21.0],
            "lower": [8.0, 18.0],
            "upper": [12.0, 22.0],
        }
    )

    metrics = ForecastMetricSet().evaluate_frame(frame)

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert set(metrics["per_horizon"]) == {"1", "2"}


def test_frame_evaluation_aligns_separate_frames_by_keys_not_row_order() -> None:
    actual = pd.DataFrame(
        {
            "series_id": ["pickup_1", "pickup_1", "pickup_2"],
            "timestamp": [1, 2, 1],
            "horizon": [1, 2, 1],
            "actual": [10.0, 20.0, 30.0],
        }
    )
    prediction = pd.DataFrame(
        {
            "series_id": ["pickup_2", "pickup_1", "pickup_1"],
            "timestamp": [1, 2, 1],
            "horizon": [1, 2, 1],
            "prediction": [29.0, 19.0, 9.0],
        }
    )

    metrics = ForecastMetricSet().evaluate_frame(actual, prediction_frame=prediction)

    assert metrics["mae"] == pytest.approx(1.0)
    assert set(metrics["per_series"]) == {"pickup_1", "pickup_2"}


def test_frame_evaluation_rejects_duplicate_metric_keys() -> None:
    frame = pd.DataFrame(
        {
            "series_id": ["pickup_1", "pickup_1"],
            "timestamp": [1, 1],
            "horizon": [1, 1],
            "actual": [10.0, 11.0],
            "prediction": [9.0, 12.0],
        }
    )

    with pytest.raises(ValueError, match="unique by series_id/timestamp/horizon"):
        ForecastMetricSet().evaluate_frame(frame)
