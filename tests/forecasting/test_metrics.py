from __future__ import annotations

import math
from datetime import datetime, timedelta

import cartoboost.metrics as metrics_module
import numpy as np
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


@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
@pytest.mark.parametrize("rows", [1, 17, 1024])
def test_native_array_probability_metrics_match_legacy_lists(layout, rows):
    from cartoboost import _native

    def values(offset):
        array = np.linspace(0.0, 0.7, rows * 2, dtype=np.float64) + offset
        if layout == "strided":
            return array[::2]
        if layout == "reversed":
            return array[:rows][::-1]
        return array[:rows].copy()

    actual, pred, weights = values(0.2), values(0.1), values(1.0)
    lower, upper = pred - 0.3, pred + 0.3
    cases = [
        ("prob_pinball_loss", (actual, pred, 0.7, "cpu", weights)),
        ("prob_interval_coverage", (actual, lower, upper, "cpu", weights)),
        ("prob_mean_interval_width", (lower, upper, "cpu", weights)),
        ("prob_brier_score", ((actual > 0.5).astype(float), pred, "cpu", weights)),
        ("prob_split_conformal_residual_quantile", (actual, pred, 0.5, 1, 1, rows + 1, rows + 1)),
        (
            "prob_weighted_conformal_residual_quantile",
            (actual, pred, weights, 0.5, 1, 1, rows + 1, rows + 1),
        ),
    ]
    for name, args in cases:
        legacy = tuple(value.tolist() if isinstance(value, np.ndarray) else value for value in args)
        expected = getattr(_native, f"{name}_value")(*legacy)
        result = getattr(_native, f"{name}_arrays")(*args)
        if "conformal" in name:
            assert result == expected
        else:
            # Parallel reductions may associate the same terms differently.
            assert result == pytest.approx(expected, rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("groups", [1, 8, 128])
def test_grouped_metrics_keep_full_scan_reference_order_and_values(groups):
    from cartoboost.forecasting.metrics import _grouped_metrics, _point_metrics

    rng = np.random.default_rng(20261124)
    truth = rng.normal(size=groups * 5)
    pred = truth + rng.normal(size=len(truth)) * 0.2
    labels = np.array([f"pickup-{index % groups}" for index in range(len(truth))])
    order = rng.permutation(len(truth))
    truth, pred, labels = truth[order], pred[order], labels[order]
    expected = {
        str(label): _point_metrics(truth[labels == label], pred[labels == label])
        for label in np.unique(labels)
    }

    assert _grouped_metrics(truth, pred, labels) == expected


@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
def test_forecast_metric_array_binding_preserves_mase_and_point_metrics(layout):
    from cartoboost import _native

    truth = np.linspace(-3.0, 5.0, 34)
    pred = truth + np.sin(truth)
    training = np.arange(20, dtype=np.float64)
    if layout == "strided":
        truth, pred, training = truth[::2], pred[::2], training[::2]
    elif layout == "reversed":
        truth, pred, training = truth[::-1], pred[::-1], training[::-1]
    timestamps = [
        (datetime(2025, 1, 1) + timedelta(hours=i)).isoformat() for i in range(len(truth))
    ]
    forecast = _native.ForecastResult(
        [
            ("pickup-a", timestamp, i + 1, "metric_set", float(pred[i]))
            for i, timestamp in enumerate(timestamps)
        ]
    )
    actuals = [
        ("pickup-a", timestamp, i + 1, float(truth[i])) for i, timestamp in enumerate(timestamps)
    ]
    training_actuals = [
        ("pickup-a", (datetime(2024, 1, 1) + timedelta(hours=i)).isoformat(), i + 1, float(value))
        for i, value in enumerate(training)
    ]
    expected = _native.forecast_evaluate_metrics(forecast, actuals, training_actuals, 2)
    actual = _native.forecast_evaluate_metrics_arrays(truth, pred, training, 2)
    for name in ("mae", "rmse", "normalized_rmse", "wape", "smape", "bias", "mase"):
        assert getattr(actual, name) == getattr(expected, name)
