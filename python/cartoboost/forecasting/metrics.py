"""Forecasting metrics with horizon and series breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class ForecastMetricSet:
    """Compute point, quantile, and interval forecast metrics."""

    seasonal_period: int = 1
    quantiles: tuple[float, ...] = field(default_factory=tuple)

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        *,
        horizon: Any | None = None,
        series_id: Any | None = None,
        y_train: Any | None = None,
        quantile_predictions: dict[float, Any] | None = None,
        lower: Any | None = None,
        upper: Any | None = None,
    ) -> dict[str, Any]:
        truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
        result = _native_point_metrics(
            truth,
            pred,
            horizon=horizon,
            series_id=series_id,
            y_train=y_train,
            seasonal_period=self.seasonal_period,
        )
        result["mape"] = _point_metrics(truth, pred)["mape"]

        if horizon is not None:
            result["per_horizon"] = _grouped_metrics(truth, pred, horizon)
        if series_id is not None:
            result["per_series"] = _grouped_metrics(truth, pred, series_id)

        quantile_predictions = quantile_predictions or {}
        quantile_scores: dict[str, float] = {}
        for q in self.quantiles:
            if q not in quantile_predictions:
                continue
            _, q_pred = _paired(truth, quantile_predictions[q], "y_true", f"q{q}")
            quantile_scores[str(q)] = pinball_loss(truth, q_pred, q)
        if quantile_scores:
            result["pinball"] = quantile_scores
            result["pinball_mean"] = float(np.mean(list(quantile_scores.values())))

        if lower is not None or upper is not None:
            if lower is None or upper is None:
                raise ValueError("lower and upper interval bounds must be provided together")
            _, lower_arr = _paired(truth, lower, "y_true", "lower")
            _, upper_arr = _paired(truth, upper, "y_true", "upper")
            if np.any(upper_arr < lower_arr):
                raise ValueError("lower bounds must be less than or equal to upper bounds")
            result["coverage"] = _native_probability_metric(
                "prob_interval_coverage_value", truth, lower_arr, upper_arr
            )
            result["interval_width"] = _native_probability_metric(
                "prob_mean_interval_width_value", lower_arr, upper_arr
            )

        return result

    def evaluate_frame(
        self,
        frame: Any,
        *,
        prediction_frame: Any | None = None,
        actual_col: str = "actual",
        prediction_col: str = "prediction",
        horizon_col: str = "horizon",
        series_id_col: str = "series_id",
        timestamp_col: str = "timestamp",
        lower_col: str | None = "lower",
        upper_col: str | None = "upper",
        y_train: Any | None = None,
    ) -> dict[str, Any]:
        if prediction_frame is not None:
            frame = _align_frames(
                frame,
                prediction_frame,
                actual_col=actual_col,
                prediction_col=prediction_col,
                horizon_col=horizon_col,
                series_id_col=series_id_col,
                timestamp_col=timestamp_col,
                lower_col=lower_col,
                upper_col=upper_col,
            )
        else:
            _validate_metric_keys(frame, series_id_col, timestamp_col, horizon_col)
            frame = _sort_frame(frame, [series_id_col, timestamp_col, horizon_col])
        lower = _optional_column(frame, lower_col)
        upper = _optional_column(frame, upper_col)
        return self.evaluate(
            _column(frame, actual_col),
            _column(frame, prediction_col),
            horizon=_optional_column(frame, horizon_col),
            series_id=_optional_column(frame, series_id_col),
            lower=lower,
            upper=upper,
            y_train=y_train,
        )


def pinball_loss(y_true: Any, y_pred: Any, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    return _native_probability_metric(
        "prob_pinball_loss_value", truth, pred, scalar=float(quantile)
    )


def _native_probability_metric(name: str, *values: Any, scalar: float | None = None) -> float:
    try:
        from cartoboost import _native
    except ImportError as exc:  # pragma: no cover - source-only installs
        raise RuntimeError(
            "cartoboost._native is required for forecasting probability metrics"
        ) from exc
    function = getattr(_native, name, None)
    if function is None:
        raise RuntimeError(f"installed CartoBoost native extension lacks {name}")
    arrays = [_vector(value, f"metric_{index}").tolist() for index, value in enumerate(values)]
    if scalar is not None:
        arrays.append(float(scalar))
    return float(function(*arrays))


def _native_point_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    horizon: Any | None,
    series_id: Any | None,
    y_train: Any | None,
    seasonal_period: int,
) -> dict[str, float]:
    """Evaluate scalar forecast metrics through the Rust forecasting core."""

    try:
        from cartoboost import _native
    except ImportError as exc:  # pragma: no cover - source-only installs
        raise RuntimeError(
            "cartoboost._native is required for ForecastMetricSet; rebuild the native extension"
        ) from exc
    result_class = getattr(_native, "ForecastResult", None)
    evaluator = getattr(_native, "forecast_evaluate_metrics", None)
    if result_class is None or evaluator is None:
        raise RuntimeError(
            "installed CartoBoost native extension lacks Rust forecast metric bindings"
        )

    rows = len(truth)
    series = _metric_series(series_id, rows)
    horizons = _metric_horizons(horizon, rows)
    timestamps = [
        (datetime(2000, 1, 1) + timedelta(seconds=index)).strftime("%Y-%m-%dT%H:%M:%S")
        for index in range(rows)
    ]
    predictions = [
        (series[index], timestamps[index], horizons[index], "metric_set", float(pred[index]))
        for index in range(rows)
    ]
    actuals = [
        (series[index], timestamps[index], horizons[index], float(truth[index]))
        for index in range(rows)
    ]
    training_actuals = None
    if y_train is not None:
        train = _vector(y_train, "y_train")
        training_timestamps = [
            (datetime(1990, 1, 1) + timedelta(seconds=index)).strftime("%Y-%m-%dT%H:%M:%S")
            for index in range(len(train))
        ]
        training_actuals = [
            ("__single__", training_timestamps[index], index + 1, float(value))
            for index, value in enumerate(train)
        ]
    native_result = evaluator(
        result_class(predictions),
        actuals,
        training_actuals,
        int(seasonal_period) if y_train is not None else None,
    )
    mase = native_result.mase
    return {
        "mae": float(native_result.mae),
        "rmse": float(native_result.rmse),
        "normalized_rmse": float(native_result.normalized_rmse),
        "wape": float(native_result.wape),
        "smape": float(native_result.smape),
        "bias": float(native_result.bias),
        "mase": float(mase) if mase is not None else float("nan"),
    }


def _metric_series(values: Any | None, rows: int) -> list[str]:
    if values is None:
        return ["__single__"] * rows
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError("series_id must be one-dimensional and match y_true")
    return [str(value) for value in array.tolist()]


def _metric_horizons(values: Any | None, rows: int) -> list[int]:
    if values is None:
        return list(range(1, rows + 1))
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError("horizon must be one-dimensional and match y_true")
    result = []
    for value in array.tolist():
        numeric = float(value)
        if not numeric.is_integer() or numeric <= 0:
            raise ValueError("horizon values must be positive integers")
        result.append(int(numeric))
    return result


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    denominator = np.abs(y_true)
    nonzero = denominator > 0
    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    smape_mask = smape_denominator > 0
    total_abs_truth = float(np.sum(denominator))
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mape": (
            float(np.mean(abs_error[nonzero] / denominator[nonzero])) if np.any(nonzero) else 0.0
        ),
        "smape": (
            float(np.mean(2.0 * abs_error[smape_mask] / smape_denominator[smape_mask]))
            if np.any(smape_mask)
            else 0.0
        ),
        "wape": float(np.sum(abs_error) / total_abs_truth) if total_abs_truth > 0 else 0.0,
        "bias": float(np.mean(error)),
    }


def _mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Any | None,
    seasonal_period: int,
) -> float:
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")
    if y_train is None:
        return float("nan")
    train = _vector(y_train, "y_train")
    if train.size <= seasonal_period:
        raise ValueError("y_train must be longer than seasonal_period for MASE")
    scale = float(np.mean(np.abs(train[seasonal_period:] - train[:-seasonal_period])))
    if scale == 0.0:
        raise ValueError("MASE scale is zero")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Any,
) -> dict[str, dict[str, float]]:
    group_arr = np.asarray(groups)
    if group_arr.shape != y_true.shape:
        raise ValueError("group arrays must match y_true shape")
    out: dict[str, dict[str, float]] = {}
    for group in np.unique(group_arr):
        mask = group_arr == group
        out[str(group)] = _point_metrics(y_true[mask], y_pred[mask])
    return out


def _paired(
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_arr = _vector(left, left_name)
    right_arr = _vector(right, right_name)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left_arr.size == 0:
        raise ValueError(f"{left_name} and {right_name} must contain at least one value")
    return left_arr, right_arr


def _vector(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _column(frame: Any, name: str) -> Any:
    try:
        return frame[name]
    except Exception as exc:
        raise ValueError(f"frame must contain column {name!r}") from exc


def _optional_column(frame: Any, name: str | None) -> Any | None:
    if name is None:
        return None
    try:
        return frame[name]
    except Exception:
        return None


def _align_frames(
    actual_frame: Any,
    prediction_frame: Any,
    *,
    actual_col: str,
    prediction_col: str,
    horizon_col: str,
    series_id_col: str,
    timestamp_col: str,
    lower_col: str | None,
    upper_col: str | None,
) -> Any:
    _validate_metric_keys(actual_frame, series_id_col, timestamp_col, horizon_col)
    _validate_metric_keys(prediction_frame, series_id_col, timestamp_col, horizon_col)
    _column(actual_frame, actual_col)
    _column(prediction_frame, prediction_col)
    if not hasattr(actual_frame, "merge") or not hasattr(prediction_frame, "merge"):
        raise TypeError("prediction_frame alignment requires pandas-like frames")

    key_cols = [series_id_col, timestamp_col, horizon_col]
    prediction_cols = [*key_cols, prediction_col]
    for bound_col in (lower_col, upper_col):
        if bound_col is not None and _optional_column(prediction_frame, bound_col) is not None:
            prediction_cols.append(bound_col)

    merged = actual_frame[[*key_cols, actual_col]].merge(
        prediction_frame[prediction_cols],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(actual_frame) or len(merged) != len(prediction_frame):
        raise ValueError(
            "actual and prediction rows must align exactly by series_id/timestamp/horizon"
        )
    return _sort_frame(merged, key_cols)


def _validate_metric_keys(
    frame: Any,
    series_id_col: str,
    timestamp_col: str,
    horizon_col: str,
) -> None:
    key_cols = [series_id_col, timestamp_col, horizon_col]
    for col in key_cols:
        _column(frame, col)
    if hasattr(frame, "duplicated"):
        duplicated = frame.duplicated(subset=key_cols)
        if bool(duplicated.any()):
            raise ValueError("metric rows must be unique by series_id/timestamp/horizon")
        return

    key_arrays = [np.asarray(_column(frame, col), dtype=object) for col in key_cols]
    keys = list(zip(*key_arrays, strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("metric rows must be unique by series_id/timestamp/horizon")


def _sort_frame(frame: Any, key_cols: list[str]) -> Any:
    if hasattr(frame, "sort_values"):
        return frame.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    return frame
