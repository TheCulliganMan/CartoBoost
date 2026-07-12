"""Rolling-origin backtesting orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import ForecastMetricSet
from .schema import ForecastFrame
from .splitters import ForecastFold, RollingOriginSplitter


@dataclass
class BacktestFoldResult:
    fold: ForecastFold
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold.fold_id,
            "metrics": _jsonable(self.metrics),
            "predictions": _jsonable(self.predictions),
            "metadata": _jsonable(
                {
                    **self.fold.metadata,
                    "train_start": self.fold.train_start,
                    "train_end": self.fold.train_end,
                    "validation_start": self.fold.validation_start,
                    "validation_end": self.fold.validation_end,
                    "horizon": self.fold.horizon,
                    "step": self.fold.step,
                }
            ),
        }


@dataclass
class BacktestResult:
    folds: list[BacktestFoldResult] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        if not self.folds:
            return {}
        keys = {
            key
            for fold in self.folds
            for key, value in fold.metrics.items()
            if isinstance(value, int | float) and np.isfinite(value)
        }
        return {
            key: float(np.mean([fold.metrics[key] for fold in self.folds if key in fold.metrics]))
            for key in sorted(keys)
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "metrics": _jsonable(self.metrics),
            "folds": [fold.to_json() for fold in self.folds],
        }

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_pandas requires pandas; install pandas to use this helper"
            ) from exc
        rows = [row for fold in self.folds for row in fold.predictions]
        return pd.DataFrame(rows)


class RollingOriginBacktester:
    """Fit a fresh model per fold and score exact-horizon predictions."""

    def __init__(
        self,
        *,
        splitter: RollingOriginSplitter | None = None,
        horizon: int | None = None,
        min_train_size: int = 1,
        step_size: int = 1,
        max_train_size: int | None = None,
        metric_set: ForecastMetricSet | None = None,
        target_col: str = "target",
        timestamp_col: str = "timestamp",
        series_id_col: str | None = "series_id",
    ) -> None:
        if splitter is None:
            if horizon is None:
                raise ValueError("either splitter or horizon is required")
            splitter = RollingOriginSplitter(
                horizon=horizon,
                step=step_size,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                timestamp_col=timestamp_col,
                series_id_col=series_id_col,
            )
        self.splitter = splitter
        self.metric_set = metric_set or ForecastMetricSet()
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.series_id_col = series_id_col

    def evaluate(self, model: Any, frame: ForecastFrame) -> BacktestResult:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("evaluate requires a ForecastFrame")
        return self._evaluate_native(model, frame)

    def _evaluate_native(self, model: Any, frame: ForecastFrame) -> BacktestResult:
        native_frame = getattr(frame, "_native_frame", None)
        new_native_model = getattr(model, "_new_native_model", None)
        if native_frame is None:
            raise RuntimeError(
                "Rust ForecastFrame binding is unavailable; rolling-origin backtesting "
                "does not run a Python fallback"
            )
        if not callable(new_native_model):
            raise RuntimeError(
                f"{model.__class__.__name__} has no Rust backtesting model binding; "
                "rolling-origin backtesting does not run a Python fallback"
            )
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise RuntimeError(
                "cartoboost._native is required for rolling-origin backtesting; "
                "install a native CartoBoost wheel"
            ) from exc
        native_splitter_class = getattr(_native, "RollingOriginSplitter", None)
        native_backtester_class = getattr(_native, "RollingOriginBacktester", None)
        if native_splitter_class is None or native_backtester_class is None:
            raise RuntimeError(
                "cartoboost._native does not expose the Rust rolling-origin backtester"
            )
        native_splitter = native_splitter_class(
            self.splitter.horizon,
            step=self.splitter.step,
            min_train_size=self.splitter.min_train_size,
            max_train_size=self.splitter.max_train_size,
            n_splits=self.splitter.n_splits,
            window=self.splitter.window,
        )
        native_backtester = native_backtester_class(native_splitter)
        native_model = new_native_model()
        runner_name = _native_backtest_runner_name(model)
        if not runner_name:
            raise RuntimeError(
                f"Rust rolling-origin backtesting does not support "
                f"{model.__class__.__name__}; no Python fallback is available"
            )
        runner = getattr(native_backtester, runner_name, None)
        if runner is None:
            raise RuntimeError(
                f"cartoboost._native.RollingOriginBacktester is missing {runner_name} "
                f"for {model.__class__.__name__}; no Python fallback is available"
            )
        return _backtest_result_from_native(runner(native_model, native_frame))

    def run(self, model: Any, data: Any) -> BacktestResult:
        del model, data
        raise RuntimeError(
            "RollingOriginBacktester.run is not available for Python-side models; "
            "construct a ForecastFrame and call evaluate() with a Rust-backed "
            "forecaster"
        )


def _native_backtest_runner_name(model: Any) -> str:
    name = getattr(model, "native_class_name", model.__class__.__name__)
    mapping = {
        "NaiveForecaster": "run_naive",
        "SeasonalNaiveForecaster": "run_seasonal_naive",
        "ThetaForecaster": "run_theta",
        "OptimizedThetaForecaster": "run_optimized_theta",
        "ETSForecaster": "run_ets",
        "ArimaForecaster": "run_arima",
        "AutoARIMAForecaster": "run_auto_arima",
        "AutoForecaster": "run_auto_forecast",
        "CartoBoostLagForecaster": "run_cartoboost_lag",
    }
    return mapping.get(name, "")


def _backtest_result_from_native(native_result: Any) -> BacktestResult:
    folds: list[BacktestFoldResult] = []
    for fold_result in native_result.folds:
        native_fold = fold_result.fold
        fold = ForecastFold(
            fold_id=native_fold.fold_id,
            train_indices=np.asarray(native_fold.train_indices, dtype=int),
            validation_indices=np.asarray(native_fold.validation_indices, dtype=int),
            train_start=native_fold.train_start,
            train_end=native_fold.train_end,
            validation_start=native_fold.validation_start,
            validation_end=native_fold.validation_end,
            horizon=int(native_fold.horizon),
            step=int(native_fold.step),
            metadata=json.loads(native_fold.metadata_json()),
        )
        predictions = [
            {
                "fold_id": fold.fold_id,
                "series_id": row[0],
                "timestamp": row[1],
                "horizon": row[2],
                "model": row[3],
                "prediction": row[4],
            }
            for row in fold_result.predictions
        ]
        native_metrics = fold_result.metrics
        metrics = {
            "mae": native_metrics.mae,
            "rmse": native_metrics.rmse,
            "normalized_rmse": native_metrics.normalized_rmse,
            "wape": native_metrics.wape,
            "smape": native_metrics.smape,
            "bias": native_metrics.bias,
        }
        if native_metrics.mase is not None:
            metrics["mase"] = native_metrics.mase
        folds.append(
            BacktestFoldResult(
                fold=fold,
                metrics=metrics,
                predictions=predictions,
            )
        )
    return BacktestResult(folds=folds)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
