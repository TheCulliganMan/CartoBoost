from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..config import Backend
from ._native_wrappers import NativeForecastWrapper


class WeightedEnsembleForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust weighted ensemble forecasting binding."""

    native_class_name = "WeightedEnsembleForecaster"

    def __init__(
        self,
        models: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        interval_level: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        if not models:
            raise ValueError("WeightedEnsembleForecaster requires at least one model")
        if interval_level is not None or lower_bound is not None or upper_bound is not None:
            raise NotImplementedError(
                "WeightedEnsembleForecaster prediction intervals are not supported yet"
            )
        weights = _normalize_weights(models, weights)
        super().__init__(
            models=dict(models),
            weights=weights,
            interval_level=interval_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={} if metadata is None else dict(metadata),
            backend=str(backend),
        )
        self.models = dict(models)
        self.weights = weights
        self.backend = str(backend)

    def _new_native_model(self) -> Any:
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise NotImplementedError(
                "Rust binding for WeightedEnsembleForecaster is not available."
            ) from exc
        native_ensemble_class = getattr(_native, "WeightedEnsembleForecaster", None)
        if native_ensemble_class is None:
            raise NotImplementedError(
                "Rust binding for WeightedEnsembleForecaster is not available."
            )

        native_members = []
        for name, model in self.models.items():
            native_model = _native_model_from_wrapper(model)
            native_members.append((name, native_model, self.weights[name]))
        return native_ensemble_class(native_members, self.backend)

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        metadata_json = getattr(self._native_model, "metadata_json", None)
        if metadata_json is None:
            return {}
        return dict(json.loads(metadata_json()))


class BacktestWeightedEnsembleForecaster(WeightedEnsembleForecaster):
    """Thin wrapper for the Rust backtest-weighted ensemble forecasting binding."""

    native_class_name = "BacktestWeightedEnsembleForecaster"

    def __init__(
        self,
        models: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        interval_level: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        backtest_horizon: int = 1,
        min_train_size: int | None = None,
        step_size: int = 1,
        error_floor: float = 1e-9,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        if backtest_horizon < 1:
            raise ValueError("backtest_horizon must be a positive integer")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        super().__init__(
            models=models,
            weights=weights,
            interval_level=interval_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata=metadata,
            backend=backend,
        )
        self._params.update(
            {
                "backtest_horizon": int(backtest_horizon),
                "min_train_size": min_train_size,
                "step_size": int(step_size),
                "error_floor": float(error_floor),
            }
        )
        self.backtest_horizon = int(backtest_horizon)
        self.min_train_size = min_train_size
        self.step_size = int(step_size)
        self.error_floor = float(error_floor)

    def _new_native_model(self) -> Any:
        raise NotImplementedError(
            "Rust binding for BacktestWeightedEnsembleForecaster is not available."
        )


class RuleBasedGating:
    """Configuration object for Rust rule-based forecast expert gating."""

    def __init__(
        self,
        *,
        metric: str,
        scores: list[Mapping[str, Any]],
        error_floor: float = 1e-9,
        top_k: int | None = None,
    ) -> None:
        if not metric:
            raise ValueError("metric must be non-empty")
        if not scores:
            raise ValueError("scores must contain at least one validation score")
        if error_floor <= 0:
            raise ValueError("error_floor must be positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when provided")
        self.metric = str(metric)
        self.scores = [dict(score) for score in scores]
        self.error_floor = float(error_floor)
        self.top_k = top_k

    def to_native_config(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "scores": self.scores,
            "error_floor": self.error_floor,
            "top_k": self.top_k,
        }


class BottomUpReconciler(NativeForecastWrapper):
    """Thin wrapper for the Rust bottom-up hierarchical reconciliation binding."""

    native_class_name = "BottomUpReconciler"

    def __init__(
        self,
        *,
        hierarchy: Mapping[str, Any] | None = None,
        summing_matrix: Any | None = None,
        series_id_column: str | None = None,
        parent_column: str | None = None,
        child_column: str | None = None,
        non_negative: bool = False,
        metadata: Mapping[str, Any] | None = None,
        backend: Backend | str = Backend.CPU,
        **params: Any,
    ) -> None:
        _validate_reconciliation_inputs(
            hierarchy=hierarchy,
            summing_matrix=summing_matrix,
            series_id_column=series_id_column,
            parent_column=parent_column,
            child_column=child_column,
        )
        super().__init__(
            hierarchy={} if hierarchy is None else dict(hierarchy),
            summing_matrix=summing_matrix,
            series_id_column=series_id_column,
            parent_column=parent_column,
            child_column=child_column,
            non_negative=bool(non_negative),
            metadata={} if metadata is None else dict(metadata),
            backend=str(backend),
            **params,
        )


class MinTraceReconciler(NativeForecastWrapper):
    """Thin wrapper for the Rust MinT hierarchical reconciliation binding."""

    native_class_name = "MinTraceReconciler"

    def __init__(
        self,
        *,
        hierarchy: Mapping[str, Any] | None = None,
        summing_matrix: Any | None = None,
        residual_covariance: Any | None = None,
        covariance_method: str = "shrunk",
        series_id_column: str | None = None,
        parent_column: str | None = None,
        child_column: str | None = None,
        non_negative: bool = False,
        metadata: Mapping[str, Any] | None = None,
        backend: Backend | str = Backend.CPU,
        **params: Any,
    ) -> None:
        if not covariance_method:
            raise ValueError("covariance_method must be non-empty")
        _validate_reconciliation_inputs(
            hierarchy=hierarchy,
            summing_matrix=summing_matrix,
            series_id_column=series_id_column,
            parent_column=parent_column,
            child_column=child_column,
        )
        super().__init__(
            hierarchy={} if hierarchy is None else dict(hierarchy),
            summing_matrix=summing_matrix,
            residual_covariance=residual_covariance,
            covariance_method=str(covariance_method),
            series_id_column=series_id_column,
            parent_column=parent_column,
            child_column=child_column,
            non_negative=bool(non_negative),
            metadata={} if metadata is None else dict(metadata),
            backend=str(backend),
            **params,
        )


def reconcile_hierarchy(
    hierarchy: Mapping[str, str | None] | list[tuple[str, str | None]],
    base_forecasts: Any,
    *,
    method: str = "bottom_up",
    variances: Any | None = None,
    residuals: Any | None = None,
    shrinkage: float = 0.5,
    level: int | None = None,
    backend: Backend | str = Backend.CPU,
) -> dict[str, Any]:
    """Reconcile a forecast panel through the shared accelerator contract."""

    from cartoboost import _native

    function = getattr(_native, "forecast_hierarchy_reconcile_value", None)
    if function is None:
        raise RuntimeError("native hierarchy reconciliation binding is unavailable")
    edges = list(hierarchy.items()) if isinstance(hierarchy, Mapping) else list(hierarchy)
    panel = np.asarray(base_forecasts, dtype=float)
    if panel.ndim != 2 or not np.isfinite(panel).all():
        raise ValueError("base_forecasts must be a finite two-dimensional array")
    residual_rows = None
    if residuals is not None:
        residual_array = np.asarray(residuals, dtype=float)
        if residual_array.ndim != 2 or not np.isfinite(residual_array).all():
            raise ValueError("residuals must be a finite two-dimensional array")
        residual_rows = residual_array.tolist()
    variance_values = None
    if variances is not None:
        variance_array = np.asarray(variances, dtype=float).reshape(-1)
        if not np.isfinite(variance_array).all():
            raise ValueError("variances must be finite")
        variance_values = variance_array.tolist()
    payload = json.loads(
        function(
            [(str(node), None if parent is None else str(parent)) for node, parent in edges],
            panel.tolist(),
            str(method),
            variance_values,
            residual_rows,
            float(shrinkage),
            level,
            str(backend),
        )
    )
    payload["values"] = np.asarray(payload["values"], dtype=float)
    return payload


__all__ = [
    "BacktestWeightedEnsembleForecaster",
    "BottomUpReconciler",
    "MinTraceReconciler",
    "RuleBasedGating",
    "WeightedEnsembleForecaster",
    "reconcile_hierarchy",
]


def _normalize_weights(
    models: Mapping[str, Any],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return {name: 1.0 for name in models}
    missing = set(models) - set(weights)
    extra = set(weights) - set(models)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing weights for {sorted(missing)}")
        if extra:
            details.append(f"unknown weights for {sorted(extra)}")
        raise ValueError("weights must match models exactly: " + "; ".join(details))
    return {name: float(weight) for name, weight in weights.items()}


def _native_model_from_wrapper(model: Any) -> Any:
    if model.__class__.__module__.endswith("._native"):
        return model
    new_native_model = getattr(model, "_new_native_model", None)
    if new_native_model is None:
        raise TypeError(
            "WeightedEnsembleForecaster only supports native CartoBoost forecasting wrappers"
        )
    return new_native_model()


def _validate_reconciliation_inputs(
    *,
    hierarchy: Mapping[str, Any] | None,
    summing_matrix: Any | None,
    series_id_column: str | None,
    parent_column: str | None,
    child_column: str | None,
) -> None:
    has_edges = parent_column is not None or child_column is not None
    if (parent_column is None) != (child_column is None):
        raise ValueError("parent_column and child_column must be provided together")
    if hierarchy is None and summing_matrix is None and not has_edges:
        raise ValueError(
            "reconciliation requires hierarchy, summing_matrix, or parent/child columns"
        )
    if series_id_column is not None and not series_id_column:
        raise ValueError("series_id_column must be non-empty")
    if parent_column is not None and not parent_column:
        raise ValueError("parent_column must be non-empty")
    if child_column is not None and not child_column:
        raise ValueError("child_column must be non-empty")
