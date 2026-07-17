"""Probabilistic helpers for quantiles, conformal intervals, and distributional reports."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cartoboost._artifacts import (
    dump_model_artifact,
    load_model_artifact,
    require_artifact_payload,
    versioned_artifact_payload,
)

from .._artifacts import ArtifactPersistenceMixin
from ..config import Backend


@dataclass(frozen=True)
class QuantileForecast:
    quantiles: tuple[float, ...]
    values: tuple[float, ...]

    def repaired(self) -> QuantileForecast:
        return QuantileForecast(self.quantiles, tuple(repair_non_crossing_quantiles(self.values)))


@dataclass(frozen=True)
class ConformalInterval:
    lower: np.ndarray
    upper: np.ndarray
    residual_quantile: float
    alpha: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributionalForecastResult:
    mean: np.ndarray
    median: np.ndarray | None = None
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    std: np.ndarray | None = None
    interval_lower: np.ndarray | None = None
    interval_upper: np.ndarray | None = None
    calibration_metadata: dict[str, Any] = field(default_factory=dict)

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("DistributionalForecastResult.to_pandas requires pandas") from exc
        data: dict[str, Any] = {"mean": self.mean}
        if self.median is not None:
            data["median"] = self.median
        if self.std is not None:
            data["std"] = self.std
        if self.interval_lower is not None:
            data["interval_lower"] = self.interval_lower
        if self.interval_upper is not None:
            data["interval_upper"] = self.interval_upper
        for level, values in sorted(self.quantiles.items()):
            data[f"quantile_{level:g}"] = values
        return pd.DataFrame(data)


class ConformalCalibrator:
    """Split-conformal calibration with explicit train/calibration/test ordering."""

    def __init__(self, *, alpha: float = 0.1) -> None:
        _validate_quantile(alpha, "alpha")
        self.alpha = float(alpha)
        self.residual_quantile_: float | None = None
        self._test_start: int | None = None

    def fit(
        self,
        calibration_actual: Any,
        calibration_prediction: Any,
        *,
        train_end_exclusive: int,
        calibration_start: int,
        calibration_end_exclusive: int,
        test_start: int,
    ) -> ConformalCalibrator:
        _validate_strict_ordering(
            train_end_exclusive,
            calibration_start,
            calibration_end_exclusive,
            test_start,
        )
        actual, prediction = _paired(
            calibration_actual,
            calibration_prediction,
            "calibration_actual",
            "calibration_prediction",
        )
        native = _native_prob_call(
            "prob_split_conformal_residual_quantile_value",
            actual.tolist(),
            prediction.tolist(),
            self.alpha,
            int(train_end_exclusive),
            int(calibration_start),
            int(calibration_end_exclusive),
            int(test_start),
        )
        self.residual_quantile_ = (
            float(native)
            if native is not None
            else _conformal_quantile(np.abs(actual - prediction), self.alpha)
        )
        self._test_start = int(test_start)
        return self

    def predict_interval(self, test_prediction: Any, *, test_start: int) -> ConformalInterval:
        if self.residual_quantile_ is None or self._test_start is None:
            raise ValueError("ConformalCalibrator must be fit before prediction")
        if int(test_start) < self._test_start:
            raise ValueError("test_start must not precede the calibrated test split")
        prediction = _vector(test_prediction, "test_prediction")
        q = self.residual_quantile_
        return ConformalInterval(
            lower=prediction - q,
            upper=prediction + q,
            residual_quantile=q,
            alpha=self.alpha,
            metadata={
                "method": "split_conformal",
                "test_start": int(test_start),
                "residual_quantile": q,
            },
        )


class ConformalIntervalRegressor(ArtifactPersistenceMixin):
    """Generic split-conformal wrapper for estimators exposing ``fit`` and ``predict``."""

    def __init__(self, estimator: Any, *, alpha: float = 0.1) -> None:
        _validate_estimator(estimator)
        _validate_quantile(alpha, "alpha")
        self.estimator = estimator
        self.alpha = float(alpha)
        self.calibrator_ = ConformalCalibrator(alpha=alpha)

    def fit(
        self,
        x_train: Any,
        y_train: Any,
        x_calibration: Any,
        y_calibration: Any,
        *,
        groups: Any | None = None,
        train_end_exclusive: int,
        calibration_start: int,
        calibration_end_exclusive: int,
        test_start: int,
    ) -> ConformalIntervalRegressor:
        del groups
        self.estimator.fit(x_train, y_train)
        calibration_prediction = self.estimator.predict(x_calibration)
        self.calibrator_.fit(
            y_calibration,
            calibration_prediction,
            train_end_exclusive=train_end_exclusive,
            calibration_start=calibration_start,
            calibration_end_exclusive=calibration_end_exclusive,
            test_start=test_start,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        return _vector(self.estimator.predict(x), "prediction")

    def predict_interval(self, x: Any, *, test_start: int) -> ConformalInterval:
        return self.calibrator_.predict_interval(self.predict(x), test_start=test_start)

    def score(self, x: Any, y: Any) -> float:
        pred = self.predict(x)
        truth = _vector(y, "y")
        if pred.shape[0] != truth.shape[0]:
            raise ValueError("prediction and y must have the same length")
        return float(np.sqrt(np.mean((truth - pred) ** 2)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"estimator": self.estimator, "alpha": self.alpha}

    def set_params(self, **params: Any) -> ConformalIntervalRegressor:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        if "alpha" in params:
            _validate_quantile(self.alpha, "alpha")
            self.calibrator_ = ConformalCalibrator(alpha=self.alpha)
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": self.__class__.__name__,
            "alpha": self.alpha,
            "fitted": self.calibrator_.residual_quantile_ is not None,
            "residual_quantile": self.calibrator_.residual_quantile_,
        }

    def save(self, path: str | Path) -> None:
        if self.calibrator_.residual_quantile_ is None:
            raise ValueError("ConformalIntervalRegressor is not fitted")
        payload = versioned_artifact_payload(
            self.__class__.__name__,
            alpha=self.alpha,
            estimator=dump_model_artifact(self.estimator, purpose="conformal artifacts"),
            residual_quantile=self.calibrator_.residual_quantile_,
            test_start=self.calibrator_._test_start,
        )
        if isinstance(self, SpatialConformalRegressor):
            payload["group_residual_quantiles"] = dict(self.group_residual_quantiles_)
            payload["backend"] = self.backend
            payload["neighbor_count"] = self.neighbor_count
            payload["calibration_actual"] = (
                None if self.calibration_actual_ is None else self.calibration_actual_.tolist()
            )
            payload["calibration_prediction"] = (
                None
                if self.calibration_prediction_ is None
                else self.calibration_prediction_.tolist()
            )
            payload["calibration_coordinates"] = (
                None
                if self.calibration_coordinates_ is None
                else self.calibration_coordinates_.tolist()
            )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ConformalIntervalRegressor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact_type = require_artifact_payload(
            payload,
            {"ConformalIntervalRegressor", "SpatialConformalRegressor"},
        )
        target_cls: type[ConformalIntervalRegressor] = (
            SpatialConformalRegressor if artifact_type == "SpatialConformalRegressor" else cls
        )
        if target_cls is SpatialConformalRegressor:
            obj = target_cls(
                load_model_artifact(payload["estimator"]),
                alpha=float(payload["alpha"]),
                neighbor_count=int(payload.get("neighbor_count", 16)),
                backend=str(payload.get("backend", Backend.CPU.value)),
            )
        else:
            obj = target_cls(
                load_model_artifact(payload["estimator"]),
                alpha=float(payload["alpha"]),
            )
        obj.calibrator_.residual_quantile_ = float(payload["residual_quantile"])
        obj.calibrator_._test_start = int(payload["test_start"])
        if isinstance(obj, SpatialConformalRegressor):
            obj.group_residual_quantiles_ = {
                str(key): float(value)
                for key, value in payload.get("group_residual_quantiles", {}).items()
            }
            obj.calibration_actual_ = _optional_vector(payload.get("calibration_actual"))
            obj.calibration_prediction_ = _optional_vector(payload.get("calibration_prediction"))
            coordinates = payload.get("calibration_coordinates")
            obj.calibration_coordinates_ = (
                None
                if coordinates is None
                else _coordinate_matrix(coordinates, "calibration_coordinates")
            )
        return obj


class QuantileCartoBoostRegressor(ArtifactPersistenceMixin):
    """Train one CartoBoost quantile regressor per requested level."""

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        backend: Backend | str = Backend.CPU,
        **kwargs: Any,
    ) -> None:
        self.quantiles = tuple(float(q) for q in quantiles)
        _validate_quantile_grid(self.quantiles)
        self.backend = str(backend)
        self.kwargs = dict(kwargs)
        self.models_: dict[float, Any] = {}
        self._native_model: Any | None = None
        from cartoboost import CartoBoostRegressor

        if (
            CartoBoostRegressor.__module__ != "cartoboost.regressor"
            or _native_quantile_regressor_class() is None
        ):
            self._build_models()

    def _build_models(self) -> None:
        from cartoboost import CartoBoostRegressor

        self.models_ = {
            q: CartoBoostRegressor(
                loss="quantile",
                quantile_alpha=q,
                backend=self.backend,
                **self.kwargs,
            )
            for q in self.quantiles
        }

    def fit(self, x: Any, y: Any) -> QuantileCartoBoostRegressor:
        native_class = _native_quantile_regressor_class()
        native_params = _native_quantile_params(self.kwargs)
        if not self.models_ and native_class is not None and native_params is not None:
            try:
                dense = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
                targets = np.ascontiguousarray(_vector(y, "y"), dtype=np.float64)
                if dense.ndim != 2:
                    raise ValueError("X must be a two-dimensional numeric matrix")
                if dense.shape[0] != targets.shape[0]:
                    raise ValueError("X and y must contain the same number of rows")
                native = native_class(
                    quantiles=list(self.quantiles),
                    backend=self.backend,
                    **native_params,
                )
                native.fit(dense, targets)
                self._native_model = native
                self.models_ = {}
                self.selected_backend_ = str(getattr(native, "selected_backend", self.backend))
                return self
            except (TypeError, ValueError):
                # Categorical/mixed inputs and options outside the native set's
                # contract retain the full per-estimator compatibility path.
                self._native_model = None
        if not self.models_:
            self._build_models()
        for model in self.models_.values():
            model.fit(x, y)
        selected = {
            str(getattr(model, "selected_backend_", self.backend))
            for model in self.models_.values()
        }
        self.selected_backend_ = next(iter(selected)) if len(selected) == 1 else self.backend
        return self

    def predict(self, x: Any) -> np.ndarray:
        if 0.5 in self.models_:
            return _vector(self.models_[0.5].predict(x), "prediction")
        columns = self.predict_quantiles(x)
        return columns[:, len(self.quantiles) // 2]

    def predict_quantiles(self, x: Any) -> np.ndarray:
        if self._native_model is not None:
            dense = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
            if dense.ndim != 2:
                raise ValueError("X must be a two-dimensional numeric matrix")
            return np.asarray(self._native_model.predict_quantiles(dense), dtype=float)
        columns = [_vector(self.models_[q].predict(x), "prediction") for q in self.quantiles]
        return np.maximum.accumulate(np.column_stack(columns), axis=1)

    def predict_distribution(self, x: Any) -> DistributionalForecastResult:
        matrix = self.predict_quantiles(x)
        quantiles = {q: matrix[:, idx] for idx, q in enumerate(self.quantiles)}
        median = quantiles.get(0.5)
        return DistributionalForecastResult(
            mean=median if median is not None else matrix[:, len(self.quantiles) // 2],
            median=median,
            quantiles=quantiles,
            interval_lower=matrix[:, 0],
            interval_upper=matrix[:, -1],
            calibration_metadata={"method": "quantile_cartoboost"},
        )

    def score(self, x: Any, y: Any) -> float:
        pred = self.predict(x)
        truth = _vector(y, "y")
        if pred.shape[0] != truth.shape[0]:
            raise ValueError("prediction and y must have the same length")
        return float(np.sqrt(np.mean((truth - pred) ** 2)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "quantiles": self.quantiles,
            "backend": self.backend,
            **dict(self.kwargs),
        }

    def set_params(self, **params: Any) -> QuantileCartoBoostRegressor:
        quantiles = params.pop("quantiles", self.quantiles)
        backend = params.pop("backend", self.backend)
        self.__init__(quantiles=quantiles, backend=backend, **{**self.kwargs, **params})
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": "QuantileCartoBoostRegressor",
            "quantiles": list(self.quantiles),
            "backend": {
                "requested": self.backend,
                "selected": (
                    {
                        str(q): str(getattr(self._native_model, "selected_backend", self.backend))
                        for q in self.quantiles
                    }
                    if self._native_model is not None
                    else {
                        str(q): str(
                            getattr(
                                getattr(model, "_model", None),
                                "selected_backend_",
                                getattr(model, "backend", self.backend),
                            )
                        )
                        for q, model in self.models_.items()
                    }
                ),
            },
            "params": dict(self.kwargs),
        }

    def save(self, path: str | Path) -> None:
        payload = versioned_artifact_payload(
            "QuantileCartoBoostRegressor",
            quantiles=list(self.quantiles),
            backend=self.backend,
            kwargs=dict(self.kwargs),
            native_model=(None if self._native_model is None else str(self._native_model.dumps())),
            models=(
                {
                    str(q): dump_model_artifact(model, purpose="quantile artifacts")
                    for q, model in self.models_.items()
                }
                if self._native_model is None
                else {}
            ),
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> QuantileCartoBoostRegressor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "QuantileCartoBoostRegressor")
        obj = cls(
            quantiles=tuple(float(q) for q in payload["quantiles"]),
            backend=str(payload.get("backend", Backend.CPU.value)),
            **dict(payload["kwargs"]),
        )
        native_payload = payload.get("native_model")
        native_class = _native_quantile_regressor_class()
        if native_payload is not None and native_class is not None:
            obj._native_model = native_class.loads(str(native_payload))
            obj.models_ = {}
            obj.selected_backend_ = str(getattr(obj._native_model, "selected_backend", obj.backend))
        else:
            obj._native_model = None
            obj.models_ = {
                float(level): load_model_artifact(model_payload)
                for level, model_payload in payload["models"].items()
            }
            selected = {
                str(getattr(model, "selected_backend_", obj.backend))
                for model in obj.models_.values()
            }
            obj.selected_backend_ = next(iter(selected)) if len(selected) == 1 else obj.backend
        return obj


class SpatialConformalRegressor(ConformalIntervalRegressor):
    def __init__(
        self,
        estimator: Any,
        *,
        alpha: float = 0.1,
        neighbor_count: int = 16,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        super().__init__(estimator, alpha=alpha)
        if int(neighbor_count) <= 0:
            raise ValueError("neighbor_count must be positive")
        self.neighbor_count = int(neighbor_count)
        self.backend = str(backend)
        self.group_residual_quantiles_: dict[str, float] = {}
        self.calibration_actual_: np.ndarray | None = None
        self.calibration_prediction_: np.ndarray | None = None
        self.calibration_coordinates_: np.ndarray | None = None

    def fit(
        self,
        x_train: Any,
        y_train: Any,
        x_calibration: Any,
        y_calibration: Any,
        *,
        groups: Any | None = None,
        calibration_coordinates: Any | None = None,
        train_end_exclusive: int,
        calibration_start: int,
        calibration_end_exclusive: int,
        test_start: int,
    ) -> SpatialConformalRegressor:
        super().fit(
            x_train,
            y_train,
            x_calibration,
            y_calibration,
            train_end_exclusive=train_end_exclusive,
            calibration_start=calibration_start,
            calibration_end_exclusive=calibration_end_exclusive,
            test_start=test_start,
        )
        actual = _vector(y_calibration, "y_calibration")
        prediction = _vector(self.estimator.predict(x_calibration), "calibration_prediction")
        if groups is None and calibration_coordinates is None:
            raise ValueError(
                "groups or calibration_coordinates are required for spatial conformal calibration"
            )
        if groups is not None:
            group_arr = np.asarray(groups)
            if group_arr.shape[0] != actual.shape[0]:
                raise ValueError("groups length must match calibration rows")
            residuals = np.abs(actual - prediction)
            self.group_residual_quantiles_ = {
                str(group): _conformal_quantile(residuals[group_arr == group], self.alpha)
                for group in np.unique(group_arr)
            }
        if calibration_coordinates is not None:
            coordinates = _coordinate_matrix(
                calibration_coordinates,
                "calibration_coordinates",
            )
            if coordinates.shape[0] != actual.shape[0]:
                raise ValueError("calibration_coordinates row count must match calibration rows")
            self.calibration_actual_ = actual
            self.calibration_prediction_ = prediction
            self.calibration_coordinates_ = coordinates
        return self

    def predict_interval(
        self,
        x: Any,
        *,
        test_start: int,
        groups: Any | None = None,
        coordinates: Any | None = None,
    ) -> ConformalInterval:
        base = super().predict_interval(x, test_start=test_start)
        if coordinates is not None:
            if (
                self.calibration_actual_ is None
                or self.calibration_prediction_ is None
                or self.calibration_coordinates_ is None
            ):
                raise ValueError(
                    "fit with calibration_coordinates before coordinate-local prediction"
                )
            prediction = self.predict(x)
            quantiles = nearest_conformal_residual_quantiles(
                self.calibration_actual_,
                self.calibration_prediction_,
                self.calibration_coordinates_,
                coordinates,
                neighbor_count=self.neighbor_count,
                alpha=self.alpha,
                backend=self.backend,
            )
            if quantiles.shape[0] != prediction.shape[0]:
                raise ValueError("coordinates length must match prediction rows")
            return ConformalInterval(
                lower=prediction - quantiles,
                upper=prediction + quantiles,
                residual_quantile=base.residual_quantile,
                alpha=base.alpha,
                metadata={
                    **base.metadata,
                    "method": "spatial_nearest_conformal",
                    "backend": self.backend,
                    "neighbor_count": self.neighbor_count,
                },
            )
        if groups is None:
            return base
        group_arr = np.asarray(groups)
        if group_arr.shape[0] != base.lower.shape[0]:
            raise ValueError("groups length must match prediction rows")
        prediction = self.predict(x)
        lower = base.lower.copy()
        upper = base.upper.copy()
        for idx, group in enumerate(group_arr):
            q = self.group_residual_quantiles_.get(str(group), base.residual_quantile)
            lower[idx] = prediction[idx] - q
            upper[idx] = prediction[idx] + q
        return ConformalInterval(
            lower=lower,
            upper=upper,
            residual_quantile=base.residual_quantile,
            alpha=base.alpha,
            metadata={**base.metadata, "method": "spatial_group_conformal"},
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "estimator": self.estimator,
            "alpha": self.alpha,
            "neighbor_count": self.neighbor_count,
            "backend": self.backend,
        }

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            **super().metadata_,
            "backend": self.backend,
            "neighbor_count": self.neighbor_count,
            "coordinate_calibration": self.calibration_coordinates_ is not None,
            "group_calibration": bool(self.group_residual_quantiles_),
        }


class ForecastConformalCalibrator:
    """Rolling-origin conformal calibration that uses only residuals before each cutoff."""

    def __init__(self, *, alpha: float = 0.1) -> None:
        _validate_quantile(alpha, "alpha")
        self.alpha = float(alpha)
        self.actual_: np.ndarray | None = None
        self.prediction_: np.ndarray | None = None
        self.cutoff_: np.ndarray | None = None

    def fit(self, actual: Any, prediction: Any, cutoff_index: Any) -> ForecastConformalCalibrator:
        actual_arr, prediction_arr = _paired(actual, prediction, "actual", "prediction")
        cutoff_arr = np.asarray(cutoff_index, dtype=int)
        if cutoff_arr.ndim != 1 or cutoff_arr.shape[0] != actual_arr.shape[0]:
            raise ValueError("cutoff_index must be one-dimensional and match actual length")
        if np.any(cutoff_arr < 0):
            raise ValueError("cutoff_index must be non-negative")
        self.actual_ = actual_arr
        self.prediction_ = prediction_arr
        self.cutoff_ = cutoff_arr
        return self

    def residual_quantile_for_cutoff(self, cutoff: int) -> float:
        if self.actual_ is None or self.prediction_ is None or self.cutoff_ is None:
            raise ValueError("ForecastConformalCalibrator must be fit before prediction")
        mask = self.cutoff_ < int(cutoff)
        if not np.any(mask):
            raise ValueError("forecast conformal calibration requires past cutoff residuals")
        return _conformal_quantile(np.abs(self.actual_[mask] - self.prediction_[mask]), self.alpha)

    def predict_interval(
        self,
        prediction: Any,
        *,
        cutoff: int,
    ) -> ConformalInterval:
        pred = _vector(prediction, "prediction")
        q = self.residual_quantile_for_cutoff(cutoff)
        return ConformalInterval(
            lower=pred - q,
            upper=pred + q,
            residual_quantile=q,
            alpha=self.alpha,
            metadata={"method": "rolling_origin_conformal", "cutoff": int(cutoff)},
        )


def pinball_loss(
    y_true: Any,
    y_pred: Any,
    quantile: float,
    *,
    backend: Backend | str = Backend.CPU,
) -> float:
    _validate_quantile(quantile, "quantile")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    native = _native_prob_call(
        "prob_pinball_loss_value",
        truth.tolist(),
        pred.tolist(),
        float(quantile),
        str(backend),
        None,
    )
    if native is not None:
        return float(native)
    residual = truth - pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def interval_coverage(
    y_true: Any,
    lower: Any,
    upper: Any,
    *,
    sample_weight: Any | None = None,
    backend: Backend | str = Backend.CPU,
) -> float:
    truth, lower_arr = _paired(y_true, lower, "y_true", "lower")
    _, upper_arr = _paired(y_true, upper, "y_true", "upper")
    _validate_interval_bounds(lower_arr, upper_arr)
    weights = _optional_vector(sample_weight)
    if weights is not None and weights.shape != truth.shape:
        raise ValueError("sample_weight must have the same length as y_true")
    native = _native_prob_call(
        "prob_interval_coverage_value",
        truth.tolist(),
        lower_arr.tolist(),
        upper_arr.tolist(),
        str(backend),
        None if weights is None else weights.tolist(),
    )
    if native is not None:
        return float(native)
    covered = ((truth >= lower_arr) & (truth <= upper_arr)).astype(float)
    return float(np.mean(covered) if weights is None else np.average(covered, weights=weights))


def mean_interval_width(
    lower: Any,
    upper: Any,
    *,
    sample_weight: Any | None = None,
    backend: Backend | str = Backend.CPU,
) -> float:
    lower_arr, upper_arr = _paired(lower, upper, "lower", "upper")
    _validate_interval_bounds(lower_arr, upper_arr)
    weights = _optional_vector(sample_weight)
    if weights is not None and weights.shape != lower_arr.shape:
        raise ValueError("sample_weight must have the same length as lower")
    native = _native_prob_call(
        "prob_mean_interval_width_value",
        lower_arr.tolist(),
        upper_arr.tolist(),
        str(backend),
        None if weights is None else weights.tolist(),
    )
    if native is not None:
        return float(native)
    widths = upper_arr - lower_arr
    return float(np.mean(widths) if weights is None else np.average(widths, weights=weights))


def weighted_conformal_residual_quantile(
    y_true: Any,
    y_pred: Any,
    weights: Any,
    alpha: float,
) -> float:
    _validate_quantile(alpha, "alpha")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    _, weight_arr = _paired(y_true, weights, "y_true", "weights")
    if np.any(weight_arr <= 0.0):
        raise ValueError("weights must be positive")
    native = _native_prob_call(
        "prob_weighted_conformal_residual_quantile_value",
        truth.tolist(),
        pred.tolist(),
        weight_arr.tolist(),
        float(alpha),
        1,
        1,
        int(truth.shape[0] + 1),
        int(truth.shape[0] + 1),
    )
    if native is not None:
        return float(native)
    order = np.argsort(np.abs(truth - pred), kind="mergesort")
    scores = np.abs(truth - pred)[order]
    sorted_weights = weight_arr[order]
    threshold = (1.0 - float(alpha)) * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(scores[min(index, scores.size - 1)])


def group_conformal_residual_quantiles(
    y_true: Any,
    y_pred: Any,
    groups: Any,
    alpha: float,
) -> dict[str, float]:
    _validate_quantile(alpha, "alpha")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    group_arr = np.asarray(groups)
    if group_arr.ndim != 1 or group_arr.shape[0] != truth.shape[0]:
        raise ValueError("groups must be one-dimensional and match y_true length")
    native = _native_prob_call(
        "prob_group_conformal_residual_quantiles_value",
        truth.tolist(),
        pred.tolist(),
        [str(group) for group in group_arr],
        float(alpha),
        1,
        1,
        int(truth.shape[0] + 1),
        int(truth.shape[0] + 1),
    )
    if native is not None:
        return {str(key): float(value) for key, value in json.loads(str(native)).items()}
    result = {}
    for group in np.unique(group_arr):
        mask = group_arr == group
        result[str(group)] = _conformal_quantile(np.abs(truth[mask] - pred[mask]), alpha)
    return result


def nearest_conformal_residual_quantiles(
    y_true: Any,
    y_pred: Any,
    calibration_coordinates: Any,
    query_coordinates: Any,
    *,
    neighbor_count: int,
    alpha: float,
    backend: Backend | str = Backend.CPU,
) -> np.ndarray:
    _validate_quantile(alpha, "alpha")
    if int(neighbor_count) <= 0:
        raise ValueError("neighbor_count must be positive")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    calibration_xy = _coordinate_matrix(calibration_coordinates, "calibration_coordinates")
    query_xy = _coordinate_matrix(query_coordinates, "query_coordinates")
    if calibration_xy.shape[0] != truth.shape[0]:
        raise ValueError("calibration_coordinates row count must match y_true length")
    native = _native_prob_call(
        "prob_nearest_conformal_residual_quantiles_value",
        truth.tolist(),
        pred.tolist(),
        calibration_xy[:, 0].tolist(),
        calibration_xy[:, 1].tolist(),
        query_xy[:, 0].tolist(),
        query_xy[:, 1].tolist(),
        int(neighbor_count),
        float(alpha),
        1,
        1,
        int(truth.shape[0] + 1),
        int(truth.shape[0] + 1),
        str(backend),
    )
    if native is not None:
        return np.asarray(native, dtype=float)
    residuals = np.abs(truth - pred)
    result = []
    for query in query_xy:
        distances = np.sum((calibration_xy - query) ** 2, axis=1)
        nearest = np.argsort(distances, kind="mergesort")[
            : min(int(neighbor_count), residuals.size)
        ]
        result.append(_conformal_quantile(residuals[nearest], alpha))
    return np.asarray(result, dtype=float)


def benchmark_calibration_report_fields(
    y_true: Any,
    lower: Any,
    upper: Any,
    horizons: Any,
    spatial_blocks: Any,
    *,
    residual_morans_i_after_calibration: float | None = None,
) -> dict[str, Any]:
    truth, lower_arr = _paired(y_true, lower, "y_true", "lower")
    _, upper_arr = _paired(y_true, upper, "y_true", "upper")
    _validate_interval_bounds(lower_arr, upper_arr)
    horizon_arr = np.asarray(horizons, dtype=int)
    block_arr = np.asarray(spatial_blocks)
    if horizon_arr.ndim != 1 or horizon_arr.shape[0] != truth.shape[0]:
        raise ValueError("horizons must be one-dimensional and match y_true length")
    if block_arr.ndim != 1 or block_arr.shape[0] != truth.shape[0]:
        raise ValueError("spatial_blocks must be one-dimensional and match y_true length")
    if residual_morans_i_after_calibration is not None and not np.isfinite(
        float(residual_morans_i_after_calibration)
    ):
        raise ValueError("residual_morans_i_after_calibration must be finite when provided")
    native = _native_prob_call(
        "prob_benchmark_calibration_report_fields_value",
        truth.tolist(),
        lower_arr.tolist(),
        upper_arr.tolist(),
        [int(value) for value in horizon_arr],
        [str(value) for value in block_arr],
        None
        if residual_morans_i_after_calibration is None
        else float(residual_morans_i_after_calibration),
    )
    if native is not None:
        decoded = json.loads(str(native))
        return {
            "coverage_by_horizon": {
                int(key): float(value) for key, value in decoded["coverage_by_horizon"].items()
            },
            "coverage_by_spatial_block": {
                str(key): float(value)
                for key, value in decoded["coverage_by_spatial_block"].items()
            },
            "width_by_horizon": {
                int(key): float(value) for key, value in decoded["width_by_horizon"].items()
            },
            "residual_morans_i_after_calibration": decoded["residual_morans_i_after_calibration"],
        }
    return {
        "coverage_by_horizon": {
            int(horizon): interval_coverage(
                truth[horizon_arr == horizon],
                lower_arr[horizon_arr == horizon],
                upper_arr[horizon_arr == horizon],
            )
            for horizon in sorted(np.unique(horizon_arr))
        },
        "coverage_by_spatial_block": {
            str(block): interval_coverage(
                truth[block_arr == block],
                lower_arr[block_arr == block],
                upper_arr[block_arr == block],
            )
            for block in sorted(np.unique(block_arr), key=str)
        },
        "width_by_horizon": {
            int(horizon): mean_interval_width(
                lower_arr[horizon_arr == horizon],
                upper_arr[horizon_arr == horizon],
            )
            for horizon in sorted(np.unique(horizon_arr))
        },
        "residual_morans_i_after_calibration": (
            None
            if residual_morans_i_after_calibration is None
            else float(residual_morans_i_after_calibration)
        ),
    }


def crps_approximation(
    y_true: Any,
    quantiles: Any,
    predictions: Any,
    *,
    backend: Backend | str = Backend.CPU,
) -> float:
    truth, levels, matrix = _quantile_matrix(y_true, quantiles, predictions)
    native = _native_prob_call(
        "prob_crps_approximation_value",
        truth.tolist(),
        levels.tolist(),
        matrix.tolist(),
        str(backend),
    )
    if native is not None:
        return float(native)
    total = 0.0
    for idx, level in enumerate(levels):
        total += 2.0 * pinball_loss(
            truth,
            matrix[:, idx],
            float(level),
            backend=backend,
        )
    return float(total / len(levels))


def weighted_interval_score(
    y_true: Any,
    median: Any,
    intervals: list[tuple[float, Any, Any]],
    *,
    backend: Backend | str = Backend.CPU,
) -> float:
    truth, med = _paired(y_true, median, "y_true", "median")
    if not intervals:
        raise ValueError("intervals must contain at least one interval")
    native_intervals = []
    total = 0.5 * np.abs(truth - med)
    weight_sum = 0.5
    for alpha, lower, upper in intervals:
        _validate_quantile(alpha, "alpha")
        lower_arr, upper_arr = _paired(lower, upper, "lower", "upper")
        _validate_interval_bounds(lower_arr, upper_arr)
        native_intervals.append((float(alpha), lower_arr.tolist(), upper_arr.tolist()))
        below = np.maximum(lower_arr - truth, 0.0) * 2.0 / float(alpha)
        above = np.maximum(truth - upper_arr, 0.0) * 2.0 / float(alpha)
        weight = float(alpha) / 2.0
        weight_sum += weight
        total += weight * (upper_arr - lower_arr + below + above)
    native = _native_prob_call(
        "prob_weighted_interval_score_value",
        truth.tolist(),
        med.tolist(),
        native_intervals,
        str(backend),
    )
    if native is not None:
        return float(native)
    return float(np.mean(total / weight_sum))


def pit_bins(
    y_true: Any,
    quantiles: Any,
    predictions: Any,
    *,
    bins: int = 10,
    backend: Backend | str = Backend.CPU,
) -> dict[str, Any]:
    if int(bins) <= 0:
        raise ValueError("bins must be positive")
    truth, levels, matrix = _quantile_matrix(y_true, quantiles, predictions)
    native = _native_prob_call(
        "prob_pit_bins_value",
        truth.tolist(),
        levels.tolist(),
        matrix.tolist(),
        int(bins),
        str(backend),
    )
    if native is not None:
        return json.loads(str(native))
    pit = np.zeros(truth.shape[0], dtype=float)
    for idx, level in enumerate(levels):
        pit = np.where(truth >= matrix[:, idx], float(level), pit)
    counts, edges = np.histogram(pit, bins=int(bins), range=(0.0, 1.0))
    return {"edges": edges.tolist(), "counts": counts.astype(int).tolist()}


def repair_non_crossing_quantiles(values: Any) -> np.ndarray:
    arr = _vector(values, "values")
    if arr.size == 0:
        raise ValueError("values must contain at least one quantile prediction")
    return np.maximum.accumulate(arr)


def rank_probability_score(probabilities: Any, observed_rank: int) -> float:
    probs = _probabilities(probabilities)
    rank = int(observed_rank)
    if rank < 0 or rank >= probs.size:
        raise ValueError("observed_rank must be a zero-based index inside probabilities")
    if probs.size == 1:
        return 0.0
    predicted_cdf = np.cumsum(probs[:-1])
    observed_cdf = (rank <= np.arange(probs.size - 1)).astype(float)
    return float(np.mean((predicted_cdf - observed_cdf) ** 2))


def _validate_strict_ordering(
    train_end_exclusive: int,
    calibration_start: int,
    calibration_end_exclusive: int,
    test_start: int,
) -> None:
    if int(train_end_exclusive) <= 0:
        raise ValueError("training split must contain at least one row")
    if int(train_end_exclusive) > int(calibration_start):
        raise ValueError("training rows must end before calibration rows start")
    if int(calibration_start) >= int(calibration_end_exclusive):
        raise ValueError("calibration split must contain at least one row")
    if int(calibration_end_exclusive) > int(test_start):
        raise ValueError("calibration rows must end before test rows start")


def _validate_quantile(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1)")


def _validate_quantile_grid(values: tuple[float, ...]) -> None:
    if not values:
        raise ValueError("quantiles must contain at least one level")
    previous = -np.inf
    for value in values:
        _validate_quantile(value, "quantile")
        if value <= previous:
            raise ValueError("quantiles must be strictly increasing")
        previous = value


def _paired(
    left: Any, right: Any, left_name: str, right_name: str
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


def _optional_vector(values: Any) -> np.ndarray | None:
    return None if values is None else _vector(values, "artifact_vector")


def _coordinate_matrix(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be a two-dimensional coordinate matrix with two columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _probabilities(values: Any) -> np.ndarray:
    arr = _vector(values, "probabilities")
    if arr.size == 0:
        raise ValueError("probabilities must contain at least one rank")
    if np.any(arr < 0.0):
        raise ValueError("probabilities must be non-negative")
    if not np.isclose(float(np.sum(arr)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("probabilities must sum to 1")
    return arr


def _validate_interval_bounds(lower: np.ndarray, upper: np.ndarray) -> None:
    if np.any(lower > upper):
        raise ValueError("lower bounds must be less than or equal to upper bounds")


def _conformal_quantile(residuals: Any, alpha: float) -> float:
    scores = np.sort(_vector(residuals, "residuals"))
    if scores.size == 0:
        raise ValueError("residuals must contain at least one value")
    rank = int(np.ceil((scores.size + 1) * (1.0 - float(alpha))))
    return float(scores[min(max(rank - 1, 0), scores.size - 1)])


def _quantile_matrix(
    y_true: Any, quantiles: Any, predictions: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = _vector(y_true, "y_true")
    levels = _vector(quantiles, "quantiles")
    _validate_quantile_grid(tuple(float(q) for q in levels))
    matrix = np.asarray(predictions, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("predictions must be a two-dimensional quantile matrix")
    if matrix.shape != (truth.shape[0], levels.shape[0]):
        raise ValueError("predictions shape must be (n_rows, n_quantiles)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("predictions must contain only finite values")
    return truth, levels, matrix


def _validate_estimator(estimator: Any) -> None:
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError("estimator must expose fit and predict methods")


def _native_prob_call(name: str, *args: Any) -> Any | None:
    try:
        native = importlib.import_module("cartoboost._native")
        function = getattr(native, name)
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None
    return function(*args)


def _native_quantile_regressor_class() -> Any | None:
    try:
        native = importlib.import_module("cartoboost._native")
        return native.QuantileRegressorSet
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None


def _native_quantile_params(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    unsupported = {
        "random_state",
        "tensorboard_log_dir",
        "tensorboard_run_name",
        "loss",
        "quantile_alpha",
        "loss_params",
        "huber_delta",
        "log_offset",
    }
    if any(key in kwargs and kwargs[key] is not None for key in unsupported):
        return None
    supported = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "min_gain",
        "leaf_predictor",
        "linear_leaf_features",
        "l2_regularization",
        "constant_l2_regularization",
        "fuzzy",
        "fuzzy_bandwidth",
        "fuzzy_kernel",
        "n_threads",
        "monotonic_constraints",
    }
    if any(key not in supported | {"split_policy", "splitters"} for key in kwargs):
        return None
    params = {key: value for key, value in kwargs.items() if key in supported}
    linear_features = params.get("linear_leaf_features")
    if linear_features is not None and any(
        not isinstance(value, (int, np.integer)) for value in linear_features
    ):
        return None
    if linear_features is not None:
        params["linear_leaf_features"] = [int(value) for value in linear_features]
    for enum_name in ("leaf_predictor", "fuzzy_kernel"):
        value = params.get(enum_name)
        if value is not None:
            params[enum_name] = getattr(value, "value", str(value))
    splitters = kwargs.get("splitters")
    if splitters is None and "split_policy" in kwargs:
        policy = getattr(kwargs["split_policy"], "value", str(kwargs["split_policy"]))
        splitters = [str(policy)]
    if splitters is not None:
        params["splitters"] = [str(getattr(value, "value", value)) for value in splitters]
    return params
