from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._artifacts import ArtifactPersistenceMixin

try:  # pragma: no cover - exercised when sklearn is installed.
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:  # pragma: no cover

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass


from ._native import (
    SpatialDurbinRegressor as _NativeSpatialDurbinRegressor,
)
from ._native import (
    SpatialErrorRegressor as _NativeSpatialErrorRegressor,
)
from ._native import (
    SpatialLagRegressor as _NativeSpatialLagRegressor,
)
from ._native import (
    SpatialTwoStageLeastSquares as _NativeSpatialTwoStageLeastSquares,
)
from ._native import (
    SpatialWeights as _NativeSpatialWeights,
)


class SpatialWeights:
    """Sparse spatial weights for classical spatial regression."""

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        rows: Any,
        cols: Any,
        values: Any,
        *,
        row_standardize: bool = True,
    ) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.values = np.asarray(values, dtype=float)
        self.row_standardize = bool(row_standardize)
        if self.rows.ndim != 1 or self.cols.ndim != 1 or self.values.ndim != 1:
            raise ValueError("rows, cols, and values must be one-dimensional")
        self._native = _NativeSpatialWeights(
            self.n_rows,
            self.n_cols,
            self.rows.tolist(),
            self.cols.tolist(),
            self.values.tolist(),
            self.row_standardize,
        )

    @classmethod
    def from_neighbors(
        cls,
        neighbors: dict[int, list[int]],
        *,
        n_rows: int | None = None,
        row_standardize: bool = True,
    ) -> SpatialWeights:
        max_id = max(neighbors.keys(), default=-1)
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for row, row_neighbors in neighbors.items():
            max_id = max(max_id, row, max(row_neighbors, default=-1))
            for col in row_neighbors:
                rows.append(int(row))
                cols.append(int(col))
                values.append(1.0)
        size = int(n_rows) if n_rows is not None else max_id + 1
        return cls(size, size, rows, cols, values, row_standardize=row_standardize)

    @property
    def isolated_rows_(self) -> list[int]:
        return list(self._native.isolated_rows())


class _SpatialRegressionBase(ArtifactPersistenceMixin, RegressorMixin, BaseEstimator):
    _native_cls: type
    model_name: str

    def __init__(self, *, row_standardize: bool = True) -> None:
        self.row_standardize = row_standardize
        self._model: Any | None = None

    def fit(self, X: Any, y: Any, *, spatial_weights: Any) -> _SpatialRegressionBase:
        x_array = _as_2d_float_array(X, "X")
        y_array = _as_1d_float_array(y, "y")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        weights = _normalize_spatial_weights(
            spatial_weights,
            n_rows=x_array.shape[0],
            row_standardize=bool(self.row_standardize),
        )
        model = self._native_cls()
        model.fit(x_array.tolist(), y_array.tolist(), weights._native)
        self._model = model
        self.n_features_in_ = x_array.shape[1]
        self.n_samples_fit_ = x_array.shape[0]
        self.spatial_weights_isolated_rows_ = weights.isolated_rows_
        self.diagnostics_ = json.loads(model.diagnostics_json())
        self.coef_ = np.asarray(model.coefficients(), dtype=float)
        self.durbin_coef_ = np.asarray(model.durbin_coefficients(), dtype=float)
        self.intercept_ = float(model.intercept())
        return self

    def predict(self, X: Any, *, spatial_weights: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        x_array = _as_2d_float_array(X, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x_array.shape[1]} features, but model was fitted with "
                f"{self.n_features_in_}"
            )
        weights = _normalize_spatial_weights(
            spatial_weights,
            n_rows=x_array.shape[0],
            row_standardize=bool(self.row_standardize),
        )
        return np.asarray(self._model.predict(x_array.tolist(), weights._native), dtype=float)

    def summary(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        return {
            "model": self.model_name,
            "intercept": self.intercept_,
            "coefficients": self.coef_.tolist(),
            "durbin_coefficients": self.durbin_coef_.tolist(),
            "diagnostics": dict(self.diagnostics_),
        }

    def score(self, X: Any, y: Any, *, spatial_weights: Any) -> float:
        """Return the coefficient of determination (R-squared)."""
        pred = self.predict(X, spatial_weights=spatial_weights)
        truth = _as_1d_float_array(y, "y")
        if pred.shape[0] != truth.shape[0]:
            raise ValueError("prediction and y must contain the same number of rows")
        residual_sum_squares = float(np.sum((truth - pred) ** 2))
        total_sum_squares = float(np.sum((truth - np.mean(truth)) ** 2))
        if total_sum_squares == 0.0:
            return 1.0 if residual_sum_squares == 0.0 else 0.0
        return 1.0 - residual_sum_squares / total_sum_squares

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"row_standardize": self.row_standardize}

    def set_params(self, **params: Any) -> _SpatialRegressionBase:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        if self._model is None:
            return {
                "model": self.model_name,
                "params": self.get_params(),
                "fitted": False,
            }
        return {
            "model": self.model_name,
            "params": self.get_params(),
            "fitted": True,
            "diagnostics": dict(self.diagnostics_),
        }

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        self._model.save(Path(path))

    @classmethod
    def load(cls, path: str | Path) -> _SpatialRegressionBase:
        obj = cls()
        obj._model = cls._native_cls.load(Path(path))
        obj.diagnostics_ = json.loads(obj._model.diagnostics_json())
        obj.coef_ = np.asarray(obj._model.coefficients(), dtype=float)
        obj.durbin_coef_ = np.asarray(obj._model.durbin_coefficients(), dtype=float)
        obj.intercept_ = float(obj._model.intercept())
        obj.n_features_in_ = int(obj.diagnostics_["n_features"])
        obj.n_samples_fit_ = int(obj.diagnostics_["n_samples"])
        obj.spatial_weights_isolated_rows_ = list(obj.diagnostics_.get("isolated_rows", []))
        return obj


class SpatialLagRegressor(_SpatialRegressionBase):
    """Spatial lag baseline: ``y = rho W y + X beta + e``."""

    _native_cls = _NativeSpatialLagRegressor
    model_name = "SpatialLagRegressor"


class SpatialErrorRegressor(_SpatialRegressionBase):
    """Spatial error baseline: ``y = X beta + u``, ``u = lambda W u + e``."""

    _native_cls = _NativeSpatialErrorRegressor
    model_name = "SpatialErrorRegressor"


class SpatialDurbinRegressor(_SpatialRegressionBase):
    """Spatial Durbin baseline: ``y = rho W y + X beta + W X theta + e``."""

    _native_cls = _NativeSpatialDurbinRegressor
    model_name = "SpatialDurbinRegressor"


class SpatialTwoStageLeastSquares(_SpatialRegressionBase):
    """Sparse instrumental-style spatial lag baseline for small fixtures."""

    _native_cls = _NativeSpatialTwoStageLeastSquares
    model_name = "SpatialTwoStageLeastSquares"


def _normalize_spatial_weights(
    spatial_weights: Any,
    *,
    n_rows: int,
    row_standardize: bool,
) -> SpatialWeights:
    if isinstance(spatial_weights, SpatialWeights):
        return spatial_weights
    if not isinstance(spatial_weights, dict):
        raise ValueError(
            "spatial_weights must be a SpatialWeights instance or a COO dict with "
            "rows, cols, and values"
        )
    rows = spatial_weights.get("rows")
    cols = spatial_weights.get("cols")
    values = spatial_weights.get("values")
    if rows is None or cols is None:
        raise ValueError("spatial_weights dict must include rows and cols")
    if values is None:
        values = np.ones(len(rows), dtype=float)
    n_cols = int(spatial_weights.get("n_cols", spatial_weights.get("shape", [n_rows, n_rows])[1]))
    return SpatialWeights(
        int(spatial_weights.get("n_rows", n_rows)),
        n_cols,
        rows,
        cols,
        values,
        row_standardize=bool(spatial_weights.get("row_standardize", row_standardize)),
    )


def _as_2d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)
