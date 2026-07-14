from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .config import Kernel

try:  # pragma: no cover - exercised when sklearn is installed.
    from sklearn.base import BaseEstimator, RegressorMixin, clone
except ImportError:  # pragma: no cover

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    def clone(estimator: Any) -> Any:  # type: ignore[no-redef]
        return estimator


from ._artifacts import (
    ArtifactPersistenceMixin,
    dump_model_artifact,
    load_model_artifact,
    require_artifact_payload,
    versioned_artifact_payload,
)
from ._native import NearestNeighborGPRegressor as _NativeNearestNeighborGPRegressor
from ._native import geostats_empirical_semivariogram_value as _native_empirical_semivariogram
from ._native import geostats_fit_variogram_wls_value as _native_fit_variogram_wls


class NearestNeighborGPRegressor(ArtifactPersistenceMixin, RegressorMixin, BaseEstimator):
    """Nearest-neighbor Gaussian process regressor for point coordinates."""

    def __init__(
        self,
        kernel: Kernel = Kernel.EXPONENTIAL,
        range: float = 1.0,
        sill: float = 1.0,
        nugget: float = 1.0e-6,
        n_neighbors: int = 16,
        anisotropy_angle_degrees: float = 0.0,
        anisotropy_scaling: float = 1.0,
        brute_force_threshold: int = 2048,
        duplicate_tolerance: float = 0.0,
    ) -> None:
        self.kernel = kernel
        self.range = range
        self.sill = sill
        self.nugget = nugget
        self.n_neighbors = n_neighbors
        self.anisotropy_angle_degrees = anisotropy_angle_degrees
        self.anisotropy_scaling = anisotropy_scaling
        self.brute_force_threshold = brute_force_threshold
        self.duplicate_tolerance = duplicate_tolerance
        self._model: Any | None = None
        self._fit_coords: np.ndarray | None = None
        self._fit_y: np.ndarray | None = None

    def fit(
        self,
        X: Iterable[Iterable[float]] | None,
        y: Iterable[float],
        *,
        coords: Iterable[Iterable[float]],
    ) -> NearestNeighborGPRegressor:
        n_features = 0 if X is None else np.asarray(X).shape[1]
        coords_array = _as_coords(coords)
        y_array = _as_vector(y, "y")
        self._model = _NativeNearestNeighborGPRegressor(
            kernel=str(self.kernel),
            range=float(self.range),
            sill=float(self.sill),
            nugget=float(self.nugget),
            n_neighbors=int(self.n_neighbors),
            anisotropy_angle_degrees=float(self.anisotropy_angle_degrees),
            anisotropy_scaling=float(self.anisotropy_scaling),
            brute_force_threshold=int(self.brute_force_threshold),
            duplicate_tolerance=float(self.duplicate_tolerance),
        )
        self._model.fit(coords_array, y_array)
        self.n_features_in_ = n_features
        self._fit_coords = coords_array
        self._fit_y = y_array
        return self

    def predict(
        self,
        X: Iterable[Iterable[float]] | None,
        *,
        coords: Iterable[Iterable[float]],
        return_std: bool = False,
        return_var: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        del X
        model = self._require_model()
        means, variances, _neighbors = model.predict(_as_coords(coords))
        mean_array = np.asarray(means, dtype=float)
        var_array = np.maximum(np.asarray(variances, dtype=float), 0.0)
        if return_var:
            return mean_array, var_array
        if return_std:
            return mean_array, np.sqrt(var_array)
        return mean_array

    def predict_interval(
        self,
        X: Iterable[Iterable[float]] | None,
        *,
        coords: Iterable[Iterable[float]],
        coverage: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.predict(X, coords=coords, return_std=True)
        z = _normal_z_for_coverage(coverage)
        return mean - z * std, mean + z * std

    def score(
        self,
        X: Iterable[Iterable[float]] | None,
        y: Iterable[float],
        *,
        coords: Iterable[Iterable[float]],
    ) -> float:
        pred = np.asarray(self.predict(X, coords=coords), dtype=float)
        truth = _as_vector(y, "y")
        if pred.shape[0] != truth.shape[0]:
            raise ValueError("prediction and y must have the same length")
        return float(np.sqrt(np.mean((truth - pred) ** 2)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "kernel": self.kernel,
            "range": self.range,
            "sill": self.sill,
            "nugget": self.nugget,
            "n_neighbors": self.n_neighbors,
            "anisotropy_angle_degrees": self.anisotropy_angle_degrees,
            "anisotropy_scaling": self.anisotropy_scaling,
            "brute_force_threshold": self.brute_force_threshold,
            "duplicate_tolerance": self.duplicate_tolerance,
        }

    def set_params(self, **params: Any) -> NearestNeighborGPRegressor:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self._model = None
        self._fit_coords = None
        self._fit_y = None
        return self

    def save(self, path: str | Path) -> None:
        self._require_model()
        if self._fit_coords is None or self._fit_y is None:
            raise ValueError("fitted training data are unavailable for artifact serialization")
        payload = versioned_artifact_payload(
            self.__class__.__name__,
            params=self.get_params(),
            coords=self._fit_coords.tolist(),
            y=self._fit_y.tolist(),
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NearestNeighborGPRegressor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(
            payload,
            {
                "NearestNeighborGPRegressor",
                "SpatialGaussianProcessRegressor",
            },
        )
        if payload.get("artifact_type") not in {
            "NearestNeighborGPRegressor",
            "SpatialGaussianProcessRegressor",
        }:
            raise ValueError("artifact is not a NearestNeighborGPRegressor")
        obj = cls(**dict(payload["params"]))
        obj.fit(None, payload["y"], coords=payload["coords"])
        return obj

    def config_(self) -> dict[str, Any]:
        return json.loads(self._require_model().config_json())

    def _require_model(self) -> Any:
        if self._model is None:
            raise ValueError("NearestNeighborGPRegressor is not fitted")
        return self._model


class ResidualNNGPRegressor(ArtifactPersistenceMixin, RegressorMixin, BaseEstimator):
    """Fit any base estimator, then model spatial residuals with an NNGP."""

    def __init__(
        self,
        base_estimator: Any,
        *,
        gp: NearestNeighborGPRegressor | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.gp = gp

    def fit(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        *,
        coords: Iterable[Iterable[float]],
    ) -> ResidualNNGPRegressor:
        x_array = np.asarray(X, dtype=float)
        y_array = _as_vector(y, "y")
        try:
            self.base_estimator_ = clone(self.base_estimator)
        except Exception:
            self.base_estimator_ = self.base_estimator
        self.base_estimator_.fit(x_array, y_array)
        base_pred = np.asarray(self.base_estimator_.predict(x_array), dtype=float)
        residuals = y_array - base_pred
        self.gp_ = self.gp or NearestNeighborGPRegressor()
        self.gp_.fit(None, residuals, coords=coords)
        self.n_features_in_ = x_array.shape[1]
        return self

    def predict(
        self,
        X: Iterable[Iterable[float]],
        *,
        coords: Iterable[Iterable[float]],
        return_std: bool = False,
        return_var: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "base_estimator_"):
            raise ValueError("ResidualNNGPRegressor is not fitted")
        base_pred = np.asarray(
            self.base_estimator_.predict(np.asarray(X, dtype=float)), dtype=float
        )
        residual = self.gp_.predict(
            None, coords=coords, return_std=return_std, return_var=return_var
        )
        if return_std or return_var:
            residual_mean, uncertainty = residual
            return base_pred + residual_mean, uncertainty
        return base_pred + residual

    def predict_interval(
        self,
        X: Iterable[Iterable[float]],
        *,
        coords: Iterable[Iterable[float]],
        coverage: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.predict(X, coords=coords, return_std=True)
        z = _normal_z_for_coverage(coverage)
        return mean - z * std, mean + z * std

    def score(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        *,
        coords: Iterable[Iterable[float]],
    ) -> float:
        pred = np.asarray(self.predict(X, coords=coords), dtype=float)
        truth = _as_vector(y, "y")
        if pred.shape[0] != truth.shape[0]:
            raise ValueError("prediction and y must have the same length")
        return float(np.sqrt(np.mean((truth - pred) ** 2)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"base_estimator": self.base_estimator, "gp": self.gp}

    def set_params(self, **params: Any) -> ResidualNNGPRegressor:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def save(self, path: str | Path) -> None:
        if not hasattr(self, "base_estimator_") or not hasattr(self, "gp_"):
            raise ValueError("ResidualNNGPRegressor is not fitted")
        payload = versioned_artifact_payload(
            "ResidualNNGPRegressor",
            base_estimator=dump_model_artifact(
                self.base_estimator_,
                purpose="residual NNGP artifacts",
            ),
            gp=dump_model_artifact(self.gp_, purpose="residual NNGP artifacts"),
            n_features_in=int(self.n_features_in_),
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ResidualNNGPRegressor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "ResidualNNGPRegressor")
        obj = cls(
            load_model_artifact(payload["base_estimator"]),
            gp=load_model_artifact(payload["gp"]),
        )
        obj.base_estimator_ = obj.base_estimator
        obj.gp_ = obj.gp
        obj.n_features_in_ = int(payload["n_features_in"])
        return obj


class SpatialGaussianProcessRegressor(NearestNeighborGPRegressor):
    """Facade for scalable spatial Gaussian process regression."""


def empirical_semivariogram(
    coords: Iterable[Iterable[float]],
    values: Iterable[float],
    *,
    bin_count: int = 12,
    max_distance: float | None = None,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
) -> list[dict[str, Any]]:
    return list(
        json.loads(
            _native_empirical_semivariogram(
                _as_coords(coords),
                _as_vector(values, "values"),
                int(bin_count),
                max_distance,
                float(anisotropy_angle_degrees),
                float(anisotropy_scaling),
            )
        )
    )


def binned_variogram(
    coords: Iterable[Iterable[float]],
    values: Iterable[float],
    *,
    bin_count: int = 12,
    max_distance: float | None = None,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
) -> list[dict[str, Any]]:
    return empirical_semivariogram(
        coords,
        values,
        bin_count=bin_count,
        max_distance=max_distance,
        anisotropy_angle_degrees=anisotropy_angle_degrees,
        anisotropy_scaling=anisotropy_scaling,
    )


def fit_variogram_wls(
    bins: Iterable[dict[str, Any]],
    *,
    kernels: Iterable[str] = ("exponential", "squared_exponential", "matern_3_2", "matern_5_2"),
    range_candidates: Iterable[float],
    sill_candidates: Iterable[float],
    nugget_candidates: Iterable[float] = (0.0, 1.0e-6, 1.0e-4),
) -> dict[str, Any]:
    numeric_bins = [
        {
            "lag_start": float(row["lag_start"]),
            "lag_end": float(row["lag_end"]),
            "lag_center": float(row["lag_center"]),
            "semivariance": float(row["semivariance"]),
            "pair_count": float(row["pair_count"]),
        }
        for row in bins
    ]
    return dict(
        json.loads(
            _native_fit_variogram_wls(
                numeric_bins,
                list(kernels),
                [float(value) for value in range_candidates],
                [float(value) for value in sill_candidates],
                [float(value) for value in nugget_candidates],
            )
        )
    )


def _as_coords(coords: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if not np.isfinite(array).all():
        raise ValueError("coords must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def _as_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def _normal_z_for_coverage(coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    # Common uncertainty-map coverages without adding a scipy dependency.
    table = {0.5: 0.67448975, 0.8: 1.28155157, 0.9: 1.64485363, 0.95: 1.95996398, 0.99: 2.5758293}
    return table.get(round(float(coverage), 2), 1.95996398)
