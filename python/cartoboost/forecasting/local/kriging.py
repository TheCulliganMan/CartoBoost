from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...config import Backend, Drift, Kernel
from .._native_wrappers import NativeForecastWrapper

CoordinateInput = Mapping[str, Sequence[float]] | Sequence[tuple[str, float, float]]


class KrigingForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust ordinary kriging forecasting binding."""

    native_class_name = "KrigingForecaster"

    def __init__(
        self,
        coordinates: CoordinateInput,
        range: float = 1.0,
        nugget: float = 1.0e-6,
        sill: float = 1.0,
        variogram_model: Kernel = Kernel.EXPONENTIAL,
        drift: Drift = Drift.ORDINARY,
        anisotropy_angle_degrees: float = 0.0,
        anisotropy_scaling: float = 1.0,
        max_neighbors: int | None = None,
        min_neighbors: int = 1,
        max_distance: float | None = None,
        backend: Backend | str = Backend.CPU,
        variogram_fit_policy: Mapping[str, Any] | None = None,
        **params: Any,
    ) -> None:
        coordinate_rows = _normalize_coordinates(coordinates)
        super().__init__(
            coordinates=coordinate_rows,
            range=float(range),
            nugget=float(nugget),
            sill=float(sill),
            variogram_model=str(variogram_model),
            drift=str(drift),
            anisotropy_angle_degrees=float(anisotropy_angle_degrees),
            anisotropy_scaling=float(anisotropy_scaling),
            max_neighbors=None if max_neighbors is None else int(max_neighbors),
            min_neighbors=int(min_neighbors),
            max_distance=None if max_distance is None else float(max_distance),
            backend=str(backend),
        )
        self.coordinates = coordinate_rows
        self.range = float(range)
        self.nugget = float(nugget)
        self.sill = float(sill)
        self.variogram_model = variogram_model
        self.drift = drift
        self.anisotropy_angle_degrees = float(anisotropy_angle_degrees)
        self.anisotropy_scaling = float(anisotropy_scaling)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.min_neighbors = int(min_neighbors)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.backend = backend
        self.variogram_fit_policy = (
            None if variogram_fit_policy is None else dict(variogram_fit_policy)
        )
        for key, value in params.items():
            setattr(self, key, value)

    def fit(self, values: Any, *args: Any, **kwargs: Any):
        """Optionally select variogram parameters using training rows only."""
        self.variogram_fit_ = None
        if self.variogram_fit_policy is not None:
            self._apply_variogram_fit_policy(values)
        return super().fit(values, *args, **kwargs)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        if self.variogram_fit_ is not None:
            metadata["variogram_fit_policy"] = self.variogram_fit_
        return metadata

    def save(self, path: str | Path) -> None:
        super().save(path)
        if self.variogram_fit_ is not None:
            _variogram_policy_sidecar(path).write_text(
                json.dumps(self.variogram_fit_, sort_keys=True), encoding="utf-8"
            )

    @classmethod
    def load(cls, path: str | Path):
        model = super().load(path)
        sidecar = _variogram_policy_sidecar(path)
        if sidecar.is_file():
            model.variogram_fit_ = json.loads(sidecar.read_text(encoding="utf-8"))
        return model

    def _apply_variogram_fit_policy(self, values: Any) -> None:
        if not isinstance(values, Mapping):
            raise ValueError("variogram_fit_policy requires mapping series training values")
        policy = dict(self.variogram_fit_policy or {})
        cutoff = policy.get("cutoff")
        selected: list[tuple[float, float, float]] = []
        coordinate_by_id = {series_id: (x, y) for series_id, x, y in self.coordinates}
        for series_id, series_values in values.items():
            numeric = np.asarray(series_values, dtype=float).reshape(-1)
            if cutoff is not None:
                numeric = numeric[: int(cutoff)]
            if numeric.size and str(series_id) in coordinate_by_id:
                x, y = coordinate_by_id[str(series_id)]
                selected.append((x, y, float(numeric[-1])))
        if len(selected) < 2:
            raise ValueError(
                "variogram_fit_policy requires at least two training series with coordinates"
            )
        from ...geostats import empirical_semivariogram, fit_variogram_wls

        coords = np.asarray([(x, y) for x, y, _ in selected], dtype=float)
        targets = np.asarray([value for _, _, value in selected], dtype=float)
        candidate_grid = {
            "kernels": list(policy.get("kernels", [str(self.variogram_model)])),
            "range_candidates": [float(value) for value in policy["range_candidates"]],
            "sill_candidates": [float(value) for value in policy["sill_candidates"]],
            "nugget_candidates": [
                float(value) for value in policy.get("nugget_candidates", [self.nugget])
            ],
        }
        bins = empirical_semivariogram(
            coords,
            targets,
            bin_count=int(policy.get("bin_count", 12)),
            max_distance=policy.get("max_distance", self.max_distance),
            anisotropy_angle_degrees=self.anisotropy_angle_degrees,
            anisotropy_scaling=self.anisotropy_scaling,
            backend=self.backend,
        )
        fit = fit_variogram_wls(bins, backend=self.backend, **candidate_grid)
        self.range = self._params["range"] = float(fit["range"])
        self.sill = self._params["sill"] = float(fit["sill"])
        self.nugget = self._params["nugget"] = float(fit["nugget"])
        self.variogram_model = self._params["variogram_model"] = str(fit["kernel"])
        self.variogram_fit_ = {
            "policy": "training_only_wls_v1",
            "cutoff": cutoff,
            "candidate_grid": candidate_grid,
            "objective": "weighted_sse",
            "selected": fit,
            "bin_count": len(bins),
            "training_series": len(selected),
        }


class SpatialPiecewiseKrigingForecaster(NativeForecastWrapper):
    """Thin wrapper for Rust-native piecewise seasonal plus spatial kriging fusion."""

    native_class_name = "SpatialPiecewiseKrigingForecaster"

    def __init__(
        self,
        coordinates: CoordinateInput,
        *,
        mode: str = "residual_kriging",
        spatial_regressors: Sequence[str] = (),
        range: float = 1.0,
        nugget: float = 1.0e-6,
        sill: float = 1.0,
        variogram_model: Kernel = Kernel.EXPONENTIAL,
        drift: Drift = Drift.ORDINARY,
        anisotropy_angle_degrees: float = 0.0,
        anisotropy_scaling: float = 1.0,
        max_neighbors: int | None = None,
        min_neighbors: int = 1,
        max_distance: float | None = None,
        residual_shrinkage: float = 1.0,
        allow_neighbor_fallback: bool = False,
        piecewise_config_json: str | None = None,
        backend: Backend | str = Backend.CPU,
        **params: Any,
    ) -> None:
        coordinate_rows = _normalize_coordinates(coordinates)
        super().__init__(
            coordinates=coordinate_rows,
            mode=str(mode),
            spatial_regressors=[str(name) for name in spatial_regressors],
            range=float(range),
            nugget=float(nugget),
            sill=float(sill),
            variogram_model=str(variogram_model),
            drift=str(drift),
            anisotropy_angle_degrees=float(anisotropy_angle_degrees),
            anisotropy_scaling=float(anisotropy_scaling),
            max_neighbors=None if max_neighbors is None else int(max_neighbors),
            min_neighbors=int(min_neighbors),
            max_distance=None if max_distance is None else float(max_distance),
            residual_shrinkage=float(residual_shrinkage),
            allow_neighbor_fallback=bool(allow_neighbor_fallback),
            piecewise_config_json=piecewise_config_json,
            backend=str(backend),
        )
        self.coordinates = coordinate_rows
        self.mode = str(mode)
        self.spatial_regressors = [str(name) for name in spatial_regressors]
        self.range = float(range)
        self.nugget = float(nugget)
        self.sill = float(sill)
        self.variogram_model = str(variogram_model)
        self.drift = str(drift)
        self.anisotropy_angle_degrees = float(anisotropy_angle_degrees)
        self.anisotropy_scaling = float(anisotropy_scaling)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.min_neighbors = int(min_neighbors)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.residual_shrinkage = float(residual_shrinkage)
        self.allow_neighbor_fallback = bool(allow_neighbor_fallback)
        self.piecewise_config_json = piecewise_config_json
        self.backend = backend
        for key, value in params.items():
            setattr(self, key, value)


def _normalize_coordinates(coordinates: CoordinateInput) -> list[tuple[str, float, float]]:
    if isinstance(coordinates, Mapping):
        rows = []
        for series_id, pair in coordinates.items():
            if len(pair) != 2:
                raise ValueError("kriging coordinates must map series_id to (x, y)")
            rows.append((str(series_id), float(pair[0]), float(pair[1])))
        return rows
    return [(str(series_id), float(x), float(y)) for series_id, x, y in coordinates]


def _variogram_policy_sidecar(path: str | Path) -> Path:
    artifact = Path(path)
    return artifact.with_name(f"{artifact.name}.variogram-policy.json")


__all__ = ["KrigingForecaster", "SpatialPiecewiseKrigingForecaster"]
