"""Accelerated direct multi-horizon forecasting wrappers."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import Backend
from ._native_wrappers import NativeForecastWrapper


class _DirectForecastWrapper(NativeForecastWrapper):
    def __init__(
        self,
        *,
        fit_horizon: int = 1,
        lags: Sequence[int] | None = None,
        rolling_windows: Sequence[int] | None = None,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int | None = None,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        if fit_horizon <= 0:
            raise ValueError("fit_horizon must be positive")
        self.fit_horizon = int(fit_horizon)
        self.lags = None if lags is None else list(lags)
        self.rolling_windows = None if rolling_windows is None else list(rolling_windows)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.backend = str(backend)
        super().__init__(
            fit_horizon=self.fit_horizon,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            backend=self.backend,
        )

    def refit_horizon(self, frame: object, horizon: int) -> _DirectForecastWrapper:
        self._check_is_fitted()
        native_frame = getattr(frame, "_native_frame", frame)
        self._native_model.refit_horizon(native_frame, int(horizon))
        self.fit_horizon = int(horizon)
        self._params["fit_horizon"] = self.fit_horizon
        return self


class CartoBoostDirectForecaster(_DirectForecastWrapper):
    """Fit one accelerator-aware boosted model per forecast horizon."""

    native_class_name = "CartoBoostDirectForecaster"


class RectifiedRecursiveForecaster(_DirectForecastWrapper):
    """Accelerated recursive forecast with horizon-specific corrections."""

    native_class_name = "RectifiedRecursiveForecaster"


__all__ = ["CartoBoostDirectForecaster", "RectifiedRecursiveForecaster"]
