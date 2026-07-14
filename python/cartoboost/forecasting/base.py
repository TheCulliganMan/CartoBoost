from __future__ import annotations

from pathlib import Path
from typing import Any

from .._artifacts import ArtifactPersistenceMixin
from .frequency import validate_horizon
from .schema import ForecastFrame


class BaseForecaster(ArtifactPersistenceMixin):
    """Base guardrails for forecasting estimators."""

    is_fitted_: bool = False

    @staticmethod
    def validate_horizon(horizon: int) -> int:
        return validate_horizon(horizon)

    def _mark_fitted(self) -> None:
        self.is_fitted_ = True

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise ValueError("forecaster is not fitted")

    def fit(self, frame: ForecastFrame, *_args: Any, **_kwargs: Any) -> BaseForecaster:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("fit requires a ForecastFrame")
        self._mark_fitted()
        return self

    def predict(self, horizon: int, *_args: Any, **_kwargs: Any) -> Any:
        self._check_is_fitted()
        return self.validate_horizon(horizon)

    def score(self, values: Any, *, horizon: int | None = None) -> float:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("forecast score requires numpy") from exc
        actual = np.asarray(values, dtype=float).reshape(-1)
        raw_prediction = self.predict(int(horizon or actual.size))
        if hasattr(raw_prediction, "predictions"):
            rows = raw_prediction.predictions
            rows = rows() if callable(rows) else rows
            prediction = np.asarray([row[-1] for row in rows], dtype=float).reshape(-1)
        else:
            prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
        if prediction.shape[0] != actual.shape[0]:
            raise ValueError("prediction and values must have the same length")
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        native_model = getattr(self, "_model", None)
        save = getattr(native_model, "save", None)
        if not callable(save):
            raise NotImplementedError(f"{self.__class__.__name__} does not expose save()")
        save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> BaseForecaster:
        raise NotImplementedError(f"{cls.__name__} does not expose load()")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {}

    def set_params(self, **params: Any) -> BaseForecaster:
        if params:
            raise ValueError(f"unknown parameters: {sorted(params)}")
        return self


class SingleSeriesForecasterMixin:
    """Mixin for estimators that only accept one time series."""

    def _validate_single_series_frame(self, frame: ForecastFrame) -> ForecastFrame:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("expected a ForecastFrame")
        if frame.is_panel:
            raise ValueError("single-series forecasters require data without series_id_col")
        return frame


class PanelForecasterMixin:
    """Mixin for estimators that require isolated panel series."""

    def _validate_panel_frame(self, frame: ForecastFrame) -> ForecastFrame:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("expected a ForecastFrame")
        if not frame.is_panel:
            raise ValueError("panel forecasters require series_id_col")
        if not frame.series_ids:
            raise ValueError("panel forecasters require at least one series")
        return frame
