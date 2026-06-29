from __future__ import annotations

from typing import Any

import numpy as np

from ._native import dumps, require_native
from .frames import DirectionalPairFrame, EntityPanelFrame


class DirectionalPairForecaster:
    def __init__(self, *, preserve_direction: bool = True, **params: Any) -> None:
        self.preserve_direction = preserve_direction
        self._params = dict(params)
        self.is_fitted_ = False

    def fit(self, frame: DirectionalPairFrame) -> DirectionalPairForecaster:
        self.feature_names_in_ = []
        self.metadata_ = {"model_class": "DirectionalPairForecaster", "preserve_direction": True}
        self.is_fitted_ = True
        return self

    def predict(self, frame: DirectionalPairFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        predict = require_native("deep_directional_pair_predict_value")
        rows = [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "timestamp": row.get("timestamp"),
                "features": row.get("features", []),
            }
            for row in frame.rows
        ]
        return np.asarray(predict(dumps(rows)), dtype=float)

    def score(self, frame: DirectionalPairFrame) -> float:
        actual = np.asarray([row["target"] for row in frame.rows], dtype=float)
        pred = self.predict(frame)
        return float(np.sqrt(np.mean((actual - pred) ** 2)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"preserve_direction": self.preserve_direction, **self._params}

    def set_params(self, **params: Any) -> DirectionalPairForecaster:
        self._params.update(params)
        return self


class TemporalEntityTransformer:
    def __init__(self, *, lookback: int = 56, horizon: int = 14, **params: Any) -> None:
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self._params = dict(params)
        self.is_fitted_ = False

    def fit(self, frame: EntityPanelFrame) -> TemporalEntityTransformer:
        if frame.y.shape[0] <= self.lookback:
            raise ValueError("frame history must exceed lookback")
        self._last_window = np.asarray(frame.y[-self.lookback :], dtype=float)
        self.metadata_ = {
            "model_class": "TemporalEntityTransformer",
            "cutoff": str(frame.timestamps[-1]),
        }
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        horizon = int(horizon or self.horizon)
        level = self._last_window[-min(7, self._last_window.shape[0]) :].mean(axis=0)
        return np.tile(level, (horizon, 1))

    def predict_quantiles(self, horizon: int | None = None) -> dict[float, np.ndarray]:
        pred = self.predict(horizon)
        return {0.1: pred, 0.5: pred, 0.9: pred}

    def backtest(self, splitter: Any = None) -> dict[str, Any]:
        del splitter
        return {"folds": []}

    def score(self, actual: Any) -> float:
        actual_arr = np.asarray(actual, dtype=float)
        pred = self.predict(actual_arr.shape[0])
        return float(np.sqrt(np.mean((actual_arr - pred) ** 2)))
