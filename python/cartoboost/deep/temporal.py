from __future__ import annotations

from typing import Any

import numpy as np

from ._native import dumps, loads, require_native
from .frames import DirectionalPairFrame, EntityPanelFrame


class DirectionalPairForecaster:
    def __init__(
        self,
        *,
        preserve_direction: bool = True,
        architecture: str = "shrinkage_effects",
        embedding_dim: int = 4,
        pair_bucket_count: int = 64,
        hidden_dim: int = 12,
        epochs: int = 700,
        learning_rate: float = 0.018,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        early_stopping_rounds: int = 80,
        seed: int = 0,
        loss: str = "squared_error",
        **params: Any,
    ) -> None:
        self.preserve_direction = preserve_direction
        self.architecture = architecture
        self.embedding_dim = int(embedding_dim)
        self.pair_bucket_count = int(pair_bucket_count)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.seed = int(seed)
        self.loss = loss
        self._params = dict(params)
        self.is_fitted_ = False

    def fit(self, frame: DirectionalPairFrame) -> DirectionalPairForecaster:
        fit = require_native("deep_directional_pair_fit_value")
        rows = [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "timestamp": row.get("timestamp"),
                "features": row.get("features", []),
                "target": row.get("target"),
            }
            for row in frame.rows
        ]
        options = {
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "pair_bucket_count": self.pair_bucket_count,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "loss": self.loss,
        }
        self._artifact_json = fit(dumps(rows), dumps(options))
        self.metadata_ = loads(self._artifact_json)
        self.metadata_["preserve_direction"] = self.preserve_direction
        self.feature_names_in_ = [
            f"feature_{idx}" for idx in range(len(rows[0].get("features", [])))
        ]
        self.is_fitted_ = True
        return self

    def predict(self, frame: DirectionalPairFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        predict = require_native("deep_directional_pair_predict_artifact_value")
        rows = [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "timestamp": row.get("timestamp"),
                "features": row.get("features", []),
            }
            for row in frame.rows
        ]
        return np.asarray(predict(self._artifact_json, dumps(rows)), dtype=float)

    def score(self, frame: DirectionalPairFrame) -> float:
        actual = np.asarray([row["target"] for row in frame.rows], dtype=float)
        pred = self.predict(frame)
        return float(np.sqrt(np.mean((actual - pred) ** 2)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "preserve_direction": self.preserve_direction,
            "architecture": self.architecture,
            "embedding_dim": self.embedding_dim,
            "pair_bucket_count": self.pair_bucket_count,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "loss": self.loss,
            **self._params,
        }

    def set_params(self, **params: Any) -> DirectionalPairForecaster:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._params[key] = value
        return self


class TemporalEntityTransformer:
    def __init__(self, *, lookback: int = 56, horizon: int = 14, **params: Any) -> None:
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self._params = dict(params)
        self.is_fitted_ = False

    def fit(self, frame: EntityPanelFrame) -> TemporalEntityTransformer:
        fit = require_native("deep_temporal_entity_fit_value")
        self._artifact_json = fit(
            dumps(np.asarray(frame.y, dtype=float).tolist()),
            self.lookback,
            self.horizon,
        )
        self.metadata_ = loads(self._artifact_json)
        self.metadata_["cutoff"] = str(frame.timestamps[-1])
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        horizon = int(horizon or self.horizon)
        predict = require_native("deep_temporal_entity_predict_value")
        return np.asarray(loads(predict(self._artifact_json, horizon)), dtype=float)

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
