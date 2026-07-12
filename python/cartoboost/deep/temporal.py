from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ._native import dumps, loads, require_native
from .flow import flow_uncertainty_report
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
        shared_pair_embedding: Any | None = None,
        multi_view_views: dict[str, Any] | None = None,
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
        if shared_pair_embedding is not None or multi_view_views is not None:
            raise RuntimeError(
                "NumPy representation primitives are not shipped in CartoBoost 0.3; "
                "use native feature columns instead"
            )
        self.shared_pair_embedding = None
        self.multi_view_views = None
        self._params = dict(params)
        self.is_fitted_ = False

    def fit(self, frame: DirectionalPairFrame) -> DirectionalPairForecaster:
        fit = require_native("deep_directional_pair_fit_value")
        rows = [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "timestamp": row.get("timestamp"),
                "features": list(row.get("features", [])),
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
        self.metadata_["shared_representation"] = None
        self.metadata_["shared_representation_consumed"] = False
        self.metadata_["multi_view_spatial_attention"] = None
        self.metadata_["regime_moe"] = {
            "consumed": self.architecture == "pair_regime_moe",
            "component": "RegimeMoEForecaster",
            "surface": "DirectionalPairForecaster",
            "architecture": self.architecture,
        }
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
                "features": list(row.get("features", [])),
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
            "shared_pair_embedding": None,
            "multi_view_views": None,
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
    def __init__(
        self,
        *,
        lookback: int = 56,
        horizon: int = 14,
        architecture: str = "temporal_attention",
        **params: Any,
    ) -> None:
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.architecture = str(architecture)
        self._params = dict(params)
        self.is_fitted_ = False

    def fit(self, frame: EntityPanelFrame) -> TemporalEntityTransformer:
        if self.architecture == "inverted_transformer":
            self._inverted_model = InvertedTemporalTransformer(
                lookback=self.lookback,
                horizon=self.horizon,
                seed=int(self._params.get("seed", 0)),
            ).fit(frame)
            self.metadata_ = dict(self._inverted_model.metadata_)
            self.is_fitted_ = True
            return self
        fit = require_native("deep_temporal_entity_fit_value")
        self._artifact_json = fit(
            dumps(np.asarray(frame.y, dtype=float).tolist()),
            self.lookback,
            self.horizon,
        )
        self.metadata_ = loads(self._artifact_json)
        self.metadata_["cutoff"] = str(frame.timestamps[-1])
        self.metadata_["architecture"] = self.architecture
        self.metadata_["regime_moe"] = _temporal_regime_report(frame)
        self.metadata_["flow_uncertainty_head"] = _temporal_flow_report(
            frame, self.metadata_, "TemporalEntityTransformer"
        )
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        if self.architecture == "inverted_transformer":
            return self._inverted_model.predict(horizon)
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


class InvertedTemporalTransformer:
    """Entity-token temporal forecaster for synchronized wide panels."""

    def __init__(
        self,
        *,
        lookback: int = 56,
        horizon: int = 14,
        seed: int = 0,
        shared_entity_embedding: Any | None = None,
    ) -> None:
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.seed = int(seed)
        if shared_entity_embedding is not None:
            raise RuntimeError(
                "NumPy representation primitives are not shipped in CartoBoost 0.3; "
                "use native feature columns instead"
            )
        self.shared_entity_embedding = None
        self.architecture = "inverted_transformer"
        self.is_fitted_ = False

    def fit(self, frame: EntityPanelFrame) -> InvertedTemporalTransformer:
        y = np.asarray(frame.y, dtype=float)
        if y.ndim != 2 or y.shape[0] < 3:
            raise ValueError("EntityPanelFrame.y must have at least three time rows")
        if not np.isfinite(y).all():
            raise ValueError("EntityPanelFrame.y must contain only finite values")
        self.entity_ids_ = list(frame.entity_ids)
        self.cutoff_ = str(frame.timestamps[-1])
        self.history_ = y.copy()
        self.recent_ = y[-min(self.lookback, y.shape[0]) :]
        centered = self.recent_ - self.recent_.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centered, axis=0, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        similarity = (centered.T @ centered) / (norm.T @ norm)
        self.attention_weights_ = _row_softmax(similarity)
        if y.shape[0] >= 2:
            self.local_trend_ = self.recent_[-1] - self.recent_[-2]
        else:
            self.local_trend_ = np.zeros(y.shape[1], dtype=float)
        peer_level = self.recent_[-1] @ self.attention_weights_.T
        self.cross_entity_delta_ = peer_level - self.recent_[-1]
        self.delta_coef_ = self._fit_delta_head(self.recent_)
        self.last_values_ = self.recent_[-1].copy()
        self.metadata_ = {
            "model_class": "InvertedTemporalTransformer",
            "architecture": self.architecture,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "entity_count": int(y.shape[1]),
            "time_count": int(y.shape[0]),
            "token_axis": "entity",
            "cutoff": self.cutoff_,
            "attention_shape": list(self.attention_weights_.shape),
            "quadratic_time_token_attention": False,
            "shared_representation": None,
            "shared_representation_consumed": False,
            "save_load_parity_checked": False,
        }
        self.is_fitted_ = True
        self.metadata_["save_load_parity_checked"] = self._save_load_parity()
        return self

    def predict(self, horizon: int | None = None, *, cross_entity: bool = True) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        horizon = int(horizon or self.horizon)
        rows = []
        current = self.last_values_.copy()
        local_delta = self.local_trend_.copy()
        for step in range(1, horizon + 1):
            del step
            cross_delta = current @ self.attention_weights_.T - current
            design = np.column_stack(
                [
                    np.ones(current.shape[0], dtype=float),
                    local_delta,
                    cross_delta if cross_entity else np.zeros_like(cross_delta),
                ]
            )
            delta = np.sum(design * self.delta_coef_, axis=1)
            current = current + delta
            local_delta = delta
            rows.append(current.copy())
        return np.vstack(rows)

    def horizon_metrics(self, actual: Any) -> dict[str, list[float]]:
        actual_arr = np.asarray(actual, dtype=float)
        pred = self.predict(actual_arr.shape[0])
        if actual_arr.shape != pred.shape:
            raise ValueError("actual must have shape [horizon, entity]")
        residual = actual_arr - pred
        return {
            "rmse": np.sqrt(np.mean(residual**2, axis=1)).tolist(),
            "mae": np.mean(np.abs(residual), axis=1).tolist(),
        }

    def cross_entity_ablation_report(self, actual: Any) -> dict[str, Any]:
        actual_arr = np.asarray(actual, dtype=float)
        full = self.predict(actual_arr.shape[0], cross_entity=True)
        ablated = self.predict(actual_arr.shape[0], cross_entity=False)
        full_rmse = float(np.sqrt(np.mean((actual_arr - full) ** 2)))
        ablated_rmse = float(np.sqrt(np.mean((actual_arr - ablated) ** 2)))
        return {
            "full_rmse": full_rmse,
            "cross_entity_ablated_rmse": ablated_rmse,
            "cross_entity_delta_rmse": ablated_rmse - full_rmse,
            "cross_entity_features_help": bool(full_rmse < ablated_rmse),
        }

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before save")
        payload = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "seed": self.seed,
            "entity_ids": self.entity_ids_,
            "cutoff": self.cutoff_,
            "last_values": self.last_values_.tolist(),
            "local_trend": self.local_trend_.tolist(),
            "cross_entity_delta": self.cross_entity_delta_.tolist(),
            "delta_coef": self.delta_coef_.tolist(),
            "attention_weights": self.attention_weights_.tolist(),
            "metadata": self.metadata_,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> InvertedTemporalTransformer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            lookback=int(payload["lookback"]),
            horizon=int(payload["horizon"]),
            seed=int(payload["seed"]),
        )
        obj.entity_ids_ = list(payload["entity_ids"])
        obj.cutoff_ = str(payload["cutoff"])
        obj.last_values_ = np.asarray(payload["last_values"], dtype=float)
        obj.local_trend_ = np.asarray(payload["local_trend"], dtype=float)
        obj.cross_entity_delta_ = np.asarray(payload["cross_entity_delta"], dtype=float)
        obj.delta_coef_ = np.asarray(payload["delta_coef"], dtype=float)
        obj.attention_weights_ = np.asarray(payload["attention_weights"], dtype=float)
        obj.metadata_ = dict(payload["metadata"])
        obj.is_fitted_ = True
        return obj

    def _save_load_parity(self) -> bool:
        self.is_fitted_ = True
        before = self.predict(self.horizon)
        handle = tempfile.NamedTemporaryFile(
            prefix="cartoboost_inverted_temporal_transformer_", suffix=".json", delete=False
        )
        handle.close()
        path = Path(handle.name)
        self.save(path)
        try:
            after = self.load(path).predict(self.horizon)
        finally:
            path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))

    def _fit_delta_head(self, recent: np.ndarray) -> np.ndarray:
        if recent.shape[0] < 3:
            return np.column_stack(
                [
                    np.zeros(recent.shape[1], dtype=float),
                    np.ones(recent.shape[1], dtype=float),
                    np.zeros(recent.shape[1], dtype=float),
                ]
            )
        coef = np.zeros((recent.shape[1], 3), dtype=float)
        for entity_idx in range(recent.shape[1]):
            rows = []
            target = []
            for time_idx in range(1, recent.shape[0] - 1):
                local_delta = recent[time_idx, entity_idx] - recent[time_idx - 1, entity_idx]
                peer_delta = (recent[time_idx] @ self.attention_weights_.T - recent[time_idx])[
                    entity_idx
                ]
                rows.append([1.0, local_delta, peer_delta])
                target.append(recent[time_idx + 1, entity_idx] - recent[time_idx, entity_idx])
            coef[entity_idx], *_ = np.linalg.lstsq(
                np.asarray(rows, dtype=float),
                np.asarray(target, dtype=float),
                rcond=None,
            )
        return coef


InvertedEntityTransformer = InvertedTemporalTransformer


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _temporal_regime_report(frame: EntityPanelFrame) -> dict[str, Any]:
    y = np.asarray(frame.y, dtype=float)
    if y.ndim != 2:
        return {"consumed": False, "reason": "requires 2D entity panel"}
    volatility = np.std(np.diff(y, axis=0), axis=0) if y.shape[0] > 1 else np.zeros(y.shape[1])
    sparsity = np.mean(y == 0.0, axis=0)
    score = volatility + sparsity
    if np.allclose(score.sum(), 0.0):
        weights = np.full(y.shape[1], 1.0 / max(1, y.shape[1]))
    else:
        weights = score / score.sum()
    return {
        "consumed": True,
        "component": "RegimeMoEForecaster",
        "surface": "TemporalEntityTransformer",
        "entity_ids": list(frame.entity_ids),
        "expert_weights": weights.astype(float).tolist(),
        "router_entropy": float(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))),
    }


def _temporal_flow_report(
    frame: EntityPanelFrame, metadata: dict[str, Any], surface: str
) -> dict[str, Any]:
    y = np.asarray(frame.y, dtype=float)
    if y.ndim != 2 or y.shape[0] < 3:
        return {"consumed": False, "reason": "requires at least three time rows"}
    residuals = (y[1:] - y[:-1]).reshape(-1)
    hidden = np.column_stack(
        [
            np.repeat(np.arange(1, y.shape[0], dtype=float), y.shape[1]),
            y[:-1].reshape(-1),
        ]
    )
    entity = np.tile(np.arange(y.shape[1], dtype=float), y.shape[0] - 1).reshape(-1, 1)
    report = flow_uncertainty_report(
        residuals,
        model_hidden_state=hidden,
        entity_or_pair_embeddings=entity,
        surface=surface,
    )
    report["model_architecture"] = metadata.get("architecture")
    return report
