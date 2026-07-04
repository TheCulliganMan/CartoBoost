from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .flow import flow_uncertainty_report
from .frames import EntityPanelFrame


class SelectiveStateSpaceBlock:
    def __init__(self, *, input_dim: int, state_dim: int = 8, seed: int = 0) -> None:
        if input_dim <= 0 or state_dim <= 0:
            raise ValueError("input_dim and state_dim must be positive")
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.seed = int(seed)
        self.architecture = "selective_ssm_lite_encoder"
        self.gate_weights = _deterministic_matrix(input_dim, state_dim, seed=seed + 11, salt="gate")
        self.delta_weights = _deterministic_matrix(
            input_dim, state_dim, seed=seed + 17, salt="delta"
        )
        self.b_weights = _deterministic_matrix(input_dim, state_dim, seed=seed + 23, salt="b")
        self.c_weights = _deterministic_matrix(input_dim, state_dim, seed=seed + 29, salt="c")
        self.direct_weights = _deterministic_matrix(
            input_dim, state_dim, seed=seed + 31, salt="direct"
        )
        self.decay = 0.1 + np.arange(state_dim, dtype=float) / float(state_dim)

    def encode(self, sequence: Any) -> np.ndarray:
        values = np.asarray(sequence, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("sequence must be a two-dimensional array matching input_dim")
        if not np.isfinite(values).all():
            raise ValueError("sequence must contain only finite values")
        state = np.zeros(self.state_dim, dtype=float)
        output = np.zeros((values.shape[0], self.state_dim), dtype=float)
        for idx, row in enumerate(values):
            gate = _sigmoid(row @ self.gate_weights)
            delta = _softplus(row @ self.delta_weights)
            b_t = row @ self.b_weights
            c_t = row @ self.c_weights
            direct = row @ self.direct_weights
            state = np.exp(-delta * self.decay) * state + gate * b_t
            output[idx] = c_t * state + direct
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "seed": self.seed,
            "architecture": self.architecture,
            "gate_weights": self.gate_weights.tolist(),
            "delta_weights": self.delta_weights.tolist(),
            "b_weights": self.b_weights.tolist(),
            "c_weights": self.c_weights.tolist(),
            "direct_weights": self.direct_weights.tolist(),
            "decay": self.decay.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SelectiveStateSpaceBlock:
        obj = cls(
            input_dim=int(payload["input_dim"]),
            state_dim=int(payload["state_dim"]),
            seed=int(payload["seed"]),
        )
        obj.architecture = str(payload["architecture"])
        obj.gate_weights = np.asarray(payload["gate_weights"], dtype=float)
        obj.delta_weights = np.asarray(payload["delta_weights"], dtype=float)
        obj.b_weights = np.asarray(payload["b_weights"], dtype=float)
        obj.c_weights = np.asarray(payload["c_weights"], dtype=float)
        obj.direct_weights = np.asarray(payload["direct_weights"], dtype=float)
        obj.decay = np.asarray(payload["decay"], dtype=float)
        return obj


class TemporalSSMForecaster:
    def __init__(
        self,
        *,
        lookback: int = 64,
        horizon: int = 1,
        state_dim: int = 8,
        seed: int = 0,
    ) -> None:
        if lookback <= 0 or horizon <= 0 or state_dim <= 0:
            raise ValueError("lookback, horizon, and state_dim must be positive")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.seed = int(seed)
        self.architecture = "selective_ssm_lite"
        self.is_fitted_ = False

    def fit(self, frame: EntityPanelFrame) -> TemporalSSMForecaster:
        y = np.asarray(frame.y, dtype=float)
        if y.ndim != 2 or y.shape[0] < 2:
            raise ValueError("EntityPanelFrame.y must have at least two time rows")
        self.block_ = SelectiveStateSpaceBlock(
            input_dim=y.shape[1],
            state_dim=self.state_dim,
            seed=self.seed,
        )
        self.encoded_ = self.block_.encode(y)
        recent = y[-max(2, min(self.lookback, y.shape[0])) :]
        self.last_values_ = y[-1].copy()
        self.trend_ = (recent[-1] - recent[0]) / max(1, recent.shape[0] - 1)
        self.decoder_weights_, self.decoder_metrics_ = _fit_horizon_decoders(
            y,
            self.encoded_,
            horizon=self.horizon,
            ridge=1e-6,
        )
        self.metadata_ = {
            "model_class": "TemporalSSMForecaster",
            "architecture": self.architecture,
            "architecture_scope": "selective_ssm_lite_not_full_mamba",
            "artifact_version": 1,
            "schema_hash": _schema_hash(y.shape, self.lookback, self.horizon, self.state_dim),
            "lookback": self.lookback,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "seed": self.seed,
            "backend": "cpu",
            "accelerated_scan": False,
            "decoder": {
                "kind": "encoded_state_horizon_specific_ridge",
                "uses_encoded_state": True,
                "uses_entity_conditioning": True,
                "rolling_origin_fit_objective": "squared_error",
                **self.decoder_metrics_,
            },
            "cutoff": str(frame.timestamps[-1]),
            "flow_uncertainty_head": _ssm_flow_report(y, self.encoded_),
            "save_load_parity_checked": False,
        }
        self.is_fitted_ = True
        self.metadata_["save_load_parity_checked"] = self._save_load_parity()
        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        horizon = int(horizon or self.horizon)
        features = _decoder_features(
            self.encoded_[-1],
            self.last_values_,
            np.arange(self.last_values_.shape[0], dtype=float),
        )
        rows = []
        for step in range(horizon):
            weight_idx = min(step, self.decoder_weights_.shape[0] - 1)
            rows.append(features @ self.decoder_weights_[weight_idx])
        return np.vstack(rows)

    def runtime_scaling_report(
        self,
        *,
        lookbacks: tuple[int, ...] = (64, 128, 256, 512, 1024),
        feature_dim: int = 4,
    ) -> list[dict[str, Any]]:
        report = []
        for lookback in lookbacks:
            sequence = _scaling_sequence(lookback, feature_dim)
            block = SelectiveStateSpaceBlock(
                input_dim=feature_dim,
                state_dim=self.state_dim,
                seed=self.seed,
            )
            started = time.perf_counter()
            encoded = block.encode(sequence)
            elapsed = time.perf_counter() - started
            report.append(
                {
                    "lookback": int(lookback),
                    "elapsed_seconds": float(elapsed),
                    "memory_bytes": int(sequence.nbytes + encoded.nbytes),
                    "architecture": self.architecture,
                    "backend": "cpu",
                }
            )
        return report

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before save")
        payload = {
            "metadata": self.metadata_,
            "block": self.block_.to_dict(),
            "encoded": self.encoded_.tolist(),
            "last_values": self.last_values_.tolist(),
            "trend": self.trend_.tolist(),
            "decoder_weights": self.decoder_weights_.tolist(),
            "decoder_metrics": self.decoder_metrics_,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> TemporalSSMForecaster:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        obj = cls(
            lookback=int(metadata["lookback"]),
            horizon=int(metadata["horizon"]),
            state_dim=int(metadata["state_dim"]),
            seed=int(metadata["seed"]),
        )
        obj.block_ = SelectiveStateSpaceBlock.from_dict(payload["block"])
        obj.encoded_ = np.asarray(payload["encoded"], dtype=float)
        obj.last_values_ = np.asarray(payload["last_values"], dtype=float)
        obj.trend_ = np.asarray(payload["trend"], dtype=float)
        obj.decoder_weights_ = np.asarray(payload["decoder_weights"], dtype=float)
        obj.decoder_metrics_ = dict(payload["decoder_metrics"])
        obj.metadata_ = dict(metadata)
        obj.is_fitted_ = True
        return obj

    def _save_load_parity(self) -> bool:
        before = self.predict(self.horizon)
        path = Path("/tmp/cartoboost_temporal_ssm_parity.json")
        self.save(path)
        try:
            after = self.load(path).predict(self.horizon)
        finally:
            path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))


EntityTemporalSSM = TemporalSSMForecaster
PairTemporalSSM = TemporalSSMForecaster
GraphTemporalSSM = TemporalSSMForecaster


def _ssm_flow_report(y: np.ndarray, encoded: np.ndarray) -> dict[str, Any]:
    if y.shape[0] < 3:
        return {"consumed": False, "reason": "requires at least three time rows"}
    residuals = (y[1:] - y[:-1]).reshape(-1)
    hidden = np.repeat(encoded[:-1], y.shape[1], axis=0)
    entity = np.tile(np.arange(y.shape[1], dtype=float), y.shape[0] - 1).reshape(-1, 1)
    return flow_uncertainty_report(
        residuals,
        model_hidden_state=hidden,
        entity_or_pair_embeddings=entity,
        surface="TemporalSSMForecaster",
    )


def _fit_horizon_decoders(
    y: np.ndarray,
    encoded: np.ndarray,
    *,
    horizon: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if y.shape[0] <= horizon:
        raise ValueError("EntityPanelFrame.y must have more rows than horizon")
    weights = []
    ssm_rmse = []
    trend_rmse = []
    conv_rmse = []
    for step in range(1, horizon + 1):
        feature_rows = []
        target_rows = []
        trend_rows = []
        conv_rows = []
        for idx in range(y.shape[0] - step):
            entity_index = np.arange(y.shape[1], dtype=float)
            feature_rows.append(_decoder_features(encoded[idx], y[idx], entity_index))
            target_rows.append(y[idx + step])
            start = max(0, idx - 3)
            recent = y[start : idx + 1]
            trend = (recent[-1] - recent[0]) / max(1, recent.shape[0] - 1)
            trend_rows.append(y[idx] + trend * step)
            conv_rows.append(np.mean(recent, axis=0))
        x = np.vstack(feature_rows)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        x = np.clip(x, -1.0e6, 1.0e6)
        target = np.concatenate(target_rows)
        coef = _ridge_fit(x, target, ridge)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = x @ coef
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        weights.append(coef)
        actual = target
        ssm_rmse.append(_rmse(actual, pred))
        trend_rmse.append(_rmse(actual, np.concatenate(trend_rows)))
        conv_rmse.append(_rmse(actual, np.concatenate(conv_rows)))
    return np.vstack(weights), {
        "rolling_origin_train_windows": int(
            sum(y.shape[0] - step for step in range(1, horizon + 1))
        ),
        "ssm_decoder_rmse": float(np.mean(ssm_rmse)),
        "trend_extrapolation_rmse": float(np.mean(trend_rmse)),
        "temporal_conv_baseline_rmse": float(np.mean(conv_rmse)),
        "beats_trend_extrapolation": bool(np.mean(ssm_rmse) < np.mean(trend_rmse)),
        "beats_temporal_conv_baseline": bool(np.mean(ssm_rmse) < np.mean(conv_rmse)),
    }


def _decoder_features(
    encoded_state: np.ndarray, current_values: np.ndarray, entity_index: np.ndarray
) -> np.ndarray:
    if current_values.ndim != 1:
        raise ValueError("current_values must be one-dimensional")
    entity_scale = entity_index / max(1.0, float(len(entity_index) - 1))
    repeated_state = np.repeat(encoded_state.reshape(1, -1), len(current_values), axis=0)
    return np.column_stack(
        [
            np.ones(len(current_values), dtype=float),
            repeated_state,
            current_values,
            entity_scale,
        ]
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    center = np.mean(x[:, 1:], axis=0)
    scale = np.std(x[:, 1:], axis=0)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    normalized = x.copy()
    normalized[:, 1:] = (normalized[:, 1:] - center) / scale
    penalty = np.sqrt(ridge) * np.eye(normalized.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    x_augmented = np.vstack([normalized, penalty])
    y_augmented = np.concatenate([y, np.zeros(normalized.shape[1], dtype=float)])
    normalized_coef = np.linalg.lstsq(x_augmented, y_augmented, rcond=None)[0]
    coef = np.zeros_like(normalized_coef)
    coef[1:] = normalized_coef[1:] / scale
    coef[0] = normalized_coef[0] - np.sum((center / scale) * normalized_coef[1:])
    return np.clip(np.nan_to_num(coef, nan=0.0, posinf=1.0e6, neginf=-1.0e6), -1.0e6, 1.0e6)


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _deterministic_matrix(rows: int, cols: int, *, seed: int, salt: str) -> np.ndarray:
    values = np.zeros((rows, cols), dtype=float)
    for row in range(rows):
        for col in range(cols):
            digest = hashlib.sha256(f"{seed}:{salt}:{row}:{col}".encode()).digest()
            integer = int.from_bytes(digest[:8], "little", signed=False)
            values[row, col] = (integer / float(2**64 - 1)) * 2.0 - 1.0
    return values


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(values, -50.0, 50.0)))


def _schema_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _scaling_sequence(lookback: int, feature_dim: int) -> np.ndarray:
    time_index = np.arange(lookback, dtype=float).reshape(-1, 1)
    cols = [np.sin(time_index[:, 0] / (idx + 2)) for idx in range(feature_dim)]
    return np.column_stack(cols)


__all__ = [
    "EntityTemporalSSM",
    "GraphTemporalSSM",
    "PairTemporalSSM",
    "SelectiveStateSpaceBlock",
    "TemporalSSMForecaster",
]
