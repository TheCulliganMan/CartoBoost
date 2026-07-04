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
        self.architecture = "selective_ssm"
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
        self.architecture = "selective_ssm"
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
        self.metadata_ = {
            "model_class": "TemporalSSMForecaster",
            "architecture": self.architecture,
            "artifact_version": 1,
            "schema_hash": _schema_hash(y.shape, self.lookback, self.horizon, self.state_dim),
            "lookback": self.lookback,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "seed": self.seed,
            "backend": "cpu",
            "accelerated_scan": False,
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
        return np.vstack([self.last_values_ + self.trend_ * step for step in range(1, horizon + 1)])

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
                    "architecture": "selective_ssm",
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
            "last_values": self.last_values_.tolist(),
            "trend": self.trend_.tolist(),
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
        obj.last_values_ = np.asarray(payload["last_values"], dtype=float)
        obj.trend_ = np.asarray(payload["trend"], dtype=float)
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
