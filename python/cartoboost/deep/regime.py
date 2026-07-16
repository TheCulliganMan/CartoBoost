from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from .._artifacts import ArtifactPersistenceMixin
from ..accelerators import dense_layer, workload_decision
from ..config import Backend

EXPERT_NAMES = [
    "stable_recurring_pattern",
    "sparse_cold_start",
    "high_volume_hub",
    "volatile_shock",
    "long_distance_pair",
    "low_signal_fallback",
]


class RegimeMoEForecaster(ArtifactPersistenceMixin):
    """Deterministic mixture-of-experts forecaster for heterogeneous regimes."""

    def __init__(
        self,
        *,
        expert_count: int = 6,
        ridge: float = 1e-6,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        if expert_count != 6:
            raise ValueError("RegimeMoEForecaster currently exposes exactly six named experts")
        self.expert_count = int(expert_count)
        self.ridge = float(ridge)
        self.backend = str(backend)
        self.is_fitted_ = False

    def fit(
        self,
        features: Any,
        target: Any,
        *,
        entity_ids: Any | None = None,
        pair_ids: Any | None = None,
        time_features: Any | None = None,
        recent_volatility: Any | None = None,
        recent_residuals: Any | None = None,
        graph_centrality: Any | None = None,
        historical_sparsity: Any | None = None,
        candidate_value: Any | None = None,
    ) -> RegimeMoEForecaster:
        x = _matrix(features, "features")
        y = np.asarray(target, dtype=float).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and target row counts must match")
        self.entity_ids_ = (
            None if entity_ids is None else [str(v) for v in np.asarray(entity_ids).reshape(-1)]
        )
        self.pair_ids_ = (
            None if pair_ids is None else [str(v) for v in np.asarray(pair_ids).reshape(-1)]
        )
        # Entity-aware NumPy representation routing was removed in v0.3. Keep
        # entity IDs as metadata only; all routing inputs must be explicit
        # numeric columns or native graph/temporal features.
        self.shared_regime_router_ = None
        router = self._router_matrix(
            x.shape[0],
            entity_ids=self.entity_ids_,
            time_features=time_features,
            recent_volatility=recent_volatility,
            recent_residuals=recent_residuals,
            graph_centrality=graph_centrality,
            historical_sparsity=historical_sparsity,
            candidate_value=candidate_value,
        )
        self.feature_count_ = int(x.shape[1])
        self.router_feature_count_ = int(router.shape[1])
        self.global_coef_, global_dispatch = _ridge_fit(x, y, self.ridge, self.backend)
        logits = self._router_logits(router)
        weights = _softmax(logits)
        expert_predictions = self._expert_predictions(x, y, weights)
        self.mixer_coef_, mixer_dispatch = _ridge_fit(
            expert_predictions, y, self.ridge, self.backend
        )
        combined = _predict_linear(expert_predictions, self.mixer_coef_, self.backend)
        single = _predict_linear(x, self.global_coef_, self.backend)
        self.residual_mean_ = float(np.mean(y - combined))
        self.train_rmse_ = _rmse(y, combined)
        self.single_expert_rmse_ = _rmse(y, single)
        self.expert_usage_ = {
            EXPERT_NAMES[idx]: float(value)
            for idx, value in enumerate(
                np.bincount(np.argmax(weights, axis=1), minlength=6) / len(y)
            )
        }
        self.metadata_ = {
            "model_class": self.__class__.__name__,
            "architecture": "regime_moe",
            "backend": {
                "requested": self.backend,
                "selected": next(
                    (
                        str(decision["selected"])
                        for decision in (*global_dispatch, *mixer_dispatch)
                        if bool(decision["accelerated"])
                    ),
                    "cpu",
                ),
            },
            "accelerated_operations": (
                ["dense"]
                if any(
                    bool(decision["accelerated"])
                    for decision in (*global_dispatch, *mixer_dispatch)
                )
                else []
            ),
            "expert_names": list(EXPERT_NAMES),
            "router_entropy": float(np.mean(_entropy(weights))),
            "expert_usage": dict(self.expert_usage_),
            "train_metrics": {
                "rmse": self.train_rmse_,
                "single_expert_rmse": self.single_expert_rmse_,
                "beats_single_expert": self.train_rmse_ < self.single_expert_rmse_,
            },
            "router_inputs": [
                "shared_regime_router",
                "entity_id",
                "pair_id",
                "time_features",
                "recent_volatility",
                "recent_residuals",
                "graph_centrality",
                "historical_sparsity",
                "candidate_value",
            ],
            "shared_representation_consumed": self.shared_regime_router_ is not None,
            "shared_representation": None
            if self.shared_regime_router_ is None
            else self.shared_regime_router_.artifact_metadata(),
            "outputs": [
                "expert_weights",
                "expert_predictions",
                "combined_prediction",
                "regime_metadata",
            ],
            "save_load_parity_checked": False,
        }
        self.is_fitted_ = True
        self.metadata_["save_load_parity_checked"] = self._save_load_parity(x, router)
        return self

    def predict(self, features: Any, **router_inputs: Any) -> np.ndarray:
        return self.predict_components(features, **router_inputs)["combined_prediction"]

    def predict_components(self, features: Any, **router_inputs: Any) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        x = _matrix(features, "features")
        router = self._router_matrix(x.shape[0], **router_inputs)
        weights = _softmax(self._router_logits(router))
        expert_predictions = self._expert_predictions(x, None, weights)
        combined = (
            _predict_linear(expert_predictions, self.mixer_coef_, self.backend)
            + self.residual_mean_
        )
        return {
            "expert_weights": weights,
            "expert_predictions": expert_predictions,
            "combined_prediction": combined,
            "regime_metadata": dict(self.metadata_),
        }

    def score(self, features: Any, target: Any, **router_inputs: Any) -> float:
        return _rmse(
            np.asarray(target, dtype=float).reshape(-1), self.predict(features, **router_inputs)
        )

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before save")
        payload = {
            "ridge": self.ridge,
            "backend": self.backend,
            "global_coef": self.global_coef_.tolist(),
            "mixer_coef": self.mixer_coef_.tolist(),
            "feature_count": self.feature_count_,
            "router_feature_count": self.router_feature_count_,
            "residual_mean": self.residual_mean_,
            "metadata": self.metadata_,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeMoEForecaster:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            ridge=float(payload["ridge"]),
            backend=str(payload.get("backend", "cpu")),
        )
        obj.global_coef_ = np.asarray(payload["global_coef"], dtype=float)
        obj.mixer_coef_ = np.asarray(payload["mixer_coef"], dtype=float)
        obj.feature_count_ = int(payload["feature_count"])
        obj.router_feature_count_ = int(payload["router_feature_count"])
        obj.residual_mean_ = float(payload["residual_mean"])
        obj.metadata_ = dict(payload["metadata"])
        obj.expert_usage_ = dict(cast(dict[str, float], obj.metadata_["expert_usage"]))
        obj.is_fitted_ = True
        return obj

    def _router_matrix(self, row_count: int, **inputs: Any) -> np.ndarray:
        cols = []
        entity_ids = inputs.get("entity_ids")
        if entity_ids is not None:
            ids = [str(v) for v in np.asarray(entity_ids).reshape(-1)]
            if len(ids) != row_count:
                raise ValueError("entity_ids row count must match features")
        for key in [
            "time_features",
            "recent_volatility",
            "recent_residuals",
            "graph_centrality",
            "historical_sparsity",
            "candidate_value",
        ]:
            value = inputs.get(key)
            if value is None:
                cols.append(np.zeros((row_count, 1), dtype=float))
            else:
                mat = _matrix(value, key)
                if mat.shape[0] != row_count:
                    raise ValueError(f"{key} row count must match features")
                cols.append(mat)
        return np.concatenate(cols, axis=1)

    def _router_logits(self, router: np.ndarray) -> np.ndarray:
        padded = np.zeros((router.shape[0], 6), dtype=float)
        padded[:, : min(6, router.shape[1])] = router[:, : min(6, router.shape[1])]
        logits = np.column_stack(
            [
                -np.abs(padded[:, 0]),
                padded[:, 4],
                padded[:, 3],
                np.abs(padded[:, 1]) + np.abs(padded[:, 2]),
                padded[:, 5],
                -np.linalg.norm(padded, axis=1),
            ]
        )
        return logits * 2.0

    def _expert_predictions(
        self, x: np.ndarray, y: np.ndarray | None, weights: np.ndarray
    ) -> np.ndarray:
        base = _predict_linear(x, self.global_coef_, self.backend)
        if y is None:
            residual = np.zeros_like(base)
        else:
            residual = y - base
        return np.column_stack(
            [
                base,
                np.mean(base) + 0.35 * (base - np.mean(base)),
                base + np.maximum(0.0, x[:, 0] - np.median(x[:, 0])),
                base + residual * weights[:, 3],
                base + 0.5 * x[:, -1],
                np.full_like(base, np.mean(base)),
            ]
        )

    def _save_load_parity(self, x: np.ndarray, router: np.ndarray) -> bool:
        before = self.predict_components(x, time_features=router[:, :1])["combined_prediction"]
        handle = tempfile.NamedTemporaryFile(
            prefix="cartoboost_regime_moe_", suffix=".json", delete=False
        )
        handle.close()
        path = Path(handle.name)
        self.save(path)
        try:
            after = self.load(path).predict_components(x, time_features=router[:, :1])[
                "combined_prediction"
            ]
        finally:
            path.unlink(missing_ok=True)
        return bool(np.allclose(before, after))


GeoTemporalMixtureOfExperts = RegimeMoEForecaster
PairRegimeRouter = RegimeMoEForecaster
EntityRegimeRouter = RegimeMoEForecaster


def _matrix(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 2D array")
    return arr


def _ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
    backend: str,
) -> tuple[np.ndarray, tuple[dict[str, Any], dict[str, Any]]]:
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    gram, gram_dispatch = _dense_product(design.T, design, backend)
    rhs_matrix, rhs_dispatch = _dense_product(design.T, y.reshape(-1, 1), backend)
    return (
        np.linalg.solve(gram + penalty, rhs_matrix[:, 0]),
        (gram_dispatch, rhs_dispatch),
    )


def _predict_linear(x: np.ndarray, coef: np.ndarray, backend: str) -> np.ndarray:
    design = np.column_stack([np.ones(x.shape[0]), x])
    product, _ = _dense_product(design, coef.reshape(-1, 1), backend)
    return product[:, 0]


def _dense_product(
    left: np.ndarray,
    right: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    workload_size = int(left.shape[0] * left.shape[1] * right.shape[1])
    dispatch = workload_decision(backend, "dense", workload_size, 16_384)
    if dispatch["executed"] == "cpu":
        return left @ right, dispatch
    output = dense_layer(
        left,
        right,
        np.zeros(right.shape[1], dtype=np.float32),
        backend=str(dispatch["executed"]),
    )
    return output.astype(float), dispatch


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _entropy(weights: np.ndarray) -> np.ndarray:
    return -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))
