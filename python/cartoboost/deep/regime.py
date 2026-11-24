from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .. import _native
from .._artifacts import ArtifactPersistenceMixin
from ..config import Backend
from ..regressor import _to_numeric_numpy

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
        y = np.asarray(_to_numeric_numpy(target), dtype=np.float64).reshape(-1)
        self.entity_ids_ = _ids(entity_ids, x.shape[0], "entity_ids")
        self.pair_ids_ = _ids(pair_ids, x.shape[0], "pair_ids")
        self.shared_regime_router_ = None
        router, widths = self._router_matrix(
            x.shape[0],
            time_features=time_features,
            recent_volatility=recent_volatility,
            recent_residuals=recent_residuals,
            graph_centrality=graph_centrality,
            historical_sparsity=historical_sparsity,
            candidate_value=candidate_value,
        )
        self._native_model = _native.RegimeMoEModel.fit_arrays(
            x, y, router, widths, self.ridge, self.backend
        )
        self._restore_attributes(json.loads(self._native_model.to_json()))
        return self

    def _prediction_inputs(self, features: Any, inputs: dict[str, Any]) -> tuple:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        x = _matrix(features, "features")
        router, widths = self._router_matrix(x.shape[0], **inputs)
        return x, router, widths

    def predict(self, features: Any, **router_inputs: Any) -> np.ndarray:
        inputs = self._prediction_inputs(features, router_inputs)
        return self._native_model.predict_arrays(*inputs)

    def predict_components(self, features: Any, **router_inputs: Any) -> dict[str, Any]:
        inputs = self._prediction_inputs(features, router_inputs)
        weights, experts, combined = self._native_model.components_arrays(*inputs)
        return {
            "expert_weights": weights,
            "expert_predictions": experts,
            "combined_prediction": combined,
            "regime_metadata": dict(self.metadata_),
        }

    def score(self, features: Any, target: Any, **router_inputs: Any) -> float:
        from ..forecasting.metrics import _paired

        truth, pred = _paired(
            target, self.predict(features, **router_inputs), "target", "prediction"
        )
        return float(_native.forecast_evaluate_metrics_arrays(truth, pred, None, None).rmse)

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before save")
        path = Path(path)
        path.write_text(self._native_model.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeMoEForecaster:
        raw = Path(path).read_text(encoding="utf-8")
        native = _native.RegimeMoEModel.from_json(raw)
        payload = json.loads(native.to_json())
        obj = cls(ridge=payload["ridge"], backend=payload["backend"])
        obj._native_model = native
        obj._restore_attributes(payload)
        if payload.get("artifact_version", 1) == 1:
            warnings.warn(
                "Legacy RegimeMoE artifact preserves its original batch-dependent predictions; "
                "refit the model to obtain batch-independent predictions.",
                UserWarning,
                stacklevel=2,
            )
        return obj

    def _restore_attributes(self, payload: dict[str, Any]) -> None:
        self.global_coef_ = np.asarray(payload["global_coef"], dtype=np.float64)
        self.mixer_coef_ = np.asarray(payload["mixer_coef"], dtype=np.float64)
        self.feature_count_ = int(payload["feature_count"])
        self.router_feature_count_ = int(payload["router_feature_count"])
        self.residual_mean_ = float(payload["residual_mean"])
        self.metadata_ = dict(payload["metadata"])
        self.expert_usage_ = dict(self.metadata_["expert_usage"])
        self.train_rmse_ = float(self.metadata_["train_metrics"]["rmse"])
        self.single_expert_rmse_ = float(self.metadata_["train_metrics"]["single_expert_rmse"])
        self.is_fitted_ = True

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        if "_cartoboost_pickle_artifact" in state:
            state["_regime_metadata"] = {
                name: self.__dict__[name]
                for name in ("entity_ids_", "pair_ids_", "shared_regime_router_")
                if name in self.__dict__
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.__dict__.update(state.get("_regime_metadata", {}))
        # Earlier pickles stored the Python fitted arrays directly. Preserve
        # their predictions using the same explicit legacy artifact path.
        if self.is_fitted_ and getattr(self, "_native_model", None) is None:
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
            self._native_model = _native.RegimeMoEModel.from_json(json.dumps(payload))
            self._restore_attributes(json.loads(self._native_model.to_json()))
            warnings.warn(
                "Legacy RegimeMoE pickle preserves batch-dependent predictions; "
                "refit the model to obtain batch-independent predictions.",
                UserWarning,
                stacklevel=2,
            )

    def _router_matrix(self, row_count: int, **inputs: Any) -> tuple[np.ndarray, list[int]]:
        _ids(inputs.get("entity_ids"), row_count, "entity_ids")
        _ids(inputs.get("pair_ids"), row_count, "pair_ids")
        keys = [
            "time_features",
            "recent_volatility",
            "recent_residuals",
            "graph_centrality",
            "historical_sparsity",
            "candidate_value",
        ]
        cols = []
        for key in keys:
            value = inputs.get(key)
            mat = (
                np.zeros((row_count, 1), dtype=np.float64) if value is None else _matrix(value, key)
            )
            if mat.shape[0] != row_count:
                raise ValueError(f"{key} row count must match features")
            cols.append(mat)
        return np.concatenate(cols, axis=1), [mat.shape[1] for mat in cols]


GeoTemporalMixtureOfExperts = RegimeMoEForecaster
PairRegimeRouter = RegimeMoEForecaster
EntityRegimeRouter = RegimeMoEForecaster


def _matrix(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(_to_numeric_numpy(values), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a finite 2D array")
    return arr


def _ids(values: Any, row_count: int, name: str) -> list[str] | None:
    if values is None:
        return None
    ids = [str(v) for v in np.asarray(values).reshape(-1)]
    if len(ids) != row_count:
        raise ValueError(f"{name} row count must match features")
    return ids
