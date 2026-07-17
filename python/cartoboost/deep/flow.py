from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._artifacts import ArtifactPersistenceMixin
from ..config import Backend
from ._native import require_native


class ConditionalFlowDistributionHead(ArtifactPersistenceMixin):
    """Rust-backed conditional residual flow distribution head."""

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
        sample_count: int = 64,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        self.quantiles = tuple(float(q) for q in quantiles)
        self.sample_count = int(sample_count)
        self.backend = str(backend)
        self.is_fitted_ = False

    def fit(
        self,
        residuals: Any,
        *,
        model_hidden_state: Any,
        horizon_embeddings: Any | None = None,
        entity_or_pair_embeddings: Any | None = None,
        graph_context: Any | None = None,
    ) -> ConditionalFlowDistributionHead:
        hidden = _combined_hidden(
            model_hidden_state,
            horizon_embeddings=horizon_embeddings,
            entity_or_pair_embeddings=entity_or_pair_embeddings,
            graph_context=graph_context,
        )
        residual_arr = np.asarray(residuals, dtype=float).reshape(-1)
        fit = require_native("prob_conditional_flow_fit_value")
        self._artifact_json = fit(
            hidden.tolist(),
            residual_arr.tolist(),
            list(self.quantiles),
            self.sample_count,
            self.backend,
        )
        self.metadata_ = json.loads(self._artifact_json)["metadata"]
        self.backend_ = str(self.metadata_["backend_selected"])
        self.hidden_dim_ = int(hidden.shape[1])
        self.is_fitted_ = True
        return self

    def predict(
        self,
        *,
        model_hidden_state: Any,
        horizon_embeddings: Any | None = None,
        entity_or_pair_embeddings: Any | None = None,
        graph_context: Any | None = None,
        actual: Any | None = None,
    ) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("flow head must be fit before prediction")
        hidden = _combined_hidden(
            model_hidden_state,
            horizon_embeddings=horizon_embeddings,
            entity_or_pair_embeddings=entity_or_pair_embeddings,
            graph_context=graph_context,
        )
        predict = require_native("prob_conditional_flow_predict_value")
        actual_values = (
            None if actual is None else np.asarray(actual, dtype=float).reshape(-1).tolist()
        )
        output = json.loads(predict(self._artifact_json, hidden.tolist(), actual_values))
        return {
            "samples": np.asarray(output["samples"], dtype=float),
            "log_likelihood": np.asarray(output["log_likelihood"], dtype=float),
            "marginal_quantiles": np.asarray(output["marginal_quantiles"], dtype=float),
            "joint_scenario_paths": np.asarray(output["joint_scenario_paths"], dtype=float),
            "tail_risk_metrics": dict(output["tail_risk_metrics"]),
            "metrics": dict(output["metrics"]),
        }

    def quantiles_for(self, **kwargs: Any) -> np.ndarray:
        return self.predict(**kwargs)["marginal_quantiles"]

    def sample(self, **kwargs: Any) -> np.ndarray:
        return self.predict(**kwargs)["samples"]

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("flow head must be fit before save")
        path = Path(path)
        path.write_text(self._artifact_json, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ConditionalFlowDistributionHead:
        artifact = Path(path).read_text(encoding="utf-8")
        payload = json.loads(artifact)
        obj = cls(
            quantiles=tuple(payload["quantiles"]),
            sample_count=int(payload["sample_count"]),
            backend=str(payload.get("backend", {}).get("selected", "cpu")),
        )
        obj._artifact_json = artifact
        obj.metadata_ = dict(payload["metadata"])
        obj.backend_ = str(obj.metadata_.get("backend_selected", obj.backend))
        obj.hidden_dim_ = len(payload["location_weights"]) - 1
        obj.is_fitted_ = True
        return obj

    def benchmark_against_baselines(
        self,
        actual: Any,
        *,
        model_hidden_state: Any,
        horizon_embeddings: Any | None = None,
        entity_or_pair_embeddings: Any | None = None,
        graph_context: Any | None = None,
    ) -> dict[str, Any]:
        actual_arr = np.asarray(actual, dtype=float).reshape(-1)
        output = self.predict(
            model_hidden_state=model_hidden_state,
            horizon_embeddings=horizon_embeddings,
            entity_or_pair_embeddings=entity_or_pair_embeddings,
            graph_context=graph_context,
            actual=actual_arr,
        )
        quantiles = output["marginal_quantiles"]
        low = quantiles[:, 0]
        high = quantiles[:, -1]
        flow_coverage = _coverage(actual_arr, low, high)
        flow_width = float(np.mean(high - low))
        baselines = _uncertainty_baselines(actual_arr)
        best_baseline_pinball = min(row["pinball_median"] for row in baselines.values())
        return {
            "flow_metrics": dict(output["metrics"]),
            "baselines": baselines,
            "independent_quantile_head": baselines["independent_quantile_head"],
            "gaussian_residual_head": baselines["gaussian_residual_head"],
            "conformal_interval_wrapper": baselines["conformal_interval_wrapper"],
            "flow_interval_coverage": flow_coverage,
            "flow_interval_width": flow_width,
            "flow_improves_calibration_or_sharpness": bool(
                abs(flow_coverage - 0.9)
                < abs(baselines["gaussian_residual_head"]["interval_coverage"] - 0.9)
                or flow_width < baselines["conformal_interval_wrapper"]["interval_width"]
                or output["metrics"]["pinball_median"] < best_baseline_pinball
            ),
        }


def flow_uncertainty_report(
    residuals: Any,
    *,
    model_hidden_state: Any,
    horizon_embeddings: Any | None = None,
    entity_or_pair_embeddings: Any | None = None,
    graph_context: Any | None = None,
    surface: str,
) -> dict[str, Any]:
    residual_arr = np.asarray(residuals, dtype=float).reshape(-1)
    head = ConditionalFlowDistributionHead(quantiles=(0.05, 0.5, 0.95), sample_count=32).fit(
        residual_arr,
        model_hidden_state=model_hidden_state,
        horizon_embeddings=horizon_embeddings,
        entity_or_pair_embeddings=entity_or_pair_embeddings,
        graph_context=graph_context,
    )
    benchmark = head.benchmark_against_baselines(
        residual_arr,
        model_hidden_state=model_hidden_state,
        horizon_embeddings=horizon_embeddings,
        entity_or_pair_embeddings=entity_or_pair_embeddings,
        graph_context=graph_context,
    )
    return {
        "consumed": True,
        "component": "ConditionalFlowDistributionHead",
        "surface": surface,
        "metadata": dict(head.metadata_),
        **benchmark,
    }


JointHorizonFlowHead = ConditionalFlowDistributionHead
ResidualFlowCalibrator = ConditionalFlowDistributionHead


def _combined_hidden(
    model_hidden_state: Any,
    *,
    horizon_embeddings: Any | None,
    entity_or_pair_embeddings: Any | None,
    graph_context: Any | None,
) -> np.ndarray:
    parts = [_matrix(model_hidden_state, "model_hidden_state")]
    row_count = parts[0].shape[0]
    for name, value in [
        ("horizon_embeddings", horizon_embeddings),
        ("entity_or_pair_embeddings", entity_or_pair_embeddings),
        ("graph_context", graph_context),
    ]:
        if value is None:
            continue
        part = _matrix(value, name)
        if part.shape[0] != row_count:
            raise ValueError(f"{name} row count must match model_hidden_state")
        parts.append(part)
    return np.concatenate(parts, axis=1)


def _uncertainty_baselines(actual: np.ndarray) -> dict[str, dict[str, float]]:
    low_q, median_q, high_q = np.quantile(actual, [0.05, 0.5, 0.95])
    std = float(np.std(actual))
    mean = float(np.mean(actual))
    conformal_q = float(np.quantile(np.abs(actual - median_q), 0.9))
    return {
        "independent_quantile_head": {
            "pinball_median": _pinball(actual, np.full_like(actual, median_q), 0.5),
            "interval_coverage": _coverage(
                actual, np.full_like(actual, low_q), np.full_like(actual, high_q)
            ),
            "interval_width": float(high_q - low_q),
        },
        "gaussian_residual_head": {
            "pinball_median": _pinball(actual, np.full_like(actual, mean), 0.5),
            "interval_coverage": _coverage(
                actual,
                np.full_like(actual, mean - 1.645 * std),
                np.full_like(actual, mean + 1.645 * std),
            ),
            "interval_width": float(3.29 * std),
        },
        "conformal_interval_wrapper": {
            "pinball_median": _pinball(actual, np.full_like(actual, median_q), 0.5),
            "interval_coverage": _coverage(
                actual,
                np.full_like(actual, median_q - conformal_q),
                np.full_like(actual, median_q + conformal_q),
            ),
            "interval_width": float(2.0 * conformal_q),
        },
    }


def _coverage(actual: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((actual >= low) & (actual <= high)))


def _pinball(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    residual = actual - pred
    return float(np.mean(np.maximum(q * residual, (q - 1.0) * residual)))


def _matrix(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 2D array")
    return arr
