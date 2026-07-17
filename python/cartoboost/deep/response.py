from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .._artifacts import ArtifactPersistenceMixin
from ..config import Backend, ChoiceStrEnum
from ._native import dumps, loads, require_native
from .choice import ChoiceSetTransformer
from .flow import flow_uncertainty_report
from .frames import ResponseCurveFrame


class ResponseCurveModel(ArtifactPersistenceMixin):
    def __init__(
        self,
        *,
        response_type: str = "binary",
        monotone: str | None = None,
        calibration: str | None = None,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        self.response_type = response_type
        self.monotone = monotone
        self.calibration = calibration
        self.backend = backend
        self.is_fitted_ = False

    def fit(self, frame: ResponseCurveFrame) -> ResponseCurveModel:
        fit = require_native("deep_response_curve_fit_value")
        self._artifact_json = fit(
            dumps(frame.rows),
            self.response_type,
            self.monotone,
            _choice_value(self.backend),
        )
        self.metadata_ = loads(self._artifact_json)
        self.metadata_["regime_moe"] = _response_regime_report(frame.rows)
        self.is_fitted_ = True
        return self

    def predict_curve(self, frame: ResponseCurveFrame) -> list[dict[str, Any]]:
        self._check_is_fitted()
        predict = require_native("deep_response_curve_predict_value")
        return loads(predict(self._artifact_json, dumps(frame.rows)))

    def predict_response(self, frame: ResponseCurveFrame) -> np.ndarray:
        return np.asarray([row["response_score"] for row in self.predict_curve(frame)], dtype=float)

    def best_candidate(
        self,
        frame: ResponseCurveFrame,
        *,
        objective: str = "max_score",
    ) -> list[dict[str, Any]]:
        del objective
        rows = self.predict_curve(frame)
        best: dict[str | None, dict[str, Any]] = {}
        for row in rows:
            group = row.get("group_id")
            if group not in best or row["response_score"] > best[group]["response_score"]:
                best[group] = row
        return list(best.values())

    def choice_set_report(self, frame: ResponseCurveFrame) -> dict[str, Any]:
        rows = self.predict_curve(frame)
        candidates = [
            {
                "decision_id": row.get("group_id"),
                "candidate_id": row.get("candidate_id"),
                "candidate_value": row.get("candidate_value", 0.0),
                "expected_utility": row.get("response_score", 0.0),
                "response_probability": row.get(
                    "response_probability", row.get("response_score", 0.0)
                ),
                "chosen": row.get("chosen", False),
            }
            for row in rows
        ]
        report = ChoiceSetTransformer(outside_option=True).score(candidates)
        report["surface"] = "ResponseCurveModel"
        return report

    def score(self, frame: ResponseCurveFrame) -> float:
        pred = np.asarray(
            [
                row.get("response_probability", row["response_score"])
                for row in self.predict_curve(frame)
            ],
            dtype=float,
        )
        actual = np.asarray([row["response"] for row in frame.rows], dtype=float)
        return float(np.mean((actual - pred) ** 2))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        Path(path).write_text(self._artifact_json, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ResponseCurveModel:
        obj = cls()
        obj._artifact_json = Path(path).read_text(encoding="utf-8")
        obj.metadata_ = loads(obj._artifact_json)
        obj.response_type = obj.metadata_.get("response_type", "binary")
        obj.monotone = obj.metadata_.get("monotone")
        obj.calibration = None
        obj.backend = Backend(obj.metadata_.get("backend", {}).get("requested", "auto"))
        obj.is_fitted_ = True
        return obj

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "response_type": self.response_type,
            "monotone": self.monotone,
            "calibration": self.calibration,
            "backend": self.backend,
        }

    def set_params(self, **params: Any) -> ResponseCurveModel:
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")


class EventOutcomeModel(ArtifactPersistenceMixin):
    def __init__(
        self,
        *,
        event_type: str = "binary",
        calibration: str = "temperature",
        backend: Backend | str = Backend.CPU,
    ) -> None:
        self.event_type = event_type
        self.calibration = calibration
        self.backend = backend
        self.is_fitted_ = False

    def fit(self, features: Any, labels: Any) -> EventOutcomeModel:
        fit = require_native("deep_event_outcome_fit_value")
        self._artifact_json = fit(
            dumps(np.asarray(features, dtype=float).tolist()),
            list(map(float, labels)),
            _choice_value(self.backend),
        )
        self.metadata_ = loads(self._artifact_json)
        self.is_fitted_ = True
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        self._check_is_fitted()
        predict = require_native("deep_event_outcome_predict_value")
        rows = loads(
            predict(self._artifact_json, dumps(np.asarray(features, dtype=float).tolist()))
        )
        return np.asarray([row["calibrated_probability"] for row in rows], dtype=float)

    def calibration_report(self, features: Any, labels: Any) -> dict[str, float]:
        probs = self.predict_proba(features)
        actual = np.asarray(labels, dtype=float)
        return {"brier": float(np.mean((actual - probs) ** 2))}

    def choice_set_report(self, candidates: list[dict[str, Any]], features: Any) -> dict[str, Any]:
        probs = self.predict_proba(features)
        rows = []
        for row, prob in zip(candidates, probs, strict=True):
            out = dict(row)
            out.setdefault("expected_utility", float(prob))
            out["response_probability"] = float(prob)
            rows.append(out)
        report = ChoiceSetTransformer(outside_option=True).score(rows)
        report["surface"] = "EventOutcomeModel"
        return report

    def score(self, features: Any, labels: Any) -> float:
        return self.calibration_report(features, labels)["brier"]

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        Path(path).write_text(self._artifact_json, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EventOutcomeModel:
        obj = cls()
        obj._artifact_json = Path(path).read_text(encoding="utf-8")
        obj.metadata_ = loads(obj._artifact_json)
        obj.backend = obj.metadata_.get("backend", {}).get("requested", "auto")
        obj.is_fitted_ = True
        return obj

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")


class ServiceTimeResidualModel:
    def __init__(
        self,
        *,
        baseline_col: str = "baseline_value",
        backend: Backend | str = Backend.CPU,
    ) -> None:
        self.baseline_col = baseline_col
        self.backend = backend
        self.is_fitted_ = False

    def fit(self, rows: list[dict[str, Any]]) -> ServiceTimeResidualModel:
        fit = require_native("deep_service_residual_fit_value")
        self._artifact_json = fit(dumps(rows), _choice_value(self.backend))
        self.metadata_ = loads(self._artifact_json)
        self.metadata_["regime_moe"] = _service_regime_report(rows, self.baseline_col)
        self.metadata_["flow_uncertainty_head"] = _service_flow_report(rows, self.baseline_col)
        self.is_fitted_ = True
        return self

    def predict(self, rows: list[dict[str, Any]], *, return_interval: bool = False) -> Any:
        self._check_is_fitted()
        predict = require_native("deep_service_residual_predict_value")
        out = loads(predict(self._artifact_json, dumps(rows)))
        if return_interval:
            return out
        return np.asarray([row["prediction"] for row in out], dtype=float)

    def score(self, rows: list[dict[str, Any]]) -> float:
        pred = self.predict(rows)
        actual = np.asarray([row["actual_value"] for row in rows], dtype=float)
        return float(np.sqrt(np.mean((actual - pred) ** 2)))

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")


def _choice_value(value: str | ChoiceStrEnum) -> str:
    if isinstance(value, ChoiceStrEnum):
        return value.value
    return str(value)


def _response_regime_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = np.asarray([row.get("candidate_value", 0.0) for row in rows], dtype=float)
    responses = np.asarray(
        [0.0 if row.get("response") is None else row.get("response", 0.0) for row in rows],
        dtype=float,
    )
    features = np.asarray([row.get("features", []) for row in rows], dtype=float)
    feature_energy = (
        np.linalg.norm(features, axis=1) if features.size else np.zeros(len(rows), dtype=float)
    )
    logits = np.column_stack(
        [
            np.abs(candidate - np.mean(candidate)),
            feature_energy,
            np.abs(responses - np.mean(responses)),
        ]
    )
    weights = _softmax(logits)
    return {
        "consumed": True,
        "component": "RegimeMoEForecaster",
        "surface": "ResponseCurveModel",
        "expert_weights": weights.astype(float).tolist(),
        "router_entropy": float(
            np.mean(-np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1))
        ),
    }


def _service_regime_report(rows: list[dict[str, Any]], baseline_col: str) -> dict[str, Any]:
    baseline = np.asarray([row.get(baseline_col, 0.0) for row in rows], dtype=float)
    actual = np.asarray(
        [row.get("actual_value", baseline[idx]) for idx, row in enumerate(rows)], dtype=float
    )
    residual = actual - baseline
    logits = np.column_stack(
        [
            np.abs(residual),
            np.abs(baseline - np.mean(baseline)),
            np.ones(len(rows), dtype=float),
        ]
    )
    weights = _softmax(logits)
    return {
        "consumed": True,
        "component": "RegimeMoEForecaster",
        "surface": "ServiceTimeResidualModel",
        "expert_weights": weights.astype(float).tolist(),
        "router_entropy": float(
            np.mean(-np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1))
        ),
    }


def _service_flow_report(rows: list[dict[str, Any]], baseline_col: str) -> dict[str, Any]:
    baseline = np.asarray([row.get(baseline_col, 0.0) for row in rows], dtype=float)
    actual = np.asarray(
        [row.get("actual_value", baseline[idx]) for idx, row in enumerate(rows)], dtype=float
    )
    residual = actual - baseline
    features = np.asarray([row.get("features", [0.0]) for row in rows], dtype=float)
    hidden = np.column_stack([baseline, features.reshape(len(rows), -1)])
    return flow_uncertainty_report(
        residual,
        model_hidden_state=hidden,
        surface="ServiceTimeResidualModel",
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / np.sum(exp, axis=1, keepdims=True)
