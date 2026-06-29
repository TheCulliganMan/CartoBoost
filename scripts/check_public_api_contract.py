#!/usr/bin/env python3
"""Audit the stable public model registry and lightweight artifact roundtrips."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost.models import GeoModelStack, ModelRegistry, ModelSpec  # noqa: E402

REQUIRED_LIFECYCLE = ("fit", "predict", "score", "save", "load", "get_params", "set_params")
ROUNDTRIP_CASES = {
    "models.cartoboost_regressor",
    "models.cartoboost_classifier",
    "models.cartoboost_ranker",
    "models.auto_geo_model",
    "models.geo_model_stack",
    "geo.nngp",
    "geo.residual_nngp",
    "prob.conformal_interval",
    "prob.spatial_conformal",
}


def main() -> int:
    registry = ModelRegistry.defaults()
    structural = [audit_spec(spec) for spec in registry.specs()]
    roundtrips = [
        run_roundtrip_case(get_registry_key(registry, name)) for name in sorted(ROUNDTRIP_CASES)
    ]
    payload = {
        "artifact_type": "cartoboost.public_api_contract_audit",
        "artifact_version": 1,
        "registry_size": len(registry.specs()),
        "required_lifecycle": list(REQUIRED_LIFECYCLE),
        "structural": structural,
        "roundtrips": roundtrips,
        "passed": all(row["passed"] for row in structural)
        and all(row["passed"] for row in roundtrips),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("public API contract audit failed")
    return 0


def audit_spec(spec: ModelSpec) -> dict[str, Any]:
    factory = spec.factory
    metadata = spec.metadata.to_dict()
    class_missing = [name for name in REQUIRED_LIFECYCLE if not hasattr(factory, name)]
    metadata_missing = [
        key
        for key in ("name", "namespace", "task_types", "capabilities", "stable", "artifact_format")
        if key not in metadata
    ]
    key = f"{spec.namespace}.{spec.name}"
    return {
        "key": key,
        "class": getattr(factory, "__name__", repr(factory)),
        "metadata": metadata,
        "missing_lifecycle": class_missing,
        "missing_metadata": metadata_missing,
        "roundtrip_case": key in ROUNDTRIP_CASES,
        "passed": not class_missing and not metadata_missing,
    }


def get_registry_key(registry: ModelRegistry, key: str) -> ModelSpec:
    namespace, name = key.split(".", 1)
    return registry.get(name, namespace=namespace)


def run_roundtrip_case(spec: ModelSpec) -> dict[str, Any]:
    key = f"{spec.namespace}.{spec.name}"
    try:
        before, after = _fit_predict_reload(spec)
        max_abs_diff = float(np.max(np.abs(before - after))) if before.size else 0.0
        return {
            "key": key,
            "rows": int(before.shape[0]),
            "roundtrip_max_abs_diff": max_abs_diff,
            "passed": max_abs_diff <= 1e-10,
        }
    except Exception as exc:  # pragma: no cover - failure payload is the product.
        return {
            "key": key,
            "error": f"{exc.__class__.__name__}: {exc}",
            "passed": False,
        }


def _fit_predict_reload(spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"{spec.namespace}-{spec.name}.json"
        model, X, kwargs = fit_registry_roundtrip_case(spec)
        before = _predict_array(model, X, kwargs)
        model.save(path)
        loaded = spec.factory.load(path)
        after = _predict_array(loaded, X, kwargs)
        return before, after


def fit_registry_roundtrip_case(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    key = f"{spec.namespace}.{spec.name}"
    if key == "models.cartoboost_regressor":
        return _fit_regressor(spec)
    if key == "models.cartoboost_classifier":
        return _fit_classifier(spec)
    if key == "models.cartoboost_ranker":
        return _fit_ranker(spec)
    if key == "models.auto_geo_model":
        return _fit_auto_geo(spec)
    if key == "models.geo_model_stack":
        return _fit_geo_stack()
    if key == "geo.nngp":
        return _fit_nngp(spec)
    if key == "geo.residual_nngp":
        return _fit_residual_nngp(spec)
    if key in {"prob.conformal_interval", "prob.spatial_conformal"}:
        return _fit_conformal(spec, spatial=key.endswith("spatial_conformal"))
    raise ValueError(f"no roundtrip case registered for {key}")


def _fit_regressor(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    X, y, _ = _sample_rows()
    model = spec.create(n_estimators=8, max_depth=2, min_samples_leaf=1, min_gain=0.0)
    model.fit(X, y)
    return model, X, {}


def _fit_classifier(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    X, _, _ = _sample_rows()
    y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)
    model = spec.create(n_estimators=8, max_depth=2, min_samples_leaf=1, min_gain=0.0)
    model.fit(X, y)
    return model, X, {"method": "predict_proba"}


def _fit_ranker(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    X = np.asarray([[0.0], [1.0], [2.0], [0.2], [1.2], [2.2]], dtype=float)
    y = np.asarray([0.0, 1.0, 3.0, 0.0, 2.0, 4.0], dtype=float)
    model = spec.create(n_estimators=8, max_depth=2, min_samples_leaf=1, min_gain=0.0)
    model.fit(X, y, groups=[3, 3])
    return model, X, {}


def _fit_auto_geo(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    X, y, coords = _sample_rows()
    model = spec.create(max_escalation_level=1)
    model.fit(
        X,
        y,
        coords=coords,
        validation={"train": [0, 1, 2, 3, 4], "holdout": [5, 6, 7]},
    )
    return model, X, {"coords": coords}


def _fit_geo_stack() -> tuple[Any, np.ndarray, dict[str, Any]]:
    X, y, coords = _sample_rows()
    model = GeoModelStack()
    model.fit(X, y, coords=coords)
    return model, X, {"coords": coords}


def _fit_nngp(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    _, y, coords = _sample_rows()
    model = spec.create(n_neighbors=3)
    model.fit(None, y, coords=coords)
    return model, coords, {"coords": coords}


def _fit_residual_nngp(spec: ModelSpec) -> tuple[Any, np.ndarray, dict[str, Any]]:
    X, y, coords = _sample_rows()
    base_spec = ModelRegistry.defaults().get("cartoboost_regressor", namespace="models")
    base = base_spec.create(n_estimators=8, max_depth=2, min_samples_leaf=1, min_gain=0.0)
    model = spec.create(base_estimator=base)
    model.fit(X, y, coords=coords)
    return model, X, {"coords": coords}


def _fit_conformal(spec: ModelSpec, *, spatial: bool) -> tuple[Any, np.ndarray, dict[str, Any]]:
    X, y, _ = _sample_rows()
    base_spec = ModelRegistry.defaults().get("cartoboost_regressor", namespace="models")
    estimator = base_spec.create(n_estimators=8, max_depth=2, min_samples_leaf=1, min_gain=0.0)
    model = spec.create(estimator=estimator, alpha=0.2)
    groups = np.asarray(["a", "a", "b", "b", "a", "a", "b", "b"], dtype=object)
    x_train, y_train = X[:5], y[:5]
    x_calibration, y_calibration = X[5:], y[5:]
    if spatial:
        model.fit(
            x_train,
            y_train,
            x_calibration,
            y_calibration,
            groups=groups[5:],
            train_end_exclusive=5,
            calibration_start=5,
            calibration_end_exclusive=8,
            test_start=8,
        )
    else:
        model.fit(
            x_train,
            y_train,
            x_calibration,
            y_calibration,
            train_end_exclusive=5,
            calibration_start=5,
            calibration_end_exclusive=8,
            test_start=8,
        )
    return model, X, {"method": "predict_interval", "test_start": 8}


def _predict_array(model: Any, X: np.ndarray, kwargs: dict[str, Any]) -> np.ndarray:
    kwargs = dict(kwargs)
    method = kwargs.pop("method", "predict")
    func = getattr(model, method)
    result = func(X, **kwargs)
    if method == "predict_interval":
        return np.column_stack([np.asarray(result.lower), np.asarray(result.upper)])
    return np.asarray(result, dtype=float)


def _sample_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(
        [[0.0], [0.4], [0.8], [1.2], [1.6], [2.0], [2.4], [2.8]],
        dtype=float,
    )
    y = 1.0 + 2.0 * X[:, 0] + 0.2 * np.sin(X[:, 0] * 3.0)
    coords = np.column_stack([X[:, 0], np.sin(X[:, 0])])
    return X, y, coords


if __name__ == "__main__":
    raise SystemExit(main())
