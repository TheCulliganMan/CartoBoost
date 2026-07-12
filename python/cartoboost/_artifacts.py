from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any


def library_version() -> str:
    """Resolve the installed package version without importing the package root."""

    try:
        from importlib.metadata import version

        return version("cartoboost")
    except Exception:  # pragma: no cover - source checkouts without metadata
        import os

        return os.environ.get("CARTOBOOST_VERSION", "0.3.0")


ARTIFACT_VERSION = 1

# Stable estimator artifacts intentionally have a separate envelope version from
# preview/experimental Python artifacts.  This lets the v0.3 stable surface make
# a clean break without silently accepting preview payloads as production models.
STABLE_ARTIFACT_FORMAT = "cartoboost.model"
STABLE_ARTIFACT_VERSION = 2
STABLE_MODEL_TYPES = {"regressor", "classifier", "ranker"}
STABLE_FORECAST_MODEL_TYPES = {"auto_forecaster", "cartoboost_lag"}
_LEGACY_STABLE_TYPES = {
    "regressor": "cartoboost.regressor",
    "classifier": "cartoboost.classifier",
    "ranker": "cartoboost.ranker",
}


def versioned_artifact_payload(artifact_type: str, **fields: Any) -> dict[str, Any]:
    """Build a Python-side CartoBoost artifact payload with a stable schema version."""

    return {
        "artifact_type": artifact_type,
        "artifact_version": ARTIFACT_VERSION,
        **fields,
    }


def require_artifact_payload(
    payload: Mapping[str, Any],
    expected_type: str | set[str],
) -> str:
    """Validate artifact type and version, returning the concrete artifact type."""

    expected = {expected_type} if isinstance(expected_type, str) else set(expected_type)
    artifact_type = str(payload.get("artifact_type", ""))
    if artifact_type not in expected:
        expected_label = ", ".join(sorted(expected))
        raise ValueError(f"artifact is not a {expected_label}")
    version = payload.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(f"unsupported {artifact_type} artifact version {version!r}")
    return artifact_type


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def schema_hash_for_payload(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash for the schema carried by a model payload."""

    native = payload.get("native_model")
    if not isinstance(native, Mapping):
        native = payload
    schema = native.get("feature_schema")
    if schema is None:
        schema = native.get("schema")
    if schema is None:
        frame = payload.get("training_frame")
        if isinstance(frame, Mapping):
            schema = frame.get("metadata")
    # Encoders change the effective feature schema even when the native schema
    # only contains the post-encoding numeric columns.
    effective = {
        "feature_schema": schema,
        "categorical_encoder": payload.get("categorical_encoder"),
    }
    return hashlib.sha256(_canonical_json(effective).encode("utf-8")).hexdigest()


def stable_model_artifact_payload(
    model_type: str,
    *,
    payload: Mapping[str, Any],
    library_version: str,
    training_config: Any = None,
) -> dict[str, Any]:
    """Build the v2 envelope used by stable CartoBoost estimators."""

    if model_type not in STABLE_MODEL_TYPES:
        raise ValueError(f"unsupported stable model type {model_type!r}")
    return {
        "format": STABLE_ARTIFACT_FORMAT,
        "artifact_version": STABLE_ARTIFACT_VERSION,
        "model_type": model_type,
        "library_version": str(library_version),
        "schema_hash": schema_hash_for_payload(payload),
        "training_config": training_config if training_config is not None else {},
        "payload": dict(payload),
    }


def stable_forecast_artifact_payload(
    model_type: str,
    *,
    payload: Mapping[str, Any],
    library_version: str,
    training_config: Any = None,
) -> dict[str, Any]:
    """Build the v2 envelope for stable Rust-backed forecasting estimators."""

    if model_type not in STABLE_FORECAST_MODEL_TYPES:
        raise ValueError(f"unsupported stable forecast model type {model_type!r}")
    return {
        "format": STABLE_ARTIFACT_FORMAT,
        "artifact_version": STABLE_ARTIFACT_VERSION,
        "model_type": model_type,
        "library_version": str(library_version),
        "schema_hash": schema_hash_for_payload(payload),
        "training_config": training_config if training_config is not None else {},
        "payload": dict(payload),
    }


def decode_stable_forecast_artifact(
    raw: Mapping[str, Any],
    model_type: str,
) -> dict[str, Any]:
    """Validate a stable forecasting envelope without accepting preview artifacts."""

    if model_type not in STABLE_FORECAST_MODEL_TYPES:
        raise ValueError(f"unsupported stable forecast model type {model_type!r}")
    if raw.get("format") != STABLE_ARTIFACT_FORMAT:
        raise ValueError("unsupported forecast artifact: expected CartoBoost v2 stable envelope")
    if raw.get("artifact_version") != STABLE_ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported {STABLE_ARTIFACT_FORMAT} artifact version {raw.get('artifact_version')!r}"
        )
    if raw.get("model_type") != model_type:
        raise ValueError(
            f"artifact model_type {raw.get('model_type')!r} does not match {model_type!r}"
        )
    required = ("library_version", "schema_hash", "training_config", "payload")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"stable forecast artifact is missing required fields: {missing}")
    if not isinstance(raw["payload"], Mapping):
        raise ValueError("stable forecast artifact payload must be an object")
    return dict(raw)


def decode_stable_model_artifact(
    raw: Mapping[str, Any],
    model_type: str,
) -> dict[str, Any]:
    """Validate a stable envelope and migrate valid v0.2 artifacts in memory.

    The returned value is always the canonical v2 envelope.  Legacy preview
    artifact types are rejected rather than being interpreted as stable models.
    Raw native v0.2 model JSON remains loadable for the three stable boosters;
    it is wrapped in memory and never rewritten on disk.
    """

    if model_type not in STABLE_MODEL_TYPES:
        raise ValueError(f"unsupported stable model type {model_type!r}")
    format_name = raw.get("format")
    if format_name == STABLE_ARTIFACT_FORMAT:
        if raw.get("artifact_version") != STABLE_ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported {STABLE_ARTIFACT_FORMAT} artifact version "
                f"{raw.get('artifact_version')!r}"
            )
        if raw.get("model_type") != model_type:
            raise ValueError(
                f"artifact model_type {raw.get('model_type')!r} does not match {model_type!r}"
            )
        required = ("library_version", "schema_hash", "training_config", "payload")
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"stable artifact is missing required fields: {missing}")
        if not isinstance(raw["payload"], Mapping):
            raise ValueError("stable artifact payload must be an object")
        return dict(raw)

    artifact_type = raw.get("artifact_type")
    expected_legacy = _LEGACY_STABLE_TYPES[model_type]
    if artifact_type is not None:
        if artifact_type != expected_legacy or raw.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported {model_type} artifact; preview and experimental artifacts "
                "are not loadable through the stable API"
            )
        native = raw.get("native_model")
        if not isinstance(native, Mapping):
            raise ValueError("legacy stable artifact is missing native_model")
        legacy_payload = dict(raw)
        return stable_model_artifact_payload(
            model_type,
            payload=legacy_payload,
            library_version="0.2.45",
            training_config=native.get("training_config", {}),
        )

    # v0.2 native booster artifacts were written directly by Rust.  Keep this
    # narrow migration path for stable booster classes only; unknown JSON is a
    # hard error so preview payloads cannot silently fall through.
    native_markers = {
        "regressor": {"init_prediction", "trees", "learning_rate"},
        "classifier": {"init_margins", "trees", "class_values"},
        "ranker": {"init_score", "trees", "learning_rate"},
    }[model_type]
    if native_markers.issubset(raw.keys()):
        native = dict(raw)
        return stable_model_artifact_payload(
            model_type,
            payload={"native_model": native},
            library_version="0.2.45",
            training_config=native.get("training_config", {}),
        )
    raise ValueError(
        "unsupported model artifact: expected CartoBoost v2 stable envelope; "
        "preview and experimental artifacts are not loadable through the stable API"
    )


def dump_model_artifact(model: Any, *, purpose: str = "model") -> dict[str, Any]:
    """Serialize a model through its public save(path) contract."""

    save = getattr(model, "save", None)
    if not callable(save):
        raise ValueError(f"{model.__class__.__name__} must expose save(path) for {purpose}")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = Path(handle.name)
    try:
        save(path)
        return {
            "module": model.__class__.__module__,
            "class": model.__class__.__name__,
            "artifact": path.read_text(encoding="utf-8"),
        }
    finally:
        path.unlink(missing_ok=True)


def load_model_artifact(payload: Mapping[str, Any]) -> Any:
    """Load a model serialized by ``dump_model_artifact``."""

    cls = getattr(import_module(str(payload["module"])), str(payload["class"]))
    load = getattr(cls, "load", None)
    if not callable(load):
        raise ValueError(f"{payload['class']} does not expose load(path)")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = Path(handle.name)
    try:
        path.write_text(str(payload["artifact"]), encoding="utf-8")
        return load(path)
    finally:
        path.unlink(missing_ok=True)
