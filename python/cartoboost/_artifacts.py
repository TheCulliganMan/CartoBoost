from __future__ import annotations

import tempfile
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = 1


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
