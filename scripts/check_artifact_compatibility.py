#!/usr/bin/env python3
"""Audit model artifact schema/version compatibility for stable registry cases."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
for path in (ROOT, PYTHON_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cartoboost.models import ModelRegistry  # noqa: E402

from scripts.check_public_api_contract import (  # noqa: E402
    ROUNDTRIP_CASES,
    _predict_array,
    fit_registry_roundtrip_case,
)


def main() -> int:
    report = check_artifact_compatibility()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        failed = [row["key"] for row in report["cases"] if not row["passed"]]
        raise SystemExit("artifact compatibility checks failed: " + ", ".join(failed))
    return 0


def check_artifact_compatibility() -> dict[str, Any]:
    registry = ModelRegistry.defaults()
    cases = [check_case(registry, key) for key in sorted(ROUNDTRIP_CASES)]
    return {
        "artifact_type": "cartoboost.artifact_compatibility_audit",
        "artifact_version": 1,
        "cases": cases,
        "passed": all(row["passed"] for row in cases),
    }


def check_case(registry: ModelRegistry, key: str) -> dict[str, Any]:
    spec = registry.get(key.split(".", 1)[1], namespace=key.split(".", 1)[0])
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"{key.replace('.', '-')}.json"
        mutated_path = Path(temp_dir) / f"{key.replace('.', '-')}-mutated.json"
        model, X, kwargs = fit_registry_roundtrip_case(spec)
        before = _predict_array(model, X, kwargs)
        model.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        markers = collect_version_markers(payload)
        loaded = spec.factory.load(path)
        after = _predict_array(loaded, X, kwargs)
        drift = float(np.max(np.abs(before - after))) if before.size else 0.0
        mutated = mutate_first_artifact_version(payload)
        mutation_error = ""
        mutation_rejected = False
        if mutated:
            mutated_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            try:
                spec.factory.load(mutated_path)
            except Exception as exc:  # noqa: BLE001 - error text is part of the audit.
                error = str(exc).lower()
                mutation_rejected = "unsupported" in error and "version" in error
                mutation_error = f"{exc.__class__.__name__}: {exc}"
        return {
            "key": key,
            "version_markers": markers,
            "roundtrip_max_abs_diff": drift,
            "unsupported_version_rejected": mutation_rejected,
            "unsupported_version_error": mutation_error,
            "passed": bool(markers) and mutated and mutation_rejected and drift <= 1e-10,
        }


def collect_version_markers(payload: Any, prefix: str = "$") -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if "artifact_version" in payload:
            markers.append(
                {
                    "path": prefix,
                    "artifact_type": payload.get("artifact_type", "native_model"),
                    "artifact_version": payload["artifact_version"],
                }
            )
        for key, value in payload.items():
            if key == "artifact" and isinstance(value, str):
                try:
                    nested = json.loads(value)
                except json.JSONDecodeError:
                    continue
                markers.extend(collect_version_markers(nested, f"{prefix}.{key}"))
            else:
                markers.extend(collect_version_markers(value, f"{prefix}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            markers.extend(collect_version_markers(value, f"{prefix}[{index}]"))
    return markers


def mutate_first_artifact_version(payload: Any) -> bool:
    if isinstance(payload, dict):
        if "artifact_version" in payload:
            payload["artifact_version"] = 999
            return True
        for key, value in payload.items():
            if key == "artifact" and isinstance(value, str):
                try:
                    nested = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if mutate_first_artifact_version(nested):
                    payload[key] = json.dumps(nested, sort_keys=True)
                    return True
            elif mutate_first_artifact_version(value):
                return True
    elif isinstance(payload, list):
        for value in payload:
            if mutate_first_artifact_version(value):
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
