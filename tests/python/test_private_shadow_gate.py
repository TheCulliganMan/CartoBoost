from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "check_private_shadow_gate.py"
    spec = importlib.util.spec_from_file_location("private_shadow_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    return {
        "schema_version": 1,
        "metric_name": "rmse",
        "candidate": {
            "primary": 0.98,
            "segments": [
                {"private_segment_name": "north-secret", "metric": 0.99, "important": True},
                {"private_segment_name": "south-secret", "metric": 1.02, "important": False},
            ],
            "artifact_roundtrip": True,
            "fit_seconds": 10.0,
            "peak_rss_mb": 128.0,
            "predict_rows_per_second": 2_000_000.0,
            "thread_speedup": 3.2,
        },
        "incumbent": {
            "primary": 1.0,
            "segments": [
                {"private_segment_name": "north-secret", "metric": 1.0},
                {"private_segment_name": "south-secret", "metric": 1.0},
            ],
        },
    }


def test_private_shadow_gate_passes_without_echoing_private_segment_names():
    report = _module().check_private_shadow(_payload())
    assert report["passed"] is True
    assert report["segment_count"] == 2
    assert "north-secret" not in str(report)
    assert "south-secret" not in str(report)


def test_private_shadow_gate_rejects_important_regression():
    payload = _payload()
    payload["candidate"]["segments"][0]["metric"] = 1.2
    report = _module().check_private_shadow(payload)
    assert report["passed"] is False
    assert report["checks"]["important_segments_within_budget"] is False


def test_private_shadow_gate_requires_thread_speedup_and_roundtrip():
    payload = _payload()
    payload["candidate"]["thread_speedup"] = 2.9
    payload["candidate"]["artifact_roundtrip"] = False
    report = _module().check_private_shadow(payload)
    assert report["passed"] is False
    assert report["checks"]["thread_speedup"] is False
    assert report["checks"]["artifact_roundtrip"] is False


def test_private_shadow_gate_rejects_unpaired_segments():
    payload = _payload()
    payload["incumbent"]["segments"] = payload["incumbent"]["segments"][:1]
    with pytest.raises(ValueError, match="segment counts"):
        _module().check_private_shadow(payload)
