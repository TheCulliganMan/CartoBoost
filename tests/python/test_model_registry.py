from __future__ import annotations

import importlib
import sys
from pathlib import Path

from cartoboost.capabilities import stable_capability_manifest, validate_capability_table
from cartoboost.models import (
    STABLE_MODEL_KEYS,
    ModelRegistry,
    model_manifest,
    native_model_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
public_api_audit = importlib.import_module("scripts.check_public_api_contract")


def test_default_registry_covers_stable_namespaces() -> None:
    registry = ModelRegistry.defaults()
    assert "forecasting.cartoboost_lag" in registry.names()
    assert "geo.residual_nngp" in registry.names()
    assert "graph.dcrnn" in registry.names()
    assert "causal.synthetic_did" in registry.names()
    assert "prob.conformal_interval" in registry.names()
    assert {item.namespace for item in registry.metadata()} >= {
        "models",
        "forecasting",
        "geo",
        "causal",
        "prob",
    }


def test_rust_manifest_is_the_registry_source_of_truth() -> None:
    registry_keys = set(ModelRegistry.defaults().names())
    native_keys = {str(row["key"]) for row in native_model_manifest()}
    assert native_keys == registry_keys


def test_auto_geo_selector_is_not_registered() -> None:
    registry = ModelRegistry.defaults()
    assert all("auto_geo" not in key and "geo_model_stack" not in key for key in registry.names())


def test_model_manifest_has_auditable_v03_tiers() -> None:
    manifest = model_manifest()
    assert manifest
    assert {row["tier"] for row in manifest} == {"stable", "preview", "experimental"}
    assert {row["key"] for row in model_manifest(tier="stable")} == set(STABLE_MODEL_KEYS)
    for row in manifest:
        assert row["artifact_version"] == 2
        assert row["backend"]
        assert row["task_types"]
        assert row["dependencies"] == row["optional_dependencies"]
        assert row["evidence_level"] in {"real_data", "synthetic", "api_only", "experimental_only"}
        assert row["stable"] is (row["tier"] == "stable")


def test_registry_tier_filters_do_not_mix_public_surface() -> None:
    registry = ModelRegistry.defaults()
    assert all(spec.metadata.tier == "stable" for spec in registry.by_tier("stable").specs())
    assert all(spec.metadata.tier == "preview" for spec in registry.by_tier("preview").specs())
    assert all(
        spec.metadata.tier == "experimental" for spec in registry.by_tier("experimental").specs()
    )


def test_capability_manifest_matches_stable_registry() -> None:
    assert validate_capability_table() == []
    assert {row["model_key"] for row in stable_capability_manifest()} == set(STABLE_MODEL_KEYS)


def test_registered_model_factories_are_constructible() -> None:
    registry = ModelRegistry.defaults()
    for spec in registry.specs():
        assert callable(spec.factory), spec.key
        assert spec.metadata.tier in {"stable", "preview", "experimental"}


def test_stable_registry_exposes_lifecycle_contract() -> None:
    required = {"fit", "predict", "score", "save", "load", "get_params", "set_params"}
    for spec in ModelRegistry.stable_defaults().specs():
        missing = sorted(name for name in required if not hasattr(spec.factory, name))
        assert not missing, f"{spec.key} missing {missing}"


def test_public_api_audit_has_no_removed_geo_selector() -> None:
    assert not hasattr(__import__("cartoboost.models", fromlist=["x"]), "AutoGeoModel")
    assert not hasattr(__import__("cartoboost.models", fromlist=["x"]), "GeoModelStack")
    assert public_api_audit.STABLE_MODEL_KEYS == STABLE_MODEL_KEYS
