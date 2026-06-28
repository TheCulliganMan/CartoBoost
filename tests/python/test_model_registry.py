from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from cartoboost.models import AutoGeoModel, GeoModelStack, ModelRegistry, model_card

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

public_api_audit = importlib.import_module("scripts.check_public_api_contract")


def _rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 1.0
    coords = np.column_stack([x[:, 0], np.sin(x[:, 0])])
    return x, y, coords


def test_default_registry_covers_stable_namespaces() -> None:
    registry = ModelRegistry.defaults()

    assert "models.auto_geo_model" in registry.names()
    assert "forecasting.cartoboost_lag" in registry.names()
    assert "geo.residual_nngp" in registry.names()
    assert "graph.dcrnn" in registry.names()
    assert "causal.synthetic_did" in registry.names()
    assert "prob.conformal_interval" in registry.names()

    metadata = {item.namespace for item in registry.metadata()}
    assert {"models", "forecasting", "geo", "causal", "prob"} <= metadata


def test_registered_model_factories_expose_lifecycle_contract() -> None:
    required = {"fit", "predict", "score", "save", "load", "get_params", "set_params"}
    registry = ModelRegistry.defaults()

    for spec in registry.specs():
        missing = {
            name
            for name in required
            if not hasattr(spec.factory, name) and not hasattr(spec.factory, "__dict__")
        }
        class_missing = {name for name in required if not hasattr(spec.factory, name)}
        assert not class_missing, f"{spec.namespace}.{spec.name} missing {class_missing}"
        assert not missing


def test_auto_geo_rejects_random_row_cv_for_spatial_claims() -> None:
    x, y, coords = _rows()

    with pytest.raises(ValueError, match="random row CV"):
        AutoGeoModel().fit(x, y, coords=coords, validation_strategy="random_row_cv")


def test_auto_geo_model_selects_and_round_trips_without_prediction_drift(tmp_path) -> None:
    x, y, coords = _rows()
    model = AutoGeoModel(max_escalation_level=1).fit(
        x,
        y,
        coords=coords,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    pred = model.predict(x, coords=coords)
    path = tmp_path / "auto-geo.pkl"
    model.save(path)
    loaded = AutoGeoModel.load(path)

    assert model.metadata_["leakage_safe"] is True
    assert loaded.selected_family_ == model.selected_family_
    np.testing.assert_allclose(loaded.predict(x, coords=coords), pred)


def test_model_card_reports_lifecycle_surface() -> None:
    card = model_card(AutoGeoModel())

    assert card["class"] == "AutoGeoModel"
    assert card["lifecycle"]["fit"] is True
    assert card["lifecycle"]["predict"] is True
    assert card["lifecycle"]["score"] is True
    assert card["lifecycle"]["save"] is True
    assert card["lifecycle"]["load"] is True
    assert card["lifecycle"]["get_params"] is True
    assert card["lifecycle"]["set_params"] is True


def test_geo_model_stack_explains_layers_and_round_trips(tmp_path) -> None:
    x, y, coords = _rows()
    stack = GeoModelStack().fit(x, y, coords=coords)

    explanation = stack.explain_layers()
    assert explanation["layers"][0]["layer"] == "tabular_booster"

    pred = stack.predict(x, coords=coords)
    path = tmp_path / "stack.pkl"
    stack.save(path)
    loaded = GeoModelStack.load(path)
    np.testing.assert_allclose(loaded.predict(x, coords=coords), pred)


def test_public_api_contract_audit_covers_registry_roundtrips() -> None:
    registry = ModelRegistry.defaults()
    structural = [public_api_audit.audit_spec(spec) for spec in registry.specs()]
    roundtrips = [
        public_api_audit.run_roundtrip_case(public_api_audit.get_registry_key(registry, key))
        for key in sorted(public_api_audit.ROUNDTRIP_CASES)
    ]

    assert all(row["passed"] for row in structural)
    assert len(roundtrips) >= 9
    assert all(row["passed"] for row in roundtrips)
    assert {
        "models.cartoboost_regressor",
        "models.auto_geo_model",
        "geo.residual_nngp",
        "prob.conformal_interval",
    } <= {row["key"] for row in roundtrips}
