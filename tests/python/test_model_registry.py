from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

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


def _evaluation_rows(model: AutoGeoModel) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], model.metadata_["evaluations"])


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


def test_auto_geo_coords_escalate_to_residual_nngp_candidate() -> None:
    x, y, coords = _rows()

    model = AutoGeoModel(max_escalation_level=1).fit(
        x,
        y,
        coords=coords,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    rows = {row["family"]: row for row in _evaluation_rows(model)}
    assert rows["geo.residual_nngp"]["status"] == "fit"
    assert rows["geo.residual_nngp"]["prediction_parity"]["passed"] is True
    assert rows["geo.nngp"]["status"] == "fit"
    assert rows["geo.nngp"]["prediction_parity"]["passed"] is True
    assert "geo.residual_nngp" in model.metadata_["evidence_card"]["candidates_tried"]
    assert "geo.nngp" in model.metadata_["evidence_card"]["candidates_tried"]


def test_auto_geo_panel_time_evaluates_forecasting_adapter() -> None:
    x, y, _coords = _rows()
    panel_id = ["pu1"] * 6 + ["pu2"] * 6
    time_index = [f"2024-01-{day:02d}" for day in range(1, 7)] * 2

    model = AutoGeoModel(max_escalation_level=2).fit(
        x,
        y,
        panel_id=panel_id,
        time_index=time_index,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    rows = {row["family"]: row for row in _evaluation_rows(model)}
    assert "forecasting.cartoboost_lag" in rows
    assert rows["forecasting.cartoboost_lag"]["status"] in {"fit", "skipped"}
    if rows["forecasting.cartoboost_lag"]["status"] == "skipped":
        assert rows["forecasting.cartoboost_lag"]["reason_code"]
    assert rows["forecasting.auto_forecaster"]["status"] == "skipped"
    assert rows["forecasting.auto_forecaster"]["reason_code"] == "missing_serialization_contract"


def test_auto_geo_graph_candidate_has_explicit_skip_reason() -> None:
    x, y, _coords = _rows()

    model = AutoGeoModel(max_escalation_level=2).fit(
        x,
        y,
        graph={"nodes": ["pu1", "pu2"]},
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    graph_rows = [row for row in _evaluation_rows(model) if row["family"] == "graph.dcrnn"]
    assert graph_rows
    assert graph_rows[0]["status"] == "skipped"
    assert graph_rows[0]["reason_code"] == "unsupported_graph_contract"


def test_auto_geo_interval_evidence_evaluates_conformal_adapter() -> None:
    x, y, _coords = _rows()

    model = AutoGeoModel(max_escalation_level=3).fit(
        x,
        y,
        interval_evidence=True,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    rows = {row["family"]: row for row in _evaluation_rows(model)}
    assert rows["prob.conformal_interval"]["status"] == "fit"
    assert rows["prob.conformal_interval"]["prediction_parity"]["passed"] is True
    assert "mean_interval_width" in rows["prob.conformal_interval"]["diagnostics"]
    assert (
        "prob.conformal_interval.mean_interval_width"
        in model.metadata_["evidence_card"]["interval_diagnostics"]
    )


def test_auto_geo_evidence_card_records_baseline_split_and_limitations() -> None:
    x, y, _coords = _rows()

    model = AutoGeoModel(max_escalation_level=3).fit(
        x,
        y,
        interval_evidence=True,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    card = model.metadata_["evidence_card"]
    assert card["selected_family"] == model.selected_family_
    assert card["baseline_comparison"]["baseline_family"] == "models.mean_baseline"
    assert len(card["split_hash"]) == 64
    assert card["leakage_policy"]["validation_strategy"] == "spatial_holdout"
    assert "forecasting candidates require both panel_id and time_index" in card["limitations"]


def test_auto_geo_evidence_card_tracks_deep_candidate_routing() -> None:
    x, y, _coords = _rows()
    panel_id = ["pu1"] * 6 + ["pu2"] * 6
    time_index = [f"2024-01-{day:02d}" for day in range(1, 7)] * 2
    source_id = ["pu1", "pu1", "pu2", "pu2"] * 3
    target_id = ["do1", "do2", "do1", "do2"] * 3

    model = AutoGeoModel(max_escalation_level=3).fit(
        x,
        y,
        graph={"nodes": ["pu1", "pu2"]},
        panel_id=panel_id,
        time_index=time_index,
        source_id=source_id,
        target_id=target_id,
        feature_roles={
            "candidate_set": ["candidate_id"],
            "decision_output": True,
            "multi_view_graph": ["physical_distance", "observed_flow"],
        },
        interval_evidence=True,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    card = model.metadata_["evidence_card"]
    expected = {
        "deep.pair_embedding_mlp",
        "deep.temporal_ssm",
        "deep.inverted_transformer",
        "deep.delay_aware_graph_transformer",
        "deep.regime_moe",
        "deep.choice_set_transformer",
        "deep.flow_uncertainty_head",
        "deep.multi_view_spatial_attention",
    }
    assert expected <= set(card["all_candidates"])
    skipped = {row["family"]: row for row in card["skipped_candidates_with_reasons"]}
    for family in expected:
        assert (
            skipped[family]["reason_code"] == "deep_candidate_not_registered_for_autogeo_selection"
        )
    assert card["split_manifest"]["strategy"] == "spatial_holdout"
    assert "models.mean_baseline" in card["claim_falsifier_baselines"]
    assert card["uncertainty_report"]["requested"] is True
    assert card["feature_roles"]["candidate_set"] == ["candidate_id"]
    assert model.metadata_["context"]["has_source_id"] is True
    assert model.metadata_["context"]["has_target_id"] is True
    assert model.metadata_["context"]["has_multi_view_graph"] is True


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


def test_fitted_auto_geo_model_card_exposes_evidence_card() -> None:
    x, y, coords = _rows()
    model = AutoGeoModel(max_escalation_level=1).fit(
        x,
        y,
        coords=coords,
        validation={"train": list(range(8)), "holdout": list(range(8, 12))},
    )

    card = model_card(model)

    assert card["evidence_card"]["selected_family"] == model.selected_family_
    assert "baseline_comparison" in card["evidence_card"]
    assert "geo.nngp" in card["evidence_card"]["candidates_tried"]


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
