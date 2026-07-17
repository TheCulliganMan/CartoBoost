from __future__ import annotations

import pytest
from cartoboost.accelerators import available_backends
from cartoboost.geo_causal import (
    CounterfactualRepresentationNet,
    DomainAdversarialGeoEncoder,
    GeoCausalPanel,
    GeoExperimentDesigner,
    InvariantRiskEncoder,
    SpatialPlaceboTester,
    SyntheticDIDEstimator,
    TreatmentEffectRepresentationHead,
)


def _known_effect_rows(effect: float = 5.0) -> list[dict[str, object]]:
    rows = []
    for unit in ("treated", "control_a", "control_b", "control_c"):
        for time in range(8):
            post = time >= 4
            unit_shift = {
                "treated": 1.0,
                "control_a": 1.0,
                "control_b": 0.8,
                "control_c": 1.2,
            }[unit]
            outcome = 10.0 + time + unit_shift
            if unit == "treated" and post:
                outcome += effect
            rows.append(
                {
                    "unit_id": unit,
                    "time": f"2026-01-{time:02}",
                    "outcome": outcome,
                    "treatment": unit == "treated" and post,
                    "latitude": 40.0 if unit == "treated" else 41.0,
                    "longitude": -73.0,
                    "region_id": unit,
                }
            )
    return rows


def _panel(effect: float = 5.0) -> GeoCausalPanel:
    return GeoCausalPanel(
        _known_effect_rows(effect),
        spatial_weights=[("treated", "control_a", 1.0)],
    )


def test_synthetic_did_recovers_known_effect() -> None:
    estimator = SyntheticDIDEstimator(intervention_time="2026-01-04", seed=7).fit(_panel(5.0))
    assert abs(estimator.estimate_effect() - 5.0) < 0.25
    assert abs(estimator.predict() - 5.0) < 0.25
    assert estimator.score() > 0.0
    assert estimator.summary()["assumptions"]


def test_synthetic_did_and_placebos_run_on_every_available_backend() -> None:
    expected_effect = None
    expected_placebos = None
    for backend in available_backends("affine"):
        estimator = SyntheticDIDEstimator(
            intervention_time="2026-01-04", seed=7, backend=backend
        ).fit(_panel(5.0))
        effect = estimator.estimate_effect()
        placebos = estimator.placebo_test(n=2)
        assert estimator.backend_ == backend
        assert estimator.summary()["backend_selected"] == backend
        if expected_effect is None:
            expected_effect = effect
            expected_placebos = placebos
        else:
            assert effect == pytest.approx(expected_effect, abs=1.0e-4)
            assert placebos == pytest.approx(expected_placebos, abs=1.0e-4)


def test_synthetic_did_save_load_preserves_effect(tmp_path) -> None:
    estimator = SyntheticDIDEstimator(intervention_time="2026-01-04", seed=7).fit(_panel(5.0))
    path = tmp_path / "synthetic-did.json"
    estimator.save(path)
    loaded = SyntheticDIDEstimator.load(path)

    assert loaded.get_params() == estimator.get_params()
    assert loaded.estimate_effect() == estimator.estimate_effect()


def test_zero_effect_placebo_is_centered_near_zero_and_deterministic() -> None:
    first = SyntheticDIDEstimator(intervention_time="2026-01-04", seed=42).fit(_panel(0.0))
    second = SyntheticDIDEstimator(intervention_time="2026-01-04", seed=42).fit(_panel(0.0))
    first_placebos = first.placebo_test(n=4)
    assert first_placebos == second.placebo_test(n=4)
    assert abs(sum(first_placebos) / len(first_placebos)) < 0.5


def test_spillover_warnings_fire_for_adjacent_units() -> None:
    summary = SpatialPlaceboTester(intervention_time="2026-01-04", seed=3).fit(_panel()).summary()
    assert summary["warnings"]
    assert summary["adjacent_treated_control_pairs"] == [["treated", "control_a", 1.0]]


def test_geo_experiment_design_runs_on_every_available_backend() -> None:
    expected = None
    for backend in available_backends("affine"):
        model = GeoExperimentDesigner(intervention_time="2026-01-04", seed=9, backend=backend).fit(
            _panel(0.0)
        )
        design = model.summary(candidate_count=1, placebo_n=2)
        assert model.get_params()["backend"] == backend
        assert design["metadata"]["backend_selected"] == backend
        comparable = (
            design["candidate_test_geos"],
            design["balance_score"],
            design["placebo_estimates"],
        )
        if expected is None:
            expected = comparable
        else:
            assert comparable[0] == expected[0]
            assert comparable[1] == pytest.approx(expected[1], abs=1.0e-4)
            assert comparable[2] == pytest.approx(expected[2], abs=1.0e-4)


def test_spillover_diagnostics_run_on_every_available_backend() -> None:
    expected = None
    for backend in available_backends("pairwise_distance"):
        model = SpatialPlaceboTester(intervention_time="2026-01-04", seed=3, backend=backend).fit(
            _panel()
        )
        summary = model.summary()
        assert model.get_params()["backend"] == backend
        distances = (
            summary["min_treated_control_distance"],
            summary["mean_treated_control_distance"],
        )
        if expected is None:
            expected = distances
        else:
            assert distances == pytest.approx(expected, abs=0.02)


def test_invariant_risk_encoder_improves_heldout_region_and_warns() -> None:
    features = []
    outcomes = []
    regions = []
    for region, shift in [("a", 0.0), ("b", 3.0), ("c", -4.0)]:
        for idx in range(8):
            stable = idx / 4.0
            features.append([stable + shift, stable * 0.5 + shift])
            outcomes.append(2.0 + 1.5 * stable)
            regions.append(region)

    report = InvariantRiskEncoder().fit_report(
        features,
        outcomes,
        regions,
        heldout_region="c",
    )

    assert report["invariant_rmse"] < report["raw_rmse"]
    assert report["improvement"] > 0.0
    assert "domain_adversarial_loss" in report["losses"]
    assert any("does not prove causal identification" in item for item in report["warnings"])
    assert report["metadata"]["supplements"] == "SyntheticDIDEstimator,GeoExperimentDesigner"
    assert DomainAdversarialGeoEncoder is InvariantRiskEncoder
    assert CounterfactualRepresentationNet is InvariantRiskEncoder
    assert TreatmentEffectRepresentationHead is InvariantRiskEncoder
