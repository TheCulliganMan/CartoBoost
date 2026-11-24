from __future__ import annotations

import importlib.util
import json

import cartoboost
import numpy as np
from cartoboost.config import BoosterConfig, SplitPolicy
from cartoboost.models import native_model_manifest
from cartoboost.regressor import _resolve_splitters
from cartoboost.schema import (
    FeatureSchema,
    NumericSpec,
    PeriodicSpec,
    SparseSetSpec,
    SpatialPairSpec,
)
from cartoboost.validation import (
    SplitManifest,
    native_buffered_spatial_split,
    native_grouped_split,
    native_out_of_time_split,
    native_spatial_split,
    native_temporal_split,
)


def test_root_surface_is_unified() -> None:
    assert {"CartoBoostRegressor", "forecasting", "deep", "graph", "geostats"} <= set(
        cartoboost.__all__
    )
    assert cartoboost.__version__.startswith("0.3.")
    assert not hasattr(cartoboost, "AutoGeoModel")
    assert importlib.util.find_spec("cartoboost.representation") is None
    assert importlib.util.find_spec("cartoboost.deep.ssm") is None
    assert cartoboost.AutoForecaster.__name__ == "AutoForecaster"


def test_forecasting_surface_exposes_all_shipped_models() -> None:
    from cartoboost import forecasting

    assert {"NaiveForecaster", "ThetaForecaster", "DCRNNForecaster"} <= set(forecasting.__all__)
    assert forecasting.ThetaForecaster.__name__ == "ThetaForecaster"


def test_rust_owned_manifest_matches_stable_surface() -> None:
    manifest = native_model_manifest()
    stable = {row["key"] for row in manifest if row["tier"] == "stable"}
    assert stable == {
        "models.cartoboost_regressor",
        "models.cartoboost_classifier",
        "models.cartoboost_ranker",
        "forecasting.auto_forecaster",
        "forecasting.cartoboost_lag",
    }


def test_typed_schema_and_split_manifest_roundtrip() -> None:
    schema = FeatureSchema.from_specs(
        [NumericSpec("distance"), PeriodicSpec("hour", 24)],
        [SparseSetSpec("zones")],
    )
    payload = schema.to_rust_payload(2, ["zones"])
    assert payload["names"] == ["distance", "hour", "zones"]
    schema.validate(2, ["zones"], payload=payload)
    assert (
        BoosterConfig(split_policy=SplitPolicy.STRUCTURED).to_dict()["split_policy"] == "structured"
    )

    from cartoboost.geo import CoordinateMatrix

    manifest = native_spatial_split(
        CoordinateMatrix([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
        n_folds=2,
        dataset_fingerprint="sha256:test",
        coordinate_crs_note="unit",
        model_version="0.3.0",
        dependency_versions={},
    )
    assert isinstance(manifest, SplitManifest)
    assert len(manifest.folds()) == 2
    assert manifest.hash().startswith("sha256:")


def test_validation_exposes_native_manifest_constructors() -> None:
    assert all(
        callable(constructor)
        for constructor in (
            native_spatial_split,
            native_buffered_spatial_split,
            native_grouped_split,
            native_temporal_split,
            native_out_of_time_split,
        )
    )


def test_structured_policy_bounds_large_row_candidate_search() -> None:
    schema = FeatureSchema.from_specs(
        [NumericSpec("distance"), PeriodicSpec("hour", 24)],
        [SparseSetSpec("zones")],
    )
    assert _resolve_splitters(SplitPolicy.STRUCTURED, schema, n_rows=99_999) == [
        "axis_histogram",
        "periodic_time",
        "sparse_set",
    ]
    assert _resolve_splitters(SplitPolicy.STRUCTURED, schema, n_rows=100_000) == [
        "axis_histogram",
        "sparse_set",
    ]


def test_structured_policy_only_enables_spatial_candidates_for_declared_pairs() -> None:
    spatial_schema = FeatureSchema.from_specs(
        [NumericSpec("distance"), SpatialPairSpec("pickup_x", "pickup_y")]
    )
    numeric_schema = FeatureSchema.from_specs([NumericSpec("distance"), NumericSpec("other")])
    assert _resolve_splitters(SplitPolicy.STRUCTURED, spatial_schema, n_rows=100) == [
        "axis_histogram",
        "diagonal_2d",
        "gaussian_2d",
    ]
    assert _resolve_splitters(SplitPolicy.STRUCTURED, numeric_schema, n_rows=100) == [
        "axis_histogram"
    ]


def test_v2_artifact_envelope_roundtrip() -> None:
    model = cartoboost.CartoBoostRegressor(
        n_estimators=4,
        max_depth=1,
        min_samples_leaf=1,
        split_policy=SplitPolicy.AXIS_ONLY,
    ).fit(np.arange(8, dtype=float).reshape(-1, 1), np.arange(8, dtype=float))
    path = __import__("tempfile").NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        model.save(path)
        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["format"] == "cartoboost.model"
        assert payload["artifact_version"] == 2
        restored = cartoboost.CartoBoostRegressor.load(path)
        np.testing.assert_allclose(model.predict([[1.5], [6.5]]), restored.predict([[1.5], [6.5]]))
    finally:
        import os

        os.unlink(path)


def test_stable_artifact_rejects_mismatched_schema_hash(tmp_path):
    import pytest

    model = cartoboost.CartoBoostRegressor(n_estimators=2, min_samples_leaf=1).fit(
        [[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0]
    )
    path = tmp_path / "taxi-model.json"
    model.save(path)
    payload = json.loads(path.read_text())
    payload["schema_hash"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_hash"):
        cartoboost.CartoBoostRegressor.load(path)


def test_root_ranking_metric_keeps_graph_query_id_call_form():
    from cartoboost.graph.eval import mean_reciprocal_rank as graph_mrr
    from cartoboost.metrics import mean_reciprocal_rank as tabular_mrr

    labels = [1, 0, 0, 1, 0, 0]
    scores = [0.9, 0.5, 0.1, 0.8, 0.7, 0.6]
    query_ids = ["a", "a", "b", "b", "c", "c"]
    assert cartoboost.mean_reciprocal_rank(labels, scores, query_ids) == graph_mrr(
        labels, scores, query_ids
    )
    assert cartoboost.mean_reciprocal_rank(labels, scores, groups=[2, 2, 2]) == tabular_mrr(
        labels, scores, groups=[2, 2, 2]
    )
