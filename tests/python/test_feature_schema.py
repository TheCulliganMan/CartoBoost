from pathlib import Path

import cartoboost._native as _native
import pytest
from cartoboost import CartoBoostRegressor
from cartoboost.schema import FeatureKind, FeatureSchema, SpatialPairSpec


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def test_feature_schema_helper_builds_rust_payload():
    schema = FeatureSchema(
        dense=[
            ("distance_m", "numeric"),
            ("pickup_centroid_x", "spatial"),
            ("hour_of_day", {"periodic": 24}),
        ],
        sparse_sets=[("route_cells", "sparse_set")],
    )

    payload = schema.to_rust_payload(dense_width=3, sparse_names=["route_cells"])

    assert payload == {
        "names": ["distance_m", "pickup_centroid_x", "hour_of_day", "route_cells"],
        "kinds": ["Numeric", "Spatial", {"Periodic": {"period": 24}}, "SparseSet"],
    }


def test_feature_schema_helper_accepts_feature_kind_enum():
    schema = FeatureSchema(
        dense=[
            ("distance_m", FeatureKind.NUMERIC),
            {"name": "hour_of_day", "kind": FeatureKind.PERIODIC, "period": 24},
        ],
        sparse_sets=[("route_cells", FeatureKind.H3_SPARSE_SET)],
    )

    payload = schema.to_rust_payload(dense_width=2, sparse_names=["route_cells"])

    assert payload == {
        "names": ["distance_m", "hour_of_day", "route_cells"],
        "kinds": ["Numeric", {"Periodic": {"period": 24}}, "SparseSet"],
    }


def test_spatial_pair_schema_marks_both_coordinates_for_native_search():
    schema = FeatureSchema.from_specs(
        [SpatialPairSpec("pickup_x", "pickup_y"), ("pickup_y", "numeric")]
    )
    payload = schema.to_rust_payload(dense_width=2, sparse_names=[])
    assert payload["names"] == ["pickup_x", "pickup_y"]
    assert payload["kinds"] == [FeatureKind.SPATIAL, FeatureKind.SPATIAL]


def test_compact_spatial_pair_schema_marks_both_coordinates_for_native_search():
    model = CartoBoostRegressor(
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        split_policy="structured",
    ).fit(
        [[-1.0, -1.0], [1.0, 1.0]],
        [-1.0, 1.0],
        feature_schema={
            "dense": [
                {"name": "pickup_x", "kind": "spatial", "other": "pickup_y"},
                {"name": "pickup_y", "kind": "numeric"},
            ]
        },
    )
    assert model._model.splitters == ["axis_histogram", "diagonal_2d", "gaussian_2d"]


def test_native_feature_schema_validator_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate feature name"):
        _native.validate_feature_schema_json('{"names":["x","x"],"kinds":["Numeric","Numeric"]}')


def test_python_schema_periodic_feature_used_without_full_period_coverage():
    x = [[7.0], [8.0], [9.0], [10.0]]
    y = [3.0, 3.0, -1.0, -1.0]
    schema = FeatureSchema(dense=[("hour", {"periodic": 24})])
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )

    _fit_or_skip(model, x, y, feature_schema=schema)

    assert model.predict(x) == pytest.approx(y)
    assert model.feature_schema_["names"] == ["hour"]
    assert model.feature_schema_["kinds"] == [{"Periodic": {"period": 24}}]


def test_python_schema_sparse_columns_restrict_sparse_splitter():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [4.0, 4.0, -2.0, -2.0]
    schema = FeatureSchema(
        dense=[("bias", "numeric")],
        sparse_sets=[("route_cells", "sparse_set")],
    )
    sparse_sets = {"route_cells": [[5], [5, 9], [2], []]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets, feature_schema=schema)

    assert model.predict(x, sparse_sets=sparse_sets) == pytest.approx(y)
    assert model.feature_schema_["names"] == ["bias", "route_cells"]
    assert model.feature_schema_["kinds"] == ["Numeric", "SparseSet"]


def test_python_schema_rejects_unknown_kind():
    model = CartoBoostRegressor(max_depth=0)
    schema = FeatureSchema(dense=[("hour", "clockish")])

    with pytest.raises(ValueError, match="unknown feature kind"):
        model.fit([[0.0]], [0.0], feature_schema=schema)


def test_python_schema_rejects_length_mismatch():
    model = CartoBoostRegressor(max_depth=0)
    schema = FeatureSchema(dense=[("only_one", "numeric")])

    with pytest.raises(ValueError, match="feature_schema length"):
        model.fit([[0.0, 1.0], [2.0, 3.0]], [1.0, 2.0], feature_schema=schema)


def test_python_schema_accepts_geographic_sparse_set_aliases():
    schema = FeatureSchema(
        dense=[("distance", "numeric")],
        sparse_sets=[("ozip_zip5", "zip_sparse_set"), ("dzip_zip5", "h3_sparse_set")],
    )

    payload = schema.to_rust_payload(dense_width=1, sparse_names=["ozip_zip5", "dzip_zip5"])

    assert payload["kinds"] == ["Numeric", "SparseSet", "SparseSet"]

    schema_with_zip3 = FeatureSchema(
        dense=[("distance", "numeric")],
        sparse_sets=[("ozip_zip_p3", "zip3_sparse_set")],
    )
    zip3_payload = schema_with_zip3.to_rust_payload(dense_width=1, sparse_names=["ozip_zip_p3"])
    assert zip3_payload["kinds"] == ["Numeric", "SparseSet"]

    schema_with_zone = FeatureSchema(
        dense=[("distance", "numeric")],
        sparse_sets=[
            ("zone_id", "zone_sparse_set"),
            ("region_id", "region_sparse_set"),
            ("geo_key", "GeoAbstractSparseSet"),
        ],
    )
    zone_payload = schema_with_zone.to_rust_payload(
        dense_width=1,
        sparse_names=["zone_id", "region_id", "geo_key"],
    )
    assert zone_payload["kinds"] == ["Numeric", "SparseSet", "SparseSet", "SparseSet"]


def test_python_schema_rejects_sparse_length_mismatch():
    model = CartoBoostRegressor(max_depth=0)
    schema = FeatureSchema(dense=[("x", "numeric")])

    with pytest.raises(ValueError, match="feature_schema length"):
        model.fit(
            [[0.0], [1.0]],
            [0.0, 1.0],
            sparse_sets={"route_cells": [[7], [8]]},
            feature_schema=schema,
        )


def test_schema_survives_save_load(tmp_path: Path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [4.0, 4.0, -2.0, -2.0]
    schema = FeatureSchema(
        dense=[("bias", "numeric")],
        sparse_sets=[("route_cells", "sparse_set")],
    )
    sparse_sets = {"route_cells": [[5], [5, 9], [2], []]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets, feature_schema=schema)
    path = tmp_path / "schema.json"

    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    assert loaded.feature_schema_ == model.feature_schema_
    assert loaded.predict(x, sparse_sets=sparse_sets) == pytest.approx(
        model.predict(x, sparse_sets=sparse_sets)
    )
