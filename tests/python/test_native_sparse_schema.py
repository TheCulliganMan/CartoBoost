from pathlib import Path

import pytest
from cartoboost import CartoBoostRegressor, FeatureKind
from cartoboost.regressor import _transform_categorical_features


def _fit_or_skip(model, *args, **kwargs):
    try:
        model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    return model


def test_python_native_sparse_list_route_cells_train_predict():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)

    assert model.predict(x, sparse_sets=sparse_sets) == pytest.approx(y)
    with pytest.raises(ValueError, match="sparse_sets"):
        model.predict(x)


def test_python_native_sparse_list_two_tree_boosting():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [10.0, 10.0, 0.0, 0.0]
    sparse_sets = {"route_cells": [[7, 11], [11], [3], []]}
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)

    assert model.predict(x, sparse_sets=sparse_sets) == pytest.approx([8.75, 8.75, 1.25, 1.25])


def test_python_native_sparse_list_save_load_identity(tmp_path: Path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)
    path = tmp_path / "sparse-list.json"
    before = model.predict(x, sparse_sets=sparse_sets)

    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    assert loaded.predict(x, sparse_sets=sparse_sets) == pytest.approx(before)
    assert loaded.feature_schema_["names"] == ["feature_0", "route_cells"]


def test_python_schema_periodic_feature_used_without_full_period_coverage():
    x = [[7.0], [8.0], [9.0], [10.0]]
    y = [3.0, 3.0, -1.0, -1.0]
    schema = {"dense": [{"name": "hour", "kind": FeatureKind.PERIODIC, "period": 24}]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["periodic:24"],
    )

    _fit_or_skip(model, x, y, feature_schema=schema)

    assert model.predict(x) == pytest.approx(y)
    assert model.feature_schema_["names"] == ["hour"]


def test_python_schema_categorical_feature_roundtrip(tmp_path: Path):
    x = [["airport"], ["airport"], ["midtown"], ["midtown"]]
    y = [12.0, 12.0, 3.0, 3.0]
    schema = {"dense": [{"name": "pickup_zone", "kind": FeatureKind.CATEGORICAL}]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis"],
    )
    _fit_or_skip(model, x, y, feature_schema=schema)
    path = tmp_path / "categorical-regressor.json"
    before = model.predict(x)

    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    assert loaded.n_features_in_ == 1
    assert loaded.categorical_encoder_["columns"][0]["strategy"] == "OneHot"
    assert loaded.predict(x) == pytest.approx(before)
    assert loaded.predict([["unknown"]]).shape == (1,)
    weights_path = tmp_path / "categorical-weights.json"
    loaded.save_weights(weights_path)
    weights_loaded = CartoBoostRegressor.load_weights(weights_path)
    assert weights_loaded.predict(x) == pytest.approx(before)


def test_python_schema_pandas_categorical_missing_values_roundtrip():
    pd = pytest.importorskip("pandas")
    x = pd.DataFrame(
        {
            "pickup_zone": pd.Series(
                ["airport", "airport", pd.NA, "midtown", pd.NA],
                dtype="category",
            ),
            "trip_distance": [8.0, 7.5, 3.0, 2.0, 3.5],
        }
    )
    y = [12.0, 12.0, 5.0, 3.0, 5.0]
    schema = {"dense": [{"name": "pickup_zone", "kind": FeatureKind.CATEGORICAL}]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis"],
    )

    _fit_or_skip(model, x, y, feature_schema=schema)

    categories = model.categorical_encoder_["columns"][0]["categories"]
    assert "<missing>" in categories
    assert model.predict(x).shape == (5,)


def test_python_schema_preserves_periodic_role_after_categorical_encoding():
    x = [
        ["airport", 23.0],
        ["airport", 0.0],
        ["midtown", 11.0],
        ["midtown", 12.0],
    ]
    y = [5.0, 5.0, -1.0, -1.0]
    schema = {
        "dense": [
            {"name": "pickup_zone", "kind": FeatureKind.CATEGORICAL},
            {"name": "pickup_hour", "kind": FeatureKind.PERIODIC, "period": 24},
        ]
    }
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis"],
    )

    _fit_or_skip(model, x, y, feature_schema=schema)

    assert model.feature_schema_["names"] == [
        "pickup_zone=str:airport",
        "pickup_zone=str:midtown",
        "pickup_hour",
    ]
    assert model.feature_schema_["kinds"] == [
        "Numeric",
        "Numeric",
        {"Periodic": {"period": 24}},
    ]


def test_python_schema_low_cardinality_categorical_uses_partition_artifact():
    x = [["airport"], ["midtown"], ["uptown"], ["downtown"]]
    y = [10.0, 3.0, 5.0, 7.0]
    schema = {"dense": [{"name": "pickup_zone", "kind": FeatureKind.CATEGORICAL}]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis"],
    )

    _fit_or_skip(model, x, y, feature_schema=schema)

    assert model.categorical_encoder_["columns"][0]["strategy"] == "Partition"
    assert len(model.categorical_encoder_["columns"][0]["partitions"]) == 7
    assert model.encoded_n_features_in_ == 7


def test_python_schema_partition_fallback_transform_accepts_enum_strategy():
    encoder = {
        "original_feature_count": 1,
        "columns": [
            {
                "index": 0,
                "strategy": "Partition",
                "partitions": [["str:airport"], ["str:midtown", "str:uptown"]],
            }
        ],
    }

    transformed = _transform_categorical_features(
        [["airport"], ["midtown"], ["downtown"]],
        encoder,
    )

    assert transformed.tolist() == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def test_python_schema_rejects_length_mismatch():
    model = CartoBoostRegressor(max_depth=0)

    with pytest.raises(ValueError, match="feature_schema length"):
        model.fit(
            [[0.0, 1.0], [2.0, 3.0]],
            [1.0, 2.0],
            feature_schema={"dense": [{"name": "only_one", "kind": FeatureKind.NUMERIC}]},
        )


def test_real_native_save_load_restores_public_params(tmp_path: Path):
    x = [[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [0.0, 3.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.25,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=0.5,
        fuzzy_kernel="tricube",
        leaf_predictor="constant",
    )
    _fit_or_skip(model, x, y)
    path = tmp_path / "native.json"

    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    assert loaded.splitters == ["gaussian_2d"]
    assert loaded.fuzzy is True
    assert loaded.fuzzy_bandwidth == 0.5
    assert loaded.fuzzy_kernel == "tricube"
    assert loaded.learning_rate == 0.25
    assert loaded.n_estimators == 2
    assert loaded.predict(x) == pytest.approx(model.predict(x))
