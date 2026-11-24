import numpy as np
import pytest
from cartoboost import CartoBoostRegressor, FeatureKind, FeatureSchema
from cartoboost.regressor import _encode_sparse_columns, _rust_feature_schema_json

try:
    from cartoboost._native import CartoBoostRegressor as NativeCartoBoostRegressor
except ImportError:  # pragma: no cover - extension may be absent in source-only runs
    NativeCartoBoostRegressor = None


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def test_numpy_dense_fast_path_matches_list_path():
    x = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]]
    y = [0.0, 0.5, 1.0, 1.5, 2.0]
    config = dict(
        n_estimators=3,
        learning_rate=0.4,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    )
    list_model = CartoBoostRegressor(**config)
    array_model = CartoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y)
    _fit_or_skip(array_model, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))

    probes = np.asarray([[0.5, 1.0], [2.5, 0.0], [4.0, 1.0]], dtype=np.float64)
    assert array_model.predict(probes) == pytest.approx(list_model.predict(probes.tolist()))


def test_numpy_sparse_fast_path_matches_list_path():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )
    list_model = CartoBoostRegressor(**config)
    array_model = CartoBoostRegressor(**config)

    _fit_or_skip(
        list_model,
        x,
        y,
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )
    _fit_or_skip(
        array_model,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )

    assert array_model.predict(np.asarray(x, dtype=np.float64), sparse_sets=sparse_sets) == (
        pytest.approx(list_model.predict(x, sparse_sets=sparse_sets))
    )


def test_numpy_sample_weight_fast_path_matches_list_path():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 10.0, 10.0]
    sample_weight = [1.0, 3.0, 1.0, 3.0]
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    )
    list_model = CartoBoostRegressor(**config)
    array_model = CartoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y, sample_weight=sample_weight)
    _fit_or_skip(
        array_model,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        sample_weight=np.asarray(sample_weight, dtype=np.float64),
    )

    assert array_model.predict(np.asarray(x, dtype=np.float64)) == pytest.approx(
        list_model.predict(x)
    )


def test_numpy_feature_schema_fast_path_matches_list_path():
    x = [[23.0], [1.0], [12.0], [15.0]]
    y = [5.0, 5.0, -5.0, -5.0]
    schema = FeatureSchema(dense=[("hour", {"periodic": 24})])
    config = dict(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )
    list_model = CartoBoostRegressor(**config)
    array_model = CartoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y, feature_schema=schema)
    _fit_or_skip(
        array_model,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        feature_schema=schema,
    )

    assert array_model.feature_schema_ == list_model.feature_schema_
    assert array_model.predict(np.asarray(x, dtype=np.float64)) == pytest.approx(
        list_model.predict(x)
    )


def test_native_numpy_fast_paths_match_native_list_paths_exactly():
    if NativeCartoBoostRegressor is None:
        pytest.skip("native extension is not available")

    x = [[23.0, 0.0], [1.0, 0.0], [12.0, 0.0], [15.0, 0.0], [2.0, 0.0]]
    y = [4.0, 4.5, -2.0, -2.5, 5.0]
    sample_weight = [1.0, 2.0, 1.0, 2.0, 1.0]
    sparse_sets = [[[10, 20], [20], [30], [], [10]]]
    sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_sets)
    schema_json = _rust_feature_schema_json(
        {
            "dense": [
                {"name": "hour", "kind": FeatureKind.PERIODIC, "period": 24},
                {"name": "dummy", "kind": FeatureKind.NUMERIC},
            ],
            "sparse_sets": [{"name": "route_cells", "kind": FeatureKind.SPARSE_SET}],
        },
        dense_width=2,
        sparse_names=["route_cells"],
    )
    config = dict(
        n_estimators=3,
        learning_rate=0.3,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis", "periodic:24", "sparse_set"],
    )
    list_model = NativeCartoBoostRegressor(**config)
    array_model = NativeCartoBoostRegressor(**config)

    list_model.fit(x, y, sample_weight, sparse_sets, schema_json)
    array_model.fit_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(sample_weight, dtype=np.float64),
        sparse_offsets,
        sparse_ids,
        schema_json,
    )

    fast_predictions = np.asarray(
        array_model.predict_arrays(np.asarray(x, dtype=np.float64), sparse_offsets, sparse_ids)
    )
    assert fast_predictions == pytest.approx(list_model.predict(x, sparse_sets))


def test_structured_policy_uses_axis_histogram():
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )

    _fit_or_skip(
        model,
        np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float64),
    )

    assert model.training_config_["splitters"] == [{"AxisHistogram": {"bins": 64}}]
    predictions = model.predict(np.asarray([[0.0], [3.0]], dtype=np.float64))
    assert predictions[0] < predictions[1]


def test_default_splitter_uses_auto_and_resolves_small_fit_to_axis():
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
    )

    _fit_or_skip(
        model,
        np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float64),
    )

    assert model.split_policy == "auto"
    assert model.training_config_["splitters"] == ["Axis"]


def test_explicit_axis_splitter_is_preserved_by_python_api():
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    )

    _fit_or_skip(
        model,
        np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float64),
    )

    assert model.split_policy == "axis_only"
    assert model.training_config_["splitters"] == ["Axis"]


@pytest.mark.parametrize("layout", ["fortran", "reversed", "strided"])
def test_native_numpy_logical_layout_matches_row_major(layout):
    assert NativeCartoBoostRegressor is not None
    x = np.arange(48, dtype=np.float64).reshape(12, 4)
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "reversed":
        x = x[::-1, ::-1]
    else:
        x = x[::2, ::2]
    y = np.ascontiguousarray(x[:, 0] * 0.25 + x[:, -1] ** 2)
    config = dict(n_estimators=3, max_depth=2, min_samples_leaf=1, splitters=["axis"])
    reference = NativeCartoBoostRegressor(**config)
    reference.fit_arrays(np.ascontiguousarray(x), y)
    model = NativeCartoBoostRegressor(**config)
    model.fit_arrays(x, y)
    expected = reference.predict_arrays(np.ascontiguousarray(x))
    np.testing.assert_array_equal(model.predict_arrays(x), expected)
    np.testing.assert_array_equal(
        model.predict_additive_arrays(x),
        reference.predict_additive_arrays(np.ascontiguousarray(x)),
    )
    np.testing.assert_array_equal(
        model.predict_feature_contributions_arrays(x),
        reference.predict_feature_contributions_arrays(np.ascontiguousarray(x)),
    )


def test_native_geospatial_arrays_preserve_fortran_coordinates():
    from cartoboost import _native

    coords = np.array([[0.0, 1.0], [3.0, 5.0], [8.0, 9.0]])
    targets = np.array([[2.0, 4.0], [7.0, 8.0]])
    expected = _native.geostats_deterministic_neighbors_value(coords, targets, 2)
    actual = _native.geostats_deterministic_neighbors_value(
        np.asfortranarray(coords), np.asfortranarray(targets), 2
    )
    assert actual == expected


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.bool_])
def test_numeric_fit_adapter_matches_object_path(dtype):
    from cartoboost.regressor import _fit_transform_categorical_features

    x = np.asarray([[0, 1], [1, 0], [2, 3]], dtype=dtype)
    y = np.array([1.0, 2.0, 4.0])
    schema = {
        "dense": [{"name": "pickup", "kind": "numeric"}, {"name": "dropoff", "kind": "numeric"}]
    }
    actual, encoder, names = _fit_transform_categorical_features(x, y, schema)
    expected, old_encoder, old_names = _fit_transform_categorical_features(
        x.astype(object), y, schema
    )
    np.testing.assert_array_equal(actual, expected)
    assert encoder == old_encoder is None
    assert names == old_names == ["pickup", "dropoff"]
    if dtype == np.float64:
        assert np.shares_memory(actual, x)


def test_numeric_fit_adapter_keeps_explicit_categorical_schema():
    from cartoboost.regressor import _fit_transform_categorical_features

    x = np.array([[1.0], [2.0], [1.0]])
    y = np.array([1.0, 2.0, 1.0])
    schema = {"dense": [{"name": "zone", "kind": "categorical"}]}
    actual, encoder, _ = _fit_transform_categorical_features(x, y, schema)
    expected, reference, _ = _fit_transform_categorical_features(x.astype(object), y, schema)
    assert encoder == reference
    assert encoder is not None
    np.testing.assert_array_equal(actual, expected)


class _CountingDuckDBInput:
    def __init__(self, columns):
        self.columns = list(columns)
        self.dtypes = [column.dtype for column in columns.values()]
        self._columns = columns
        self.fetches = 0

    def fetchnumpy(self):
        self.fetches += 1
        if self.fetches != 1:
            raise AssertionError("a query result must be materialized only once per operation")
        return self._columns


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("family", ["regressor", "classifier", "ranker"])
def test_fit_materializes_each_query_input_once(categorical, family):
    from cartoboost import CartoBoostClassifier, CartoBoostRanker

    model_type = {
        "regressor": CartoBoostRegressor,
        "classifier": CartoBoostClassifier,
        "ranker": CartoBoostRanker,
    }[family]
    x = _CountingDuckDBInput(
        {"zone": np.array(["a", "b", "a", "b"]) if categorical else np.arange(4.0)}
    )
    y = _CountingDuckDBInput({"target": np.array([0.0, 1.0, 0.0, 1.0])})
    weights = _CountingDuckDBInput({"weight": np.array([1.0, 2.0, 1.0, 2.0])})
    kwargs = {"groups": [2, 2]} if family == "ranker" else {}
    model = model_type(n_estimators=1, min_samples_leaf=1).fit(
        x, y, sample_weight=weights, **kwargs
    )
    assert (x.fetches, y.fetches, weights.fetches) == (1, 1, 1)
    assert list(model.feature_names_in_) == ["zone"]
    assert bool(model.categorical_encoder_) == categorical


@pytest.mark.parametrize("table_format", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("family", ["regressor", "classifier", "ranker"])
def test_numeric_table_predictions_match_numpy_and_single_rows(table_format, family):
    from cartoboost import CartoBoostClassifier, CartoBoostRanker

    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    module = pytest.importorskip(table_format)

    def table(array):
        columns = {"distance": array[:, 0], "hour": array[:, 1]}
        if table_format == "duckdb":
            pd = pytest.importorskip("pandas")
            return module.from_df(pd.DataFrame(columns))
        return module.DataFrame(columns)

    model_type = {
        "regressor": CartoBoostRegressor,
        "classifier": CartoBoostClassifier,
        "ranker": CartoBoostRanker,
    }[family]
    kwargs = {"groups": [2, 2]} if family == "ranker" else {}
    config = dict(n_estimators=2, max_depth=2, min_samples_leaf=1, split_policy="axis_only")
    model = model_type(**config).fit(table(x), y, **kwargs)
    reference = model_type(**config).fit(x, y, **kwargs)
    assert model.categorical_encoder_ is None
    assert list(model.feature_names_in_) == ["distance", "hour"]
    methods = ["predict"]
    if family == "regressor":
        methods += ["predict_additive_values", "predict_feature_contributions"]
    elif family == "classifier":
        methods += ["predict_proba", "decision_function"]
    for method in methods:
        predict = getattr(model, method)
        batch = predict(table(x))
        np.testing.assert_array_equal(batch, getattr(reference, method)(x))
        rows = np.concatenate([predict(table(x[index : index + 1])) for index in range(len(x))])
        np.testing.assert_array_equal(batch, rows)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_nullable_numeric_frames_preserve_numeric_missing_value_semantics(dtype):
    pd = pytest.importorskip("pandas")
    from cartoboost.regressor import _as_2d_float_array, _fit_transform_categorical_features

    frame = pd.DataFrame({"value": pd.Series([0, 1, 0, 1], dtype=dtype)})
    y = np.array([0.0, 2.0, 0.0, 2.0])
    actual, encoder, names = _fit_transform_categorical_features(frame, y, None)
    assert encoder is None
    assert names == ["value"]
    np.testing.assert_array_equal(actual[:, 0], [0.0, 1.0, 0.0, 1.0])
    model = CartoBoostRegressor(n_estimators=2, min_samples_leaf=1).fit(frame, y)
    missing = frame.copy()
    missing.loc[0, "value"] = pd.NA
    expected = actual.copy()
    expected[0, 0] = np.nan
    np.testing.assert_array_equal(_as_2d_float_array(missing, check_finite=False), expected)
    for values in (missing, expected):
        with pytest.raises(ValueError, match="finite"):
            model.predict(values)
    with pytest.raises(ValueError, match="finite"):
        CartoBoostRegressor().fit(missing, y)


def test_typed_categorical_sparse_schema_preserves_numeric_tokens():
    from cartoboost import SparseSetSpec
    from cartoboost.regressor import _fit_transform_categorical_features

    schema = FeatureSchema(
        dense=[("zone", "categorical"), ("distance", "numeric")],
        sparse_sets=[SparseSetSpec("cells")],
    )
    x = [[1, 1.5], [2, 2.5], [1, 3.5]]
    y = np.array([0.0, 1.0, 0.0])
    actual, encoder, names = _fit_transform_categorical_features(x, y, schema)
    expected, reference, _ = _fit_transform_categorical_features(
        np.asarray(x, dtype=object), y, schema.to_dict()
    )
    assert names == ["zone", "distance"]
    assert encoder == reference
    assert encoder is not None
    np.testing.assert_array_equal(actual, expected)


def test_typed_categorical_sparse_schema_retains_spatial_pair_kinds():
    from cartoboost import CategoricalSpec, NumericSpec, SparseSetSpec, SpatialPairSpec
    from cartoboost.regressor import _encoded_feature_schema, _fit_transform_categorical_features

    schema = FeatureSchema(
        dense=[SpatialPairSpec("x", "y"), NumericSpec("y"), CategoricalSpec("zone")],
        sparse_sets=[SparseSetSpec("cells")],
    )
    encoded, encoder, _ = _fit_transform_categorical_features(
        [[0.0, 1.0, "a"], [1.0, 2.0, "b"], [2.0, 3.0, "a"]], [0.0, 1.0, 0.0], schema
    )
    payload = _encoded_feature_schema(schema, encoder, encoded.shape[1], ["cells"])
    assert payload["kinds"][:2] == ["Spatial", "Spatial"]
    assert payload["kinds"][-1] == "SparseSet"


@pytest.mark.parametrize("dtype_name", ["Categorical", "Enum", "String"])
def test_polars_declared_categories_are_not_reinterpreted_as_numeric(dtype_name):
    pl = pytest.importorskip("polars")
    from cartoboost.regressor import _fit_transform_categorical_features

    dtype = pl.Enum(["1", "2"]) if dtype_name == "Enum" else getattr(pl, dtype_name)
    frame = pl.DataFrame({"zone": pl.Series(["1", "2", "1", "2"], dtype=dtype)})
    _, encoder, names = _fit_transform_categorical_features(frame, [0.0, 1.0, 0.0, 1.0], None)
    assert names == ["zone"]
    assert encoder is not None
    assert encoder["columns"][0]["index"] == 0


def test_constant_prediction_reads_table_shape_without_materializing_values():
    class ShapeOnlyTable:
        shape = (100_000, 2)

        def to_numpy(self):
            raise AssertionError("constant prediction does not require feature values")

    model = CartoBoostRegressor(n_estimators=2, max_depth=0).fit([[0.0, 1.0]], [3.0])
    prediction = model.predict(ShapeOnlyTable())
    assert prediction.shape == (100_000,)
    np.testing.assert_array_equal(prediction, 3.0)


@pytest.mark.parametrize(
    "kind,threshold,cardinality",
    [("categorical", 16, 6), ("categorical", 16, 3), ("categorical", 2, 6), ("ordinal", 16, 6)],
)
def test_typed_categorical_columns_preserve_legacy_encodings_and_artifacts(
    kind, threshold, cardinality, monkeypatch
):
    from cartoboost import _native
    from cartoboost.regressor import (
        _fit_transform_categorical_features,
        _transform_categorical_features,
    )

    if not hasattr(_native, "categorical_fit_transform_arrays"):
        pytest.skip("typed categorical bindings require the rebuilt extension")
    raw = np.empty((48, 3), dtype=object)
    categories = [None, "red", "blue", 2, 2.5, "green"][:cardinality]
    raw[:, 0] = [categories[index % cardinality] for index in range(len(raw))]
    raw[:, 1] = np.arange(48) * 0.125
    raw[:, 2] = np.arange(48) % 3
    targets = np.sin(np.arange(48, dtype=float))
    weights = np.arange(48, dtype=float) % 4
    schema = FeatureSchema(dense=[("zone", kind), ("distance", "numeric"), ("hour", "numeric")])
    options = dict(sample_weight=weights, low_cardinality_threshold=threshold, smoothing=1.5)
    actual, encoder, names = _fit_transform_categorical_features(raw, targets, schema, **options)
    probes = raw[[0, 1, 2, 10]].copy()
    probes[1, 0] = "unseen-category"
    transformed = _transform_categorical_features(probes, encoder)
    with monkeypatch.context() as patch:
        patch.setattr(_native, "categorical_fit_transform_arrays", None)
        patch.setattr(_native, "categorical_transform_arrays", None)
        expected, legacy_encoder, legacy_names = _fit_transform_categorical_features(
            raw, targets, schema, **options
        )
        legacy_transformed = _transform_categorical_features(probes, legacy_encoder)
    assert names == legacy_names
    assert encoder == legacy_encoder
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(transformed, legacy_transformed)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        np.nan,
        np.inf,
        "not-a-number",
        complex(3, 4),
        np.complex64(3 + 4j),
        np.complex128(3 + 4j),
        np.datetime64("2026-01-01"),
        np.timedelta64(3, "D"),
        10**400,
    ],
)
def test_categorical_numeric_columns_preserve_invalid_input_rejection(invalid, monkeypatch):
    from cartoboost import _native
    from cartoboost.regressor import (
        _fit_transform_categorical_features,
        _transform_categorical_features,
    )

    if not hasattr(_native, "categorical_fit_transform_arrays"):
        pytest.skip("typed categorical bindings require the rebuilt extension")
    raw = np.array([["red", 1.0], ["blue", 2.0]], dtype=object)
    _, encoder, _ = _fit_transform_categorical_features(raw, [0.0, 1.0], None)
    raw[0, 1] = invalid
    with pytest.raises(ValueError):
        _transform_categorical_features(raw, encoder)
    with monkeypatch.context() as patch:
        patch.setattr(_native, "categorical_transform_arrays", None)
        with pytest.raises(ValueError):
            _transform_categorical_features(raw, encoder)


@pytest.mark.parametrize("family", ["regressor", "binary", "multiclass", "ranker"])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "reversed", "strided"])
def test_native_numpy_outputs_preserve_legacy_values_and_layout(family, layout):
    from cartoboost import _native

    model_type = {
        "regressor": _native.CartoBoostRegressor,
        "binary": _native.CartoBoostClassifier,
        "multiclass": _native.CartoBoostClassifier,
        "ranker": _native.CartoBoostRanker,
    }[family]
    x = np.arange(96, dtype=np.float64).reshape(24, 4) / 7.0
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "reversed":
        x = x[::-1, ::-1]
    elif layout == "strided":
        x = x[::2, ::2]
    y = np.arange(len(x), dtype=np.float64) % (3 if family == "multiclass" else 2)
    config = dict(n_estimators=3, max_depth=2, min_samples_leaf=1, splitters=["axis"])
    if family == "multiclass":
        config.update(objective="multiclass_logloss", class_count=3)
    model = model_type(**config)
    if family == "ranker":
        model.fit_arrays(x, y, [len(x)])
    else:
        model.fit_arrays(x, y)
    if family == "regressor":
        operations = ["predict_additive", "predict_feature_contributions"]
    elif family == "ranker":
        operations = ["predict"]
    else:
        operations = ["predict", "predict_proba", "decision_function"]
    for operation in operations:
        actual = getattr(model, f"{operation}_numpy")(x)
        expected = getattr(model, f"{operation}_arrays")(np.ascontiguousarray(x))
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float64
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(getattr(model, f"{operation}_numpy")(x[:1]), actual[:1])
    if family == "ranker":
        # The score path also accepts target views without a logical-order copy bug.
        target_storage = np.repeat(y, 2)
        assert model.metrics_arrays(x, target_storage[::2], [len(x)]) == model.metrics_arrays(
            np.ascontiguousarray(x), y, [len(x)]
        )


@pytest.mark.parametrize("family", ["binary", "multiclass", "ranker"])
def test_native_numpy_outputs_preserve_validation_errors(family):
    from cartoboost import _native

    model_type = _native.CartoBoostRanker if family == "ranker" else _native.CartoBoostClassifier
    config = dict(n_estimators=1, min_samples_leaf=1)
    if family == "multiclass":
        config.update(objective="multiclass_logloss", class_count=3)
    model = model_type(**config)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_numpy(np.ones((1, 2)))
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.arange(6, dtype=np.float64) % (3 if family == "multiclass" else 2)
    if family == "ranker":
        model.fit_arrays(x, y, [3, 3])
    else:
        model.fit_arrays(x, y)
    methods = [model.predict_numpy]
    if family != "ranker":
        methods += [model.predict_proba_numpy, model.decision_function_numpy]
    for method in methods:
        for invalid in [
            np.empty((0, 2)),
            np.empty((1, 0)),
            np.ones((1, 3)),
            np.array([[np.nan, 1.0]]),
        ]:
            with pytest.raises(ValueError):
                method(invalid)


@pytest.mark.parametrize("family", ["regressor", "classifier", "ranker"])
def test_native_fit_accepts_strided_targets_and_weights(family):
    from cartoboost import _native

    model_type = getattr(_native, f"CartoBoost{family.title()}")
    config = dict(n_estimators=3, max_depth=2, min_samples_leaf=1, splitters=["axis"])
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    targets = (np.arange(24, dtype=np.float64) % 3)[::2]
    if family == "classifier":
        targets = targets % 2
        storage = np.empty(24, dtype=np.float64)
        storage[::2] = targets
        targets = storage[::2]
    weights = np.linspace(1.0, 3.0, 24)[::-2]
    actual = model_type(**config)
    expected = model_type(**config)
    kwargs = {"groups": [6, 6]} if family == "ranker" else {}
    actual.fit_arrays(x, targets, sample_weight=weights, **kwargs)
    expected.fit_arrays(
        x, np.ascontiguousarray(targets), sample_weight=np.ascontiguousarray(weights), **kwargs
    )
    np.testing.assert_array_equal(actual.predict_arrays(x), expected.predict_arrays(x))


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize(
    "dtype,data",
    [
        ("float64", [1.25, 2.5]),
        ("float32", [1.25, 2.5]),
        ("int64", [1, 2]),
        ("uint64", [2**63, 2**63 + 1024]),
        ("bool", [True, False]),
        ("object", ["1.25", "2"]),
        ("object", [1, "2"]),
        ("Int64", [1, None]),
        ("UInt64", [2**63, None]),
        ("Float64", [1.25, None]),
        ("boolean", [True, None]),
        ("string", ["1.25", None]),
        ("category", ["1", "2"]),
        ("datetime64[ns]", ["2026-01-01", None]),
        ("timedelta64[ns]", [1, None]),
    ],
)
def test_known_numeric_pandas_conversion_preserves_discovery_values(dtype, data, as_frame):
    pd = pytest.importorskip("pandas")
    from cartoboost.regressor import _to_numeric_numpy, _to_numpy

    values = pd.Series(data, dtype=dtype)
    if as_frame:
        values = values.to_frame()
    expected = np.asarray(_to_numpy(values, numeric=True), dtype=np.float64)
    np.testing.assert_array_equal(_to_numeric_numpy(values), expected)


def test_known_numeric_pandas_conversion_preserves_float64_buffers():
    pd = pytest.importorskip("pandas")
    from cartoboost.regressor import _as_2d_float_array, _to_numeric_numpy

    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    frame = pd.DataFrame(values, copy=False)
    assert _to_numeric_numpy(values) is values
    assert np.shares_memory(_to_numeric_numpy(frame), values)
    assert np.shares_memory(_as_2d_float_array(frame), values)


@pytest.mark.parametrize(
    "adapter", ["_as_2d_float_array", "_as_1d_float_array", "_as_sample_weight_array"]
)
def test_known_numeric_pandas_conversion_rejects_nonnumeric_inputs(adapter):
    pd = pytest.importorskip("pandas")
    from cartoboost import regressor

    values = pd.DataFrame({"value": ["not-a-number"]})
    args = (values, 1) if adapter == "_as_sample_weight_array" else (values,)
    with pytest.raises(ValueError):
        getattr(regressor, adapter)(*args)


@pytest.mark.parametrize("n_threads", [1, 4, None])
def test_native_ranker_small_batch_parallel_boundary_keeps_exact_predictions(n_threads):
    from cartoboost import _native

    train = np.arange(48, dtype=np.float64).reshape(12, 4) / 9.0
    target = np.arange(12, dtype=np.float64) % 4
    model = _native.CartoBoostRanker(
        n_estimators=128, max_depth=2, min_samples_leaf=1, splitters=["axis"], n_threads=n_threads
    )
    model.fit_arrays(train, target, [4, 4, 4])
    x = np.arange(132, dtype=np.float64).reshape(33, 4) / 11.0
    singleton = np.array([model.predict_numpy(row[None, :])[0] for row in x])
    for rows in [1, 31, 32, 33]:
        values = x[:rows]
        np.testing.assert_array_equal(model.predict_numpy(values), singleton[:rows])
        np.testing.assert_array_equal(model.predict_arrays(values), singleton[:rows])
        np.testing.assert_array_equal(model.predict(values.tolist()), singleton[:rows])


def _large_category_table(table_format, columns):
    if table_format == "counting_query":
        return _CountingDuckDBInput(columns)
    if table_format == "polars":
        return pytest.importorskip("polars").DataFrame(columns)
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(columns)
    if table_format == "pandas_nullable":
        dtype = "UInt64" if frame["PULocationID"].dtype.kind == "u" else "Int64"
        frame["PULocationID"] = frame["PULocationID"].astype(dtype)
    if table_format.startswith("pandas"):
        return frame
    duckdb = pytest.importorskip("duckdb")
    if table_format == "duckdb_connection":
        connection = duckdb.connect()
        connection.register("pickup_rows", frame)
        return connection.execute("SELECT * FROM pickup_rows")
    return duckdb.from_df(frame)


@pytest.mark.parametrize("family", ["regressor", "classifier", "ranker"])
@pytest.mark.parametrize(
    "table_format",
    ["pandas", "pandas_nullable", "polars", "duckdb", "duckdb_connection", "counting_query"],
)
@pytest.mark.parametrize("dtype,base", [(np.int64, 2**53), (np.uint64, 2**63)])
def test_explicit_categorical_tables_preserve_adjacent_large_ids(family, table_format, dtype, base):
    from cartoboost import CartoBoostClassifier, CartoBoostRanker

    ids = np.array([base + 1, base + 2] * 3, dtype=dtype)
    columns = {"PULocationID": ids, "trip_distance": np.full(6, 1.25)}
    raw = np.empty((6, 2), dtype=object)
    raw[:, 0], raw[:, 1] = ids, columns["trip_distance"]
    targets = np.array([0.0, 1.0] * 3)
    schema = FeatureSchema(dense=[("PULocationID", "categorical"), ("trip_distance", "numeric")])
    cls = {
        "regressor": CartoBoostRegressor,
        "classifier": CartoBoostClassifier,
        "ranker": CartoBoostRanker,
    }[family]
    config = dict(n_estimators=3, max_depth=2, min_samples_leaf=1, split_policy="axis_only")
    kwargs = {"groups": [2, 2, 2]} if family == "ranker" else {}
    reference = cls(**config).fit(raw, targets, feature_schema=schema, **kwargs)
    training = _large_category_table(table_format, columns)
    prediction = _large_category_table(table_format, columns)
    try:
        candidate = cls(**config).fit(training, targets, feature_schema=schema, **kwargs)
        assert candidate.categorical_encoder_ == reference.categorical_encoder_
        categories = candidate.categorical_encoder_["columns"][0]["categories"]
        assert categories == [f"int:{base + 1}", f"int:{base + 2}"]
        actual = candidate.predict(prediction)
        np.testing.assert_array_equal(actual, reference.predict(raw))
        assert actual[0] != actual[1]
        if table_format == "counting_query":
            assert training.fetches == prediction.fetches == 1
    finally:
        if table_format == "duckdb_connection":
            training.close()
            prediction.close()


@pytest.mark.parametrize("table_format", ["pandas", "polars", "duckdb", "counting_query"])
def test_ranker_group_extraction_preserves_large_categorical_ids(table_format):
    from cartoboost import CartoBoostRanker

    ids = np.array([2**63 + 1, 2**63 + 2] * 3, dtype=np.uint64)
    columns = {
        "query_id": np.repeat(["morning", "midday", "evening"], 2),
        "PULocationID": ids,
        "trip_distance": np.full(6, 1.25),
    }
    schema = FeatureSchema(
        dense=[
            ("query_id", "categorical"),
            ("PULocationID", "categorical"),
            ("trip_distance", "numeric"),
        ]
    )
    raw = np.empty((6, 3), dtype=object)
    for index, column in enumerate(columns.values()):
        raw[:, index] = column
    targets = np.array([0.0, 1.0] * 3)
    config = dict(n_estimators=3, max_depth=2, min_samples_leaf=1, split_policy="axis_only")
    reference = CartoBoostRanker(group_col=0, **config).fit(raw, targets, feature_schema=schema)
    training = _large_category_table(table_format, columns)
    prediction = _large_category_table(table_format, columns)
    candidate = CartoBoostRanker(group_col="query_id", **config).fit(
        training, targets, feature_schema=schema
    )
    assert candidate.categorical_encoder_ == reference.categorical_encoder_
    np.testing.assert_array_equal(candidate.predict(prediction), reference.predict(raw))
    if table_format == "counting_query":
        assert training.fetches == prediction.fetches == 1


def test_duckdb_nullable_categorical_ids_preserve_integer_values():
    pd = pytest.importorskip("pandas")
    duckdb = pytest.importorskip("duckdb")
    from cartoboost.regressor import (
        _fit_transform_categorical_features,
        _transform_categorical_features,
    )

    frame = pd.DataFrame(
        {
            "PULocationID": pd.array([2**63 + 1, None, 2**63 + 2], dtype="UInt64"),
            "trip_distance": [1.25] * 3,
        }
    )
    schema = FeatureSchema(dense=[("PULocationID", "categorical"), ("trip_distance", "numeric")])
    expected, encoder, _ = _fit_transform_categorical_features(frame, [0.0, 1.0, 2.0], schema)
    actual, candidate, _ = _fit_transform_categorical_features(
        duckdb.from_df(frame), [0.0, 1.0, 2.0], schema
    )
    assert candidate == encoder
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        _transform_categorical_features(duckdb.from_df(frame), encoder),
        _transform_categorical_features(frame, encoder),
    )


@pytest.mark.parametrize("table_format", ["pandas", "polars", "duckdb", "counting_query"])
def test_small_mixed_categorical_table_tokens_remain_compatible(table_format):
    from cartoboost.regressor import _fit_transform_categorical_features

    columns = {"PULocationID": np.array([1, 2, 1]), "trip_distance": np.array([1.25, 2.5, 1.25])}
    schema = FeatureSchema(dense=[("PULocationID", "categorical"), ("trip_distance", "numeric")])
    raw = np.column_stack(list(columns.values()))
    expected, encoder, _ = _fit_transform_categorical_features(raw, [0.0, 1.0, 0.0], schema)
    actual, candidate, _ = _fit_transform_categorical_features(
        _large_category_table(table_format, columns), [0.0, 1.0, 0.0], schema
    )
    assert candidate == encoder
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("table_format", ["numpy", "pandas"])
@pytest.mark.parametrize(
    "adapter", ["_as_2d_float_array", "_as_1d_float_array", "_as_sample_weight_array"]
)
def test_known_numeric_adapters_reject_complex_values(table_format, adapter):
    from cartoboost import regressor

    values = np.array([[1.0 + 2.0j], [3.0 + 4.0j]])
    if table_format == "pandas":
        values = pytest.importorskip("pandas").DataFrame(values)
    args = (values, 2) if adapter == "_as_sample_weight_array" else (values,)
    with pytest.raises(ValueError):
        getattr(regressor, adapter)(*args)


@pytest.mark.parametrize(
    "table_format", ["pandas", "polars", "duckdb", "counting_query", "ranker_columns"]
)
def test_large_numeric_columns_retain_numeric_conversion_path(table_format):
    from cartoboost.ranker import _ColumnArray
    from cartoboost.regressor import _fit_transform_categorical_features, _to_numpy

    columns = {
        "PULocationID": np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64),
        "trip_distance": np.array([1.25, 2.5]),
    }

    def table():
        return (
            _ColumnArray(columns)
            if table_format == "ranker_columns"
            else _large_category_table(table_format, columns)
        )

    numeric = _to_numpy(table(), numeric=True)
    assert numeric.dtype == np.float64
    np.testing.assert_array_equal(numeric, np.column_stack(list(columns.values())))
    transformed, encoder, _ = _fit_transform_categorical_features(table(), [0.0, 1.0], None)
    assert encoder is None
    np.testing.assert_array_equal(transformed, numeric)
