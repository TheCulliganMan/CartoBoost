import numpy as np
import pytest
from cartoboost import CartoBoostRegressor

duckdb = pytest.importorskip("duckdb")


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def test_duckdb_relation_fit_predict_preserves_feature_names():
    rel = duckdb.sql(
        """
        select *
        from (
            values
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.5),
                (2.0, 0.0, 2.0),
                (3.0, 1.0, 3.5)
        ) as rows(distance_m, hour, target)
        """
    )
    x = rel.select("distance_m, hour")
    y = rel.select("target")
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
    )

    model.fit(x, y)
    predictions = model.predict(x)

    assert list(model.feature_names_in_) == ["distance_m", "hour"]
    assert predictions.shape == (4,)
    assert predictions[0] < predictions[-1]


def test_duckdb_relation_sample_weight_matches_numpy_path():
    rel = duckdb.sql(
        """
        select *
        from (
            values
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 3.0),
                (2.0, 10.0, 1.0),
                (3.0, 10.0, 3.0)
        ) as rows(feature, target, weight)
        """
    )
    x = rel.select("feature")
    y = rel.select("target")
    sample_weight = rel.select("weight")
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    )
    duckdb_model = CartoBoostRegressor(**config)
    numpy_model = CartoBoostRegressor(**config)

    _fit_or_skip(duckdb_model, x, y, sample_weight=sample_weight)
    _fit_or_skip(
        numpy_model,
        x.fetchnumpy()["feature"].reshape(-1, 1),
        y.fetchnumpy()["target"],
        sample_weight=sample_weight.fetchnumpy()["weight"],
    )

    assert duckdb_model.predict(x) == pytest.approx(
        numpy_model.predict(x.fetchnumpy()["feature"].reshape(-1, 1))
    )


def test_duckdb_sparse_set_relation_train_predict_matches_dict_path():
    x = duckdb.sql("select * from (values (0.0), (0.0), (0.0), (0.0)) as rows(bias)")
    y = duckdb.sql("select * from (values (7.0), (7.0), (-2.0), (-2.0)) as rows(target)")
    sparse_sets = duckdb.sql(
        """
        select *
        from (
            values
                ([10, 20]::integer[]),
                ([20, 30]::integer[]),
                ([40]::integer[]),
                ([]::integer[])
        ) as rows(taxi_zones)
        """
    )
    dict_sparse_sets = {"taxi_zones": sparse_sets.fetchnumpy()["taxi_zones"].tolist()}
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )
    duckdb_model = CartoBoostRegressor(**config)
    dict_model = CartoBoostRegressor(**config)

    _fit_or_skip(
        duckdb_model,
        x,
        y,
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("taxi_zones", "sparse_set")],
        },
    )
    _fit_or_skip(
        dict_model,
        x.fetchnumpy()["bias"].reshape(-1, 1),
        y.fetchnumpy()["target"],
        sparse_sets=dict_sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("taxi_zones", "sparse_set")],
        },
    )

    assert duckdb_model.sparse_set_names_ == ["taxi_zones"]
    assert duckdb_model.predict(x, sparse_sets=sparse_sets) == pytest.approx(
        dict_model.predict(x.fetchnumpy()["bias"].reshape(-1, 1), sparse_sets=dict_sparse_sets)
    )


def test_duckdb_sparse_shap_adapter_accepts_relations():
    shap = pytest.importorskip("shap")
    x = duckdb.sql("select * from (values (0.0), (0.0), (0.0), (0.0)) as rows(bias)")
    y = np.asarray([10.0, 10.0, 0.0, 0.0], dtype=float)
    sparse_sets = duckdb.sql(
        """
        select *
        from (
            values
                ([7]::integer[]),
                ([7, 11]::integer[]),
                ([3]::integer[]),
                ([]::integer[])
        ) as rows(taxi_zones)
        """
    )
    sparse_head = duckdb.sql(
        """
        select *
        from (
            values
                ([7]::integer[]),
                ([7, 11]::integer[])
        ) as rows(taxi_zones)
        """
    )
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        split_policy="structured",
    )
    _fit_or_skip(
        model,
        x,
        y,
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )

    explanation = model.explain_shap(
        x.limit(2),
        background=x,
        sparse_sets=sparse_head,
        background_sparse_sets=sparse_sets,
        algorithm="exact",
    )

    reconstructed = np.asarray(explanation.base_values) + explanation.values.sum(axis=1)
    assert isinstance(explanation, shap.Explanation)
    assert list(explanation.feature_names) == [
        "bias",
        "taxi_zones=3",
        "taxi_zones=7",
        "taxi_zones=11",
    ]
    assert reconstructed == pytest.approx(model.predict(x.limit(2), sparse_sets=sparse_head))


@pytest.mark.parametrize("input_name", ["X", "y", "sample_weight"])
def test_duckdb_numeric_nulls_are_not_silently_replaced_by_masked_backing_data(input_name):
    inputs = {"X": [[0.0], [1.0]], "y": [0.0, 1.0], "sample_weight": [1.0, 1.0]}
    inputs[input_name] = duckdb.sql("select * from (values (1.0), (NULL)) as rows(value)")
    with pytest.raises(ValueError, match="finite"):
        CartoBoostRegressor(n_estimators=1).fit(**inputs)


def test_duckdb_prediction_null_matches_numpy_nan():
    model = CartoBoostRegressor(n_estimators=2, min_samples_leaf=1).fit(
        [[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0]
    )
    relation = duckdb.sql("select * from (values (1.0), (NULL)) as rows(value)")
    for values in (relation, [[1.0], [np.nan]]):
        with pytest.raises(ValueError, match="finite"):
            model.predict(values)


def test_duckdb_connection_result_can_be_consumed_once_for_fit():
    connection = duckdb.connect()
    try:
        query = connection.execute("select * from (values (0.0), (1.0)) as rows(value)")
        model = CartoBoostRegressor(n_estimators=1, min_samples_leaf=1).fit(query, [0.0, 1.0])
        assert model.predict([[0.0], [1.0]]).shape == (2,)
    finally:
        connection.close()
