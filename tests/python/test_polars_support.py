import numpy as np
import pytest
from cartoboost import CartoBoostRegressor

pl = pytest.importorskip("polars")


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def test_polars_dataframe_fit_predict_preserves_feature_names():
    x = pl.DataFrame(
        {
            "distance_m": [0.0, 1.0, 2.0, 3.0],
            "hour": [0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pl.Series("target", [0.0, 1.5, 2.0, 3.5])
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
    )

    model.fit(x, y)
    predictions = model.predict(x.select(["distance_m", "hour"]))

    assert list(model.feature_names_in_) == ["distance_m", "hour"]
    assert predictions.shape == (4,)
    assert predictions[0] < predictions[-1]


def test_polars_series_sample_weight_matches_numpy_path():
    x = pl.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pl.Series("target", [0.0, 0.0, 10.0, 10.0])
    sample_weight = pl.Series("weight", [1.0, 3.0, 1.0, 3.0])
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    )
    polars_model = CartoBoostRegressor(**config)
    numpy_model = CartoBoostRegressor(**config)

    _fit_or_skip(polars_model, x, y, sample_weight=sample_weight)
    _fit_or_skip(
        numpy_model,
        x.to_numpy(),
        y.to_numpy(),
        sample_weight=sample_weight.to_numpy(),
    )

    assert polars_model.predict(x) == pytest.approx(numpy_model.predict(x.to_numpy()))


def test_polars_sparse_set_dataframe_train_predict_matches_dict_path():
    x = pl.DataFrame({"bias": [0.0, 0.0, 0.0, 0.0]})
    y = pl.Series("target", [7.0, 7.0, -2.0, -2.0])
    sparse_sets = pl.DataFrame({"route_cells": [[10, 20], [20, 30], [40], []]})
    dict_sparse_sets = {"route_cells": sparse_sets["route_cells"].to_list()}
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )
    polars_model = CartoBoostRegressor(**config)
    dict_model = CartoBoostRegressor(**config)

    _fit_or_skip(
        polars_model,
        x,
        y,
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )
    _fit_or_skip(
        dict_model,
        x.to_numpy(),
        y.to_numpy(),
        sparse_sets=dict_sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )

    assert polars_model.sparse_set_names_ == ["route_cells"]
    assert polars_model.predict(x, sparse_sets=sparse_sets) == pytest.approx(
        dict_model.predict(x.to_numpy(), sparse_sets=dict_sparse_sets)
    )


def test_polars_sparse_shap_adapter_accepts_dataframes():
    shap = pytest.importorskip("shap")
    x = pl.DataFrame({"bias": [0.0, 0.0, 0.0, 0.0]})
    y = np.asarray([10.0, 10.0, 0.0, 0.0], dtype=float)
    sparse_sets = pl.DataFrame({"route_cells": [[7], [7, 11], [3], []]})
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
        x.head(2),
        background=x,
        sparse_sets=sparse_sets.head(2),
        background_sparse_sets=sparse_sets,
        algorithm="exact",
    )

    reconstructed = np.asarray(explanation.base_values) + explanation.values.sum(axis=1)
    assert isinstance(explanation, shap.Explanation)
    assert list(explanation.feature_names) == [
        "bias",
        "route_cells=3",
        "route_cells=7",
        "route_cells=11",
    ]
    assert reconstructed == pytest.approx(model.predict(x.head(2), sparse_sets=sparse_sets.head(2)))
    np.testing.assert_array_equal(explanation.data, [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])


@pytest.mark.parametrize("use_background", [False, True])
def test_polars_shap_keeps_dense_feature_names_and_values_aligned(use_background):
    pytest.importorskip("shap")
    x = pl.DataFrame({"trip_distance": [1.0, 2.0, 4.0, 8.0], "pickup_hour": [7, 8, 19, 20]})
    model = CartoBoostRegressor(
        n_estimators=3,
        max_depth=2,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit(x, [4.0, 7.0, 12.0, 22.0])

    explanation = model.explain_shap(x.head(2), background=x if use_background else None)

    assert list(explanation.feature_names) == ["trip_distance", "pickup_hour"]
    np.testing.assert_array_equal(explanation.data, x.head(2).to_numpy())
    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        model.predict(x.head(2)),
        rtol=1e-6,
    )
