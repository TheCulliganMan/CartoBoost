import pickle
from pathlib import Path

import numpy as np
import pytest
from cartoboost import CartoBoostRegressor
from cartoboost.explain import _tree_shap_ensemble, make_shap_explainer

shap = pytest.importorskip("shap")


def _assert_additive(explanation, prediction):
    reconstructed = np.asarray(explanation.base_values) + explanation.values.sum(axis=1)
    assert reconstructed == pytest.approx(prediction)


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))


def test_shap_explainer_accepts_cartoboost_estimator_directly():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
    ).fit(X, y)

    explainer = shap.Explainer(model, X, algorithm="exact")
    explanation = explainer(X[:2])

    assert isinstance(explanation, shap.Explanation)
    assert explanation.values.shape == (2, 1)
    _assert_additive(explanation, model.predict(X[:2]))


def test_explain_shap_returns_additive_explanation_for_native_backend():
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([0.0, 0.5, 2.0, 2.5, 4.0, 4.5], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
    ).fit(X, y)

    explanation = model.explain_shap(X[:3], background=X, algorithm="exact")

    assert explanation.values.shape == (3, 2)
    assert explanation.data.shape == (3, 2)
    _assert_additive(explanation, model.predict(X[:3]))


def test_dense_axis_models_use_shap_tree_explainer():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=8,
        learning_rate=0.3,
        max_depth=2,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit(X, y)

    explainer = model.make_shap_explainer(X)
    explanation = explainer(X[:2])

    assert type(explainer.explainer).__name__ == "TreeExplainer"
    assert explanation.values.shape == (2, 1)
    _assert_additive(explanation, model.predict(X[:2]))


def test_native_path_dependent_contributions_match_lightgbm_layout():
    X = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=float,
    )
    y = np.asarray([0.0, 2.0, 8.0, 10.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=2,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit(X, y)

    direct = model.predict_feature_contributions(X)
    compatible = model.predict(X, pred_contrib=True)

    assert direct.shape == (4, 3)
    assert compatible == pytest.approx(direct)
    assert direct.sum(axis=1) == pytest.approx(model.predict(X))
    assert direct[:, -1] == pytest.approx(direct[0, -1])
    assert model.feature_name_ == ["feature_0", "feature_1"]


def test_native_path_dependent_contributions_match_shap_tree_explainer():
    X = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=float,
    )
    y = np.asarray([0.0, 1.0, 5.0, 9.0, 12.0], dtype=float)
    sample_weight = np.asarray([1.0, 2.0, 1.0, 3.0, 2.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.4,
        max_depth=2,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit(X, y, sample_weight=sample_weight)
    ensemble = _tree_shap_ensemble(model)
    assert ensemble is not None
    reference = shap.TreeExplainer(
        ensemble,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    native = model.predict_feature_contributions(X)
    reference_values = np.asarray(reference.shap_values(X), dtype=float)

    assert native[:, :-1] == pytest.approx(reference_values, abs=1.0e-10)
    reference_base = float(np.asarray(reference.expected_value).reshape(-1)[0])
    assert native[:, -1] == pytest.approx(reference_base, abs=1.0e-10)


def test_background_free_explain_shap_returns_standard_explanation():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=2,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit(X, y)

    explainer = model.make_shap_explainer()
    explanation = model.explain_shap(X[:2])

    assert isinstance(explanation, shap.Explanation)
    assert explanation.values == pytest.approx(explainer.shap_values(X[:2]))
    assert explanation.base_values == pytest.approx(explainer.expected_value)
    assert explanation.feature_names == ["feature_0"]
    assert explainer.expected_value == pytest.approx(
        model.predict_feature_contributions(X[:1])[0, -1]
    )
    _assert_additive(explanation, model.predict(X[:2]))


def test_categorical_contributions_aggregate_to_original_dataframe_columns(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "LEADTIME_CATEGORICAL": pd.Series(["0", "0.5", "1", "2"] * 2, dtype="category"),
            "DAYS_ELAPSED": np.arange(8, dtype=float),
        }
    )
    y = np.asarray([8.0, 7.0, 5.0, 1.0, 9.0, 8.0, 6.0, 2.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=2,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit(X, y)

    contributions = model.predict(X.iloc[:2], pred_contrib=True)
    columns = [*model.feature_name_, "base_value"]
    frame = pd.DataFrame(contributions, columns=columns)
    path = tmp_path / "categorical.cartoboost.json"
    model.save(path)
    loaded = CartoBoostRegressor.load(path)
    unpickled = pickle.loads(pickle.dumps(model))

    assert contributions.shape == (2, 3)
    assert list(frame.columns) == [
        "LEADTIME_CATEGORICAL",
        "DAYS_ELAPSED",
        "base_value",
    ]
    assert contributions.sum(axis=1) == pytest.approx(model.predict(X.iloc[:2]))
    assert loaded.feature_name_ == model.feature_name_
    assert loaded.predict(X.iloc[:2], pred_contrib=True) == pytest.approx(contributions)
    assert unpickled.feature_name_ == model.feature_name_
    assert unpickled.predict(X.iloc[:2], pred_contrib=True) == pytest.approx(contributions)


@pytest.mark.parametrize(
    "split_policy, monotonic_constraints",
    [("auto", None), ("structured", None), ("axis_only", [1, 0])],
)
def test_feature_contributions_support_default_histogram_and_monotonic_axis_fits(
    split_policy, monotonic_constraints
):
    X = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype=float,
    )
    y = np.asarray([0.0, 1.0, 3.0, 4.0, 8.0, 9.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        max_depth=2,
        min_samples_leaf=1,
        split_policy=split_policy,
        monotonic_constraints=monotonic_constraints,
    ).fit(X, y)

    contributions = model.predict_feature_contributions(X)

    assert contributions.sum(axis=1) == pytest.approx(model.predict(X))


@pytest.mark.parametrize(
    "params, message",
    [
        ({"fuzzy": True, "fuzzy_bandwidth": 1.0}, "axis-aligned"),
        (
            {"leaf_predictor": "linear", "linear_leaf_features": ["0"]},
            "constant leaves",
        ),
        ({"loss": "log_l2"}, "identity"),
    ],
)
def test_background_free_feature_contributions_reject_unsupported_models(params, message):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        **params,
    )
    _fit_or_skip(model, X, y)

    with pytest.raises(ValueError, match=message):
        model.predict(X[:1], pred_contrib=True)


def test_make_shap_explainer_public_helper_matches_method():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
    ).fit(X, y)

    helper_explainer = make_shap_explainer(model, X, algorithm="exact")
    method_explainer = model.make_shap_explainer(X, algorithm="exact")

    helper_explanation = helper_explainer(X[:2])
    method_explanation = method_explainer(X[:2])

    assert helper_explanation.values == pytest.approx(method_explanation.values)
    assert helper_explanation.base_values == pytest.approx(method_explanation.base_values)


def test_shap_decomposes_additive_weights_for_weighted_native_backend():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=float)
    sample_weight = np.asarray([4.0, 1.0, 1.0, 2.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
    ).fit(X, y, sample_weight=sample_weight)

    additive = model.predict_additive_values(X[:2])
    assert additive.shape[0] == 2
    assert additive.sum(axis=1) == pytest.approx(model.predict(X[:2]))

    explanation = model.explain_shap(
        X[:2],
        background=X,
        algorithm="exact",
        decomposition="weights",
    )

    assert list(explanation.feature_names) == [
        "init_prediction",
        *[f"tree_{idx}" for idx in range(additive.shape[1] - 1)],
    ]
    assert explanation.values.shape == additive.shape
    _assert_additive(explanation, model.predict(X[:2]))


def test_weight_decomposition_is_direct_exact_shap_without_generic_explainer(monkeypatch):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
    ).fit(X, y)

    def fail_if_generic_explainer_is_used(*args, **kwargs):
        raise AssertionError("weight decomposition must not construct shap.Explainer")

    monkeypatch.setattr(shap, "Explainer", fail_if_generic_explainer_is_used)
    explainer = model.make_shap_explainer(X, decomposition="weights")
    explanation = explainer(X[:2])
    additive = model.predict_additive_values(X[:2])
    background_mean = model.predict_additive_values(X).mean(axis=0)

    assert explanation.values == pytest.approx(additive - background_mean)
    assert explanation.base_values == pytest.approx(background_mean.sum())
    assert explainer.shap_values(X[:2]) == pytest.approx(explanation.values)
    _assert_additive(explanation, model.predict(X[:2]))


def test_shap_preserves_pandas_feature_names():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"distance_m": [0.0, 1.0, 2.0, 3.0], "hour": [0.0, 1.0, 0.0, 1.0]})
    y = np.asarray([0.0, 1.5, 2.0, 3.5], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
    ).fit(X, y)

    explanation = model.explain_shap(X.iloc[:2], background=X, algorithm="exact")

    assert list(explanation.feature_names) == ["distance_m", "hour"]
    _assert_additive(explanation, model.predict(X.iloc[:2]))


@pytest.mark.parametrize(
    ("split_policy", "X", "y", "extra"),
    [
        ("axis_only", [[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 10.0, 10.0], {}),
        (
            "auto",
            [[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]],
            [-10.0, -10.0, 10.0, 10.0],
            {},
        ),
        (
            "structured",
            [[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 3.1]],
            [5.0, 5.0, -5.0, -5.0],
            {},
        ),
        (
            "auto",
            [[0.0], [1.0], [12.0], [13.0]],
            [10.0, 10.0, 0.0, 0.0],
            {"feature_schema": {"dense": [{"name": "hour", "kind": "periodic", "period": 24}]}},
        ),
    ],
)
def test_shap_additivity_for_native_split_policies(split_policy, X, y, extra):
    rows = np.asarray(X, dtype=float)
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        split_policy=split_policy,
    )
    _fit_or_skip(model, rows, y, **extra)

    explanation = model.explain_shap(rows[:2], background=rows, algorithm="exact")

    assert isinstance(explanation, shap.Explanation)
    _assert_additive(explanation, model.predict(rows[:2]))


@pytest.mark.parametrize(
    "params",
    [
        {"fuzzy": True, "fuzzy_bandwidth": 1.0},
        {"leaf_predictor": "linear", "linear_leaf_features": ["0"], "l2_regularization": 0.0},
    ],
)
def test_shap_additivity_for_rust_fuzzy_and_linear_leaf(params):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        **params,
    )
    _fit_or_skip(model, X, y)

    explanation = model.explain_shap(X[:2], background=X, algorithm="exact")

    _assert_additive(explanation, model.predict(X[:2]))


def test_shap_additivity_after_save_load(tmp_path: Path):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        min_samples_leaf=1,
    )
    _fit_or_skip(model, X, y)
    path = tmp_path / "model.cartoboost.json"
    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    explanation = loaded.explain_shap(X[:2], background=X, algorithm="exact")

    _assert_additive(explanation, loaded.predict(X[:2]))


def test_shap_supports_sparse_set_models_with_augmented_features():
    X = np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=float)
    y = np.asarray([10.0, 10.0, 0.0, 0.0], dtype=float)
    sparse_sets = {"route_cells": [[7], [7, 11], [3], []]}
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
    )
    _fit_or_skip(model, X, y, sparse_sets=sparse_sets)

    explanation = model.explain_shap(
        X[:2],
        background=X,
        sparse_sets={"route_cells": sparse_sets["route_cells"][:2]},
        background_sparse_sets=sparse_sets,
        algorithm="exact",
    )

    assert list(explanation.feature_names) == [
        "feature_0",
        "route_cells=3",
        "route_cells=7",
        "route_cells=11",
    ]
    assert explanation.values.shape == (2, 4)
    _assert_additive(
        explanation,
        model.predict(X[:2], sparse_sets={"route_cells": sparse_sets["route_cells"][:2]}),
    )


def test_shap_decomposes_additive_weights_for_sparse_set_models():
    X = np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=float)
    y = np.asarray([10.0, 10.0, 0.0, 0.0], dtype=float)
    sparse_sets = {"route_cells": [[7], [7, 11], [3], []]}
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
    )
    _fit_or_skip(model, X, y, sparse_sets=sparse_sets)

    additive = model.predict_additive_values(
        X[:2],
        sparse_sets={"route_cells": sparse_sets["route_cells"][:2]},
    )
    assert additive.shape == (2, 3)
    assert additive.sum(axis=1) == pytest.approx(
        model.predict(X[:2], sparse_sets={"route_cells": sparse_sets["route_cells"][:2]})
    )

    explanation = model.explain_shap(
        X[:2],
        background=X,
        sparse_sets={"route_cells": sparse_sets["route_cells"][:2]},
        background_sparse_sets=sparse_sets,
        algorithm="exact",
        decomposition="weights",
    )

    assert list(explanation.feature_names) == ["init_prediction", "tree_0", "tree_1"]
    assert explanation.values.shape == additive.shape
    _assert_additive(
        explanation,
        model.predict(X[:2], sparse_sets={"route_cells": sparse_sets["route_cells"][:2]}),
    )
