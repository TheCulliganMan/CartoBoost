import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from cartoboost import CartoBoostRanker
from cartoboost.ranker import _normalize_groups
from cartoboost.schema import FeatureKind


def test_ranker_fit_predict_metrics_and_roundtrip(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]]
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    groups = [3, 3]
    ranker = CartoBoostRanker(
        n_estimators=8,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y, groups=groups)

    scores = ranker.predict(X)
    metrics = ranker.score_groups(X, y, groups=groups)

    assert scores.shape == (6,)
    assert scores[2] > scores[0]
    assert scores[5] > scores[3]
    assert metrics["ndcg"] > 0.9
    assert metrics["map"] > 0.9
    assert metrics["mrr"] > 0.9
    assert {row["name"] for row in ranker.training_history_} >= {
        "train/ndcg",
        "train/map",
        "train/mrr",
    }

    model_path = tmp_path / "ranker.json"
    ranker.save(model_path)
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    assert artifact["format"] == "cartoboost.model"
    assert artifact["artifact_version"] == 2
    assert artifact["model_type"] == "ranker"
    loaded = CartoBoostRanker.load(model_path)

    assert loaded.predict(X) == pytest.approx(scores)


def test_ranker_accepts_contiguous_query_ids_as_groups():
    X = np.asarray([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]], dtype=float)
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    query_ids = ["a", "a", "a", "b", "b", "b"]
    ranker = CartoBoostRanker(
        n_estimators=4,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
        objective="pairwise_logit",
    ).fit(X, y, groups=query_ids)

    assert ranker.score_groups(X, y, groups=query_ids)["ndcg"] > 0.8


def test_ranker_treats_same_length_numeric_groups_as_query_ids():
    X = np.asarray([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]], dtype=float)
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    query_ids = [1, 1, 1, 2, 2, 2]
    ranker = CartoBoostRanker(
        n_estimators=4,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y, groups=query_ids)

    assert ranker.groups_ == [3, 3]
    assert ranker.score_groups(X, y, groups=query_ids)["ndcg"] > 0.8


def test_ranker_explicit_groups_prefer_group_sizes_when_ambiguous():
    assert _normalize_groups([1, 1], 2) == [1, 1]
    assert _normalize_groups([2, 1], 3) == [2, 1]


def test_ranker_group_col_values_prefer_query_ids_when_ambiguous():
    assert _normalize_groups([1, 1], 2, prefer_query_ids=True) == [2]
    assert _normalize_groups([1, 1, 2], 3, prefer_query_ids=True) == [2, 1]


def test_ranker_accepts_tuple_query_ids_as_groups():
    X = np.asarray([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]], dtype=float)
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    query_ids = [
        ("pickup", "a"),
        ("pickup", "a"),
        ("pickup", "a"),
        ("pickup", "b"),
        ("pickup", "b"),
        ("pickup", "b"),
    ]
    ranker = CartoBoostRanker(
        n_estimators=4,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y, groups=query_ids)

    assert ranker.groups_ == [3, 3]
    assert ranker.score_groups(X, y, groups=query_ids)["ndcg"] > 0.8


def test_ranker_accepts_native_categorical_features(tmp_path: Path):
    X = [["bad"], ["ok"], ["great"], ["bad"], ["ok"], ["great"]]
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    groups = [3, 3]
    schema = {"dense": [{"name": "route_quality", "kind": FeatureKind.CATEGORICAL}]}
    ranker = CartoBoostRanker(
        n_estimators=6,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y, groups=groups, feature_schema=schema)
    scores = ranker.predict(X)
    model_path = tmp_path / "categorical-ranker.json"

    ranker.save(model_path)
    loaded = CartoBoostRanker.load(model_path)

    assert ranker.n_features_in_ == 1
    assert loaded.categorical_encoder_["columns"][0]["strategy"] == "Partition"
    assert loaded.predict(X) == pytest.approx(scores)
    assert loaded.score_groups(X, y, groups=groups)["ndcg"] > 0.8


def test_ranker_group_col_is_removed_from_feature_matrix(tmp_path: Path):
    X = np.asarray(
        [
            [10.0, 0.0],
            [10.0, 1.0],
            [10.0, 2.0],
            [20.0, 0.0],
            [20.0, 1.0],
            [20.0, 2.0],
        ],
        dtype=float,
    )
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    ranker = CartoBoostRanker(
        n_estimators=4,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        group_col=0,
        split_policy="axis_only",
    ).fit(X, y)

    assert ranker.n_features_in_ == 1
    assert ranker.predict([[0.0], [1.0], [2.0]]).shape == (3,)

    model_path = tmp_path / "group-col-ranker.json"
    ranker.save(model_path)
    loaded = CartoBoostRanker.load(model_path)

    assert loaded.group_col == 0
    assert loaded.predict(X) == pytest.approx(ranker.predict(X))


def test_ranker_group_col_removes_matching_feature_schema_entry():
    X = np.asarray(
        [
            [10.0, 0.0],
            [10.0, 1.0],
            [10.0, 2.0],
            [20.0, 0.0],
            [20.0, 1.0],
            [20.0, 2.0],
        ],
        dtype=float,
    )
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    schema = {
        "dense": [
            {"name": "query_id", "kind": FeatureKind.NUMERIC},
            {"name": "route_score", "kind": FeatureKind.NUMERIC},
        ]
    }
    ranker = CartoBoostRanker(
        n_estimators=4,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        group_col=0,
        split_policy="axis_only",
    ).fit(X, y, feature_schema=schema)

    assert ranker.feature_schema_["names"] == ["route_score"]
    assert ranker.n_features_in_ == 1


def test_ranker_group_col_with_pandas_categorical_schema():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "route_quality": pd.Series(
                ["bad", "ok", "great", "bad", "ok", "great"],
                dtype="category",
            ),
            "trip_distance": [8.0, 5.0, 3.0, 8.5, 4.0, 2.5],
        }
    )
    y = [0.0, 1.0, 3.0, 0.0, 2.0, 4.0]
    schema = {
        "dense": [
            {"name": "query_id", "kind": FeatureKind.CATEGORICAL},
            {"name": "route_quality", "kind": FeatureKind.CATEGORICAL},
            {"name": "trip_distance", "kind": FeatureKind.NUMERIC},
        ]
    }
    ranker = CartoBoostRanker(
        n_estimators=6,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        group_col="query_id",
        split_policy="axis_only",
    ).fit(X, y, feature_schema=schema)

    full_scores = ranker.predict(X)
    feature_scores = ranker.predict(X.drop(columns=["query_id"]))

    assert ranker.n_features_in_ == 2
    assert ranker.categorical_encoder_["columns"][0]["index"] == 0
    assert ranker.feature_schema_["names"][-1] == "trip_distance"
    assert feature_scores == pytest.approx(full_scores)
    assert ranker.score_groups(X, y)["ndcg"] > 0.8


def test_ranker_rejects_noncontiguous_query_ids():
    with pytest.raises(ValueError, match="contiguous"):
        CartoBoostRanker().fit(
            [[0.0], [1.0], [2.0]],
            [0.0, 1.0, 2.0],
            groups=["a", "b", "a"],
        )


def test_ranker_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        CartoBoostRanker().predict([[1.0]])


def test_ranker_weights_and_pickle_roundtrip(tmp_path: Path):
    X = [[0.0], [1.0], [0.0], [1.0]]
    y = [0.0, 1.0, 0.0, 2.0]
    ranker = CartoBoostRanker(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
    ).fit(X, y, groups=[2, 2])

    expected = ranker.predict(X)
    path = tmp_path / "ranker.weights.json"
    ranker.save_weights(path)

    loaded = CartoBoostRanker.load_weights(path)
    restored = pickle.loads(pickle.dumps(ranker))

    assert loaded.predict(X) == pytest.approx(expected)
    assert restored.predict(X) == pytest.approx(expected)
    with pytest.raises(NotImplementedError, match="JSON only"):
        ranker.save_weights(tmp_path / "ranker.onnx", format="onnx")


def test_ranker_graph_smoothing_roundtrip(tmp_path: Path):
    ranker = CartoBoostRanker(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        graph_indptr=[0, 1, 2, 3, 4],
        graph_indices=[1, 0, 3, 2],
        graph_weights=[1.0, 1.0, 1.0, 1.0],
        graph_smoothing=0.5,
        graph_smoothing_iterations=2,
        backend="cpu",
    ).fit([[0.0], [1.0], [0.0], [1.0]], [0.0, 2.0, 0.0, 3.0], groups=[2, 2])
    path = tmp_path / "ranker-graph.json"
    expected = ranker.predict([[0.0], [1.0]])
    ranker.save(path)
    loaded = CartoBoostRanker.load(path)

    assert loaded.graph_indices == [1, 0, 3, 2]
    assert loaded.graph_smoothing_iterations == 2
    assert ranker.selected_backend_ == "cpu"
    assert loaded.selected_backend_ == "cpu"
    assert loaded.predict([[0.0], [1.0]]) == pytest.approx(expected)


def test_ranker_fit_group_column_is_retained_without_changing_constructor(tmp_path: Path):
    X = np.asarray([[10.0, 0.0], [10.0, 1.0], [20.0, 0.0], [20.0, 1.0]])
    y = [0.0, 1.0, 0.0, 2.0]
    model = CartoBoostRanker(n_estimators=2, min_samples_leaf=1).fit(X, y, group_col=0)

    assert model.get_params()["group_col"] is None
    assert model.group_col_ == 0
    expected = model.predict(X[:, 1:])
    np.testing.assert_array_equal(model.predict(X), expected)
    assert model.score_groups(X, y) == model.score_groups(X[:, 1:], y, groups=[2, 2])
    path = tmp_path / "ranker.json"
    model.save(path)
    restored = CartoBoostRanker.load(path)
    assert restored.get_params()["group_col"] is None
    assert restored.group_col_ == 0
    np.testing.assert_array_equal(restored.predict(X), expected)


@pytest.mark.parametrize("groups", [[2.9, 2.1], [True, 3], [float("inf"), 2], [float("nan"), 2]])
def test_ranker_rejects_noninteger_group_sizes(groups):
    with pytest.raises(ValueError, match="positive integers"):
        _normalize_groups(groups, 4)


@pytest.mark.parametrize("table_format", ["polars", "duckdb"])
def test_ranker_group_column_accepts_optional_tables_and_preserves_names(table_format, tmp_path):
    module = pytest.importorskip(table_format)
    columns = {
        "query": ["a", "a", "b", "b"],
        "distance": [0.0, 1.0, 0.0, 1.0],
    }
    if table_format == "duckdb":
        pd = pytest.importorskip("pandas")
        frame = module.from_df(pd.DataFrame(columns))
        features = frame.select("distance")
    else:
        frame = module.DataFrame(columns)
        features = frame.drop("query")
    y = [0.0, 1.0, 0.0, 2.0]
    model = CartoBoostRanker(n_estimators=2, min_samples_leaf=1).fit(frame, y, group_col="query")
    assert list(model.feature_names_in_) == ["distance"]
    np.testing.assert_array_equal(model.predict(frame), model.predict(features))
    assert model.score_groups(frame, y) == model.score_groups(features, y, groups=[2, 2])
    path = tmp_path / "ranker.json"
    model.save(path)
    np.testing.assert_array_equal(CartoBoostRanker.load(path).predict(frame), model.predict(frame))


@pytest.mark.parametrize("layout", ["fortran", "reversed", "strided"])
def test_ranker_wrapper_layout_and_row_batch_parity(layout):
    X = np.arange(48, dtype=float).reshape(12, 4)
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "reversed":
        X = X[::-1, ::-1]
    else:
        X = X[::2, ::2]
    model = CartoBoostRanker(n_estimators=2, min_samples_leaf=1).fit(
        X, [index % 2 for index in range(len(X))], groups=[2] * (len(X) // 2)
    )
    expected = model.predict(np.ascontiguousarray(X))
    np.testing.assert_array_equal(model.predict(X), expected)
    np.testing.assert_array_equal(
        np.concatenate([model.predict(X[index : index + 1]) for index in range(len(X))]), expected
    )
    with pytest.raises(ValueError, match="empty"):
        model.predict(X[:0])


def test_ranker_accepts_query_group_sizes_from_a_duckdb_relation():
    duckdb = pytest.importorskip("duckdb")
    X = [[0.0], [1.0], [0.0], [1.0]]
    y = [0.0, 1.0, 0.0, 2.0]
    groups = duckdb.sql("select * from (values (2), (2)) as rows(size)")
    model = CartoBoostRanker(n_estimators=2, min_samples_leaf=1).fit(X, y, groups=groups)
    assert model.groups_ == [2, 2]
    assert model.score_groups(X, y, groups=groups) == model.score_groups(X, y, groups=[2, 2])
