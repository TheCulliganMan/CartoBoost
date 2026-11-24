import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from cartoboost import CartoBoostClassifier, CartoBoostRanker, CartoBoostRegressor
from cartoboost.schema import FeatureKind


def test_binary_classifier_fit_predict_proba_and_roundtrip(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = ["low", "low", "high", "high"]
    classifier = CartoBoostClassifier(
        n_estimators=8,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y)

    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    decisions = classifier.decision_function(X)

    assert predictions.tolist() == y
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert decisions.shape == (4,)
    assert classifier.training_history_
    assert classifier.training_history_[0]["name"] == "train/logloss"
    assert (
        probabilities[0, classifier.classes_.tolist().index("high")]
        < probabilities[3, classifier.classes_.tolist().index("high")]
    )

    model_path = tmp_path / "classifier.json"
    classifier.save(model_path)
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    assert artifact["format"] == "cartoboost.model"
    assert artifact["artifact_version"] == 2
    assert artifact["model_type"] == "classifier"
    loaded = CartoBoostClassifier.load(model_path)

    assert loaded.classes_.tolist() == classifier.classes_.tolist()
    assert loaded.predict(X).tolist() == y
    assert loaded.predict_proba(X) == pytest.approx(probabilities)


def test_multiclass_classifier_learns_ordered_regions():
    X = [[0.0], [0.2], [2.0], [2.2], [4.0], [4.2]]
    y = [0, 0, 1, 1, 2, 2]
    classifier = CartoBoostClassifier(
        n_estimators=12,
        learning_rate=0.4,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
        objective="multiclass_logloss",
        split_policy="axis_only",
    ).fit(X, y)

    probabilities = classifier.predict_proba(X)

    assert classifier.predict(X).tolist() == y
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert classifier.decision_function(X).shape == (6, 3)


def test_classifier_preserves_first_seen_mixed_label_order():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = ["airport", "airport", 1, 1]
    classifier = CartoBoostClassifier(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y)

    assert classifier.classes_.tolist() == ["airport", 1]
    assert classifier.predict(X).tolist() == y


def test_classifier_roundtrip_preserves_tuple_labels(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [("zone", "airport"), ("zone", "airport"), ("zone", "midtown"), ("zone", "midtown")]
    classifier = CartoBoostClassifier(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y)
    model_path = tmp_path / "tuple-label-classifier.json"

    classifier.save(model_path)
    loaded = CartoBoostClassifier.load(model_path)

    assert loaded.classes_.tolist() == classifier.classes_.tolist()
    assert loaded.predict(X).tolist() == y


def test_classifier_accepts_native_categorical_features(tmp_path: Path):
    X = [["airport"], ["airport"], ["midtown"], ["midtown"]]
    y = ["long", "long", "short", "short"]
    schema = {"dense": [{"name": "pickup_zone", "kind": FeatureKind.CATEGORICAL}]}
    classifier = CartoBoostClassifier(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y, feature_schema=schema)
    probabilities = classifier.predict_proba(X)
    model_path = tmp_path / "categorical-classifier.json"

    classifier.save(model_path)
    loaded = CartoBoostClassifier.load(model_path)

    assert classifier.n_features_in_ == 1
    assert loaded.categorical_encoder_["columns"][0]["strategy"] == "OneHot"
    assert loaded.predict(X).tolist() == y
    assert loaded.predict_proba(X) == pytest.approx(probabilities)
    assert loaded.predict([["unknown"]]).shape == (1,)


def test_classifier_accepts_pandas_categorical_and_string_features():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "pickup_zone": pd.Series(
                ["airport", "airport", "midtown", "midtown"],
                dtype="category",
            ),
            "daypart": ["night", "night", "rush", "rush"],
        }
    )
    y = ["long", "long", "short", "short"]
    classifier = CartoBoostClassifier(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(X, y)

    assert classifier.feature_names_in_.tolist() == ["pickup_zone", "daypart"]
    assert classifier.categorical_encoder_["columns"][0]["name"] == "pickup_zone"
    assert classifier.predict(X).tolist() == y


def test_classifier_class_weight_changes_native_training_weights():
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 0, 0, 0, 0, 1]
    unweighted = CartoBoostClassifier(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        class_weight=None,
    ).fit(X, y)
    weighted = CartoBoostClassifier(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        class_weight={0: 1.0, 1: 20.0},
    ).fit(X, y)

    positive_index = weighted.classes_.tolist().index(1)

    assert (
        weighted.predict_proba([[0.0]])[0, positive_index]
        > unweighted.predict_proba([[0.0]])[0, positive_index]
    )


def test_classifier_balanced_class_weight_increases_rare_class_probability():
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0, 0, 0, 0, 0, 1]
    unweighted = CartoBoostClassifier(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        class_weight=None,
    ).fit(X, y)
    balanced = CartoBoostClassifier(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        class_weight="balanced",
    ).fit(X, y)

    positive_index = balanced.classes_.tolist().index(1)

    assert (
        balanced.predict_proba([[0.0]])[0, positive_index]
        > unweighted.predict_proba([[0.0]])[0, positive_index]
    )


def test_classifier_rejects_invalid_class_weight_type():
    with pytest.raises(ValueError, match="class_weight"):
        CartoBoostClassifier(class_weight=[1.0, 2.0]).fit(
            [[0.0], [1.0]],
            [0, 1],
        )


def test_classifier_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        CartoBoostClassifier().predict([[1.0]])


def test_classifier_weights_and_pickle_roundtrip(tmp_path: Path):
    classifier = CartoBoostClassifier(
        n_estimators=1,
        max_depth=0,
    ).fit([[0.0], [1.0]], ["airport", "midtown"])

    X = [[0.0], [1.0]]
    expected = classifier.predict_proba(X)
    path = tmp_path / "classifier.weights.json"
    classifier.save_weights(path)

    loaded = CartoBoostClassifier.load_weights(path)
    restored = pickle.loads(pickle.dumps(classifier))

    assert loaded.predict_proba(X) == pytest.approx(expected)
    assert restored.predict_proba(X) == pytest.approx(expected)
    assert loaded.predict(X).tolist() == classifier.predict(X).tolist()
    with pytest.raises(NotImplementedError, match="JSON only"):
        classifier.save_weights(tmp_path / "classifier.onnx", format="onnx")


def test_classifier_rejects_single_class_training_data():
    with pytest.raises(ValueError, match="at least two classes"):
        CartoBoostClassifier().fit([[0.0], [1.0]], ["airport", "airport"])


def test_classifier_rejects_2d_label_array():
    with pytest.raises(ValueError, match="1D array"):
        CartoBoostClassifier().fit([[0.0], [1.0]], np.asarray([[0], [1]], dtype=object))


def test_classifier_graph_smoothing_roundtrip(tmp_path: Path):
    classifier = CartoBoostClassifier(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        graph_indptr=[0, 1, 2, 3, 4],
        graph_indices=[1, 0, 3, 2],
        graph_weights=[1.0, 1.0, 1.0, 1.0],
        graph_smoothing=0.5,
        graph_smoothing_iterations=2,
        backend="cpu",
    ).fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    path = tmp_path / "classifier-graph.json"
    expected = classifier.predict_proba([[0.0], [3.0]])
    classifier.save(path)
    loaded = CartoBoostClassifier.load(path)

    assert loaded.graph_indptr == [0, 1, 2, 3, 4]
    assert loaded.graph_smoothing == pytest.approx(0.5)
    assert classifier.selected_backend_ == "cpu"
    assert loaded.selected_backend_ == "cpu"
    assert loaded.predict_proba([[0.0], [3.0]]) == pytest.approx(expected)


def test_classifier_loads_legacy_native_numeric_labels(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    model = CartoBoostClassifier(n_estimators=2, min_samples_leaf=1).fit(X, [0, 0, 1, 1])
    path = tmp_path / "native-classifier.json"
    model._model.save(path)
    original = path.read_bytes()

    restored = CartoBoostClassifier.load(path)

    assert restored.classes_.tolist() == model.classes_.tolist()
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    assert path.read_bytes() == original


@pytest.mark.parametrize("classes", [None, ["only-one"]])
def test_classifier_rejects_incomplete_label_envelope(tmp_path: Path, classes):
    model = CartoBoostClassifier(n_estimators=1).fit([[0.0], [1.0]], [0, 1])
    path = tmp_path / "classifier.json"
    model.save(path)
    payload = json.loads(path.read_text())
    if classes is None:
        del payload["payload"]["classes"]
    else:
        payload["payload"]["classes"] = classes
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="classes"):
        CartoBoostClassifier.load(path)


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_classifier_array_margins_match_native_list_path(classes, use_sparse):
    X = np.arange(12, dtype=float).reshape(6, 2)[:, ::-1]
    y = [row % classes for row in range(6)]
    sparse = {"zones": [[value] for value in y]} if use_sparse else None
    schema = {"dense": ["a", "b"], "sparse_sets": ["zones"]} if use_sparse else None
    model = CartoBoostClassifier(
        n_estimators=2,
        min_samples_leaf=1,
        split_policy="structured" if use_sparse else "axis_only",
    ).fit(X, y, sparse_sets=sparse, feature_schema=schema)
    native_sparse = [sparse["zones"]] if use_sparse else None
    expected = np.asarray(model._model.decision_function(X.tolist(), native_sparse))
    if classes == 2:
        expected = expected[:, 0]
    np.testing.assert_array_equal(model.decision_function(X, sparse_sets=sparse), expected)


@pytest.mark.parametrize(
    "model_type", [CartoBoostClassifier, CartoBoostRanker, CartoBoostRegressor]
)
@pytest.mark.parametrize("explicit_schema", [False, True])
def test_estimators_combine_categorical_and_sparse_features(tmp_path, model_type, explicit_schema):
    X = [["a"], ["b"], ["a"], ["b"]]
    sparse = {"zones": [[10], [20], [10], [20]]}
    schema = (
        {"dense": [{"name": "label", "kind": "Categorical"}], "sparse_sets": ["zones"]}
        if explicit_schema
        else None
    )
    kwargs = {"groups": [2, 2]} if model_type is CartoBoostRanker else {}
    model = model_type(n_estimators=2, min_samples_leaf=1, split_policy="structured").fit(
        X, [0, 1, 0, 1], feature_schema=schema, sparse_sets=sparse, **kwargs
    )
    assert model.feature_schema_["names"][-1] == "zones"
    assert model.feature_schema_["kinds"][-1] == "SparseSet"
    expected = model.predict(X, sparse_sets=sparse)
    path = tmp_path / "combined.json"
    model.save(path)
    np.testing.assert_array_equal(model_type.load(path).predict(X, sparse_sets=sparse), expected)


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("layout", ["fortran", "reversed", "strided"])
def test_classifier_wrapper_layout_and_row_batch_parity(classes, layout):
    X = np.arange(48, dtype=float).reshape(12, 4)
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "reversed":
        X = X[::-1, ::-1]
    else:
        X = X[::2, ::2]
    labels = [index % classes for index in range(len(X))]
    model = CartoBoostClassifier(n_estimators=2, min_samples_leaf=1).fit(X, labels)
    for method in (model.predict, model.predict_proba, model.decision_function):
        expected = method(np.ascontiguousarray(X))
        np.testing.assert_array_equal(method(X), expected)
        np.testing.assert_array_equal(
            np.concatenate([method(X[index : index + 1]) for index in range(len(X))]), expected
        )
        with pytest.raises(ValueError, match="empty"):
            method(X[:0])


def test_classifier_scores_tuple_labels_and_nullable_integer_labels():
    pd = pytest.importorskip("pandas")
    X = [[0.0], [1.0], [2.0], [3.0]]
    tuples = [("low", 0), ("low", 0), ("high", 1), ("high", 1)]
    config = dict(n_estimators=3, min_samples_leaf=1, learning_rate=1.0)
    model = CartoBoostClassifier(**config).fit(X, tuples)
    assert model.score(X, tuples) == 1.0
    labels = pd.Series([0, 0, 1, 1], dtype="Int64")
    model = CartoBoostClassifier(**config).fit(X, labels)
    assert all(type(label) is int for label in model.classes_)
    assert model.score(X, labels.to_frame()) == 1.0
