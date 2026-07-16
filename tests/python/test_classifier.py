import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from cartoboost import CartoBoostClassifier
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
        splitters=["axis"],
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
        splitters=["axis"],
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
        splitters=["axis"],
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
        splitters=["axis"],
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
        splitters=["axis"],
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
        splitters=["axis"],
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
    assert loaded.predict_proba([[0.0], [3.0]]) == pytest.approx(expected)
