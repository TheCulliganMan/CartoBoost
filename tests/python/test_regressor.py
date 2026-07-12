import json
from pathlib import Path

import cartoboost.tensorboard as tensorboard
import pytest
from cartoboost import CartoBoostRegressor


def test_get_params_and_set_params_reset_model():
    regressor = CartoBoostRegressor(n_estimators=3)
    assert regressor.get_params()["n_estimators"] == 3

    returned = regressor.set_params(learning_rate=0.2)

    assert returned is regressor
    assert regressor.learning_rate == 0.2


def test_fit_predict_and_roundtrip_native_backend(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    regressor = CartoBoostRegressor(
        n_estimators=8,
        learning_rate=0.4,
        min_samples_leaf=1,
    )

    regressor.fit(X, y)
    predictions = regressor.predict([[0.0], [3.0]])

    assert predictions[0] < predictions[1]

    model_path = tmp_path / "model.json"
    regressor.save(model_path)
    loaded = CartoBoostRegressor.load(model_path)

    assert loaded.predict([[0.0], [3.0]]) == pytest.approx(predictions)


def test_stable_model_save_uses_v2_envelope_and_migrates_v1(tmp_path: Path):
    model = CartoBoostRegressor(n_estimators=2, min_samples_leaf=1).fit(
        [[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0]
    )
    path = tmp_path / "model.json"
    model.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "cartoboost.model"
    assert payload["artifact_version"] == 2
    assert payload["model_type"] == "regressor"
    assert payload["schema_hash"]
    assert set(("library_version", "training_config", "payload")) <= set(payload)

    # v0.2.45's Python envelope is accepted only for the matching stable model
    # and is migrated in memory; it is never rewritten as a compatibility alias.
    legacy = dict(payload["payload"])
    legacy.update({"artifact_type": "cartoboost.regressor", "artifact_version": 1})
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    restored = CartoBoostRegressor.load(legacy_path)
    assert restored.predict([[0.0], [2.0]]) == pytest.approx(model.predict([[0.0], [2.0]]))


def test_regressor_records_training_history_and_tensorboard_scalars(monkeypatch, tmp_path: Path):
    calls = []

    class Writer:
        def __init__(self, path):
            calls.append(("path", path))

        def add_scalar(self, name, value, step):
            calls.append((name, value, step))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(tensorboard, "_summary_writer_class", lambda: Writer)
    model = CartoBoostRegressor(
        n_estimators=3,
        min_samples_leaf=1,
        tensorboard_log_dir=tmp_path,
        tensorboard_run_name="taxi-fare",
    ).fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0])

    assert model.training_history_
    assert model.training_history_[0]["name"] == "train/rmse"
    assert ("path", str(tmp_path / "taxi-fare")) in calls
    assert any(call[0] == "train/rmse" and call[2] == 1 for call in calls)


def test_quantile_native_backend_uses_quantile_initial_prediction_and_roundtrips(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 10.0, 20.0, 30.0]
    regressor = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        loss="quantile",
        quantile_alpha=0.8,
    ).fit(X, y)

    assert regressor.predict([[0.0], [3.0]]) == pytest.approx([30.0, 30.0])

    model_path = tmp_path / "quantile.json"
    regressor.save(model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    loaded = CartoBoostRegressor.load(model_path)

    assert "constant_prediction_value" not in payload
    assert loaded.loss == "quantile"
    assert loaded.quantile_alpha == pytest.approx(0.8)
    assert loaded.predict([[0.0], [3.0]]) == pytest.approx([30.0, 30.0])


def test_l1_native_backend_uses_weighted_median_initial_prediction_and_roundtrips(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 10.0, 20.0, 30.0]
    regressor = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        loss="mae",
    ).fit(X, y, sample_weight=[1.0, 1.0, 10.0, 1.0])

    assert regressor.predict([[0.0], [3.0]]) == pytest.approx([20.0, 20.0])

    model_path = tmp_path / "l1.json"
    regressor.save(model_path)
    loaded = CartoBoostRegressor.load(model_path)

    assert loaded.loss == "l1"
    assert loaded.predict([[0.0], [3.0]]) == pytest.approx([20.0, 20.0])


def test_native_backend_monotonic_constraint_blocks_decreasing_stump():
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        monotonic_constraints=[1],
    ).fit([[0.0], [1.0], [2.0], [3.0]], [10.0, 10.0, 0.0, 0.0])

    predictions = model.predict([[0.0], [3.0]])

    assert predictions[0] == pytest.approx(predictions[1])


def test_native_backend_save_weights_roundtrip_is_versioned_json(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    regressor = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.3,
        min_samples_leaf=1,
    ).fit(X, y)
    predictions = regressor.predict([[0.0], [3.0]])
    path = tmp_path / "model.weights.json"

    regressor.save_weights(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = CartoBoostRegressor.load_weights(path)

    assert payload["artifact_type"] == "cartoboost.weights"
    assert payload["weights_artifact_version"] == 1
    assert payload["model_artifact_version"] == 1
    assert payload["backend"] == "rust"
    assert loaded.predict([[0.0], [3.0]]) == pytest.approx(predictions)


def test_native_backend_save_weights_onnx_when_optional_dependency_is_available(tmp_path: Path):
    pytest.importorskip("onnx")
    regressor = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=0.3,
        min_samples_leaf=1,
    ).fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0])
    path = tmp_path / "model.onnx"

    regressor.save_weights(path)

    assert path.stat().st_size > 0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        CartoBoostRegressor().predict([[1.0]])


def test_rust_backend_accepts_special_splitters():
    regressor = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        splitters=["diagonal_2d"],
    )
    try:
        regressor.fit(
            [[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]], [-10.0, -10.0, 10.0, 10.0]
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    assert regressor.predict([[-2.0, -1.0], [2.0, 1.0]]) == pytest.approx([-10.0, 10.0])


def test_rust_backend_accepts_linear_fuzzy_and_sparse_options():
    linear = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=3,
        leaf_predictor="linear",
        linear_leaf_features=["0"],
        l2_regularization=0.0,
    )
    sparse = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        splitters=["sparse_set"],
    )
    fuzzy = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        fuzzy=True,
        fuzzy_bandwidth=1.0,
    )

    try:
        linear.fit([[0.0], [1.0], [2.0], [3.0]], [3.0, 5.0, 7.0, 9.0])
        sparse.fit([[7.0], [7.0], [3.0], [4.0]], [25.0, 25.0, -5.0, -5.0])
        fuzzy.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 10.0, 10.0])
    except ImportError as exc:
        pytest.skip(str(exc))

    assert linear.predict([[0.0], [3.0]]) == pytest.approx([4.5, 7.5])
    assert sparse.predict([[7.0], [3.0]]) == pytest.approx([25.0, -5.0])
    assert fuzzy.predict([[1.5]]) == pytest.approx([5.0])


def test_rust_backend_preserves_fuzzy_kernel_roundtrip(tmp_path: Path):
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        fuzzy=True,
        fuzzy_bandwidth=1.0,
        fuzzy_kernel="gaussian",
    )
    try:
        model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 10.0, 10.0])
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "fuzzy-kernel.json"
    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    assert loaded.fuzzy_kernel == "gaussian"
    assert loaded.predict([[1.5]]) == pytest.approx(model.predict([[1.5]]))


def test_rust_backend_quantile_and_monotonic_roundtrip(tmp_path: Path):
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        loss="quantile",
        quantile_alpha=0.8,
        monotonic_constraints=[1],
    )
    try:
        model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 10.0, 20.0, 30.0])
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native-quantile.json"
    predictions = model.predict([[0.0], [3.0]])
    model.save(path)
    loaded = CartoBoostRegressor.load(path)

    assert loaded.loss == "quantile"
    assert loaded.quantile_alpha == pytest.approx(0.8)
    assert loaded.monotonic_constraints == [1]
    assert loaded.predict([[0.0], [3.0]]) == pytest.approx(predictions)
