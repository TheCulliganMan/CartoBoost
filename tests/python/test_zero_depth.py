import numpy as np
import pytest
from cartoboost import CartoBoostRegressor
from cartoboost._native import CartoBoostRegressor as NativeCartoBoostRegressor


def test_zero_depth_predicts_constant_mean_native_backend():
    model = CartoBoostRegressor(max_depth=0)

    model.fit([[0.0], [1.0], [2.0]], [3.0, 6.0, 9.0])

    assert model.predict([[0.0], [10.0]]) == pytest.approx([6.0, 6.0])


def test_zero_depth_roundtrip_preserves_constant_model(tmp_path):
    model = CartoBoostRegressor(max_depth=0)
    model.fit([[0.0], [1.0]], [2.0, 8.0])
    model_path = tmp_path / "zero_depth.json"

    model.save(model_path)
    restored = CartoBoostRegressor.load(model_path)

    assert restored.max_depth == 0
    assert restored.predict([[0.0], [1.0]]) == pytest.approx([5.0, 5.0])


def test_zero_depth_constant_model_accepts_valid_native_only_splitters():
    model = NativeCartoBoostRegressor(
        max_depth=0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=1.0,
    )

    model.fit([[0.0, 0.0], [1.0, 1.0]], [2.0, 6.0])

    assert model.splitters == ["gaussian_2d"]
    assert model.predict([[10.0, 10.0]]) == pytest.approx([4.0])


def test_negative_max_depth_is_rejected():
    model = CartoBoostRegressor(max_depth=-1)

    with pytest.raises(ValueError, match="non-negative"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


@pytest.mark.parametrize("loss", ["l2", "l1", "huber", "quantile", "log_l2"])
@pytest.mark.parametrize("weighted", [False, True])
def test_zero_depth_cached_prediction_matches_native_loss_and_roundtrip(tmp_path, loss, weighted):
    x = np.arange(5, dtype=np.float64).reshape(-1, 1)
    y = np.array([0.0, 1.0, 4.0, 10.0, 30.0])
    weights = np.array([1.0, 0.0, 5.0, 1.0, 2.0]) if weighted else None
    model = CartoBoostRegressor(
        n_estimators=3,
        max_depth=0,
        loss=loss,
        quantile_alpha=0.8,
        huber_delta=2.0,
        learning_rate=0.3,
    ).fit(x, y, sample_weight=weights)
    native = model._model.predict_arrays(x)
    np.testing.assert_array_equal(model.predict(x), native)
    path = tmp_path / "zero-depth.json"
    model.save(path)
    restored = CartoBoostRegressor.load(path)
    np.testing.assert_array_equal(restored.predict(x), native)
    np.testing.assert_array_equal(restored._model.predict_arrays(x), native)
