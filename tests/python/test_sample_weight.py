import cartoboost.regressor as regressor_module
import numpy as np
import pytest
from cartoboost import CartoBoostRegressor


def test_native_backend_uses_weighted_initial_prediction():
    model = CartoBoostRegressor(max_depth=0)

    model.fit([[0.0], [1.0]], [0.0, 10.0], sample_weight=[3.0, 1.0])

    assert model.predict([[0.0], [1.0]]) == pytest.approx([2.5, 2.5])


def test_sample_weight_validation():
    model = CartoBoostRegressor()

    with pytest.raises(ValueError, match="length"):
        model.fit([[0.0], [1.0]], [0.0, 1.0], sample_weight=[1.0])
    with pytest.raises(ValueError, match="finite non-negative"):
        model.fit([[0.0], [1.0]], [0.0, 1.0], sample_weight=[1.0, -1.0])


def test_sample_weight_is_passed_to_native_when_supported(monkeypatch):
    calls = {}

    class NativeWithWeights:
        def __init__(self, **params):
            calls["params"] = params

        def fit_arrays(self, rows, targets, sample_weight, offsets, ids, schema):
            calls["fit"] = (rows, targets, sample_weight)
            assert offsets == ids == []
            assert schema is None

        def predict(self, rows):
            return [0.0 for _ in rows]

    monkeypatch.setattr(regressor_module, "_NativeRegressorModel", NativeWithWeights)

    model = CartoBoostRegressor(n_estimators=1)
    model.fit([[0.0], [1.0]], [0.0, 1.0], sample_weight=[0.25, 0.75])

    for actual, expected in zip(
        calls["fit"], ([[0.0], [1.0]], [0.0, 1.0], [0.25, 0.75]), strict=True
    ):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    assert model._backend_used == "rust"


def test_native_backend_requires_typed_array_training_support(monkeypatch):
    class NativeWithoutWeights:
        def __init__(self, **params):
            pass

        def fit(self, rows, targets):
            pass

    monkeypatch.setattr(regressor_module, "_NativeRegressorModel", NativeWithoutWeights)

    model = CartoBoostRegressor(max_depth=1)

    with pytest.raises(RuntimeError, match="missing the typed fit_arrays binding"):
        model.fit([[0.0], [1.0]], [0.0, 10.0], sample_weight=[1.0, 3.0])
