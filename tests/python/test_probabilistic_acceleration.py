from __future__ import annotations

import numpy as np
from cartoboost.forecasting.probabilistic import QuantileCartoBoostRegressor


def test_quantile_booster_preserves_backend_through_artifact(tmp_path) -> None:
    model = QuantileCartoBoostRegressor(
        quantiles=(0.25, 0.75),
        backend="cpu",
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
    )
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    model.fit(x, y)

    artifact = tmp_path / "quantile.json"
    model.save(artifact)
    loaded = QuantileCartoBoostRegressor.load(artifact)

    assert model.get_params()["backend"] == "cpu"
    assert model.metadata_["backend"]["requested"] == "cpu"
    assert set(model.metadata_["backend"]["selected"].values()) == {"cpu"}
    assert loaded.get_params()["backend"] == "cpu"
    assert np.allclose(loaded.predict_quantiles(x), model.predict_quantiles(x))


def test_quantile_booster_uses_native_batched_set_for_numeric_data() -> None:
    model = QuantileCartoBoostRegressor(
        quantiles=(0.25, 0.5, 0.75),
        backend="cpu",
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
    )
    x = np.arange(12.0, dtype=np.float64).reshape(-1, 1)
    y = np.linspace(1.0, 8.0, 12)

    model.fit(x, y)
    predictions = model.predict_quantiles(x)

    assert model._native_model is not None
    assert model.models_ == {}
    assert predictions.shape == (12, 3)
    assert np.all(predictions[:, :-1] <= predictions[:, 1:])
