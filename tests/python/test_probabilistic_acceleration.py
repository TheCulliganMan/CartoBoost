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
