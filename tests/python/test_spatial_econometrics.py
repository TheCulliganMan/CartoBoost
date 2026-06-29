from __future__ import annotations

import numpy as np
import pytest
from cartoboost import (
    SpatialDurbinRegressor,
    SpatialErrorRegressor,
    SpatialLagRegressor,
    SpatialTwoStageLeastSquares,
    SpatialWeights,
)


def _taxi_zone_chain_weights() -> SpatialWeights:
    return SpatialWeights(
        4,
        4,
        rows=[0, 1, 1, 2, 2, 3],
        cols=[1, 0, 2, 1, 3, 2],
        values=[1.0] * 6,
    )


def test_spatial_lag_predicts_and_summarizes() -> None:
    weights = _taxi_zone_chain_weights()
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.6, 4.4, 5.8])

    model = SpatialLagRegressor().fit(x, y, spatial_weights=weights)

    pred = model.predict(x, spatial_weights=weights)
    summary = model.summary()
    assert pred.shape == (4,)
    assert summary["model"] == "SpatialLagRegressor"
    assert summary["diagnostics"]["rho"] is not None
    assert np.isfinite(summary["diagnostics"]["residual_morans_i"])
    assert model.get_params() == {"row_standardize": True}
    assert model.score(x, y, spatial_weights=weights) >= 0.0
    assert model.metadata_["fitted"] is True


def test_spatial_error_reports_lambda() -> None:
    weights = _taxi_zone_chain_weights()
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.2, 4.0, 5.3])

    model = SpatialErrorRegressor().fit(x, y, spatial_weights=weights)

    assert model.summary()["diagnostics"]["lambda"] is not None


def test_spatial_durbin_effects_and_save_load(tmp_path) -> None:
    weights = _taxi_zone_chain_weights()
    x = np.array([[1.0], [2.0], [4.0], [8.0]])
    y = np.array([2.0, 3.0, 6.0, 10.0])
    model = SpatialDurbinRegressor().fit(x, y, spatial_weights=weights)
    before = model.predict(x, spatial_weights=weights)

    path = tmp_path / "spatial-durbin.json"
    model.save(path)
    loaded = SpatialDurbinRegressor.load(path)

    np.testing.assert_allclose(before, loaded.predict(x, spatial_weights=weights))
    assert loaded.summary()["diagnostics"]["total_effects"] is not None


def test_spatial_2sls_uses_same_public_contract() -> None:
    weights = _taxi_zone_chain_weights()
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 2.6, 4.4, 5.8]

    model = SpatialTwoStageLeastSquares().fit(x, y, spatial_weights=weights)

    assert model.predict(x, spatial_weights=weights).shape == (4,)


def test_invalid_weights_fail_clearly() -> None:
    with pytest.raises(ValueError, match="square"):
        SpatialWeights(2, 3, rows=[0], cols=[1], values=[1.0])


def test_isolated_nodes_are_explicit() -> None:
    weights = SpatialWeights(3, 3, rows=[0, 1], cols=[1, 0], values=[1.0, 1.0])

    assert weights.isolated_rows_ == [2]


def test_spatial_lag_small_fixture_matches_pysal_prediction_shape_when_available() -> None:
    libpysal = pytest.importorskip("libpysal")
    spreg = pytest.importorskip("spreg")
    weights = _taxi_zone_chain_weights()
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.6, 4.4, 5.8])

    model = SpatialLagRegressor().fit(x, y, spatial_weights=weights)
    cartoboost_pred = model.predict(x, spatial_weights=weights)
    pysal_weights = libpysal.weights.W(
        {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2],
        }
    )
    pysal_weights.transform = "r"
    pysal_model = spreg.ML_Lag(y.reshape(-1, 1), x, w=pysal_weights)

    assert cartoboost_pred.shape == np.asarray(pysal_model.predy).reshape(-1).shape
    assert np.isfinite(cartoboost_pred).all()
