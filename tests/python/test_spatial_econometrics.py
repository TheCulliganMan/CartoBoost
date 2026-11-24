from __future__ import annotations

import pickle

import numpy as np
import pytest
from cartoboost.accelerators import available_backends
from cartoboost.spatial_econometrics import (
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


def _identified_ring_fixture() -> tuple[SpatialWeights, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = 12
    rows: list[int] = []
    cols: list[int] = []
    dense_weights = np.zeros((n_rows, n_rows), dtype=float)
    for row in range(n_rows):
        neighbors = ((row - 1) % n_rows, (row + 1) % n_rows)
        for col in neighbors:
            rows.append(row)
            cols.append(col)
            dense_weights[row, col] = 0.5
    weights = SpatialWeights(
        n_rows,
        n_rows,
        rows=rows,
        cols=cols,
        values=[1.0] * len(rows),
    )
    x = np.asarray([0.0, 1.0, 4.0, 2.0, 7.0, 3.0, 9.0, 5.0, 11.0, 6.0, 10.0, 8.0]).reshape(-1, 1)
    innovations = np.asarray(
        [0.30, -0.20, 0.10, -0.35, 0.25, 0.05, -0.15, 0.40, -0.25, 0.15, -0.05, -0.10]
    )
    return weights, dense_weights, x, innovations


def _spatial_lag_target(
    dense_weights: np.ndarray,
    x: np.ndarray,
    innovations: np.ndarray,
    *,
    rho: float,
    beta: float,
    theta: float = 0.0,
) -> np.ndarray:
    structural_mean = 1.5 + beta * x[:, 0] + theta * (dense_weights @ x[:, 0])
    return np.linalg.solve(
        np.eye(x.shape[0]) - rho * dense_weights,
        structural_mean + innovations,
    )


@pytest.mark.parametrize(
    "model_class",
    [
        SpatialLagRegressor,
        SpatialErrorRegressor,
        SpatialDurbinRegressor,
        SpatialTwoStageLeastSquares,
    ],
)
def test_spatial_array_paths_preserve_legacy_and_changed_network_predictions(model_class, tmp_path):
    weights, dense, x, noise = _identified_ring_fixture()
    y = _spatial_lag_target(dense, x, noise, rho=0.35, beta=1.2)
    features = np.column_stack([x[:, 0], np.sin(x[:, 0])])
    model = model_class().fit(np.asfortranarray(features), y, spatial_weights=weights)
    native = model._native_cls("cpu")
    native.fit(features.tolist(), y.tolist(), weights._native)
    original = np.asarray(native.predict(features.tolist(), weights._native))
    for values in [
        features,
        np.asfortranarray(features),
        features[::-1],
        np.repeat(features, 2, axis=1)[:, ::2],
    ]:
        expected = np.asarray(native.predict(values.tolist(), weights._native))
        np.testing.assert_array_equal(model.predict(values, spatial_weights=weights), expected)
    np.testing.assert_array_equal(model.predict(features, spatial_weights=weights), original)
    path = tmp_path / "spatial.json"
    model.save(path)
    restored = model_class.load(path)
    np.testing.assert_array_equal(restored.predict(features, spatial_weights=weights), original)
    # Full-network inference must invalidate cached solves for different CSR weights.
    rows = np.arange(len(x))
    for offset in [2, 3, 1]:
        changed = SpatialWeights(len(x), len(x), rows, (rows + offset) % len(x), np.ones(len(x)))
        w = np.zeros_like(dense)
        w[rows, (rows + offset) % len(x)] = 1.0
        structural = model.intercept_ + features @ model.coef_
        if model.durbin_coef_.size:
            structural += (w @ features) @ model.durbin_coef_
        rho = model.diagnostics_.get("rho")
        oracle = (
            structural if rho is None else np.linalg.solve(np.eye(len(x)) - rho * w, structural)
        )
        for _ in range(2):
            np.testing.assert_allclose(
                model.predict(features, spatial_weights=changed), oracle, atol=1e-10, rtol=1e-10
            )


def test_native_r_squared_validates_and_matches_constant_target_conventions():
    from cartoboost import _native

    values = np.arange(12, dtype=float)
    prediction = values * 0.7 + 0.5
    expected = 1.0 - np.sum((values - prediction) ** 2) / np.sum((values - values.mean()) ** 2)
    assert _native.r_squared_arrays(values[::-1], prediction[::-1]) == pytest.approx(expected)
    assert _native.r_squared_arrays(np.ones(3), np.ones(3)) == 1.0
    assert _native.r_squared_arrays(np.ones(3), np.zeros(3)) == 0.0
    decimal = np.full(3, 0.1)
    assert _native.r_squared_arrays(decimal, decimal) == 1.0
    assert _native.r_squared_arrays(decimal, np.zeros(3)) == 0.0
    for actual, pred in [
        (values, prediction[:-1]),
        (np.array([]), np.array([])),
        (np.array([np.nan]), np.array([0.0])),
    ]:
        with pytest.raises(ValueError):
            _native.r_squared_arrays(actual, pred)


def test_spatial_lag_predicts_and_summarizes() -> None:
    weights, dense_weights, x, innovations = _identified_ring_fixture()
    y = _spatial_lag_target(dense_weights, x, innovations, rho=0.35, beta=1.2)

    model = SpatialLagRegressor().fit(x, y, spatial_weights=weights)

    pred = model.predict(x, spatial_weights=weights)
    summary = model.summary()
    assert pred.shape == (12,)
    assert summary["model"] == "SpatialLagRegressor"
    assert summary["diagnostics"]["rho"] is not None
    assert np.isfinite(summary["diagnostics"]["residual_morans_i"])
    assert model.get_params() == {"row_standardize": True, "backend": "cpu"}
    expected_r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert model.score(x, y, spatial_weights=weights) == pytest.approx(expected_r2)
    assert model.metadata_["fitted"] is True
    assert model.metadata_["backend"] == {"requested": "cpu", "selected": "cpu"}


def test_spatial_lag_public_model_accepts_every_complete_backend() -> None:
    weights, dense_weights, x, innovations = _identified_ring_fixture()
    y = _spatial_lag_target(dense_weights, x, innovations, rho=0.35, beta=1.2)
    expected = (
        SpatialLagRegressor(backend="cpu")
        .fit(x, y, spatial_weights=weights)
        .predict(x, spatial_weights=weights)
    )

    for backend in available_backends("csr_diffusion"):
        model = SpatialLagRegressor(backend=backend).fit(x, y, spatial_weights=weights)
        actual = model.predict(x, spatial_weights=weights)
        assert model.backend_ == backend
        np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-4)


def test_spatial_error_reports_lambda() -> None:
    weights, dense_weights, x, innovations = _identified_ring_fixture()
    disturbances = np.linalg.solve(np.eye(x.shape[0]) - 0.4 * dense_weights, innovations)
    y = 2.0 + 1.1 * x[:, 0] + disturbances

    model = SpatialErrorRegressor().fit(x, y, spatial_weights=weights)

    assert model.summary()["diagnostics"]["lambda"] is not None


def test_spatial_durbin_effects_and_save_load(tmp_path) -> None:
    weights, dense_weights, x, innovations = _identified_ring_fixture()
    y = _spatial_lag_target(dense_weights, x, innovations, rho=0.25, beta=1.1, theta=0.4)
    model = SpatialDurbinRegressor().fit(x, y, spatial_weights=weights)
    before = model.predict(x, spatial_weights=weights)

    path = tmp_path / "spatial-durbin.json"
    model.save(path)
    loaded = SpatialDurbinRegressor.load(path)
    weights_path = tmp_path / "spatial-durbin.weights.json"
    model.save_weights(weights_path)
    weights_loaded = SpatialDurbinRegressor.load_weights(weights_path)
    pickled = pickle.loads(pickle.dumps(model))

    np.testing.assert_allclose(before, loaded.predict(x, spatial_weights=weights))
    np.testing.assert_allclose(before, weights_loaded.predict(x, spatial_weights=weights))
    np.testing.assert_allclose(before, pickled.predict(x, spatial_weights=weights))
    assert loaded.summary()["diagnostics"]["total_effects"] is not None
    assert model.durbin_coef_.shape == (1,)
    np.testing.assert_allclose(model.durbin_coef_, loaded.durbin_coef_)
    assert loaded.backend_ == "cpu"
    assert model.summary()["durbin_coefficients"] == pytest.approx(model.durbin_coef_.tolist())
    with pytest.raises(ValueError, match="SpatialLagRegressor requires SpatialLag"):
        SpatialLagRegressor.load(path)


def test_spatial_2sls_uses_same_public_contract() -> None:
    weights = _taxi_zone_chain_weights()
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 2.6, 4.4, 5.8]

    model = SpatialTwoStageLeastSquares().fit(x, y, spatial_weights=weights)

    assert model.predict(x, spatial_weights=weights).shape == (4,)


def test_invalid_weights_fail_clearly() -> None:
    with pytest.raises(ValueError, match="square"):
        SpatialWeights(2, 3, rows=[0], cols=[1], values=[1.0])
    with pytest.raises(ValueError, match="zero diagonal"):
        SpatialWeights(2, 2, rows=[0], cols=[0], values=[1.0])


def test_isolated_nodes_are_explicit() -> None:
    weights = SpatialWeights(3, 3, rows=[0, 1], cols=[1, 0], values=[1.0, 1.0])

    assert weights.isolated_rows_ == [2]


def test_spatial_lag_small_fixture_matches_pysal_prediction_shape_when_available() -> None:
    libpysal = pytest.importorskip("libpysal")
    spreg = pytest.importorskip("spreg")
    weights, dense_weights, x, innovations = _identified_ring_fixture()
    y = _spatial_lag_target(dense_weights, x, innovations, rho=0.35, beta=1.2)

    model = SpatialLagRegressor().fit(x, y, spatial_weights=weights)
    cartoboost_pred = model.predict(x, spatial_weights=weights)
    pysal_weights = libpysal.weights.W(
        {row: [(row - 1) % x.shape[0], (row + 1) % x.shape[0]] for row in range(x.shape[0])}
    )
    pysal_weights.transform = "r"
    pysal_model = spreg.ML_Lag(y.reshape(-1, 1), x, w=pysal_weights)

    assert cartoboost_pred.shape == np.asarray(pysal_model.predy).reshape(-1).shape
    assert np.isfinite(cartoboost_pred).all()
