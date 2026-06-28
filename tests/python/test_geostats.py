import numpy as np
import pytest
from cartoboost import (
    NearestNeighborGPRegressor,
    ResidualNNGPRegressor,
    binned_variogram,
    empirical_semivariogram,
    fit_variogram_wls,
)


class MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.mean_)


def test_nearest_neighbor_gp_recovers_synthetic_field_and_uncertainty():
    x = np.linspace(0.0, 1.0, 30)
    coords = np.column_stack([x, np.sin(x)])
    y = np.sin(3.0 * x) + 0.25 * coords[:, 1]
    model = NearestNeighborGPRegressor(
        kernel="matern_3_2",
        range=0.25,
        sill=1.0,
        nugget=1e-6,
        n_neighbors=10,
    ).fit(None, y, coords=coords)

    mean, std = model.predict(None, coords=coords[[10]], return_std=True)
    far_mean, far_var = model.predict(None, coords=np.array([[5.0, 5.0]]), return_var=True)

    assert abs(mean[0] - y[10]) < 1e-3
    assert std[0] >= 0.0
    assert far_var[0] >= std[0] ** 2
    assert np.isfinite(far_mean[0])
    assert model.get_params()["kernel"] == "matern_3_2"
    assert model.score(None, y, coords=coords) < 1e-6


def test_nearest_neighbor_gp_save_load_preserves_predictions(tmp_path):
    coords = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
    y = np.array([1.0, 1.5, 2.0])
    model = NearestNeighborGPRegressor(range=0.5, n_neighbors=2).fit(None, y, coords=coords)
    before = model.predict(None, coords=coords)

    path = tmp_path / "nngp.json"
    model.save(path)
    loaded = NearestNeighborGPRegressor.load(path)

    np.testing.assert_allclose(loaded.predict(None, coords=coords), before)


def test_nearest_neighbor_gp_rejects_duplicate_coordinates():
    model = NearestNeighborGPRegressor()
    with pytest.raises(ValueError, match="duplicate coordinates"):
        model.fit(None, [1.0, 2.0], coords=[[0.0, 0.0], [0.0, 0.0]])


def test_variogram_utilities_return_weighted_fit():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    values = np.array([0.0, 1.0, 1.5, 1.75])
    bins = empirical_semivariogram(coords, values, bin_count=3)
    assert binned_variogram(coords, values, bin_count=3) == bins
    fit = fit_variogram_wls(
        bins,
        range_candidates=[1.0, 2.0],
        sill_candidates=[0.5, 1.0],
        nugget_candidates=[0.0, 0.05],
    )
    assert fit["kernel"] in {"exponential", "squared_exponential", "matern_3_2", "matern_5_2"}
    assert fit["weighted_sse"] >= 0.0


def test_residual_nngp_adds_base_prediction_and_returns_std():
    coords = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0]])
    X = coords[:, :1]
    y = np.array([2.0, 2.4, 1.8, 2.2])
    model = ResidualNNGPRegressor(
        MeanRegressor(),
        gp=NearestNeighborGPRegressor(range=0.5, n_neighbors=3),
    ).fit(X, y, coords=coords)

    pred, std = model.predict(X, coords=coords, return_std=True)
    assert pred.shape == y.shape
    assert np.all(std >= 0.0)
    assert model.score(X, y, coords=coords) >= 0.0
