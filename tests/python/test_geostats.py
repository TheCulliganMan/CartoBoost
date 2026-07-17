import numpy as np
import pytest
from cartoboost import (
    CartoBoostRegressor,
    NearestNeighborGPRegressor,
    ResidualNNGPRegressor,
    binned_variogram,
    deterministic_neighbors,
    empirical_semivariogram,
    fit_variogram_wls,
)
from cartoboost.accelerators import available_backends
from cartoboost.geostats import directional_lane_distance_matrix


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


def test_directional_lane_metric_preserves_direction_and_supports_crossed_weights():
    lanes = np.array([[0.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    forward = directional_lane_distance_matrix(lanes)
    crossed = directional_lane_distance_matrix(lanes, mode="crossed")
    weighted = directional_lane_distance_matrix(
        lanes, origin_weight=2.0, destination_weight=0.5
    )
    np.testing.assert_allclose(np.diag(forward), 0.0)
    assert forward[0, 1] > 0.0  # A→B is distinct from B→A.
    assert crossed[0, 1] == 0.0
    assert weighted[0, 1] != forward[0, 1]


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


@pytest.mark.parametrize("backend", available_backends("dense"))
def test_variogram_fit_accepts_every_dense_backend(backend):
    bins = [
        {
            "lag_start": 0.0,
            "lag_end": 1.0,
            "lag_center": 0.5,
            "semivariance": 0.2,
            "pair_count": 4,
        },
        {
            "lag_start": 1.0,
            "lag_end": 2.0,
            "lag_center": 1.5,
            "semivariance": 0.7,
            "pair_count": 3,
        },
    ]
    fit = fit_variogram_wls(
        bins,
        kernels=["exponential"],
        range_candidates=[0.5, 1.0],
        sill_candidates=[0.5, 1.0],
        nugget_candidates=[0.0, 0.05],
        backend=backend,
    )
    assert fit["weighted_sse"] >= 0.0


def test_empirical_variogram_runs_on_every_available_backend():
    coords = np.array([[0.0, 0.0], [0.6, 0.2], [1.4, -0.1], [2.2, 0.4], [3.0, 0.0]])
    values = np.array([0.0, 1.0, 1.4, 1.8, 2.1])
    expected = empirical_semivariogram(
        coords,
        values,
        bin_count=2,
        max_distance=10.0,
        anisotropy_angle_degrees=23.0,
        anisotropy_scaling=1.3,
        backend="cpu",
    )
    for backend in available_backends("pairwise_distance"):
        actual = empirical_semivariogram(
            coords,
            values,
            bin_count=2,
            max_distance=10.0,
            anisotropy_angle_degrees=23.0,
            anisotropy_scaling=1.3,
            backend=backend,
        )
        assert actual == expected


def test_deterministic_neighbors_runs_on_every_available_backend():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    targets = np.array([[0.25, 0.0], [4.0, 0.0]])
    expected = deterministic_neighbors(coords, targets, k=2, backend="cpu")
    for backend in available_backends("pairwise_distance"):
        assert deterministic_neighbors(coords, targets, k=2, backend=backend) == expected


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


@pytest.mark.parametrize("backend", available_backends("pairwise_distance"))
def test_residual_nngp_constructs_spatial_stage_on_every_backend(backend):
    coords = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0]])
    x = coords[:, :1]
    y = np.array([2.0, 2.4, 1.8, 2.2])
    supplied_gp = NearestNeighborGPRegressor(n_neighbors=3)
    model = ResidualNNGPRegressor(
        MeanRegressor(), gp=supplied_gp, backend=backend
    ).fit(x, y, coords=coords)
    assert model.gp_.backend_ == backend
    assert supplied_gp.backend == "cpu"
    assert np.all(np.isfinite(model.predict(x, coords=coords)))


@pytest.mark.parametrize("backend", available_backends("pairwise_distance"))
def test_residual_nngp_preserves_backend_through_artifact(tmp_path, backend):
    coords = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0]])
    x = coords[:, :1]
    y = np.array([2.0, 2.4, 1.8, 2.2])
    model = ResidualNNGPRegressor(
        CartoBoostRegressor(n_estimators=2, min_samples_leaf=1),
        backend=backend,
    ).fit(x, y, coords=coords)
    path = tmp_path / "residual-nngp.json"
    model.save(path)

    restored = ResidualNNGPRegressor.load(path)

    assert restored.backend == backend
    assert restored.gp_.backend_ == backend
    np.testing.assert_allclose(restored.predict(x, coords=coords), model.predict(x, coords=coords))
