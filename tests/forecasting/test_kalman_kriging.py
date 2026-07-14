import importlib.util
import sys
from pathlib import Path

import pytest
from cartoboost.forecasting import AutoKalmanForecaster as PublicAutoKalmanForecaster
from cartoboost.forecasting import (
    AutoLocalLevelKalmanForecaster as PublicAutoLocalLevelKalmanForecaster,
)
from cartoboost.forecasting import KalmanForecaster as PublicKalmanForecaster
from cartoboost.forecasting import KrigingForecaster as PublicKrigingForecaster
from cartoboost.forecasting import (
    LocalLevelKalmanForecaster as PublicLocalLevelKalmanForecaster,
)
from cartoboost.forecasting import (
    SpatialPiecewiseKrigingForecaster as PublicSpatialPiecewiseKrigingForecaster,
)
from cartoboost.forecasting.local import (
    AutoKalmanForecaster,
    AutoLocalLevelKalmanForecaster,
    KalmanForecaster,
    KrigingForecaster,
    LocalLevelKalmanForecaster,
    SpatialPiecewiseKrigingForecaster,
)


def test_models_are_public_forecasting_imports():
    assert PublicAutoKalmanForecaster is AutoKalmanForecaster
    assert PublicAutoLocalLevelKalmanForecaster is AutoLocalLevelKalmanForecaster
    assert PublicKalmanForecaster is KalmanForecaster
    assert PublicLocalLevelKalmanForecaster is LocalLevelKalmanForecaster
    assert PublicKrigingForecaster is KrigingForecaster
    assert PublicSpatialPiecewiseKrigingForecaster is SpatialPiecewiseKrigingForecaster


def test_kalman_converts_panel_and_delegates_to_native(install_fake_native):
    native = install_fake_native("KalmanForecaster")

    result = (
        KalmanForecaster(
            level_process_variance=0.1,
            trend_process_variance=0.01,
            observation_variance=0.5,
        )
        .fit({"pickup_1": [10, 12, 14], "pickup_2": [30, 29, 28]})
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "level_process_variance": 0.1,
            "trend_process_variance": 0.01,
            "observation_variance": 0.5,
        },
    )
    assert native.calls[1][1].rows[0] == (
        "pickup_1",
        "1970-01-01T00:00:00",
        10.0,
    )


def test_kalman_fit_predict_uses_rust_binding():
    model = KalmanForecaster(
        level_process_variance=0.01,
        trend_process_variance=0.001,
        observation_variance=0.1,
    )

    model.fit([12.0, 14.0, 16.0, 18.0])
    result = model.predict(2)

    predictions = result.predictions()
    assert [row[3] for row in predictions] == ["kalman", "kalman"]
    assert [row[2] for row in predictions] == [1, 2]
    assert predictions[1][-1] > predictions[0][-1]


def test_auto_kalman_converts_panel_and_delegates_to_native(install_fake_native):
    native = install_fake_native("AutoKalmanForecaster")

    result = (
        AutoKalmanForecaster(
            level_process_variance_grid=[0.01, 0.1],
            trend_process_variance_grid=[0.001],
            observation_variance_grid=[0.5, 1.0],
            validation_window=2,
        )
        .fit({"pickup_1": [10, 12, 14, 16], "pickup_2": [30, 29, 28, 27]})
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "level_process_variance_grid": [0.01, 0.1],
            "trend_process_variance_grid": [0.001],
            "observation_variance_grid": [0.5, 1.0],
            "validation_window": 2,
        },
    )
    assert native.calls[1][1].rows[0] == (
        "pickup_1",
        "1970-01-01T00:00:00",
        10.0,
    )


def test_auto_kalman_fit_predict_uses_rust_binding():
    model = AutoKalmanForecaster(
        level_process_variance_grid=[0.001, 0.01],
        trend_process_variance_grid=[0.0001, 0.001],
        observation_variance_grid=[0.1, 1.0],
        validation_window=2,
    )

    model.fit([12.0, 14.0, 16.0, 18.0, 20.0, 22.0])
    result = model.predict(2)
    metadata = model.metadata_

    predictions = result.predictions()
    assert [row[3] for row in predictions] == ["auto_kalman", "auto_kalman"]
    assert [row[2] for row in predictions] == [1, 2]
    assert metadata["selected_params"]["level_process_variance"] in [0.001, 0.01]
    assert len(metadata["validation_scores"]) == 8


def test_local_level_kalman_fit_predict_uses_rust_binding():
    model = LocalLevelKalmanForecaster(
        level_process_variance=0.01,
        observation_variance=0.1,
    )

    model.fit([12.0, 13.0, 12.5, 13.5])
    result = model.predict(2)

    predictions = result.predictions()
    assert [row[3] for row in predictions] == ["local_level_kalman", "local_level_kalman"]
    assert [row[2] for row in predictions] == [1, 2]
    assert predictions[0][-1] == predictions[1][-1]


def test_auto_local_level_kalman_fit_predict_uses_rust_binding():
    model = AutoLocalLevelKalmanForecaster(
        level_process_variance_grid=[0.001, 0.01],
        observation_variance_grid=[0.1, 1.0],
        validation_window=2,
    )

    model.fit([12.0, 13.0, 12.5, 13.5, 13.0, 12.8])
    result = model.predict(2)
    metadata = model.metadata_

    predictions = result.predictions()
    assert [row[3] for row in predictions] == [
        "auto_local_level_kalman",
        "auto_local_level_kalman",
    ]
    assert metadata["selected_params"]["observation_variance"] in [0.1, 1.0]
    assert len(metadata["validation_scores"]) == 4


def test_kriging_normalizes_mapping_coordinates_and_delegates_to_native(install_fake_native):
    native = install_fake_native("KrigingForecaster")

    result = (
        KrigingForecaster(
            coordinates={"pickup_1": (0.0, 0.0), "pickup_2": (1.0, 0.0)},
            range=2.0,
            nugget=0.05,
        )
        .fit({"pickup_1": [10, 11], "pickup_2": [20, 21]})
        .predict(1)
    )

    assert result == {"args": (1,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "coordinates": [("pickup_1", 0.0, 0.0), ("pickup_2", 1.0, 0.0)],
            "range": 2.0,
            "nugget": 0.05,
            "sill": 1.0,
            "variogram_model": "exponential",
            "drift": "ordinary",
            "anisotropy_angle_degrees": 0.0,
            "anisotropy_scaling": 1.0,
            "max_neighbors": None,
            "min_neighbors": 1,
            "max_distance": None,
        },
    )


def test_kriging_fit_predict_uses_rust_binding():
    model = KrigingForecaster(
        coordinates={
            "PULocationID_142": (0.0, 0.0),
            "PULocationID_236": (10.0, 0.0),
        },
        range=1.0,
        nugget=1.0e-9,
    )

    model.fit({"PULocationID_142": [10.0, 12.0], "PULocationID_236": [40.0, 42.0]})
    result = model.predict(1)

    predictions = result.predictions()
    assert [row[3] for row in predictions] == ["kriging", "kriging"]
    assert [row[2] for row in predictions] == [1, 1]
    assert abs(predictions[0][-1] - 42.0) < 1.0e-4
    assert abs(predictions[1][-1] - 12.0) < 1.0e-4
    assert model.get_metadata()["target_policy"] == "leave_one_series_out"


def test_kriging_accepts_coordinate_triples(install_fake_native):
    native = install_fake_native("KrigingForecaster")

    KrigingForecaster(
        coordinates=[("pickup_1", 0, 0), ("pickup_2", 2, 1)],
        range=1.5,
        nugget=0.0,
    ).fit({"pickup_1": [10, 11], "pickup_2": [20, 21]})

    assert native.calls[0] == (
        "init",
        {
            "coordinates": [("pickup_1", 0.0, 0.0), ("pickup_2", 2.0, 1.0)],
            "range": 1.5,
            "nugget": 0.0,
            "sill": 1.0,
            "variogram_model": "exponential",
            "drift": "ordinary",
            "anisotropy_angle_degrees": 0.0,
            "anisotropy_scaling": 1.0,
            "max_neighbors": None,
            "min_neighbors": 1,
            "max_distance": None,
        },
    )


def test_kriging_passes_extended_native_parameters(install_fake_native):
    native = install_fake_native("KrigingForecaster")

    KrigingForecaster(
        coordinates=[("pickup_1", 0, 0), ("pickup_2", 2, 1), ("pickup_3", 3, 1)],
        range=1.5,
        nugget=0.01,
        sill=2.0,
        variogram_model="spherical",
        drift="linear",
        anisotropy_angle_degrees=30.0,
        anisotropy_scaling=0.5,
        max_neighbors=2,
        min_neighbors=2,
        max_distance=5.0,
    ).fit({"pickup_1": [10, 11], "pickup_2": [20, 21], "pickup_3": [30, 31]})

    assert native.calls[0] == (
        "init",
        {
            "coordinates": [
                ("pickup_1", 0.0, 0.0),
                ("pickup_2", 2.0, 1.0),
                ("pickup_3", 3.0, 1.0),
            ],
            "range": 1.5,
            "nugget": 0.01,
            "sill": 2.0,
            "variogram_model": "spherical",
            "drift": "linear",
            "anisotropy_angle_degrees": 30.0,
            "anisotropy_scaling": 0.5,
            "max_neighbors": 2,
            "min_neighbors": 2,
            "max_distance": 5.0,
        },
    )


def test_kriging_rejects_bad_mapping_coordinate_shape(install_fake_native):
    install_fake_native("KrigingForecaster")

    with pytest.raises(ValueError, match="series_id to \\(x, y\\)"):
        KrigingForecaster(coordinates={"pickup_1": (0.0, 0.0, 1.0)})


def test_spatial_piecewise_kriging_delegates_to_native(install_fake_native):
    native = install_fake_native("SpatialPiecewiseKrigingForecaster")

    result = (
        SpatialPiecewiseKrigingForecaster(
            coordinates={"pickup_1": (0.0, 0.0), "pickup_2": (1.0, 0.0)},
            mode="hybrid",
            spatial_regressors=["traffic_density"],
            range=2.0,
            nugget=0.05,
            residual_shrinkage=0.8,
        )
        .fit({"pickup_1": [10, 11], "pickup_2": [20, 21]})
        .predict(1)
    )

    assert result == {"args": (1,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "coordinates": [("pickup_1", 0.0, 0.0), ("pickup_2", 1.0, 0.0)],
            "mode": "hybrid",
            "spatial_regressors": ["traffic_density"],
            "range": 2.0,
            "nugget": 0.05,
            "sill": 1.0,
            "variogram_model": "exponential",
            "drift": "ordinary",
            "anisotropy_angle_degrees": 0.0,
            "anisotropy_scaling": 1.0,
            "max_neighbors": None,
            "min_neighbors": 1,
            "max_distance": None,
            "residual_shrinkage": 0.8,
            "allow_neighbor_fallback": False,
            "piecewise_config_json": None,
        },
    )


def test_spatial_piecewise_benchmark_reports_required_roster_and_deltas():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "forecasting_benchmark.py"
    spec = importlib.util.spec_from_file_location("forecasting_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    forecasting_benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = forecasting_benchmark
    spec.loader.exec_module(forecasting_benchmark)

    frame = forecasting_benchmark._spatial_piecewise_kriging_fixture(days=60, series_count=4)
    required = {
        "naive",
        "seasonal_naive",
        "piecewise_linear_seasonal",
        "kriging",
        "spatial_piecewise_kriging_hybrid",
    }
    assert required <= set(forecasting_benchmark._models(frame))
    assert forecasting_benchmark._coordinates_from_frame(frame)

    aggregate = forecasting_benchmark._aggregate_dataset_scores(
        {
            "rolling_origin_1": {
                "models": {
                    "piecewise_linear_seasonal": {
                        "status": "ok",
                        "rmse": 2.0,
                        "mae": 1.5,
                        "wape": 0.03,
                        "fit_seconds": 0.1,
                        "predict_seconds": 0.01,
                    },
                    "seasonal_naive": {
                        "status": "ok",
                        "rmse": 3.0,
                        "mae": 2.0,
                        "wape": 0.04,
                        "fit_seconds": 0.01,
                        "predict_seconds": 0.01,
                    },
                    "spatial_piecewise_kriging_hybrid": {
                        "status": "ok",
                        "rmse": 1.8,
                        "mae": 1.3,
                        "wape": 0.025,
                        "fit_seconds": 0.2,
                        "predict_seconds": 0.02,
                    },
                }
            }
        }
    )
    hybrid = aggregate["spatial_piecewise_kriging_hybrid"]
    assert hybrid["rmse"] < aggregate["piecewise_linear_seasonal"]["rmse"]
    assert "rmse_delta_vs_piecewise_linear_seasonal" in hybrid
    assert "fit_seconds" in hybrid
    assert "predict_seconds" in hybrid

    split_models = {
        "piecewise_linear_seasonal": {
            "status": "ok",
            "rmse": 2.0,
            "mae": 1.5,
            "wape": 0.03,
            "metadata": {},
        },
        "spatial_piecewise_kriging_hybrid": {
            "status": "ok",
            "rmse": 1.8,
            "mae": 1.3,
            "wape": 0.025,
            "metadata": {},
        },
    }
    forecasting_benchmark._add_split_baseline_delta_metadata(
        split_models,
        baseline="piecewise_linear_seasonal",
    )
    baseline_deltas = split_models["spatial_piecewise_kriging_hybrid"]["metadata"][
        "baseline_deltas"
    ]["piecewise_linear_seasonal"]
    assert baseline_deltas == {
        "rmse": pytest.approx(-0.2),
        "mae": pytest.approx(-0.2),
        "wape": pytest.approx(-0.005),
    }
