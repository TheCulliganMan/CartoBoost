from __future__ import annotations

import inspect

import pytest
from cartoboost import BoosterConfig
from cartoboost.forecasting import CartoBoostLagForecaster, LagConfig
from cartoboost.forecasting.lag_features import (
    CalendarFeatureConfig,
    LagFeatureConfig,
    RollingFeatureConfig,
)

pd = pytest.importorskip("pandas")


def test_lag_forecaster_uses_typed_configuration_without_generic_kwargs() -> None:
    signature = inspect.signature(CartoBoostLagForecaster)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    model = CartoBoostLagForecaster(
        booster_config=BoosterConfig(n_estimators=8),
        lag_config=LagConfig(lags=(1, 7)),
    )
    assert model.get_params()["n_estimators"] == 8


def test_cartoboost_lag_converts_panel_and_delegates_to_native(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    result = (
        CartoBoostLagForecaster(lags=[1], rolling_windows=[], calendar_features=False)
        .fit({"pickup_1": [10, 11, 12, 13], "pickup_2": [20, 22, 24, 26]})
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {"backend": "cpu", "lags": [1], "rolling_windows": [], "calendar_features": False},
    )
    assert native.calls[1][1].rows[:2] == [
        ("pickup_1", "1970-01-01T00:00:00", 10.0),
        ("pickup_1", "1970-01-02T00:00:00", 11.0),
    ]
    assert native.calls[2] == ("predict", (2,), {})


def test_cartoboost_lag_converts_supported_feature_configs(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    CartoBoostLagForecaster(
        lag_config=LagFeatureConfig(
            lags=[1, 24],
            difference_lags=[24],
            rolling_trend_windows=[3],
        ),
        rolling_config=RollingFeatureConfig(
            windows=[3],
            aggregations=["mean", "std", "min", "max"],
        ),
        calendar_config=CalendarFeatureConfig(features=["dayofweek", "month", "day"]),
        target_mode="seasonal_delta_7",
    ).fit({"pickup_1": [10, 11, 12, 13]})

    assert native.calls[0] == (
        "init",
        {
            "backend": "cpu",
            "lags": [1, 24],
            "difference_lags": [24],
            "rolling_trend_windows": [3],
            "partial_rolling_mean_windows": [],
            "rolling_windows": [3],
            "rolling_std_windows": [3],
            "rolling_min_windows": [3],
            "rolling_max_windows": [3],
            "calendar_features": True,
            "target_mode": "seasonal_delta_7",
        },
    )


def test_cartoboost_lag_maps_min_periods_one_to_partial_rolling_mean(
    install_fake_native,
):
    native = install_fake_native("CartoBoostLagForecaster")

    CartoBoostLagForecaster(
        rolling_config=RollingFeatureConfig(
            windows=[7, 14],
            aggregations=["mean"],
            min_periods=1,
        ),
        lags=[1],
        rolling_windows=[],
        calendar_features=False,
    ).fit({"pickup_1": [10, 11, 12, 13]})

    assert native.calls[0][1]["partial_rolling_mean_windows"] == [7, 14]
    assert native.calls[0][1]["rolling_windows"] == []


def test_cartoboost_lag_passes_supported_regressor_params(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    CartoBoostLagForecaster(
        lags=[1, 7],
        rolling_windows=[7],
        difference_lags=[7],
        rolling_trend_windows=[7],
        calendar_features=True,
        trend_features=True,
        target_mode="delta_from_last",
        split_policy="axis_only",
        regressor_params={
            "n_estimators": 32,
            "learning_rate": 0.08,
            "max_depth": 3,
            "min_samples_leaf": 4,
            "min_gain": 0.0,
        },
    ).fit({"pickup_1": [10, 11, 12, 13, 15, 18, 21, 24]})

    assert native.calls[0] == (
        "init",
        {
            "backend": "cpu",
            "lags": [1, 7],
            "rolling_windows": [7],
            "difference_lags": [7],
            "rolling_trend_windows": [7],
            "calendar_features": True,
            "trend_features": True,
            "target_mode": "delta_from_last",
            "n_estimators": 32,
            "learning_rate": 0.08,
            "max_depth": 3,
            "min_samples_leaf": 4,
            "min_gain": 0.0,
            "splitters": ["axis"],
        },
    )


def test_cartoboost_lag_passes_elapsed_calendar_feature_flag(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    CartoBoostLagForecaster(
        lags=[1],
        rolling_windows=[],
        calendar_features=True,
        elapsed_calendar_features=True,
        elapsed_calendar_periods=[7],
    ).fit({"pickup_1": [10, 11, 12, 13]})

    assert native.calls[0] == (
        "init",
        {
            "backend": "cpu",
            "lags": [1],
            "rolling_windows": [],
            "calendar_features": True,
            "elapsed_calendar_features": True,
            "elapsed_calendar_periods": [7],
        },
    )


def test_cartoboost_lag_dataframe_coercion_passes_static_covariates_to_native(
    install_fake_native,
):
    native = install_fake_native("CartoBoostLagForecaster")
    training = pd.DataFrame(
        {
            "lane_id": ["PU1-DO2", "PU1-DO2", "PU1-DO2"],
            "pickup_day": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "trips": [10.0, 12.0, 14.0],
            "distance_miles": [2.5, 2.5, 2.5],
            "airport_lane": [0.0, 0.0, 0.0],
        }
    )

    CartoBoostLagForecaster(
        time_col="pickup_day",
        target_col="trips",
        panel_cols=["lane_id"],
        covariate_features=["distance_miles", "airport_lane"],
        lags=[1],
        rolling_windows=[],
        calendar_features=False,
    ).fit(training)

    assert native.calls[0] == (
        "init",
        {
            "backend": "cpu",
            "covariate_features": ["distance_miles", "airport_lane"],
            "lags": [1],
            "rolling_windows": [],
            "calendar_features": False,
        },
    )
    assert native.calls[1][1].kwargs["row_covariates"] == [
        {"distance_miles": 2.5, "airport_lane": 0.0},
        {"distance_miles": 2.5, "airport_lane": 0.0},
        {"distance_miles": 2.5, "airport_lane": 0.0},
    ]


def test_cartoboost_lag_predict_passes_known_future_covariates_to_native(
    install_fake_native,
):
    native = install_fake_native("CartoBoostLagForecaster")
    training = pd.DataFrame(
        {
            "lane_id": ["PU1-DO2", "PU1-DO2", "PU1-DO2"],
            "pickup_day": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "trips": [10.0, 12.0, 14.0],
            "event_code": [0.0, 0.0, 1.0],
            "sell_price": [2.5, 2.5, 2.75],
        }
    )
    known_future = pd.DataFrame(
        {
            "lane_id": ["PU1-DO2"],
            "pickup_day": pd.to_datetime(["2026-01-04"]),
            "event_code": [3.0],
            "sell_price": [3.25],
        }
    )

    result = (
        CartoBoostLagForecaster(
            time_col="pickup_day",
            target_col="trips",
            panel_cols=["lane_id"],
            covariate_features=["event_code", "sell_price"],
            lags=[1],
            rolling_windows=[],
            calendar_features=False,
        )
        .fit(training)
        .predict(1, known_future=known_future)
    )

    assert result == {"args": (1, native.calls[2][1][1]), "kwargs": {}}
    assert native.calls[2][0] == "predict_with_known_future"
    assert native.calls[2][1][1].rows == [("PU1-DO2", "2026-01-04T00:00:00", 0.0)]
    assert native.calls[2][1][1].kwargs["row_covariates"] == [
        {"event_code": 3.0, "sell_price": 3.25}
    ]


def test_cartoboost_lag_rejects_unsupported_regressor_params() -> None:
    with pytest.raises(ValueError, match="unsupported CartoBoostLagForecaster regressor_params"):
        CartoBoostLagForecaster(regressor_params={"subsample": 0.8})


def test_cartoboost_lag_rejects_unsupported_calendar_config() -> None:
    with pytest.raises(ValueError, match="unsupported: \\['hour'\\]"):
        CartoBoostLagForecaster(calendar_config=CalendarFeatureConfig(features=["hour"]))


def test_cartoboost_lag_dataframe_coercion_rejects_duplicate_panel_timestamps(
    install_fake_native,
):
    install_fake_native("CartoBoostLagForecaster")
    training = pd.DataFrame(
        {
            "PULocationID": ["zone_4", "zone_4"],
            "pickup_hour": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
            "pickup_trips": [42.0, 45.0],
        }
    )
    forecaster = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_trips",
        panel_cols=["PULocationID"],
    )

    with pytest.raises(ValueError, match="unique timestamps within each panel"):
        forecaster.fit(training)
