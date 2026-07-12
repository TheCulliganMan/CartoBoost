from __future__ import annotations

import inspect

import pandas as pd
import pytest
from cartoboost import BoosterConfig
from cartoboost.forecasting import AutoForecastConfig, AutoForecaster, ForecastFrame


def test_auto_forecaster_uses_typed_configuration_without_generic_kwargs() -> None:
    signature = inspect.signature(AutoForecaster)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    config = AutoForecastConfig(booster_config=BoosterConfig(n_estimators=12))
    model = AutoForecaster(config=config)
    assert model.config.booster_config.n_estimators == 12


def test_auto_forecaster_delegates_to_native_auto_model(install_fake_native):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "pickup_hour": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
            }
        ),
        timestamp_col="pickup_hour",
        target_col="pickup_trips",
        freq="D",
    )

    result = (
        AutoForecaster(
            season_length=7,
            objective="wape",
            validation_window=2,
            validation_origin_count=3,
            baseline_displacement_gain=0.04,
            hard_winner_relative_gain=0.06,
            min_blend_weight=0.2,
            max_blend_weight=0.8,
            max_direct_horizon=14,
            n_estimators=16,
        )
        .fit(frame)
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "lags": [1, 2, 3, 7, 14, 28],
            "rolling_windows": [7, 14, 28],
            "partial_rolling_mean_windows": [],
            "rolling_std_windows": [7, 14, 28],
            "rolling_min_windows": [7, 14, 28],
            "rolling_max_windows": [7, 14, 28],
            "ewm_alpha_percents": [],
            "calendar_features": True,
            "rich_calendar_features": False,
            "elapsed_calendar_features": False,
            "elapsed_calendar_periods": [],
            "covariate_features": [],
            "covariate_calendar_interactions": False,
            "season_length": 7,
            "validation_window": 2,
            "validation_origin_count": 3,
            "objective": "wape",
            "baseline_displacement_gain": 0.04,
            "hard_winner_relative_gain": 0.06,
            "min_blend_weight": 0.2,
            "max_blend_weight": 0.8,
            "max_direct_horizon": 14,
            "max_candidate_count": None,
            "n_estimators": 16,
        },
    )
    assert native.calls[1][0] == "fit"
    assert native.calls[2] == ("predict", (2,), {})


def test_auto_forecaster_rejects_invalid_ewm_alpha_percents():
    with pytest.raises(ValueError, match="ewm_alpha_percents"):
        AutoForecaster(ewm_alpha_percents=[0])
    with pytest.raises(ValueError, match="duplicate"):
        AutoForecaster(ewm_alpha_percents=[90, 90])


def test_auto_forecaster_can_opt_into_partial_rolling_means(install_fake_native):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "lane_id": ["PU1-DO2"] * 8,
                "pickup_day": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
            }
        ),
        timestamp_col="pickup_day",
        target_col="pickup_trips",
        series_id_col="lane_id",
        freq="D",
    )

    AutoForecaster(partial_rolling_mean_windows=[7, 14]).fit(frame)

    assert native.calls[0][1]["partial_rolling_mean_windows"] == [7, 14]


def test_auto_forecaster_rejects_invalid_partial_rolling_mean_windows():
    with pytest.raises(ValueError, match="partial_rolling_mean_windows"):
        AutoForecaster(partial_rolling_mean_windows=[0])
    with pytest.raises(ValueError, match="duplicate"):
        AutoForecaster(partial_rolling_mean_windows=[7, 7])


def test_auto_forecaster_rejects_invalid_validation_origin_count():
    with pytest.raises(ValueError, match="validation_origin_count"):
        AutoForecaster(validation_origin_count=0)


def test_auto_forecaster_rejects_invalid_max_candidate_count():
    with pytest.raises(ValueError, match="max_candidate_count"):
        AutoForecaster(max_candidate_count=0)


def test_auto_forecaster_can_opt_into_ewm_alpha_percents(install_fake_native):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "lane_id": ["PU1-DO2"] * 8,
                "pickup_day": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
            }
        ),
        timestamp_col="pickup_day",
        target_col="pickup_trips",
        series_id_col="lane_id",
        freq="D",
    )

    AutoForecaster(ewm_alpha_percents=[90]).fit(frame)

    assert native.calls[0][1]["ewm_alpha_percents"] == [90]


def test_auto_forecaster_can_opt_into_elapsed_calendar_features(install_fake_native):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "lane_id": ["PU1-DO2"] * 8,
                "pickup_day": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
            }
        ),
        timestamp_col="pickup_day",
        target_col="pickup_trips",
        series_id_col="lane_id",
        freq="D",
    )

    model = AutoForecaster(
        elapsed_calendar_features=True,
        elapsed_calendar_periods=[7],
    ).fit(frame)

    assert native.calls[0][1]["elapsed_calendar_features"] is True
    assert native.calls[0][1]["elapsed_calendar_periods"] == [7]
    assert model.get_metadata()["auto_forecaster"]["elapsed_calendar_features"] is True
    assert model.get_metadata()["auto_forecaster"]["elapsed_calendar_periods"] == [7]


def test_auto_forecaster_rejects_invalid_elapsed_calendar_periods():
    with pytest.raises(ValueError, match="elapsed_calendar_periods"):
        AutoForecaster(elapsed_calendar_periods=[1])
    with pytest.raises(ValueError, match="duplicate"):
        AutoForecaster(elapsed_calendar_periods=[7, 7])
    with pytest.raises(ValueError, match="at most one"):
        AutoForecaster(elapsed_calendar_periods=[7, 12])


def test_auto_forecaster_uses_static_covariates_by_default(install_fake_native):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "lane_id": ["PU1-DO2"] * 8,
                "pickup_day": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
                "distance_miles": [2.5] * 8,
                "airport_lane": [0.0] * 8,
            }
        ),
        timestamp_col="pickup_day",
        target_col="pickup_trips",
        series_id_col="lane_id",
        static_covariates=["distance_miles", "airport_lane"],
        freq="D",
    )

    model = AutoForecaster(
        season_length=7,
        rich_calendar_features=True,
        covariate_calendar_interactions=True,
    ).fit(frame)

    assert native.calls[0][1]["covariate_features"] == ["distance_miles", "airport_lane"]
    assert native.calls[0][1]["rich_calendar_features"] is True
    assert native.calls[0][1]["calendar_features"] is True
    assert native.calls[0][1]["covariate_calendar_interactions"] is True
    metadata = model.get_metadata()["auto_forecaster"]
    assert metadata["covariate_features"] is None
    assert metadata["covariate_calendar_interactions"] is True
    assert metadata["effective_covariate_features"] == ["distance_miles", "airport_lane"]
    assert metadata["rich_calendar_features"] is True


def test_auto_forecaster_covariate_features_override_frame_static_covariates(
    install_fake_native,
):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "lane_id": ["PU1-DO2"] * 8,
                "pickup_day": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
                "distance_miles": [2.5] * 8,
            }
        ),
        timestamp_col="pickup_day",
        target_col="pickup_trips",
        series_id_col="lane_id",
        static_covariates=["distance_miles"],
        freq="D",
    )

    model = AutoForecaster(season_length=7, covariate_features=[]).fit(frame)

    assert native.calls[0][1]["covariate_features"] == []
    metadata = model.get_metadata()["auto_forecaster"]
    assert metadata["covariate_features"] == []
    assert metadata["effective_covariate_features"] == []


def test_auto_forecaster_rejects_invalid_covariate_feature_names():
    with pytest.raises(ValueError, match="sequence of column names"):
        AutoForecaster(covariate_features="distance_miles")
    with pytest.raises(ValueError, match="must not contain duplicate"):
        AutoForecaster(covariate_features=["distance_miles", "distance_miles"])
