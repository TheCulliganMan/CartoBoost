from __future__ import annotations

import json

import numpy as np
import pytest
from cartoboost import CartoBoostRegressor, Prophet
from cartoboost.forecasting import ForecastFrame, PiecewiseLinearSeasonalForecaster


def _radial_fixture() -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-3.0, 3.0, 17)
    xx, yy = np.meshgrid(values, values)
    x = np.column_stack([xx.ravel(), yy.ravel()])
    y = np.where(np.hypot(x[:, 0], x[:, 1]) <= 1.5, 10.0, -10.0)
    return x, y


def _assert_json_close(left, right) -> None:
    if isinstance(left, float) or isinstance(right, float):
        assert left == pytest.approx(right)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_json_close(left[key], right[key])
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_json_close(left_item, right_item)
    else:
        assert left == right


def test_prophet_compatibility_surface_accepts_pandas_and_polars():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("polars")
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=36, freq="D"),
            "y": np.linspace(10.0, 30.0, 36) + np.sin(np.arange(36)),
            "promo": np.arange(36) % 2,
        }
    )
    model = (
        Prophet(
            yearly_seasonality=False,
            weekly_seasonality=2,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        .add_regressor("promo")
        .fit(frame)
    )
    future = model.make_future_dataframe(5)
    future["promo"] = 0.0
    forecast = model.predict(future)
    assert list(forecast.columns[:4]) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(forecast) == 41
    assert np.isfinite(forecast["yhat"].to_numpy()).all()

    import polars as pl

    polars_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).fit(pl.from_pandas(frame[["ds", "y"]]))
    polars_forecast = polars_model.predict(polars_model.make_future_dataframe(3))
    assert len(polars_forecast) == 39
    assert np.isfinite(polars_forecast["yhat"].to_numpy()).all()


def test_prophet_compatibility_surface_matches_prophet_changepoint_spacing():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=30, freq="D"),
            "y": np.arange(30, dtype=float),
        }
    )
    model = Prophet(
        n_changepoints=5,
        changepoint_range=0.8,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).fit(frame)
    assert model.changepoints.dt.day.tolist() == [6, 10, 15, 19, 24]
    assert model._model.metadata_["changepoint_timestamps"] == [
        "2026-01-06T00:00:00",
        "2026-01-10T00:00:00",
        "2026-01-15T00:00:00",
        "2026-01-19T00:00:00",
        "2026-01-24T00:00:00",
    ]


def test_prophet_compatibility_surface_implements_deterministic_helper_contracts():
    pd = pytest.importorskip("pandas")
    dates = pd.Series(pd.date_range("2026-01-01", periods=30, freq="D"))
    holidays = pd.DataFrame(
        {
            "holiday": ["launch"],
            "ds": [pd.Timestamp("2026-01-10")],
            "lower_window": [-1],
            "upper_window": [1],
            "prior_scale": [4.0],
        }
    )
    frame = pd.DataFrame({"ds": dates, "y": np.linspace(10.0, 20.0, len(dates))})
    model = Prophet(
        holidays=holidays,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        uncertainty_samples=0,
    )
    features, priors, component_cols, modes = model.make_all_seasonality_features(frame)
    assert list(features.columns) == [
        "weekly_delim_1",
        "weekly_delim_2",
        "weekly_delim_3",
        "weekly_delim_4",
        "weekly_delim_5",
        "weekly_delim_6",
        "launch_delim_+0",
        "launch_delim_+1",
        "launch_delim_-1",
    ]
    assert priors[-3:] == [4.0, 4.0, 4.0]
    assert {"additive_terms", "multiplicative_terms"}.issubset(component_cols.columns)
    assert "launch" in modes["additive"]
    assert (
        model.add_group_component(
            pd.DataFrame({"col": [0, 1], "component": ["weekly", "launch"]}),
            "all_terms",
            ["weekly", "launch"],
        ).shape[0]
        == 4
    )
    prepared = model.setup_dataframe(frame, initialize_scales=True)
    assert {"t", "y_scaled", "floor"}.issubset(prepared.columns)
    model.preprocess(frame)
    params = model.calculate_initial_params(features.shape[1])
    assert params.beta.shape == (features.shape[1],)


def test_piecewise_linear_seasonal_python_wrapper_uses_native_features():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 30,
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "fare": [50.0 + day + (20.0 if day % 5 == 0 else 0.0) for day in range(1, 31)],
            "airport_queue": [1.0 if day % 5 == 0 else 0.0 for day in range(1, 31)],
            "rush_hour": [1.0 if day % 2 == 0 else 0.0 for day in range(1, 31)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
        known_future_covariates=["airport_queue", "rush_hour"],
    )
    model = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        changepoint_range=0.8,
        changepoint_timestamps=["2026-01-15T00:00:00"],
        changepoint_l1_regularization=0.01,
        weekly_fourier_order=0,
        events=[
            {
                "name": "airport_surge",
                "timestamp": "2026-02-01T00:00:00",
                "lower_window": 0,
                "upper_window": 0,
            }
        ],
        event_l2_regularization_by_name={"airport_surge": 0.02},
        extra_regressors=["airport_queue"],
        regressor_l2_regularization_by_name={"airport_queue": 0.001},
        future_regressors={"airport_queue": [1.0, 0.0, 0.0], "rush_hour": [0.0, 1.0, 0.0]},
        custom_seasonalities=[
            {
                "name": "biweekly_pickup_cycle",
                "period_days": 14.0,
                "fourier_order": 2,
                "mode": "additive",
                "condition_name": "rush_hour",
                "l2_regularization": 0.002,
            }
        ],
        regressor_modes={"airport_queue": "additive"},
        prediction_interval_levels=[0.8],
        uncertainty_samples=64,
        trend_uncertainty_scale=0.5,
        coefficient_uncertainty_scale=1.5,
        uncertainty_seed=7,
        fit_loss="huber",
        huber_delta=1.25,
        irls_iterations=4,
    ).fit(forecast_frame)

    result = model.predict(3)
    components = model.components(3)
    samples = model.samples(3)
    restored = PiecewiseLinearSeasonalForecaster.from_json(model.to_json())
    restored_result = restored.predict(3)
    restored_components = restored.components(3)
    restored_samples = restored.samples(3)
    columns = result.columns()
    records = json.loads(result.to_json())["records"]
    result_roundtrip = type(result).from_json(result.to_json())
    result_roundtrip_payload = json.loads(result_roundtrip.to_json())
    restored_records = json.loads(restored_result.to_json())["records"]

    assert "prediction_lower_p80" in columns
    assert "prediction_upper_p80" in result_roundtrip.columns()
    _assert_json_close(result_roundtrip_payload["records"], records)
    _assert_json_close(records, restored_records)
    _assert_json_close(components, restored_components)
    _assert_json_close(samples, restored_samples)
    assert components["records"][0]["prediction"] == pytest.approx(records[0]["prediction"])
    assert samples["sample_count"] == 64
    assert len(samples["records"]) == 3 * 64
    assert {"prediction", "mean", "residual_draw", "coefficient_draw", "trend_draw"}.issubset(
        samples["records"][0]
    )
    assert any(abs(record["coefficient_draw"]) > 0.0 for record in samples["records"])
    assert any(abs(record["residual_draw"]) > 0.0 for record in samples["records"])
    assert components["records"][0]["components"]["regressors"]["airport_queue"] > 10.0
    assert "biweekly_pickup_cycle" in components["records"][0]["components"]
    assert records[0]["model"] == "piecewise_linear_seasonal"
    assert records[0]["prediction"] > records[1]["prediction"]
    assert model.metadata_["extra_regressors"] == ["airport_queue"]
    assert model.metadata_["changepoint_range"] == pytest.approx(0.8)
    assert model.metadata_["changepoint_l1_regularization"] == pytest.approx(0.01)
    assert model.metadata_["changepoint_timestamps"] == ["2026-01-15T00:00:00"]
    assert model.metadata_["custom_seasonalities"][0]["name"] == "biweekly_pickup_cycle"
    assert model.metadata_["custom_seasonalities"][0]["mode"] == "additive"
    assert model.metadata_["custom_seasonalities"][0]["condition_name"] == "rush_hour"
    assert model.metadata_["custom_seasonalities"][0]["l2_regularization"] == pytest.approx(0.002)
    assert model.metadata_["event_l2_regularization_by_name"] == {
        "airport_surge": pytest.approx(0.02)
    }
    assert model.metadata_["regressor_l2_regularization_by_name"] == {
        "airport_queue": pytest.approx(0.001)
    }
    assert model.metadata_["regressor_modes"] == {"airport_queue": "additive"}
    assert model.metadata_["uncertainty_samples"] == 64
    assert model.metadata_["trend_uncertainty_policy"] == "laplace"
    assert model.metadata_["trend_uncertainty_scale"] == pytest.approx(0.5)
    assert model.metadata_["coefficient_uncertainty_scale"] == pytest.approx(1.5)
    assert model.metadata_["uncertainty_seed"] == 7
    assert model.metadata_["fit_loss"] == "huber"
    assert model.metadata_["huber_delta"] == pytest.approx(1.25)
    assert model.metadata_["irls_iterations"] == 4
    with pytest.raises(ValueError, match="fit_loss"):
        PiecewiseLinearSeasonalForecaster(fit_loss="absolute")
    with pytest.raises(ValueError, match="trend_uncertainty_policy"):
        PiecewiseLinearSeasonalForecaster(trend_uncertainty_policy="student_t")


def test_piecewise_linear_seasonal_python_wrapper_supports_prophet_holidays_and_changepoints():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 30,
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "fare": [50.0 + day + (30.0 if day == 15 else 0.0) for day in range(1, 31)],
        }
    )
    holidays = pd.DataFrame(
        {
            "holiday": ["airport_queue_surge", "airport_queue_surge"],
            "ds": [pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-31")],
            "lower_window": [0, 0],
            "upper_window": [0, 0],
            "prior_scale": [100.0, 100.0],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
    )
    model = PiecewiseLinearSeasonalForecaster(
        changepoints=[pd.Timestamp("2026-01-10")],
        n_changepoints=4,
        changepoint_l2_regularization=0.001,
        changepoint_prior_scale=0.5,
        weekly_fourier_order=0,
        auto_weekly_seasonality=False,
        holidays=holidays,
        holidays_prior_scale=100.0,
    ).fit(forecast_frame)

    components = model.components(2)
    first_components = components["records"][0]["components"]

    assert model.metadata_["changepoints"] == 0
    assert model.metadata_["changepoint_timestamps"] == ["2026-01-10T00:00:00"]
    assert model.metadata_["changepoint_l1_regularization"] == pytest.approx(2.0)
    assert model.metadata_["event_l2_regularization"] == pytest.approx(0.0001)
    assert model.metadata_["event_l2_regularization_by_name"] == {
        "airport_queue_surge": pytest.approx(0.0001)
    }
    assert model.metadata_["events"][0]["name"] == "airport_queue_surge"
    assert first_components["events"]["airport_queue_surge"] > 10.0
    assert first_components["event_window_offsets"]["airport_queue_surge[+0]"] == pytest.approx(
        first_components["events"]["airport_queue_surge"]
    )


def test_piecewise_linear_seasonal_python_wrapper_accepts_flat_growth():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 14,
            "timestamp": pd.date_range("2026-01-01", periods=14, freq="D"),
            "fare": [40.0 + 2.0 * day for day in range(1, 15)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
    )
    model = PiecewiseLinearSeasonalForecaster(
        growth="flat",
        changepoints=0,
        weekly_fourier_order=0,
    ).fit(forecast_frame)

    assert model.metadata_["growth"] == "flat"
    assert len(json.loads(model.predict(2).to_json())["records"]) == 2


def test_piecewise_linear_seasonal_python_wrapper_exposes_regressor_standardization():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 30,
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "fare": [40.0 + day + 1.5 * (100.0 + 4.0 * day) for day in range(1, 31)],
            "traffic_index": [100.0 + 4.0 * day for day in range(1, 31)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
        known_future_covariates=["traffic_index"],
    )
    model = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=0,
        extra_regressors=["traffic_index"],
        future_regressors={"traffic_index": [224.0, 228.0]},
    ).fit(forecast_frame)
    raw_model = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=0,
        extra_regressors=["traffic_index"],
        regressor_standardization="none",
        future_regressors={"traffic_index": [224.0, 228.0]},
    ).fit(forecast_frame)

    assert model.metadata_["regressor_standardization"] == "auto"
    assert raw_model.metadata_["regressor_standardization"] == "none"
    with pytest.raises(ValueError, match="regressor_standardization"):
        PiecewiseLinearSeasonalForecaster(regressor_standardization="standardize")


def test_piecewise_linear_seasonal_python_predict_accepts_future_regressors():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 30,
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "fare": [50.0 + day + (20.0 if day % 5 == 0 else 0.0) for day in range(1, 31)],
            "airport_queue": [1.0 if day % 5 == 0 else 0.0 for day in range(1, 31)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
        known_future_covariates=["airport_queue"],
    )
    future_regressors = {"airport_queue": [1.0, 0.0, 0.0]}
    constructor_model = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=0,
        extra_regressors=["airport_queue"],
        future_regressors=future_regressors,
    ).fit(forecast_frame)
    predict_model = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=0,
        extra_regressors=["airport_queue"],
    ).fit(forecast_frame)

    constructor_records = json.loads(constructor_model.predict(3).to_json())["records"]
    predict_records = json.loads(
        predict_model.predict(3, future_regressors=future_regressors).to_json()
    )["records"]
    restored_model = PiecewiseLinearSeasonalForecaster.from_json(predict_model.to_json())
    restored_records = json.loads(
        restored_model.predict(3, future_regressors=future_regressors).to_json()
    )["records"]
    predict_components = predict_model.components(3, future_regressors=future_regressors)
    restored_components = restored_model.components(3, future_regressors=future_regressors)

    _assert_json_close(constructor_records, predict_records)
    _assert_json_close(constructor_records, restored_records)
    _assert_json_close(predict_components, restored_components)
    assert predict_components["records"][0]["components"]["regressors"]["airport_queue"] > 10.0


def test_piecewise_linear_seasonal_python_wrapper_exposes_trend_and_shock_controls():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 24,
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="D"),
            "fare": [20.0 + 0.5 * day + (12.0 if day >= 22 else 0.0) for day in range(1, 25)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
    )
    baseline = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=0,
        auto_weekly_seasonality=False,
    ).fit(forecast_frame)
    adjusted = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=0,
        auto_weekly_seasonality=False,
        trend_adjustments={2: 1.10},
        residual_shock_window=3,
        residual_shock_scale=0.8,
        residual_shock_decay=0.5,
    ).fit(forecast_frame)

    baseline_records = json.loads(baseline.predict(2).to_json())["records"]
    adjusted_records = json.loads(adjusted.predict(2).to_json())["records"]
    override_records = json.loads(baseline.predict(2, trend_adjustments={2: 1.10}).to_json())[
        "records"
    ]
    components = adjusted.components(2)

    assert adjusted_records[0]["prediction"] > baseline_records[0]["prediction"]
    assert adjusted_records[1]["prediction"] > override_records[1]["prediction"]
    assert components["records"][1]["trend_adjustment_multiplier"] == pytest.approx(1.10)
    assert components["records"][0]["residual_shock"] > components["records"][1]["residual_shock"]
    assert adjusted.metadata_["trend_adjustments"] == {"2": pytest.approx(1.10)}
    assert adjusted.metadata_["residual_shock_window"] == 3
    assert adjusted.metadata_["residual_shock_scale"] == pytest.approx(0.8)
    assert adjusted.metadata_["residual_shock_decay"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="trend_adjustments"):
        PiecewiseLinearSeasonalForecaster(trend_adjustments={0: 1.0})
    with pytest.raises(ValueError, match="residual_shock_decay"):
        PiecewiseLinearSeasonalForecaster(residual_shock_decay=1.5)


def test_piecewise_linear_seasonal_python_wrapper_exposes_builtin_seasonality_l2():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 28,
            "timestamp": pd.date_range("2026-01-01", periods=28, freq="D"),
            "fare": [60.0 + day + (12.0 if day % 7 == 0 else 0.0) for day in range(1, 29)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
    )
    model = PiecewiseLinearSeasonalForecaster(
        changepoints=0,
        weekly_fourier_order=3,
        auto_weekly_seasonality=False,
        weekly_l2_regularization=123.0,
        yearly_l2_regularization=456.0,
    ).fit(forecast_frame)

    assert model.metadata_["weekly_l2_regularization"] == pytest.approx(123.0)
    assert model.metadata_["yearly_l2_regularization"] == pytest.approx(456.0)
    assert model.metadata_["daily_l2_regularization"] is None
    assert model.metadata_["auto_weekly_seasonality"] is False


def test_piecewise_linear_seasonal_python_wrapper_uses_dynamic_logistic_cap():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    days = np.arange(1, 29, dtype=float)
    caps = 80.0 + days
    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * len(days),
            "timestamp": pd.date_range("2026-01-01", periods=len(days), freq="D"),
            "fare": caps / (1.0 + np.exp(-0.2 * (days - 14.0))),
            "zone_capacity": caps,
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
        known_future_covariates=["zone_capacity"],
    )
    future_caps = [109.0, 110.0, 111.0]
    model = PiecewiseLinearSeasonalForecaster(
        growth="logistic",
        changepoints=3,
        weekly_fourier_order=0,
        cap_regressor="zone_capacity",
        future_regressors={"zone_capacity": future_caps},
    ).fit(forecast_frame)

    records = json.loads(model.predict(3).to_json())["records"]

    assert model.metadata_["cap_regressor"] == "zone_capacity"
    assert all(
        0.0 < record["prediction"] < cap for record, cap in zip(records, future_caps, strict=True)
    )


def test_piecewise_linear_seasonal_python_wrapper_uses_dynamic_logistic_floor():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    days = np.arange(1, 29, dtype=float)
    cap = 140.0
    floors = 20.0 + days * 0.25
    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * len(days),
            "timestamp": pd.date_range("2026-01-01", periods=len(days), freq="D"),
            "fare": floors + (cap - floors) / (1.0 + np.exp(-0.2 * (days - 14.0))),
            "service_floor": floors,
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
        known_future_covariates=["service_floor"],
    )
    high_floor = PiecewiseLinearSeasonalForecaster(
        growth="logistic",
        changepoints=3,
        weekly_fourier_order=0,
        cap=cap,
        floor_regressor="service_floor",
        future_regressors={"service_floor": [32.0, 33.0, 34.0]},
    ).fit(forecast_frame)
    restored = PiecewiseLinearSeasonalForecaster.from_json(high_floor.to_json())
    low_floor_records = json.loads(
        restored.predict(3, future_regressors={"service_floor": [5.0, 6.0, 7.0]}).to_json()
    )["records"]
    high_floor_records = json.loads(high_floor.predict(3).to_json())["records"]

    assert high_floor.metadata_["floor_regressor"] == "service_floor"
    assert all(
        floor < record["prediction"] < cap
        for record, floor in zip(high_floor_records, [32.0, 33.0, 34.0], strict=True)
    )
    assert all(
        high["prediction"] > low["prediction"]
        for high, low in zip(high_floor_records, low_floor_records, strict=True)
    )


def test_piecewise_linear_seasonal_python_wrapper_supports_monotone_regressors_and_quantiles():
    pd = pytest.importorskip("pandas")
    try:
        from cartoboost import _native  # noqa: F401
    except ImportError as exc:
        pytest.skip(str(exc))

    frame = pd.DataFrame(
        {
            "series_id": ["pickup_zone_1"] * 30,
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
            "fare": [100.0 - 4.0 * (10.0 if day % 2 == 0 else 0.0) for day in range(1, 31)],
            "traffic": [10.0 if day % 2 == 0 else 0.0 for day in range(1, 31)],
        }
    )
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="timestamp",
        target_col="fare",
        series_id_col="series_id",
        freq="D",
        known_future_covariates=["traffic"],
    )
    model = PiecewiseLinearSeasonalForecaster(
        growth="flat",
        changepoints=0,
        weekly_fourier_order=0,
        regressor_l2_regularization=0.0,
        extra_regressors=["traffic"],
        extra_regressor_monotonic_constraints={"traffic": 1},
        future_regressors={"traffic": [0.0, 10.0]},
        quantile_levels=[0.25, 0.75],
        uncertainty_samples=32,
        uncertainty_seed=11,
    ).fit(forecast_frame)

    components = model.components(2)
    low = components["records"][0]["components"]["regressors"]["traffic"]
    high = components["records"][1]["components"]["regressors"]["traffic"]
    default_quantiles = model.quantiles(2)
    override_quantiles = model.quantiles(2, quantile_levels=[0.1, 0.5, 0.9])

    assert high >= low
    assert model.metadata_["extra_regressor_monotonic_constraints"] == {"traffic": 1}
    assert [row["quantile"] for row in default_quantiles["records"][:2]] == [0.25, 0.75]
    assert [row["quantile"] for row in override_quantiles["records"][:3]] == [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match="monotonic constraint"):
        PiecewiseLinearSeasonalForecaster(
            extra_regressors=["traffic"],
            extra_regressor_monotonic_constraints={"traffic": 2},
        )


def test_native_fuzzy_gaussian_serialization_preserves_predictions(tmp_path):
    x, y = _radial_fixture()
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.75,
        max_depth=1,
        min_samples_leaf=4,
        min_gain=0.0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=0.5,
    )

    try:
        model.fit(x, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    probes = np.array(
        [
            [0.0, 0.0],
            [1.45, 0.0],
            [1.55, 0.0],
            [3.0, 3.0],
        ]
    )
    before = model.predict(probes)
    model_path = tmp_path / "fuzzy_gaussian.cartoboost"
    model.save(model_path)

    restored = CartoBoostRegressor.load(model_path)

    assert restored.predict(probes) == pytest.approx(before)
    assert restored.n_features_in_ == 2


def test_real_native_save_load_restores_public_params_and_metadata(tmp_path):
    x = np.array([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [0.0, 3.0]])
    y = np.array([5.0, 5.0, -1.0, -1.0])
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.25,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=0.5,
        leaf_predictor="constant",
    )

    try:
        model.fit(x, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native-cartoboost.json"
    before = model.predict(x)
    model.save(path)

    restored = CartoBoostRegressor.load(path)

    assert restored.get_params() == {
        "n_estimators": 2,
        "learning_rate": 0.25,
        "max_depth": 1,
        "min_samples_leaf": 1,
        "min_gain": 0.0,
        "loss": "l2",
        "quantile_alpha": 0.5,
        "huber_delta": 1.0,
        "log_offset": 1.0,
        "loss_params": None,
        "splitters": ["gaussian_2d"],
        "leaf_predictor": "constant",
        "linear_leaf_features": [],
        "fuzzy": True,
        "fuzzy_bandwidth": 0.5,
        "fuzzy_kernel": "linear",
        "l2_regularization": 1.0,
        "constant_l2_regularization": 0.0,
        "random_state": None,
        "n_threads": None,
        "monotonic_constraints": None,
        "tensorboard_log_dir": None,
        "tensorboard_run_name": None,
    }
    assert restored.metadata_["library_name"] == "cartoboost-core"
    assert restored.training_config_["splitters"] == ["Gaussian2D"]
    assert restored.requires_sparse_sets_ is False
    assert restored.predict(x) == pytest.approx(before)


def test_real_native_save_weights_load_weights_restores_predictions(tmp_path):
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 5.0, 5.0])
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
    )

    try:
        model.fit(x, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native.weights.json"
    before = model.predict(x)
    model.save_weights(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    restored = CartoBoostRegressor.load_weights(path)

    assert payload["artifact_type"] == "cartoboost.weights"
    assert payload["weights_artifact_version"] == 1
    assert payload["backend"] == "rust"
    assert restored.predict(x) == pytest.approx(before)


def test_native_sparse_list_prediction_requires_sparse_sets_after_load(tmp_path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )

    try:
        model.fit(x, y, sparse_sets=sparse_sets)
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native-sparse-cartoboost.json"
    model.save(path)
    restored = CartoBoostRegressor.load(path)

    assert restored.requires_sparse_sets_ is True
    with pytest.raises(ValueError, match="sparse_sets are required"):
        restored.predict(x)
    assert restored.predict(x, sparse_sets=sparse_sets) == pytest.approx(y)


def test_native_artifact_version_mismatch_errors_clearly(tmp_path):
    path = tmp_path / "future-native-artifact.json"
    path.write_text(
        json.dumps(
            {
                "artifact_version": 999,
                "init_prediction": 0.0,
                "learning_rate": 0.1,
                "feature_count": 1,
                "target_name": None,
                "trees": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported model artifact version 999"):
        CartoBoostRegressor.load(path)


def test_python_api_rejects_unsupported_objectives():
    with pytest.raises(ValueError, match="loss"):
        CartoBoostRegressor(loss="poisson").fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="quantile_alpha"):
        CartoBoostRegressor(loss="quantile", quantile_alpha=1.0).fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="leaf_predictor"):
        CartoBoostRegressor(leaf_predictor="spline").fit([[0.0], [1.0]], [0.0, 1.0])


def test_python_api_rejects_invalid_training_arrays():
    model = CartoBoostRegressor()

    with pytest.raises(ValueError, match="same number of rows"):
        model.fit([[0.0], [1.0]], [0.0])
    with pytest.raises(ValueError, match="rectangular"):
        model.fit([[0.0], [1.0, 2.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[0.0], [float("nan")]], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[0.0], [1.0]], [0.0, float("inf")])


def test_linear_leaf_feature_indices_are_validated():
    with pytest.raises(ValueError, match="stringified integer"):
        CartoBoostRegressor(
            leaf_predictor="linear",
            linear_leaf_features=["distance"],
        ).fit([[0.0], [1.0]], [0.0, 1.0])

    with pytest.raises(ValueError, match="out of bounds"):
        CartoBoostRegressor(
            leaf_predictor="linear",
            linear_leaf_features=["2"],
        ).fit([[0.0], [1.0]], [0.0, 1.0])
