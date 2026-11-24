from __future__ import annotations

import builtins
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
        split_policy="structured",
        fuzzy=True,
        fuzzy_bandwidth=0.5,
    )

    try:
        model.fit(
            x, y, feature_schema={"dense": [("pickup_x", "spatial"), ("pickup_y", "spatial")]}
        )
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
        split_policy="structured",
        fuzzy=True,
        fuzzy_bandwidth=0.5,
        leaf_predictor="constant",
    )

    try:
        model.fit(
            x, y, feature_schema={"dense": [("pickup_x", "spatial"), ("pickup_y", "spatial")]}
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native-cartoboost.json"
    before = model.predict(x)
    model.save(path)

    restored = CartoBoostRegressor.load(path)

    assert restored.get_params() == {**model.get_params(), "linear_leaf_features": []}
    assert restored.metadata_["library_name"] == "cartoboost-core"
    assert restored.training_config_["splitters"] == model.training_config_["splitters"]
    assert "Gaussian2D" in restored.training_config_["splitters"]
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
    assert payload["backend"] == "cpu"
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
        split_policy="structured",
    )

    try:
        model.fit(
            x,
            y,
            sparse_sets=sparse_sets,
            feature_schema={
                "dense": [("feature_0", "numeric")],
                "sparse_sets": [("route_cells", "sparse_set")],
            },
        )
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


def test_feature_contributions_do_not_import_optional_shap(monkeypatch):
    original_import = builtins.__import__

    def reject_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise AssertionError("native feature contributions must not import shap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_shap)
    model = CartoBoostRegressor(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        split_policy="axis_only",
    ).fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 4.0, 4.0])

    values = model.predict([[0.0], [3.0]], pred_contrib=True)

    assert values.shape == (2, 2)
    assert values.sum(axis=1) == pytest.approx(model.predict([[0.0], [3.0]]))


@pytest.mark.parametrize("explicit_dates", [False, True])
@pytest.mark.parametrize("sample_count", [0, 13])
@pytest.mark.parametrize("panel", [False, True])
def test_native_piecewise_combined_outputs_and_sample_arrays_preserve_legacy_results(
    explicit_dates, sample_count, panel
):
    pd = pytest.importorskip("pandas")
    zones = ["pickup_zone_1", "pickup_zone_2"] if panel else ["pickup_zone_1"]
    records = [
        {
            "zone": zone,
            "ds": timestamp,
            "fare": 20.0 + 0.3 * day + np.sin(day * 0.7) + zone_index,
            "airport_queue": float(day % 3),
        }
        for zone_index, zone in enumerate(zones)
        for day, timestamp in enumerate(pd.date_range("2026-01-01", periods=32, freq="D"))
    ]
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(records),
        timestamp_col="ds",
        target_col="fare",
        series_id_col="zone",
        freq="D",
        known_future_covariates=["airport_queue"],
    )
    model = PiecewiseLinearSeasonalForecaster(
        changepoints=3,
        weekly_fourier_order=2,
        extra_regressors=["airport_queue"],
        future_regressors={"airport_queue": [0.0, 1.0, 2.0]},
        prediction_interval_levels=[0.8],
        uncertainty_samples=sample_count,
        uncertainty_seed=17,
    ).fit(frame)
    native = model._native_model
    artifact_before = native.to_json()
    options = {}
    if explicit_dates:
        options["future_timestamps"] = [
            "2026-02-04T00:00:00",
            "2026-02-07T00:00:00",
            "2026-02-12T00:00:00",
        ]
    expected_result = native.predict(3, **options)
    expected_components = json.loads(native.components_json(3, **options))
    result, components_json = native.predict_with_components(3, **options)
    assert result.records() == json.loads(expected_result.to_json())["records"]
    assert result.records() == expected_result.records()
    assert json.loads(components_json) == expected_components
    series_ids, timestamps, values = native.samples_arrays(3, **options)
    samples = json.loads(native.samples_json(3, **options))
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64 and values.flags.c_contiguous
    assert values.shape == (len(zones) * 3, sample_count)
    assert len(series_ids) == len(timestamps) == len(values)
    if sample_count:
        row_indices = {
            (series, timestamp): index
            for index, (series, timestamp) in enumerate(zip(series_ids, timestamps, strict=True))
        }
        expected = np.empty_like(values)
        for record in samples["records"]:
            index = row_indices[(record["series_id"], record["timestamp"])]
            expected[index, record["sample"]] = record["prediction"]
        np.testing.assert_array_equal(values, expected)
    else:
        assert samples["records"] == []
    # Prediction overrides operate on a view/clone and cannot change persisted fitted state.
    native.predict_with_components(
        3,
        future_regressors={"airport_queue": [2.0, 0.0, 1.0]},
        prediction_interval_levels=[0.9],
        uncertainty_samples=7,
        **options,
    )
    native.samples_arrays(3, uncertainty_samples=5, **options)
    assert native.to_json() == artifact_before
    assert native.predict(3, **options).records() == expected_result.records()
    restored = type(native).from_json(artifact_before)
    assert restored.predict(3, **options).records() == expected_result.records()


def test_native_piecewise_new_outputs_keep_strict_schedule_validation():
    pd = pytest.importorskip("pandas")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "ds": pd.date_range("2026-01-01", periods=16, freq="D"),
                "fare": np.arange(16, dtype=float),
            }
        ),
        timestamp_col="ds",
        target_col="fare",
        freq="D",
    )
    native = (
        PiecewiseLinearSeasonalForecaster(
            changepoints=0, weekly_fourier_order=0, uncertainty_samples=3
        )
        .fit(frame)
        ._native_model
    )
    for method in [
        native.predict_with_components,
        native.samples_arrays,
        native.components_json,
        native.samples_json,
    ]:
        with pytest.raises(ValueError):
            method(0)
        with pytest.raises(ValueError, match="length must match horizon"):
            method(2, future_timestamps=["2026-01-17"])
        with pytest.raises(ValueError, match="not both"):
            method(
                1,
                future_timestamps=["2026-01-17"],
                future_timestamps_by_series={"zone": ["2026-01-17"]},
            )
        with pytest.raises(ValueError, match="strictly after"):
            method(1, future_timestamps=["2026-01-01"])


class _ProphetFacadeResult:
    def __init__(self, records):
        self._records = records

    def records(self):
        return self._records

    def to_json(self):
        raise AssertionError("forecast records should not roundtrip through JSON")


class _ProphetFacadeNative:
    def __init__(self, dates, shift=0.0):
        self._native_model = object()
        self.dates = dates
        self.shift = shift
        self.calls = {"history": 0, "forecast": 0, "components": 0, "samples": 0}
        self.schedule = []

    def _records(self, dates):
        return [
            {
                "timestamp": value.isoformat(),
                "prediction": float(value.day) + self.shift,
                "prediction_lower_p80": float(value.day) - 1.0 + self.shift,
                "prediction_upper_p80": float(value.day) + 1.0 + self.shift,
                "components": {"trend_linear": float(value.day), "non_trend_total": self.shift},
            }
            for value in dates
        ]

    def history_components(self):
        self.calls["history"] += 1
        return {"records": self._records(self.dates)}

    def predict_with_components(self, horizon, **kwargs):
        self.calls["forecast"] += 1
        self.schedule = kwargs["future_timestamps"]
        records = self._records(self.schedule)
        return _ProphetFacadeResult(records), {"records": records}

    def components(self, horizon, **kwargs):
        self.calls["components"] += 1
        self.schedule = kwargs["future_timestamps"]
        return {"records": self._records(self.schedule)}

    def samples_arrays(self, horizon, **kwargs):
        self.calls["samples"] += 1
        dates = kwargs["future_timestamps"]
        count = kwargs["uncertainty_samples"]
        values = np.array([value.day for value in dates], dtype=float)[:, None]
        return (
            ["series"] * horizon,
            [value.isoformat() for value in dates],
            values + np.arange(count),
        )


def _prophet_facade_fixture():
    pd = pytest.importorskip("pandas")
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    model = Prophet(uncertainty_samples=4)
    model.history = pd.DataFrame({"ds": dates, "y": np.arange(5, dtype=float)})
    model._model = _ProphetFacadeNative(dates)
    return model


def test_prophet_history_cache_is_reused_and_invalidated_by_model_replacement():
    model = _prophet_facade_fixture()
    frame = model.history.iloc[[3, 0, 3]]
    expected = model.predict(frame)
    actual = model.predict(frame)
    assert model._model.calls["history"] == 1
    np.testing.assert_array_equal(actual["yhat"], expected["yhat"])
    model._model = _ProphetFacadeNative(model.history["ds"], shift=10.0)
    changed = model.predict(frame)
    np.testing.assert_array_equal(changed["yhat"], expected["yhat"] + 10.0)
    assert model._model.calls["history"] == 1


def test_prophet_future_schedule_preserves_reordered_duplicate_rows_and_arbitrary_index():
    pd = pytest.importorskip("pandas")
    model = _prophet_facade_fixture()
    frame = pd.DataFrame(
        {"ds": pd.to_datetime(["2026-01-09", "2026-01-02", "2026-01-07", "2026-01-09"])},
        index=[90, 2, 2, -5],
    )
    forecast = model.predict(frame)
    assert forecast["ds"].tolist() == frame["ds"].tolist()
    np.testing.assert_array_equal(forecast["yhat"], [9.0, 2.0, 7.0, 9.0])
    assert model._model.schedule == pd.to_datetime(["2026-01-07", "2026-01-09"]).tolist()
    assert model._model.calls["forecast"] == 1
    assert model._model.calls["components"] == 0
    assert list(frame.index) == [90, 2, 2, -5]


def test_prophet_samples_scatter_by_position_without_repredicting_uncertainty():
    pd = pytest.importorskip("pandas")
    model = _prophet_facade_fixture()
    frame = pd.DataFrame(
        {"ds": pd.to_datetime(["2026-01-09", "2026-01-02", "2026-01-07", "2026-01-09"])},
        index=[90, 2, 2, -5],
    )
    samples = model.predictive_samples(frame)
    np.testing.assert_array_equal(
        samples["yhat"], [[9, 10, 11, 12], [2, 2, 2, 2], [7, 8, 9, 10], [9, 10, 11, 12]]
    )
    np.testing.assert_array_equal(samples["trend"], [[9] * 4, [2] * 4, [7] * 4, [9] * 4])
    assert model._model.calls == {"history": 1, "forecast": 0, "components": 1, "samples": 1}
    model.predict_trend(frame)
    model.predict_seasonal_components(frame)
    assert model._model.calls["forecast"] == 0
    assert model._model.calls["samples"] == 1


def test_prophet_duplicate_future_dates_reject_conflicting_regressors():
    pd = pytest.importorskip("pandas")
    model = _prophet_facade_fixture()
    model.extra_regressors = {"promo": {}}
    frame = pd.DataFrame({"ds": pd.to_datetime(["2026-01-07"] * 2), "promo": [0.0, 1.0]})
    with pytest.raises(ValueError, match="identical regressor values"):
        model.predict(frame)


def test_prophet_column_assembly_preserves_component_order_and_missing_values():
    pd = pytest.importorskip("pandas")
    from cartoboost.prophet import _prophet_result_frame

    frame = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3)})
    result = [
        {"prediction": value, "prediction_lower_p80": value - 1.0} for value in [1.0, 2.0, 3.0]
    ]
    components = [
        {"components": {"trend_linear": 1.0, "regressors": {"promo": 2.0}}},
        {
            "components": {
                "trend_linear": 2.0,
                "non_trend_total": 1.0,
                "regressors": {"holiday": 3.0},
            }
        },
        {"components": {"trend_linear": 3.0, "regressors": {"promo": 4.0, "trend": 5.0}}},
    ]
    actual = _prophet_result_frame(frame, result, components, 0.8)
    assert list(actual.columns) == [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "trend",
        "additive_terms",
        "promo",
        "holiday",
    ]
    np.testing.assert_array_equal(actual["trend"], [1.0, 2.0, 5.0])
    np.testing.assert_array_equal(actual["additive_terms"], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(actual["promo"], [2.0, np.nan, 4.0])
    np.testing.assert_array_equal(actual["holiday"], [np.nan, 3.0, np.nan])


def test_prophet_holiday_features_match_calendar_days_and_repeated_occurrences():
    pd = pytest.importorskip("pandas")
    dates = pd.to_datetime(
        ["2026-01-01", "2026-01-02 06:00", "2026-01-02 18:00", "2026-01-03", "2027-01-02"],
        format="mixed",
    )
    holidays = pd.DataFrame(
        {
            "holiday": ["launch", "launch"],
            "ds": pd.to_datetime(["2026-01-02", "2027-01-02"]),
            "lower_window": [-1, -1],
            "upper_window": [1, 1],
        }
    )
    features, _, names = Prophet().make_holiday_features(dates, holidays)
    np.testing.assert_array_equal(features["launch_delim_+0"], [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(features["launch_delim_-1"], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(features["launch_delim_+1"], [0, 0, 0, 1, 0])
    assert names == ["launch"]


def test_prophet_zero_period_future_frame_is_independent_and_keeps_datetime_dtype():
    pd = pytest.importorskip("pandas")
    model = _prophet_facade_fixture()
    empty = model.make_future_dataframe(0, include_history=False)
    assert empty.shape == (0, 1)
    assert empty["ds"].dtype == model.history["ds"].dtype
    copied = model.make_future_dataframe(0)
    original = model.history["ds"].copy()
    copied.loc[0, "ds"] = pd.Timestamp("2030-01-01")
    pd.testing.assert_series_equal(model.history["ds"], original)


def test_prophet_native_irregular_forecast_and_samples_keep_requested_row_order():
    pd = pytest.importorskip("pandas")
    dates = pd.date_range("2026-01-01 00:00:00.125", periods=32, freq="D")
    model = Prophet(n_changepoints=2, uncertainty_samples=8, yearly_seasonality=False).fit(
        pd.DataFrame({"ds": dates, "y": np.arange(32, dtype=float) + np.sin(np.arange(32))})
    )
    future_dates = [dates[-1] + pd.Timedelta(days=2), dates[-1] + pd.Timedelta(days=5)]
    request = pd.DataFrame(
        {"ds": [future_dates[1], dates[3], future_dates[0], future_dates[1]]}, index=[40, -2, 40, 7]
    )
    forecast = model.predict(request)
    reference = model._model.predict(2, future_timestamps=future_dates).records()
    np.testing.assert_array_equal(
        forecast["yhat"].iloc[[0, 2, 3]],
        [reference[1]["prediction"], reference[0]["prediction"], reference[1]["prediction"]],
    )
    assert forecast["ds"].tolist() == request["ds"].tolist()
    samples = model.predictive_samples(request)
    _, _, expected = model._model.samples_arrays(2, future_timestamps=future_dates)
    np.testing.assert_array_equal(samples["yhat"][[0, 2, 3]], np.asarray(expected)[[1, 0, 1]])
    np.testing.assert_array_equal(samples["yhat"][1], forecast["yhat"].iloc[1])
    cached = model.predict(model.history.iloc[[0]])["yhat"].to_numpy()
    model.history.loc[:, "y"] = 999.0
    np.testing.assert_array_equal(model.predict(model.history.iloc[[0]])["yhat"], cached)


def test_prophet_future_only_samples_reuse_native_matrix(monkeypatch):
    pd = pytest.importorskip("pandas")
    model = _prophet_facade_fixture()
    original = model._model.samples_arrays
    captured = {}

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["values"] = result[2]
        return result

    monkeypatch.setattr(model._model, "samples_arrays", capture)
    request = pd.DataFrame({"ds": pd.to_datetime(["2026-01-07", "2026-01-09"])})
    result = model.predictive_samples(request)
    assert np.shares_memory(result["yhat"], captured["values"])
    np.testing.assert_array_equal(result["yhat"], [[7, 8, 9, 10], [9, 10, 11, 12]])


def _prophet_direct_reference(frame, *, growth="linear", covariates=(), **kwargs):
    forecast_frame = ForecastFrame.from_pandas(
        frame,
        timestamp_col="ds",
        target_col="y",
        freq="D",
        known_future_covariates=list(covariates),
        allow_irregular=True,
        allow_missing_targets=True,
    )
    return PiecewiseLinearSeasonalForecaster(
        growth=growth,
        changepoints=0,
        changepoint_range=0.8,
        yearly_fourier_order=0,
        weekly_fourier_order=0,
        daily_fourier_order=0,
        auto_yearly_seasonality=False,
        auto_weekly_seasonality=False,
        auto_daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_mode="additive",
        prediction_interval_levels=[0.8],
        uncertainty_samples=0,
        **kwargs,
    ).fit(forecast_frame)


def _small_prophet(**kwargs):
    return Prophet(
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
        **kwargs,
    )


@pytest.mark.parametrize("use_floor", [False, True])
def test_prophet_logistic_bounds_match_direct_piecewise_configuration(use_floor):
    pd = pytest.importorskip("pandas")
    t = np.arange(24, dtype=float)
    cap = 80.0 + t
    floor = 5.0 + 0.1 * t if use_floor else np.zeros(len(t))
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=len(t)),
            "cap": cap,
            "y": floor + (cap - floor) / (1.0 + np.exp(-0.2 * (t - 12.0))),
        }
    )
    covariates = ["cap"]
    if use_floor:
        frame["floor"] = floor
        covariates.append("floor")
    model = _small_prophet(growth="logistic").fit(frame)
    direct = _prophet_direct_reference(
        frame,
        growth="logistic",
        covariates=covariates,
        cap_regressor="cap",
        floor_regressor="floor" if use_floor else None,
    )
    future = model.make_future_dataframe(3, include_history=False)
    future["cap"] = [104.0, 105.0, 106.0]
    if use_floor:
        future["floor"] = [7.4, 7.5, 7.6]
    forecast = model.predict(future)
    prediction = forecast["yhat"].to_numpy()
    np.testing.assert_array_equal(forecast["cap"], future["cap"])
    if use_floor:
        np.testing.assert_array_equal(forecast["floor"], future["floor"])
    expected = direct.predict(
        3, future_regressors={name: future[name].tolist() for name in covariates}
    ).records()
    np.testing.assert_allclose(
        prediction, [row["prediction"] for row in expected], rtol=0.0, atol=1e-12
    )
    expected_trend = np.asarray(
        [
            row["trend"]
            for row in direct.components(
                3, future_regressors={name: future[name].tolist() for name in covariates}
            )["records"]
        ]
    )
    np.testing.assert_array_equal(forecast["trend"], expected_trend)
    np.testing.assert_array_equal(model.predict_trend(future), expected_trend)
    np.testing.assert_array_equal(
        model.predictive_samples(future)["trend"], expected_trend[:, None]
    )
    assert np.all(prediction < future["cap"].to_numpy())
    assert np.all(prediction > (future["floor"].to_numpy() if use_floor else 0.0))
    assert model.logistic_floor == use_floor
    np.testing.assert_array_equal(model.history["floor"], floor)
    prepared = model.setup_dataframe(future)
    np.testing.assert_array_equal(prepared["floor"], future["floor"] if use_floor else np.zeros(3))
    with pytest.raises(ValueError, match="cap"):
        model.predict(future.drop(columns="cap"))


def test_prophet_logistic_forecast_plot_preserves_capacity_and_floor_traces():
    pd = pytest.importorskip("pandas")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    t = np.arange(16, dtype=float)
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=len(t)),
            "cap": 80.0 + t,
            "floor": 5.0 + 0.1 * t,
            "y": 5.0 + 0.1 * t + (75.0 + 0.9 * t) / (1.0 + np.exp(-0.2 * (t - 8.0))),
        }
    )
    model = _small_prophet(growth="logistic").fit(frame)
    future = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-19", "2026-01-17"]),
            "cap": [98.0, 96.0],
            "floor": [6.8, 6.6],
        },
        index=[4, -2],
    )
    forecast = model.predict(future)
    figure = model.plot(forecast, uncertainty=False)
    try:
        traces = {line.get_label(): line for line in figure.axes[0].lines}
        for name, label in (("cap", "Maximum capacity"), ("floor", "Minimum capacity")):
            np.testing.assert_array_equal(forecast[name], future[name])
            np.testing.assert_array_equal(traces[label].get_ydata(), future[name])
            np.testing.assert_array_equal(traces[label].get_xdata(), future["ds"].to_numpy())
    finally:
        pyplot.close(figure)


def test_prophet_regressor_prior_scales_match_direct_native_penalties():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(734)
    x = rng.uniform(0.0, 100.0, 32)
    binary = np.arange(32) % 2
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=32),
            "y": 4.0 + 0.3 * x + binary,
            "x": x,
            "binary": binary,
        }
    )
    model = (
        _small_prophet()
        .add_regressor("x", prior_scale=2.0)
        .add_regressor("binary", prior_scale=5.0)
        .fit(frame)
    )
    direct = _prophet_direct_reference(
        frame,
        covariates=["x", "binary"],
        extra_regressors=["x", "binary"],
        regressor_l2_regularization_by_name={"x": 0.25, "binary": 0.04},
    )
    future = model.make_future_dataframe(3, include_history=False)
    future["x"], future["binary"] = [10.0, 40.0, 90.0], [0.0, 1.0, 0.0]
    expected = direct.predict(
        3, future_regressors={name: future[name].tolist() for name in ["x", "binary"]}
    ).records()
    np.testing.assert_allclose(
        model.predict(future)["yhat"], [row["prediction"] for row in expected], rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("prior_scale", [0.0, -1.0, np.nan, np.inf, 1e-300])
def test_prophet_rejects_invalid_regressor_prior_scales(prior_scale):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=4),
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="prior_scale"):
        _small_prophet().add_regressor("x", prior_scale=prior_scale).fit(frame)


def test_prophet_mixed_regressor_standardization_matches_native_overrides():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1734)
    count = 24
    raw, scaled = rng.uniform(0.0, 100.0, count), rng.uniform(1.0, 5.0, count)
    binary_auto, binary_forced = np.arange(count) % 2, (np.arange(count) // 2) % 2
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=count),
            "y": 5.0 + 0.1 * raw + scaled + 2.0 * binary_auto + 4.0 * binary_forced,
            "raw": raw,
            "scaled": scaled,
            "binary_auto": binary_auto,
            "binary_forced": binary_forced,
        }
    )
    names = ["raw", "scaled", "binary_auto", "binary_forced"]
    model = (
        _small_prophet()
        .add_regressor("raw", standardize=False)
        .add_regressor("scaled", standardize=True)
        .add_regressor("binary_auto", standardize="auto")
        .add_regressor("binary_forced", standardize=True)
        .fit(frame)
    )
    overrides = {"raw": "none", "scaled": "standardize", "binary_forced": "standardize"}
    direct = _prophet_direct_reference(
        frame,
        covariates=names,
        extra_regressors=names,
        regressor_standardization_by_name=overrides,
        regressor_l2_regularization_by_name={name: 0.01 for name in names},
    )
    native_payload = json.loads(model._model.to_json())["model"]
    assert native_payload["regressor_standardization_by_name"] == overrides
    stats = next(iter(native_payload["fitted"]["series"].values()))["regressor_stats"]
    assert not stats["raw"]["standardized"]
    assert not stats["binary_auto"]["standardized"]
    assert stats["scaled"]["standardized"]
    assert stats["binary_forced"]["standardized"]
    future = model.make_future_dataframe(2, include_history=False)
    future["raw"], future["scaled"] = [10.0, 90.0], [2.0, 4.0]
    future["binary_auto"], future["binary_forced"] = [0.0, 1.0], [1.0, 0.0]
    values = {name: future[name].tolist() for name in names}
    expected = direct.predict(2, future_regressors=values).records()
    np.testing.assert_allclose(
        model.predict(future)["yhat"], [row["prediction"] for row in expected], rtol=0.0, atol=1e-12
    )


def test_native_forecast_timestamp_accessors_keep_nanosecond_precision():
    from cartoboost import _native

    timestamps = ["2026-01-01T00:00:00.123456789", "2026-01-02T00:00:00.123456789"]
    rows = [("pickup_zone_132", stamp, 12.0 + index) for index, stamp in enumerate(timestamps)]
    frame = _native.ForecastFrame(rows, "D")
    assert frame.rows() == rows
    predictions = [
        (zone, timestamp, index + 1, "taxi_fare", value)
        for index, (zone, timestamp, value) in enumerate(rows)
    ]
    result = _native.ForecastResult(predictions)
    assert result.predictions() == predictions
    assert [row["timestamp"] for row in result.records()] == timestamps
    restored = _native.ForecastResult.from_json(result.to_json())
    assert restored.predictions() == predictions
    assert restored.records() == result.records()


def test_native_piecewise_per_regressor_standardization_bindings_roundtrip():
    from cartoboost import _native

    options = {"airport_queue": "none", "pickup_indicator": "standardize"}
    model = _native.PiecewiseLinearSeasonalForecaster(
        extra_regressors=list(options), regressor_standardization_by_name=options
    )
    assert model.regressor_standardization_by_name() == options
    restored = _native.PiecewiseLinearSeasonalForecaster.from_json(model.to_json())
    assert restored.regressor_standardization_by_name() == options
    clone = restored.unfitted_clone()
    assert clone.regressor_standardization_by_name() == options
    clone.set_regressor_standardization_by_name({"airport_queue": "auto"})
    assert clone.regressor_standardization_by_name() == {"airport_queue": "auto"}
    assert restored.regressor_standardization_by_name() == options
    with pytest.raises(ValueError, match="standardization"):
        clone.set_regressor_standardization_by_name({"airport_queue": "invalid"})
    assert clone.regressor_standardization_by_name() == {"airport_queue": "auto"}
