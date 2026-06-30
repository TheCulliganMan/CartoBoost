import importlib.util
import sys
import types
from datetime import date
from pathlib import Path

import pytest
from cartoboost.forecasting import (
    CrostonForecaster,
    NBeatsForecaster,
    NeuralPanelForecaster,
    NHiTSForecaster,
    SbaForecaster,
    TsbForecaster,
)
from cartoboost.forecasting.local import PiecewiseLinearSeasonalForecaster
from cartoboost.forecasting.registry import ForecastModelSpec, ForecastRegistry
from cartoboost.forecasting.schema import ForecastFrame


def test_default_registry_contains_forecasting_v1_models():
    registry = ForecastRegistry.defaults()

    assert registry.names() == (
        "naive",
        "seasonal_naive",
        "theta",
        "optimized_theta",
        "piecewise_linear_seasonal",
        "ets",
        "arima",
        "auto_arima",
        "autostats_bank",
        "croston",
        "sba",
        "tsb",
        "kalman",
        "local_level_kalman",
        "auto_kalman",
        "auto_local_level_kalman",
        "cartoboost_lag",
        "auto_forecaster",
    )


def test_deferred_model_names_are_absent_from_default_registry():
    registry = ForecastRegistry.defaults()

    deferred = {
        "local_linear_trend_kalman",
        "unobserved_components",
        "sarimax",
        "dynamic_regression",
        "mstl_ets",
        "stl_arima",
        "quantile_carto_boost_lag",
        "conformal_forecaster",
        "bottom_up_reconciler",
        "min_trace_reconciler",
        "foundation_model_adapter_optional",
    }

    assert deferred.isdisjoint(registry.names())


def test_registry_prevents_duplicate_names_unless_override():
    registry = ForecastRegistry()
    registry.register(ForecastModelSpec("taxi_demand", factory=lambda: "first"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ForecastModelSpec("taxi_demand", factory=lambda: "second"))

    registry.register(ForecastModelSpec("taxi_demand", factory=lambda: "second"), override=True)

    assert registry.create("taxi_demand") == "second"


def test_model_spec_reports_missing_optional_dependencies(monkeypatch):
    def missing(name):
        if name == "missing_backend":
            return None
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", missing)
    spec = ForecastModelSpec(
        "taxi_theta",
        factory=lambda: object(),
        optional_dependencies=("missing_backend",),
    )

    with pytest.raises(ImportError, match="missing_backend"):
        spec.create()


def test_neural_forecaster_wrappers_delegate_to_native_bindings(install_fake_native):
    native = install_fake_native("NBeatsForecaster")

    model = NBeatsForecaster(input_size=3, hidden_size=4, epochs=5, learning_rate=0.2)

    assert model.fit([1.0, 2.0, 3.0, 4.0]).predict(2) == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "input_size": 3,
            "hidden_size": 4,
            "epochs": 5,
            "learning_rate": 0.2,
            "backend": "cpu",
        },
    )
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-04T00:00:00", 4.0)


def test_nhits_forecaster_wrapper_delegates_to_native_binding(install_fake_native):
    native = install_fake_native("NHiTSForecaster")

    model = NHiTSForecaster(
        input_size=4,
        hidden_size=5,
        epochs=6,
        learning_rate=0.1,
        pooling_size=2,
    )

    model.fit({"PULocationID=237": [8.0, 9.0, 10.0, 11.0]}).predict(3)
    assert native.calls[0] == (
        "init",
        {
            "input_size": 4,
            "hidden_size": 5,
            "epochs": 6,
            "learning_rate": 0.1,
            "pooling_size": 2,
            "backend": "cpu",
        },
    )
    assert native.calls[1][1].rows[-1] == (
        "PULocationID=237",
        "1970-01-04T00:00:00",
        11.0,
    )


def test_neural_panel_forecaster_wrapper_delegates_backend_to_native_binding(
    install_fake_native,
):
    native = install_fake_native("NeuralPanelForecaster")

    model = NeuralPanelForecaster(n_lags=3, n_forecasts=2, epochs=4, backend="metal")

    model.fit({"PULocationID=237:DOLocationID=161": [8.0, 9.0, 10.0, 11.0]}).predict(2)
    assert native.calls[0][0] == "init"
    assert native.calls[0][1]["n_lags"] == 3
    assert native.calls[0][1]["n_forecasts"] == 2
    assert native.calls[0][1]["epochs"] == 4
    assert native.calls[0][1]["backend"] == "metal"


def test_intermittent_forecaster_wrappers_validate_parameters_and_delegate(install_fake_native):
    native = install_fake_native("TsbForecaster")

    model = TsbForecaster(alpha=0.3, beta=0.4)

    assert model.fit([0.0, 2.0, 0.0, 4.0]).predict(2) == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == ("init", {"alpha": 0.3, "beta": 0.4})
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-04T00:00:00", 4.0)

    with pytest.raises(ValueError, match="alpha"):
        CrostonForecaster(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        SbaForecaster(alpha=1.2)
    with pytest.raises(ValueError, match="beta"):
        TsbForecaster(beta=float("nan"))


def test_default_registry_models_are_native_wrappers(install_fake_native):
    native = install_fake_native("NaiveForecaster")
    registry = ForecastRegistry.defaults()

    naive = registry.create("naive")

    assert naive.fit([1.0, 2.0, 3.0]).predict(1) == {"args": (1,), "kwargs": {}}
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-03T00:00:00", 3.0)


def test_every_default_registry_model_constructs_and_runs(monkeypatch):
    import cartoboost
    import pandas as pd

    calls: list[tuple[str, object]] = []

    class RecordingForecastFrame:
        def __init__(self, rows, freq, **kwargs):
            self.rows = rows
            self.freq = freq
            self.kwargs = kwargs

    def recording_model_class(class_name: str):
        class RecordingModel:
            def __init__(self, **params):
                calls.append((f"{class_name}.init", params))

            def fit(self, frame):
                calls.append((f"{class_name}.fit", frame))
                return self

            def predict(self, *args, **kwargs):
                calls.append((f"{class_name}.predict", (args, kwargs)))
                return {"model": class_name, "horizon": args[0]}

            def metadata_json(self):
                return f'{{"model": "{class_name}"}}'

        return RecordingModel

    native_class_names = {
        "NaiveForecaster",
        "SeasonalNaiveForecaster",
        "ThetaForecaster",
        "OptimizedThetaForecaster",
        "PiecewiseLinearSeasonalForecaster",
        "ETSForecaster",
        "ArimaForecaster",
        "AutoARIMAForecaster",
        "AutoStatsBank",
        "CrostonForecaster",
        "SbaForecaster",
        "TsbForecaster",
        "KalmanForecaster",
        "LocalLevelKalmanForecaster",
        "AutoKalmanForecaster",
        "AutoLocalLevelKalmanForecaster",
        "CartoBoostLagForecaster",
        "AutoForecastModel",
    }
    native = type("Native", (), {"ForecastFrame": RecordingForecastFrame})()
    for class_name in native_class_names:
        setattr(native, class_name, recording_model_class(class_name))
    monkeypatch.setattr(cartoboost, "_native", native, raising=False)

    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "pickup_hour": pd.date_range("2026-01-01", periods=40, freq="D"),
                "pickup_count": [float(10 + idx) for idx in range(40)],
                "PULocationID": ["237"] * 40,
            }
        ),
        timestamp_col="pickup_hour",
        target_col="pickup_count",
        series_id_col="PULocationID",
        freq="D",
    )
    registry = ForecastRegistry.defaults()

    for name in registry.names():
        model = registry.create(name)
        model.fit(frame)
        result = model.predict(2)
        assert result["horizon"] == 2

    initialized = {name.split(".")[0] for name, _ in calls if name.endswith(".init")}
    assert native_class_names == initialized


def test_removed_default_model_name_fails_clearly():
    registry = ForecastRegistry.defaults()

    with pytest.raises(KeyError, match="sarimax"):
        registry.create("sarimax")


def test_removed_default_model_names_are_absent_from_maintained_docs():
    root = Path(__file__).resolve().parents[2]
    docs = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "docs/user-guide/model-types.md",
            "docs/user-guide/forecasting-models/index.md",
            "docs/reference/python-api.md",
            "docs/llms.txt",
        )
    )

    for removed in (
        "sarimax",
        "dynamic_regression",
        "unobserved_components",
        "mstl_ets",
        "stl_arima",
        "quantile_carto_boost_lag",
        "conformal_forecaster",
        "foundation_model_adapter_optional",
    ):
        assert removed not in docs


def test_piecewise_linear_seasonal_wrapper_normalizes_native_overrides(install_fake_native):
    native = install_fake_native("PiecewiseLinearSeasonalForecaster")
    model = PiecewiseLinearSeasonalForecaster(
        growth="flat",
        component_mode="multiplicative",
        changepoints=2,
        changepoint_range=0.9,
        changepoint_timestamps=("2026-01-08T00:00:00",),
        custom_seasonalities=[
            {
                "name": "biweekly_pickup_cycle",
                "periodDays": 14,
                "fourierOrder": 2,
                "mode": "additive",
                "conditionName": "rush_hour",
                "l2Regularization": 0.05,
            }
        ],
        events=[("airport_surge", "2026-02-01T00:00:00", -1, 1)],
        event_mode="additive",
        extra_regressors=("airport_queue",),
        regressor_modes={"airport_queue": "additive"},
        extra_regressor_monotonic_constraints={"airport_queue": 1},
        future_regressors={"airport_queue": (1, 0)},
        future_regressors_by_series={"PULocationID=132": {"airport_queue": (0, 1)}},
        trend_adjustments={"2": 1.1},
        trend_adjustments_by_series={"PULocationID=132": {"1": 1.2}},
        prediction_interval_levels=(0.8,),
        quantile_levels=(0.25, 0.75),
        uncertainty_samples=16,
        trend_uncertainty_policy="normal",
        fit_loss="huber",
        huber_delta=1.25,
        irls_iterations=4,
    ).fit({"PULocationID=132": [10.0, 12.0, 14.0]})

    model.predict(
        2,
        future_regressors={"airport_queue": [0, 1]},
        future_regressors_by_series={"PULocationID=132": {"airport_queue": [1, 0]}},
        prediction_interval_levels=[0.9],
        uncertainty_samples=8,
        trend_adjustments={1: 1.05},
        trend_adjustments_by_series={"PULocationID=132": {2: 1.15}},
    )
    model.components(
        2,
        future_regressors={"airport_queue": [1, 0]},
        trend_adjustments={2: 1.1},
    )
    model.history_components()
    history_frame = model.history_components_frame()
    model.samples(
        2,
        future_regressors_by_series={"PULocationID=132": {"airport_queue": [1, 1]}},
        uncertainty_samples=4,
    )
    model.quantiles(2, [0.1, 0.9], future_regressors={"airport_queue": [0, 0]})
    restored = PiecewiseLinearSeasonalForecaster.from_json(model.to_json())

    assert restored.is_fitted_
    assert native.calls[0] == (
        "init",
        {
            "growth": "flat",
            "component_mode": "multiplicative",
            "changepoints": 2,
            "changepoint_range": 0.9,
            "changepoint_timestamps": ["2026-01-08T00:00:00"],
            "yearly_fourier_order": 0,
            "weekly_fourier_order": 3,
            "daily_fourier_order": 0,
            "auto_yearly_seasonality": True,
            "auto_weekly_seasonality": True,
            "auto_daily_seasonality": True,
            "custom_seasonalities": [
                ("biweekly_pickup_cycle", 14.0, 2, "additive", "rush_hour", 0.05)
            ],
            "changepoint_l2_regularization": 0.05,
            "changepoint_l1_regularization": 0.0,
            "seasonality_l2_regularization": 0.01,
            "yearly_l2_regularization": None,
            "weekly_l2_regularization": None,
            "daily_l2_regularization": None,
            "event_l2_regularization": 0.01,
            "regressor_l2_regularization": 0.01,
            "event_l2_regularization_by_name": {},
            "regressor_l2_regularization_by_name": {},
            "events": [("airport_surge", "2026-02-01T00:00:00", -1, 1)],
            "event_mode": "additive",
            "extra_regressors": ["airport_queue"],
            "regressor_modes": {"airport_queue": "additive"},
            "extra_regressor_monotonic_constraints": {"airport_queue": 1},
            "regressor_standardization": "auto",
            "future_regressors": {"airport_queue": [1.0, 0.0]},
            "future_regressors_by_series": {"PULocationID=132": {"airport_queue": [0.0, 1.0]}},
            "trend_adjustments": {2: 1.1},
            "trend_adjustments_by_series": {"PULocationID=132": {1: 1.2}},
            "residual_shock_window": 0,
            "residual_shock_scale": 0.0,
            "residual_shock_decay": 1.0,
            "prediction_interval_levels": (0.8,),
            "quantile_levels": (0.25, 0.75),
            "uncertainty_samples": 16,
            "trend_uncertainty_policy": "normal",
            "trend_uncertainty_scale": 1.0,
            "coefficient_uncertainty_scale": 1.0,
            "uncertainty_seed": 14172834030107287843,
            "cap": None,
            "floor": 0.0,
            "cap_regressor": None,
            "floor_regressor": None,
            "fit_loss": "huber",
            "huber_delta": 1.25,
            "irls_iterations": 4,
        },
    )
    assert native.calls[1][1].rows[-1] == ("PULocationID=132", "1970-01-03T00:00:00", 14.0)
    assert native.calls[2] == (
        "predict",
        (
            2,
            {"airport_queue": [0.0, 1.0]},
            {"PULocationID=132": {"airport_queue": [1.0, 0.0]}},
            (0.9,),
            8,
            {1: 1.05},
            {"PULocationID=132": {2: 1.15}},
            None,
            None,
        ),
        {},
    )
    assert native.calls[3] == (
        "components_json",
        (2, {"airport_queue": [1.0, 0.0]}, None, {2: 1.1}, None),
        {},
    )
    assert native.calls[4] == (
        "history_components_json",
        (),
        {},
    )
    assert history_frame.loc[0, "trend"] == 12.0
    assert history_frame.loc[0, "components.seasonal_total"] == 1.5
    assert history_frame.loc[0, "components.weekly"] == 1.25
    assert history_frame.loc[0, "components.events.airport_surge"] == 0.25
    assert native.calls[5] == (
        "history_components_json",
        (),
        {},
    )
    assert native.calls[6] == (
        "samples_json",
        (2, None, {"PULocationID=132": {"airport_queue": [1.0, 1.0]}}, 4, None, None),
        {},
    )
    assert native.calls[7] == (
        "quantiles_json",
        (2, (0.1, 0.9), {"airport_queue": [0.0, 0.0]}, None, None, None, None),
        {},
    )
    assert native.calls[-3][0] == "to_json"
    assert native.calls[-2][0] == "from_json"
    assert native.calls[-1][0] == "init"


def test_piecewise_linear_seasonal_wrapper_accepts_prophet_holiday_and_changepoint_aliases(
    install_fake_native,
):
    native = install_fake_native("PiecewiseLinearSeasonalForecaster")
    PiecewiseLinearSeasonalForecaster(
        n_changepoints=4,
        changepoints=("2026-01-10T00:00:00", "2026-01-20T00:00:00"),
        holidays=[
            {
                "holiday": "airport_queue_surge",
                "ds": "2026-02-01T00:00:00",
                "lower_window": -1,
                "upper_window": 2,
                "prior_scale": 5.0,
            }
        ],
        holidays_prior_scale=10.0,
    ).fit({"PULocationID=132": [10.0, 12.0, 14.0]})

    init_params = native.calls[0][1]
    assert init_params["changepoints"] == 0
    assert init_params["changepoint_timestamps"] == [
        "2026-01-10T00:00:00",
        "2026-01-20T00:00:00",
    ]
    assert init_params["events"] == [("airport_queue_surge", "2026-02-01T00:00:00", -1, 2)]
    assert init_params["event_l2_regularization"] == pytest.approx(0.01)
    assert init_params["event_l2_regularization_by_name"] == {
        "airport_queue_surge": pytest.approx(0.04)
    }


def test_piecewise_linear_seasonal_predict_accepts_future_timestamps(install_fake_native):
    pd = pytest.importorskip("pandas")
    native = install_fake_native("PiecewiseLinearSeasonalForecaster")
    model = PiecewiseLinearSeasonalForecaster().fit([10.0, 12.0, 14.0])

    model.predict(2, future_timestamps=pd.bdate_range("2026-01-05", periods=2))
    model.predict(
        2,
        future_timestamps_by_series={"__single__": ["2026-01-05", "2026-01-06"]},
    )

    assert native.calls[2] == (
        "predict",
        (
            2,
            None,
            None,
            None,
            None,
            None,
            None,
            ["2026-01-05T00:00:00", "2026-01-06T00:00:00"],
            None,
        ),
        {},
    )
    assert native.calls[3] == (
        "predict",
        (
            2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {"__single__": ["2026-01-05T00:00:00", "2026-01-06T00:00:00"]},
        ),
        {},
    )


def test_piecewise_linear_seasonal_wrapper_accepts_prophet_country_holiday_and_prior_aliases(
    install_fake_native,
    monkeypatch,
):
    class FakeUSCalendar:
        def __init__(self, *, years, **_kwargs):
            self._holidays = {
                date(2026, 1, 1): ["New Year's Day"],
                date(2026, 7, 4): ["Independence Day"],
            }
            assert years == [2026]

        def __iter__(self):
            return iter(self._holidays)

        def get_list(self, holiday_date):
            return self._holidays[holiday_date]

    fake_holidays = types.SimpleNamespace(US=FakeUSCalendar)
    monkeypatch.setitem(sys.modules, "holidays", fake_holidays)
    native = install_fake_native("PiecewiseLinearSeasonalForecaster")
    PiecewiseLinearSeasonalForecaster(
        seasonality_mode="multiplicative",
        holidays_mode="additive",
        country_holidays="US",
        country_holiday_years=[2026],
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=5.0,
    ).fit({"PULocationID=132": [10.0, 12.0, 14.0]})

    init_params = native.calls[0][1]
    assert init_params["component_mode"] == "multiplicative"
    assert init_params["event_mode"] == "additive"
    assert init_params["changepoint_l1_regularization"] == pytest.approx(5.0)
    assert init_params["seasonality_l2_regularization"] == pytest.approx(0.04)
    assert ("New Year's Day", "2026-01-01T00:00:00", 0, 0) in init_params["events"]
    assert ("Independence Day", "2026-07-04T00:00:00", 0, 0) in init_params["events"]
