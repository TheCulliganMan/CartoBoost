from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from cartoboost.forecasting import LaneNeuralPanelForecaster, NeuralPanelForecaster


def test_neural_panel_wrapper_uses_native_forecaster_syntax(install_fake_native):
    native = install_fake_native("NeuralPanelForecaster")

    model = NeuralPanelForecaster(
        n_lags=24,
        n_forecasts=6,
        quantiles=[0.1, 0.5, 0.9],
        daily_fourier_order=3,
        future_regressors={"is_airport_event": "additive"},
        lagged_regressors={"avg_trip_distance": 24},
        ar_layers=[16],
        trend_mode="glocal",
        local_l2=0.1,
        seed=42,
    )
    model.fit({"PU1:DO2": [1.0, 2.0, 3.0, 4.0]})

    init_name, params = native.calls[0]
    assert init_name == "init"
    assert params["n_lags"] == 24
    assert params["n_forecasts"] == 6
    assert params["quantiles"] == [0.1, 0.5, 0.9]
    assert params["future_regressors"] == {"is_airport_event": "additive"}
    assert params["lagged_regressors"] == {"avg_trip_distance": 24}
    assert params["trend_mode"] == "glocal"
    assert native.calls[1][0] == "fit"


def test_neural_panel_wrapper_normalizes_conditional_custom_seasonalities(
    install_fake_native,
):
    native = install_fake_native("NeuralPanelForecaster")

    model = NeuralPanelForecaster(
        n_lags=12,
        n_forecasts=2,
        custom_seasonalities=[
            ("taxi_cycle", 24.0, 2),
            ("rush_hour_cycle", 12.0, 1, "rush_hour"),
        ],
    )
    model.fit({"PU1:DO2": [1.0, 2.0, 3.0, 4.0]})

    _init_name, params = native.calls[0]
    assert params["custom_seasonalities"] == [
        ("taxi_cycle", 24.0, 2, None),
        ("rush_hour_cycle", 12.0, 1, "rush_hour"),
    ]


def test_neural_panel_builder_methods_flow_into_native_params(
    install_fake_native,
    monkeypatch,
):
    class FakeUSCalendar:
        def __init__(self, *, years, **_kwargs):
            self._holidays = {date(2026, 1, 1): ["New Year's Day"]}
            assert years == [2026]

        def __iter__(self):
            return iter(self._holidays)

        def get_list(self, holiday_date):
            return self._holidays[holiday_date]

    monkeypatch.setitem(sys.modules, "holidays", SimpleNamespace(US=FakeUSCalendar))
    native = install_fake_native("NeuralPanelForecaster")

    model = (
        NeuralPanelForecaster(n_lags=2, n_forecasts=1)
        .add_seasonality("taxi_cycle", 24.0, 2)
        .add_seasonality("rush_hour_cycle", 12.0, 1, condition_name="rush_hour")
        .add_future_regressor("promo")
        .add_lagged_regressor("avg_trip_distance", 24)
        .add_events("airport_surge", -1, 2)
        .add_country_holidays("US", years=[2026])
    )
    model._new_native_model()

    init_params = native.calls[0][1]
    assert init_params["future_regressors"]["promo"] == "additive"
    assert init_params["future_regressors"]["New Year's Day"] == "additive"
    assert init_params["lagged_regressors"] == {"avg_trip_distance": 24}
    assert init_params["events"]["airport_surge"] == [-1, 0, 1, 2]
    assert init_params["custom_seasonalities"] == [
        ("taxi_cycle", 24.0, 2, None),
        ("rush_hour_cycle", 12.0, 1, "rush_hour"),
    ]


def test_lane_neural_panel_wrapper_passes_embedding_dim(install_fake_native):
    native = install_fake_native("LaneNeuralPanelForecaster")

    model = LaneNeuralPanelForecaster(n_lags=12, n_forecasts=3, embedding_dim=4)
    model.fit({"PU1:DO2": [1.0, 2.0, 3.0, 4.0]})

    _init_name, params = native.calls[0]
    assert params["n_lags"] == 12
    assert params["n_forecasts"] == 3
    assert params["embedding_dim"] == 4


def test_lane_neural_panel_predict_for_lanes_delegates_lane_ids(install_fake_native):
    native = install_fake_native("LaneNeuralPanelForecaster")

    model = LaneNeuralPanelForecaster(n_lags=2, n_forecasts=2)
    model.fit({"A:B": [1.0, 2.0, 3.0, 4.0]})
    result = model.predict_for_lanes(2, ["A:B", "A:C"])

    assert result["args"] == (2, ["A:B", "A:C"])
    assert native.calls[-1][0] == "predict_for_lanes"


def test_neural_panel_predict_with_known_future_delegates_frame(install_fake_native):
    native = install_fake_native("NeuralPanelForecaster")
    future = native.frame_class(
        [("A:B", "1970-01-05T00:00:00", 0.0)],
        "D",
        row_covariates=[{"promo": 1.0}],
    )

    model = NeuralPanelForecaster(
        n_lags=2,
        n_forecasts=1,
        future_regressors={"promo": "additive"},
    )
    model.fit({"A:B": [1.0, 2.0, 3.0, 4.0]})
    result = model.predict(1, known_future=SimpleNamespace(_native_frame=future))

    assert result["args"] == (1, future)
    assert native.calls[-1][0] == "predict_with_known_future"


def test_neural_panel_components_json_delegates_frame(install_fake_native):
    native = install_fake_native("NeuralPanelForecaster")
    future = native.frame_class(
        [("A:B", "1970-01-05T00:00:00", 0.0)],
        "D",
        row_covariates=[{"promo": 1.0}],
    )

    model = NeuralPanelForecaster(n_lags=2, n_forecasts=1, future_regressors={"promo": "additive"})
    model.fit({"A:B": [1.0, 2.0, 3.0, 4.0]})
    result = model.components_json(1, known_future=SimpleNamespace(_native_frame=future))
    payload = json.loads(result)

    assert native.calls[-1][0] == "components_json"
    assert payload["args"][0] == 1
    assert payload["kwargs"] == {}


def test_neural_panel_components_dict_and_history_frame(install_fake_native):
    native = install_fake_native("NeuralPanelForecaster")

    model = NeuralPanelForecaster(n_lags=2, n_forecasts=1)
    model.fit({"A:B": [1.0, 2.0, 3.0, 4.0]})

    components = model.components(1)
    history = model.history_components()
    history_frame = model.history_components_frame()

    assert components["args"] == [1]
    assert history["model"] == "NeuralPanelForecaster"
    assert not history_frame.empty
    assert "feature_contributions.weekly" in history_frame.columns
    assert native.calls[-1][0] == "history_components_json"


def test_neural_panel_history_components_json_delegates(install_fake_native):
    native = install_fake_native("NeuralPanelForecaster")

    model = NeuralPanelForecaster(n_lags=2, n_forecasts=1)
    model.fit({"A:B": [1.0, 2.0, 3.0, 4.0]})
    result = json.loads(model.history_components_json())

    assert native.calls[-1][0] == "history_components_json"
    assert result["model"] == "NeuralPanelForecaster"
    assert result["records"]


def test_neural_panel_benchmark_split_suite_records_required_artifact(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "neural_panel_split_suite.json"

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "forecasting_library_benchmark.py"),
            "--source",
            "polars",
            "--model-roster",
            "neural-panel",
            "--neural-panel-splits",
            "--lanes",
            "8",
            "--days",
            "64",
            "--horizon",
            "3",
            "--suite-folds",
            "1",
            "--cartoboost-n-estimators",
            "5",
            "--cartoboost-max-depth",
            "2",
            "--cartoboost-min-samples-leaf",
            "2",
            "--output",
            str(output),
        ],
        cwd=repo,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(repo / "python"), os.environ.get("PYTHONPATH", "")]),
        },
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "neural_panel_taxi_lane_split_suite"
    assert payload["invocation"]["command"]
    assert payload["artifact_paths"]["json"] == str(output)
    assert set(payload["splits"]) == {
        "rolling_origin",
        "cold_lane",
        "cold_origin",
        "sparse_tail",
    }
    for split in payload["splits"].values():
        assert split["metrics"]["cartoboost_neural_panel"]["rmse"] >= 0.0
        assert split["metrics"]["cartoboost_lag"]["rmse"] >= 0.0
        assert split["metrics"]["seasonal_naive"]["rmse"] >= 0.0
    assert "splits" in payload["timing"]
