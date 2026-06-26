from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cartoboost.forecasting import LaneNeuralPairwiseForecaster, NeuralPairwiseForecaster


def test_neural_pairwise_wrapper_uses_native_forecaster_syntax(install_fake_native):
    native = install_fake_native("NeuralPairwiseForecaster")

    model = NeuralPairwiseForecaster(
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


def test_lane_neural_pairwise_wrapper_passes_embedding_dim(install_fake_native):
    native = install_fake_native("LaneNeuralPairwiseForecaster")

    model = LaneNeuralPairwiseForecaster(n_lags=12, n_forecasts=3, embedding_dim=4)
    model.fit({"PU1:DO2": [1.0, 2.0, 3.0, 4.0]})

    _init_name, params = native.calls[0]
    assert params["n_lags"] == 12
    assert params["n_forecasts"] == 3
    assert params["embedding_dim"] == 4


def test_lane_neural_pairwise_predict_for_lanes_delegates_lane_ids(install_fake_native):
    native = install_fake_native("LaneNeuralPairwiseForecaster")

    model = LaneNeuralPairwiseForecaster(n_lags=2, n_forecasts=2)
    model.fit({"A:B": [1.0, 2.0, 3.0, 4.0]})
    result = model.predict_for_lanes(2, ["A:B", "A:C"])

    assert result["args"] == (2, ["A:B", "A:C"])
    assert native.calls[-1][0] == "predict_for_lanes"


def test_neural_pairwise_benchmark_split_suite_records_required_artifact(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "neural_pairwise_split_suite.json"

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "forecasting_library_benchmark.py"),
            "--source",
            "polars",
            "--model-roster",
            "neural-pairwise",
            "--neural-pairwise-splits",
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
    assert payload["benchmark"] == "neural_pairwise_taxi_lane_split_suite"
    assert payload["invocation"]["command"]
    assert payload["artifact_paths"]["json"] == str(output)
    assert set(payload["splits"]) == {
        "rolling_origin",
        "cold_lane",
        "cold_origin",
        "sparse_tail",
    }
    for split in payload["splits"].values():
        assert split["metrics"]["cartoboost_neural_pairwise"]["rmse"] >= 0.0
        assert split["metrics"]["cartoboost_lag"]["rmse"] >= 0.0
        assert split["metrics"]["seasonal_naive"]["rmse"] >= 0.0
    assert "splits" in payload["timing"]
