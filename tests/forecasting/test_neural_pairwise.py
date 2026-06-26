from __future__ import annotations

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
