from __future__ import annotations

from cartoboost.accelerators import available_backends
from cartoboost.forecasting import NBeatsForecaster, NHiTSForecaster


def test_neural_forecasters_report_every_selected_backend() -> None:
    values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for backend in available_backends("tanh_mlp_training"):
        nbeats = NBeatsForecaster(
            input_size=3,
            hidden_size=2,
            epochs=1,
            learning_rate=0.01,
            backend=backend,
        ).fit(values)
        nhits = NHiTSForecaster(
            input_size=4,
            hidden_size=2,
            epochs=1,
            learning_rate=0.01,
            pooling_size=2,
            backend=backend,
        ).fit(values)

        assert nbeats.selected_backend_ == backend
        assert nhits.selected_backend_ == backend
        assert nbeats.get_metadata()["backend"]["selected"] == backend
        assert nhits.get_metadata()["backend"]["selected"] == backend
