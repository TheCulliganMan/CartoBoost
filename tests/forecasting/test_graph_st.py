from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from cartoboost.forecasting import (
    DCRNNForecaster,
    GraphTemporalFrame,
    RollingOriginSplitter,
    available_graph_st_backends,
)


def test_graph_temporal_frame_and_dcrnn_delegate_to_native(monkeypatch):
    calls = []

    class NativeGraphTemporalFrame:
        def __init__(
            self,
            node_ids,
            timestamps,
            target,
            indptr,
            indices,
            data,
            horizon,
            frequency,
            covariates,
        ):
            calls.append(
                (
                    "frame",
                    node_ids,
                    timestamps,
                    target,
                    indptr,
                    indices,
                    data,
                    horizon,
                    frequency,
                    covariates,
                )
            )
            self.node_ids = node_ids
            self.horizon = horizon
            self.frequency = frequency

    class NativeDcrnnForecaster:
        def __init__(self, *args):
            calls.append(("init", args))

        def fit(self, frame):
            calls.append(("fit", frame))

        def predict(self, horizon):
            calls.append(("predict", horizon))
            return [[1.0, 2.0], [3.0, 4.0]][:horizon]

        def backtest(self, frame, train_size):
            calls.append(("backtest", frame, train_size))
            return json.dumps({"by_horizon": [{"horizon": 1, "mae": 0.1}]})

    import cartoboost

    monkeypatch.setattr(
        cartoboost,
        "_native",
        SimpleNamespace(
            GraphTemporalFrame=NativeGraphTemporalFrame,
            DCRNNForecaster=NativeDcrnnForecaster,
        ),
        raising=False,
    )

    frame = GraphTemporalFrame(
        node_ids=["sensor_a", "sensor_b"],
        timestamps=[0, 1, 2],
        target=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        indptr=[0, 1, 2],
        indices=[1, 0],
        data=[1.0, 1.0],
        horizon=1,
        frequency="hourly",
    )
    model = DCRNNForecaster(diffusion_steps=3, hidden_size=5, epochs=7, learning_rate=0.2)

    assert model.fit(frame).predict(2).shape == (2, 2)
    assert model.backtest(frame, train_size=2)["by_horizon"][0]["mae"] == 0.1
    assert calls[0][0] == "frame"
    assert calls[1] == ("init", (3, 5, 7, 0.2, 1.0, 0.2, 0.0001, "auto"))
    assert calls[2][0] == "fit"


def test_dcrnn_backtest_accepts_rolling_origin_splitter(monkeypatch):
    class NativeGraphTemporalFrame:
        def __init__(
            self,
            node_ids,
            timestamps,
            target,
            indptr,
            indices,
            data,
            horizon,
            frequency,
            covariates,
        ):
            self.node_ids = node_ids
            self.horizon = horizon
            self.frequency = frequency

    class NativeDcrnnForecaster:
        def __init__(self, *_args):
            pass

        def fit(self, _frame):
            return None

        def predict(self, horizon):
            return [[float(horizon), float(horizon + 1)] for _ in range(horizon)]

    import cartoboost

    monkeypatch.setattr(
        cartoboost,
        "_native",
        SimpleNamespace(
            GraphTemporalFrame=NativeGraphTemporalFrame,
            DCRNNForecaster=NativeDcrnnForecaster,
        ),
        raising=False,
    )

    frame = GraphTemporalFrame(
        node_ids=["a", "b"],
        timestamps=list(range(8)),
        target=np.arange(16, dtype=float).reshape(8, 2),
        indptr=[0, 1, 2],
        indices=[1, 0],
        data=[1.0, 1.0],
        horizon=2,
        frequency="hourly",
    )
    model = DCRNNForecaster(epochs=3).fit(frame)
    splitter = RollingOriginSplitter(horizon=2, min_train_size=4, n_splits=2)

    result = model.backtest(splitter)

    assert len(result["folds"]) == 2
    assert result["folds"][0]["by_horizon"][0]["horizon"] == 1
    assert {"mae", "rmse", "wape"}.issubset(result["folds"][0]["by_horizon"][0])


def test_dcrnn_backend_parameter_and_availability_delegate(monkeypatch):
    class NativeDcrnnForecaster:
        def __init__(self, *_args):
            self._args = _args

        def backend(self):
            return self._args[-1]

    import cartoboost

    monkeypatch.setattr(
        cartoboost,
        "_native",
        SimpleNamespace(
            DCRNNForecaster=NativeDcrnnForecaster,
            graph_st_available_backends_value=lambda: ["cpu", "cuda"],
        ),
        raising=False,
    )

    model = DCRNNForecaster(backend="cuda")

    assert model.get_params()["backend"] == "cuda"
    assert model.metadata_["backend"] == "cuda"
    assert available_graph_st_backends() == ["cpu", "cuda"]
