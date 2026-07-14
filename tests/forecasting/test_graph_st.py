from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from cartoboost.forecasting import (
    DCRNNForecaster,
    GraphTemporalFrame,
    GraphWaveNetForecaster,
    LSTTNForecaster,
    MarketPanelFrame,
    MarketStructureForecaster,
    RollingOriginSplitter,
    SpatialShiftGraphonMoEForecaster,
    SpatialTemporalGraphGatedTransformerForecaster,
    STAEformerForecaster,
    STGformerForecaster,
    STGormerForecaster,
    available_graph_st_backends,
)


def test_market_structure_wrappers_keep_targets_generic(monkeypatch):
    calls = []

    class NativeMarketPanelFrame:
        def __init__(self, *args):
            calls.append(("frame", args))
            self.lane_ids = args[0]
            self.target_names = args[2]

    class NativeMarketStructureForecaster:
        def __init__(self, **params):
            calls.append(("init", params))

        def fit(self, frame):
            calls.append(("fit", frame))

        def predict_json(self, horizon, calendar):
            calls.append(("predict", horizon, calendar))
            return json.dumps([{"primary": 1.0, "secondary": 2.0}])

        def nowcast_json(self):
            return json.dumps([{"shift": "no_shift"}])

        def weekly_rollups_json(self, horizon, calendar):
            calls.append(("weekly_rollups", horizon, calendar))
            return json.dumps([{"days": 2, "primary": 1.1, "secondary": 4.0}])

        def relationships_json(self):
            return json.dumps([])

        def explorer_json(self, horizon):
            calls.append(("explorer", horizon))
            return json.dumps({"lanes": [], "forecasts": [], "explanations": [], "kernels": []})

    import cartoboost

    monkeypatch.setattr(
        cartoboost,
        "_native",
        SimpleNamespace(
            MarketPanelFrame=NativeMarketPanelFrame,
            MarketStructureForecaster=NativeMarketStructureForecaster,
        ),
        raising=False,
    )
    frame = MarketPanelFrame(
        lane_ids=["a:b", "a:c"],
        timestamps=[0, 1, 2],
        target_names=["benchmark", "supporting_measure"],
        primary=[[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        secondary=[[3.0, 4.0], [3.1, 4.1], [3.2, 4.2]],
        origin_ids=["a", "a"],
        destination_ids=["b", "c"],
        coordinates=[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]],
        hierarchy_groups=[["parent:a"], ["parent:a"]],
    )
    model = MarketStructureForecaster(
        head_epochs=12,
        huber_delta=1.5,
        quantile_levels=(0.1, 0.5, 0.9),
    ).fit(frame)

    assert frame.target_names == ["benchmark", "supporting_measure"]
    assert model.predict(1) == [{"primary": 1.0, "secondary": 2.0}]
    assert model.nowcast() == [{"shift": "no_shift"}]
    assert model.weekly_rollups(2) == [{"days": 2, "primary": 1.1, "secondary": 4.0}]
    assert model.explorer_payload(3)["kernels"] == []
    assert calls[0][1][2] == ["benchmark", "supporting_measure"]
    assert calls[0][1][9] == [["parent:a"], ["parent:a"]]
    assert calls[1][1]["head_epochs"] == 12
    assert calls[1][1]["huber_delta"] == 1.5
    assert calls[1][1]["quantile_levels"] == [0.1, 0.5, 0.9]


def test_market_panel_graph_adapter_requires_explicit_complete_inputs(monkeypatch):
    class NativeMarketPanelFrame:
        def __init__(self, *args):
            self.lane_ids = args[0]
            self.target_names = args[2]

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

    import cartoboost

    monkeypatch.setattr(
        cartoboost,
        "_native",
        SimpleNamespace(
            MarketPanelFrame=NativeMarketPanelFrame,
            GraphTemporalFrame=NativeGraphTemporalFrame,
        ),
        raising=False,
    )
    frame = MarketPanelFrame(
        lane_ids=["a:b", "a:c"],
        timestamps=[0, 1, 2],
        target_names=["fare", "duration"],
        primary=[[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        secondary=[[3.0, 4.0], [3.1, 4.1], [3.2, 4.2]],
        origin_ids=["a", "a"],
        destination_ids=["b", "c"],
        coordinates=[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]],
    )
    graph_frame = frame.as_graph_temporal_frame(indptr=[0, 1, 2], indices=[1, 0], data=[1.0, 1.0])
    assert graph_frame.node_ids == ["a:b", "a:c"]

    missing = MarketPanelFrame(
        lane_ids=["a:b", "a:c"],
        timestamps=[0, 1, 2],
        target_names=["fare", "duration"],
        primary=[[1.0, 2.0], [np.nan, 2.1], [1.2, 2.2]],
        secondary=[[3.0, 4.0], [3.1, 4.1], [3.2, 4.2]],
        origin_ids=["a", "a"],
        destination_ids=["b", "c"],
        coordinates=[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]],
    )
    with np.testing.assert_raises_regex(ValueError, "complete observed target"):
        missing.as_graph_temporal_frame(indptr=[0, 1, 2], indices=[1, 0], data=[1.0, 1.0])


def test_paper_graph_transformers_fit_a_complete_market_lane_panel_through_native_adapter():
    timestamps = list(range(24))
    primary = [[10.0 + np.sin(time / 3.0), 12.0 + np.cos(time / 4.0)] for time in timestamps]
    secondary = [[20.0 + np.cos(time / 5.0), 18.0 + np.sin(time / 6.0)] for time in timestamps]
    market_frame = MarketPanelFrame(
        lane_ids=["pickup_a:dropoff_b", "pickup_b:dropoff_a"],
        timestamps=timestamps,
        target_names=["fare", "trip_duration"],
        primary=primary,
        secondary=secondary,
        origin_ids=["pickup_a", "pickup_b"],
        destination_ids=["dropoff_b", "dropoff_a"],
        coordinates=[[-73.99, 40.75, -73.97, 40.73], [-73.97, 40.73, -73.99, 40.75]],
        horizon=2,
        frequency="hourly",
    )
    # The adapter owns only explicit market-to-graph conversion. Every paper
    # profile must then consume that same native graph frame without Python
    # model logic or a profile-specific market fallback.
    for target in ("primary", "secondary"):
        graph_frame = market_frame.as_graph_temporal_frame(
            indptr=[0, 1, 2], indices=[1, 0], data=[1.0, 1.0], target=target
        )
        for model_type in (
            STGormerForecaster,
            STGformerForecaster,
            LSTTNForecaster,
            SpatialTemporalGraphGatedTransformerForecaster,
            SpatialShiftGraphonMoEForecaster,
        ):
            model = model_type(
                lookback=8,
                hidden_size=4,
                attention_heads=2,
                graph_order=1,
                experts=2,
                periodicity=6,
                epochs=1,
            ).fit(graph_frame)
            prediction = model.predict(2)
            assert prediction.shape == (2, 2)
            assert np.isfinite(prediction).all()


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
    assert calls[1] == ("init", (3, 5, 7, 0.2, 1.0, 0.2, 0.0001, "cpu"))
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


def test_staeformer_native_fit_predict_and_save_load(tmp_path):
    target = []
    for t in range(24):
        target.append(
            [
                10.0 + np.sin(t / 3.0),
                12.0 + np.sin((t - 1) / 3.0),
                14.0 + np.sin((t - 2) / 3.0),
            ]
        )
    frame = GraphTemporalFrame(
        node_ids=["a", "b", "c"],
        timestamps=list(range(24)),
        target=np.asarray(target, dtype=float),
        indptr=[0, 1, 2, 3],
        indices=[1, 2, 0],
        data=[1.0, 1.0, 1.0],
        horizon=3,
        frequency="hourly",
    )
    model = STAEformerForecaster(lookback=5, attention_heads=2, hidden_size=4).fit(frame)

    prediction = model.predict(3)
    assert prediction.shape == (3, 3)
    assert np.isfinite(model.score(np.asarray(target[-3:], dtype=float)))
    path = tmp_path / "staeformer.json"
    model.save(path)
    loaded = STAEformerForecaster.load(path)
    assert loaded.predict(3).shape == (3, 3)


def test_graph_wavenet_native_fit_predict_and_save_load(tmp_path):
    target = []
    for t in range(24):
        target.append(
            [
                10.0 + np.sin(t / 3.0),
                12.0 + np.sin((t - 1) / 3.0),
                14.0 + np.sin((t - 2) / 3.0),
            ]
        )
    frame = GraphTemporalFrame(
        node_ids=["a", "b", "c"],
        timestamps=list(range(24)),
        target=np.asarray(target, dtype=float),
        indptr=[0, 1, 2, 3],
        indices=[1, 2, 0],
        data=[1.0, 1.0, 1.0],
        horizon=3,
        frequency="hourly",
    )
    model = GraphWaveNetForecaster(lookback=5, dilation_depth=2, hidden_size=4).fit(frame)

    prediction = model.predict(3)
    assert prediction.shape == (3, 3)
    assert np.isfinite(model.score(np.asarray(target[-3:], dtype=float)))
    path = tmp_path / "graph_wavenet.json"
    model.save(path)
    loaded = GraphWaveNetForecaster.load(path)
    assert loaded.predict(3).shape == (3, 3)


def test_paper_graph_transformers_are_native_backed_and_persistent(tmp_path):
    target = np.asarray(
        [
            [10.0 + np.sin(t / 3.0), 12.0 + np.sin((t - 1) / 3.0), 14.0 + np.sin((t - 2) / 3.0)]
            for t in range(36)
        ],
        dtype=float,
    )
    frame = GraphTemporalFrame(
        node_ids=["pickup", "midtown", "dropoff"],
        timestamps=list(range(36)),
        target=target,
        indptr=[0, 1, 2, 3],
        indices=[1, 2, 0],
        data=[1.0, 1.0, 1.0],
        horizon=3,
        frequency="hourly",
    )
    models = [
        STGormerForecaster,
        STGformerForecaster,
        LSTTNForecaster,
        SpatialTemporalGraphGatedTransformerForecaster,
        SpatialShiftGraphonMoEForecaster,
    ]
    for model_type in models:
        model = model_type(
            lookback=8,
            hidden_size=8,
            attention_heads=2,
            graph_order=2,
            experts=3,
            periodicity=6,
            epochs=8,
        ).fit(frame)
        prediction = model.predict(3)
        assert prediction.shape == (3, 3)
        assert np.isfinite(prediction).all()
        assert np.isfinite(model.score(target[-3:]))
        assert model.metadata_["fitted"] is True
        report = model.metadata_["architecture_report"]
        assert report["direct_multi_horizon"] is True
        assert report["trainable_forecast_head"] is True
        path = tmp_path / f"{model_type.__name__}.json"
        model.save(path)
        np.testing.assert_allclose(model_type.load(path).predict(3), prediction, atol=1e-12)


def test_lsttn_uses_long_history_defaults():
    model = LSTTNForecaster()

    assert model.get_params()["lookback"] == 24 * 28
    assert model.get_params()["periodicity"] == 24
    assert model.get_params()["recent_window"] == 24 * 7
    assert model.get_params()["horizon"] == 24 * 7


def test_lsttn_temporal_widths_are_configurable_in_frame_rows():
    model = LSTTNForecaster(
        lookback=24 * 28,
        periodicity=24,
        recent_window=24 * 7,
        horizon=24 * 7,
    )

    assert model.get_params()["lookback"] == 24 * 28
    assert model.get_params()["periodicity"] == 24
    assert model.get_params()["recent_window"] == 24 * 7
    assert model.get_params()["horizon"] == 24 * 7
