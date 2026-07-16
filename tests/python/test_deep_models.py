from __future__ import annotations

import json
import pickle
import types

import cartoboost.deep._native as native_helpers
import numpy as np
import pytest
from cartoboost.config import GraphBackbone
from cartoboost.deep import (
    ChoiceSetTransformer,
    ConditionalFlowDistributionHead,
    ConditionalResidualDiffusion,
    ConstrainedDecisionOptimizer,
    CounterfactualCandidateScorer,
    DirectionalPairForecaster,
    DirectionalPairFrame,
    EntityPanelFrame,
    EventOutcomeModel,
    FlowScenarioGenerator,
    FourierGeoOperator,
    GeoTemporalDiffusionScenarioModel,
    GraphNeuralOperator,
    GraphTemporalFrame,
    InvertedTemporalTransformer,
    NestedChoiceHead,
    PropagationDelayGraphForecaster,
    RegimeMoEForecaster,
    ResponseCurveFrame,
    ResponseCurveModel,
    ServiceTimeResidualModel,
    SpatioTemporalGraphForecaster,
    SpatioTemporalOperator,
    TemporalEntityTransformer,
    UtilityNet,
    available_deep_backends,
    backend_dispatch_report,
)


class _Frame:
    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return self._rows


def test_response_curve_monotone_decreasing_with_native():
    rows = [
        {
            "features": [1.0],
            "candidate_value": 1.0,
            "response": 0.9,
            "group_id": "g",
            "candidate_id": "a",
        },
        {
            "features": [1.0],
            "candidate_value": 2.0,
            "response": 0.2,
            "group_id": "g",
            "candidate_id": "b",
        },
    ]

    model = ResponseCurveModel(response_type="binary", monotone="decreasing")
    model.fit(ResponseCurveFrame(rows))
    curve = model.predict_curve(ResponseCurveFrame(rows))
    choice = model.choice_set_report(ResponseCurveFrame(rows))

    assert curve[0]["response_score"] > curve[1]["response_score"]
    assert model.best_candidate(ResponseCurveFrame(rows))[0]["candidate_id"] == "a"
    assert choice["surface"] == "ResponseCurveModel"
    assert any(row["candidate_id"] == "__outside__" for row in choice["predictions"])
    assert model.metadata_["hidden_weights"]
    assert model.metadata_["regime_moe"]["consumed"] is True
    assert model.metadata_["regime_moe"]["surface"] == "ResponseCurveModel"


def test_directional_pair_preserves_order_with_native_fit():
    frame = DirectionalPairFrame(
        [
            {"source_id": "A", "target_id": "B", "features": [], "target": 1.0},
            {"source_id": "B", "target_id": "A", "features": [], "target": 2.0},
        ]
    )

    pred = DirectionalPairForecaster().fit(frame).predict(frame)

    assert pred[1] > pred[0]


def test_directional_pair_runs_on_every_available_backend():
    frame = DirectionalPairFrame(
        [
            {
                "source_id": source,
                "target_id": target,
                "features": [float(step), float(step % 2)],
                "target": offset + 0.4 * step,
            }
            for source, target, offset in [("A", "B", 1.0), ("B", "A", -1.0)]
            for step in range(5)
        ]
    )
    expected = DirectionalPairForecaster(backend="cpu").fit(frame).predict(frame)
    for backend in available_deep_backends():
        model = DirectionalPairForecaster(backend=backend).fit(frame)
        actual = model.predict(frame)
        assert model.backend_ == backend
        assert np.allclose(actual, expected, rtol=1.0e-4, atol=1.0e-4), backend


def test_continuous_tanh_models_train_on_every_available_backend():
    response_rows = [
        {
            "features": [x, np.sin(x)],
            "candidate_value": 1.0 + 0.2 * x,
            "response": 0.7 + 0.4 * x + 0.2 * np.sin(x),
        }
        for x in np.linspace(0.0, 3.0, 16)
    ]
    response_frame = ResponseCurveFrame(response_rows)
    service_rows = [
        {
            "baseline_value": 10.0,
            "actual_value": 10.0 + row["response"],
            "features": row["features"],
        }
        for row in response_rows
    ]
    expected_response = ResponseCurveModel(response_type="continuous", backend="cpu").fit(
        response_frame
    )
    expected_response_values = expected_response.predict_response(response_frame)
    expected_service = ServiceTimeResidualModel(backend="cpu").fit(service_rows)
    expected_service_values = expected_service.predict(service_rows)
    for backend in available_deep_backends():
        response = ResponseCurveModel(response_type="continuous", backend=backend).fit(
            response_frame
        )
        service = ServiceTimeResidualModel(backend=backend).fit(service_rows)
        assert response.metadata_["backend"]["selected"] == backend
        assert service.metadata_["backend"]["selected"] == backend
        assert np.allclose(
            response.predict_response(response_frame),
            expected_response_values,
            rtol=0.02,
            atol=0.08,
        ), backend
        assert np.allclose(
            service.predict(service_rows), expected_service_values, rtol=0.02, atol=0.08
        ), backend


def test_binary_tanh_models_train_on_every_available_backend():
    features = np.asarray([[x, x * x] for x in np.linspace(0.0, 1.0, 20)])
    labels = (features[:, 0] >= 0.5).astype(float)
    response_frame = ResponseCurveFrame(
        [
            {
                "features": row.tolist(),
                "candidate_value": float(row[0]),
                "response": float(label),
            }
            for row, label in zip(features, labels, strict=True)
        ]
    )
    for backend in available_deep_backends():
        event = EventOutcomeModel(backend=backend).fit(features, labels)
        event_probability = event.predict_proba(features)
        response = ResponseCurveModel(
            response_type="binary", monotone="increasing", backend=backend
        ).fit(response_frame)
        response_probability = np.asarray(
            [row["response_probability"] for row in response.predict_curve(response_frame)]
        )
        assert event.metadata_["backend"]["selected"] == backend
        assert response.metadata_["backend"]["selected"] == backend
        assert event_probability[-1] > event_probability[0], backend
        assert response_probability[-1] > response_probability[0], backend


def test_directional_pair_embedding_mlp_public_wrapper():
    rows = []
    for source, target, direction in [("A", "B", 1.0), ("B", "A", -1.0), ("A", "C", 0.6)]:
        for step in range(12):
            x = step / 11.0
            rows.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "features": [x],
                    "target": 2.0 + direction * (1.0 + (x - 0.5) ** 2),
                }
            )
    frame = DirectionalPairFrame(rows)
    model = DirectionalPairForecaster(
        architecture="pair_embedding_mlp",
        embedding_dim=4,
        pair_bucket_count=24,
        hidden_dim=12,
        epochs=260,
        seed=13,
    ).fit(frame)
    pred = model.predict(frame)
    unseen = model.predict(
        DirectionalPairFrame(
            [{"source_id": "A", "target_id": "Z", "features": [0.2], "target": None}]
        )
    )

    assert model.metadata_["architecture"] == "pair_embedding_mlp"
    assert model.metadata_["source_id_map"]["__unknown__"] == 0
    assert model.metadata_["target_id_map"]["__unknown__"] == 0
    assert model.metadata_["pair_global_bucket"] == 0
    assert model.metadata_["pair_bucket_count"] == 24
    assert model.metadata_["embedding_dim"] == 4
    assert model.metadata_["loss"] == "squared_error"
    assert model.metadata_["seed"] == 13
    assert model.metadata_["schema_hash"].startswith("directional_pair:")
    assert model.metadata_["shared_representation_consumed"] is False
    assert model.metadata_["shared_representation"] is None
    assert model.metadata_["train_metrics"]["rmse"] >= 0.0
    assert np.isfinite(unseen[0])
    assert pred[0] > pred[12]


def test_directional_pair_forecaster_rejects_removed_multi_view_representation():
    views = {
        "physical_distance": [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
        "observed_flow": [[0.0, 2.0], [1.5, 0.2], [0.1, 1.7]],
    }

    with pytest.raises(RuntimeError, match="representation primitives are not shipped"):
        DirectionalPairForecaster(
            architecture="pair_embedding_mlp",
            epochs=80,
            seed=17,
            multi_view_views=views,
        )


def test_directional_pair_embedding_mlp_beats_shrinkage_public_wrapper():
    rows = []
    for source, target, direction in [("A", "B", 1.0), ("B", "A", -1.0), ("A", "C", 0.4)]:
        for step in range(18):
            x = step / 17.0
            rows.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "features": [x, x * x],
                    "target": 1.5 + direction * (2.0 * np.sin(np.pi * x) + (x - 0.5) ** 2),
                }
            )
    frame = DirectionalPairFrame(rows)

    shrink = DirectionalPairForecaster(architecture="shrinkage_effects").fit(frame)
    embed = DirectionalPairForecaster(
        architecture="pair_embedding_mlp",
        embedding_dim=5,
        pair_bucket_count=32,
        hidden_dim=16,
        epochs=420,
        learning_rate=0.012,
        seed=19,
    ).fit(frame)

    assert np.isfinite(embed.score(frame))
    assert np.isfinite(shrink.score(frame))


def test_directional_pair_temporal_ssm_and_regime_moe_architectures() -> None:
    rows = []
    for step in range(18):
        rows.append(
            {
                "source_id": "A" if step % 2 == 0 else "B",
                "target_id": "B" if step % 2 == 0 else "A",
                "timestamp": step * 3600,
                "features": [step / 17.0],
                "target": 2.0 + np.sin(step / 3.0) + (0.5 if step % 2 == 0 else -0.5),
            }
        )
    frame = DirectionalPairFrame(rows)

    temporal = DirectionalPairForecaster(architecture="pair_temporal_ssm", epochs=120, seed=31).fit(
        frame
    )
    regime = DirectionalPairForecaster(architecture="pair_regime_moe", epochs=120, seed=37).fit(
        frame
    )

    assert temporal.metadata_["architecture"] == "pair_temporal_ssm"
    assert regime.metadata_["architecture"] == "pair_regime_moe"
    assert regime.metadata_["regime_moe"]["consumed"] is True
    assert regime.metadata_["regime_moe"]["surface"] == "DirectionalPairForecaster"
    assert temporal.metadata_["train_metrics"]["expanded_feature_count"] >= 1
    assert regime.metadata_["train_metrics"]["expanded_feature_count"] >= 1
    assert np.isfinite(temporal.predict(frame)).all()
    assert np.isfinite(regime.predict(frame)).all()


def test_inverted_transformer_runs_without_removed_representation() -> None:
    y = np.asarray(
        [
            [1.0, 2.0],
            [1.2, 1.9],
            [1.5, 2.1],
            [1.8, 2.4],
            [2.0, 2.6],
        ],
        dtype=float,
    )
    frame = EntityPanelFrame(y=y, timestamps=list(range(len(y))), entity_ids=["a", "b"])

    model = InvertedTemporalTransformer(lookback=4, horizon=2, seed=7).fit(frame)

    assert model.metadata_["shared_representation_consumed"] is False
    assert model.metadata_["shared_representation"] is None
    assert model.predict().shape == (2, 2)


def test_deep_frame_builders_reject_non_finite_values():
    panel = _Frame(
        [
            {"ts": 0, "entity": "a", "target": 1.0},
            {"ts": 0, "entity": "b", "target": np.inf},
        ]
    )
    with pytest.raises(ValueError, match="targets must contain only finite values"):
        EntityPanelFrame.from_pandas(
            panel,
            timestamp_col="ts",
            entity_col="entity",
            target_col="target",
        )

    directional = _Frame([{"source": "a", "target": "b", "distance": np.nan, "y": 1.0}])
    with pytest.raises(ValueError, match="directional pair features"):
        DirectionalPairFrame.from_pandas(
            directional,
            source_col="source",
            target_col="target",
            target_value_col="y",
            numeric_covariates=["distance"],
        )

    response = _Frame([{"price": 10.0, "distance": np.inf, "response": 0.2}])
    with pytest.raises(ValueError, match="response curve features"):
        ResponseCurveFrame.from_pandas(
            response,
            feature_cols=["distance"],
            candidate_value_col="price",
            response_col="response",
        )


def test_temporal_entity_transformer_uses_native_attention_fit():
    y = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 12.0],
            [3.0, 14.0],
            [4.0, 16.0],
            [5.0, 18.0],
            [6.0, 20.0],
        ]
    )
    frame = EntityPanelFrame(y=y, timestamps=list(range(len(y))), entity_ids=["a", "b"])
    model = TemporalEntityTransformer(lookback=2, horizon=2).fit(frame)

    pred = model.predict()
    tiled_recent_mean = np.tile(y[-2:].mean(axis=0), (2, 1))

    assert pred.shape == (2, 2)
    assert model.metadata_["attention_queries"]
    assert model.metadata_["decoder_weights"]
    assert model.metadata_["regime_moe"]["consumed"] is True
    assert model.metadata_["regime_moe"]["surface"] == "TemporalEntityTransformer"
    assert model.metadata_["flow_uncertainty_head"]["consumed"] is True
    assert model.metadata_["flow_uncertainty_head"]["surface"] == "TemporalEntityTransformer"
    assert not np.allclose(pred, tiled_recent_mean)


def test_temporal_entity_transformer_runs_on_every_available_backend():
    y = np.asarray(
        [[1.0 + step * 0.2, 3.0 + np.sin(step * 0.3)] for step in range(10)],
        dtype=float,
    )
    frame = EntityPanelFrame(y=y, timestamps=list(range(len(y))), entity_ids=["a", "b"])
    expected = TemporalEntityTransformer(lookback=3, horizon=2, backend="cpu").fit(frame)
    expected_prediction = expected.predict()
    for backend in available_deep_backends():
        model = TemporalEntityTransformer(lookback=3, horizon=2, backend=backend).fit(frame)
        actual = model.predict()
        assert model.backend_ == backend
        assert actual.shape == expected_prediction.shape
        assert np.allclose(actual, expected_prediction, rtol=0.02, atol=0.1), backend


def test_inverted_temporal_transformer_beats_lag_and_reports_horizon_metrics(tmp_path):
    steps = 60
    y = np.zeros((steps, 3), dtype=float)
    y[0] = [1.0, 0.0, 0.0]
    for step in range(1, steps):
        y[step, 1] = 0.8 * y[step - 1, 1] + 0.2 * np.sin(step / 4.0)
        y[step, 0] = 0.6 * y[step - 1, 0] + 0.6 * y[step - 1, 1] + 0.05 * step
        y[step, 2] = 0.8 * y[step - 1, 2] + 0.1 * y[step - 1, 0]
    train = y[:-3]
    actual = y[-3:]
    frame = EntityPanelFrame(
        y=train, timestamps=list(range(len(train))), entity_ids=["a", "b", "c"]
    )
    model = InvertedTemporalTransformer(lookback=24, horizon=3).fit(frame)
    pred = model.predict()
    lag = np.tile(train[-1], (3, 1))
    report = model.cross_entity_ablation_report(actual)
    path = tmp_path / "inverted.json"
    model.save(path)
    loaded = InvertedTemporalTransformer.load(path)

    assert pred.shape == (3, 3)
    assert np.sqrt(np.mean((actual - pred) ** 2)) < np.sqrt(np.mean((actual - lag) ** 2))
    assert report["cross_entity_features_help"] is True
    assert len(model.horizon_metrics(actual)["rmse"]) == 3
    assert model.metadata_["architecture"] == "inverted_transformer"
    assert model.metadata_["token_axis"] == "entity"
    assert model.metadata_["quadratic_time_token_attention"] is False
    assert model.metadata_["save_load_parity_checked"] is True
    np.testing.assert_array_equal(loaded.predict(), pred)


def test_temporal_entity_transformer_routes_inverted_architecture():
    y = np.column_stack(
        [
            np.arange(12, dtype=float),
            np.arange(12, dtype=float) * 2.0,
        ]
    )
    frame = EntityPanelFrame(y=y, timestamps=list(range(len(y))), entity_ids=["a", "b"])
    model = TemporalEntityTransformer(
        architecture="inverted_transformer",
        lookback=6,
        horizon=2,
    ).fit(frame)

    assert model.predict().shape == (2, 2)
    assert model.metadata_["architecture"] == "inverted_transformer"


def test_spatiotemporal_graph_facade_routes_graph_wavenet():
    model = SpatioTemporalGraphForecaster(
        backbone=GraphBackbone.GRAPH_WAVENET,
        lookback=3,
        dilation_depth=2,
    )

    assert model.metadata_["backbone"] == "graph_wavenet"


def test_spatiotemporal_graph_facade_routes_temporal_attention():
    model = SpatioTemporalGraphForecaster(
        backbone=GraphBackbone.TEMPORAL_GRAPH_ATTENTION,
        lookback=3,
        attention_heads=2,
    )

    assert model.metadata_["backbone"] == "temporal_graph_attention"


def test_spatiotemporal_graph_forecaster_rejects_removed_multi_view_representation():
    views = {
        "physical_distance": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        "historical_similarity": [[0.9], [0.4], [0.2]],
    }

    with pytest.raises(RuntimeError, match="representation primitives are not shipped"):
        SpatioTemporalGraphForecaster(
            backbone=GraphBackbone.DELAY_AWARE_GRAPH_TRANSFORMER,
            horizon=2,
            edge_delay_prior=[1, 1],
            multi_view_views=views,
        )


def test_delay_aware_graph_transformer_direction_delay_and_roundtrip(tmp_path):
    steps = 90
    y = np.zeros((steps, 3), dtype=float)
    for step in range(steps):
        y[step, 0] = np.sin(step / 4.0) + 0.02 * step
        if step >= 2:
            y[step, 1] = 0.35 * y[step - 1, 1] + 0.9 * y[step - 2, 0]
        if step >= 3:
            y[step, 2] = 0.25 * y[step - 1, 2] + 0.7 * y[step - 1, 1]

    train = y[:-4]
    actual = y[-4:]
    timestamps = list(range(len(train)))
    correct = GraphTemporalFrame(
        y=train,
        timestamps=timestamps,
        node_ids=["pickup", "midway", "dropoff"],
        edges=[(0, 1), (1, 2)],
        edge_weights=[1.0, 1.0],
        edge_distances=[0.4, 0.9],
        node_covariates=np.asarray([[1.0, 0.0], [0.6, 0.3], [0.1, 1.0]], dtype=float),
        known_future_covariates=np.ones((4, 3, 1), dtype=float) * 0.2,
        directed=True,
    )
    reversed_graph = GraphTemporalFrame(
        y=train,
        timestamps=timestamps,
        node_ids=["pickup", "midway", "dropoff"],
        edges=[(1, 0), (2, 1)],
        edge_weights=[1.0, 1.0],
        directed=True,
    )

    model = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1]).fit(correct)
    reversed_model = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1]).fit(
        reversed_graph
    )
    no_delay = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[1, 1]).fit(correct)
    path = tmp_path / "delay-graph.json"
    model.save(path)
    loaded = PropagationDelayGraphForecaster.load(path)
    falsifiers = model.falsifier_report(actual)
    routed = SpatioTemporalGraphForecaster(
        backbone=GraphBackbone.DELAY_AWARE_GRAPH_TRANSFORMER,
        horizon=4,
        edge_delay_prior=[2, 1],
    ).fit(correct)
    routed_path = tmp_path / "routed-delay-graph.weights.json"
    routed.save_weights(routed_path)
    routed_loaded = SpatioTemporalGraphForecaster.load_weights(routed_path)
    routed_pickled = pickle.loads(pickle.dumps(routed))

    assert model.score(actual) < reversed_model.score(actual)
    assert model.score(actual) < no_delay.score(actual)
    assert falsifiers["delay_beats_non_graph_temporal"] is True
    assert falsifiers["delay_beats_static_adjacency_only"] is True
    assert model.edge_delay_sensitivity()["delay_counts"] == {"1": 1, "2": 1}
    assert model.metadata_["architecture"] == "delay_aware_graph_transformer"
    assert model.metadata_["backend"]["supported"] == [
        "cpu",
        "cuda",
        "rocm",
        "metal",
        "directml",
        "webgpu",
    ]
    assert model.metadata_["backend"]["accelerated"] is False
    assert model.metadata_["shared_representation_consumed"] is False
    assert model.metadata_["shared_representation"] is None
    assert model.metadata_["inputs"] == {
        "edge_distances": True,
        "node_covariates": True,
        "known_future_covariates": True,
    }
    attention = model.metadata_["attention_blocks"]
    assert len(attention["edge_distance_embedding"]) == 2
    assert len(attention["dynamic_attention_mask"]) == 2
    assert len(attention["short_range_graph_attention"]) == 2
    assert len(attention["long_range_semantic_attention"]) == 2
    assert attention["temporal_attention"]["lookback"] == 8
    assert model.metadata_["falsifier_baselines"] == [
        "non_graph_temporal_model",
        "static_adjacency_only_graph_model",
    ]
    assert model.metadata_["flow_uncertainty_head"]["consumed"] is True
    assert model.metadata_["flow_uncertainty_head"]["surface"] == "SpatioTemporalGraphForecaster"
    assert model.metadata_["save_load_parity_checked"] is True
    assert routed.metadata_["backbone"] == "delay_aware_graph_transformer"
    np.testing.assert_array_equal(routed_loaded.predict(), routed.predict())
    np.testing.assert_array_equal(routed_pickled.predict(), routed.predict())
    np.testing.assert_array_equal(loaded.predict(), model.predict())

    with pytest.raises(ValueError, match="backend must be one of"):
        PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1], backend="mlx")


def test_regime_moe_reports_usage_and_beats_single_expert(tmp_path):
    steps = 72
    x0 = np.linspace(-2.0, 2.0, steps)
    distance = np.linspace(0.0, 1.0, steps)
    features = np.column_stack([x0, x0**2, distance])
    volatility = np.where(np.arange(steps) % 9 == 0, 2.0, 0.1)
    sparsity = np.where(np.arange(steps) < 12, 2.5, 0.1)
    centrality = np.where(np.arange(steps) > 48, 2.0, 0.2)
    residuals = np.sin(np.arange(steps) / 2.0) * volatility
    y = (
        1.2
        + 0.8 * x0
        + 1.5 * centrality
        + 1.1 * volatility * np.sign(np.sin(np.arange(steps)))
        + 1.6 * distance
        - 0.6 * sparsity
    )
    model = RegimeMoEForecaster().fit(
        features,
        y,
        entity_ids=[f"entity_{idx % 8}" for idx in range(steps)],
        pair_ids=[f"lane_{idx % 5}" for idx in range(steps)],
        time_features=np.column_stack([np.sin(np.arange(steps) / 6.0)]),
        recent_volatility=volatility,
        recent_residuals=residuals,
        graph_centrality=centrality,
        historical_sparsity=sparsity,
        candidate_value=distance,
    )
    components = model.predict_components(
        features,
        time_features=np.column_stack([np.sin(np.arange(steps) / 6.0)]),
        recent_volatility=volatility,
        recent_residuals=residuals,
        graph_centrality=centrality,
        historical_sparsity=sparsity,
        candidate_value=distance,
    )
    path = tmp_path / "regime-moe.json"
    model.save(path)
    loaded = RegimeMoEForecaster.load(path)
    weights_path = tmp_path / "regime-moe.weights.json"
    model.save_weights(weights_path)
    weights_loaded = RegimeMoEForecaster.load_weights(weights_path)

    assert components["expert_weights"].shape == (steps, 6)
    assert components["expert_predictions"].shape == (steps, 6)
    assert components["combined_prediction"].shape == (steps,)
    assert model.metadata_["router_entropy"] > 0.0
    assert model.metadata_["shared_representation_consumed"] is False
    assert model.metadata_["shared_representation"] is None
    assert sum(value > 0.0 for value in model.metadata_["expert_usage"].values()) >= 2
    assert model.metadata_["train_metrics"]["beats_single_expert"] is True
    assert model.metadata_["expert_usage"]["sparse_cold_start"] > 0.0
    assert model.metadata_["outputs"] == [
        "expert_weights",
        "expert_predictions",
        "combined_prediction",
        "regime_metadata",
    ]
    np.testing.assert_allclose(
        loaded.predict(
            features,
            time_features=np.column_stack([np.sin(np.arange(steps) / 6.0)]),
            recent_volatility=volatility,
            recent_residuals=residuals,
            graph_centrality=centrality,
            historical_sparsity=sparsity,
            candidate_value=distance,
        ),
        components["combined_prediction"],
    )
    np.testing.assert_allclose(
        pickle.loads(pickle.dumps(model)).predict(
            features,
            time_features=np.column_stack([np.sin(np.arange(steps) / 6.0)]),
            recent_volatility=volatility,
            recent_residuals=residuals,
            graph_centrality=centrality,
            historical_sparsity=sparsity,
            candidate_value=distance,
        ),
        components["combined_prediction"],
    )
    np.testing.assert_allclose(
        weights_loaded.predict(
            features,
            time_features=np.column_stack([np.sin(np.arange(steps) / 6.0)]),
            recent_volatility=volatility,
            recent_residuals=residuals,
            graph_centrality=centrality,
            historical_sparsity=sparsity,
            candidate_value=distance,
        ),
        components["combined_prediction"],
    )


def test_conditional_flow_head_outputs_joint_uncertainty_metrics(tmp_path):
    steps = 16
    hidden = np.column_stack([np.linspace(0.0, 1.0, steps), np.sin(np.arange(steps) / 3.0)])
    horizon = np.column_stack([np.arange(steps) % 4])
    entity = np.column_stack([np.linspace(1.0, 2.0, steps)])
    graph = np.column_stack([np.cos(np.arange(steps) / 5.0)])
    residuals = 0.2 * hidden[:, 0] + 0.1 * np.sin(np.arange(steps))
    head = ConditionalFlowDistributionHead(quantiles=(0.05, 0.5, 0.95), sample_count=12).fit(
        residuals,
        model_hidden_state=hidden,
        horizon_embeddings=horizon,
        entity_or_pair_embeddings=entity,
        graph_context=graph,
    )
    output = head.predict(
        model_hidden_state=hidden,
        horizon_embeddings=horizon,
        entity_or_pair_embeddings=entity,
        graph_context=graph,
        actual=residuals,
    )
    path = tmp_path / "flow.json"
    head.save(path)
    loaded = ConditionalFlowDistributionHead.load(path)
    weights_path = tmp_path / "flow.weights.json"
    head.save_weights(weights_path)
    weights_loaded = ConditionalFlowDistributionHead.load_weights(weights_path)
    loaded_output = loaded.predict(
        model_hidden_state=hidden,
        horizon_embeddings=horizon,
        entity_or_pair_embeddings=entity,
        graph_context=graph,
        actual=residuals,
    )
    weights_output = weights_loaded.predict(
        model_hidden_state=hidden,
        horizon_embeddings=horizon,
        entity_or_pair_embeddings=entity,
        graph_context=graph,
        actual=residuals,
    )
    pickled_output = pickle.loads(pickle.dumps(head)).predict(
        model_hidden_state=hidden,
        horizon_embeddings=horizon,
        entity_or_pair_embeddings=entity,
        graph_context=graph,
        actual=residuals,
    )

    assert output["samples"].shape == (steps, 12)
    assert output["marginal_quantiles"].shape == (steps, 3)
    assert output["joint_scenario_paths"].shape == (12, steps)
    assert output["log_likelihood"].shape == (steps,)
    np.testing.assert_allclose(weights_output["marginal_quantiles"], output["marginal_quantiles"])
    np.testing.assert_allclose(pickled_output["marginal_quantiles"], output["marginal_quantiles"])
    assert "expected_shortfall_low" in output["tail_risk_metrics"]
    assert "crps" in output["metrics"]
    assert "pinball_median" in output["metrics"]
    assert "interval_coverage" in output["metrics"]
    assert "joint_path_calibration" in output["metrics"]
    assert "tail_event_calibration" in output["metrics"]
    assert head.metadata_["architecture"] == "conditional_residual_sampler"
    assert head.metadata_["backend_requested"] == "cpu"
    assert head.backend_ == "cpu"
    assert loaded.backend_ == "cpu"
    benchmark = head.benchmark_against_baselines(
        residuals,
        model_hidden_state=hidden,
        horizon_embeddings=horizon,
        entity_or_pair_embeddings=entity,
        graph_context=graph,
    )
    assert "independent_quantile_head" in benchmark
    assert "gaussian_residual_head" in benchmark
    assert "conformal_interval_wrapper" in benchmark
    assert benchmark["flow_improves_calibration_or_sharpness"] is True
    np.testing.assert_array_equal(loaded_output["marginal_quantiles"], output["marginal_quantiles"])


def test_conditional_flow_training_and_inference_support_every_backend():
    hidden = np.column_stack([np.linspace(0.0, 1.0, 12), np.arange(12) % 3])
    residuals = 0.4 * hidden[:, 0] - 0.15 * hidden[:, 1]
    expected = (
        ConditionalFlowDistributionHead(sample_count=8)
        .fit(residuals, model_hidden_state=hidden)
        .predict(model_hidden_state=hidden)
    )
    for backend in available_deep_backends():
        head = ConditionalFlowDistributionHead(sample_count=8, backend=backend).fit(
            residuals, model_hidden_state=hidden
        )
        assert head.backend_ == backend
        actual = head.predict(model_hidden_state=hidden)
        np.testing.assert_allclose(actual["samples"], expected["samples"], rtol=2e-3, atol=2e-3)


def test_diffusion_scenario_generator_reports_experimental_summaries():
    point_forecast = np.array(
        [
            [10.0, 12.0, 13.0],
            [11.0, 12.5, 14.0],
            [12.0, 13.0, 15.0],
        ]
    )
    edges = [
        {"source": 0, "target": 1, "weight": 1.0},
        {"source": 1, "target": 2, "weight": 0.7},
    ]
    model = GeoTemporalDiffusionScenarioModel(
        scenario_count=7,
        diffusion_steps=2,
        shock_scale=0.4,
    )
    output = model.generate(point_forecast, edges)

    assert output["scenarios"].shape == (7, 3, 3)
    assert output["scenario_mean"].shape == point_forecast.shape
    assert output["scenario_variance"].shape == point_forecast.shape
    assert np.isfinite(output["spatial_correlation"])
    assert "mean_absolute_delta" in output["point_forecast_comparison"]
    assert "mean_variance" in output["point_forecast_comparison"]
    assert output["metadata"]["capability_tier"] == "experimental"
    assert output["metadata"]["auto_geo_enabled"] == "false"
    assert output["metadata"]["primary_benchmark_evidence"] == "false"
    assert output["metadata"]["backend_requested"] == "cpu"
    assert model.backend_ == "cpu"
    assert FlowScenarioGenerator is GeoTemporalDiffusionScenarioModel
    assert ConditionalResidualDiffusion is GeoTemporalDiffusionScenarioModel


def test_graph_neural_operator_outputs_fields_and_benchmark_lift():
    fields = np.array([[1.0, 2.0, 3.0], [1.2, 2.1, 3.4]])
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    edges = [
        {"source": 0, "target": 1, "weight": 1.0},
        {"source": 1, "target": 2, "weight": 0.5},
    ]
    exogenous = np.full_like(fields, 0.2)
    model = GraphNeuralOperator(smoothing=0.25, coordinate_scale=0.1)
    output = model.predict(
        field_values=fields,
        coordinates=coords,
        edges=edges,
        exogenous_fields=exogenous,
    )
    benchmark = GraphNeuralOperator.synthetic_benchmark()

    assert output["future_field"].shape == fields.shape
    assert output["residual_field"].shape == fields.shape
    assert output["uncertainty_field"].shape == fields.shape
    assert output["metadata"]["capability_tier"] == "advanced_experimental"
    assert benchmark["operator_rmse"] < benchmark["pointwise_mlp_rmse"]
    assert benchmark["improvement"] > 0.0
    assert FourierGeoOperator is GraphNeuralOperator
    assert SpatioTemporalOperator is GraphNeuralOperator


def test_choice_set_transformer_competes_candidates_and_reports_calibration():
    candidates = [
        {
            "decision_id": "d1",
            "candidate_id": "a",
            "candidate_value": 1.0,
            "expected_utility": 2.0,
            "response_probability": 0.8,
            "candidate_features": [1.0, 0.0],
            "context_features": [0.5],
            "entity_or_pair_embeddings": [0.2],
            "nest_id": "n",
            "chosen": True,
        },
        {
            "decision_id": "d1",
            "candidate_id": "b",
            "candidate_value": 1.5,
            "expected_utility": 0.2,
            "response_probability": 0.2,
            "candidate_features": [0.0, 1.0],
            "context_features": [0.5],
            "entity_or_pair_embeddings": [0.1],
            "nest_id": "n",
            "chosen": False,
        },
    ]
    report = ChoiceSetTransformer(
        temperature=0.7,
        monotone_candidate_value="increasing",
        outside_option=True,
    ).score(candidates)
    reversed_report = ChoiceSetTransformer(
        temperature=0.7,
        monotone_candidate_value="increasing",
    ).score(list(reversed(candidates)))

    assert report["counterfactual_best"][0]["candidate_id"] == "a"
    assert reversed_report["counterfactual_best"][0]["candidate_id"] == "a"
    assert any(row["candidate_id"] == "__outside__" for row in report["predictions"])
    assert all(row["choice_probability"] > 0.0 for row in report["predictions"])
    assert any(row["nested_probability"] is not None for row in report["predictions"])
    assert "brier" in report["calibration"]
    assert "ece" in report["calibration"]
    assert report["metadata"]["architecture"] == "choice_set_utility_softmax"
    assert report["metadata"]["candidate_candidate_attention"] == "false"
    assert (
        report["benchmark"]["choice_set_log_loss"]
        < report["benchmark"]["independent_response_log_loss"]
    )
    assert UtilityNet is ChoiceSetTransformer
    assert NestedChoiceHead is ChoiceSetTransformer
    assert CounterfactualCandidateScorer is ChoiceSetTransformer


def test_event_residual_and_decision_wrappers_with_native():
    event = EventOutcomeModel().fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0])
    probs = event.predict_proba([[0.0], [3.0]])
    event_choice = event.choice_set_report(
        [
            {"decision_id": "d", "candidate_id": "low", "candidate_value": 1.0},
            {"decision_id": "d", "candidate_id": "high", "candidate_value": 2.0},
        ],
        [[0.0], [3.0]],
    )
    assert probs[1] > probs[0]
    assert event.metadata_["hidden_weights"]
    assert event_choice["surface"] == "EventOutcomeModel"

    rows = [
        {"baseline_value": 10.0, "actual_value": 10.0, "features": [0.0]},
        {"baseline_value": 10.0, "actual_value": 13.0, "features": [2.0]},
    ]
    residual = ServiceTimeResidualModel().fit(rows)
    pred = residual.predict(rows)
    assert pred[1] > pred[0]
    assert residual.metadata_["hidden_weights"]
    assert residual.metadata_["regime_moe"]["consumed"] is True
    assert residual.metadata_["regime_moe"]["surface"] == "ServiceTimeResidualModel"
    assert residual.metadata_["flow_uncertainty_head"]["consumed"] is True
    assert residual.metadata_["flow_uncertainty_head"]["surface"] == "ServiceTimeResidualModel"

    choice = ConstrainedDecisionOptimizer().select(
        [{"decision_id": "d", "candidate_id": "c", "candidate_value": 1.0, "expected_utility": 3.0}]
    )
    assert choice[0]["candidate_id"] == "c"
    decision_choice = ConstrainedDecisionOptimizer().choice_set_report(
        [{"decision_id": "d", "candidate_id": "c", "candidate_value": 1.0, "expected_utility": 3.0}]
    )
    assert decision_choice["surface"] == "ConstrainedDecisionOptimizer"
    assert any(row["candidate_id"] == "__outside__" for row in decision_choice["predictions"])
    decision_flow = ConstrainedDecisionOptimizer().flow_uncertainty_report(
        [
            {
                "decision_id": "d",
                "candidate_id": "a",
                "candidate_value": 1.0,
                "expected_utility": 3.0,
            },
            {
                "decision_id": "d",
                "candidate_id": "b",
                "candidate_value": 2.0,
                "expected_utility": 1.0,
            },
            {
                "decision_id": "d",
                "candidate_id": "c",
                "candidate_value": 3.0,
                "expected_utility": 5.0,
            },
        ]
    )
    assert decision_flow["consumed"] is True
    assert decision_flow["surface"] == "ConstrainedDecisionOptimizer"


def test_decision_optimizer_merges_predictions_and_applies_risk_aversion():
    candidates = [
        {"decision_id": "d", "candidate_id": "fast", "candidate_value": 2.0},
        {"decision_id": "d", "candidate_id": "steady", "candidate_value": 2.0},
    ]
    predictions = [
        {
            "candidate_id": "fast",
            "expected_utility": 10.0,
            "response_probability": 0.9,
            "risk_score": 8.0,
        },
        {
            "candidate_id": "steady",
            "expected_utility": 7.0,
            "response_probability": 0.8,
            "risk_score": 1.0,
        },
    ]

    risk_neutral = ConstrainedDecisionOptimizer(risk_aversion=0.0).select(candidates, predictions)
    risk_averse = ConstrainedDecisionOptimizer(risk_aversion=1.0).select(candidates, predictions)

    assert risk_neutral[0]["candidate_id"] == "fast"
    assert risk_averse[0]["candidate_id"] == "steady"


def test_deep_backend_parameter_and_availability_delegate(monkeypatch):
    calls = []
    artifact = {
        "response_type": "binary",
        "backend": {"requested": "cuda", "selected": "cuda", "available": ["cpu", "cuda"]},
    }

    monkeypatch.setattr(
        native_helpers,
        "_native",
        types.SimpleNamespace(
            deep_available_backends_value=lambda: ["cpu", "cuda"],
            deep_backend_dispatch_report_value=lambda backend=None, len=4096: json.dumps(
                {
                    "requested": backend or "auto",
                    "selected": "cuda",
                    "operation": "vector_add_f32",
                    "len": len,
                    "checksum": 12.0,
                    "expected_checksum": 12.0,
                    "elapsed_ms": 1.0,
                    "accelerated": True,
                }
            ),
            deep_response_curve_fit_value=lambda *args: calls.append(args) or json.dumps(artifact),
        ),
    )

    model = ResponseCurveModel(backend="cuda").fit(
        ResponseCurveFrame(
            [
                {
                    "features": [1.0],
                    "candidate_value": 1.0,
                    "response": 1.0,
                    "group_id": "g",
                    "candidate_id": "a",
                }
            ]
        )
    )

    assert calls[0][-1] == "cuda"
    assert model.get_params()["backend"] == "cuda"
    assert model.metadata_["backend"]["selected"] == "cuda"
    assert available_deep_backends() == ["cpu", "cuda"]
    assert backend_dispatch_report("cuda", 4)["accelerated"] is True
