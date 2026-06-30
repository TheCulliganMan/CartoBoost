from __future__ import annotations

import json
import types

import cartoboost.deep._native as native_helpers
import numpy as np
from cartoboost.config import GraphBackbone
from cartoboost.deep import (
    ConstrainedDecisionOptimizer,
    DirectionalPairForecaster,
    DirectionalPairFrame,
    EntityPanelFrame,
    EventOutcomeModel,
    ResponseCurveFrame,
    ResponseCurveModel,
    ServiceTimeResidualModel,
    SpatioTemporalGraphForecaster,
    TemporalEntityTransformer,
    available_deep_backends,
    backend_dispatch_report,
)


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

    assert curve[0]["response_score"] > curve[1]["response_score"]
    assert model.best_candidate(ResponseCurveFrame(rows))[0]["candidate_id"] == "a"
    assert model.metadata_["hidden_weights"]


def test_directional_pair_preserves_order_with_native_fit():
    frame = DirectionalPairFrame(
        [
            {"source_id": "A", "target_id": "B", "features": [], "target": 1.0},
            {"source_id": "B", "target_id": "A", "features": [], "target": 2.0},
        ]
    )

    pred = DirectionalPairForecaster().fit(frame).predict(frame)

    assert pred[1] > pred[0]


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
    assert model.metadata_["pair_global_bucket"] == 0
    assert np.isfinite(unseen[0])
    assert pred[0] > pred[12]


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
    assert not np.allclose(pred, tiled_recent_mean)


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


def test_event_residual_and_decision_wrappers_with_native():
    event = EventOutcomeModel().fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0])
    probs = event.predict_proba([[0.0], [3.0]])
    assert probs[1] > probs[0]
    assert event.metadata_["hidden_weights"]

    rows = [
        {"baseline_value": 10.0, "actual_value": 10.0, "features": [0.0]},
        {"baseline_value": 10.0, "actual_value": 13.0, "features": [2.0]},
    ]
    residual = ServiceTimeResidualModel().fit(rows)
    pred = residual.predict(rows)
    assert pred[1] > pred[0]
    assert residual.metadata_["hidden_weights"]

    choice = ConstrainedDecisionOptimizer().select(
        [{"decision_id": "d", "candidate_id": "c", "candidate_value": 1.0, "expected_utility": 3.0}]
    )
    assert choice[0]["candidate_id"] == "c"


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
