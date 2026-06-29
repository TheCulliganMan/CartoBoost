from __future__ import annotations

import json
import types

import cartoboost.deep._native as native_helpers
import numpy as np
from cartoboost.deep import (
    ConstrainedDecisionOptimizer,
    DirectionalPairForecaster,
    DirectionalPairFrame,
    EventOutcomeModel,
    ResponseCurveFrame,
    ResponseCurveModel,
    ServiceTimeResidualModel,
    available_deep_backends,
    backend_dispatch_report,
)


def test_response_curve_monotone_decreasing_with_native_stub(monkeypatch):
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
    artifact = {
        "response_type": "binary",
        "monotone": "decreasing",
        "feature_means": [1.0],
        "feature_weights": [0.0],
        "intercept": 1.0,
        "candidate_slope": -1.0,
    }

    monkeypatch.setattr(
        native_helpers,
        "_native",
        types.SimpleNamespace(
            deep_response_curve_fit_value=lambda *_: json.dumps(artifact),
            deep_response_curve_predict_value=lambda artifact_json, rows_json: json.dumps(
                [
                    {
                        "group_id": row["group_id"],
                        "candidate_id": row["candidate_id"],
                        "candidate_value": row["candidate_value"],
                        "response_score": artifact["intercept"]
                        + artifact["candidate_slope"] * row["candidate_value"],
                        "response_probability": 0.5,
                        "calibrated_probability": 0.5,
                    }
                    for row in json.loads(rows_json)
                ]
            ),
        ),
    )

    model = ResponseCurveModel(response_type="binary", monotone="decreasing")
    model.fit(ResponseCurveFrame(rows))
    curve = model.predict_curve(ResponseCurveFrame(rows))

    assert curve[0]["response_score"] > curve[1]["response_score"]
    assert model.best_candidate(ResponseCurveFrame(rows))[0]["candidate_id"] == "a"


def test_directional_pair_preserves_order_with_native_stub(monkeypatch):
    monkeypatch.setattr(
        native_helpers,
        "_native",
        types.SimpleNamespace(
            deep_directional_pair_predict_value=lambda rows_json: [
                1.0 if row["source_id"] == "A" and row["target_id"] == "B" else 2.0
                for row in json.loads(rows_json)
            ]
        ),
    )
    frame = DirectionalPairFrame(
        [
            {"source_id": "A", "target_id": "B", "features": [], "target": 1.0},
            {"source_id": "B", "target_id": "A", "features": [], "target": 2.0},
        ]
    )

    pred = DirectionalPairForecaster().fit(frame).predict(frame)

    assert pred.tolist() == [1.0, 2.0]


def test_event_residual_and_decision_wrappers_with_native_stub(monkeypatch):
    monkeypatch.setattr(
        native_helpers,
        "_native",
        types.SimpleNamespace(
            deep_event_outcome_fit_value=lambda *_: json.dumps(
                {"model_class": "EventOutcomeModel"}
            ),
            deep_event_outcome_predict_value=lambda *_: json.dumps(
                [{"logit": 0.0, "probability": 0.5, "calibrated_probability": 0.5}]
            ),
            deep_service_residual_fit_value=lambda *_: json.dumps(
                {"model_class": "ServiceTimeResidualModel"}
            ),
            deep_service_residual_predict_value=lambda *_: json.dumps(
                [
                    {
                        "prediction": 11.0,
                        "residual_mean": 1.0,
                        "lower_quantile": 10.0,
                        "upper_quantile": 12.0,
                    }
                ]
            ),
            deep_constrained_decision_select_value=lambda candidates, *_: json.dumps(
                [
                    {
                        "decision_id": json.loads(candidates)[0]["decision_id"],
                        "candidate_id": json.loads(candidates)[0]["candidate_id"],
                        "candidate_value": json.loads(candidates)[0]["candidate_value"],
                        "score": json.loads(candidates)[0]["expected_utility"],
                        "reason_code": "constraints_satisfied",
                    }
                ]
            ),
        ),
    )

    event = EventOutcomeModel().fit([[0.0]], [1.0])
    assert np.allclose(event.predict_proba([[0.0]]), [0.5])

    rows = [{"baseline_value": 10.0, "actual_value": 11.0, "features": [1.0]}]
    residual = ServiceTimeResidualModel().fit(rows)
    assert residual.predict(rows).tolist() == [11.0]

    choice = ConstrainedDecisionOptimizer().select(
        [{"decision_id": "d", "candidate_id": "c", "candidate_value": 1.0, "expected_utility": 3.0}]
    )
    assert choice[0]["candidate_id"] == "c"


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
