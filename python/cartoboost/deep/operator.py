from __future__ import annotations

import json
from typing import Any

import numpy as np

from ._native import require_native


class GraphNeuralOperator:
    """Advanced experimental native-backed spatial field operator."""

    capability_tier = "advanced_experimental"

    def __init__(self, *, smoothing: float = 0.35, coordinate_scale: float = 0.08) -> None:
        self.smoothing = float(smoothing)
        self.coordinate_scale = float(coordinate_scale)
        self.metadata_ = {"capability_tier": self.capability_tier}

    def predict(
        self,
        *,
        field_values: Any,
        coordinates: Any,
        edges: Any,
        exogenous_fields: Any | None = None,
    ) -> dict[str, Any]:
        fields = _matrix(field_values, "field_values")
        coords = _matrix(coordinates, "coordinates")
        exogenous = (
            np.empty((0, 0), dtype=float)
            if exogenous_fields is None
            else _matrix(exogenous_fields, "exogenous_fields")
        )
        predict = require_native("deep_graph_neural_operator_predict_value")
        output = json.loads(
            predict(
                json.dumps(fields.tolist()),
                json.dumps(coords.tolist()),
                _edges(edges),
                json.dumps(exogenous.tolist()),
                self.smoothing,
                self.coordinate_scale,
            )
        )
        self.metadata_ = dict(output["metadata"])
        return {
            "future_field": np.asarray(output["future_field"], dtype=float),
            "residual_field": np.asarray(output["residual_field"], dtype=float),
            "uncertainty_field": np.asarray(output["uncertainty_field"], dtype=float),
            "metadata": dict(output["metadata"]),
        }

    @staticmethod
    def synthetic_benchmark() -> dict[str, Any]:
        benchmark = require_native("deep_neural_operator_synthetic_benchmark_value")
        return json.loads(benchmark())


FourierGeoOperator = GraphNeuralOperator
SpatioTemporalOperator = GraphNeuralOperator


def _matrix(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 2D array")
    return arr


def _edges(values: Any) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for edge in values:
        if isinstance(edge, dict):
            source = edge["source"]
            target = edge["target"]
            weight = edge.get("weight", 1.0)
        else:
            source, target, weight = edge
        weight = float(weight)
        if not np.isfinite(weight):
            raise ValueError("edge weights must be finite")
        out.append((int(source), int(target), weight))
    return out
