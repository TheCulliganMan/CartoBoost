from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import Backend
from ._native import require_native


class GeoTemporalDiffusionScenarioModel:
    """Experimental Rust-backed graph diffusion scenario generator."""

    capability_tier = "experimental"

    def __init__(
        self,
        *,
        scenario_count: int = 64,
        diffusion_steps: int = 2,
        shock_scale: float = 1.0,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        self.scenario_count = int(scenario_count)
        self.diffusion_steps = int(diffusion_steps)
        self.shock_scale = float(shock_scale)
        self.backend = str(backend)
        self.metadata_ = {
            "capability_tier": self.capability_tier,
            "auto_geo_enabled": "false",
            "primary_benchmark_evidence": "false",
        }

    def generate(self, point_forecast: Any, edges: Any) -> dict[str, Any]:
        panel = _panel(point_forecast, "point_forecast")
        edge_list = _edges(edges)
        generate = require_native("prob_diffusion_scenario_generate_value")
        output = json.loads(
            generate(
                panel.tolist(),
                edge_list,
                self.scenario_count,
                self.diffusion_steps,
                self.shock_scale,
                self.backend,
            )
        )
        self.metadata_ = dict(output["metadata"])
        self.backend_ = str(self.metadata_["backend_selected"])
        return {
            "scenarios": np.asarray(output["scenarios"], dtype=float),
            "scenario_mean": np.asarray(output["scenario_mean"], dtype=float),
            "scenario_variance": np.asarray(output["scenario_variance"], dtype=float),
            "spatial_correlation": float(output["spatial_correlation"]),
            "point_forecast_comparison": dict(output["point_forecast_comparison"]),
            "metadata": dict(output["metadata"]),
        }


FlowScenarioGenerator = GeoTemporalDiffusionScenarioModel
ConditionalResidualDiffusion = GeoTemporalDiffusionScenarioModel


def _panel(values: Any, name: str) -> np.ndarray:
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
