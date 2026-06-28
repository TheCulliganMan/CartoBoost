from __future__ import annotations

from typing import Any

from cartoboost.forecasting.graph_st import DCRNNForecaster as _DCRNNForecaster

from ..config import GraphBackbone


class SpatioTemporalGraphForecaster(_DCRNNForecaster):
    """Generic graph sequence forecaster facade over the native DCRNN core."""

    def __init__(self, *, backbone: GraphBackbone = GraphBackbone.DCRNN, **params: Any) -> None:
        if backbone not in {
            GraphBackbone.DCRNN,
            GraphBackbone.GRAPH_WAVENET,
            GraphBackbone.TEMPORAL_GRAPH_ATTENTION,
        }:
            raise ValueError("unknown graph backbone")
        self.backbone = backbone
        super().__init__(**{k: v for k, v in params.items() if k in _DCRNN_PARAMS})


_DCRNN_PARAMS = {
    "diffusion_steps",
    "hidden_size",
    "epochs",
    "learning_rate",
    "teacher_forcing_start",
    "teacher_forcing_end",
    "ridge",
    "backend",
}
