from __future__ import annotations

from typing import Any

from cartoboost.forecasting.graph_st import DCRNNForecaster as _DCRNNForecaster
from cartoboost.forecasting.graph_st import GraphWaveNetForecaster as _GraphWaveNetForecaster
from cartoboost.forecasting.graph_st import STAEformerForecaster as _STAEformerForecaster

from ..config import GraphBackbone


class SpatioTemporalGraphForecaster:
    """Generic graph sequence forecaster facade over implemented native graph cores."""

    def __init__(self, *, backbone: GraphBackbone = GraphBackbone.DCRNN, **params: Any) -> None:
        try:
            backbone = GraphBackbone(backbone)
        except ValueError as exc:
            raise ValueError("unknown graph backbone") from exc
        self.backbone = backbone
        if backbone is GraphBackbone.DCRNN:
            self._model = _DCRNNForecaster(
                **{k: v for k, v in params.items() if k in _DCRNN_PARAMS}
            )
        elif backbone is GraphBackbone.GRAPH_WAVENET:
            self._model = _GraphWaveNetForecaster(
                **{k: v for k, v in params.items() if k in _GRAPH_WAVENET_PARAMS}
            )
        else:
            self._model = _STAEformerForecaster(
                **{k: v for k, v in params.items() if k in _STAEFORMER_PARAMS}
            )

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        self._model.fit(*args, **kwargs)
        return self

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(*args, **kwargs)

    def score(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.score(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.save(*args, **kwargs)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"backbone": self.backbone, **self._model.get_params(deep=deep)}

    def set_params(self, **params: Any) -> SpatioTemporalGraphForecaster:
        backbone = params.pop("backbone", self.backbone)
        self.__init__(backbone=backbone, **{**self._model.get_params(), **params})
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        metadata = dict(self._model.metadata_)
        metadata["backbone"] = self.backbone.value
        return metadata


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

_STAEFORMER_PARAMS = {
    "lookback",
    "attention_heads",
    "hidden_size",
    "epochs",
    "learning_rate",
    "ridge",
    "backend",
}

_GRAPH_WAVENET_PARAMS = {
    "lookback",
    "dilation_depth",
    "hidden_size",
    "epochs",
    "learning_rate",
    "ridge",
    "backend",
}
