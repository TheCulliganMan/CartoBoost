"""Unstable research adapters kept outside the stable CartoBoost API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentalAdapterSpec:
    name: str
    family: str
    stability: str = "experimental"
    notes: str = ""


def available_adapters() -> tuple[ExperimentalAdapterSpec, ...]:
    return (
        ExperimentalAdapterSpec(
            "foundation_time_series_adapter",
            "foundation_time_series",
            notes="Adapter scaffold for external foundation time-series models.",
        ),
        ExperimentalAdapterSpec(
            "raster_embedding_adapter",
            "remote_sensing",
            notes="Adapter scaffold for raster or remote-sensing embeddings.",
        ),
        ExperimentalAdapterSpec(
            "neural_operator_adapter",
            "neural_operator",
            notes="Scaffold for spatial neural operator experiments.",
        ),
        ExperimentalAdapterSpec(
            "transformer_graph_forecaster",
            "graph_transformer",
            notes="Scaffold for transformer graph forecasting experiments.",
        ),
    )


class ExperimentalAdapter:
    """Explicit placeholder for research integrations that are not stable API."""

    def __init__(self, spec: ExperimentalAdapterSpec, backend: Any | None = None) -> None:
        self.spec = spec
        self.backend = backend

    def fit(self, *args: Any, **kwargs: Any) -> ExperimentalAdapter:
        if self.backend is None:
            raise NotImplementedError(
                f"{self.spec.name} is experimental and requires an explicit backend"
            )
        self.backend.fit(*args, **kwargs)
        return self

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        if self.backend is None:
            raise NotImplementedError(
                f"{self.spec.name} is experimental and requires an explicit backend"
            )
        return self.backend.predict(*args, **kwargs)


__all__ = ["ExperimentalAdapter", "ExperimentalAdapterSpec", "available_adapters"]
