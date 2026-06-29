"""Generic native-backed deep modeling surfaces."""

from ._native import available_deep_backends, backend_dispatch_report
from .decision import ConstrainedDecisionOptimizer
from .frames import DirectionalPairFrame, EntityPanelFrame, GraphTemporalFrame, ResponseCurveFrame
from .graph import SpatioTemporalGraphForecaster
from .response import EventOutcomeModel, ResponseCurveModel, ServiceTimeResidualModel
from .temporal import DirectionalPairForecaster, TemporalEntityTransformer

__all__ = [
    "ConstrainedDecisionOptimizer",
    "DirectionalPairForecaster",
    "DirectionalPairFrame",
    "EntityPanelFrame",
    "EventOutcomeModel",
    "available_deep_backends",
    "backend_dispatch_report",
    "GraphTemporalFrame",
    "ResponseCurveFrame",
    "ResponseCurveModel",
    "ServiceTimeResidualModel",
    "SpatioTemporalGraphForecaster",
    "TemporalEntityTransformer",
]
