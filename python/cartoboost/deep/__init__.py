"""Generic native-backed deep modeling surfaces."""

from ._native import available_deep_backends, backend_dispatch_report
from .choice import (
    ChoiceSetTransformer,
    CounterfactualCandidateScorer,
    NestedChoiceHead,
    UtilityNet,
)
from .decision import ConstrainedDecisionOptimizer
from .flow import ConditionalFlowDistributionHead, JointHorizonFlowHead, ResidualFlowCalibrator
from .frames import DirectionalPairFrame, EntityPanelFrame, GraphTemporalFrame, ResponseCurveFrame
from .graph import (
    DelayAwareGraphTransformer,
    DynamicAdjacencyTransformer,
    PropagationDelayGraphForecaster,
    SpatioTemporalGraphForecaster,
)
from .operator import FourierGeoOperator, GraphNeuralOperator, SpatioTemporalOperator
from .regime import (
    EntityRegimeRouter,
    GeoTemporalMixtureOfExperts,
    PairRegimeRouter,
    RegimeMoEForecaster,
)
from .response import EventOutcomeModel, ResponseCurveModel, ServiceTimeResidualModel
from .scenario import (
    ConditionalResidualDiffusion,
    FlowScenarioGenerator,
    GeoTemporalDiffusionScenarioModel,
)
from .ssm import (
    EntityTemporalSSM,
    GraphTemporalSSM,
    PairTemporalSSM,
    SelectiveStateSpaceBlock,
    TemporalSSMForecaster,
)
from .temporal import (
    DirectionalPairForecaster,
    InvertedEntityTransformer,
    InvertedTemporalTransformer,
    TemporalEntityTransformer,
)

__all__ = [
    "ConstrainedDecisionOptimizer",
    "ConditionalFlowDistributionHead",
    "ConditionalResidualDiffusion",
    "ChoiceSetTransformer",
    "CounterfactualCandidateScorer",
    "DirectionalPairForecaster",
    "DirectionalPairFrame",
    "EntityPanelFrame",
    "EventOutcomeModel",
    "FourierGeoOperator",
    "available_deep_backends",
    "backend_dispatch_report",
    "GraphTemporalFrame",
    "GeoTemporalDiffusionScenarioModel",
    "GraphNeuralOperator",
    "JointHorizonFlowHead",
    "NestedChoiceHead",
    "ResponseCurveFrame",
    "ResponseCurveModel",
    "FlowScenarioGenerator",
    "ResidualFlowCalibrator",
    "ServiceTimeResidualModel",
    "SelectiveStateSpaceBlock",
    "SpatioTemporalGraphForecaster",
    "SpatioTemporalOperator",
    "TemporalSSMForecaster",
    "UtilityNet",
    "TemporalEntityTransformer",
    "EntityTemporalSSM",
    "PairTemporalSSM",
    "GraphTemporalSSM",
    "InvertedEntityTransformer",
    "InvertedTemporalTransformer",
    "DelayAwareGraphTransformer",
    "DynamicAdjacencyTransformer",
    "PropagationDelayGraphForecaster",
    "RegimeMoEForecaster",
    "GeoTemporalMixtureOfExperts",
    "PairRegimeRouter",
    "EntityRegimeRouter",
]
