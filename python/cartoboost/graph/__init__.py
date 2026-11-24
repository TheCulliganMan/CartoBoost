"""Graph feature subpackage."""

from .builder import (
    HeterogeneousGraph,
    HomogeneousGraph,
    SourceTargetPairNodes,
    materialize_source_target_pair_nodes,
    normalize_heterogeneous_graph,
    normalize_homogeneous_graph,
)
from .config import (
    DirectionalFeature,
    GraphBackend,
    GraphEmbeddingsConfig,
    GraphEncoderConfig,
    GraphEncoderFamily,
    GraphFeatureConfig,
    GraphOutputsConfig,
    GraphTaskMode,
)
from .encoders import (
    GraphFeatureTransformer,
    GraphSageConfig,
    GraphSageFeatureEncoder,
    HeteroGraphSageConfig,
    HeteroGraphSageFeatureEncoder,
    HinSageConfig,
    HinSageFeatureEncoder,
    Node2VecConfig,
    Node2VecFeatureEncoder,
)
from .eval import (
    binary_auc,
    binary_average_precision,
    link_prediction_report,
    mean_reciprocal_rank,
    top_k_metrics,
)
from .features import GraphFeatureBundle
from .schema import (
    DirectedMetaPath,
    DirectionalityConfig,
    EdgeType,
    GraphSchema,
    TemporalEdge,
    as_edge_types,
)
from .walks import MetaPathWalkGenerator, SignedEdgeSampler, TemporalWalkGenerator

_STANDALONE_EXPORTS = {
    "Node2VecStandaloneRegressor",
    "GraphSageStandaloneRegressor",
    "HeteroGraphSageStandaloneRegressor",
    "HinSageStandaloneRegressor",
    "Node2VecLinkPredictor",
    "GraphSageLinkPredictor",
    "HeteroGraphSageLinkPredictor",
    "HinSageLinkPredictor",
}

_NATIVE_ENCODERS = {
    "Node2VecEncoder",
    "GraphSageEncoder",
    "HeteroGraphSageEncoder",
    "HinSageEncoder",
}


def __getattr__(name: str) -> object:
    if name in _NATIVE_ENCODERS:
        from .. import _native

        value = getattr(_native, name)
        globals()[name] = value
        return value
    if name in _STANDALONE_EXPORTS:
        from .. import standalone

        value = getattr(standalone, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EdgeType",
    "DirectionalityConfig",
    "DirectedMetaPath",
    "GraphSchema",
    "GraphFeatureConfig",
    "GraphEmbeddingsConfig",
    "GraphEncoderConfig",
    "GraphEncoderFamily",
    "GraphBackend",
    "GraphTaskMode",
    "GraphOutputsConfig",
    "DirectionalFeature",
    "TemporalEdge",
    "as_edge_types",
    "HomogeneousGraph",
    "HeterogeneousGraph",
    "SourceTargetPairNodes",
    "materialize_source_target_pair_nodes",
    "normalize_homogeneous_graph",
    "normalize_heterogeneous_graph",
    "GraphSageConfig",
    "HeteroGraphSageConfig",
    "HinSageConfig",
    "Node2VecConfig",
    "GraphSageFeatureEncoder",
    "HeteroGraphSageFeatureEncoder",
    "HinSageFeatureEncoder",
    "Node2VecFeatureEncoder",
    "Node2VecEncoder",
    "GraphSageEncoder",
    "HeteroGraphSageEncoder",
    "HinSageEncoder",
    "Node2VecStandaloneRegressor",
    "GraphSageStandaloneRegressor",
    "HeteroGraphSageStandaloneRegressor",
    "HinSageStandaloneRegressor",
    "Node2VecLinkPredictor",
    "GraphSageLinkPredictor",
    "HeteroGraphSageLinkPredictor",
    "HinSageLinkPredictor",
    "GraphFeatureTransformer",
    "GraphFeatureBundle",
    "MetaPathWalkGenerator",
    "TemporalWalkGenerator",
    "SignedEdgeSampler",
    "binary_auc",
    "binary_average_precision",
    "link_prediction_report",
    "top_k_metrics",
    "mean_reciprocal_rank",
]
