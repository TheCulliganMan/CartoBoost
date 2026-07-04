mod implementation;

use serde::{Deserialize, Serialize};

pub mod artifact;
pub mod entity;
pub mod graph_context;
pub mod pair;
pub mod pretraining;
pub mod regime;
pub mod retrieval;
pub mod spatiotemporal;

pub use artifact::{
    resolve_backend, BackendMetadata, RepresentationArtifact, RepresentationError, Result,
    REPRESENTATION_ARTIFACT_VERSION,
};
pub use entity::EntityEmbedding;
pub use graph_context::{MultiViewAttentionOutput, MultiViewSpatialAttention, ViewAblationReport};
pub use pair::PairEmbedding;
pub use pretraining::{
    SelfSupervisedPretrainer, FUTURE_PATCH_RECONSTRUCTION, GRAPH_EDGE_DENOISING,
    MASKED_ENTITY_TIME_MODELING, MASKED_PAIR_TIME_MODELING, SPATIAL_NEIGHBOR_CONTRASTIVE_LOSS,
    TEMPORAL_ORDER_CONTRASTIVE_LOSS,
};
pub use regime::{RegimeRoute, RegimeRouter};
pub use retrieval::{AnalogQueryResult, HistoricalAnalogRetriever};
pub use spatiotemporal::{
    EntityTimeAdaptiveEmbedding, NodeTimeAdaptiveEmbedding, PairTimeAdaptiveEmbedding,
    SpatioTemporalAdaptiveEmbedding,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaskedEntityTimeModeling;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaskedPairTimeModeling;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphEdgeDenoising;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalOrderContrastiveLoss;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpatialNeighborContrastiveLoss;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FuturePatchReconstruction;
