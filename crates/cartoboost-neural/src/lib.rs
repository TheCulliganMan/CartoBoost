pub mod artifact;
pub mod backend;
pub mod deep;
pub mod encoder;
mod error;
pub mod features;
pub mod forecasting;
pub mod graph_features;
pub mod graphsage;
pub mod node2vec;
pub mod standalone;
mod trainer;

pub use artifact::{
    build_embedding_table_artifact, write_embedding_table_artifact, ArtifactFallbackKind,
    EmbeddingChecksum, EmbeddingIdType, EmbeddingRow, EmbeddingTable, EmbeddingTableArtifact,
    EmbeddingTableMetadata, FallbackStrategy,
};
pub use backend::{
    available_backends, backend_affine_scores, backend_dense_layer_f32, backend_dispatch_report,
    backend_pair_sigmoid_scores_f32, select_backend, BackendDispatchReport, BackendSelection,
    ComputeBackend,
};
pub use deep::{
    constrained_decision_select, constrained_decision_select_with_options, directional_pair_fit,
    directional_pair_fit_with_options, directional_pair_predict, directional_pair_predictions,
    event_outcome_fit, event_outcome_fit_with_backend, event_outcome_predict, response_curve_fit,
    response_curve_fit_with_backend, response_curve_predict, service_residual_fit,
    service_residual_fit_with_backend, service_residual_predict, temporal_entity_fit,
    temporal_entity_predict, DeepDecisionChoice, DeepDirectionalPairArtifact,
    DeepDirectionalPairRow, DeepEventArtifact, DeepEventPrediction, DeepResponseArtifact,
    DeepResponsePrediction, DeepResponseRow, DeepServiceResidualArtifact,
    DeepServiceResidualPrediction, DeepServiceResidualRow, DeepTemporalEntityArtifact,
    DirectionalPairFitOptions,
};
pub use encoder::{EmbeddingTableEncoder, NeuralEncoder};
pub use error::{NeuralError, Result};
pub use features::NeuralFeatureBlock;
pub use forecasting::{
    ComponentMode, ForecastWindow, LaneNeuralPanelConfig, LaneNeuralPanelForecaster, NBeatsConfig,
    NBeatsForecaster, NHiTSConfig, NHiTSForecaster, NeuralPanelConfig, NeuralPanelForecaster,
    NeuralPanelLoss, NeuralPanelMode, NeuralPanelWindow, NeuralPanelWindowDataset, StandardScaler,
    TrendMode, WindowDataset,
};
pub use graph_features::{
    compute_directional_features, materialize_source_target_pair_nodes, validate_directed_metapath,
    DirectionalFeatureBlock, SourceTargetPairExpansion,
};
pub use graphsage::{
    GraphSageConfig, GraphSageEncoder, GraphSageEncoderArtifact, GraphSageLoss,
    GraphSageModelArtifact, HeteroGraph, HeteroGraphSageConfig, HeteroGraphSageEncoder,
    HeteroGraphSageEncoderArtifact, HeteroTypedEdge, HinSageConfig, HinSageEncoder,
    HinSageEncoderArtifact, HinSageGraph, HomogeneousGraph,
};
pub use node2vec::{
    AliasSampler, EdgeEmbeddingFeatures, EdgeEmbeddingModel, EmbeddingFeatureTransformer,
    Node2VecConfig, Node2VecEncoder, Node2VecEncoderArtifact, Node2VecLoss, Node2VecTrainer,
    RandomWalkGenerator,
};
pub use standalone::{
    GraphRegressionMode, GraphSageLinkPredictor, GraphSageLinkPredictorArtifact,
    GraphSageRegressor, GraphSageRegressorArtifact, HeteroGraphSageLinkPredictor,
    HeteroGraphSageLinkPredictorArtifact, HeteroGraphSageRegressor,
    HeteroGraphSageRegressorArtifact, HinSageLinkPredictor, HinSageLinkPredictorArtifact,
    HinSageRegressor, HinSageRegressorArtifact, NeuralEmbeddingRegressor,
    NeuralEmbeddingRegressorArtifact, Node2VecLinkPredictor, Node2VecLinkPredictorArtifact,
    Node2VecRegressor, Node2VecRegressorArtifact, StandaloneBoosterConfig,
    STANDALONE_ARTIFACT_VERSION,
};
pub use trainer::{fit_embedding_table, fit_embedding_table_with_options};
