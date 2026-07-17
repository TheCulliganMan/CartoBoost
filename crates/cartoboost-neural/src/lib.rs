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
pub mod operator;
pub mod standalone;
mod trainer;

pub use artifact::{
    build_embedding_table_artifact, write_embedding_table_artifact, ArtifactFallbackKind,
    EmbeddingChecksum, EmbeddingIdType, EmbeddingRow, EmbeddingTable, EmbeddingTableArtifact,
    EmbeddingTableMetadata, FallbackStrategy,
};
#[cfg(all(feature = "cuda", not(feature = "cuda-oxide"), target_os = "linux"))]
pub use backend::CudaCsrDiffusionWorkspace;
#[cfg(all(feature = "cuda", not(feature = "cuda-oxide"), target_os = "linux"))]
pub use backend::CudaTensorArena;
pub use backend::{
    available_backends, backend_adamw_step_f32, backend_affine_scores,
    backend_csr_diffusion_backward_f32, backend_csr_diffusion_f32,
    backend_csr_row_softmax_backward_f32, backend_csr_row_softmax_f32, backend_dense_layer_f32,
    backend_dispatch_report, backend_layer_norm_f32, backend_pair_sigmoid_scores_f32,
    backend_pairwise_squared_distances_f32, backend_scalar_graph_f32,
    backend_scalar_graph_train_step_f32, backend_supports_operation, backend_train_tanh_mlp_f32,
    backend_workload_decision, masked_inverse_scale_mae_f32, select_backend, select_backend_for,
    select_backend_for_operations, BackendDispatchReport, BackendOperation, BackendSelection,
    BackendWorkloadDecision, ComputeBackend, CsrDiffusionBackward,
};
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub use cartoboost_accelerator::{
    webgpu_adamw_f32_async, webgpu_affine_scores_f32_async,
    webgpu_csr_diffusion_backward_f32_async, webgpu_csr_diffusion_f32_async,
    webgpu_csr_row_softmax_backward_f32_async, webgpu_csr_row_softmax_f32_async,
    webgpu_dense_layer_f32_async, webgpu_dispatch_report_async, webgpu_layer_norm_f32_async,
    webgpu_pair_sigmoid_scores_f32_async, webgpu_pairwise_squared_distances_f32_async,
    webgpu_scalar_graph_f32_async, webgpu_scalar_graph_train_step_f32_async,
    webgpu_train_tanh_mlp_f32_async,
};
#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
pub use cartoboost_accelerator::{CudaCsrDiffusionWorkspace, CudaTensorArena};
pub use deep::{
    choice_set_transformer_report, choice_set_transformer_report_json,
    choice_set_transformer_report_json_with_backend, choice_set_transformer_report_with_backend,
    constrained_decision_select, constrained_decision_select_with_options, directional_pair_fit,
    directional_pair_fit_with_options, directional_pair_fit_with_options_and_backend,
    directional_pair_predict, directional_pair_predictions, event_outcome_fit,
    event_outcome_fit_with_backend, event_outcome_predict, response_curve_fit,
    response_curve_fit_with_backend, response_curve_predict, service_residual_fit,
    service_residual_fit_with_backend, service_residual_predict, temporal_entity_fit,
    temporal_entity_fit_with_backend, temporal_entity_predict, ChoiceSetTransformer,
    CounterfactualCandidateScorer, DeepChoiceSetPrediction, DeepChoiceSetReport,
    DeepCounterfactualCandidate, DeepDecisionChoice, DeepDirectionalPairArtifact,
    DeepDirectionalPairRow, DeepEventArtifact, DeepEventPrediction, DeepResponseArtifact,
    DeepResponsePrediction, DeepResponseRow, DeepServiceResidualArtifact,
    DeepServiceResidualPrediction, DeepServiceResidualRow, DeepTemporalEntityArtifact,
    DirectionalPairFitOptions, NestedChoiceHead, UtilityNet,
};
pub use encoder::{EmbeddingTableEncoder, NeuralEncoder};
pub use error::{NeuralError, Result};
pub use features::NeuralFeatureBlock;
pub use forecasting::{
    fit_dense_regressor, fit_dense_regressor_with_backend, ComponentMode, DenseRegressorConfig,
    ForecastWindow, LaneNeuralPanelConfig, LaneNeuralPanelForecaster, MlpState, NBeatsConfig,
    NBeatsForecaster, NHiTSConfig, NHiTSForecaster, NeuralPanelConfig, NeuralPanelForecaster,
    NeuralPanelLoss, NeuralPanelMode, NeuralPanelWindow, NeuralPanelWindowDataset, StandardScaler,
    TrendMode, WindowDataset,
};
pub use graph_features::{
    compute_directional_features, compute_directional_features_with_backend,
    materialize_source_target_pair_nodes, validate_directed_metapath, DirectionalFeatureBlock,
    SourceTargetPairExpansion,
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
pub use operator::{
    graph_neural_operator_predict_json, neural_operator_synthetic_benchmark_json,
    FourierGeoOperator, GraphNeuralOperator, NeuralOperatorPrediction,
    NeuralOperatorSyntheticBenchmark, SpatialOperatorEdge, SpatioTemporalOperator,
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
pub use trainer::{
    fit_embedding_table, fit_embedding_table_with_options,
    fit_embedding_table_with_options_and_backend,
};
