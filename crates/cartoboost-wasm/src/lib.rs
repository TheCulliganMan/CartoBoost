use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::data::{Dataset, FeatureKind, FeatureSchema, SparseSetColumn};
use cartoboost_core::forecasting::{
    crossing_rate, default_quantile_levels, interval_diagnostics, pinball_loss,
    regime_adjusted_intervals, repair_non_crossing_quantiles, rolling_mad_residual,
    rolling_median_residual, AutoARIMAForecaster, AutoETSForecaster, AutoForecastConfig,
    AutoForecastModel, AutoKalmanForecaster, AutoLocalLevelKalmanForecaster, AutoStatsBank,
    CalendarFeature, CartoBoostDirectForecaster, CartoBoostLagForecaster, ClassicalExpertBank,
    CrostonForecaster, CusumConfig, ETSForecaster, EwmaVolatility, EwmaVolatilityConfig,
    ForecastFrame, ForecastFrameMetadata, ForecastFrequency, ForecastResult, Forecaster,
    IntermittentDemandConfig, IntermittentDemandForecaster, KalmanForecaster,
    KalmanResidualCorrector, KrigingForecaster, LagFeatureConfig, LagPlusConfig, LagPlusForecaster,
    LocalLevelKalmanForecaster, LocalStandardScaledForecaster, Log1pForecaster,
    MSTLCartoBoostForecaster, NaiveForecaster, OptimizedThetaForecaster, PageHinkley,
    PageHinkleyConfig, PiecewiseLinearComponentMode, PiecewiseLinearEvent, PiecewiseLinearFitLoss,
    PiecewiseLinearGrowth, PiecewiseLinearSeasonalConfig, PiecewiseLinearSeasonalForecaster,
    PiecewiseLinearSeasonality, RectifiedRecursiveForecaster, ReferencePathConfig, ReferenceSignal,
    RegimeIntervalPolicy, ResidualStateKey, STLCartoBoostForecaster, SbaForecaster,
    SeasonalNaiveForecaster, SeasonalWindowAverageForecaster, SequenceCandidate,
    SequenceCandidateEnsemble, SequenceCandidatePrediction, SequenceFrame, SequenceGroupPrediction,
    SequenceOofCandidateRow, SequenceOofFold, SequenceSeries, SequenceStateSpaceConfig,
    SpatialPiecewiseKrigingConfig, SpatialPiecewiseKrigingForecaster, SpatialPiecewiseKrigingMode,
    StateFilter, StateObservation, ThetaForecaster, ThetaSeasonality, TsbForecaster,
    WindowAverageForecaster, CUSUM,
};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::objectives::{
    calibration_improvement, calibration_metrics, escalation_risk_event, event_within_horizon,
    failure_risk_event, success_within_threshold, IsotonicCalibrator, ProbabilityCalibrator,
    SigmoidCalibrator, TemperatureCalibrator,
};
use cartoboost_core::tree::{Node, Split, SplitterKind};
use cartoboost_core::Booster;
use cartoboost_core::{CartoBoostError, Result};
use cartoboost_geo_causal::{
    GeoCausalPanel, GeoCausalRow, SpatialWeight, SyntheticDIDConfig,
    SyntheticDIDEstimator as CoreSyntheticDIDEstimator,
};
use cartoboost_geo_core::{
    clockwise_bearing_unit_vector, initial_bearing_unit_vector_latlng, local_frame_features,
    radial_anchor_distances, rbf_anchor_features, route_feature_vector,
    SplitManifest as GeoCoreSplitManifest,
};
use cartoboost_geo_st::{
    graph_metrics as graph_st_metrics, select_compute_backend as graph_st_select_backend,
    CsrAdjacency as GraphStCsrAdjacency, DcrnnConfig as GraphStDcrnnConfig,
    DcrnnForecaster as GraphStDcrnnForecaster, GraphTemporalFrame as GraphStTemporalFrame,
    GraphTransformerProfile as BrowserGraphTransformerProfile,
    MarketPanelFrame as BrowserMarketPanelFrame,
    MarketStructureConfig as BrowserMarketStructureConfig,
    MarketStructureForecaster as BrowserMarketStructureForecaster,
    PaperGraphTransformerConfig as BrowserPaperGraphTransformerConfig,
    PaperGraphTransformerForecaster as BrowserPaperGraphTransformerForecaster,
};
use cartoboost_geostats::{
    Anisotropy as GeostatsAnisotropy, CovarianceKernel,
    NearestNeighborGPRegressor as WasmNearestNeighborGPRegressor, NngpConfig,
};
use cartoboost_neural::{
    available_backends as deep_available_backends,
    choice_set_transformer_report_json as deep_choice_set_transformer_report_json,
    constrained_decision_select as deep_constrained_decision_select,
    directional_pair_predictions as deep_directional_pair_predictions,
    event_outcome_fit_with_backend as deep_event_outcome_fit,
    event_outcome_predict as deep_event_outcome_predict,
    graph_neural_operator_predict_json as deep_graph_neural_operator_predict_json,
    neural_operator_synthetic_benchmark_json as deep_neural_operator_synthetic_benchmark_json,
    response_curve_fit_with_backend as deep_response_curve_fit,
    response_curve_predict as deep_response_curve_predict,
    service_residual_fit_with_backend as deep_service_residual_fit,
    service_residual_predict as deep_service_residual_predict,
    temporal_entity_fit as deep_temporal_entity_fit,
    temporal_entity_predict as deep_temporal_entity_predict, ArtifactFallbackKind,
    BackendSelection, ComponentMode as NeuralComponentMode, DeepDirectionalPairRow,
    DeepEventArtifact, DeepResponseArtifact, DeepResponseRow, DeepServiceResidualArtifact,
    DeepServiceResidualRow, DeepTemporalEntityArtifact, GraphSageConfig, GraphSageRegressor,
    HeteroGraphSageConfig, HeteroGraphSageRegressor, HinSageConfig, HinSageRegressor, NBeatsConfig,
    NBeatsForecaster, NHiTSConfig, NHiTSForecaster, NeuralEmbeddingRegressor, NeuralPanelConfig,
    NeuralPanelForecaster, NeuralPanelLoss, NeuralPanelMode, Node2VecConfig, Node2VecRegressor,
    SpatialOperatorEdge as DeepSpatialOperatorEdge, StandaloneBoosterConfig,
    TrendMode as NeuralTrendMode,
};
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use cartoboost_neural::{webgpu_dense_layer_f32_async, webgpu_dispatch_report_async};
use cartoboost_prob::{
    conditional_flow_fit_json as deep_conditional_flow_fit_json,
    conditional_flow_predict_json as deep_conditional_flow_predict_json,
    diffusion_scenario_generate_json as deep_diffusion_scenario_generate_json,
    ConditionalFlowDistributionHead as DeepConditionalFlowDistributionHead,
    DiffusionEdge as DeepDiffusionEdge,
};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use wasm_bindgen::prelude::*;

type BrowserNeuralPipelineOutput = (Vec<f64>, Vec<String>, Vec<cartoboost_core::Tree>, Value);

// Browser API families share one namespace; exports remain unchanged.
include!("browser/types.rs");
include!("browser/exports.rs");
include!("browser/geo.rs");
include!("browser/forecast.rs");
include!("browser/modeling.rs");
include!("browser/forecaster_factory.rs");
include!("browser/tests.rs");
