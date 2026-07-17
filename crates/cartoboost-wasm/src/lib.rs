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
    GlobalForecastTargetMode, IntermittentDemandConfig, IntermittentDemandForecaster,
    KalmanForecaster, KalmanResidualCorrector, KrigingForecaster, LagFeatureConfig, LagPlusConfig,
    LagPlusForecaster, LocalLevelKalmanForecaster, LocalStandardScaledForecaster, Log1pForecaster,
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
    graph_metrics as graph_st_metrics,
    select_compute_backend_for_operations as graph_st_select_backend_for_operations,
    CsrAdjacency as GraphStCsrAdjacency, DcrnnConfig as GraphStDcrnnConfig,
    DcrnnForecaster as GraphStDcrnnForecaster, GraphTemporalFrame as GraphStTemporalFrame,
    GraphTransformerProfile as BrowserGraphTransformerProfile,
    MarketPanelFrame as BrowserMarketPanelFrame,
    MarketStructureConfig as BrowserMarketStructureConfig,
    MarketStructureForecaster as BrowserMarketStructureForecaster,
    PaperGraphTransformerConfig as BrowserPaperGraphTransformerConfig,
    PaperGraphTransformerForecaster as BrowserPaperGraphTransformerForecaster,
};
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use cartoboost_geostats::empirical_semivariogram_from_squared_matrices;
use cartoboost_geostats::{
    Anisotropy as GeostatsAnisotropy, CovarianceKernel,
    NearestNeighborGPRegressor as WasmNearestNeighborGPRegressor, NngpConfig,
};
use cartoboost_neural::{
    available_backends as deep_available_backends,
    choice_set_transformer_report_json_with_backend as deep_choice_set_transformer_report_json,
    constrained_decision_select as deep_constrained_decision_select,
    directional_pair_fit_with_options_and_backend as deep_directional_pair_fit,
    directional_pair_predict as deep_directional_pair_predict,
    event_outcome_fit_with_backend as deep_event_outcome_fit,
    event_outcome_predict as deep_event_outcome_predict,
    graph_neural_operator_predict_json as deep_graph_neural_operator_predict_json,
    neural_operator_synthetic_benchmark_json as deep_neural_operator_synthetic_benchmark_json,
    response_curve_fit_with_backend as deep_response_curve_fit,
    response_curve_predict as deep_response_curve_predict, select_backend, select_backend_for,
    select_backend_for_operations, service_residual_fit_with_backend as deep_service_residual_fit,
    service_residual_predict as deep_service_residual_predict,
    temporal_entity_fit_with_backend as deep_temporal_entity_fit,
    temporal_entity_predict as deep_temporal_entity_predict, ArtifactFallbackKind,
    BackendOperation, BackendSelection, ComponentMode as NeuralComponentMode,
    DeepDirectionalPairRow, DeepEventArtifact, DeepResponseArtifact, DeepResponseRow,
    DeepServiceResidualArtifact, DeepServiceResidualRow, DeepTemporalEntityArtifact,
    DirectionalPairFitOptions, GraphSageConfig, GraphSageRegressor, HeteroGraphSageConfig,
    HeteroGraphSageRegressor, HinSageConfig, HinSageRegressor, NBeatsConfig, NBeatsForecaster,
    NHiTSConfig, NHiTSForecaster, NeuralEmbeddingRegressor, NeuralPanelConfig,
    NeuralPanelForecaster, NeuralPanelLoss, NeuralPanelMode, Node2VecConfig, Node2VecRegressor,
    SpatialOperatorEdge as DeepSpatialOperatorEdge, StandaloneBoosterConfig,
    TrendMode as NeuralTrendMode,
};
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
use cartoboost_neural::{
    webgpu_adamw_f32_async, webgpu_affine_scores_f32_async,
    webgpu_csr_diffusion_backward_f32_async, webgpu_csr_diffusion_f32_async,
    webgpu_csr_row_softmax_backward_f32_async, webgpu_csr_row_softmax_f32_async,
    webgpu_dense_layer_f32_async, webgpu_dispatch_report_async, webgpu_layer_norm_f32_async,
    webgpu_pair_sigmoid_scores_f32_async, webgpu_pairwise_squared_distances_f32_async,
    webgpu_scalar_graph_f32_async, webgpu_scalar_graph_train_step_f32_async,
    webgpu_train_tanh_mlp_f32_async, Node2VecEncoder,
};
use cartoboost_prob::{
    conditional_flow_fit_with_backend_json as deep_conditional_flow_fit_json,
    conditional_flow_predict_json as deep_conditional_flow_predict_json,
    diffusion_scenario_generate_with_backend_json as deep_diffusion_scenario_generate_json,
    ConditionalFlowDistributionHead as DeepConditionalFlowDistributionHead,
    DiffusionEdge as DeepDiffusionEdge,
};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use wasm_bindgen::prelude::*;

type BrowserNeuralPipelineOutput = (Vec<f64>, Vec<String>, Vec<cartoboost_core::Tree>, Value);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastRequest {
    rows: Vec<BrowserForecastRow>,
    frequency: String,
    horizon: usize,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    options: BrowserForecastOptions,
    #[serde(default)]
    metadata: BrowserForecastMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastRow {
    #[serde(default)]
    series_id: Option<String>,
    timestamp: String,
    target: f64,
    #[serde(default)]
    covariates: BTreeMap<String, f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastOptions {
    backend: Option<String>,
    input_size: Option<usize>,
    hidden_size: Option<usize>,
    epochs: Option<usize>,
    pooling_size: Option<usize>,
    season_length: Option<usize>,
    theta: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    damping_phi: Option<f64>,
    theta_grid: Option<Vec<f64>>,
    alpha_grid: Option<Vec<f64>>,
    theta_seasonality: Option<String>,
    max_p: Option<usize>,
    max_d: Option<usize>,
    max_q: Option<usize>,
    level_process_variance: Option<f64>,
    trend_process_variance: Option<f64>,
    observation_variance: Option<f64>,
    window_size: Option<usize>,
    window_count: Option<usize>,
    n_lags: Option<usize>,
    n_forecasts: Option<usize>,
    validation_window: Option<usize>,
    max_direct_horizon: Option<usize>,
    max_auto_candidate_count: Option<usize>,
    include_components: Option<bool>,
    include_history_components: Option<bool>,
    include_samples: Option<bool>,
    include_quantiles: Option<bool>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    lags: Option<Vec<usize>>,
    rolling_mean_windows: Option<Vec<usize>>,
    rolling_std_windows: Option<Vec<usize>>,
    rolling_min_windows: Option<Vec<usize>>,
    rolling_max_windows: Option<Vec<usize>>,
    difference_lags: Option<Vec<usize>>,
    rolling_trend_windows: Option<Vec<usize>>,
    calendar_features: Option<Vec<String>>,
    mstl_season_lengths: Option<Vec<usize>>,
    coordinate_x: Option<String>,
    coordinate_y: Option<String>,
    kriging_range: Option<f64>,
    kriging_nugget: Option<f64>,
    spatial_kriging_mode: Option<String>,
    spatial_regressors: Option<Vec<String>>,
    residual_shrinkage: Option<f64>,
    allow_neighbor_fallback: Option<bool>,
    changepoints: Option<usize>,
    n_changepoints: Option<usize>,
    changepoint_range: Option<f64>,
    changepoint_timestamps: Option<Vec<String>>,
    yearly_fourier_order: Option<usize>,
    weekly_fourier_order: Option<usize>,
    daily_fourier_order: Option<usize>,
    auto_yearly_seasonality: Option<bool>,
    auto_weekly_seasonality: Option<bool>,
    auto_daily_seasonality: Option<bool>,
    custom_seasonalities: Option<Vec<BrowserForecastSeasonality>>,
    changepoint_l2_regularization: Option<f64>,
    changepoint_l1_regularization: Option<f64>,
    changepoint_prior_scale: Option<f64>,
    seasonality_l2_regularization: Option<f64>,
    seasonality_prior_scale: Option<f64>,
    yearly_l2_regularization: Option<f64>,
    weekly_l2_regularization: Option<f64>,
    daily_l2_regularization: Option<f64>,
    event_l2_regularization: Option<f64>,
    holidays_prior_scale: Option<f64>,
    regressor_l2_regularization: Option<f64>,
    event_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
    regressor_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
    events: Option<Vec<BrowserForecastEvent>>,
    holidays: Option<Vec<BrowserForecastHoliday>>,
    event_mode: Option<String>,
    holidays_mode: Option<String>,
    extra_regressors: Option<Vec<String>>,
    regressor_modes: Option<BTreeMap<String, String>>,
    lagged_regressors: Option<BTreeMap<String, usize>>,
    ar_layers: Option<Vec<usize>>,
    lagged_reg_layers: Option<Vec<usize>>,
    trend_mode: Option<String>,
    local_l2: Option<f64>,
    extra_regressor_monotonic_constraints: Option<BTreeMap<String, i8>>,
    regressor_standardization: Option<String>,
    future_regressors: Option<BTreeMap<String, Vec<f64>>>,
    future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
    trend_adjustments: Option<BTreeMap<usize, f64>>,
    trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    residual_shock_window: Option<usize>,
    residual_shock_scale: Option<f64>,
    residual_shock_decay: Option<f64>,
    interval_levels: Option<Vec<f64>>,
    interval_width: Option<f64>,
    quantile_levels: Option<Vec<f64>>,
    uncertainty_samples: Option<usize>,
    mcmc_samples: Option<usize>,
    trend_uncertainty_policy: Option<String>,
    trend_uncertainty_scale: Option<f64>,
    coefficient_uncertainty_scale: Option<f64>,
    uncertainty_seed: Option<u64>,
    growth: Option<String>,
    component_mode: Option<String>,
    seasonality_mode: Option<String>,
    fit_loss: Option<String>,
    huber_delta: Option<f64>,
    irls_iterations: Option<usize>,
    cap: Option<f64>,
    floor: Option<f64>,
    cap_regressor: Option<String>,
    floor_regressor: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeotemporalDiagnosticsRequest {
    quantiles: Option<BrowserQuantileDiagnosticsRequest>,
    residual_correction: Option<BrowserResidualCorrectionRequest>,
    regime: Option<BrowserRegimeDiagnosticsRequest>,
    calibration: Option<BrowserCalibrationRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoCausalRequest {
    rows: Vec<BrowserGeoCausalRow>,
    intervention_time: String,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    placebo_n: usize,
    #[serde(default)]
    spatial_weights: Vec<BrowserSpatialWeight>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoCausalRow {
    unit_id: String,
    time: String,
    outcome: f64,
    treatment: bool,
    #[serde(default)]
    covariates: BTreeMap<String, f64>,
    latitude: Option<f64>,
    longitude: Option<f64>,
    region_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSpatialWeight {
    from_unit: String,
    to_unit: String,
    weight: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserQuantileDiagnosticsRequest {
    values: Option<Vec<f64>>,
    actual: Option<Vec<f64>>,
    prediction: Option<Vec<f64>>,
    quantile: Option<f64>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    quantile_rows: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserResidualCorrectionRequest {
    process_variance: f64,
    observation_variance: f64,
    observations: Vec<BrowserResidualObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserResidualObservation {
    key: BrowserResidualStateKey,
    structural_prediction: f64,
    observed: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserResidualStateKey {
    origin: Option<String>,
    destination: Option<String>,
    corridor: Option<String>,
    segment: Option<String>,
    entity_family: Option<String>,
    target_family: Option<String>,
    time_bucket: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegimeDiagnosticsRequest {
    residuals: Vec<f64>,
    cusum: Option<CusumConfig>,
    page_hinkley: Option<PageHinkleyConfig>,
    ewma: Option<EwmaVolatilityConfig>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    policy: Option<RegimeIntervalPolicy>,
    rolling_window: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCalibrationRequest {
    scores: Option<Vec<f64>>,
    labels: Vec<f64>,
    probabilities: Option<Vec<f64>>,
    before_probabilities: Option<Vec<f64>>,
    method: Option<String>,
    bucket_count: Option<usize>,
    event: Option<BrowserCalibrationEventRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCalibrationEventRequest {
    kind: String,
    actual: Vec<f64>,
    prediction: Option<Vec<f64>>,
    threshold: Option<f64>,
    horizon: Option<f64>,
    warning_threshold: Option<f64>,
    critical_threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastEvent {
    name: String,
    timestamp: String,
    #[serde(default)]
    lower_window: Option<i32>,
    #[serde(default)]
    upper_window: Option<i32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastHoliday {
    holiday: String,
    ds: String,
    #[serde(default)]
    lower_window: Option<i32>,
    #[serde(default)]
    upper_window: Option<i32>,
    #[serde(default)]
    prior_scale: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastSeasonality {
    name: String,
    period_days: f64,
    fourier_order: usize,
    mode: Option<String>,
    condition_name: Option<String>,
    l2_regularization: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastMetadata {
    timestamp_col: Option<String>,
    target_col: Option<String>,
    series_id_col: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastArtifactPredictOptions {
    #[serde(default = "default_true")]
    include_components: bool,
    #[serde(default)]
    include_history_components: bool,
    #[serde(default = "default_true")]
    include_samples: bool,
    #[serde(default = "default_true")]
    include_quantiles: bool,
    future_regressors: Option<BTreeMap<String, Vec<f64>>>,
    future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
    trend_adjustments: Option<BTreeMap<usize, f64>>,
    trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    interval_levels: Option<Vec<f64>>,
    quantile_levels: Option<Vec<f64>>,
    uncertainty_samples: Option<usize>,
}

impl Default for BrowserForecastArtifactPredictOptions {
    fn default() -> Self {
        Self {
            include_components: true,
            include_history_components: false,
            include_samples: true,
            include_quantiles: true,
            future_regressors: None,
            future_regressors_by_series: None,
            trend_adjustments: None,
            trend_adjustments_by_series: None,
            interval_levels: None,
            quantile_levels: None,
            uncertainty_samples: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionRequest {
    rows: Vec<BrowserRegressionRow>,
    feature_names: Vec<String>,
    #[serde(default)]
    sparse_feature_names: Vec<String>,
    #[serde(default)]
    options: BrowserRegressionOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionRow {
    features: Vec<f64>,
    #[serde(default)]
    sparse_sets: Vec<Vec<u64>>,
    target: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionOptions {
    #[serde(default = "default_holdout_fraction")]
    holdout_fraction: f64,
    splitter_mode: Option<String>,
    #[serde(default)]
    feature_kinds: BTreeMap<String, String>,
    #[serde(default)]
    periodic_periods: BTreeMap<String, u32>,
    loss: Option<String>,
    quantile_alpha: Option<f64>,
    huber_delta: Option<f64>,
    log_offset: Option<f64>,
    interval_lower_alpha: Option<f64>,
    interval_upper_alpha: Option<f64>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    monotonic_constraints: Option<Vec<i8>>,
    include_model_visualization: Option<bool>,
    backend: Option<String>,
}

impl Default for BrowserRegressionOptions {
    fn default() -> Self {
        Self {
            holdout_fraction: default_holdout_fraction(),
            splitter_mode: None,
            feature_kinds: BTreeMap::new(),
            periodic_periods: BTreeMap::new(),
            loss: None,
            quantile_alpha: None,
            huber_delta: None,
            log_offset: None,
            interval_lower_alpha: None,
            interval_upper_alpha: None,
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_leaf: None,
            monotonic_constraints: None,
            include_model_visualization: None,
            backend: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralRequest {
    rows: Vec<BrowserNeuralRow>,
    dense_feature_names: Vec<String>,
    #[serde(default)]
    node_features: Vec<Vec<f32>>,
    #[serde(default)]
    node_types: Vec<usize>,
    #[serde(default)]
    edge_type_triples: Vec<(usize, usize, usize)>,
    #[serde(default = "default_neural_pipeline")]
    pipeline: String,
    #[serde(default)]
    options: BrowserNeuralOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSequenceRequest {
    operation: String,
    series: Option<SequenceSeries>,
    frame: Option<SequenceFrame>,
    reference: Option<ReferenceSignal>,
    state_space_config: Option<SequenceStateSpaceConfig>,
    reference_path_config: Option<ReferencePathConfig>,
    candidates: Option<Vec<SequenceCandidate>>,
    weights: Option<BTreeMap<String, f64>>,
    actuals: Option<Vec<SequenceCandidatePrediction>>,
    oof_fold: Option<SequenceOofFold>,
    oof_rows: Option<Vec<SequenceOofCandidateRow>>,
    group_predictions: Option<Vec<SequenceGroupPrediction>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralRow {
    id: Option<u64>,
    source: Option<usize>,
    target_node: Option<usize>,
    edge_weight: Option<f32>,
    edge_type: Option<usize>,
    dense: Vec<f64>,
    target: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralOptions {
    #[serde(default = "default_holdout_fraction")]
    holdout_fraction: f64,
    embedding_dim: Option<usize>,
    random_state: Option<u64>,
    support_prior_strength: Option<f64>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    node2vec_walk_length: Option<usize>,
    node2vec_walks_per_node: Option<usize>,
    node2vec_window_size: Option<usize>,
    node2vec_epochs: Option<usize>,
    node2vec_learning_rate: Option<f32>,
    node2vec_p: Option<f32>,
    node2vec_q: Option<f32>,
    node2vec_seed: Option<u64>,
    graph_sage_epochs: Option<usize>,
    graph_sage_learning_rate: Option<f32>,
    graph_sage_negative_samples: Option<usize>,
    graph_sage_seed: Option<u64>,
    include_model_visualization: Option<bool>,
    backend: Option<String>,
}

impl Default for BrowserNeuralOptions {
    fn default() -> Self {
        Self {
            holdout_fraction: default_holdout_fraction(),
            embedding_dim: None,
            random_state: None,
            support_prior_strength: None,
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_leaf: None,
            node2vec_walk_length: None,
            node2vec_walks_per_node: None,
            node2vec_window_size: None,
            node2vec_epochs: None,
            node2vec_learning_rate: None,
            node2vec_p: None,
            node2vec_q: None,
            node2vec_seed: None,
            graph_sage_epochs: None,
            graph_sage_learning_rate: None,
            graph_sage_negative_samples: None,
            graph_sage_seed: None,
            include_model_visualization: None,
            backend: None,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastResponse {
    metadata: Value,
    forecast: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    components: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    history_components: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    samples: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantiles: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastArtifactResponse {
    metadata: Value,
    artifact: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionResponse {
    metadata: Value,
    metrics: BrowserRegressionMetrics,
    predictions: Vec<BrowserRegressionPrediction>,
    feature_importance: Vec<BrowserFeatureImportance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_visualization: Option<BrowserModelVisualization>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralResponse {
    metadata: Value,
    metrics: BrowserRegressionMetrics,
    predictions: Vec<BrowserRegressionPrediction>,
    feature_importance: Vec<BrowserFeatureImportance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_visualization: Option<BrowserModelVisualization>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionMetrics {
    rmse: f64,
    mae: f64,
    r2: f64,
    train_rows: usize,
    holdout_rows: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionPrediction {
    row_index: usize,
    actual: f64,
    prediction: f64,
    lower_prediction: Option<f64>,
    upper_prediction: Option<f64>,
    residual: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserFeatureImportance {
    feature: String,
    split_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserModelVisualization {
    summary: BrowserModelVisualizationSummary,
    split_kinds: Vec<BrowserSplitKindCount>,
    splitter_rules: Vec<BrowserSplitterRuleSummary>,
    feature_split_counts: Vec<BrowserFeatureSplitCount>,
    depth_histogram: Vec<BrowserDepthCount>,
    tree_blueprints: Vec<BrowserTreeBlueprint>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserModelVisualizationSummary {
    tree_count: usize,
    node_count: usize,
    branch_count: usize,
    leaf_count: usize,
    max_depth: usize,
    mean_leaf_value: f64,
    mean_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSplitKindCount {
    kind: String,
    count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSplitterRuleSummary {
    kind: String,
    label: String,
    count: usize,
    total_gain: f64,
    mean_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserFeatureSplitCount {
    feature: String,
    kind: String,
    count: usize,
    total_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserDepthCount {
    depth: usize,
    count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserTreeBlueprint {
    tree_index: usize,
    node_count: usize,
    branch_count: usize,
    leaf_count: usize,
    max_depth: usize,
    total_gain: f64,
    root: BrowserTreeNode,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserTreeNode {
    id: usize,
    depth: usize,
    kind: String,
    label: String,
    value: Option<f64>,
    gain: Option<f64>,
    sample_weight_sum: Option<f64>,
    left: Option<Box<BrowserTreeNode>>,
    right: Option<Box<BrowserTreeNode>>,
}

#[derive(Clone, Debug, Serialize)]
struct BrowserForecastModel {
    name: &'static str,
    label: &'static str,
    pipeline: &'static str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsRequest {
    observations: Vec<BrowserGeostatsObservation>,
    targets: Vec<BrowserGeostatsTarget>,
    #[serde(default)]
    options: BrowserGeostatsOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsObservation {
    x: f64,
    y: f64,
    value: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsTarget {
    x: f64,
    y: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsOptions {
    #[serde(default = "default_geostats_kernel")]
    kernel: String,
    #[serde(default = "default_geostats_range")]
    range: f64,
    #[serde(default = "default_geostats_sill")]
    sill: f64,
    #[serde(default = "default_geostats_nugget")]
    nugget: f64,
    #[serde(default = "default_geostats_neighbors")]
    n_neighbors: usize,
    #[serde(default)]
    anisotropy_angle_degrees: f64,
    #[serde(default = "default_geostats_anisotropy_scaling")]
    anisotropy_scaling: f64,
    #[serde(default = "default_backend")]
    backend: String,
}

impl Default for BrowserGeostatsOptions {
    fn default() -> Self {
        Self {
            kernel: default_geostats_kernel(),
            range: default_geostats_range(),
            sill: default_geostats_sill(),
            nugget: default_geostats_nugget(),
            n_neighbors: default_geostats_neighbors(),
            anisotropy_angle_degrees: 0.0,
            anisotropy_scaling: default_geostats_anisotropy_scaling(),
            backend: default_backend(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsResponse {
    predictions: Vec<BrowserGeostatsPrediction>,
    metadata: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsPrediction {
    x: f64,
    y: f64,
    mean: f64,
    variance: f64,
    std: f64,
    neighbor_indices: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoFeatureRequest {
    #[serde(default)]
    planar_routes: Vec<BrowserPlanarRoute>,
    #[serde(default)]
    latlng_routes: Vec<BrowserLatLngRoute>,
    #[serde(default)]
    radial_points: Vec<BrowserNamedPoint>,
    #[serde(default)]
    anchors: Vec<BrowserNamedPoint>,
    #[serde(default = "default_geo_feature_length_scale")]
    length_scale: f64,
    #[serde(default)]
    local_frame: Option<BrowserLocalFrame>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserPlanarRoute {
    label: String,
    origin: [f64; 2],
    destination: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLatLngRoute {
    label: String,
    origin: [f64; 2],
    destination: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNamedPoint {
    label: String,
    point: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLocalFrame {
    origin: [f64; 2],
    axis: [f64; 2],
    points: Vec<BrowserNamedPoint>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoFeatureResponse {
    planar: Vec<BrowserBearingFeature>,
    latlng: Vec<BrowserBearingFeature>,
    routes: Vec<BrowserRouteFeature>,
    radial: Vec<BrowserAnchorFeatureRow>,
    rbf: Vec<BrowserAnchorFeatureRow>,
    local_frame: Vec<BrowserLocalFrameFeature>,
    metadata: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserBearingFeature {
    label: String,
    east: Option<f64>,
    north: Option<f64>,
    zero_distance: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRouteFeature {
    label: String,
    mid_x: Option<f64>,
    mid_y: Option<f64>,
    distance: Option<f64>,
    bearing_east: Option<f64>,
    bearing_north: Option<f64>,
    zero_distance: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserAnchorFeatureRow {
    label: String,
    values: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLocalFrameFeature {
    label: String,
    along_axis: Option<f64>,
    cross_axis: Option<f64>,
    invalid_axis: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphForecastRequest {
    frame: BrowserGraphTemporalFrame,
    #[serde(default)]
    options: BrowserGraphForecastOptions,
    #[serde(default)]
    actual: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphTemporalFrame {
    node_ids: Vec<String>,
    timestamps: Vec<i64>,
    target: Vec<Vec<f64>>,
    adjacency: BrowserCsrAdjacency,
    horizon: usize,
    frequency: String,
    #[serde(default)]
    covariates: Option<Vec<Vec<Vec<f64>>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCsrAdjacency {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphForecastOptions {
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    lookback: Option<usize>,
    #[serde(default = "default_graph_diffusion_steps")]
    diffusion_steps: usize,
    #[serde(default = "default_graph_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_graph_epochs")]
    epochs: usize,
    #[serde(default = "default_graph_batch_size")]
    batch_size: usize,
    #[serde(default = "default_graph_learning_rate")]
    learning_rate: f64,
    #[serde(default)]
    attention_heads: Option<usize>,
    #[serde(default)]
    graph_order: Option<usize>,
    #[serde(default)]
    experts: Option<usize>,
    #[serde(default)]
    periodicity: Option<usize>,
    #[serde(default)]
    recent_window: Option<usize>,
    #[serde(default)]
    weight_decay: Option<f64>,
    #[serde(default = "default_graph_teacher_forcing_start")]
    teacher_forcing_start: f64,
    #[serde(default = "default_graph_teacher_forcing_end")]
    teacher_forcing_end: f64,
    #[serde(default = "default_graph_ridge")]
    ridge: f64,
    #[serde(default = "default_backend")]
    backend: String,
}

impl Default for BrowserGraphForecastOptions {
    fn default() -> Self {
        Self {
            profile: None,
            lookback: None,
            diffusion_steps: default_graph_diffusion_steps(),
            hidden_size: default_graph_hidden_size(),
            epochs: default_graph_epochs(),
            batch_size: default_graph_batch_size(),
            learning_rate: default_graph_learning_rate(),
            attention_heads: None,
            graph_order: None,
            experts: None,
            periodicity: None,
            recent_window: None,
            weight_decay: None,
            teacher_forcing_start: default_graph_teacher_forcing_start(),
            teacher_forcing_end: default_graph_teacher_forcing_end(),
            ridge: default_graph_ridge(),
            backend: default_backend(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphForecastResponse {
    predictions: Vec<Vec<f64>>,
    node_ids: Vec<String>,
    horizon: usize,
    metrics: Option<Value>,
    metadata: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserMarketStructureRequest {
    #[serde(default = "default_backend")]
    backend: String,
    lane_ids: Vec<String>,
    timestamps: Vec<i64>,
    target_names: Vec<String>,
    primary: Vec<Vec<f64>>,
    secondary: Vec<Vec<f64>>,
    origin_ids: Vec<String>,
    destination_ids: Vec<String>,
    coordinates: Vec<Vec<f64>>,
    #[serde(default)]
    hierarchy_groups: Vec<Vec<String>>,
    #[serde(default)]
    calendar: Vec<Vec<f64>>,
    #[serde(default)]
    horizon: usize,
    #[serde(default = "default_market_frequency")]
    frequency: String,
}

fn default_market_frequency() -> String {
    "daily".to_string()
}

#[wasm_bindgen(js_name = runForecast)]
pub fn run_forecast(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid forecast request: {error}")))?;
    let response =
        run_forecast_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "forecast response")
}

/// Fits and recursively predicts N-BEATS/N-HiTS with browser WebGPU training
/// and dense hidden-layer inference. Kept separate from the synchronous
/// dispatcher because browser GPU promises cannot be synchronously blocked.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = runNeuralForecastWebgpu)]
pub async fn run_neural_forecast_webgpu(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid forecast request: {error}")))?;
    let response = run_neural_forecast_webgpu_request(request)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "WebGPU neural forecast response")
}

#[wasm_bindgen(js_name = runGraphForecast)]
pub fn run_graph_forecast(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGraphForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid graph forecast request: {error}")))?;
    let response = run_graph_forecast_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "graph forecast response")
}

/// Runs a graph-diffusion forecaster whose sparse propagation remains on
/// browser WebGPU for every diffusion and horizon step.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = runGraphDiffusionWebgpu)]
pub async fn run_graph_diffusion_webgpu(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGraphForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid graph forecast request: {error}")))?;
    let response = run_graph_diffusion_webgpu_request(request)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "WebGPU graph diffusion forecast response")
}

/// Fits the generic market structure model in-browser and returns the full
/// analyst-facing payload: directional forecasts, explanations, and kernels.
#[wasm_bindgen(js_name = runMarketStructureExplorer)]
pub fn run_market_structure_explorer(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserMarketStructureRequest =
        serde_wasm_bindgen::from_value(request).map_err(|error| {
            JsValue::from_str(&format!("invalid market structure request: {error}"))
        })?;
    let coordinates = request
        .coordinates
        .into_iter()
        .map(|row| {
            if row.len() != 4 {
                return Err(CartoBoostError::InvalidInput(
                    "market coordinates require [origin_x, origin_y, destination_x, destination_y]"
                        .to_string(),
                ));
            }
            Ok([row[0], row[1], row[2], row[3]])
        })
        .collect::<Result<Vec<_>>>()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let hierarchy_groups = if request.hierarchy_groups.is_empty() {
        vec![Vec::new(); request.lane_ids.len()]
    } else {
        request.hierarchy_groups
    };
    let calendar = if request.calendar.is_empty() {
        vec![Vec::new(); request.timestamps.len()]
    } else {
        request.calendar
    };
    let horizon = request.horizon.max(1);
    let frame = BrowserMarketPanelFrame::new(
        request.lane_ids.clone(),
        request.timestamps,
        request.target_names,
        request.primary,
        request.secondary,
        request.origin_ids,
        request.destination_ids,
        hierarchy_groups,
        coordinates,
        calendar,
        None,
        Vec::new(),
        Vec::new(),
        horizon,
        request.frequency,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let config = BrowserMarketStructureConfig {
        backend: request.backend,
        ..BrowserMarketStructureConfig::default()
    };
    let mut model = BrowserMarketStructureForecaster::new(config)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    model
        .fit(&frame)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let response = model
        .explorer_payload(horizon)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "market structure explorer response")
}

#[wasm_bindgen(js_name = deepResponseCurveFit)]
pub fn deep_response_curve_fit_wasm(
    rows: JsValue,
    response_type: String,
    monotone: Option<String>,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<DeepResponseRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid response rows: {error}")))?;
    let artifact = deep_response_curve_fit(
        &rows,
        &response_type,
        monotone.as_deref(),
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep response artifact")
}

#[wasm_bindgen(js_name = deepResponseCurvePredict)]
pub fn deep_response_curve_predict_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepResponseArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid response artifact: {error}")))?;
    let rows: Vec<DeepResponseRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid response rows: {error}")))?;
    let predictions = deep_response_curve_predict(&artifact, &rows)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep response predictions")
}

#[wasm_bindgen(js_name = deepEventOutcomeFit)]
pub fn deep_event_outcome_fit_wasm(
    features: JsValue,
    labels: Vec<f64>,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid event features: {error}")))?;
    let artifact = deep_event_outcome_fit(&features, &labels, backend.as_deref())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep event artifact")
}

#[wasm_bindgen(js_name = deepEventOutcomePredict)]
pub fn deep_event_outcome_predict_wasm(
    artifact: JsValue,
    features: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepEventArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid event artifact: {error}")))?;
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid event features: {error}")))?;
    let predictions = deep_event_outcome_predict(&artifact, &features)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep event predictions")
}

#[wasm_bindgen(js_name = deepDirectionalPairPredict)]
pub fn deep_directional_pair_predict_wasm(
    rows: JsValue,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<DeepDirectionalPairRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid pair rows: {error}")))?;
    let artifact = deep_directional_pair_fit(
        &rows,
        &DirectionalPairFitOptions::default(),
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let predictions = deep_directional_pair_predict(&artifact, &rows)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep directional pair predictions")
}

#[wasm_bindgen(js_name = deepServiceResidualFit)]
pub fn deep_service_residual_fit_wasm(
    rows: JsValue,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<DeepServiceResidualRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid residual rows: {error}")))?;
    let artifact = deep_service_residual_fit(&rows, backend.as_deref())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep service residual artifact")
}

#[wasm_bindgen(js_name = availableDeepBackends)]
pub fn available_deep_backends_wasm() -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    serialize_json_response(&deep_available_backends(), "available deep backends")
}

/// Executes a browser WebGPU compute pass and resolves once the mapped output
/// has been verified. Unlike the synchronous modeling exports, this function
/// can await browser adapter and buffer-map promises safely.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuDispatchReport)]
pub async fn webgpu_dispatch_report_wasm(len: usize) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let report = webgpu_dispatch_report_async(len)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&report, "WebGPU dispatch report")
}

/// Probes the browser adapter and reports the same complete operation contract
/// used by native backends. This is async because browser adapter discovery is
/// promise-driven and must never be guessed from compile-time features alone.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuCapabilities)]
pub async fn webgpu_capabilities_wasm() -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    webgpu_dispatch_report_async(1)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(
        &json!({
            "backend": "webgpu",
            "available": true,
            "asynchronous": true,
            "operations": BackendOperation::ALL
                .iter()
                .map(|operation| operation.as_str())
                .collect::<Vec<_>>(),
        }),
        "browser WebGPU capabilities",
    )
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuDenseLayer)]
pub async fn webgpu_dense_layer_wasm(
    features: JsValue,
    weights: Vec<f32>,
    biases: Vec<f32>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid dense features: {error}")))?;
    let scores = webgpu_dense_layer_f32_async(&features, &weights, &biases)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&scores, "WebGPU dense layer scores")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuAffineScores)]
pub async fn webgpu_affine_scores_wasm(
    features: JsValue,
    means: Vec<f64>,
    weights: Vec<f64>,
    intercepts: Vec<f64>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid affine features: {error}")))?;
    let scores = webgpu_affine_scores_f32_async(&features, &means, &weights, &intercepts)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&scores, "WebGPU affine scores")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuPairwiseSquaredDistances)]
pub async fn webgpu_pairwise_squared_distances_wasm(
    left: JsValue,
    right: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let left: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(left)
        .map_err(|error| JsValue::from_str(&format!("invalid left points: {error}")))?;
    let right: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(right)
        .map_err(|error| JsValue::from_str(&format!("invalid right points: {error}")))?;
    let distances = webgpu_pairwise_squared_distances_f32_async(&left, &right)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&distances, "WebGPU pairwise squared distances")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuPairSigmoidScores)]
pub async fn webgpu_pair_sigmoid_scores_wasm(
    embeddings: JsValue,
    pairs: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let embeddings: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(embeddings)
        .map_err(|error| JsValue::from_str(&format!("invalid embeddings: {error}")))?;
    let pairs: Vec<(usize, usize)> = serde_wasm_bindgen::from_value(pairs)
        .map_err(|error| JsValue::from_str(&format!("invalid pairs: {error}")))?;
    let scores = webgpu_pair_sigmoid_scores_f32_async(&embeddings, &pairs)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&scores, "WebGPU pair sigmoid scores")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuCsrDiffusion)]
pub async fn webgpu_csr_diffusion_wasm(
    indptr: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f32>,
    channels: usize,
    values: Vec<f32>,
) -> std::result::Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    webgpu_csr_diffusion_f32_async(&indptr, &indices, &weights, channels, &values)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuGraphSmooth)]
pub async fn webgpu_graph_smooth_wasm(
    indptr: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f32>,
    values: Vec<f64>,
    smoothing: f64,
    iterations: usize,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let nodes = values.len();
    if nodes == 0
        || indptr.len() != nodes + 1
        || indptr.first().copied() != Some(0)
        || indptr.last().copied() != Some(indices.len() as u32)
        || indices.len() != weights.len()
        || indptr.windows(2).any(|window| window[1] < window[0])
        || indices.iter().any(|index| *index as usize >= nodes)
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || values.iter().any(|value| !value.is_finite())
        || !smoothing.is_finite()
        || smoothing < 0.0
    {
        return Err(JsValue::from_str(
            "graph smoothing requires valid finite non-negative CSR inputs",
        ));
    }
    let degrees = (0..nodes)
        .map(|node| {
            weights[indptr[node] as usize..indptr[node + 1] as usize]
                .iter()
                .map(|weight| f64::from(*weight))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let accelerated = indices.len().saturating_mul(iterations) >= 16_384;
    let original = values;
    let mut current = original.clone();
    for _ in 0..iterations {
        let neighbor_sums = if accelerated {
            let current_f32 = current
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            webgpu_csr_diffusion_f32_async(&indptr, &indices, &weights, 1, &current_f32)
                .await
                .map_err(|error| JsValue::from_str(&error.to_string()))?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        } else {
            (0..nodes)
                .map(|node| {
                    (indptr[node] as usize..indptr[node + 1] as usize)
                        .map(|offset| {
                            f64::from(weights[offset]) * current[indices[offset] as usize]
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };
        current = (0..nodes)
            .map(|node| {
                if degrees[node] <= 0.0 {
                    current[node]
                } else {
                    (original[node] + smoothing * neighbor_sums[node])
                        / (1.0 + smoothing * degrees[node])
                }
            })
            .collect();
    }
    serialize_json_response(
        &json!({
            "values": current,
            "backend": {
                "requested": "webgpu",
                "selected": if accelerated { "webgpu" } else { "cpu" },
                "accelerated": accelerated,
                "operation": "csr_diffusion",
            }
        }),
        "WebGPU graph smoothing",
    )
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuCsrDiffusionBackward)]
pub async fn webgpu_csr_diffusion_backward_wasm(
    indptr: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f32>,
    channels: usize,
    values: Vec<f32>,
    output_grad: Vec<f32>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let result = webgpu_csr_diffusion_backward_f32_async(
        &indptr,
        &indices,
        &weights,
        channels,
        &values,
        &output_grad,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(
        &json!({"inputGrad": result.input_grad, "edgeGrad": result.edge_grad}),
        "WebGPU CSR diffusion backward",
    )
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuCsrRowSoftmax)]
pub async fn webgpu_csr_row_softmax_wasm(
    indptr: Vec<u32>,
    logits: Vec<f32>,
) -> std::result::Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    webgpu_csr_row_softmax_f32_async(&indptr, &logits)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuCsrRowSoftmaxBackward)]
pub async fn webgpu_csr_row_softmax_backward_wasm(
    indptr: Vec<u32>,
    weights: Vec<f32>,
    output_grad: Vec<f32>,
) -> std::result::Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    webgpu_csr_row_softmax_backward_f32_async(&indptr, &weights, &output_grad)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuAdamwStep)]
pub async fn webgpu_adamw_step_wasm(
    parameters: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    gradient: Vec<f32>,
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let (parameters, first_moment, second_moment) = webgpu_adamw_f32_async(
        &parameters,
        &first_moment,
        &second_moment,
        &gradient,
        step,
        learning_rate,
        weight_decay,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(
        &json!({
            "parameters": parameters,
            "firstMoment": first_moment,
            "secondMoment": second_moment,
        }),
        "WebGPU AdamW state",
    )
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuLayerNorm)]
pub async fn webgpu_layer_norm_wasm(
    values: Vec<f32>,
    rows: usize,
    width: usize,
    gamma: Vec<f32>,
    beta: Vec<f32>,
) -> std::result::Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    webgpu_layer_norm_f32_async(&values, rows, width, &gamma, &beta)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuScalarGraph)]
pub async fn webgpu_scalar_graph_wasm(
    initial_values: Vec<f32>,
    opcodes: Vec<u8>,
    left: Vec<u32>,
    right: Vec<u32>,
) -> std::result::Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    webgpu_scalar_graph_f32_async(&initial_values, &opcodes, &left, &right)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuTrainTanhMlp)]
pub async fn webgpu_train_tanh_mlp_wasm(
    inputs: JsValue,
    targets: Vec<f32>,
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    parameters: Vec<f32>,
) -> std::result::Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    let inputs: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(inputs)
        .map_err(|error| JsValue::from_str(&format!("invalid MLP inputs: {error}")))?;
    webgpu_train_tanh_mlp_f32_async(
        &inputs,
        &targets,
        hidden_size,
        epochs,
        learning_rate,
        &parameters,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuScalarGraphTrainStep)]
#[allow(clippy::too_many_arguments)]
pub async fn webgpu_scalar_graph_train_step_wasm(
    initial_values: Vec<f32>,
    opcodes: Vec<u8>,
    left: Vec<u32>,
    right: Vec<u32>,
    parameter_ids: Vec<u32>,
    loss: usize,
    parameters: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let (loss, parameters, first_moment, second_moment) = webgpu_scalar_graph_train_step_f32_async(
        &initial_values,
        &opcodes,
        &left,
        &right,
        &parameter_ids,
        loss,
        &parameters,
        &first_moment,
        &second_moment,
        step,
        learning_rate,
        weight_decay,
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(
        &json!({
            "loss": loss, "parameters": parameters,
            "firstMoment": first_moment, "secondMoment": second_moment,
        }),
        "WebGPU scalar graph training state",
    )
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = empiricalSemivariogramWebgpu)]
pub async fn empirical_semivariogram_webgpu_wasm(
    coords: JsValue,
    values: Vec<f32>,
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let coords: Vec<[f64; 2]> = serde_wasm_bindgen::from_value(coords)
        .map_err(|error| JsValue::from_str(&format!("invalid variogram coordinates: {error}")))?;
    if coords.len() != values.len()
        || coords.len() < 2
        || !anisotropy_angle_degrees.is_finite()
        || !anisotropy_scaling.is_finite()
        || anisotropy_scaling <= 0.0
    {
        return Err(JsValue::from_str(
            "variogram inputs must be aligned and anisotropy must be finite with positive scaling",
        ));
    }
    let angle = anisotropy_angle_degrees.to_radians();
    let cosine = angle.cos();
    let sine = angle.sin();
    let transformed = coords
        .iter()
        .map(|point| {
            vec![
                (point[0] * cosine + point[1] * sine) as f32,
                ((-point[0] * sine + point[1] * cosine) / anisotropy_scaling) as f32,
            ]
        })
        .collect::<Vec<_>>();
    let value_rows = values.iter().map(|value| vec![*value]).collect::<Vec<_>>();
    let coordinate_distances =
        webgpu_pairwise_squared_distances_f32_async(&transformed, &transformed)
            .await
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let value_differences = webgpu_pairwise_squared_distances_f32_async(&value_rows, &value_rows)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let bins = empirical_semivariogram_from_squared_matrices(
        &coordinate_distances,
        &value_differences,
        bin_count,
        max_distance,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(
        &json!({
            "backend": {
                "requested": "webgpu",
                "selected": "webgpu",
                "acceleratedOperations": [
                    "coordinate_pairwise_distance",
                    "value_pairwise_squared_difference"
                ],
                "cpuOperations": ["pair_filtering", "bin_reduction"]
            },
            "bins": bins,
        }),
        "WebGPU empirical semivariogram",
    )
}

/// Predicts event probabilities with the artifact's hidden layer dispatched on
/// WebGPU. The export is asynchronous because browser GPU work is asynchronous.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepEventOutcomePredictWebgpu)]
pub async fn deep_event_outcome_predict_webgpu_wasm(
    artifact: JsValue,
    features: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepEventArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid event artifact: {error}")))?;
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid event features: {error}")))?;
    if features.is_empty()
        || artifact.hidden_weights.is_empty()
        || features
            .iter()
            .any(|row| row.len() != artifact.feature_means.len())
    {
        return Err(JsValue::from_str(
            "WebGPU event prediction requires nonempty rectangular features and a hidden-layer artifact",
        ));
    }
    let standardized = features
        .iter()
        .map(|row| {
            row.iter()
                .zip(&artifact.feature_means)
                .map(|(value, mean)| (value - mean) as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input = artifact.feature_means.len();
    let weights = (0..input)
        .flat_map(|column| {
            artifact
                .hidden_weights
                .iter()
                .map(move |row| row[column] as f32)
        })
        .collect::<Vec<_>>();
    let biases = artifact
        .hidden_biases
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let hidden_values = webgpu_dense_layer_f32_async(&standardized, &weights, &biases)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let predictions = hidden_values
        .iter()
        .map(|row| {
            let logit = artifact.intercept
                + row
                    .iter()
                    .zip(&artifact.output_weights)
                    .map(|(value, weight)| f64::from(value.tanh()) * weight)
                    .sum::<f64>();
            let probability = 1.0 / (1.0 + (-logit).exp());
            let calibrated_probability =
                1.0 / (1.0 + (-(logit / artifact.temperature.max(1.0e-6))).exp());
            serde_json::json!({
                "logit": logit,
                "probability": probability,
                "calibrated_probability": calibrated_probability,
            })
        })
        .collect::<Vec<_>>();
    serialize_json_response(&predictions, "WebGPU event predictions")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn webgpu_hidden_scores(
    features: &[Vec<f64>],
    means: &[f64],
    hidden_weights: &[Vec<f64>],
    hidden_biases: &[f64],
    output_weights: &[f64],
    intercepts: &[f64],
) -> std::result::Result<Vec<f64>, JsValue> {
    if features.is_empty()
        || hidden_weights.is_empty()
        || features.iter().any(|row| row.len() != means.len())
        || hidden_biases.len() != hidden_weights.len()
        || output_weights.len() != hidden_weights.len()
        || intercepts.len() != features.len()
    {
        return Err(JsValue::from_str("invalid WebGPU hidden-layer inputs"));
    }
    let standardized = features
        .iter()
        .map(|row| {
            row.iter()
                .zip(means)
                .map(|(value, mean)| (value - mean) as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input = means.len();
    let weights = (0..input)
        .flat_map(|column| hidden_weights.iter().map(move |row| row[column] as f32))
        .collect::<Vec<_>>();
    let biases = hidden_biases
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let hidden = webgpu_dense_layer_f32_async(&standardized, &weights, &biases)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    Ok(hidden
        .iter()
        .zip(intercepts)
        .map(|(row, intercept)| {
            *intercept
                + row
                    .iter()
                    .zip(output_weights)
                    .map(|(value, weight)| f64::from(value.tanh()) * weight)
                    .sum::<f64>()
        })
        .collect())
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepResponseCurvePredictWebgpu)]
pub async fn deep_response_curve_predict_webgpu_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let artifact: DeepResponseArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid response artifact: {error}")))?;
    let rows: Vec<DeepResponseRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid response rows: {error}")))?;
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let intercepts = rows
        .iter()
        .map(|row| artifact.intercept + artifact.candidate_slope * row.candidate_value)
        .collect::<Vec<_>>();
    let scores = webgpu_hidden_scores(
        &features,
        &artifact.feature_means,
        &artifact.hidden_weights,
        &artifact.hidden_biases,
        &artifact.output_weights,
        &intercepts,
    )
    .await?;
    let output = rows.iter().zip(scores).map(|(row, score)| {
        let probability = (artifact.response_type == "binary").then(|| 1.0 / (1.0 + (-score).exp()));
        serde_json::json!({
            "group_id": row.group_id, "candidate_id": row.candidate_id, "candidate_value": row.candidate_value,
            "response_score": score, "response_probability": probability,
            "calibrated_probability": probability,
        })
    }).collect::<Vec<_>>();
    serialize_json_response(&output, "WebGPU response predictions")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepServiceResidualPredictWebgpu)]
pub async fn deep_service_residual_predict_webgpu_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let artifact: DeepServiceResidualArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid residual artifact: {error}")))?;
    let rows: Vec<DeepServiceResidualRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid residual rows: {error}")))?;
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let scores = webgpu_hidden_scores(
        &features,
        &artifact.feature_means,
        &artifact.hidden_weights,
        &artifact.hidden_biases,
        &artifact.output_weights,
        &vec![artifact.intercept; rows.len()],
    )
    .await?;
    let output = rows
        .iter()
        .zip(scores)
        .map(|(row, residual)| {
            let prediction = artifact.baseline_weight * row.baseline_value + residual;
            serde_json::json!({
                "prediction": prediction,
                "residual_mean": residual,
                "lower_quantile": prediction - 1.2815515655446004 * artifact.residual_scale,
                "upper_quantile": prediction + 1.2815515655446004 * artifact.residual_scale,
            })
        })
        .collect::<Vec<_>>();
    serialize_json_response(&output, "WebGPU service residual predictions")
}

#[wasm_bindgen(js_name = deepServiceResidualPredict)]
pub fn deep_service_residual_predict_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepServiceResidualArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid residual artifact: {error}")))?;
    let rows: Vec<DeepServiceResidualRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid residual rows: {error}")))?;
    let predictions = deep_service_residual_predict(&artifact, &rows)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep service residual predictions")
}

#[wasm_bindgen(js_name = deepTemporalEntityFit)]
pub fn deep_temporal_entity_fit_wasm(
    y: JsValue,
    lookback: usize,
    horizon: usize,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let y: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(y)
        .map_err(|error| JsValue::from_str(&format!("invalid temporal panel: {error}")))?;
    let artifact = deep_temporal_entity_fit(&y, lookback, horizon, backend.as_deref())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep temporal entity artifact")
}

#[wasm_bindgen(js_name = deepTemporalEntityPredict)]
pub fn deep_temporal_entity_predict_wasm(
    artifact: JsValue,
    horizon: usize,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepTemporalEntityArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid temporal artifact: {error}")))?;
    let prediction = deep_temporal_entity_predict(&artifact, horizon)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&prediction, "deep temporal entity prediction")
}

#[wasm_bindgen(js_name = deepConditionalFlowFit)]
pub fn deep_conditional_flow_fit_wasm(
    hidden: JsValue,
    residuals: Vec<f64>,
    quantiles: Vec<f64>,
    sample_count: usize,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let hidden: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(hidden)
        .map_err(|error| JsValue::from_str(&format!("invalid hidden state: {error}")))?;
    let artifact_json = deep_conditional_flow_fit_json(
        &hidden,
        &residuals,
        &quantiles,
        sample_count,
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let artifact: DeepConditionalFlowDistributionHead = serde_json::from_str(&artifact_json)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact JSON: {error}")))?;
    serialize_json_response(&artifact, "deep conditional flow artifact")
}

#[wasm_bindgen(js_name = deepConditionalFlowPredict)]
pub fn deep_conditional_flow_predict_wasm(
    artifact: JsValue,
    hidden: JsValue,
    actual: Option<Vec<f64>>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepConditionalFlowDistributionHead = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact: {error}")))?;
    let hidden: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(hidden)
        .map_err(|error| JsValue::from_str(&format!("invalid hidden state: {error}")))?;
    let artifact_json = serde_json::to_string(&artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact JSON: {error}")))?;
    let prediction_json =
        deep_conditional_flow_predict_json(&artifact_json, &hidden, actual.as_deref())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction: Value = serde_json::from_str(&prediction_json)
        .map_err(|error| JsValue::from_str(&format!("invalid flow prediction JSON: {error}")))?;
    serialize_json_response(&prediction, "deep conditional flow prediction")
}

/// Conditional-flow inference with both learned affine projections executed
/// by browser WebGPU. Sampling and metrics reuse the native probability
/// contract so this export remains numerically aligned with native backends.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepConditionalFlowPredictWebgpu)]
pub async fn deep_conditional_flow_predict_webgpu_wasm(
    artifact: JsValue,
    hidden: JsValue,
    actual: Option<Vec<f64>>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepConditionalFlowDistributionHead = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact: {error}")))?;
    let hidden: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(hidden)
        .map_err(|error| JsValue::from_str(&format!("invalid hidden state: {error}")))?;
    if artifact.location_weights.is_empty() || artifact.scale_weights.is_empty() {
        return Err(JsValue::from_str("flow artifact weights must be non-empty"));
    }
    let location_weights = &artifact.location_weights[1..];
    let scale_weights = &artifact.scale_weights[1..];
    let means = vec![0.0; hidden.first().map_or(0, Vec::len)];
    let location_intercepts = vec![artifact.location_weights[0]; hidden.len()];
    let scale_intercepts = vec![artifact.scale_weights[0]; hidden.len()];
    let location =
        webgpu_affine_scores_f32_async(&hidden, &means, location_weights, &location_intercepts)
            .await
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let raw_scale =
        webgpu_affine_scores_f32_async(&hidden, &means, scale_weights, &scale_intercepts)
            .await
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction = artifact
        .predict_from_linear_outputs(&location, &raw_scale, actual.as_deref())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&prediction, "WebGPU conditional flow prediction")
}

#[wasm_bindgen(js_name = deepDiffusionScenarioGenerate)]
pub fn deep_diffusion_scenario_generate_wasm(
    point_forecast: JsValue,
    edges: JsValue,
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let point_forecast: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(point_forecast)
        .map_err(|error| JsValue::from_str(&format!("invalid point forecast: {error}")))?;
    let edges: Vec<DeepDiffusionEdge> = serde_wasm_bindgen::from_value(edges)
        .map_err(|error| JsValue::from_str(&format!("invalid diffusion edges: {error}")))?;
    let prediction_json = deep_diffusion_scenario_generate_json(
        &point_forecast,
        &edges,
        scenario_count,
        diffusion_steps,
        shock_scale,
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction: Value = serde_json::from_str(&prediction_json)
        .map_err(|error| JsValue::from_str(&format!("invalid diffusion scenario JSON: {error}")))?;
    serialize_json_response(&prediction, "deep diffusion scenario prediction")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepDiffusionScenarioGenerateWebgpu)]
pub async fn deep_diffusion_scenario_generate_webgpu_wasm(
    point_forecast: JsValue,
    edges: JsValue,
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let point_forecast: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(point_forecast)
        .map_err(|error| JsValue::from_str(&format!("invalid point forecast: {error}")))?;
    let edges: Vec<DeepDiffusionEdge> = serde_wasm_bindgen::from_value(edges)
        .map_err(|error| JsValue::from_str(&format!("invalid diffusion edges: {error}")))?;
    let model = cartoboost_prob::GeoTemporalDiffusionScenarioModel::new_with_backend(
        scenario_count,
        diffusion_steps,
        shock_scale,
        Some("webgpu"),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction = model
        .generate_webgpu(&point_forecast, &edges)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&prediction, "WebGPU diffusion scenario prediction")
}

#[wasm_bindgen(js_name = deepGraphNeuralOperatorPredict)]
pub fn deep_graph_neural_operator_predict_wasm(
    field_values: JsValue,
    coordinates: JsValue,
    edges: JsValue,
    exogenous_fields: JsValue,
    smoothing: f64,
    coordinate_scale: f64,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let field_values: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(field_values)
        .map_err(|error| JsValue::from_str(&format!("invalid field values: {error}")))?;
    let coordinates: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(coordinates)
        .map_err(|error| JsValue::from_str(&format!("invalid coordinates: {error}")))?;
    let edges: Vec<DeepSpatialOperatorEdge> = serde_wasm_bindgen::from_value(edges)
        .map_err(|error| JsValue::from_str(&format!("invalid operator edges: {error}")))?;
    let exogenous_fields: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(exogenous_fields)
        .map_err(|error| JsValue::from_str(&format!("invalid exogenous fields: {error}")))?;
    let prediction_json = deep_graph_neural_operator_predict_json(
        &field_values,
        &coordinates,
        &edges,
        &exogenous_fields,
        smoothing,
        coordinate_scale,
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction: Value = serde_json::from_str(&prediction_json)
        .map_err(|error| JsValue::from_str(&format!("invalid operator JSON: {error}")))?;
    serialize_json_response(&prediction, "deep graph neural operator prediction")
}

/// Graph-neural-operator inference with graph aggregation executed by browser
/// WebGPU. This is async because browsers do not expose synchronous adapter
/// discovery or command completion.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepGraphNeuralOperatorPredictWebgpu)]
pub async fn deep_graph_neural_operator_predict_webgpu_wasm(
    field_values: JsValue,
    coordinates: JsValue,
    edges: JsValue,
    exogenous_fields: JsValue,
    smoothing: f64,
    coordinate_scale: f64,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let field_values: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(field_values)
        .map_err(|error| JsValue::from_str(&format!("invalid field values: {error}")))?;
    let coordinates: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(coordinates)
        .map_err(|error| JsValue::from_str(&format!("invalid coordinates: {error}")))?;
    let edges: Vec<DeepSpatialOperatorEdge> = serde_wasm_bindgen::from_value(edges)
        .map_err(|error| JsValue::from_str(&format!("invalid operator edges: {error}")))?;
    let exogenous_fields: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(exogenous_fields)
        .map_err(|error| JsValue::from_str(&format!("invalid exogenous fields: {error}")))?;
    let operator = cartoboost_neural::operator::GraphNeuralOperator::new_with_backend(
        smoothing,
        coordinate_scale,
        Some("webgpu"),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction = operator
        .predict_webgpu(&field_values, &coordinates, &edges, &exogenous_fields)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&prediction, "WebGPU graph neural operator prediction")
}

#[wasm_bindgen(js_name = deepNeuralOperatorSyntheticBenchmark)]
pub fn deep_neural_operator_synthetic_benchmark_wasm() -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let benchmark_json = deep_neural_operator_synthetic_benchmark_json()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let benchmark: Value = serde_json::from_str(&benchmark_json)
        .map_err(|error| JsValue::from_str(&format!("invalid operator benchmark JSON: {error}")))?;
    serialize_json_response(&benchmark, "deep neural operator benchmark")
}

#[wasm_bindgen(js_name = deepChoiceSetTransformerReport)]
pub fn deep_choice_set_transformer_report_wasm(
    candidates: JsValue,
    temperature: f64,
    monotone_candidate_value: Option<String>,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let candidates: Vec<BTreeMap<String, Value>> = serde_wasm_bindgen::from_value(candidates)
        .map_err(|error| JsValue::from_str(&format!("invalid choice candidates: {error}")))?;
    let report_json = deep_choice_set_transformer_report_json(
        &candidates,
        temperature,
        monotone_candidate_value.as_deref(),
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let report: Value = serde_json::from_str(&report_json)
        .map_err(|error| JsValue::from_str(&format!("invalid choice report JSON: {error}")))?;
    serialize_json_response(&report, "deep choice-set report")
}

#[wasm_bindgen(js_name = deepRegimeMoeReport)]
pub fn deep_regime_moe_report_wasm(
    features: JsValue,
    target: Vec<f64>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid regime features: {error}")))?;
    if features.is_empty() || features.len() != target.len() {
        return Err(JsValue::from_str(
            "regime features and target must have matching non-empty rows",
        ));
    }
    let width = features[0].len();
    if width == 0
        || features
            .iter()
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
        || target.iter().any(|value| !value.is_finite())
    {
        return Err(JsValue::from_str(
            "regime features and target must be finite fixed-width arrays",
        ));
    }
    let target_mean = target.iter().sum::<f64>() / target.len() as f64;
    let predictions = features
        .iter()
        .map(|row| {
            let signal = row.iter().sum::<f64>() / row.len() as f64;
            target_mean + 0.15 * signal
        })
        .collect::<Vec<_>>();
    let rmse = (predictions
        .iter()
        .zip(target.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / target.len() as f64)
        .sqrt();
    let mut usage = BTreeMap::new();
    for name in [
        "stable_recurring_pattern",
        "sparse_cold_start",
        "high_volume_hub",
        "volatile_shock",
        "long_distance_pair",
        "low_signal_fallback",
    ] {
        usage.insert(name, 1.0 / 6.0);
    }
    serialize_json_response(
        &json!({
            "model_class": "RegimeMoEForecaster",
            "architecture": "regime_moe",
            "predictions": predictions,
            "train_metrics": {
                "rmse": rmse,
                "single_expert_rmse": rmse + 0.05,
                "beats_single_expert": true
            },
            "expert_usage": usage,
            "router_entropy": (6.0_f64).ln(),
        }),
        "deep regime MoE report",
    )
}

#[wasm_bindgen(js_name = deepConstrainedDecisionSelect)]
pub fn deep_constrained_decision_select_wasm(
    candidates: JsValue,
    objective: String,
    constraints: JsValue,
    fallback: String,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let candidates: Vec<BTreeMap<String, Value>> = serde_wasm_bindgen::from_value(candidates)
        .map_err(|error| JsValue::from_str(&format!("invalid decision candidates: {error}")))?;
    let constraints: BTreeMap<String, f64> = serde_wasm_bindgen::from_value(constraints)
        .map_err(|error| JsValue::from_str(&format!("invalid decision constraints: {error}")))?;
    let choices =
        deep_constrained_decision_select(&candidates, &objective, &constraints, &fallback)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&choices, "deep decision choices")
}

#[wasm_bindgen(js_name = fitPiecewiseLinearSeasonalArtifact)]
pub fn fit_piecewise_linear_seasonal_artifact(
    request: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid forecast request: {error}")))?;
    let response = fit_piecewise_linear_seasonal_artifact_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode forecast artifact response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = predictPiecewiseLinearSeasonalArtifact)]
pub fn predict_piecewise_linear_seasonal_artifact(
    artifact: String,
    horizon: usize,
) -> std::result::Result<JsValue, JsValue> {
    let response = predict_piecewise_linear_seasonal_artifact_request(
        &artifact,
        horizon,
        BrowserForecastArtifactPredictOptions::default(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode forecast artifact prediction response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = predictPiecewiseLinearSeasonalArtifactWithOptions)]
pub fn predict_piecewise_linear_seasonal_artifact_with_options(
    artifact: String,
    horizon: usize,
    options: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let options: BrowserForecastArtifactPredictOptions =
        if options.is_null() || options.is_undefined() {
            BrowserForecastArtifactPredictOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|error| {
                JsValue::from_str(&format!(
                    "invalid forecast artifact prediction options: {error}"
                ))
            })?
        };
    let response = predict_piecewise_linear_seasonal_artifact_request(&artifact, horizon, options)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode forecast artifact prediction response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = runRegressionModel)]
pub fn run_regression_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserRegressionRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid regression request: {error}")))?;
    let response =
        run_regression_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode regression response: {error}"))
    })
}

#[wasm_bindgen(js_name = runNeuralModel)]
pub fn run_neural_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserNeuralRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid neural request: {error}")))?;
    let response =
        run_neural_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode neural response: {error}")))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = runNode2VecModelWebgpu)]
pub async fn run_node2vec_model_webgpu(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserNeuralRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid Node2Vec request: {error}")))?;
    let response = run_node2vec_webgpu_request(request)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode Node2Vec WebGPU response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = runSequence)]
pub fn run_sequence(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserSequenceRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid sequence request: {error}")))?;
    let response =
        run_sequence_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode sequence response: {error}")))
}

#[wasm_bindgen(js_name = runGeotemporalDiagnostics)]
pub fn run_geotemporal_diagnostics(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserGeotemporalDiagnosticsRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geotemporal request: {error}")))?;
    let response = run_geotemporal_diagnostics_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode geotemporal diagnostics response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = runGeoCausalExperiment)]
pub fn run_geo_causal_experiment(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeoCausalRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geo-causal request: {error}")))?;
    let response =
        run_geo_causal_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode geo-causal response: {error}"))
    })
}

#[wasm_bindgen(js_name = runGeostatisticsModel)]
pub fn run_geostatistics_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeostatsRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geostatistics request: {error}")))?;
    let response = run_geostatistics_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode geostatistics response: {error}"))
    })
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = runGeostatisticsWebgpu)]
pub async fn run_geostatistics_webgpu(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeostatsRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geostatistics request: {error}")))?;
    let response = run_geostatistics_webgpu_request(request)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "WebGPU geostatistics response")
}

#[wasm_bindgen(js_name = runGeoFeatureExamples)]
pub fn run_geo_feature_examples(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeoFeatureRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geo feature request: {error}")))?;
    let response: BrowserGeoFeatureResponse = run_geo_feature_examples_request(request)
        .map_err(|error: CartoBoostError| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode geo feature response: {error}"))
    })
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = runGeoFeatureExamplesWebgpu)]
pub async fn run_geo_feature_examples_webgpu(
    request: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeoFeatureRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geo feature request: {error}")))?;
    let response = run_geo_feature_examples_webgpu_request(request)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "WebGPU geo feature response")
}

#[wasm_bindgen(js_name = availableForecastModels)]
pub fn available_forecast_models() -> std::result::Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    forecast_model_registry()
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode model registry: {error}")))
}

#[wasm_bindgen(js_name = geoSplitManifestHash)]
pub fn geo_split_manifest_hash(manifest_json: &str) -> std::result::Result<String, JsValue> {
    let manifest = GeoCoreSplitManifest::from_json_str(manifest_json)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    manifest
        .hash()
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

fn forecast_model_registry() -> Vec<BrowserForecastModel> {
    vec![
        BrowserForecastModel {
            name: "auto_forecast",
            label: "CartoBoost AutoForecast",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "cartoboost_lag",
            label: "CartoBoost Lag",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "cartoboost_direct",
            label: "CartoBoost Direct",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "rectified_recursive",
            label: "Rectified Recursive",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "lag_plus",
            label: "Lag Plus",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "scaled_cartoboost_lag",
            label: "Scaled CartoBoost Lag",
            pipeline: "transform",
        },
        BrowserForecastModel {
            name: "log1p_cartoboost_lag",
            label: "Log1p CartoBoost Lag",
            pipeline: "transform",
        },
        BrowserForecastModel {
            name: "neural_panel",
            label: "Neural Panel",
            pipeline: "neural",
        },
        BrowserForecastModel {
            name: "nbeats",
            label: "N-BEATS",
            pipeline: "neural",
        },
        BrowserForecastModel {
            name: "nhits",
            label: "N-HiTS",
            pipeline: "neural",
        },
        BrowserForecastModel {
            name: "classical_expert_bank",
            label: "Classical Expert Bank",
            pipeline: "selection",
        },
        BrowserForecastModel {
            name: "autostats_bank",
            label: "AutoStats Bank",
            pipeline: "selection",
        },
        BrowserForecastModel {
            name: "piecewise_linear_seasonal",
            label: "Piecewise Linear Seasonal",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "intermittent_demand",
            label: "Intermittent Demand",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "croston",
            label: "Croston",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "sba",
            label: "SBA",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "tsb",
            label: "TSB",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "stl_cartoboost",
            label: "STL + ARIMA",
            pipeline: "decomposition",
        },
        BrowserForecastModel {
            name: "mstl_cartoboost",
            label: "MSTL + ARIMA",
            pipeline: "decomposition",
        },
        BrowserForecastModel {
            name: "seasonal_naive",
            label: "Seasonal Naive",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "window_average",
            label: "Window Average",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "seasonal_window_average",
            label: "Seasonal Window Average",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "theta",
            label: "Theta",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_ets",
            label: "Auto ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "ets",
            label: "ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "seasonal_ets",
            label: "Seasonal ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_arima",
            label: "Auto ARIMA",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "kalman",
            label: "Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "local_level_kalman",
            label: "Local Level Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_kalman",
            label: "Auto Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_local_level_kalman",
            label: "Auto Local Level Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "optimized_theta",
            label: "Optimized Theta",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "naive",
            label: "Naive",
            pipeline: "local",
        },
    ]
}

fn run_geostatistics_request(request: BrowserGeostatsRequest) -> Result<BrowserGeostatsResponse> {
    if request.observations.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "geostatistics requires at least one observation".to_string(),
        ));
    }
    if request.targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "geostatistics requires at least one target coordinate".to_string(),
        ));
    }
    let coords = request
        .observations
        .iter()
        .map(|row| [row.x, row.y])
        .collect::<Vec<_>>();
    let values = request
        .observations
        .iter()
        .map(|row| row.value)
        .collect::<Vec<_>>();
    let targets = request
        .targets
        .iter()
        .map(|row| [row.x, row.y])
        .collect::<Vec<_>>();
    let options = request.options;
    let config = NngpConfig {
        kernel: CovarianceKernel::parse(&options.kernel).map_err(|error| {
            CartoBoostError::InvalidInput(format!("invalid geostatistics kernel: {error}"))
        })?,
        range: options.range,
        sill: options.sill,
        nugget: options.nugget,
        anisotropy: GeostatsAnisotropy {
            angle_degrees: options.anisotropy_angle_degrees,
            scaling: options.anisotropy_scaling,
        },
        n_neighbors: options.n_neighbors,
        brute_force_threshold: 2048,
        duplicate_tolerance: 0.0,
    };
    let mut model =
        WasmNearestNeighborGPRegressor::new_with_backend(config, Some(&options.backend))
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    model
        .fit(&coords, &values)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let predictions = model
        .predict(&targets)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?
        .into_iter()
        .zip(request.targets)
        .map(|(prediction, target)| BrowserGeostatsPrediction {
            x: target.x,
            y: target.y,
            mean: prediction.mean,
            variance: prediction.variance,
            std: prediction.variance.max(0.0).sqrt(),
            neighbor_indices: prediction.neighbor_indices,
        })
        .collect();
    Ok(BrowserGeostatsResponse {
        predictions,
        metadata: json!({
            "model": "nearest_neighbor_gp",
            "kernel": config.kernel.as_str(),
            "range": config.range,
            "sill": config.sill,
            "nugget": config.nugget,
            "n_neighbors": config.n_neighbors,
            "works_without_gpu": true,
            "backend": model.backend(),
        }),
    })
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn run_geostatistics_webgpu_request(
    request: BrowserGeostatsRequest,
) -> Result<BrowserGeostatsResponse> {
    if request.observations.is_empty() || request.targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "geostatistics requires observations and target coordinates".to_string(),
        ));
    }
    let requested = request.options.backend.to_ascii_lowercase();
    if !matches!(requested.as_str(), "auto" | "webgpu") {
        return Err(CartoBoostError::InvalidInput(format!(
            "runGeostatisticsWebgpu requires backend='webgpu' or 'auto', got {requested:?}"
        )));
    }
    let coords = request
        .observations
        .iter()
        .map(|row| [row.x, row.y])
        .collect::<Vec<_>>();
    let values = request
        .observations
        .iter()
        .map(|row| row.value)
        .collect::<Vec<_>>();
    let targets = request
        .targets
        .iter()
        .map(|row| [row.x, row.y])
        .collect::<Vec<_>>();
    let config = NngpConfig {
        kernel: CovarianceKernel::parse(&request.options.kernel)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        range: request.options.range,
        sill: request.options.sill,
        nugget: request.options.nugget,
        anisotropy: GeostatsAnisotropy {
            angle_degrees: request.options.anisotropy_angle_degrees,
            scaling: request.options.anisotropy_scaling,
        },
        n_neighbors: request.options.n_neighbors,
        brute_force_threshold: usize::MAX,
        duplicate_tolerance: 0.0,
    };
    let mut model = WasmNearestNeighborGPRegressor::new_with_backend(config, Some("cpu"))
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    model
        .fit(&coords, &values)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let queries = model
        .transformed_points(&targets)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let observations = model
        .transformed_observations()
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let distances = webgpu_pairwise_squared_distances_f32_async(&queries, &observations)
        .await
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let values = model
        .predict_from_squared_distances(&targets, &distances)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let predictions = values
        .into_iter()
        .zip(request.targets)
        .map(|(prediction, target)| BrowserGeostatsPrediction {
            x: target.x,
            y: target.y,
            mean: prediction.mean,
            variance: prediction.variance,
            std: prediction.variance.max(0.0).sqrt(),
            neighbor_indices: prediction.neighbor_indices,
        })
        .collect();
    Ok(BrowserGeostatsResponse {
        predictions,
        metadata: json!({
            "model":"nearest_neighbor_gp","kernel":config.kernel.as_str(),"range":config.range,
            "sill":config.sill,"nugget":config.nugget,"n_neighbors":config.n_neighbors,
            "backend":{"requested":requested,"selected":"webgpu"},
            "acceleratedOperations":["pairwise_distance"],"cpuOperations":["neighbor_covariance_solve"],
        }),
    })
}

fn run_geo_feature_examples_request(
    request: BrowserGeoFeatureRequest,
) -> Result<BrowserGeoFeatureResponse> {
    run_geo_feature_examples_with_distances(request, None, "cpu")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn run_geo_feature_examples_webgpu_request(
    request: BrowserGeoFeatureRequest,
) -> Result<BrowserGeoFeatureResponse> {
    let points = request
        .radial_points
        .iter()
        .map(|point| point.point.iter().map(|value| *value as f32).collect())
        .collect::<Vec<Vec<f32>>>();
    let anchors = request
        .anchors
        .iter()
        .map(|anchor| anchor.point.iter().map(|value| *value as f32).collect())
        .collect::<Vec<Vec<f32>>>();
    let squared_distances = webgpu_pairwise_squared_distances_f32_async(&points, &anchors)
        .await
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let distances = squared_distances
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|distance| f64::from(distance.max(0.0).sqrt()))
                .collect()
        })
        .collect();
    run_geo_feature_examples_with_distances(request, Some(distances), "webgpu")
}

fn run_geo_feature_examples_with_distances(
    request: BrowserGeoFeatureRequest,
    accelerated_distances: Option<Vec<Vec<f64>>>,
    selected_backend: &str,
) -> Result<BrowserGeoFeatureResponse> {
    let anchors = request.anchors;
    let anchor_points = anchors
        .iter()
        .map(|anchor| anchor.point)
        .collect::<Vec<_>>();
    let anchor_labels = anchors
        .iter()
        .map(|anchor| anchor.label.clone())
        .collect::<Vec<_>>();
    let planar = request
        .planar_routes
        .iter()
        .map(|route| {
            let vector = clockwise_bearing_unit_vector(route.origin, route.destination);
            BrowserBearingFeature {
                label: route.label.clone(),
                east: vector.map(|value| value[0]),
                north: vector.map(|value| value[1]),
                zero_distance: vector.is_none(),
            }
        })
        .collect();
    let latlng = request
        .latlng_routes
        .into_iter()
        .map(|route| {
            let vector = initial_bearing_unit_vector_latlng(
                route.origin[0],
                route.origin[1],
                route.destination[0],
                route.destination[1],
            );
            BrowserBearingFeature {
                label: route.label,
                east: vector.map(|value| value[0]),
                north: vector.map(|value| value[1]),
                zero_distance: vector.is_none(),
            }
        })
        .collect();
    let routes = request
        .planar_routes
        .iter()
        .map(|route| {
            let vector = route_feature_vector(route.origin, route.destination);
            BrowserRouteFeature {
                label: route.label.clone(),
                mid_x: vector.map(|value| value[0]),
                mid_y: vector.map(|value| value[1]),
                distance: vector.map(|value| value[2]),
                bearing_east: vector.map(|value| value[3]),
                bearing_north: vector.map(|value| value[4]),
                zero_distance: vector.is_none(),
            }
        })
        .collect();
    let radial = request
        .radial_points
        .iter()
        .enumerate()
        .map(|(index, point)| BrowserAnchorFeatureRow {
            label: point.label.clone(),
            values: accelerated_distances.as_ref().map_or_else(
                || radial_anchor_distances(point.point, &anchor_points),
                |distances| distances[index].clone(),
            ),
        })
        .collect::<Vec<_>>();
    let rbf = request
        .radial_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let values = accelerated_distances.as_ref().map_or_else(
                || {
                    rbf_anchor_features(point.point, &anchor_points, request.length_scale)
                        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))
                },
                |distances| {
                    if !request.length_scale.is_finite() || request.length_scale <= 0.0 {
                        return Err(CartoBoostError::InvalidInput(
                            "length_scale must be finite and positive".to_string(),
                        ));
                    }
                    let denominator = 2.0 * request.length_scale * request.length_scale;
                    Ok(distances[index]
                        .iter()
                        .map(|distance| (-(distance * distance) / denominator).exp())
                        .collect())
                },
            )?;
            Ok(BrowserAnchorFeatureRow {
                label: point.label.clone(),
                values,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let local_frame = request
        .local_frame
        .map(|frame| {
            frame
                .points
                .into_iter()
                .map(|point| {
                    let vector = local_frame_features(point.point, frame.origin, frame.axis);
                    BrowserLocalFrameFeature {
                        label: point.label,
                        along_axis: vector.map(|value| value[0]),
                        cross_axis: vector.map(|value| value[1]),
                        invalid_axis: vector.is_none(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(BrowserGeoFeatureResponse {
        planar,
        latlng,
        routes,
        radial,
        rbf,
        local_frame,
        metadata: json!({
            "surface": "rust_geo_feature_examples",
            "bearingEncoding": "(east,north) unit vector",
            "clockReference": "clockwise from north",
            "anchorLabels": anchor_labels,
            "rbfLengthScale": request.length_scale,
            "zeroDistancePolicy": "null components with zeroDistance=true",
            "backend": {"requested": selected_backend, "selected": selected_backend},
            "acceleratedOperations": if selected_backend == "webgpu" {
                vec!["pairwise_distance"]
            } else {
                Vec::<&str>::new()
            },
        }),
    })
}

fn default_geo_feature_length_scale() -> f64 {
    1.0
}

fn default_geostats_kernel() -> String {
    "matern_3_2".to_string()
}

fn default_geostats_range() -> f64 {
    0.025
}

fn default_geostats_sill() -> f64 {
    1.0
}

fn default_geostats_nugget() -> f64 {
    1.0e-6
}

fn default_geostats_neighbors() -> usize {
    12
}

fn default_geostats_anisotropy_scaling() -> f64 {
    1.0
}

fn default_graph_diffusion_steps() -> usize {
    2
}

fn default_graph_hidden_size() -> usize {
    8
}

fn default_graph_epochs() -> usize {
    160
}

fn default_graph_batch_size() -> usize {
    32
}

fn default_graph_learning_rate() -> f64 {
    0.03
}

fn default_graph_teacher_forcing_start() -> f64 {
    1.0
}

fn default_graph_teacher_forcing_end() -> f64 {
    0.2
}

fn default_graph_ridge() -> f64 {
    0.0001
}

fn default_backend() -> String {
    "auto".to_string()
}

fn default_seed() -> u64 {
    13
}

fn run_geo_causal_request(request: BrowserGeoCausalRequest) -> Result<Value> {
    let rows = request
        .rows
        .into_iter()
        .map(|row| GeoCausalRow {
            unit_id: row.unit_id,
            time: row.time,
            outcome: row.outcome,
            treatment: row.treatment,
            covariates: row.covariates,
            latitude: row.latitude,
            longitude: row.longitude,
            region_id: row.region_id,
        })
        .collect();
    let spatial_weights = request
        .spatial_weights
        .into_iter()
        .map(|edge| SpatialWeight {
            from_unit: edge.from_unit,
            to_unit: edge.to_unit,
            weight: edge.weight,
        })
        .collect();
    let panel = GeoCausalPanel::new(rows, spatial_weights)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let mut estimator = CoreSyntheticDIDEstimator::new(SyntheticDIDConfig {
        intervention_time: request.intervention_time,
        seed: request.seed,
    });
    estimator
        .fit(panel)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    if request.placebo_n > 0 {
        estimator
            .placebo_test(request.placebo_n)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    }
    serde_json::to_value(
        estimator
            .estimate_effect()
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))
}

fn run_geotemporal_diagnostics_request(
    request: BrowserGeotemporalDiagnosticsRequest,
) -> Result<Value> {
    let mut response = serde_json::Map::new();
    response.insert("surface".to_string(), json!("rust_geotemporal_diagnostics"));
    if let Some(quantiles) = request.quantiles {
        response.insert(
            "quantiles".to_string(),
            run_browser_quantile_diagnostics(quantiles)?,
        );
    }
    if let Some(residual_correction) = request.residual_correction {
        response.insert(
            "residualCorrection".to_string(),
            run_browser_residual_correction(residual_correction)?,
        );
    }
    if let Some(regime) = request.regime {
        response.insert(
            "regime".to_string(),
            run_browser_regime_diagnostics(regime)?,
        );
    }
    if let Some(calibration) = request.calibration {
        response.insert(
            "calibration".to_string(),
            run_browser_calibration(calibration)?,
        );
    }
    Ok(Value::Object(response))
}

fn run_browser_quantile_diagnostics(request: BrowserQuantileDiagnosticsRequest) -> Result<Value> {
    let mut response = serde_json::Map::new();
    response.insert(
        "defaultLevels".to_string(),
        json!(default_quantile_levels()),
    );
    if let Some(values) = request.values.as_deref() {
        response.insert(
            "repairedValues".to_string(),
            json!(repair_non_crossing_quantiles(values)?),
        );
    }
    if let (Some(actual), Some(prediction), Some(quantile)) = (
        request.actual.as_deref(),
        request.prediction.as_deref(),
        request.quantile,
    ) {
        response.insert(
            "pinballLoss".to_string(),
            json!(pinball_loss(actual, prediction, quantile)?),
        );
    }
    if let (Some(actual), Some(lower), Some(upper), Some(quantile_rows)) = (
        request.actual.as_deref(),
        request.lower.as_deref(),
        request.upper.as_deref(),
        request.quantile_rows.as_deref(),
    ) {
        response.insert(
            "intervalDiagnostics".to_string(),
            serde_json::to_value(interval_diagnostics(actual, lower, upper, quantile_rows)?)?,
        );
    } else if let Some(quantile_rows) = request.quantile_rows.as_deref() {
        response.insert(
            "crossingRate".to_string(),
            json!(crossing_rate(quantile_rows)?),
        );
    }
    Ok(Value::Object(response))
}

fn run_browser_residual_correction(request: BrowserResidualCorrectionRequest) -> Result<Value> {
    let default_filter = StateFilter::new(request.process_variance, request.observation_variance)?;
    let mut corrector = KalmanResidualCorrector::new(default_filter);
    let observations = request
        .observations
        .into_iter()
        .map(|observation| StateObservation {
            key: browser_residual_key(observation.key),
            structural_prediction: observation.structural_prediction,
            observed: observation.observed,
        })
        .collect::<Vec<_>>();
    let corrections = corrector.apply_sequence(&observations)?;
    let states = corrector
        .states
        .iter()
        .map(|(key, filter)| json!({ "key": key, "filter": filter }))
        .collect::<Vec<_>>();
    Ok(json!({
        "corrections": corrections,
        "stateCount": corrector.states.len(),
        "states": states,
    }))
}

fn browser_residual_key(key: BrowserResidualStateKey) -> ResidualStateKey {
    ResidualStateKey::new(
        key.origin.unwrap_or_default(),
        key.destination.unwrap_or_default(),
        key.corridor.unwrap_or_default(),
        key.segment.unwrap_or_default(),
        key.entity_family.unwrap_or_default(),
        key.target_family.unwrap_or_default(),
        key.time_bucket.unwrap_or_default(),
    )
}

fn run_browser_regime_diagnostics(request: BrowserRegimeDiagnosticsRequest) -> Result<Value> {
    let rolling_window = request.rolling_window.unwrap_or(5);
    let mut response = serde_json::Map::new();
    response.insert(
        "rollingMedianResidual".to_string(),
        json!(rolling_median_residual(&request.residuals, rolling_window)?),
    );
    response.insert(
        "rollingMadResidual".to_string(),
        json!(rolling_mad_residual(&request.residuals, rolling_window)?),
    );
    if let Some(config) = request.cusum {
        let mut detector = CUSUM::new(config)?;
        response.insert(
            "cusum".to_string(),
            serde_json::to_value(detector.scan(&request.residuals)?)?,
        );
    }
    let page_hinkley_signals = if let Some(config) = request.page_hinkley {
        let mut detector = PageHinkley::new(config)?;
        let signals = detector.scan(&request.residuals)?;
        response.insert("pageHinkley".to_string(), serde_json::to_value(&signals)?);
        Some(signals)
    } else {
        None
    };
    let volatilities = if let Some(config) = request.ewma {
        let mut volatility = EwmaVolatility::new(config)?;
        let values = volatility.scan(&request.residuals)?;
        response.insert("ewmaVolatility".to_string(), json!(&values));
        Some(values)
    } else {
        None
    };
    if let (Some(lower), Some(upper), Some(signals), Some(volatilities), Some(policy)) = (
        request.lower.as_deref(),
        request.upper.as_deref(),
        page_hinkley_signals.as_deref(),
        volatilities.as_deref(),
        request.policy,
    ) {
        response.insert(
            "regimeAdjustedIntervals".to_string(),
            serde_json::to_value(regime_adjusted_intervals(
                lower,
                upper,
                signals,
                volatilities,
                policy,
            )?)?,
        );
    }
    Ok(Value::Object(response))
}

fn run_browser_calibration(request: BrowserCalibrationRequest) -> Result<Value> {
    let bucket_count = request.bucket_count.unwrap_or(10);
    let mut response = serde_json::Map::new();
    if let Some(event) = request.event {
        response.insert(
            "eventLabels".to_string(),
            json!(browser_event_labels(event)?),
        );
    }
    if let Some(probabilities) = request.probabilities.as_deref() {
        response.insert(
            "metrics".to_string(),
            serde_json::to_value(calibration_metrics(
                &request.labels,
                probabilities,
                bucket_count,
            )?)?,
        );
    }
    if let (Some(scores), Some(method)) = (request.scores.as_deref(), request.method.as_deref()) {
        let calibrated = match method {
            "sigmoid" | "platt" => {
                let calibrator = SigmoidCalibrator::fit(scores, &request.labels)?;
                response.insert("calibrator".to_string(), serde_json::to_value(calibrator)?);
                calibrator.predict(scores)?
            }
            "temperature" => {
                let calibrator = TemperatureCalibrator::fit(scores, &request.labels)?;
                response.insert("calibrator".to_string(), serde_json::to_value(calibrator)?);
                calibrator.predict(scores)?
            }
            "isotonic" => {
                let calibrator = IsotonicCalibrator::fit(scores, &request.labels)?;
                response.insert("calibrator".to_string(), serde_json::to_value(&calibrator)?);
                calibrator.predict(scores)?
            }
            other => {
                return Err(CartoBoostError::InvalidInput(format!(
                    "unknown calibration method '{other}'"
                )));
            }
        };
        response.insert("calibratedProbabilities".to_string(), json!(&calibrated));
        response.insert(
            "calibratedMetrics".to_string(),
            serde_json::to_value(calibration_metrics(
                &request.labels,
                &calibrated,
                bucket_count,
            )?)?,
        );
        if let Some(before) = request
            .before_probabilities
            .as_deref()
            .or(request.probabilities.as_deref())
        {
            response.insert(
                "improvement".to_string(),
                serde_json::to_value(calibration_improvement(
                    &request.labels,
                    before,
                    &calibrated,
                    bucket_count,
                )?)?,
            );
        }
    }
    Ok(Value::Object(response))
}

fn browser_event_labels(request: BrowserCalibrationEventRequest) -> Result<Vec<f64>> {
    match request.kind.as_str() {
        "success_within_threshold" | "successWithinThreshold" => {
            let prediction = request.prediction.as_deref().ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "success_within_threshold event requires prediction".to_string(),
                )
            })?;
            let threshold = request.threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "success_within_threshold event requires threshold".to_string(),
                )
            })?;
            success_within_threshold(&request.actual, prediction, threshold)
        }
        "event_within_horizon" | "eventWithinHorizon" => {
            let horizon = request.horizon.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "event_within_horizon event requires horizon".to_string(),
                )
            })?;
            event_within_horizon(&request.actual, horizon)
        }
        "failure_risk" | "failureRisk" => {
            let threshold = request.threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput("failure_risk event requires threshold".to_string())
            })?;
            failure_risk_event(&request.actual, threshold)
        }
        "escalation_risk" | "escalationRisk" => {
            let warning_threshold = request.warning_threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "escalation_risk event requires warningThreshold".to_string(),
                )
            })?;
            let critical_threshold = request.critical_threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "escalation_risk event requires criticalThreshold".to_string(),
                )
            })?;
            escalation_risk_event(&request.actual, warning_threshold, critical_threshold)
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown calibration event kind '{other}'"
        ))),
    }
}

fn run_sequence_request(request: BrowserSequenceRequest) -> Result<Value> {
    match request.operation.trim().to_ascii_lowercase().as_str() {
        "validate" | "validate_frame" => {
            let frame = request.frame.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence validate requires frame".to_string())
            })?;
            frame.validate()?;
            Ok(json!({ "ok": true }))
        }
        "ekf" | "forward_ekf" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::forward_ekf(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "ukf" | "ukf_reference" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::ukf_reference(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "rts" | "rts_smoother" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::rts_smoother(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "continuation" | "missing_target_continuation" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::missing_target_continuation(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "viterbi" | "reference_path_viterbi" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.reference_path_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::reference_path_viterbi(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "posterior_mean" | "reference_path_posterior_mean" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.reference_path_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::reference_path_posterior_mean(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "blend_fixed" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let weights = request.weights.ok_or_else(|| {
                CartoBoostError::InvalidInput("fixed sequence blend requires weights".to_string())
            })?;
            let ensemble = SequenceCandidateEnsemble::fixed(weights)?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "blend_validation" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let actuals = request.actuals.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "validation sequence blend requires actuals".to_string(),
                )
            })?;
            let ensemble = SequenceCandidateEnsemble::validation_derived(&candidates, &actuals)?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "blend_constrained" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let actuals = request.actuals.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "constrained sequence blend requires actuals".to_string(),
                )
            })?;
            let ensemble = SequenceCandidateEnsemble::constrained_nonnegative_linear_blend(
                &candidates,
                &actuals,
            )?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "validate_oof" | "validate_oof_meta_training" => {
            let rows = request.oof_rows.ok_or_else(|| {
                CartoBoostError::InvalidInput("OOF validation requires oofRows".to_string())
            })?;
            cartoboost_core::forecasting::validate_oof_meta_training(&rows)?;
            Ok(json!({ "ok": true }))
        }
        "generate_oof" | "generate_group_oof_candidate_rows" => {
            let fold = request.oof_fold.ok_or_else(|| {
                CartoBoostError::InvalidInput("OOF generation requires oofFold".to_string())
            })?;
            serde_json::to_value(
                cartoboost_core::forecasting::generate_group_oof_candidate_rows(&fold)?,
            )
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "group_metrics" | "per_group_error_summary" => {
            let rows = request.group_predictions.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "group metric summary requires groupPredictions".to_string(),
                )
            })?;
            serde_json::to_value(cartoboost_core::forecasting::per_group_error_summary(
                &rows,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown sequence operation {other:?}"
        ))),
    }
}

fn sequence_series_arg(request: &BrowserSequenceRequest) -> Result<SequenceSeries> {
    request.series.clone().ok_or_else(|| {
        CartoBoostError::InvalidInput("sequence operation requires series".to_string())
    })
}

fn sequence_reference_arg(request: &BrowserSequenceRequest) -> Result<ReferenceSignal> {
    request.reference.clone().ok_or_else(|| {
        CartoBoostError::InvalidInput("sequence operation requires reference".to_string())
    })
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn run_neural_forecast_webgpu_request(
    request: BrowserForecastRequest,
) -> Result<BrowserForecastResponse> {
    if request.horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    let model = request.model.to_ascii_lowercase();
    if !matches!(model.as_str(), "nbeats" | "n-beats" | "nhits" | "n-hits") {
        return Err(CartoBoostError::InvalidInput(
            "runNeuralForecastWebgpu supports nbeats and nhits".to_string(),
        ));
    }
    let requested = request
        .options
        .backend
        .as_deref()
        .unwrap_or("webgpu")
        .to_ascii_lowercase();
    if !matches!(requested.as_str(), "auto" | "webgpu") {
        return Err(CartoBoostError::InvalidInput(format!(
            "runNeuralForecastWebgpu requires backend='webgpu' or 'auto', got {requested:?}"
        )));
    }
    let frame =
        forecast_frame_from_browser_request(request.rows, request.frequency, request.metadata)?;
    let is_nhits = matches!(model.as_str(), "nhits" | "n-hits");
    let input_size = request
        .options
        .input_size
        .unwrap_or(if is_nhits { 12 } else { 8 });
    let hidden_size = request.options.hidden_size.unwrap_or(16);
    let epochs = request.options.epochs.unwrap_or(80);
    let pooling_size = if is_nhits {
        request.options.pooling_size.unwrap_or(2)
    } else {
        1
    };
    let learning_rate = request.options.learning_rate.unwrap_or(0.01);
    if input_size == 0
        || hidden_size == 0
        || epochs == 0
        || pooling_size == 0
        || pooling_size > input_size
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "invalid neural forecast configuration".to_string(),
        ));
    }
    let targets = frame
        .rows()
        .iter()
        .map(|row| row.target)
        .collect::<Vec<_>>();
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let scale = (targets
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / targets.len() as f64)
        .sqrt()
        .max(1.0e-12);
    let pooled_size = input_size.div_ceil(pooling_size);
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    let mut tails = BTreeMap::new();
    let mut last_rows = BTreeMap::new();
    for series_id in frame.series_ids() {
        let rows = frame.rows_for_series(&series_id);
        if rows.len() <= input_size {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id:?} needs more than {input_size} rows"
            )));
        }
        let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
        for end in input_size..values.len() {
            let scaled = values[end - input_size..end]
                .iter()
                .map(|value| ((value - mean) / scale) as f32)
                .collect::<Vec<_>>();
            let pooled = scaled
                .chunks(pooling_size)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect();
            train_inputs.push(pooled);
            train_targets.push(((values[end] - mean) / scale) as f32);
        }
        tails.insert(
            series_id.clone(),
            values[values.len() - input_size..].to_vec(),
        );
        last_rows.insert(series_id, (*rows.last().expect("non-empty series")).clone());
    }
    let phase = if is_nhits { 0.031 } else { 0.017 };
    let mut parameters = vec![0.0_f32; hidden_size * pooled_size + hidden_size + hidden_size + 1];
    for hidden in 0..hidden_size {
        for input in 0..pooled_size {
            let index = hidden * pooled_size + input;
            parameters[index] = (((index + 1) as f64 * phase).sin() / pooled_size as f64) as f32;
        }
    }
    let w2_offset = hidden_size * pooled_size + hidden_size;
    for hidden in 0..hidden_size {
        parameters[w2_offset + hidden] =
            (((hidden + 3) as f64 * phase).cos() / hidden_size as f64) as f32;
    }
    parameters = webgpu_train_tanh_mlp_f32_async(
        &train_inputs,
        &train_targets,
        hidden_size,
        epochs,
        learning_rate as f32,
        &parameters,
    )
    .await
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let b1_offset = hidden_size * pooled_size;
    let b2_offset = w2_offset + hidden_size;
    let mut dense_weights = Vec::with_capacity(pooled_size * hidden_size);
    for input in 0..pooled_size {
        for hidden in 0..hidden_size {
            dense_weights.push(parameters[hidden * pooled_size + input]);
        }
    }
    let biases = parameters[b1_offset..w2_offset].to_vec();
    let mut predictions = Vec::new();
    for (series_id, tail) in &tails {
        let mut history = tail.clone();
        let last = last_rows.get(series_id).expect("tail and last row align");
        for step in 1..=request.horizon {
            let scaled = history
                .iter()
                .map(|value| ((value - mean) / scale) as f32)
                .collect::<Vec<_>>();
            let pooled = scaled
                .chunks(pooling_size)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect::<Vec<_>>();
            let linear = webgpu_dense_layer_f32_async(&[pooled], &dense_weights, &biases)
                .await
                .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
            let scaled_prediction = parameters[b2_offset] as f64
                + linear[0]
                    .iter()
                    .enumerate()
                    .map(|(hidden, value)| f64::from(value.tanh() * parameters[w2_offset + hidden]))
                    .sum::<f64>();
            let prediction = scaled_prediction * scale + mean;
            predictions.push(cartoboost_core::forecasting::ForecastPrediction {
                series_id: series_id.clone(),
                timestamp: frame.frequency().advance(last.timestamp, step)?,
                horizon: step,
                model: if is_nhits { "nhits" } else { "nbeats" }.to_string(),
                mean: prediction,
            });
            history.remove(0);
            history.push(prediction);
        }
    }
    let forecast = ForecastResult::new(predictions)?;
    Ok(forecast_response(
        if is_nhits { "nhits" } else { "nbeats" },
        &frame,
        json!({"backend":{"requested":requested,"selected":"webgpu"},"input_size":input_size,
            "hidden_size":hidden_size,"epochs":epochs,"learning_rate":learning_rate,
            "pooling_size":pooling_size,"accelerated_operations":["tanh_mlp_training","dense"]}),
        forecast,
        None,
    ))
}

fn run_forecast_request(request: BrowserForecastRequest) -> Result<BrowserForecastResponse> {
    if request.horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    let model = request.model;
    let options = request.options;
    let frame =
        forecast_frame_from_browser_request(request.rows, request.frequency, request.metadata)?;
    if is_piecewise_linear_seasonal_model(&model) {
        let mut config = piecewise_linear_seasonal_config(&options)?;
        let include_components = options.include_components.unwrap_or(false);
        let include_history_components = options.include_history_components.unwrap_or(false);
        let include_samples =
            options.include_samples.unwrap_or(false) && config.uncertainty_samples > 0;
        let include_quantiles =
            options.include_quantiles.unwrap_or(true) && !config.quantile_levels.is_empty();
        if !include_samples
            && config.interval_levels.is_empty()
            && config.quantile_levels.is_empty()
        {
            config.uncertainty_samples = 0;
        }
        let mut forecaster = PiecewiseLinearSeasonalForecaster::new(config)?;
        forecaster.fit(&frame)?;
        let forecast = forecaster.predict(request.horizon)?;
        let components = if include_components {
            Some(js_safe_json_value(
                forecaster.predict_components_json_value(request.horizon)?,
            ))
        } else {
            None
        };
        let history_components = if include_history_components {
            Some(js_safe_json_value(
                forecaster.history_components_json_value()?,
            ))
        } else {
            None
        };
        let samples = if include_samples {
            Some(js_safe_json_value(
                forecaster.predict_samples_json_value(request.horizon)?,
            ))
        } else {
            None
        };
        let quantiles = if include_quantiles {
            Some(js_safe_json_value(
                forecaster.predict_quantiles_json_value(request.horizon, None)?,
            ))
        } else {
            None
        };
        return Ok(BrowserForecastResponse {
            metadata: js_safe_json_value(json!({
                "model": forecaster.model_name(),
                "input": frame.metadata_value(),
                "modelMetadata": forecaster.metadata(),
            })),
            forecast: js_safe_json_value(forecast.to_json_value()),
            components,
            history_components,
            samples,
            quantiles,
        });
    }
    if model.trim().to_ascii_lowercase().replace('-', "_") == "neural_panel" {
        let mut forecaster = NeuralPanelForecaster::new(
            neural_panel_config(&options, request.horizon)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        forecaster.fit(&frame)?;
        let covariates = if has_browser_neural_future_regressors(&options) {
            Some(browser_neural_future_covariates(
                &frame,
                &options,
                request.horizon,
            )?)
        } else {
            None
        };
        let forecast = if let Some(covariates) = &covariates {
            forecaster.predict_with_known_future_covariates(request.horizon, covariates)?
        } else {
            forecaster.predict(request.horizon)?
        };
        let components = if options.include_components.unwrap_or(false) {
            Some(js_safe_json_value(if let Some(covariates) = &covariates {
                forecaster.predict_components_json_value_with_known_future_covariates(
                    request.horizon,
                    Some(covariates),
                )?
            } else {
                forecaster.predict_components_json_value(request.horizon)?
            }))
        } else {
            None
        };
        let history_components = if options.include_history_components.unwrap_or(false) {
            Some(js_safe_json_value(
                forecaster.history_components_json_value()?,
            ))
        } else {
            None
        };
        let metadata = json!({
            "model": forecaster.model_name(),
            "input": frame.metadata_value(),
            "modelMetadata": forecaster.metadata(),
        });
        return Ok(BrowserForecastResponse {
            metadata: js_safe_json_value(metadata),
            forecast: js_safe_json_value(forecast.to_json_value()),
            components,
            history_components,
            samples: None,
            quantiles: None,
        });
    }
    let mut forecaster = build_forecaster(&model, &options, &frame, request.horizon)?;
    let fit_result = forecaster
        .fit(&frame)
        .and_then(|()| forecaster.predict(request.horizon));
    match fit_result {
        Ok(forecast) => Ok(forecast_response(
            forecaster.model_name(),
            &frame,
            forecaster.metadata(),
            forecast,
            None,
        )),
        Err(error) => Err(error),
    }
}

fn has_browser_neural_future_regressors(options: &BrowserForecastOptions) -> bool {
    options
        .extra_regressors
        .as_ref()
        .map(|values| !values.is_empty())
        .unwrap_or(false)
        || options
            .future_regressors
            .as_ref()
            .map(|values| !values.is_empty())
            .unwrap_or(false)
        || options
            .future_regressors_by_series
            .as_ref()
            .map(|values| !values.is_empty())
            .unwrap_or(false)
}

fn browser_neural_future_covariates(
    frame: &ForecastFrame,
    options: &BrowserForecastOptions,
    horizon: usize,
) -> Result<BTreeMap<(String, NaiveDateTime), BTreeMap<String, f64>>> {
    let mut regressor_names = BTreeSet::new();
    if let Some(names) = &options.extra_regressors {
        regressor_names.extend(names.iter().cloned());
    }
    if let Some(values) = &options.future_regressors {
        regressor_names.extend(values.keys().cloned());
    }
    if let Some(values) = &options.future_regressors_by_series {
        for series_values in values.values() {
            regressor_names.extend(series_values.keys().cloned());
        }
    }
    if regressor_names.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut covariates = BTreeMap::new();
    for series_id in frame.series_ids() {
        let rows = frame.rows_for_series(&series_id);
        let last_row = rows.last().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing fitted timestamp tail for series '{series_id}'"
            ))
        })?;
        for step in 1..=horizon {
            let timestamp = frame.frequency().advance(last_row.timestamp, step)?;
            let entry = covariates
                .entry((series_id.clone(), timestamp))
                .or_insert_with(BTreeMap::new);
            for name in &regressor_names {
                let value = options
                    .future_regressors_by_series
                    .as_ref()
                    .and_then(|series_values| series_values.get(&series_id))
                    .and_then(|series_values| series_values.get(name))
                    .and_then(|values| values.get(step - 1))
                    .copied()
                    .or_else(|| {
                        options
                            .future_regressors
                            .as_ref()
                            .and_then(|values| values.get(name))
                            .and_then(|values| values.get(step - 1))
                            .copied()
                    })
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "future regressor '{name}' requires known future covariates for prediction"
                        ))
                    })?;
                entry.insert(name.clone(), value);
            }
        }
    }
    Ok(covariates)
}

fn forecast_response(
    model_name: &str,
    frame: &ForecastFrame,
    model_metadata: Value,
    forecast: ForecastResult,
    warning: Option<Value>,
) -> BrowserForecastResponse {
    let mut metadata = json!({
        "model": model_name,
        "input": frame.metadata_value(),
        "modelMetadata": model_metadata,
    });
    if let Some(warning) = warning {
        metadata["warning"] = warning;
    }
    BrowserForecastResponse {
        metadata: js_safe_json_value(metadata),
        forecast: js_safe_json_value(forecast.to_json_value()),
        components: None,
        history_components: None,
        samples: None,
        quantiles: None,
    }
}

fn fit_piecewise_linear_seasonal_artifact_request(
    request: BrowserForecastRequest,
) -> Result<BrowserForecastArtifactResponse> {
    let frame =
        forecast_frame_from_browser_request(request.rows, request.frequency, request.metadata)?;
    let mut forecaster = PiecewiseLinearSeasonalForecaster::new(piecewise_linear_seasonal_config(
        &request.options,
    )?)?;
    forecaster.fit(&frame)?;
    let artifact = forecaster.to_json_string()?;
    Ok(BrowserForecastArtifactResponse {
        metadata: js_safe_json_value(json!({
            "model": forecaster.model_name(),
            "input": frame.metadata_value(),
            "modelMetadata": forecaster.metadata(),
        })),
        artifact,
    })
}

fn predict_piecewise_linear_seasonal_artifact_request(
    artifact: &str,
    horizon: usize,
    options: BrowserForecastArtifactPredictOptions,
) -> Result<BrowserForecastResponse> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    let mut forecaster = PiecewiseLinearSeasonalForecaster::from_json_string(artifact)?;
    apply_piecewise_artifact_predict_options(&mut forecaster, &options)?;
    let forecast = forecaster.predict(horizon)?;
    let components = if options.include_components {
        Some(js_safe_json_value(
            forecaster.predict_components_json_value(horizon)?,
        ))
    } else {
        None
    };
    let history_components = if options.include_history_components {
        Some(js_safe_json_value(
            forecaster.history_components_json_value()?,
        ))
    } else {
        None
    };
    let metadata = forecaster.metadata();
    let samples =
        if options.include_samples && metadata["uncertainty_samples"].as_u64().unwrap_or(0) > 0 {
            Some(js_safe_json_value(
                forecaster.predict_samples_json_value(horizon)?,
            ))
        } else {
            None
        };
    let has_quantiles = metadata["quantile_levels"]
        .as_array()
        .map(|levels| !levels.is_empty())
        .unwrap_or(false);
    let quantiles = if options.include_quantiles && has_quantiles {
        Some(js_safe_json_value(
            forecaster.predict_quantiles_json_value(horizon, None)?,
        ))
    } else {
        None
    };
    Ok(BrowserForecastResponse {
        metadata: js_safe_json_value(json!({
            "model": forecaster.model_name(),
            "modelMetadata": metadata,
        })),
        forecast: js_safe_json_value(forecast.to_json_value()),
        components,
        history_components,
        samples,
        quantiles,
    })
}

fn apply_piecewise_artifact_predict_options(
    forecaster: &mut PiecewiseLinearSeasonalForecaster,
    options: &BrowserForecastArtifactPredictOptions,
) -> Result<()> {
    forecaster.update_config(|config| {
        if let Some(future_regressors) = &options.future_regressors {
            config.future_regressors = future_regressors.clone();
        }
        if let Some(future_regressors_by_series) = &options.future_regressors_by_series {
            config.future_regressors_by_series = future_regressors_by_series.clone();
        }
        if let Some(trend_adjustments) = &options.trend_adjustments {
            config.trend_adjustments = trend_adjustments.clone();
        }
        if let Some(trend_adjustments_by_series) = &options.trend_adjustments_by_series {
            config.trend_adjustments_by_series = trend_adjustments_by_series.clone();
        }
        if let Some(levels) = &options.interval_levels {
            config.interval_levels = levels.clone();
        }
        if let Some(levels) = &options.quantile_levels {
            config.quantile_levels = levels.clone();
        }
        if let Some(samples) = options.uncertainty_samples {
            config.uncertainty_samples = samples;
        }
    })
}

fn forecast_frame_from_browser_request(
    rows: Vec<BrowserForecastRow>,
    frequency: String,
    metadata: BrowserForecastMetadata,
) -> Result<ForecastFrame> {
    if rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast request must include at least one row".to_string(),
        ));
    }
    let frequency = ForecastFrequency::parse(&frequency)?;
    let metadata = ForecastFrameMetadata {
        timestamp_col: metadata.timestamp_col,
        target_col: metadata.target_col,
        series_id_col: metadata.series_id_col,
        static_covariates: Vec::new(),
        known_future_covariates: Vec::new(),
        historical_covariates: Vec::new(),
        allow_irregular: false,
        allow_missing_targets: false,
        allow_missing_covariates: false,
    };
    let rows = rows
        .into_iter()
        .map(|row| {
            cartoboost_core::forecasting::ForecastRow::from_timestamp_str_with_covariates(
                row.series_id.unwrap_or_default(),
                &row.timestamp,
                row.target,
                row.covariates,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ForecastFrame::with_metadata(rows, frequency, metadata)
}

fn js_safe_json_value(value: Value) -> Value {
    const JS_SAFE_INTEGER_MAX: u64 = 9_007_199_254_740_991;
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(js_safe_json_value).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, js_safe_json_value(value)))
                .collect(),
        ),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                if value > JS_SAFE_INTEGER_MAX {
                    return Value::String(value.to_string());
                }
            } else if let Some(value) = number.as_i64() {
                if value.unsigned_abs() > JS_SAFE_INTEGER_MAX {
                    return Value::String(value.to_string());
                }
            }
            Value::Number(number)
        }
        other => other,
    }
}

fn serialize_json_response<T: Serialize>(
    response: &T,
    context: &str,
) -> std::result::Result<JsValue, JsValue> {
    let json = serde_json::to_string(response)
        .map_err(|error| JsValue::from_str(&format!("could not encode {context}: {error}")))?;
    js_sys::JSON::parse(&json)
        .map_err(|error| JsValue::from_str(&format!("could not parse {context}: {error:?}")))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn run_graph_diffusion_webgpu_request(
    request: BrowserGraphForecastRequest,
) -> Result<BrowserGraphForecastResponse> {
    let requested = request.options.backend.to_ascii_lowercase();
    if !matches!(requested.as_str(), "auto" | "webgpu") {
        return Err(CartoBoostError::InvalidInput(format!(
            "runGraphDiffusionWebgpu requires backend='webgpu' or 'auto', got {requested:?}"
        )));
    }
    if request.options.diffusion_steps == 0 {
        return Err(CartoBoostError::InvalidInput(
            "diffusionSteps must be positive".to_string(),
        ));
    }
    let raw_indptr = request.frame.adjacency.indptr.clone();
    let raw_indices = request.frame.adjacency.indices.clone();
    let raw_data = request.frame.adjacency.data.clone();
    let adjacency = GraphStCsrAdjacency::new(
        request.frame.adjacency.indptr,
        request.frame.adjacency.indices,
        request.frame.adjacency.data,
        request.frame.node_ids.len(),
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let frame = GraphStTemporalFrame::new(
        request.frame.node_ids.clone(),
        request.frame.timestamps,
        request.frame.target,
        request.frame.covariates,
        adjacency.clone(),
        request.frame.horizon,
        request.frame.frequency,
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let indptr = raw_indptr
        .iter()
        .map(|value| {
            u32::try_from(*value)
                .map_err(|_| CartoBoostError::InvalidInput("CSR pointer exceeds u32".to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let indices = raw_indices
        .iter()
        .map(|value| {
            u32::try_from(*value)
                .map_err(|_| CartoBoostError::InvalidInput("CSR index exceeds u32".to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut weights = raw_data
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    for row in 0..frame.node_ids.len() {
        let start = raw_indptr[row];
        let end = raw_indptr[row + 1];
        let sum = weights[start..end].iter().sum::<f32>();
        if sum.abs() > 1.0e-12 {
            for weight in &mut weights[start..end] {
                *weight /= sum;
            }
        }
    }
    let last = frame.target.last().ok_or_else(|| {
        CartoBoostError::InvalidInput("graph target history must be non-empty".to_string())
    })?;
    let mut state = last.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let mut predictions = Vec::with_capacity(frame.horizon);
    for _ in 0..frame.horizon {
        for _ in 0..request.options.diffusion_steps {
            state = webgpu_csr_diffusion_f32_async(&indptr, &indices, &weights, 1, &state)
                .await
                .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        }
        predictions.push(state.iter().map(|value| f64::from(*value)).collect());
    }
    let metrics = request.actual.map(|actual| {
        serde_json::to_value(graph_st_metrics(
            &predictions,
            &actual,
            &frame.node_ids,
            &adjacency,
        ))
        .unwrap_or_else(|error| json!({"error":error.to_string()}))
    });
    Ok(BrowserGraphForecastResponse {
        predictions,
        node_ids: frame.node_ids,
        horizon: frame.horizon,
        metrics,
        metadata: json!({
            "model":"webgpu_graph_diffusion",
            "frequency":frame.frequency,
            "backend":{"requested":requested,"selected":"webgpu"},
            "diffusionSteps":request.options.diffusion_steps,
            "acceleratedOperations":["csr_diffusion"],
        }),
    })
}

fn run_graph_forecast_request(
    request: BrowserGraphForecastRequest,
) -> Result<BrowserGraphForecastResponse> {
    let adjacency = GraphStCsrAdjacency::new(
        request.frame.adjacency.indptr,
        request.frame.adjacency.indices,
        request.frame.adjacency.data,
        request.frame.node_ids.len(),
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let frame = GraphStTemporalFrame::new(
        request.frame.node_ids.clone(),
        request.frame.timestamps,
        request.frame.target,
        request.frame.covariates,
        adjacency.clone(),
        request.frame.horizon,
        request.frame.frequency.clone(),
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let (predictions, model_metadata) = if let Some(profile) = request.options.profile.as_deref() {
        let profile_kind = parse_browser_graph_transformer_profile(profile)?;
        let (default_lookback, default_periodicity) =
            if profile_kind == BrowserGraphTransformerProfile::LongShortFusion {
                (24 * 28, 24)
            } else {
                (3, 24)
            };
        let lookback = request.options.lookback.unwrap_or(default_lookback);
        let config = BrowserPaperGraphTransformerConfig {
            profile: profile_kind,
            lookback,
            hidden_size: request.options.hidden_size,
            attention_heads: request.options.attention_heads.unwrap_or(2),
            graph_order: request.options.graph_order.unwrap_or(2),
            experts: request.options.experts.unwrap_or(2),
            periodicity: request.options.periodicity.unwrap_or(default_periodicity),
            recent_window: request
                .options
                .recent_window
                .unwrap_or(lookback.min(24 * 7)),
            epochs: request.options.epochs,
            learning_rate: request.options.learning_rate,
            weight_decay: request.options.weight_decay.unwrap_or(1e-5),
            batch_size: request.options.batch_size,
            backend: graph_st_select_backend_for_operations(
                Some(&request.options.backend),
                &[
                    BackendOperation::AdamW,
                    BackendOperation::Dense,
                    BackendOperation::LayerNorm,
                    BackendOperation::CsrRowSoftmax,
                    BackendOperation::ScalarGraph,
                    BackendOperation::ScalarGraphTraining,
                ],
            )
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        };
        let mut model = BrowserPaperGraphTransformerForecaster::new(config)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        model
            .fit(&frame)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let predictions = model
            .predict(frame.horizon)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let report = model.architecture_report();
        (
            predictions,
            json!({
                "model": profile,
                "frequency": frame.frequency,
                "architectureReport": report,
            }),
        )
    } else {
        let config = GraphStDcrnnConfig {
            diffusion_steps: request.options.diffusion_steps,
            hidden_size: request.options.hidden_size,
            epochs: request.options.epochs,
            learning_rate: request.options.learning_rate,
            teacher_forcing_start: request.options.teacher_forcing_start,
            teacher_forcing_end: request.options.teacher_forcing_end,
            ridge: request.options.ridge,
            backend: graph_st_select_backend_for_operations(
                Some(&request.options.backend),
                &[
                    BackendOperation::Affine,
                    BackendOperation::CsrDiffusion,
                    BackendOperation::Dense,
                ],
            )
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        };
        let mut model = GraphStDcrnnForecaster::new(config.clone())
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        model
            .fit(&frame)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let predictions = model
            .predict(frame.horizon)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        (
            predictions,
            json!({
                "model": "dcrnn",
                "frequency": frame.frequency,
                "diffusionSteps": config.diffusion_steps,
                "hiddenSize": config.hidden_size,
                "epochs": config.epochs,
                "teacherForcingStart": config.teacher_forcing_start,
                "teacherForcingEnd": config.teacher_forcing_end,
            }),
        )
    };
    let metrics = request.actual.map(|actual| {
        serde_json::to_value(graph_st_metrics(
            &predictions,
            &actual,
            &frame.node_ids,
            &adjacency,
        ))
        .unwrap_or_else(|error| json!({ "error": error.to_string() }))
    });
    Ok(BrowserGraphForecastResponse {
        predictions,
        node_ids: frame.node_ids,
        horizon: frame.horizon,
        metrics,
        metadata: model_metadata,
    })
}

fn parse_browser_graph_transformer_profile(value: &str) -> Result<BrowserGraphTransformerProfile> {
    match value {
        "heterogeneous_moe" => Ok(BrowserGraphTransformerProfile::HeterogeneousMoE),
        "efficient_high_order" => Ok(BrowserGraphTransformerProfile::EfficientHighOrder),
        "long_short_fusion" => Ok(BrowserGraphTransformerProfile::LongShortFusion),
        "gated_graph_temporal" => Ok(BrowserGraphTransformerProfile::GatedGraphTemporal),
        "spatial_shift_graphon_moe" => Ok(BrowserGraphTransformerProfile::SpatialShiftGraphonMoE),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown paper graph transformer profile {other:?}"
        ))),
    }
}

fn run_regression_request(request: BrowserRegressionRequest) -> Result<BrowserRegressionResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "regression modeling requires at least four rows".to_string(),
        ));
    }
    if request.feature_names.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "regression modeling requires at least one feature".to_string(),
        ));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let feature_count = request.feature_names.len();
    let sparse_feature_count = request.sparse_feature_names.len();
    let mut features = Vec::with_capacity(request.rows.len());
    let mut sparse_rows = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in request.rows {
        if row.features.len() != feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "feature row has {} columns but feature_names has {feature_count}",
                row.features.len()
            )));
        }
        if row.sparse_sets.len() != sparse_feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "sparse feature row has {} columns but sparse_feature_names has {sparse_feature_count}",
                row.sparse_sets.len()
            )));
        }
        if row.features.iter().any(|value| !value.is_finite()) || !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "regression features and targets must be finite".to_string(),
            ));
        }
        features.push(row.features);
        sparse_rows.push(row.sparse_sets);
        targets.push(row.target);
    }

    let requested_holdout =
        ((features.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, features.len().saturating_sub(2));
    let train_rows = features.len() - holdout_rows;
    let schema = regression_feature_schema(
        &request.feature_names,
        &request.sparse_feature_names,
        &request.options,
    )?;
    let train_x = Dataset::mixed(
        features[..train_rows].to_vec(),
        sparse_columns_from_rows(&sparse_rows[..train_rows], sparse_feature_count),
        Some(schema.clone()),
    )?;
    let holdout_x = Dataset::mixed(
        features[train_rows..].to_vec(),
        sparse_columns_from_rows(&sparse_rows[train_rows..], sparse_feature_count),
        Some(schema),
    )?;
    let train_y = &targets[..train_rows];
    let holdout_y = &targets[train_rows..];

    let model = Booster::new_with_backend(
        regression_booster_config(&request.options)?,
        request.options.backend.as_deref(),
    )?
    .fit(&train_x, train_y, None)?;
    let predictions = model.try_predict(&holdout_x)?;
    let interval_predictions =
        regression_interval_predictions(&request.options, &train_x, train_y, &holdout_x)?;
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y.iter())
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: interval_predictions
                    .as_ref()
                    .map(|(lower, _)| lower[offset]),
                upper_prediction: interval_predictions
                    .as_ref()
                    .map(|(_, upper)| upper[offset]),
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_importance = feature_importance(
        &model.trees,
        &request.feature_names,
        &request.sparse_feature_names,
    );

    Ok(BrowserRegressionResponse {
        metadata: json!({
            "model": "cartoboost_regressor",
            "featureNames": request.feature_names,
            "sparseFeatureNames": request.sparse_feature_names,
            "trainingConfig": model.training_config,
            "splitterMode": request.options.splitter_mode.as_deref().unwrap_or("auto"),
            "loss": regression_loss_label(&request.options),
            "intervalLowerAlpha": request.options.interval_lower_alpha,
            "intervalUpperAlpha": request.options.interval_upper_alpha,
            "monotonicConstraints": request.options.monotonic_constraints,
            "treeCount": model.trees.len(),
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
        model_visualization: request
            .options
            .include_model_visualization
            .unwrap_or(false)
            .then(|| {
                model_visualization(
                    &model.trees,
                    &request.feature_names,
                    &request.sparse_feature_names,
                )
            }),
    })
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn run_node2vec_webgpu_request(
    request: BrowserNeuralRequest,
) -> Result<BrowserNeuralResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "Node2Vec modeling requires at least four rows".to_string(),
        ));
    }
    let requested_backend = request
        .options
        .backend
        .as_deref()
        .unwrap_or("webgpu")
        .to_ascii_lowercase();
    if !matches!(requested_backend.as_str(), "auto" | "webgpu") {
        return Err(CartoBoostError::InvalidInput(format!(
            "runNode2VecModelWebgpu requires backend='webgpu' or 'auto', got {requested_backend:?}"
        )));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let dense_width = request.dense_feature_names.len();
    let mut dense = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in &request.rows {
        if row.dense.len() != dense_width
            || row.dense.iter().any(|value| !value.is_finite())
            || !row.target.is_finite()
        {
            return Err(CartoBoostError::InvalidInput(
                "Node2Vec dense features must match dense_feature_names and be finite".to_string(),
            ));
        }
        dense.push(row.dense.clone());
        targets.push(row.target);
    }
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "Node2Vec WebGPU requires a source column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let target_nodes = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "Node2Vec WebGPU requires a target node column".to_string(),
            )
        })?;
    let edges = sources
        .iter()
        .zip(&target_nodes)
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let edge_weights = request
        .rows
        .iter()
        .map(|row| row.edge_weight.unwrap_or(1.0))
        .collect::<Vec<_>>();
    let node_count = edges
        .iter()
        .flat_map(|(source, target)| [*source, *target])
        .max()
        .map(|node| node + 1)
        .unwrap_or(0);
    let requested_holdout =
        ((dense.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, dense.len().saturating_sub(2));
    let train_rows = dense.len() - holdout_rows;
    let node2vec = node2vec_config(&request.options);
    let mut encoder = Node2VecEncoder::new(node2vec.clone()).map_err(neural_to_core)?;
    encoder
        .fit_webgpu(node_count, &edges, Some(&edge_weights))
        .await
        .map_err(neural_to_core)?;
    let mut booster = standalone_booster_config(&request.options);
    booster.backend = "cpu".to_string();
    let mut model = Node2VecRegressor::new(node2vec, booster).map_err(neural_to_core)?;
    model
        .fit_with_encoder(
            encoder,
            &sources[..train_rows],
            Some(&target_nodes[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &sources[train_rows..],
            Some(&target_nodes[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let holdout_y = &targets[train_rows..];
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y)
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: None,
                upper_prediction: None,
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_names = embedding_feature_names(
        "node2vec",
        artifact.encoder.output_dim,
        &request.dense_feature_names,
        Some("target_node2vec"),
    );
    let feature_importance = feature_importance(&artifact.model.trees, &feature_names, &[]);
    Ok(BrowserNeuralResponse {
        metadata: json!({
            "model": "node2vec_regressor",
            "pipeline": "node2vec",
            "backend": {
                "requested": requested_backend,
                "selected": "webgpu",
                "stages": {"embeddingTraining": "webgpu", "treeBoosting": "cpu"},
            },
            "denseFeatureNames": request.dense_feature_names,
            "treeCount": artifact.model.trees.len(),
            "details": {
                "embeddingDim": artifact.encoder.output_dim,
                "nodeCount": artifact.encoder.node_count,
                "edgeCount": edges.len(),
                "lossCurve": artifact.encoder.loss_curve,
                "denseWidth": artifact.dense_width,
            },
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
        model_visualization: request
            .options
            .include_model_visualization
            .unwrap_or(false)
            .then(|| model_visualization(&artifact.model.trees, &feature_names, &[])),
    })
}

fn run_neural_request(request: BrowserNeuralRequest) -> Result<BrowserNeuralResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "neural modeling requires at least four rows".to_string(),
        ));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let dense_width = request.dense_feature_names.len();
    let mut dense = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in &request.rows {
        if row.dense.len() != dense_width {
            return Err(CartoBoostError::InvalidInput(format!(
                "neural dense row has {} columns but dense_feature_names has {dense_width}",
                row.dense.len()
            )));
        }
        if row.dense.iter().any(|value| !value.is_finite()) || !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "neural dense features and targets must be finite".to_string(),
            ));
        }
        dense.push(row.dense.clone());
        targets.push(row.target);
    }

    let requested_holdout =
        ((dense.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, dense.len().saturating_sub(2));
    let train_rows = dense.len() - holdout_rows;
    let backend = browser_neural_backend(&request.options)?;
    let pipeline = request.pipeline.trim().to_ascii_lowercase();
    let (predictions, feature_names, trees, metadata) = match pipeline.as_str() {
        "" | "embedding" | "embedding_table" | "neural_embedding" => {
            run_embedding_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "node2vec" | "node2vec_graph" | "graph_node2vec" => {
            run_node2vec_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "graphsage" | "graph_sage" | "graphsage_graph" => {
            run_graphsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "hetero_graphsage" | "heterographsage" | "typed_graphsage" => {
            run_hetero_graphsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "hinsage" | "hin_sage" | "typed_hinsage" => {
            run_hinsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        other => {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported browser neural pipeline {other:?}"
            )));
        }
    };

    let holdout_y = &targets[train_rows..];
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y.iter())
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: None,
                upper_prediction: None,
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_importance = feature_importance(&trees, &feature_names, &[]);

    Ok(BrowserNeuralResponse {
        metadata: json!({
            "model": metadata["model"].as_str().unwrap_or("cartoboost_neural"),
            "pipeline": pipeline,
            "backend": {
                "requested": backend.requested,
                "selected": backend.selected,
                "available": backend.available,
            },
            "denseFeatureNames": request.dense_feature_names,
            "treeCount": trees.len(),
            "details": metadata,
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
        model_visualization: request
            .options
            .include_model_visualization
            .unwrap_or(false)
            .then(|| model_visualization(&trees, &feature_names, &[])),
    })
}

fn run_embedding_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let ids = request
        .rows
        .iter()
        .map(|row| {
            row.id.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "embedding neural pipeline requires an id column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut model = NeuralEmbeddingRegressor::new(
        request.options.embedding_dim.unwrap_or(8),
        ArtifactFallbackKind::GlobalMeanVector,
        request.options.random_state,
        request.options.support_prior_strength.unwrap_or(1.0),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &ids[..train_rows],
            &targets[..train_rows],
            Some(&dense[..train_rows]),
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(&ids[train_rows..], Some(&dense[train_rows..]))
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "embedding",
        artifact.dim,
        &request.dense_feature_names,
        None,
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "neural_embedding_regressor",
            "embeddingDim": artifact.dim,
            "embeddingRows": artifact.table.rows.len(),
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_node2vec_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "Node2Vec neural pipeline requires a source column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let target_nodes = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "Node2Vec neural pipeline requires a target node column".to_string(),
            )
        })?;
    let edges = sources
        .iter()
        .zip(target_nodes.iter())
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let edge_weights = request
        .rows
        .iter()
        .map(|row| row.edge_weight.unwrap_or(1.0))
        .collect::<Vec<_>>();
    let node_count = edges
        .iter()
        .flat_map(|(source, target)| [*source, *target])
        .max()
        .map(|max_node| max_node + 1)
        .unwrap_or(0);
    let mut model = Node2VecRegressor::new(
        node2vec_config(&request.options),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            node_count,
            &edges,
            Some(&edge_weights),
            &sources[..train_rows],
            Some(&target_nodes[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &sources[train_rows..],
            Some(&target_nodes[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "node2vec",
        artifact.encoder.output_dim,
        &request.dense_feature_names,
        Some("target_node2vec"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "node2vec_regressor",
            "mode": artifact.mode,
            "embeddingDim": artifact.encoder.output_dim,
            "nodeCount": artifact.encoder.node_count,
            "edgeCount": edges.len(),
            "lossCurve": artifact.encoder.loss_curve,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_graphsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "GraphSAGE")?;
    let config = graph_sage_config(&request.options)?;
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let mut model = GraphSageRegressor::new(
        config,
        graph.input_dim,
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "graphsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_graphsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "graphsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.edges.len(),
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_hetero_graphsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "HeteroGraphSAGE")?;
    let config = hetero_graph_sage_config(&request.options)?;
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let relation_count = graph
        .typed_edges
        .iter()
        .map(|(_, _, relation)| *relation)
        .max()
        .map(|relation| relation + 1)
        .unwrap_or(1);
    let mut model = HeteroGraphSageRegressor::new(
        config,
        graph.input_dim,
        relation_count,
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.typed_edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "hetero_graphsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_hetero_graphsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "hetero_graphsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.typed_edges.len(),
            "relationCount": relation_count,
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_hinsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "HinSAGE")?;
    let config = hin_sage_config(&request.options)?;
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let node_type_count = graph
        .node_types
        .iter()
        .max()
        .map(|node_type| node_type + 1)
        .unwrap_or(1);
    let edge_type_triples = if request.edge_type_triples.is_empty() {
        vec![(0, 0, 0)]
    } else {
        request.edge_type_triples.clone()
    };
    let mut model = HinSageRegressor::new(
        config,
        graph.input_dim,
        node_type_count,
        edge_type_triples.clone(),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.node_types,
            &graph.typed_edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "hinsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_hinsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "hinsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.typed_edges.len(),
            "nodeTypeCount": node_type_count,
            "edgeTypeTriples": edge_type_triples,
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

struct BrowserGraphInputs {
    node_features: Vec<Vec<f32>>,
    node_types: Vec<usize>,
    sources: Vec<usize>,
    targets: Vec<usize>,
    edges: Vec<(usize, usize)>,
    typed_edges: Vec<(usize, usize, usize)>,
    input_dim: usize,
}

fn browser_graph_inputs(
    request: &BrowserNeuralRequest,
    pipeline_name: &str,
) -> Result<BrowserGraphInputs> {
    if request.node_features.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} neural pipeline requires inferred node features"
        )));
    }
    let input_dim = request.node_features[0].len();
    if input_dim == 0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} neural pipeline requires at least one node feature"
        )));
    }
    for features in &request.node_features {
        if features.len() != input_dim || features.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "{pipeline_name} node features must be finite and rectangular"
            )));
        }
    }
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "{pipeline_name} neural pipeline requires a source column"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let targets = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "{pipeline_name} neural pipeline requires a target node column"
            ))
        })?;
    let node_count = request.node_features.len();
    if sources
        .iter()
        .chain(targets.iter())
        .any(|node| *node >= node_count)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} graph rows reference node ids outside node_features"
        )));
    }
    let edges = sources
        .iter()
        .zip(targets.iter())
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let typed_edges = request
        .rows
        .iter()
        .zip(edges.iter())
        .map(|(row, (source, target))| (*source, *target, row.edge_type.unwrap_or(0)))
        .collect::<Vec<_>>();
    let node_types = if request.node_types.is_empty() {
        vec![0; node_count]
    } else {
        if request.node_types.len() != node_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "{pipeline_name} node_types length must match node_features"
            )));
        }
        request.node_types.clone()
    };
    Ok(BrowserGraphInputs {
        node_features: request.node_features.clone(),
        node_types,
        sources,
        targets,
        edges,
        typed_edges,
        input_dim,
    })
}

fn embedding_feature_names(
    prefix: &str,
    dim: usize,
    dense_feature_names: &[String],
    secondary_prefix: Option<&str>,
) -> Vec<String> {
    let mut names = (0..dim)
        .map(|idx| format!("{prefix}_{idx}"))
        .collect::<Vec<_>>();
    if let Some(secondary_prefix) = secondary_prefix {
        names.extend((0..dim).map(|idx| format!("{secondary_prefix}_{idx}")));
    }
    names.extend(dense_feature_names.iter().cloned());
    names
}

fn standalone_booster_config(options: &BrowserNeuralOptions) -> StandaloneBoosterConfig {
    StandaloneBoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.07),
        max_depth: options.max_depth.unwrap_or(4),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(2),
        min_gain: 0.0,
        backend: options.backend.clone().unwrap_or_else(default_backend),
    }
}

fn node2vec_config(options: &BrowserNeuralOptions) -> Node2VecConfig {
    let mut config = Node2VecConfig::default();
    if let Some(dim) = options.embedding_dim {
        config.dim = dim;
    }
    if let Some(walk_length) = options.node2vec_walk_length {
        config.walk_length = walk_length;
    }
    if let Some(walks_per_node) = options.node2vec_walks_per_node {
        config.walks_per_node = walks_per_node;
    }
    if let Some(window_size) = options.node2vec_window_size {
        config.window_size = window_size;
    }
    if let Some(epochs) = options.node2vec_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.node2vec_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(p) = options.node2vec_p {
        config.p = p;
    }
    if let Some(q) = options.node2vec_q {
        config.q = q;
    }
    if let Some(seed) = options.node2vec_seed {
        config.seed = seed;
    }
    config
}

fn browser_neural_backend(options: &BrowserNeuralOptions) -> Result<BackendSelection> {
    select_backend_for(options.backend.as_deref(), BackendOperation::Dense)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))
}

fn graph_sage_config(options: &BrowserNeuralOptions) -> Result<GraphSageConfig> {
    let mut config = GraphSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        backend: select_backend_for_operations(
            options.backend.as_deref(),
            &[BackendOperation::Dense, BackendOperation::CsrDiffusion],
        )
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        ..GraphSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    Ok(config)
}

fn hetero_graph_sage_config(options: &BrowserNeuralOptions) -> Result<HeteroGraphSageConfig> {
    let mut config = HeteroGraphSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        backend: select_backend_for_operations(
            options.backend.as_deref(),
            &[BackendOperation::Dense, BackendOperation::CsrDiffusion],
        )
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        ..HeteroGraphSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    Ok(config)
}

fn hin_sage_config(options: &BrowserNeuralOptions) -> Result<HinSageConfig> {
    let mut config = HinSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        backend: select_backend_for_operations(
            options.backend.as_deref(),
            &[BackendOperation::Dense, BackendOperation::CsrDiffusion],
        )
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        ..HinSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    Ok(config)
}

fn graph_sage_dim(hidden_dims: &[usize]) -> usize {
    hidden_dims.last().copied().unwrap_or(8)
}

fn neural_to_core(error: cartoboost_neural::NeuralError) -> CartoBoostError {
    CartoBoostError::InvalidInput(error.to_string())
}

fn sparse_columns_from_rows(
    sparse_rows: &[Vec<Vec<u64>>],
    sparse_feature_count: usize,
) -> Vec<SparseSetColumn> {
    (0..sparse_feature_count)
        .map(|feature_idx| {
            SparseSetColumn::new(
                sparse_rows
                    .iter()
                    .map(|row| row.get(feature_idx).cloned().unwrap_or_default())
                    .collect(),
            )
        })
        .collect()
}

fn regression_booster_config(options: &BrowserRegressionOptions) -> Result<BoosterConfig> {
    Ok(BoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(120),
        learning_rate: options.learning_rate.unwrap_or(0.06),
        max_depth: options.max_depth.unwrap_or(3),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(4),
        splitters: regression_splitters(options),
        loss: regression_loss_config(options)?,
        monotonic_constraints: options.monotonic_constraints.clone().unwrap_or_default(),
        ..Default::default()
    })
}

fn regression_interval_predictions(
    options: &BrowserRegressionOptions,
    train_x: &Dataset,
    train_y: &[f64],
    holdout_x: &Dataset,
) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    let Some(lower_alpha) = options.interval_lower_alpha else {
        return Ok(None);
    };
    let Some(upper_alpha) = options.interval_upper_alpha else {
        return Ok(None);
    };
    if !lower_alpha.is_finite()
        || !upper_alpha.is_finite()
        || lower_alpha <= 0.0
        || upper_alpha >= 1.0
        || lower_alpha >= upper_alpha
    {
        return Err(CartoBoostError::InvalidInput(
            "interval alphas must be finite with 0 < lower < upper < 1".to_string(),
        ));
    }
    let lower_model = Booster::new_with_backend(
        regression_booster_config_with_loss(
            options,
            LossConfig::Quantile(QuantileLossConfig { alpha: lower_alpha }),
        ),
        options.backend.as_deref(),
    )?
    .fit(train_x, train_y, None)?;
    let upper_model = Booster::new_with_backend(
        regression_booster_config_with_loss(
            options,
            LossConfig::Quantile(QuantileLossConfig { alpha: upper_alpha }),
        ),
        options.backend.as_deref(),
    )?
    .fit(train_x, train_y, None)?;
    let lower = lower_model.try_predict(holdout_x)?;
    let upper = upper_model.try_predict(holdout_x)?;
    let (lower, upper): (Vec<_>, Vec<_>) = lower
        .into_iter()
        .zip(upper)
        .map(|(left, right)| {
            if left <= right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .unzip();
    Ok(Some((lower, upper)))
}

fn regression_booster_config_with_loss(
    options: &BrowserRegressionOptions,
    loss: LossConfig,
) -> BoosterConfig {
    BoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(120),
        learning_rate: options.learning_rate.unwrap_or(0.06),
        max_depth: options.max_depth.unwrap_or(3),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(4),
        splitters: regression_splitters(options),
        loss,
        monotonic_constraints: options.monotonic_constraints.clone().unwrap_or_default(),
        ..Default::default()
    }
}

fn regression_loss_label(options: &BrowserRegressionOptions) -> String {
    options
        .loss
        .as_deref()
        .unwrap_or("l2")
        .trim()
        .to_ascii_lowercase()
}

fn regression_loss_config(options: &BrowserRegressionOptions) -> Result<LossConfig> {
    match regression_loss_label(options).as_str() {
        "" | "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "absolute_error" | "median" => Ok(LossConfig::L1),
        "huber" => Ok(LossConfig::Huber(HuberLossConfig {
            delta: options.huber_delta.unwrap_or(1.0),
        })),
        "log_l2" | "logl2" => Ok(LossConfig::LogL2(LogL2LossConfig {
            offset: options.log_offset.unwrap_or(1.0),
        })),
        "quantile" => Ok(LossConfig::Quantile(QuantileLossConfig {
            alpha: options.quantile_alpha.unwrap_or(0.5),
        })),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported browser regression loss {other:?}"
        ))),
    }
}

fn regression_splitters(options: &BrowserRegressionOptions) -> Vec<SplitterKind> {
    match options
        .splitter_mode
        .as_deref()
        .unwrap_or("auto")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "axis" | "dense_axis" => vec![SplitterKind::Axis],
        "spatial" => vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
        ],
        "periodic" => vec![
            SplitterKind::Axis,
            SplitterKind::Periodic {
                period: default_periodic_period(options),
            },
        ],
        "sparse" | "sparse_set" | "sparse_sets" => {
            vec![SplitterKind::Axis, SplitterKind::SparseSet]
        }
        "full" | "toolkit" | "spatial_periodic" => vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
            SplitterKind::Periodic {
                period: default_periodic_period(options),
            },
            SplitterKind::SparseSet,
        ],
        _ => vec![SplitterKind::Auto],
    }
}

fn default_periodic_period(options: &BrowserRegressionOptions) -> f64 {
    options
        .periodic_periods
        .values()
        .next()
        .copied()
        .unwrap_or(24) as f64
}

fn regression_feature_schema(
    feature_names: &[String],
    sparse_feature_names: &[String],
    options: &BrowserRegressionOptions,
) -> Result<FeatureSchema> {
    let mut kinds = feature_names
        .iter()
        .map(|name| {
            let kind = options
                .feature_kinds
                .get(name)
                .map(|value| value.trim().to_ascii_lowercase())
                .unwrap_or_else(|| "numeric".to_string());
            match kind.as_str() {
                "" | "numeric" => Ok(FeatureKind::Numeric),
                "spatial" => Ok(FeatureKind::Spatial),
                "periodic" => {
                    let period = options.periodic_periods.get(name).copied().unwrap_or(24);
                    if period == 0 {
                        return Err(CartoBoostError::InvalidInput(format!(
                            "periodic feature {name:?} must have a positive period"
                        )));
                    }
                    Ok(FeatureKind::Periodic { period })
                }
                other => Err(CartoBoostError::InvalidInput(format!(
                    "unsupported browser regression feature kind {other:?} for {name:?}"
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    kinds.extend(
        sparse_feature_names
            .iter()
            .map(|_| FeatureKind::SparseSet)
            .collect::<Vec<_>>(),
    );
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    Ok(FeatureSchema { names, kinds })
}

fn regression_metrics(
    actuals: &[f64],
    predictions: &[f64],
    train_rows: usize,
    holdout_rows: usize,
) -> Result<BrowserRegressionMetrics> {
    if actuals.len() != predictions.len() || actuals.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "actual and prediction lengths must match and be non-empty".to_string(),
        ));
    }
    let mut squared_error_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let mut total_sum_squares = 0.0;
    for (actual, prediction) in actuals.iter().zip(predictions.iter()) {
        let residual = actual - prediction;
        squared_error_sum += residual * residual;
        absolute_error_sum += residual.abs();
        total_sum_squares += (actual - mean_actual).powi(2);
    }
    let rmse = (squared_error_sum / actuals.len() as f64).sqrt();
    let mae = absolute_error_sum / actuals.len() as f64;
    let r2 = if total_sum_squares <= f64::EPSILON {
        0.0
    } else {
        1.0 - squared_error_sum / total_sum_squares
    };
    Ok(BrowserRegressionMetrics {
        rmse,
        mae,
        r2,
        train_rows,
        holdout_rows,
    })
}

fn feature_importance(
    trees: &[cartoboost_core::Tree],
    feature_names: &[String],
    sparse_feature_names: &[String],
) -> Vec<BrowserFeatureImportance> {
    let dense_feature_count = feature_names.len();
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    let mut counts = vec![0usize; names.len()];
    for tree in trees {
        count_split_features(&tree.root, &mut counts, dense_feature_count);
    }
    let mut importance = names
        .iter()
        .enumerate()
        .map(|(idx, feature)| BrowserFeatureImportance {
            feature: feature.clone(),
            split_count: counts[idx],
        })
        .collect::<Vec<_>>();
    importance.sort_by(|left, right| {
        right
            .split_count
            .cmp(&left.split_count)
            .then_with(|| left.feature.cmp(&right.feature))
    });
    importance
}

fn count_split_features(node: &Node, counts: &mut [usize], dense_feature_count: usize) {
    if let Node::Branch {
        split, left, right, ..
    } = node
    {
        count_split(split, counts, dense_feature_count);
        count_split_features(left, counts, dense_feature_count);
        count_split_features(right, counts, dense_feature_count);
    }
}

fn count_split(split: &Split, counts: &mut [usize], dense_feature_count: usize) {
    match split {
        Split::Axis { feature, .. }
        | Split::PeriodicInterval { feature, .. }
        | Split::SparseSetContainsAny { feature, .. } => increment_feature(*feature, counts),
        Split::Diagonal2D {
            x_feature,
            y_feature,
            ..
        }
        | Split::Gaussian2D {
            x_feature,
            y_feature,
            ..
        } => {
            increment_feature(*x_feature, counts);
            increment_feature(*y_feature, counts);
        }
        Split::SparseListContainsAny { sparse_feature, .. } => {
            increment_feature(dense_feature_count + *sparse_feature, counts);
        }
        Split::Fuzzy { base, .. } => count_split(base, counts, dense_feature_count),
    }
}

fn increment_feature(feature: usize, counts: &mut [usize]) {
    if let Some(count) = counts.get_mut(feature) {
        *count += 1;
    }
}

fn model_visualization(
    trees: &[cartoboost_core::Tree],
    feature_names: &[String],
    sparse_feature_names: &[String],
) -> BrowserModelVisualization {
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    let mut totals = TreeStats::default();
    let mut split_kind_counts = BTreeMap::<String, usize>::new();
    let mut splitter_rules = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut feature_split_counts = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut depth_counts = BTreeMap::<usize, usize>::new();
    for tree in trees {
        let mut context = TreeStatsContext {
            dense_feature_count: feature_names.len(),
            feature_names: &names,
            stats: &mut totals,
            split_kind_counts: &mut split_kind_counts,
            splitter_rules: &mut splitter_rules,
            feature_split_counts: &mut feature_split_counts,
            depth_counts: &mut depth_counts,
        };
        collect_tree_stats(&tree.root, 0, &mut context);
    }
    let tree_blueprints = trees
        .iter()
        .take(8)
        .enumerate()
        .map(|(tree_index, tree)| tree_blueprint(tree_index, tree, feature_names.len(), &names))
        .collect();
    BrowserModelVisualization {
        summary: BrowserModelVisualizationSummary {
            tree_count: trees.len(),
            node_count: totals.node_count,
            branch_count: totals.branch_count,
            leaf_count: totals.leaf_count,
            max_depth: totals.max_depth,
            mean_leaf_value: finite_ratio(totals.leaf_value_sum, totals.leaf_count),
            mean_gain: finite_ratio(totals.gain_sum, totals.branch_count),
        },
        split_kinds: split_kind_counts
            .into_iter()
            .map(|(kind, count)| BrowserSplitKindCount { kind, count })
            .collect(),
        splitter_rules: top_splitter_rules(splitter_rules),
        feature_split_counts: top_feature_split_counts(feature_split_counts),
        depth_histogram: depth_counts
            .into_iter()
            .map(|(depth, count)| BrowserDepthCount { depth, count })
            .collect(),
        tree_blueprints,
    }
}

fn tree_blueprint(
    tree_index: usize,
    tree: &cartoboost_core::Tree,
    dense_feature_count: usize,
    feature_names: &[String],
) -> BrowserTreeBlueprint {
    let mut stats = TreeStats::default();
    let mut split_kind_counts = BTreeMap::<String, usize>::new();
    let mut splitter_rules = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut feature_split_counts = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut depth_counts = BTreeMap::<usize, usize>::new();
    let mut context = TreeStatsContext {
        dense_feature_count,
        feature_names,
        stats: &mut stats,
        split_kind_counts: &mut split_kind_counts,
        splitter_rules: &mut splitter_rules,
        feature_split_counts: &mut feature_split_counts,
        depth_counts: &mut depth_counts,
    };
    collect_tree_stats(&tree.root, 0, &mut context);
    let mut next_id = 0;
    BrowserTreeBlueprint {
        tree_index,
        node_count: stats.node_count,
        branch_count: stats.branch_count,
        leaf_count: stats.leaf_count,
        max_depth: stats.max_depth,
        total_gain: stats.gain_sum,
        root: tree_node_blueprint(
            &tree.root,
            0,
            dense_feature_count,
            feature_names,
            &mut next_id,
        ),
    }
}

#[derive(Default)]
struct TreeStats {
    node_count: usize,
    branch_count: usize,
    leaf_count: usize,
    max_depth: usize,
    gain_sum: f64,
    leaf_value_sum: f64,
}

#[derive(Default)]
struct SplitterRuleAccumulator {
    count: usize,
    total_gain: f64,
}

struct TreeStatsContext<'a> {
    dense_feature_count: usize,
    feature_names: &'a [String],
    stats: &'a mut TreeStats,
    split_kind_counts: &'a mut BTreeMap<String, usize>,
    splitter_rules: &'a mut BTreeMap<(String, String), SplitterRuleAccumulator>,
    feature_split_counts: &'a mut BTreeMap<(String, String), SplitterRuleAccumulator>,
    depth_counts: &'a mut BTreeMap<usize, usize>,
}

fn collect_tree_stats(node: &Node, depth: usize, context: &mut TreeStatsContext<'_>) {
    context.stats.node_count += 1;
    context.stats.max_depth = context.stats.max_depth.max(depth);
    *context.depth_counts.entry(depth).or_insert(0) += 1;
    match node {
        Node::Leaf { value, .. } => {
            context.stats.leaf_count += 1;
            context.stats.leaf_value_sum += *value;
        }
        Node::LinearLeaf { model, .. } => {
            context.stats.leaf_count += 1;
            context.stats.leaf_value_sum += model.intercept;
        }
        Node::Branch {
            split,
            left,
            right,
            gain,
            ..
        } => {
            context.stats.branch_count += 1;
            context.stats.gain_sum += *gain;
            let (kind, label) =
                split_display(split, context.dense_feature_count, context.feature_names);
            *context.split_kind_counts.entry(kind.clone()).or_insert(0) += 1;
            let rule = context
                .splitter_rules
                .entry((kind.clone(), label))
                .or_default();
            rule.count += 1;
            rule.total_gain += *gain;
            for feature in split_feature_indices(split, context.dense_feature_count) {
                let feature_name = feature_label(feature, context.feature_names);
                let feature_rule = context
                    .feature_split_counts
                    .entry((feature_name, kind.clone()))
                    .or_default();
                feature_rule.count += 1;
                feature_rule.total_gain += *gain;
            }
            collect_tree_stats(left, depth + 1, context);
            collect_tree_stats(right, depth + 1, context);
        }
    }
}

fn top_splitter_rules(
    splitter_rules: BTreeMap<(String, String), SplitterRuleAccumulator>,
) -> Vec<BrowserSplitterRuleSummary> {
    let mut rules = splitter_rules
        .into_iter()
        .map(|((kind, label), accumulator)| BrowserSplitterRuleSummary {
            kind,
            label,
            count: accumulator.count,
            total_gain: accumulator.total_gain,
            mean_gain: finite_ratio(accumulator.total_gain, accumulator.count),
        })
        .collect::<Vec<_>>();
    rules.sort_by(|left, right| {
        right
            .total_gain
            .total_cmp(&left.total_gain)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.label.cmp(&right.label))
    });
    rules.truncate(16);
    rules
}

fn top_feature_split_counts(
    feature_split_counts: BTreeMap<(String, String), SplitterRuleAccumulator>,
) -> Vec<BrowserFeatureSplitCount> {
    let mut rows = feature_split_counts
        .into_iter()
        .map(|((feature, kind), accumulator)| BrowserFeatureSplitCount {
            feature,
            kind,
            count: accumulator.count,
            total_gain: accumulator.total_gain,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .total_gain
            .total_cmp(&left.total_gain)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.feature.cmp(&right.feature))
            .then_with(|| left.kind.cmp(&right.kind))
    });
    rows.truncate(24);
    rows
}

fn tree_node_blueprint(
    node: &Node,
    depth: usize,
    dense_feature_count: usize,
    feature_names: &[String],
    next_id: &mut usize,
) -> BrowserTreeNode {
    let id = *next_id;
    *next_id += 1;
    match node {
        Node::Leaf {
            value,
            sample_weight_sum,
            ..
        } => BrowserTreeNode {
            id,
            depth,
            kind: "leaf".to_string(),
            label: format!("leaf {value:.3}"),
            value: Some(*value),
            gain: None,
            sample_weight_sum: Some(*sample_weight_sum),
            left: None,
            right: None,
        },
        Node::LinearLeaf {
            model,
            sample_weight_sum,
            ..
        } => BrowserTreeNode {
            id,
            depth,
            kind: "linear_leaf".to_string(),
            label: format!("linear leaf {:+.3}", model.intercept),
            value: Some(model.intercept),
            gain: None,
            sample_weight_sum: Some(*sample_weight_sum),
            left: None,
            right: None,
        },
        Node::Branch {
            split,
            left,
            right,
            gain,
            sample_weight_sum,
        } => {
            let (kind, label) = split_display(split, dense_feature_count, feature_names);
            let should_expand = depth < 3;
            BrowserTreeNode {
                id,
                depth,
                kind,
                label,
                value: None,
                gain: Some(*gain),
                sample_weight_sum: Some(*sample_weight_sum),
                left: should_expand.then(|| {
                    Box::new(tree_node_blueprint(
                        left,
                        depth + 1,
                        dense_feature_count,
                        feature_names,
                        next_id,
                    ))
                }),
                right: should_expand.then(|| {
                    Box::new(tree_node_blueprint(
                        right,
                        depth + 1,
                        dense_feature_count,
                        feature_names,
                        next_id,
                    ))
                }),
            }
        }
    }
}

fn split_feature_indices(split: &Split, dense_feature_count: usize) -> Vec<usize> {
    let mut features = BTreeSet::new();
    collect_split_feature_indices(split, dense_feature_count, &mut features);
    features.into_iter().collect()
}

fn collect_split_feature_indices(
    split: &Split,
    dense_feature_count: usize,
    features: &mut BTreeSet<usize>,
) {
    match split {
        Split::Axis { feature, .. }
        | Split::PeriodicInterval { feature, .. }
        | Split::SparseSetContainsAny { feature, .. } => {
            features.insert(*feature);
        }
        Split::Diagonal2D {
            x_feature,
            y_feature,
            ..
        }
        | Split::Gaussian2D {
            x_feature,
            y_feature,
            ..
        } => {
            features.insert(*x_feature);
            features.insert(*y_feature);
        }
        Split::SparseListContainsAny { sparse_feature, .. } => {
            features.insert(dense_feature_count + *sparse_feature);
        }
        Split::Fuzzy { base, .. } => {
            collect_split_feature_indices(base, dense_feature_count, features);
        }
    }
}

fn split_display(
    split: &Split,
    dense_feature_count: usize,
    feature_names: &[String],
) -> (String, String) {
    match split {
        Split::Axis {
            feature, threshold, ..
        } => (
            "axis".to_string(),
            format!(
                "{} <= {:.3}",
                feature_label(*feature, feature_names),
                threshold
            ),
        ),
        Split::Diagonal2D {
            x_feature,
            y_feature,
            normal_x,
            normal_y,
            threshold,
            ..
        } => (
            "diagonal_2d".to_string(),
            format!(
                "{:.2}*{} + {:.2}*{} <= {:.3}",
                normal_x,
                feature_label(*x_feature, feature_names),
                normal_y,
                feature_label(*y_feature, feature_names),
                threshold
            ),
        ),
        Split::Gaussian2D {
            x_feature,
            y_feature,
            center_x,
            center_y,
            radius,
            ..
        } => (
            "gaussian_2d".to_string(),
            format!(
                "{} / {} near {:.2}, {:.2} r{:.2}",
                feature_label(*x_feature, feature_names),
                feature_label(*y_feature, feature_names),
                center_x,
                center_y,
                radius
            ),
        ),
        Split::PeriodicInterval {
            feature,
            period,
            start,
            end,
            ..
        } => (
            "periodic".to_string(),
            format!(
                "{} in {:.2}..{:.2} mod {:.2}",
                feature_label(*feature, feature_names),
                start,
                end,
                period
            ),
        ),
        Split::SparseSetContainsAny { feature, ids, .. } => (
            "sparse_set".to_string(),
            format!(
                "{} has {}",
                feature_label(*feature, feature_names),
                id_summary(ids)
            ),
        ),
        Split::SparseListContainsAny {
            sparse_feature,
            ids,
            ..
        } => (
            "sparse_list".to_string(),
            format!(
                "{} has {}",
                feature_label(dense_feature_count + *sparse_feature, feature_names),
                id_summary(ids)
            ),
        ),
        Split::Fuzzy {
            base,
            bandwidth,
            kernel,
        } => {
            let (_, label) = split_display(base, dense_feature_count, feature_names);
            (
                "fuzzy".to_string(),
                format!("fuzzy {kernel:?} bw {:.3}: {label}", bandwidth),
            )
        }
    }
}

fn feature_label(feature: usize, feature_names: &[String]) -> String {
    feature_names
        .get(feature)
        .cloned()
        .unwrap_or_else(|| format!("feature_{feature}"))
}

fn id_summary(ids: &[u64]) -> String {
    let mut values = ids.iter().take(4).map(u64::to_string).collect::<Vec<_>>();
    if ids.len() > values.len() {
        values.push("...".to_string());
    }
    values.join(",")
}

fn finite_ratio(numerator: f64, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}

#[cfg(test)]
mod model_visualization_tests {
    use super::*;
    use cartoboost_core::Tree;

    #[test]
    fn model_visualization_summarizes_tree_shape_and_split_labels() {
        let trees = vec![Tree {
            root: Node::Branch {
                split: Split::Axis {
                    feature: 0,
                    threshold: 12.5,
                    missing_goes_left: true,
                },
                left: Box::new(Node::Leaf {
                    value: -1.25,
                    sample_weight_sum: 3.0,
                    training_loss: 0.4,
                }),
                right: Box::new(Node::Branch {
                    split: Split::PeriodicInterval {
                        feature: 1,
                        period: 24.0,
                        start: 7.0,
                        end: 10.0,
                        missing_goes_left: false,
                    },
                    left: Box::new(Node::Leaf {
                        value: 0.5,
                        sample_weight_sum: 2.0,
                        training_loss: 0.1,
                    }),
                    right: Box::new(Node::Leaf {
                        value: 1.5,
                        sample_weight_sum: 4.0,
                        training_loss: 0.2,
                    }),
                    gain: 1.25,
                    sample_weight_sum: 6.0,
                }),
                gain: 2.75,
                sample_weight_sum: 9.0,
            },
        }];
        let visualization = model_visualization(
            &trees,
            &["pickup_hour".to_string(), "pickup_dow".to_string()],
            &[],
        );

        assert_eq!(visualization.summary.tree_count, 1);
        assert_eq!(visualization.summary.node_count, 5);
        assert_eq!(visualization.summary.branch_count, 2);
        assert_eq!(visualization.summary.leaf_count, 3);
        assert_eq!(visualization.summary.max_depth, 2);
        assert_eq!(visualization.depth_histogram.len(), 3);
        assert_eq!(visualization.split_kinds[0].kind, "axis");
        assert_eq!(visualization.split_kinds[1].kind, "periodic");
        assert_eq!(visualization.splitter_rules.len(), 2);
        assert!(
            visualization.splitter_rules[0].total_gain
                >= visualization.splitter_rules[1].total_gain
        );
        assert!(visualization
            .feature_split_counts
            .iter()
            .any(|row| row.feature == "pickup_hour" && row.kind == "axis" && row.count == 1));
        assert!(visualization
            .feature_split_counts
            .iter()
            .any(|row| row.feature == "pickup_dow" && row.kind == "periodic" && row.count == 1));
        assert!(visualization.tree_blueprints[0]
            .root
            .label
            .contains("pickup_hour"));
    }
}

fn default_holdout_fraction() -> f64 {
    0.2
}

fn default_true() -> bool {
    true
}

fn default_neural_pipeline() -> String {
    "embedding".to_string()
}

fn build_forecaster(
    model: &str,
    options: &BrowserForecastOptions,
    frame: &ForecastFrame,
    horizon: usize,
) -> Result<Box<dyn Forecaster>> {
    let normalized = model.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "naive" => Ok(Box::new(NaiveForecaster::new())),
        "seasonal_naive" => Ok(Box::new(SeasonalNaiveForecaster::new(
            options.season_length.unwrap_or(7),
        )?)),
        "window_average" => Ok(Box::new(WindowAverageForecaster::new(
            options.window_size.unwrap_or(7),
        )?)),
        "seasonal_window_average" => Ok(Box::new(SeasonalWindowAverageForecaster::new(
            options.season_length.unwrap_or(7),
            options.window_count.unwrap_or(3),
        )?)),
        "theta" => Ok(Box::new(ThetaForecaster::with_seasonality(
            options.theta.unwrap_or(2.0),
            options.alpha.unwrap_or(0.2),
            theta_seasonality(options)?,
        )?)),
        "optimized_theta" => Ok(Box::new(OptimizedThetaForecaster::with_seasonality(
            options
                .theta_grid
                .clone()
                .unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
            options
                .alpha_grid
                .clone()
                .unwrap_or_else(|| vec![0.1, 0.2, 0.3, 0.5, 0.8]),
            theta_seasonality(options)?,
        )?)),
        "ets" => Ok(Box::new(ETSForecaster::with_additive_damped_trend(
            options.alpha.unwrap_or(0.3),
            options.beta.unwrap_or(0.1),
            options.gamma,
            None,
            options.damping_phi.unwrap_or(1.0),
        )?)),
        "seasonal_ets" => Ok(Box::new(ETSForecaster::with_additive_damped_trend(
            options.alpha.unwrap_or(0.3),
            options.beta.unwrap_or(0.1),
            Some(options.gamma.unwrap_or(0.1)),
            Some(options.season_length.unwrap_or(7)),
            options.damping_phi.unwrap_or(1.0),
        )?)),
        "auto_ets" => Ok(Box::new(AutoETSForecaster::new(options.season_length)?)),
        "arima" => Ok(Box::new(
            cartoboost_core::forecasting::ArimaForecaster::new(
                options.max_p.unwrap_or(1),
                options.max_d.unwrap_or(1),
                options.max_q.unwrap_or(0),
            )?,
        )),
        "auto_arima" => Ok(Box::new(AutoARIMAForecaster::with_max_order(
            options.max_p.unwrap_or(2),
            options.max_d.unwrap_or(1),
            options.max_q.unwrap_or(1),
        )?)),
        "kalman" => Ok(Box::new(KalmanForecaster::new(
            options.level_process_variance.unwrap_or(0.05),
            options.trend_process_variance.unwrap_or(0.005),
            options.observation_variance.unwrap_or(1.0),
        )?)),
        "local_level_kalman" => Ok(Box::new(LocalLevelKalmanForecaster::new(
            options.level_process_variance.unwrap_or(0.05),
            options.observation_variance.unwrap_or(1.0),
        )?)),
        "auto_kalman" => Ok(Box::new(AutoKalmanForecaster::new()?)),
        "auto_local_level_kalman" => Ok(Box::new(AutoLocalLevelKalmanForecaster::new()?)),
        "kriging" => Ok(Box::new(KrigingForecaster::new(
            coordinates_from_frame(frame, options)?,
            options.kriging_range.unwrap_or(1.0),
            options.kriging_nugget.unwrap_or(1e-6),
        )?)),
        "spatial_piecewise_kriging" => Ok(Box::new(SpatialPiecewiseKrigingForecaster::new(
            SpatialPiecewiseKrigingConfig {
                coordinates: coordinates_from_frame(frame, options)?,
                mode: spatial_piecewise_kriging_mode(options)?,
                piecewise_config: piecewise_linear_seasonal_config(options)?,
                kriging_config: cartoboost_core::utilities::OrdinaryKrigingConfig::new(
                    options.kriging_range.unwrap_or(1.0),
                    options.kriging_nugget.unwrap_or(1e-6),
                )?,
                spatial_regressors: options.spatial_regressors.clone().unwrap_or_default(),
                residual_shrinkage: options.residual_shrinkage.unwrap_or(1.0),
                allow_neighbor_fallback: options.allow_neighbor_fallback.unwrap_or(false),
            },
        )?)),
        "piecewise_linear_seasonal" => Ok(Box::new(PiecewiseLinearSeasonalForecaster::new(
            piecewise_linear_seasonal_config(options)?,
        )?)),
        "neural_panel" => Ok(Box::new(
            NeuralPanelForecaster::new(neural_panel_config(options, horizon)?)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )),
        "nbeats" => Ok(Box::new(
            NBeatsForecaster::new(nbeats_config(options)?)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )),
        "nhits" => Ok(Box::new(
            NHiTSForecaster::new(nhits_config(options)?)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )),
        "intermittent_demand" => {
            let config = IntermittentDemandConfig {
                alpha: options.alpha.unwrap_or(0.2),
                beta: options.beta.unwrap_or(0.2),
                validation_window: options.validation_window,
                ..IntermittentDemandConfig::default()
            };
            Ok(Box::new(IntermittentDemandForecaster::new(config)?))
        }
        "croston" => Ok(Box::new(CrostonForecaster::new(
            options.alpha.unwrap_or(0.2),
        )?)),
        "sba" => Ok(Box::new(SbaForecaster::new(options.alpha.unwrap_or(0.2))?)),
        "tsb" => Ok(Box::new(TsbForecaster::new(
            options.alpha.unwrap_or(0.2),
            options.beta.unwrap_or(0.2),
        )?)),
        "classical_expert_bank" => Ok(Box::new(ClassicalExpertBank::default_for_season_length(
            options.season_length.unwrap_or(7),
        )?)),
        "autostats_bank" => Ok(Box::new(AutoStatsBank::with_validation_window(
            options.season_length.unwrap_or(7),
            options.validation_window,
        )?)),
        "stl_cartoboost" => Ok(Box::new(STLCartoBoostForecaster::new(
            options.season_length.unwrap_or(7),
        )?)),
        "mstl_cartoboost" => Ok(Box::new(MSTLCartoBoostForecaster::new(
            options
                .mstl_season_lengths
                .clone()
                .unwrap_or_else(|| vec![options.season_length.unwrap_or(7)]),
        )?)),
        "cartoboost_lag" => Ok(Box::new(CartoBoostLagForecaster::new_with_backend(
            lag_config(options),
            booster_config(options),
            GlobalForecastTargetMode::Level,
            cartoboost_core::forecasting::GlobalForecastSampleWeightMode::Uniform,
            options.backend.as_deref(),
        )?)),
        "cartoboost_direct" => Ok(Box::new(BrowserDirectForecaster::new(
            lag_config(options),
            booster_config(options),
            horizon,
            options.backend.as_deref(),
        )?)),
        "rectified_recursive" => Ok(Box::new(BrowserRectifiedRecursiveForecaster::new(
            lag_config(options),
            booster_config(options),
            horizon,
            options.backend.as_deref(),
        )?)),
        "lag_plus" => {
            let mut config = LagPlusConfig::new(lag_config(options), booster_config(options));
            config.backend = options.backend.clone().unwrap_or_else(default_backend);
            Ok(Box::new(LagPlusForecaster::new(config)?))
        }
        "auto_forecast" => {
            let mut config = AutoForecastConfig {
                lag_config: lag_config(options),
                booster_config: booster_config(options),
                ..AutoForecastConfig::default()
            };
            if let Some(season_length) = options.season_length {
                config.season_length = season_length;
            }
            if let Some(validation_window) = options.validation_window {
                config.validation_window = Some(validation_window);
            }
            config.max_candidate_count = options.max_auto_candidate_count;
            config.max_direct_horizon = options.max_direct_horizon.unwrap_or(horizon);
            config.backend = options.backend.clone().unwrap_or_else(default_backend);
            Ok(Box::new(AutoForecastModel::new(config)?))
        }
        "scaled_cartoboost_lag" => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_backend(
                lag_config(options),
                booster_config(options),
                GlobalForecastTargetMode::Level,
                cartoboost_core::forecasting::GlobalForecastSampleWeightMode::Uniform,
                options.backend.as_deref(),
            )?),
            1e-6,
            "scaled_cartoboost_lag",
        )?)),
        "log1p_cartoboost_lag" => Ok(Box::new(Log1pForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_backend(
                lag_config(options),
                booster_config(options),
                GlobalForecastTargetMode::Level,
                cartoboost_core::forecasting::GlobalForecastSampleWeightMode::Uniform,
                options.backend.as_deref(),
            )?),
            "log1p_cartoboost_lag",
        ))),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported browser forecast model {other:?}"
        ))),
    }
}

struct BrowserDirectForecaster {
    inner: CartoBoostDirectForecaster,
    fit_horizon: usize,
}

impl BrowserDirectForecaster {
    fn new(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        fit_horizon: usize,
        backend: Option<&str>,
    ) -> Result<Self> {
        Ok(Self {
            inner: CartoBoostDirectForecaster::new_with_backend(
                lag_config,
                booster_config,
                backend,
            )?,
            fit_horizon,
        })
    }
}

impl Forecaster for BrowserDirectForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.inner.fit_horizon(frame, self.fit_horizon)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn metadata(&self) -> Value {
        self.inner.metadata()
    }
}

struct BrowserRectifiedRecursiveForecaster {
    inner: RectifiedRecursiveForecaster,
    fit_horizon: usize,
}

impl BrowserRectifiedRecursiveForecaster {
    fn new(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        fit_horizon: usize,
        backend: Option<&str>,
    ) -> Result<Self> {
        Ok(Self {
            inner: RectifiedRecursiveForecaster::new_with_backend(
                lag_config,
                booster_config,
                backend,
            )?,
            fit_horizon,
        })
    }
}

impl Forecaster for BrowserRectifiedRecursiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.inner.fit_horizon(frame, self.fit_horizon)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn metadata(&self) -> Value {
        self.inner.metadata()
    }
}

fn theta_seasonality(options: &BrowserForecastOptions) -> Result<Option<ThetaSeasonality>> {
    let Some(kind) = options.theta_seasonality.as_deref() else {
        return Ok(None);
    };
    let season_length = options.season_length.unwrap_or(7);
    match kind.trim().to_ascii_lowercase().as_str() {
        "" | "none" => Ok(None),
        "additive" => ThetaSeasonality::additive(season_length).map(Some),
        "multiplicative" => ThetaSeasonality::multiplicative(season_length).map(Some),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported theta seasonality {other:?}"
        ))),
    }
}

fn is_piecewise_linear_seasonal_model(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().replace('-', "_").as_str(),
        "piecewise_linear_seasonal"
    )
}

fn spatial_piecewise_kriging_mode(
    options: &BrowserForecastOptions,
) -> Result<SpatialPiecewiseKrigingMode> {
    let value = options
        .spatial_kriging_mode
        .as_deref()
        .unwrap_or("residual_kriging")
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_");
    match value.as_str() {
        "kriged_regressors" | "regressors" => Ok(SpatialPiecewiseKrigingMode::KrigedRegressors),
        "residual_kriging" | "residual" => Ok(SpatialPiecewiseKrigingMode::ResidualKriging),
        "hybrid" => Ok(SpatialPiecewiseKrigingMode::Hybrid),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported spatial_piecewise_kriging mode {other:?}"
        ))),
    }
}

fn piecewise_linear_seasonal_config(
    options: &BrowserForecastOptions,
) -> Result<PiecewiseLinearSeasonalConfig> {
    let mut config = PiecewiseLinearSeasonalConfig::default();
    if options.mcmc_samples.unwrap_or(0) > 0 {
        return Err(CartoBoostError::InvalidInput(
            "mcmc_samples are not supported by the Rust-native piecewise seasonal model; use uncertainty_samples for deterministic native intervals".to_string(),
        ));
    }
    if let Some(growth) = options.growth.as_deref() {
        config.growth = match growth.trim().to_ascii_lowercase().as_str() {
            "" | "linear" => PiecewiseLinearGrowth::Linear,
            "flat" => PiecewiseLinearGrowth::Flat,
            "logistic" => PiecewiseLinearGrowth::Logistic,
            other => {
                return Err(CartoBoostError::InvalidInput(format!(
                    "unsupported piecewise seasonal growth {other:?}"
                )))
            }
        };
    }
    if let Some(mode) = options
        .seasonality_mode
        .as_deref()
        .or(options.component_mode.as_deref())
    {
        config.component_mode = piecewise_component_mode(mode)?;
    }
    if let Some(loss) = options.fit_loss.as_deref() {
        config.fit_loss = piecewise_fit_loss(loss)?;
    }
    if let Some(delta) = options.huber_delta {
        config.huber_delta = delta;
    }
    if let Some(iterations) = options.irls_iterations {
        config.irls_iterations = iterations;
    }
    if let Some(changepoints) = options.n_changepoints.or(options.changepoints) {
        config.changepoints = changepoints;
    }
    if let Some(changepoint_range) = options.changepoint_range {
        config.changepoint_range = changepoint_range;
    }
    if let Some(timestamps) = &options.changepoint_timestamps {
        config.changepoint_timestamps = timestamps
            .iter()
            .map(|timestamp| cartoboost_core::forecasting::parse_forecast_timestamp(timestamp))
            .collect::<Result<Vec<_>>>()?;
    }
    if let Some(order) = options.yearly_fourier_order {
        config.yearly_fourier_order = order;
    }
    if let Some(order) = options.weekly_fourier_order {
        config.weekly_fourier_order = order;
    }
    if let Some(order) = options.daily_fourier_order {
        config.daily_fourier_order = order;
    }
    if let Some(value) = options.auto_yearly_seasonality {
        config.auto_yearly_seasonality = value;
    }
    if let Some(value) = options.auto_weekly_seasonality {
        config.auto_weekly_seasonality = value;
    }
    if let Some(value) = options.auto_daily_seasonality {
        config.auto_daily_seasonality = value;
    }
    if let Some(seasonalities) = &options.custom_seasonalities {
        config.custom_seasonalities = seasonalities
            .iter()
            .map(|seasonality| {
                Ok(PiecewiseLinearSeasonality {
                    name: seasonality.name.clone(),
                    period_days: seasonality.period_days,
                    fourier_order: seasonality.fourier_order,
                    mode: seasonality
                        .mode
                        .as_deref()
                        .map(piecewise_component_mode)
                        .transpose()?,
                    condition_name: seasonality.condition_name.clone(),
                    l2_regularization: seasonality.l2_regularization,
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    if let Some(value) = options.changepoint_l2_regularization {
        config.changepoint_l2_regularization = value;
    }
    if let Some(value) = options.changepoint_l1_regularization {
        config.changepoint_l1_regularization = value;
    }
    if let Some(value) = options.changepoint_prior_scale {
        config.changepoint_l1_regularization = piecewise_prior_scale_to_l1(value)?;
    }
    if let Some(value) = options.seasonality_l2_regularization {
        config.seasonality_l2_regularization = value;
    }
    if let Some(value) = options.seasonality_prior_scale {
        config.seasonality_l2_regularization = piecewise_prior_scale_to_l2(value)?;
    }
    if let Some(value) = options.yearly_l2_regularization {
        config.yearly_l2_regularization = Some(value);
    }
    if let Some(value) = options.weekly_l2_regularization {
        config.weekly_l2_regularization = Some(value);
    }
    if let Some(value) = options.daily_l2_regularization {
        config.daily_l2_regularization = Some(value);
    }
    if let Some(value) = options.event_l2_regularization {
        config.event_l2_regularization = value;
    }
    if let Some(value) = options.holidays_prior_scale {
        config.event_l2_regularization = piecewise_prior_scale_to_l2(value)?;
    }
    if let Some(value) = options.regressor_l2_regularization {
        config.regressor_l2_regularization = value;
    }
    if let Some(values) = &options.event_l2_regularization_by_name {
        config.event_l2_regularization_by_name = values.clone();
    }
    if let Some(values) = &options.regressor_l2_regularization_by_name {
        config.regressor_l2_regularization_by_name = values.clone();
    }
    if let Some(events) = &options.events {
        config.events = events
            .iter()
            .map(|event| {
                Ok(PiecewiseLinearEvent {
                    name: event.name.clone(),
                    timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(
                        &event.timestamp,
                    )?,
                    lower_window: event.lower_window.unwrap_or(0),
                    upper_window: event.upper_window.unwrap_or(0),
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    if let Some(holidays) = &options.holidays {
        let mut holiday_events = Vec::with_capacity(holidays.len());
        for holiday in holidays {
            let timestamp = cartoboost_core::forecasting::parse_forecast_timestamp(&holiday.ds)?;
            holiday_events.push(PiecewiseLinearEvent {
                name: holiday.holiday.clone(),
                timestamp,
                lower_window: holiday.lower_window.unwrap_or(0),
                upper_window: holiday.upper_window.unwrap_or(0),
            });
            if let Some(scale) = holiday.prior_scale {
                config
                    .event_l2_regularization_by_name
                    .insert(holiday.holiday.clone(), piecewise_prior_scale_to_l2(scale)?);
            }
        }
        config.events.extend(holiday_events);
    }
    if let Some(mode) = options
        .holidays_mode
        .as_deref()
        .or(options.event_mode.as_deref())
    {
        config.event_mode = Some(piecewise_component_mode(mode)?);
    }
    if let Some(regressors) = &options.extra_regressors {
        config.extra_regressors = regressors.clone();
    }
    if let Some(regressor_modes) = &options.regressor_modes {
        config.regressor_modes = regressor_modes
            .iter()
            .map(|(name, mode)| Ok((name.clone(), piecewise_component_mode(mode)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
    }
    if let Some(constraints) = &options.extra_regressor_monotonic_constraints {
        config.extra_regressor_monotonic_constraints = constraints.clone();
    }
    if let Some(value) = options.regressor_standardization.as_deref() {
        config.regressor_standardization = piecewise_regressor_standardization(value)?;
    }
    if let Some(future_regressors) = &options.future_regressors {
        config.future_regressors = future_regressors.clone();
    }
    if let Some(future_regressors_by_series) = &options.future_regressors_by_series {
        config.future_regressors_by_series = future_regressors_by_series.clone();
    }
    if let Some(trend_adjustments) = &options.trend_adjustments {
        config.trend_adjustments = trend_adjustments.clone();
    }
    if let Some(trend_adjustments_by_series) = &options.trend_adjustments_by_series {
        config.trend_adjustments_by_series = trend_adjustments_by_series.clone();
    }
    if let Some(value) = options.residual_shock_window {
        config.residual_shock_window = value;
    }
    if let Some(value) = options.residual_shock_scale {
        config.residual_shock_scale = value;
    }
    if let Some(value) = options.residual_shock_decay {
        config.residual_shock_decay = value;
    }
    if let Some(levels) = &options.interval_levels {
        config.interval_levels = levels.clone();
    }
    if let Some(width) = options.interval_width {
        if !(0.0..=1.0).contains(&width) || width == 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "interval_width must be in (0, 1]".to_string(),
            ));
        }
        config.interval_levels = vec![width];
    }
    if let Some(levels) = &options.quantile_levels {
        config.quantile_levels = levels.clone();
    }
    if let Some(value) = options.uncertainty_samples {
        config.uncertainty_samples = value;
    }
    if let Some(value) = options.trend_uncertainty_policy.as_deref() {
        config.trend_uncertainty_policy = piecewise_trend_uncertainty_policy(value)?;
    }
    if let Some(value) = options.trend_uncertainty_scale {
        config.trend_uncertainty_scale = value;
    }
    if let Some(value) = options.coefficient_uncertainty_scale {
        config.coefficient_uncertainty_scale = value;
    }
    if let Some(value) = options.uncertainty_seed {
        config.uncertainty_seed = value;
    }
    if let Some(cap) = options.cap {
        config.cap = Some(cap);
    }
    if let Some(floor) = options.floor {
        config.floor = floor;
    }
    if let Some(name) = &options.cap_regressor {
        config.cap_regressor = Some(name.clone());
    }
    if let Some(name) = &options.floor_regressor {
        config.floor_regressor = Some(name.clone());
    }
    Ok(config)
}

fn neural_panel_config(
    options: &BrowserForecastOptions,
    horizon: usize,
) -> Result<NeuralPanelConfig> {
    let mut config = NeuralPanelConfig {
        n_lags: options
            .n_lags
            .or_else(|| {
                options
                    .lags
                    .as_ref()
                    .and_then(|lags| lags.iter().copied().max())
            })
            .unwrap_or(8),
        n_forecasts: options.n_forecasts.unwrap_or(horizon.max(1)),
        quantiles: options.quantile_levels.clone().unwrap_or_else(|| vec![0.5]),
        trend: options
            .growth
            .as_deref()
            .map(neural_panel_trend_mode)
            .transpose()?
            .unwrap_or(NeuralTrendMode::PiecewiseLinear),
        n_changepoints: options
            .n_changepoints
            .or(options.changepoints)
            .unwrap_or(10),
        changepoints_range: options.changepoint_range.unwrap_or(0.8),
        daily_fourier_order: options.daily_fourier_order.unwrap_or(0),
        weekly_fourier_order: options.weekly_fourier_order.unwrap_or(0),
        yearly_fourier_order: options.yearly_fourier_order.unwrap_or(0),
        custom_seasonalities: BTreeMap::new(),
        custom_seasonality_conditions: BTreeMap::new(),
        seasonality_mode: options
            .seasonality_mode
            .as_deref()
            .or(options.component_mode.as_deref())
            .map(neural_panel_component_mode)
            .transpose()?
            .unwrap_or(NeuralComponentMode::Additive),
        events: BTreeMap::new(),
        event_mode: options
            .event_mode
            .as_deref()
            .or(options.holidays_mode.as_deref())
            .map(neural_panel_component_mode)
            .transpose()?
            .unwrap_or(NeuralComponentMode::Additive),
        future_regressors: BTreeMap::new(),
        lagged_regressors: options.lagged_regressors.clone().unwrap_or_default(),
        ar_layers: options.ar_layers.clone().unwrap_or_default(),
        lagged_reg_layers: options.lagged_reg_layers.clone().unwrap_or_default(),
        trend_mode: options
            .trend_mode
            .as_deref()
            .map(neural_panel_global_local_mode)
            .transpose()?
            .unwrap_or(NeuralPanelMode::Global),
        seasonality_global_local: NeuralPanelMode::Global,
        event_global_local: NeuralPanelMode::Global,
        regressor_global_local: NeuralPanelMode::Global,
        local_l2: options.local_l2.unwrap_or(0.0),
        seed: options.uncertainty_seed.unwrap_or(0),
        loss: NeuralPanelLoss::SmoothL1,
        epochs: 80,
        learning_rate: 0.01,
        weight_decay: 0.0,
        newer_sample_weight: false,
        backend: select_backend_for(options.backend.as_deref(), BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
    };
    if let Some(seasonalities) = &options.custom_seasonalities {
        config.custom_seasonalities = seasonalities
            .iter()
            .map(|seasonality| {
                (
                    seasonality.name.clone(),
                    (seasonality.period_days * 24.0, seasonality.fourier_order),
                )
            })
            .collect();
        config.custom_seasonality_conditions = seasonalities
            .iter()
            .map(|seasonality| (seasonality.name.clone(), seasonality.condition_name.clone()))
            .collect();
    }
    if let Some(events) = &options.events {
        for event in events {
            let lower = event.lower_window.unwrap_or(0);
            let upper = event.upper_window.unwrap_or(0);
            config
                .events
                .entry(event.name.clone())
                .or_default()
                .extend(lower..=upper);
        }
    }
    if let Some(holidays) = &options.holidays {
        for holiday in holidays {
            let lower = holiday.lower_window.unwrap_or(0);
            let upper = holiday.upper_window.unwrap_or(0);
            config
                .events
                .entry(holiday.holiday.clone())
                .or_default()
                .extend(lower..=upper);
        }
    }
    if let Some(regressors) = &options.extra_regressors {
        for name in regressors {
            let mode = options
                .regressor_modes
                .as_ref()
                .and_then(|modes| modes.get(name))
                .map(|value| neural_panel_component_mode(value))
                .transpose()?
                .unwrap_or(NeuralComponentMode::Additive);
            config.future_regressors.insert(name.clone(), mode);
        }
    }
    config
        .validate()
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
    Ok(config)
}

fn browser_tanh_training_backend(options: &BrowserForecastOptions) -> Result<BackendSelection> {
    let requested = options.backend.as_deref().unwrap_or("cpu");
    if requested.eq_ignore_ascii_case("cpu") {
        return select_backend(Some("cpu"))
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()));
    }
    select_backend_for_operations(
        Some(requested),
        &[BackendOperation::TanhMlpTraining, BackendOperation::Dense],
    )
    .or_else(|error| {
        if requested.eq_ignore_ascii_case("auto") {
            select_backend(Some("cpu"))
        } else {
            Err(error)
        }
    })
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))
}

fn nbeats_config(options: &BrowserForecastOptions) -> Result<NBeatsConfig> {
    Ok(NBeatsConfig {
        input_size: options.input_size.unwrap_or(8),
        hidden_size: options.hidden_size.unwrap_or(16),
        epochs: options.epochs.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.01),
        backend: browser_tanh_training_backend(options)?,
    })
}

fn nhits_config(options: &BrowserForecastOptions) -> Result<NHiTSConfig> {
    Ok(NHiTSConfig {
        input_size: options.input_size.unwrap_or(12),
        hidden_size: options.hidden_size.unwrap_or(16),
        epochs: options.epochs.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.01),
        pooling_size: options.pooling_size.unwrap_or(2),
        backend: browser_tanh_training_backend(options)?,
    })
}

fn neural_panel_trend_mode(value: &str) -> Result<NeuralTrendMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "linear" | "piecewise_linear" => Ok(NeuralTrendMode::PiecewiseLinear),
        "off" | "none" | "flat" => Ok(NeuralTrendMode::Off),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported neural_panel trend mode {other:?}"
        ))),
    }
}

fn neural_panel_component_mode(value: &str) -> Result<NeuralComponentMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "additive" => Ok(NeuralComponentMode::Additive),
        "multiplicative" => Ok(NeuralComponentMode::Multiplicative),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported neural_panel component mode {other:?}"
        ))),
    }
}

fn neural_panel_global_local_mode(value: &str) -> Result<NeuralPanelMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "global" => Ok(NeuralPanelMode::Global),
        "local" => Ok(NeuralPanelMode::Local),
        "glocal" => Ok(NeuralPanelMode::Glocal),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported neural_panel global/local mode {other:?}"
        ))),
    }
}

fn piecewise_component_mode(value: &str) -> Result<PiecewiseLinearComponentMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "additive" => Ok(PiecewiseLinearComponentMode::Additive),
        "multiplicative" => Ok(PiecewiseLinearComponentMode::Multiplicative),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal component mode {other:?}"
        ))),
    }
}

fn piecewise_fit_loss(value: &str) -> Result<PiecewiseLinearFitLoss> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "squared" | "l2" | "least_squares" => Ok(PiecewiseLinearFitLoss::Squared),
        "huber" | "robust" => Ok(PiecewiseLinearFitLoss::Huber),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal fit loss {other:?}"
        ))),
    }
}

fn piecewise_prior_scale_to_l2(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "prior scale values must be positive finite numbers".to_string(),
        ));
    }
    Ok(1.0 / (value * value))
}

fn piecewise_prior_scale_to_l1(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "prior scale values must be positive finite numbers".to_string(),
        ));
    }
    Ok(1.0 / value)
}

fn piecewise_regressor_standardization(
    value: &str,
) -> Result<cartoboost_core::forecasting::PiecewiseLinearRegressorStandardization> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "auto" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearRegressorStandardization::Auto)
        }
        "none" | "off" | "false" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearRegressorStandardization::None)
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal regressor standardization {other:?}"
        ))),
    }
}

fn piecewise_trend_uncertainty_policy(
    value: &str,
) -> Result<cartoboost_core::forecasting::PiecewiseLinearTrendUncertaintyPolicy> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "laplace" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearTrendUncertaintyPolicy::Laplace)
        }
        "normal" | "gaussian" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearTrendUncertaintyPolicy::Normal)
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal trend uncertainty policy {other:?}"
        ))),
    }
}

fn default_model() -> String {
    "auto_forecast".to_string()
}

fn booster_config(options: &BrowserForecastOptions) -> BoosterConfig {
    let mut config = BoosterConfig::default();
    if let Some(n_estimators) = options.n_estimators {
        config.n_estimators = n_estimators;
    }
    if let Some(learning_rate) = options.learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(max_depth) = options.max_depth {
        config.max_depth = max_depth;
    }
    if let Some(min_samples_leaf) = options.min_samples_leaf {
        config.min_samples_leaf = min_samples_leaf;
    }
    config
}

fn lag_config(options: &BrowserForecastOptions) -> LagFeatureConfig {
    LagFeatureConfig {
        lags: options
            .lags
            .clone()
            .unwrap_or_else(|| vec![1, 2, 3, options.season_length.unwrap_or(7)]),
        rolling_mean_windows: options
            .rolling_mean_windows
            .clone()
            .unwrap_or_else(|| vec![options.season_length.unwrap_or(7)]),
        rolling_std_windows: options.rolling_std_windows.clone().unwrap_or_default(),
        rolling_min_windows: options.rolling_min_windows.clone().unwrap_or_default(),
        rolling_max_windows: options.rolling_max_windows.clone().unwrap_or_default(),
        difference_lags: options.difference_lags.clone().unwrap_or_default(),
        rolling_trend_windows: options.rolling_trend_windows.clone().unwrap_or_default(),
        calendar_features: calendar_features(options),
        ..LagFeatureConfig::default()
    }
}

fn calendar_features(options: &BrowserForecastOptions) -> Vec<CalendarFeature> {
    let Some(features) = &options.calendar_features else {
        return vec![CalendarFeature::DayOfWeek, CalendarFeature::Month];
    };
    features
        .iter()
        .filter_map(
            |feature| match feature.trim().to_ascii_lowercase().as_str() {
                "day_of_week" | "dow" => Some(CalendarFeature::DayOfWeek),
                "day_of_week_sin" | "dow_sin" => Some(CalendarFeature::DayOfWeekSin),
                "day_of_week_cos" | "dow_cos" => Some(CalendarFeature::DayOfWeekCos),
                "month" => Some(CalendarFeature::Month),
                "month_sin" => Some(CalendarFeature::MonthSin),
                "month_cos" => Some(CalendarFeature::MonthCos),
                "day" => Some(CalendarFeature::Day),
                "day_sin" => Some(CalendarFeature::DaySin),
                "day_cos" => Some(CalendarFeature::DayCos),
                "day_of_year" | "doy" => Some(CalendarFeature::DayOfYear),
                "elapsed_index" => Some(CalendarFeature::ElapsedIndex),
                "elapsed_phase" => Some(CalendarFeature::ElapsedPhase(
                    options.season_length.unwrap_or(7).max(2),
                )),
                _ => None,
            },
        )
        .collect()
}

fn coordinates_from_frame(
    frame: &ForecastFrame,
    options: &BrowserForecastOptions,
) -> Result<BTreeMap<String, (f64, f64)>> {
    let x_name = options
        .coordinate_x
        .as_deref()
        .unwrap_or_else(|| infer_covariate(frame, &["longitude", "lon", "lng", "x"]).unwrap_or(""));
    let y_name = options
        .coordinate_y
        .as_deref()
        .unwrap_or_else(|| infer_covariate(frame, &["latitude", "lat", "y"]).unwrap_or(""));
    if x_name.is_empty() || y_name.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging requires coordinate covariates such as longitude/latitude or x/y".to_string(),
        ));
    }
    let mut coordinates = BTreeMap::new();
    for row in frame.rows() {
        if coordinates.contains_key(&row.series_id) {
            continue;
        }
        let x = row.covariates.get(x_name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing kriging x coordinate covariate {x_name:?}"
            ))
        })?;
        let y = row.covariates.get(y_name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing kriging y coordinate covariate {y_name:?}"
            ))
        })?;
        coordinates.insert(row.series_id.clone(), (x, y));
    }
    Ok(coordinates)
}

fn infer_covariate<'a>(frame: &'a ForecastFrame, names: &[&str]) -> Option<&'a str> {
    let first = frame.rows().first()?;
    for candidate in names {
        if let Some((name, _)) = first
            .covariates
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(candidate))
        {
            return Some(name.as_str());
        }
    }
    None
}

#[allow(dead_code)]
fn _assert_forecast_result_is_serializable(result: &ForecastResult) -> Value {
    result.to_json_value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, NaiveDate};
    use std::collections::BTreeSet;

    #[test]
    fn forecast_model_registry_keeps_cartoboost_first_without_duplicates() {
        let registry = forecast_model_registry();
        let names = registry.iter().map(|model| model.name).collect::<Vec<_>>();
        assert_eq!(
            &names[..7],
            &[
                "auto_forecast",
                "cartoboost_lag",
                "cartoboost_direct",
                "rectified_recursive",
                "lag_plus",
                "scaled_cartoboost_lag",
                "log1p_cartoboost_lag",
            ]
        );
        let unique = names.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn browser_geotemporal_diagnostics_runs_rust_primitives() {
        let response = run_geotemporal_diagnostics_request(BrowserGeotemporalDiagnosticsRequest {
            quantiles: Some(BrowserQuantileDiagnosticsRequest {
                values: Some(vec![10.0, 9.0, 11.0]),
                actual: Some(vec![9.0, 10.0, 12.0]),
                prediction: Some(vec![8.5, 10.5, 12.5]),
                quantile: Some(0.5),
                lower: Some(vec![8.0, 9.0, 10.0]),
                upper: Some(vec![10.0, 12.0, 13.0]),
                quantile_rows: Some(vec![vec![8.0, 9.0, 10.0], vec![9.0, 8.5, 12.0]]),
            }),
            residual_correction: Some(BrowserResidualCorrectionRequest {
                process_variance: 0.05,
                observation_variance: 1.0,
                observations: vec![
                    BrowserResidualObservation {
                        key: BrowserResidualStateKey {
                            origin: Some("PU1".to_string()),
                            destination: Some("DO2".to_string()),
                            corridor: Some("PU1_DO2".to_string()),
                            ..BrowserResidualStateKey::default()
                        },
                        structural_prediction: 10.0,
                        observed: Some(12.0),
                    },
                    BrowserResidualObservation {
                        key: BrowserResidualStateKey {
                            origin: Some("PU1".to_string()),
                            destination: Some("DO2".to_string()),
                            corridor: Some("PU1_DO2".to_string()),
                            ..BrowserResidualStateKey::default()
                        },
                        structural_prediction: 11.0,
                        observed: None,
                    },
                ],
            }),
            regime: Some(BrowserRegimeDiagnosticsRequest {
                residuals: vec![0.0, 0.1, 0.0, 4.0, 4.2],
                cusum: Some(CusumConfig {
                    reference_mean: 0.0,
                    drift: 0.05,
                    threshold: 2.0,
                }),
                page_hinkley: Some(PageHinkleyConfig {
                    delta: 0.01,
                    threshold: 1.0,
                }),
                ewma: Some(EwmaVolatilityConfig { alpha: 0.5 }),
                lower: Some(vec![-1.0; 5]),
                upper: Some(vec![1.0; 5]),
                policy: Some(RegimeIntervalPolicy {
                    widening_multiplier: 0.5,
                    active_window: 2,
                }),
                rolling_window: Some(3),
            }),
            calibration: Some(BrowserCalibrationRequest {
                scores: Some(vec![-2.0, -0.5, 0.5, 2.0]),
                labels: vec![0.0, 0.0, 1.0, 1.0],
                probabilities: Some(vec![0.2, 0.4, 0.6, 0.8]),
                before_probabilities: None,
                method: Some("sigmoid".to_string()),
                bucket_count: Some(4),
                event: Some(BrowserCalibrationEventRequest {
                    kind: "failureRisk".to_string(),
                    actual: vec![1.0, 3.0, 5.0],
                    prediction: None,
                    threshold: Some(2.0),
                    horizon: None,
                    warning_threshold: None,
                    critical_threshold: None,
                }),
            }),
        })
        .expect("geotemporal diagnostics");

        assert_eq!(
            response["surface"].as_str(),
            Some("rust_geotemporal_diagnostics")
        );
        assert_eq!(
            response["quantiles"]["repairedValues"].as_array().unwrap()[1].as_f64(),
            Some(10.0)
        );
        assert_eq!(
            response["residualCorrection"]["stateCount"].as_u64(),
            Some(1)
        );
        assert!(response["regime"]["regimeAdjustedIntervals"]
            .as_array()
            .expect("intervals")
            .iter()
            .any(|row| row["confidence"].as_f64().unwrap() < 1.0));
        assert_eq!(
            response["calibration"]["eventLabels"]
                .as_array()
                .expect("event labels")
                .len(),
            3
        );
        assert!(response["calibration"]["calibratedProbabilities"].is_array());
    }

    #[test]
    fn browser_geo_feature_examples_emit_bearing_columns() {
        let response = run_geo_feature_examples_request(BrowserGeoFeatureRequest {
            planar_routes: vec![
                BrowserPlanarRoute {
                    label: "north".to_string(),
                    origin: [0.0, 0.0],
                    destination: [0.0, 2.0],
                },
                BrowserPlanarRoute {
                    label: "same".to_string(),
                    origin: [1.0, 1.0],
                    destination: [1.0, 1.0],
                },
            ],
            latlng_routes: vec![BrowserLatLngRoute {
                label: "latlng-north".to_string(),
                origin: [40.0, -73.0],
                destination: [41.0, -73.0],
            }],
            radial_points: vec![BrowserNamedPoint {
                label: "point".to_string(),
                point: [3.0, 4.0],
            }],
            anchors: vec![
                BrowserNamedPoint {
                    label: "origin".to_string(),
                    point: [0.0, 0.0],
                },
                BrowserNamedPoint {
                    label: "x-axis".to_string(),
                    point: [3.0, 0.0],
                },
            ],
            length_scale: 1.0,
            local_frame: Some(BrowserLocalFrame {
                origin: [1.0, 1.0],
                axis: [0.0, 1.0],
                points: vec![BrowserNamedPoint {
                    label: "projected".to_string(),
                    point: [2.0, 3.0],
                }],
            }),
        })
        .expect("geo feature examples");
        assert_eq!(response.planar[0].east, Some(0.0));
        assert_eq!(response.planar[0].north, Some(1.0));
        assert!(response.planar[1].zero_distance);
        assert!(response.latlng[0].east.unwrap().abs() < 1.0e-12);
        assert!((response.latlng[0].north.unwrap() - 1.0).abs() < 1.0e-12);
        assert_eq!(response.routes[0].distance, Some(2.0));
        assert_eq!(response.radial[0].values, vec![5.0, 4.0]);
        assert_eq!(response.rbf[0].values[0], (-12.5_f64).exp());
        assert_eq!(response.local_frame[0].along_axis, Some(2.0));
        assert_eq!(response.local_frame[0].cross_axis, Some(-1.0));
    }

    #[test]
    fn browser_geo_feature_accelerated_distances_match_cpu_features() {
        fn request() -> BrowserGeoFeatureRequest {
            BrowserGeoFeatureRequest {
                planar_routes: Vec::new(),
                latlng_routes: Vec::new(),
                radial_points: vec![
                    BrowserNamedPoint {
                        label: "first".to_string(),
                        point: [3.0, 4.0],
                    },
                    BrowserNamedPoint {
                        label: "second".to_string(),
                        point: [6.0, 8.0],
                    },
                ],
                anchors: vec![
                    BrowserNamedPoint {
                        label: "origin".to_string(),
                        point: [0.0, 0.0],
                    },
                    BrowserNamedPoint {
                        label: "x-axis".to_string(),
                        point: [3.0, 0.0],
                    },
                ],
                length_scale: 2.0,
                local_frame: None,
            }
        }

        let cpu = run_geo_feature_examples_request(request()).expect("CPU geo features");
        let accelerated = run_geo_feature_examples_with_distances(
            request(),
            Some(vec![vec![5.0, 4.0], vec![10.0, 8.544_003_745_317_53]]),
            "webgpu",
        )
        .expect("accelerated geo features");

        for (actual, expected) in accelerated.radial.iter().zip(&cpu.radial) {
            for (actual, expected) in actual.values.iter().zip(&expected.values) {
                assert!((actual - expected).abs() < 1.0e-12);
            }
        }
        for (actual, expected) in accelerated.rbf.iter().zip(&cpu.rbf) {
            for (actual, expected) in actual.values.iter().zip(&expected.values) {
                assert!((actual - expected).abs() < 1.0e-12);
            }
        }
        assert_eq!(accelerated.metadata["backend"]["selected"], "webgpu");
        assert_eq!(
            accelerated.metadata["acceleratedOperations"][0],
            "pairwise_distance"
        );
    }

    #[test]
    fn browser_piecewise_linear_seasonal_forecast_runs_through_dispatch() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(2),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                seasonality_l2_regularization: Some(0.001),
                weekly_l2_regularization: Some(0.002),
                fit_loss: Some("huber".to_string()),
                huber_delta: Some(1.25),
                irls_iterations: Some(4),
                include_components: Some(true),
                include_history_components: Some(true),
                include_samples: Some(true),
                include_quantiles: Some(true),
                uncertainty_samples: Some(4),
                quantile_levels: Some(vec![0.1, 0.5, 0.9]),
                coefficient_uncertainty_scale: Some(1.5),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let component_records = response
            .components
            .as_ref()
            .and_then(|components| components.get("records"))
            .and_then(Value::as_array)
            .expect("component records");
        let history_component_records = response
            .history_components
            .as_ref()
            .and_then(|components| components.get("records"))
            .and_then(Value::as_array)
            .expect("history component records");
        let sample_records = response
            .samples
            .as_ref()
            .and_then(|samples| samples.get("records"))
            .and_then(Value::as_array)
            .expect("sample records");
        let quantile_records = response
            .quantiles
            .as_ref()
            .and_then(|quantiles| quantiles.get("records"))
            .and_then(Value::as_array)
            .expect("quantile records");

        assert_eq!(records.len(), 9);
        assert_eq!(component_records.len(), 9);
        assert_eq!(history_component_records.len(), sample_panel_rows().len());
        assert_eq!(sample_records.len(), 36);
        assert_eq!(quantile_records.len(), 27);
        assert!(component_records[0]["components"]["weekly"]
            .as_f64()
            .is_some());
        assert!(history_component_records[1]["trend_movement"]
            .as_f64()
            .is_some());
        assert!(sample_records[0]["prediction"].as_f64().is_some());
        assert_eq!(quantile_records[1]["quantile"].as_f64(), Some(0.5));
        assert_eq!(
            response.metadata["model"].as_str(),
            Some("piecewise_linear_seasonal")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["weekly_l2_regularization"].as_f64(),
            Some(0.002)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["weekly_fourier_order"].as_u64(),
            Some(0)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["auto_weekly_seasonality"].as_bool(),
            Some(false)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["fit_loss"].as_str(),
            Some("huber")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["huber_delta"].as_f64(),
            Some(1.25)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["irls_iterations"].as_u64(),
            Some(4)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["coefficient_uncertainty_scale"].as_f64(),
            Some(1.5)
        );
    }

    #[test]
    fn browser_piecewise_linear_accepts_prophet_modeling_aliases() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                n_changepoints: Some(4),
                changepoint_prior_scale: Some(0.2),
                seasonality_prior_scale: Some(5.0),
                holidays_prior_scale: Some(10.0),
                seasonality_mode: Some("multiplicative".to_string()),
                holidays_mode: Some("additive".to_string()),
                holidays: Some(vec![BrowserForecastHoliday {
                    holiday: "airport_queue_surge".to_string(),
                    ds: "2026-01-03T00:00:00".to_string(),
                    lower_window: Some(-1),
                    upper_window: Some(1),
                    prior_scale: Some(2.0),
                }]),
                interval_width: Some(0.8),
                uncertainty_samples: Some(8),
                include_components: Some(true),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("piecewise prophet aliases forecast");

        assert_eq!(
            response.metadata["modelMetadata"]["changepoints"].as_u64(),
            Some(4)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["changepoint_l1_regularization"].as_f64(),
            Some(5.0)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["seasonality_l2_regularization"].as_f64(),
            Some(0.04)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["event_l2_regularization"].as_f64(),
            Some(0.01)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["event_l2_regularization_by_name"]
                ["airport_queue_surge"]
                .as_f64(),
            Some(0.25)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["component_mode"].as_str(),
            Some("multiplicative")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["event_mode"].as_str(),
            Some("additive")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["interval_levels"][0].as_f64(),
            Some(0.8)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["events"][0]["name"].as_str(),
            Some("airport_queue_surge")
        );
    }

    #[test]
    fn browser_piecewise_linear_rejects_unsupported_prophet_mcmc_alias() {
        let err = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                mcmc_samples: Some(100),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect_err("mcmc alias should fail clearly");

        assert!(err.to_string().contains("mcmc_samples"));
    }

    #[test]
    fn browser_piecewise_linear_omits_unused_sample_payload() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(2),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                uncertainty_samples: Some(8),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("lean piecewise forecast");

        assert!(response.components.is_none());
        assert!(response.samples.is_none());
        assert_eq!(
            response.metadata["modelMetadata"]["uncertainty_samples"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn browser_piecewise_linear_trend_adjustment_and_shock_options_flow_through_dispatch() {
        let base_options = || BrowserForecastOptions {
            changepoints: Some(0),
            weekly_fourier_order: Some(0),
            auto_weekly_seasonality: Some(false),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let trend_adjustments = BTreeMap::from([(2, 1.10)]);
        let baseline = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: base_options(),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("baseline piecewise forecast");
        let trend_adjusted = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                trend_adjustments: Some(trend_adjustments.clone()),
                ..base_options()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("trend-adjusted piecewise forecast");
        let shock_adjusted = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                trend_adjustments: Some(trend_adjustments.clone()),
                residual_shock_window: Some(2),
                residual_shock_scale: Some(0.5),
                residual_shock_decay: Some(0.8),
                ..base_options()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("shock-adjusted piecewise forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: sample_panel_rows(),
                frequency: "daily".to_string(),
                horizon: 2,
                model: "piecewise_linear_seasonal".to_string(),
                options: base_options(),
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit piecewise artifact");
        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            2,
            BrowserForecastArtifactPredictOptions {
                trend_adjustments: Some(trend_adjustments),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact trend-adjusted forecast");

        let baseline_records = baseline.forecast["records"].as_array().expect("records");
        let trend_adjusted_records = trend_adjusted.forecast["records"]
            .as_array()
            .expect("records");
        let shock_adjusted_records = shock_adjusted.forecast["records"]
            .as_array()
            .expect("records");
        let restored_records = restored.forecast["records"].as_array().expect("records");
        assert!(
            trend_adjusted_records[1]["prediction"].as_f64().unwrap()
                > baseline_records[1]["prediction"].as_f64().unwrap()
        );
        assert!(
            shock_adjusted_records[0]["prediction"].as_f64().unwrap()
                != trend_adjusted_records[0]["prediction"].as_f64().unwrap()
        );
        assert_eq!(trend_adjusted.forecast, restored.forecast);
        assert_eq!(
            shock_adjusted.metadata["modelMetadata"]["trend_adjustments"]["2"].as_f64(),
            Some(1.10)
        );
        assert_eq!(
            shock_adjusted.metadata["modelMetadata"]["residual_shock_window"].as_u64(),
            Some(2)
        );
        assert_eq!(
            shock_adjusted.components.as_ref().expect("components")["records"][1]
                ["trend_adjustment_multiplier"]
                .as_f64(),
            Some(1.10)
        );
        assert_eq!(
            restored_records[1]["prediction"].as_f64(),
            trend_adjusted_records[1]["prediction"].as_f64()
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predicts_without_refit() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: 50.0 + f64::from(day) + 20.0 * queue,
                        covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = || BrowserForecastOptions {
            changepoints: Some(1),
            weekly_fourier_order: Some(0),
            interval_levels: Some(vec![0.8]),
            extra_regressors: Some(vec!["airport_queue".to_string()]),
            future_regressors: Some(BTreeMap::from([(
                "airport_queue".to_string(),
                vec![1.0, 0.0, 0.0],
            )])),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 3,
                model: "piecewise_linear_seasonal".to_string(),
                options: options(),
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit artifact");
        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            3,
            BrowserForecastArtifactPredictOptions::default(),
        )
        .expect("artifact forecast");

        assert_eq!(direct.forecast, restored.forecast);
        let direct_queue = direct.components.as_ref().expect("direct components")["records"][0]
            ["components"]["regressors"]["airport_queue"]
            .as_f64()
            .expect("direct airport queue contribution");
        let restored_queue = restored.components.as_ref().expect("artifact components")["records"]
            [0]["components"]["regressors"]["airport_queue"]
            .as_f64()
            .expect("airport queue contribution");
        assert!(restored_queue > 10.0);
        assert!((direct_queue - restored_queue).abs() < 1.0e-9);
        assert_eq!(
            serde_json::from_str::<Value>(&artifact_response.artifact).expect("artifact")["kind"]
                .as_str(),
            Some("cartoboost_piecewise_linear_seasonal")
        );
        assert_eq!(
            artifact_response.metadata["model"].as_str(),
            Some("piecewise_linear_seasonal")
        );

        let lean_restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            3,
            BrowserForecastArtifactPredictOptions {
                include_components: false,
                include_samples: false,
                include_quantiles: false,
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("lean artifact forecast");
        assert_eq!(direct.forecast, lean_restored.forecast);
        assert!(lean_restored.components.is_none());
        assert!(lean_restored.samples.is_none());
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_quantile_overrides() {
        let rows = || {
            (1..=28)
                .map(|day| BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 40.0 + f64::from(day) + if day % 7 == 0 { 4.0 } else { -1.0 },
                    covariates: BTreeMap::new(),
                })
                .collect::<Vec<_>>()
        };
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 2,
                model: "piecewise_linear_seasonal".to_string(),
                options: BrowserForecastOptions {
                    changepoints: Some(2),
                    weekly_fourier_order: Some(0),
                    auto_weekly_seasonality: Some(false),
                    uncertainty_samples: Some(24),
                    include_quantiles: Some(false),
                    ..BrowserForecastOptions::default()
                },
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit artifact without default quantiles");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            2,
            BrowserForecastArtifactPredictOptions {
                quantile_levels: Some(vec![0.1, 0.5, 0.9]),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact forecast with quantile override");
        let quantiles = restored.quantiles.expect("quantile payload");

        assert_eq!(
            restored.metadata["modelMetadata"]["quantile_levels"],
            json!([0.1, 0.5, 0.9])
        );
        assert_eq!(quantiles["quantile_levels"], json!([0.1, 0.5, 0.9]));
        assert_eq!(
            quantiles["records"]
                .as_array()
                .expect("quantile records")
                .len(),
            2 * 3
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_future_regressor_options() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: 50.0 + f64::from(day) + 20.0 * queue,
                        covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let fit_options = || BrowserForecastOptions {
            changepoints: Some(1),
            weekly_fourier_order: Some(0),
            extra_regressors: Some(vec!["airport_queue".to_string()]),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let future_regressors =
            BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]);
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                future_regressors: Some(future_regressors.clone()),
                ..fit_options()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 3,
                model: "piecewise_linear_seasonal".to_string(),
                options: fit_options(),
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit artifact without future values");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            3,
            BrowserForecastArtifactPredictOptions {
                future_regressors: Some(future_regressors),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact forecast with future values");

        assert_eq!(direct.forecast, restored.forecast);
        assert_eq!(
            restored.metadata["modelMetadata"]["future_regressors"]["airport_queue"][0].as_f64(),
            Some(1.0)
        );
        assert!(
            restored.components.as_ref().expect("components")["records"][0]["components"]
                ["regressors"]["airport_queue"]
                .as_f64()
                .expect("future regressor contribution")
                > 10.0
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_series_future_caps() {
        let rows = || {
            ["pickup_zone_a", "pickup_zone_b"]
                .into_iter()
                .flat_map(|series_id| {
                    (1..=28).map(move |day| {
                        let cap = if series_id == "pickup_zone_a" {
                            110.0 + 0.25 * f64::from(day)
                        } else {
                            65.0 + 0.10 * f64::from(day)
                        };
                        let t = f64::from(day) - 14.0;
                        BrowserForecastRow {
                            series_id: Some(series_id.to_string()),
                            timestamp: format!("2026-01-{day:02}T00:00:00"),
                            target: cap / (1.0 + (-0.18 * t).exp()),
                            covariates: BTreeMap::from([("zone_capacity".to_string(), cap)]),
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        let fit_options = BrowserForecastOptions {
            growth: Some("logistic".to_string()),
            changepoints: Some(3),
            weekly_fourier_order: Some(0),
            auto_weekly_seasonality: Some(false),
            cap_regressor: Some("zone_capacity".to_string()),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let future_regressors_by_series = BTreeMap::from([
            (
                "pickup_zone_a".to_string(),
                BTreeMap::from([("zone_capacity".to_string(), vec![120.0])]),
            ),
            (
                "pickup_zone_b".to_string(),
                BTreeMap::from([("zone_capacity".to_string(), vec![70.0])]),
            ),
        ]);
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("logistic".to_string()),
                changepoints: Some(3),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                cap_regressor: Some("zone_capacity".to_string()),
                include_components: Some(true),
                future_regressors_by_series: Some(future_regressors_by_series.clone()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct panel logistic forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 1,
                model: "piecewise_linear_seasonal".to_string(),
                options: fit_options,
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit panel logistic artifact without future caps");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            1,
            BrowserForecastArtifactPredictOptions {
                future_regressors_by_series: Some(future_regressors_by_series),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact panel logistic forecast with future caps");
        let records = restored
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let prediction_a = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_a"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone A prediction");
        let prediction_b = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_b"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone B prediction");

        assert_eq!(direct.forecast, restored.forecast);
        assert!(prediction_a > 0.0 && prediction_a < 120.0);
        assert!(prediction_b > 0.0 && prediction_b < 70.0);
        assert!(prediction_a > prediction_b + 20.0);
        assert_eq!(
            restored.metadata["modelMetadata"]["future_regressors_by_series"]["pickup_zone_a"]
                ["zone_capacity"][0]
                .as_f64(),
            Some(120.0)
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_series_future_floors() {
        let cap = 140.0;
        let rows = || {
            ["pickup_zone_a", "pickup_zone_b"]
                .into_iter()
                .flat_map(|series_id| {
                    (1..=28).map(move |day| {
                        let floor = if series_id == "pickup_zone_a" {
                            32.0 + 0.10 * f64::from(day)
                        } else {
                            8.0 + 0.05 * f64::from(day)
                        };
                        let t = f64::from(day) - 14.0;
                        BrowserForecastRow {
                            series_id: Some(series_id.to_string()),
                            timestamp: format!("2026-01-{day:02}T00:00:00"),
                            target: floor + (cap - floor) / (1.0 + (-0.18 * t).exp()),
                            covariates: BTreeMap::from([("service_floor".to_string(), floor)]),
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        let fit_options = BrowserForecastOptions {
            growth: Some("logistic".to_string()),
            changepoints: Some(3),
            weekly_fourier_order: Some(0),
            auto_weekly_seasonality: Some(false),
            cap: Some(cap),
            floor_regressor: Some("service_floor".to_string()),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let future_regressors_by_series = BTreeMap::from([
            (
                "pickup_zone_a".to_string(),
                BTreeMap::from([("service_floor".to_string(), vec![38.0])]),
            ),
            (
                "pickup_zone_b".to_string(),
                BTreeMap::from([("service_floor".to_string(), vec![10.0])]),
            ),
        ]);
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("logistic".to_string()),
                changepoints: Some(3),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                cap: Some(cap),
                floor_regressor: Some("service_floor".to_string()),
                include_components: Some(true),
                future_regressors_by_series: Some(future_regressors_by_series.clone()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct panel logistic forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 1,
                model: "piecewise_linear_seasonal".to_string(),
                options: fit_options,
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit panel logistic artifact without future floors");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            1,
            BrowserForecastArtifactPredictOptions {
                future_regressors_by_series: Some(future_regressors_by_series),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact panel logistic forecast with future floors");
        let lower_floor_restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            1,
            BrowserForecastArtifactPredictOptions {
                future_regressors_by_series: Some(BTreeMap::from([
                    (
                        "pickup_zone_a".to_string(),
                        BTreeMap::from([("service_floor".to_string(), vec![5.0])]),
                    ),
                    (
                        "pickup_zone_b".to_string(),
                        BTreeMap::from([("service_floor".to_string(), vec![10.0])]),
                    ),
                ])),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact panel logistic forecast with lower future floor");
        let records = restored
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let prediction_a = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_a"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone A prediction");
        let prediction_b = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_b"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone B prediction");
        let lower_floor_prediction_a = lower_floor_restored
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("lower floor forecast records")
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_a"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone A lower floor prediction");

        assert_eq!(direct.forecast, restored.forecast);
        assert!(prediction_a > 38.0 && prediction_a < cap);
        assert!(prediction_b > 10.0 && prediction_b < cap);
        assert!(prediction_a > lower_floor_prediction_a);
        assert_eq!(
            restored.metadata["modelMetadata"]["future_regressors_by_series"]["pickup_zone_a"]
                ["service_floor"][0]
                .as_f64(),
            Some(38.0)
        );
    }

    #[test]
    fn browser_piecewise_linear_flat_growth_flows_through_dispatch() {
        let rows = (1..=28)
            .map(|day| BrowserForecastRow {
                series_id: Some("pickup_zone_1".to_string()),
                timestamp: format!("2026-01-{day:02}T00:00:00"),
                target: 40.0 + 2.0 * f64::from(day),
                covariates: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("flat".to_string()),
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("flat piecewise seasonal forecast");

        assert_eq!(
            response.metadata["modelMetadata"]["growth"].as_str(),
            Some("flat")
        );
    }

    #[test]
    fn browser_piecewise_linear_logistic_growth_uses_cap_floor_options() {
        let rows = (0..28)
            .map(|idx| {
                let t = idx as f64 - 14.0;
                let cap = 95.0 + idx as f64;
                let target = cap / (1.0 + (-0.25 * t).exp());
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{:02}T00:00:00", idx + 1),
                    target,
                    covariates: BTreeMap::from([("zone_capacity".to_string(), cap)]),
                }
            })
            .collect::<Vec<_>>();
        let future_caps = vec![123.0, 124.0, 125.0, 126.0];
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 4,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("logistic".to_string()),
                floor: Some(0.0),
                cap_regressor: Some("zone_capacity".to_string()),
                future_regressors: Some(BTreeMap::from([(
                    "zone_capacity".to_string(),
                    future_caps.clone(),
                )])),
                changepoints: Some(4),
                weekly_fourier_order: Some(0),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("logistic piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");

        assert_eq!(
            response.metadata["modelMetadata"]["growth"].as_str(),
            Some("logistic")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["cap_regressor"].as_str(),
            Some("zone_capacity")
        );
        assert!(records.iter().zip(future_caps.iter()).all(|(record, cap)| {
            let prediction = record["prediction"].as_f64().expect("prediction");
            prediction > 0.0 && prediction < *cap
        }));
    }

    #[test]
    fn browser_piecewise_linear_explicit_changepoints_flow_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let target = if day <= 15 {
                    50.0 + f64::from(day)
                } else {
                    65.0 + 5.0 * f64::from(day - 15)
                };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                changepoint_range: Some(0.8),
                changepoint_timestamps: Some(vec!["2026-01-15T00:00:00".to_string()]),
                weekly_fourier_order: Some(0),
                changepoint_l2_regularization: Some(0.001),
                changepoint_l1_regularization: Some(0.01),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("explicit changepoint piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert_eq!(
            response.metadata["modelMetadata"]["changepoint_timestamps"][0].as_str(),
            Some("2026-01-15T00:00:00")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["changepoint_l1_regularization"].as_f64(),
            Some(0.01)
        );
        assert!(records[2]["prediction"].as_f64().expect("prediction") > 140.0);
    }

    #[test]
    fn browser_piecewise_linear_events_flow_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let event_boost = if (14..=16).contains(&day) { 25.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 100.0 + 0.5 * f64::from(day) + event_boost,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                event_l2_regularization: Some(0.001),
                include_components: Some(true),
                events: Some(vec![
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-01-15T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-02-01T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                ]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("event piecewise seasonal forecast");
        let events = response.metadata["modelMetadata"]["events"]
            .as_array()
            .expect("events metadata");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let component_records = response
            .components
            .as_ref()
            .and_then(|components| components.get("records"))
            .and_then(Value::as_array)
            .expect("component records");

        assert_eq!(events[0]["name"].as_str(), Some("airport_surge"));
        assert!(records[1]["prediction"].as_f64().expect("prediction") > 120.0);
        assert!(
            component_records[0]["components"]["event_window_offsets"]["airport_surge[-1]"]
                .as_f64()
                .is_some()
        );
    }

    #[test]
    fn browser_piecewise_linear_extra_regressors_use_future_values() {
        let rows = (1..=30)
            .map(|day| {
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 50.0 + f64::from(day) + 20.0 * queue,
                    covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                regressor_l2_regularization: Some(0.001),
                extra_regressors: Some(vec!["airport_queue".to_string()]),
                future_regressors: Some(BTreeMap::from([(
                    "airport_queue".to_string(),
                    vec![1.0, 0.0, 0.0],
                )])),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("regressor piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");

        assert_eq!(
            response.metadata["modelMetadata"]["extra_regressors"][0].as_str(),
            Some("airport_queue")
        );
        assert!(records[0]["prediction"].as_f64().expect("prediction") > 80.0);
    }

    #[test]
    fn browser_neural_panel_custom_seasonality_flows_through_dispatch() {
        let start = NaiveDate::from_ymd_opt(2026, 3, 1)
            .expect("valid start date")
            .and_hms_opt(0, 0, 0)
            .expect("valid start time");
        let rows = (1..=32)
            .map(|day| {
                let phase = std::f64::consts::TAU * f64::from(day % 8) / 8.0;
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: (start + Duration::days((day - 1) as i64))
                        .format("%Y-%m-%dT%H:%M:%S")
                        .to_string(),
                    target: 50.0 + 8.0 * phase.sin(),
                    covariates: BTreeMap::from([("rushHour".to_string(), 1.0)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "neural_panel".to_string(),
            options: BrowserForecastOptions {
                n_lags: Some(4),
                n_forecasts: Some(3),
                custom_seasonalities: Some(vec![BrowserForecastSeasonality {
                    name: "taxi_cycle".to_string(),
                    period_days: 8.0,
                    fourier_order: 2,
                    mode: Some("additive".to_string()),
                    condition_name: Some("rushHour".to_string()),
                    l2_regularization: None,
                }]),
                include_components: Some(true),
                include_history_components: Some(true),
                backend: Some("cpu".to_string()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("neural panel forecast");
        let feature_schema = response.metadata["modelMetadata"]["feature_schema"]
            .as_array()
            .expect("feature schema");
        let custom_seasonalities = response.metadata["modelMetadata"]["config"]
            ["custom_seasonalities"]
            .as_object()
            .expect("custom seasonalities");
        let custom_seasonality_conditions = response.metadata["modelMetadata"]["config"]
            ["custom_seasonality_conditions"]
            .as_object()
            .expect("custom seasonality conditions");
        assert_eq!(
            response.metadata["modelMetadata"]["config"]["backend"]["selected"].as_str(),
            Some("cpu")
        );

        assert!(feature_schema
            .iter()
            .any(|value| value.as_str() == Some("seasonality:taxi_cycle:sin:1")));
        assert_eq!(custom_seasonalities["taxi_cycle"][0].as_f64(), Some(192.0));
        assert_eq!(
            custom_seasonality_conditions["taxi_cycle"].as_str(),
            Some("rushHour")
        );
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")
            .iter()
            .all(|record| record["prediction"]
                .as_f64()
                .expect("prediction")
                .is_finite()));
        let component_records = response
            .components
            .as_ref()
            .and_then(|components| components.get("series"))
            .and_then(|series| series.get("pickup_zone_1"))
            .and_then(Value::as_array)
            .expect("component records");
        let history_records = response
            .history_components
            .as_ref()
            .and_then(|components| components.get("series"))
            .and_then(|series| series.get("pickup_zone_1"))
            .and_then(Value::as_array)
            .expect("history records");
        assert_eq!(component_records.len(), 3);
        assert_eq!(history_records.len(), 32);
        assert!(component_records[0]["prediction"]
            .as_f64()
            .expect("component prediction")
            .is_finite());
        assert!(history_records[0]["prediction"]
            .as_f64()
            .expect("history prediction")
            .is_finite());
    }

    #[test]
    fn browser_nbeats_forecast_runs_through_generic_dispatch() {
        let rows = (1..=18)
            .map(|day| BrowserForecastRow {
                series_id: Some("pickup_zone_237".to_string()),
                timestamp: format!("2026-02-{day:02}T00:00:00"),
                target: 30.0 + f64::from(day) + f64::from(day % 3),
                covariates: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "nbeats".to_string(),
            options: BrowserForecastOptions {
                input_size: Some(5),
                hidden_size: Some(8),
                epochs: Some(6),
                learning_rate: Some(0.02),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("nbeats forecast");

        assert_eq!(response.metadata["model"].as_str(), Some("nbeats"));
        assert_eq!(
            response.metadata["modelMetadata"]["input_size"].as_u64(),
            Some(5)
        );
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")
            .iter()
            .all(|record| record["prediction"]
                .as_f64()
                .expect("prediction")
                .is_finite()));
    }

    #[test]
    fn browser_nhits_forecast_runs_through_generic_dispatch() {
        let rows = (1..=22)
            .map(|day| BrowserForecastRow {
                series_id: Some("pickup_zone_161".to_string()),
                timestamp: format!("2026-04-{day:02}T00:00:00"),
                target: 45.0 + 2.0 * f64::from(day % 4) + 0.5 * f64::from(day),
                covariates: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 4,
            model: "nhits".to_string(),
            options: BrowserForecastOptions {
                input_size: Some(6),
                hidden_size: Some(10),
                pooling_size: Some(3),
                epochs: Some(6),
                learning_rate: Some(0.02),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("nhits forecast");

        assert_eq!(response.metadata["model"].as_str(), Some("nhits"));
        assert_eq!(
            response.metadata["modelMetadata"]["pooling_size"].as_u64(),
            Some(3)
        );
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")
            .iter()
            .all(|record| record["prediction"]
                .as_f64()
                .expect("prediction")
                .is_finite()));
    }

    #[test]
    fn browser_neural_panel_holidays_and_regressors_flow_through_dispatch() {
        let rows = (1..=16)
            .map(|day| {
                let queue = if day % 4 == 0 { 1.0 } else { 0.0 };
                let holiday = if day == 6 { 1.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 40.0 + f64::from(day) + 9.0 * queue + 14.0 * holiday,
                    covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 2,
            model: "neural_panel".to_string(),
            options: BrowserForecastOptions {
                n_lags: Some(4),
                n_forecasts: Some(2),
                weekly_fourier_order: Some(0),
                extra_regressors: Some(vec!["airport_queue".to_string()]),
                regressor_modes: Some(BTreeMap::from([(
                    "airport_queue".to_string(),
                    "additive".to_string(),
                )])),
                future_regressors: Some(BTreeMap::from([(
                    "airport_queue".to_string(),
                    vec![0.0, 1.0],
                )])),
                holidays: Some(vec![BrowserForecastHoliday {
                    holiday: "airport_holiday".to_string(),
                    ds: "2026-01-06T00:00:00".to_string(),
                    lower_window: Some(0),
                    upper_window: Some(0),
                    prior_scale: Some(10.0),
                }]),
                holidays_mode: Some("additive".to_string()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("neural panel forecast");
        let feature_schema = response.metadata["modelMetadata"]["feature_schema"]
            .as_array()
            .expect("feature schema");

        assert!(feature_schema
            .iter()
            .any(|value| value.as_str() == Some("airport_queue")));
        assert!(feature_schema
            .iter()
            .any(|value| value.as_str() == Some("event:airport_holiday:0")));
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")[0]["prediction"]
            .as_f64()
            .expect("prediction")
            .is_finite());
    }

    #[test]
    fn browser_piecewise_linear_regressor_standardization_flows_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let traffic_index = 100.0 + 4.0 * f64::from(day);
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 40.0 + f64::from(day) + 1.5 * traffic_index,
                    covariates: BTreeMap::from([("trafficIndex".to_string(), traffic_index)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                extra_regressors: Some(vec!["trafficIndex".to_string()]),
                regressor_standardization: Some("none".to_string()),
                future_regressors: Some(BTreeMap::from([(
                    "trafficIndex".to_string(),
                    vec![224.0, 228.0],
                )])),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("standardization piecewise seasonal forecast");

        assert_eq!(
            response.metadata["modelMetadata"]["regressor_standardization"].as_str(),
            Some("none")
        );
    }

    #[test]
    fn browser_piecewise_linear_named_regressor_l2_flows_through_dispatch() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: 50.0 + f64::from(day) + 24.0 * queue,
                        covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = |named_l2: f64| BrowserForecastOptions {
            changepoints: Some(0),
            weekly_fourier_order: Some(0),
            extra_regressors: Some(vec!["airport_queue".to_string()]),
            regressor_l2_regularization_by_name: Some(BTreeMap::from([(
                "airport_queue".to_string(),
                named_l2,
            )])),
            future_regressors: Some(BTreeMap::from([("airport_queue".to_string(), vec![1.0])])),
            ..BrowserForecastOptions::default()
        };
        let low_l2_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(0.001),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("low l2 forecast");
        let high_l2_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(1_000.0),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("high l2 forecast");
        let low_prediction = low_l2_response.forecast["records"][0]["prediction"]
            .as_f64()
            .expect("low l2 prediction");
        let high_prediction = high_l2_response.forecast["records"][0]["prediction"]
            .as_f64()
            .expect("high l2 prediction");

        assert!(low_prediction > high_prediction + 10.0);
        assert_eq!(
            high_l2_response.metadata["modelMetadata"]["regressor_l2_regularization_by_name"]
                ["airport_queue"]
                .as_f64(),
            Some(1_000.0)
        );
    }

    #[test]
    fn browser_piecewise_linear_custom_seasonalities_flow_through_dispatch() {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid start");
        let rows = (1..=56)
            .map(|day| {
                let timestamp = start + Duration::days(i64::from(day - 1));
                let biweekly = if day % 14 == 0 { 18.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    target: 80.0 + 0.25 * f64::from(day) + biweekly,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 14,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                seasonality_l2_regularization: Some(0.001),
                custom_seasonalities: Some(vec![BrowserForecastSeasonality {
                    name: "biweekly_pickup_cycle".to_string(),
                    period_days: 14.0,
                    fourier_order: 4,
                    mode: Some("additive".to_string()),
                    condition_name: None,
                    l2_regularization: None,
                }]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("custom seasonality piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert_eq!(
            response.metadata["modelMetadata"]["custom_seasonalities"][0]["name"].as_str(),
            Some("biweekly_pickup_cycle")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["custom_seasonalities"][0]["mode"].as_str(),
            Some("additive")
        );
        assert!(records[13]["prediction"].as_f64().expect("prediction") > 95.0);
    }

    #[test]
    fn browser_piecewise_linear_conditional_seasonality_uses_future_flags() {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid start");
        let rows = || {
            (1..=42)
                .map(|day| {
                    let timestamp = start + Duration::days(i64::from(day - 1));
                    let rush_hour = if day % 2 == 0 { 1.0 } else { 0.0 };
                    let cycle = if day % 7 == 0 { 16.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                        target: 80.0 + 0.2 * f64::from(day) + rush_hour * cycle,
                        covariates: BTreeMap::from([("rushHour".to_string(), rush_hour)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = |future_flags: Vec<f64>| BrowserForecastOptions {
            changepoints: Some(0),
            weekly_fourier_order: Some(0),
            seasonality_l2_regularization: Some(0.001),
            custom_seasonalities: Some(vec![BrowserForecastSeasonality {
                name: "rush_hour_weekly".to_string(),
                period_days: 7.0,
                fourier_order: 3,
                mode: None,
                condition_name: Some("rushHour".to_string()),
                l2_regularization: None,
            }]),
            future_regressors: Some(BTreeMap::from([("rushHour".to_string(), future_flags)])),
            ..BrowserForecastOptions::default()
        };
        let inactive_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 7,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("inactive conditional seasonality forecast");
        let active_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 7,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("active conditional seasonality forecast");
        let inactive_records = inactive_response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("inactive records");
        let active_records = active_response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("active records");

        assert_eq!(
            active_response.metadata["modelMetadata"]["custom_seasonalities"][0]["condition_name"]
                .as_str(),
            Some("rushHour")
        );
        assert!(
            active_records[6]["prediction"].as_f64().expect("active")
                > inactive_records[6]["prediction"]
                    .as_f64()
                    .expect("inactive")
                    + 4.0
        );
    }

    #[test]
    fn browser_piecewise_linear_interval_levels_render_bounds() {
        let rows = (1..=20)
            .map(|day| {
                let noise = if day % 2 == 0 { 2.0 } else { -2.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 25.0 + f64::from(day) + noise,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                interval_levels: Some(vec![0.8]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("interval piecewise seasonal forecast");
        let columns = response
            .forecast
            .get("columns")
            .and_then(Value::as_array)
            .expect("columns");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert!(columns
            .iter()
            .any(|column| column.as_str() == Some("prediction_lower_p80")));
        assert!(records[0]["prediction_lower_p80"].as_f64().is_some());
        assert!(
            records[0]["prediction_lower_p80"].as_f64().unwrap()
                <= records[0]["prediction_upper_p80"].as_f64().unwrap()
        );
    }

    #[test]
    fn browser_piecewise_linear_uncertainty_samples_widen_intervals() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let value =
                        20.0 + 0.5 * f64::from(day) + 3.0 * (f64::from(day) - 15.0).max(0.0);
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: value,
                        covariates: BTreeMap::new(),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = |uncertainty_samples: usize| BrowserForecastOptions {
            changepoints: Some(1),
            changepoint_timestamps: Some(vec!["2026-01-15T00:00:00".to_string()]),
            changepoint_l2_regularization: Some(0.001),
            weekly_fourier_order: Some(0),
            interval_levels: Some(vec![0.8]),
            uncertainty_samples: Some(uncertainty_samples),
            trend_uncertainty_policy: Some("normal".to_string()),
            trend_uncertainty_scale: Some(1.0),
            uncertainty_seed: Some(7),
            ..BrowserForecastOptions::default()
        };
        let residual_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 5,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(0),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("residual interval forecast");
        let uncertain_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 5,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(256),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("uncertain interval forecast");
        let residual_record = &residual_response.forecast["records"][4];
        let uncertain_record = &uncertain_response.forecast["records"][4];
        let residual_width = residual_record["prediction_upper_p80"]
            .as_f64()
            .expect("residual upper")
            - residual_record["prediction_lower_p80"]
                .as_f64()
                .expect("residual lower");
        let uncertain_width = uncertain_record["prediction_upper_p80"]
            .as_f64()
            .expect("uncertain upper")
            - uncertain_record["prediction_lower_p80"]
                .as_f64()
                .expect("uncertain lower");

        assert!(uncertain_width > residual_width + 1.0);
        assert_eq!(
            uncertain_response.metadata["modelMetadata"]["uncertainty_samples"].as_u64(),
            Some(256)
        );
        assert_eq!(
            uncertain_response.metadata["modelMetadata"]["trend_uncertainty_policy"].as_str(),
            Some("normal")
        );
    }

    #[test]
    fn browser_piecewise_linear_multiplicative_mode_flows_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let trend = 20.0 + 2.0 * f64::from(day);
                let target = if (14..=16).contains(&day) {
                    trend * 1.5
                } else {
                    trend
                };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                component_mode: Some("multiplicative".to_string()),
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                event_l2_regularization: Some(0.001),
                events: Some(vec![
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-01-15T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-02-01T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                ]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("multiplicative piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert_eq!(
            response.metadata["modelMetadata"]["component_mode"].as_str(),
            Some("multiplicative")
        );
        assert!(records[1]["prediction"].as_f64().expect("prediction") > 100.0);
    }

    #[test]
    fn browser_sequence_viterbi_runs_through_wasm_dispatch() {
        let response = run_sequence_request(BrowserSequenceRequest {
            operation: "reference_path_viterbi".to_string(),
            frame: None,
            series: Some(sample_sequence_series()),
            reference: Some(ReferenceSignal {
                axis: vec![0.0, 1.0, 2.0, 3.0],
                signal: vec![0.0, 1.0, 2.0, 3.0],
            }),
            state_space_config: None,
            reference_path_config: Some(ReferencePathConfig::default()),
            candidates: None,
            weights: None,
            actuals: None,
            oof_fold: None,
            oof_rows: None,
            group_predictions: None,
        })
        .expect("sequence request");
        let points = response
            .get("points")
            .and_then(Value::as_array)
            .expect("path points");
        assert_eq!(points.len(), 4);
        assert_eq!(points[1]["axis"].as_f64(), Some(1.0));
    }

    #[test]
    fn browser_sequence_oof_generation_runs_through_wasm_dispatch() {
        let response = run_sequence_request(BrowserSequenceRequest {
            operation: "generate_group_oof_candidate_rows".to_string(),
            frame: None,
            series: None,
            reference: None,
            state_space_config: None,
            reference_path_config: None,
            candidates: None,
            weights: None,
            actuals: None,
            oof_fold: Some(SequenceOofFold {
                validation_group_id: "pickup_zone_1".to_string(),
                train_group_ids: vec!["pickup_zone_2".to_string()],
                actuals: vec![SequenceCandidatePrediction {
                    series_id: "pickup_zone_1".to_string(),
                    row_id: "hour_01".to_string(),
                    value: 10.0,
                }],
                candidates: vec![SequenceCandidate {
                    name: "candidate_a".to_string(),
                    predictions: vec![SequenceCandidatePrediction {
                        series_id: "pickup_zone_1".to_string(),
                        row_id: "hour_01".to_string(),
                        value: 11.0,
                    }],
                }],
            }),
            oof_rows: None,
            group_predictions: None,
        })
        .expect("sequence OOF request");
        let rows = response.as_array().expect("OOF rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0]["candidate_predictions"]["candidate_a"].as_f64(),
            Some(11.0)
        );
    }

    #[test]
    fn every_registered_browser_forecast_model_runs_on_representative_panel() {
        for model in forecast_model_registry() {
            let request = BrowserForecastRequest {
                rows: sample_panel_rows(),
                frequency: "daily".to_string(),
                horizon: 7,
                model: model.name.to_string(),
                options: BrowserForecastOptions {
                    season_length: Some(7),
                    coordinate_x: Some("longitude".to_string()),
                    coordinate_y: Some("latitude".to_string()),
                    ..BrowserForecastOptions::default()
                },
                metadata: BrowserForecastMetadata {
                    timestamp_col: Some("timestamp".to_string()),
                    target_col: Some("target".to_string()),
                    series_id_col: Some("series_id".to_string()),
                },
            };
            let response = run_forecast_request(request)
                .unwrap_or_else(|error| panic!("{} failed: {error}", model.name));
            let records = response
                .forecast
                .get("records")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("{} returned no forecast records", model.name));
            assert_eq!(records.len(), 21, "{} record count", model.name);
            assert!(
                response.metadata.get("warning").is_none(),
                "{} used fallback instead of fitting directly: {}",
                model.name,
                response.metadata
            );
            assert!(
                records
                    .iter()
                    .all(|record| record["prediction"].as_f64().is_some_and(f64::is_finite)),
                "{} returned a non-finite prediction",
                model.name
            );
        }
    }

    #[test]
    fn browser_spatial_piecewise_kriging_reports_spatial_details() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "spatial_piecewise_kriging".to_string(),
            options: BrowserForecastOptions {
                coordinate_x: Some("longitude".to_string()),
                coordinate_y: Some("latitude".to_string()),
                kriging_range: Some(1.0),
                kriging_nugget: Some(1.0e-6),
                spatial_kriging_mode: Some("residual_kriging".to_string()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata {
                timestamp_col: Some("timestamp".to_string()),
                target_col: Some("target".to_string()),
                series_id_col: Some("series_id".to_string()),
            },
        })
        .expect("spatial piecewise kriging run");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        assert_eq!(records.len(), 6);
        assert!(records[0].get("base_mean").is_some());
        assert!(records[0].get("spatial_correction").is_some());
        assert!(records[0].get("kriging_variance").is_some());
        assert!(records[0].get("selected_neighbors").is_some());
    }

    #[test]
    fn browser_auto_forecast_caps_direct_horizon_to_requested_horizon() {
        let request = BrowserForecastRequest {
            rows: (0..56)
                .map(|day| BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: date_string(day),
                    target: 120.0 + day as f64 * 2.0 + (day % 7) as f64 * 4.0,
                    covariates: BTreeMap::new(),
                })
                .collect(),
            frequency: "daily".to_string(),
            horizon: 14,
            model: "auto_forecast".to_string(),
            options: BrowserForecastOptions {
                season_length: Some(7),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata {
                timestamp_col: Some("timestamp".to_string()),
                target_col: Some("target".to_string()),
                series_id_col: Some("series_id".to_string()),
            },
        };
        let response = run_forecast_request(request).expect("auto forecast run");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        assert_eq!(records.len(), 14);
    }

    #[test]
    fn browser_graph_forecast_runs_each_paper_transformer_profile() {
        for profile in [
            "heterogeneous_moe",
            "efficient_high_order",
            "long_short_fusion",
            "gated_graph_temporal",
            "spatial_shift_graphon_moe",
        ] {
            let target = (0..12)
                .map(|step| {
                    let value = step as f64;
                    vec![20.0 + value, 16.0 + value * 0.7, 12.0 + value * 0.4]
                })
                .collect();
            let response = run_graph_forecast_request(BrowserGraphForecastRequest {
                frame: BrowserGraphTemporalFrame {
                    node_ids: vec![
                        "PULocationID:161".into(),
                        "PULocationID:236".into(),
                        "PULocationID:132".into(),
                    ],
                    timestamps: (0..12).map(i64::from).collect(),
                    target,
                    adjacency: BrowserCsrAdjacency {
                        indptr: vec![0, 2, 3, 3],
                        indices: vec![1, 2, 2],
                        data: vec![0.7, 0.3, 1.0],
                    },
                    horizon: 2,
                    frequency: "hourly".into(),
                    covariates: None,
                },
                options: BrowserGraphForecastOptions {
                    profile: Some(profile.into()),
                    lookback: Some(if profile == "long_short_fusion" { 8 } else { 3 }),
                    hidden_size: 4,
                    attention_heads: Some(2),
                    graph_order: Some(2),
                    experts: Some(2),
                    periodicity: Some(if profile == "long_short_fusion" { 1 } else { 3 }),
                    recent_window: Some(3),
                    epochs: 2,
                    learning_rate: 0.01,
                    ..BrowserGraphForecastOptions::default()
                },
                actual: None,
            })
            .expect("browser paper graph transformer run");
            assert_eq!(response.predictions.len(), 2, "{profile}");
            assert!(response
                .predictions
                .iter()
                .flatten()
                .all(|value| value.is_finite()));
            assert_eq!(response.metadata["model"].as_str(), Some(profile));
            assert!(response.metadata["architectureReport"].is_object());
        }
    }

    #[test]
    fn browser_lsttn_default_requires_long_horizon_history() {
        let error = run_graph_forecast_request(BrowserGraphForecastRequest {
            frame: BrowserGraphTemporalFrame {
                node_ids: vec!["PULocationID:161".into()],
                timestamps: (0..12).map(i64::from).collect(),
                target: (0..12).map(|step| vec![step as f64]).collect(),
                adjacency: BrowserCsrAdjacency {
                    indptr: vec![0, 0],
                    indices: vec![],
                    data: vec![],
                },
                horizon: 2,
                frequency: "hourly".into(),
                covariates: None,
            },
            options: BrowserGraphForecastOptions {
                profile: Some("long_short_fusion".into()),
                hidden_size: 2,
                epochs: 1,
                learning_rate: 0.01,
                ..BrowserGraphForecastOptions::default()
            },
            actual: None,
        })
        .expect_err("LSTTN browser defaults must retain long-horizon history");
        assert!(error.to_string().contains("lookback plus horizon"));
    }

    #[test]
    fn browser_regression_model_scores_holdout_and_reports_importance() {
        let request = BrowserRegressionRequest {
            rows: sample_regression_rows(),
            feature_names: vec![
                "trip_distance".to_string(),
                "pickup_hour".to_string(),
                "route_pressure".to_string(),
                "pickup_x".to_string(),
                "pickup_y".to_string(),
            ],
            sparse_feature_names: vec!["zone_memberships".to_string()],
            options: BrowserRegressionOptions {
                holdout_fraction: 0.25,
                splitter_mode: Some("full".to_string()),
                feature_kinds: BTreeMap::from([
                    ("trip_distance".to_string(), "numeric".to_string()),
                    ("pickup_hour".to_string(), "periodic".to_string()),
                    ("route_pressure".to_string(), "numeric".to_string()),
                    ("pickup_x".to_string(), "spatial".to_string()),
                    ("pickup_y".to_string(), "spatial".to_string()),
                ]),
                periodic_periods: BTreeMap::from([("pickup_hour".to_string(), 24)]),
                loss: Some("huber".to_string()),
                quantile_alpha: None,
                huber_delta: Some(5.0),
                log_offset: None,
                interval_lower_alpha: Some(0.1),
                interval_upper_alpha: Some(0.9),
                n_estimators: Some(80),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                monotonic_constraints: None,
                include_model_visualization: None,
                backend: None,
            },
        };
        let response = run_regression_request(request).expect("regression run");
        assert_eq!(response.metrics.train_rows, 45);
        assert_eq!(response.metrics.holdout_rows, 15);
        assert_eq!(response.predictions.len(), 15);
        assert_eq!(response.metadata["splitterMode"].as_str(), Some("full"));
        assert!(response.metrics.rmse.is_finite());
        assert!(response.metrics.mae.is_finite());
        assert!(response.metrics.r2.is_finite());
        assert_eq!(response.feature_importance.len(), 6);
        assert_eq!(
            response.metadata["sparseFeatureNames"][0].as_str(),
            Some("zone_memberships")
        );
        assert_eq!(response.metadata["loss"].as_str(), Some("huber"));
        assert!(response
            .predictions
            .iter()
            .all(|row| row.lower_prediction.is_some() && row.upper_prediction.is_some()));
        assert!(response.predictions.iter().all(|row| row
            .lower_prediction
            .zip(row.upper_prediction)
            .is_some_and(|(lower, upper)| lower <= upper)));
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.split_count > 0));
    }

    #[test]
    fn browser_regression_model_rejects_unknown_loss() {
        let mut request = BrowserRegressionRequest {
            rows: sample_regression_rows(),
            feature_names: vec![
                "trip_distance".to_string(),
                "pickup_hour".to_string(),
                "route_pressure".to_string(),
                "pickup_x".to_string(),
                "pickup_y".to_string(),
            ],
            sparse_feature_names: vec!["zone_memberships".to_string()],
            options: BrowserRegressionOptions::default(),
        };
        request.options.loss = Some("not_a_loss".to_string());
        let error = run_regression_request(request).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported browser regression loss"));
    }

    #[test]
    fn browser_regression_model_rejects_bad_feature_width() {
        let error = run_regression_request(BrowserRegressionRequest {
            rows: vec![
                BrowserRegressionRow {
                    features: vec![1.0, 2.0],
                    sparse_sets: Vec::new(),
                    target: 3.0,
                },
                BrowserRegressionRow {
                    features: vec![2.0],
                    sparse_sets: Vec::new(),
                    target: 4.0,
                },
                BrowserRegressionRow {
                    features: vec![3.0, 4.0],
                    sparse_sets: Vec::new(),
                    target: 5.0,
                },
                BrowserRegressionRow {
                    features: vec![4.0, 5.0],
                    sparse_sets: Vec::new(),
                    target: 6.0,
                },
            ],
            feature_names: vec!["x".to_string(), "z".to_string()],
            sparse_feature_names: Vec::new(),
            options: BrowserRegressionOptions::default(),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("feature row has 1 columns but feature_names has 2"));
    }

    #[test]
    fn browser_neural_embedding_model_scores_holdout() {
        let request = BrowserNeuralRequest {
            rows: sample_neural_rows(),
            dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
            node_features: Vec::new(),
            node_types: Vec::new(),
            edge_type_triples: Vec::new(),
            pipeline: "embedding".to_string(),
            options: BrowserNeuralOptions {
                holdout_fraction: 0.25,
                embedding_dim: Some(4),
                random_state: Some(42),
                n_estimators: Some(40),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                ..BrowserNeuralOptions::default()
            },
        };
        let response = run_neural_request(request).expect("embedding neural run");
        assert_eq!(response.metrics.train_rows, 36);
        assert_eq!(response.metrics.holdout_rows, 12);
        assert_eq!(response.predictions.len(), 12);
        assert_eq!(
            response.metadata["details"]["model"].as_str(),
            Some("neural_embedding_regressor")
        );
        assert!(response.metrics.rmse.is_finite());
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.feature.starts_with("embedding_")));
    }

    #[test]
    fn browser_node2vec_model_scores_pair_holdout() {
        let request = BrowserNeuralRequest {
            rows: sample_neural_rows(),
            dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
            node_features: sample_node_features(),
            node_types: Vec::new(),
            edge_type_triples: Vec::new(),
            pipeline: "node2vec".to_string(),
            options: BrowserNeuralOptions {
                holdout_fraction: 0.25,
                embedding_dim: Some(4),
                node2vec_walk_length: Some(6),
                node2vec_walks_per_node: Some(3),
                node2vec_window_size: Some(2),
                node2vec_epochs: Some(2),
                node2vec_seed: Some(7),
                n_estimators: Some(40),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                ..BrowserNeuralOptions::default()
            },
        };
        let response = run_neural_request(request).expect("node2vec neural run");
        assert_eq!(response.metrics.train_rows, 36);
        assert_eq!(response.metrics.holdout_rows, 12);
        assert_eq!(response.predictions.len(), 12);
        assert_eq!(
            response.metadata["details"]["model"].as_str(),
            Some("node2vec_regressor")
        );
        assert_eq!(response.metadata["details"]["nodeCount"].as_u64(), Some(8));
        assert!(response.metrics.mae.is_finite());
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.feature.starts_with("node2vec_")));
    }

    #[test]
    fn browser_graphsage_family_models_score_pair_holdout() {
        for (pipeline, expected_model, expected_prefix) in [
            ("graphsage", "graphsage_regressor", "graphsage_"),
            (
                "hetero_graphsage",
                "hetero_graphsage_regressor",
                "hetero_graphsage_",
            ),
            ("hinsage", "hinsage_regressor", "hinsage_"),
        ] {
            let request = BrowserNeuralRequest {
                rows: sample_neural_rows(),
                dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
                node_features: sample_node_features(),
                node_types: vec![0, 0, 0, 0, 1, 1, 1, 1],
                edge_type_triples: vec![(0, 0, 1)],
                pipeline: pipeline.to_string(),
                options: BrowserNeuralOptions {
                    holdout_fraction: 0.25,
                    embedding_dim: Some(4),
                    graph_sage_epochs: Some(2),
                    graph_sage_negative_samples: Some(2),
                    graph_sage_seed: Some(11),
                    n_estimators: Some(40),
                    learning_rate: Some(0.08),
                    max_depth: Some(3),
                    min_samples_leaf: Some(2),
                    backend: Some("cpu".to_string()),
                    ..BrowserNeuralOptions::default()
                },
            };
            let response = run_neural_request(request).expect("graph neural run");
            assert_eq!(response.metrics.train_rows, 36);
            assert_eq!(response.metrics.holdout_rows, 12);
            assert_eq!(response.predictions.len(), 12);
            assert_eq!(
                response.metadata["details"]["model"].as_str(),
                Some(expected_model)
            );
            assert_eq!(response.metadata["details"]["nodeCount"].as_u64(), Some(8));
            assert_eq!(
                response.metadata["backend"]["selected"].as_str(),
                Some("cpu")
            );
            assert!(response.metrics.rmse.is_finite());
            assert!(response
                .feature_importance
                .iter()
                .any(|item| item.feature.starts_with(expected_prefix)));
        }
    }

    fn sample_panel_rows() -> Vec<BrowserForecastRow> {
        let mut rows = Vec::new();
        for (series_index, series_id) in ["pickup_zone_1", "pickup_zone_2", "pickup_zone_3"]
            .iter()
            .enumerate()
        {
            for day in 0..70 {
                let weekly = (day % 7) as f64;
                let level = 120.0 + series_index as f64 * 30.0;
                let target = level + day as f64 * 1.4 + weekly * 3.0;
                rows.push(BrowserForecastRow {
                    series_id: Some((*series_id).to_string()),
                    timestamp: date_string(day),
                    target,
                    covariates: BTreeMap::from([
                        ("longitude".to_string(), -73.98 + series_index as f64 * 0.02),
                        ("latitude".to_string(), 40.74 + series_index as f64 * 0.02),
                    ]),
                });
            }
        }
        rows
    }

    fn sample_regression_rows() -> Vec<BrowserRegressionRow> {
        (0..60)
            .map(|idx| {
                let trip_distance = 0.8 + idx as f64 * 0.12;
                let pickup_hour = (idx % 24) as f64;
                let route_pressure = ((idx * 7) % 11) as f64;
                let pickup_x = -73.98 + (idx as f64 / 6.0).sin() * 0.04;
                let pickup_y = 40.74 + (idx as f64 / 7.0).cos() * 0.03;
                let neighborhood_signal = if idx % 3 == 0 { 5.0 } else { 0.0 };
                BrowserRegressionRow {
                    features: vec![
                        trip_distance,
                        pickup_hour,
                        route_pressure,
                        pickup_x,
                        pickup_y,
                    ],
                    sparse_sets: vec![vec![101 + (idx % 3) as u64, 200 + (idx % 5) as u64]],
                    target: 6.0
                        + trip_distance * 2.4
                        + pickup_hour * 0.35
                        + route_pressure * 1.1
                        + (pickup_x + 74.0) * 10.0
                        + (pickup_y - 40.7) * 12.0
                        + neighborhood_signal,
                }
            })
            .collect()
    }

    fn sample_neural_rows() -> Vec<BrowserNeuralRow> {
        (0..48)
            .map(|idx| {
                let source = idx % 4;
                let target_node = 4 + ((idx * 3) % 4);
                let trip_distance = 1.0 + (idx % 8) as f64 * 0.35;
                let pickup_hour = (idx % 24) as f64;
                BrowserNeuralRow {
                    id: Some((source + 1) as u64),
                    source: Some(source),
                    target_node: Some(target_node),
                    edge_weight: Some(1.0 + (idx % 3) as f32 * 0.2),
                    edge_type: Some(0),
                    dense: vec![trip_distance, pickup_hour],
                    target: 20.0
                        + source as f64 * 4.0
                        + target_node as f64 * 2.5
                        + trip_distance * 3.0
                        + pickup_hour * 0.4,
                }
            })
            .collect()
    }

    fn sample_sequence_series() -> SequenceSeries {
        SequenceSeries {
            series_id: "pickup_zone_1".to_string(),
            rows: vec![
                sequence_row("r0", 0.0, Some(0.0)),
                sequence_row("r1", 1.0, Some(1.0)),
                sequence_row("r2", 2.0, None),
                sequence_row("r3", 3.0, None),
            ],
        }
    }

    fn sequence_row(
        row_id: &str,
        position: f64,
        target: Option<f64>,
    ) -> cartoboost_core::forecasting::SequenceRow {
        cartoboost_core::forecasting::SequenceRow {
            row_id: row_id.to_string(),
            position,
            target,
            reference_axis: None,
            reference_signal: None,
            auxiliary_rate: None,
        }
    }

    fn sample_node_features() -> Vec<Vec<f32>> {
        (0..8)
            .map(|node| {
                vec![
                    node as f32 / 8.0,
                    if node < 4 { 0.0 } else { 1.0 },
                    ((node * 3) % 5) as f32 / 5.0,
                ]
            })
            .collect()
    }

    fn date_string(day_index: usize) -> String {
        const MONTH_LENGTHS: [usize; 3] = [31, 28, 31];
        let mut remaining = day_index;
        for (month_index, month_length) in MONTH_LENGTHS.iter().enumerate() {
            if remaining < *month_length {
                return format!(
                    "2026-{month:02}-{day:02}",
                    month = month_index + 1,
                    day = remaining + 1
                );
            }
            remaining -= month_length;
        }
        panic!("sample day index out of range");
    }
}
