use cartoboost_core::data::{FeatureSchema, SparseSetColumn};
use cartoboost_core::forecasting::{
    calendar_profile_candidate_prediction as core_calendar_profile_candidate_prediction,
    candidate_complexity_rank as core_candidate_complexity_rank, evaluate_competition_metrics,
    forecast_magnitude_guard_allows,
    include_autostats_candidate as core_include_autostats_candidate,
    lag_origin_consistency_guard as core_lag_origin_consistency_guard,
    missing_target_continuation as core_missing_target_continuation,
    native_auto_raw_candidate_is_confident as core_native_auto_raw_candidate_is_confident,
    parse_forecast_timestamp,
    proportional_total_reconciliation as core_proportional_total_reconciliation,
    reference_path_posterior_mean as core_reference_path_posterior_mean,
    reference_path_viterbi as core_reference_path_viterbi,
    relative_loss_displacement_allowed as core_relative_loss_displacement_allowed,
    requires_lag_spine as core_requires_lag_spine,
    seasonal_naive_candidate_prediction as core_seasonal_naive_candidate_prediction,
    selectable_candidate_names as core_selectable_candidate_names,
    shared_candidate_names as core_shared_candidate_names,
    stable_magnitude_candidate_choice as core_stable_magnitude_candidate_choice,
    trend_candidate_prediction as core_trend_candidate_prediction,
    validation_ensemble_weights as core_validation_ensemble_weights,
    validation_unavailable_candidate_choice as core_validation_unavailable_candidate_choice,
    weighted_blend_candidate_forecast as core_weighted_blend_candidate_forecast,
    ArimaForecaster as CoreArimaForecaster, AutoARIMAForecaster as CoreAutoARIMAForecaster,
    AutoForecastConfig as CoreAutoForecastConfig, AutoForecastModel as CoreAutoForecastModel,
    AutoKalmanForecaster as CoreAutoKalmanForecaster,
    AutoLocalLevelKalmanForecaster as CoreAutoLocalLevelKalmanForecaster,
    AutoStatsBank as CoreAutoStatsBank, BacktestFoldResult as CoreBacktestFoldResult,
    BacktestResult as CoreBacktestResult, CalendarFeature,
    CandidateSelectionPolicy as CoreCandidateSelectionPolicy,
    CandidateValidationCutoffSchedule as CoreCandidateValidationCutoffSchedule,
    CartoBoostLagForecaster as CoreCartoBoostLagForecaster, ClassicalExpertValidationObjective,
    CrostonForecaster as CoreCrostonForecaster, ETSForecaster as CoreETSForecaster, ForecastActual,
    ForecastFold as CoreForecastFold, ForecastFrame as CoreForecastFrame, ForecastFrameMetadata,
    ForecastFrequency, ForecastMetricSet as CoreForecastMetricSet,
    ForecastObjective as CoreForecastObjective, ForecastPrediction,
    ForecastResult as CoreForecastResult, ForecastRow as CoreForecastRow, ForecastWindow,
    Forecaster, GlobalForecastTargetMode, KalmanForecaster as CoreKalmanForecaster,
    KrigingForecaster as CoreKrigingForecaster, LagFeatureConfig,
    LocalLevelKalmanForecaster as CoreLocalLevelKalmanForecaster,
    NaiveForecaster as CoreNaiveForecaster,
    OptimizedThetaForecaster as CoreOptimizedThetaForecaster, PiecewiseLinearComponentMode,
    PiecewiseLinearEvent, PiecewiseLinearFitLoss, PiecewiseLinearGrowth,
    PiecewiseLinearRegressorStandardization,
    PiecewiseLinearSeasonalConfig as CorePiecewiseLinearSeasonalConfig,
    PiecewiseLinearSeasonalForecaster as CorePiecewiseLinearSeasonalForecaster,
    PiecewiseLinearSeasonality, PiecewiseLinearTrendUncertaintyPolicy, ReferencePathConfig,
    ReferenceSignal, RollingOriginBacktester as CoreRollingOriginBacktester,
    RollingOriginSplitter as CoreRollingOriginSplitter, SbaForecaster as CoreSbaForecaster,
    SeasonalNaiveForecaster as CoreSeasonalNaiveForecaster, SequenceCandidate,
    SequenceCandidateEnsemble, SequenceCandidatePrediction, SequenceFrame, SequenceGroupPrediction,
    SequenceOofCandidateRow, SequenceOofFold, SequenceSeries, SequenceStateSpaceConfig,
    SpatialPiecewiseKrigingConfig as CoreSpatialPiecewiseKrigingConfig,
    SpatialPiecewiseKrigingForecaster as CoreSpatialPiecewiseKrigingForecaster,
    SpatialPiecewiseKrigingMode, ThetaForecaster as CoreThetaForecaster, ThetaSeasonality,
    TsbForecaster as CoreTsbForecaster,
    WeightedEnsembleForecaster as CoreWeightedEnsembleForecaster,
};
use cartoboost_core::geo::{
    assemble_route_sparse_rows, assemble_sparse_column, assemble_sparse_row,
    expand_h3_sparse_set as core_expand_h3_sparse_set,
    normalize_coordinate as core_normalize_coordinate, normalize_h3_id_text,
    normalize_h3_resolution, normalize_s2_id_text, normalize_s2_level, scaffold_h3_parent_id,
    validate_equal_row_count, validate_parent_levels, GeoGridKind,
};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::manifest::model_manifest_json as core_model_manifest_json;
use cartoboost_core::metrics::{
    aggregate_equal_level_wrmsse as core_aggregate_equal_level_wrmsse,
    calibrated_rank_bucket_probabilities, extreme_portfolio_decisions,
    ordered_nonnegative_weights as core_ordered_nonnegative_weights, portfolio_summary,
    rank_buckets, rank_hit_rates, rank_portfolio_decision_loss, rank_portfolio_summary,
    rank_probability_calibration, rank_scored_assets, rmsse_scale as core_rmsse_scale,
    wrmsse as core_wrmsse, PortfolioAsset, PortfolioDecision, PortfolioSide, RankBucketPrediction,
    WrmsseSeries,
};
use cartoboost_core::tree::{FlatAxisPredictor, FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::utilities::{
    empirical_variogram, fit_local_level_kalman, fit_local_linear_kalman,
    fit_ordinary_kriging_variogram, intermittent_demand_forecast, local_level_kalman_forecast,
    local_level_kalman_forecast_distribution, local_linear_kalman_forecast,
    local_linear_kalman_forecast_distribution, ordinary_kriging_leave_one_out,
    ordinary_kriging_leave_one_out_diagnostics, ordinary_kriging_predict_many,
    IntermittentDemandMethod, KrigingDrift, KrigingObservation, KrigingVariogramModel,
    LocalLevelKalmanConfig, LocalLinearKalmanConfig, OrdinaryKrigingConfig,
};
use cartoboost_core::{
    Booster, BoosterConfig, CartoBoostError, CategoricalEncoder, CategoricalEncodingConfig,
    ClassificationObjective, Classifier, ClassifierConfig, ClassifierModel, Dataset, Model, Ranker,
    RankerConfig, RankerModel, RankingObjective,
};
use cartoboost_geo_causal::{
    causal_representation_report_json as core_geo_causal_representation_report_json,
    spillover_diagnostics as core_geo_causal_spillover_diagnostics, GeoCausalPanel, GeoCausalRow,
    GeoExperimentDesigner as CoreGeoExperimentDesigner, SpatialPlaceboTester, SpatialWeight,
    SyntheticDIDConfig, SyntheticDIDEstimator as CoreSyntheticDIDEstimator,
};
use cartoboost_geo_core::{
    buffered_spatial_cv as core_buffered_spatial_cv, group_spatial_cv as core_group_spatial_cv,
    rolling_origin_panel_split as core_rolling_origin_panel_split,
    spatial_block_cv as core_spatial_block_cv,
    spatial_temporal_blocked_split as core_spatial_temporal_blocked_split,
    CoordinateMatrix as CoreCoordinateMatrix, GeoFrameMeta as CoreGeoFrameMeta,
    PanelIndex as CorePanelIndex, SpatialWeights as CoreGeoSpatialWeights,
    SplitManifest as CoreSplitManifest, TimeIndex as CoreTimeIndex,
};
use cartoboost_geo_st::{
    available_compute_backends as graph_st_available_compute_backends,
    select_compute_backend as graph_st_select_compute_backend, CsrAdjacency as CoreStCsrAdjacency,
    DcrnnConfig as CoreDcrnnConfig, DcrnnForecaster as CoreDcrnnForecaster,
    DelayAwareGraphConfig as CoreDelayAwareGraphConfig,
    DelayAwareGraphTransformer as CoreDelayAwareGraphTransformer,
    ExpertEventLabel as CoreExpertEventLabel,
    ExpertRelationshipPrior as CoreExpertRelationshipPrior,
    GraphTemporalFrame as CoreGraphTemporalFrame,
    GraphTransformerProfile as CoreGraphTransformerProfile,
    GraphWaveNetConfig as CoreGraphWaveNetConfig,
    GraphWaveNetForecaster as CoreGraphWaveNetForecaster, MarketPanelFrame as CoreMarketPanelFrame,
    MarketStructureConfig as CoreMarketStructureConfig,
    MarketStructureForecaster as CoreMarketStructureForecaster,
    PaperGraphTransformerConfig as CorePaperGraphTransformerConfig,
    PaperGraphTransformerForecaster as CorePaperGraphTransformerForecaster,
    STAEformerConfig as CoreSTAEformerConfig, STAEformerForecaster as CoreSTAEformerForecaster,
};
use cartoboost_geostats::{
    empirical_semivariogram as geostats_empirical_semivariogram,
    fit_variogram_wls as geostats_fit_variogram_wls, Anisotropy as CoreGeostatsAnisotropy,
    CovarianceKernel as CoreCovarianceKernel,
    NearestNeighborGPRegressor as CoreNearestNeighborGPRegressor, NngpConfig as CoreNngpConfig,
};
use cartoboost_neural::{
    available_backends as neural_available_backends,
    backend_dispatch_report as neural_backend_dispatch_report, build_embedding_table_artifact,
    choice_set_transformer_report_json as core_choice_set_transformer_report_json,
    compute_directional_features,
    constrained_decision_select_with_options as core_deep_constrained_decision_select,
    directional_pair_fit as core_deep_directional_pair_fit,
    directional_pair_predict as core_deep_directional_pair_predict,
    directional_pair_predictions as core_deep_directional_pair_predictions,
    event_outcome_fit_with_backend as core_deep_event_outcome_fit,
    event_outcome_predict as core_deep_event_outcome_predict, fit_embedding_table_with_options,
    materialize_source_target_pair_nodes,
    response_curve_fit_with_backend as core_deep_response_curve_fit,
    response_curve_predict as core_deep_response_curve_predict,
    select_backend as neural_select_backend,
    service_residual_fit_with_backend as core_deep_service_residual_fit,
    service_residual_predict as core_deep_service_residual_predict,
    temporal_entity_fit as core_deep_temporal_entity_fit,
    temporal_entity_predict as core_deep_temporal_entity_predict, validate_directed_metapath,
    write_embedding_table_artifact, ArtifactFallbackKind, DeepDirectionalPairArtifact,
    DeepDirectionalPairRow, DeepEventArtifact, DeepResponseArtifact, DeepResponseRow,
    DeepServiceResidualArtifact, DeepServiceResidualRow, DeepTemporalEntityArtifact,
    DirectionalPairFitOptions, EmbeddingTable, GraphSageConfig, GraphSageEncoder,
    GraphSageLinkPredictor, GraphSageRegressor, HeteroGraph, HeteroGraphSageConfig,
    HeteroGraphSageEncoder, HeteroGraphSageLinkPredictor, HeteroGraphSageRegressor,
    HeteroTypedEdge, HinSageConfig, HinSageEncoder, HinSageGraph, HinSageLinkPredictor,
    HinSageRegressor, HomogeneousGraph,
    NeuralEmbeddingRegressor as StandaloneNeuralEmbeddingRegressor, Node2VecConfig,
    Node2VecEncoder, Node2VecLinkPredictor, Node2VecRegressor, StandaloneBoosterConfig,
};
use cartoboost_neural::{
    graph_neural_operator_predict_json as core_graph_neural_operator_predict_json,
    neural_operator_synthetic_benchmark_json as core_neural_operator_synthetic_benchmark_json,
    SpatialOperatorEdge as CoreSpatialOperatorEdge,
};
use cartoboost_neural::{
    ComponentMode as CoreNeuralPanelComponentMode,
    LaneNeuralPanelConfig as CoreLaneNeuralPanelConfig,
    LaneNeuralPanelForecaster as CoreLaneNeuralPanelForecaster, NBeatsConfig as CoreNBeatsConfig,
    NBeatsForecaster as CoreNBeatsForecaster, NHiTSConfig as CoreNHiTSConfig,
    NHiTSForecaster as CoreNHiTSForecaster, NeuralPanelConfig as CoreNeuralPanelConfig,
    NeuralPanelForecaster as CoreNeuralPanelForecaster, NeuralPanelLoss as CoreNeuralPanelLoss,
    NeuralPanelMode as CoreNeuralPanelMode, TrendMode as CoreNeuralPanelTrendMode,
};
use cartoboost_prob::{
    benchmark_calibration_report_fields as core_prob_benchmark_calibration_report_fields,
    conditional_flow_fit_json as core_prob_conditional_flow_fit_json,
    conditional_flow_predict_json as core_prob_conditional_flow_predict_json,
    crps_approximation as core_prob_crps_approximation,
    diffusion_scenario_generate_json as core_prob_diffusion_scenario_generate_json,
    group_conformal_residual_quantiles as core_prob_group_conformal_residual_quantiles,
    interval_coverage as core_prob_interval_coverage,
    mean_interval_width as core_prob_mean_interval_width,
    nearest_calibration_residual_quantiles as core_prob_nearest_calibration_residual_quantiles,
    pinball_loss as core_prob_pinball_loss, pit_bins as core_prob_pit_bins,
    rolling_origin_conformal_residual_quantiles as core_prob_rolling_origin_conformal_residual_quantiles,
    split_conformal_residual_quantile as core_prob_split_conformal_residual_quantile,
    weighted_conformal_residual_quantile as core_prob_weighted_conformal_residual_quantile,
    weighted_interval_score as core_prob_weighted_interval_score,
    DiffusionEdge as CoreDiffusionEdge, SplitOrder as CoreProbSplitOrder,
};
use cartoboost_spatial_econ::{
    spatial_weights_from_coo, SpatialEconError, SpatialModelKind, SpatialRegressionModel,
    SpatialWeights,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyType};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

type StringTypedEdges = Vec<(String, String, String)>;
type PyWrmsseSeries = (String, Vec<f64>, Vec<f64>, Vec<f64>, f64);
type CustomSeasonalitySpec = (String, f64, usize, Option<String>);
type PyPortfolioDecisionRow = (String, String, f64, f64, f64);
type PyKrigingPrediction = (f64, f64, f64, Vec<f64>);
type PyDetailedKrigingPrediction = (f64, f64, f64, f64, Vec<f64>, Vec<usize>);
type PyNngpPrediction = (Vec<f64>, Vec<f64>, Vec<Vec<usize>>);
type PyPiecewiseEvent = (String, String, Option<i32>, Option<i32>);
type PyPiecewiseSeasonality = (
    String,
    f64,
    usize,
    Option<String>,
    Option<String>,
    Option<f64>,
);
type PyGeoCausalRow = (
    String,
    String,
    f64,
    bool,
    BTreeMap<String, f64>,
    Option<f64>,
    Option<f64>,
    Option<String>,
);

// Binding families share this crate-level namespace so PyO3 registration stays explicit.
include!("bindings/geo.rs");
include!("bindings/forecasting.rs");
include!("bindings/estimators.rs");
include!("bindings/neural.rs");
include!("bindings/functions.rs");

#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(model_manifest_json, m)?)?;
    m.add_class::<NativeCartoBoostRegressor>()?;
    m.add_class::<NativeNearestNeighborGPRegressor>()?;
    m.add_class::<NativeCartoBoostClassifier>()?;
    m.add_class::<NativeCartoBoostRanker>()?;
    m.add_class::<NativeCoordinateMatrix>()?;
    m.add_class::<NativeTimeIndex>()?;
    m.add_class::<NativePanelIndex>()?;
    m.add_class::<NativeGeoSpatialWeights>()?;
    m.add_class::<NativeSplitManifest>()?;
    m.add_class::<NativeSpatialWeights>()?;
    m.add_class::<NativeSpatialLagRegressor>()?;
    m.add_class::<NativeSpatialErrorRegressor>()?;
    m.add_class::<NativeSpatialDurbinRegressor>()?;
    m.add_class::<NativeSpatialTwoStageLeastSquares>()?;
    m.add_function(wrap_pyfunction!(categorical_fit_transform, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_transform, m)?)?;
    m.add_function(wrap_pyfunction!(validate_feature_schema_json, m)?)?;
    m.add_function(wrap_pyfunction!(geo_spatial_block_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geo_buffered_spatial_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geo_group_spatial_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geo_rolling_origin_panel_split, m)?)?;
    m.add_function(wrap_pyfunction!(geo_spatial_temporal_blocked_split, m)?)?;
    m.add_class::<NativeForecastFrame>()?;
    m.add_class::<NativeForecastResult>()?;
    m.add_class::<NativeForecastFold>()?;
    m.add_class::<NativeRollingOriginSplitter>()?;
    m.add_class::<NativeForecastMetricSet>()?;
    m.add_class::<NativeBacktestFoldResult>()?;
    m.add_class::<NativeBacktestResult>()?;
    m.add_class::<NativeRollingOriginBacktester>()?;
    m.add_class::<NativeNaiveForecaster>()?;
    m.add_class::<NativeSeasonalNaiveForecaster>()?;
    m.add_class::<NativeThetaForecaster>()?;
    m.add_class::<NativeOptimizedThetaForecaster>()?;
    m.add_class::<NativePiecewiseLinearSeasonalForecaster>()?;
    m.add_class::<NativeETSForecaster>()?;
    m.add_class::<NativeArimaForecaster>()?;
    m.add_class::<NativeAutoARIMAForecaster>()?;
    m.add_class::<NativeAutoStatsBank>()?;
    m.add_class::<NativeCrostonForecaster>()?;
    m.add_class::<NativeSbaForecaster>()?;
    m.add_class::<NativeTsbForecaster>()?;
    m.add_class::<NativeKalmanForecaster>()?;
    m.add_class::<NativeLocalLevelKalmanForecaster>()?;
    m.add_class::<NativeAutoKalmanForecaster>()?;
    m.add_class::<NativeAutoLocalLevelKalmanForecaster>()?;
    m.add_class::<NativeKrigingForecaster>()?;
    m.add_class::<NativeSpatialPiecewiseKrigingForecaster>()?;
    m.add_class::<NativeGraphTemporalFrame>()?;
    m.add_class::<NativeMarketPanelFrame>()?;
    m.add_class::<NativeMarketStructureForecaster>()?;
    m.add_class::<NativeDcrnnForecaster>()?;
    m.add_class::<NativeSTAEformerForecaster>()?;
    m.add_class::<NativeGraphWaveNetForecaster>()?;
    m.add_class::<NativePropagationDelayGraphForecaster>()?;
    m.add_class::<NativePaperGraphTransformerForecaster>()?;
    m.add_class::<NativeNBeatsForecaster>()?;
    m.add_class::<NativeNHiTSForecaster>()?;
    m.add_class::<NativeNeuralPanelForecaster>()?;
    m.add_class::<NativeLaneNeuralPanelForecaster>()?;
    m.add_class::<NativeAutoForecastModel>()?;
    m.add_class::<NativeCartoBoostLagForecaster>()?;
    m.add_class::<NativeWeightedEnsembleForecaster>()?;
    m.add_class::<NativeNeuralEmbeddingFeatures>()?;
    m.add_class::<NativeGraphSageEncoder>()?;
    m.add_class::<NativeNode2VecEncoder>()?;
    m.add_class::<NativeStandaloneNeuralEmbeddingRegressor>()?;
    m.add_class::<NativeStandaloneNode2VecRegressor>()?;
    m.add_class::<NativeStandaloneGraphSageRegressor>()?;
    m.add_class::<NativeStandaloneHeteroGraphSageRegressor>()?;
    m.add_class::<NativeStandaloneHinSageRegressor>()?;
    m.add_class::<NativeStandaloneNode2VecLinkPredictor>()?;
    m.add_class::<NativeStandaloneGraphSageLinkPredictor>()?;
    m.add_class::<NativeStandaloneHeteroGraphSageLinkPredictor>()?;
    m.add_class::<NativeStandaloneHinSageLinkPredictor>()?;
    m.add_class::<NativeHeteroGraphSageEncoder>()?;
    m.add_class::<NativeHinSageEncoder>()?;
    m.add_function(wrap_pyfunction!(utility_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(utility_local_level_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(utility_intermittent_demand_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(utility_ordinary_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(
        utility_ordinary_kriging_predict_detailed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utility_ordinary_kriging_leave_one_out, m)?)?;
    m.add_function(wrap_pyfunction!(utility_empirical_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(utility_fit_ordinary_kriging_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(
        utility_ordinary_kriging_leave_one_out_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utility_series_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(graph_compute_directional_features, m)?)?;
    m.add_function(wrap_pyfunction!(graph_validate_directed_metapath, m)?)?;
    m.add_function(wrap_pyfunction!(
        graph_materialize_source_target_pair_nodes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rmsse_scale_value, m)?)?;
    m.add_function(wrap_pyfunction!(wrmsse_value, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_equal_level_wrmsse_value, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_nonnegative_weights_value, m)?)?;
    m.add_function(wrap_pyfunction!(competition_forecast_metrics_value, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_candidate_choice_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_validation_unavailable_candidate_choice_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_candidate_validation_cutoff_indices_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forecast_magnitude_guard_allows_value, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_requires_lag_spine_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_seasonal_naive_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forecast_trend_candidate_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_calendar_profile_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_validation_ensemble_weights_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forecast_shared_candidate_names_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_selectable_candidate_names_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_include_autostats_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_candidate_complexity_rank_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_native_auto_raw_candidate_is_confident_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_lag_origin_consistency_guard_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_relative_loss_displacement_allowed_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_stable_magnitude_candidate_choice_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_proportional_total_reconciliation_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_weighted_blend_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(prob_pinball_loss_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_interval_coverage_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_mean_interval_width_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_crps_approximation_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_weighted_interval_score_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_pit_bins_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_conditional_flow_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_conditional_flow_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_diffusion_scenario_generate_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        prob_split_conformal_residual_quantile_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_weighted_conformal_residual_quantile_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_group_conformal_residual_quantiles_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_rolling_origin_conformal_residual_quantiles_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_nearest_conformal_residual_quantiles_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_benchmark_calibration_report_fields_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(portfolio_summary_value, m)?)?;
    m.add_function(wrap_pyfunction!(extreme_portfolio_decisions_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_hit_rates_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_buckets_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_scored_assets_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_portfolio_summary_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_portfolio_decision_loss_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_probability_calibration_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        calibrated_rank_bucket_probabilities_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sequence_validate_value, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_state_space_value, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_reference_path_viterbi_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        sequence_reference_path_posterior_mean_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sequence_blend_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        sequence_validate_oof_meta_training_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sequence_generate_group_oof_candidate_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sequence_group_error_summary_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_normalize_id_text, m)?)?;
    m.add_function(wrap_pyfunction!(s2_normalize_id_text, m)?)?;
    m.add_function(wrap_pyfunction!(h3_normalize_resolution_value, m)?)?;
    m.add_function(wrap_pyfunction!(s2_normalize_level_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_normalize_coordinate_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        geo_clockwise_bearing_unit_vector_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        geo_initial_bearing_unit_vector_latlng_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(geo_route_feature_vector_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_radial_anchor_distances_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_rbf_anchor_features_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_local_frame_features_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_validate_parent_resolutions_value, m)?)?;
    m.add_function(wrap_pyfunction!(s2_validate_parent_levels_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_scaffold_parent_id_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_expand_sparse_set_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_sparse_row_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_sparse_column_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_route_sparse_rows_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_validate_equal_row_count_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_synthetic_did_summary, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_design_summary, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_spatial_placebos, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_spillover_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_representation_report_value, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_overlay, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_parse_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_evaluate_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(graph_st_available_backends_value, m)?)?;
    m.add_function(wrap_pyfunction!(geostats_empirical_semivariogram_value, m)?)?;
    m.add_function(wrap_pyfunction!(geostats_fit_variogram_wls_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_event_outcome_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_event_outcome_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_directional_pair_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_directional_pair_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_directional_pair_predict_artifact_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_available_backends_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_backend_dispatch_report_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_constrained_decision_select_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_choice_set_transformer_report_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_graph_neural_operator_predict_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deep_neural_operator_synthetic_benchmark_value,
        m
    )?)?;
    Ok(())
}
