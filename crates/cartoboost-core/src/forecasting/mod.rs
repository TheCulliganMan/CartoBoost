//! Rust-native forecasting primitives.
//!
//! The Python forecasting API should wrap these types instead of implementing
//! forecasting behavior itself.

mod artifacts;
mod auto;
mod autostats;
mod backtesting;
mod classical_bank;
mod config;
mod conformal;
mod decomposition;
mod diagnostics;
mod direct;
mod ensemble;
mod features;
mod frequency;
mod gating;
mod global;
mod hierarchy;
mod horizon;
mod intermittent;
mod lag_features;
mod lag_plus;
mod local;
mod metrics;
mod mstl;
mod objective;
mod probabilistic;
mod quantiles;
mod rank_probability;
mod reconciliation;
mod registry;
mod residual_correction;
mod result;
mod schema;
mod sequence;
pub(crate) mod splitters;
mod stl;
mod target_transform;
mod temporal_hierarchy;
mod traits;

pub use artifacts::{
    ForecastArtifact, ForecastArtifactManifest, DEFAULT_FORECAST_FILE, DEFAULT_MANIFEST_FILE,
    FORECAST_ARTIFACT_SCHEMA_VERSION,
};
pub use auto::{AutoForecastConfig, AutoForecastModel, AutoForecastObjective};
pub use autostats::AutoStatsBank;
pub use backtesting::{BacktestFoldResult, BacktestResult, RollingOriginBacktester};
pub use classical_bank::{
    ClassicalExpert, ClassicalExpertBank, ClassicalExpertScore, ClassicalExpertValidationObjective,
};
pub use config::{ForecastModelConfig, ForecastingConfig};
pub use conformal::{ConformalCalibrator, ConformalInterval};
pub use decomposition::{MSTLCartoBoostForecaster, STLCartoBoostForecaster};
pub use diagnostics::{
    regime_adjusted_intervals, rolling_mad_residual, rolling_median_residual,
    widen_interval_for_regime, CusumConfig, EwmaVolatility, EwmaVolatilityConfig,
    ForecastDiagnostics, PageHinkley, PageHinkleyConfig, RegimeIntervalAdjustment,
    RegimeIntervalPolicy, RegimeSignal, SeriesDiagnostics, CUSUM,
};
pub use direct::{
    CartoBoostDirectForecaster, DirectForecastStrategy, RectifiedRecursiveForecaster,
};
pub use ensemble::{ForecastEnsemble, GatedEnsembleForecaster, WeightedEnsembleForecaster};
pub use features::{DirectFeatureMatrix, ForecastFeatureFactory};
pub use frequency::{parse_forecast_timestamp, ForecastFrequency};
pub use gating::{
    calendar_profile_candidate_prediction, candidate_complexity_rank,
    forecast_magnitude_guard_allows, include_autostats_candidate, lag_origin_consistency_guard,
    native_auto_raw_candidate_is_confident, relative_loss_displacement_allowed, requires_lag_spine,
    seasonal_naive_candidate_prediction, selectable_candidate_names, shared_candidate_names,
    stable_magnitude_candidate_choice, trend_candidate_prediction, validation_ensemble_weights,
    validation_unavailable_candidate_choice, weighted_blend_candidate_forecast, CandidateSelection,
    CandidateSelectionPolicy, ExpertScore, RuleBasedGating, RuleBasedGatingGuardrails,
    ValidationScoreTable,
};
pub use global::{
    CartoBoostLagForecaster, GlobalForecastSampleWeightMode, GlobalForecastTargetMode,
};
pub use hierarchy::{HierarchyNode, HierarchySpec};
pub use horizon::{ForecastOutput, ForecastRequest, ForecastStrategy, QuantileForecastOutput};
pub use intermittent::{
    adida_forecast, croston_forecast, sba_forecast, tsb_forecast, CrostonForecaster,
    IntermittentDemandConfig, IntermittentDemandForecaster, IntermittentDemandMethod,
    SbaForecaster, TsbForecaster,
};
pub use lag_features::{CalendarFeature, LagFeatureBuilder, LagFeatureConfig, LagFeatureRow};
pub use lag_plus::{LagPlusConfig, LagPlusForecaster};
pub use local::{
    ArimaForecaster, ArimaValidationScore, AutoARIMAForecaster, AutoETSForecaster,
    AutoKalmanForecaster, AutoLocalLevelKalmanForecaster, ETSForecaster, ETSParameterSet,
    ETSValidationScore, KalmanForecaster, KalmanParameterSet, KalmanValidationScore,
    KrigingForecaster, LocalLevelKalmanForecaster, LocalLevelKalmanParameterSet,
    LocalLevelKalmanValidationScore, NaiveForecaster, OptimizedThetaForecaster,
    PiecewiseLinearComponentMode, PiecewiseLinearEvent, PiecewiseLinearFitLoss,
    PiecewiseLinearGrowth, PiecewiseLinearRegressorStandardization, PiecewiseLinearSeasonalConfig,
    PiecewiseLinearSeasonalForecaster, PiecewiseLinearSeasonality,
    PiecewiseLinearTrendUncertaintyPolicy, SeasonalNaiveForecaster,
    SeasonalWindowAverageForecaster, SpatialPiecewiseKrigingConfig,
    SpatialPiecewiseKrigingForecaster, SpatialPiecewiseKrigingMode, ThetaForecaster,
    ThetaSeasonality, WindowAverageForecaster,
};
pub use metrics::{
    competition_mase_scale, evaluate_competition_metrics, evaluate_forecast,
    evaluate_forecast_with_training, CompetitionMetricSet, ForecastActual, ForecastMetricSet,
};
pub use mstl::MSTLDecomposition;
pub use objective::ForecastObjective;
pub use probabilistic::{
    ProbabilisticDirectForecaster, ProbabilisticForecaster, QuantileRegressorSet,
    QuantileRegressorSetConfig,
};
pub use quantiles::{
    crossing_rate, default_quantile_levels, interval_coverage, interval_diagnostics,
    mean_interval_width, pinball_loss, repair_non_crossing_quantiles, IntervalDiagnostics,
    QuantileForecast, DEFAULT_QUANTILE_LEVELS,
};
pub use rank_probability::{rank_probability_score, RankProbabilityForecast};
pub use reconciliation::{proportional_total_reconciliation, Reconciler, ReconciliationMethod};
pub use registry::{ForecastModelSpec, ForecastRegistry, RegisterMode, RegisteredForecastModel};
pub use residual_correction::{
    KalmanResidualCorrector, ResidualStateKey, StateCorrectedBooster, StateCorrection, StateFilter,
    StateObservation, StatePrediction,
};
pub use result::{
    ForecastIntervalPrediction, ForecastPrediction, ForecastPredictionDetail, ForecastResult,
};
pub use schema::{ForecastFrame, ForecastFrameMetadata, ForecastRow, SINGLE_SERIES_ID};
pub use sequence::{
    forward_ekf, generate_group_oof_candidate_rows, missing_target_continuation,
    per_group_error_summary, reference_path_posterior_mean, reference_path_viterbi, rts_smoother,
    ukf_reference, validate_oof_meta_training, KnownPrefix, PredictionMask, ReferencePathConfig,
    ReferencePathPoint, ReferencePathResult, ReferenceSignal, SequenceBlendPrediction,
    SequenceCandidate, SequenceCandidateEnsemble, SequenceCandidatePrediction, SequenceFrame,
    SequenceGroupMetric, SequenceGroupPrediction, SequenceKalmanPoint, SequenceKalmanResult,
    SequenceOofCandidateRow, SequenceOofFold, SequenceRow, SequenceSeries,
    SequenceStateSpaceConfig,
};
pub use splitters::{
    CandidateValidationCutoffSchedule, ForecastFold, ForecastFoldMetadata, ForecastWindow,
    RollingOriginSplitter,
};
pub use stl::STLDecomposition;
pub use target_transform::{
    LocalScaleStats, LocalStandardScaledForecaster, LocalStandardScaler, Log1pForecaster,
};
pub use temporal_hierarchy::{TemporalAggregation, TemporalHierarchy};
pub use traits::Forecaster;
