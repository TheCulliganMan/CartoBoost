#![allow(dead_code)]

use crate::forecasting::{
    ForecastFrame, ForecastFrequency, ForecastIntervalPrediction, ForecastPrediction,
    ForecastPredictionDetail, ForecastResult, ForecastRow, Forecaster,
};
use crate::loss::huber_irls_weights;
use crate::utilities::{
    fit_local_level_kalman, fit_local_linear_kalman, local_level_kalman_forecast_distribution,
    local_linear_kalman_forecast_distribution, ordinary_kriging_leave_one_out_with_backend,
    ordinary_kriging_predict_many_with_backend, KrigingObservation, LocalLevelKalmanConfig,
    LocalLinearKalmanConfig, OrdinaryKrigingConfig,
};
use crate::{CartoBoostError, Result};
use cartoboost_accelerator::{select_backend_for, BackendOperation, BackendSelection};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const MAX_ARIMA_ORDER: usize = 8;
const MAX_ARIMA_COLUMNS: usize = MAX_ARIMA_ORDER * 2 + 1;
const PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND: &str = "cartoboost_piecewise_linear_seasonal";
const PIECEWISE_LINEAR_SEASONAL_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NaiveForecaster {
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalNaiveForecaster {
    season_length: usize,
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone)]
pub struct WindowAverageForecaster {
    window_size: usize,
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone)]
pub struct SeasonalWindowAverageForecaster {
    season_length: usize,
    window_count: usize,
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone)]
pub struct ThetaForecaster {
    theta: f64,
    alpha: f64,
    seasonality: Option<ThetaSeasonality>,
    fitted: Option<FittedThetaState>,
}

#[derive(Debug, Clone)]
pub struct OptimizedThetaForecaster {
    theta_grid: Vec<f64>,
    alpha_grid: Vec<f64>,
    seasonality: Option<ThetaSeasonality>,
    selected_theta: Option<f64>,
    selected_alpha: Option<f64>,
    validation_window: Option<usize>,
    validation_scores: Vec<ThetaValidationScore>,
    fitted: Option<ThetaForecaster>,
}

#[derive(Debug, Clone)]
pub struct ETSForecaster {
    alpha: f64,
    beta: f64,
    gamma: Option<f64>,
    season_length: Option<usize>,
    damping_phi: f64,
    fitted: Option<FittedETSState>,
}

#[derive(Debug, Clone)]
pub struct AutoETSForecaster {
    alpha_grid: Vec<f64>,
    beta_grid: Vec<f64>,
    gamma_grid: Vec<Option<f64>>,
    damping_phi_grid: Vec<f64>,
    season_length: Option<usize>,
    selected_params: Option<ETSParameterSet>,
    validation_window: Option<usize>,
    validation_scores: Vec<ETSValidationScore>,
    fitted: Option<ETSForecaster>,
}

#[derive(Debug, Clone)]
pub struct ArimaForecaster {
    p: usize,
    d: usize,
    q: usize,
    fitted: Option<FittedArimaState>,
}

#[derive(Debug, Clone)]
pub struct AutoARIMAForecaster {
    max_p: usize,
    max_d: usize,
    max_q: usize,
    selected_order: Option<(usize, usize, usize)>,
    validation_scores: Vec<ArimaValidationScore>,
    fitted: Option<ArimaForecaster>,
}

#[derive(Debug, Clone)]
pub struct KalmanForecaster {
    level_process_variance: f64,
    trend_process_variance: f64,
    observation_variance: f64,
    fitted: Option<FittedKalmanState>,
}

#[derive(Debug, Clone)]
pub struct LocalLevelKalmanForecaster {
    level_process_variance: f64,
    observation_variance: f64,
    fitted: Option<FittedLocalLevelKalmanState>,
}

#[derive(Debug, Clone)]
pub struct AutoKalmanForecaster {
    level_process_variance_grid: Vec<f64>,
    trend_process_variance_grid: Vec<f64>,
    observation_variance_grid: Vec<f64>,
    validation_window: Option<usize>,
    selected_params: Option<KalmanParameterSet>,
    validation_scores: Vec<KalmanValidationScore>,
    fitted: Option<KalmanForecaster>,
}

#[derive(Debug, Clone)]
pub struct AutoLocalLevelKalmanForecaster {
    level_process_variance_grid: Vec<f64>,
    observation_variance_grid: Vec<f64>,
    validation_window: Option<usize>,
    selected_params: Option<LocalLevelKalmanParameterSet>,
    validation_scores: Vec<LocalLevelKalmanValidationScore>,
    fitted: Option<LocalLevelKalmanForecaster>,
}

#[derive(Debug, Clone)]
pub struct KrigingForecaster {
    coordinates: BTreeMap<String, (f64, f64)>,
    config: OrdinaryKrigingConfig,
    backend: BackendSelection,
    fitted: Option<FittedKrigingState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPiecewiseKrigingForecaster {
    config: SpatialPiecewiseKrigingConfig,
    backend: BackendSelection,
    fitted: Option<FittedSpatialPiecewiseKrigingState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialPiecewiseKrigingConfig {
    pub coordinates: BTreeMap<String, (f64, f64)>,
    pub mode: SpatialPiecewiseKrigingMode,
    pub piecewise_config: PiecewiseLinearSeasonalConfig,
    pub kriging_config: OrdinaryKrigingConfig,
    pub spatial_regressors: Vec<String>,
    pub residual_shrinkage: f64,
    pub allow_neighbor_fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialPiecewiseKrigingMode {
    KrigedRegressors,
    ResidualKriging,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseLinearSeasonalForecaster {
    config: PiecewiseLinearSeasonalConfig,
    fitted: Option<FittedPiecewiseLinearSeasonalState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiecewiseLinearSeasonalConfig {
    pub growth: PiecewiseLinearGrowth,
    pub component_mode: PiecewiseLinearComponentMode,
    pub fit_loss: PiecewiseLinearFitLoss,
    pub huber_delta: f64,
    pub irls_iterations: usize,
    pub changepoints: usize,
    pub changepoint_range: f64,
    pub changepoint_timestamps: Vec<chrono::NaiveDateTime>,
    pub yearly_fourier_order: usize,
    pub weekly_fourier_order: usize,
    pub daily_fourier_order: usize,
    pub auto_yearly_seasonality: bool,
    pub auto_weekly_seasonality: bool,
    pub auto_daily_seasonality: bool,
    pub custom_seasonalities: Vec<PiecewiseLinearSeasonality>,
    pub changepoint_l2_regularization: f64,
    pub changepoint_l1_regularization: f64,
    pub seasonality_l2_regularization: f64,
    pub yearly_l2_regularization: Option<f64>,
    pub weekly_l2_regularization: Option<f64>,
    pub daily_l2_regularization: Option<f64>,
    pub event_l2_regularization: f64,
    pub regressor_l2_regularization: f64,
    pub event_l2_regularization_by_name: BTreeMap<String, f64>,
    pub regressor_l2_regularization_by_name: BTreeMap<String, f64>,
    pub events: Vec<PiecewiseLinearEvent>,
    pub event_mode: Option<PiecewiseLinearComponentMode>,
    pub extra_regressors: Vec<String>,
    pub regressor_modes: BTreeMap<String, PiecewiseLinearComponentMode>,
    pub extra_regressor_monotonic_constraints: BTreeMap<String, i8>,
    pub regressor_standardization: PiecewiseLinearRegressorStandardization,
    pub future_regressors: BTreeMap<String, Vec<f64>>,
    pub future_regressors_by_series: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
    pub trend_adjustments: BTreeMap<usize, f64>,
    pub trend_adjustments_by_series: BTreeMap<String, BTreeMap<usize, f64>>,
    pub residual_shock_window: usize,
    pub residual_shock_scale: f64,
    pub residual_shock_decay: f64,
    pub interval_levels: Vec<f64>,
    pub quantile_levels: Vec<f64>,
    pub uncertainty_samples: usize,
    pub trend_uncertainty_policy: PiecewiseLinearTrendUncertaintyPolicy,
    pub trend_uncertainty_scale: f64,
    pub coefficient_uncertainty_scale: f64,
    pub uncertainty_seed: u64,
    pub cap: Option<f64>,
    pub floor: f64,
    pub cap_regressor: Option<String>,
    pub floor_regressor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiecewiseLinearEvent {
    pub name: String,
    pub timestamp: chrono::NaiveDateTime,
    pub lower_window: i32,
    pub upper_window: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiecewiseLinearSeasonality {
    pub name: String,
    pub period_days: f64,
    pub fourier_order: usize,
    pub mode: Option<PiecewiseLinearComponentMode>,
    pub condition_name: Option<String>,
    pub l2_regularization: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiecewiseLinearGrowth {
    Linear,
    Flat,
    Logistic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiecewiseLinearComponentMode {
    Additive,
    Multiplicative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiecewiseLinearFitLoss {
    Squared,
    Huber,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiecewiseLinearRegressorStandardization {
    None,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiecewiseLinearTrendUncertaintyPolicy {
    Normal,
    Laplace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaSeasonalityKind {
    Additive,
    Multiplicative,
}

#[derive(Debug, Clone, Copy)]
pub struct ThetaSeasonality {
    kind: ThetaSeasonalityKind,
    season_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedLocalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    anchor_timestamp_by_series: BTreeMap<String, chrono::NaiveDateTime>,
}

#[derive(Debug, Clone)]
struct FittedThetaState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedThetaSeries>,
}

#[derive(Debug, Clone)]
struct FittedETSState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedETSSeries>,
}

#[derive(Debug, Clone)]
struct FittedETSSeries {
    last_timestamp: chrono::NaiveDateTime,
    n_obs: usize,
    level: f64,
    trend: f64,
    damping_phi: f64,
    seasonals: Option<Vec<f64>>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
    level_values: Vec<f64>,
    trend_values: Vec<f64>,
    seasonal_values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FittedArimaState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedArimaSeries>,
}

#[derive(Debug, Clone)]
struct FittedArimaSeries {
    last_timestamp: chrono::NaiveDateTime,
    intercept: f64,
    ar_coefficients: Vec<f64>,
    ma_coefficients: Vec<f64>,
    score_start: usize,
    differenced_history: Vec<f64>,
    residual_history: Vec<f64>,
    last_differences: Vec<f64>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
}

type ArimaComponents = (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(Debug, Clone)]
struct FittedKalmanState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedKalmanSeries>,
}

#[derive(Debug, Clone)]
struct FittedLocalLevelKalmanState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedLocalLevelKalmanSeries>,
}

#[derive(Debug, Clone)]
struct FittedKalmanSeries {
    last_timestamp: chrono::NaiveDateTime,
    level: f64,
    trend: f64,
}

#[derive(Debug, Clone)]
struct FittedLocalLevelKalmanSeries {
    last_timestamp: chrono::NaiveDateTime,
    level: f64,
}

#[derive(Debug, Clone)]
struct FittedKrigingState {
    frame: ForecastFrame,
    levels: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedSpatialPiecewiseKrigingState {
    frame: ForecastFrame,
    base: PiecewiseLinearSeasonalForecaster,
    residual_levels: BTreeMap<String, f64>,
    residual_observation_series: Vec<String>,
    cutoff_timestamps: BTreeMap<String, chrono::NaiveDateTime>,
    fit_metadata: Value,
}

#[derive(Debug, Clone)]
struct SpatialKrigingCorrection {
    prediction: crate::utilities::KrigingPrediction,
    used_neighbor_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedPiecewiseLinearSeasonalState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedPiecewiseLinearSeasonalSeries>,
    #[serde(default)]
    history_frame: Option<ForecastFrame>,
    #[serde(default)]
    anchor_timestamp_by_series: BTreeMap<String, chrono::NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedPiecewiseLinearSeasonalSeries {
    start_timestamp: chrono::NaiveDateTime,
    last_timestamp: chrono::NaiveDateTime,
    last_elapsed_days: f64,
    changepoints: Vec<f64>,
    coefficients: Vec<f64>,
    coefficient_covariance: Vec<Vec<f64>>,
    feature_count: usize,
    residuals: Vec<f64>,
    transformed_residual_scale: f64,
    trend_delta_scale: f64,
    regressor_stats: BTreeMap<String, PiecewiseLinearRegressorStats>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct PiecewiseLinearRegressorStats {
    mean: f64,
    scale: f64,
    standardized: bool,
}

struct PiecewiseLinearFeatureContext<'a> {
    series_id: Option<&'a str>,
    timestamp: chrono::NaiveDateTime,
    covariates: Option<&'a BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    component_multiplier: f64,
    changepoints: &'a [f64],
    config: &'a PiecewiseLinearSeasonalConfig,
    regressor_stats: Option<&'a BTreeMap<String, PiecewiseLinearRegressorStats>>,
}

struct PiecewiseLinearFitResult {
    coefficients: Vec<f64>,
    coefficient_covariance: Vec<Vec<f64>>,
}

struct PiecewisePredictionTerms {
    mean: f64,
    linear_predictor: f64,
    coefficient_scale: f64,
    linear_coefficient_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PiecewiseEventTerm {
    name: String,
    offset: i32,
}

// Local forecasting families share this module namespace.
include!("local/configuration.rs");
include!("local/forecasters.rs");
include!("local/fitted.rs");
include!("local/algorithms.rs");
include!("local/tests.rs");
