#![allow(dead_code)]

use crate::forecasting::{
    ForecastFrame, ForecastFrequency, ForecastIntervalPrediction, ForecastPrediction,
    ForecastPredictionDetail, ForecastResult, ForecastRow, Forecaster,
};
use crate::loss::huber_irls_weights;
use crate::utilities::{
    fit_local_level_kalman, fit_local_linear_kalman, ordinary_kriging_predict,
    ordinary_kriging_predict_many, KrigingObservation, LocalLevelKalmanConfig,
    LocalLinearKalmanConfig, OrdinaryKrigingConfig,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const MAX_ARIMA_ORDER: usize = 8;
const MAX_ARIMA_COLUMNS: usize = MAX_ARIMA_ORDER * 2 + 1;
const PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND: &str = "cartoboost_piecewise_linear_seasonal";
const PIECEWISE_LINEAR_SEASONAL_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Default)]
pub struct NaiveForecaster {
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone)]
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
    fitted: Option<FittedKrigingState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPiecewiseKrigingForecaster {
    config: SpatialPiecewiseKrigingConfig,
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

#[derive(Debug, Clone)]
struct FittedLocalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
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

impl PiecewiseEventTerm {
    fn label(&self) -> String {
        format!("{}[{:+}]", self.name, self.offset)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PiecewiseLinearSeasonalArtifact {
    kind: String,
    schema_version: u32,
    model: PiecewiseLinearSeasonalForecaster,
}

#[derive(Debug, Clone)]
struct FittedThetaSeries {
    last_timestamp: chrono::NaiveDateTime,
    n_obs: usize,
    component: ThetaComponent,
    seasonal_pattern: Option<Vec<f64>>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ThetaComponent {
    last_level: f64,
    slope: f64,
    theta: f64,
}

#[derive(Debug, Clone)]
pub struct ThetaValidationScore {
    pub theta: f64,
    pub alpha: f64,
    pub mse: f64,
}

#[derive(Debug, Clone)]
pub struct ArimaValidationScore {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub mse: f64,
    pub stable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ETSParameterSet {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: Option<f64>,
    pub damping_phi: f64,
}

#[derive(Debug, Clone)]
pub struct ETSValidationScore {
    pub params: ETSParameterSet,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanParameterSet {
    pub level_process_variance: f64,
    pub trend_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone)]
pub struct KalmanValidationScore {
    pub params: KalmanParameterSet,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelKalmanParameterSet {
    pub level_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone)]
pub struct LocalLevelKalmanValidationScore {
    pub params: LocalLevelKalmanParameterSet,
    pub mse: f64,
}

impl Default for PiecewiseLinearSeasonalConfig {
    fn default() -> Self {
        Self {
            growth: PiecewiseLinearGrowth::Linear,
            component_mode: PiecewiseLinearComponentMode::Additive,
            fit_loss: PiecewiseLinearFitLoss::Squared,
            huber_delta: 1.345,
            irls_iterations: 5,
            changepoints: 25,
            changepoint_range: 1.0,
            changepoint_timestamps: Vec::new(),
            yearly_fourier_order: 0,
            weekly_fourier_order: 3,
            daily_fourier_order: 0,
            auto_yearly_seasonality: true,
            auto_weekly_seasonality: true,
            auto_daily_seasonality: true,
            custom_seasonalities: Vec::new(),
            changepoint_l2_regularization: 0.05,
            changepoint_l1_regularization: 0.0,
            seasonality_l2_regularization: 0.01,
            yearly_l2_regularization: None,
            weekly_l2_regularization: None,
            daily_l2_regularization: None,
            event_l2_regularization: 0.01,
            regressor_l2_regularization: 0.01,
            event_l2_regularization_by_name: BTreeMap::new(),
            regressor_l2_regularization_by_name: BTreeMap::new(),
            events: Vec::new(),
            event_mode: None,
            extra_regressors: Vec::new(),
            regressor_modes: BTreeMap::new(),
            extra_regressor_monotonic_constraints: BTreeMap::new(),
            regressor_standardization: PiecewiseLinearRegressorStandardization::Auto,
            future_regressors: BTreeMap::new(),
            future_regressors_by_series: BTreeMap::new(),
            trend_adjustments: BTreeMap::new(),
            trend_adjustments_by_series: BTreeMap::new(),
            residual_shock_window: 0,
            residual_shock_scale: 0.0,
            residual_shock_decay: 1.0,
            interval_levels: Vec::new(),
            quantile_levels: Vec::new(),
            uncertainty_samples: 0,
            trend_uncertainty_policy: PiecewiseLinearTrendUncertaintyPolicy::Laplace,
            trend_uncertainty_scale: 1.0,
            coefficient_uncertainty_scale: 1.0,
            uncertainty_seed: 0xC4B0_0575_A11C_E123,
            cap: None,
            floor: 0.0,
            cap_regressor: None,
            floor_regressor: None,
        }
    }
}

impl NaiveForecaster {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PiecewiseLinearSeasonalForecaster {
    pub fn new(config: PiecewiseLinearSeasonalConfig) -> Result<Self> {
        validate_piecewise_linear_seasonal_config(&config)?;
        Ok(Self {
            config,
            fitted: None,
        })
    }

    pub fn config(&self) -> &PiecewiseLinearSeasonalConfig {
        &self.config
    }

    pub fn update_config<F>(&mut self, update: F) -> Result<()>
    where
        F: FnOnce(&mut PiecewiseLinearSeasonalConfig),
    {
        let mut config = self.config.clone();
        update(&mut config);
        validate_piecewise_linear_seasonal_config(&config)?;
        self.config = config;
        Ok(())
    }

    pub fn to_json_string(&self) -> Result<String> {
        let artifact = PiecewiseLinearSeasonalArtifact {
            kind: PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND.to_string(),
            schema_version: PIECEWISE_LINEAR_SEASONAL_ARTIFACT_VERSION,
            model: self.clone(),
        };
        serde_json::to_string(&artifact).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal artifact: {err}"
            ))
        })
    }

    pub fn from_json_string(payload: &str) -> Result<Self> {
        let artifact =
            serde_json::from_str::<PiecewiseLinearSeasonalArtifact>(payload).map_err(|err| {
                CartoBoostError::InvalidInput(format!(
                    "failed to parse piecewise linear seasonal artifact: {err}"
                ))
            })?;
        if artifact.kind != PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported piecewise linear seasonal artifact kind {:?}",
                artifact.kind
            )));
        }
        if artifact.schema_version != PIECEWISE_LINEAR_SEASONAL_ARTIFACT_VERSION {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported piecewise linear seasonal artifact schema_version {}",
                artifact.schema_version
            )));
        }
        validate_piecewise_linear_seasonal_config(&artifact.model.config)?;
        Ok(artifact.model)
    }

    pub fn fitted_series_ids(&self) -> Result<Vec<String>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.series.keys().cloned().collect())
    }

    pub fn predict_components_json_value(&self, horizon: usize) -> Result<Value> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let records = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        let timestamp = fitted
                            .frame
                            .frequency()
                            .advance(series.last_timestamp, step)?;
                        let elapsed = elapsed_days(series.start_timestamp, timestamp);
                        series.predict_component_record(
                            series_id,
                            elapsed,
                            timestamp,
                            step,
                            &self.config,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(json!({
            "model": self.model_name(),
            "columns": [
                "series_id",
                "timestamp",
                "horizon",
                "prediction",
                "trend",
                "adjusted_trend",
                "trend_adjustment_multiplier",
                "trend_adjustment",
                "residual_shock",
                "linear_predictor",
                "components",
            ],
            "records": records,
        }))
    }

    pub fn history_components_json_value(&self) -> Result<Value> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut history_by_series: BTreeMap<String, Vec<&ForecastRow>> = BTreeMap::new();
        for row in fitted.frame.rows() {
            history_by_series
                .entry(row.series_id.clone())
                .or_default()
                .push(row);
        }
        let records = history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let series = fitted.series.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing fitted piecewise linear seasonal state for series {series_id:?}"
                    ))
                })?;
                series.history_component_records(series_id, history, &self.config)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(json!({
            "model": self.model_name(),
            "columns": [
                "series_id",
                "timestamp",
                "index",
                "actual",
                "fitted",
                "residual",
                "trend",
                "adjusted_trend",
                "trend_adjustment_multiplier",
                "trend_adjustment",
                "trend_movement",
                "fitted_movement",
                "linear_predictor",
                "components",
            ],
            "records": records,
        }))
    }

    pub fn history_components_json_string(&self) -> Result<String> {
        serde_json::to_string(&self.history_components_json_value()?).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal history components: {err}"
            ))
        })
    }

    pub fn predict_components_json_string(&self, horizon: usize) -> Result<String> {
        serde_json::to_string(&self.predict_components_json_value(horizon)?).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal components: {err}"
            ))
        })
    }

    pub fn predict_samples_json_value(&self, horizon: usize) -> Result<Value> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let sample_count = self.config.uncertainty_samples;
        let records = if sample_count == 0 {
            Vec::new()
        } else {
            fitted
                .series
                .iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(series_id, series)| {
                    let residual_scale = series.residual_scale();
                    (1..=horizon)
                        .map(|step| {
                            let timestamp = fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?;
                            let elapsed = elapsed_days(series.start_timestamp, timestamp);
                            let bounds =
                                piecewise_bounds(Some(series_id), None, Some(step), &self.config)?;
                            let terms = series.prediction_terms_at(
                                series_id,
                                elapsed,
                                timestamp,
                                step,
                                bounds,
                                &self.config,
                            )?;
                            let trend_offsets = series.trend_uncertainty_offsets(
                                series_id,
                                elapsed,
                                timestamp,
                                step,
                                &self.config,
                            )?;
                            let linear_trend_offsets = series.trend_uncertainty_linear_offsets(
                                series_id,
                                elapsed,
                                step,
                                &self.config,
                            );
                            Ok((0..sample_count)
                                .map(|sample| {
                                    series.predictive_sample_record(
                                        series_id,
                                        timestamp,
                                        step,
                                        sample,
                                        terms.mean,
                                        terms.linear_predictor,
                                        bounds,
                                        residual_scale,
                                        terms.coefficient_scale,
                                        terms.linear_coefficient_scale,
                                        trend_offsets.get(sample).copied().unwrap_or(0.0),
                                        linear_trend_offsets.get(sample).copied().unwrap_or(0.0),
                                        &self.config,
                                    )
                                })
                                .collect::<Vec<_>>())
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>()
        };
        Ok(json!({
            "model": self.model_name(),
            "sample_count": sample_count,
            "columns": [
                "series_id",
                "timestamp",
                "horizon",
                "sample",
                "prediction",
                "mean",
                "residual_draw",
                "coefficient_draw",
                "trend_draw",
            ],
            "records": records,
        }))
    }

    pub fn predict_samples_json_string(&self, horizon: usize) -> Result<String> {
        serde_json::to_string(&self.predict_samples_json_value(horizon)?).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal posterior samples: {err}"
            ))
        })
    }

    pub fn predict_at_timestamps(
        &self,
        timestamps_by_series: BTreeMap<String, Vec<chrono::NaiveDateTime>>,
    ) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let schedule = self.validate_prediction_schedule(timestamps_by_series)?;
        self.predict_with_schedule(fitted, &schedule)
    }

    fn horizon_schedule(
        &self,
        fitted: &FittedPiecewiseLinearSeasonalState,
        horizon: usize,
    ) -> Result<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
        validate_horizon(horizon)?;
        fitted
            .series
            .iter()
            .map(|(series_id, series)| {
                let timestamps = (1..=horizon)
                    .map(|step| {
                        fitted
                            .frame
                            .frequency()
                            .advance(series.last_timestamp, step)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((series_id.clone(), timestamps))
            })
            .collect()
    }

    fn validate_prediction_schedule(
        &self,
        timestamps_by_series: BTreeMap<String, Vec<chrono::NaiveDateTime>>,
    ) -> Result<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        if timestamps_by_series.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "future_timestamps must contain at least one timestamp".to_string(),
            ));
        }
        for series_id in fitted.series.keys() {
            let timestamps = timestamps_by_series.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "future_timestamps_by_series is missing fitted series {series_id:?}"
                ))
            })?;
            if timestamps.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "future_timestamps for series {series_id:?} must not be empty"
                )));
            }
            let series = fitted.series.get(series_id).expect("series key exists");
            let mut previous = series.last_timestamp;
            for timestamp in timestamps {
                if *timestamp <= previous {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "future_timestamps for series {series_id:?} must be strictly after the last training timestamp and increasing"
                    )));
                }
                previous = *timestamp;
            }
        }
        for series_id in timestamps_by_series.keys() {
            if !fitted.series.contains_key(series_id) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "future_timestamps_by_series contains unknown series {series_id:?}"
                )));
            }
        }
        Ok(timestamps_by_series)
    }

    fn predict_with_schedule(
        &self,
        fitted: &FittedPiecewiseLinearSeasonalState,
        schedule: &BTreeMap<String, Vec<chrono::NaiveDateTime>>,
    ) -> Result<ForecastResult> {
        let per_series = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                let emit_intervals = !self.config.interval_levels.is_empty();
                let residual_scale = if emit_intervals {
                    series.residual_scale()
                } else {
                    0.0
                };
                let timestamps = schedule.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing prediction timestamps for series {series_id:?}"
                    ))
                })?;
                timestamps
                    .iter()
                    .enumerate()
                    .map(|(idx, &timestamp)| {
                        let step = idx + 1;
                        let elapsed = elapsed_days(series.start_timestamp, timestamp);
                        let bounds =
                            piecewise_bounds(Some(series_id), None, Some(step), &self.config)?;
                        let terms = series.prediction_terms_at(
                            series_id,
                            elapsed,
                            timestamp,
                            step,
                            bounds,
                            &self.config,
                        )?;
                        let prediction = ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean: terms.mean,
                        };
                        let intervals = if emit_intervals {
                            piecewise_prediction_intervals(
                                &prediction,
                                residual_scale,
                                terms.coefficient_scale,
                                if series.transformed_residual_scale > 0.0 {
                                    series.transformed_residual_scale
                                } else {
                                    residual_scale
                                },
                                terms.linear_predictor,
                                terms.linear_coefficient_scale,
                                series.trend_uncertainty_offsets(
                                    series_id,
                                    elapsed,
                                    timestamp,
                                    step,
                                    &self.config,
                                )?,
                                series.trend_uncertainty_linear_offsets(
                                    series_id,
                                    elapsed,
                                    step,
                                    &self.config,
                                ),
                                &self.config.interval_levels,
                                bounds,
                                &self.config,
                            )?
                        } else {
                            Vec::new()
                        };
                        Ok((prediction, intervals))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut predictions = Vec::with_capacity(per_series.len());
        let mut intervals = Vec::new();
        for (prediction, prediction_intervals) in per_series {
            predictions.push(prediction);
            intervals.extend(prediction_intervals);
        }
        ForecastResult::new_with_intervals(predictions, intervals)
    }

    pub fn predict_quantiles_json_value(
        &self,
        horizon: usize,
        quantile_levels: Option<Vec<f64>>,
    ) -> Result<Value> {
        validate_horizon(horizon)?;
        let levels = quantile_levels.unwrap_or_else(|| self.config.quantile_levels.clone());
        validate_piecewise_quantile_levels(&levels)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let records = if levels.is_empty() {
            Vec::new()
        } else {
            fitted
                .series
                .iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(series_id, series)| {
                    let residual_scale = series.residual_scale();
                    (1..=horizon)
                        .map(|step| {
                            let timestamp = fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?;
                            let elapsed = elapsed_days(series.start_timestamp, timestamp);
                            let bounds =
                                piecewise_bounds(Some(series_id), None, Some(step), &self.config)?;
                            let terms = series.prediction_terms_at(
                                series_id,
                                elapsed,
                                timestamp,
                                step,
                                bounds,
                                &self.config,
                            )?;
                            let prediction = ForecastPrediction {
                                series_id: series_id.clone(),
                                timestamp,
                                horizon: step,
                                model: self.model_name().to_string(),
                                mean: terms.mean,
                            };
                            Ok(piecewise_prediction_quantiles(
                                &prediction,
                                residual_scale,
                                terms.coefficient_scale,
                                if series.transformed_residual_scale > 0.0 {
                                    series.transformed_residual_scale
                                } else {
                                    residual_scale
                                },
                                terms.linear_predictor,
                                terms.linear_coefficient_scale,
                                series.trend_uncertainty_offsets(
                                    series_id,
                                    elapsed,
                                    timestamp,
                                    step,
                                    &self.config,
                                )?,
                                series.trend_uncertainty_linear_offsets(
                                    series_id,
                                    elapsed,
                                    step,
                                    &self.config,
                                ),
                                &levels,
                                bounds,
                                &self.config,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>()
        };
        Ok(json!({
            "model": self.model_name(),
            "quantile_levels": levels,
            "columns": [
                "series_id",
                "timestamp",
                "horizon",
                "quantile",
                "prediction",
                "mean",
            ],
            "records": records,
        }))
    }

    pub fn predict_quantiles_json_string(
        &self,
        horizon: usize,
        quantile_levels: Option<Vec<f64>>,
    ) -> Result<String> {
        serde_json::to_string(&self.predict_quantiles_json_value(horizon, quantile_levels)?)
            .map_err(|err| {
                CartoBoostError::InvalidInput(format!(
                    "failed to serialize piecewise linear seasonal quantiles: {err}"
                ))
            })
    }
}

impl PiecewiseLinearGrowth {
    fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Flat => "flat",
            Self::Logistic => "logistic",
        }
    }
}

impl PiecewiseLinearComponentMode {
    fn name(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::Multiplicative => "multiplicative",
        }
    }
}

impl SpatialPiecewiseKrigingMode {
    fn name(self) -> &'static str {
        match self {
            Self::KrigedRegressors => "kriged_regressors",
            Self::ResidualKriging => "residual_kriging",
            Self::Hybrid => "hybrid",
        }
    }
}

impl SpatialPiecewiseKrigingConfig {
    fn uses_kriged_regressors(&self) -> bool {
        matches!(
            self.mode,
            SpatialPiecewiseKrigingMode::KrigedRegressors | SpatialPiecewiseKrigingMode::Hybrid
        )
    }

    fn uses_residual_kriging(&self) -> bool {
        matches!(
            self.mode,
            SpatialPiecewiseKrigingMode::ResidualKriging | SpatialPiecewiseKrigingMode::Hybrid
        )
    }

    fn piecewise_config_metadata(&self) -> Value {
        let piecewise_config = spatial_piecewise_base_config(self);
        json!({
            "growth": piecewise_config.growth.name(),
            "component_mode": piecewise_config.component_mode.name(),
            "changepoints": piecewise_config.changepoints,
            "weekly_fourier_order": piecewise_config.weekly_fourier_order,
            "daily_fourier_order": piecewise_config.daily_fourier_order,
            "yearly_fourier_order": piecewise_config.yearly_fourier_order,
            "extra_regressors": piecewise_config.extra_regressors,
        })
    }
}

impl PiecewiseLinearFitLoss {
    fn name(self) -> &'static str {
        match self {
            Self::Squared => "squared",
            Self::Huber => "huber",
        }
    }
}

impl PiecewiseLinearRegressorStandardization {
    fn name(self) -> &'static str {
        match self {
            PiecewiseLinearRegressorStandardization::None => "none",
            PiecewiseLinearRegressorStandardization::Auto => "auto",
        }
    }
}

impl PiecewiseLinearTrendUncertaintyPolicy {
    fn name(self) -> &'static str {
        match self {
            PiecewiseLinearTrendUncertaintyPolicy::Normal => "normal",
            PiecewiseLinearTrendUncertaintyPolicy::Laplace => "laplace",
        }
    }
}

impl SeasonalNaiveForecaster {
    pub fn new(season_length: usize) -> Result<Self> {
        if season_length == 0 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be positive".to_string(),
            ));
        }
        Ok(Self {
            season_length,
            fitted: None,
        })
    }
}

impl WindowAverageForecaster {
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(CartoBoostError::InvalidInput(
                "window_size must be positive".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            fitted: None,
        })
    }
}

impl SeasonalWindowAverageForecaster {
    pub fn new(season_length: usize, window_count: usize) -> Result<Self> {
        if season_length == 0 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be positive".to_string(),
            ));
        }
        if window_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "seasonal window_count must be positive".to_string(),
            ));
        }
        Ok(Self {
            season_length,
            window_count,
            fitted: None,
        })
    }
}

impl ThetaSeasonality {
    pub fn additive(season_length: usize) -> Result<Self> {
        Self::new(ThetaSeasonalityKind::Additive, season_length)
    }

    pub fn multiplicative(season_length: usize) -> Result<Self> {
        Self::new(ThetaSeasonalityKind::Multiplicative, season_length)
    }

    fn new(kind: ThetaSeasonalityKind, season_length: usize) -> Result<Self> {
        if season_length <= 1 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be greater than 1 for theta seasonality".to_string(),
            ));
        }
        Ok(Self {
            kind,
            season_length,
        })
    }

    fn name(self) -> &'static str {
        match self.kind {
            ThetaSeasonalityKind::Additive => "additive",
            ThetaSeasonalityKind::Multiplicative => "multiplicative",
        }
    }
}

impl ThetaForecaster {
    pub fn new(theta: f64, alpha: f64) -> Result<Self> {
        Self::with_seasonality(theta, alpha, None)
    }

    pub fn with_seasonality(
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        validate_theta_params(theta, alpha)?;
        Ok(Self {
            theta,
            alpha,
            seasonality,
            fitted: None,
        })
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }
}

impl OptimizedThetaForecaster {
    pub fn new(theta_grid: Vec<f64>, alpha_grid: Vec<f64>) -> Result<Self> {
        Self::with_seasonality(theta_grid, alpha_grid, None)
    }

    pub fn with_seasonality(
        theta_grid: Vec<f64>,
        alpha_grid: Vec<f64>,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        if theta_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "theta_grid must not be empty".to_string(),
            ));
        }
        if alpha_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "alpha_grid must not be empty".to_string(),
            ));
        }
        for &theta in &theta_grid {
            validate_theta_params(theta, 0.5)?;
        }
        for &alpha in &alpha_grid {
            validate_theta_params(1.0, alpha)?;
        }
        Ok(Self {
            theta_grid,
            alpha_grid,
            seasonality,
            selected_theta: None,
            selected_alpha: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_theta(&self) -> Option<f64> {
        self.selected_theta
    }

    pub fn selected_alpha(&self) -> Option<f64> {
        self.selected_alpha
    }

    pub fn validation_scores(&self) -> &[ThetaValidationScore] {
        &self.validation_scores
    }
}

impl ETSForecaster {
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        Self::with_additive_seasonality(alpha, beta, None, None)
    }

    pub fn with_additive_seasonality(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        Self::with_additive_damped_trend(alpha, beta, gamma, season_length, 1.0)
    }

    pub fn with_additive_damped_trend(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
        damping_phi: f64,
    ) -> Result<Self> {
        validate_ets_params(alpha, beta, gamma, season_length, damping_phi)?;
        Ok(Self {
            alpha,
            beta,
            gamma,
            season_length,
            damping_phi,
            fitted: None,
        })
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }

    pub fn level_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.level_values.as_slice())
    }

    pub fn trend_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.trend_values.as_slice())
    }

    pub fn seasonal_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.seasonal_values.as_slice())
    }
}

impl AutoETSForecaster {
    pub fn new(season_length: Option<usize>) -> Result<Self> {
        Self::with_grids(
            vec![0.1, 0.2, 0.3, 0.5, 0.8, 0.95],
            vec![0.0, 0.05, 0.1, 0.2, 0.4],
            match season_length {
                Some(_) => vec![
                    Some(0.0),
                    Some(0.05),
                    Some(0.1),
                    Some(0.2),
                    Some(0.3),
                    Some(0.5),
                ],
                None => vec![None],
            },
            vec![0.8, 0.9, 0.95, 0.98, 1.0],
            season_length,
        )
    }

    pub fn with_grids(
        alpha_grid: Vec<f64>,
        beta_grid: Vec<f64>,
        gamma_grid: Vec<Option<f64>>,
        damping_phi_grid: Vec<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        if alpha_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets alpha_grid must not be empty".to_string(),
            ));
        }
        if beta_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets beta_grid must not be empty".to_string(),
            ));
        }
        if gamma_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets gamma_grid must not be empty".to_string(),
            ));
        }
        if damping_phi_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets damping_phi_grid must not be empty".to_string(),
            ));
        }
        for &alpha in &alpha_grid {
            validate_ets_params(alpha, 0.0, None, None, 1.0)?;
        }
        for &beta in &beta_grid {
            validate_ets_params(0.5, beta, None, None, 1.0)?;
        }
        for &gamma in &gamma_grid {
            validate_ets_params(0.5, 0.1, gamma, season_length, 1.0)?;
        }
        for &damping_phi in &damping_phi_grid {
            validate_ets_params(0.5, 0.1, None, None, damping_phi)?;
        }
        Ok(Self {
            alpha_grid,
            beta_grid,
            gamma_grid,
            damping_phi_grid,
            season_length,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<ETSParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[ETSValidationScore] {
        &self.validation_scores
    }
}

impl ArimaForecaster {
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        validate_arima_order(p, d, q)?;
        Ok(Self {
            p,
            d,
            q,
            fitted: None,
        })
    }

    pub fn order(&self) -> (usize, usize, usize) {
        (self.p, self.d, self.q)
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }
}

impl AutoARIMAForecaster {
    pub fn new(max_p: usize, max_d: usize) -> Result<Self> {
        Self::with_max_order(max_p, max_d, 2)
    }

    pub fn with_max_order(max_p: usize, max_d: usize, max_q: usize) -> Result<Self> {
        if max_p > 8 {
            return Err(CartoBoostError::InvalidInput(
                "max_p must be <= 8 for auto_arima".to_string(),
            ));
        }
        if max_d > 2 {
            return Err(CartoBoostError::InvalidInput(
                "max_d must be <= 2 for auto_arima".to_string(),
            ));
        }
        if max_q > 8 {
            return Err(CartoBoostError::InvalidInput(
                "max_q must be <= 8 for auto_arima".to_string(),
            ));
        }
        Ok(Self {
            max_p,
            max_d,
            max_q,
            selected_order: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_order(&self) -> Option<(usize, usize, usize)> {
        self.selected_order
    }

    pub fn validation_scores(&self) -> &[ArimaValidationScore] {
        &self.validation_scores
    }
}

impl KalmanForecaster {
    pub fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        LocalLinearKalmanConfig::new(
            level_process_variance,
            trend_process_variance,
            observation_variance,
        )?;
        Ok(Self {
            level_process_variance,
            trend_process_variance,
            observation_variance,
            fitted: None,
        })
    }
}

impl LocalLevelKalmanForecaster {
    pub fn new(level_process_variance: f64, observation_variance: f64) -> Result<Self> {
        LocalLevelKalmanConfig::new(level_process_variance, observation_variance)?;
        Ok(Self {
            level_process_variance,
            observation_variance,
            fitted: None,
        })
    }
}

impl AutoKalmanForecaster {
    pub fn new() -> Result<Self> {
        Self::with_grids(
            vec![0.001, 0.01, 0.05, 0.1],
            vec![0.0001, 0.001, 0.005, 0.01],
            vec![0.1, 0.5, 1.0, 2.0],
            None,
        )
    }

    pub fn with_grids(
        level_process_variance_grid: Vec<f64>,
        trend_process_variance_grid: Vec<f64>,
        observation_variance_grid: Vec<f64>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        validate_kalman_grid("level_process_variance_grid", &level_process_variance_grid)?;
        validate_kalman_grid("trend_process_variance_grid", &trend_process_variance_grid)?;
        validate_kalman_grid("observation_variance_grid", &observation_variance_grid)?;
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "auto_kalman validation_window must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            level_process_variance_grid,
            trend_process_variance_grid,
            observation_variance_grid,
            validation_window,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<KalmanParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[KalmanValidationScore] {
        &self.validation_scores
    }
}

impl AutoLocalLevelKalmanForecaster {
    pub fn new() -> Result<Self> {
        Self::with_grids(vec![0.001, 0.01, 0.05, 0.1], vec![0.1, 0.5, 1.0, 2.0], None)
    }

    pub fn with_grids(
        level_process_variance_grid: Vec<f64>,
        observation_variance_grid: Vec<f64>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        validate_kalman_grid("level_process_variance_grid", &level_process_variance_grid)?;
        validate_kalman_grid("observation_variance_grid", &observation_variance_grid)?;
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "auto_local_level_kalman validation_window must be positive when provided"
                    .to_string(),
            ));
        }
        Ok(Self {
            level_process_variance_grid,
            observation_variance_grid,
            validation_window,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<LocalLevelKalmanParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[LocalLevelKalmanValidationScore] {
        &self.validation_scores
    }
}

impl Default for AutoLocalLevelKalmanForecaster {
    fn default() -> Self {
        Self::new().expect("default auto_local_level_kalman grid is valid")
    }
}

impl Default for AutoKalmanForecaster {
    fn default() -> Self {
        Self::new().expect("default auto_kalman grid is valid")
    }
}

impl Default for KalmanForecaster {
    fn default() -> Self {
        Self {
            level_process_variance: 0.05,
            trend_process_variance: 0.005,
            observation_variance: 1.0,
            fitted: None,
        }
    }
}

impl Default for LocalLevelKalmanForecaster {
    fn default() -> Self {
        Self {
            level_process_variance: 0.05,
            observation_variance: 1.0,
            fitted: None,
        }
    }
}

impl KrigingForecaster {
    pub fn new(coordinates: BTreeMap<String, (f64, f64)>, range: f64, nugget: f64) -> Result<Self> {
        Self::with_config(coordinates, OrdinaryKrigingConfig::new(range, nugget)?)
    }

    pub fn with_config(
        coordinates: BTreeMap<String, (f64, f64)>,
        config: OrdinaryKrigingConfig,
    ) -> Result<Self> {
        if coordinates.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "kriging coordinates must not be empty".to_string(),
            ));
        }
        for (series_id, (x, y)) in &coordinates {
            if !x.is_finite() || !y.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "kriging coordinate for series {series_id} must be finite"
                )));
            }
        }
        Ok(Self {
            coordinates,
            config,
            fitted: None,
        })
    }
}

impl SpatialPiecewiseKrigingForecaster {
    pub fn new(config: SpatialPiecewiseKrigingConfig) -> Result<Self> {
        validate_spatial_piecewise_kriging_config(&config)?;
        Ok(Self {
            config,
            fitted: None,
        })
    }

    pub fn config(&self) -> &SpatialPiecewiseKrigingConfig {
        &self.config
    }
}

impl Forecaster for NaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedLocalState::from_frame(frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: last.target,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "naive"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name()})
    }
}

impl Forecaster for SeasonalNaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedLocalState::from_frame(frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let effective_season_length = self.season_length.min(history.len()).max(1);
                let base = history.len() - effective_season_length;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let seasonal_index = base + ((step - 1) % effective_season_length);
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: history[seasonal_index].target,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "seasonal_naive"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name(), "season_length": self.season_length})
    }
}

impl Forecaster for WindowAverageForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedLocalState::from_frame(frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let effective_window_size = self.window_size.min(history.len()).max(1);
                let start = history.len() - effective_window_size;
                let mean = history[start..].iter().map(|row| row.target).sum::<f64>()
                    / effective_window_size as f64;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "window_average"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "window_size": self.window_size,
        })
    }
}

impl Forecaster for SeasonalWindowAverageForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedLocalState::from_frame(frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let model = self.model_name().to_string();
                let effective_season_length = self.season_length.min(history.len()).max(1);
                let effective_window_count = self
                    .window_count
                    .min(history.len() / effective_season_length)
                    .max(1);
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let phase_offset = (step - 1) % effective_season_length;
                    let mut sum = 0.0;
                    for window in 0..effective_window_count {
                        let base = history.len() - effective_season_length * (window + 1);
                        sum += history[base + phase_offset].target;
                    }
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: sum / effective_window_count as f64,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "seasonal_window_average"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "season_length": self.season_length,
            "window_count": self.window_count,
        })
    }
}

impl Forecaster for ThetaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedThetaState::from_frame(
            frame,
            self.theta,
            self.alpha,
            self.seasonality,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "theta"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "theta": self.theta,
            "alpha": self.alpha,
            "seasonality": self.seasonality.map(ThetaSeasonality::name),
            "season_length": self.seasonality.map(|seasonality| seasonality.season_length),
        })
    }
}

impl Forecaster for OptimizedThetaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let candidates = self
            .theta_grid
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(theta_idx, theta)| {
                self.alpha_grid
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(alpha_idx, alpha)| (theta_idx, alpha_idx, theta, alpha))
            })
            .collect::<Vec<_>>();
        let scored = candidates
            .into_par_iter()
            .map(|(theta_idx, alpha_idx, theta, alpha)| {
                let fitted = FittedThetaState::from_frame(frame, theta, alpha, self.seasonality)?;
                let mse = fitted.mean_squared_residual();
                Ok((
                    theta_idx,
                    alpha_idx,
                    ThetaValidationScore { theta, alpha, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(theta_idx, alpha_idx, _)| (*theta_idx, *alpha_idx));
        let scores = scored
            .into_iter()
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores
            .iter()
            .map(|score| {
                (
                    OrderedF64(score.mse),
                    OrderedF64(score.theta),
                    OrderedF64(score.alpha),
                )
            })
            .min();
        let (_, theta, alpha) = best.ok_or_else(|| {
            CartoBoostError::InvalidInput("theta validation grid must not be empty".to_string())
        })?;
        let mut fitted = ThetaForecaster::with_seasonality(theta.0, alpha.0, self.seasonality)?;
        fitted.fit(frame)?;
        self.selected_theta = Some(theta.0);
        self.selected_alpha = Some(alpha.0);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "optimized_theta"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "selected_theta": self.selected_theta,
            "selected_alpha": self.selected_alpha,
            "seasonality": self.seasonality.map(ThetaSeasonality::name),
            "season_length": self.seasonality.map(|seasonality| seasonality.season_length),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({"theta": score.theta, "alpha": score.alpha, "mse": score.mse})
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for ETSForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedETSState::from_frame(
            frame,
            self.alpha,
            self.beta,
            self.gamma,
            self.season_length,
            self.damping_phi,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        let seasonal = series
                            .seasonals
                            .as_ref()
                            .map(|seasonals| seasonals[(series.n_obs + step - 1) % seasonals.len()])
                            .unwrap_or(0.0);
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean: series.level
                                + damped_trend_multiplier(series.damping_phi, step) * series.trend
                                + seasonal,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "ets"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "season_length": self.season_length,
            "damping_phi": self.damping_phi,
        })
    }
}

impl Forecaster for AutoETSForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let mut candidates = Vec::new();
        for (alpha_idx, alpha) in self.alpha_grid.iter().copied().enumerate() {
            for (beta_idx, beta) in self.beta_grid.iter().copied().enumerate() {
                for (gamma_idx, gamma) in self.gamma_grid.iter().copied().enumerate() {
                    for (damping_idx, damping_phi) in
                        self.damping_phi_grid.iter().copied().enumerate()
                    {
                        candidates.push((
                            alpha_idx,
                            beta_idx,
                            gamma_idx,
                            damping_idx,
                            alpha,
                            beta,
                            gamma,
                            damping_phi,
                        ));
                    }
                }
            }
        }
        let scored = candidates
            .into_par_iter()
            .map(
                |(alpha_idx, beta_idx, gamma_idx, damping_idx, alpha, beta, gamma, damping_phi)| {
                    let fitted = FittedETSState::from_frame(
                        frame,
                        alpha,
                        beta,
                        gamma,
                        self.season_length,
                        damping_phi,
                    )?;
                    let params = ETSParameterSet {
                        alpha,
                        beta,
                        gamma,
                        damping_phi,
                    };
                    Ok((
                        alpha_idx,
                        beta_idx,
                        gamma_idx,
                        damping_idx,
                        ETSValidationScore {
                            params,
                            mse: fitted.mean_squared_residual(),
                        },
                    ))
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(alpha_idx, beta_idx, gamma_idx, damping_idx, _)| {
            (*alpha_idx, *beta_idx, *gamma_idx, *damping_idx)
        });
        let scores = scored
            .into_iter()
            .map(|(_, _, _, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by(|left, right| {
            left.mse
                .total_cmp(&right.mse)
                .then_with(|| left.params.alpha.total_cmp(&right.params.alpha))
                .then_with(|| left.params.beta.total_cmp(&right.params.beta))
                .then_with(|| left.params.damping_phi.total_cmp(&right.params.damping_phi))
                .then_with(|| match (left.params.gamma, right.params.gamma) {
                    (Some(left), Some(right)) => left.total_cmp(&right),
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                })
        });
        let params = best.map(|score| score.params).ok_or_else(|| {
            CartoBoostError::InvalidInput("auto_ets candidate grid must not be empty".to_string())
        })?;
        let mut fitted = ETSForecaster::with_additive_damped_trend(
            params.alpha,
            params.beta,
            params.gamma,
            self.season_length,
            params.damping_phi,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.predict(horizon)?;
        let predictions = result
            .predictions()
            .iter()
            .map(|prediction| ForecastPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: self.model_name().to_string(),
                mean: prediction.mean,
            })
            .collect::<Vec<_>>();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "auto_ets"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "season_length": self.season_length,
            "selected_params": self.selected_params.map(|params| {
                json!({
                    "alpha": params.alpha,
                    "beta": params.beta,
                    "gamma": params.gamma,
                    "damping_phi": params.damping_phi,
                })
            }),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "alpha": score.params.alpha,
                    "beta": score.params.beta,
                    "gamma": score.params.gamma,
                    "damping_phi": score.params.damping_phi,
                    "mse": score.mse,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for ArimaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedArimaState::from_frame(frame, self.p, self.d, self.q)?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "arima"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name(), "p": self.p, "d": self.d, "q": self.q})
    }
}

impl Forecaster for AutoARIMAForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let max_p = self.max_p;
        let max_d = self.max_d;
        let max_q = self.max_q;
        let min_history_len = FittedLocalState::from_frame(frame)
            .history_by_series
            .values()
            .map(Vec::len)
            .min()
            .unwrap_or(0);
        let mut candidate_orders = BTreeSet::new();
        for d in 0..=max_d {
            for p in 0..=max_p {
                for q in 0..=max_q {
                    candidate_orders.insert(arima_order_supported_by_history(
                        min_history_len,
                        p,
                        d,
                        q,
                    ));
                }
            }
        }
        let mut scores = candidate_orders
            .into_par_iter()
            .map(|(p, d, q)| {
                let fitted = FittedArimaState::from_frame(frame, p, d, q)?;
                let mse = fitted.mean_squared_residual();
                let stable = fitted.has_stable_ar_recursions();
                Ok(ArimaValidationScore {
                    p,
                    d,
                    q,
                    mse,
                    stable,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        scores.sort_by_key(|score| (score.d, score.p, score.q));
        let best = scores
            .iter()
            .filter(|score| score.stable)
            .map(|score| (OrderedF64(score.mse), score.p, score.d, score.q))
            .min();
        let (_, p, d, q) = best.ok_or_else(|| {
            CartoBoostError::InvalidInput("auto_arima candidate grid must not be empty".to_string())
        })?;
        let mut fitted = ArimaForecaster::new(p, d, q)?;
        fitted.fit(frame)?;
        self.selected_order = Some((p, d, q));
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_arima"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "max_p": self.max_p,
            "max_d": self.max_d,
            "max_q": self.max_q,
            "selected_order": self.selected_order.map(|(p, d, q)| json!({"p": p, "d": d, "q": q})),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({"p": score.p, "d": score.d, "q": score.q, "mse": score.mse, "stable": score.stable})
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for KalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedKalmanState::from_frame(
            frame,
            self.level_process_variance,
            self.trend_process_variance,
            self.observation_variance,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "level_process_variance": self.level_process_variance,
            "trend_process_variance": self.trend_process_variance,
            "observation_variance": self.observation_variance,
        })
    }
}

impl Forecaster for LocalLevelKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedLocalLevelKalmanState::from_frame(
            frame,
            self.level_process_variance,
            self.observation_variance,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "local_level_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "level_process_variance": self.level_process_variance,
            "observation_variance": self.observation_variance,
        })
    }
}

impl Forecaster for AutoKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let local = FittedLocalState::from_frame(frame);
        let level_grid = self.level_process_variance_grid.clone();
        let trend_grid = self.trend_process_variance_grid.clone();
        let observation_grid = self.observation_variance_grid.clone();
        let validation_window = self.validation_window;
        let mut candidates = Vec::new();
        for (level_idx, level_process_variance) in level_grid.iter().copied().enumerate() {
            for (trend_idx, trend_process_variance) in trend_grid.iter().copied().enumerate() {
                for (observation_idx, observation_variance) in
                    observation_grid.iter().copied().enumerate()
                {
                    candidates.push((
                        level_idx,
                        trend_idx,
                        observation_idx,
                        KalmanParameterSet {
                            level_process_variance,
                            trend_process_variance,
                            observation_variance,
                        },
                    ));
                }
            }
        }
        let scored = candidates
            .into_par_iter()
            .map(|(level_idx, trend_idx, observation_idx, params)| {
                let mse = score_kalman_params(&local.history_by_series, params, validation_window)?;
                Ok((
                    level_idx,
                    trend_idx,
                    observation_idx,
                    KalmanValidationScore { params, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(level_idx, trend_idx, observation_idx, _)| {
            (*level_idx, *trend_idx, *observation_idx)
        });
        let scores = scored
            .into_iter()
            .map(|(_, _, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by_key(|score| {
            (
                OrderedF64(score.mse),
                OrderedF64(score.params.level_process_variance),
                OrderedF64(score.params.trend_process_variance),
                OrderedF64(score.params.observation_variance),
            )
        });
        let params = best
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_kalman candidate grid must not be empty".to_string(),
                )
            })?
            .params;
        let mut fitted = KalmanForecaster::new(
            params.level_process_variance,
            params.trend_process_variance,
            params.observation_variance,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| json!({
                "level_process_variance": params.level_process_variance,
                "trend_process_variance": params.trend_process_variance,
                "observation_variance": params.observation_variance,
            })),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "level_process_variance": score.params.level_process_variance,
                    "trend_process_variance": score.params.trend_process_variance,
                    "observation_variance": score.params.observation_variance,
                    "mse": score.mse,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for AutoLocalLevelKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let local = FittedLocalState::from_frame(frame);
        let candidates = self
            .level_process_variance_grid
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(level_idx, level_process_variance)| {
                self.observation_variance_grid
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(observation_idx, observation_variance)| {
                        (
                            level_idx,
                            observation_idx,
                            LocalLevelKalmanParameterSet {
                                level_process_variance,
                                observation_variance,
                            },
                        )
                    })
            })
            .collect::<Vec<_>>();
        let scored = candidates
            .into_par_iter()
            .map(|(level_idx, observation_idx, params)| {
                let mse = score_local_level_kalman_params(
                    &local.history_by_series,
                    params,
                    self.validation_window,
                )?;
                Ok((
                    level_idx,
                    observation_idx,
                    LocalLevelKalmanValidationScore { params, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(level_idx, observation_idx, _)| (*level_idx, *observation_idx));
        let scores = scored
            .into_iter()
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by_key(|score| {
            (
                OrderedF64(score.mse),
                OrderedF64(score.params.level_process_variance),
                OrderedF64(score.params.observation_variance),
            )
        });
        let params = best
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_local_level_kalman candidate grid must not be empty".to_string(),
                )
            })?
            .params;
        let mut fitted = LocalLevelKalmanForecaster::new(
            params.level_process_variance,
            params.observation_variance,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_local_level_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| json!({
                "level_process_variance": params.level_process_variance,
                "observation_variance": params.observation_variance,
            })),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "level_process_variance": score.params.level_process_variance,
                    "observation_variance": score.params.observation_variance,
                    "mse": score.mse,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for KrigingForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedKrigingState::from_frame(frame, &self.coordinates)?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let observations = fitted
            .levels
            .iter()
            .map(|(series_id, value)| {
                let coord = self.coordinates.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing kriging coordinate for series {series_id}"
                    ))
                })?;
                Ok(KrigingObservation {
                    x: coord.0,
                    y: coord.1,
                    value: *value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let series_ids = fitted.frame.series_ids();
        let targets = series_ids
            .iter()
            .map(|series_id| {
                self.coordinates.get(series_id).copied().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing kriging coordinate for series {series_id}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let means = ordinary_kriging_predict_many(&observations, &targets, self.config)?
            .into_iter()
            .map(|prediction| prediction.mean)
            .collect::<Vec<_>>();
        let predictions = series_ids
            .into_par_iter()
            .zip(means)
            .map(|(series_id, mean)| {
                let history = fitted.frame.rows_for_series(&series_id);
                let last_timestamp = history
                    .last()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput("empty series history".to_string())
                    })?
                    .timestamp;
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted.frame.frequency().advance(last_timestamp, step)?,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "kriging"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "range": self.config.range,
            "nugget": self.config.nugget,
            "sill": self.config.sill,
            "variogram_model": format!("{:?}", self.config.variogram_model).to_lowercase(),
            "drift": format!("{:?}", self.config.drift).to_lowercase(),
            "anisotropy_angle_degrees": self.config.anisotropy_angle_degrees,
            "anisotropy_scaling": self.config.anisotropy_scaling,
            "max_neighbors": self.config.max_neighbors,
            "min_neighbors": self.config.min_neighbors,
            "max_distance": self.config.max_distance,
            "series_count": self.coordinates.len(),
        })
    }
}

impl Forecaster for SpatialPiecewiseKrigingForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        validate_spatial_piecewise_frame(frame, &self.config)?;
        self.fitted = Some(FittedSpatialPiecewiseKrigingState::from_frame(
            frame,
            &self.config,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut base = fitted.base.clone();
        if self.config.uses_kriged_regressors() {
            let future_regressors = fitted.future_spatial_regressors(horizon, &self.config)?;
            base.update_config(|config| {
                for (series_id, regressors) in future_regressors {
                    config
                        .future_regressors_by_series
                        .entry(series_id)
                        .or_default()
                        .extend(regressors);
                }
            })?;
        }
        let base_result = base.predict(horizon)?;
        let corrections = if self.config.uses_residual_kriging() {
            fitted.residual_kriging_predictions(&self.config)?
        } else {
            BTreeMap::new()
        };
        let component_records = base
            .predict_components_json_value(horizon)
            .ok()
            .and_then(|value| value.get("records").and_then(Value::as_array).cloned())
            .unwrap_or_default()
            .into_iter()
            .map(|record| {
                let key = component_record_key(&record);
                (key, record)
            })
            .collect::<BTreeMap<_, _>>();
        let mut predictions = Vec::with_capacity(base_result.predictions().len());
        let mut details = Vec::with_capacity(base_result.predictions().len());
        for base_prediction in base_result.predictions() {
            let correction = corrections.get(&base_prediction.series_id);
            let shrinkage = self
                .config
                .residual_shrinkage
                .powi((base_prediction.horizon - 1) as i32);
            let spatial_correction =
                correction.map(|correction| correction.prediction.mean * shrinkage);
            let final_mean = base_prediction.mean + spatial_correction.unwrap_or(0.0);
            predictions.push(ForecastPrediction {
                series_id: base_prediction.series_id.clone(),
                timestamp: base_prediction.timestamp,
                horizon: base_prediction.horizon,
                model: self.model_name().to_string(),
                mean: final_mean,
            });
            let selected_neighbors = correction
                .map(|prediction| {
                    prediction
                        .prediction
                        .neighbor_indices
                        .iter()
                        .filter_map(|idx| fitted.residual_observation_series.get(*idx).cloned())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let neighbor_count = selected_neighbors.len();
            let kriging_variance = correction.map(|prediction| prediction.prediction.variance);
            let correction_magnitude = spatial_correction.map(f64::abs);
            let detail_key = prediction_lookup_key(
                &base_prediction.series_id,
                base_prediction.timestamp,
                base_prediction.horizon,
            );
            details.push(ForecastPredictionDetail {
                series_id: base_prediction.series_id.clone(),
                timestamp: base_prediction.timestamp,
                horizon: base_prediction.horizon,
                model: self.model_name().to_string(),
                base_mean: Some(base_prediction.mean),
                spatial_correction,
                kriging_variance,
                selected_neighbors,
                component_decomposition: component_records.get(&detail_key).cloned(),
                metadata: Some(json!({
                    "mode": self.config.mode.name(),
                    "residual_shrinkage": self.config.residual_shrinkage,
                    "neighbor_fallback": correction.is_some_and(|prediction| prediction.used_neighbor_fallback),
                    "neighbor_count": neighbor_count,
                    "correction_magnitude": correction_magnitude,
                    "kriging_variance": kriging_variance,
                    "fit_runtime_seconds": fitted.fit_metadata.get("runtime_seconds").cloned(),
                    "cutoff": fitted.cutoff_timestamps.get(&base_prediction.series_id).map(|timestamp| {
                        timestamp.format("%Y-%m-%dT%H:%M:%S").to_string()
                    }),
                })),
            });
        }
        let intervals = base_result
            .intervals()
            .iter()
            .map(|interval| ForecastIntervalPrediction {
                series_id: interval.series_id.clone(),
                timestamp: interval.timestamp,
                horizon: interval.horizon,
                model: self.model_name().to_string(),
                level: interval.level,
                lower: interval.lower,
                upper: interval.upper,
            })
            .collect::<Vec<_>>();
        ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
    }

    fn model_name(&self) -> &'static str {
        "spatial_piecewise_kriging"
    }

    fn metadata(&self) -> Value {
        let fit_metadata = self
            .fitted
            .as_ref()
            .map(|fitted| fitted.fit_metadata.clone())
            .unwrap_or_else(|| json!({}));
        json!({
            "model": self.model_name(),
            "mode": self.config.mode.name(),
            "piecewise": self.config.piecewise_config_metadata(),
            "variogram": kriging_config_metadata(self.config.kriging_config),
            "spatial_regressors": self.config.spatial_regressors,
            "series_count": self.config.coordinates.len(),
            "residual_shrinkage": self.config.residual_shrinkage,
            "allow_neighbor_fallback": self.config.allow_neighbor_fallback,
            "fit": fit_metadata,
        })
    }
}

impl Forecaster for PiecewiseLinearSeasonalForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let resolved_config = resolve_piecewise_auto_seasonalities(frame, &self.config);
        self.fitted = Some(FittedPiecewiseLinearSeasonalState::from_frame(
            frame,
            resolved_config.clone(),
        )?);
        self.config = resolved_config;
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let schedule = self.horizon_schedule(fitted, horizon)?;
        self.predict_with_schedule(fitted, &schedule)
    }

    fn model_name(&self) -> &'static str {
        "piecewise_linear_seasonal"
    }

    fn metadata(&self) -> Value {
        let residual_rmse = self
            .fitted
            .as_ref()
            .map(FittedPiecewiseLinearSeasonalState::root_mean_squared_residual);
        let mut metadata = json!({
            "model": self.model_name(),
            "growth": self.config.growth.name(),
            "component_mode": self.config.component_mode.name(),
            "changepoints": self.config.changepoints,
            "changepoint_range": self.config.changepoint_range,
            "changepoint_timestamps": self.config.changepoint_timestamps.iter().map(|timestamp| {
                timestamp.format("%Y-%m-%dT%H:%M:%S").to_string()
            }).collect::<Vec<_>>(),
            "yearly_fourier_order": self.config.yearly_fourier_order,
            "weekly_fourier_order": self.config.weekly_fourier_order,
            "daily_fourier_order": self.config.daily_fourier_order,
            "auto_yearly_seasonality": self.config.auto_yearly_seasonality,
            "auto_weekly_seasonality": self.config.auto_weekly_seasonality,
            "auto_daily_seasonality": self.config.auto_daily_seasonality,
            "custom_seasonalities": self.config.custom_seasonalities.iter().map(|seasonality| {
                json!({
                    "name": seasonality.name,
                    "period_days": seasonality.period_days,
                    "fourier_order": seasonality.fourier_order,
                    "mode": seasonality.mode.map(PiecewiseLinearComponentMode::name),
                    "condition_name": seasonality.condition_name,
                    "l2_regularization": seasonality.l2_regularization,
                })
            }).collect::<Vec<_>>(),
            "changepoint_l2_regularization": self.config.changepoint_l2_regularization,
            "changepoint_l1_regularization": self.config.changepoint_l1_regularization,
            "seasonality_l2_regularization": self.config.seasonality_l2_regularization,
            "yearly_l2_regularization": self.config.yearly_l2_regularization,
            "weekly_l2_regularization": self.config.weekly_l2_regularization,
            "daily_l2_regularization": self.config.daily_l2_regularization,
            "event_l2_regularization": self.config.event_l2_regularization,
            "regressor_l2_regularization": self.config.regressor_l2_regularization,
            "event_l2_regularization_by_name": self.config.event_l2_regularization_by_name,
            "regressor_l2_regularization_by_name": self.config.regressor_l2_regularization_by_name,
            "events": self.config.events.iter().map(|event| {
                json!({
                    "name": event.name,
                    "timestamp": event.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    "lower_window": event.lower_window,
                    "upper_window": event.upper_window,
                })
            }).collect::<Vec<_>>(),
            "event_mode": self.config.event_mode.map(PiecewiseLinearComponentMode::name),
            "extra_regressors": self.config.extra_regressors,
            "regressor_modes": self.config.regressor_modes.iter().map(|(name, mode)| {
                (name.clone(), mode.name())
            }).collect::<BTreeMap<_, _>>(),
            "extra_regressor_monotonic_constraints": self.config.extra_regressor_monotonic_constraints,
            "regressor_standardization": self.config.regressor_standardization.name(),
            "future_regressors": self.config.future_regressors,
            "interval_levels": self.config.interval_levels,
            "quantile_levels": self.config.quantile_levels,
            "uncertainty_samples": self.config.uncertainty_samples,
            "trend_uncertainty_policy": self.config.trend_uncertainty_policy.name(),
            "trend_uncertainty_scale": self.config.trend_uncertainty_scale,
            "uncertainty_seed": self.config.uncertainty_seed,
            "cap": self.config.cap,
            "floor": self.config.floor,
            "cap_regressor": self.config.cap_regressor,
            "floor_regressor": self.config.floor_regressor,
            "residual_rmse": residual_rmse,
        });
        if let Value::Object(values) = &mut metadata {
            values.insert(
                "future_regressors_by_series".to_string(),
                json!(self.config.future_regressors_by_series),
            );
            values.insert(
                "trend_adjustments".to_string(),
                json!(self.config.trend_adjustments),
            );
            values.insert(
                "trend_adjustments_by_series".to_string(),
                json!(self.config.trend_adjustments_by_series),
            );
            values.insert(
                "residual_shock_window".to_string(),
                json!(self.config.residual_shock_window),
            );
            values.insert(
                "residual_shock_scale".to_string(),
                json!(self.config.residual_shock_scale),
            );
            values.insert(
                "residual_shock_decay".to_string(),
                json!(self.config.residual_shock_decay),
            );
            values.insert("fit_loss".to_string(), json!(self.config.fit_loss.name()));
            values.insert("huber_delta".to_string(), json!(self.config.huber_delta));
            values.insert(
                "irls_iterations".to_string(),
                json!(self.config.irls_iterations),
            );
            values.insert(
                "coefficient_uncertainty_scale".to_string(),
                json!(self.config.coefficient_uncertainty_scale),
            );
        }
        metadata
    }
}

impl ThetaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        let adjusted = forecast_theta_component(&series.component, step);
                        let fitted_seasonality =
                            series.seasonal_pattern.as_ref().and(self.seasonality);
                        let mean = reseasonalize_value(
                            adjusted,
                            series.n_obs + step - 1,
                            fitted_seasonality,
                            series.seasonal_pattern.as_deref(),
                        )?;
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl ArimaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                series
                    .forecast_values(horizon)
                    .into_iter()
                    .enumerate()
                    .map(|(idx, mean)| {
                        let step = idx + 1;
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl FittedLocalState {
    fn from_frame(frame: &ForecastFrame) -> Self {
        let mut history_by_series: BTreeMap<String, Vec<ForecastRow>> = BTreeMap::new();
        for row in frame.rows() {
            history_by_series
                .entry(row.series_id.clone())
                .or_default()
                .push(row.clone());
        }
        Self {
            frame: frame.clone(),
            history_by_series,
        }
    }
}

impl FittedETSState {
    fn from_frame(
        frame: &ForecastFrame,
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
        damping_phi: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let effective_season_length =
            effective_full_cycle_season_length(&local.history_by_series, season_length);
        let effective_gamma = gamma.filter(|_| effective_season_length.is_some());
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedETSSeries::fit(
                        series_id,
                        history,
                        alpha,
                        beta,
                        effective_gamma,
                        effective_season_length,
                        damping_phi,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

impl FittedArimaState {
    fn from_frame(frame: &ForecastFrame, p: usize, d: usize, q: usize) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedArimaSeries::fit(series_id, history, p, d, q)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(series.score_start)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    fn has_stable_ar_recursions(&self) -> bool {
        self.series
            .values()
            .all(|series| ar_recursion_is_stable(&series.ar_coefficients))
    }
}

impl FittedPiecewiseLinearSeasonalState {
    fn from_frame(frame: &ForecastFrame, config: PiecewiseLinearSeasonalConfig) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedPiecewiseLinearSeasonalSeries::fit(series_id, history, &config)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn root_mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            (sum / count as f64).sqrt()
        }
    }
}

impl FittedPiecewiseLinearSeasonalSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for piecewise linear seasonal forecasting"
            )));
        }
        let start_timestamp = history[0].timestamp;
        let last_timestamp = history[history.len() - 1].timestamp;
        let elapsed = history
            .iter()
            .map(|row| elapsed_days(start_timestamp, row.timestamp))
            .collect::<Vec<_>>();
        let changepoints = select_piecewise_changepoints(
            series_id,
            start_timestamp,
            last_timestamp,
            &elapsed,
            config,
        )?;
        let regressor_stats = piecewise_regressor_stats(history, config)?;
        let feature_count = piecewise_linear_seasonal_feature_count(config, changepoints.len());
        let trend_coefficients =
            fit_piecewise_trend_coefficients(history, &elapsed, &changepoints, config)?;
        let compute_covariance = piecewise_needs_coefficient_covariance(config);
        let mut fit_result = fit_piecewise_linear_weighted_coefficients(
            history,
            &elapsed,
            &changepoints,
            config,
            &regressor_stats,
            &trend_coefficients,
            None,
            compute_covariance && config.fit_loss != PiecewiseLinearFitLoss::Huber,
        )?
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "could not solve piecewise linear seasonal normal equations for series {series_id}"
            ))
        })?;
        let mut coefficients = fit_result.coefficients.clone();
        if config.fit_loss == PiecewiseLinearFitLoss::Huber && config.irls_iterations > 0 {
            for iteration in 0..config.irls_iterations {
                let residuals = piecewise_transformed_residuals(
                    history,
                    &elapsed,
                    &changepoints,
                    config,
                    &regressor_stats,
                    &trend_coefficients,
                    &coefficients,
                )?;
                let weights = huber_irls_weights(&residuals, config.huber_delta);
                fit_result = fit_piecewise_linear_weighted_coefficients(
                    history,
                    &elapsed,
                    &changepoints,
                    config,
                    &regressor_stats,
                    &trend_coefficients,
                    Some(&weights),
                    compute_covariance && iteration + 1 == config.irls_iterations,
                )?
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "could not solve robust piecewise linear seasonal normal equations for series {series_id}"
                    ))
                })?;
                let max_delta =
                    max_abs_difference(coefficients.as_slice(), fit_result.coefficients.as_slice());
                coefficients = fit_result.coefficients.clone();
                if max_delta < 1.0e-8 {
                    if compute_covariance && fit_result.coefficient_covariance.is_empty() {
                        fit_result = fit_piecewise_linear_weighted_coefficients(
                            history,
                            &elapsed,
                            &changepoints,
                            config,
                            &regressor_stats,
                            &trend_coefficients,
                            Some(&weights),
                            true,
                        )?
                        .ok_or_else(|| {
                            CartoBoostError::InvalidInput(format!(
                                "could not solve robust piecewise linear covariance for series {series_id}"
                            ))
                        })?;
                        coefficients = fit_result.coefficients.clone();
                    }
                    break;
                }
            }
        }
        let transformed_residual_scale = if compute_covariance
            || !config.interval_levels.is_empty()
            || !config.quantile_levels.is_empty()
            || config.uncertainty_samples > 0
        {
            piecewise_transformed_residual_scale(
                history,
                &elapsed,
                &changepoints,
                config,
                &regressor_stats,
                &coefficients,
            )?
        } else {
            0.0
        };
        let residuals = history
            .iter()
            .zip(elapsed.iter())
            .map(|(row, &t)| {
                let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
                Ok(row.target
                    - inverse_piecewise_target(
                        predict_piecewise_linear_value(
                            t,
                            &coefficients,
                            &PiecewiseLinearFeatureContext {
                                series_id: None,
                                timestamp: row.timestamp,
                                covariates: Some(&row.covariates),
                                horizon_step: None,
                                component_multiplier: fit_component_multiplier(
                                    t,
                                    &coefficients,
                                    &changepoints,
                                    bounds,
                                    config,
                                ),
                                changepoints: &changepoints,
                                config,
                                regressor_stats: Some(&regressor_stats),
                            },
                        )?,
                        bounds,
                        config,
                    ))
            })
            .collect::<Result<Vec<_>>>()?;
        let trend_delta_scale = piecewise_trend_delta_scale(&coefficients, changepoints.len());
        Ok(Self {
            start_timestamp,
            last_timestamp,
            last_elapsed_days: elapsed.last().copied().unwrap_or(0.0),
            changepoints,
            coefficients,
            coefficient_covariance: fit_result.coefficient_covariance,
            feature_count,
            residuals,
            transformed_residual_scale,
            trend_delta_scale,
            regressor_stats,
        })
    }

    fn predict_component_record(
        &self,
        series_id: &str,
        elapsed_days: f64,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Value> {
        debug_assert_eq!(self.feature_count, self.coefficients.len());
        let bounds = piecewise_bounds(Some(series_id), None, Some(step), config)?;
        let component_multiplier = self.component_multiplier(elapsed_days, bounds, config);
        let context = PiecewiseLinearFeatureContext {
            series_id: Some(series_id),
            timestamp,
            covariates: None,
            horizon_step: Some(step),
            component_multiplier,
            changepoints: &self.changepoints,
            config,
            regressor_stats: Some(&self.regressor_stats),
        };
        let mut features =
            vec![0.0; piecewise_linear_seasonal_feature_count(config, self.changepoints.len())];
        fill_piecewise_linear_seasonal_features(&mut features, elapsed_days, &context)?;
        let linear_predictor = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<f64>();
        let trend_linear =
            piecewise_trend_value(elapsed_days, &self.coefficients, &self.changepoints, config);
        let trend = inverse_piecewise_target(trend_linear, bounds, config);
        let trend_adjustment_multiplier = piecewise_trend_adjustment_multiplier(
            series_id,
            step,
            &config.trend_adjustments,
            &config.trend_adjustments_by_series,
        );
        let adjusted_trend = trend * trend_adjustment_multiplier;
        let trend_adjustment = adjusted_trend - trend;
        let residual_shock = self.residual_shock(step, config);
        let prediction = inverse_piecewise_target(linear_predictor, bounds, config)
            + trend_adjustment
            + residual_shock;
        let components = piecewise_component_contributions(
            &features,
            &self.coefficients,
            self.changepoints.len(),
            config,
        )?;
        Ok(json!({
            "series_id": series_id,
            "timestamp": timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "horizon": step,
            "prediction": prediction,
            "trend": trend,
            "adjusted_trend": adjusted_trend,
            "trend_adjustment_multiplier": trend_adjustment_multiplier,
            "trend_adjustment": trend_adjustment,
            "residual_shock": residual_shock,
            "linear_predictor": linear_predictor,
            "trend_linear": trend_linear,
            "component_scale": if config.growth == PiecewiseLinearGrowth::Logistic {
                "logistic_linear_predictor"
            } else {
                "prediction"
            },
            "components": components,
        }))
    }

    fn history_component_records(
        &self,
        series_id: &str,
        history: &[&ForecastRow],
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Vec<Value>> {
        let mut records = Vec::with_capacity(history.len());
        let mut previous_trend = None;
        let mut previous_fitted = None;
        for (idx, row) in history.iter().enumerate() {
            let record = self.history_component_record(
                series_id,
                row,
                idx,
                previous_trend,
                previous_fitted,
                config,
            )?;
            previous_trend = record["trend"].as_f64();
            previous_fitted = record["fitted"].as_f64();
            records.push(record);
        }
        Ok(records)
    }

    fn history_component_record(
        &self,
        series_id: &str,
        row: &ForecastRow,
        idx: usize,
        previous_trend: Option<f64>,
        previous_fitted: Option<f64>,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Value> {
        debug_assert_eq!(self.feature_count, self.coefficients.len());
        let elapsed = elapsed_days(self.start_timestamp, row.timestamp);
        let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
        let component_multiplier = self.component_multiplier(elapsed, bounds, config);
        let context = PiecewiseLinearFeatureContext {
            series_id: Some(series_id),
            timestamp: row.timestamp,
            covariates: Some(&row.covariates),
            horizon_step: None,
            component_multiplier,
            changepoints: &self.changepoints,
            config,
            regressor_stats: Some(&self.regressor_stats),
        };
        let mut features =
            vec![0.0; piecewise_linear_seasonal_feature_count(config, self.changepoints.len())];
        fill_piecewise_linear_seasonal_features(&mut features, elapsed, &context)?;
        let linear_predictor = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<f64>();
        let trend_linear =
            piecewise_trend_value(elapsed, &self.coefficients, &self.changepoints, config);
        let trend = inverse_piecewise_target(trend_linear, bounds, config);
        let fitted = inverse_piecewise_target(linear_predictor, bounds, config);
        let components = piecewise_component_contributions(
            &features,
            &self.coefficients,
            self.changepoints.len(),
            config,
        )?;
        Ok(json!({
            "series_id": series_id,
            "timestamp": row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "index": idx,
            "actual": row.target,
            "fitted": fitted,
            "residual": row.target - fitted,
            "trend": trend,
            "adjusted_trend": trend,
            "trend_adjustment_multiplier": 1.0,
            "trend_adjustment": 0.0,
            "trend_movement": previous_trend.map(|previous| trend - previous),
            "fitted_movement": previous_fitted.map(|previous| fitted - previous),
            "linear_predictor": linear_predictor,
            "trend_linear": trend_linear,
            "component_scale": if config.growth == PiecewiseLinearGrowth::Logistic {
                "logistic_linear_predictor"
            } else {
                "prediction"
            },
            "components": components,
        }))
    }

    fn residual_scale(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        let mse = self
            .residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>()
            / self.residuals.len() as f64;
        mse.sqrt()
    }

    fn trend_uncertainty_offsets(
        &self,
        series_id: &str,
        elapsed_days: f64,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Vec<f64>> {
        if config.uncertainty_samples == 0
            || config.trend_uncertainty_scale <= 0.0
            || self.trend_delta_scale <= 0.0
            || config.growth == PiecewiseLinearGrowth::Flat
        {
            return Ok(Vec::new());
        }
        let future_elapsed = (elapsed_days - self.last_elapsed_days).max(0.0);
        if future_elapsed <= 0.0 {
            return Ok(Vec::new());
        }
        let scale = self.trend_delta_scale
            * config.trend_uncertainty_scale
            * future_elapsed
            * (step as f64).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Ok(Vec::new());
        }
        let bounds = piecewise_bounds(Some(series_id), None, Some(step), config)?;
        let linear_predictor = predict_piecewise_linear_value(
            elapsed_days,
            &self.coefficients,
            &PiecewiseLinearFeatureContext {
                series_id: Some(series_id),
                timestamp,
                covariates: None,
                horizon_step: Some(step),
                component_multiplier: self.component_multiplier(elapsed_days, bounds, config),
                changepoints: &self.changepoints,
                config,
                regressor_stats: Some(&self.regressor_stats),
            },
        )?;
        let derivative =
            inverse_piecewise_target_derivative(linear_predictor, bounds, config).abs();
        if !derivative.is_finite() || derivative <= 0.0 {
            return Ok(Vec::new());
        }
        let series_hash = stable_hash64(series_id.as_bytes());
        Ok((0..config.uncertainty_samples)
            .map(|sample| {
                let draw = deterministic_trend_uncertainty_draw(
                    config.uncertainty_seed ^ series_hash,
                    step as u64,
                    sample as u64,
                    config.trend_uncertainty_policy,
                );
                draw * scale * derivative
            })
            .collect())
    }

    fn trend_uncertainty_linear_offsets(
        &self,
        series_id: &str,
        elapsed_days: f64,
        step: usize,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Vec<f64> {
        if config.uncertainty_samples == 0
            || config.trend_uncertainty_scale <= 0.0
            || self.trend_delta_scale <= 0.0
            || config.growth == PiecewiseLinearGrowth::Flat
        {
            return Vec::new();
        }
        let future_elapsed = (elapsed_days - self.last_elapsed_days).max(0.0);
        if future_elapsed <= 0.0 {
            return Vec::new();
        }
        let scale = self.trend_delta_scale
            * config.trend_uncertainty_scale
            * future_elapsed
            * (step as f64).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Vec::new();
        }
        let series_hash = stable_hash64(series_id.as_bytes());
        (0..config.uncertainty_samples)
            .map(|sample| {
                deterministic_trend_uncertainty_draw(
                    config.uncertainty_seed ^ series_hash,
                    step as u64,
                    sample as u64,
                    config.trend_uncertainty_policy,
                ) * scale
            })
            .collect()
    }

    fn prediction_terms_at(
        &self,
        series_id: &str,
        elapsed_days: f64,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        bounds: PiecewiseBounds,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<PiecewisePredictionTerms> {
        let context = PiecewiseLinearFeatureContext {
            series_id: Some(series_id),
            timestamp,
            covariates: None,
            horizon_step: Some(step),
            component_multiplier: self.component_multiplier(elapsed_days, bounds, config),
            changepoints: &self.changepoints,
            config,
            regressor_stats: Some(&self.regressor_stats),
        };
        let mut features =
            vec![0.0; piecewise_linear_seasonal_feature_count(config, self.changepoints.len())];
        fill_piecewise_linear_seasonal_features(&mut features, elapsed_days, &context)?;
        let linear_predictor = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<f64>();
        let mean = inverse_piecewise_target(linear_predictor, bounds, config);
        let trend_linear =
            piecewise_trend_value(elapsed_days, &self.coefficients, &self.changepoints, config);
        let trend = inverse_piecewise_target(trend_linear, bounds, config);
        let trend_adjustment_multiplier = piecewise_trend_adjustment_multiplier(
            series_id,
            step,
            &config.trend_adjustments,
            &config.trend_adjustments_by_series,
        );
        let adjusted_trend = trend * trend_adjustment_multiplier;
        let residual_shock = self.residual_shock(step, config);
        let adjusted_mean = mean + adjusted_trend - trend + residual_shock;
        let variance = if config.coefficient_uncertainty_scale > 0.0
            && !self.coefficient_covariance.is_empty()
            && self.transformed_residual_scale > 0.0
        {
            quadratic_form(&features, &self.coefficient_covariance).max(0.0)
        } else {
            0.0
        };
        let linear_coefficient_scale = if variance.is_finite() && variance > 0.0 {
            config.coefficient_uncertainty_scale * self.transformed_residual_scale * variance.sqrt()
        } else {
            0.0
        };
        let derivative = inverse_piecewise_target_derivative(linear_predictor, bounds, config);
        let coefficient_scale = linear_coefficient_scale * derivative.abs();
        Ok(PiecewisePredictionTerms {
            mean: adjusted_mean,
            linear_predictor,
            coefficient_scale: if coefficient_scale.is_finite() {
                coefficient_scale
            } else {
                0.0
            },
            linear_coefficient_scale: if linear_coefficient_scale.is_finite() {
                linear_coefficient_scale
            } else {
                0.0
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn predictive_sample_record(
        &self,
        series_id: &str,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        sample: usize,
        mean: f64,
        linear_predictor: f64,
        bounds: PiecewiseBounds,
        residual_scale: f64,
        coefficient_scale: f64,
        linear_coefficient_scale: f64,
        trend_draw: f64,
        linear_trend_draw: f64,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Value {
        let series_hash = stable_hash64(series_id.as_bytes());
        let residual_z = deterministic_standard_normal(
            config.uncertainty_seed ^ series_hash ^ 0x8f6b_3c2d_19e0_a457,
            step as u64,
            sample as u64,
        );
        let coefficient_z = deterministic_standard_normal(
            config.uncertainty_seed ^ series_hash ^ 0x4d31_2f9a_b875_c913,
            step as u64,
            sample as u64,
        );
        let residual_draw = residual_scale * residual_z;
        let coefficient_draw = coefficient_scale * coefficient_z;
        let prediction = match config.growth {
            PiecewiseLinearGrowth::Logistic => clamp_piecewise_logistic_interior_value(
                inverse_piecewise_target(
                    linear_predictor
                        + self.transformed_residual_scale * residual_z
                        + linear_coefficient_scale * coefficient_z
                        + linear_trend_draw,
                    bounds,
                    config,
                ),
                bounds,
                config,
            ),
            PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => {
                mean + residual_draw + coefficient_draw + trend_draw
            }
        };
        json!({
            "series_id": series_id,
            "timestamp": timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "horizon": step,
            "sample": sample,
            "prediction": prediction,
            "mean": mean,
            "residual_draw": residual_draw,
            "coefficient_draw": coefficient_draw,
            "trend_draw": trend_draw,
        })
    }

    fn component_multiplier(
        &self,
        elapsed_days: f64,
        bounds: PiecewiseBounds,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> f64 {
        match config.component_mode {
            PiecewiseLinearComponentMode::Additive => 1.0,
            PiecewiseLinearComponentMode::Multiplicative => {
                piecewise_component_multiplier_from_trend(
                    self.trend_component(elapsed_days, config),
                    bounds,
                    config,
                )
            }
        }
    }

    fn trend_component(&self, elapsed_days: f64, config: &PiecewiseLinearSeasonalConfig) -> f64 {
        piecewise_trend_value(elapsed_days, &self.coefficients, &self.changepoints, config)
    }

    fn residual_shock(&self, step: usize, config: &PiecewiseLinearSeasonalConfig) -> f64 {
        if config.residual_shock_window == 0 || config.residual_shock_scale <= 0.0 {
            return 0.0;
        }
        let window = config.residual_shock_window.min(self.residuals.len());
        if window == 0 {
            return 0.0;
        }
        let recent = &self.residuals[self.residuals.len() - window..];
        let first = recent[0];
        if first == 0.0 || !first.is_finite() {
            return 0.0;
        }
        let sign = first.signum();
        if recent
            .iter()
            .any(|residual| !residual.is_finite() || *residual == 0.0 || residual.signum() != sign)
        {
            return 0.0;
        }
        let average = recent.iter().sum::<f64>() / window as f64;
        average
            * config.residual_shock_scale
            * config
                .residual_shock_decay
                .powi(step.saturating_sub(1) as i32)
    }
}

fn piecewise_trend_adjustment_multiplier(
    series_id: &str,
    step: usize,
    global: &BTreeMap<usize, f64>,
    by_series: &BTreeMap<String, BTreeMap<usize, f64>>,
) -> f64 {
    by_series
        .get(series_id)
        .and_then(|values| values.get(&step))
        .or_else(|| global.get(&step))
        .copied()
        .unwrap_or(1.0)
}

#[allow(clippy::too_many_arguments)]
fn piecewise_prediction_intervals(
    prediction: &ForecastPrediction,
    residual_scale: f64,
    coefficient_scale: f64,
    linear_residual_scale: f64,
    linear_predictor: f64,
    linear_coefficient_scale: f64,
    mut trend_offsets: Vec<f64>,
    mut linear_trend_offsets: Vec<f64>,
    levels: &[f64],
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Vec<ForecastIntervalPrediction>> {
    if levels.is_empty() {
        return Ok(Vec::new());
    }
    if !residual_scale.is_finite() || residual_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal residual scale must be finite and nonnegative".to_string(),
        ));
    }
    if !coefficient_scale.is_finite() || coefficient_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal coefficient uncertainty scale must be finite and nonnegative"
                .to_string(),
        ));
    }
    let predictive_scale =
        (residual_scale * residual_scale + coefficient_scale * coefficient_scale).sqrt();
    let linear_predictive_scale = (linear_residual_scale * linear_residual_scale
        + linear_coefficient_scale * linear_coefficient_scale)
        .sqrt();
    if !trend_offsets.is_empty() {
        trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    if !linear_trend_offsets.is_empty() {
        linear_trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    levels
        .iter()
        .map(|&level| {
            let (lower, upper) = if config.growth == PiecewiseLinearGrowth::Logistic {
                piecewise_logistic_interval_bounds(
                    linear_predictor,
                    linear_predictive_scale,
                    &linear_trend_offsets,
                    level,
                    bounds,
                    config,
                )
            } else if trend_offsets.is_empty() {
                let alpha = (1.0 + level) / 2.0;
                let width = inverse_standard_normal_cdf(alpha) * predictive_scale;
                (prediction.mean - width, prediction.mean + width)
            } else {
                piecewise_sampled_interval_bounds(
                    prediction.mean,
                    predictive_scale,
                    &trend_offsets,
                    level,
                )
            };
            let (lower, upper) = clamp_piecewise_interval_bounds(lower, upper, bounds, config);
            Ok(ForecastIntervalPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: prediction.model.clone(),
                level,
                lower,
                upper,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn piecewise_prediction_quantiles(
    prediction: &ForecastPrediction,
    residual_scale: f64,
    coefficient_scale: f64,
    linear_residual_scale: f64,
    linear_predictor: f64,
    linear_coefficient_scale: f64,
    mut trend_offsets: Vec<f64>,
    mut linear_trend_offsets: Vec<f64>,
    levels: &[f64],
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> Vec<Value> {
    let predictive_scale =
        (residual_scale * residual_scale + coefficient_scale * coefficient_scale).sqrt();
    let linear_predictive_scale = (linear_residual_scale * linear_residual_scale
        + linear_coefficient_scale * linear_coefficient_scale)
        .sqrt();
    if !trend_offsets.is_empty() {
        trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    if !linear_trend_offsets.is_empty() {
        linear_trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    levels
        .iter()
        .map(|&level| {
            let z = inverse_standard_normal_cdf(level);
            let value = if config.growth == PiecewiseLinearGrowth::Logistic {
                let linear_value = if linear_trend_offsets.is_empty() {
                    linear_predictor + z * linear_predictive_scale
                } else {
                    linear_predictor
                        + quantile_from_sorted(&linear_trend_offsets, level)
                        + z * linear_predictive_scale
                };
                inverse_piecewise_target(linear_value, bounds, config)
            } else if trend_offsets.is_empty() {
                prediction.mean + z * predictive_scale
            } else {
                prediction.mean + quantile_from_sorted(&trend_offsets, level) + z * predictive_scale
            };
            let value = clamp_piecewise_logistic_interior_value(value, bounds, config);
            json!({
                "series_id": prediction.series_id.clone(),
                "timestamp": prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                "horizon": prediction.horizon,
                "quantile": level,
                "prediction": value,
                "mean": prediction.mean,
            })
        })
        .collect()
}

fn clamp_piecewise_logistic_interior_value(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    if config.growth != PiecewiseLinearGrowth::Logistic {
        return value;
    }
    let cap = bounds.cap.expect("validated logistic cap");
    let span = cap - bounds.floor;
    let epsilon = (span.abs() * 1.0e-12).max(f64::EPSILON);
    value.max(bounds.floor + epsilon).min(cap - epsilon)
}

fn clamp_piecewise_interval_bounds(
    lower: f64,
    upper: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> (f64, f64) {
    if config.growth != PiecewiseLinearGrowth::Logistic {
        return (lower, upper);
    }
    let cap = bounds.cap.expect("validated logistic cap");
    (lower.max(bounds.floor), upper.min(cap))
}

fn piecewise_logistic_interval_bounds(
    linear_predictor: f64,
    linear_predictive_scale: f64,
    sorted_linear_trend_offsets: &[f64],
    level: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> (f64, f64) {
    let residual_width = inverse_standard_normal_cdf((1.0 + level) / 2.0) * linear_predictive_scale;
    let (linear_lower, linear_upper) = if sorted_linear_trend_offsets.is_empty() {
        (
            linear_predictor - residual_width,
            linear_predictor + residual_width,
        )
    } else {
        let lower_q = (1.0 - level) / 2.0;
        let upper_q = 1.0 - lower_q;
        (
            linear_predictor + quantile_from_sorted(sorted_linear_trend_offsets, lower_q)
                - residual_width,
            linear_predictor
                + quantile_from_sorted(sorted_linear_trend_offsets, upper_q)
                + residual_width,
        )
    };
    (
        inverse_piecewise_target(linear_lower, bounds, config),
        inverse_piecewise_target(linear_upper, bounds, config),
    )
}

fn piecewise_sampled_interval_bounds(
    mean: f64,
    residual_scale: f64,
    sorted_trend_offsets: &[f64],
    level: f64,
) -> (f64, f64) {
    let residual_width = inverse_standard_normal_cdf((1.0 + level) / 2.0) * residual_scale;
    let lower_q = (1.0 - level) / 2.0;
    let upper_q = 1.0 - lower_q;
    (
        mean + quantile_from_sorted(sorted_trend_offsets, lower_q) - residual_width,
        mean + quantile_from_sorted(sorted_trend_offsets, upper_q) + residual_width,
    )
}

fn quantile_from_sorted(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let bounded = probability.clamp(0.0, 1.0);
    let position = bounded * (values.len().saturating_sub(1)) as f64;
    let lower_idx = position.floor() as usize;
    let upper_idx = position.ceil() as usize;
    if lower_idx == upper_idx {
        values[lower_idx]
    } else {
        let weight = position - lower_idx as f64;
        values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight
    }
}

fn inverse_standard_normal_cdf(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if probability <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability >= 1.0 {
        return f64::INFINITY;
    }
    if probability < P_LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability <= P_HIGH {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

impl KalmanForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean: series.level + step as f64 * series.trend,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl LocalLevelKalmanForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean: series.level,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl FittedKalmanState {
    fn from_frame(
        frame: &ForecastFrame,
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedKalmanSeries::fit(
                        series_id,
                        history,
                        level_process_variance,
                        trend_process_variance,
                        observation_variance,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedLocalLevelKalmanState {
    fn from_frame(
        frame: &ForecastFrame,
        level_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedLocalLevelKalmanSeries::fit(
                        series_id,
                        history,
                        level_process_variance,
                        observation_variance,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedKalmanSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least one row for kalman forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let config = LocalLinearKalmanConfig::new(
            level_process_variance,
            trend_process_variance,
            observation_variance,
        )?;
        if values.len() == 1 {
            return Ok(Self {
                last_timestamp: history.last().expect("history length checked").timestamp,
                level: values[0],
                trend: 0.0,
            });
        }
        let result = fit_local_linear_kalman(&values, config)
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
        Ok(Self {
            last_timestamp: history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
                .timestamp,
            level: result.final_state.level,
            trend: result.final_state.trend,
        })
    }
}

impl FittedLocalLevelKalmanSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        level_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least one row for local level kalman forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let config = LocalLevelKalmanConfig::new(level_process_variance, observation_variance)?;
        let result = fit_local_level_kalman(&values, config)
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            level: result.final_level,
        })
    }
}

impl FittedKrigingState {
    fn from_frame(
        frame: &ForecastFrame,
        coordinates: &BTreeMap<String, (f64, f64)>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let mut levels = BTreeMap::new();
        for (series_id, history) in &local.history_by_series {
            if !coordinates.contains_key(series_id) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "missing kriging coordinate for series {series_id}"
                )));
            }
            let last = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?;
            levels.insert(series_id.clone(), last.target);
        }
        Ok(Self {
            frame: frame.clone(),
            levels,
        })
    }
}

impl FittedSpatialPiecewiseKrigingState {
    fn from_frame(frame: &ForecastFrame, config: &SpatialPiecewiseKrigingConfig) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        #[cfg(target_arch = "wasm32")]
        let started = ();
        let piecewise_config = spatial_piecewise_base_config(config);
        let modeled_frame = if config.uses_kriged_regressors() {
            kriged_regressor_frame(frame, config)?
        } else {
            frame.clone()
        };
        let mut base = PiecewiseLinearSeasonalForecaster::new(piecewise_config)?;
        base.fit(&modeled_frame)?;
        let base_fitted = base.fitted.as_ref().ok_or_else(not_fitted)?;
        let local = FittedLocalState::from_frame(&modeled_frame);
        let mut residual_levels = BTreeMap::new();
        let mut residual_observation_series = Vec::new();
        let mut cutoff_timestamps = BTreeMap::new();
        for (series_id, history) in &local.history_by_series {
            let fitted_series = base_fitted.series.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted base series for spatial piecewise kriging series {series_id}"
                ))
            })?;
            let last_residual = fitted_series.residuals.last().copied().ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "series {series_id} has no residuals for residual kriging"
                ))
            })?;
            if !last_residual.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} residual is not finite"
                )));
            }
            let last_timestamp = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
                .timestamp;
            residual_levels.insert(series_id.clone(), last_residual);
            residual_observation_series.push(series_id.clone());
            cutoff_timestamps.insert(series_id.clone(), last_timestamp);
        }
        residual_observation_series.sort();
        let residual_rmse = base_fitted.root_mean_squared_residual();
        let fit_metadata = json!({
            "cutoffs": cutoff_timestamps.iter().map(|(series_id, timestamp)| {
                (series_id.clone(), timestamp.format("%Y-%m-%dT%H:%M:%S").to_string())
            }).collect::<BTreeMap<_, _>>(),
            "residual_rmse": residual_rmse,
            "runtime_seconds": spatial_piecewise_runtime_seconds(&started),
        });
        Ok(Self {
            frame: modeled_frame,
            base,
            residual_levels,
            residual_observation_series,
            cutoff_timestamps,
            fit_metadata,
        })
    }

    fn residual_kriging_predictions(
        &self,
        config: &SpatialPiecewiseKrigingConfig,
    ) -> Result<BTreeMap<String, SpatialKrigingCorrection>> {
        let observations = self
            .residual_observation_series
            .iter()
            .map(|series_id| {
                let coord = config.coordinates.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing spatial piecewise kriging coordinate for series {series_id}"
                    ))
                })?;
                let value = self
                    .residual_levels
                    .get(series_id)
                    .copied()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing residual kriging level for series {series_id}"
                        ))
                    })?;
                Ok(KrigingObservation {
                    x: coord.0,
                    y: coord.1,
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let series_ids = self.frame.series_ids();
        let targets = series_ids
            .iter()
            .map(|series_id| {
                config.coordinates.get(series_id).copied().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing spatial piecewise kriging coordinate for series {series_id}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !config.allow_neighbor_fallback {
            let predictions =
                ordinary_kriging_predict_many(&observations, &targets, config.kriging_config)?;
            return Ok(series_ids
                .into_iter()
                .zip(predictions)
                .map(|(series_id, prediction)| {
                    (
                        series_id,
                        SpatialKrigingCorrection {
                            prediction,
                            used_neighbor_fallback: false,
                        },
                    )
                })
                .collect());
        }
        let mut corrections = BTreeMap::new();
        for (series_id, target) in series_ids.into_iter().zip(targets) {
            let correction =
                match ordinary_kriging_predict(&observations, target, config.kriging_config) {
                    Ok(prediction) => SpatialKrigingCorrection {
                        prediction,
                        used_neighbor_fallback: false,
                    },
                    Err(error) if is_neighbor_rule_error(&error) => SpatialKrigingCorrection {
                        prediction: crate::utilities::KrigingPrediction {
                            x: target.0,
                            y: target.1,
                            mean: 0.0,
                            variance: residual_level_variance(&self.residual_levels),
                            weights: Vec::new(),
                            neighbor_indices: Vec::new(),
                        },
                        used_neighbor_fallback: true,
                    },
                    Err(error) => return Err(error),
                };
            corrections.insert(series_id, correction);
        }
        Ok(corrections)
    }

    fn future_spatial_regressors(
        &self,
        horizon: usize,
        config: &SpatialPiecewiseKrigingConfig,
    ) -> Result<BTreeMap<String, BTreeMap<String, Vec<f64>>>> {
        let mut latest_by_series: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for row in self.frame.rows() {
            let entry = latest_by_series.entry(row.series_id.clone()).or_default();
            for name in &config.spatial_regressors {
                if let Some(value) = row.covariates.get(name) {
                    entry.insert(name.clone(), *value);
                }
            }
        }
        Ok(latest_by_series
            .into_iter()
            .map(|(series_id, values)| {
                let repeated = values
                    .into_iter()
                    .map(|(name, value)| (name, vec![value; horizon]))
                    .collect::<BTreeMap<_, _>>();
                (series_id, repeated)
            })
            .collect())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spatial_piecewise_runtime_seconds(started: &std::time::Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn spatial_piecewise_runtime_seconds(_: &()) -> f64 {
    0.0
}

impl FittedETSSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
        damping_phi: f64,
    ) -> Result<Self> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least one row for ETS forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let mut seasonals = match season_length {
            Some(length) => {
                if values.len() < length * 2 {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "series {series_id} requires at least two full seasonal cycles for ETS seasonality"
                    )));
                }
                let (_, pattern) = deseasonalize(
                    series_id,
                    &values,
                    Some(ThetaSeasonality::additive(length)?),
                )?;
                pattern
            }
            None => None,
        };
        if values.len() == 1 {
            let initial_seasonal = seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0);
            let level = values[0] - initial_seasonal;
            return Ok(Self {
                last_timestamp: history.last().expect("history length checked").timestamp,
                n_obs: history.len(),
                level,
                trend: 0.0,
                damping_phi,
                seasonals,
                fitted_values: vec![values[0]],
                residuals: vec![0.0],
                level_values: vec![level],
                trend_values: vec![0.0],
                seasonal_values: vec![initial_seasonal],
            });
        }

        let mut level = values[0] - seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0);
        let mut trend = initial_trend(&values, seasonals.as_deref());
        let mut fitted_values = Vec::with_capacity(values.len());
        let mut residuals = Vec::with_capacity(values.len());
        let mut level_values = Vec::with_capacity(values.len());
        let mut trend_values = Vec::with_capacity(values.len());
        let mut seasonal_values = Vec::with_capacity(values.len());
        fitted_values.push(values[0]);
        residuals.push(0.0);
        level_values.push(level);
        trend_values.push(trend);
        seasonal_values.push(seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0));

        for (idx, value) in values.iter().enumerate().skip(1) {
            let seasonal_idx = seasonals.as_ref().map(|seasonals| idx % seasonals.len());
            let seasonal = seasonal_idx
                .and_then(|seasonal_idx| {
                    seasonals.as_ref().map(|seasonals| seasonals[seasonal_idx])
                })
                .unwrap_or(0.0);
            let fitted = level + damping_phi * trend + seasonal;
            fitted_values.push(fitted);
            residuals.push(*value - fitted);

            let previous_level = level;
            level = alpha * (*value - seasonal) + (1.0 - alpha) * (level + damping_phi * trend);
            trend = beta * (level - previous_level) + (1.0 - beta) * damping_phi * trend;
            if let (Some(gamma), Some(seasonal_idx), Some(seasonals)) =
                (gamma, seasonal_idx, seasonals.as_mut())
            {
                seasonals[seasonal_idx] =
                    gamma * (*value - level) + (1.0 - gamma) * seasonals[seasonal_idx];
            }
            level_values.push(level);
            trend_values.push(trend);
            seasonal_values.push(seasonal);
        }

        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            n_obs: history.len(),
            level,
            trend,
            damping_phi,
            seasonals,
            fitted_values,
            residuals,
            level_values,
            trend_values,
            seasonal_values,
        })
    }
}

impl FittedArimaSeries {
    fn fit(series_id: &str, history: &[ForecastRow], p: usize, d: usize, q: usize) -> Result<Self> {
        validate_arima_order(p, d, q)?;
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let (effective_p, effective_d, effective_q) =
            arima_order_supported_by_history(values.len(), p, d, q);
        let differences = difference_series(&values, effective_d)?;
        let required_lags = effective_p.max(effective_q);
        if differences.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has no differenced rows available for ARIMA fitting",
            )));
        }
        let (intercept, ar_coefficients, ma_coefficients, fitted_diff, residuals) =
            fit_arima_components(&differences, effective_p, effective_q)?;
        let fitted_values = undifference_fitted_values(&values, &fitted_diff, effective_d);
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            intercept,
            ar_coefficients,
            ma_coefficients,
            score_start: required_lags,
            differenced_history: differences,
            residual_history: residuals.clone(),
            last_differences: last_differences(&values, effective_d)?,
            fitted_values,
            residuals,
        })
    }

    fn forecast_values(&self, horizon: usize) -> Vec<f64> {
        let p = self.ar_coefficients.len();
        let q = self.ma_coefficients.len();
        let mut differenced = tail_values(&self.differenced_history, p);
        let mut residuals = tail_values(&self.residual_history, q);
        let mut levels = self.last_differences.clone();
        let mut forecasts = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let next_diff = forecast_arima_next(
                &differenced,
                &residuals,
                self.intercept,
                &self.ar_coefficients,
                &self.ma_coefficients,
            );
            push_tail(&mut differenced, p, next_diff);
            push_tail(&mut residuals, q, 0.0);
            let mut value = next_diff;
            for idx in (0..(levels.len() - 1)).rev() {
                levels[idx] += value;
                value = levels[idx];
            }
            forecasts.push(value);
        }
        forecasts
    }
}

impl FittedThetaState {
    fn from_frame(
        frame: &ForecastFrame,
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let effective_seasonality =
            effective_theta_seasonality(&local.history_by_series, seasonality);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedThetaSeries::fit(
                        series_id,
                        history,
                        theta,
                        alpha,
                        effective_seasonality,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

impl FittedThetaSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for theta forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let (adjusted, pattern) = deseasonalize(series_id, &values, seasonality)?;
        let component = fit_theta_component(&adjusted, theta, alpha);
        let fitted_adjusted = fitted_theta_values(&adjusted, alpha);
        let mut fitted_values = Vec::with_capacity(values.len());
        let mut residuals = Vec::with_capacity(values.len());
        for (idx, fitted) in fitted_adjusted.into_iter().enumerate() {
            let reseasonalized = reseasonalize_value(fitted, idx, seasonality, pattern.as_deref())?;
            fitted_values.push(reseasonalized);
            residuals.push(values[idx] - reseasonalized);
        }
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            n_obs: history.len(),
            component,
            seasonal_pattern: pattern,
            fitted_values,
            residuals,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("theta grid scores are finite")
    }
}

fn validate_theta_params(theta: f64, alpha: f64) -> Result<()> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "theta must be a positive finite value".to_string(),
        ));
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "alpha must be finite and in (0, 1]".to_string(),
        ));
    }
    Ok(())
}

fn validate_piecewise_linear_seasonal_config(config: &PiecewiseLinearSeasonalConfig) -> Result<()> {
    if !config.changepoint_range.is_finite()
        || config.changepoint_range <= 0.0
        || config.changepoint_range > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_range must be finite and in (0, 1]".to_string(),
        ));
    }
    if !config.changepoint_l2_regularization.is_finite()
        || config.changepoint_l2_regularization < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.changepoint_l1_regularization.is_finite()
        || config.changepoint_l1_regularization < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_l1_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.huber_delta.is_finite() || config.huber_delta <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "huber_delta must be a finite positive value".to_string(),
        ));
    }
    if !config.seasonality_l2_regularization.is_finite()
        || config.seasonality_l2_regularization < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "seasonality_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    validate_optional_nonnegative(config.yearly_l2_regularization, "yearly_l2_regularization")?;
    validate_optional_nonnegative(config.weekly_l2_regularization, "weekly_l2_regularization")?;
    validate_optional_nonnegative(config.daily_l2_regularization, "daily_l2_regularization")?;
    if !config.event_l2_regularization.is_finite() || config.event_l2_regularization < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "event_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.regressor_l2_regularization.is_finite() || config.regressor_l2_regularization < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "regressor_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.trend_uncertainty_scale.is_finite() || config.trend_uncertainty_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "trend_uncertainty_scale must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.coefficient_uncertainty_scale.is_finite()
        || config.coefficient_uncertainty_scale < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "coefficient_uncertainty_scale must be a finite nonnegative value".to_string(),
        ));
    }
    let mut changepoint_timestamps = config.changepoint_timestamps.clone();
    changepoint_timestamps.sort();
    if changepoint_timestamps
        .windows(2)
        .any(|window| window[0] == window[1])
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_timestamps must be unique".to_string(),
        ));
    }
    let mut seasonality_names = BTreeSet::new();
    for seasonality in &config.custom_seasonalities {
        if seasonality.name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "custom seasonality names must not be empty".to_string(),
            ));
        }
        if !seasonality_names.insert(seasonality.name.clone()) {
            return Err(CartoBoostError::InvalidInput(
                "custom seasonality names must be unique".to_string(),
            ));
        }
        if !seasonality.period_days.is_finite() || seasonality.period_days <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "custom seasonality {:?} period_days must be a positive finite value",
                seasonality.name
            )));
        }
        if seasonality.fourier_order == 0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "custom seasonality {:?} fourier_order must be positive",
                seasonality.name
            )));
        }
        if let Some(condition_name) = &seasonality.condition_name {
            if condition_name.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "custom seasonality {:?} condition_name must not be empty",
                    seasonality.name
                )));
            }
        }
        if let Some(value) = seasonality.l2_regularization {
            if !value.is_finite() || value < 0.0 {
                return Err(CartoBoostError::InvalidInput(format!(
                    "custom seasonality {:?} l2_regularization must be a finite nonnegative value",
                    seasonality.name
                )));
            }
        }
    }
    for event in &config.events {
        if event.name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "piecewise seasonal event names must not be empty".to_string(),
            ));
        }
        if event.lower_window > event.upper_window {
            return Err(CartoBoostError::InvalidInput(format!(
                "piecewise seasonal event {:?} lower_window must be <= upper_window",
                event.name
            )));
        }
    }
    let mut regressors = config.extra_regressors.clone();
    regressors.sort();
    for name in &regressors {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "extra regressor names must not be empty".to_string(),
            ));
        }
    }
    if regressors.windows(2).any(|window| window[0] == window[1]) {
        return Err(CartoBoostError::InvalidInput(
            "extra regressor names must be unique".to_string(),
        ));
    }
    for name in config.regressor_modes.keys() {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "regressor mode names must not be empty".to_string(),
            ));
        }
        if !regressors.iter().any(|regressor| regressor == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "regressor mode {name:?} does not match an extra regressor"
            )));
        }
    }
    for (name, direction) in &config.extra_regressor_monotonic_constraints {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "extra regressor monotonic constraint names must not be empty".to_string(),
            ));
        }
        if !regressors.iter().any(|regressor| regressor == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "extra regressor monotonic constraint {name:?} does not match an extra regressor"
            )));
        }
        if !matches!(*direction, -1..=1) {
            return Err(CartoBoostError::InvalidInput(
                "extra regressor monotonic constraints must be -1, 0, or 1".to_string(),
            ));
        }
    }
    for (name, values) in &config.future_regressors {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "future regressor names must not be empty".to_string(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "future regressor {name:?} values must be finite"
            )));
        }
    }
    for (series_id, regressors_by_name) in &config.future_regressors_by_series {
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "future regressor series ids must not be empty".to_string(),
            ));
        }
        for (name, values) in regressors_by_name {
            if name.is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "per-series future regressor names must not be empty".to_string(),
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "per-series future regressor {name:?} for series {series_id:?} values must be finite"
                )));
            }
        }
    }
    validate_piecewise_trend_adjustments(&config.trend_adjustments, "trend_adjustments")?;
    for (series_id, adjustments) in &config.trend_adjustments_by_series {
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "trend adjustment series ids must not be empty".to_string(),
            ));
        }
        validate_piecewise_trend_adjustments(
            adjustments,
            &format!("trend_adjustments_by_series[{series_id:?}]"),
        )?;
    }
    if !config.residual_shock_scale.is_finite() || config.residual_shock_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "residual_shock_scale must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.residual_shock_decay.is_finite()
        || config.residual_shock_decay < 0.0
        || config.residual_shock_decay > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "residual_shock_decay must be finite and in [0, 1]".to_string(),
        ));
    }
    let event_names = piecewise_event_names(config);
    for (name, value) in &config.event_l2_regularization_by_name {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "event l2 regularization names must not be empty".to_string(),
            ));
        }
        if !event_names.iter().any(|event_name| event_name == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "event l2 regularization {name:?} does not match a configured event"
            )));
        }
        if !value.is_finite() || *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "event l2 regularization {name:?} must be a finite nonnegative value"
            )));
        }
    }
    for (name, value) in &config.regressor_l2_regularization_by_name {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "regressor l2 regularization names must not be empty".to_string(),
            ));
        }
        if !regressors.iter().any(|regressor| regressor == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "regressor l2 regularization {name:?} does not match an extra regressor"
            )));
        }
        if !value.is_finite() || *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "regressor l2 regularization {name:?} must be a finite nonnegative value"
            )));
        }
    }
    for level in &config.interval_levels {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(CartoBoostError::InvalidInput(
                "prediction interval levels must be finite and in (0, 1)".to_string(),
            ));
        }
    }
    let mut interval_levels = config.interval_levels.clone();
    interval_levels.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("interval levels are finite after validation")
    });
    if interval_levels
        .windows(2)
        .any(|window| (window[0] - window[1]).abs() < 1.0e-12)
    {
        return Err(CartoBoostError::InvalidInput(
            "prediction interval levels must be unique".to_string(),
        ));
    }
    validate_piecewise_quantile_levels(&config.quantile_levels)?;
    if !config.floor.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal floor must be finite".to_string(),
        ));
    }
    if let Some(name) = &config.cap_regressor {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "cap_regressor must not be empty".to_string(),
            ));
        }
    }
    if let Some(name) = &config.floor_regressor {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "floor_regressor must not be empty".to_string(),
            ));
        }
    }
    if let PiecewiseLinearGrowth::Logistic = config.growth {
        if config.cap.is_none() && config.cap_regressor.is_none() {
            return Err(CartoBoostError::InvalidInput(
                "logistic piecewise seasonal growth requires cap or cap_regressor".to_string(),
            ));
        }
        if let Some(cap) = config.cap {
            if !cap.is_finite() || cap <= config.floor {
                return Err(CartoBoostError::InvalidInput(
                    "logistic piecewise seasonal cap must be finite and greater than floor"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_piecewise_trend_adjustments(
    adjustments: &BTreeMap<usize, f64>,
    name: &str,
) -> Result<()> {
    for (horizon, multiplier) in adjustments {
        if *horizon == 0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} horizon keys must be positive"
            )));
        }
        if !multiplier.is_finite() || *multiplier <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} multipliers must be positive finite values"
            )));
        }
    }
    Ok(())
}

fn validate_optional_nonnegative(value: Option<f64>, name: &str) -> Result<()> {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} must be a finite nonnegative value"
            )));
        }
    }
    Ok(())
}

fn validate_piecewise_quantile_levels(levels: &[f64]) -> Result<()> {
    for level in levels {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(CartoBoostError::InvalidInput(
                "quantile levels must be finite and in (0, 1)".to_string(),
            ));
        }
    }
    let mut sorted = levels.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted
        .windows(2)
        .any(|window| (window[0] - window[1]).abs() < 1.0e-12)
    {
        return Err(CartoBoostError::InvalidInput(
            "quantile levels must be unique".to_string(),
        ));
    }
    Ok(())
}

fn resolve_piecewise_auto_seasonalities(
    frame: &ForecastFrame,
    config: &PiecewiseLinearSeasonalConfig,
) -> PiecewiseLinearSeasonalConfig {
    let mut resolved = config.clone();
    let Some((start, end)) = forecast_frame_timestamp_span(frame) else {
        return resolved;
    };
    let span_days = (end - start).num_seconds() as f64 / 86_400.0;
    if resolved.auto_yearly_seasonality && resolved.yearly_fourier_order == 0 && span_days >= 730.0
    {
        resolved.yearly_fourier_order = 10;
    }
    if resolved.auto_weekly_seasonality
        && resolved.weekly_fourier_order == 0
        && matches!(
            frame.frequency(),
            ForecastFrequency::Daily | ForecastFrequency::Hourly
        )
        && span_days >= 14.0
    {
        resolved.weekly_fourier_order = 3;
    }
    if resolved.auto_daily_seasonality
        && resolved.daily_fourier_order == 0
        && frame.frequency() == ForecastFrequency::Hourly
        && span_days >= 2.0
    {
        resolved.daily_fourier_order = 4;
    }
    resolved
}

fn forecast_frame_timestamp_span(
    frame: &ForecastFrame,
) -> Option<(chrono::NaiveDateTime, chrono::NaiveDateTime)> {
    let mut iter = frame.rows().iter().map(|row| row.timestamp);
    let first = iter.next()?;
    let (min_timestamp, max_timestamp) = iter.fold(
        (first, first),
        |(min_timestamp, max_timestamp), timestamp| {
            (min_timestamp.min(timestamp), max_timestamp.max(timestamp))
        },
    );
    Some((min_timestamp, max_timestamp))
}

fn elapsed_days(start_timestamp: chrono::NaiveDateTime, timestamp: chrono::NaiveDateTime) -> f64 {
    (timestamp - start_timestamp).num_seconds() as f64 / 86_400.0
}

fn select_piecewise_changepoints(
    series_id: &str,
    start_timestamp: chrono::NaiveDateTime,
    last_timestamp: chrono::NaiveDateTime,
    elapsed: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Vec<f64>> {
    if !config.changepoint_timestamps.is_empty() {
        let mut changepoints = config
            .changepoint_timestamps
            .iter()
            .map(|&timestamp| {
                if timestamp <= start_timestamp || timestamp >= last_timestamp {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "changepoint timestamp {timestamp} must be inside the training range for series {series_id:?}"
                    )));
                }
                Ok(elapsed_days(start_timestamp, timestamp))
            })
            .collect::<Result<Vec<_>>>()?;
        changepoints.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("explicit changepoint elapsed values are finite")
        });
        return Ok(changepoints);
    }
    Ok(select_even_changepoints(
        elapsed,
        config.changepoints,
        config.changepoint_range,
    ))
}

fn select_even_changepoints(elapsed: &[f64], requested: usize, changepoint_range: f64) -> Vec<f64> {
    if requested == 0 || elapsed.len() <= 2 {
        return Vec::new();
    }
    let last_idx = elapsed.len() - 1;
    let cutoff_idx = ((last_idx as f64) * changepoint_range).floor().max(1.0) as usize;
    let cutoff_idx = cutoff_idx.min(last_idx.saturating_sub(1));
    let count = requested.min(cutoff_idx.saturating_sub(1).max(1));
    (1..=count)
        .map(|idx| {
            let position = (idx * cutoff_idx) / (count + 1);
            elapsed[position.max(1).min(cutoff_idx)]
        })
        .collect()
}

fn fit_piecewise_trend_coefficients(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Vec<f64>> {
    if !piecewise_uses_multiplicative_components(config) {
        return Ok(Vec::new());
    }
    let feature_count = 2 + changepoints.len();
    let mut xtx = vec![vec![0.0; feature_count]; feature_count];
    let mut xty = vec![0.0; feature_count];
    let mut features = vec![0.0; feature_count];
    for (row, &t) in history.iter().zip(elapsed.iter()) {
        let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
        let target = transform_piecewise_target(row.target, bounds, config)?;
        fill_piecewise_trend_features(&mut features, t, changepoints, config);
        for i in 0..feature_count {
            xty[i] += features[i] * target;
            for j in i..feature_count {
                xtx[i][j] += features[i] * features[j];
            }
        }
    }
    for i in 0..feature_count {
        let (previous_rows, current_and_after) = xtx.split_at_mut(i);
        let current_row = &mut current_and_after[0];
        for (j, previous_row) in previous_rows.iter().enumerate() {
            current_row[j] = previous_row[i];
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        let _ = idx;
        row[idx] += config.changepoint_l2_regularization;
    }
    solve_piecewise_linear_coefficients(xtx, xty, changepoints.len(), config).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "could not solve piecewise linear seasonal trend prefit normal equations".to_string(),
        )
    })
}

fn piecewise_uses_multiplicative_components(config: &PiecewiseLinearSeasonalConfig) -> bool {
    config.component_mode == PiecewiseLinearComponentMode::Multiplicative
        || config.custom_seasonalities.iter().any(|seasonality| {
            seasonality.mode == Some(PiecewiseLinearComponentMode::Multiplicative)
        })
        || config.event_mode == Some(PiecewiseLinearComponentMode::Multiplicative)
        || config
            .regressor_modes
            .values()
            .any(|mode| *mode == PiecewiseLinearComponentMode::Multiplicative)
}

fn fill_piecewise_trend_features(
    features: &mut [f64],
    elapsed_days: f64,
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) {
    features.fill(0.0);
    features[0] = 1.0;
    if config.growth == PiecewiseLinearGrowth::Flat {
        return;
    }
    features[1] = elapsed_days;
    for (idx, &changepoint) in changepoints.iter().enumerate() {
        features[2 + idx] = (elapsed_days - changepoint).max(0.0);
    }
}

fn fit_component_multiplier(
    elapsed_days: f64,
    coefficients: &[f64],
    changepoints: &[f64],
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    match config.component_mode {
        PiecewiseLinearComponentMode::Additive => 1.0,
        PiecewiseLinearComponentMode::Multiplicative => piecewise_component_multiplier_from_trend(
            piecewise_trend_value(elapsed_days, coefficients, changepoints, config),
            bounds,
            config,
        ),
    }
}

fn piecewise_component_multiplier_from_trend(
    trend_linear: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    let trend = match config.growth {
        PiecewiseLinearGrowth::Logistic => inverse_piecewise_target(trend_linear, bounds, config),
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => trend_linear,
    };
    trend.abs().max(1.0e-9)
}

fn piecewise_trend_value(
    elapsed_days: f64,
    coefficients: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    let mut value = coefficients.first().copied().unwrap_or(0.0)
        + if config.growth == PiecewiseLinearGrowth::Flat {
            0.0
        } else {
            coefficients.get(1).copied().unwrap_or(0.0) * elapsed_days
        };
    if config.growth == PiecewiseLinearGrowth::Flat {
        return value;
    }
    for (idx, &changepoint) in changepoints.iter().enumerate() {
        value += coefficients.get(2 + idx).copied().unwrap_or(0.0)
            * (elapsed_days - changepoint).max(0.0);
    }
    value
}

fn piecewise_linear_seasonal_feature_count(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> usize {
    2 + changepoint_count
        + 2 * config.yearly_fourier_order
        + 2 * config.weekly_fourier_order
        + 2 * config.daily_fourier_order
        + 2 * config
            .custom_seasonalities
            .iter()
            .map(|seasonality| seasonality.fourier_order)
            .sum::<usize>()
        + piecewise_event_terms(config).len()
        + config.extra_regressors.len()
}

fn fill_piecewise_linear_seasonal_features(
    features: &mut [f64],
    elapsed_days: f64,
    context: &PiecewiseLinearFeatureContext<'_>,
) -> Result<()> {
    features.fill(0.0);
    features[0] = 1.0;
    features[1] = elapsed_days;
    let mut col = 2;
    if context.config.growth == PiecewiseLinearGrowth::Flat {
        features[1] = 0.0;
    }
    for &changepoint in context.changepoints {
        features[col] = if context.config.growth == PiecewiseLinearGrowth::Flat {
            0.0
        } else {
            (elapsed_days - changepoint).max(0.0)
        };
        col += 1;
    }
    let seasonal_start = col;
    col = fill_fourier_terms(
        features,
        col,
        elapsed_days,
        365.25,
        context.config.yearly_fourier_order,
    );
    col = fill_fourier_terms(
        features,
        col,
        elapsed_days,
        7.0,
        context.config.weekly_fourier_order,
    );
    col = fill_fourier_terms(
        features,
        col,
        elapsed_days,
        1.0,
        context.config.daily_fourier_order,
    );
    apply_component_mode_to_features(
        features,
        seasonal_start,
        col,
        context.component_multiplier,
        context.config.component_mode,
    );
    for seasonality in &context.config.custom_seasonalities {
        let seasonality_start = col;
        col = fill_fourier_terms(
            features,
            col,
            elapsed_days,
            seasonality.period_days,
            seasonality.fourier_order,
        );
        apply_component_mode_to_features(
            features,
            seasonality_start,
            col,
            context.component_multiplier,
            seasonality.mode.unwrap_or(context.config.component_mode),
        );
        if !piecewise_seasonality_condition_is_active(seasonality, context)? {
            for feature in features.iter_mut().take(col).skip(seasonality_start) {
                *feature = 0.0;
            }
        }
    }
    for term in piecewise_event_terms(context.config) {
        features[col] = if event_term_is_active(context.timestamp, &term, &context.config.events) {
            component_multiplier_for_mode(
                context
                    .config
                    .event_mode
                    .unwrap_or(context.config.component_mode),
                context.component_multiplier,
            )
        } else {
            0.0
        };
        col += 1;
    }
    for name in &context.config.extra_regressors {
        let mode = context
            .config
            .regressor_modes
            .get(name)
            .copied()
            .unwrap_or(context.config.component_mode);
        features[col] = component_multiplier_for_mode(mode, context.component_multiplier)
            * piecewise_extra_regressor_value(
                name,
                context.series_id,
                context.covariates,
                context.horizon_step,
                context.config,
                context.regressor_stats,
            )?;
        col += 1;
    }
    Ok(())
}

fn apply_component_mode_to_features(
    features: &mut [f64],
    start: usize,
    end: usize,
    component_multiplier: f64,
    mode: PiecewiseLinearComponentMode,
) {
    let multiplier = component_multiplier_for_mode(mode, component_multiplier);
    for feature in features.iter_mut().take(end).skip(start) {
        *feature *= multiplier;
    }
}

fn component_multiplier_for_mode(mode: PiecewiseLinearComponentMode, multiplier: f64) -> f64 {
    match mode {
        PiecewiseLinearComponentMode::Additive => 1.0,
        PiecewiseLinearComponentMode::Multiplicative => multiplier,
    }
}

fn fill_fourier_terms(
    features: &mut [f64],
    mut col: usize,
    elapsed_days: f64,
    period: f64,
    order: usize,
) -> usize {
    for harmonic in 1..=order {
        let angle = std::f64::consts::TAU * harmonic as f64 * elapsed_days / period;
        features[col] = angle.sin();
        features[col + 1] = angle.cos();
        col += 2;
    }
    col
}

fn apply_piecewise_linear_ridge(
    xtx: &mut [Vec<f64>],
    changepoint_count: usize,
    config: &PiecewiseLinearSeasonalConfig,
) {
    let penalties = piecewise_linear_l2_penalties(config, changepoint_count);
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        let penalty = penalties.get(idx).copied().unwrap_or(0.0);
        row[idx] += penalty;
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_piecewise_linear_weighted_coefficients(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: &BTreeMap<String, PiecewiseLinearRegressorStats>,
    trend_coefficients: &[f64],
    weights: Option<&[f64]>,
    compute_covariance: bool,
) -> Result<Option<PiecewiseLinearFitResult>> {
    if let Some(weights) = weights {
        if weights.len() != history.len() {
            return Err(CartoBoostError::InvalidInput(
                "piecewise robust fit received mismatched weight count".to_string(),
            ));
        }
    }
    let feature_count = piecewise_linear_seasonal_feature_count(config, changepoints.len());
    let mut xtx = vec![vec![0.0; feature_count]; feature_count];
    let mut xty = vec![0.0; feature_count];
    let mut features = vec![0.0; feature_count];
    for (idx, (row, &t)) in history.iter().zip(elapsed.iter()).enumerate() {
        let weight = weights.map(|values| values[idx]).unwrap_or(1.0);
        if !weight.is_finite() || weight < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "piecewise robust fit weights must be finite nonnegative values".to_string(),
            ));
        }
        if weight <= 0.0 {
            continue;
        }
        let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
        let target = transform_piecewise_target(row.target, bounds, config)?;
        fill_piecewise_linear_seasonal_features(
            &mut features,
            t,
            &PiecewiseLinearFeatureContext {
                series_id: None,
                timestamp: row.timestamp,
                covariates: Some(&row.covariates),
                horizon_step: None,
                component_multiplier: fit_component_multiplier(
                    t,
                    trend_coefficients,
                    changepoints,
                    bounds,
                    config,
                ),
                changepoints,
                config,
                regressor_stats: Some(regressor_stats),
            },
        )?;
        for i in 0..feature_count {
            xty[i] += weight * features[i] * target;
            for j in i..feature_count {
                xtx[i][j] += weight * features[i] * features[j];
            }
        }
    }
    for i in 0..feature_count {
        let (previous_rows, current_and_after) = xtx.split_at_mut(i);
        let current_row = &mut current_and_after[0];
        for (j, previous_row) in previous_rows.iter().enumerate() {
            current_row[j] = previous_row[i];
        }
    }
    apply_piecewise_linear_ridge(&mut xtx, changepoints.len(), config);
    let Some(coefficients) =
        solve_piecewise_linear_coefficients(xtx.clone(), xty, changepoints.len(), config)
    else {
        return Ok(None);
    };
    let coefficient_covariance = if compute_covariance {
        invert_piecewise_linear_normal_matrix(&xtx).ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "could not invert piecewise linear covariance normal matrix".to_string(),
            )
        })?
    } else {
        Vec::new()
    };
    Ok(Some(PiecewiseLinearFitResult {
        coefficients,
        coefficient_covariance,
    }))
}

fn piecewise_needs_coefficient_covariance(config: &PiecewiseLinearSeasonalConfig) -> bool {
    config.coefficient_uncertainty_scale > 0.0
        && (!config.interval_levels.is_empty()
            || !config.quantile_levels.is_empty()
            || config.uncertainty_samples > 0)
}

fn invert_piecewise_linear_normal_matrix(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut left = matrix.to_vec();
    let mut inverse = vec![vec![0.0; n]; n];
    for (idx, row) in inverse.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if left[row][pivot_idx].abs() > left[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if left[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        left.swap(pivot_idx, pivot_row);
        inverse.swap(pivot_idx, pivot_row);

        let pivot = left[pivot_idx][pivot_idx];
        for cell in left[pivot_idx].iter_mut() {
            *cell /= pivot;
        }
        for cell in inverse[pivot_idx].iter_mut() {
            *cell /= pivot;
        }
        let pivot_left = left[pivot_idx].clone();
        let pivot_inverse = inverse[pivot_idx].clone();

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = left[row][pivot_idx];
            if factor == 0.0 {
                continue;
            }
            for (cell, pivot_cell) in left[row].iter_mut().zip(pivot_left.iter()) {
                *cell -= factor * pivot_cell;
            }
            for (cell, pivot_cell) in inverse[row].iter_mut().zip(pivot_inverse.iter()) {
                *cell -= factor * pivot_cell;
            }
        }
    }
    Some(inverse)
}

fn piecewise_transformed_residual_scale(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: &BTreeMap<String, PiecewiseLinearRegressorStats>,
    coefficients: &[f64],
) -> Result<f64> {
    let residuals = piecewise_transformed_residuals(
        history,
        elapsed,
        changepoints,
        config,
        regressor_stats,
        coefficients,
        coefficients,
    )?;
    Ok(piecewise_residual_rmse_scale(&residuals).unwrap_or(0.0))
}

fn piecewise_transformed_residuals(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: &BTreeMap<String, PiecewiseLinearRegressorStats>,
    trend_coefficients: &[f64],
    coefficients: &[f64],
) -> Result<Vec<f64>> {
    let mut features =
        vec![0.0; piecewise_linear_seasonal_feature_count(config, changepoints.len())];
    history
        .iter()
        .zip(elapsed.iter())
        .map(|(row, &t)| {
            let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
            let target = transform_piecewise_target(row.target, bounds, config)?;
            fill_piecewise_linear_seasonal_features(
                &mut features,
                t,
                &PiecewiseLinearFeatureContext {
                    series_id: None,
                    timestamp: row.timestamp,
                    covariates: Some(&row.covariates),
                    horizon_step: None,
                    component_multiplier: fit_component_multiplier(
                        t,
                        trend_coefficients,
                        changepoints,
                        bounds,
                        config,
                    ),
                    changepoints,
                    config,
                    regressor_stats: Some(regressor_stats),
                },
            )?;
            let fitted = features
                .iter()
                .zip(coefficients.iter())
                .map(|(feature, coefficient)| feature * coefficient)
                .sum::<f64>();
            Ok(target - fitted)
        })
        .collect()
}

fn piecewise_residual_rmse_scale(residuals: &[f64]) -> Option<f64> {
    let mut count = 0usize;
    let mut sum_squared = 0.0;
    for residual in residuals.iter().copied().filter(|value| value.is_finite()) {
        count += 1;
        sum_squared += residual * residual;
    }
    (count > 0 && sum_squared > 0.0).then_some((sum_squared / count as f64).sqrt())
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn piecewise_linear_l2_penalties(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> Vec<f64> {
    let feature_count = piecewise_linear_seasonal_feature_count(config, changepoint_count);
    let mut penalties = vec![0.0; feature_count];
    let mut col = 2;
    penalties[1] = config.changepoint_l2_regularization;
    for _ in 0..changepoint_count {
        penalties[col] = config.changepoint_l2_regularization;
        col += 1;
    }
    col = fill_penalty_range(
        &mut penalties,
        col,
        2 * config.yearly_fourier_order,
        config
            .yearly_l2_regularization
            .unwrap_or(config.seasonality_l2_regularization),
    );
    col = fill_penalty_range(
        &mut penalties,
        col,
        2 * config.weekly_fourier_order,
        config
            .weekly_l2_regularization
            .unwrap_or(config.seasonality_l2_regularization),
    );
    col = fill_penalty_range(
        &mut penalties,
        col,
        2 * config.daily_fourier_order,
        config
            .daily_l2_regularization
            .unwrap_or(config.seasonality_l2_regularization),
    );
    for seasonality in &config.custom_seasonalities {
        col = fill_penalty_range(
            &mut penalties,
            col,
            2 * seasonality.fourier_order,
            seasonality
                .l2_regularization
                .unwrap_or(config.seasonality_l2_regularization),
        );
    }
    for term in piecewise_event_terms(config) {
        penalties[col] = config
            .event_l2_regularization_by_name
            .get(&term.name)
            .copied()
            .unwrap_or(config.event_l2_regularization);
        col += 1;
    }
    for name in &config.extra_regressors {
        penalties[col] = config
            .regressor_l2_regularization_by_name
            .get(name)
            .copied()
            .unwrap_or(config.regressor_l2_regularization);
        col += 1;
    }
    debug_assert_eq!(col, feature_count);
    penalties
}

fn fill_penalty_range(penalties: &mut [f64], start: usize, len: usize, penalty: f64) -> usize {
    for value in penalties.iter_mut().skip(start).take(len) {
        *value = penalty;
    }
    start + len
}

fn solve_piecewise_linear_coefficients(
    xtx: Vec<Vec<f64>>,
    xty: Vec<f64>,
    changepoint_count: usize,
    config: &PiecewiseLinearSeasonalConfig,
) -> Option<Vec<f64>> {
    let mut coefficients = solve_linear_system(xtx.clone(), xty.clone())?;
    let monotonic_constraints =
        piecewise_linear_coefficient_monotonic_constraints(config, changepoint_count);
    let has_monotonic_constraints = monotonic_constraints
        .iter()
        .any(|direction| *direction != 0);
    if (config.changepoint_l1_regularization <= 0.0 || changepoint_count == 0)
        && !has_monotonic_constraints
    {
        return Some(coefficients);
    }
    let penalized_start = 2;
    let penalized_end = 2 + changepoint_count;
    for _ in 0..100 {
        let mut max_delta = 0.0_f64;
        for j in 0..coefficients.len() {
            let diagonal = xtx[j][j];
            if diagonal.abs() <= 1.0e-12 || !diagonal.is_finite() {
                return None;
            }
            let without_j = xty[j]
                - xtx[j]
                    .iter()
                    .zip(coefficients.iter())
                    .enumerate()
                    .filter(|(idx, _)| *idx != j)
                    .map(|(_, (x, coefficient))| x * coefficient)
                    .sum::<f64>();
            let mut updated = if (penalized_start..penalized_end).contains(&j) {
                soft_threshold(without_j, config.changepoint_l1_regularization) / diagonal
            } else {
                without_j / diagonal
            };
            updated = match monotonic_constraints.get(j).copied().unwrap_or(0) {
                1 => updated.max(0.0),
                -1 => updated.min(0.0),
                _ => updated,
            };
            max_delta = max_delta.max((updated - coefficients[j]).abs());
            coefficients[j] = updated;
        }
        if max_delta < 1.0e-10 {
            break;
        }
    }
    Some(coefficients)
}

fn piecewise_linear_coefficient_monotonic_constraints(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> Vec<i8> {
    let feature_count = piecewise_linear_seasonal_feature_count(config, changepoint_count);
    let mut constraints = vec![0; feature_count];
    let start = piecewise_extra_regressor_start_col(config, changepoint_count);
    for (offset, name) in config.extra_regressors.iter().enumerate() {
        constraints[start + offset] = config
            .extra_regressor_monotonic_constraints
            .get(name)
            .copied()
            .unwrap_or(0);
    }
    constraints
}

fn piecewise_extra_regressor_start_col(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> usize {
    2 + changepoint_count
        + 2 * config.yearly_fourier_order
        + 2 * config.weekly_fourier_order
        + 2 * config.daily_fourier_order
        + 2 * config
            .custom_seasonalities
            .iter()
            .map(|seasonality| seasonality.fourier_order)
            .sum::<usize>()
        + piecewise_event_terms(config).len()
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

fn piecewise_trend_delta_scale(coefficients: &[f64], changepoint_count: usize) -> f64 {
    if changepoint_count == 0 {
        return 0.0;
    }
    let mut deltas = coefficients
        .iter()
        .skip(2)
        .take(changepoint_count)
        .map(|value| value.abs())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if deltas.is_empty() {
        return 0.0;
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    quantile_from_sorted(&deltas, 0.5)
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn deterministic_standard_normal(seed: u64, step: u64, sample: u64) -> f64 {
    let state = splitmix64(
        seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ sample.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    );
    let uniform = uniform_open_0_1(state);
    inverse_standard_normal_cdf(uniform)
}

fn deterministic_trend_uncertainty_draw(
    seed: u64,
    step: u64,
    sample: u64,
    policy: PiecewiseLinearTrendUncertaintyPolicy,
) -> f64 {
    let state = splitmix64(
        seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ sample.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    );
    let uniform = uniform_open_0_1(state);
    match policy {
        PiecewiseLinearTrendUncertaintyPolicy::Normal => inverse_standard_normal_cdf(uniform),
        PiecewiseLinearTrendUncertaintyPolicy::Laplace => {
            if uniform < 0.5 {
                (2.0 * uniform).ln()
            } else {
                -(2.0 * (1.0 - uniform)).ln()
            }
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn uniform_open_0_1(value: u64) -> f64 {
    let mantissa = value >> 11;
    ((mantissa as f64) + 0.5) / ((1_u64 << 53) as f64)
}

fn predict_piecewise_linear_value(
    elapsed_days: f64,
    coefficients: &[f64],
    context: &PiecewiseLinearFeatureContext<'_>,
) -> Result<f64> {
    let mut features =
        vec![
            0.0;
            piecewise_linear_seasonal_feature_count(context.config, context.changepoints.len())
        ];
    fill_piecewise_linear_seasonal_features(&mut features, elapsed_days, context)?;
    Ok(features
        .iter()
        .zip(coefficients.iter())
        .map(|(feature, coefficient)| feature * coefficient)
        .sum())
}

fn piecewise_component_contributions(
    features: &[f64],
    coefficients: &[f64],
    changepoint_count: usize,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Value> {
    let expected = piecewise_linear_seasonal_feature_count(config, changepoint_count);
    if features.len() != expected || coefficients.len() != expected {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal component decomposition received mismatched feature dimensions"
                .to_string(),
        ));
    }
    let mut col = 2 + changepoint_count;
    let mut components = serde_json::Map::new();
    let trend = dot_range(features, coefficients, 0, col);
    components.insert("trend_linear".to_string(), json!(trend));
    components.insert(
        "changepoint_delta".to_string(),
        json!(dot_range(features, coefficients, 2, 2 + changepoint_count)),
    );
    col = insert_fourier_component(
        &mut components,
        "yearly",
        features,
        coefficients,
        col,
        config.yearly_fourier_order,
    );
    col = insert_fourier_component(
        &mut components,
        "weekly",
        features,
        coefficients,
        col,
        config.weekly_fourier_order,
    );
    col = insert_fourier_component(
        &mut components,
        "daily",
        features,
        coefficients,
        col,
        config.daily_fourier_order,
    );
    for seasonality in &config.custom_seasonalities {
        col = insert_fourier_component(
            &mut components,
            &seasonality.name,
            features,
            coefficients,
            col,
            seasonality.fourier_order,
        );
    }
    let mut event_components = serde_json::Map::new();
    let mut event_offset_components = serde_json::Map::new();
    for term in piecewise_event_terms(config) {
        let contribution = features[col] * coefficients[col];
        let event_total = event_components
            .get(&term.name)
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            + contribution;
        event_components.insert(term.name.clone(), json!(event_total));
        event_offset_components.insert(term.label(), json!(contribution));
        col += 1;
    }
    components.insert("events".to_string(), Value::Object(event_components));
    components.insert(
        "event_window_offsets".to_string(),
        Value::Object(event_offset_components),
    );
    let mut regressor_components = serde_json::Map::new();
    for name in &config.extra_regressors {
        regressor_components.insert(name.clone(), json!(features[col] * coefficients[col]));
        col += 1;
    }
    components.insert(
        "regressors".to_string(),
        Value::Object(regressor_components),
    );
    let seasonal_total = components
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "trend_linear"
                    | "changepoint_delta"
                    | "events"
                    | "event_window_offsets"
                    | "regressors"
            )
        })
        .filter_map(|(_, value)| value.as_f64())
        .sum::<f64>();
    let event_total = components["events"]
        .as_object()
        .map(|events| events.values().filter_map(Value::as_f64).sum::<f64>())
        .unwrap_or(0.0);
    let regressor_total = components["regressors"]
        .as_object()
        .map(|regressors| regressors.values().filter_map(Value::as_f64).sum::<f64>())
        .unwrap_or(0.0);
    components.insert("seasonal_total".to_string(), json!(seasonal_total));
    components.insert("event_total".to_string(), json!(event_total));
    components.insert("regressor_total".to_string(), json!(regressor_total));
    components.insert(
        "non_trend_total".to_string(),
        json!(seasonal_total + event_total + regressor_total),
    );
    debug_assert_eq!(col, expected);
    Ok(Value::Object(components))
}

fn insert_fourier_component(
    components: &mut serde_json::Map<String, Value>,
    name: &str,
    features: &[f64],
    coefficients: &[f64],
    col: usize,
    order: usize,
) -> usize {
    let len = 2 * order;
    components.insert(
        name.to_string(),
        json!(dot_range(features, coefficients, col, col + len)),
    );
    col + len
}

fn dot_range(features: &[f64], coefficients: &[f64], start: usize, end: usize) -> f64 {
    features
        .iter()
        .zip(coefficients.iter())
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|(feature, coefficient)| feature * coefficient)
        .sum()
}

fn piecewise_event_names(config: &PiecewiseLinearSeasonalConfig) -> Vec<String> {
    let mut names = config
        .events
        .iter()
        .map(|event| event.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn piecewise_event_terms(config: &PiecewiseLinearSeasonalConfig) -> Vec<PiecewiseEventTerm> {
    let mut terms = config
        .events
        .iter()
        .flat_map(|event| {
            (event.lower_window..=event.upper_window).map(|offset| PiecewiseEventTerm {
                name: event.name.clone(),
                offset,
            })
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn event_term_is_active(
    timestamp: chrono::NaiveDateTime,
    term: &PiecewiseEventTerm,
    events: &[PiecewiseLinearEvent],
) -> bool {
    events
        .iter()
        .filter(|event| event.name == term.name)
        .any(|event| {
            let days = (timestamp.date() - event.timestamp.date()).num_days();
            days == i64::from(term.offset)
                && days >= i64::from(event.lower_window)
                && days <= i64::from(event.upper_window)
        })
}

fn piecewise_regressor_stats(
    history: &[ForecastRow],
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<BTreeMap<String, PiecewiseLinearRegressorStats>> {
    config
        .extra_regressors
        .iter()
        .map(|name| {
            let values = history
                .iter()
                .map(|row| {
                    piecewise_named_value(
                        name,
                        None,
                        Some(&row.covariates),
                        None,
                        config,
                        "extra regressor",
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((
                name.clone(),
                piecewise_regressor_stat(&values, config.regressor_standardization),
            ))
        })
        .collect()
}

fn piecewise_regressor_stat(
    values: &[f64],
    standardization: PiecewiseLinearRegressorStandardization,
) -> PiecewiseLinearRegressorStats {
    if standardization == PiecewiseLinearRegressorStandardization::None || values.is_empty() {
        return PiecewiseLinearRegressorStats {
            mean: 0.0,
            scale: 1.0,
            standardized: false,
        };
    }
    let is_binary = values
        .iter()
        .all(|value| (*value - 0.0).abs() < 1.0e-12 || (*value - 1.0).abs() < 1.0e-12);
    if is_binary {
        return PiecewiseLinearRegressorStats {
            mean: 0.0,
            scale: 1.0,
            standardized: false,
        };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    let scale = variance.sqrt();
    if !scale.is_finite() || scale <= 1.0e-12 {
        PiecewiseLinearRegressorStats {
            mean: 0.0,
            scale: 1.0,
            standardized: false,
        }
    } else {
        PiecewiseLinearRegressorStats {
            mean,
            scale,
            standardized: true,
        }
    }
}

fn piecewise_regressor_value(
    name: &str,
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<f64> {
    piecewise_named_value(
        name,
        series_id,
        covariates,
        horizon_step,
        config,
        "extra regressor",
    )
}

fn piecewise_extra_regressor_value(
    name: &str,
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: Option<&BTreeMap<String, PiecewiseLinearRegressorStats>>,
) -> Result<f64> {
    let value = piecewise_regressor_value(name, series_id, covariates, horizon_step, config)?;
    Ok(match regressor_stats.and_then(|stats| stats.get(name)) {
        Some(stats) if stats.standardized => (value - stats.mean) / stats.scale,
        _ => value,
    })
}

fn piecewise_seasonality_condition_is_active(
    seasonality: &PiecewiseLinearSeasonality,
    context: &PiecewiseLinearFeatureContext<'_>,
) -> Result<bool> {
    let Some(condition_name) = &seasonality.condition_name else {
        return Ok(true);
    };
    let value = piecewise_named_value(
        condition_name,
        context.series_id,
        context.covariates,
        context.horizon_step,
        context.config,
        "seasonality condition",
    )?;
    Ok(value > 0.0)
}

fn piecewise_named_value(
    name: &str,
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
    role: &str,
) -> Result<f64> {
    if let Some(covariates) = covariates {
        let value = covariates.get(name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "piecewise seasonal {role} {name:?} missing from training covariates"
            ))
        })?;
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "piecewise seasonal {role} {name:?} training values must be finite"
            )));
        }
        return Ok(value);
    }
    let step = horizon_step.ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "piecewise seasonal prediction requires a future regressor horizon step".to_string(),
        )
    })?;
    let values = series_id
        .and_then(|series_id| config.future_regressors_by_series.get(series_id))
        .and_then(|values_by_name| values_by_name.get(name))
        .or_else(|| config.future_regressors.get(name))
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "piecewise seasonal future {role} {name:?} is missing"
            ))
        })?;
    let value = values.get(step - 1).copied().ok_or_else(|| {
        CartoBoostError::InvalidInput(format!(
            "piecewise seasonal future {role} {name:?} has fewer than {step} values"
        ))
    })?;
    if !value.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "piecewise seasonal future {role} {name:?} values must be finite"
        )));
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy)]
struct PiecewiseBounds {
    floor: f64,
    cap: Option<f64>,
}

fn piecewise_bounds(
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<PiecewiseBounds> {
    let floor = match &config.floor_regressor {
        Some(name) => piecewise_named_value(
            name,
            series_id,
            covariates,
            horizon_step,
            config,
            "floor_regressor",
        )?,
        None => config.floor,
    };
    let cap = match &config.cap_regressor {
        Some(name) => Some(piecewise_named_value(
            name,
            series_id,
            covariates,
            horizon_step,
            config,
            "cap_regressor",
        )?),
        None => config.cap,
    };
    if !floor.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal floor must be finite".to_string(),
        ));
    }
    if let Some(cap) = cap {
        if !cap.is_finite() || cap <= floor {
            return Err(CartoBoostError::InvalidInput(
                "piecewise seasonal cap must be finite and greater than floor".to_string(),
            ));
        }
    }
    Ok(PiecewiseBounds { floor, cap })
}

fn transform_piecewise_target(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<f64> {
    match config.growth {
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => Ok(value),
        PiecewiseLinearGrowth::Logistic => {
            let cap = bounds.cap.expect("validated logistic cap");
            if value <= bounds.floor || value >= cap {
                return Err(CartoBoostError::InvalidInput(
                    "logistic piecewise seasonal targets must be strictly between floor and cap"
                        .to_string(),
                ));
            }
            let scaled = (value - bounds.floor) / (cap - bounds.floor);
            Ok((scaled / (1.0 - scaled)).ln())
        }
    }
}

fn inverse_piecewise_target(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    match config.growth {
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => value,
        PiecewiseLinearGrowth::Logistic => {
            let cap = bounds.cap.expect("validated logistic cap");
            let scaled = if value >= 0.0 {
                let z = (-value).exp();
                1.0 / (1.0 + z)
            } else {
                let z = value.exp();
                z / (1.0 + z)
            };
            bounds.floor + (cap - bounds.floor) * scaled
        }
    }
}

fn inverse_piecewise_target_derivative(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    match config.growth {
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => 1.0,
        PiecewiseLinearGrowth::Logistic => {
            let cap = bounds.cap.expect("validated logistic cap");
            let scaled = if value >= 0.0 {
                let z = (-value).exp();
                1.0 / (1.0 + z)
            } else {
                let z = value.exp();
                z / (1.0 + z)
            };
            (cap - bounds.floor) * scaled * (1.0 - scaled)
        }
    }
}

fn quadratic_form(vector: &[f64], matrix: &[Vec<f64>]) -> f64 {
    if matrix.len() != vector.len() || matrix.iter().any(|row| row.len() != vector.len()) {
        return 0.0;
    }
    vector
        .iter()
        .enumerate()
        .map(|(i, left)| {
            let row_dot = matrix[i]
                .iter()
                .zip(vector.iter())
                .map(|(matrix_value, right)| matrix_value * right)
                .sum::<f64>();
            left * row_dot
        })
        .sum()
}

fn validate_kalman_grid(name: &str, values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "auto_kalman {name} must not be empty"
        )));
    }
    for &value in values {
        if !value.is_finite() || value <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "auto_kalman {name} values must be positive finite numbers"
            )));
        }
    }
    Ok(())
}

fn score_kalman_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    params: KalmanParameterSet,
    validation_window: Option<usize>,
) -> Result<f64> {
    let config = LocalLinearKalmanConfig::new(
        params.level_process_variance,
        params.trend_process_variance,
        params.observation_variance,
    )?;
    let (sum_squared_error, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            if history.len() < 2 {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} has {} rows, but auto_kalman requires at least two rows",
                    history.len()
                )));
            }
            let requested_window = validation_window.unwrap_or_else(|| {
                let suggested = history.len() / 5;
                suggested.clamp(1, 12)
            });
            let window = requested_window.min(history.len().saturating_sub(2));
            if window == 0 {
                let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
                let result = fit_local_linear_kalman(&values, config)
                    .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
                return Ok((no_holdout_validation_score(result.residual_summary.mse), 1));
            }
            let train_len = history.len() - window;
            let train = history[..train_len]
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>();
            let result = fit_local_linear_kalman(&train, config)
                .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            let mut sum = 0.0;
            for (idx, row) in history[train_len..].iter().enumerate() {
                let step = idx + 1;
                let mean = result.final_state.level + step as f64 * result.final_state.trend;
                let residual = row.target - mean;
                sum += residual * residual;
            }
            Ok((sum, window))
        })
        .reduce(
            || Ok((0.0, 0usize)),
            |left: Result<(f64, usize)>, right: Result<(f64, usize)>| {
                let (left_sum, left_count) = left?;
                let (right_sum, right_count) = right?;
                Ok((left_sum + right_sum, left_count + right_count))
            },
        )?;
    if count == 0 {
        if sum_squared_error.is_finite() {
            return Ok(sum_squared_error);
        }
        return Err(CartoBoostError::InvalidInput(
            "auto_kalman validation score must be finite".to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    if !mse.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_kalman validation score must be finite".to_string(),
        ));
    }
    Ok(mse)
}

fn score_local_level_kalman_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    params: LocalLevelKalmanParameterSet,
    validation_window: Option<usize>,
) -> Result<f64> {
    let config =
        LocalLevelKalmanConfig::new(params.level_process_variance, params.observation_variance)?;
    let (sum_squared_error, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            if history.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} has no rows, but auto_local_level_kalman requires at least one row"
                )));
            }
            let requested_window = validation_window.unwrap_or_else(|| {
                let suggested = history.len() / 5;
                suggested.clamp(1, 12)
            });
            let window = requested_window.min(history.len().saturating_sub(1));
            if window == 0 {
                let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
                let result = fit_local_level_kalman(&values, config)
                    .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
                return Ok((no_holdout_validation_score(result.residual_summary.mse), 1));
            }
            let train_len = history.len() - window;
            let train = history[..train_len]
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>();
            let result = fit_local_level_kalman(&train, config)
                .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            let mut sum = 0.0;
            for row in &history[train_len..] {
                let residual = row.target - result.final_level;
                sum += residual * residual;
            }
            Ok((sum, window))
        })
        .reduce(
            || Ok((0.0, 0usize)),
            |left: Result<(f64, usize)>, right: Result<(f64, usize)>| {
                let (left_sum, left_count) = left?;
                let (right_sum, right_count) = right?;
                Ok((left_sum + right_sum, left_count + right_count))
            },
        )?;
    if count == 0 {
        if sum_squared_error.is_finite() {
            return Ok(sum_squared_error);
        }
        return Err(CartoBoostError::InvalidInput(
            "auto_local_level_kalman validation score must be finite".to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    if !mse.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_local_level_kalman validation score must be finite".to_string(),
        ));
    }
    Ok(mse)
}

fn no_holdout_validation_score(in_sample_mse: f64) -> f64 {
    if in_sample_mse.is_finite() {
        in_sample_mse
    } else {
        0.0
    }
}

fn validate_ets_params(
    alpha: f64,
    beta: f64,
    gamma: Option<f64>,
    season_length: Option<usize>,
    damping_phi: f64,
) -> Result<()> {
    validate_unit_interval("alpha", alpha, false)?;
    validate_unit_interval("beta", beta, true)?;
    validate_unit_interval("damping_phi", damping_phi, false)?;
    if let Some(gamma) = gamma {
        validate_unit_interval("gamma", gamma, true)?;
    }
    match (gamma, season_length) {
        (Some(_), Some(length)) if length > 1 => Ok(()),
        (None, None) => Ok(()),
        (Some(_), None) => Err(CartoBoostError::InvalidInput(
            "ETS gamma requires season_length".to_string(),
        )),
        (None, Some(_)) => Err(CartoBoostError::InvalidInput(
            "ETS season_length requires gamma".to_string(),
        )),
        (Some(_), Some(_)) => Err(CartoBoostError::InvalidInput(
            "ETS season_length must be greater than 1".to_string(),
        )),
    }
}

fn damped_trend_multiplier(damping_phi: f64, step: usize) -> f64 {
    if (damping_phi - 1.0).abs() <= f64::EPSILON {
        step as f64
    } else {
        damping_phi * (1.0 - damping_phi.powi(step as i32)) / (1.0 - damping_phi)
    }
}

fn validate_unit_interval(name: &str, value: f64, allow_zero: bool) -> Result<()> {
    let lower_ok = if allow_zero {
        value >= 0.0
    } else {
        value > 0.0
    };
    if !value.is_finite() || !lower_ok || value > 1.0 {
        let range = if allow_zero { "[0, 1]" } else { "(0, 1]" };
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite and in {range}"
        )));
    }
    Ok(())
}

fn validate_arima_order(p: usize, d: usize, q: usize) -> Result<()> {
    if p > MAX_ARIMA_ORDER {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA p must be <= 8".to_string(),
        ));
    }
    if d > 2 {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        ));
    }
    if q > MAX_ARIMA_ORDER {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA q must be <= 8".to_string(),
        ));
    }
    Ok(())
}

fn arima_order_supported_by_history(
    history_len: usize,
    p: usize,
    d: usize,
    q: usize,
) -> (usize, usize, usize) {
    let effective_d = d.min(2).min(history_len.saturating_sub(1));
    let differenced_len = history_len.saturating_sub(effective_d);
    let max_lag_order = differenced_len.saturating_sub(1).min(MAX_ARIMA_ORDER);
    (p.min(max_lag_order), effective_d, q.min(max_lag_order))
}

fn initial_trend(values: &[f64], seasonals: Option<&[f64]>) -> f64 {
    match seasonals {
        Some(seasonals) if values.len() > seasonals.len() => {
            let length = seasonals.len();
            let mut sum = 0.0;
            for idx in 0..length {
                sum += (values[idx + length] - values[idx]) / length as f64;
            }
            sum / length as f64
        }
        Some(seasonals) => {
            (values[1] - seasonals[1 % seasonals.len()]) - (values[0] - seasonals[0])
        }
        None => values[1] - values[0],
    }
}

fn difference_series(values: &[f64], d: usize) -> Result<Vec<f64>> {
    if values.len() <= d {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA differencing order leaves no observations".to_string(),
        ));
    }
    match d {
        0 => Ok(values.to_vec()),
        1 => {
            let mut differences = Vec::with_capacity(values.len() - 1);
            for idx in 1..values.len() {
                differences.push(values[idx] - values[idx - 1]);
            }
            Ok(differences)
        }
        2 => {
            let mut differences = Vec::with_capacity(values.len() - 2);
            for idx in 2..values.len() {
                differences.push(values[idx] - 2.0 * values[idx - 1] + values[idx - 2]);
            }
            Ok(differences)
        }
        _ => Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        )),
    }
}

fn last_differences(values: &[f64], d: usize) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA requires at least one observation".to_string(),
        ));
    }
    if values.len() <= d {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA differencing order leaves no observations".to_string(),
        ));
    }
    let last = values[values.len() - 1];
    match d {
        0 => Ok(vec![last]),
        1 => Ok(vec![last, last - values[values.len() - 2]]),
        2 => {
            let n = values.len();
            Ok(vec![
                last,
                last - values[n - 2],
                last - 2.0 * values[n - 2] + values[n - 3],
            ])
        }
        _ => Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        )),
    }
}

fn tail_values(values: &[f64], length: usize) -> Vec<f64> {
    if length == 0 {
        Vec::new()
    } else {
        values[values.len().saturating_sub(length)..].to_vec()
    }
}

fn push_tail(values: &mut Vec<f64>, max_len: usize, value: f64) {
    if max_len == 0 {
        return;
    }
    if values.len() == max_len {
        values.rotate_left(1);
        if let Some(last) = values.last_mut() {
            *last = value;
        }
    } else {
        values.push(value);
    }
}

fn arima_feature(values: &[f64], residuals: &[f64], idx: usize, p: usize, col: usize) -> f64 {
    if col == 0 {
        1.0
    } else if col <= p {
        values[idx - col]
    } else {
        residuals[idx - (col - p)]
    }
}

fn solve_arima_normal_equations(
    mut matrix: [[f64; MAX_ARIMA_COLUMNS]; MAX_ARIMA_COLUMNS],
    mut rhs: [f64; MAX_ARIMA_COLUMNS],
    n: usize,
) -> Option<Vec<f64>> {
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if matrix[row][pivot_idx].abs() > matrix[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if matrix[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_idx, pivot_row);
        rhs.swap(pivot_idx, pivot_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for value in matrix[pivot_idx].iter_mut().take(n).skip(pivot_idx) {
            *value /= pivot;
        }
        rhs[pivot_idx] /= pivot;
        let pivot_tail = matrix[pivot_idx];
        let pivot_rhs = rhs[pivot_idx];

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = matrix[row][pivot_idx];
            for (col, pivot_cell) in pivot_tail.iter().enumerate().take(n).skip(pivot_idx) {
                matrix[row][col] -= factor * pivot_cell;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    Some(rhs[..n].to_vec())
}

fn fit_arima_components(values: &[f64], p: usize, q: usize) -> Result<ArimaComponents> {
    let mut residuals = vec![0.0; values.len()];
    let mut intercept = values.iter().sum::<f64>() / values.len() as f64;
    let mut ar_coefficients = vec![0.0; p];
    let mut ma_coefficients = vec![0.0; q];
    let iterations = if q == 0 { 1 } else { 6 };
    for _ in 0..iterations {
        let (next_intercept, next_ar, next_ma) =
            fit_arima_coefficients_once(values, &residuals, p, q)?;
        intercept = next_intercept;
        ar_coefficients = next_ar;
        ma_coefficients = next_ma;
        let (fitted, next_residuals) = fitted_arima_values(
            values,
            intercept,
            &ar_coefficients,
            &ma_coefficients,
            &residuals,
        );
        let _ = fitted;
        residuals = next_residuals;
    }
    let (fitted, residuals) = fitted_arima_values(
        values,
        intercept,
        &ar_coefficients,
        &ma_coefficients,
        &residuals,
    );
    Ok((
        intercept,
        ar_coefficients,
        ma_coefficients,
        fitted,
        residuals,
    ))
}

fn fit_arima_coefficients_once(
    values: &[f64],
    residuals: &[f64],
    p: usize,
    q: usize,
) -> Result<(f64, Vec<f64>, Vec<f64>)> {
    if p == 0 && q == 0 {
        return Ok((
            values.iter().sum::<f64>() / values.len() as f64,
            Vec::new(),
            Vec::new(),
        ));
    }
    let cols = p + q + 1;
    let mut xtx = [[0.0; MAX_ARIMA_COLUMNS]; MAX_ARIMA_COLUMNS];
    let mut xty = [0.0; MAX_ARIMA_COLUMNS];
    let start = p.max(q);
    for idx in start..values.len() {
        for row in 0..cols {
            let row_value = arima_feature(values, residuals, idx, p, row);
            xty[row] += row_value * values[idx];
            for (col, cell) in xtx[row].iter_mut().enumerate().take(cols) {
                *cell += row_value * arima_feature(values, residuals, idx, p, col);
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().take(cols) {
        row[idx] += 1.0e-8;
    }
    let solution = solve_arima_normal_equations(xtx, xty, cols).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "could not solve ARIMA normal equations for coefficient fit".to_string(),
        )
    })?;
    Ok((
        solution[0],
        solution[1..=p].to_vec(),
        solution[(p + 1)..].to_vec(),
    ))
}

fn fitted_arima_values(
    values: &[f64],
    intercept: f64,
    ar_coefficients: &[f64],
    ma_coefficients: &[f64],
    residual_history: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let p = ar_coefficients.len();
    let q = ma_coefficients.len();
    let mut fitted = Vec::with_capacity(values.len());
    let mut residuals = vec![0.0; values.len()];
    for idx in 0..values.len() {
        let start = p.max(q);
        let mean = if idx < start {
            values[idx]
        } else {
            let mut mean = intercept;
            for (coef_idx, coef) in ar_coefficients.iter().enumerate() {
                mean += coef * values[idx - coef_idx - 1];
            }
            for (coef_idx, coef) in ma_coefficients.iter().enumerate() {
                mean += coef * residual_history[idx - coef_idx - 1];
            }
            mean
        };
        fitted.push(mean);
        if idx >= start {
            residuals[idx] = values[idx] - mean;
        }
    }
    (fitted, residuals)
}

fn forecast_arima_next(
    history: &[f64],
    residuals: &[f64],
    intercept: f64,
    ar_coefficients: &[f64],
    ma_coefficients: &[f64],
) -> f64 {
    let mut forecast = intercept;
    for (idx, coef) in ar_coefficients.iter().enumerate() {
        forecast += coef * history[history.len() - idx - 1];
    }
    for (idx, coef) in ma_coefficients.iter().enumerate() {
        forecast += coef * residuals[residuals.len() - idx - 1];
    }
    forecast
}

fn ar_recursion_is_stable(ar_coefficients: &[f64]) -> bool {
    let p = ar_coefficients.len();
    if p == 0 {
        return true;
    }
    for basis_idx in 0..p {
        let mut state = vec![0.0; p];
        state[basis_idx] = 1.0;
        let mut max_abs = 1.0;
        for _ in 0..128 {
            let next = ar_coefficients
                .iter()
                .enumerate()
                .map(|(idx, coef)| coef * state[p - idx - 1])
                .sum::<f64>();
            if !next.is_finite() || next.abs() > 16.0 || next.abs() > max_abs * 4.0 {
                return false;
            }
            max_abs = max_abs.max(next.abs());
            push_tail(&mut state, p, next);
        }
    }
    true
}

fn undifference_fitted_values(values: &[f64], fitted_diff: &[f64], d: usize) -> Vec<f64> {
    match d {
        0 => fitted_diff.to_vec(),
        1 => {
            let mut fitted = vec![values[0]];
            for idx in 1..values.len() {
                fitted.push(values[idx - 1] + fitted_diff[idx - 1]);
            }
            fitted
        }
        2 => {
            let first_diff = values
                .windows(2)
                .map(|window| window[1] - window[0])
                .collect::<Vec<_>>();
            let mut fitted = vec![values[0], values[1]];
            for idx in 2..values.len() {
                fitted.push(values[idx - 1] + first_diff[idx - 2] + fitted_diff[idx - 2]);
            }
            fitted
        }
        _ => values.to_vec(),
    }
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if matrix[row][pivot_idx].abs() > matrix[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if matrix[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_idx, pivot_row);
        rhs.swap(pivot_idx, pivot_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for cell in matrix[pivot_idx].iter_mut().take(n).skip(pivot_idx) {
            *cell /= pivot;
        }
        rhs[pivot_idx] /= pivot;
        let pivot_tail = matrix[pivot_idx][pivot_idx..n].to_vec();

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = matrix[row][pivot_idx];
            for (cell, pivot_cell) in matrix[row]
                .iter_mut()
                .take(n)
                .skip(pivot_idx)
                .zip(pivot_tail.iter())
            {
                *cell -= factor * pivot_cell;
            }
            rhs[row] -= factor * rhs[pivot_idx];
        }
    }
    Some(rhs)
}

fn deseasonalize(
    series_id: &str,
    values: &[f64],
    seasonality: Option<ThetaSeasonality>,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let Some(seasonality) = seasonality else {
        return Ok((values.to_vec(), None));
    };
    if values.len() < seasonality.season_length * 2 {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} requires at least two full seasonal cycles for theta seasonality"
        )));
    }
    if seasonality.kind == ThetaSeasonalityKind::Multiplicative
        && values.iter().any(|value| *value <= 0.0)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} uses multiplicative seasonality but contains non-positive values"
        )));
    }

    let mut pattern = vec![0.0; seasonality.season_length];
    let mut counts = vec![0usize; seasonality.season_length];
    for (idx, value) in values.iter().enumerate() {
        let season_idx = idx % seasonality.season_length;
        pattern[season_idx] += *value;
        counts[season_idx] += 1;
    }
    for (slot, count) in pattern.iter_mut().zip(counts) {
        *slot /= count as f64;
    }

    match seasonality.kind {
        ThetaSeasonalityKind::Additive => {
            let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
            for slot in &mut pattern {
                *slot -= mean;
            }
            let adjusted = values
                .iter()
                .enumerate()
                .map(|(idx, value)| value - pattern[idx % pattern.len()])
                .collect();
            Ok((adjusted, Some(pattern)))
        }
        ThetaSeasonalityKind::Multiplicative => {
            let series_mean = values.iter().sum::<f64>() / values.len() as f64;
            for slot in &mut pattern {
                *slot /= series_mean;
            }
            let pattern_mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
            for slot in &mut pattern {
                *slot /= pattern_mean;
            }
            let adjusted = values
                .iter()
                .enumerate()
                .map(|(idx, value)| value / pattern[idx % pattern.len()])
                .collect();
            Ok((adjusted, Some(pattern)))
        }
    }
}

fn effective_theta_seasonality(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    seasonality: Option<ThetaSeasonality>,
) -> Option<ThetaSeasonality> {
    seasonality.filter(|seasonality| {
        history_by_series
            .values()
            .all(|history| supports_full_season_cycles(history.len(), seasonality.season_length))
    })
}

fn effective_full_cycle_season_length(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    season_length: Option<usize>,
) -> Option<usize> {
    season_length.filter(|season_length| {
        history_by_series
            .values()
            .all(|history| supports_full_season_cycles(history.len(), *season_length))
    })
}

fn supports_full_season_cycles(history_len: usize, season_length: usize) -> bool {
    season_length > 1 && history_len >= season_length.saturating_mul(2)
}

fn fit_theta_component(values: &[f64], theta: f64, alpha: f64) -> ThetaComponent {
    let slope = linear_slope(values);
    let levels = ses_one_step_levels(values, alpha);
    let last_level = alpha * values[values.len() - 1] + (1.0 - alpha) * levels[values.len() - 1];
    ThetaComponent {
        last_level,
        slope,
        theta,
    }
}

fn fitted_theta_values(values: &[f64], alpha: f64) -> Vec<f64> {
    let levels = ses_one_step_levels(values, alpha);
    let slope = linear_slope(values);
    let intercept = values.iter().sum::<f64>() / values.len() as f64
        - slope * ((values.len() - 1) as f64 / 2.0);
    levels
        .iter()
        .enumerate()
        .map(|(idx, level)| 0.5 * (level + intercept + slope * idx as f64))
        .collect()
}

fn ses_one_step_levels(values: &[f64], alpha: f64) -> Vec<f64> {
    let mut levels = Vec::with_capacity(values.len());
    levels.push(values[0]);
    for idx in 1..values.len() {
        levels.push(alpha * values[idx - 1] + (1.0 - alpha) * levels[idx - 1]);
    }
    levels
}

fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let x_mean = (values.len() - 1) as f64 / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (idx, value) in values.iter().enumerate() {
        let x_delta = idx as f64 - x_mean;
        numerator += x_delta * (value - y_mean);
        denominator += x_delta * x_delta;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn forecast_theta_component(component: &ThetaComponent, step: usize) -> f64 {
    let drift = (1.0 - 1.0 / component.theta) * component.slope * step as f64;
    component.last_level + drift
}

fn reseasonalize_value(
    value: f64,
    position: usize,
    seasonality: Option<ThetaSeasonality>,
    pattern: Option<&[f64]>,
) -> Result<f64> {
    let Some(seasonality) = seasonality else {
        return Ok(value);
    };
    let pattern = pattern.ok_or_else(|| {
        CartoBoostError::InvalidInput("theta seasonal pattern is missing".to_string())
    })?;
    let seasonal = pattern[position % seasonality.season_length];
    match seasonality.kind {
        ThetaSeasonalityKind::Additive => Ok(value + seasonal),
        ThetaSeasonalityKind::Multiplicative => Ok(value * seasonal),
    }
}

fn validate_spatial_piecewise_kriging_config(config: &SpatialPiecewiseKrigingConfig) -> Result<()> {
    if config.coordinates.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "spatial piecewise kriging coordinates must not be empty".to_string(),
        ));
    }
    for (series_id, (x, y)) in &config.coordinates {
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "spatial piecewise kriging series ids must not be empty".to_string(),
            ));
        }
        if !x.is_finite() || !y.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "spatial piecewise kriging coordinate for series {series_id} must be finite"
            )));
        }
    }
    if !config.residual_shrinkage.is_finite()
        || config.residual_shrinkage < 0.0
        || config.residual_shrinkage > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "residual_shrinkage must be finite and in [0, 1]".to_string(),
        ));
    }
    if config.uses_kriged_regressors() && config.spatial_regressors.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriged_regressors and hybrid modes require at least one spatial_regressor".to_string(),
        ));
    }
    let spatial_regressors = config
        .spatial_regressors
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if spatial_regressors.len() != config.spatial_regressors.len() {
        return Err(CartoBoostError::InvalidInput(
            "spatial_regressors must not contain duplicates".to_string(),
        ));
    }
    for regressor in &config.spatial_regressors {
        if regressor.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "spatial_regressors must not contain empty names".to_string(),
            ));
        }
        if config
            .piecewise_config
            .future_regressors
            .contains_key(regressor)
            || config
                .piecewise_config
                .future_regressors_by_series
                .values()
                .any(|values| values.contains_key(regressor))
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "future spatial regressor {regressor:?} would leak future observations"
            )));
        }
    }
    validate_piecewise_linear_seasonal_config(&config.piecewise_config)?;
    Ok(())
}

fn validate_spatial_piecewise_frame(
    frame: &ForecastFrame,
    config: &SpatialPiecewiseKrigingConfig,
) -> Result<()> {
    for series_id in frame.series_ids() {
        if !config.coordinates.contains_key(&series_id) {
            return Err(CartoBoostError::InvalidInput(format!(
                "missing spatial piecewise kriging coordinate for series {series_id}"
            )));
        }
    }
    Ok(())
}

fn spatial_piecewise_base_config(
    config: &SpatialPiecewiseKrigingConfig,
) -> PiecewiseLinearSeasonalConfig {
    let mut piecewise_config = config.piecewise_config.clone();
    if config.uses_kriged_regressors() {
        for regressor in &config.spatial_regressors {
            if !piecewise_config.extra_regressors.contains(regressor) {
                piecewise_config.extra_regressors.push(regressor.clone());
            }
        }
    }
    piecewise_config
}

fn kriged_regressor_frame(
    frame: &ForecastFrame,
    config: &SpatialPiecewiseKrigingConfig,
) -> Result<ForecastFrame> {
    let mut rows = frame.rows().to_vec();
    let mut rows_by_timestamp: BTreeMap<chrono::NaiveDateTime, Vec<usize>> = BTreeMap::new();
    for (idx, row) in rows.iter().enumerate() {
        rows_by_timestamp
            .entry(row.timestamp)
            .or_default()
            .push(idx);
    }
    for regressor in &config.spatial_regressors {
        for indices in rows_by_timestamp.values() {
            let observations = indices
                .iter()
                .filter_map(|idx| {
                    let row = &rows[*idx];
                    row.covariates.get(regressor).map(|value| (row, *value))
                })
                .map(|(row, value)| {
                    let coord = config.coordinates.get(&row.series_id).ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing spatial piecewise kriging coordinate for series {}",
                            row.series_id
                        ))
                    })?;
                    Ok(KrigingObservation {
                        x: coord.0,
                        y: coord.1,
                        value,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if observations.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "spatial regressor {regressor:?} has no observed cutoff-safe values"
                )));
            }
            let targets = indices
                .iter()
                .map(|idx| {
                    let row = &rows[*idx];
                    config
                        .coordinates
                        .get(&row.series_id)
                        .copied()
                        .ok_or_else(|| {
                            CartoBoostError::InvalidInput(format!(
                                "missing spatial piecewise kriging coordinate for series {}",
                                row.series_id
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let predictions =
                ordinary_kriging_predict_many(&observations, &targets, config.kriging_config)?;
            for (idx, prediction) in indices.iter().zip(predictions) {
                rows[*idx]
                    .covariates
                    .insert(regressor.clone(), prediction.mean);
            }
        }
    }
    let mut metadata = frame.metadata().clone();
    for regressor in &config.spatial_regressors {
        if !metadata.known_future_covariates.contains(regressor) {
            metadata.known_future_covariates.push(regressor.clone());
        }
    }
    ForecastFrame::with_metadata(rows, frame.frequency(), metadata)
}

fn kriging_config_metadata(config: OrdinaryKrigingConfig) -> Value {
    json!({
        "range": config.range,
        "nugget": config.nugget,
        "sill": config.sill,
        "variogram_model": format!("{:?}", config.variogram_model).to_lowercase(),
        "drift": format!("{:?}", config.drift).to_lowercase(),
        "anisotropy_angle_degrees": config.anisotropy_angle_degrees,
        "anisotropy_scaling": config.anisotropy_scaling,
        "max_neighbors": config.max_neighbors,
        "min_neighbors": config.min_neighbors,
        "max_distance": config.max_distance,
    })
}

fn is_neighbor_rule_error(error: &CartoBoostError) -> bool {
    error.to_string().contains("neighbors")
}

fn residual_level_variance(levels: &BTreeMap<String, f64>) -> f64 {
    if levels.is_empty() {
        return 0.0;
    }
    let mean = levels.values().sum::<f64>() / levels.len() as f64;
    levels
        .values()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / levels.len() as f64
}

fn prediction_lookup_key(
    series_id: &str,
    timestamp: chrono::NaiveDateTime,
    horizon: usize,
) -> String {
    format!(
        "{}\x1f{}\x1f{}",
        series_id,
        timestamp.format("%Y-%m-%dT%H:%M:%S"),
        horizon
    )
}

fn component_record_key(record: &Value) -> String {
    let series_id = record
        .get("series_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let timestamp = record
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let horizon = record.get("horizon").and_then(Value::as_u64).unwrap_or(0);
    format!("{series_id}\x1f{timestamp}\x1f{horizon}")
}

fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{ForecastFrameMetadata, ForecastFrequency};
    use chrono::{Duration, NaiveDate, NaiveDateTime};

    fn ts(day: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 1, day)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid fixture timestamp")
    }

    #[test]
    fn piecewise_linear_predicts_irregular_history_at_future_timestamps() {
        let frame = ForecastFrame::with_metadata(
            vec![
                ForecastRow::new("__single__", ts(1), 10.0),
                ForecastRow::new("__single__", ts(8), 14.0),
                ForecastRow::new("__single__", ts(9), 15.0),
                ForecastRow::new("__single__", ts(12), 17.0),
            ],
            ForecastFrequency::Daily,
            ForecastFrameMetadata {
                allow_irregular: true,
                ..ForecastFrameMetadata::default()
            },
        )
        .expect("irregular frame");
        let mut model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig::default())
                .expect("valid piecewise model");
        model.fit(&frame).expect("fit irregular history");

        let result = model
            .predict_at_timestamps(BTreeMap::from([(
                "__single__".to_string(),
                vec![ts(13), ts(14)],
            )]))
            .expect("predict at explicit timestamps");

        let predictions = result.predictions();
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].timestamp, ts(13));
        assert_eq!(predictions[1].timestamp, ts(14));
        assert!(predictions
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
    }

    #[test]
    fn theta_forecasts_panel_series_without_bleeding() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 12.0),
                ForecastRow::new("PU1->DO2", ts(3), 15.0),
                ForecastRow::new("PU1->DO2", ts(4), 19.0),
                ForecastRow::new("PU9->DO8", ts(1), 30.0),
                ForecastRow::new("PU9->DO8", ts(2), 29.0),
                ForecastRow::new("PU9->DO8", ts(3), 27.0),
                ForecastRow::new("PU9->DO8", ts(4), 24.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ThetaForecaster::new(2.0, 0.4).expect("valid theta");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.series_id.as_str(), row.horizon, row.mean))
            .collect::<Vec<_>>();
        assert_eq!(means.len(), 4);
        assert_eq!(means[0].0, "PU1->DO2");
        assert_eq!(means[2].0, "PU9->DO8");
        assert!(means[0].2 > means[1].2 - 10.0);
        assert_ne!(means[0].2, means[2].2);
        assert_eq!(model.fitted_values("PU1->DO2").expect("fitted").len(), 4);
        assert_eq!(model.residuals("PU9->DO8").expect("residuals").len(), 4);
    }

    #[test]
    fn spatial_piecewise_residual_kriging_improves_synthetic_spatial_panel() {
        let mut rows = Vec::new();
        let coordinates = BTreeMap::from([
            ("PU1->DO1".to_string(), (0.0, 0.0)),
            ("PU2->DO2".to_string(), (1.0, 0.0)),
            ("PU3->DO3".to_string(), (2.0, 0.0)),
        ]);
        let offsets = BTreeMap::from([("PU1->DO1", 0.0), ("PU2->DO2", 4.0), ("PU3->DO3", 8.0)]);
        for day in 1..=8 {
            for (series_id, offset) in &offsets {
                rows.push(ForecastRow::new(
                    *series_id,
                    ts(day),
                    20.0 + f64::from(day) + offset,
                ));
            }
        }
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let piecewise_config = PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Flat,
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            auto_yearly_seasonality: false,
            auto_daily_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut base =
            PiecewiseLinearSeasonalForecaster::new(piecewise_config.clone()).expect("base config");
        base.fit(&frame).expect("base fit");
        let base_result = base.predict(1).expect("base predict");
        let mut fused = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates,
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config,
            kriging_config: OrdinaryKrigingConfig::new(3.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect("spatial config");
        fused.fit(&frame).expect("spatial fit");
        let fused_result = fused.predict(1).expect("spatial predict");
        let actuals = BTreeMap::from([
            ("PU1->DO1".to_string(), 29.0),
            ("PU2->DO2".to_string(), 33.0),
            ("PU3->DO3".to_string(), 37.0),
        ]);
        let base_mae = base_result
            .predictions()
            .iter()
            .map(|prediction| (prediction.mean - actuals[&prediction.series_id]).abs())
            .sum::<f64>()
            / actuals.len() as f64;
        let fused_mae = fused_result
            .predictions()
            .iter()
            .map(|prediction| (prediction.mean - actuals[&prediction.series_id]).abs())
            .sum::<f64>()
            / actuals.len() as f64;
        assert!(fused_mae < base_mae);
        assert_eq!(
            fused_result.details().len(),
            fused_result.predictions().len()
        );
        let json = fused_result.to_json_value();
        let first = &json["records"][0];
        assert!(first.get("base_mean").is_some());
        assert!(first.get("spatial_correction").is_some());
        assert!(first.get("kriging_variance").is_some());
        assert!(first.get("selected_neighbors").is_some());
        assert_eq!(first["metadata"]["neighbor_count"].as_u64(), Some(3));
        assert!(first["metadata"]["correction_magnitude"].is_number());
        assert!(first["metadata"]["kriging_variance"].is_number());
        assert!(first["metadata"]["fit_runtime_seconds"].is_number());
    }

    #[test]
    fn spatial_piecewise_kriging_errors_for_missing_coordinate() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO1", ts(1), 10.0),
                ForecastRow::new("PU1->DO1", ts(2), 11.0),
                ForecastRow::new("PU2->DO2", ts(1), 20.0),
                ForecastRow::new("PU2->DO2", ts(2), 21.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates: BTreeMap::from([("PU1->DO1".to_string(), (0.0, 0.0))]),
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config: PiecewiseLinearSeasonalConfig::default(),
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect("spatial config");
        let err = model
            .fit(&frame)
            .expect_err("missing coordinate should fail");
        assert!(err
            .to_string()
            .contains("missing spatial piecewise kriging coordinate"));
    }

    #[test]
    fn spatial_piecewise_kriging_rejects_future_spatial_regressor_leakage() {
        let mut piecewise_config = PiecewiseLinearSeasonalConfig::default();
        piecewise_config
            .future_regressors
            .insert("traffic_density".to_string(), vec![1.0]);
        let err = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates: BTreeMap::from([("PU1->DO1".to_string(), (0.0, 0.0))]),
            mode: SpatialPiecewiseKrigingMode::KrigedRegressors,
            piecewise_config,
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: vec!["traffic_density".to_string()],
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect_err("future spatial regressor should fail");
        assert!(err.to_string().contains("would leak future observations"));
    }

    #[test]
    fn spatial_piecewise_kriging_neighbor_fallback_is_opt_in_and_flagged() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO1", ts(1), 10.0),
                ForecastRow::new("PU1->DO1", ts(2), 11.0),
                ForecastRow::new("PU2->DO2", ts(1), 20.0),
                ForecastRow::new("PU2->DO2", ts(2), 21.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = SpatialPiecewiseKrigingConfig {
            coordinates: BTreeMap::from([
                ("PU1->DO1".to_string(), (0.0, 0.0)),
                ("PU2->DO2".to_string(), (100.0, 0.0)),
            ]),
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config: PiecewiseLinearSeasonalConfig::default(),
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6)
                .and_then(|config| config.with_neighbor_limits(None, 3, None))
                .expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        };
        let mut strict =
            SpatialPiecewiseKrigingForecaster::new(base_config.clone()).expect("strict config");
        strict.fit(&frame).expect("strict fit");
        strict
            .predict(1)
            .expect_err("neighbor rule should fail without fallback");

        let mut fallback = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            allow_neighbor_fallback: true,
            ..base_config
        })
        .expect("fallback config");
        fallback.fit(&frame).expect("fallback fit");
        let result = fallback.predict(1).expect("fallback predict");
        assert!(result.details().iter().all(|detail| {
            detail
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("neighbor_fallback"))
                .and_then(Value::as_bool)
                == Some(true)
        }));
    }

    #[test]
    fn theta_additive_seasonality_reseasons_forecast() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    let base = f64::from(day);
                    let seasonal = if day % 2 == 0 { 5.0 } else { -5.0 };
                    ForecastRow::single(ts(day), 20.0 + base + seasonal)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::additive(2).expect("valid season");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.5, Some(seasonality)).expect("valid theta");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| row.mean)
            .collect::<Vec<_>>();

        assert_eq!(means.len(), 2);
        assert!(means[1] > means[0]);
    }

    #[test]
    fn theta_multiplicative_rejects_non_positive_values() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 0.0),
                ForecastRow::single(ts(4), 4.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::multiplicative(2).expect("valid season");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.5, Some(seasonality)).expect("valid theta");

        let err = model.fit(&frame).expect_err("non-positive values rejected");

        assert!(err.to_string().contains("non-positive"));
    }

    #[test]
    fn piecewise_linear_seasonal_forecaster_projects_trend_and_weekly_pattern() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let weekly = if day % 7 == 0 { 8.0 } else { 0.0 };
                    ForecastRow::single(ts(day), 30.0 + 1.5 * f64::from(day) + weekly)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 3,
            weekly_fourier_order: 3,
            seasonality_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid piecewise seasonal config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(7).expect("predict");
        let predictions = forecast.predictions();

        assert_eq!(predictions.len(), 7);
        assert!(predictions[6].mean > predictions[0].mean);
        assert_eq!(predictions[0].model, "piecewise_linear_seasonal");
        assert_eq!(model.metadata()["weekly_fourier_order"].as_u64(), Some(3));
    }

    #[test]
    fn piecewise_linear_default_changepoint_range_tracks_late_weekly_lane_movement() {
        let start = ts(1);
        let rows = (0..140)
            .map(|week| {
                let late_ramp = if week >= 92 {
                    55.0 * f64::from(week - 92)
                } else {
                    0.0
                };
                let current_ramp = if week >= 118 {
                    140.0 * f64::from(week - 118)
                } else {
                    0.0
                };
                let annual = 70.0 * (std::f64::consts::TAU * f64::from(week) / 52.18).sin();
                ForecastRow::new(
                    "PULocationID=132->DOLocationID=236",
                    start + Duration::weeks(i64::from(week)),
                    1100.0 + 2.2 * f64::from(week) + late_ramp + current_ramp + annual,
                )
            })
            .collect::<Vec<_>>();
        let default_rmse = rolling_one_week_rmse(
            &rows,
            PiecewiseLinearSeasonalConfig {
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                changepoint_l2_regularization: 0.001,
                ..PiecewiseLinearSeasonalConfig::default()
            },
        );
        let strict_prophet_range_rmse = rolling_one_week_rmse(
            &rows,
            PiecewiseLinearSeasonalConfig {
                changepoint_range: 0.8,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                changepoint_l2_regularization: 0.001,
                ..PiecewiseLinearSeasonalConfig::default()
            },
        );

        assert!(default_rmse < strict_prophet_range_rmse * 0.4);
    }

    fn rolling_one_week_rmse(rows: &[ForecastRow], config: PiecewiseLinearSeasonalConfig) -> f64 {
        let mut sum_squared = 0.0;
        for holdout_weeks_back in (1..=12).rev() {
            let cutoff = rows.len() - holdout_weeks_back;
            let train = rows[..cutoff].to_vec();
            let actual = rows[cutoff].target;
            let frame = ForecastFrame::new(train, ForecastFrequency::Weekly).expect("valid frame");
            let mut model =
                PiecewiseLinearSeasonalForecaster::new(config.clone()).expect("valid config");
            model.fit(&frame).expect("fit");
            let prediction = model.predict(1).expect("predict").predictions()[0].mean;
            let error = prediction - actual;
            sum_squared += error * error;
        }
        (sum_squared / 12.0).sqrt()
    }

    #[test]
    fn piecewise_linear_auto_seasonalities_resolve_from_training_span() {
        let hourly_frame = ForecastFrame::new(
            (0..72)
                .map(|hour| {
                    ForecastRow::single(
                        ts(1) + Duration::hours(i64::from(hour)),
                        50.0 + 0.1 * f64::from(hour),
                    )
                })
                .collect(),
            ForecastFrequency::Hourly,
        )
        .expect("valid hourly frame");
        let long_daily_frame = ForecastFrame::new(
            (0..800)
                .map(|day| {
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day)),
                        80.0 + 0.05 * f64::from(day),
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid long daily frame");
        let mut hourly = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            weekly_fourier_order: 0,
            daily_fourier_order: 0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid hourly auto config");
        let mut long_daily =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                yearly_fourier_order: 0,
                weekly_fourier_order: 0,
                daily_fourier_order: 0,
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid long daily auto config");
        let mut disabled = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            yearly_fourier_order: 0,
            weekly_fourier_order: 0,
            auto_yearly_seasonality: false,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid disabled auto config");

        hourly.fit(&hourly_frame).expect("hourly fit");
        long_daily.fit(&long_daily_frame).expect("long daily fit");
        disabled.fit(&long_daily_frame).expect("disabled fit");

        assert_eq!(hourly.metadata()["daily_fourier_order"].as_u64(), Some(4));
        assert_eq!(
            long_daily.metadata()["yearly_fourier_order"].as_u64(),
            Some(10)
        );
        assert_eq!(
            long_daily.metadata()["weekly_fourier_order"].as_u64(),
            Some(3)
        );
        assert_eq!(
            disabled.metadata()["yearly_fourier_order"].as_u64(),
            Some(0)
        );
        assert_eq!(
            disabled.metadata()["weekly_fourier_order"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn piecewise_linear_seasonal_components_reconstruct_predictions() {
        let rows = (1..=35)
            .map(|day| {
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                let timestamp = ts(1) + Duration::days(i64::from(day - 1));
                ForecastRow::with_covariates(
                    "PU1->DO2",
                    timestamp,
                    50.0 + f64::from(day) + 20.0 * queue,
                    BTreeMap::from([("airport_queue".to_string(), queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 2,
            seasonality_l2_regularization: 0.001,
            regressor_l2_regularization: 0.001,
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid piecewise config");
        model.fit(&frame).expect("fit");

        let forecast = model.predict(3).expect("predict");
        let components = model
            .predict_components_json_value(3)
            .expect("component forecast");
        let history_components = model
            .history_components_json_value()
            .expect("history components");
        let records = components["records"].as_array().expect("records");
        let history_records = history_components["records"]
            .as_array()
            .expect("history records");

        assert_eq!(records.len(), 3);
        assert_eq!(history_records.len(), 35);
        assert_eq!(
            records[0]["prediction"]
                .as_f64()
                .expect("component prediction"),
            forecast.predictions()[0].mean
        );
        assert!(records[0]["components"]["weekly"].as_f64().is_some());
        assert!(history_records[0]["components"]["weekly"]
            .as_f64()
            .is_some());
        assert!(history_records[1]["trend_movement"].as_f64().is_some());
        let reconstructed_residual = history_records[0]["actual"]
            .as_f64()
            .expect("history actual")
            - history_records[0]["fitted"]
                .as_f64()
                .expect("history fitted");
        let emitted_residual = history_records[0]["residual"]
            .as_f64()
            .expect("history residual");
        assert!((reconstructed_residual - emitted_residual).abs() < 1.0e-9);
        assert!(
            records[0]["components"]["regressors"]["airport_queue"]
                .as_f64()
                .expect("airport queue contribution")
                > 10.0
        );
        assert!(components["columns"]
            .as_array()
            .expect("columns")
            .iter()
            .any(|column| column.as_str() == Some("components")));
    }

    #[test]
    fn piecewise_linear_seasonal_flat_growth_suppresses_trend_extrapolation() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| ForecastRow::single(ts(day), 40.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut linear = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid linear config");
        let mut flat = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Flat,
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid flat config");

        linear.fit(&frame).expect("linear fit");
        flat.fit(&frame).expect("flat fit");
        let linear_forecast = linear.predict(3).expect("linear predict");
        let flat_forecast = flat.predict(3).expect("flat predict");

        assert!(linear_forecast.predictions()[2].mean > flat_forecast.predictions()[2].mean + 20.0);
        assert_eq!(flat.metadata()["growth"].as_str(), Some("flat"));
    }

    #[test]
    fn piecewise_linear_seasonal_forecaster_emits_prediction_intervals() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let noise = if day % 2 == 0 { 0.4 } else { -0.4 };
                    ForecastRow::single(ts(day), 30.0 + 1.5 * f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 2,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8, 0.95],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid piecewise seasonal config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.intervals().len(), 4);
        assert!(forecast
            .intervals()
            .iter()
            .all(|interval| interval.lower <= interval.upper));
        assert!(forecast
            .intervals()
            .iter()
            .any(|interval| (interval.level - 0.95).abs() < 1.0e-12));
    }

    #[test]
    fn piecewise_linear_skips_coefficient_covariance_for_point_only_fit() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| ForecastRow::single(ts(day), 40.0 + 0.8 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut point_only =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 2,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("point-only config");
        let mut interval_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 2,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                interval_levels: vec![0.8],
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("interval config");

        point_only.fit(&frame).expect("point fit");
        interval_model.fit(&frame).expect("interval fit");
        let point_series = point_only
            .fitted
            .as_ref()
            .expect("point fitted")
            .series
            .get("__single__")
            .expect("point series");
        let interval_series = interval_model
            .fitted
            .as_ref()
            .expect("interval fitted")
            .series
            .get("__single__")
            .expect("interval series");

        assert!(point_series.coefficient_covariance.is_empty());
        assert!(!interval_series.coefficient_covariance.is_empty());
    }

    #[test]
    fn piecewise_linear_coefficient_uncertainty_widens_intervals() {
        let frame = ForecastFrame::new(
            (1..=12)
                .map(|day| {
                    let noise = if day % 2 == 0 { 1.0 } else { -1.0 };
                    ForecastRow::single(ts(day), 25.0 + 1.2 * f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut residual_only =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                coefficient_uncertainty_scale: 0.0,
                ..base_config.clone()
            })
            .expect("residual interval config");
        let mut posterior_like =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                coefficient_uncertainty_scale: 3.0,
                ..base_config
            })
            .expect("coefficient uncertainty config");

        residual_only.fit(&frame).expect("residual fit");
        posterior_like
            .fit(&frame)
            .expect("coefficient uncertainty fit");
        let residual_forecast = residual_only.predict(5).expect("residual predict");
        let posterior_forecast = posterior_like
            .predict(5)
            .expect("coefficient uncertainty predict");
        let residual_interval = &residual_forecast.intervals()[4];
        let posterior_interval = &posterior_forecast.intervals()[4];
        let residual_width = residual_interval.upper - residual_interval.lower;
        let posterior_width = posterior_interval.upper - posterior_interval.lower;

        assert_eq!(
            residual_forecast.predictions()[4].mean,
            posterior_forecast.predictions()[4].mean
        );
        assert!(posterior_width > residual_width);
        assert_eq!(
            posterior_like.metadata()["coefficient_uncertainty_scale"].as_f64(),
            Some(3.0)
        );
    }

    #[test]
    fn piecewise_linear_uncertainty_samples_widen_future_intervals() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| {
                    let t = f64::from(day);
                    let after_break = (t - 15.0).max(0.0);
                    ForecastRow::single(ts(day), 20.0 + 0.5 * t + 3.0 * after_break)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 1,
            changepoint_timestamps: vec![ts(15)],
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            changepoint_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut residual_only = PiecewiseLinearSeasonalForecaster::new(base_config.clone())
            .expect("valid residual interval config");
        let mut trend_uncertain =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                uncertainty_samples: 256,
                trend_uncertainty_scale: 1.0,
                uncertainty_seed: 7,
                ..base_config.clone()
            })
            .expect("valid trend uncertainty config");
        let mut normal_uncertain =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                uncertainty_samples: 256,
                trend_uncertainty_policy: PiecewiseLinearTrendUncertaintyPolicy::Normal,
                trend_uncertainty_scale: 1.0,
                uncertainty_seed: 7,
                ..base_config
            })
            .expect("valid normal trend uncertainty config");

        residual_only.fit(&frame).expect("residual fit");
        trend_uncertain.fit(&frame).expect("uncertain fit");
        normal_uncertain.fit(&frame).expect("normal uncertain fit");
        let residual_forecast = residual_only.predict(5).expect("residual predict");
        let uncertain_forecast = trend_uncertain.predict(5).expect("uncertain predict");
        let normal_forecast = normal_uncertain
            .predict(5)
            .expect("normal uncertain predict");
        let residual_interval = &residual_forecast.intervals()[4];
        let uncertain_interval = &uncertain_forecast.intervals()[4];
        let normal_interval = &normal_forecast.intervals()[4];
        let residual_width = residual_interval.upper - residual_interval.lower;
        let uncertain_width = uncertain_interval.upper - uncertain_interval.lower;
        let normal_width = normal_interval.upper - normal_interval.lower;

        assert!(uncertain_width > residual_width + 1.0);
        assert!((uncertain_width - normal_width).abs() > 1.0e-6);
        assert_eq!(
            trend_uncertain.metadata()["uncertainty_samples"].as_u64(),
            Some(256)
        );
        assert_eq!(
            trend_uncertain.metadata()["trend_uncertainty_policy"].as_str(),
            Some("laplace")
        );
        assert_eq!(
            normal_uncertain.metadata()["trend_uncertainty_policy"].as_str(),
            Some("normal")
        );
    }

    #[test]
    fn piecewise_linear_predictive_samples_round_trip_with_artifact() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| {
                    let noise = if day % 2 == 0 { 0.6 } else { -0.6 };
                    ForecastRow::single(ts(day), 40.0 + 0.8 * f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 1,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            uncertainty_samples: 8,
            uncertainty_seed: 11,
            coefficient_uncertainty_scale: 1.5,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid samples config");

        model.fit(&frame).expect("fit");
        let samples = model.predict_samples_json_value(3).expect("samples");
        let records = samples["records"].as_array().expect("sample records");
        let payload = model.to_json_string().expect("serialize artifact");
        let restored = PiecewiseLinearSeasonalForecaster::from_json_string(&payload)
            .expect("restore artifact");
        let restored_samples = restored
            .predict_samples_json_value(3)
            .expect("restored samples");
        let restored_records = restored_samples["records"]
            .as_array()
            .expect("restored sample records");

        assert_eq!(samples["sample_count"].as_u64(), Some(8));
        assert_eq!(restored_samples["sample_count"].as_u64(), Some(8));
        assert_eq!(records.len(), 24);
        assert_eq!(restored_records.len(), records.len());
        for (record, restored_record) in records.iter().zip(restored_records.iter()) {
            assert_eq!(record["series_id"], restored_record["series_id"]);
            assert_eq!(record["timestamp"], restored_record["timestamp"]);
            assert_eq!(record["horizon"], restored_record["horizon"]);
            assert_eq!(record["sample"], restored_record["sample"]);
            for field in [
                "prediction",
                "mean",
                "residual_draw",
                "coefficient_draw",
                "trend_draw",
            ] {
                let left = record[field].as_f64().expect("numeric sample field");
                let right = restored_record[field]
                    .as_f64()
                    .expect("restored numeric sample field");
                assert!((left - right).abs() < 1.0e-12);
            }
        }
        assert!(records
            .iter()
            .any(|record| record["coefficient_draw"].as_f64().unwrap().abs() > 0.0));
        assert!(records
            .iter()
            .any(|record| record["residual_draw"].as_f64().unwrap().abs() > 0.0));
    }

    #[test]
    fn piecewise_linear_artifact_round_trips_fitted_state() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    ForecastRow::with_covariates(
                        "__single__",
                        ts(day),
                        50.0 + f64::from(day) + 20.0 * queue,
                        BTreeMap::from([("airport_queue".to_string(), queue)]),
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 1,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid config");

        model.fit(&frame).expect("fit");
        let before = model.predict(3).expect("predict before");
        let payload = model.to_json_string().expect("serialize artifact");
        let loaded =
            PiecewiseLinearSeasonalForecaster::from_json_string(&payload).expect("load artifact");
        let after = loaded.predict(3).expect("predict after");

        assert_eq!(before.to_json_value(), after.to_json_value());
        assert_eq!(
            serde_json::from_str::<Value>(&payload).expect("artifact json")["kind"].as_str(),
            Some(PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND)
        );
    }

    #[test]
    fn piecewise_linear_seasonal_logistic_growth_respects_bounds() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 14.0;
                    let value = 5.0 + 90.0 / (1.0 + (-0.25 * t).exp());
                    ForecastRow::single(ts(day), value)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 4,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(7).expect("predict");

        assert_eq!(model.metadata()["growth"].as_str(), Some("logistic"));
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean > 0.0 && prediction.mean < 100.0));
    }

    #[test]
    fn piecewise_linear_logistic_trend_uncertainty_uses_inverse_link_scale() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 8.0;
                    let value = 100.0 / (1.0 + (-0.55 * t).exp());
                    ForecastRow::single(ts(day), value.clamp(0.001, 99.999))
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid saturated logistic frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 2,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            uncertainty_samples: 16,
            trend_uncertainty_scale: 10.0,
            uncertainty_seed: 13,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic uncertainty config");

        model.fit(&frame).expect("fit");
        let fitted = model.fitted.as_ref().expect("fitted");
        let series = fitted.series.get("__single__").expect("single series");
        let timestamp = ForecastFrequency::Daily
            .advance(series.last_timestamp, 3)
            .expect("future timestamp");
        let elapsed = elapsed_days(series.start_timestamp, timestamp);
        let bounds =
            piecewise_bounds(Some("__single__"), None, Some(3), &model.config).expect("bounds");
        let linear_predictor = predict_piecewise_linear_value(
            elapsed,
            &series.coefficients,
            &PiecewiseLinearFeatureContext {
                series_id: Some("__single__"),
                timestamp,
                covariates: None,
                horizon_step: Some(3),
                component_multiplier: series.component_multiplier(elapsed, bounds, &model.config),
                changepoints: &series.changepoints,
                config: &model.config,
                regressor_stats: Some(&series.regressor_stats),
            },
        )
        .expect("linear predictor");
        let derivative =
            inverse_piecewise_target_derivative(linear_predictor, bounds, &model.config);
        let offsets = series
            .trend_uncertainty_offsets("__single__", elapsed, timestamp, 3, &model.config)
            .expect("trend offsets");

        assert!(derivative < 1.0);
        assert_eq!(offsets.len(), 16);
        assert!(offsets.iter().all(|offset| offset.is_finite()));
    }

    #[test]
    fn piecewise_linear_logistic_predictive_samples_respect_bounds() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 12.0;
                    let value = 5.0 + 90.0 / (1.0 + (-0.35 * t).exp());
                    ForecastRow::single(ts(day), value)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid logistic frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            uncertainty_samples: 32,
            trend_uncertainty_scale: 20.0,
            coefficient_uncertainty_scale: 8.0,
            uncertainty_seed: 17,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic sample config");

        model.fit(&frame).expect("fit");
        let samples = model.predict_samples_json_value(5).expect("samples");
        let records = samples["records"].as_array().expect("sample records");

        assert_eq!(records.len(), 5 * 32);
        assert!(records.iter().all(|record| {
            let prediction = record["prediction"].as_f64().expect("sample prediction");
            prediction > 0.0 && prediction < 100.0
        }));
    }

    #[test]
    fn piecewise_linear_logistic_quantiles_respect_inverse_link_bounds() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 12.0;
                    let value = 5.0 + 90.0 / (1.0 + (-0.35 * t).exp());
                    ForecastRow::single(ts(day), value)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid logistic frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            quantile_levels: vec![0.05, 0.5, 0.95],
            uncertainty_samples: 32,
            trend_uncertainty_scale: 20.0,
            coefficient_uncertainty_scale: 8.0,
            uncertainty_seed: 23,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic quantile config");

        model.fit(&frame).expect("fit");
        let quantiles = model
            .predict_quantiles_json_value(5, None)
            .expect("quantiles");
        let records = quantiles["records"].as_array().expect("quantile records");

        assert_eq!(records.len(), 5 * 3);
        assert!(records.iter().all(|record| {
            let prediction = record["prediction"].as_f64().expect("quantile prediction");
            prediction > 0.0 && prediction < 100.0
        }));
    }

    #[test]
    fn piecewise_linear_logistic_prediction_intervals_respect_bounds() {
        let frame = ForecastFrame::new(
            (1..=24)
                .map(|day| {
                    let t = f64::from(day) - 10.0;
                    let noise = if day % 2 == 0 { 1.0 } else { -1.0 };
                    let value = 5.0 + 90.0 / (1.0 + (-0.4 * t).exp()) + noise;
                    ForecastRow::single(ts(day), value.clamp(5.001, 94.999))
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid logistic interval frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8, 0.95],
            uncertainty_samples: 32,
            trend_uncertainty_scale: 20.0,
            coefficient_uncertainty_scale: 8.0,
            uncertainty_seed: 19,
            cap: Some(95.0),
            floor: 5.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic interval config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(5).expect("predict");

        assert_eq!(forecast.intervals().len(), 10);
        assert!(forecast
            .intervals()
            .iter()
            .all(|interval| interval.lower >= 5.0 && interval.upper <= 95.0));
    }

    #[test]
    fn piecewise_linear_seasonal_logistic_growth_uses_dynamic_capacity() {
        let rows = (1..=28)
            .map(|day| {
                let cap = 80.0 + f64::from(day);
                let t = f64::from(day) - 14.0;
                let target = cap / (1.0 + (-0.2 * t).exp());
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    target,
                    BTreeMap::from([("zone_capacity".to_string(), cap)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let future_caps = vec![109.0, 110.0, 111.0];
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap_regressor: Some("zone_capacity".to_string()),
            future_regressors: BTreeMap::from([("zone_capacity".to_string(), future_caps.clone())]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid dynamic cap config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");

        assert_eq!(
            model.metadata()["cap_regressor"].as_str(),
            Some("zone_capacity")
        );
        assert!(forecast
            .predictions()
            .iter()
            .zip(future_caps.iter())
            .all(|(prediction, cap)| prediction.mean > 0.0 && prediction.mean < *cap));
    }

    #[test]
    fn piecewise_linear_logistic_cap_regressor_can_be_series_specific() {
        let rows = ["A", "B"]
            .into_iter()
            .flat_map(|series| {
                (1..=28).map(move |day| {
                    let cap = if series == "A" {
                        110.0 + f64::from(day) * 0.25
                    } else {
                        65.0 + f64::from(day) * 0.10
                    };
                    let t = f64::from(day) - 14.0;
                    let target = cap / (1.0 + (-0.18 * t).exp());
                    ForecastRow::with_covariates(
                        series,
                        ts(day),
                        target,
                        BTreeMap::from([("zone_capacity".to_string(), cap)]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap_regressor: Some("zone_capacity".to_string()),
            future_regressors_by_series: BTreeMap::from([
                (
                    "A".to_string(),
                    BTreeMap::from([("zone_capacity".to_string(), vec![120.0])]),
                ),
                (
                    "B".to_string(),
                    BTreeMap::from([("zone_capacity".to_string(), vec![70.0])]),
                ),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid panel dynamic cap config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(1).expect("predict");
        let a = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A forecast")
            .mean;
        let b = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "B")
            .expect("series B forecast")
            .mean;

        assert!(a > 0.0 && a < 120.0);
        assert!(b > 0.0 && b < 70.0);
        assert!(a > b + 20.0);
    }

    #[test]
    fn piecewise_linear_logistic_floor_regressor_can_be_series_specific() {
        let cap = 140.0;
        let rows = ["A", "B"]
            .into_iter()
            .flat_map(|series| {
                (1..=28).map(move |day| {
                    let floor = if series == "A" {
                        32.0 + f64::from(day) * 0.10
                    } else {
                        8.0 + f64::from(day) * 0.05
                    };
                    let t = f64::from(day) - 14.0;
                    let target = floor + (cap - floor) / (1.0 + (-0.18 * t).exp());
                    ForecastRow::with_covariates(
                        series,
                        ts(day),
                        target,
                        BTreeMap::from([("service_floor".to_string(), floor)]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap: Some(cap),
            floor_regressor: Some("service_floor".to_string()),
            future_regressors_by_series: BTreeMap::from([
                (
                    "A".to_string(),
                    BTreeMap::from([("service_floor".to_string(), vec![38.0])]),
                ),
                (
                    "B".to_string(),
                    BTreeMap::from([("service_floor".to_string(), vec![10.0])]),
                ),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid panel dynamic floor config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(1).expect("predict");
        let a = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A forecast")
            .mean;
        let b = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "B")
            .expect("series B forecast")
            .mean;
        let mut lower_floor_model = model.clone();
        lower_floor_model
            .update_config(|config| {
                config.future_regressors_by_series.insert(
                    "A".to_string(),
                    BTreeMap::from([("service_floor".to_string(), vec![5.0])]),
                );
            })
            .expect("lower future floor config");
        let lower_floor_a = lower_floor_model
            .predict(1)
            .expect("lower floor predict")
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A lower floor forecast")
            .mean;

        assert!(a > 38.0 && a < cap);
        assert!(b > 10.0 && b < cap);
        assert!(a > lower_floor_a);
        assert_eq!(
            model.metadata()["floor_regressor"].as_str(),
            Some("service_floor")
        );
    }

    #[test]
    fn piecewise_linear_seasonal_explicit_changepoint_projects_break() {
        let rows = (1..=30)
            .map(|day| {
                let target = if day <= 15 {
                    50.0 + f64::from(day)
                } else {
                    65.0 + 5.0 * f64::from(day - 15)
                };
                ForecastRow::single(ts(day), target)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut explicit = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            changepoint_l2_regularization: 0.001,
            changepoint_timestamps: vec![ts(15)],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid explicit changepoint config");

        baseline.fit(&frame).expect("baseline fit");
        explicit.fit(&frame).expect("explicit fit");
        let baseline_forecast = baseline.predict(3).expect("baseline predict");
        let explicit_forecast = explicit.predict(3).expect("explicit predict");

        assert!(
            explicit_forecast.predictions()[2].mean > baseline_forecast.predictions()[2].mean + 8.0
        );
        assert_eq!(
            explicit.metadata()["changepoint_timestamps"][0].as_str(),
            Some("2026-01-15T00:00:00")
        );
    }

    #[test]
    fn piecewise_linear_seasonal_changepoint_l1_shrinks_deltas() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| ForecastRow::single(ts(day), 20.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 5,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            changepoint_l2_regularization: 0.001,
            changepoint_l1_regularization: 10.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid sparse changepoint config");

        model.fit(&frame).expect("fit");
        let fitted = model.fitted.as_ref().expect("fitted");
        let series = fitted.series.get("__single__").expect("single series");
        let max_delta = series.coefficients[2..2 + series.changepoints.len()]
            .iter()
            .map(|coefficient| coefficient.abs())
            .fold(0.0_f64, f64::max);

        assert!(max_delta < 1.0e-6);
        assert_eq!(
            model.metadata()["changepoint_l1_regularization"].as_f64(),
            Some(10.0)
        );
    }

    #[test]
    fn piecewise_linear_huber_loss_resists_large_outlier() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| {
                    let clean = 50.0 + 1.5 * f64::from(day);
                    let outlier = if day == 30 { 180.0 } else { 0.0 };
                    ForecastRow::single(ts(day), clean + outlier)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            changepoint_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut squared =
            PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("squared config");
        let mut huber = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            fit_loss: PiecewiseLinearFitLoss::Huber,
            huber_delta: 1.345,
            irls_iterations: 8,
            ..base_config
        })
        .expect("huber config");

        squared.fit(&frame).expect("squared fit");
        huber.fit(&frame).expect("huber fit");
        let squared_prediction = squared.predict(1).expect("squared predict").predictions()[0].mean;
        let huber_prediction = huber.predict(1).expect("huber predict").predictions()[0].mean;
        let clean_next = 50.0 + 1.5 * 31.0;

        assert!((huber_prediction - clean_next).abs() < (squared_prediction - clean_next).abs());
        assert_eq!(huber.metadata()["fit_loss"].as_str(), Some("huber"));
        assert_eq!(huber.metadata()["irls_iterations"].as_u64(), Some(8));
    }

    #[test]
    fn piecewise_linear_seasonal_rejects_invalid_changepoint_range() {
        let err = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoint_range: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect_err("invalid range rejected");

        assert!(err.to_string().contains("changepoint_range"));
    }

    #[test]
    fn piecewise_linear_seasonal_event_window_carries_future_effect() {
        let train_event_timestamp = ts(15);
        let future_event_timestamp = NaiveDate::from_ymd_opt(2026, 2, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid event timestamp");
        let rows = (1..=30)
            .map(|day| {
                let event_boost = if (14..=16).contains(&day) { 25.0 } else { 0.0 };
                ForecastRow::single(ts(day), 100.0 + 0.5 * f64::from(day) + event_boost)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut event_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 0,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                event_l2_regularization: 0.001,
                events: vec![
                    PiecewiseLinearEvent {
                        name: "airport_surge".to_string(),
                        timestamp: train_event_timestamp,
                        lower_window: -1,
                        upper_window: 1,
                    },
                    PiecewiseLinearEvent {
                        name: "airport_surge".to_string(),
                        timestamp: future_event_timestamp,
                        lower_window: -1,
                        upper_window: 1,
                    },
                ],
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid event config");

        baseline.fit(&frame).expect("baseline fit");
        event_model.fit(&frame).expect("event fit");
        let baseline_forecast = baseline.predict(3).expect("baseline predict");
        let event_forecast = event_model.predict(3).expect("event predict");

        assert!(
            event_forecast.predictions()[1].mean > baseline_forecast.predictions()[1].mean + 10.0
        );
        assert_eq!(
            event_model.metadata()["events"][0]["name"].as_str(),
            Some("airport_surge")
        );
    }

    #[test]
    fn piecewise_linear_event_window_offsets_get_separate_effects() {
        let future_event_timestamp = NaiveDate::from_ymd_opt(2026, 2, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid event timestamp");
        let rows = (1..=30)
            .map(|day| {
                let event_offset = [10, 20]
                    .iter()
                    .find_map(|event_day| {
                        let offset = day - event_day;
                        (-1..=1).contains(&offset).then_some(offset)
                    })
                    .unwrap_or(99);
                let event_effect = match event_offset {
                    -1 => 4.0,
                    0 => 25.0,
                    1 => -12.0,
                    _ => 0.0,
                };
                ForecastRow::single(ts(day as u32), 100.0 + 0.1 * f64::from(day) + event_effect)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            event_l2_regularization: 0.001,
            events: vec![
                PiecewiseLinearEvent {
                    name: "airport_surge".to_string(),
                    timestamp: ts(10),
                    lower_window: -1,
                    upper_window: 1,
                },
                PiecewiseLinearEvent {
                    name: "airport_surge".to_string(),
                    timestamp: ts(20),
                    lower_window: -1,
                    upper_window: 1,
                },
                PiecewiseLinearEvent {
                    name: "airport_surge".to_string(),
                    timestamp: future_event_timestamp,
                    lower_window: -1,
                    upper_window: 1,
                },
            ],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid event window config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");
        let predictions = forecast.predictions();
        let components = model
            .predict_components_json_value(3)
            .expect("component forecast");
        let offset_components = components["records"][0]["components"]["event_window_offsets"]
            .as_object()
            .expect("event offset components");

        assert!(predictions[1].mean > predictions[0].mean + 12.0);
        assert!(predictions[0].mean > predictions[2].mean + 8.0);
        assert!(
            offset_components
                .get("airport_surge[-1]")
                .and_then(Value::as_f64)
                .expect("day-before contribution")
                > 2.0
        );
    }

    #[test]
    fn piecewise_linear_seasonal_extra_regressor_uses_future_values() {
        let rows = (1..=30)
            .map(|day| {
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    50.0 + f64::from(day) + 20.0 * queue,
                    BTreeMap::from([("airport_queue".to_string(), queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut regressor_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 0,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                regressor_l2_regularization: 0.001,
                extra_regressors: vec!["airport_queue".to_string()],
                future_regressors: BTreeMap::from([(
                    "airport_queue".to_string(),
                    vec![1.0, 0.0, 0.0],
                )]),
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid regressor config");

        baseline.fit(&frame).expect("baseline fit");
        regressor_model.fit(&frame).expect("regressor fit");
        let baseline_forecast = baseline.predict(3).expect("baseline predict");
        let regressor_forecast = regressor_model.predict(3).expect("regressor predict");

        assert!(
            regressor_forecast.predictions()[0].mean
                > baseline_forecast.predictions()[0].mean + 10.0
        );
        assert_eq!(
            regressor_model.metadata()["extra_regressors"][0].as_str(),
            Some("airport_queue")
        );
    }

    #[test]
    fn piecewise_linear_extra_regressor_future_values_can_be_series_specific() {
        let rows = ["A", "B"]
            .into_iter()
            .flat_map(|series| {
                (1..=30).map(move |day| {
                    let queue = if day % 3 == 0 { 1.0 } else { 0.0 };
                    ForecastRow::with_covariates(
                        series,
                        ts(day),
                        75.0 + f64::from(day) + 18.0 * queue,
                        BTreeMap::from([("airport_queue".to_string(), queue)]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            regressor_l2_regularization: 0.001,
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors_by_series: BTreeMap::from([
                (
                    "A".to_string(),
                    BTreeMap::from([("airport_queue".to_string(), vec![1.0])]),
                ),
                (
                    "B".to_string(),
                    BTreeMap::from([("airport_queue".to_string(), vec![0.0])]),
                ),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid per-series regressor config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(1).expect("predict");
        let a = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A forecast")
            .mean;
        let b = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "B")
            .expect("series B forecast")
            .mean;

        assert!(a > b + 10.0);
        assert!(model.metadata()["future_regressors"]
            .as_object()
            .unwrap()
            .is_empty());
        assert_eq!(
            model.metadata()["future_regressors_by_series"]["A"]["airport_queue"][0].as_f64(),
            Some(1.0)
        );
    }

    #[test]
    fn piecewise_linear_extra_regressor_mode_can_be_multiplicative() {
        let rows = (1..=30)
            .map(|day| {
                let trend = 30.0 + 2.0 * f64::from(day);
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    if queue > 0.0 { trend * 1.4 } else { trend },
                    BTreeMap::from([("airport_queue".to_string(), queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            regressor_l2_regularization: 0.001,
            extra_regressors: vec!["airport_queue".to_string()],
            regressor_modes: BTreeMap::from([(
                "airport_queue".to_string(),
                PiecewiseLinearComponentMode::Multiplicative,
            )]),
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid multiplicative regressor config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");

        assert!(forecast.predictions()[0].mean > forecast.predictions()[1].mean + 15.0);
        assert_eq!(
            model.metadata()["regressor_modes"]["airport_queue"].as_str(),
            Some("multiplicative")
        );
    }

    #[test]
    fn piecewise_linear_extra_regressor_monotonic_constraint_clamps_effect() {
        let rows = (1..=30)
            .map(|day| {
                let traffic = if day % 2 == 0 { 10.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    100.0 - 4.0 * traffic,
                    BTreeMap::from([("traffic".to_string(), traffic)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Flat,
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            regressor_l2_regularization: 0.0,
            extra_regressors: vec!["traffic".to_string()],
            extra_regressor_monotonic_constraints: BTreeMap::from([("traffic".to_string(), 1)]),
            future_regressors: BTreeMap::from([("traffic".to_string(), vec![0.0, 10.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid monotone regressor config");

        model.fit(&frame).expect("fit");
        let components = model
            .predict_components_json_value(2)
            .expect("component forecast");
        let records = components["records"].as_array().expect("component records");
        let low = records[0]["components"]["regressors"]["traffic"]
            .as_f64()
            .expect("low traffic contribution");
        let high = records[1]["components"]["regressors"]["traffic"]
            .as_f64()
            .expect("high traffic contribution");

        assert!(high >= low);
        assert_eq!(
            model.metadata()["extra_regressor_monotonic_constraints"]["traffic"].as_i64(),
            Some(1)
        );
    }

    #[test]
    fn piecewise_linear_per_regressor_l2_shrinks_named_effect() {
        let rows = (1..=30)
            .map(|day| {
                let airport_queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    50.0 + f64::from(day) + 24.0 * airport_queue,
                    BTreeMap::from([("airport_queue".to_string(), airport_queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut low_l2 = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            regressor_l2_regularization_by_name: BTreeMap::from([(
                "airport_queue".to_string(),
                0.001,
            )]),
            ..base_config.clone()
        })
        .expect("valid low l2 config");
        let mut high_l2 = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            regressor_l2_regularization_by_name: BTreeMap::from([(
                "airport_queue".to_string(),
                1_000.0,
            )]),
            ..base_config
        })
        .expect("valid high l2 config");

        low_l2.fit(&frame).expect("low l2 fit");
        high_l2.fit(&frame).expect("high l2 fit");
        let low_prediction = low_l2.predict(1).expect("low predict").predictions()[0].mean;
        let high_prediction = high_l2.predict(1).expect("high predict").predictions()[0].mean;

        assert!(low_prediction > high_prediction + 10.0);
        assert_eq!(
            high_l2.metadata()["regressor_l2_regularization_by_name"]["airport_queue"].as_f64(),
            Some(1_000.0)
        );
    }

    #[test]
    fn piecewise_linear_auto_standardizes_continuous_extra_regressors_only() {
        let rows = (1..=30)
            .map(|day| {
                let traffic_index = 100.0 + 4.0 * f64::from(day);
                let airport_queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    20.0 + 0.3 * f64::from(day) + 1.5 * traffic_index + 12.0 * airport_queue,
                    BTreeMap::from([
                        ("traffic_index".to_string(), traffic_index),
                        ("airport_queue".to_string(), airport_queue),
                    ]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            extra_regressors: vec!["traffic_index".to_string(), "airport_queue".to_string()],
            future_regressors: BTreeMap::from([
                ("traffic_index".to_string(), vec![224.0]),
                ("airport_queue".to_string(), vec![1.0]),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid standardized regressor config");

        model.fit(&frame).expect("fit");
        let fitted = model.fitted.as_ref().expect("fitted");
        let series = fitted.series.get("__single__").expect("single series");
        let traffic_stats = series
            .regressor_stats
            .get("traffic_index")
            .expect("traffic stats");
        let queue_stats = series
            .regressor_stats
            .get("airport_queue")
            .expect("queue stats");
        let payload = serde_json::from_str::<Value>(&model.to_json_string().expect("artifact"))
            .expect("artifact json");

        assert!(traffic_stats.standardized);
        assert!(traffic_stats.scale > 1.0);
        assert!(!queue_stats.standardized);
        assert_eq!(
            model.metadata()["regressor_standardization"].as_str(),
            Some("auto")
        );
        assert_eq!(
            payload["model"]["fitted"]["series"]["__single__"]["regressor_stats"]["traffic_index"]
                ["standardized"]
                .as_bool(),
            Some(true)
        );
    }

    #[test]
    fn piecewise_linear_trend_adjustments_shift_future_trend() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut baseline =
            PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("baseline config");
        let mut adjusted = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            trend_adjustments: BTreeMap::from([(2, 1.20)]),
            ..base_config
        })
        .expect("adjusted config");

        baseline.fit(&frame).expect("baseline fit");
        adjusted.fit(&frame).expect("adjusted fit");
        let baseline_forecast = baseline.predict(2).expect("baseline predict");
        let adjusted_forecast = adjusted.predict(2).expect("adjusted predict");
        let baseline_second = baseline_forecast.predictions()[1].mean;
        let adjusted_second = adjusted_forecast.predictions()[1].mean;
        let components = adjusted
            .predict_components_json_value(2)
            .expect("components");
        let second_record = &components["records"][1];

        assert!(adjusted_second > baseline_second + 7.0);
        assert_eq!(
            second_record["trend_adjustment_multiplier"].as_f64(),
            Some(1.20)
        );
        assert!(
            second_record["adjusted_trend"].as_f64().unwrap()
                > second_record["trend"].as_f64().unwrap()
        );
    }

    #[test]
    fn piecewise_linear_residual_shock_passes_recent_signed_residuals_forward() {
        let rows = (1..=24)
            .map(|day| {
                let shock = if day >= 22 { 12.0 } else { 0.0 };
                ForecastRow::single(ts(day), 20.0 + 0.5 * f64::from(day) + shock)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut baseline =
            PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("baseline config");
        let mut shock_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                residual_shock_window: 3,
                residual_shock_scale: 0.8,
                residual_shock_decay: 0.5,
                ..base_config
            })
            .expect("shock config");

        baseline.fit(&frame).expect("baseline fit");
        shock_model.fit(&frame).expect("shock fit");
        let baseline_forecast = baseline.predict(2).expect("baseline predict");
        let shock_forecast = shock_model.predict(2).expect("shock predict");
        let components = shock_model
            .predict_components_json_value(2)
            .expect("components");

        assert!(shock_forecast.predictions()[0].mean > baseline_forecast.predictions()[0].mean);
        assert!(shock_forecast.predictions()[1].mean > baseline_forecast.predictions()[1].mean);
        assert!(
            components["records"][0]["residual_shock"].as_f64().unwrap()
                > components["records"][1]["residual_shock"].as_f64().unwrap()
        );
        assert_eq!(
            shock_model.metadata()["residual_shock_window"].as_u64(),
            Some(3)
        );
    }

    #[test]
    fn piecewise_linear_seasonal_prediction_intervals_render_columns() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| {
                    let noise = if day % 2 == 0 { 2.0 } else { -2.0 };
                    ForecastRow::single(ts(day), 25.0 + f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid interval config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let json = forecast.to_json_value();
        let records = json["records"].as_array().expect("records");

        assert_eq!(forecast.intervals().len(), 2);
        assert!(json["columns"]
            .as_array()
            .expect("columns")
            .iter()
            .any(|column| column.as_str() == Some("prediction_lower_p80")));
        assert!(records[0]["prediction_lower_p80"].as_f64().is_some());
        assert!(
            records[0]["prediction_lower_p80"].as_f64().unwrap()
                <= records[0]["prediction_upper_p80"].as_f64().unwrap()
        );
    }

    #[test]
    fn piecewise_linear_seasonal_custom_fourier_period_projects_cycle() {
        let frame = ForecastFrame::new(
            (1..=56)
                .map(|day| {
                    let biweekly = if day % 14 == 0 { 18.0 } else { 0.0 };
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day - 1)),
                        80.0 + 0.25 * f64::from(day) + biweekly,
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut custom = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            seasonality_l2_regularization: 0.001,
            custom_seasonalities: vec![PiecewiseLinearSeasonality {
                name: "biweekly_pickup_cycle".to_string(),
                period_days: 14.0,
                fourier_order: 4,
                mode: None,
                condition_name: None,
                l2_regularization: None,
            }],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid custom seasonality config");

        baseline.fit(&frame).expect("baseline fit");
        custom.fit(&frame).expect("custom fit");
        let baseline_forecast = baseline.predict(14).expect("baseline predict");
        let custom_forecast = custom.predict(14).expect("custom predict");

        assert!(
            custom_forecast.predictions()[13].mean > baseline_forecast.predictions()[13].mean + 8.0
        );
        assert_eq!(
            custom.metadata()["custom_seasonalities"][0]["name"].as_str(),
            Some("biweekly_pickup_cycle")
        );
    }

    #[test]
    fn piecewise_linear_builtin_seasonality_l2_can_target_weekly_terms() {
        let frame = ForecastFrame::new(
            (1..=56)
                .map(|day| {
                    let weekly = if day % 7 == 0 { 18.0 } else { 0.0 };
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day - 1)),
                        60.0 + 0.2 * f64::from(day) + weekly,
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 3,
            seasonality_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut low_l2 = PiecewiseLinearSeasonalForecaster::new(base_config.clone())
            .expect("valid low l2 config");
        let mut high_l2 = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            weekly_l2_regularization: Some(1_000.0),
            ..base_config
        })
        .expect("valid high weekly l2 config");

        low_l2.fit(&frame).expect("low l2 fit");
        high_l2.fit(&frame).expect("high l2 fit");
        let low_components = low_l2
            .predict_components_json_value(7)
            .expect("low components");
        let high_components = high_l2
            .predict_components_json_value(7)
            .expect("high components");
        let low_weekly = low_components["records"][6]["components"]["weekly"]
            .as_f64()
            .expect("low weekly contribution")
            .abs();
        let high_weekly = high_components["records"][6]["components"]["weekly"]
            .as_f64()
            .expect("high weekly contribution")
            .abs();

        assert!(low_weekly > high_weekly + 8.0);
        assert_eq!(
            high_l2.metadata()["weekly_l2_regularization"].as_f64(),
            Some(1_000.0)
        );
    }

    #[test]
    fn piecewise_linear_custom_seasonality_mode_can_be_multiplicative() {
        let frame = ForecastFrame::new(
            (1..=56)
                .map(|day| {
                    let trend = 40.0 + f64::from(day);
                    let multiplier = if day % 14 == 0 { 1.35 } else { 1.0 };
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day - 1)),
                        trend * multiplier,
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            seasonality_l2_regularization: 0.001,
            custom_seasonalities: vec![PiecewiseLinearSeasonality {
                name: "biweekly_pickup_multiplier".to_string(),
                period_days: 14.0,
                fourier_order: 4,
                mode: Some(PiecewiseLinearComponentMode::Multiplicative),
                condition_name: None,
                l2_regularization: None,
            }],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid custom seasonality config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(14).expect("predict");

        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
        assert_eq!(
            model.metadata()["custom_seasonalities"][0]["mode"].as_str(),
            Some("multiplicative")
        );
    }

    #[test]
    fn piecewise_linear_custom_seasonality_condition_gates_fourier_terms() {
        let start = ts(1);
        let rows = (1..=42)
            .map(|day| {
                let rush_hour = if day % 2 == 0 { 1.0 } else { 0.0 };
                let cycle = if day % 7 == 0 { 16.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    start + Duration::days(i64::from(day - 1)),
                    80.0 + 0.2 * f64::from(day) + rush_hour * cycle,
                    BTreeMap::from([("rush_hour".to_string(), rush_hour)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            seasonality_l2_regularization: 0.001,
            custom_seasonalities: vec![PiecewiseLinearSeasonality {
                name: "rush_hour_weekly".to_string(),
                period_days: 7.0,
                fourier_order: 3,
                mode: None,
                condition_name: Some("rush_hour".to_string()),
                l2_regularization: None,
            }],
            future_regressors: BTreeMap::from([(
                "rush_hour".to_string(),
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )]),
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut inactive = PiecewiseLinearSeasonalForecaster::new(config.clone())
            .expect("valid inactive conditional seasonality config");
        let mut active = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            future_regressors: BTreeMap::from([(
                "rush_hour".to_string(),
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )]),
            ..config
        })
        .expect("valid active conditional seasonality config");

        inactive.fit(&frame).expect("inactive fit");
        active.fit(&frame).expect("active fit");
        let inactive_forecast = inactive.predict(7).expect("inactive predict");
        let active_forecast = active.predict(7).expect("active predict");

        assert!(
            active_forecast.predictions()[6].mean > inactive_forecast.predictions()[6].mean + 4.0,
            "active conditional seasonality should lift matching future periods"
        );
        assert_eq!(
            active.metadata()["custom_seasonalities"][0]["condition_name"].as_str(),
            Some("rush_hour")
        );
    }

    #[test]
    fn piecewise_linear_seasonal_multiplicative_event_scales_with_trend() {
        let future_event_timestamp = NaiveDate::from_ymd_opt(2026, 2, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid event timestamp");
        let rows = (1..=30)
            .map(|day| {
                let trend = 20.0 + 2.0 * f64::from(day);
                let target = if (14..=16).contains(&day) {
                    trend * 1.5
                } else {
                    trend
                };
                ForecastRow::single(ts(day), target)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let event_config = vec![
            PiecewiseLinearEvent {
                name: "airport_surge".to_string(),
                timestamp: ts(15),
                lower_window: -1,
                upper_window: 1,
            },
            PiecewiseLinearEvent {
                name: "airport_surge".to_string(),
                timestamp: future_event_timestamp,
                lower_window: -1,
                upper_window: 1,
            },
        ];
        let mut additive = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            event_l2_regularization: 0.001,
            events: event_config.clone(),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid additive config");
        let mut multiplicative =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                component_mode: PiecewiseLinearComponentMode::Multiplicative,
                changepoints: 0,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                event_l2_regularization: 0.001,
                events: event_config,
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid multiplicative config");

        additive.fit(&frame).expect("additive fit");
        multiplicative.fit(&frame).expect("multiplicative fit");
        let additive_forecast = additive.predict(3).expect("additive predict");
        let multiplicative_forecast = multiplicative.predict(3).expect("multiplicative predict");

        assert!(
            multiplicative_forecast.predictions()[1].mean
                > additive_forecast.predictions()[1].mean + 5.0
        );
        assert_eq!(
            multiplicative.metadata()["component_mode"].as_str(),
            Some("multiplicative")
        );
    }

    #[test]
    fn optimized_theta_selects_from_grid_deterministically() {
        let frame = ForecastFrame::new(
            (1..=6)
                .map(|day| ForecastRow::single(ts(day), f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            OptimizedThetaForecaster::new(vec![1.0, 2.0], vec![0.2, 0.8]).expect("valid grid");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(model.selected_theta(), Some(1.0 | 2.0)));
        assert!(matches!(model.selected_alpha(), Some(0.2 | 0.8)));
        assert_eq!(model.validation_scores().len(), 4);
        assert_eq!(forecast.predictions().len(), 2);
    }

    #[test]
    fn auto_kalman_selects_variances_and_predicts_with_auto_name() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 20.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoKalmanForecaster::with_grids(
            vec![0.001, 0.01],
            vec![0.0001, 0.001],
            vec![0.1, 1.0],
            Some(2),
        )
        .expect("valid auto kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let metadata = model.metadata();

        assert!(model.selected_params().is_some());
        assert_eq!(model.validation_scores().len(), 8);
        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.predictions()[0].model, "auto_kalman");
        assert!(
            metadata["validation_scores"]
                .as_array()
                .expect("scores")
                .len()
                == 8
        );
    }

    #[test]
    fn auto_kalman_rejects_empty_grid_and_caps_short_validation_history() {
        assert!(
            AutoKalmanForecaster::with_grids(Vec::new(), vec![0.001], vec![1.0], Some(1),).is_err()
        );

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            AutoKalmanForecaster::with_grids(vec![0.001], vec![0.0001], vec![1.0], Some(1))
                .expect("valid grid");

        model
            .fit(&frame)
            .expect("fit with capped validation window");
        let forecast = model.predict(2).expect("forecast");

        assert_eq!(forecast.predictions().len(), 2);
        assert!(model
            .validation_scores()
            .iter()
            .all(|score| score.mse.is_finite()));
    }

    #[test]
    fn kalman_degrades_single_observation_to_zero_trend_level_fit() {
        let frame = ForecastFrame::new(
            vec![ForecastRow::single(ts(1), 12.5)],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = KalmanForecaster::new(0.01, 0.001, 0.1).expect("valid kalman");

        model.fit(&frame).expect("fit single observation");
        let forecast = model.predict(2).expect("forecast");

        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean == 12.5));
    }

    #[test]
    fn local_level_kalman_forecasts_flat_panel_levels() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PULocationID=1", ts(1), 10.0),
                ForecastRow::new("PULocationID=1", ts(2), 11.0),
                ForecastRow::new("PULocationID=2", ts(1), 30.0),
                ForecastRow::new("PULocationID=2", ts(2), 31.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = LocalLevelKalmanForecaster::new(0.01, 0.1).expect("valid kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(forecast.predictions().len(), 4);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.model == "local_level_kalman"));
    }

    #[test]
    fn auto_local_level_kalman_selects_variances() {
        let frame = ForecastFrame::new(
            (1..=6)
                .map(|day| ForecastRow::single(ts(day), 20.0 + f64::from(day % 2)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            AutoLocalLevelKalmanForecaster::with_grids(vec![0.001, 0.01], vec![0.1, 1.0], Some(2))
                .expect("valid auto kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(model.selected_params().is_some());
        assert_eq!(model.validation_scores().len(), 4);
        assert_eq!(forecast.predictions()[0].model, "auto_local_level_kalman");
    }

    #[test]
    fn auto_local_level_kalman_caps_short_validation_history() {
        let frame = ForecastFrame::new(
            vec![ForecastRow::single(ts(1), 10.0)],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoLocalLevelKalmanForecaster::with_grids(vec![0.001], vec![0.1], Some(3))
            .expect("valid auto kalman");

        model.fit(&frame).expect("fit with no holdout");
        let forecast = model.predict(2).expect("forecast");

        assert_eq!(forecast.predictions().len(), 2);
        assert!(model
            .validation_scores()
            .iter()
            .all(|score| score.mse.is_finite()));
    }

    #[test]
    fn ets_forecasts_panel_series_with_daily_timestamps() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PULocationID=1", ts(1), 10.0),
                ForecastRow::new("PULocationID=1", ts(2), 12.0),
                ForecastRow::new("PULocationID=1", ts(3), 14.0),
                ForecastRow::new("PULocationID=1", ts(4), 16.0),
                ForecastRow::new("PULocationID=2", ts(1), 30.0),
                ForecastRow::new("PULocationID=2", ts(2), 29.0),
                ForecastRow::new("PULocationID=2", ts(3), 28.0),
                ForecastRow::new("PULocationID=2", ts(4), 27.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::new(0.6, 0.2).expect("valid ets");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        let predictions = forecast.predictions();
        assert_eq!(predictions.len(), 4);
        assert_eq!(predictions[0].series_id, "PULocationID=1");
        assert_eq!(predictions[0].timestamp, ts(5));
        assert_eq!(predictions[1].horizon, 2);
        assert_eq!(predictions[2].series_id, "PULocationID=2");
        assert!(predictions[0].mean > 16.0);
        assert!(predictions[2].mean < 27.0);
        assert_eq!(
            model.fitted_values("PULocationID=1").expect("fitted").len(),
            4
        );
        assert_eq!(
            model.level_values("PULocationID=1").expect("levels").len(),
            4
        );
        assert_eq!(
            model.trend_values("PULocationID=1").expect("trends").len(),
            4
        );
        assert_eq!(
            model
                .seasonal_values("PULocationID=1")
                .expect("seasonals")
                .len(),
            4
        );
    }

    #[test]
    fn ets_additive_seasonality_repeats_pattern() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    let seasonal = if day % 2 == 0 { 4.0 } else { -4.0 };
                    ForecastRow::single(ts(day), 50.0 + seasonal)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::with_additive_seasonality(0.5, 0.0, Some(0.5), Some(2))
            .expect("valid seasonal ets");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| row.mean)
            .collect::<Vec<_>>();

        assert_eq!(forecast.predictions()[0].timestamp, ts(9));
        assert!(means[1] > means[0]);
        let seasonals = model
            .seasonal_values("__single__")
            .expect("seasonal contributions");
        assert_eq!(seasonals.len(), 8);
        assert!(seasonals[1] > seasonals[0]);
    }

    #[test]
    fn ets_rejects_invalid_params_and_degrades_short_seasonal_history() {
        assert!(ETSForecaster::new(0.0, 0.2).is_err());
        assert!(ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), None).is_err());

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 3.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), Some(2))
            .expect("valid seasonal ets");
        model
            .fit(&frame)
            .expect("short seasonal history fits non-seasonally");
        let forecast = model.predict(2).expect("forecast");
        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
    }

    #[test]
    fn ets_and_auto_ets_degrade_single_observation_to_level_fit() {
        let frame = ForecastFrame::new(
            vec![ForecastRow::single(ts(1), 42.0)],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut ets = ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), Some(7))
            .expect("valid seasonal ets");
        let mut auto_ets = AutoETSForecaster::with_grids(
            vec![0.3, 0.5],
            vec![0.0, 0.1],
            vec![Some(0.0), Some(0.2)],
            vec![0.9, 1.0],
            Some(7),
        )
        .expect("valid auto ets");

        ets.fit(&frame).expect("ets fit");
        auto_ets.fit(&frame).expect("auto ets fit");
        let ets_forecast = ets.predict(2).expect("ets forecast");
        let auto_forecast = auto_ets.predict(2).expect("auto ets forecast");

        assert_eq!(ets_forecast.predictions().len(), 2);
        assert_eq!(auto_forecast.predictions().len(), 2);
        assert!(ets_forecast
            .predictions()
            .iter()
            .chain(auto_forecast.predictions())
            .all(|prediction| prediction.mean == 42.0));
        assert!(auto_ets
            .validation_scores()
            .iter()
            .all(|score| score.mse.is_finite()));
    }

    #[test]
    fn arima_forecasts_differenced_linear_series() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day) * 3.0))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(0, 1, 0).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.timestamp, row.horizon, row.mean))
            .collect::<Vec<_>>();

        assert_eq!(means[0].0, ts(9));
        assert_eq!(means[2].1, 3);
        assert!((means[0].2 - 37.0).abs() < 1.0e-6);
        assert!((means[2].2 - 43.0).abs() < 1.0e-6);
        assert_eq!(model.residuals("__single__").expect("residuals").len(), 7);
    }

    #[test]
    fn arima_forecasts_each_panel_series_without_bleeding() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 13.0),
                ForecastRow::new("PU1->DO2", ts(3), 16.0),
                ForecastRow::new("PU1->DO2", ts(4), 19.0),
                ForecastRow::new("PU9->DO8", ts(1), 40.0),
                ForecastRow::new("PU9->DO8", ts(2), 38.0),
                ForecastRow::new("PU9->DO8", ts(3), 36.0),
                ForecastRow::new("PU9->DO8", ts(4), 34.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(0, 1, 0).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.series_id.as_str(), row.timestamp, row.mean))
            .collect::<Vec<_>>();

        assert_eq!(means.len(), 4);
        assert_eq!(means[0].0, "PU1->DO2");
        assert_eq!(means[0].1, ts(5));
        assert_eq!(means[2].0, "PU9->DO8");
        assert!(means[0].2 > 19.0);
        assert!(means[2].2 < 34.0);
    }

    #[test]
    fn arima_rejects_invalid_order_and_prunes_unsupported_terms() {
        assert!(ArimaForecaster::new(9, 0, 0).is_err());
        assert!(ArimaForecaster::new(1, 0, 9).is_err());

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(2, 0, 0).expect("valid arima");
        model.fit(&frame).expect("fit pruned order");
        let forecast = model.predict(2).expect("forecast");

        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
    }

    #[test]
    fn arima_supports_moving_average_terms() {
        let frame = ForecastFrame::new(
            (1..=10)
                .map(|day| {
                    let shock = if day % 3 == 0 { 2.0 } else { -1.0 };
                    ForecastRow::single(ts(day), 20.0 + f64::from(day) + shock)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(1, 0, 1).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(model.order(), (1, 0, 1));
        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));
    }

    #[test]
    fn arima_candidate_score_excludes_warmup_residuals() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 20.0 + f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let state = FittedArimaState::from_frame(&frame, 2, 0, 1).expect("fit arima");
        let series = state.series.get("__single__").expect("single series");

        let expected = series
            .residuals
            .iter()
            .skip(2)
            .map(|residual| residual * residual)
            .sum::<f64>()
            / 6.0;

        assert_eq!(series.score_start, 2);
        assert!((state.mean_squared_residual() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn auto_arima_rejects_unstable_ar_recursions() {
        let values = [
            36.893, 37.770, 38.912, 33.241, 32.972, 34.398, 35.728, 36.870, 37.747, 45.299, 45.490,
            41.305, 40.759, 39.888, 35.752, 34.428, 33.008, 31.587, 30.264, 29.127, 31.256, 30.710,
            30.526, 27.716, 35.268, 36.145, 37.288, 31.617, 33.044, 34.470, 35.800, 36.942, 37.819,
            45.371, 45.562, 41.377, 40.831, 39.960, 35.824, 34.500, 33.080, 31.659, 30.336, 29.199,
            31.328, 30.782, 30.598, 27.788, 35.340, 36.217, 37.360, 31.689, 32.812, 34.238, 35.568,
            36.710, 37.587, 45.139, 45.330, 41.145, 40.599, 39.728, 35.592, 34.268, 32.848, 31.427,
            30.104, 28.968, 31.097, 30.550, 30.366, 27.556, 35.109, 35.986, 37.128, 31.457, 34.999,
            36.425, 37.755, 38.897, 39.774, 47.326, 47.517, 43.332, 42.786, 41.915, 37.779, 36.455,
            35.035, 33.614, 32.291, 31.155, 33.284, 32.737, 32.553, 29.743,
        ];
        let start = ts(1);
        let frame = ForecastFrame::new(
            values
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    ForecastRow::single(start + Duration::hours(idx as i64), *value)
                })
                .collect(),
            ForecastFrequency::Hourly,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(2, 1, 1).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(14).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|prediction| prediction.mean)
            .collect::<Vec<_>>();

        assert_eq!(model.selected_order(), Some((1, 0, 1)));
        assert!(model.validation_scores().iter().any(|score| !score.stable));
        assert!(means.iter().all(|mean| mean.is_finite() && *mean < 60.0));
        assert!(means.iter().all(|mean| *mean > 0.0));
    }

    #[test]
    fn auto_arima_selects_candidate_and_predicts_with_model_name() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(2, 1, 1).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(
            model.selected_order(),
            Some((0..=2, 0..=1, 0..=1))
        ));
        assert_eq!(model.validation_scores().len(), 12);
        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.predictions()[0].model, "auto_arima");
        assert_eq!(forecast.predictions()[0].timestamp, ts(9));
    }

    #[test]
    fn auto_arima_deduplicates_orders_after_short_history_pruning() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(3, 1, 2).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(
            model.selected_order(),
            Some((0..=1, 0..=1, 0..=1))
        ));
        assert!(model.validation_scores().len() <= 8);
        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.model == "auto_arima" && prediction.mean.is_finite()));
    }

    #[test]
    fn local_seasonal_and_window_models_use_available_short_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 20.0),
                ForecastRow::new("PU1->DO2", ts(3), 30.0),
                ForecastRow::new("PU9->DO8", ts(1), 4.0),
                ForecastRow::new("PU9->DO8", ts(2), 6.0),
                ForecastRow::new("PU9->DO8", ts(3), 8.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");

        let mut seasonal_naive = SeasonalNaiveForecaster::new(24).expect("seasonal naive");
        seasonal_naive.fit(&frame).expect("fit seasonal naive");
        let seasonal = seasonal_naive.predict(4).expect("predict seasonal naive");
        let pu1 = seasonal
            .predictions()
            .iter()
            .filter(|row| row.series_id == "PU1->DO2")
            .map(|row| row.mean)
            .collect::<Vec<_>>();
        assert_eq!(pu1, vec![10.0, 20.0, 30.0, 10.0]);

        let mut window = WindowAverageForecaster::new(24).expect("window average");
        window.fit(&frame).expect("fit window average");
        let averaged = window.predict(2).expect("predict window average");
        assert_eq!(averaged.predictions().len(), 4);
        assert!(averaged
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));
        assert_eq!(averaged.predictions()[0].mean, 20.0);

        let mut seasonal_window =
            SeasonalWindowAverageForecaster::new(24, 3).expect("seasonal window average");
        seasonal_window.fit(&frame).expect("fit seasonal window");
        let seasonal_averaged = seasonal_window
            .predict(4)
            .expect("predict seasonal window average");
        let pu9 = seasonal_averaged
            .predictions()
            .iter()
            .filter(|row| row.series_id == "PU9->DO8")
            .map(|row| row.mean)
            .collect::<Vec<_>>();
        assert_eq!(pu9, vec![4.0, 6.0, 8.0, 4.0]);
    }

    #[test]
    fn theta_degrades_unsupported_seasonality_to_nonseasonal_fit() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 11.0),
                ForecastRow::new("PU1->DO2", ts(3), 12.0),
                ForecastRow::new("PU9->DO8", ts(1), 30.0),
                ForecastRow::new("PU9->DO8", ts(2), 29.0),
                ForecastRow::new("PU9->DO8", ts(3), 28.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::additive(24).expect("seasonality");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.3, Some(seasonality)).expect("theta");

        model
            .fit(&frame)
            .expect("fit theta without supported seasonality");
        let forecast = model.predict(2).expect("forecast");
        assert_eq!(forecast.predictions().len(), 4);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
    }
}
