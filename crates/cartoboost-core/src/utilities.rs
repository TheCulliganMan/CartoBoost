use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct LocalLinearKalmanConfig {
    pub level_process_variance: f64,
    pub trend_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLinearKalmanState {
    pub level: f64,
    pub trend: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLinearKalmanEstimate {
    pub step: usize,
    pub observed: f64,
    pub prior_level: f64,
    pub prior_trend: f64,
    pub prior_level_variance: f64,
    pub prior_trend_variance: f64,
    pub prior_covariance: [[f64; 2]; 2],
    pub level: f64,
    pub trend: f64,
    pub level_variance: f64,
    pub trend_variance: f64,
    pub covariance: [[f64; 2]; 2],
    pub innovation: f64,
    pub innovation_variance: f64,
    pub level_gain: f64,
    pub trend_gain: f64,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLinearKalmanSmoothedState {
    pub step: usize,
    pub level: f64,
    pub trend: f64,
    pub covariance: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalLinearKalmanResult {
    pub final_state: LocalLinearKalmanState,
    pub final_covariance: [[f64; 2]; 2],
    pub estimates: Vec<LocalLinearKalmanEstimate>,
    pub smoothed_states: Vec<LocalLinearKalmanSmoothedState>,
    pub residual_summary: KalmanResidualSummary,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LocalLevelKalmanConfig {
    pub level_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelKalmanEstimate {
    pub step: usize,
    pub observed: f64,
    pub prior_level: f64,
    pub prior_variance: f64,
    pub level: f64,
    pub variance: f64,
    pub innovation: f64,
    pub innovation_variance: f64,
    pub gain: f64,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelKalmanSmoothedState {
    pub step: usize,
    pub level: f64,
    pub variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalLevelKalmanResult {
    pub final_level: f64,
    pub final_variance: f64,
    pub estimates: Vec<LocalLevelKalmanEstimate>,
    pub smoothed_states: Vec<LocalLevelKalmanSmoothedState>,
    pub residual_summary: KalmanResidualSummary,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanResidualSummary {
    pub observation_count: usize,
    pub fitted_count: usize,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mean_standardized_innovation: f64,
    pub max_abs_standardized_innovation: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanForecastPoint {
    pub step: usize,
    pub mean: f64,
    pub variance: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntermittentDemandMethod {
    Croston,
    Sba,
    Tsb,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrdinaryKrigingConfig {
    pub range: f64,
    pub nugget: f64,
    pub sill: f64,
    pub variogram_model: KrigingVariogramModel,
    pub drift: KrigingDrift,
    pub anisotropy_angle_degrees: f64,
    pub anisotropy_scaling: f64,
    pub max_neighbors: Option<usize>,
    pub min_neighbors: usize,
    pub max_distance: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KrigingObservation {
    pub x: f64,
    pub y: f64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KrigingPrediction {
    pub x: f64,
    pub y: f64,
    pub mean: f64,
    pub variance: f64,
    pub weights: Vec<f64>,
    pub neighbor_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalVariogramBin {
    pub lag_min: f64,
    pub lag_max: f64,
    pub lag_center: f64,
    pub mean_distance: f64,
    pub semivariance: f64,
    pub pair_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KrigingVariogramFit {
    pub config: OrdinaryKrigingConfig,
    pub bins: Vec<EmpiricalVariogramBin>,
    pub weighted_sse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KrigingLooDiagnostics {
    pub observation_count: usize,
    pub mean_error: f64,
    pub mae: f64,
    pub rmse: f64,
    pub mean_standardized_error: f64,
    pub rmse_standardized_error: f64,
    pub max_abs_standardized_error: f64,
    pub interval_coverage_95: f64,
    pub average_variance: f64,
}

#[derive(Debug, Clone)]
pub struct OrdinaryKrigingSystem {
    observations: Vec<KrigingObservation>,
    config: OrdinaryKrigingConfig,
    factorization: LinearSystemFactorization,
    drift_terms: usize,
}

#[derive(Debug, Clone)]
struct LinearSystemFactorization {
    lu: Vec<Vec<f64>>,
    permutation: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KrigingVariogramModel {
    Exponential,
    Gaussian,
    Spherical,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KrigingDrift {
    Ordinary,
    Linear,
}

impl LocalLinearKalmanConfig {
    pub fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        Self {
            level_process_variance,
            trend_process_variance,
            observation_variance,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self> {
        validate_positive_finite(self.level_process_variance, "level_process_variance")?;
        validate_positive_finite(self.trend_process_variance, "trend_process_variance")?;
        validate_positive_finite(self.observation_variance, "observation_variance")?;
        Ok(self)
    }
}

impl LocalLevelKalmanConfig {
    pub fn new(level_process_variance: f64, observation_variance: f64) -> Result<Self> {
        Self {
            level_process_variance,
            observation_variance,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self> {
        validate_positive_finite(self.level_process_variance, "level_process_variance")?;
        validate_positive_finite(self.observation_variance, "observation_variance")?;
        Ok(self)
    }
}

impl OrdinaryKrigingConfig {
    pub fn new(range: f64, nugget: f64) -> Result<Self> {
        Self {
            range,
            nugget,
            sill: 1.0,
            variogram_model: KrigingVariogramModel::Exponential,
            drift: KrigingDrift::Ordinary,
            anisotropy_angle_degrees: 0.0,
            anisotropy_scaling: 1.0,
            max_neighbors: None,
            min_neighbors: 1,
            max_distance: None,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self> {
        validate_positive_finite(self.range, "range")?;
        if !self.nugget.is_finite() || self.nugget < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        validate_positive_finite(self.sill, "sill")?;
        if !self.anisotropy_angle_degrees.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "anisotropy_angle_degrees must be finite".to_string(),
            ));
        }
        validate_positive_finite(self.anisotropy_scaling, "anisotropy_scaling")?;
        if self.min_neighbors == 0 {
            return Err(CartoBoostError::InvalidInput(
                "min_neighbors must be positive".to_string(),
            ));
        }
        if let Some(max_neighbors) = self.max_neighbors {
            if max_neighbors == 0 {
                return Err(CartoBoostError::InvalidInput(
                    "max_neighbors must be positive when provided".to_string(),
                ));
            }
            if self.min_neighbors > max_neighbors {
                return Err(CartoBoostError::InvalidInput(
                    "min_neighbors must be <= max_neighbors".to_string(),
                ));
            }
            let drift_terms = drift_term_count(self.drift);
            if max_neighbors < drift_terms {
                return Err(CartoBoostError::InvalidInput(format!(
                    "max_neighbors must be at least {drift_terms} for {:?} kriging drift",
                    self.drift
                )));
            }
        }
        if let Some(max_distance) = self.max_distance {
            validate_positive_finite(max_distance, "max_distance")?;
        }
        Ok(self)
    }

    pub fn with_sill(mut self, sill: f64) -> Result<Self> {
        self.sill = sill;
        self.validate()
    }

    pub fn with_variogram_model(mut self, variogram_model: KrigingVariogramModel) -> Self {
        self.variogram_model = variogram_model;
        self
    }

    pub fn with_drift(mut self, drift: KrigingDrift) -> Self {
        self.drift = drift;
        self
    }

    pub fn with_anisotropy(mut self, angle_degrees: f64, scaling: f64) -> Result<Self> {
        if !angle_degrees.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "anisotropy_angle_degrees must be finite".to_string(),
            ));
        }
        validate_positive_finite(scaling, "anisotropy_scaling")?;
        self.anisotropy_angle_degrees = angle_degrees;
        self.anisotropy_scaling = scaling;
        self.validate()
    }

    pub fn with_neighbor_limits(
        mut self,
        max_neighbors: Option<usize>,
        min_neighbors: usize,
        max_distance: Option<f64>,
    ) -> Result<Self> {
        self.max_neighbors = max_neighbors;
        self.min_neighbors = min_neighbors;
        self.max_distance = max_distance;
        self.validate()
    }
}

impl OrdinaryKrigingSystem {
    pub fn new(observations: &[KrigingObservation], config: OrdinaryKrigingConfig) -> Result<Self> {
        let config = config.validate()?;
        validate_kriging_observations(observations)?;
        if uses_local_neighbors(config) {
            return Err(CartoBoostError::InvalidInput(
                "OrdinaryKrigingSystem requires all-neighbor config; use ordinary_kriging_predict_many for max_neighbors or max_distance".to_string(),
            ));
        }
        if observations.len() < config.min_neighbors {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging found {} neighbors, but min_neighbors is {}",
                observations.len(),
                config.min_neighbors
            )));
        }
        let drift_terms = drift_term_count(config.drift);
        if observations.len() < drift_terms {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging drift {:?} requires at least {drift_terms} observations; got {}",
                config.drift,
                observations.len()
            )));
        }
        let matrix = build_kriging_system_matrix(observations, config);
        let factorization = LinearSystemFactorization::factor(matrix).ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "kriging system is singular or numerically ill-conditioned; adjust coordinates, variogram scale, or nugget".to_string(),
            )
        })?;
        Ok(Self {
            observations: observations.to_vec(),
            config,
            factorization,
            drift_terms,
        })
    }

    pub fn predict(&self, target: (f64, f64)) -> Result<KrigingPrediction> {
        validate_kriging_target(target)?;
        let rhs = build_kriging_rhs(&self.observations, target, self.config);
        let solution = self.factorization.solve(&rhs).ok_or_else(|| {
            CartoBoostError::InvalidInput("kriging solve produced a non-finite result".to_string())
        })?;
        kriging_prediction_from_solution(
            &self.observations,
            target,
            self.config,
            &rhs,
            &solution,
            (0..self.observations.len()).collect(),
        )
    }

    pub fn predict_many(&self, targets: &[(f64, f64)]) -> Result<Vec<KrigingPrediction>> {
        if targets.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "kriging targets must not be empty".to_string(),
            ));
        }
        targets
            .par_iter()
            .map(|target| self.predict(*target))
            .collect()
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn drift_terms(&self) -> usize {
        self.drift_terms
    }
}

pub fn fit_local_level_kalman(
    values: &[f64],
    config: LocalLevelKalmanConfig,
) -> Result<LocalLevelKalmanResult> {
    let config = config.validate()?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman filter requires at least one observation".to_string(),
        ));
    }
    validate_numeric_series(values, "kalman observation")?;
    let mut level = values[0];
    let mut variance = config.observation_variance;
    let mut estimates = Vec::with_capacity(values.len().saturating_sub(1));
    let mut filtered_states = Vec::with_capacity(values.len());
    filtered_states.push(LocalLevelKalmanSmoothedState {
        step: 0,
        level,
        variance,
    });
    let mut total_log_likelihood = 0.0;
    for (idx, observed) in values.iter().enumerate().skip(1) {
        let prior_level = level;
        let prior_variance = variance + config.level_process_variance;
        let innovation = observed - prior_level;
        let innovation_variance = prior_variance + config.observation_variance;
        if innovation_variance <= 0.0 || !innovation_variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "local level kalman innovation variance is not positive".to_string(),
            ));
        }
        let gain = prior_variance / innovation_variance;
        let log_likelihood = gaussian_log_likelihood(innovation, innovation_variance);
        level = prior_level + gain * innovation;
        variance =
            (1.0 - gain).powi(2) * prior_variance + gain.powi(2) * config.observation_variance;
        total_log_likelihood += log_likelihood;
        estimates.push(LocalLevelKalmanEstimate {
            step: idx,
            observed: *observed,
            prior_level,
            prior_variance,
            level,
            variance,
            innovation,
            innovation_variance,
            gain,
            log_likelihood,
        });
        filtered_states.push(LocalLevelKalmanSmoothedState {
            step: idx,
            level,
            variance,
        });
    }
    let smoothed_states = smooth_local_level_states(&filtered_states, &estimates)?;
    let residual_summary =
        kalman_residual_summary(values.len(), &estimates, total_log_likelihood, 2);
    Ok(LocalLevelKalmanResult {
        final_level: level,
        final_variance: variance,
        estimates,
        smoothed_states,
        residual_summary,
        log_likelihood: total_log_likelihood,
    })
}

pub fn local_level_kalman_forecast(final_level: f64, horizon: usize) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !final_level.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman final level must be finite".to_string(),
        ));
    }
    Ok(vec![final_level; horizon])
}

pub fn fit_local_linear_kalman(
    values: &[f64],
    config: LocalLinearKalmanConfig,
) -> Result<LocalLinearKalmanResult> {
    let config = config.validate()?;
    if values.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "local linear kalman filter requires at least two observations".to_string(),
        ));
    }
    validate_numeric_series(values, "kalman observation")?;
    let mut level = values[0];
    // The first observation conditions the initial level. The trend remains an
    // unobserved zero-mean state until later observations update it; deriving
    // it from values[1] and then filtering values[1] double-counts that value.
    let mut trend = 0.0;
    let mut p00 = config.observation_variance;
    let mut p01 = 0.0;
    let mut p10 = 0.0;
    let mut p11 = config.observation_variance;
    let mut estimates = Vec::with_capacity(values.len() - 1);
    let mut filtered_states = Vec::with_capacity(values.len());
    filtered_states.push(LocalLinearKalmanSmoothedState {
        step: 0,
        level,
        trend,
        covariance: [[p00, p01], [p10, p11]],
    });
    let mut total_log_likelihood = 0.0;
    for (idx, observed) in values.iter().enumerate().skip(1) {
        let prior_level = level + trend;
        let prior_trend = trend;
        let pp00 = p00 + p01 + p10 + p11 + config.level_process_variance;
        let pp01 = p01 + p11;
        let pp10 = p10 + p11;
        let pp11 = p11 + config.trend_process_variance;

        let innovation = observed - prior_level;
        let innovation_variance = pp00 + config.observation_variance;
        if innovation_variance <= 0.0 || !innovation_variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "kalman innovation variance is not positive".to_string(),
            ));
        }
        let k0 = pp00 / innovation_variance;
        let k1 = pp10 / innovation_variance;
        let log_likelihood = gaussian_log_likelihood(innovation, innovation_variance);
        level = prior_level + k0 * innovation;
        trend = prior_trend + k1 * innovation;
        let posterior_covariance = kalman_joseph_covariance_update(
            [[pp00, pp01], [pp10, pp11]],
            [k0, k1],
            config.observation_variance,
        )?;
        p00 = posterior_covariance[0][0];
        p01 = posterior_covariance[0][1];
        p10 = posterior_covariance[1][0];
        p11 = posterior_covariance[1][1];
        total_log_likelihood += log_likelihood;
        estimates.push(LocalLinearKalmanEstimate {
            step: idx,
            observed: *observed,
            prior_level,
            prior_trend,
            prior_level_variance: pp00,
            prior_trend_variance: pp11,
            prior_covariance: [[pp00, pp01], [pp10, pp11]],
            level,
            trend,
            level_variance: p00,
            trend_variance: p11,
            covariance: [[p00, p01], [p10, p11]],
            innovation,
            innovation_variance,
            level_gain: k0,
            trend_gain: k1,
            log_likelihood,
        });
        filtered_states.push(LocalLinearKalmanSmoothedState {
            step: idx,
            level,
            trend,
            covariance: [[p00, p01], [p10, p11]],
        });
    }
    let smoothed_states = smooth_local_linear_states(&filtered_states, &estimates)?;
    let residual_summary =
        kalman_residual_summary(values.len(), &estimates, total_log_likelihood, 3);
    Ok(LocalLinearKalmanResult {
        final_state: LocalLinearKalmanState { level, trend },
        final_covariance: [[p00, p01], [p10, p11]],
        estimates,
        smoothed_states,
        residual_summary,
        log_likelihood: total_log_likelihood,
    })
}

pub fn intermittent_demand_forecast(
    values: &[f64],
    horizon: usize,
    alpha: f64,
    beta: f64,
    method: IntermittentDemandMethod,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand forecast horizon must be positive".to_string(),
        ));
    }
    validate_unit_interval("alpha", alpha)?;
    validate_unit_interval("beta", beta)?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand forecast requires at least one observation".to_string(),
        ));
    }
    validate_numeric_series(values, "intermittent demand observation")?;
    for (idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent demand observation at index {idx} must be non-negative"
            )));
        }
    }
    match method {
        IntermittentDemandMethod::Croston | IntermittentDemandMethod::Sba => {
            let estimate = croston_level(values, alpha)?;
            let adjusted = if method == IntermittentDemandMethod::Sba {
                estimate * (1.0 - alpha / 2.0)
            } else {
                estimate
            };
            Ok(vec![adjusted; horizon])
        }
        IntermittentDemandMethod::Tsb => {
            let estimate = tsb_level(values, alpha, beta)?;
            Ok(vec![estimate; horizon])
        }
    }
}

pub fn local_linear_kalman_forecast(
    state: LocalLinearKalmanState,
    horizon: usize,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !state.level.is_finite() || !state.trend.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast state must be finite".to_string(),
        ));
    }
    Ok((1..=horizon)
        .map(|step| state.level + step as f64 * state.trend)
        .collect())
}

pub fn local_level_kalman_forecast_distribution(
    final_level: f64,
    final_variance: f64,
    config: LocalLevelKalmanConfig,
    horizon: usize,
    interval_z: f64,
) -> Result<Vec<KalmanForecastPoint>> {
    let config = config.validate()?;
    validate_kalman_forecast_inputs(horizon, interval_z)?;
    if !final_level.is_finite() || !final_variance.is_finite() || final_variance < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman final state must be finite with non-negative variance".to_string(),
        ));
    }
    Ok((1..=horizon)
        .map(|step| {
            let variance = final_variance
                + step as f64 * config.level_process_variance
                + config.observation_variance;
            forecast_point(step, final_level, variance, interval_z)
        })
        .collect())
}

pub fn local_linear_kalman_forecast_distribution(
    state: LocalLinearKalmanState,
    covariance: [[f64; 2]; 2],
    config: LocalLinearKalmanConfig,
    horizon: usize,
    interval_z: f64,
) -> Result<Vec<KalmanForecastPoint>> {
    let config = config.validate()?;
    validate_kalman_forecast_inputs(horizon, interval_z)?;
    if !state.level.is_finite() || !state.trend.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kalman final state must be finite".to_string(),
        ));
    }
    let covariance = validate_covariance_2x2(covariance, "kalman final covariance")?;
    (1..=horizon)
        .map(|step| {
            let h = step as f64;
            let mean = state.level + h * state.trend;
            let trend_noise_multiplier = if step <= 1 {
                0.0
            } else {
                (h - 1.0) * h * (2.0 * h - 1.0) / 6.0
            };
            let state_variance = covariance[0][0]
                + h * (covariance[0][1] + covariance[1][0])
                + h * h * covariance[1][1]
                + h * config.level_process_variance
                + trend_noise_multiplier * config.trend_process_variance;
            let state_variance = checked_non_negative_variance(
                state_variance,
                covariance_scale(covariance)
                    + h * config.level_process_variance
                    + trend_noise_multiplier * config.trend_process_variance,
                "kalman forecast state variance",
            )?;
            let variance = state_variance + config.observation_variance;
            Ok(forecast_point(step, mean, variance, interval_z))
        })
        .collect()
}

fn gaussian_log_likelihood(innovation: f64, innovation_variance: f64) -> f64 {
    -0.5 * ((2.0 * std::f64::consts::PI).ln()
        + innovation_variance.ln()
        + innovation * innovation / innovation_variance)
}

fn kalman_joseph_covariance_update(
    prior: [[f64; 2]; 2],
    gain: [f64; 2],
    observation_variance: f64,
) -> Result<[[f64; 2]; 2]> {
    let update = [[1.0 - gain[0], 0.0], [-gain[1], 1.0]];
    let measurement = [
        [
            gain[0] * observation_variance * gain[0],
            gain[0] * observation_variance * gain[1],
        ],
        [
            gain[1] * observation_variance * gain[0],
            gain[1] * observation_variance * gain[1],
        ],
    ];
    let posterior = mat2_add(
        mat2_mul(mat2_mul(update, prior), mat2_transpose(update)),
        measurement,
    );
    validate_covariance_2x2(posterior, "kalman posterior covariance")
}

fn covariance_scale(matrix: [[f64; 2]; 2]) -> f64 {
    matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
}

fn checked_non_negative_variance(value: f64, scale: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be finite"
        )));
    }
    let tolerance = 128.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    if value < -tolerance {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} is negative ({value:e}); covariance inputs are invalid or numerically unstable"
        )));
    }
    Ok(value.max(0.0))
}

fn validate_covariance_2x2(matrix: [[f64; 2]; 2], label: &str) -> Result<[[f64; 2]; 2]> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be finite"
        )));
    }
    let scale = covariance_scale(matrix);
    let tolerance = 128.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    if (matrix[0][1] - matrix[1][0]).abs() > tolerance {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be symmetric"
        )));
    }
    let p00 = checked_non_negative_variance(matrix[0][0], scale, label)?;
    let p11 = checked_non_negative_variance(matrix[1][1], scale, label)?;
    let mut off_diagonal = 0.5 * (matrix[0][1] + matrix[1][0]);
    let determinant = p00 * p11 - off_diagonal * off_diagonal;
    let determinant_tolerance = 256.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE).powi(2);
    if determinant < -determinant_tolerance {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be positive semidefinite"
        )));
    }
    if determinant < 0.0 {
        let bound = (p00 * p11).sqrt();
        off_diagonal = off_diagonal.clamp(-bound, bound);
    }
    Ok([[p00, off_diagonal], [off_diagonal, p11]])
}

fn validate_kalman_forecast_inputs(horizon: usize, interval_z: f64) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !interval_z.is_finite() || interval_z < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast interval_z must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn forecast_point(step: usize, mean: f64, variance: f64, interval_z: f64) -> KalmanForecastPoint {
    debug_assert!(variance >= 0.0 && variance.is_finite());
    let standard_error = variance.sqrt();
    KalmanForecastPoint {
        step,
        mean,
        variance,
        lower: mean - interval_z * standard_error,
        upper: mean + interval_z * standard_error,
    }
}

trait KalmanInnovation {
    fn innovation(&self) -> f64;
    fn innovation_variance(&self) -> f64;
}

impl KalmanInnovation for LocalLevelKalmanEstimate {
    fn innovation(&self) -> f64 {
        self.innovation
    }

    fn innovation_variance(&self) -> f64 {
        self.innovation_variance
    }
}

impl KalmanInnovation for LocalLinearKalmanEstimate {
    fn innovation(&self) -> f64 {
        self.innovation
    }

    fn innovation_variance(&self) -> f64 {
        self.innovation_variance
    }
}

fn kalman_residual_summary<T: KalmanInnovation>(
    observation_count: usize,
    estimates: &[T],
    log_likelihood: f64,
    parameter_count: usize,
) -> KalmanResidualSummary {
    let fitted_count = estimates.len();
    if fitted_count == 0 {
        return KalmanResidualSummary {
            observation_count,
            fitted_count,
            log_likelihood,
            aic: 2.0 * parameter_count as f64 - 2.0 * log_likelihood,
            bic: f64::NAN,
            mse: f64::NAN,
            rmse: f64::NAN,
            mae: f64::NAN,
            mean_standardized_innovation: f64::NAN,
            max_abs_standardized_innovation: f64::NAN,
        };
    }
    let mut sse = 0.0;
    let mut sae = 0.0;
    let mut standardized_sum = 0.0;
    let mut standardized_abs_max = 0.0_f64;
    for estimate in estimates {
        let innovation = estimate.innovation();
        let standardized = innovation / estimate.innovation_variance().sqrt();
        sse += innovation * innovation;
        sae += innovation.abs();
        standardized_sum += standardized;
        standardized_abs_max = standardized_abs_max.max(standardized.abs());
    }
    let n = fitted_count as f64;
    let mse = sse / n;
    KalmanResidualSummary {
        observation_count,
        fitted_count,
        log_likelihood,
        aic: 2.0 * parameter_count as f64 - 2.0 * log_likelihood,
        bic: (parameter_count as f64) * n.ln() - 2.0 * log_likelihood,
        mse,
        rmse: mse.sqrt(),
        mae: sae / n,
        mean_standardized_innovation: standardized_sum / n,
        max_abs_standardized_innovation: standardized_abs_max,
    }
}

fn smooth_local_level_states(
    filtered_states: &[LocalLevelKalmanSmoothedState],
    estimates: &[LocalLevelKalmanEstimate],
) -> Result<Vec<LocalLevelKalmanSmoothedState>> {
    if filtered_states.is_empty() {
        return Ok(Vec::new());
    }
    let mut smoothed = filtered_states.to_vec();
    for idx in (0..filtered_states.len().saturating_sub(1)).rev() {
        let next_estimate = &estimates[idx];
        let filtered = filtered_states[idx];
        let next_smoothed = smoothed[idx + 1];
        if !next_estimate.prior_variance.is_finite() || next_estimate.prior_variance <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "local level kalman smoother prior variance must be positive and finite"
                    .to_string(),
            ));
        }
        let smoother_gain = filtered.variance / next_estimate.prior_variance;
        let level =
            filtered.level + smoother_gain * (next_smoothed.level - next_estimate.prior_level);
        let variance = filtered.variance
            + smoother_gain
                * smoother_gain
                * (next_smoothed.variance - next_estimate.prior_variance);
        let variance = checked_non_negative_variance(
            variance,
            filtered
                .variance
                .abs()
                .max(next_estimate.prior_variance.abs())
                .max(next_smoothed.variance.abs()),
            "local level kalman smoothed variance",
        )?;
        smoothed[idx] = LocalLevelKalmanSmoothedState {
            step: filtered.step,
            level,
            variance,
        };
    }
    Ok(smoothed)
}

fn smooth_local_linear_states(
    filtered_states: &[LocalLinearKalmanSmoothedState],
    estimates: &[LocalLinearKalmanEstimate],
) -> Result<Vec<LocalLinearKalmanSmoothedState>> {
    if filtered_states.is_empty() {
        return Ok(Vec::new());
    }
    let mut smoothed = filtered_states.to_vec();
    for idx in (0..filtered_states.len().saturating_sub(1)).rev() {
        let filtered = filtered_states[idx];
        let next_estimate = &estimates[idx];
        let predicted_inverse = invert_2x2(next_estimate.prior_covariance).ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "kalman smoother prior covariance is singular or numerically ill-conditioned"
                    .to_string(),
            )
        })?;
        let gain = mat2_mul(
            mat2_mul(filtered.covariance, [[1.0, 0.0], [1.0, 1.0]]),
            predicted_inverse,
        );
        let predicted_state = [next_estimate.prior_level, next_estimate.prior_trend];
        let next_smoothed = smoothed[idx + 1];
        let state_delta = [
            next_smoothed.level - predicted_state[0],
            next_smoothed.trend - predicted_state[1],
        ];
        let correction = mat2_vec_mul(gain, state_delta);
        let covariance_delta = mat2_sub(next_smoothed.covariance, next_estimate.prior_covariance);
        let covariance = mat2_add(
            filtered.covariance,
            mat2_mul(mat2_mul(gain, covariance_delta), mat2_transpose(gain)),
        );
        smoothed[idx] = LocalLinearKalmanSmoothedState {
            step: filtered.step,
            level: filtered.level + correction[0],
            trend: filtered.trend + correction[1],
            covariance: validate_covariance_2x2(covariance, "kalman smoothed covariance")?,
        };
    }
    Ok(smoothed)
}

fn invert_2x2(matrix: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    let scale = covariance_scale(matrix);
    if scale == 0.0 {
        return None;
    }
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if !determinant.is_finite() || determinant.abs() <= 128.0 * f64::EPSILON * scale.powi(2) {
        return None;
    }
    Some([
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ])
}

fn mat2_mul(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn mat2_vec_mul(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn mat2_add(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [left[0][0] + right[0][0], left[0][1] + right[0][1]],
        [left[1][0] + right[1][0], left[1][1] + right[1][1]],
    ]
}

fn mat2_sub(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [left[0][0] - right[0][0], left[0][1] - right[0][1]],
        [left[1][0] - right[1][0], left[1][1] - right[1][1]],
    ]
}

fn mat2_transpose(matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

pub fn ordinary_kriging_predict_many(
    observations: &[KrigingObservation],
    targets: &[(f64, f64)],
    config: OrdinaryKrigingConfig,
) -> Result<Vec<KrigingPrediction>> {
    let config = config.validate()?;
    validate_kriging_observations(observations)?;
    if targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging targets must not be empty".to_string(),
        ));
    }
    if !uses_local_neighbors(config) {
        return OrdinaryKrigingSystem::new(observations, config)?.predict_many(targets);
    }
    targets
        .par_iter()
        .map(|target| ordinary_kriging_predict_unchecked(observations, *target, config))
        .collect()
}

pub fn ordinary_kriging_predict(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<KrigingPrediction> {
    let config = config.validate()?;
    validate_kriging_observations(observations)?;
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    if !uses_local_neighbors(config) {
        return OrdinaryKrigingSystem::new(observations, config)?.predict(target);
    }
    ordinary_kriging_predict_unchecked(observations, target, config)
}

fn ordinary_kriging_predict_unchecked(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<KrigingPrediction> {
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    let selected = select_kriging_neighbors(observations, target, config)?;
    let n = selected.len();
    let drift_terms = drift_term_count(config.drift);
    if n < drift_terms {
        return Err(CartoBoostError::InvalidInput(format!(
            "kriging drift {:?} requires at least {drift_terms} neighbors; got {n}",
            config.drift
        )));
    }
    let selected_observations = selected
        .iter()
        .map(|(_, observation)| **observation)
        .collect::<Vec<_>>();
    let matrix = build_kriging_system_matrix(&selected_observations, config);
    let rhs = build_kriging_rhs(&selected_observations, target, config);
    let factorization = LinearSystemFactorization::factor(matrix).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "kriging system is singular or numerically ill-conditioned; adjust coordinates, variogram scale, or nugget".to_string(),
        )
    })?;
    let solution = factorization.solve(&rhs).ok_or_else(|| {
        CartoBoostError::InvalidInput("kriging solve produced a non-finite result".to_string())
    })?;
    kriging_prediction_from_solution(
        &selected_observations,
        target,
        config,
        &rhs,
        &solution,
        selected.into_iter().map(|(idx, _)| idx).collect(),
    )
}

pub fn ordinary_kriging_leave_one_out(
    observations: &[KrigingObservation],
    config: OrdinaryKrigingConfig,
) -> Result<Vec<KrigingPrediction>> {
    let config = config.validate()?;
    validate_kriging_observations(observations)?;
    if observations.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "kriging leave-one-out requires at least two observations".to_string(),
        ));
    }
    observations
        .par_iter()
        .enumerate()
        .map(|(held_out_idx, held_out)| {
            let training_rows = observations
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != held_out_idx)
                .map(|(idx, observation)| (idx, *observation))
                .collect::<Vec<_>>();
            let training = training_rows
                .iter()
                .map(|(_, observation)| *observation)
                .collect::<Vec<_>>();
            let mut prediction =
                ordinary_kriging_predict_unchecked(&training, (held_out.x, held_out.y), config)?;
            prediction.neighbor_indices = prediction
                .neighbor_indices
                .iter()
                .map(|local_idx| training_rows[*local_idx].0)
                .collect();
            Ok(prediction)
        })
        .collect()
}

pub fn ordinary_kriging_leave_one_out_diagnostics(
    observations: &[KrigingObservation],
    config: OrdinaryKrigingConfig,
) -> Result<(Vec<KrigingPrediction>, KrigingLooDiagnostics)> {
    let predictions = ordinary_kriging_leave_one_out(observations, config)?;
    let diagnostics = kriging_loo_diagnostics(observations, &predictions)?;
    Ok((predictions, diagnostics))
}

pub fn empirical_variogram(
    observations: &[KrigingObservation],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> Result<Vec<EmpiricalVariogramBin>> {
    validate_kriging_observations(observations)?;
    if observations.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "empirical variogram requires at least two observations".to_string(),
        ));
    }
    if bin_count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "variogram bin_count must be positive".to_string(),
        ));
    }
    let distance_config = OrdinaryKrigingConfig::new(1.0, 0.0)?
        .with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling)?;
    let pairs = variogram_pairs(observations, distance_config, max_distance)?;
    if pairs.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "empirical variogram has no coordinate pairs within max_distance".to_string(),
        ));
    }
    let max_lag = max_distance.unwrap_or_else(|| {
        pairs
            .iter()
            .map(|(distance, _)| *distance)
            .fold(0.0, f64::max)
    });
    if max_lag <= 0.0 || !max_lag.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "empirical variogram max lag must be positive".to_string(),
        ));
    }
    let width = max_lag / bin_count as f64;
    let mut counts = vec![0usize; bin_count];
    let mut distance_sums = vec![0.0; bin_count];
    let mut semivariance_sums = vec![0.0; bin_count];
    for (distance, semivariance) in pairs {
        let mut bin = (distance / width).floor() as usize;
        if bin >= bin_count {
            bin = bin_count - 1;
        }
        counts[bin] += 1;
        distance_sums[bin] += distance;
        semivariance_sums[bin] += semivariance;
    }
    Ok((0..bin_count)
        .filter_map(|bin| {
            let pair_count = counts[bin];
            if pair_count == 0 {
                return None;
            }
            let lag_min = bin as f64 * width;
            let lag_max = (bin + 1) as f64 * width;
            Some(EmpiricalVariogramBin {
                lag_min,
                lag_max,
                lag_center: 0.5 * (lag_min + lag_max),
                mean_distance: distance_sums[bin] / pair_count as f64,
                semivariance: semivariance_sums[bin] / pair_count as f64,
                pair_count,
            })
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn fit_ordinary_kriging_variogram(
    observations: &[KrigingObservation],
    variogram_models: &[KrigingVariogramModel],
    range_candidates: &[f64],
    nugget_candidates: &[f64],
    sill_candidates: &[f64],
    bin_count: usize,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> Result<KrigingVariogramFit> {
    let bins = empirical_variogram(
        observations,
        bin_count,
        None,
        anisotropy_angle_degrees,
        anisotropy_scaling,
    )?;
    let models = if variogram_models.is_empty() {
        vec![
            KrigingVariogramModel::Exponential,
            KrigingVariogramModel::Gaussian,
            KrigingVariogramModel::Spherical,
            KrigingVariogramModel::Linear,
        ]
    } else {
        variogram_models.to_vec()
    };
    let ranges = if range_candidates.is_empty() {
        default_variogram_ranges(&bins)
    } else {
        validate_variogram_candidates(range_candidates, "range_candidates")?;
        range_candidates.to_vec()
    };
    let nuggets = if nugget_candidates.is_empty() {
        default_variogram_nuggets(&bins)
    } else {
        validate_non_negative_candidates(nugget_candidates, "nugget_candidates")?;
        nugget_candidates.to_vec()
    };
    let sills = if sill_candidates.is_empty() {
        default_variogram_sills(&bins)
    } else {
        validate_variogram_candidates(sill_candidates, "sill_candidates")?;
        sill_candidates.to_vec()
    };
    let mut candidates =
        Vec::with_capacity(models.len() * ranges.len() * nuggets.len() * sills.len());
    for model in models {
        for &range in &ranges {
            for &nugget in &nuggets {
                for &sill in &sills {
                    candidates.push((model, range, nugget, sill));
                }
            }
        }
    }

    candidates
        .par_iter()
        .enumerate()
        .filter_map(|(index, &(model, range, nugget, sill))| {
            let config = OrdinaryKrigingConfig::new(range, nugget)
                .and_then(|config| config.with_sill(sill))
                .and_then(|config| {
                    config.with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling)
                })
                .ok()?
                .with_variogram_model(model);
            let weighted_sse = variogram_weighted_sse(&bins, config);
            weighted_sse
                .is_finite()
                .then_some((index, config, weighted_sse))
        })
        .reduce_with(|left, right| {
            if right.2 < left.2 || (right.2 == left.2 && right.0 < left.0) {
                right
            } else {
                left
            }
        })
        .map(|(_, config, weighted_sse)| KrigingVariogramFit {
            config,
            bins,
            weighted_sse,
        })
        .ok_or_else(|| {
            CartoBoostError::InvalidInput("variogram fit found no valid candidate".to_string())
        })
}
pub fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be positive and finite"
        )));
    }
    Ok(())
}

fn validate_kriging_target(target: (f64, f64)) -> Result<()> {
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    Ok(())
}

fn uses_local_neighbors(config: OrdinaryKrigingConfig) -> bool {
    config.max_neighbors.is_some() || config.max_distance.is_some()
}

fn kriging_loo_diagnostics(
    observations: &[KrigingObservation],
    predictions: &[KrigingPrediction],
) -> Result<KrigingLooDiagnostics> {
    if observations.len() != predictions.len() {
        return Err(CartoBoostError::InvalidInput(
            "kriging diagnostics require one prediction per observation".to_string(),
        ));
    }
    let n = observations.len();
    if n == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kriging diagnostics require at least one prediction".to_string(),
        ));
    }
    let mut error_sum = 0.0;
    let mut abs_error_sum = 0.0;
    let mut squared_error_sum = 0.0;
    let mut standardized_sum = 0.0;
    let mut standardized_squared_sum = 0.0;
    let mut standardized_abs_max = 0.0_f64;
    let mut covered_95 = 0usize;
    let mut variance_sum = 0.0;
    for (observation, prediction) in observations.iter().zip(predictions.iter()) {
        let error = observation.value - prediction.mean;
        let variance = prediction.variance.max(f64::EPSILON);
        let standardized = error / variance.sqrt();
        error_sum += error;
        abs_error_sum += error.abs();
        squared_error_sum += error * error;
        standardized_sum += standardized;
        standardized_squared_sum += standardized * standardized;
        standardized_abs_max = standardized_abs_max.max(standardized.abs());
        variance_sum += prediction.variance;
        if standardized.abs() <= 1.959_963_984_540_054 {
            covered_95 += 1;
        }
    }
    let n_f64 = n as f64;
    Ok(KrigingLooDiagnostics {
        observation_count: n,
        mean_error: error_sum / n_f64,
        mae: abs_error_sum / n_f64,
        rmse: (squared_error_sum / n_f64).sqrt(),
        mean_standardized_error: standardized_sum / n_f64,
        rmse_standardized_error: (standardized_squared_sum / n_f64).sqrt(),
        max_abs_standardized_error: standardized_abs_max,
        interval_coverage_95: covered_95 as f64 / n_f64,
        average_variance: variance_sum / n_f64,
    })
}

fn variogram_pairs(
    observations: &[KrigingObservation],
    distance_config: OrdinaryKrigingConfig,
    max_distance: Option<f64>,
) -> Result<Vec<(f64, f64)>> {
    if let Some(max_distance) = max_distance {
        validate_positive_finite(max_distance, "max_distance")?;
    }
    Ok((0..observations.len())
        .into_par_iter()
        .flat_map_iter(|left_idx| {
            ((left_idx + 1)..observations.len()).filter_map(move |right_idx| {
                let left = observations[left_idx];
                let right = observations[right_idx];
                let distance =
                    transformed_distance((left.x, left.y), (right.x, right.y), distance_config);
                if max_distance
                    .map(|max_distance| distance > max_distance)
                    .unwrap_or(false)
                {
                    return None;
                }
                let diff = left.value - right.value;
                Some((distance, 0.5 * diff * diff))
            })
        })
        .collect())
}

fn validate_variogram_candidates(values: &[f64], name: &str) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    for value in values {
        validate_positive_finite(*value, name)?;
    }
    Ok(())
}

fn validate_non_negative_candidates(values: &[f64], name: &str) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    for value in values {
        if !value.is_finite() || *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} must contain finite non-negative values"
            )));
        }
    }
    Ok(())
}

fn default_variogram_ranges(bins: &[EmpiricalVariogramBin]) -> Vec<f64> {
    let max_distance = bins
        .iter()
        .map(|bin| bin.mean_distance)
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    [0.25, 0.5, 1.0, 1.5, 2.0]
        .iter()
        .map(|factor| max_distance * factor)
        .collect()
}

fn default_variogram_nuggets(bins: &[EmpiricalVariogramBin]) -> Vec<f64> {
    let first = bins
        .first()
        .map(|bin| bin.semivariance)
        .unwrap_or(0.0)
        .max(0.0);
    vec![0.0, first * 0.25, first * 0.5]
}

fn default_variogram_sills(bins: &[EmpiricalVariogramBin]) -> Vec<f64> {
    let max_semivariance = bins
        .iter()
        .map(|bin| bin.semivariance)
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    [0.5, 1.0, 1.5, 2.0]
        .iter()
        .map(|factor| max_semivariance * factor)
        .collect()
}

fn variogram_weighted_sse(bins: &[EmpiricalVariogramBin], config: OrdinaryKrigingConfig) -> f64 {
    bins.par_iter()
        .map(|bin| {
            let fitted = theoretical_semivariogram(bin.mean_distance, config);
            let residual = bin.semivariance - fitted;
            bin.pair_count as f64 * residual * residual
        })
        .sum()
}

fn theoretical_semivariogram(distance: f64, config: OrdinaryKrigingConfig) -> f64 {
    let ratio = distance / config.range;
    match config.variogram_model {
        KrigingVariogramModel::Exponential => config.nugget + config.sill * (1.0 - (-ratio).exp()),
        KrigingVariogramModel::Gaussian => {
            config.nugget + config.sill * (1.0 - (-(ratio * ratio)).exp())
        }
        KrigingVariogramModel::Spherical => {
            let structural = if ratio >= 1.0 {
                1.0
            } else {
                1.5 * ratio - 0.5 * ratio.powi(3)
            };
            config.nugget + config.sill * structural
        }
        // A linear variogram has no bounded covariance counterpart. `sill`
        // is its structural contribution at `range`, so sill/range is slope.
        KrigingVariogramModel::Linear => config.nugget + config.sill * ratio,
    }
}

fn build_kriging_system_matrix(
    observations: &[KrigingObservation],
    config: OrdinaryKrigingConfig,
) -> Vec<Vec<f64>> {
    let n = observations.len();
    let drift_terms = drift_term_count(config.drift);
    let system_size = n + drift_terms;
    (0..system_size)
        .into_par_iter()
        .map(|row_idx| {
            let mut row = vec![0.0; system_size];
            if row_idx < n {
                let left = observations[row_idx];
                for (col_idx, right) in observations.iter().enumerate() {
                    row[col_idx] = if row_idx == col_idx {
                        0.0
                    } else {
                        theoretical_semivariogram(
                            transformed_distance((left.x, left.y), (right.x, right.y), config),
                            config,
                        )
                    };
                }
                let basis = drift_basis((left.x, left.y), config.drift);
                for (basis_idx, basis_value) in basis.iter().enumerate() {
                    row[n + basis_idx] = *basis_value;
                }
            } else {
                let basis_idx = row_idx - n;
                for (col_idx, observation) in observations.iter().enumerate() {
                    row[col_idx] = drift_basis((observation.x, observation.y), config.drift)
                        .get(basis_idx)
                        .copied()
                        .unwrap_or(0.0);
                }
            }
            row
        })
        .collect()
}

fn build_kriging_rhs(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Vec<f64> {
    let drift_terms = drift_term_count(config.drift);
    let mut rhs = observations
        .par_iter()
        .map(|observation| {
            theoretical_semivariogram(
                transformed_distance((observation.x, observation.y), target, config),
                config,
            )
        })
        .collect::<Vec<_>>();
    rhs.extend(
        drift_basis(target, config.drift)
            .into_iter()
            .take(drift_terms),
    );
    rhs
}

fn kriging_prediction_from_solution(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
    rhs: &[f64],
    solution: &[f64],
    neighbor_indices: Vec<usize>,
) -> Result<KrigingPrediction> {
    let n = observations.len();
    let weights = solution.iter().copied().take(n).collect::<Vec<_>>();
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "kriging weights are not finite".to_string(),
        ));
    }
    validate_kriging_drift_constraints(observations, target, config.drift, &weights)?;
    let mean = observations
        .iter()
        .enumerate()
        .map(|(idx, observation)| weights[idx] * observation.value)
        .sum::<f64>();
    if !mean.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging estimate is not finite".to_string(),
        ));
    }
    let raw_variance = solution
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let variance_scale = solution
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left * right).abs())
        .sum::<f64>()
        .max(config.sill + config.nugget);
    let variance =
        checked_non_negative_variance(raw_variance, variance_scale, "kriging prediction variance")?;
    Ok(KrigingPrediction {
        x: target.0,
        y: target.1,
        mean,
        variance,
        weights,
        neighbor_indices,
    })
}

fn validate_numeric_series(values: &[f64], label: &str) -> Result<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "{label} at index {idx} must be finite"
            )));
        }
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be in (0, 1]"
        )));
    }
    Ok(())
}

fn croston_level(values: &[f64], alpha: f64) -> Result<f64> {
    let first_nonzero = values
        .iter()
        .position(|value| *value > 0.0)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "intermittent demand forecast requires at least one non-zero observation"
                    .to_string(),
            )
        })?;
    let mut demand = values[first_nonzero];
    let mut interval = (first_nonzero + 1) as f64;
    let mut elapsed = 0usize;
    for value in values.iter().skip(first_nonzero + 1) {
        elapsed += 1;
        if *value > 0.0 {
            demand += alpha * (*value - demand);
            interval += alpha * (elapsed as f64 - interval);
            elapsed = 0;
        }
    }
    if interval <= 0.0 || !interval.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand interval estimate is invalid".to_string(),
        ));
    }
    Ok(demand / interval)
}

fn tsb_level(values: &[f64], alpha: f64, beta: f64) -> Result<f64> {
    let first_nonzero = values.iter().find(|value| **value > 0.0).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "TSB forecast requires at least one non-zero observation".to_string(),
        )
    })?;
    let mut demand = *first_nonzero;
    let mut probability = 0.0;
    for value in values {
        let occurrence = if *value > 0.0 { 1.0 } else { 0.0 };
        probability += beta * (occurrence - probability);
        if *value > 0.0 {
            demand += alpha * (*value - demand);
        }
    }
    Ok(probability * demand)
}

fn validate_kriging_observations(observations: &[KrigingObservation]) -> Result<()> {
    if observations.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging observations must not be empty".to_string(),
        ));
    }
    for (idx, observation) in observations.iter().enumerate() {
        if !observation.x.is_finite() || !observation.y.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging observation {idx} coordinates must be finite"
            )));
        }
        if !observation.value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging observation {idx} value must be finite"
            )));
        }
    }
    Ok(())
}

fn select_kriging_neighbors(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<Vec<(usize, &KrigingObservation)>> {
    let mut candidates = observations
        .iter()
        .enumerate()
        .map(|(idx, observation)| {
            (
                idx,
                observation,
                transformed_distance((observation.x, observation.y), target, config),
            )
        })
        .filter(|(_, _, distance)| {
            config
                .max_distance
                .map(|max_distance| *distance <= max_distance)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.2
            .partial_cmp(&right.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    if let Some(max_neighbors) = config.max_neighbors {
        candidates.truncate(max_neighbors);
    }
    if candidates.len() < config.min_neighbors {
        return Err(CartoBoostError::InvalidInput(format!(
            "kriging found {} neighbors, but min_neighbors is {}",
            candidates.len(),
            config.min_neighbors
        )));
    }
    Ok(candidates
        .into_iter()
        .map(|(idx, observation, _)| (idx, observation))
        .collect())
}

fn transformed_distance(left: (f64, f64), right: (f64, f64), config: OrdinaryKrigingConfig) -> f64 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    if (config.anisotropy_angle_degrees.abs() <= f64::EPSILON)
        && ((config.anisotropy_scaling - 1.0).abs() <= f64::EPSILON)
    {
        return (dx * dx + dy * dy).sqrt();
    }
    let angle = config.anisotropy_angle_degrees.to_radians();
    let cos = angle.cos();
    let sin = angle.sin();
    let rotated_x = cos * dx + sin * dy;
    let rotated_y = -sin * dx + cos * dy;
    (rotated_x * rotated_x + (rotated_y / config.anisotropy_scaling).powi(2)).sqrt()
}

fn drift_term_count(drift: KrigingDrift) -> usize {
    match drift {
        KrigingDrift::Ordinary => 1,
        KrigingDrift::Linear => 3,
    }
}

fn drift_basis(point: (f64, f64), drift: KrigingDrift) -> Vec<f64> {
    match drift {
        KrigingDrift::Ordinary => vec![1.0],
        KrigingDrift::Linear => vec![1.0, point.0, point.1],
    }
}

fn validate_kriging_drift_constraints(
    observations: &[KrigingObservation],
    target: (f64, f64),
    drift: KrigingDrift,
    weights: &[f64],
) -> Result<()> {
    let target_basis = drift_basis(target, drift);
    for (basis_idx, expected) in target_basis.iter().enumerate() {
        let mut actual = 0.0;
        let mut scale = expected.abs().max(1.0);
        for (observation, weight) in observations.iter().zip(weights) {
            let basis = drift_basis((observation.x, observation.y), drift)[basis_idx];
            actual += weight * basis;
            scale += (weight * basis).abs();
        }
        if !actual.is_finite() || (actual - expected).abs() > 1.0e-9 * scale {
            return Err(CartoBoostError::InvalidInput(
                "kriging solution does not satisfy its unbiasedness constraints; system is numerically unstable"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

impl LinearSystemFactorization {
    fn factor(mut matrix: Vec<Vec<f64>>) -> Option<Self> {
        let n = matrix.len();
        if n == 0
            || matrix
                .iter()
                .any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        {
            return None;
        }
        let mut permutation = (0..n).collect::<Vec<_>>();
        let mut row_scales = matrix
            .iter()
            .map(|row| row.iter().map(|value| value.abs()).fold(0.0, f64::max))
            .collect::<Vec<_>>();
        if row_scales.contains(&0.0) {
            return None;
        }
        for pivot in 0..n {
            let best = (pivot..n).max_by(|left, right| {
                let left_score = matrix[*left][pivot].abs() / row_scales[*left];
                let right_score = matrix[*right][pivot].abs() / row_scales[*right];
                left_score.total_cmp(&right_score)
            })?;
            let pivot_value = matrix[best][pivot].abs();
            let tolerance = 128.0 * f64::EPSILON * row_scales[best].max(f64::MIN_POSITIVE);
            if !pivot_value.is_finite() || pivot_value <= tolerance {
                return None;
            }
            matrix.swap(pivot, best);
            permutation.swap(pivot, best);
            row_scales.swap(pivot, best);
            #[allow(clippy::needless_range_loop)]
            for row in (pivot + 1)..n {
                let factor = matrix[row][pivot] / matrix[pivot][pivot];
                if !factor.is_finite() {
                    return None;
                }
                matrix[row][pivot] = factor;
                for column in (pivot + 1)..n {
                    matrix[row][column] -= factor * matrix[pivot][column];
                    if !matrix[row][column].is_finite() {
                        return None;
                    }
                }
            }
        }
        Some(Self {
            lu: matrix,
            permutation,
        })
    }

    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let n = self.lu.len();
        if rhs.len() != n || rhs.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut solution = self
            .permutation
            .iter()
            .map(|index| rhs[*index])
            .collect::<Vec<_>>();
        for row in 0..n {
            for column in 0..row {
                solution[row] -= self.lu[row][column] * solution[column];
            }
        }
        for row in (0..n).rev() {
            for column in (row + 1)..n {
                solution[row] -= self.lu[row][column] * solution[column];
            }
            solution[row] /= self.lu[row][row];
        }
        solution
            .iter()
            .all(|value| value.is_finite())
            .then_some(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_filter_tracks_linear_trend_and_forecasts() {
        let config = LocalLinearKalmanConfig::new(0.01, 0.001, 0.1).expect("config");
        let result = fit_local_linear_kalman(&[12.0, 14.0, 16.0, 18.0], config).expect("filter");
        let forecast = local_linear_kalman_forecast(result.final_state, 2).expect("forecast");

        assert_eq!(result.estimates.len(), 3);
        assert_eq!(result.smoothed_states.len(), 4);
        assert!(result.final_state.trend > 0.0);
        assert!(result.final_covariance[0][0] > 0.0);
        assert!(result.estimates.last().unwrap().innovation_variance > 0.0);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.residual_summary.fitted_count, 3);
        assert!(result.residual_summary.rmse >= 0.0);
        assert!(forecast[1] > forecast[0]);

        let distribution = local_linear_kalman_forecast_distribution(
            result.final_state,
            result.final_covariance,
            config,
            2,
            1.96,
        )
        .expect("forecast distribution");
        assert_eq!(distribution.len(), 2);
        assert_eq!(distribution[1].mean, forecast[1]);
        assert!(distribution[1].lower < distribution[1].mean);
        assert!(distribution[1].upper > distribution[1].mean);
    }

    #[test]
    fn kalman_filter_rejects_bad_inputs() {
        let config = LocalLinearKalmanConfig::new(0.01, 0.001, 0.1).expect("config");

        assert!(fit_local_linear_kalman(&[1.0], config).is_err());
        assert!(fit_local_linear_kalman(&[1.0, f64::NAN], config).is_err());
        assert!(LocalLinearKalmanConfig::new(0.0, 0.001, 0.1).is_err());
        assert!(local_linear_kalman_forecast(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0
            },
            0
        )
        .is_err());
        assert!(local_linear_kalman_forecast(
            LocalLinearKalmanState {
                level: f64::NAN,
                trend: 0.0,
            },
            1,
        )
        .is_err());
        let invalid_config = LocalLinearKalmanConfig {
            level_process_variance: -1.0,
            trend_process_variance: 0.001,
            observation_variance: 0.1,
        };
        assert!(fit_local_linear_kalman(&[1.0, 2.0], invalid_config).is_err());
        assert!(local_linear_kalman_forecast_distribution(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0,
            },
            [[1.0, 0.0], [0.0, 1.0]],
            invalid_config,
            1,
            1.96,
        )
        .is_err());
        assert!(local_linear_kalman_forecast_distribution(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0,
            },
            [[1.0, 2.0], [0.0, 1.0]],
            config,
            1,
            1.96,
        )
        .is_err());
    }

    #[test]
    fn local_linear_kalman_filters_second_observation_once() {
        let config = LocalLinearKalmanConfig::new(1.0, 1.0, 1.0).expect("config");

        let result = fit_local_linear_kalman(&[0.0, 10.0], config).expect("filter");
        let estimate = result.estimates.first().expect("second observation update");

        assert_eq!(estimate.prior_level, 0.0);
        assert_eq!(estimate.prior_trend, 0.0);
        assert_eq!(estimate.innovation, 10.0);
        assert!((result.final_state.level - 7.5).abs() < 1.0e-12);
        assert!((result.final_state.trend - 2.5).abs() < 1.0e-12);
        assert!((result.final_covariance[0][0] - 0.75).abs() < 1.0e-12);
        assert!((result.final_covariance[0][1] - 0.25).abs() < 1.0e-12);
        assert!((result.final_covariance[1][1] - 1.75).abs() < 1.0e-12);
    }

    #[test]
    fn local_linear_kalman_smoothing_is_scale_invariant() {
        let values = [0.0, 10.0, 0.0, 10.0];
        let base = fit_local_linear_kalman(
            &values,
            LocalLinearKalmanConfig::new(0.1, 0.01, 1.0).expect("base config"),
        )
        .expect("base filter");
        let scaled = fit_local_linear_kalman(
            &values,
            LocalLinearKalmanConfig::new(1.0e-7, 1.0e-8, 1.0e-6).expect("scaled config"),
        )
        .expect("scaled filter");

        for (base, scaled) in base.smoothed_states.iter().zip(&scaled.smoothed_states) {
            assert!((base.level - scaled.level).abs() < 1.0e-8);
            assert!((base.trend - scaled.trend).abs() < 1.0e-8);
        }
    }

    #[test]
    fn local_linear_kalman_forecast_variance_propagates_process_noise_exactly() {
        let config = LocalLinearKalmanConfig::new(2.0, 3.0, 5.0).expect("config");
        let distribution = local_linear_kalman_forecast_distribution(
            LocalLinearKalmanState {
                level: 0.0,
                trend: 0.0,
            },
            [[0.0, 0.0], [0.0, 0.0]],
            config,
            3,
            1.96,
        )
        .expect("forecast distribution");

        assert_eq!(distribution[0].variance, 7.0);
        assert_eq!(distribution[1].variance, 12.0);
        assert_eq!(distribution[2].variance, 26.0);
    }

    #[test]
    fn local_level_kalman_forecasts_flat_level() {
        let config = LocalLevelKalmanConfig::new(0.01, 0.1).expect("config");
        let result = fit_local_level_kalman(&[12.0, 13.0, 13.5], config).expect("filter");
        let forecast = local_level_kalman_forecast(result.final_level, 3).expect("forecast");

        assert_eq!(result.estimates.len(), 2);
        assert_eq!(result.smoothed_states.len(), 3);
        assert!(result.final_variance > 0.0);
        assert!(result.estimates.last().unwrap().gain > 0.0);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.residual_summary.fitted_count, 2);
        assert_eq!(forecast, vec![result.final_level; 3]);

        let distribution = local_level_kalman_forecast_distribution(
            result.final_level,
            result.final_variance,
            config,
            3,
            1.96,
        )
        .expect("forecast distribution");
        assert_eq!(distribution.len(), 3);
        assert_eq!(distribution[0].mean, result.final_level);
        assert!(distribution[0].variance > result.final_variance);
    }

    #[test]
    fn intermittent_demand_methods_are_positive_and_bias_adjusted() {
        let values = [0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0];
        let croston =
            intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Croston)
                .expect("croston");
        let sba = intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Sba)
            .expect("sba");
        let tsb = intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Tsb)
            .expect("tsb");

        assert_eq!(croston.len(), 2);
        assert!(croston[0] > 0.0);
        assert!(sba[0] < croston[0]);
        assert!(tsb[0] > 0.0);
    }

    #[test]
    fn intermittent_demand_rejects_invalid_inputs() {
        assert!(intermittent_demand_forecast(
            &[0.0, -1.0],
            1,
            0.1,
            0.1,
            IntermittentDemandMethod::Croston,
        )
        .is_err());
        assert!(intermittent_demand_forecast(
            &[0.0, 0.0],
            1,
            0.1,
            0.1,
            IntermittentDemandMethod::Tsb,
        )
        .is_err());
        assert!(intermittent_demand_forecast(
            &[1.0],
            0,
            0.1,
            0.1,
            IntermittentDemandMethod::Croston,
        )
        .is_err());
    }

    #[test]
    fn ordinary_kriging_returns_exact_known_coordinate_with_tiny_nugget() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(1.0, 1.0e-9).expect("config");

        let prediction =
            ordinary_kriging_predict(&observations, (0.0, 0.0), config).expect("kriging");

        assert!((prediction.mean - 12.0).abs() < 1.0e-4);
        assert_eq!(prediction.weights.len(), 2);
    }

    #[test]
    fn bounded_ordinary_kriging_matches_reference_equations() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 1.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 3.0,
            },
            KrigingObservation {
                x: 0.0,
                y: 1.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 1.0,
                value: 5.0,
            },
        ];
        let cases = [
            (
                KrigingVariogramModel::Exponential,
                2.058_716_343_389_453_8,
                0.916_622_073_252_181_6,
                [
                    0.406_016_481_098_025_74,
                    0.305_494_690_594_402_15,
                    0.235_409_450_343_212_84,
                    0.053_079_377_964_359_19,
                ],
            ),
            (
                KrigingVariogramModel::Gaussian,
                1.771_379_181_116_378_6,
                0.405_754_171_709_639_1,
                [
                    0.433_091_393_303_780_1,
                    0.374_703_977_216_747_64,
                    0.248_949_097_078_335_18,
                    -0.056_744_467_598_862_924,
                ],
            ),
            (
                KrigingVariogramModel::Spherical,
                2.085_389_982_235_605,
                1.517_206_693_789_695_7,
                [
                    0.437_944_953_865_678_67,
                    0.290_440_969_324_087,
                    0.193_982_754_551_168_7,
                    0.077_631_322_259_065_66,
                ],
            ),
        ];

        for (model, expected_mean, expected_variance, expected_weights) in cases {
            let config = OrdinaryKrigingConfig::new(1.2, 0.15)
                .expect("config")
                .with_sill(1.7)
                .expect("sill")
                .with_variogram_model(model);
            let prediction = ordinary_kriging_predict(&observations, (0.4, 0.3), config)
                .expect("reference prediction");

            assert!((prediction.mean - expected_mean).abs() < 1.0e-12);
            assert!((prediction.variance - expected_variance).abs() < 1.0e-12);
            for (actual, expected) in prediction.weights.iter().zip(expected_weights) {
                assert!((actual - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn linear_variogram_is_unbounded_and_has_known_midpoint_solution() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 0.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 2.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(2.0, 0.0)
            .expect("config")
            .with_sill(4.0)
            .expect("sill")
            .with_variogram_model(KrigingVariogramModel::Linear);

        assert_eq!(theoretical_semivariogram(3.0, config), 6.0);
        let prediction =
            ordinary_kriging_predict(&observations, (1.0, 0.0), config).expect("prediction");
        assert!((prediction.mean - 1.0).abs() < 1.0e-12);
        assert!((prediction.variance - 2.0).abs() < 1.0e-12);
        assert!((prediction.weights[0] - 0.5).abs() < 1.0e-12);
        assert!((prediction.weights[1] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn ordinary_kriging_supports_variogram_neighbors_and_variance() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 0.0,
                value: 50.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6)
            .expect("config")
            .with_variogram_model(KrigingVariogramModel::Spherical)
            .with_neighbor_limits(Some(2), 2, None)
            .expect("neighbors");

        let prediction =
            ordinary_kriging_predict(&observations, (10.0, 0.0), config).expect("kriging");

        assert!(prediction.variance >= 0.0);
        assert_eq!(prediction.weights.len(), 2);
        assert_eq!(prediction.neighbor_indices.len(), 2);
    }

    #[test]
    fn ordinary_kriging_system_matches_predict_many() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 5.0,
                value: 55.0,
            },
        ];
        let targets = vec![(0.0, 0.0), (5.0, 2.0), (20.0, 5.0)];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6)
            .expect("config")
            .with_variogram_model(KrigingVariogramModel::Gaussian);

        let system = OrdinaryKrigingSystem::new(&observations, config).expect("system");
        let cached = system.predict_many(&targets).expect("cached");
        let direct = targets
            .iter()
            .map(|target| ordinary_kriging_predict_unchecked(&observations, *target, config))
            .collect::<Result<Vec<_>>>()
            .expect("direct");

        assert_eq!(system.observation_count(), observations.len());
        assert_eq!(system.drift_terms(), 1);
        assert_eq!(cached.len(), direct.len());
        for (cached, direct) in cached.iter().zip(direct.iter()) {
            assert!((cached.mean - direct.mean).abs() < 1.0e-8);
            assert!((cached.variance - direct.variance).abs() < 1.0e-8);
            assert_eq!(cached.neighbor_indices, vec![0, 1, 2]);
        }
    }

    #[test]
    fn ordinary_kriging_system_rejects_local_neighbor_config() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6)
            .expect("config")
            .with_neighbor_limits(Some(1), 1, None)
            .expect("neighbors");

        assert!(OrdinaryKrigingSystem::new(&observations, config).is_err());
    }

    #[test]
    fn ordinary_kriging_leave_one_out_returns_all_observations() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 0.0,
                value: 50.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6).expect("config");

        let diagnostics = ordinary_kriging_leave_one_out(&observations, config).expect("loo");

        assert_eq!(diagnostics.len(), observations.len());
        assert!(diagnostics
            .iter()
            .all(|prediction| prediction.variance >= 0.0));
    }

    #[test]
    fn ordinary_kriging_leave_one_out_reports_original_neighbor_indices() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 0.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 4.0,
                y: 0.0,
                value: 4.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(2.0, 1.0e-6)
            .expect("config")
            .with_neighbor_limits(Some(1), 1, None)
            .expect("neighbors");

        let predictions = ordinary_kriging_leave_one_out(&observations, config).expect("LOO");

        assert_eq!(predictions[0].neighbor_indices, vec![1]);
        assert_eq!(predictions[1].neighbor_indices, vec![0]);
        assert_eq!(predictions[2].neighbor_indices, vec![1]);
    }

    #[test]
    fn empirical_variogram_bins_coordinate_pairs() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 10.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 16.0,
            },
        ];

        let bins = empirical_variogram(&observations, 2, None, 0.0, 1.0).expect("variogram");

        assert!(!bins.is_empty());
        assert_eq!(bins.iter().map(|bin| bin.pair_count).sum::<usize>(), 3);
        assert!(bins.iter().all(|bin| bin.semivariance >= 0.0));
    }

    #[test]
    fn empirical_variogram_keeps_collocated_pairs_for_nugget_evidence() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 0.0,
            },
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 1.0,
            },
        ];

        let bins = empirical_variogram(&observations, 2, None, 0.0, 1.0)
            .expect("variogram with collocated observations");

        assert_eq!(bins.iter().map(|bin| bin.pair_count).sum::<usize>(), 3);
        assert!(bins
            .iter()
            .any(|bin| bin.lag_min == 0.0 && bin.semivariance >= 2.0));
    }

    #[test]
    fn variogram_fit_selects_candidate_config() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 10.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 16.0,
            },
            KrigingObservation {
                x: 3.0,
                y: 0.0,
                value: 20.0,
            },
        ];

        let fit = fit_ordinary_kriging_variogram(
            &observations,
            &[
                KrigingVariogramModel::Exponential,
                KrigingVariogramModel::Spherical,
            ],
            &[1.0, 2.0],
            &[0.0, 0.1],
            &[1.0, 5.0],
            3,
            0.0,
            1.0,
        )
        .expect("fit");

        assert!(fit.weighted_sse.is_finite());
        assert!([1.0, 2.0].contains(&fit.config.range));
        assert!([1.0, 5.0].contains(&fit.config.sill));
    }

    #[test]
    fn kriging_leave_one_out_diagnostics_summarize_residuals() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 0.0,
                value: 50.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6).expect("config");

        let (predictions, diagnostics) =
            ordinary_kriging_leave_one_out_diagnostics(&observations, config).expect("diagnostics");

        assert_eq!(predictions.len(), observations.len());
        assert_eq!(diagnostics.observation_count, observations.len());
        assert!(diagnostics.rmse >= 0.0);
        assert!((0.0..=1.0).contains(&diagnostics.interval_coverage_95));
    }

    #[test]
    fn universal_kriging_linear_drift_reproduces_plane() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 5.0,
            },
            KrigingObservation {
                x: 0.0,
                y: 1.0,
                value: 7.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(10.0, 1.0e-9)
            .expect("config")
            .with_drift(KrigingDrift::Linear);

        let prediction =
            ordinary_kriging_predict(&observations, (0.5, 0.5), config).expect("kriging");

        assert!((prediction.mean - 6.0).abs() < 1.0e-6);
    }

    #[test]
    fn ordinary_kriging_rejects_bad_inputs() {
        let config = OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("config");

        assert!(ordinary_kriging_predict(&[], (0.0, 0.0), config).is_err());
        assert!(OrdinaryKrigingConfig::new(0.0, 1.0e-6).is_err());
        assert!(OrdinaryKrigingConfig::new(1.0, -1.0).is_err());
        let invalid_public_config = OrdinaryKrigingConfig {
            range: 1.0,
            nugget: 0.0,
            sill: f64::NAN,
            variogram_model: KrigingVariogramModel::Exponential,
            drift: KrigingDrift::Ordinary,
            anisotropy_angle_degrees: 0.0,
            anisotropy_scaling: 1.0,
            max_neighbors: None,
            min_neighbors: 1,
            max_distance: None,
        };
        assert!(ordinary_kriging_predict(
            &[KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 1.0,
            }],
            (0.0, 0.0),
            invalid_public_config,
        )
        .is_err());
        let impossible_linear_neighbors = OrdinaryKrigingConfig::new(1.0, 0.0)
            .expect("config")
            .with_neighbor_limits(Some(2), 1, None)
            .expect("ordinary neighbor config")
            .with_drift(KrigingDrift::Linear);
        assert!(impossible_linear_neighbors.validate().is_err());
        assert!(ordinary_kriging_predict(
            &[KrigingObservation {
                x: 0.0,
                y: f64::NAN,
                value: 1.0,
            }],
            (0.0, 0.0),
            config,
        )
        .is_err());
    }
}
