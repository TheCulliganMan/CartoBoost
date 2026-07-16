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

// Statistical utility families share this module namespace.
include!("utilities/configuration.rs");
include!("utilities/kalman.rs");
include!("utilities/kriging.rs");
include!("utilities/demand.rs");
include!("utilities/kriging_system.rs");
include!("utilities/tests.rs");
