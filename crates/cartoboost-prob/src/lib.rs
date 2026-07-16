use cartoboost_core::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    pub method: String,
    pub alpha: f64,
    pub train_end_exclusive: Option<usize>,
    pub calibration_start: Option<usize>,
    pub calibration_end_exclusive: Option<usize>,
    pub test_start: Option<usize>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub groups: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionalPrediction {
    pub mean: f64,
    pub median: Option<f64>,
    pub quantiles: BTreeMap<String, f64>,
    pub std: Option<f64>,
    pub interval_lower: Option<f64>,
    pub interval_upper: Option<f64>,
    pub calibration: Option<CalibrationMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionalForecastResult {
    pub predictions: Vec<DistributionalPrediction>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkCalibrationReportFields {
    pub coverage_by_horizon: BTreeMap<usize, f64>,
    pub coverage_by_spatial_block: BTreeMap<String, f64>,
    pub width_by_horizon: BTreeMap<usize, f64>,
    pub residual_morans_i_after_calibration: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntervalMetrics {
    pub coverage: f64,
    pub mean_width: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PitBins {
    pub edges: Vec<f64>,
    pub counts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionalFlowDistributionHead {
    pub quantiles: Vec<f64>,
    pub sample_count: usize,
    pub location_weights: Vec<f64>,
    pub scale_weights: Vec<f64>,
    pub residual_scale: f64,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowPrediction {
    pub samples: Vec<Vec<f64>>,
    pub log_likelihood: Vec<f64>,
    pub marginal_quantiles: Vec<Vec<f64>>,
    pub joint_scenario_paths: Vec<Vec<f64>>,
    pub tail_risk_metrics: BTreeMap<String, f64>,
    pub metrics: BTreeMap<String, f64>,
}

pub type JointHorizonFlowHead = ConditionalFlowDistributionHead;
pub type ResidualFlowCalibrator = ConditionalFlowDistributionHead;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DiffusionEdge {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoTemporalDiffusionScenarioModel {
    pub scenario_count: usize,
    pub diffusion_steps: usize,
    pub shock_scale: f64,
    pub capability_tier: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionScenarioPrediction {
    pub scenarios: Vec<Vec<Vec<f64>>>,
    pub scenario_mean: Vec<Vec<f64>>,
    pub scenario_variance: Vec<Vec<f64>>,
    pub spatial_correlation: f64,
    pub point_forecast_comparison: BTreeMap<String, f64>,
    pub metadata: BTreeMap<String, String>,
}

pub type FlowScenarioGenerator = GeoTemporalDiffusionScenarioModel;
pub type ConditionalResidualDiffusion = GeoTemporalDiffusionScenarioModel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplitOrder {
    pub train_end_exclusive: usize,
    pub calibration_start: usize,
    pub calibration_end_exclusive: usize,
    pub test_start: usize,
}

impl SplitOrder {
    pub fn validate(&self) -> Result<()> {
        if self.train_end_exclusive == 0 {
            return invalid("training split must contain at least one row");
        }
        if self.train_end_exclusive > self.calibration_start {
            return invalid("training rows must end before calibration rows start");
        }
        if self.calibration_start >= self.calibration_end_exclusive {
            return invalid("calibration split must contain at least one row");
        }
        if self.calibration_end_exclusive > self.test_start {
            return invalid("calibration rows must end before test rows start");
        }
        Ok(())
    }
}

// Cohesive implementation families share the crate namespace.
include!("probability/metrics.rs");
include!("probability/distribution.rs");
include!("probability/conformal.rs");
include!("probability/helpers.rs");
include!("probability/tests.rs");
