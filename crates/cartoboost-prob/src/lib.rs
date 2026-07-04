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

pub fn pinball_loss(actual: &[f64], prediction: &[f64], quantile: f64) -> Result<f64> {
    cartoboost_core::forecasting::pinball_loss(actual, prediction, quantile)
}

pub fn interval_coverage(actual: &[f64], lower: &[f64], upper: &[f64]) -> Result<f64> {
    cartoboost_core::forecasting::interval_coverage(actual, lower, upper)
}

pub fn mean_interval_width(lower: &[f64], upper: &[f64]) -> Result<f64> {
    cartoboost_core::forecasting::mean_interval_width(lower, upper)
}

pub fn interval_metrics(actual: &[f64], lower: &[f64], upper: &[f64]) -> Result<IntervalMetrics> {
    Ok(IntervalMetrics {
        coverage: interval_coverage(actual, lower, upper)?,
        mean_width: mean_interval_width(lower, upper)?,
    })
}

pub fn crps_approximation(
    actual: &[f64],
    quantiles: &[f64],
    predictions: &[Vec<f64>],
) -> Result<f64> {
    validate_quantile_rows(actual, quantiles, predictions)?;
    let mut total = 0.0;
    for (idx, row) in predictions.iter().enumerate() {
        for (&level, &prediction) in quantiles.iter().zip(row) {
            total += 2.0
                * cartoboost_core::forecasting::pinball_loss(&[actual[idx]], &[prediction], level)?;
        }
    }
    Ok(total / (actual.len() * quantiles.len()) as f64)
}

pub fn weighted_interval_score(
    actual: &[f64],
    median: &[f64],
    intervals: &[(f64, Vec<f64>, Vec<f64>)],
) -> Result<f64> {
    validate_same_non_empty(actual, median, "actual", "median")?;
    if intervals.is_empty() {
        return invalid("intervals must contain at least one central interval");
    }
    let mut score = 0.5
        * actual
            .iter()
            .zip(median)
            .map(|(&y, &m)| (y - m).abs())
            .sum::<f64>();
    let mut weight_sum = 0.5;
    for (alpha, lower, upper) in intervals {
        validate_alpha(*alpha)?;
        validate_same_non_empty(actual, lower, "actual", "lower")?;
        validate_same_non_empty(actual, upper, "actual", "upper")?;
        let weight = *alpha / 2.0;
        weight_sum += weight;
        for ((&y, &lo), &hi) in actual.iter().zip(lower).zip(upper) {
            if lo > hi {
                return invalid("lower bounds must be less than or equal to upper bounds");
            }
            let below = if y < lo { (lo - y) * 2.0 / alpha } else { 0.0 };
            let above = if y > hi { (y - hi) * 2.0 / alpha } else { 0.0 };
            score += weight * (hi - lo + below + above);
        }
    }
    Ok(score / (actual.len() as f64 * weight_sum))
}

pub fn pit_bins(
    actual: &[f64],
    quantiles: &[f64],
    predictions: &[Vec<f64>],
    bins: usize,
) -> Result<PitBins> {
    validate_quantile_rows(actual, quantiles, predictions)?;
    if bins == 0 {
        return invalid("bins must be positive");
    }
    let mut counts = vec![0usize; bins];
    for (&y, row) in actual.iter().zip(predictions) {
        let mut pit = 0.0;
        for (&level, &prediction) in quantiles.iter().zip(row) {
            if y >= prediction {
                pit = level;
            } else {
                break;
            }
        }
        let idx = ((pit * bins as f64).floor() as usize).min(bins - 1);
        counts[idx] += 1;
    }
    let edges = (0..=bins).map(|idx| idx as f64 / bins as f64).collect();
    Ok(PitBins { edges, counts })
}

pub fn conditional_flow_fit_json(
    hidden: &[Vec<f64>],
    residuals: &[f64],
    quantiles: &[f64],
    sample_count: usize,
) -> Result<String> {
    let model = ConditionalFlowDistributionHead::fit(hidden, residuals, quantiles, sample_count)?;
    serde_json::to_string(&model).map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
}

pub fn conditional_flow_predict_json(
    artifact_json: &str,
    hidden: &[Vec<f64>],
    actual: Option<&[f64]>,
) -> Result<String> {
    let model: ConditionalFlowDistributionHead = serde_json::from_str(artifact_json)
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
    let prediction = model.predict(hidden, actual)?;
    serde_json::to_string(&prediction).map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
}

pub fn diffusion_scenario_generate_json(
    point_forecast: &[Vec<f64>],
    edges: &[DiffusionEdge],
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
) -> Result<String> {
    let model =
        GeoTemporalDiffusionScenarioModel::new(scenario_count, diffusion_steps, shock_scale)?;
    let prediction = model.generate(point_forecast, edges)?;
    serde_json::to_string(&prediction).map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
}

impl ConditionalFlowDistributionHead {
    pub fn fit(
        hidden: &[Vec<f64>],
        residuals: &[f64],
        quantiles: &[f64],
        sample_count: usize,
    ) -> Result<Self> {
        validate_same_non_empty_matrix(hidden, residuals, "hidden", "residuals")?;
        validate_quantile_grid(quantiles)?;
        if sample_count == 0 {
            return invalid("sample_count must be positive");
        }
        let location_weights = ridge_fit(hidden, residuals, 1.0e-6);
        let predicted = predict_linear(hidden, &location_weights)?;
        let abs_residuals = residuals
            .iter()
            .zip(predicted.iter())
            .map(|(&actual, &pred)| (actual - pred).abs().ln_1p())
            .collect::<Vec<_>>();
        let scale_weights = ridge_fit(hidden, &abs_residuals, 1.0e-6);
        let residual_scale = (residuals.iter().map(|v| v * v).sum::<f64>()
            / residuals.len() as f64)
            .sqrt()
            .max(1.0e-9);
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "architecture".to_string(),
            "conditional_residual_sampler".to_string(),
        );
        metadata.insert("invertible_flow".to_string(), "false".to_string());
        metadata.insert("sample_count".to_string(), sample_count.to_string());
        metadata.insert(
            "quantiles".to_string(),
            serde_json::to_string(quantiles).unwrap(),
        );
        Ok(Self {
            quantiles: quantiles.to_vec(),
            sample_count,
            location_weights,
            scale_weights,
            residual_scale,
            metadata,
        })
    }

    pub fn predict(&self, hidden: &[Vec<f64>], actual: Option<&[f64]>) -> Result<FlowPrediction> {
        if hidden.is_empty() {
            return invalid("hidden must contain at least one row");
        }
        if let Some(actual) = actual {
            validate_same_non_empty(actual, &vec![0.0; hidden.len()], "actual", "hidden")?;
        }
        let location = predict_linear(hidden, &self.location_weights)?;
        let raw_scale = predict_linear(hidden, &self.scale_weights)?;
        let scale = raw_scale
            .iter()
            .map(|value| value.exp().max(1.0e-6) * self.residual_scale)
            .collect::<Vec<_>>();
        let samples = location
            .iter()
            .zip(scale.iter())
            .enumerate()
            .map(|(row, (&loc, &s))| {
                (0..self.sample_count)
                    .map(|sample| loc + s * deterministic_standard_sample(row, sample))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let marginal_quantiles = location
            .iter()
            .zip(scale.iter())
            .map(|(&loc, &s)| {
                self.quantiles
                    .iter()
                    .map(|&q| loc + s * normal_quantile_proxy(q))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let joint_scenario_paths = (0..self.sample_count)
            .map(|sample| samples.iter().map(|row| row[sample]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let log_likelihood = match actual {
            Some(values) => values
                .iter()
                .zip(location.iter())
                .zip(scale.iter())
                .map(|((&y, &loc), &s)| gaussian_log_likelihood(y, loc, s))
                .collect(),
            None => vec![0.0; hidden.len()],
        };
        let mut tail_risk_metrics = BTreeMap::new();
        tail_risk_metrics.insert(
            "p05_mean".to_string(),
            quantile_mean(&marginal_quantiles, 0)?,
        );
        tail_risk_metrics.insert(
            "p95_mean".to_string(),
            quantile_mean(&marginal_quantiles, self.quantiles.len() - 1)?,
        );
        tail_risk_metrics.insert(
            "expected_shortfall_low".to_string(),
            samples
                .iter()
                .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                .sum::<f64>()
                / samples.len() as f64,
        );
        let mut metrics = BTreeMap::new();
        if let Some(values) = actual {
            metrics.insert(
                "log_likelihood_mean".to_string(),
                log_likelihood.iter().sum::<f64>() / log_likelihood.len() as f64,
            );
            metrics.insert(
                "crps".to_string(),
                crps_approximation(values, &self.quantiles, &marginal_quantiles)?,
            );
            let median_idx = self
                .quantiles
                .iter()
                .position(|q| (*q - 0.5).abs() < 1.0e-12)
                .unwrap_or(self.quantiles.len() / 2);
            let median = marginal_quantiles
                .iter()
                .map(|row| row[median_idx])
                .collect::<Vec<_>>();
            metrics.insert(
                "pinball_median".to_string(),
                pinball_loss(values, &median, self.quantiles[median_idx])?,
            );
            let lower = marginal_quantiles
                .iter()
                .map(|row| row[0])
                .collect::<Vec<_>>();
            let upper = marginal_quantiles
                .iter()
                .map(|row| row[self.quantiles.len() - 1])
                .collect::<Vec<_>>();
            metrics.insert(
                "interval_coverage".to_string(),
                interval_coverage(values, &lower, &upper)?,
            );
            metrics.insert(
                "interval_width".to_string(),
                mean_interval_width(&lower, &upper)?,
            );
            metrics.insert(
                "joint_path_calibration".to_string(),
                joint_path_calibration(values, &joint_scenario_paths),
            );
            metrics.insert(
                "tail_event_calibration".to_string(),
                tail_event_calibration(values, &upper),
            );
        }
        Ok(FlowPrediction {
            samples,
            log_likelihood,
            marginal_quantiles,
            joint_scenario_paths,
            tail_risk_metrics,
            metrics,
        })
    }
}

impl GeoTemporalDiffusionScenarioModel {
    pub fn new(scenario_count: usize, diffusion_steps: usize, shock_scale: f64) -> Result<Self> {
        if scenario_count == 0 {
            return invalid("scenario_count must be positive");
        }
        if diffusion_steps == 0 {
            return invalid("diffusion_steps must be positive");
        }
        if !shock_scale.is_finite() || shock_scale <= 0.0 {
            return invalid("shock_scale must be finite and positive");
        }
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "architecture".to_string(),
            "geo_temporal_diffusion_scenario".to_string(),
        );
        metadata.insert("capability_tier".to_string(), "experimental".to_string());
        metadata.insert("auto_geo_enabled".to_string(), "false".to_string());
        metadata.insert(
            "primary_benchmark_evidence".to_string(),
            "false".to_string(),
        );
        Ok(Self {
            scenario_count,
            diffusion_steps,
            shock_scale,
            capability_tier: "experimental".to_string(),
            metadata,
        })
    }

    pub fn generate(
        &self,
        point_forecast: &[Vec<f64>],
        edges: &[DiffusionEdge],
    ) -> Result<DiffusionScenarioPrediction> {
        validate_panel(point_forecast, "point_forecast")?;
        validate_edges(edges, point_forecast[0].len())?;
        let horizon = point_forecast.len();
        let nodes = point_forecast[0].len();
        let mut scenarios = Vec::with_capacity(self.scenario_count);
        for scenario_idx in 0..self.scenario_count {
            let mut scenario = point_forecast.to_vec();
            let mut residual_field = (0..horizon)
                .map(|t| {
                    (0..nodes)
                        .map(|node| {
                            self.shock_scale
                                * deterministic_standard_sample(
                                    scenario_idx * horizon + t,
                                    node + self.diffusion_steps,
                                )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for _ in 0..self.diffusion_steps {
                residual_field = diffuse_residual_field(&residual_field, edges, nodes);
            }
            for t in 0..horizon {
                for node in 0..nodes {
                    scenario[t][node] += residual_field[t][node];
                }
            }
            scenarios.push(scenario);
        }
        let scenario_mean = scenario_panel_mean(&scenarios, horizon, nodes);
        let scenario_variance = scenario_panel_variance(&scenarios, &scenario_mean, horizon, nodes);
        let spatial_correlation = scenario_spatial_correlation(&scenario_mean, edges);
        let mut point_forecast_comparison = BTreeMap::new();
        point_forecast_comparison.insert(
            "mean_absolute_delta".to_string(),
            mean_abs_panel_delta(&scenario_mean, point_forecast),
        );
        point_forecast_comparison.insert(
            "mean_variance".to_string(),
            scenario_variance.iter().flatten().sum::<f64>() / (horizon * nodes) as f64,
        );
        let mut metadata = self.metadata.clone();
        metadata.insert(
            "scenario_count".to_string(),
            self.scenario_count.to_string(),
        );
        metadata.insert(
            "diffusion_steps".to_string(),
            self.diffusion_steps.to_string(),
        );
        metadata.insert("shock_scale".to_string(), self.shock_scale.to_string());
        Ok(DiffusionScenarioPrediction {
            scenarios,
            scenario_mean,
            scenario_variance,
            spatial_correlation,
            point_forecast_comparison,
            metadata,
        })
    }
}

pub fn split_conformal_residual_quantile(
    actual: &[f64],
    prediction: &[f64],
    alpha: f64,
    order: SplitOrder,
) -> Result<f64> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    let mut residuals = actual
        .iter()
        .zip(prediction)
        .map(|(&y, &p)| (y - p).abs())
        .collect::<Vec<_>>();
    residuals.sort_by(f64::total_cmp);
    let rank = (((residuals.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    Ok(residuals[rank.saturating_sub(1).min(residuals.len() - 1)])
}

pub fn weighted_conformal_residual_quantile(
    actual: &[f64],
    prediction: &[f64],
    weights: &[f64],
    alpha: f64,
    order: SplitOrder,
) -> Result<f64> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    validate_same_non_empty(actual, weights, "actual", "weights")?;
    if weights.iter().any(|w| *w <= 0.0) {
        return invalid("weights must be positive");
    }
    let mut pairs = actual
        .iter()
        .zip(prediction)
        .zip(weights)
        .map(|((&y, &p), &w)| ((y - p).abs(), w))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_weight = pairs.iter().map(|(_, w)| *w).sum::<f64>();
    let threshold = (1.0 - alpha) * total_weight;
    let mut cumulative = 0.0;
    for (residual, weight) in pairs {
        cumulative += weight;
        if cumulative >= threshold {
            return Ok(residual);
        }
    }
    invalid("weighted conformal calibration failed")
}

pub fn group_conformal_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    groups: &[String],
    alpha: f64,
    order: SplitOrder,
) -> Result<BTreeMap<String, f64>> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    if groups.len() != actual.len() {
        return invalid("groups length must match actual length");
    }
    let mut grouped: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for ((&y, &p), group) in actual.iter().zip(prediction).zip(groups) {
        let entry = grouped.entry(group.clone()).or_default();
        entry.0.push(y);
        entry.1.push(p);
    }
    grouped
        .into_iter()
        .map(|(group, (y, p))| {
            split_conformal_residual_quantile(&y, &p, alpha, order).map(|q| (group, q))
        })
        .collect()
}

pub fn spatial_block_conformal_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    block_ids: &[String],
    alpha: f64,
    order: SplitOrder,
) -> Result<BTreeMap<String, f64>> {
    group_conformal_residual_quantiles(actual, prediction, block_ids, alpha, order)
}

#[allow(clippy::too_many_arguments)]
pub fn nearest_calibration_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    calibration_x: &[f64],
    calibration_y: &[f64],
    query_x: &[f64],
    query_y: &[f64],
    neighbor_count: usize,
    alpha: f64,
    order: SplitOrder,
) -> Result<Vec<f64>> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    validate_same_non_empty(
        calibration_x,
        calibration_y,
        "calibration_x",
        "calibration_y",
    )?;
    validate_same_non_empty(calibration_x, actual, "calibration_x", "actual")?;
    validate_same_non_empty(query_x, query_y, "query_x", "query_y")?;
    if neighbor_count == 0 {
        return invalid("neighbor_count must be positive");
    }
    let residuals = actual
        .iter()
        .zip(prediction)
        .map(|(&y, &p)| (y - p).abs())
        .collect::<Vec<_>>();
    query_x
        .iter()
        .zip(query_y)
        .map(|(&x, &y)| {
            let mut distances = calibration_x
                .iter()
                .zip(calibration_y)
                .zip(&residuals)
                .map(|((&cx, &cy), &residual)| {
                    let dx = cx - x;
                    let dy = cy - y;
                    (dx * dx + dy * dy, residual)
                })
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            let local = distances
                .iter()
                .take(neighbor_count.min(distances.len()))
                .map(|(_, residual)| *residual)
                .collect::<Vec<_>>();
            conformal_quantile(&local, alpha)
        })
        .collect()
}

pub fn rolling_origin_conformal_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    cutoffs: &[usize],
    alpha: f64,
) -> Result<Vec<f64>> {
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    if cutoffs.is_empty() {
        return invalid("cutoffs must contain at least one cutoff");
    }
    let mut result = Vec::with_capacity(cutoffs.len());
    for &cutoff in cutoffs {
        if cutoff == 0 || cutoff > actual.len() {
            return invalid("each cutoff must be inside the calibration history");
        }
        let order = SplitOrder {
            train_end_exclusive: 1,
            calibration_start: 1,
            calibration_end_exclusive: cutoff,
            test_start: cutoff,
        };
        result.push(split_conformal_residual_quantile(
            &actual[..cutoff],
            &prediction[..cutoff],
            alpha,
            order,
        )?);
    }
    Ok(result)
}

pub fn benchmark_calibration_report_fields(
    actual: &[f64],
    lower: &[f64],
    upper: &[f64],
    horizons: &[usize],
    spatial_blocks: &[String],
    residual_morans_i_after_calibration: Option<f64>,
) -> Result<BenchmarkCalibrationReportFields> {
    validate_same_non_empty(actual, lower, "actual", "lower")?;
    validate_same_non_empty(actual, upper, "actual", "upper")?;
    if horizons.len() != actual.len() {
        return invalid("horizons length must match actual length");
    }
    if spatial_blocks.len() != actual.len() {
        return invalid("spatial_blocks length must match actual length");
    }
    if let Some(value) = residual_morans_i_after_calibration {
        if !value.is_finite() {
            return invalid("residual_morans_i_after_calibration must be finite when provided");
        }
    }
    let mut coverage_by_horizon = BTreeMap::new();
    let mut width_by_horizon = BTreeMap::new();
    for horizon in horizons
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let mut y = Vec::new();
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for idx in 0..actual.len() {
            if horizons[idx] == horizon {
                y.push(actual[idx]);
                lo.push(lower[idx]);
                hi.push(upper[idx]);
            }
        }
        coverage_by_horizon.insert(horizon, interval_coverage(&y, &lo, &hi)?);
        width_by_horizon.insert(horizon, mean_interval_width(&lo, &hi)?);
    }
    let mut coverage_by_spatial_block = BTreeMap::new();
    for block in spatial_blocks
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let mut y = Vec::new();
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for idx in 0..actual.len() {
            if spatial_blocks[idx] == block {
                y.push(actual[idx]);
                lo.push(lower[idx]);
                hi.push(upper[idx]);
            }
        }
        coverage_by_spatial_block.insert(block, interval_coverage(&y, &lo, &hi)?);
    }
    Ok(BenchmarkCalibrationReportFields {
        coverage_by_horizon,
        coverage_by_spatial_block,
        width_by_horizon,
        residual_morans_i_after_calibration,
    })
}

fn validate_quantile_rows(
    actual: &[f64],
    quantiles: &[f64],
    predictions: &[Vec<f64>],
) -> Result<()> {
    if actual.is_empty() {
        return invalid("actual must contain at least one value");
    }
    if actual.len() != predictions.len() {
        return invalid("actual and predictions must have the same row count");
    }
    for &q in quantiles {
        validate_alpha(q)?;
    }
    for row in predictions {
        if row.len() != quantiles.len() {
            return invalid("each prediction row must match quantiles length");
        }
        validate_finite(row, "prediction row")?;
    }
    validate_finite(actual, "actual")?;
    Ok(())
}

fn conformal_quantile(residuals: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_finite(residuals, "residuals")?;
    if residuals.is_empty() {
        return invalid("residuals must contain at least one value");
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (((sorted.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    Ok(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

fn validate_same_non_empty(
    left: &[f64],
    right: &[f64],
    left_name: &str,
    right_name: &str,
) -> Result<()> {
    validate_finite(left, left_name)?;
    validate_finite(right, right_name)?;
    if left.len() != right.len() {
        return invalid(&format!(
            "{left_name} and {right_name} must have the same length"
        ));
    }
    if left.is_empty() {
        return invalid(&format!(
            "{left_name} and {right_name} must contain at least one value"
        ));
    }
    Ok(())
}

fn validate_same_non_empty_matrix(
    matrix: &[Vec<f64>],
    values: &[f64],
    matrix_name: &str,
    values_name: &str,
) -> Result<()> {
    if matrix.is_empty() || values.is_empty() {
        return invalid(&format!(
            "{matrix_name} and {values_name} must contain at least one row"
        ));
    }
    if matrix.len() != values.len() {
        return invalid(&format!(
            "{matrix_name} and {values_name} must have the same row count"
        ));
    }
    let cols = matrix[0].len();
    if cols == 0 || matrix.iter().any(|row| row.len() != cols) {
        return invalid(&format!("{matrix_name} must have a fixed positive width"));
    }
    for row in matrix {
        validate_finite(row, matrix_name)?;
    }
    validate_finite(values, values_name)?;
    Ok(())
}

fn validate_panel(panel: &[Vec<f64>], name: &str) -> Result<()> {
    if panel.is_empty() {
        return invalid(&format!("{name} must contain at least one horizon row"));
    }
    let cols = panel[0].len();
    if cols == 0 {
        return invalid(&format!("{name} must contain at least one node column"));
    }
    for row in panel {
        if row.len() != cols {
            return invalid(&format!("{name} must have a fixed node width"));
        }
        validate_finite(row, name)?;
    }
    Ok(())
}

fn validate_edges(edges: &[DiffusionEdge], node_count: usize) -> Result<()> {
    for edge in edges {
        if edge.source >= node_count || edge.target >= node_count {
            return invalid("edge source and target must reference point_forecast columns");
        }
        if !edge.weight.is_finite() {
            return invalid("edge weights must be finite");
        }
    }
    Ok(())
}

fn validate_quantile_grid(quantiles: &[f64]) -> Result<()> {
    if quantiles.is_empty() {
        return invalid("quantiles must contain at least one value");
    }
    for pair in quantiles.windows(2) {
        if pair[0] >= pair[1] {
            return invalid("quantiles must be strictly increasing");
        }
    }
    for &q in quantiles {
        validate_alpha(q)?;
    }
    Ok(())
}

fn ridge_fit(features: &[Vec<f64>], target: &[f64], ridge: f64) -> Vec<f64> {
    let cols = features[0].len() + 1;
    let mut xtx = vec![vec![0.0; cols]; cols];
    let mut xty = vec![0.0; cols];
    for (row, &y) in features.iter().zip(target) {
        let mut x = vec![1.0];
        x.extend_from_slice(row);
        for r in 0..cols {
            xty[r] += x[r] * y;
            for c in 0..cols {
                xtx[r][c] += x[r] * x[c];
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        row[idx] += ridge.max(1.0e-12);
    }
    solve_linear_system(xtx, xty)
}

fn predict_linear(features: &[Vec<f64>], weights: &[f64]) -> Result<Vec<f64>> {
    if weights.is_empty() {
        return invalid("linear weights must be non-empty");
    }
    let expected_width = weights.len() - 1;
    if features.iter().any(|row| row.len() != expected_width) {
        return invalid("feature width must match fitted flow artifact");
    }
    Ok(features
        .iter()
        .map(|row| {
            weights[0]
                + row
                    .iter()
                    .zip(&weights[1..])
                    .map(|(x, w)| x * w)
                    .sum::<f64>()
        })
        .collect())
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if best != pivot {
            matrix.swap(pivot, best);
            rhs.swap(pivot, best);
        }
        let diag = matrix[pivot][pivot];
        if diag.abs() < 1.0e-12 {
            continue;
        }
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= diag;
        }
        rhs[pivot] /= diag;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for col in pivot..n {
                matrix[row][col] -= factor * pivot_row[col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    rhs
}

fn deterministic_standard_sample(row: usize, sample: usize) -> f64 {
    let value = ((row as u64 + 1) * 1_103_515_245 + (sample as u64 + 17) * 12_345) % 10_000;
    (value as f64 / 10_000.0 - 0.5) * 2.0
}

fn diffuse_residual_field(
    residual_field: &[Vec<f64>],
    edges: &[DiffusionEdge],
    node_count: usize,
) -> Vec<Vec<f64>> {
    let mut next = residual_field.to_vec();
    let mut inbound_weight = vec![0.0; node_count];
    for edge in edges {
        inbound_weight[edge.target] += edge.weight.abs();
    }
    for (t, row) in residual_field.iter().enumerate() {
        for edge in edges {
            let denom = inbound_weight[edge.target].max(1.0);
            next[t][edge.target] += 0.5 * edge.weight * row[edge.source] / denom;
        }
    }
    next
}

fn scenario_panel_mean(scenarios: &[Vec<Vec<f64>>], horizon: usize, nodes: usize) -> Vec<Vec<f64>> {
    let mut mean = vec![vec![0.0; nodes]; horizon];
    for scenario in scenarios {
        for t in 0..horizon {
            for node in 0..nodes {
                mean[t][node] += scenario[t][node] / scenarios.len() as f64;
            }
        }
    }
    mean
}

fn scenario_panel_variance(
    scenarios: &[Vec<Vec<f64>>],
    mean: &[Vec<f64>],
    horizon: usize,
    nodes: usize,
) -> Vec<Vec<f64>> {
    let mut variance = vec![vec![0.0; nodes]; horizon];
    for scenario in scenarios {
        for t in 0..horizon {
            for node in 0..nodes {
                let delta = scenario[t][node] - mean[t][node];
                variance[t][node] += delta * delta / scenarios.len() as f64;
            }
        }
    }
    variance
}

fn mean_abs_panel_delta(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let count = a.iter().map(Vec::len).sum::<usize>().max(1);
    a.iter()
        .zip(b)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(&x, &y)| (x - y).abs())
                .sum::<f64>()
        })
        .sum::<f64>()
        / count as f64
}

fn scenario_spatial_correlation(panel: &[Vec<f64>], edges: &[DiffusionEdge]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in panel {
        let mean = row.iter().sum::<f64>() / row.len() as f64;
        let centered = row.iter().map(|value| value - mean).collect::<Vec<_>>();
        let variance = centered.iter().map(|value| value * value).sum::<f64>();
        if variance <= 1.0e-12 {
            continue;
        }
        for edge in edges {
            numerator += edge.weight * centered[edge.source] * centered[edge.target];
            denominator += edge.weight.abs() * variance;
        }
    }
    if denominator <= 1.0e-12 {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

fn normal_quantile_proxy(q: f64) -> f64 {
    // Smooth symmetric approximation sufficient for deterministic quantile surfaces.
    ((q / (1.0 - q)).ln() * 0.5513).clamp(-4.0, 4.0)
}

fn gaussian_log_likelihood(y: f64, loc: f64, scale: f64) -> f64 {
    let var = scale * scale;
    -0.5 * (((y - loc) * (y - loc)) / var + var.ln() + (2.0 * std::f64::consts::PI).ln())
}

fn quantile_mean(rows: &[Vec<f64>], idx: usize) -> Result<f64> {
    if rows.is_empty() || rows.iter().any(|row| idx >= row.len()) {
        return invalid("quantile index is out of bounds");
    }
    Ok(rows.iter().map(|row| row[idx]).sum::<f64>() / rows.len() as f64)
}

fn joint_path_calibration(actual: &[f64], paths: &[Vec<f64>]) -> f64 {
    if paths.is_empty() {
        return 0.0;
    }
    let actual_sum = actual.iter().sum::<f64>();
    let covered = paths
        .iter()
        .filter(|path| path.iter().sum::<f64>() >= actual_sum)
        .count();
    covered as f64 / paths.len() as f64
}

fn tail_event_calibration(actual: &[f64], upper: &[f64]) -> f64 {
    actual.iter().zip(upper).filter(|(y, hi)| y > hi).count() as f64 / actual.len() as f64
}

fn validate_finite(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return invalid(&format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> Result<()> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return invalid("alpha must be finite and in (0, 1)");
    }
    Ok(())
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(CartoBoostError::InvalidInput(message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_conformal_covers_synthetic_holdout_without_holdout_training() {
        let calibration_prediction = vec![10.0; 20];
        let calibration_actual = (0..20)
            .map(|idx| 10.0 + if idx % 2 == 0 { 1.0 } else { -1.0 } * (idx % 5) as f64)
            .collect::<Vec<_>>();
        let order = SplitOrder {
            train_end_exclusive: 10,
            calibration_start: 10,
            calibration_end_exclusive: 30,
            test_start: 30,
        };
        let q = split_conformal_residual_quantile(
            &calibration_actual,
            &calibration_prediction,
            0.1,
            order,
        )
        .unwrap();
        let lower = vec![10.0 - q; 10];
        let upper = vec![10.0 + q; 10];
        let actual = vec![9.0, 11.0, 10.0, 8.0, 12.0, 10.5, 9.5, 11.5, 8.5, 12.5];
        assert!(interval_coverage(&actual, &lower, &upper).unwrap() >= 0.9);
    }

    #[test]
    fn rolling_origin_uses_only_past_cutoff_residuals() {
        let actual = vec![10.0, 11.0, 14.0, 50.0];
        let prediction = vec![10.0, 10.0, 10.0, 10.0];
        let qs = rolling_origin_conformal_residual_quantiles(&actual, &prediction, &[2, 3], 0.1)
            .unwrap();
        assert_eq!(qs, vec![1.0, 4.0]);
    }

    #[test]
    fn distributional_metrics_validate_and_score() {
        let actual = vec![1.0, 2.0];
        let quantiles = vec![0.1, 0.5, 0.9];
        let predictions = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]];
        assert!(crps_approximation(&actual, &quantiles, &predictions).unwrap() >= 0.0);
        let pits = pit_bins(&actual, &quantiles, &predictions, 5).unwrap();
        assert_eq!(pits.counts.iter().sum::<usize>(), 2);
        let wis = weighted_interval_score(
            &actual,
            &[1.0, 2.0],
            &[(0.2, vec![0.0, 1.0], vec![2.0, 3.0])],
        )
        .unwrap();
        assert!(wis >= 0.0);
    }

    #[test]
    fn conditional_flow_head_emits_joint_distribution_outputs_and_metrics() {
        let hidden = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.5],
            vec![2.0, 0.0],
            vec![3.0, 1.0],
        ];
        let residuals = vec![-0.5, 0.1, 0.4, 0.9];
        let artifact =
            conditional_flow_fit_json(&hidden, &residuals, &[0.05, 0.5, 0.95], 8).unwrap();
        let output_json =
            conditional_flow_predict_json(&artifact, &hidden, Some(&residuals)).unwrap();
        let output: FlowPrediction = serde_json::from_str(&output_json).unwrap();

        assert_eq!(output.samples.len(), hidden.len());
        assert_eq!(output.samples[0].len(), 8);
        assert_eq!(output.marginal_quantiles[0].len(), 3);
        assert_eq!(output.joint_scenario_paths.len(), 8);
        assert_eq!(output.log_likelihood.len(), hidden.len());
        assert!(output
            .tail_risk_metrics
            .contains_key("expected_shortfall_low"));
        assert!(output.metrics.contains_key("crps"));
        assert!(output.metrics.contains_key("pinball_median"));
        assert!(output.metrics.contains_key("interval_coverage"));
        assert!(output.metrics.contains_key("joint_path_calibration"));
        assert!(output.metrics.contains_key("tail_event_calibration"));
    }

    #[test]
    fn diffusion_scenario_generator_reports_shape_variance_and_spatial_correlation() {
        let point_forecast = vec![
            vec![10.0, 12.0, 13.0],
            vec![11.0, 12.5, 14.0],
            vec![12.0, 13.0, 15.0],
        ];
        let edges = vec![
            DiffusionEdge {
                source: 0,
                target: 1,
                weight: 1.0,
            },
            DiffusionEdge {
                source: 1,
                target: 2,
                weight: 0.7,
            },
        ];
        let output_json =
            diffusion_scenario_generate_json(&point_forecast, &edges, 6, 2, 0.4).unwrap();
        let output: DiffusionScenarioPrediction = serde_json::from_str(&output_json).unwrap();

        assert_eq!(output.scenarios.len(), 6);
        assert_eq!(output.scenarios[0].len(), point_forecast.len());
        assert_eq!(output.scenarios[0][0].len(), point_forecast[0].len());
        assert_eq!(output.scenario_mean.len(), point_forecast.len());
        assert_eq!(output.scenario_variance[0].len(), point_forecast[0].len());
        assert!(output.spatial_correlation.is_finite());
        assert!(output
            .point_forecast_comparison
            .contains_key("mean_absolute_delta"));
        assert_eq!(
            output.metadata.get("capability_tier").map(String::as_str),
            Some("experimental")
        );
        assert_eq!(
            output.metadata.get("auto_geo_enabled").map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn benchmark_report_fields_group_coverage_by_horizon_and_block() {
        let report = benchmark_calibration_report_fields(
            &[10.0, 12.0, 20.0, 25.0],
            &[9.0, 11.0, 15.0, 24.0],
            &[11.0, 13.0, 22.0, 24.5],
            &[1, 1, 2, 2],
            &[
                "pickup_142".into(),
                "pickup_142".into(),
                "pickup_236".into(),
                "pickup_236".into(),
            ],
            Some(0.05),
        )
        .unwrap();
        assert_eq!(report.coverage_by_horizon[&1], 1.0);
        assert_eq!(report.coverage_by_spatial_block["pickup_236"], 0.5);
        assert_eq!(report.residual_morans_i_after_calibration, Some(0.05));
    }

    #[test]
    fn nearest_calibration_residuals_use_local_neighbors() {
        let q = nearest_calibration_residual_quantiles(
            &[10.0, 20.0, 100.0],
            &[9.0, 18.0, 90.0],
            &[0.0, 1.0, 100.0],
            &[0.0, 1.0, 100.0],
            &[0.1, 99.0],
            &[0.1, 99.0],
            1,
            0.1,
            SplitOrder {
                train_end_exclusive: 1,
                calibration_start: 1,
                calibration_end_exclusive: 4,
                test_start: 4,
            },
        )
        .unwrap();
        assert_eq!(q, vec![1.0, 10.0]);
    }
}
