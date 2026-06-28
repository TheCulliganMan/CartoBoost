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
