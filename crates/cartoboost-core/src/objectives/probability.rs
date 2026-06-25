use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1.0e-15;

pub trait ProbabilityCalibrator {
    fn fit(scores: &[f64], labels: &[f64]) -> Result<Self>
    where
        Self: Sized;
    fn predict_one(&self, score: f64) -> f64;
    fn predict(&self, scores: &[f64]) -> Result<Vec<f64>> {
        validate_finite(scores, "scores")?;
        Ok(scores
            .iter()
            .map(|score| self.predict_one(*score))
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SigmoidCalibrator {
    pub slope: f64,
    pub intercept: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemperatureCalibrator {
    pub temperature: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    pub thresholds: Vec<f64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThresholdEvent {
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FailureRiskEvent {
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EscalationRiskEvent {
    pub warning_threshold: f64,
    pub critical_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBucket {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub mean_probability: f64,
    pub event_rate: f64,
    pub absolute_gap: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub brier_score: f64,
    pub log_loss: f64,
    pub expected_calibration_error: f64,
    pub buckets: Vec<CalibrationBucket>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationImprovement {
    pub before: CalibrationMetrics,
    pub after: CalibrationMetrics,
    pub brier_score_reduction: f64,
    pub log_loss_reduction: f64,
    pub expected_calibration_error_reduction: f64,
}

impl ProbabilityCalibrator for SigmoidCalibrator {
    fn fit(scores: &[f64], labels: &[f64]) -> Result<Self> {
        validate_scores_labels(scores, labels)?;
        let mut slope = 1.0;
        let prior = labels.iter().sum::<f64>() / labels.len() as f64;
        let mut intercept = logit(prior.clamp(1.0e-6, 1.0 - 1.0e-6));
        let ridge = 1.0e-6;
        for _ in 0..64 {
            let mut grad_slope = ridge * (slope - 1.0);
            let mut grad_intercept = ridge * intercept;
            let mut h00 = ridge;
            let mut h01 = 0.0;
            let mut h11 = ridge;
            for (&score, &label) in scores.iter().zip(labels) {
                let z = slope * score + intercept;
                let p = sigmoid(z);
                let grad = p - label;
                let h = (p * (1.0 - p)).max(1.0e-12);
                grad_slope += grad * score;
                grad_intercept += grad;
                h00 += h * score * score;
                h01 += h * score;
                h11 += h;
            }
            let det = h00 * h11 - h01 * h01;
            if det.abs() <= 1.0e-18 {
                break;
            }
            let step_slope = (h11 * grad_slope - h01 * grad_intercept) / det;
            let step_intercept = (-h01 * grad_slope + h00 * grad_intercept) / det;
            slope -= step_slope.clamp(-4.0, 4.0);
            intercept -= step_intercept.clamp(-4.0, 4.0);
            if step_slope.abs() + step_intercept.abs() < 1.0e-10 {
                break;
            }
        }
        Ok(Self { slope, intercept })
    }

    fn predict_one(&self, score: f64) -> f64 {
        sigmoid(self.slope * score + self.intercept)
    }
}

impl ProbabilityCalibrator for TemperatureCalibrator {
    fn fit(scores: &[f64], labels: &[f64]) -> Result<Self> {
        validate_scores_labels(scores, labels)?;
        let mut log_temperature = 0.0_f64;
        for _ in 0..80 {
            let temperature = log_temperature.exp().clamp(1.0e-3, 1.0e3);
            let mut grad = 0.0;
            let mut hess = 1.0e-8;
            for (&score, &label) in scores.iter().zip(labels) {
                let z = score / temperature;
                let p = sigmoid(z);
                let dz = -z;
                let d2z = z;
                grad += (p - label) * dz;
                hess += (p * (1.0 - p)) * dz * dz + (p - label) * d2z;
            }
            let step = (grad / hess.max(1.0e-8)).clamp(-1.0, 1.0);
            log_temperature -= step;
            log_temperature = log_temperature.clamp(-6.0, 6.0);
            if step.abs() < 1.0e-10 {
                break;
            }
        }
        Ok(Self {
            temperature: log_temperature.exp().clamp(1.0e-3, 1.0e3),
        })
    }

    fn predict_one(&self, score: f64) -> f64 {
        sigmoid(score / self.temperature.max(1.0e-12))
    }
}

impl ProbabilityCalibrator for IsotonicCalibrator {
    fn fit(scores: &[f64], labels: &[f64]) -> Result<Self> {
        validate_scores_labels(scores, labels)?;
        let probabilities = scores
            .iter()
            .map(|score| probability_from_score(*score))
            .collect::<Vec<_>>();
        let mut rows = probabilities
            .iter()
            .copied()
            .zip(labels.iter().copied())
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| left.0.total_cmp(&right.0));
        let sorted_scores = rows.iter().map(|(score, _)| *score).collect::<Vec<_>>();
        let sorted_labels = rows.iter().map(|(_, label)| *label).collect::<Vec<_>>();
        let values = pool_adjacent_violators(&sorted_labels)?;
        Ok(Self {
            thresholds: sorted_scores,
            values,
        })
    }

    fn predict_one(&self, score: f64) -> f64 {
        if self.thresholds.is_empty() {
            return probability_from_score(score);
        }
        let probability = probability_from_score(score);
        match self
            .thresholds
            .binary_search_by(|threshold| threshold.total_cmp(&probability))
        {
            Ok(idx) => self.values[idx].clamp(0.0, 1.0),
            Err(0) => self.values[0].clamp(0.0, 1.0),
            Err(idx) if idx >= self.values.len() => {
                self.values[self.values.len() - 1].clamp(0.0, 1.0)
            }
            Err(idx) => {
                let lo_x = self.thresholds[idx - 1];
                let hi_x = self.thresholds[idx];
                let lo_y = self.values[idx - 1];
                let hi_y = self.values[idx];
                if (hi_x - lo_x).abs() <= 1.0e-15 {
                    hi_y.clamp(0.0, 1.0)
                } else {
                    let t = ((probability - lo_x) / (hi_x - lo_x)).clamp(0.0, 1.0);
                    (lo_y + t * (hi_y - lo_y)).clamp(0.0, 1.0)
                }
            }
        }
    }
}

impl ThresholdEvent {
    pub fn labels(&self, actual: &[f64]) -> Result<Vec<f64>> {
        validate_finite(actual, "actual")?;
        Ok(actual
            .iter()
            .map(|value| (*value <= self.threshold) as u8 as f64)
            .collect())
    }
}

impl FailureRiskEvent {
    pub fn labels(&self, actual: &[f64]) -> Result<Vec<f64>> {
        validate_finite(actual, "actual")?;
        Ok(actual
            .iter()
            .map(|value| (*value > self.threshold) as u8 as f64)
            .collect())
    }
}

impl EscalationRiskEvent {
    pub fn labels(&self, actual: &[f64]) -> Result<Vec<f64>> {
        validate_finite(actual, "actual")?;
        if self.warning_threshold >= self.critical_threshold {
            return Err(CartoBoostError::InvalidInput(
                "warning_threshold must be less than critical_threshold".to_string(),
            ));
        }
        Ok(actual
            .iter()
            .map(|value| (*value >= self.critical_threshold) as u8 as f64)
            .collect())
    }
}

pub fn success_within_threshold(
    actual: &[f64],
    prediction: &[f64],
    threshold: f64,
) -> Result<Vec<f64>> {
    validate_paired(actual, prediction, "actual", "prediction")?;
    validate_nonnegative_finite(threshold, "threshold")?;
    Ok(actual
        .iter()
        .zip(prediction)
        .map(|(&a, &p)| ((a - p).abs() <= threshold) as u8 as f64)
        .collect())
}

pub fn event_within_horizon(event_times: &[f64], horizon: f64) -> Result<Vec<f64>> {
    validate_finite(event_times, "event_times")?;
    validate_nonnegative_finite(horizon, "horizon")?;
    Ok(event_times
        .iter()
        .map(|time| (*time >= 0.0 && *time <= horizon) as u8 as f64)
        .collect())
}

pub fn failure_risk_event(values: &[f64], threshold: f64) -> Result<Vec<f64>> {
    FailureRiskEvent { threshold }.labels(values)
}

pub fn escalation_risk_event(
    values: &[f64],
    warning_threshold: f64,
    critical_threshold: f64,
) -> Result<Vec<f64>> {
    EscalationRiskEvent {
        warning_threshold,
        critical_threshold,
    }
    .labels(values)
}

pub fn calibration_metrics(
    labels: &[f64],
    probabilities: &[f64],
    bucket_count: usize,
) -> Result<CalibrationMetrics> {
    validate_probabilities_labels(probabilities, labels)?;
    if bucket_count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "bucket_count must be positive".to_string(),
        ));
    }
    let n = labels.len() as f64;
    let brier_score = labels
        .iter()
        .zip(probabilities)
        .map(|(&label, &probability)| (probability - label).powi(2))
        .sum::<f64>()
        / n;
    let log_loss = labels
        .iter()
        .zip(probabilities)
        .map(|(&label, &probability)| {
            let p = probability.clamp(EPSILON, 1.0 - EPSILON);
            -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / n;
    let mut counts = vec![0_usize; bucket_count];
    let mut prob_sums = vec![0.0; bucket_count];
    let mut label_sums = vec![0.0; bucket_count];
    for (&label, &probability) in labels.iter().zip(probabilities) {
        let idx = ((probability.clamp(0.0, 1.0 - f64::EPSILON) * bucket_count as f64).floor()
            as usize)
            .min(bucket_count - 1);
        counts[idx] += 1;
        prob_sums[idx] += probability;
        label_sums[idx] += label;
    }
    let mut expected_calibration_error = 0.0;
    let mut buckets = Vec::with_capacity(bucket_count);
    for idx in 0..bucket_count {
        let count = counts[idx];
        let lower = idx as f64 / bucket_count as f64;
        let upper = (idx + 1) as f64 / bucket_count as f64;
        let (mean_probability, event_rate, absolute_gap) = if count == 0 {
            (0.5 * (lower + upper), 0.0, 0.0)
        } else {
            let mean_probability = prob_sums[idx] / count as f64;
            let event_rate = label_sums[idx] / count as f64;
            let absolute_gap = (mean_probability - event_rate).abs();
            expected_calibration_error += count as f64 / n * absolute_gap;
            (mean_probability, event_rate, absolute_gap)
        };
        buckets.push(CalibrationBucket {
            lower,
            upper,
            count,
            mean_probability,
            event_rate,
            absolute_gap,
        });
    }
    Ok(CalibrationMetrics {
        brier_score,
        log_loss,
        expected_calibration_error,
        buckets,
    })
}

pub fn calibration_improvement(
    labels: &[f64],
    before_probabilities: &[f64],
    after_probabilities: &[f64],
    bucket_count: usize,
) -> Result<CalibrationImprovement> {
    let before = calibration_metrics(labels, before_probabilities, bucket_count)?;
    let after = calibration_metrics(labels, after_probabilities, bucket_count)?;
    Ok(CalibrationImprovement {
        brier_score_reduction: before.brier_score - after.brier_score,
        log_loss_reduction: before.log_loss - after.log_loss,
        expected_calibration_error_reduction: before.expected_calibration_error
            - after.expected_calibration_error,
        before,
        after,
    })
}

pub fn pool_adjacent_violators(values: &[f64]) -> Result<Vec<f64>> {
    validate_binary_labels(values)?;
    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        weight: f64,
        value: f64,
    }
    let mut blocks: Vec<Block> = Vec::new();
    for (idx, &value) in values.iter().enumerate() {
        blocks.push(Block {
            start: idx,
            end: idx + 1,
            weight: 1.0,
            value,
        });
        while blocks.len() >= 2 {
            let last = blocks[blocks.len() - 1];
            let prev = blocks[blocks.len() - 2];
            if prev.value <= last.value {
                break;
            }
            let merged_weight = prev.weight + last.weight;
            let merged_value =
                (prev.value * prev.weight + last.value * last.weight) / merged_weight;
            let merged = Block {
                start: prev.start,
                end: last.end,
                weight: merged_weight,
                value: merged_value,
            };
            blocks.pop();
            blocks.pop();
            blocks.push(merged);
        }
    }
    let mut fitted = vec![0.0; values.len()];
    for block in blocks {
        for value in &mut fitted[block.start..block.end] {
            *value = block.value.clamp(0.0, 1.0);
        }
    }
    Ok(fitted)
}

fn probability_from_score(score: f64) -> f64 {
    if (0.0..=1.0).contains(&score) {
        score
    } else {
        sigmoid(score)
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

fn validate_scores_labels(scores: &[f64], labels: &[f64]) -> Result<()> {
    validate_paired(scores, labels, "scores", "labels")?;
    validate_binary_labels(labels)
}

fn validate_probabilities_labels(probabilities: &[f64], labels: &[f64]) -> Result<()> {
    validate_paired(probabilities, labels, "probabilities", "labels")?;
    validate_binary_labels(labels)?;
    if probabilities
        .iter()
        .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must be finite and in [0, 1]".to_string(),
        ));
    }
    Ok(())
}

fn validate_paired(left: &[f64], right: &[f64], left_name: &str, right_name: &str) -> Result<()> {
    validate_finite(left, left_name)?;
    validate_finite(right, right_name)?;
    if left.len() != right.len() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{left_name} and {right_name} must have the same length"
        )));
    }
    if left.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{left_name} and {right_name} must contain at least one value"
        )));
    }
    Ok(())
}

fn validate_finite(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_binary_labels(labels: &[f64]) -> Result<()> {
    validate_finite(labels, "labels")?;
    if labels.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "labels must contain at least one value".to_string(),
        ));
    }
    if labels
        .iter()
        .any(|label| (*label - 0.0).abs() > 1.0e-12 && (*label - 1.0).abs() > 1.0e-12)
    {
        return Err(CartoBoostError::InvalidInput(
            "labels must be binary values encoded as 0 or 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_nonnegative_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotonic_uses_pool_adjacent_violators() {
        let fitted = pool_adjacent_violators(&[0.0, 1.0, 0.0, 1.0]).unwrap();

        assert_eq!(fitted, vec![0.0, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn calibrators_keep_probabilities_bounded() {
        let scores = [-4.0, -1.0, 0.0, 1.0, 4.0];
        let labels = [0.0, 0.0, 0.0, 1.0, 1.0];

        let sigmoid = SigmoidCalibrator::fit(&scores, &labels).unwrap();
        let temperature = TemperatureCalibrator::fit(&scores, &labels).unwrap();
        let isotonic = IsotonicCalibrator::fit(&scores, &labels).unwrap();

        for probability in sigmoid
            .predict(&scores)
            .unwrap()
            .into_iter()
            .chain(temperature.predict(&scores).unwrap())
            .chain(isotonic.predict(&scores).unwrap())
        {
            assert!((0.0..=1.0).contains(&probability));
        }
    }

    #[test]
    fn calibration_metrics_report_reliability_curve() {
        let labels = [0.0, 0.0, 1.0, 1.0];
        let probabilities = [0.1, 0.4, 0.6, 0.9];

        let metrics = calibration_metrics(&labels, &probabilities, 2).unwrap();

        assert_eq!(metrics.buckets.len(), 2);
        assert_eq!(metrics.buckets[0].count, 2);
        assert_eq!(metrics.buckets[1].count, 2);
        assert!(metrics.brier_score < 0.17);
        assert!(metrics.log_loss < 0.6);
        assert!(metrics.expected_calibration_error >= 0.0);
    }

    #[test]
    fn isotonic_calibration_improves_distorted_synthetic_reliability() {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for i in 0..200 {
            let base = (i as f64 + 0.5) / 200.0;
            let distorted = (0.15 + 0.7 * base).clamp(0.0, 1.0);
            scores.push(distorted);
            labels.push((base > 0.5) as u8 as f64);
        }
        let before = calibration_metrics(&labels, &scores, 10)
            .unwrap()
            .expected_calibration_error;
        let calibrator = IsotonicCalibrator::fit(&scores, &labels).unwrap();
        let calibrated = calibrator.predict(&scores).unwrap();
        let after = calibration_metrics(&labels, &calibrated, 10)
            .unwrap()
            .expected_calibration_error;

        assert!(after < before);
    }

    #[test]
    fn calibration_improvement_reports_brier_logloss_and_ece_reductions() {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for i in 0..240 {
            let base = (i as f64 + 0.5) / 240.0;
            let distorted = (0.2 + 0.6 * base).clamp(0.0, 1.0);
            scores.push(distorted);
            labels.push((base >= 0.5) as u8 as f64);
        }
        let calibrator = IsotonicCalibrator::fit(&scores, &labels).unwrap();
        let calibrated = calibrator.predict(&scores).unwrap();
        let improvement = calibration_improvement(&labels, &scores, &calibrated, 12).unwrap();

        assert!(improvement.brier_score_reduction > 0.0);
        assert!(improvement.log_loss_reduction > 0.0);
        assert!(improvement.expected_calibration_error_reduction > 0.0);
        assert!(improvement.after.brier_score < improvement.before.brier_score);
        assert!(improvement.after.log_loss < improvement.before.log_loss);
        assert!(
            improvement.after.expected_calibration_error
                < improvement.before.expected_calibration_error
        );
    }

    #[test]
    fn probability_calibrators_round_trip_exactly() {
        let scores = [-3.0, -1.0, 0.0, 1.0, 3.0];
        let labels = [0.0, 0.0, 0.0, 1.0, 1.0];

        let sigmoid = SigmoidCalibrator::fit(&scores, &labels).unwrap();
        let temperature = TemperatureCalibrator::fit(&scores, &labels).unwrap();
        let isotonic = IsotonicCalibrator::fit(&scores, &labels).unwrap();

        let sigmoid_round_trip: SigmoidCalibrator =
            serde_json::from_str(&serde_json::to_string(&sigmoid).unwrap()).unwrap();
        let temperature_round_trip: TemperatureCalibrator =
            serde_json::from_str(&serde_json::to_string(&temperature).unwrap()).unwrap();
        let isotonic_round_trip: IsotonicCalibrator =
            serde_json::from_str(&serde_json::to_string(&isotonic).unwrap()).unwrap();

        assert!((sigmoid_round_trip.slope - sigmoid.slope).abs() <= 1.0e-12);
        assert!((sigmoid_round_trip.intercept - sigmoid.intercept).abs() <= 1.0e-12);
        assert!((temperature_round_trip.temperature - temperature.temperature).abs() <= 1.0e-12);
        assert_eq!(isotonic_round_trip, isotonic);
        assert_probabilities_close(
            &sigmoid_round_trip.predict(&scores).unwrap(),
            &sigmoid.predict(&scores).unwrap(),
        );
        assert_probabilities_close(
            &temperature_round_trip.predict(&scores).unwrap(),
            &temperature.predict(&scores).unwrap(),
        );
        assert_eq!(
            isotonic_round_trip.predict(&scores).unwrap(),
            isotonic.predict(&scores).unwrap()
        );
    }

    #[test]
    fn event_helpers_compile_threshold_and_horizon_labels() {
        assert_eq!(
            success_within_threshold(&[10.0, 14.0], &[11.0, 10.0], 2.0).unwrap(),
            vec![1.0, 0.0]
        );
        assert_eq!(
            event_within_horizon(&[0.0, 3.0, 5.0], 3.0).unwrap(),
            vec![1.0, 1.0, 0.0]
        );
        assert_eq!(
            failure_risk_event(&[0.1, 0.9], 0.5).unwrap(),
            vec![0.0, 1.0]
        );
        assert_eq!(
            escalation_risk_event(&[0.1, 0.6, 0.9], 0.5, 0.8).unwrap(),
            vec![0.0, 0.0, 1.0]
        );
        assert!(escalation_risk_event(&[0.1], 1.0, 0.8).is_err());
    }

    fn assert_probabilities_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-12);
        }
    }
}
