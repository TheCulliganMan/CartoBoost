mod hurdle;
mod negative_binomial;
mod poisson;
mod probability;
mod tweedie;

use crate::loss::{weighted_quantile, HuberLoss, L1Loss, L2Loss, Loss, LossConfig};
use crate::{CartoBoostError, Result};

pub use hurdle::{HurdleDerivatives, HurdleObjective};
pub use negative_binomial::NegativeBinomialObjective;
pub use poisson::PoissonObjective;
pub use probability::{
    calibration_improvement, calibration_metrics, escalation_risk_event, event_within_horizon,
    failure_risk_event, pool_adjacent_violators, success_within_threshold, CalibrationBucket,
    CalibrationImprovement, CalibrationMetrics, EscalationRiskEvent, FailureRiskEvent,
    IsotonicCalibrator, ProbabilityCalibrator, SigmoidCalibrator, TemperatureCalibrator,
    ThresholdEvent,
};
pub use tweedie::TweedieObjective;

/// Logistic boosting objective for binary probability forecasting.
pub type LogisticBoostingObjective = BinaryLogLossObjective;

/// Binary probability booster using CartoBoost's Rust-native classifier.
pub type ProbabilityBooster = crate::booster::Classifier;

const PROBABILITY_EPSILON: f64 = 1.0e-15;
const LOGIT_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveDerivatives {
    pub value: f64,
    pub gradient: f64,
    pub hessian: f64,
}

pub trait CountObjective {
    fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives>;
}

/// Broad modeling family served by an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveTask {
    /// Continuous target regression.
    Regression,
    /// Two-class classification with one raw margin per row.
    BinaryClassification,
    /// Multiclass classification with row-major class margins.
    MulticlassClassification,
    /// Grouped relevance ranking.
    Ranking,
}

/// Transform applied to raw model margins before user-facing prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionTransformKind {
    /// Return raw margins unchanged.
    Identity,
    /// Apply a logistic sigmoid to each raw margin.
    Sigmoid,
    /// Apply row-wise softmax across the objective output dimension.
    Softmax,
}

/// First- and second-order derivative pair for one objective output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientPair {
    /// Objective derivative with respect to the raw margin.
    pub gradient: f64,
    /// Non-negative second derivative approximation.
    pub hessian: f64,
}

/// Named scalar metric produced by an objective.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricValue {
    /// Stable metric identifier.
    pub name: &'static str,
    /// Metric value where lower or higher direction depends on metric semantics.
    pub value: f64,
}

/// Shared interface for CartoBoost training objectives.
///
/// Implementations define how raw margins are initialized, how gradients and
/// Hessians are computed, how leaf updates are derived, how predictions are
/// transformed, and which default metric represents the objective.
pub trait Objective {
    /// Stable objective identifier.
    fn name(&self) -> &'static str;
    /// Modeling task family for this objective.
    fn task(&self) -> ObjectiveTask;
    /// Number of raw outputs per training row.
    fn output_dimension(&self) -> usize {
        1
    }
    /// User-facing prediction transform for raw margins.
    fn prediction_transform(&self) -> PredictionTransformKind {
        PredictionTransformKind::Identity
    }
    /// Initial raw margin vector before any trees are fitted.
    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>>;
    /// Per-output gradients and Hessians for the current raw predictions.
    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>>;
    /// Newton-style constant leaf update from summed derivatives.
    fn leaf_value(&self, grad_sum: f64, hess_sum: f64, lambda_l2: f64) -> f64 {
        if hess_sum + lambda_l2 <= 0.0 {
            0.0
        } else {
            -grad_sum / (hess_sum + lambda_l2)
        }
    }
    /// Transform raw margins into the objective's prediction space.
    fn transform_predictions(&self, raw_predictions: &[f64]) -> Result<Vec<f64>> {
        transform_predictions(
            self.prediction_transform(),
            raw_predictions,
            self.output_dimension(),
        )
    }
    /// Compute the objective's default evaluation metric.
    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        groups: Option<&[usize]>,
    ) -> Result<MetricValue>;
    /// Convert gradients into pseudo-residual targets for first-order tree fitting.
    fn pseudo_residuals(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        groups: Option<&[usize]>,
    ) -> Result<Vec<f64>> {
        Ok(self
            .gradients_hessians(targets, raw_predictions, weights, groups)?
            .into_iter()
            .map(|pair| -pair.gradient)
            .collect())
    }
}

/// Squared-error regression objective.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct L2Objective;

impl Objective for L2Objective {
    fn name(&self) -> &'static str {
        "l2"
    }

    fn task(&self) -> ObjectiveTask {
        ObjectiveTask::Regression
    }

    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        validate_targets(targets)?;
        validate_objective_weights(weights, targets.len())?;
        Ok(vec![L2Loss.initial_prediction(targets, weights)])
    }

    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        _groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>> {
        validate_prediction_shape(targets, raw_predictions, 1)?;
        validate_objective_weights(weights, targets.len())?;
        Ok(targets
            .iter()
            .zip(raw_predictions)
            .enumerate()
            .map(|(idx, (target, prediction))| {
                let weight = weight_at(weights, idx);
                GradientPair {
                    gradient: weight * (prediction - target),
                    hessian: weight,
                }
            })
            .collect())
    }

    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        _groups: Option<&[usize]>,
    ) -> Result<MetricValue> {
        validate_prediction_shape(targets, raw_predictions, 1)?;
        Ok(MetricValue {
            name: "rmse",
            value: mean_squared_error(targets, raw_predictions).sqrt(),
        })
    }
}

/// Pinball-loss regression objective for a requested quantile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantileObjective {
    /// Target quantile in the open interval `(0, 1)`.
    pub alpha: f64,
}

impl QuantileObjective {
    pub fn new(alpha: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(CartoBoostError::InvalidInput(
                "quantile alpha must be finite and in (0, 1)".to_string(),
            ));
        }
        Ok(Self { alpha })
    }
}

impl Objective for QuantileObjective {
    fn name(&self) -> &'static str {
        "quantile"
    }

    fn task(&self) -> ObjectiveTask {
        ObjectiveTask::Regression
    }

    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        validate_targets(targets)?;
        validate_objective_weights(weights, targets.len())?;
        let unit_weights;
        let weights = match weights {
            Some(weights) => weights,
            None => {
                unit_weights = vec![1.0; targets.len()];
                &unit_weights
            }
        };
        Ok(vec![weighted_quantile(targets, weights, self.alpha)])
    }

    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        _groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>> {
        validate_prediction_shape(targets, raw_predictions, 1)?;
        validate_objective_weights(weights, targets.len())?;
        Ok(targets
            .iter()
            .zip(raw_predictions)
            .enumerate()
            .map(|(idx, (target, prediction))| {
                let gradient = if target > prediction {
                    -self.alpha
                } else {
                    1.0 - self.alpha
                };
                let weight = weight_at(weights, idx);
                GradientPair {
                    gradient: weight * gradient,
                    hessian: weight,
                }
            })
            .collect())
    }

    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        _groups: Option<&[usize]>,
    ) -> Result<MetricValue> {
        validate_prediction_shape(targets, raw_predictions, 1)?;
        Ok(MetricValue {
            name: "pinball_loss",
            value: targets
                .iter()
                .zip(raw_predictions)
                .map(|(target, prediction)| {
                    let residual = target - prediction;
                    if residual >= 0.0 {
                        self.alpha * residual
                    } else {
                        (self.alpha - 1.0) * residual
                    }
                })
                .sum::<f64>()
                / targets.len().max(1) as f64,
        })
    }
}

/// Binary logistic loss objective for labels encoded as `0.0` or `1.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BinaryLogLossObjective;

impl Objective for BinaryLogLossObjective {
    fn name(&self) -> &'static str {
        "binary_logloss"
    }

    fn task(&self) -> ObjectiveTask {
        ObjectiveTask::BinaryClassification
    }

    fn prediction_transform(&self) -> PredictionTransformKind {
        PredictionTransformKind::Sigmoid
    }

    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        validate_binary_targets(targets)?;
        validate_objective_weights(weights, targets.len())?;
        let positive_rate =
            weighted_mean(targets, weights).clamp(LOGIT_EPSILON, 1.0 - LOGIT_EPSILON);
        Ok(vec![(positive_rate / (1.0 - positive_rate)).ln()])
    }

    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        _groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>> {
        validate_binary_targets(targets)?;
        validate_prediction_shape(targets, raw_predictions, 1)?;
        validate_objective_weights(weights, targets.len())?;
        Ok(targets
            .iter()
            .zip(raw_predictions)
            .enumerate()
            .map(|(idx, (target, raw_prediction))| {
                let probability = sigmoid(*raw_prediction);
                let weight = weight_at(weights, idx);
                GradientPair {
                    gradient: weight * (probability - target),
                    hessian: weight * probability * (1.0 - probability),
                }
            })
            .collect())
    }

    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        _groups: Option<&[usize]>,
    ) -> Result<MetricValue> {
        validate_binary_targets(targets)?;
        validate_prediction_shape(targets, raw_predictions, 1)?;
        let probabilities = self.transform_predictions(raw_predictions)?;
        Ok(MetricValue {
            name: "logloss",
            value: binary_logloss(targets, &probabilities),
        })
    }
}

/// Multiclass logistic loss objective for integer class-id targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MulticlassLogLossObjective {
    /// Number of classes represented in each row-major raw prediction block.
    pub class_count: usize,
}

impl MulticlassLogLossObjective {
    pub fn new(class_count: usize) -> Result<Self> {
        if class_count < 2 {
            return Err(CartoBoostError::InvalidInput(
                "multiclass_logloss requires at least two classes".to_string(),
            ));
        }
        Ok(Self { class_count })
    }
}

impl Objective for MulticlassLogLossObjective {
    fn name(&self) -> &'static str {
        "multiclass_logloss"
    }

    fn task(&self) -> ObjectiveTask {
        ObjectiveTask::MulticlassClassification
    }

    fn output_dimension(&self) -> usize {
        self.class_count
    }

    fn prediction_transform(&self) -> PredictionTransformKind {
        PredictionTransformKind::Softmax
    }

    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        validate_multiclass_targets(targets, self.class_count)?;
        validate_objective_weights(weights, targets.len())?;
        let mut counts = vec![0.0; self.class_count];
        let mut total = 0.0;
        for (idx, target) in targets.iter().enumerate() {
            let class = *target as usize;
            let weight = weight_at(weights, idx);
            counts[class] += weight;
            total += weight;
        }
        if total <= 0.0 {
            return Ok(vec![0.0; self.class_count]);
        }
        Ok(counts
            .into_iter()
            .map(|count| (count.max(LOGIT_EPSILON) / total).ln())
            .collect())
    }

    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        _groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>> {
        validate_multiclass_targets(targets, self.class_count)?;
        validate_prediction_shape(targets, raw_predictions, self.class_count)?;
        validate_objective_weights(weights, targets.len())?;
        let mut pairs = Vec::with_capacity(raw_predictions.len());
        if let Some(weights) = weights {
            for ((row, target), weight) in raw_predictions
                .chunks_exact(self.class_count)
                .zip(targets)
                .zip(weights)
            {
                let class = *target as usize;
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_sum = 0.0;
                for value in row {
                    exp_sum += (value - max).exp();
                }
                let inv_sum = 1.0 / exp_sum;
                for (output, value) in row.iter().enumerate() {
                    let probability = (value - max).exp() * inv_sum;
                    let label = if output == class { 1.0 } else { 0.0 };
                    pairs.push(GradientPair {
                        gradient: weight * (probability - label),
                        hessian: weight * probability * (1.0 - probability),
                    });
                }
            }
        } else {
            for (row, target) in raw_predictions.chunks_exact(self.class_count).zip(targets) {
                let class = *target as usize;
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_sum = 0.0;
                for value in row {
                    exp_sum += (value - max).exp();
                }
                let inv_sum = 1.0 / exp_sum;
                for (output, value) in row.iter().enumerate() {
                    let probability = (value - max).exp() * inv_sum;
                    let label = if output == class { 1.0 } else { 0.0 };
                    pairs.push(GradientPair {
                        gradient: probability - label,
                        hessian: probability * (1.0 - probability),
                    });
                }
            }
        }
        Ok(pairs)
    }

    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        _groups: Option<&[usize]>,
    ) -> Result<MetricValue> {
        validate_multiclass_targets(targets, self.class_count)?;
        validate_prediction_shape(targets, raw_predictions, self.class_count)?;
        let loss = targets
            .iter()
            .zip(raw_predictions.chunks_exact(self.class_count))
            .map(|(target, row)| {
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_sum = 0.0;
                for value in row {
                    exp_sum += (value - max).exp();
                }
                let probability = ((row[*target as usize] - max).exp() / exp_sum)
                    .clamp(PROBABILITY_EPSILON, 1.0)
                    .ln();
                -probability
            })
            .sum::<f64>()
            / targets.len().max(1) as f64;
        Ok(MetricValue {
            name: "logloss",
            value: loss,
        })
    }
}

/// Pairwise logistic ranking objective over query/document groups.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PairwiseLogitObjective;

impl Objective for PairwiseLogitObjective {
    fn name(&self) -> &'static str {
        "pairwise_logit"
    }

    fn task(&self) -> ObjectiveTask {
        ObjectiveTask::Ranking
    }

    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        validate_targets(targets)?;
        validate_objective_weights(weights, targets.len())?;
        Ok(vec![0.0])
    }

    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>> {
        pairwise_gradients(targets, raw_predictions, weights, groups, false)
    }

    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        groups: Option<&[usize]>,
    ) -> Result<MetricValue> {
        Ok(MetricValue {
            name: "ndcg",
            value: mean_ndcg(targets, raw_predictions, groups, None)?,
        })
    }
}

/// Pairwise ranking objective with NDCG-delta weighted gradients.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LambdaRankObjective;

impl Objective for LambdaRankObjective {
    fn name(&self) -> &'static str {
        "lambdarank"
    }

    fn task(&self) -> ObjectiveTask {
        ObjectiveTask::Ranking
    }

    fn initial_margin(&self, targets: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        validate_targets(targets)?;
        validate_objective_weights(weights, targets.len())?;
        Ok(vec![0.0])
    }

    fn gradients_hessians(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        weights: Option<&[f64]>,
        groups: Option<&[usize]>,
    ) -> Result<Vec<GradientPair>> {
        pairwise_gradients(targets, raw_predictions, weights, groups, true)
    }

    fn default_metric(
        &self,
        targets: &[f64],
        raw_predictions: &[f64],
        groups: Option<&[usize]>,
    ) -> Result<MetricValue> {
        Ok(MetricValue {
            name: "ndcg",
            value: mean_ndcg(targets, raw_predictions, groups, None)?,
        })
    }
}

pub(crate) fn initial_margin_for_loss(
    loss: &LossConfig,
    targets: &[f64],
    weights: Option<&[f64]>,
) -> Result<f64> {
    validate_targets(targets)?;
    let value = match loss {
        LossConfig::L2 | LossConfig::LogL2(_) => L2Objective.initial_margin(targets, weights)?[0],
        LossConfig::L1 => L1Loss.initial_prediction(targets, weights),
        LossConfig::Huber(config) => {
            HuberLoss::new(config.delta).initial_prediction(targets, weights)
        }
        LossConfig::Quantile(config) => {
            QuantileObjective::new(config.alpha)?.initial_margin(targets, weights)?[0]
        }
    };
    Ok(value)
}

pub(crate) fn pseudo_residual_for_loss(loss: &LossConfig, target: f64, prediction: f64) -> f64 {
    match loss {
        LossConfig::L2 | LossConfig::LogL2(_) => target - prediction,
        LossConfig::L1 => target - prediction,
        LossConfig::Huber(config) => -HuberLoss::new(config.delta).gradient(target, prediction),
        LossConfig::Quantile(config) => {
            if target > prediction {
                config.alpha
            } else {
                config.alpha - 1.0
            }
        }
    }
}

pub(crate) fn validate_non_negative_target(target: f64) -> Result<()> {
    if !target.is_finite() || target < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "count objective target must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn finite_exp(raw_prediction: f64) -> Result<f64> {
    if !raw_prediction.is_finite() {
        return Err(crate::CartoBoostError::InvalidInput(
            "count objective raw prediction must be finite".to_string(),
        ));
    }
    let clipped = raw_prediction.clamp(-35.0, 35.0);
    Ok(clipped.exp())
}

pub(crate) fn validate_derivatives(
    value: f64,
    gradient: f64,
    hessian: f64,
) -> Result<ObjectiveDerivatives> {
    if !value.is_finite() || !gradient.is_finite() || !hessian.is_finite() || hessian < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "count objective derivatives must be finite with non-negative hessian".to_string(),
        ));
    }
    Ok(ObjectiveDerivatives {
        value,
        gradient,
        hessian,
    })
}

fn validate_targets(targets: &[f64]) -> Result<()> {
    if targets.iter().any(|target| !target.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "objective targets must be finite".to_string(),
        ));
    }
    Ok(())
}

fn validate_binary_targets(targets: &[f64]) -> Result<()> {
    validate_targets(targets)?;
    if targets
        .iter()
        .any(|target| (*target - 0.0).abs() > 1e-12 && (*target - 1.0).abs() > 1e-12)
    {
        return Err(CartoBoostError::InvalidInput(
            "binary_logloss targets must be 0 or 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_multiclass_targets(targets: &[f64], class_count: usize) -> Result<()> {
    validate_targets(targets)?;
    if targets
        .iter()
        .any(|target| target.fract() != 0.0 || *target < 0.0 || *target as usize >= class_count)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "multiclass_logloss targets must be integer class ids in [0, {class_count})"
        )));
    }
    Ok(())
}

fn validate_prediction_shape(
    targets: &[f64],
    raw_predictions: &[f64],
    output_dimension: usize,
) -> Result<()> {
    if output_dimension == 0 || raw_predictions.len() != targets.len() * output_dimension {
        return Err(CartoBoostError::InvalidInput(format!(
            "raw prediction length {} does not match {} targets with output dimension {}",
            raw_predictions.len(),
            targets.len(),
            output_dimension
        )));
    }
    if raw_predictions
        .iter()
        .any(|prediction| !prediction.is_finite())
    {
        return Err(CartoBoostError::InvalidInput(
            "objective raw predictions must be finite".to_string(),
        ));
    }
    Ok(())
}

fn weight_at(weights: Option<&[f64]>, index: usize) -> f64 {
    weights.map_or(1.0, |weights| weights[index])
}

fn validate_objective_weights(weights: Option<&[f64]>, row_count: usize) -> Result<()> {
    let Some(weights) = weights else {
        return Ok(());
    };
    if weights.len() != row_count {
        return Err(CartoBoostError::InvalidInput(format!(
            "objective weight length {} does not match {row_count} targets",
            weights.len()
        )));
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(CartoBoostError::InvalidInput(
            "objective weights must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn weighted_mean(values: &[f64], weights: Option<&[f64]>) -> f64 {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (idx, value) in values.iter().enumerate() {
        let weight = weight_at(weights, idx);
        weighted_sum += weight * value;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        0.0
    } else {
        weighted_sum / weight_sum
    }
}

fn sigmoid(raw_prediction: f64) -> f64 {
    if raw_prediction >= 0.0 {
        1.0 / (1.0 + (-raw_prediction).exp())
    } else {
        let exp_value = raw_prediction.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn transform_predictions(
    transform: PredictionTransformKind,
    raw_predictions: &[f64],
    output_dimension: usize,
) -> Result<Vec<f64>> {
    match transform {
        PredictionTransformKind::Identity => Ok(raw_predictions.to_vec()),
        PredictionTransformKind::Sigmoid => Ok(raw_predictions
            .iter()
            .map(|raw_prediction| sigmoid(*raw_prediction))
            .collect()),
        PredictionTransformKind::Softmax => {
            if output_dimension < 2 || !raw_predictions.len().is_multiple_of(output_dimension) {
                return Err(CartoBoostError::InvalidInput(
                    "softmax predictions must be a row-major matrix with at least two outputs"
                        .to_string(),
                ));
            }
            let mut probabilities = Vec::with_capacity(raw_predictions.len());
            for row in raw_predictions.chunks_exact(output_dimension) {
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let start = probabilities.len();
                let mut total = 0.0;
                for value in row {
                    let exp_value = (value - max).exp();
                    probabilities.push(exp_value);
                    total += exp_value;
                }
                let inv_total = 1.0 / total;
                for probability in &mut probabilities[start..] {
                    *probability *= inv_total;
                }
            }
            Ok(probabilities)
        }
    }
}

fn mean_squared_error(targets: &[f64], predictions: &[f64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| (target - prediction).powi(2))
        .sum::<f64>()
        / targets.len() as f64
}

fn binary_logloss(targets: &[f64], probabilities: &[f64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    targets
        .iter()
        .zip(probabilities)
        .map(|(target, probability)| {
            let probability = probability.clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON);
            -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln())
        })
        .sum::<f64>()
        / targets.len() as f64
}

fn pairwise_gradients(
    targets: &[f64],
    raw_predictions: &[f64],
    weights: Option<&[f64]>,
    groups: Option<&[usize]>,
    scale_by_ndcg_delta: bool,
) -> Result<Vec<GradientPair>> {
    pairwise_gradients_with_rhos(
        targets,
        raw_predictions,
        weights,
        groups,
        scale_by_ndcg_delta,
        None,
    )
}

pub(crate) fn pairwise_gradients_with_accelerated_rhos(
    targets: &[f64],
    raw_predictions: &[f64],
    weights: Option<&[f64]>,
    groups: Option<&[usize]>,
    scale_by_ndcg_delta: bool,
    rhos: &[f64],
) -> Result<Vec<GradientPair>> {
    pairwise_gradients_with_rhos(
        targets,
        raw_predictions,
        weights,
        groups,
        scale_by_ndcg_delta,
        Some(rhos),
    )
}

fn pairwise_gradients_with_rhos(
    targets: &[f64],
    raw_predictions: &[f64],
    weights: Option<&[f64]>,
    groups: Option<&[usize]>,
    scale_by_ndcg_delta: bool,
    accelerated_rhos: Option<&[f64]>,
) -> Result<Vec<GradientPair>> {
    validate_targets(targets)?;
    validate_prediction_shape(targets, raw_predictions, 1)?;
    validate_objective_weights(weights, targets.len())?;
    let ranges = group_ranges(groups, targets.len())?;
    let mut pairs = vec![
        GradientPair {
            gradient: 0.0,
            hessian: 0.0,
        };
        targets.len()
    ];
    let mut position_by_row = if scale_by_ndcg_delta {
        vec![0usize; targets.len()]
    } else {
        Vec::new()
    };
    let mut rho_index = 0usize;
    for range in ranges {
        if range.len() < 2 {
            continue;
        }
        let ideal_dcg = if scale_by_ndcg_delta {
            dcg_for_sorted_targets(targets, range.clone(), None)
        } else {
            0.0
        };
        if scale_by_ndcg_delta {
            let mut ranks = (range.clone()).collect::<Vec<_>>();
            ranks.sort_by(|left, right| raw_predictions[*right].total_cmp(&raw_predictions[*left]));
            for (position, row) in ranks.iter().enumerate() {
                position_by_row[*row] = position;
            }
        }
        for high in range.clone() {
            let high_target = targets[high];
            let high_prediction = raw_predictions[high];
            let high_weight = weights.map_or(1.0, |values| values[high]);
            for low in range.clone() {
                let low_target = targets[low];
                if high_target <= low_target {
                    continue;
                }
                let score_delta = high_prediction - raw_predictions[low];
                let rho = if let Some(rhos) = accelerated_rhos {
                    let value = rhos.get(rho_index).copied().ok_or_else(|| {
                        CartoBoostError::InvalidInput(
                            "accelerated pair probabilities are shorter than ranking pairs"
                                .to_string(),
                        )
                    })?;
                    rho_index += 1;
                    value
                } else {
                    sigmoid(-score_delta)
                };
                let pair_weight = if let Some(values) = weights {
                    0.5 * (high_weight + values[low])
                } else {
                    1.0
                };
                let ndcg_weight = if scale_by_ndcg_delta && ideal_dcg > 0.0 {
                    ndcg_swap_delta(
                        high_target,
                        low_target,
                        position_by_row[high],
                        position_by_row[low],
                    ) / ideal_dcg
                } else {
                    1.0
                };
                let lambda = pair_weight * ndcg_weight * rho;
                let hessian = pair_weight * ndcg_weight * rho * (1.0 - rho);
                pairs[high].gradient -= lambda;
                pairs[low].gradient += lambda;
                pairs[high].hessian += hessian;
                pairs[low].hessian += hessian;
            }
        }
    }
    if accelerated_rhos.is_some_and(|rhos| rho_index != rhos.len()) {
        return Err(CartoBoostError::InvalidInput(
            "accelerated pair probabilities do not match ranking pairs".to_string(),
        ));
    }
    Ok(pairs)
}

fn group_ranges(groups: Option<&[usize]>, row_count: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let Some(groups) = groups else {
        return Ok(std::iter::once(0..row_count).collect());
    };
    let mut start = 0usize;
    let mut ranges = Vec::with_capacity(groups.len());
    for &size in groups {
        if size == 0 {
            return Err(CartoBoostError::InvalidInput(
                "ranking groups must be positive sizes".to_string(),
            ));
        }
        let end = start + size;
        if end > row_count {
            return Err(CartoBoostError::InvalidInput(
                "ranking group sizes exceed target length".to_string(),
            ));
        }
        ranges.push(start..end);
        start = end;
    }
    if start != row_count {
        return Err(CartoBoostError::InvalidInput(
            "ranking group sizes must sum to target length".to_string(),
        ));
    }
    Ok(ranges)
}

fn mean_ndcg(
    targets: &[f64],
    raw_predictions: &[f64],
    groups: Option<&[usize]>,
    k: Option<usize>,
) -> Result<f64> {
    validate_targets(targets)?;
    validate_prediction_shape(targets, raw_predictions, 1)?;
    let ranges = group_ranges(groups, targets.len())?;
    if ranges.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0;
    for range in &ranges {
        let limit = k.unwrap_or(range.len()).min(range.len());
        let mut predicted = range.clone().collect::<Vec<_>>();
        predicted.sort_by(|left, right| raw_predictions[*right].total_cmp(&raw_predictions[*left]));
        let dcg = predicted
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(position, row)| gain(targets[row]) * discount(position))
            .sum::<f64>();
        let ideal = dcg_for_sorted_targets(targets, range.clone(), Some(limit));
        total += if ideal > 0.0 { dcg / ideal } else { 0.0 };
    }
    Ok(total / ranges.len() as f64)
}

fn dcg_for_sorted_targets(
    targets: &[f64],
    range: std::ops::Range<usize>,
    limit: Option<usize>,
) -> f64 {
    let limit = limit.unwrap_or(range.len()).min(range.len());
    let mut rows = range.collect::<Vec<_>>();
    rows.sort_by(|left, right| targets[*right].total_cmp(&targets[*left]));
    rows.into_iter()
        .take(limit)
        .enumerate()
        .map(|(position, row)| gain(targets[row]) * discount(position))
        .sum()
}

fn ndcg_swap_delta(
    high_target: f64,
    low_target: f64,
    high_position: usize,
    low_position: usize,
) -> f64 {
    ((gain(high_target) - gain(low_target)) * (discount(high_position) - discount(low_position)))
        .abs()
}

fn gain(label: f64) -> f64 {
    2.0_f64.powf(label.max(0.0)) - 1.0
}

fn discount(position: usize) -> f64 {
    1.0 / ((position + 2) as f64).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_objective_initializes_weighted_mean_and_residuals() {
        let objective = L2Objective;
        let targets = [0.0, 10.0];

        assert_eq!(
            objective
                .initial_margin(&targets, Some(&[3.0, 1.0]))
                .unwrap(),
            vec![2.5]
        );
        assert_eq!(
            objective
                .pseudo_residuals(&targets, &[2.5, 2.5], None, None)
                .unwrap(),
            vec![-2.5, 7.5]
        );
    }

    #[test]
    fn quantile_objective_uses_weighted_quantile() {
        let objective = QuantileObjective::new(0.75).unwrap();

        assert_eq!(
            objective
                .initial_margin(&[0.0, 10.0, 20.0], Some(&[10.0, 1.0, 1.0]))
                .unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn quantile_loss_pseudo_residual_uses_pinball_gradient() {
        let loss = LossConfig::Quantile(crate::loss::QuantileLossConfig { alpha: 0.8 });

        assert!((pseudo_residual_for_loss(&loss, 10.0, 5.0) - 0.8).abs() < 1e-12);
        assert!((pseudo_residual_for_loss(&loss, 5.0, 10.0) + 0.2).abs() < 1e-12);
        assert!((pseudo_residual_for_loss(&loss, 5.0, 5.0) + 0.2).abs() < 1e-12);
    }

    #[test]
    fn binary_logloss_objective_exposes_sigmoid_probabilities() {
        let objective = BinaryLogLossObjective;
        let probabilities = objective.transform_predictions(&[0.0, 2.0]).unwrap();

        assert!((probabilities[0] - 0.5).abs() < 1e-12);
        assert!(probabilities[1] > 0.88 && probabilities[1] < 0.89);
        assert!(objective
            .default_metric(&[0.0, 1.0], &[0.0, 2.0], None)
            .unwrap()
            .value
            .is_finite());
    }

    #[test]
    fn multiclass_logloss_objective_uses_row_major_softmax() {
        let objective = MulticlassLogLossObjective::new(3).unwrap();
        let probabilities = objective
            .transform_predictions(&[2.0, 1.0, 0.0, 0.0, 1.0, 2.0])
            .unwrap();

        assert_eq!(probabilities.len(), 6);
        assert!((probabilities[..3].iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[5] > probabilities[4]);
    }

    #[test]
    fn pairwise_logit_prefers_higher_labels_with_higher_scores() {
        let objective = PairwiseLogitObjective;
        let gradients = objective
            .gradients_hessians(&[2.0, 0.0], &[0.0, 0.0], None, Some(&[2]))
            .unwrap();

        assert!(gradients[0].gradient < 0.0);
        assert!(gradients[1].gradient > 0.0);
        assert!(gradients.iter().all(|pair| pair.hessian > 0.0));
    }

    #[test]
    fn lambdarank_scales_pairwise_gradients_by_ndcg_delta() {
        let objective = LambdaRankObjective;
        let gradients = objective
            .gradients_hessians(&[3.0, 2.0, 0.0], &[0.0, 0.0, 0.0], None, Some(&[3]))
            .unwrap();

        assert!(gradients[0].gradient < 0.0);
        assert!(gradients[2].gradient > 0.0);
        assert!(
            objective
                .default_metric(&[3.0, 2.0, 0.0], &[3.0, 2.0, 0.0], Some(&[3]))
                .unwrap()
                .value
                > 0.999
        );
    }

    #[test]
    fn ndcg_at_k_uses_top_k_ideal_dcg() {
        let targets = [0.0, 3.0, 2.0];
        let scores = [0.0, 3.0, 2.0];

        assert!((mean_ndcg(&targets, &scores, Some(&[3]), Some(1)).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ranking_gain_treats_negative_relevance_as_zero() {
        let targets = [-2.0, 1.0];
        let good_scores = [0.0, 1.0];
        let bad_scores = [1.0, 0.0];

        assert_eq!(gain(-2.0), 0.0);
        assert!(mean_ndcg(&targets, &good_scores, Some(&[2]), None).unwrap() > 0.999);
        assert_eq!(
            mean_ndcg(&targets, &bad_scores, Some(&[2]), Some(1)).unwrap(),
            0.0
        );
    }

    #[test]
    fn objectives_reject_mismatched_or_invalid_weights() {
        let binary = BinaryLogLossObjective;

        assert!(binary
            .initial_margin(&[0.0, 1.0], Some(&[1.0]))
            .unwrap_err()
            .to_string()
            .contains("weight length"));
        assert!(binary
            .gradients_hessians(&[0.0, 1.0], &[0.0, 0.0], Some(&[1.0, f64::NAN]), None,)
            .unwrap_err()
            .to_string()
            .contains("finite and non-negative"));

        let ranker = PairwiseLogitObjective;
        assert!(ranker
            .initial_margin(&[0.0, 1.0], Some(&[1.0]))
            .unwrap_err()
            .to_string()
            .contains("weight length"));
    }
}
