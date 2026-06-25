use super::Loss;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantileLossConfig {
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HuberQuantileLossConfig {
    pub alpha: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeQuantileLossConfig {
    pub quantiles: Vec<f64>,
    #[serde(default)]
    pub weights: Vec<f64>,
}

impl QuantileLossConfig {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl HuberQuantileLossConfig {
    pub fn new(alpha: f64, delta: f64) -> Self {
        Self { alpha, delta }
    }
}

impl CompositeQuantileLossConfig {
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self {
            quantiles,
            weights: Vec::new(),
        }
    }

    pub fn with_weights(quantiles: Vec<f64>, weights: Vec<f64>) -> Self {
        Self { quantiles, weights }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct HuberQuantileLoss {
    pub alpha: f64,
    pub delta: f64,
}

#[derive(Debug, Clone)]
pub struct CompositeQuantileLoss {
    pub quantiles: Vec<f64>,
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct L1Loss;

impl QuantileLoss {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn value(&self, y: f64, pred: f64) -> f64 {
        pinball_loss(y, pred, self.alpha)
    }
}

impl HuberQuantileLoss {
    pub fn new(alpha: f64, delta: f64) -> Self {
        Self { alpha, delta }
    }

    pub fn value(&self, y: f64, pred: f64) -> f64 {
        huber_quantile_loss(y, pred, self.alpha, self.delta)
    }
}

impl CompositeQuantileLoss {
    pub fn new(quantiles: Vec<f64>, weights: Vec<f64>) -> Self {
        Self { quantiles, weights }
    }

    pub fn from_config(config: CompositeQuantileLossConfig) -> Self {
        Self {
            quantiles: config.quantiles,
            weights: config.weights,
        }
    }

    pub fn value(&self, y: f64, predictions: &[f64]) -> f64 {
        let weights = self.resolved_weights();
        let total_weight = weights.iter().sum::<f64>().max(1.0e-12);
        self.quantiles
            .iter()
            .zip(predictions)
            .zip(weights.iter())
            .map(|((&alpha, &prediction), &weight)| weight * pinball_loss(y, prediction, alpha))
            .sum::<f64>()
            / total_weight
    }

    pub fn values(&self, actual: &[f64], predictions: &[Vec<f64>]) -> Vec<f64> {
        actual
            .iter()
            .zip(predictions)
            .map(|(&y, row)| self.value(y, row))
            .collect()
    }

    fn resolved_weights(&self) -> Vec<f64> {
        if self.weights.len() == self.quantiles.len()
            && self
                .weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0)
        {
            self.weights.clone()
        } else {
            vec![1.0; self.quantiles.len()]
        }
    }
}

impl Loss for L1Loss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64 {
        let unit_weights;
        let weights = match w {
            Some(weights) => weights,
            None => {
                unit_weights = vec![1.0; y.len()];
                &unit_weights
            }
        };
        weighted_quantile(y, weights, 0.5)
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        if y > pred {
            -1.0
        } else {
            1.0
        }
    }
}

impl Loss for QuantileLoss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64 {
        let unit_weights;
        let weights = match w {
            Some(weights) => weights,
            None => {
                unit_weights = vec![1.0; y.len()];
                &unit_weights
            }
        };
        weighted_quantile(y, weights, self.alpha)
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        if y > pred {
            -self.alpha
        } else {
            1.0 - self.alpha
        }
    }
}

impl Loss for HuberQuantileLoss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64 {
        let unit_weights;
        let weights = match w {
            Some(weights) => weights,
            None => {
                unit_weights = vec![1.0; y.len()];
                &unit_weights
            }
        };
        weighted_quantile(y, weights, self.alpha)
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        let residual = y - pred;
        let delta = self.delta.max(1.0e-12);
        if residual >= delta {
            -self.alpha
        } else if residual > 0.0 {
            -self.alpha * residual / delta
        } else if residual <= -delta {
            1.0 - self.alpha
        } else {
            -(1.0 - self.alpha) * residual / delta
        }
    }

    fn hessian(&self, y: f64, pred: f64) -> f64 {
        let residual = y - pred;
        let delta = self.delta.max(1.0e-12);
        if residual > 0.0 && residual < delta {
            self.alpha / delta
        } else if residual < 0.0 && residual > -delta {
            (1.0 - self.alpha) / delta
        } else {
            0.0
        }
    }
}

pub fn pinball_loss(value: f64, prediction: f64, alpha: f64) -> f64 {
    let residual = value - prediction;
    if residual >= 0.0 {
        alpha * residual
    } else {
        (alpha - 1.0) * residual
    }
}

pub fn huber_quantile_loss(value: f64, prediction: f64, alpha: f64, delta: f64) -> f64 {
    let residual = value - prediction;
    let delta = delta.max(1.0e-12);
    if residual >= delta {
        alpha * (residual - 0.5 * delta)
    } else if residual >= 0.0 {
        0.5 * alpha * residual * residual / delta
    } else if residual <= -delta {
        (1.0 - alpha) * (-residual - 0.5 * delta)
    } else {
        0.5 * (1.0 - alpha) * residual * residual / delta
    }
}

pub fn absolute_loss(value: f64, prediction: f64) -> f64 {
    (value - prediction).abs()
}

pub fn weighted_absolute_loss(values: &[f64], weights: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let selected_values = indices.iter().map(|&idx| values[idx]).collect::<Vec<_>>();
    let selected_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
    let prediction = weighted_quantile(&selected_values, &selected_weights, 0.5);
    indices
        .iter()
        .map(|&idx| weights[idx] * absolute_loss(values[idx], prediction))
        .sum()
}

pub fn weighted_pinball_loss(
    values: &[f64],
    weights: &[f64],
    indices: &[usize],
    alpha: f64,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let selected_values = indices.iter().map(|&idx| values[idx]).collect::<Vec<_>>();
    let selected_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
    let prediction = weighted_quantile(&selected_values, &selected_weights, alpha);
    indices
        .iter()
        .map(|&idx| weights[idx] * pinball_loss(values[idx], prediction, alpha))
        .sum()
}

pub fn weighted_quantile(values: &[f64], weights: &[f64], alpha: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut pairs = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(value, weight)| value.is_finite() && weight.is_finite() && *weight > 0.0)
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight = pairs.iter().map(|(_, weight)| *weight).sum::<f64>();
    let threshold = alpha.clamp(0.0, 1.0) * total_weight;
    let mut cumulative = 0.0;
    for (value, weight) in pairs {
        cumulative += weight;
        if cumulative >= threshold {
            return value;
        }
    }
    values[values.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_weighted_quantile() {
        let loss = QuantileLoss::new(0.75);

        assert_eq!(loss.initial_prediction(&[0.0, 10.0, 20.0], None), 20.0);
        assert_eq!(
            loss.initial_prediction(&[0.0, 10.0, 20.0], Some(&[10.0, 1.0, 1.0])),
            0.0
        );
    }

    #[test]
    fn pinball_loss_is_asymmetric() {
        assert_eq!(pinball_loss(10.0, 8.0, 0.8), 1.6);
        assert_eq!(pinball_loss(8.0, 10.0, 0.8), 0.3999999999999999);
    }

    #[test]
    fn huber_quantile_loss_is_smooth_near_zero_and_asymmetric_in_tails() {
        let loss = HuberQuantileLoss::new(0.8, 2.0);

        assert!((loss.value(11.0, 10.0) - 0.2).abs() < 1.0e-12);
        assert!((loss.value(14.0, 10.0) - 2.4).abs() < 1.0e-12);
        assert!((loss.gradient(11.0, 10.0) + 0.4).abs() < 1.0e-12);
        assert!((loss.gradient(8.0, 10.0) - 0.2).abs() < 1.0e-12);
        assert!((loss.hessian(11.0, 10.0) - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn composite_quantile_loss_averages_requested_quantiles() {
        let loss = CompositeQuantileLoss::new(vec![0.1, 0.5, 0.9], Vec::new());

        let value = loss.value(10.0, &[8.0, 9.0, 12.0]);

        assert!((value - ((0.2 + 0.5 + 0.2) / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn l1_initial_prediction_is_weighted_median() {
        let loss = L1Loss;

        assert_eq!(loss.initial_prediction(&[0.0, 10.0, 20.0], None), 10.0);
        assert_eq!(
            loss.initial_prediction(&[0.0, 10.0, 20.0], Some(&[10.0, 1.0, 1.0])),
            0.0
        );
    }
}
