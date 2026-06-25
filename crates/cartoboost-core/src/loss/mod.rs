mod l2;
mod quantile;

pub use l2::{huber_irls_weights, HuberLoss, HuberLossConfig, L2Loss, LogL2LossConfig};
pub use quantile::{
    absolute_loss, huber_quantile_loss, pinball_loss, weighted_absolute_loss,
    weighted_pinball_loss, weighted_quantile, CompositeQuantileLoss, CompositeQuantileLossConfig,
    HuberQuantileLoss, HuberQuantileLossConfig, L1Loss, QuantileLoss, QuantileLossConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum LossConfig {
    #[default]
    L2,
    L1,
    Huber(HuberLossConfig),
    LogL2(LogL2LossConfig),
    Quantile(QuantileLossConfig),
}

pub trait Loss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64;
    fn gradient(&self, y: f64, pred: f64) -> f64;
    fn hessian(&self, _y: f64, _pred: f64) -> f64 {
        1.0
    }
    fn leaf_value(&self, grad_sum: f64, hess_sum: f64, lambda_l2: f64) -> f64 {
        -grad_sum / (hess_sum + lambda_l2)
    }
}
