use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StandardScaler {
    mean: f64,
    scale: f64,
}

impl StandardScaler {
    pub fn fit(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "scaler requires at least one value".to_string(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(NeuralError::InvalidArgument(
                "scaler values must be finite".to_string(),
            ));
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
        let scale = variance.sqrt().max(1e-12);
        Ok(Self { mean, scale })
    }

    pub fn transform(&self, value: f64) -> f64 {
        (value - self.mean) / self.scale
    }

    pub fn inverse_transform(&self, value: f64) -> f64 {
        value * self.scale + self.mean
    }

    pub fn transform_slice(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|value| self.transform(*value)).collect()
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}
