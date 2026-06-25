#![allow(clippy::items_after_test_module)]

use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_QUANTILE_LEVELS: [f64; 5] = [0.10, 0.25, 0.50, 0.75, 0.90];

#[derive(Debug, Clone, PartialEq)]
pub struct QuantileForecast {
    pub quantiles: Vec<f64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntervalDiagnostics {
    pub coverage: f64,
    pub mean_width: f64,
    pub crossing_rate: f64,
}

impl QuantileForecast {
    pub fn new(quantiles: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        validate_finite_values(&values, "values")?;
        if quantiles.len() != values.len() {
            return Err(CartoBoostError::InvalidInput(
                "quantiles and values must have the same length".to_string(),
            ));
        }
        Ok(Self { quantiles, values })
    }

    pub fn repaired(quantiles: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        let values = repair_non_crossing_quantiles(&values)?;
        Self::new(quantiles, values)
    }
}

pub fn default_quantile_levels() -> Vec<f64> {
    DEFAULT_QUANTILE_LEVELS.to_vec()
}

pub fn pinball_loss(actual: &[f64], prediction: &[f64], quantile: f64) -> Result<f64> {
    validate_quantile(quantile)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    Ok(actual
        .iter()
        .zip(prediction)
        .map(|(&y, &q)| {
            let residual = y - q;
            (quantile * residual).max((quantile - 1.0) * residual)
        })
        .sum::<f64>()
        / actual.len() as f64)
}

pub fn mean_interval_width(lower: &[f64], upper: &[f64]) -> Result<f64> {
    validate_same_non_empty(lower, upper, "lower", "upper")?;
    if lower.iter().zip(upper).any(|(&lo, &hi)| lo > hi) {
        return Err(CartoBoostError::InvalidInput(
            "lower bounds must be less than or equal to upper bounds".to_string(),
        ));
    }
    Ok(lower
        .iter()
        .zip(upper)
        .map(|(&lo, &hi)| hi - lo)
        .sum::<f64>()
        / lower.len() as f64)
}

pub fn interval_coverage(actual: &[f64], lower: &[f64], upper: &[f64]) -> Result<f64> {
    validate_same_non_empty(actual, lower, "actual", "lower")?;
    validate_same_non_empty(actual, upper, "actual", "upper")?;
    if lower.iter().zip(upper).any(|(&lo, &hi)| lo > hi) {
        return Err(CartoBoostError::InvalidInput(
            "lower bounds must be less than or equal to upper bounds".to_string(),
        ));
    }
    Ok(actual
        .iter()
        .zip(lower)
        .zip(upper)
        .filter(|((&value, &lo), &hi)| value >= lo && value <= hi)
        .count() as f64
        / actual.len() as f64)
}

pub fn crossing_rate(quantile_rows: &[Vec<f64>]) -> Result<f64> {
    if quantile_rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "quantile_rows must contain at least one row".to_string(),
        ));
    }
    let mut crossing_rows = 0usize;
    for row in quantile_rows {
        validate_finite_values(row, "quantile row")?;
        if row.windows(2).any(|window| window[1] < window[0]) {
            crossing_rows += 1;
        }
    }
    Ok(crossing_rows as f64 / quantile_rows.len() as f64)
}

pub fn interval_diagnostics(
    actual: &[f64],
    lower: &[f64],
    upper: &[f64],
    quantile_rows: &[Vec<f64>],
) -> Result<IntervalDiagnostics> {
    Ok(IntervalDiagnostics {
        coverage: interval_coverage(actual, lower, upper)?,
        mean_width: mean_interval_width(lower, upper)?,
        crossing_rate: crossing_rate(quantile_rows)?,
    })
}

pub fn repair_non_crossing_quantiles(values: &[f64]) -> Result<Vec<f64>> {
    validate_finite_values(values, "values")?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "values must contain at least one quantile prediction".to_string(),
        ));
    }
    let mut repaired = values.to_vec();
    for idx in 1..repaired.len() {
        if repaired[idx] < repaired[idx - 1] {
            repaired[idx] = repaired[idx - 1];
        }
    }
    Ok(repaired)
}

pub(crate) fn validate_quantile_grid(quantiles: &[f64]) -> Result<()> {
    if quantiles.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "quantiles must contain at least one level".to_string(),
        ));
    }
    let mut previous = f64::NEG_INFINITY;
    for &q in quantiles {
        validate_quantile(q)?;
        if q <= previous {
            return Err(CartoBoostError::InvalidInput(
                "quantiles must be strictly increasing".to_string(),
            ));
        }
        previous = q;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grid_matches_requested_quantile_regressors() {
        assert_eq!(
            default_quantile_levels(),
            vec![0.10, 0.25, 0.50, 0.75, 0.90]
        );
    }

    #[test]
    fn interval_diagnostics_measure_coverage_width_and_crossing() {
        let actual = [10.0, 20.0, 30.0, 40.0];
        let lower = [9.0, 18.0, 29.0, 35.0];
        let upper = [11.0, 19.0, 35.0, 50.0];
        let quantile_rows = vec![
            vec![9.0, 10.0, 11.0],
            vec![18.0, 20.0, 19.0],
            vec![29.0, 31.0, 35.0],
            vec![35.0, 42.0, 50.0],
        ];

        let diagnostics =
            interval_diagnostics(&actual, &lower, &upper, &quantile_rows).expect("diagnostics");

        assert_eq!(diagnostics.coverage, 0.75);
        assert_eq!(diagnostics.mean_width, 6.0);
        assert_eq!(diagnostics.crossing_rate, 0.25);
    }

    #[test]
    fn repaired_quantiles_have_zero_crossing_rate() {
        let repaired = vec![
            repair_non_crossing_quantiles(&[12.0, 10.0, 13.0]).unwrap(),
            repair_non_crossing_quantiles(&[1.0, 2.0, 3.0]).unwrap(),
        ];

        assert_eq!(crossing_rate(&repaired).unwrap(), 0.0);
    }
}

pub(crate) fn validate_quantile(quantile: f64) -> Result<()> {
    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "quantile must be finite and in (0, 1)".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_finite_values(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

pub(crate) fn validate_same_non_empty(
    left: &[f64],
    right: &[f64],
    left_name: &str,
    right_name: &str,
) -> Result<()> {
    validate_finite_values(left, left_name)?;
    validate_finite_values(right, right_name)?;
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
