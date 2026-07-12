use crate::{CartoBoostError, Result};
use serde_json::{json, Value};

pub(crate) const DEFAULT_SEASONAL_WINDOW: usize = 7;
const INNER_ITERATIONS: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct STLDecomposition {
    season_length: usize,
    trend_window: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct STLDecompositionResult {
    pub observed: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub remainder: Vec<f64>,
}

impl STLDecomposition {
    pub fn new(season_length: usize) -> Result<Self> {
        Self::with_trend_window(season_length, None)
    }

    pub fn with_trend_window(season_length: usize, trend_window: Option<usize>) -> Result<Self> {
        validate_season_length(season_length)?;
        if let Some(window) = trend_window {
            validate_trend_window(window)?;
        }
        Ok(Self {
            season_length,
            trend_window,
        })
    }

    pub fn season_length(&self) -> usize {
        self.season_length
    }

    pub fn trend_window(&self) -> Option<usize> {
        self.trend_window
    }

    pub fn decompose(&self, values: &[f64]) -> Result<STLDecompositionResult> {
        validate_values(values)?;
        validate_history_length(values.len(), self.season_length, "stl")?;
        if let Some(window) = self.trend_window {
            if window > values.len() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "stl trend_window={window} exceeds the available history of {} observations",
                    values.len()
                )));
            }
        }

        let trend_window = self.effective_trend_window()?;
        let low_pass_window = next_odd(self.season_length)?;
        let mut trend = vec![0.0; values.len()];
        let mut seasonal = vec![0.0; values.len()];

        // Cleveland-style STL inner loop: smooth each detrended cycle subseries,
        // remove its low-pass component, then LOESS-smooth the deseasonalized data.
        for _ in 0..INNER_ITERATIONS {
            let detrended = values
                .iter()
                .zip(&trend)
                .map(|(value, trend_value)| value - trend_value)
                .collect::<Vec<_>>();
            let extended =
                smooth_cycle_subseries(&detrended, self.season_length, DEFAULT_SEASONAL_WINDOW)?;
            let low_pass = stl_low_pass(&extended, self.season_length, low_pass_window)?;
            seasonal = extended[self.season_length..self.season_length + values.len()]
                .iter()
                .zip(low_pass)
                .map(|(cycle, low_pass_value)| cycle - low_pass_value)
                .collect();
            let deseasonalized = values
                .iter()
                .zip(&seasonal)
                .map(|(value, seasonal_value)| value - seasonal_value)
                .collect::<Vec<_>>();
            trend = loess_smooth(&deseasonalized, trend_window)?;
        }

        let remainder = values
            .iter()
            .zip(&trend)
            .zip(&seasonal)
            .map(|((value, trend_value), seasonal_value)| value - trend_value - seasonal_value)
            .collect::<Vec<_>>();
        validate_components(&trend, &seasonal, &remainder)?;
        Ok(STLDecompositionResult {
            observed: values.to_vec(),
            trend,
            seasonal,
            remainder,
        })
    }

    pub fn metadata(&self) -> Value {
        json!({
            "method": "stl",
            "season_length": self.season_length,
            "seasonal_window": DEFAULT_SEASONAL_WINDOW,
            "trend_window": self.trend_window,
            "effective_trend_window": self.effective_trend_window().ok(),
            "inner_iterations": INNER_ITERATIONS,
        })
    }

    pub(crate) fn effective_trend_window(&self) -> Result<usize> {
        match self.trend_window {
            Some(window) => Ok(window),
            None => default_trend_window(self.season_length),
        }
    }
}

impl STLDecompositionResult {
    pub fn len(&self) -> usize {
        self.observed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observed.is_empty()
    }

    pub fn recompose(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(&self.seasonal)
            .zip(&self.remainder)
            .map(|((trend, seasonal), remainder)| trend + seasonal + remainder)
            .collect()
    }

    pub fn max_abs_recomposition_error(&self) -> f64 {
        self.observed
            .iter()
            .zip(self.recompose())
            .map(|(observed, recomposed)| (observed - recomposed).abs())
            .fold(0.0, f64::max)
    }

    pub fn seasonal_pattern(&self, season_length: usize) -> Vec<f64> {
        seasonal_pattern(&self.seasonal, season_length)
    }
}

pub(crate) fn validate_values(values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "decomposition requires at least one observation".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "decomposition values must be finite".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_history_length(
    history_len: usize,
    season_length: usize,
    method: &str,
) -> Result<()> {
    let minimum = season_length
        .checked_mul(2)
        .ok_or_else(|| CartoBoostError::InvalidInput("season_length is too large".to_string()))?;
    if history_len < minimum {
        return Err(CartoBoostError::InvalidInput(format!(
            "{method} requires at least two complete seasonal cycles ({minimum} observations for season_length={season_length}), got {history_len}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_season_length(season_length: usize) -> Result<()> {
    if season_length <= 1 {
        return Err(CartoBoostError::InvalidInput(
            "season_length must be greater than 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_trend_window(window: usize) -> Result<()> {
    if window < 3 || window.is_multiple_of(2) {
        return Err(CartoBoostError::InvalidInput(
            "trend_window must be an odd integer of at least 3".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn default_trend_window(season_length: usize) -> Result<usize> {
    // The standard STL default is the smallest odd integer at least
    // 1.5 * period / (1 - 1.5 / seasonal_window).
    let denominator = 1.0 - 1.5 / DEFAULT_SEASONAL_WINDOW as f64;
    let requested = (1.5 * season_length as f64 / denominator).ceil();
    if !requested.is_finite() || requested > usize::MAX as f64 {
        return Err(CartoBoostError::InvalidInput(
            "season_length is too large".to_string(),
        ));
    }
    next_odd(requested as usize)
}

fn next_odd(value: usize) -> Result<usize> {
    let value = value.max(3);
    if value.is_multiple_of(2) {
        value.checked_add(1).ok_or_else(|| {
            CartoBoostError::InvalidInput("smoothing window is too large".to_string())
        })
    } else {
        Ok(value)
    }
}

pub(crate) fn loess_smooth(values: &[f64], window: usize) -> Result<Vec<f64>> {
    validate_values(values)?;
    validate_trend_window(window)?;
    let x = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    (0..values.len())
        .map(|idx| local_linear_predict(&x, values, idx as f64, window))
        .collect()
}

fn smooth_cycle_subseries(values: &[f64], season_length: usize, window: usize) -> Result<Vec<f64>> {
    let n = values.len();
    let extended_len = n
        .checked_add(season_length.checked_mul(2).ok_or_else(|| {
            CartoBoostError::InvalidInput("season_length is too large".to_string())
        })?)
        .ok_or_else(|| CartoBoostError::InvalidInput("series is too large".to_string()))?;
    let mut extended = vec![0.0; extended_len];

    for phase in 0..season_length {
        let phase_values = (phase..n)
            .step_by(season_length)
            .map(|idx| values[idx])
            .collect::<Vec<_>>();
        if phase_values.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "stl seasonal phase {phase} has fewer than two observations"
            )));
        }
        let cycles = (0..phase_values.len())
            .map(|cycle| cycle as f64)
            .collect::<Vec<_>>();
        let phase_i = phase as i128;
        let period_i = season_length as i128;
        for (extended_idx, value) in extended.iter_mut().enumerate().take(extended_len) {
            let time = extended_idx as i128 - period_i;
            if time.rem_euclid(period_i) == phase_i {
                let cycle = (time - phase_i).div_euclid(period_i) as f64;
                *value = local_linear_predict(&cycles, &phase_values, cycle, window)?;
            }
        }
    }
    Ok(extended)
}

fn stl_low_pass(
    extended_cycle: &[f64],
    season_length: usize,
    low_pass_window: usize,
) -> Result<Vec<f64>> {
    let first = moving_average_valid(extended_cycle, season_length)?;
    let second = moving_average_valid(&first, season_length)?;
    let third = moving_average_valid(&second, 3)?;
    loess_smooth(&third, low_pass_window)
}

fn moving_average_valid(values: &[f64], window: usize) -> Result<Vec<f64>> {
    if window == 0 || window > values.len() {
        return Err(CartoBoostError::InvalidInput(format!(
            "moving-average window {window} exceeds series length {}",
            values.len()
        )));
    }
    let mut sum = values[..window].iter().sum::<f64>();
    let mut result = Vec::with_capacity(values.len() - window + 1);
    result.push(sum / window as f64);
    for idx in window..values.len() {
        sum += values[idx] - values[idx - window];
        result.push(sum / window as f64);
    }
    Ok(result)
}

fn local_linear_predict(x: &[f64], y: &[f64], target: f64, window: usize) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() || y.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "LOESS inputs must be non-empty, finite, and have matching lengths".to_string(),
        ));
    }
    let count = window.min(x.len());
    let insertion = x.partition_point(|candidate| *candidate < target);
    let mut start = insertion.saturating_sub(count / 2);
    start = start.min(x.len() - count);
    let end = start + count;
    let max_distance = x[start..end]
        .iter()
        .map(|value| (value - target).abs())
        .fold(0.0, f64::max);
    if max_distance <= f64::EPSILON {
        return Ok(y[start]);
    }

    let mut sum_weight = 0.0;
    let mut sum_weight_x = 0.0;
    let mut sum_weight_xx = 0.0;
    let mut sum_weight_y = 0.0;
    let mut sum_weight_xy = 0.0;
    for idx in start..end {
        let distance_ratio = ((x[idx] - target).abs() / max_distance).min(1.0);
        let weight = (1.0 - distance_ratio.powi(3)).powi(3);
        let centered_x = x[idx] - target;
        sum_weight += weight;
        sum_weight_x += weight * centered_x;
        sum_weight_xx += weight * centered_x * centered_x;
        sum_weight_y += weight * y[idx];
        sum_weight_xy += weight * centered_x * y[idx];
    }
    if !sum_weight.is_finite() || sum_weight <= f64::EPSILON {
        return Err(CartoBoostError::InvalidInput(
            "LOESS smoothing produced no positive finite weights".to_string(),
        ));
    }
    let determinant = sum_weight * sum_weight_xx - sum_weight_x * sum_weight_x;
    let scale = (sum_weight * sum_weight_xx)
        .abs()
        .max(sum_weight_x.abs().powi(2))
        .max(1.0);
    let prediction = if determinant.abs() <= scale * 1.0e-12 {
        sum_weight_y / sum_weight
    } else {
        (sum_weight_xx * sum_weight_y - sum_weight_x * sum_weight_xy) / determinant
    };
    if !prediction.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "LOESS smoothing produced a non-finite value".to_string(),
        ));
    }
    Ok(prediction)
}

fn validate_components(trend: &[f64], seasonal: &[f64], remainder: &[f64]) -> Result<()> {
    if trend
        .iter()
        .chain(seasonal)
        .chain(remainder)
        .any(|value| !value.is_finite())
    {
        return Err(CartoBoostError::InvalidInput(
            "stl decomposition produced non-finite components".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn seasonal_pattern(values: &[f64], season_length: usize) -> Vec<f64> {
    let mut sums = vec![0.0; season_length];
    let mut counts = vec![0usize; season_length];
    for (idx, value) in values.iter().enumerate() {
        let phase = idx % season_length;
        sums[phase] += value;
        counts[phase] += 1;
    }
    let mut pattern = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| if count == 0 { 0.0 } else { sum / count as f64 })
        .collect::<Vec<_>>();
    let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
    for value in &mut pattern {
        *value -= mean;
    }
    pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stl_requires_two_complete_cycles() {
        let error = STLDecomposition::new(4)
            .expect("valid configuration")
            .decompose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect_err("incomplete history must fail");
        assert!(error.to_string().contains("two complete seasonal cycles"));
    }

    #[test]
    fn stl_rejects_even_explicit_trend_window() {
        let error = STLDecomposition::with_trend_window(4, Some(6))
            .expect_err("even LOESS window must fail");
        assert!(error.to_string().contains("odd integer"));
    }

    #[test]
    fn stl_rejects_explicit_trend_window_larger_than_history() {
        let error = STLDecomposition::with_trend_window(3, Some(9))
            .expect("valid configuration")
            .decompose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect_err("unsupported explicit window must fail");
        assert!(error.to_string().contains("exceeds the available history"));
    }

    #[test]
    fn stl_extracts_stable_periodic_signal() {
        let values = (0..36)
            .map(|idx| 20.0 + 0.25 * idx as f64 + [3.0, -2.0, -1.0][idx % 3])
            .collect::<Vec<_>>();
        let result = STLDecomposition::new(3)
            .expect("valid stl")
            .decompose(&values)
            .expect("decomposition succeeds");

        assert!(result.max_abs_recomposition_error() <= 1.0e-10);
        let seasonal = result.seasonal_pattern(3);
        assert!((seasonal[0] - 3.0).abs() < 0.15);
        assert!((seasonal[1] + 2.0).abs() < 0.15);
        assert!((seasonal[2] + 1.0).abs() < 0.15);
        assert!(
            result
                .remainder
                .iter()
                .map(|value| value.abs())
                .sum::<f64>()
                < 0.5
        );
    }

    #[test]
    fn stl_matches_statsmodels_reference_for_linear_period_two_series() {
        let values = [10.0, 14.0, 11.0, 15.0, 12.0, 16.0, 13.0, 17.0];
        let result = STLDecomposition::new(2)
            .expect("valid stl")
            .decompose(&values)
            .expect("decomposition succeeds");
        let expected_trend = [11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25];
        let expected_seasonal = [-1.75, 1.75, -1.75, 1.75, -1.75, 1.75, -1.75, 1.75];
        for (actual, expected) in result.trend.iter().zip(expected_trend) {
            assert!(
                (actual - expected).abs() <= 1.0e-9,
                "trend mismatch: actual={actual}, expected={expected}"
            );
        }
        for (actual, expected) in result.seasonal.iter().zip(expected_seasonal) {
            assert!(
                (actual - expected).abs() <= 1.0e-9,
                "seasonal mismatch: actual={actual}, expected={expected}"
            );
        }
        assert!(result.remainder.iter().all(|value| value.abs() <= 1.0e-9));
    }

    #[test]
    fn local_linear_smoother_reproduces_a_line_including_boundaries() {
        let values = (0..20)
            .map(|idx| 4.0 + 1.75 * idx as f64)
            .collect::<Vec<_>>();
        let smoothed = loess_smooth(&values, 7).expect("smooth line");
        for (actual, expected) in smoothed.iter().zip(values) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
    }
}
