use crate::forecasting::stl::{
    default_trend_window, loess_smooth, validate_history_length, validate_season_length,
    validate_trend_window, validate_values, STLDecomposition, DEFAULT_SEASONAL_WINDOW,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;

const MSTL_ITERATIONS: usize = 2;

#[derive(Debug, Clone, PartialEq)]
pub struct MSTLDecomposition {
    season_lengths: Vec<usize>,
    trend_window: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MSTLSeasonalComponent {
    pub season_length: usize,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MSTLDecompositionResult {
    pub observed: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal_components: Vec<MSTLSeasonalComponent>,
    pub remainder: Vec<f64>,
}

impl MSTLDecomposition {
    pub fn new(season_lengths: Vec<usize>) -> Result<Self> {
        Self::with_trend_window(season_lengths, None)
    }

    pub fn with_trend_window(
        mut season_lengths: Vec<usize>,
        trend_window: Option<usize>,
    ) -> Result<Self> {
        if season_lengths.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "mstl requires at least one season length".to_string(),
            ));
        }
        let mut unique = BTreeSet::new();
        for &season_length in &season_lengths {
            validate_season_length(season_length)?;
            if !unique.insert(season_length) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "mstl season lengths must be unique; duplicate {season_length}"
                )));
            }
        }
        season_lengths.sort_unstable();
        if let Some(window) = trend_window {
            validate_trend_window(window)?;
        }
        Ok(Self {
            season_lengths,
            trend_window,
        })
    }

    pub fn season_lengths(&self) -> &[usize] {
        &self.season_lengths
    }

    pub fn trend_window(&self) -> Option<usize> {
        self.trend_window
    }

    pub fn decompose(&self, values: &[f64]) -> Result<MSTLDecompositionResult> {
        validate_values(values)?;
        for &season_length in &self.season_lengths {
            validate_history_length(values.len(), season_length, "mstl")?;
        }

        let mut deseasonalized = values.to_vec();
        let mut seasonal_values = vec![vec![0.0; values.len()]; self.season_lengths.len()];
        for _ in 0..MSTL_ITERATIONS {
            for (component_idx, &season_length) in self.season_lengths.iter().enumerate() {
                // Restore the previous estimate for this period before re-estimating it,
                // while retaining the removal of every other seasonal component.
                for (value, previous) in deseasonalized
                    .iter_mut()
                    .zip(&seasonal_values[component_idx])
                {
                    *value += previous;
                }
                let decomposition =
                    STLDecomposition::with_trend_window(season_length, self.trend_window)?
                        .decompose(&deseasonalized)?;
                seasonal_values[component_idx] = decomposition.seasonal;
                for (value, seasonal) in deseasonalized
                    .iter_mut()
                    .zip(&seasonal_values[component_idx])
                {
                    *value -= seasonal;
                }
            }
        }

        let trend_window = match self.trend_window {
            Some(window) => window,
            None => default_trend_window(
                *self
                    .season_lengths
                    .last()
                    .expect("constructor requires at least one season length"),
            )?,
        };
        let trend = loess_smooth(&deseasonalized, trend_window)?;
        let remainder = deseasonalized
            .iter()
            .zip(&trend)
            .map(|(value, trend_value)| value - trend_value)
            .collect::<Vec<_>>();
        if seasonal_values
            .iter()
            .flatten()
            .chain(&trend)
            .chain(&remainder)
            .any(|value| !value.is_finite())
        {
            return Err(CartoBoostError::InvalidInput(
                "mstl decomposition produced non-finite components".to_string(),
            ));
        }
        let seasonal_components = self
            .season_lengths
            .iter()
            .copied()
            .zip(seasonal_values)
            .map(|(season_length, values)| MSTLSeasonalComponent {
                season_length,
                values,
            })
            .collect();
        Ok(MSTLDecompositionResult {
            observed: values.to_vec(),
            trend,
            seasonal_components,
            remainder,
        })
    }

    pub fn metadata(&self) -> Value {
        let effective_trend_window = self.trend_window.or_else(|| {
            self.season_lengths
                .last()
                .and_then(|season_length| default_trend_window(*season_length).ok())
        });
        json!({
            "method": "mstl",
            "season_lengths": self.season_lengths,
            "seasonal_window": DEFAULT_SEASONAL_WINDOW,
            "trend_window": self.trend_window,
            "effective_trend_window": effective_trend_window,
            "iterations": MSTL_ITERATIONS,
        })
    }
}

impl MSTLDecompositionResult {
    pub fn len(&self) -> usize {
        self.observed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observed.is_empty()
    }

    pub fn total_seasonal(&self) -> Vec<f64> {
        let mut total = vec![0.0; self.observed.len()];
        for component in &self.seasonal_components {
            for (sum, value) in total.iter_mut().zip(&component.values) {
                *sum += value;
            }
        }
        total
    }

    pub fn recompose(&self) -> Vec<f64> {
        let seasonal = self.total_seasonal();
        self.trend
            .iter()
            .zip(seasonal)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mstl_rejects_duplicate_periods() {
        let error =
            MSTLDecomposition::new(vec![24, 7, 24]).expect_err("duplicate periods must fail");
        assert!(error.to_string().contains("duplicate 24"));
    }

    #[test]
    fn mstl_requires_two_cycles_of_every_period() {
        let error = MSTLDecomposition::new(vec![2, 6])
            .expect("valid configuration")
            .decompose(&[1.0; 11])
            .expect_err("incomplete longest cycle must fail");
        assert!(error.to_string().contains("12 observations"));
    }

    #[test]
    fn mstl_separates_multiple_periodic_components() {
        let short = [2.0, -2.0];
        let long = [3.0, 1.0, -1.0, -3.0];
        let values = (0..80)
            .map(|idx| 50.0 + idx as f64 * 0.1 + short[idx % 2] + long[idx % 4])
            .collect::<Vec<_>>();
        let result = MSTLDecomposition::new(vec![2, 4])
            .expect("valid mstl")
            .decompose(&values)
            .expect("decomposition succeeds");

        assert_eq!(result.seasonal_components.len(), 2);
        assert!(result.max_abs_recomposition_error() <= 1.0e-10);
        assert!(
            result
                .remainder
                .iter()
                .map(|value| value.abs())
                .sum::<f64>()
                < 2.0
        );
    }
}
