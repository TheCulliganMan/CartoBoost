use crate::forecasting::ForecastFrame;
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CusumConfig {
    pub reference_mean: f64,
    pub drift: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PageHinkleyConfig {
    pub delta: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EwmaVolatilityConfig {
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimeSignal {
    pub index: usize,
    pub triggered: bool,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimeIntervalAdjustment {
    pub lower: f64,
    pub upper: f64,
    pub confidence: f64,
    pub process_variance_multiplier: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimeIntervalPolicy {
    pub widening_multiplier: f64,
    pub active_window: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CUSUM {
    pub config: CusumConfig,
    pub positive_sum: f64,
    pub negative_sum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PageHinkley {
    pub config: PageHinkleyConfig,
    pub count: usize,
    pub mean: f64,
    pub cumulative: f64,
    pub min_cumulative: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EwmaVolatility {
    pub config: EwmaVolatilityConfig,
    pub variance: f64,
    pub initialized: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastDiagnostics {
    pub n_rows: usize,
    pub n_series: usize,
    pub zero_count: usize,
    pub zero_fraction: f64,
    pub intermittent_series_count: usize,
    pub series: Vec<SeriesDiagnostics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesDiagnostics {
    pub series_id: String,
    pub n_rows: usize,
    pub start_timestamp: NaiveDateTime,
    pub end_timestamp: NaiveDateTime,
    pub min_target: f64,
    pub max_target: f64,
    pub mean_target: f64,
    pub zero_count: usize,
    pub nonzero_count: usize,
    pub zero_fraction: f64,
    pub intermittency_ratio: Option<f64>,
    pub mean_nonzero_interval: Option<f64>,
    pub max_zero_run: usize,
    pub is_intermittent: bool,
}

impl CUSUM {
    pub fn new(config: CusumConfig) -> Result<Self> {
        if !config.reference_mean.is_finite()
            || !config.drift.is_finite()
            || config.drift < 0.0
            || !config.threshold.is_finite()
            || config.threshold <= 0.0
        {
            return Err(CartoBoostError::InvalidInput(
                "CUSUM requires finite reference_mean, non-negative drift, and positive threshold"
                    .to_string(),
            ));
        }
        Ok(Self {
            config,
            positive_sum: 0.0,
            negative_sum: 0.0,
        })
    }

    pub fn update(&mut self, value: f64, index: usize) -> Result<RegimeSignal> {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "CUSUM value must be finite".to_string(),
            ));
        }
        let centered = value - self.config.reference_mean;
        self.positive_sum = (self.positive_sum + centered - self.config.drift).max(0.0);
        self.negative_sum = (self.negative_sum - centered - self.config.drift).max(0.0);
        let score = self.positive_sum.max(self.negative_sum);
        let triggered = score > self.config.threshold;
        if triggered {
            self.positive_sum = 0.0;
            self.negative_sum = 0.0;
        }
        Ok(RegimeSignal {
            index,
            triggered,
            score,
        })
    }

    pub fn scan(&mut self, values: &[f64]) -> Result<Vec<RegimeSignal>> {
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| self.update(*value, idx))
            .collect()
    }
}

impl PageHinkley {
    pub fn new(config: PageHinkleyConfig) -> Result<Self> {
        if !config.delta.is_finite()
            || config.delta < 0.0
            || !config.threshold.is_finite()
            || config.threshold <= 0.0
        {
            return Err(CartoBoostError::InvalidInput(
                "PageHinkley requires non-negative finite delta and positive threshold".to_string(),
            ));
        }
        Ok(Self {
            config,
            count: 0,
            mean: 0.0,
            cumulative: 0.0,
            min_cumulative: 0.0,
        })
    }

    pub fn update(&mut self, value: f64, index: usize) -> Result<RegimeSignal> {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "PageHinkley value must be finite".to_string(),
            ));
        }
        self.count += 1;
        self.mean += (value - self.mean) / self.count as f64;
        self.cumulative += value - self.mean - self.config.delta;
        self.min_cumulative = self.min_cumulative.min(self.cumulative);
        let score = self.cumulative - self.min_cumulative;
        let triggered = score > self.config.threshold;
        if triggered {
            self.count = 0;
            self.mean = 0.0;
            self.cumulative = 0.0;
            self.min_cumulative = 0.0;
        }
        Ok(RegimeSignal {
            index,
            triggered,
            score,
        })
    }

    pub fn scan(&mut self, values: &[f64]) -> Result<Vec<RegimeSignal>> {
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| self.update(*value, idx))
            .collect()
    }
}

impl EwmaVolatility {
    pub fn new(config: EwmaVolatilityConfig) -> Result<Self> {
        if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha > 1.0 {
            return Err(CartoBoostError::InvalidInput(
                "EWMA volatility alpha must be finite and in (0, 1]".to_string(),
            ));
        }
        Ok(Self {
            config,
            variance: 0.0,
            initialized: false,
        })
    }

    pub fn update(&mut self, residual: f64) -> Result<f64> {
        if !residual.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "EWMA residual must be finite".to_string(),
            ));
        }
        let squared = residual * residual;
        self.variance = if self.initialized {
            self.config.alpha * squared + (1.0 - self.config.alpha) * self.variance
        } else {
            self.initialized = true;
            squared
        };
        Ok(self.variance.sqrt())
    }

    pub fn scan(&mut self, residuals: &[f64]) -> Result<Vec<f64>> {
        residuals
            .iter()
            .map(|residual| self.update(*residual))
            .collect()
    }
}

pub fn rolling_median_residual(residuals: &[f64], window: usize) -> Result<Vec<f64>> {
    rolling_statistic(residuals, window, median)
}

pub fn rolling_mad_residual(residuals: &[f64], window: usize) -> Result<Vec<f64>> {
    validate_residual_window(residuals, window)?;
    Ok((0..residuals.len())
        .map(|idx| {
            let start = (idx + 1).saturating_sub(window);
            let values = &residuals[start..=idx];
            let center = median(values.to_vec());
            let deviations = values
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>();
            median(deviations)
        })
        .collect())
}

pub fn widen_interval_for_regime(
    lower: f64,
    upper: f64,
    signal: RegimeSignal,
    volatility: f64,
    widening_multiplier: f64,
) -> Result<RegimeIntervalAdjustment> {
    if !lower.is_finite()
        || !upper.is_finite()
        || lower > upper
        || !signal.score.is_finite()
        || !volatility.is_finite()
        || volatility < 0.0
        || !widening_multiplier.is_finite()
        || widening_multiplier < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "regime interval adjustment requires finite ordered interval and non-negative volatility/multiplier"
                .to_string(),
        ));
    }
    let extra = if signal.triggered {
        widening_multiplier * volatility.max(signal.score)
    } else {
        0.0
    };
    let process_variance_multiplier = if signal.triggered {
        1.0 + widening_multiplier.max(0.0)
    } else {
        1.0
    };
    let confidence = if signal.triggered {
        (1.0 / process_variance_multiplier).clamp(0.0, 1.0)
    } else {
        1.0
    };
    Ok(RegimeIntervalAdjustment {
        lower: lower - extra,
        upper: upper + extra,
        confidence,
        process_variance_multiplier,
    })
}

pub fn regime_adjusted_intervals(
    lower: &[f64],
    upper: &[f64],
    signals: &[RegimeSignal],
    volatilities: &[f64],
    policy: RegimeIntervalPolicy,
) -> Result<Vec<RegimeIntervalAdjustment>> {
    if lower.len() != upper.len()
        || lower.len() != signals.len()
        || lower.len() != volatilities.len()
    {
        return Err(CartoBoostError::InvalidInput(
            "regime interval inputs must have the same length".to_string(),
        ));
    }
    if policy.active_window == 0
        || !policy.widening_multiplier.is_finite()
        || policy.widening_multiplier < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "regime interval policy requires positive active_window and non-negative finite multiplier"
                .to_string(),
        ));
    }

    let mut active_remaining = 0usize;
    let mut active_score = 0.0;
    lower
        .iter()
        .zip(upper)
        .zip(signals)
        .zip(volatilities)
        .map(|(((lower, upper), signal), volatility)| {
            if signal.triggered {
                active_remaining = policy.active_window;
                active_score = signal.score;
            }
            let active_signal = RegimeSignal {
                index: signal.index,
                triggered: active_remaining > 0,
                score: if active_remaining > 0 {
                    active_score.max(signal.score)
                } else {
                    signal.score
                },
            };
            let adjustment = widen_interval_for_regime(
                *lower,
                *upper,
                active_signal,
                *volatility,
                policy.widening_multiplier,
            )?;
            active_remaining = active_remaining.saturating_sub(1);
            Ok(adjustment)
        })
        .collect()
}

impl ForecastDiagnostics {
    pub fn from_frame(frame: &ForecastFrame) -> Self {
        let series = frame
            .series_ids()
            .into_iter()
            .map(|series_id| {
                let rows = frame.rows_for_series(&series_id);
                SeriesDiagnostics::from_rows(series_id, &rows)
            })
            .collect::<Vec<_>>();
        let n_rows = frame.rows().len();
        let zero_count = series.iter().map(|diag| diag.zero_count).sum::<usize>();
        let zero_fraction = fraction(zero_count, n_rows);
        let intermittent_series_count = series.iter().filter(|diag| diag.is_intermittent).count();
        Self {
            n_rows,
            n_series: series.len(),
            zero_count,
            zero_fraction,
            intermittent_series_count,
            series,
        }
    }

    pub fn series(&self, series_id: &str) -> Option<&SeriesDiagnostics> {
        self.series
            .iter()
            .find(|diagnostics| diagnostics.series_id == series_id)
    }
}

impl SeriesDiagnostics {
    fn from_rows(series_id: String, rows: &[&crate::forecasting::ForecastRow]) -> Self {
        let n_rows = rows.len();
        let start_timestamp = rows
            .first()
            .map(|row| row.timestamp)
            .expect("ForecastFrame guarantees non-empty series");
        let end_timestamp = rows
            .last()
            .map(|row| row.timestamp)
            .expect("ForecastFrame guarantees non-empty series");
        let mut min_target = f64::INFINITY;
        let mut max_target = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut zero_count = 0;
        let mut max_zero_run = 0;
        let mut current_zero_run = 0;
        let mut nonzero_positions = Vec::new();

        for (idx, row) in rows.iter().enumerate() {
            min_target = min_target.min(row.target);
            max_target = max_target.max(row.target);
            sum += row.target;
            if row.target == 0.0 {
                zero_count += 1;
                current_zero_run += 1;
                max_zero_run = max_zero_run.max(current_zero_run);
            } else {
                current_zero_run = 0;
                nonzero_positions.push(idx);
            }
        }

        let nonzero_count = n_rows - zero_count;
        let zero_fraction = fraction(zero_count, n_rows);
        let intermittency_ratio = if nonzero_count == 0 {
            None
        } else {
            Some(zero_count as f64 / nonzero_count as f64)
        };
        let mean_nonzero_interval = if nonzero_positions.len() < 2 {
            None
        } else {
            let first = *nonzero_positions.first().expect("non-empty positions") as f64;
            let last = *nonzero_positions.last().expect("non-empty positions") as f64;
            Some((last - first) / (nonzero_positions.len() - 1) as f64)
        };

        Self {
            series_id,
            n_rows,
            start_timestamp,
            end_timestamp,
            min_target,
            max_target,
            mean_target: sum / n_rows as f64,
            zero_count,
            nonzero_count,
            zero_fraction,
            intermittency_ratio,
            mean_nonzero_interval,
            max_zero_run,
            is_intermittent: zero_count > 0,
        }
    }
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn rolling_statistic(
    residuals: &[f64],
    window: usize,
    statistic: fn(Vec<f64>) -> f64,
) -> Result<Vec<f64>> {
    validate_residual_window(residuals, window)?;
    Ok((0..residuals.len())
        .map(|idx| {
            let start = (idx + 1).saturating_sub(window);
            statistic(residuals[start..=idx].to_vec())
        })
        .collect())
}

fn validate_residual_window(residuals: &[f64], window: usize) -> Result<()> {
    if window == 0 {
        return Err(CartoBoostError::InvalidInput(
            "rolling residual window must be positive".to_string(),
        ));
    }
    if residuals.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "residuals must contain only finite values".to_string(),
        ));
    }
    Ok(())
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::quantiles::interval_coverage;

    #[test]
    fn cusum_catches_injected_mean_shift_with_low_stationary_triggers() {
        let stationary = (0..80)
            .map(|idx| {
                if idx % 11 == 0 {
                    2.5
                } else {
                    0.05 * ((idx % 5) as f64 - 2.0)
                }
            })
            .collect::<Vec<_>>();
        let mut detector = CUSUM::new(CusumConfig {
            reference_mean: 0.0,
            drift: 0.2,
            threshold: 8.0,
        })
        .unwrap();
        let stationary_signals = detector.scan(&stationary).unwrap();
        assert_eq!(
            stationary_signals
                .iter()
                .filter(|signal| signal.triggered)
                .count(),
            0
        );

        let shifted = stationary
            .into_iter()
            .chain((0..20).map(|_| 1.5))
            .collect::<Vec<_>>();
        let mut detector = CUSUM::new(CusumConfig {
            reference_mean: 0.0,
            drift: 0.2,
            threshold: 8.0,
        })
        .unwrap();
        let shifted_signals = detector.scan(&shifted).unwrap();
        assert!(shifted_signals
            .iter()
            .any(|signal| signal.triggered && signal.index >= 80));
    }

    #[test]
    fn page_hinkley_catches_mean_shift() {
        let values = (0..50)
            .map(|idx| 0.02 * ((idx % 7) as f64 - 3.0))
            .chain((0..30).map(|_| 1.0))
            .collect::<Vec<_>>();
        let mut detector = PageHinkley::new(PageHinkleyConfig {
            delta: 0.05,
            threshold: 6.0,
        })
        .unwrap();
        let signals = detector.scan(&values).unwrap();

        assert!(signals
            .iter()
            .any(|signal| signal.triggered && signal.index >= 50));
    }

    #[test]
    fn page_hinkley_has_low_false_triggers_on_stationary_heavy_tailed_noise() {
        let values = (0..160)
            .map(|idx| match idx % 29 {
                0 => 2.5,
                1 => -2.5,
                _ => 0.03 * ((idx % 9) as f64 - 4.0),
            })
            .collect::<Vec<_>>();
        let mut detector = PageHinkley::new(PageHinkleyConfig {
            delta: 0.1,
            threshold: 7.5,
        })
        .unwrap();
        let signals = detector.scan(&values).unwrap();

        assert!(
            signals.iter().filter(|signal| signal.triggered).count() <= 1,
            "stationary heavy-tailed noise should have low false triggers"
        );
    }

    #[test]
    fn rolling_residual_statistics_are_deterministic() {
        let residuals = [1.0, 100.0, 2.0, 3.0, 4.0];

        assert_eq!(
            rolling_median_residual(&residuals, 3).unwrap(),
            vec![1.0, 50.5, 2.0, 3.0, 3.0]
        );
        assert_eq!(
            rolling_mad_residual(&residuals, 3).unwrap(),
            vec![0.0, 49.5, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn ewma_volatility_and_regime_adjustment_widen_intervals() {
        let mut ewma = EwmaVolatility::new(EwmaVolatilityConfig { alpha: 0.5 }).unwrap();
        let vol = ewma.scan(&[1.0, 3.0]).unwrap();
        assert!(vol[1] > vol[0]);

        let adjusted = widen_interval_for_regime(
            10.0,
            12.0,
            RegimeSignal {
                index: 5,
                triggered: true,
                score: 2.0,
            },
            vol[1],
            1.5,
        )
        .unwrap();

        assert!(adjusted.lower < 10.0);
        assert!(adjusted.upper > 12.0);
        assert!(adjusted.confidence < 1.0);
        assert!(adjusted.process_variance_multiplier > 1.0);
    }

    #[test]
    fn regime_aware_intervals_raise_coverage_during_shift() {
        let actual = (0..80)
            .map(|idx| {
                if idx < 50 {
                    0.4 * ((idx % 5) as f64 - 2.0)
                } else {
                    3.0 + 0.4 * ((idx % 5) as f64 - 2.0)
                }
            })
            .collect::<Vec<_>>();
        let lower = vec![-1.0; actual.len()];
        let upper = vec![1.0; actual.len()];
        let residuals = actual.clone();
        let mut cusum = CUSUM::new(CusumConfig {
            reference_mean: 0.0,
            drift: 0.1,
            threshold: 2.0,
        })
        .unwrap();
        let signals = cusum.scan(&residuals).unwrap();
        assert!(signals
            .iter()
            .any(|signal| signal.triggered && signal.index >= 50));

        let mut ewma = EwmaVolatility::new(EwmaVolatilityConfig { alpha: 0.25 }).unwrap();
        let volatilities = ewma.scan(&residuals).unwrap();
        let adjusted = regime_adjusted_intervals(
            &lower,
            &upper,
            &signals,
            &volatilities,
            RegimeIntervalPolicy {
                widening_multiplier: 1.0,
                active_window: 12,
            },
        )
        .unwrap();
        let adjusted_lower = adjusted
            .iter()
            .map(|interval| interval.lower)
            .collect::<Vec<_>>();
        let adjusted_upper = adjusted
            .iter()
            .map(|interval| interval.upper)
            .collect::<Vec<_>>();

        let base_shift_coverage =
            interval_coverage(&actual[50..], &lower[50..], &upper[50..]).unwrap();
        let adjusted_shift_coverage =
            interval_coverage(&actual[50..], &adjusted_lower[50..], &adjusted_upper[50..]).unwrap();

        assert!(adjusted_shift_coverage > base_shift_coverage);
        assert!(adjusted
            .iter()
            .skip(50)
            .any(|interval| interval.process_variance_multiplier > 1.0));
        assert!(adjusted
            .iter()
            .skip(50)
            .any(|interval| interval.confidence < 1.0));
    }
}
