use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

use super::quantiles::{validate_finite_values, validate_quantile, validate_same_non_empty};
use super::{ForecastIntervalPrediction, ForecastPrediction, ForecastResult};
use std::collections::BTreeMap;

/// Gaussian residual approximation. This does not claim split-conformal coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResidualForecastIntervals {
    pub(crate) levels: Vec<f64>,
    scales: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) origin_age_steps: BTreeMap<String, f64>,
}

impl ResidualForecastIntervals {
    pub(crate) fn reset(&self) -> Self {
        Self {
            levels: self.levels.clone(),
            scales: BTreeMap::new(),
            origin_age_steps: BTreeMap::new(),
        }
    }
    pub(crate) fn configured(mut levels: Vec<f64>) -> Result<Option<Self>> {
        for &level in &levels {
            validate_quantile(level)?;
        }
        levels.sort_by(f64::total_cmp);
        if levels.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(CartoBoostError::InvalidInput(
                "prediction interval levels must be unique".into(),
            ));
        }
        Ok((!levels.is_empty()).then(|| Self {
            levels,
            scales: BTreeMap::new(),
            origin_age_steps: BTreeMap::new(),
        }))
    }

    pub(crate) fn fit_series(
        &mut self,
        series_id: &str,
        residuals: impl Iterator<Item = f64>,
    ) -> Result<()> {
        let mut sum = 0.0;
        let mut count = 0usize;
        for residual in residuals {
            sum += residual * residual;
            count += 1;
        }
        if count < 2 || !sum.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} prediction intervals require at least two finite residuals with finite squared sum"
            )));
        }
        self.scales
            .insert(series_id.to_string(), (sum / count as f64).sqrt());
        Ok(())
    }

    pub(crate) fn metadata(&self, horizon_scaling: &str) -> serde_json::Value {
        serde_json::json!({
            "method": "residual_gaussian", "levels": self.levels,
            "scale": "training_residual_root_mean_square", "horizon_scaling": horizon_scaling,
            "coverage_assumption": "Gaussian residual approximation; not held-out conformal calibration"
        })
    }
}

pub(crate) fn residual_interval_result(
    predictions: Vec<ForecastPrediction>,
    calibration: Option<&ResidualForecastIntervals>,
    recurrence_period: usize,
) -> Result<ForecastResult> {
    let Some(calibration) = calibration else {
        return ForecastResult::new(predictions);
    };
    let mut intervals =
        Vec::with_capacity(predictions.len().saturating_mul(calibration.levels.len()));
    let quantiles: Vec<_> = calibration
        .levels
        .iter()
        .map(|&level| {
            (
                level,
                -super::local::inverse_standard_normal_cdf((1.0 - level) / 2.0),
            )
        })
        .collect();
    for prediction in &predictions {
        let scale = calibration
            .scales
            .get(&prediction.series_id)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing residual scale for series {}",
                    prediction.series_id
                ))
            })?;
        let growth = if recurrence_period == 0 {
            1.0
        } else if recurrence_period == 1 {
            (prediction.horizon as f64
                + calibration
                    .origin_age_steps
                    .get(&prediction.series_id)
                    .copied()
                    .unwrap_or(0.0))
            .sqrt()
        } else {
            (prediction.horizon.div_ceil(recurrence_period) as f64).sqrt()
        };
        for &(level, quantile) in &quantiles {
            let width = quantile * scale * growth;
            intervals.push(ForecastIntervalPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: prediction.model.clone(),
                level,
                lower: prediction.mean - width,
                upper: prediction.mean + width,
            });
        }
    }
    ForecastResult::new_with_intervals(predictions, intervals)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalInterval {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub residual_quantile: f64,
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformalSplitOrder {
    pub train_end_exclusive: usize,
    pub calibration_start: usize,
    pub calibration_end_exclusive: usize,
    pub test_start: usize,
}

impl ConformalSplitOrder {
    pub fn validate(&self) -> Result<()> {
        if self.train_end_exclusive == 0 {
            return Err(CartoBoostError::InvalidInput(
                "training split must contain at least one row".to_string(),
            ));
        }
        if self.train_end_exclusive > self.calibration_start {
            return Err(CartoBoostError::InvalidInput(
                "training rows must end before calibration rows start".to_string(),
            ));
        }
        if self.calibration_start >= self.calibration_end_exclusive {
            return Err(CartoBoostError::InvalidInput(
                "calibration split must contain at least one row".to_string(),
            ));
        }
        if self.calibration_end_exclusive > self.test_start {
            return Err(CartoBoostError::InvalidInput(
                "calibration rows must end before test rows start".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalCalibrator {
    alpha: f64,
    residual_quantile: Option<f64>,
    split_order: Option<ConformalSplitOrder>,
}

impl ConformalCalibrator {
    pub fn new(alpha: f64) -> Result<Self> {
        validate_quantile(alpha)?;
        Ok(Self {
            alpha,
            residual_quantile: None,
            split_order: None,
        })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn residual_quantile(&self) -> Option<f64> {
        self.residual_quantile
    }

    pub fn fit(
        &mut self,
        calibration_actual: &[f64],
        calibration_prediction: &[f64],
        split_order: ConformalSplitOrder,
    ) -> Result<()> {
        validate_quantile(self.alpha)?;
        split_order.validate()?;
        validate_same_non_empty(
            calibration_actual,
            calibration_prediction,
            "calibration_actual",
            "calibration_prediction",
        )?;
        let expected_calibration_len = split_order
            .calibration_end_exclusive
            .checked_sub(split_order.calibration_start)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "calibration split bounds must be ordered".to_string(),
                )
            })?;
        if calibration_actual.len() != expected_calibration_len {
            return Err(CartoBoostError::InvalidInput(format!(
                "calibration arrays contain {} rows but the declared calibration split contains {expected_calibration_len}",
                calibration_actual.len()
            )));
        }
        let mut scores = calibration_actual
            .iter()
            .zip(calibration_prediction)
            .map(|(&actual, &prediction)| (actual - prediction).abs())
            .collect::<Vec<_>>();
        validate_finite_values(&scores, "calibration residuals")?;
        let rank = (((scores.len() + 1) as f64) * (1.0 - self.alpha)).ceil() as usize;
        if rank > scores.len() {
            return Err(CartoBoostError::InvalidInput(format!(
                "{} calibration residuals are insufficient for a finite split-conformal interval at alpha {}; require alpha >= {}",
                scores.len(),
                self.alpha,
                1.0 / (scores.len() + 1) as f64
            )));
        }
        let index = rank.saturating_sub(1);
        let (_, quantile, _) = scores.select_nth_unstable_by(index, f64::total_cmp);
        self.residual_quantile = Some(*quantile);
        self.split_order = Some(split_order);
        Ok(())
    }

    pub fn fit_with_strict_ordering(
        &mut self,
        calibration_actual: &[f64],
        calibration_prediction: &[f64],
        train_end_exclusive: usize,
        calibration_start: usize,
        calibration_end_exclusive: usize,
        test_start: usize,
    ) -> Result<()> {
        self.fit(
            calibration_actual,
            calibration_prediction,
            ConformalSplitOrder {
                train_end_exclusive,
                calibration_start,
                calibration_end_exclusive,
                test_start,
            },
        )
    }

    pub fn predict_interval(
        &self,
        test_prediction: &[f64],
        test_start: usize,
    ) -> Result<ConformalInterval> {
        validate_quantile(self.alpha)?;
        validate_finite_values(test_prediction, "test_prediction")?;
        if test_prediction.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "test_prediction must contain at least one value".to_string(),
            ));
        }
        let order = self.split_order.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "conformal calibrator must be fit before prediction".into(),
            )
        })?;
        order.validate()?;
        if test_start < order.test_start {
            return Err(CartoBoostError::InvalidInput(
                "test_start must not precede the validated test split".to_string(),
            ));
        }
        let residual_quantile = self.residual_quantile.ok_or_else(|| {
            CartoBoostError::InvalidInput("conformal calibrator has no residual quantile".into())
        })?;
        if !residual_quantile.is_finite() || residual_quantile < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "conformal residual quantile must be finite and nonnegative".into(),
            ));
        }
        let lower = test_prediction
            .iter()
            .map(|prediction| prediction - residual_quantile)
            .collect::<Vec<_>>();
        let upper = test_prediction
            .iter()
            .map(|prediction| prediction + residual_quantile)
            .collect::<Vec<_>>();
        validate_finite_values(&lower, "conformal lower bounds")?;
        validate_finite_values(&upper, "conformal upper bounds")?;
        Ok(ConformalInterval {
            lower,
            upper,
            residual_quantile,
            alpha: self.alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::quantiles::interval_coverage;

    fn heteroscedastic_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let calibration_prediction = vec![100.0; 100];
        let test_prediction = vec![100.0; 100];
        let calibration_actual = (0..100)
            .map(|idx| {
                let scale = 1.0 + (idx % 10) as f64;
                let magnitude = ((idx / 10) + 1) as f64;
                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                100.0 + sign * scale * magnitude
            })
            .collect::<Vec<_>>();
        let test_actual = (0..100)
            .map(|idx| {
                let scale = 1.0 + ((99 - idx) % 10) as f64;
                let magnitude = (((99 - idx) / 10) + 1) as f64;
                let sign = if idx % 2 == 0 { -1.0 } else { 1.0 };
                100.0 + sign * scale * magnitude
            })
            .collect::<Vec<_>>();
        (
            calibration_actual,
            calibration_prediction,
            test_actual,
            test_prediction,
        )
    }

    #[test]
    fn conformal_intervals_cover_heteroscedastic_synthetic_at_80_and_90_percent() {
        let (calibration_actual, calibration_prediction, test_actual, test_prediction) =
            heteroscedastic_fixture();

        for (alpha, expected) in [(0.2, 0.8), (0.1, 0.9)] {
            let mut calibrator = ConformalCalibrator::new(alpha).unwrap();
            calibrator
                .fit_with_strict_ordering(
                    &calibration_actual,
                    &calibration_prediction,
                    50,
                    50,
                    150,
                    150,
                )
                .unwrap();
            let interval = calibrator.predict_interval(&test_prediction, 150).unwrap();
            let coverage =
                interval_coverage(&test_actual, &interval.lower, &interval.upper).unwrap();

            assert!(
                coverage >= expected,
                "coverage {coverage} did not reach {expected}"
            );
        }
    }

    #[test]
    fn fitted_conformal_calibrator_and_interval_round_trip_exactly() {
        let (calibration_actual, calibration_prediction, _test_actual, test_prediction) =
            heteroscedastic_fixture();
        let mut calibrator = ConformalCalibrator::new(0.1).unwrap();
        calibrator
            .fit_with_strict_ordering(
                &calibration_actual,
                &calibration_prediction,
                50,
                50,
                150,
                150,
            )
            .unwrap();
        let interval = calibrator.predict_interval(&test_prediction, 150).unwrap();

        let decoded_calibrator: ConformalCalibrator =
            serde_json::from_str(&serde_json::to_string(&calibrator).unwrap()).unwrap();
        let decoded_interval: ConformalInterval =
            serde_json::from_str(&serde_json::to_string(&interval).unwrap()).unwrap();

        assert_eq!(decoded_calibrator, calibrator);
        assert_eq!(decoded_interval, interval);
        assert_eq!(
            decoded_calibrator
                .predict_interval(&test_prediction, 150)
                .unwrap(),
            interval
        );
    }

    #[test]
    fn calibration_arrays_must_match_the_declared_split() {
        let mut calibrator = ConformalCalibrator::new(0.2).expect("calibrator");
        let error = calibrator
            .fit_with_strict_ordering(&[1.0, 2.0], &[1.0, 2.0], 5, 5, 8, 8)
            .expect_err("declared three-row calibration split must reject two rows");

        assert!(error.to_string().contains("declared calibration split"));
    }

    #[test]
    fn finite_sample_rank_must_exist_for_the_requested_alpha() {
        let mut calibrator = ConformalCalibrator::new(0.1).expect("calibrator");
        let error = calibrator
            .fit_with_strict_ordering(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0], 4, 4, 8, 8)
            .expect_err("four residuals cannot support a finite 90% split-conformal interval");

        assert!(error.to_string().contains("insufficient"));
        assert!(calibrator.residual_quantile().is_none());
    }

    #[test]
    fn calibration_rejects_overflow_and_invalid_deserialized_alpha() {
        let mut calibrator = ConformalCalibrator::new(0.5).expect("calibrator");
        assert!(calibrator
            .fit_with_strict_ordering(&[f64::MAX], &[-f64::MAX], 1, 1, 2, 2)
            .is_err());
        assert!(calibrator.residual_quantile().is_none());
        let mut invalid: ConformalCalibrator = serde_json::from_value(serde_json::json!({
            "alpha": -0.5, "residual_quantile": null, "split_order": null
        }))
        .expect("deserialized state");
        assert!(invalid
            .fit_with_strict_ordering(&[1.0], &[1.0], 1, 1, 2, 2)
            .is_err());
    }

    #[test]
    fn interval_prediction_rejects_nonfinite_bounds_and_invalid_state() {
        let mut calibrator = ConformalCalibrator::new(0.5).expect("calibrator");
        calibrator
            .fit_with_strict_ordering(&[f64::MAX], &[0.0], 1, 1, 2, 2)
            .expect("finite residual");
        assert!(calibrator.predict_interval(&[f64::MAX], 2).is_err());
        calibrator.residual_quantile = Some(-1.0);
        assert!(calibrator.predict_interval(&[1.0], 2).is_err());
    }

    #[test]
    fn selected_conformal_quantile_matches_sorted_reference() {
        let actual = (0..101)
            .map(|index| ((index * 37) % 23) as f64)
            .collect::<Vec<_>>();
        let mut sorted = actual.clone();
        sorted.sort_by(f64::total_cmp);
        for alpha in [0.01, 0.1, 0.2, 0.5, 0.9] {
            let mut calibrator = ConformalCalibrator::new(alpha).expect("calibrator");
            calibrator
                .fit_with_strict_ordering(&actual, &vec![0.0; actual.len()], 1, 1, 102, 102)
                .expect("fit");
            let rank = ((102.0_f64 * (1.0 - alpha)).ceil() as usize).saturating_sub(1);
            assert_eq!(calibrator.residual_quantile(), Some(sorted[rank]));
        }
    }
}
