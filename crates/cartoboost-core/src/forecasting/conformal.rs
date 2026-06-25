use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

use super::quantiles::{validate_finite_values, validate_quantile, validate_same_non_empty};

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
        split_order.validate()?;
        validate_same_non_empty(
            calibration_actual,
            calibration_prediction,
            "calibration_actual",
            "calibration_prediction",
        )?;
        let mut scores = calibration_actual
            .iter()
            .zip(calibration_prediction)
            .map(|(&actual, &prediction)| (actual - prediction).abs())
            .collect::<Vec<_>>();
        scores.sort_by(f64::total_cmp);
        let rank = (((scores.len() + 1) as f64) * (1.0 - self.alpha)).ceil() as usize;
        let index = rank.saturating_sub(1).min(scores.len() - 1);
        self.residual_quantile = Some(scores[index]);
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
        if test_start < order.test_start {
            return Err(CartoBoostError::InvalidInput(
                "test_start must not precede the validated test split".to_string(),
            ));
        }
        let residual_quantile = self.residual_quantile.ok_or_else(|| {
            CartoBoostError::InvalidInput("conformal calibrator has no residual quantile".into())
        })?;
        Ok(ConformalInterval {
            lower: test_prediction
                .iter()
                .map(|prediction| prediction - residual_quantile)
                .collect(),
            upper: test_prediction
                .iter()
                .map(|prediction| prediction + residual_quantile)
                .collect(),
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
}
