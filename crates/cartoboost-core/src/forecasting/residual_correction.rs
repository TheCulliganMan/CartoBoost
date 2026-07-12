use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResidualStateKey {
    pub origin: String,
    pub destination: String,
    pub corridor: String,
    pub segment: String,
    pub entity_family: String,
    pub target_family: String,
    pub time_bucket: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "UncheckedStateFilter")]
pub struct StateFilter {
    pub mean: f64,
    pub variance: f64,
    pub process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct UncheckedStateFilter {
    mean: f64,
    variance: f64,
    process_variance: f64,
    observation_variance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatePrediction {
    pub key: ResidualStateKey,
    pub structural_prediction: f64,
    pub residual_adjustment: f64,
    pub corrected_prediction: f64,
    pub residual_variance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateObservation {
    pub key: ResidualStateKey,
    pub structural_prediction: f64,
    pub observed: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateCorrection {
    pub prediction: StatePrediction,
    pub updated: bool,
    pub residual: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanResidualCorrector {
    pub default_filter: StateFilter,
    #[serde(with = "state_map_serde")]
    pub states: BTreeMap<ResidualStateKey, StateFilter>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateCorrectedBooster {
    pub corrector: KalmanResidualCorrector,
}

impl ResidualStateKey {
    pub fn new(
        origin: impl Into<String>,
        destination: impl Into<String>,
        corridor: impl Into<String>,
        segment: impl Into<String>,
        entity_family: impl Into<String>,
        target_family: impl Into<String>,
        time_bucket: impl Into<String>,
    ) -> Self {
        Self {
            origin: origin.into(),
            destination: destination.into(),
            corridor: corridor.into(),
            segment: segment.into(),
            entity_family: entity_family.into(),
            target_family: target_family.into(),
            time_bucket: time_bucket.into(),
        }
    }
}

impl StateFilter {
    pub fn new(process_variance: f64, observation_variance: f64) -> Result<Self> {
        let filter = Self {
            mean: 0.0,
            variance: observation_variance,
            process_variance,
            observation_variance,
        };
        filter.validate()?;
        Ok(filter)
    }

    pub fn with_state(
        mean: f64,
        variance: f64,
        process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let filter = Self {
            mean,
            variance,
            process_variance,
            observation_variance,
        };
        filter.validate()?;
        Ok(filter)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.mean.is_finite() || !self.variance.is_finite() || self.variance < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "state filter mean and variance must be finite with variance >= 0".to_string(),
            ));
        }
        if !self.process_variance.is_finite()
            || self.process_variance < 0.0
            || !self.observation_variance.is_finite()
            || self.observation_variance <= 0.0
        {
            return Err(CartoBoostError::InvalidInput(
                "state filter variances must be finite, process >= 0, observation > 0".to_string(),
            ));
        }
        Ok(())
    }

    pub fn predict(&self) -> Result<Self> {
        self.validate()?;
        let variance = self.variance + self.process_variance;
        if !variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "state filter prediction variance overflowed".to_string(),
            ));
        }
        let predicted = Self {
            mean: self.mean,
            variance,
            process_variance: self.process_variance,
            observation_variance: self.observation_variance,
        };
        predicted.validate()?;
        Ok(predicted)
    }

    pub fn update(&self, residual: f64) -> Result<Self> {
        if !residual.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "residual observation must be finite".to_string(),
            ));
        }
        let predicted = self.predict()?;
        let innovation_variance = predicted.variance + predicted.observation_variance;
        if !innovation_variance.is_finite() || innovation_variance <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "state filter innovation variance must be finite and positive".to_string(),
            ));
        }
        let gain = predicted.variance / innovation_variance;
        let innovation = residual - predicted.mean;
        if !gain.is_finite() || !innovation.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "state filter update produced a non-finite gain or innovation".to_string(),
            ));
        }
        let posterior_mean = predicted.mean + gain * innovation;
        let retained = 1.0 - gain;
        // Joseph's scalar covariance update is robust to roundoff and preserves
        // non-negative variance for a valid gain.
        let posterior_variance =
            retained * retained * predicted.variance + gain * gain * predicted.observation_variance;
        let updated = Self {
            mean: posterior_mean,
            variance: posterior_variance,
            process_variance: self.process_variance,
            observation_variance: self.observation_variance,
        };
        updated.validate()?;
        Ok(updated)
    }
}

impl TryFrom<UncheckedStateFilter> for StateFilter {
    type Error = CartoBoostError;

    fn try_from(value: UncheckedStateFilter) -> Result<Self> {
        Self::with_state(
            value.mean,
            value.variance,
            value.process_variance,
            value.observation_variance,
        )
    }
}

impl KalmanResidualCorrector {
    pub fn new(default_filter: StateFilter) -> Self {
        Self {
            default_filter,
            states: BTreeMap::new(),
        }
    }

    pub fn predict(
        &self,
        key: &ResidualStateKey,
        structural_prediction: f64,
    ) -> Result<StatePrediction> {
        if !structural_prediction.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "structural prediction must be finite".to_string(),
            ));
        }
        self.default_filter.validate()?;
        let filter = self
            .states
            .get(key)
            .copied()
            .unwrap_or(self.default_filter)
            .predict()?;
        let corrected_prediction = structural_prediction + filter.mean;
        if !corrected_prediction.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "state-corrected prediction overflowed".to_string(),
            ));
        }
        Ok(StatePrediction {
            key: key.clone(),
            structural_prediction,
            residual_adjustment: filter.mean,
            corrected_prediction,
            residual_variance: filter.variance,
        })
    }

    pub fn update(
        &mut self,
        key: ResidualStateKey,
        structural_prediction: f64,
        observed: Option<f64>,
    ) -> Result<StateCorrection> {
        let prediction = self.predict(&key, structural_prediction)?;
        let Some(observed) = observed else {
            let predicted_filter = self
                .states
                .get(&key)
                .copied()
                .unwrap_or(self.default_filter)
                .predict()?;
            self.states.insert(key, predicted_filter);
            return Ok(StateCorrection {
                prediction,
                updated: false,
                residual: None,
            });
        };
        if !observed.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "observed value must be finite when present".to_string(),
            ));
        }
        let residual = observed - structural_prediction;
        let current = self
            .states
            .get(&key)
            .copied()
            .unwrap_or(self.default_filter);
        let updated = current.update(residual)?;
        self.states.insert(key, updated);
        Ok(StateCorrection {
            prediction,
            updated: true,
            residual: Some(residual),
        })
    }

    pub fn apply_sequence(
        &mut self,
        observations: &[StateObservation],
    ) -> Result<Vec<StateCorrection>> {
        observations
            .iter()
            .map(|observation| {
                self.update(
                    observation.key.clone(),
                    observation.structural_prediction,
                    observation.observed,
                )
            })
            .collect()
    }
}

impl StateCorrectedBooster {
    pub fn new(corrector: KalmanResidualCorrector) -> Self {
        Self { corrector }
    }

    pub fn predict(
        &self,
        key: &ResidualStateKey,
        structural_prediction: f64,
    ) -> Result<StatePrediction> {
        self.corrector.predict(key, structural_prediction)
    }

    pub fn predict_update(
        &mut self,
        key: ResidualStateKey,
        structural_prediction: f64,
        observed: Option<f64>,
    ) -> Result<StateCorrection> {
        self.corrector.update(key, structural_prediction, observed)
    }
}

mod state_map_serde {
    use super::{ResidualStateKey, StateFilter};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S>(
        states: &BTreeMap<ResidualStateKey, StateFilter>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        states.iter().collect::<Vec<_>>().serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<ResidualStateKey, StateFilter>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<(ResidualStateKey, StateFilter)>::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(bucket: &str) -> ResidualStateKey {
        ResidualStateKey::new(
            "PULocationID=132",
            "DOLocationID=237",
            "132->237",
            "airport_lane",
            "corridor",
            "fare",
            bucket,
        )
    }

    #[test]
    fn predict_before_update_prevents_target_leakage() {
        let filter = StateFilter::new(0.1, 0.1).unwrap();
        let mut corrector = KalmanResidualCorrector::new(filter);
        let key = key("hour=08");

        let correction = corrector.update(key.clone(), 10.0, Some(20.0)).unwrap();

        assert_eq!(correction.prediction.corrected_prediction, 10.0);
        assert_eq!(correction.residual, Some(10.0));
        let next = corrector.predict(&key, 10.0).unwrap();
        assert!(next.corrected_prediction > 10.0);
    }

    #[test]
    fn missing_observations_advance_and_persist_process_uncertainty() {
        let filter = StateFilter::with_state(2.0, 1.0, 0.25, 0.5).unwrap();
        let mut corrector = KalmanResidualCorrector::new(filter);
        let key = key("hour=09");

        let first = corrector.update(key.clone(), 10.0, None).unwrap();

        assert!(!first.updated);
        assert!(first.residual.is_none());
        assert_eq!(first.prediction.residual_adjustment, 2.0);
        assert_eq!(first.prediction.residual_variance, 1.25);
        assert_eq!(corrector.states[&key].mean, 2.0);
        assert_eq!(corrector.states[&key].variance, 1.25);

        let second = corrector.update(key.clone(), 10.0, None).unwrap();

        assert!(!second.updated);
        assert_eq!(second.prediction.residual_variance, 1.5);
        assert_eq!(corrector.states[&key].variance, 1.5);

        // Read-only prediction projects one step but does not mutate the clock.
        let projected = corrector.predict(&key, 10.0).unwrap();
        assert_eq!(projected.residual_variance, 1.75);
        assert_eq!(corrector.states[&key].variance, 1.5);
    }

    #[test]
    fn state_filter_update_matches_scalar_kalman_equations() {
        let filter = StateFilter::with_state(2.0, 3.0, 1.0, 4.0).unwrap();

        let updated = filter.update(10.0).unwrap();

        assert_eq!(updated.mean, 6.0);
        assert_eq!(updated.variance, 2.0);
    }

    #[test]
    fn public_state_filter_fields_cannot_bypass_runtime_validation() {
        let invalid_filters = [
            StateFilter {
                mean: f64::NAN,
                variance: 1.0,
                process_variance: 0.1,
                observation_variance: 1.0,
            },
            StateFilter {
                mean: 0.0,
                variance: -1.0,
                process_variance: 0.1,
                observation_variance: 1.0,
            },
            StateFilter {
                mean: 0.0,
                variance: 1.0,
                process_variance: -0.1,
                observation_variance: 1.0,
            },
            StateFilter {
                mean: 0.0,
                variance: 1.0,
                process_variance: 0.1,
                observation_variance: 0.0,
            },
        ];

        for filter in invalid_filters {
            assert!(filter.validate().is_err());
            assert!(filter.predict().is_err());
            assert!(filter.update(1.0).is_err());
        }
    }

    #[test]
    fn state_filter_rejects_numerical_overflow() {
        let overflowing_variance = StateFilter::with_state(0.0, f64::MAX, f64::MAX, 1.0).unwrap();
        assert!(overflowing_variance.predict().is_err());

        let overflowing_innovation = StateFilter::with_state(-f64::MAX, 1.0, 0.0, 1.0).unwrap();
        assert!(overflowing_innovation.update(f64::MAX).is_err());

        let overflowing_prediction = StateFilter::with_state(f64::MAX, 1.0, 0.0, 1.0).unwrap();
        let corrector = KalmanResidualCorrector::new(overflowing_prediction);
        assert!(corrector.predict(&key("hour=overflow"), f64::MAX).is_err());
    }

    #[test]
    fn deserialization_rejects_invalid_state_filters() {
        let serialized = r#"{
            "mean": 0.0,
            "variance": -1.0,
            "process_variance": 0.1,
            "observation_variance": 1.0
        }"#;

        assert!(serde_json::from_str::<StateFilter>(serialized).is_err());
    }

    #[test]
    fn distinct_state_keys_do_not_bleed_across_families() {
        let filter = StateFilter::new(0.1, 0.1).unwrap();
        let mut corrector = KalmanResidualCorrector::new(filter);
        let fare = key("hour=10");
        let demand = ResidualStateKey::new(
            "PULocationID=132",
            "DOLocationID=237",
            "132->237",
            "airport_lane",
            "corridor",
            "demand",
            "hour=10",
        );

        corrector.update(fare.clone(), 10.0, Some(20.0)).unwrap();
        let demand_prediction = corrector.predict(&demand, 10.0).unwrap();

        assert_eq!(demand_prediction.corrected_prediction, 10.0);
    }

    #[test]
    fn state_keys_isolate_every_geotemporal_dimension() {
        let filter = StateFilter::new(0.1, 0.1).unwrap();
        let mut corrector = KalmanResidualCorrector::new(filter);
        let base = ResidualStateKey::new(
            "PULocationID=132",
            "DOLocationID=237",
            "132->237",
            "airport_lane",
            "corridor",
            "fare",
            "hour=10",
        );
        corrector.update(base.clone(), 100.0, Some(130.0)).unwrap();
        assert!(
            corrector
                .predict(&base, 100.0)
                .unwrap()
                .corrected_prediction
                > 100.0
        );

        let variants = [
            ResidualStateKey::new(
                "PULocationID=138",
                "DOLocationID=237",
                "132->237",
                "airport_lane",
                "corridor",
                "fare",
                "hour=10",
            ),
            ResidualStateKey::new(
                "PULocationID=132",
                "DOLocationID=161",
                "132->237",
                "airport_lane",
                "corridor",
                "fare",
                "hour=10",
            ),
            ResidualStateKey::new(
                "PULocationID=132",
                "DOLocationID=237",
                "132->161",
                "airport_lane",
                "corridor",
                "fare",
                "hour=10",
            ),
            ResidualStateKey::new(
                "PULocationID=132",
                "DOLocationID=237",
                "132->237",
                "midtown_lane",
                "corridor",
                "fare",
                "hour=10",
            ),
            ResidualStateKey::new(
                "PULocationID=132",
                "DOLocationID=237",
                "132->237",
                "airport_lane",
                "segment",
                "fare",
                "hour=10",
            ),
            ResidualStateKey::new(
                "PULocationID=132",
                "DOLocationID=237",
                "132->237",
                "airport_lane",
                "corridor",
                "duration",
                "hour=10",
            ),
            ResidualStateKey::new(
                "PULocationID=132",
                "DOLocationID=237",
                "132->237",
                "airport_lane",
                "corridor",
                "fare",
                "hour=11",
            ),
        ];

        for variant in variants {
            assert_eq!(
                corrector
                    .predict(&variant, 100.0)
                    .unwrap()
                    .corrected_prediction,
                100.0
            );
        }
    }

    #[test]
    fn state_correction_lowers_post_shift_error_vs_static_predictions() {
        let filter = StateFilter::new(0.5, 0.2).unwrap();
        let mut corrector = KalmanResidualCorrector::new(filter);
        let key = key("hour=11");
        let observations = (0..20)
            .map(|idx| StateObservation {
                key: key.clone(),
                structural_prediction: 10.0,
                observed: Some(if idx < 5 { 10.0 } else { 15.0 }),
            })
            .collect::<Vec<_>>();
        let corrections = corrector.apply_sequence(&observations).unwrap();
        let static_abs_error = observations[10..]
            .iter()
            .map(|row| (row.observed.unwrap() - row.structural_prediction).abs())
            .sum::<f64>();
        let corrected_abs_error = corrections[10..]
            .iter()
            .zip(&observations[10..])
            .map(|(correction, row)| {
                (row.observed.unwrap() - correction.prediction.corrected_prediction).abs()
            })
            .sum::<f64>();

        assert!(corrected_abs_error < static_abs_error);
    }

    #[test]
    fn state_corrected_booster_predicts_structural_plus_residual_adjustment() {
        let filter = StateFilter::new(0.1, 0.1).unwrap();
        let mut booster = StateCorrectedBooster::new(KalmanResidualCorrector::new(filter));
        let key = key("hour=13");

        let first = booster
            .predict_update(key.clone(), 50.0, Some(60.0))
            .unwrap();
        assert_eq!(first.prediction.corrected_prediction, 50.0);
        assert!(first.updated);

        let second = booster.predict(&key, 50.0).unwrap();
        assert!(second.residual_adjustment > 0.0);
        assert_eq!(
            second.corrected_prediction,
            second.structural_prediction + second.residual_adjustment
        );
    }

    #[test]
    fn residual_corrector_serializes_exactly() {
        let filter = StateFilter::new(0.1, 0.2).unwrap();
        let mut corrector = KalmanResidualCorrector::new(filter);
        corrector
            .update(key("hour=12"), 100.0, Some(104.0))
            .unwrap();

        let text = serde_json::to_string(&corrector).unwrap();
        let restored: KalmanResidualCorrector = serde_json::from_str(&text).unwrap();

        assert_eq!(restored, corrector);
    }
}
