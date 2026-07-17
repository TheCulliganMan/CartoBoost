use crate::forecasting::{
    ForecastFrame, ForecastIntervalPrediction, ForecastPrediction, ForecastPredictionDetail,
    ForecastResult, Forecaster, RuleBasedGating,
};
use crate::{CartoBoostError, Result};
use cartoboost_accelerator::{
    backend_dense_layer_f32, select_backend_for, BackendOperation, BackendSelection,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const ENSEMBLE_DENSE_DISPATCH_MIN_OPS: usize = 16_384;

pub struct WeightedEnsembleForecaster {
    members: Vec<WeightedMember>,
    backend: BackendSelection,
}

pub type ForecastEnsemble = WeightedEnsembleForecaster;

struct WeightedMember {
    name: String,
    weight: f64,
    forecaster: Box<dyn Forecaster>,
}

pub struct GatedEnsembleForecaster {
    members: Vec<NamedMember>,
    gating: RuleBasedGating,
    weights: Option<BTreeMap<String, f64>>,
    backend: BackendSelection,
}

struct NamedMember {
    name: String,
    forecaster: Box<dyn Forecaster>,
}

#[derive(Debug, Clone, PartialEq)]
struct ForecastKey {
    series_id: String,
    timestamp: chrono::NaiveDateTime,
    horizon: usize,
}

impl Ord for ForecastKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.series_id
            .cmp(&other.series_id)
            .then_with(|| self.timestamp.cmp(&other.timestamp))
            .then_with(|| self.horizon.cmp(&other.horizon))
    }
}

impl PartialOrd for ForecastKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ForecastKey {}

#[derive(Debug, Clone, PartialEq)]
struct ForecastIntervalKey {
    forecast: ForecastKey,
    level: f64,
}

impl Eq for ForecastIntervalKey {}

impl WeightedEnsembleForecaster {
    pub fn new(members: Vec<(String, Box<dyn Forecaster>, f64)>) -> Result<Self> {
        Self::new_with_backend(members, Some("cpu"))
    }

    pub fn new_with_backend(
        members: Vec<(String, Box<dyn Forecaster>, f64)>,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = select_backend_for(backend, BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        if members.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "weighted ensemble requires at least one member".to_string(),
            ));
        }
        let mut cleaned = Vec::with_capacity(members.len());
        for (name, forecaster, weight) in members {
            if name.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "weighted ensemble member names must be non-empty".to_string(),
                ));
            }
            if cleaned
                .iter()
                .any(|member: &WeightedMember| member.name == name)
            {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate weighted ensemble member name '{name}'"
                )));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "weighted ensemble weights must be finite and non-negative".to_string(),
                ));
            }
            cleaned.push(WeightedMember {
                name,
                weight,
                forecaster,
            });
        }
        let max_weight = cleaned
            .iter()
            .map(|member| member.weight)
            .fold(0.0_f64, f64::max);
        if max_weight <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "weighted ensemble requires at least one positive weight".to_string(),
            ));
        }
        // Scale before summing so valid weights near f64::MAX cannot overflow
        // the normalization denominator.
        let scaled_total = cleaned
            .iter()
            .map(|member| member.weight / max_weight)
            .sum::<f64>();
        if !scaled_total.is_finite() || scaled_total <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "weighted ensemble weights could not be normalized".to_string(),
            ));
        }
        for member in &mut cleaned {
            member.weight = (member.weight / max_weight) / scaled_total;
        }
        Ok(Self {
            members: cleaned,
            backend,
        })
    }

    pub fn weights(&self) -> BTreeMap<String, f64> {
        self.members
            .iter()
            .map(|member| (member.name.clone(), member.weight))
            .collect()
    }
}

impl GatedEnsembleForecaster {
    pub fn new(
        members: Vec<(String, Box<dyn Forecaster>)>,
        gating: RuleBasedGating,
    ) -> Result<Self> {
        Self::new_with_backend(members, gating, Some("cpu"))
    }

    pub fn new_with_backend(
        members: Vec<(String, Box<dyn Forecaster>)>,
        gating: RuleBasedGating,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = select_backend_for(backend, BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        if members.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "gated ensemble requires at least one member".to_string(),
            ));
        }
        let mut cleaned = Vec::with_capacity(members.len());
        for (name, forecaster) in members {
            if name.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "gated ensemble member names must be non-empty".to_string(),
                ));
            }
            if cleaned
                .iter()
                .any(|member: &NamedMember| member.name == name)
            {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate gated ensemble member name '{name}'"
                )));
            }
            cleaned.push(NamedMember { name, forecaster });
        }
        Ok(Self {
            members: cleaned,
            gating,
            weights: None,
            backend,
        })
    }

    pub fn weights(&self) -> Option<&BTreeMap<String, f64>> {
        self.weights.as_ref()
    }

    fn weighted_result(
        &self,
        weights: &BTreeMap<String, f64>,
        horizon: usize,
    ) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        for expert in weights.keys() {
            if !self.members.iter().any(|member| &member.name == expert) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "gating weights reference unknown ensemble member '{expert}'"
                )));
            }
        }
        validate_normalized_weights(weights, "gating")?;
        let active_members = self
            .members
            .iter()
            .filter_map(|member| {
                let weight = weights.get(&member.name).copied().unwrap_or(0.0);
                (weight > 0.0).then_some((member, weight))
            })
            .collect::<Vec<_>>();
        let member_results = active_members
            .par_iter()
            .map(|(member, _)| member.forecaster.predict(horizon))
            .collect::<Result<Vec<_>>>()?;
        aggregate_member_results(
            active_members
                .into_iter()
                .zip(member_results)
                .map(|((member, weight), result)| (member.name.as_str(), weight, result))
                .collect(),
            self.model_name(),
            &self.backend,
        )
    }
}

impl Forecaster for WeightedEnsembleForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.members
            .par_iter_mut()
            .filter(|member| member.weight > 0.0)
            .map(|member| member.forecaster.fit(frame))
            .collect()
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let active_members = self
            .members
            .iter()
            .filter(|member| member.weight > 0.0)
            .collect::<Vec<_>>();
        let member_results = active_members
            .par_iter()
            .map(|member| member.forecaster.predict(horizon))
            .collect::<Result<Vec<_>>>()?;
        aggregate_member_results(
            active_members
                .into_iter()
                .zip(member_results)
                .map(|(member, result)| (member.name.as_str(), member.weight, result))
                .collect(),
            self.model_name(),
            &self.backend,
        )
    }

    fn model_name(&self) -> &'static str {
        "weighted_ensemble"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "weights": self.weights(),
            "backend": self.backend,
        })
    }
}

impl Forecaster for GatedEnsembleForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.members
            .par_iter_mut()
            .map(|member| member.forecaster.fit(frame))
            .collect::<Result<Vec<_>>>()?;
        let weights = self.gating.weights_for_frame(frame)?;
        for expert in weights.keys() {
            if !self.members.iter().any(|member| &member.name == expert) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "gating weights reference unknown ensemble member '{expert}'"
                )));
            }
        }
        validate_normalized_weights(&weights, "gating")?;
        self.weights = Some(weights);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            CartoBoostError::InvalidInput("gated ensemble must be fit before predict".to_string())
        })?;
        self.weighted_result(weights, horizon)
    }

    fn model_name(&self) -> &'static str {
        "gated_ensemble"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "weights": self.weights,
            "gating": self.gating.metadata(),
            "backend": self.backend,
        })
    }
}

fn validate_normalized_weights(weights: &BTreeMap<String, f64>, source: &str) -> Result<()> {
    if weights.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{source} produced no ensemble weights"
        )));
    }
    let mut total = 0.0;
    for (name, weight) in weights {
        if name.trim().is_empty() || !weight.is_finite() || *weight < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{source} ensemble weights must have non-empty names and finite non-negative values"
            )));
        }
        total += weight;
    }
    if !total.is_finite() || (total - 1.0).abs() > 1.0e-9 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{source} ensemble weights must sum to one; received {total}"
        )));
    }
    Ok(())
}

fn forecast_key(prediction: &ForecastPrediction) -> ForecastKey {
    ForecastKey {
        series_id: prediction.series_id.clone(),
        timestamp: prediction.timestamp,
        horizon: prediction.horizon,
    }
}

fn detail_forecast_key(detail: &ForecastPredictionDetail) -> ForecastKey {
    ForecastKey {
        series_id: detail.series_id.clone(),
        timestamp: detail.timestamp,
        horizon: detail.horizon,
    }
}

fn interval_key(interval: &ForecastIntervalPrediction) -> ForecastIntervalKey {
    ForecastIntervalKey {
        forecast: ForecastKey {
            series_id: interval.series_id.clone(),
            timestamp: interval.timestamp,
            horizon: interval.horizon,
        },
        level: interval.level,
    }
}

fn aggregate_member_results(
    member_results: Vec<(&str, f64, ForecastResult)>,
    model_name: &str,
    backend: &BackendSelection,
) -> Result<ForecastResult> {
    if member_results.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ensemble requires at least one positive-weight member".to_string(),
        ));
    }
    let member_count = member_results.len();

    let mut weighted: BTreeMap<ForecastKey, f64> = BTreeMap::new();
    let mut contributions: BTreeMap<ForecastKey, Vec<Value>> = BTreeMap::new();
    let mut expected_keys: Option<Vec<ForecastKey>> = None;
    let mut expected_interval_keys: Option<Vec<ForecastIntervalKey>> = None;
    let mut weighted_intervals: Vec<(ForecastIntervalKey, f64, f64)> = Vec::new();
    let mut has_member_details = false;
    let mut accelerator_columns = Vec::<Vec<f32>>::new();
    let mut accelerator_weights = Vec::<f32>::new();
    let mut use_accelerator = false;

    for (member_name, weight, result) in member_results {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "ensemble member '{member_name}' must have a finite positive aggregation weight"
            )));
        }
        let current_keys = result
            .predictions()
            .iter()
            .map(forecast_key)
            .collect::<Vec<_>>();
        if current_keys.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(CartoBoostError::InvalidInput(format!(
                "ensemble member '{member_name}' produced duplicate forecast index values"
            )));
        }
        if let Some(expected) = &expected_keys {
            if expected != &current_keys {
                return Err(CartoBoostError::InvalidInput(format!(
                    "ensemble member '{member_name}' produced a mismatched forecast index"
                )));
            }
        } else {
            use_accelerator = should_accelerate_ensemble(backend, current_keys.len(), member_count);
            expected_keys = Some(current_keys);
        }

        let details = result
            .details()
            .iter()
            .map(|detail| {
                Ok((
                    detail_forecast_key(detail),
                    serde_json::to_value(detail).map_err(CartoBoostError::from)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        has_member_details |= !details.is_empty();
        if use_accelerator {
            accelerator_columns.push(
                result
                    .predictions()
                    .iter()
                    .map(|prediction| prediction.mean as f32)
                    .collect(),
            );
            accelerator_weights.push(weight as f32);
        }
        for prediction in result.predictions() {
            let key = forecast_key(prediction);
            let weighted_mean = weight * prediction.mean;
            *weighted.entry(key.clone()).or_insert(0.0) += weighted_mean;
            contributions.entry(key.clone()).or_default().push(json!({
                "member": member_name,
                "weight": weight,
                "mean": prediction.mean,
                "weighted_mean": weighted_mean,
                "detail": details.get(&key),
            }));
        }

        let current_interval_keys = result
            .intervals()
            .iter()
            .map(interval_key)
            .collect::<Vec<_>>();
        if let Some(expected) = &expected_interval_keys {
            if expected != &current_interval_keys {
                return Err(CartoBoostError::InvalidInput(format!(
                    "ensemble member '{member_name}' produced mismatched interval levels or indices"
                )));
            }
            for (aggregate, interval) in weighted_intervals.iter_mut().zip(result.intervals()) {
                aggregate.1 += weight * interval.lower;
                aggregate.2 += weight * interval.upper;
            }
        } else {
            weighted_intervals = result
                .intervals()
                .iter()
                .map(|interval| {
                    (
                        interval_key(interval),
                        weight * interval.lower,
                        weight * interval.upper,
                    )
                })
                .collect();
            expected_interval_keys = Some(current_interval_keys);
        }
    }

    if use_accelerator {
        let keys = expected_keys.as_ref().expect("validated ensemble keys");
        let features = (0..keys.len())
            .map(|row| {
                accelerator_columns
                    .iter()
                    .map(|column| column[row])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let accelerated = backend_dense_layer_f32(backend, &features, &accelerator_weights, &[0.0])
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        for (key, output) in keys.iter().cloned().zip(accelerated) {
            weighted.insert(key, f64::from(output[0]));
        }
    }

    let predictions = weighted
        .into_iter()
        .map(|(key, mean)| ForecastPrediction {
            series_id: key.series_id,
            timestamp: key.timestamp,
            horizon: key.horizon,
            model: model_name.to_string(),
            mean,
        })
        .collect::<Vec<_>>();
    let intervals = weighted_intervals
        .into_iter()
        .map(|(key, lower, upper)| ForecastIntervalPrediction {
            series_id: key.forecast.series_id,
            timestamp: key.forecast.timestamp,
            horizon: key.forecast.horizon,
            model: model_name.to_string(),
            level: key.level,
            lower,
            upper,
        })
        .collect::<Vec<_>>();
    let details = if has_member_details {
        contributions
            .into_iter()
            .map(|(key, members)| ForecastPredictionDetail {
                series_id: key.series_id,
                timestamp: key.timestamp,
                horizon: key.horizon,
                model: model_name.to_string(),
                base_mean: None,
                spatial_correction: None,
                kriging_variance: None,
                selected_neighbors: Vec::new(),
                component_decomposition: Some(json!({
                    "aggregation": "weighted_member_contributions",
                    "members": members,
                })),
                metadata: Some(json!({
                    "interval_aggregation": "weighted_quantile_average",
                })),
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
}

fn should_accelerate_ensemble(
    backend: &BackendSelection,
    forecast_count: usize,
    member_count: usize,
) -> bool {
    backend.selected != "cpu"
        && forecast_count.saturating_mul(member_count) >= ENSEMBLE_DENSE_DISPATCH_MIN_OPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{
        ExpertScore, ForecastFrequency, ForecastRow, NaiveForecaster, SeasonalNaiveForecaster,
        ValidationScoreTable,
    };
    use chrono::NaiveDate;
    use serde_json::Value;

    struct FixedForecaster {
        predictions: Vec<ForecastPrediction>,
        name: &'static str,
    }

    impl Forecaster for FixedForecaster {
        fn fit(&mut self, _frame: &ForecastFrame) -> Result<()> {
            Ok(())
        }

        fn predict(&self, _horizon: usize) -> Result<ForecastResult> {
            ForecastResult::new(self.predictions.clone())
        }

        fn model_name(&self) -> &'static str {
            self.name
        }

        fn metadata(&self) -> Value {
            json!({"model": self.name})
        }
    }

    struct RichFixedForecaster {
        result: ForecastResult,
        name: &'static str,
    }

    impl Forecaster for RichFixedForecaster {
        fn fit(&mut self, _frame: &ForecastFrame) -> Result<()> {
            Ok(())
        }

        fn predict(&self, _horizon: usize) -> Result<ForecastResult> {
            Ok(self.result.clone())
        }

        fn model_name(&self) -> &'static str {
            self.name
        }

        fn metadata(&self) -> Value {
            json!({"model": self.name})
        }
    }

    #[test]
    fn weighted_ensemble_averages_forecast_means() {
        let rows = vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 12.0),
            ForecastRow::single(ts(3), 14.0),
        ];
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
        let mut ensemble = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
            (
                "seasonal".to_string(),
                Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
                3.0,
            ),
        ])
        .expect("ensemble");

        ensemble.fit(&frame).expect("fit");
        let result = ensemble.predict(2).expect("predict");
        let means: Vec<f64> = result
            .predictions()
            .iter()
            .map(|prediction| prediction.mean)
            .collect();

        assert_eq!(means, vec![12.5, 14.0]);
    }

    #[test]
    fn accelerator_selection_preserves_small_ensemble_results() {
        for backend in cartoboost_accelerator::available_backends() {
            if backend == "cpu"
                || !cartoboost_accelerator::backend_supports_operation(
                    &backend,
                    BackendOperation::Dense,
                )
            {
                continue;
            }
            let ensemble = WeightedEnsembleForecaster::new_with_backend(
                vec![
                    (
                        "left".to_string(),
                        Box::new(FixedForecaster {
                            predictions: vec![prediction("PU1", 4, 1, 10.0)],
                            name: "left",
                        }),
                        0.25,
                    ),
                    (
                        "right".to_string(),
                        Box::new(FixedForecaster {
                            predictions: vec![prediction("PU1", 4, 1, 30.0)],
                            name: "right",
                        }),
                        0.75,
                    ),
                ],
                Some(&backend),
            )
            .unwrap();
            let result = ensemble.predict(1).unwrap();
            assert!(
                (result.predictions()[0].mean - 25.0).abs() < 1.0e-4,
                "backend {backend}"
            );
        }
    }

    #[test]
    fn ensemble_dense_dispatch_requires_a_profitable_workload() {
        for backend in cartoboost_accelerator::available_backends() {
            let selection =
                select_backend_for(Some(&backend), BackendOperation::Dense).expect("selection");
            assert!(!should_accelerate_ensemble(&selection, 1, 2));
            assert_eq!(
                should_accelerate_ensemble(&selection, 8_192, 2),
                backend != "cpu"
            );
        }
    }

    #[test]
    fn large_accelerated_weighted_ensemble_matches_cpu() {
        let left = (0..8_192)
            .map(|index| prediction(&format!("lane-{index:05}"), 4, 1, index as f64 * 0.25))
            .collect::<Vec<_>>();
        let right = (0..8_192)
            .map(|index| {
                prediction(
                    &format!("lane-{index:05}"),
                    4,
                    1,
                    100.0 - index as f64 * 0.125,
                )
            })
            .collect::<Vec<_>>();
        let cpu = WeightedEnsembleForecaster::new_with_backend(
            vec![
                (
                    "left".to_string(),
                    Box::new(FixedForecaster {
                        predictions: left.clone(),
                        name: "left",
                    }),
                    0.25,
                ),
                (
                    "right".to_string(),
                    Box::new(FixedForecaster {
                        predictions: right.clone(),
                        name: "right",
                    }),
                    0.75,
                ),
            ],
            Some("cpu"),
        )
        .unwrap()
        .predict(1)
        .unwrap();
        for backend in cartoboost_accelerator::available_backends()
            .into_iter()
            .filter(|backend| backend != "cpu")
        {
            let accelerated = WeightedEnsembleForecaster::new_with_backend(
                vec![
                    (
                        "left".to_string(),
                        Box::new(FixedForecaster {
                            predictions: left.clone(),
                            name: "left",
                        }),
                        0.25,
                    ),
                    (
                        "right".to_string(),
                        Box::new(FixedForecaster {
                            predictions: right.clone(),
                            name: "right",
                        }),
                        0.75,
                    ),
                ],
                Some(&backend),
            )
            .unwrap()
            .predict(1)
            .unwrap();
            for (expected, actual) in cpu.predictions().iter().zip(accelerated.predictions()) {
                assert!(
                    (expected.mean - actual.mean).abs() < 1.0e-4,
                    "{backend}: {} != {}",
                    expected.mean,
                    actual.mean
                );
            }
        }
    }

    #[test]
    fn weighted_ensemble_rejects_invalid_weights() {
        let err = WeightedEnsembleForecaster::new(vec![(
            "last".to_string(),
            Box::new(NaiveForecaster::new()),
            0.0,
        )])
        .err()
        .expect("invalid weights");

        assert!(err.to_string().contains("at least one positive weight"));
    }

    #[test]
    fn weighted_ensemble_rejects_duplicate_member_names() {
        let err = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
        ])
        .err()
        .expect("duplicate name");

        assert!(err
            .to_string()
            .contains("duplicate weighted ensemble member name"));
    }

    #[test]
    fn weighted_ensemble_aligns_panel_forecasts() {
        let rows = vec![
            ForecastRow::new("PU1->DO2", ts(1), 10.0),
            ForecastRow::new("PU1->DO2", ts(2), 12.0),
            ForecastRow::new("PU1->DO2", ts(3), 14.0),
            ForecastRow::new("PU9->DO8", ts(1), 30.0),
            ForecastRow::new("PU9->DO8", ts(2), 28.0),
            ForecastRow::new("PU9->DO8", ts(3), 26.0),
        ];
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
        let mut ensemble = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 0.5),
            (
                "seasonal".to_string(),
                Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
                0.5,
            ),
        ])
        .expect("ensemble");

        ensemble.fit(&frame).expect("fit");
        let result = ensemble.predict(1).expect("predict");
        let predictions = result.predictions();

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].series_id, "PU1->DO2");
        assert_eq!(predictions[0].mean, 13.0);
        assert_eq!(predictions[1].series_id, "PU9->DO8");
        assert_eq!(predictions[1].mean, 27.0);
    }

    #[test]
    fn weighted_ensemble_rejects_mismatched_forecast_index() {
        let first = FixedForecaster {
            name: "first",
            predictions: vec![prediction("PU1->DO2", 4, 1, 10.0)],
        };
        let second = FixedForecaster {
            name: "second",
            predictions: vec![prediction("PU9->DO8", 4, 1, 20.0)],
        };
        let ensemble = WeightedEnsembleForecaster::new(vec![
            ("first".to_string(), Box::new(first), 1.0),
            ("second".to_string(), Box::new(second), 1.0),
        ])
        .expect("ensemble");

        let err = ensemble.predict(1).expect_err("mismatched index");

        assert!(err.to_string().contains("mismatched forecast index"));
    }

    #[test]
    fn weighted_ensemble_metadata_exposes_normalized_weights() {
        let ensemble = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
            (
                "seasonal".to_string(),
                Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
                3.0,
            ),
        ])
        .expect("ensemble");
        let metadata = ensemble.metadata();

        assert_eq!(metadata["model"], "weighted_ensemble");
        assert_eq!(metadata["weights"]["last"], 0.25);
        assert_eq!(metadata["weights"]["seasonal"], 0.75);
    }

    #[test]
    fn weighted_ensemble_normalizes_extreme_finite_weights_without_overflow() {
        let ensemble = WeightedEnsembleForecaster::new(vec![
            (
                "first".to_string(),
                Box::new(NaiveForecaster::new()),
                f64::MAX,
            ),
            (
                "second".to_string(),
                Box::new(NaiveForecaster::new()),
                f64::MAX,
            ),
        ])
        .expect("finite weights are normalizable");

        assert_eq!(ensemble.weights()["first"], 0.5);
        assert_eq!(ensemble.weights()["second"], 0.5);
    }

    #[test]
    fn weighted_ensemble_aggregates_intervals_and_preserves_member_details() {
        let first = rich_result("first", 10.0, 8.0, 12.0, 9.0, 1.0);
        let second = rich_result("second", 20.0, 16.0, 24.0, 18.0, 2.0);
        let ensemble = WeightedEnsembleForecaster::new(vec![
            (
                "first".to_string(),
                Box::new(RichFixedForecaster {
                    result: first,
                    name: "first",
                }),
                1.0,
            ),
            (
                "second".to_string(),
                Box::new(RichFixedForecaster {
                    result: second,
                    name: "second",
                }),
                3.0,
            ),
        ])
        .expect("ensemble");

        let result = ensemble.predict(1).expect("predict");

        assert_eq!(result.predictions()[0].mean, 17.5);
        assert_eq!(result.intervals().len(), 1);
        assert_eq!(result.intervals()[0].lower, 14.0);
        assert_eq!(result.intervals()[0].upper, 21.0);
        let decomposition = result.details()[0]
            .component_decomposition
            .as_ref()
            .expect("member contributions");
        assert_eq!(
            decomposition["aggregation"],
            "weighted_member_contributions"
        );
        let members = decomposition["members"].as_array().expect("members");
        assert_eq!(members.len(), 2);
        assert_eq!(members[0]["detail"]["base_mean"], 9.0);
        assert_eq!(members[1]["detail"]["spatial_correction"], 2.0);
    }

    #[test]
    fn weighted_ensemble_rejects_partial_interval_coverage() {
        let rich = rich_result("first", 10.0, 8.0, 12.0, 9.0, 1.0);
        let plain =
            ForecastResult::new(vec![prediction("PU1->DO2", 4, 1, 20.0)]).expect("plain result");
        let ensemble = WeightedEnsembleForecaster::new(vec![
            (
                "first".to_string(),
                Box::new(RichFixedForecaster {
                    result: rich,
                    name: "first",
                }),
                1.0,
            ),
            (
                "second".to_string(),
                Box::new(RichFixedForecaster {
                    result: plain,
                    name: "second",
                }),
                1.0,
            ),
        ])
        .expect("ensemble");

        let error = ensemble
            .predict(1)
            .expect_err("partial interval grids must not be silently discarded");
        assert!(error
            .to_string()
            .contains("mismatched interval levels or indices"));
    }

    #[test]
    fn gated_ensemble_top_k_can_select_a_strict_member_subset() {
        let table = ValidationScoreTable::new(vec![
            ExpertScore::global("first", "rmse", 1.0),
            ExpertScore::global("second", "rmse", 2.0),
        ])
        .expect("score table");
        let gating = RuleBasedGating::with_options("rmse", table, 1.0e-9, Some(1)).expect("gating");
        let mut ensemble = GatedEnsembleForecaster::new(
            vec![
                (
                    "first".to_string(),
                    Box::new(FixedForecaster {
                        predictions: vec![prediction("PU1->DO2", 4, 1, 10.0)],
                        name: "first",
                    }) as Box<dyn Forecaster>,
                ),
                (
                    "second".to_string(),
                    Box::new(FixedForecaster {
                        predictions: vec![prediction("PU1->DO2", 4, 1, 20.0)],
                        name: "second",
                    }) as Box<dyn Forecaster>,
                ),
            ],
            gating,
        )
        .expect("ensemble");
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("frame");

        ensemble.fit(&frame).expect("fit");
        let result = ensemble.predict(1).expect("predict");

        assert_eq!(
            ensemble.weights().unwrap(),
            &BTreeMap::from([("first".to_string(), 1.0)])
        );
        assert_eq!(result.predictions()[0].mean, 10.0);
    }

    fn rich_result(
        model: &str,
        mean: f64,
        lower: f64,
        upper: f64,
        base_mean: f64,
        spatial_correction: f64,
    ) -> ForecastResult {
        let prediction = ForecastPrediction {
            model: model.to_string(),
            ..prediction("PU1->DO2", 4, 1, mean)
        };
        ForecastResult::new_with_intervals_and_details(
            vec![prediction],
            vec![ForecastIntervalPrediction {
                series_id: "PU1->DO2".to_string(),
                timestamp: ts(4),
                horizon: 1,
                model: model.to_string(),
                level: 0.8,
                lower,
                upper,
            }],
            vec![ForecastPredictionDetail {
                series_id: "PU1->DO2".to_string(),
                timestamp: ts(4),
                horizon: 1,
                model: model.to_string(),
                base_mean: Some(base_mean),
                spatial_correction: Some(spatial_correction),
                kriging_variance: Some(4.0),
                selected_neighbors: vec![format!("{model}_neighbor")],
                component_decomposition: Some(json!({"trend": base_mean})),
                metadata: Some(json!({"source": model})),
            }],
        )
        .expect("rich forecast result")
    }

    fn prediction(series_id: &str, day: u32, horizon: usize, mean: f64) -> ForecastPrediction {
        ForecastPrediction {
            series_id: series_id.to_string(),
            timestamp: ts(day),
            horizon,
            model: "fixed".to_string(),
            mean,
        }
    }

    fn ts(day: u32) -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, day)
            .expect("date")
            .and_hms_opt(0, 0, 0)
            .expect("time")
    }
}
