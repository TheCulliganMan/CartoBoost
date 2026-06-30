use crate::booster::{Booster, BoosterConfig};
use crate::data::{Dataset, FeatureKind, FeatureSchema};
use crate::forecasting::lag_features::{
    history_by_series, lag_config_supported_by_history, LagFeatureBuilder, LagFeatureConfig,
};
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::tree::Model;
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const RECURSIVE_FORECAST_RANGE_MARGIN_MULTIPLIER: f64 = 3.0;
const RECURSIVE_FORECAST_SCALE_MARGIN_FRACTION: f64 = 0.25;
const RECURSIVE_FORECAST_ABSOLUTE_MARGIN_FLOOR: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GlobalForecastTargetMode {
    Level,
    DeltaFromLast,
    SeasonalDelta { season_length: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GlobalForecastSampleWeightMode {
    Uniform,
    ExponentialRecency { half_life: usize },
}

#[derive(Debug, Clone)]
pub struct CartoBoostLagForecaster {
    lag_builder: LagFeatureBuilder,
    booster_config: BoosterConfig,
    target_mode: GlobalForecastTargetMode,
    sample_weight_mode: GlobalForecastSampleWeightMode,
    fitted: Option<FittedGlobalState>,
}

#[derive(Debug, Clone)]
struct FittedGlobalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    model: Model,
    training_rows: usize,
}

impl CartoBoostLagForecaster {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Result<Self> {
        Self::new_with_target_mode(lag_config, booster_config, GlobalForecastTargetMode::Level)
    }

    pub fn new_with_target_mode(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        target_mode: GlobalForecastTargetMode,
    ) -> Result<Self> {
        Self::new_with_target_mode_and_sample_weight(
            lag_config,
            booster_config,
            target_mode,
            GlobalForecastSampleWeightMode::Uniform,
        )
    }

    pub fn new_with_target_mode_and_sample_weight(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        target_mode: GlobalForecastTargetMode,
        sample_weight_mode: GlobalForecastSampleWeightMode,
    ) -> Result<Self> {
        validate_target_mode(target_mode)?;
        validate_sample_weight_mode(sample_weight_mode)?;
        Ok(Self {
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            target_mode,
            sample_weight_mode,
            fitted: None,
        })
    }

    pub fn lag_builder(&self) -> &LagFeatureBuilder {
        &self.lag_builder
    }

    pub fn booster_config(&self) -> &BoosterConfig {
        &self.booster_config
    }

    pub fn target_mode(&self) -> GlobalForecastTargetMode {
        self.target_mode
    }

    pub fn sample_weight_mode(&self) -> GlobalForecastSampleWeightMode {
        self.sample_weight_mode
    }

    pub fn model(&self) -> Option<&Model> {
        self.fitted.as_ref().map(|state| &state.model)
    }

    pub fn training_rows(&self) -> Option<usize> {
        self.fitted.as_ref().map(|state| state.training_rows)
    }

    pub fn predict_with_known_future_covariates(
        &self,
        horizon: usize,
        known_future_covariates: &BTreeMap<(String, chrono::NaiveDateTime), BTreeMap<String, f64>>,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, fitted_history)| {
                let mut history = fitted_history.clone();
                let value_bounds = forecast_value_bounds(fitted_history)?;
                let last = history
                    .last()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput("empty series history".to_string())
                    })?
                    .clone();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let timestamp = fitted.frame.frequency().advance(last.timestamp, step)?;
                    let future_covariates =
                        known_future_covariates.get(&(series_id.clone(), timestamp));
                    let features = self
                        .lag_builder
                        .transform_next_sorted_prior_with_covariates(
                            series_id,
                            &history,
                            timestamp,
                            future_covariates,
                        )?;
                    let raw_prediction = fitted.model.predict_one(&features);
                    let mean = match self.target_mode {
                        GlobalForecastTargetMode::Level => raw_prediction,
                        GlobalForecastTargetMode::DeltaFromLast => {
                            let previous = history.last().ok_or_else(|| {
                                CartoBoostError::InvalidInput(format!(
                                    "series {series_id} has no previous target for delta forecast"
                                ))
                            })?;
                            previous.target + raw_prediction
                        }
                        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
                            let seasonal = seasonal_target_before(&history, season_length)
                                .ok_or_else(|| {
                                    CartoBoostError::InvalidInput(format!(
                                        "series {series_id} does not have enough seasonal history"
                                    ))
                                })?;
                            seasonal + raw_prediction
                        }
                    };
                    let mean = clamp_forecast_value(mean, &value_bounds);
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp,
                        horizon: step,
                        model: self.model_name().to_string(),
                        mean,
                    });
                    let covariates = future_covariates
                        .cloned()
                        .or_else(|| history.last().map(|row| row.covariates.clone()))
                        .unwrap_or_default();
                    history.push(ForecastRow::with_covariates(
                        series_id.clone(),
                        timestamp,
                        mean,
                        covariates,
                    ));
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl Forecaster for CartoBoostLagForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let effective_lag_config =
            lag_config_supported_by_history(self.lag_builder.config(), frame);
        self.lag_builder = LagFeatureBuilder::new(effective_lag_config)?;
        let feature_rows = self.lag_builder.transform_frame(frame)?;
        if feature_rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "not enough history to build lag training rows".to_string(),
            ));
        }
        let feature_count = self.lag_builder.feature_names().len();
        let x = Dataset::from_rows(
            feature_rows
                .iter()
                .map(|row| row.features.clone())
                .collect::<Vec<_>>(),
        )?
        .with_schema(FeatureSchema {
            names: self.lag_builder.feature_names().to_vec(),
            kinds: vec![FeatureKind::Numeric; feature_count],
        })?;
        let history_by_series = history_by_series(frame.rows());
        let y = feature_rows
            .iter()
            .map(|row| target_for_mode(&self.target_mode, &history_by_series, row))
            .collect::<Result<Vec<_>>>()?;
        let sample_weights = sample_weights_for_feature_rows(
            &feature_rows,
            &history_by_series,
            self.sample_weight_mode,
        )?;
        let model =
            Booster::new(self.booster_config.clone()).fit(&x, &y, sample_weights.as_deref())?;
        self.fitted = Some(FittedGlobalState {
            frame: frame.clone(),
            history_by_series,
            model,
            training_rows: feature_rows.len(),
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, fitted_history)| {
                let mut history = fitted_history.clone();
                let value_bounds = forecast_value_bounds(fitted_history)?;
                let last = history
                    .last()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput("empty series history".to_string())
                    })?
                    .clone();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let timestamp = fitted.frame.frequency().advance(last.timestamp, step)?;
                    let features = self
                        .lag_builder
                        .transform_next_sorted_prior(series_id, &history, timestamp)?;
                    let raw_prediction = fitted.model.predict_one(&features);
                    let mean = match self.target_mode {
                        GlobalForecastTargetMode::Level => raw_prediction,
                        GlobalForecastTargetMode::DeltaFromLast => {
                            let previous = history.last().ok_or_else(|| {
                                CartoBoostError::InvalidInput(format!(
                                    "series {series_id} has no previous target for delta forecast"
                                ))
                            })?;
                            previous.target + raw_prediction
                        }
                        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
                            let seasonal = seasonal_target_before(&history, season_length)
                                .ok_or_else(|| {
                                    CartoBoostError::InvalidInput(format!(
                                        "series {series_id} does not have enough seasonal history"
                                    ))
                                })?;
                            seasonal + raw_prediction
                        }
                    };
                    let mean = clamp_forecast_value(mean, &value_bounds);
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp,
                        horizon: step,
                        model: self.model_name().to_string(),
                        mean,
                    });
                    let covariates = history
                        .last()
                        .map(|row| row.covariates.clone())
                        .unwrap_or_default();
                    history.push(ForecastRow::with_covariates(
                        series_id.clone(),
                        timestamp,
                        mean,
                        covariates,
                    ));
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "cartoboost_lag"
    }

    fn metadata(&self) -> Value {
        let mut payload = json!({
            "model": self.model_name(),
            "feature_names": self.lag_builder.feature_names(),
            "lag_config": self.lag_builder.config(),
            "booster_config": self.booster_config,
            "target_mode": target_mode_name(self.target_mode),
            "sample_weight_mode": sample_weight_mode_name(self.sample_weight_mode),
        });
        if let Some(fitted) = &self.fitted {
            payload["training_rows"] = json!(fitted.training_rows);
            payload["series_count"] = json!(fitted.history_by_series.len());
            payload["native_model_metadata"] = json!(fitted.model.metadata);
        }
        payload
    }
}

fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ForecastValueBounds {
    lower: f64,
    upper: f64,
}

fn forecast_value_bounds(history: &[ForecastRow]) -> Result<ForecastValueBounds> {
    if history.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast bounds require non-empty history".to_string(),
        ));
    }
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let mut max_abs = 0.0_f64;
    for row in history {
        if !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "forecast bounds require finite history targets".to_string(),
            ));
        }
        min_value = min_value.min(row.target);
        max_value = max_value.max(row.target);
        max_abs = max_abs.max(row.target.abs());
    }
    let range = max_value - min_value;
    let margin = (range * RECURSIVE_FORECAST_RANGE_MARGIN_MULTIPLIER)
        .max(max_abs * RECURSIVE_FORECAST_SCALE_MARGIN_FRACTION)
        .max(RECURSIVE_FORECAST_ABSOLUTE_MARGIN_FLOOR);
    Ok(ForecastValueBounds {
        lower: min_value - margin,
        upper: max_value + margin,
    })
}

fn clamp_forecast_value(value: f64, bounds: &ForecastValueBounds) -> f64 {
    if value.is_finite() {
        value.clamp(bounds.lower, bounds.upper)
    } else if value.is_sign_negative() {
        bounds.lower
    } else {
        bounds.upper
    }
}

fn validate_target_mode(target_mode: GlobalForecastTargetMode) -> Result<()> {
    if matches!(
        target_mode,
        GlobalForecastTargetMode::SeasonalDelta { season_length: 0 }
    ) {
        return Err(CartoBoostError::InvalidInput(
            "seasonal_delta target mode requires a positive season length".to_string(),
        ));
    }
    Ok(())
}

fn validate_sample_weight_mode(mode: GlobalForecastSampleWeightMode) -> Result<()> {
    match mode {
        GlobalForecastSampleWeightMode::Uniform => Ok(()),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life } => {
            if half_life == 0 {
                return Err(CartoBoostError::InvalidInput(
                    "exponential recency half_life must be positive".to_string(),
                ));
            }
            Ok(())
        }
    }
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

fn sample_weights_for_feature_rows(
    feature_rows: &[crate::forecasting::LagFeatureRow],
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    mode: GlobalForecastSampleWeightMode,
) -> Result<Option<Vec<f64>>> {
    match mode {
        GlobalForecastSampleWeightMode::Uniform => Ok(None),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life } => {
            let mut weights = Vec::with_capacity(feature_rows.len());
            for row in feature_rows {
                let history = history_by_series.get(&row.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing history for series {}",
                        row.series_id
                    ))
                })?;
                let row_index = history
                    .iter()
                    .position(|history_row| history_row.timestamp == row.timestamp)
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing history timestamp {} for series {}",
                            row.timestamp, row.series_id
                        ))
                    })?;
                let age = history.len().saturating_sub(row_index + 1);
                let exponent = -(age as f64) / half_life as f64;
                weights.push(2.0_f64.powf(exponent));
            }
            normalize_sample_weights(weights).map(Some)
        }
    }
}

fn normalize_sample_weights(mut weights: Vec<f64>) -> Result<Vec<f64>> {
    if weights.is_empty() {
        return Ok(weights);
    }
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "sample weights must have positive finite mean".to_string(),
        ));
    }
    for weight in &mut weights {
        *weight /= mean;
    }
    Ok(weights)
}

fn target_for_mode(
    target_mode: &GlobalForecastTargetMode,
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    row: &crate::forecasting::LagFeatureRow,
) -> Result<f64> {
    match target_mode {
        GlobalForecastTargetMode::Level => Ok(row.target),
        GlobalForecastTargetMode::DeltaFromLast => {
            let history = history_by_series.get(&row.series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing history for series {}",
                    row.series_id
                ))
            })?;
            let prior_target = prior_target_before(history, row.timestamp).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing prior target for series {} at {}",
                    row.series_id, row.timestamp
                ))
            })?;
            Ok(row.target - prior_target)
        }
        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
            let history = history_by_series.get(&row.series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing history for series {}",
                    row.series_id
                ))
            })?;
            let seasonal_target =
                seasonal_target_before_timestamp(history, row.timestamp, *season_length)
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing seasonal target for series {} at {}",
                            row.series_id, row.timestamp
                        ))
                    })?;
            Ok(row.target - seasonal_target)
        }
    }
}

fn prior_target_before(history: &[ForecastRow], timestamp: chrono::NaiveDateTime) -> Option<f64> {
    history
        .iter()
        .rev()
        .find(|row| row.timestamp < timestamp)
        .map(|row| row.target)
}

fn seasonal_target_before_timestamp(
    history: &[ForecastRow],
    timestamp: chrono::NaiveDateTime,
    season_length: usize,
) -> Option<f64> {
    if season_length == 0 {
        return None;
    }
    let prior = history
        .iter()
        .filter(|row| row.timestamp < timestamp)
        .collect::<Vec<_>>();
    if prior.len() < season_length {
        return None;
    }
    Some(prior[prior.len() - season_length].target)
}

fn seasonal_target_before(history: &[ForecastRow], season_length: usize) -> Option<f64> {
    if season_length == 0 || history.len() < season_length {
        return None;
    }
    Some(history[history.len() - season_length].target)
}

pub(crate) fn target_mode_name(target_mode: GlobalForecastTargetMode) -> String {
    match target_mode {
        GlobalForecastTargetMode::Level => "level".to_string(),
        GlobalForecastTargetMode::DeltaFromLast => "delta_from_last".to_string(),
        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
            format!("seasonal_delta_{season_length}")
        }
    }
}

pub(crate) fn sample_weight_mode_name(mode: GlobalForecastSampleWeightMode) -> String {
    match mode {
        GlobalForecastSampleWeightMode::Uniform => "uniform".to_string(),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life } => {
            format!("exponential_recency_half_life_{half_life}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn ts(day: u32) -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 1, day)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid timestamp")
    }

    #[test]
    fn recursive_forecast_value_bounds_limit_unstable_extrapolation() {
        let history = vec![
            ForecastRow::new("series", ts(1), 10.0),
            ForecastRow::new("series", ts(2), 20.0),
            ForecastRow::new("series", ts(3), 30.0),
        ];
        let bounds = forecast_value_bounds(&history).expect("bounds");

        assert_eq!(clamp_forecast_value(25.0, &bounds), 25.0);
        assert_eq!(clamp_forecast_value(1_000_000.0, &bounds), bounds.upper);
        assert_eq!(clamp_forecast_value(f64::INFINITY, &bounds), bounds.upper);
        assert_eq!(
            clamp_forecast_value(f64::NEG_INFINITY, &bounds),
            bounds.lower
        );
        assert!(bounds.upper < 100.0);
        assert!(bounds.lower < 10.0);
    }

    #[test]
    fn lag_forecaster_prunes_unsupported_lag_features_for_short_panels() {
        let frame = ForecastFrame::new(
            ["PU1->DO2", "PU9->DO8"]
                .into_iter()
                .flat_map(|series_id| {
                    (1..=8).map(move |day| {
                        ForecastRow::new(
                            series_id,
                            ts(day),
                            f64::from(day) + f64::from(series_id.len() as u32),
                        )
                    })
                })
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("valid short panel");
        let mut model = CartoBoostLagForecaster::new(
            LagFeatureConfig {
                lags: vec![1, 24],
                rolling_mean_windows: vec![24],
                partial_rolling_mean_windows: Vec::new(),
                rolling_std_windows: vec![24],
                rolling_min_windows: vec![24],
                rolling_max_windows: vec![24],
                ewm_alpha_percents: Vec::new(),
                calendar_features: Vec::new(),
                difference_lags: vec![24],
                rolling_trend_windows: vec![24],
                covariate_features: Vec::new(),
                covariate_indicator_values: Default::default(),
                covariate_calendar_interactions: false,
            },
            BoosterConfig {
                n_estimators: 3,
                max_depth: 2,
                min_samples_leaf: 1,
                ..BoosterConfig::default()
            },
        )
        .expect("model");

        model.fit(&frame).expect("fit lag model");
        let metadata = model.metadata();
        assert_eq!(metadata["lag_config"]["lags"], serde_json::json!([1]));
        assert_eq!(
            metadata["lag_config"]["rolling_mean_windows"],
            serde_json::json!([])
        );
        let forecast = model.predict(3).expect("forecast");
        assert_eq!(forecast.predictions().len(), 6);
        assert!(forecast
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));
    }
}
