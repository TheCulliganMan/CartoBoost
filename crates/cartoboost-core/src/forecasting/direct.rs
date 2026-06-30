use crate::booster::{Booster, BoosterConfig};
use crate::data::{Dataset, FeatureKind, FeatureSchema};
use crate::forecasting::horizon::validate_horizon;
use crate::forecasting::lag_features::{
    history_by_series, lag_config_supported_by_history, LagFeatureBuilder, LagFeatureConfig,
};
use crate::forecasting::{
    CartoBoostLagForecaster, ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow,
    Forecaster, GlobalForecastTargetMode,
};
use crate::tree::Model;
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectForecastStrategy {
    Direct,
    Recursive,
    RectifiedRecursive,
}

#[derive(Debug, Clone)]
pub struct CartoBoostDirectForecaster {
    lag_builder: LagFeatureBuilder,
    booster_config: BoosterConfig,
    fitted: Option<FittedDirectState>,
}

#[derive(Debug, Clone)]
struct FittedDirectState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    models: Vec<Model>,
    training_rows_by_horizon: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct RectifiedRecursiveForecaster {
    recursive: CartoBoostLagForecaster,
    lag_builder: LagFeatureBuilder,
    booster_config: BoosterConfig,
    fitted: Option<FittedRectifiedState>,
}

#[derive(Debug, Clone)]
struct FittedRectifiedState {
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    corrections: Vec<Model>,
    training_rows_by_horizon: Vec<usize>,
}

impl CartoBoostDirectForecaster {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Result<Self> {
        Ok(Self {
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            fitted: None,
        })
    }

    pub fn lag_builder(&self) -> &LagFeatureBuilder {
        &self.lag_builder
    }

    pub fn booster_config(&self) -> &BoosterConfig {
        &self.booster_config
    }

    pub fn models(&self) -> Option<&[Model]> {
        self.fitted.as_ref().map(|state| state.models.as_slice())
    }

    pub fn training_rows_by_horizon(&self) -> Option<&[usize]> {
        self.fitted
            .as_ref()
            .map(|state| state.training_rows_by_horizon.as_slice())
    }

    pub fn fit_horizon(&mut self, frame: &ForecastFrame, horizon: usize) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        validate_horizon(horizon)?;
        let effective_lag_config =
            lag_config_supported_by_history(self.lag_builder.config(), frame);
        self.lag_builder = LagFeatureBuilder::new(effective_lag_config)?;
        let mut models = Vec::with_capacity(horizon);
        let mut training_rows_by_horizon = Vec::with_capacity(horizon);
        for step in 1..=horizon {
            let training = build_direct_training(frame, &self.lag_builder, step)?;
            let model =
                Booster::new(self.booster_config.clone()).fit(&training.x, &training.y, None)?;
            training_rows_by_horizon.push(training.y.len());
            models.push(model);
        }
        self.fitted = Some(FittedDirectState {
            frame: frame.clone(),
            history_by_series: history_by_series(frame.rows()),
            models,
            training_rows_by_horizon,
        });
        Ok(())
    }
}

impl Forecaster for CartoBoostDirectForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fit_horizon(frame, 1)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        if horizon > fitted.models.len() {
            return Err(CartoBoostError::InvalidInput(format!(
                "direct forecaster was fitted for {} horizons but {horizon} were requested",
                fitted.models.len()
            )));
        }
        let predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!("series {series_id} has no history"))
                })?;
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let timestamp = fitted.frame.frequency().advance(last.timestamp, step)?;
                    let features = self
                        .lag_builder
                        .transform_next_sorted_prior(series_id, history, timestamp)?;
                    let mean = fitted.models[step - 1].predict_one(&features);
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp,
                        horizon: step,
                        model: self.model_name().to_string(),
                        mean,
                    });
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
        "cartoboost_direct"
    }

    fn metadata(&self) -> Value {
        let mut payload = json!({
            "model": self.model_name(),
            "strategy": DirectForecastStrategy::Direct,
            "feature_names": self.lag_builder.feature_names(),
            "lag_config": self.lag_builder.config(),
            "booster_config": self.booster_config,
        });
        if let Some(fitted) = &self.fitted {
            payload["fitted_horizon"] = json!(fitted.models.len());
            payload["training_rows_by_horizon"] = json!(fitted.training_rows_by_horizon);
        }
        payload
    }
}

impl RectifiedRecursiveForecaster {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Result<Self> {
        Ok(Self {
            recursive: CartoBoostLagForecaster::new_with_target_mode(
                lag_config.clone(),
                booster_config.clone(),
                GlobalForecastTargetMode::Level,
            )?,
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            fitted: None,
        })
    }

    pub fn fit_horizon(&mut self, frame: &ForecastFrame, horizon: usize) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        validate_horizon(horizon)?;
        let effective_lag_config =
            lag_config_supported_by_history(self.lag_builder.config(), frame);
        self.lag_builder = LagFeatureBuilder::new(effective_lag_config.clone())?;
        self.recursive = CartoBoostLagForecaster::new_with_target_mode(
            effective_lag_config,
            self.booster_config.clone(),
            GlobalForecastTargetMode::Level,
        )?;
        self.recursive.fit(frame)?;
        let recursive_baselines = recursive_training_predictions(
            frame,
            &self.lag_builder,
            &self.booster_config,
            horizon,
        )?;
        let mut corrections = Vec::with_capacity(horizon);
        let mut training_rows_by_horizon = Vec::with_capacity(horizon);
        for step in 1..=horizon {
            let training = build_rectification_training(
                frame,
                &self.lag_builder,
                step,
                &recursive_baselines[step - 1],
            )?;
            let model =
                Booster::new(self.booster_config.clone()).fit(&training.x, &training.y, None)?;
            training_rows_by_horizon.push(training.y.len());
            corrections.push(model);
        }
        self.fitted = Some(FittedRectifiedState {
            history_by_series: history_by_series(frame.rows()),
            corrections,
            training_rows_by_horizon,
        });
        Ok(())
    }

    pub fn training_rows_by_horizon(&self) -> Option<&[usize]> {
        self.fitted
            .as_ref()
            .map(|state| state.training_rows_by_horizon.as_slice())
    }
}

impl Forecaster for RectifiedRecursiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fit_horizon(frame, 1)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        if horizon > fitted.corrections.len() {
            return Err(CartoBoostError::InvalidInput(format!(
                "rectified recursive forecaster was fitted for {} horizons but {horizon} were requested",
                fitted.corrections.len()
            )));
        }
        let baseline = self.recursive.predict(horizon)?;
        let base_predictions = baseline.predictions();
        let corrected = base_predictions
            .par_iter()
            .map(|prediction| {
                let history = fitted
                    .history_by_series
                    .get(&prediction.series_id)
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing history for series {}",
                            prediction.series_id
                        ))
                    })?;
                let features = self.lag_builder.transform_next_sorted_prior(
                    &prediction.series_id,
                    history,
                    prediction.timestamp,
                )?;
                let correction = fitted.corrections[prediction.horizon - 1].predict_one(&features);
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: self.model_name().to_string(),
                    mean: prediction.mean + correction,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new(corrected)
    }

    fn model_name(&self) -> &'static str {
        "cartoboost_rectified_recursive"
    }

    fn metadata(&self) -> Value {
        let mut payload = json!({
            "model": self.model_name(),
            "strategy": DirectForecastStrategy::RectifiedRecursive,
            "feature_names": self.lag_builder.feature_names(),
            "lag_config": self.lag_builder.config(),
            "booster_config": self.booster_config,
            "recursive": self.recursive.metadata(),
        });
        if let Some(fitted) = &self.fitted {
            payload["fitted_horizon"] = json!(fitted.corrections.len());
            payload["training_rows_by_horizon"] = json!(fitted.training_rows_by_horizon);
        }
        payload
    }
}

struct DirectTraining {
    x: Dataset,
    y: Vec<f64>,
}

fn build_direct_training(
    frame: &ForecastFrame,
    lag_builder: &LagFeatureBuilder,
    horizon: usize,
) -> Result<DirectTraining> {
    let mut x_rows = Vec::new();
    let mut y = Vec::new();
    for (_series_id, history) in history_by_series(frame.rows()) {
        for origin_idx in 0..history.len() {
            let target_idx = origin_idx + horizon;
            if target_idx >= history.len() {
                continue;
            }
            let origin_timestamp = history[origin_idx].timestamp;
            let prior = &history[..=origin_idx];
            let features = match lag_builder.transform_next_sorted_prior(
                &history[origin_idx].series_id,
                prior,
                frame.frequency().advance(origin_timestamp, horizon)?,
            ) {
                Ok(features) => features,
                Err(err) if is_incomplete_lag_history(&err) => continue,
                Err(err) => return Err(err),
            };
            x_rows.push(features);
            y.push(history[target_idx].target);
        }
    }
    dataset_from_lag_rows(x_rows, y, lag_builder)
}

fn build_rectification_training(
    frame: &ForecastFrame,
    lag_builder: &LagFeatureBuilder,
    horizon: usize,
    recursive_baselines: &[Option<f64>],
) -> Result<DirectTraining> {
    let mut x_rows = Vec::new();
    let mut y = Vec::new();
    let mut global_idx = 0usize;
    for (_series_id, history) in history_by_series(frame.rows()) {
        for origin_idx in 0..history.len() {
            let target_idx = origin_idx + horizon;
            if target_idx >= history.len() {
                global_idx += 1;
                continue;
            }
            if let Some(baseline) = recursive_baselines.get(global_idx).and_then(|v| *v) {
                let origin_timestamp = history[origin_idx].timestamp;
                let prior = &history[..=origin_idx];
                let features = match lag_builder.transform_next_sorted_prior(
                    &history[origin_idx].series_id,
                    prior,
                    frame.frequency().advance(origin_timestamp, horizon)?,
                ) {
                    Ok(features) => features,
                    Err(err) if is_incomplete_lag_history(&err) => {
                        global_idx += 1;
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                x_rows.push(features);
                y.push(history[target_idx].target - baseline);
            }
            global_idx += 1;
        }
    }
    dataset_from_lag_rows(x_rows, y, lag_builder)
}

fn recursive_training_predictions(
    frame: &ForecastFrame,
    lag_builder: &LagFeatureBuilder,
    booster_config: &BoosterConfig,
    horizon: usize,
) -> Result<Vec<Vec<Option<f64>>>> {
    let mut result = vec![Vec::new(); horizon];
    let training = build_direct_training(frame, lag_builder, 1)?;
    let one_step = Booster::new(booster_config.clone()).fit(&training.x, &training.y, None)?;
    for (_series_id, history) in history_by_series(frame.rows()) {
        for origin_idx in 0..history.len() {
            let mut recursive_history = history[..=origin_idx].to_vec();
            let mut incomplete_history = false;
            for step in 1..=horizon {
                if origin_idx + step >= history.len() || incomplete_history {
                    result[step - 1].push(None);
                    continue;
                }
                let timestamp = frame
                    .frequency()
                    .advance(history[origin_idx].timestamp, step)?;
                let features = match lag_builder.transform_next_sorted_prior(
                    &history[origin_idx].series_id,
                    &recursive_history,
                    timestamp,
                ) {
                    Ok(features) => features,
                    Err(err) if is_incomplete_lag_history(&err) => {
                        incomplete_history = true;
                        result[step - 1].push(None);
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                let mean = one_step.predict_one(&features);
                result[step - 1].push(Some(mean));
                recursive_history.push(ForecastRow::new(
                    history[origin_idx].series_id.clone(),
                    timestamp,
                    mean,
                ));
            }
        }
    }
    Ok(result)
}

fn dataset_from_lag_rows(
    x_rows: Vec<Vec<f64>>,
    y: Vec<f64>,
    lag_builder: &LagFeatureBuilder,
) -> Result<DirectTraining> {
    if x_rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "not enough history to build direct forecast training rows".to_string(),
        ));
    }
    let feature_count = lag_builder.feature_names().len();
    let x = Dataset::from_rows(x_rows)?.with_schema(FeatureSchema {
        names: lag_builder.feature_names().to_vec(),
        kinds: vec![FeatureKind::Numeric; feature_count],
    })?;
    Ok(DirectTraining { x, y })
}

fn is_incomplete_lag_history(err: &CartoBoostError) -> bool {
    matches!(err, CartoBoostError::InvalidInput(message) if message.contains("does not have enough prior history"))
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::ForecastFrequency;
    use chrono::{NaiveDate, NaiveDateTime};

    fn ts(day: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 1, day)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid timestamp")
    }

    fn short_panel_frame() -> ForecastFrame {
        ForecastFrame::new(
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
            ForecastFrequency::Daily,
        )
        .expect("valid short panel")
    }

    fn oversized_lag_config() -> LagFeatureConfig {
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
        }
    }

    #[test]
    fn direct_models_prune_unsupported_lag_features_for_short_panels() {
        let frame = short_panel_frame();
        let booster = BoosterConfig {
            n_estimators: 3,
            max_depth: 2,
            min_samples_leaf: 1,
            ..BoosterConfig::default()
        };

        let mut direct = CartoBoostDirectForecaster::new(oversized_lag_config(), booster.clone())
            .expect("direct");
        direct.fit_horizon(&frame, 3).expect("fit direct");
        let direct_metadata = direct.metadata();
        assert_eq!(
            direct_metadata["lag_config"]["lags"],
            serde_json::json!([1])
        );
        assert_eq!(
            direct_metadata["lag_config"]["rolling_mean_windows"],
            serde_json::json!([])
        );
        let direct_forecast = direct.predict(3).expect("direct forecast");
        assert_eq!(direct_forecast.predictions().len(), 6);
        assert!(direct_forecast
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));

        let mut rectified =
            RectifiedRecursiveForecaster::new(oversized_lag_config(), booster).expect("rectified");
        rectified
            .fit_horizon(&frame, 3)
            .expect("fit rectified recursive");
        let rectified_metadata = rectified.metadata();
        assert_eq!(
            rectified_metadata["lag_config"]["lags"],
            serde_json::json!([1])
        );
        let rectified_forecast = rectified.predict(3).expect("rectified forecast");
        assert_eq!(rectified_forecast.predictions().len(), 6);
        assert!(rectified_forecast
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));
    }
}
