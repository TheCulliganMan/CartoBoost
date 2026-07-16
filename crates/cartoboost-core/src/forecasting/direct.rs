use crate::booster::{Booster, BoosterConfig};
use crate::data::{Dataset, FeatureKind, FeatureSchema};
use crate::forecasting::horizon::validate_horizon;
use crate::forecasting::lag_features::{
    history_by_series, minimum_prior_len, validate_lag_config_supported_by_prior,
    LagFeatureBuilder, LagFeatureConfig,
};
use crate::forecasting::{
    CartoBoostLagForecaster, ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow,
    Forecaster, GlobalForecastTargetMode,
};
use crate::tree::Model;
use crate::{CartoBoostError, Result};
use cartoboost_accelerator::backend::{select_backend_for, BackendOperation, BackendSelection};
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
    backend: BackendSelection,
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
    backend: BackendSelection,
}

#[derive(Debug, Clone)]
struct FittedRectifiedState {
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    corrections: Vec<Model>,
    training_rows_by_horizon: Vec<usize>,
    validation_window: usize,
}

impl CartoBoostDirectForecaster {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Result<Self> {
        Self::new_with_backend(lag_config, booster_config, Some("cpu"))
    }

    pub fn new_with_backend(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = select_backend_for(backend, BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(Self {
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            fitted: None,
            backend,
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
        validate_direct_lag_history(frame, self.lag_builder.config(), horizon, self.model_name())?;
        let fit_step = |step| {
            let training = build_direct_training(frame, &self.lag_builder, step)?;
            let training_rows = training.y.len();
            let model = Booster::new_with_backend(
                self.booster_config.clone(),
                Some(&self.backend.selected),
            )?
            .fit(&training.x, &training.y, None)?;
            Ok((model, training_rows))
        };
        let fitted_steps = if self.backend.selected == "cpu" {
            (1..=horizon)
                .into_par_iter()
                .map(fit_step)
                .collect::<Result<Vec<_>>>()?
        } else {
            (1..=horizon).map(fit_step).collect::<Result<Vec<_>>>()?
        };
        let (models, training_rows_by_horizon) = fitted_steps.into_iter().unzip();
        self.fitted = Some(FittedDirectState {
            frame: frame.clone(),
            history_by_series: history_by_series(frame.rows()),
            models,
            training_rows_by_horizon,
        });
        Ok(())
    }

    fn predict_accelerated(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series = fitted.history_by_series.iter().collect::<Vec<_>>();
        let mut predictions = Vec::with_capacity(series.len() * horizon);
        for step in 1..=horizon {
            let mut rows = Vec::with_capacity(series.len());
            let mut timestamps = Vec::with_capacity(series.len());
            for (series_id, history) in &series {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!("series {series_id} has no history"))
                })?;
                let timestamp = fitted.frame.frequency().advance(last.timestamp, step)?;
                rows.push(
                    self.lag_builder
                        .transform_next_sorted_prior(series_id, history, timestamp)?,
                );
                timestamps.push(timestamp);
            }
            let dataset = Dataset::from_rows(rows)?;
            let means = fitted.models[step - 1].try_predict(&dataset)?;
            for (((series_id, _), timestamp), mean) in series.iter().zip(timestamps).zip(means) {
                predictions.push(ForecastPrediction {
                    series_id: (*series_id).clone(),
                    timestamp,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean,
                });
            }
        }
        ForecastResult::new(predictions)
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
        if self.backend.selected != "cpu" {
            return self.predict_accelerated(horizon);
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
            "backend": self.backend,
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
        Self::new_with_backend(lag_config, booster_config, Some("cpu"))
    }

    pub fn new_with_backend(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = select_backend_for(backend, BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(Self {
            recursive: CartoBoostLagForecaster::new_with_backend(
                lag_config.clone(),
                booster_config.clone(),
                GlobalForecastTargetMode::Level,
                crate::forecasting::GlobalForecastSampleWeightMode::Uniform,
                Some(&backend.selected),
            )?,
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            fitted: None,
            backend,
        })
    }

    pub fn fit_horizon(&mut self, frame: &ForecastFrame, horizon: usize) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        validate_horizon(horizon)?;
        validate_direct_lag_history(frame, self.lag_builder.config(), horizon, self.model_name())?;
        self.recursive = CartoBoostLagForecaster::new_with_backend(
            self.lag_builder.config().clone(),
            self.booster_config.clone(),
            GlobalForecastTargetMode::Level,
            crate::forecasting::GlobalForecastSampleWeightMode::Uniform,
            Some(&self.backend.selected),
        )?;
        self.recursive.fit(frame)?;
        let (recursive_baselines, validation_window) = recursive_training_predictions(
            frame,
            &self.lag_builder,
            &self.booster_config,
            horizon,
            Some(&self.backend.selected),
        )?;
        let fit_step = |step| {
            let training = build_rectification_training(
                frame,
                &self.lag_builder,
                step,
                &recursive_baselines[step - 1],
            )?;
            let training_rows = training.y.len();
            let model = Booster::new_with_backend(
                self.booster_config.clone(),
                Some(&self.backend.selected),
            )?
            .fit(&training.x, &training.y, None)?;
            Ok((model, training_rows))
        };
        let fitted_steps = if self.backend.selected == "cpu" {
            (1..=horizon)
                .into_par_iter()
                .map(fit_step)
                .collect::<Result<Vec<_>>>()?
        } else {
            (1..=horizon).map(fit_step).collect::<Result<Vec<_>>>()?
        };
        let (corrections, training_rows_by_horizon) = fitted_steps.into_iter().unzip();
        self.fitted = Some(FittedRectifiedState {
            history_by_series: history_by_series(frame.rows()),
            corrections,
            training_rows_by_horizon,
            validation_window,
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
        if self.backend.selected != "cpu" {
            let mut corrected = base_predictions.to_vec();
            for step in 1..=horizon {
                let indices = base_predictions
                    .iter()
                    .enumerate()
                    .filter_map(|(index, prediction)| (prediction.horizon == step).then_some(index))
                    .collect::<Vec<_>>();
                let rows = indices
                    .iter()
                    .map(|index| {
                        let prediction = &base_predictions[*index];
                        let history = fitted
                            .history_by_series
                            .get(&prediction.series_id)
                            .ok_or_else(|| {
                                CartoBoostError::InvalidInput(format!(
                                    "missing history for series {}",
                                    prediction.series_id
                                ))
                            })?;
                        self.lag_builder.transform_next_sorted_prior(
                            &prediction.series_id,
                            history,
                            prediction.timestamp,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let corrections =
                    fitted.corrections[step - 1].try_predict(&Dataset::from_rows(rows)?)?;
                for (index, correction) in indices.into_iter().zip(corrections) {
                    corrected[index].model = self.model_name().to_string();
                    corrected[index].mean += correction;
                }
            }
            return ForecastResult::new(corrected);
        }
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
            "backend": self.backend,
        });
        if let Some(fitted) = &self.fitted {
            payload["fitted_horizon"] = json!(fitted.corrections.len());
            payload["training_rows_by_horizon"] = json!(fitted.training_rows_by_horizon);
            payload["validation_window"] = json!(fitted.validation_window);
        }
        payload
    }
}

struct DirectTraining {
    x: Dataset,
    y: Vec<f64>,
}

fn validate_direct_lag_history(
    frame: &ForecastFrame,
    config: &LagFeatureConfig,
    horizon: usize,
    model_name: &str,
) -> Result<()> {
    let minimum_history = history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    validate_lag_config_supported_by_prior(
        config,
        minimum_history.saturating_sub(horizon),
        model_name,
    )
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
    backend: Option<&str>,
) -> Result<(Vec<Vec<Option<f64>>>, usize)> {
    let histories = history_by_series(frame.rows());
    let minimum_history = histories.values().map(Vec::len).min().unwrap_or(0);
    let required_training_rows = minimum_prior_len(lag_builder.config()).saturating_add(1);
    let maximum_validation_window = minimum_history
        .checked_sub(required_training_rows)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "rectified recursive cross-validation requires at least {required_training_rows} training rows per series before its holdout"
            ))
        })?;
    let automatic_window = (minimum_history / 5).clamp(1, 28);
    let validation_window = automatic_window.max(horizon).min(maximum_validation_window);
    if validation_window < horizon {
        return Err(CartoBoostError::InvalidInput(format!(
            "rectified recursive fitting requires a holdout of at least {horizon} rows while retaining {required_training_rows} training rows; shortest series has {minimum_history} rows"
        )));
    }

    let mut result = vec![Vec::new(); horizon];
    let one_step = fit_recursive_prefix_model(
        frame,
        lag_builder,
        booster_config,
        validation_window,
        backend,
    )?;
    for (_series_id, history) in histories {
        let first_validation_origin = history.len() - validation_window - 1;
        for origin_idx in 0..history.len() {
            if origin_idx < first_validation_origin {
                for step_result in &mut result {
                    step_result.push(None);
                }
                continue;
            }
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
                if !mean.is_finite() {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "rectified recursive baseline produced a non-finite forecast for series {} at {timestamp}",
                        history[origin_idx].series_id
                    )));
                }
                result[step - 1].push(Some(mean));
                let covariates = recursive_history
                    .last()
                    .map(|row| row.covariates.clone())
                    .unwrap_or_default();
                recursive_history.push(ForecastRow::with_covariates(
                    history[origin_idx].series_id.clone(),
                    timestamp,
                    mean,
                    covariates,
                ));
            }
        }
    }
    Ok((result, validation_window))
}

fn fit_recursive_prefix_model(
    frame: &ForecastFrame,
    lag_builder: &LagFeatureBuilder,
    booster_config: &BoosterConfig,
    validation_window: usize,
    backend: Option<&str>,
) -> Result<Model> {
    let mut prefix_rows = Vec::new();
    for (_series_id, history) in history_by_series(frame.rows()) {
        let train_len = history
            .len()
            .checked_sub(validation_window)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "rectified recursive validation window exceeds a series history".to_string(),
                )
            })?;
        prefix_rows.extend(history[..train_len].iter().cloned());
    }
    let prefix =
        ForecastFrame::with_metadata(prefix_rows, frame.frequency(), frame.metadata().clone())?;
    let feature_rows = lag_builder.transform_frame(&prefix)?;
    if feature_rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "rectified recursive training prefix produced no lag-feature rows".to_string(),
        ));
    }
    let feature_count = lag_builder.feature_names().len();
    let x = Dataset::from_rows(
        feature_rows
            .iter()
            .map(|row| row.features.clone())
            .collect(),
    )?
    .with_schema(FeatureSchema {
        names: lag_builder.feature_names().to_vec(),
        kinds: vec![FeatureKind::Numeric; feature_count],
    })?;
    let y = feature_rows
        .iter()
        .map(|row| row.target)
        .collect::<Vec<_>>();
    Booster::new_with_backend(booster_config.clone(), backend)?.fit(&x, &y, None)
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
    use cartoboost_accelerator::backend::available_backends;
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
    fn direct_models_reject_unsupported_lag_features_for_short_panels() {
        let frame = short_panel_frame();
        let booster = BoosterConfig {
            n_estimators: 3,
            max_depth: 2,
            min_samples_leaf: 1,
            ..BoosterConfig::default()
        };

        let mut direct = CartoBoostDirectForecaster::new(oversized_lag_config(), booster.clone())
            .expect("direct");
        let error = direct
            .fit_horizon(&frame, 3)
            .expect_err("explicit direct lag features must not be silently removed");
        assert!(error
            .to_string()
            .contains("requires at least 25 prior observations"));

        let mut rectified =
            RectifiedRecursiveForecaster::new(oversized_lag_config(), booster).expect("rectified");
        let error = rectified
            .fit_horizon(&frame, 3)
            .expect_err("explicit rectified lag features must not be silently removed");
        assert!(error
            .to_string()
            .contains("requires at least 25 prior observations"));
    }

    #[test]
    fn available_accelerators_match_cpu_direct_forecasting() {
        let frame = short_panel_frame();
        let lag_config = LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        };
        let booster = BoosterConfig {
            n_estimators: 4,
            max_depth: 2,
            min_samples_leaf: 1,
            ..BoosterConfig::default()
        };
        let mut cpu = CartoBoostDirectForecaster::new(lag_config.clone(), booster.clone()).unwrap();
        cpu.fit_horizon(&frame, 2).unwrap();
        let expected = cpu.predict(2).unwrap();
        for backend in available_backends()
            .into_iter()
            .filter(|name| name != "cpu")
        {
            let mut accelerated = CartoBoostDirectForecaster::new_with_backend(
                lag_config.clone(),
                booster.clone(),
                Some(&backend),
            )
            .unwrap_or_else(|error| panic!("{backend} direct setup failed: {error}"));
            accelerated
                .fit_horizon(&frame, 2)
                .unwrap_or_else(|error| panic!("{backend} direct fit failed: {error}"));
            let actual = accelerated.predict(2).unwrap();
            for (actual, expected) in actual.predictions().iter().zip(expected.predictions()) {
                assert!((actual.mean - expected.mean).abs() <= 1.0e-4);
            }
            assert_eq!(accelerated.metadata()["backend"]["selected"], backend);
        }
    }

    #[test]
    fn available_accelerators_match_cpu_rectified_forecasting() {
        let frame = short_panel_frame();
        let lag_config = LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            ..LagFeatureConfig::default()
        };
        let booster = BoosterConfig {
            n_estimators: 3,
            max_depth: 2,
            min_samples_leaf: 1,
            ..BoosterConfig::default()
        };
        let mut cpu =
            RectifiedRecursiveForecaster::new(lag_config.clone(), booster.clone()).unwrap();
        cpu.fit_horizon(&frame, 2).unwrap();
        let expected = cpu.predict(2).unwrap();
        for backend in available_backends()
            .into_iter()
            .filter(|name| name != "cpu")
        {
            let mut accelerated = RectifiedRecursiveForecaster::new_with_backend(
                lag_config.clone(),
                booster.clone(),
                Some(&backend),
            )
            .unwrap();
            accelerated.fit_horizon(&frame, 2).unwrap();
            let actual = accelerated.predict(2).unwrap();
            for (actual, expected) in actual.predictions().iter().zip(expected.predictions()) {
                assert!((actual.mean - expected.mean).abs() <= 1.0e-4, "{backend}");
            }
        }
    }

    #[test]
    fn rectified_recursive_validation_baselines_are_causal() {
        let original = ForecastFrame::new(
            (1..=10)
                .map(|day| ForecastRow::single(ts(day), f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("frame");
        let changed_future = ForecastFrame::new(
            (1..=10)
                .map(|day| {
                    ForecastRow::single(
                        ts(day),
                        if day <= 8 {
                            f64::from(day)
                        } else {
                            10_000.0 + f64::from(day)
                        },
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig::default()).expect("builder");
        let booster = BoosterConfig {
            n_estimators: 3,
            max_depth: 2,
            min_samples_leaf: 1,
            ..BoosterConfig::default()
        };

        let (original_baselines, original_window) =
            recursive_training_predictions(&original, &builder, &booster, 1, Some("cpu"))
                .expect("baselines");
        let (changed_baselines, changed_window) =
            recursive_training_predictions(&changed_future, &builder, &booster, 1, Some("cpu"))
                .expect("baselines");

        assert_eq!(original_window, 2);
        assert_eq!(changed_window, original_window);
        assert_eq!(original_baselines[0][7], changed_baselines[0][7]);
        assert!(original_baselines[0][7].is_some());
    }
}
