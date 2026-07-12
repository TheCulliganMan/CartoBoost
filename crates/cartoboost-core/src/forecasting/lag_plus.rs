use crate::booster::BoosterConfig;
use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::{
    CartoBoostLagForecaster, ForecastActual, ForecastFrame, ForecastObjective, ForecastPrediction,
    ForecastResult, ForecastRow, Forecaster, GlobalForecastTargetMode, LagFeatureConfig,
};
use crate::{CartoBoostError, Result};
use chrono::{Datelike, NaiveDateTime};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagPlusConfig {
    pub lag_config: LagFeatureConfig,
    pub booster_config: BoosterConfig,
    pub target_mode: GlobalForecastTargetMode,
    pub validation_window: Option<usize>,
    pub objective: ForecastObjective,
    pub shrinkage_strength: f64,
    pub seasonal_bucket_period: Option<usize>,
}

impl LagPlusConfig {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Self {
        Self {
            lag_config,
            booster_config,
            target_mode: GlobalForecastTargetMode::Level,
            validation_window: None,
            objective: ForecastObjective::Rmse,
            shrinkage_strength: 4.0,
            seasonal_bucket_period: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LagPlusForecaster {
    config: LagPlusConfig,
    base: CartoBoostLagForecaster,
    fitted: Option<FittedLagPlus>,
}

#[derive(Debug, Clone)]
struct FittedLagPlus {
    corrections: BTreeMap<usize, f64>,
    seasonal_corrections: BTreeMap<usize, f64>,
    series_corrections: BTreeMap<String, f64>,
    validation_window: usize,
    base_rmse: f64,
    corrected_rmse: f64,
    base_wape: f64,
    corrected_wape: f64,
    objective: ForecastObjective,
    base_objective: f64,
    corrected_objective: f64,
    enabled: bool,
}

impl LagPlusForecaster {
    pub fn new(config: LagPlusConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            base: build_base(&config)?,
            config,
            fitted: None,
        })
    }

    pub fn corrections(&self) -> BTreeMap<usize, f64> {
        self.fitted
            .as_ref()
            .map(|state| state.corrections.clone())
            .unwrap_or_default()
    }

    pub fn seasonal_corrections(&self) -> BTreeMap<usize, f64> {
        self.fitted
            .as_ref()
            .map(|state| state.seasonal_corrections.clone())
            .unwrap_or_default()
    }

    pub fn series_corrections(&self) -> BTreeMap<String, f64> {
        self.fitted
            .as_ref()
            .map(|state| state.series_corrections.clone())
            .unwrap_or_default()
    }
}

impl Forecaster for LagPlusForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let validation_window = effective_validation_window(frame, self.config.validation_window)?;
        let split = split_validation_frame(frame, validation_window)?;
        let calibration = calibrate_corrections(&self.config, &split)?;
        self.base = build_base(&self.config)?;
        self.base.fit(frame)?;
        self.fitted = Some(calibration);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let state = self
            .fitted
            .as_ref()
            .ok_or_else(|| CartoBoostError::InvalidInput("LagPlus must be fitted".to_string()))?;
        let base = self.base.predict(horizon)?;
        let predictions = base
            .predictions()
            .iter()
            .map(|prediction| ForecastPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: self.model_name().to_string(),
                mean: prediction.mean
                    + state
                        .corrections
                        .get(&prediction.horizon)
                        .copied()
                        .unwrap_or(0.0)
                    + seasonal_correction_for(
                        &state.seasonal_corrections,
                        self.config.seasonal_bucket_period,
                        prediction.timestamp,
                    )
                    + state
                        .series_corrections
                        .get(&prediction.series_id)
                        .copied()
                        .unwrap_or(0.0),
            })
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "lag_plus"
    }

    fn metadata(&self) -> Value {
        let fitted = self.fitted.as_ref();
        json!({
            "model": self.model_name(),
            "base_model": "cartoboost_lag",
            "validation_window": fitted.map(|state| state.validation_window),
            "enabled": fitted.map(|state| state.enabled),
            "objective": fitted.map(|state| state.objective.as_str()),
            "base_objective": fitted.map(|state| state.base_objective),
            "corrected_objective": fitted.map(|state| state.corrected_objective),
            "base_rmse": fitted.map(|state| state.base_rmse),
            "corrected_rmse": fitted.map(|state| state.corrected_rmse),
            "base_wape": fitted.map(|state| state.base_wape),
            "corrected_wape": fitted.map(|state| state.corrected_wape),
            "corrections": fitted.map(|state| &state.corrections),
            "seasonal_corrections": fitted.map(|state| &state.seasonal_corrections),
            "series_corrections": fitted.map(|state| &state.series_corrections),
            "seasonal_bucket_period": self.config.seasonal_bucket_period,
            "shrinkage_strength": self.config.shrinkage_strength,
            "base_metadata": self.base.metadata(),
        })
    }
}

struct ValidationSplit {
    train: ForecastFrame,
    validation: Vec<ForecastRow>,
}

fn build_base(config: &LagPlusConfig) -> Result<CartoBoostLagForecaster> {
    CartoBoostLagForecaster::new_with_target_mode(
        config.lag_config.clone(),
        config.booster_config.clone(),
        config.target_mode,
    )
}

fn calibrate_corrections(config: &LagPlusConfig, split: &ValidationSplit) -> Result<FittedLagPlus> {
    let mut base = build_base(config)?;
    base.fit(&split.train)?;
    let horizon = validation_horizon(&split.validation);
    let forecast = base.predict(horizon)?;
    let actuals = validation_actuals(&split.validation);
    let residuals = validation_residuals(&forecast, &actuals)?;
    let mut by_horizon: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for residual in &residuals {
        by_horizon
            .entry(residual.horizon)
            .or_default()
            .push(residual.residual);
    }
    let mut corrections = shrink_mean_corrections(by_horizon, config.shrinkage_strength);
    let mut by_seasonal_bucket: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    if let Some(period) = config.seasonal_bucket_period {
        for residual in &residuals {
            let horizon_correction = corrections.get(&residual.horizon).copied().unwrap_or(0.0);
            by_seasonal_bucket
                .entry(seasonal_bucket(residual.timestamp, period))
                .or_default()
                .push(residual.residual - horizon_correction);
        }
    }
    let mut seasonal_corrections =
        shrink_mean_corrections(by_seasonal_bucket, config.shrinkage_strength);
    let mut by_series: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for residual in &residuals {
        let horizon_correction = corrections.get(&residual.horizon).copied().unwrap_or(0.0);
        let seasonal_correction = seasonal_correction_for(
            &seasonal_corrections,
            config.seasonal_bucket_period,
            residual.timestamp,
        );
        by_series
            .entry(residual.series_id.clone())
            .or_default()
            .push(residual.residual - horizon_correction - seasonal_correction);
    }
    let mut series_corrections =
        shrink_mean_string_corrections(by_series, config.shrinkage_strength);
    let comparison = validation_comparison(
        &forecast,
        &actuals,
        &corrections,
        &seasonal_corrections,
        &series_corrections,
        config.seasonal_bucket_period,
    )?;
    let (base_objective, corrected_objective) = match config.objective {
        ForecastObjective::Rmse => (comparison.base_rmse, comparison.corrected_rmse),
        ForecastObjective::Wape => (comparison.base_wape, comparison.corrected_wape),
        ForecastObjective::RmseWape => (
            0.5 * (comparison.base_normalized_rmse + comparison.base_wape),
            0.5 * (comparison.corrected_normalized_rmse + comparison.corrected_wape),
        ),
    };
    let enabled = corrected_objective <= base_objective;
    if !enabled {
        corrections.clear();
        seasonal_corrections.clear();
        series_corrections.clear();
    }
    Ok(FittedLagPlus {
        corrections,
        seasonal_corrections,
        series_corrections,
        validation_window: split.validation.len() / split.train.series_ids().len().max(1),
        base_rmse: comparison.base_rmse,
        corrected_rmse: comparison.corrected_rmse,
        base_wape: comparison.base_wape,
        corrected_wape: comparison.corrected_wape,
        objective: config.objective,
        base_objective,
        corrected_objective,
        enabled,
    })
}

fn split_validation_frame(
    frame: &ForecastFrame,
    validation_window: usize,
) -> Result<ValidationSplit> {
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for (_, rows) in history_by_series(frame.rows()) {
        if rows.len() <= validation_window {
            return Err(CartoBoostError::InvalidInput(
                "not enough history for LagPlus validation".to_string(),
            ));
        }
        let split_at = rows.len() - validation_window;
        train_rows.extend(rows[..split_at].iter().cloned());
        validation_rows.extend(rows[split_at..].iter().cloned());
    }
    Ok(ValidationSplit {
        train: ForecastFrame::with_metadata(
            train_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?,
        validation: validation_rows,
    })
}

fn validation_actuals(validation: &[ForecastRow]) -> Vec<ForecastActual> {
    let mut actuals = Vec::with_capacity(validation.len());
    for (_, rows) in history_by_series(validation) {
        for (index, row) in rows.iter().enumerate() {
            actuals.push(ForecastActual {
                series_id: row.series_id.clone(),
                timestamp: row.timestamp,
                horizon: index + 1,
                actual: row.target,
            });
        }
    }
    actuals
}

#[derive(Debug, Clone)]
struct ValidationResidual {
    series_id: String,
    horizon: usize,
    timestamp: NaiveDateTime,
    residual: f64,
}

fn validation_residuals(
    forecast: &ForecastResult,
    actuals: &[ForecastActual],
) -> Result<Vec<ValidationResidual>> {
    let mut predictions = BTreeMap::new();
    for prediction in forecast.predictions() {
        predictions.insert(
            (
                prediction.series_id.clone(),
                prediction.timestamp,
                prediction.horizon,
            ),
            prediction.mean,
        );
    }
    let mut residuals = Vec::with_capacity(actuals.len());
    for actual in actuals {
        let key = (actual.series_id.clone(), actual.timestamp, actual.horizon);
        let predicted = predictions.get(&key).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing LagPlus validation forecast for series {}, timestamp {}, horizon {}",
                actual.series_id, actual.timestamp, actual.horizon
            ))
        })?;
        residuals.push(ValidationResidual {
            series_id: actual.series_id.clone(),
            horizon: actual.horizon,
            timestamp: actual.timestamp,
            residual: actual.actual - predicted,
        });
    }
    Ok(residuals)
}

fn shrink_mean_corrections(
    values_by_key: BTreeMap<usize, Vec<f64>>,
    shrinkage_strength: f64,
) -> BTreeMap<usize, f64> {
    let mut corrections = BTreeMap::new();
    for (key, values) in values_by_key {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let reliability = values.len() as f64 / (values.len() as f64 + shrinkage_strength);
        let correction = mean * reliability;
        if correction.is_finite() {
            corrections.insert(key, correction);
        }
    }
    corrections
}

fn shrink_mean_string_corrections(
    values_by_key: BTreeMap<String, Vec<f64>>,
    shrinkage_strength: f64,
) -> BTreeMap<String, f64> {
    let mut corrections = BTreeMap::new();
    for (key, values) in values_by_key {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let reliability = values.len() as f64 / (values.len() as f64 + shrinkage_strength);
        let correction = mean * reliability;
        if correction.is_finite() {
            corrections.insert(key, correction);
        }
    }
    corrections
}

struct ValidationComparison {
    base_rmse: f64,
    corrected_rmse: f64,
    base_normalized_rmse: f64,
    corrected_normalized_rmse: f64,
    base_wape: f64,
    corrected_wape: f64,
}

fn validation_comparison(
    forecast: &ForecastResult,
    actuals: &[ForecastActual],
    corrections: &BTreeMap<usize, f64>,
    seasonal_corrections: &BTreeMap<usize, f64>,
    series_corrections: &BTreeMap<String, f64>,
    seasonal_bucket_period: Option<usize>,
) -> Result<ValidationComparison> {
    let mut predictions = BTreeMap::new();
    for prediction in forecast.predictions() {
        predictions.insert(
            (
                prediction.series_id.clone(),
                prediction.timestamp,
                prediction.horizon,
            ),
            prediction.mean,
        );
    }
    let mut base_sq = 0.0;
    let mut corrected_sq = 0.0;
    let mut base_abs = 0.0;
    let mut corrected_abs = 0.0;
    let mut actual_abs = 0.0;
    for actual in actuals {
        let key = (actual.series_id.clone(), actual.timestamp, actual.horizon);
        let predicted = predictions.get(&key).ok_or_else(|| {
            CartoBoostError::InvalidInput("missing LagPlus validation prediction".to_string())
        })?;
        let base_error = predicted - actual.actual;
        let corrected = predicted
            + corrections.get(&actual.horizon).copied().unwrap_or(0.0)
            + seasonal_correction_for(
                seasonal_corrections,
                seasonal_bucket_period,
                actual.timestamp,
            )
            + series_corrections
                .get(&actual.series_id)
                .copied()
                .unwrap_or(0.0);
        let corrected_error = corrected - actual.actual;
        base_sq += base_error * base_error;
        corrected_sq += corrected_error * corrected_error;
        base_abs += base_error.abs();
        corrected_abs += corrected_error.abs();
        actual_abs += actual.actual.abs();
    }
    let count = actuals.len() as f64;
    let base_rmse = (base_sq / count).sqrt();
    let corrected_rmse = (corrected_sq / count).sqrt();
    let mean_abs_actual = actual_abs / count;
    Ok(ValidationComparison {
        base_rmse,
        corrected_rmse,
        base_normalized_rmse: normalized_rmse(base_rmse, mean_abs_actual),
        corrected_normalized_rmse: normalized_rmse(corrected_rmse, mean_abs_actual),
        base_wape: if actual_abs > 0.0 {
            base_abs / actual_abs
        } else {
            0.0
        },
        corrected_wape: if actual_abs > 0.0 {
            corrected_abs / actual_abs
        } else {
            0.0
        },
    })
}

fn normalized_rmse(rmse: f64, mean_abs_actual: f64) -> f64 {
    if mean_abs_actual > 0.0 {
        rmse / mean_abs_actual
    } else if rmse == 0.0 {
        0.0
    } else {
        rmse / 1e-12
    }
}

fn seasonal_correction_for(
    seasonal_corrections: &BTreeMap<usize, f64>,
    seasonal_bucket_period: Option<usize>,
    timestamp: NaiveDateTime,
) -> f64 {
    seasonal_bucket_period
        .and_then(|period| seasonal_corrections.get(&seasonal_bucket(timestamp, period)))
        .copied()
        .unwrap_or(0.0)
}

fn seasonal_bucket(timestamp: NaiveDateTime, period: usize) -> usize {
    timestamp
        .date()
        .num_days_from_ce()
        .rem_euclid(period as i32) as usize
}

fn validation_horizon(validation: &[ForecastRow]) -> usize {
    history_by_series(validation)
        .values()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
}

fn effective_validation_window(frame: &ForecastFrame, configured: Option<usize>) -> Result<usize> {
    let min_history = history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "LagPlus requires at least one series for validation".to_string(),
            )
        })?;
    if min_history < 2 {
        return Err(CartoBoostError::InvalidInput(
            "LagPlus requires at least two rows per series for a real holdout".to_string(),
        ));
    }
    match configured {
        Some(window) if window < min_history => Ok(window),
        Some(window) => Err(CartoBoostError::InvalidInput(format!(
            "LagPlus validation_window must be smaller than every series history; minimum history is {min_history}, got {window}"
        ))),
        None => Ok((min_history / 5).clamp(1, 8)),
    }
}

fn validate_config(config: &LagPlusConfig) -> Result<()> {
    if matches!(config.validation_window, Some(0)) {
        return Err(CartoBoostError::InvalidInput(
            "LagPlus validation_window must be positive".to_string(),
        ));
    }
    if !config.shrinkage_strength.is_finite() || config.shrinkage_strength < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "LagPlus shrinkage_strength must be finite and non-negative".to_string(),
        ));
    }
    if matches!(config.seasonal_bucket_period, Some(0)) {
        return Err(CartoBoostError::InvalidInput(
            "LagPlus seasonal_bucket_period must be positive".to_string(),
        ));
    }
    Ok(())
}
