use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::{
    ForecastFrame, ForecastObjective, ForecastPrediction, ForecastResult, Forecaster,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermittentDemandConfig {
    pub alpha: f64,
    pub beta: f64,
    pub adida_bucket_size: usize,
    pub validation_window: Option<usize>,
    pub objective: ForecastObjective,
}

impl Default for IntermittentDemandConfig {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.2,
            adida_bucket_size: 7,
            validation_window: None,
            objective: ForecastObjective::Wape,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntermittentDemandMethod {
    Zero,
    Croston,
    Sba,
    Tsb,
    Adida,
}

impl IntermittentDemandMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Croston => "croston",
            Self::Sba => "sba",
            Self::Tsb => "tsb",
            Self::Adida => "adida",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct IntermittentDemandForecaster {
    config: IntermittentDemandConfig,
    fitted: Option<FittedIntermittentDemand>,
}

#[derive(Debug, Clone)]
struct FittedIntermittentDemand {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedIntermittentSeries>,
    validation_window: usize,
}

#[derive(Debug, Clone)]
struct FittedIntermittentSeries {
    method: IntermittentDemandMethod,
    level: f64,
    validation_loss: f64,
    zero_fraction: f64,
}

#[derive(Debug, Clone)]
struct FittedFixedIntermittentDemand {
    frame: ForecastFrame,
    levels: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CrostonForecaster {
    alpha: f64,
    fitted: Option<FittedFixedIntermittentDemand>,
}

#[derive(Debug, Clone)]
pub struct SbaForecaster {
    alpha: f64,
    fitted: Option<FittedFixedIntermittentDemand>,
}

#[derive(Debug, Clone)]
pub struct TsbForecaster {
    alpha: f64,
    beta: f64,
    fitted: Option<FittedFixedIntermittentDemand>,
}

impl IntermittentDemandForecaster {
    pub fn new(config: IntermittentDemandConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            fitted: None,
        })
    }

    pub fn fitted_methods(&self) -> BTreeMap<String, IntermittentDemandMethod> {
        self.fitted
            .as_ref()
            .map(|state| {
                state
                    .series
                    .iter()
                    .map(|(series_id, fitted)| (series_id.clone(), fitted.method))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl CrostonForecaster {
    pub fn new(alpha: f64) -> Result<Self> {
        validate_unit_interval("alpha", alpha)?;
        Ok(Self {
            alpha,
            fitted: None,
        })
    }
}

impl Default for CrostonForecaster {
    fn default() -> Self {
        Self::new(0.2).expect("default alpha is valid")
    }
}

impl SbaForecaster {
    pub fn new(alpha: f64) -> Result<Self> {
        validate_unit_interval("alpha", alpha)?;
        Ok(Self {
            alpha,
            fitted: None,
        })
    }
}

impl Default for SbaForecaster {
    fn default() -> Self {
        Self::new(0.2).expect("default alpha is valid")
    }
}

impl TsbForecaster {
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        validate_unit_interval("alpha", alpha)?;
        validate_unit_interval("beta", beta)?;
        Ok(Self {
            alpha,
            beta,
            fitted: None,
        })
    }
}

impl Default for TsbForecaster {
    fn default() -> Self {
        Self::new(0.2, 0.2).expect("default smoothing parameters are valid")
    }
}

impl Forecaster for IntermittentDemandForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        validate_nonnegative_frame(frame)?;
        let validation_window = effective_validation_window(frame, self.config.validation_window);
        let mut series = BTreeMap::new();
        for (series_id, rows) in history_by_series(frame.rows()) {
            if rows.len() <= validation_window {
                return Err(CartoBoostError::InvalidInput(
                    "not enough history for intermittent demand validation".to_string(),
                ));
            }
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let fitted = fit_series(&values, validation_window, &self.config)?;
            series.insert(series_id, fitted);
        }
        self.fitted = Some(FittedIntermittentDemand {
            frame: frame.clone(),
            series,
            validation_window,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            CartoBoostError::InvalidInput("intermittent demand forecaster must be fitted".into())
        })?;
        let histories = history_by_series(fitted.frame.rows());
        let series_predictions = histories
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let state = fitted.series.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing intermittent fitted state for series {series_id}"
                    ))
                })?;
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: self.model_name().to_string(),
                        mean: state.level.max(0.0),
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions = Vec::with_capacity(histories.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "intermittent_demand"
    }

    fn metadata(&self) -> Value {
        let fitted = self.fitted.as_ref();
        json!({
            "model": self.model_name(),
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "adida_bucket_size": self.config.adida_bucket_size,
            "objective": self.config.objective.as_str(),
            "validation_window": fitted.map(|state| state.validation_window),
            "series": fitted.map(|state| {
                state.series.iter().map(|(series_id, series)| {
                    json!({
                        "series_id": series_id,
                        "method": series.method.as_str(),
                        "level": series.level,
                        "validation_loss": series.validation_loss,
                        "zero_fraction": series.zero_fraction,
                    })
                }).collect::<Vec<_>>()
            }),
        })
    }
}

impl Forecaster for CrostonForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(fit_fixed_intermittent(
            frame,
            IntermittentDemandMethod::Croston,
            self.alpha,
            None,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        predict_fixed_intermittent(self.model_name(), self.fitted.as_ref(), horizon)
    }

    fn model_name(&self) -> &'static str {
        "croston"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "alpha": self.alpha,
            "series_count": self.fitted.as_ref().map(|state| state.levels.len()),
        })
    }
}

impl Forecaster for SbaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(fit_fixed_intermittent(
            frame,
            IntermittentDemandMethod::Sba,
            self.alpha,
            None,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        predict_fixed_intermittent(self.model_name(), self.fitted.as_ref(), horizon)
    }

    fn model_name(&self) -> &'static str {
        "sba"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "alpha": self.alpha,
            "series_count": self.fitted.as_ref().map(|state| state.levels.len()),
        })
    }
}

impl Forecaster for TsbForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(fit_fixed_intermittent(
            frame,
            IntermittentDemandMethod::Tsb,
            self.alpha,
            Some(self.beta),
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        predict_fixed_intermittent(self.model_name(), self.fitted.as_ref(), horizon)
    }

    fn model_name(&self) -> &'static str {
        "tsb"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "alpha": self.alpha,
            "beta": self.beta,
            "series_count": self.fitted.as_ref().map(|state| state.levels.len()),
        })
    }
}

pub fn croston_forecast(values: &[f64], horizon: usize, alpha: f64) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, None)?;
    Ok(vec![croston_level(values, alpha)?; horizon])
}

pub fn sba_forecast(values: &[f64], horizon: usize, alpha: f64) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, None)?;
    Ok(vec![
        croston_level(values, alpha)? * (1.0 - alpha / 2.0);
        horizon
    ])
}

pub fn tsb_forecast(values: &[f64], horizon: usize, alpha: f64, beta: f64) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, Some(beta))?;
    Ok(vec![tsb_level(values, alpha, beta)?; horizon])
}

pub fn adida_forecast(
    values: &[f64],
    horizon: usize,
    bucket_size: usize,
    alpha: f64,
) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, None)?;
    if bucket_size == 0 {
        return Err(CartoBoostError::InvalidInput(
            "ADIDA bucket_size must be positive".to_string(),
        ));
    }
    let mut buckets = Vec::new();
    for chunk in values.chunks(bucket_size) {
        buckets.push(chunk.iter().sum::<f64>());
    }
    let aggregate = croston_level(&buckets, alpha)? / bucket_size as f64;
    Ok(vec![aggregate; horizon])
}

fn fit_series(
    values: &[f64],
    validation_window: usize,
    config: &IntermittentDemandConfig,
) -> Result<FittedIntermittentSeries> {
    let split_at = values.len() - validation_window;
    let train = &values[..split_at];
    let validation = &values[split_at..];
    let zero_fraction =
        values.iter().filter(|value| **value == 0.0).count() as f64 / values.len().max(1) as f64;
    let candidates = candidate_methods(values);
    let mut best: Option<(IntermittentDemandMethod, f64)> = None;
    for method in candidates {
        let loss = if validation_window == 0 {
            method_forecast(method, values, 1, config)?;
            0.0
        } else {
            let forecast = method_forecast(method, train, validation.len(), config)?;
            objective_loss(validation, &forecast, config.objective)?
        };
        if !loss.is_finite() {
            continue;
        }
        let replace = best
            .as_ref()
            .map(|(best_method, best_loss)| {
                loss < *best_loss || (loss == *best_loss && method < *best_method)
            })
            .unwrap_or(true);
        if replace {
            best = Some((method, loss));
        }
    }
    let (method, validation_loss) = best.ok_or_else(|| {
        CartoBoostError::InvalidInput("no intermittent demand method could validate".to_string())
    })?;
    let level = method_forecast(method, values, 1, config)?
        .into_iter()
        .next()
        .unwrap_or(0.0)
        .max(0.0);
    Ok(FittedIntermittentSeries {
        method,
        level,
        validation_loss,
        zero_fraction,
    })
}

fn candidate_methods(values: &[f64]) -> Vec<IntermittentDemandMethod> {
    if values.iter().all(|value| *value == 0.0) {
        return vec![IntermittentDemandMethod::Zero];
    }
    vec![
        IntermittentDemandMethod::Tsb,
        IntermittentDemandMethod::Sba,
        IntermittentDemandMethod::Croston,
        IntermittentDemandMethod::Adida,
    ]
}

fn method_forecast(
    method: IntermittentDemandMethod,
    values: &[f64],
    horizon: usize,
    config: &IntermittentDemandConfig,
) -> Result<Vec<f64>> {
    match method {
        IntermittentDemandMethod::Zero => Ok(vec![0.0; horizon]),
        IntermittentDemandMethod::Croston => croston_forecast(values, horizon, config.alpha),
        IntermittentDemandMethod::Sba => sba_forecast(values, horizon, config.alpha),
        IntermittentDemandMethod::Tsb => tsb_forecast(values, horizon, config.alpha, config.beta),
        IntermittentDemandMethod::Adida => {
            adida_forecast(values, horizon, config.adida_bucket_size, config.alpha)
        }
    }
}

fn fit_fixed_intermittent(
    frame: &ForecastFrame,
    method: IntermittentDemandMethod,
    alpha: f64,
    beta: Option<f64>,
) -> Result<FittedFixedIntermittentDemand> {
    validate_nonnegative_frame(frame)?;
    let config = IntermittentDemandConfig {
        alpha,
        beta: beta.unwrap_or(alpha),
        ..IntermittentDemandConfig::default()
    };
    let mut levels = BTreeMap::new();
    for (series_id, rows) in history_by_series(frame.rows()) {
        let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
        let level = method_forecast(method, &values, 1, &config)?
            .into_iter()
            .next()
            .unwrap_or(0.0)
            .max(0.0);
        levels.insert(series_id, level);
    }
    Ok(FittedFixedIntermittentDemand {
        frame: frame.clone(),
        levels,
    })
}

fn predict_fixed_intermittent(
    model_name: &'static str,
    fitted: Option<&FittedFixedIntermittentDemand>,
    horizon: usize,
) -> Result<ForecastResult> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    let fitted = fitted.ok_or_else(|| {
        CartoBoostError::InvalidInput(format!("{model_name} forecaster must be fitted"))
    })?;
    let histories = history_by_series(fitted.frame.rows());
    let mut predictions = Vec::with_capacity(histories.len().saturating_mul(horizon));
    for (series_id, history) in histories {
        let last = history
            .last()
            .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?;
        let level = *fitted.levels.get(&series_id).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing intermittent fitted level for series {series_id}"
            ))
        })?;
        for step in 1..=horizon {
            predictions.push(ForecastPrediction {
                series_id: series_id.clone(),
                timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                horizon: step,
                model: model_name.to_string(),
                mean: level,
            });
        }
    }
    ForecastResult::new(predictions)
}

fn objective_loss(
    actuals: &[f64],
    predictions: &[f64],
    objective: ForecastObjective,
) -> Result<f64> {
    if actuals.len() != predictions.len() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent validation length mismatch".to_string(),
        ));
    }
    match objective {
        ForecastObjective::Rmse => {
            let mse = actuals
                .iter()
                .zip(predictions)
                .map(|(actual, predicted)| {
                    let error = predicted - actual;
                    error * error
                })
                .sum::<f64>()
                / actuals.len().max(1) as f64;
            Ok(mse.sqrt())
        }
        ForecastObjective::Wape => wape_loss(actuals, predictions),
        ForecastObjective::RmseWape => {
            let rmse = objective_loss(actuals, predictions, ForecastObjective::Rmse)?;
            let mean_abs_actual = actuals.iter().map(|actual| actual.abs()).sum::<f64>()
                / actuals.len().max(1) as f64;
            let normalized_rmse = if mean_abs_actual > 0.0 {
                rmse / mean_abs_actual
            } else if rmse == 0.0 {
                0.0
            } else {
                rmse / 1e-12
            };
            Ok(0.5 * (normalized_rmse + wape_loss(actuals, predictions)?))
        }
    }
}

fn wape_loss(actuals: &[f64], predictions: &[f64]) -> Result<f64> {
    let denominator = actuals.iter().map(|actual| actual.abs()).sum::<f64>();
    if denominator == 0.0 {
        return Ok(predictions.iter().map(|prediction| prediction.abs()).sum());
    }
    let numerator = actuals
        .iter()
        .zip(predictions)
        .map(|(actual, predicted)| (predicted - actual).abs())
        .sum::<f64>();
    Ok(numerator / denominator)
}

fn effective_validation_window(frame: &ForecastFrame, configured: Option<usize>) -> usize {
    if let Some(window) = configured {
        return window;
    }
    let min_history = history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    if min_history < 2 {
        return 0;
    }
    (min_history / 5).clamp(1, 8)
}

fn validate_nonnegative_frame(frame: &ForecastFrame) -> Result<()> {
    for (index, row) in frame.rows().iter().enumerate() {
        if !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent demand target at row {index} must be finite"
            )));
        }
        if row.target < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent demand target at row {index} must be non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_config(config: &IntermittentDemandConfig) -> Result<()> {
    validate_unit_interval("alpha", config.alpha)?;
    validate_unit_interval("beta", config.beta)?;
    if config.adida_bucket_size == 0 {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand ADIDA bucket_size must be positive".to_string(),
        ));
    }
    if matches!(config.validation_window, Some(0)) {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand validation_window must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_inputs(values: &[f64], horizon: usize, alpha: f64, beta: Option<f64>) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "intermittent forecast horizon must be positive".to_string(),
        ));
    }
    validate_unit_interval("alpha", alpha)?;
    if let Some(beta) = beta {
        validate_unit_interval("beta", beta)?;
    }
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent forecast requires at least one observation".to_string(),
        ));
    }
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent observation at index {idx} must be finite"
            )));
        }
        if *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent observation at index {idx} must be non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be in (0, 1]"
        )));
    }
    Ok(())
}

fn croston_level(values: &[f64], alpha: f64) -> Result<f64> {
    let first_nonzero = values
        .iter()
        .position(|value| *value > 0.0)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "intermittent forecast requires at least one non-zero observation".to_string(),
            )
        })?;
    let mut demand = values[first_nonzero];
    let mut interval = (first_nonzero + 1) as f64;
    let mut elapsed = 0usize;
    for value in values.iter().skip(first_nonzero + 1) {
        elapsed += 1;
        if *value > 0.0 {
            demand += alpha * (*value - demand);
            interval += alpha * (elapsed as f64 - interval);
            elapsed = 0;
        }
    }
    if interval <= 0.0 || !interval.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent interval estimate is invalid".to_string(),
        ));
    }
    Ok(demand / interval)
}

fn tsb_level(values: &[f64], alpha: f64, beta: f64) -> Result<f64> {
    let first_nonzero = values.iter().find(|value| **value > 0.0).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "TSB forecast requires at least one non-zero observation".to_string(),
        )
    })?;
    let mut demand = *first_nonzero;
    let mut probability = 0.0;
    for value in values {
        let occurrence = if *value > 0.0 { 1.0 } else { 0.0 };
        probability += beta * (occurrence - probability);
        if *value > 0.0 {
            demand += alpha * (*value - demand);
        }
    }
    Ok(probability * demand)
}
