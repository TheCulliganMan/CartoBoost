use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalStandardScaler {
    min_scale: f64,
    stats: BTreeMap<String, LocalScaleStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalScaleStats {
    pub mean: f64,
    pub scale: f64,
}

impl LocalStandardScaler {
    pub fn new(min_scale: f64) -> Result<Self> {
        if !min_scale.is_finite() || min_scale <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "local standard scaler min_scale must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            min_scale,
            stats: BTreeMap::new(),
        })
    }

    pub fn fit_transform(&mut self, frame: &ForecastFrame) -> Result<ForecastFrame> {
        let mut rows = Vec::with_capacity(frame.rows().len());
        let mut stats = BTreeMap::new();
        for (series_id, history) in history_by_series(frame.rows()) {
            let stat = fit_stats(&history, self.min_scale)?;
            stats.insert(series_id.clone(), stat);
            rows.extend(history.into_iter().map(|row| {
                ForecastRow::new(
                    row.series_id,
                    row.timestamp,
                    (row.target - stat.mean) / stat.scale,
                )
            }));
        }
        self.stats = stats;
        ForecastFrame::with_metadata(rows, frame.frequency(), frame.metadata().clone())
    }

    pub fn inverse_result(
        &self,
        result: &ForecastResult,
        model_name: &str,
    ) -> Result<ForecastResult> {
        let predictions = result
            .predictions()
            .iter()
            .map(|prediction| {
                let stat = self.stats.get(&prediction.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing local scale stats for series {}",
                        prediction.series_id
                    ))
                })?;
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: model_name.to_string(),
                    mean: prediction.mean * stat.scale + stat.mean,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new(predictions)
    }

    pub fn stats(&self) -> &BTreeMap<String, LocalScaleStats> {
        &self.stats
    }

    pub fn metadata(&self) -> Value {
        json!({
            "transform": "local_standard_scaler",
            "min_scale": self.min_scale,
            "series_count": self.stats.len(),
            "stats": self.stats,
        })
    }
}

fn fit_stats(history: &[ForecastRow], min_scale: f64) -> Result<LocalScaleStats> {
    if history.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "local standard scaler requires at least one row per series".to_string(),
        ));
    }
    let mean = history.iter().map(|row| row.target).sum::<f64>() / history.len() as f64;
    let variance = history
        .iter()
        .map(|row| {
            let centered = row.target - mean;
            centered * centered
        })
        .sum::<f64>()
        / history.len() as f64;
    let scale = variance.sqrt().max(min_scale);
    if !mean.is_finite() || !scale.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "local standard scaler produced non-finite stats".to_string(),
        ));
    }
    Ok(LocalScaleStats { mean, scale })
}

pub struct LocalStandardScaledForecaster {
    scaler: LocalStandardScaler,
    inner: Box<dyn Forecaster>,
    model_name: &'static str,
}

impl LocalStandardScaledForecaster {
    pub fn new(
        inner: Box<dyn Forecaster>,
        min_scale: f64,
        model_name: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            scaler: LocalStandardScaler::new(min_scale)?,
            inner,
            model_name,
        })
    }
}

impl Forecaster for LocalStandardScaledForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_observed_targets_for_model(self.model_name())?;
        let transformed = self.scaler.fit_transform(frame)?;
        self.inner.fit(&transformed)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let transformed = self.inner.predict(horizon)?;
        self.scaler.inverse_result(&transformed, self.model_name)
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "target_transform": self.scaler.metadata(),
            "inner": self.inner.metadata(),
        })
    }
}

pub struct Log1pForecaster {
    inner: Box<dyn Forecaster>,
    model_name: &'static str,
}

impl Log1pForecaster {
    pub fn new(inner: Box<dyn Forecaster>, model_name: &'static str) -> Self {
        Self { inner, model_name }
    }
}

impl Forecaster for Log1pForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_observed_targets_for_model(self.model_name())?;
        let rows = frame
            .rows()
            .iter()
            .map(|row| {
                if row.target < 0.0 {
                    return Err(CartoBoostError::InvalidInput(
                        "log1p target transform requires nonnegative targets".to_string(),
                    ));
                }
                Ok(ForecastRow::new(
                    row.series_id.clone(),
                    row.timestamp,
                    row.target.ln_1p(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let transformed =
            ForecastFrame::with_metadata(rows, frame.frequency(), frame.metadata().clone())?;
        self.inner.fit(&transformed)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let transformed = self.inner.predict(horizon)?;
        let predictions = transformed
            .predictions()
            .iter()
            .map(|prediction| ForecastPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: self.model_name.to_string(),
                mean: prediction.mean.exp_m1().max(0.0),
            })
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "target_transform": {
                "transform": "log1p",
                "inverse": "expm1_clamped_nonnegative",
            },
            "inner": self.inner.metadata(),
        })
    }
}
