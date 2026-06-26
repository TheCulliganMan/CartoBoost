use cartoboost_core::forecasting::{
    ForecastFrame, ForecastFrequency, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use cartoboost_core::{CartoBoostError, Result as CoreResult};
use chrono::{Datelike, NaiveDateTime, Timelike};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

use super::scaler::StandardScaler;
use crate::{NeuralError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendMode {
    Off,
    PiecewiseLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentMode {
    Additive,
    Multiplicative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuralPanelMode {
    Global,
    Local,
    Glocal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralPanelConfig {
    pub n_lags: usize,
    pub n_forecasts: usize,
    pub quantiles: Vec<f64>,
    pub trend: TrendMode,
    pub n_changepoints: usize,
    pub changepoints_range: f64,
    pub daily_fourier_order: usize,
    pub weekly_fourier_order: usize,
    pub yearly_fourier_order: usize,
    pub custom_seasonalities: BTreeMap<String, (f64, usize)>,
    pub seasonality_mode: ComponentMode,
    pub events: BTreeMap<String, Vec<i32>>,
    pub event_mode: ComponentMode,
    pub future_regressors: BTreeMap<String, ComponentMode>,
    pub lagged_regressors: BTreeMap<String, usize>,
    pub ar_layers: Vec<usize>,
    pub lagged_reg_layers: Vec<usize>,
    pub trend_mode: NeuralPanelMode,
    pub seasonality_global_local: NeuralPanelMode,
    pub local_l2: f64,
    pub seed: u64,
}

impl Default for NeuralPanelConfig {
    fn default() -> Self {
        Self {
            n_lags: 8,
            n_forecasts: 1,
            quantiles: vec![0.5],
            trend: TrendMode::PiecewiseLinear,
            n_changepoints: 10,
            changepoints_range: 0.8,
            daily_fourier_order: 0,
            weekly_fourier_order: 0,
            yearly_fourier_order: 0,
            custom_seasonalities: BTreeMap::new(),
            seasonality_mode: ComponentMode::Additive,
            events: BTreeMap::new(),
            event_mode: ComponentMode::Additive,
            future_regressors: BTreeMap::new(),
            lagged_regressors: BTreeMap::new(),
            ar_layers: Vec::new(),
            lagged_reg_layers: Vec::new(),
            trend_mode: NeuralPanelMode::Global,
            seasonality_global_local: NeuralPanelMode::Global,
            local_l2: 0.0,
            seed: 0,
        }
    }
}

impl NeuralPanelConfig {
    pub fn validate(&mut self) -> Result<()> {
        if self.n_forecasts == 0 {
            return Err(NeuralError::InvalidArgument(
                "n_forecasts must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.changepoints_range) || self.changepoints_range == 0.0 {
            return Err(NeuralError::InvalidArgument(
                "changepoints_range must be in (0, 1]".to_string(),
            ));
        }
        if !self.local_l2.is_finite() || self.local_l2 < 0.0 {
            return Err(NeuralError::InvalidArgument(
                "local_l2 must be finite and non-negative".to_string(),
            ));
        }
        for hidden in self.ar_layers.iter().chain(self.lagged_reg_layers.iter()) {
            if *hidden == 0 {
                return Err(NeuralError::InvalidArgument(
                    "network layer sizes must be positive".to_string(),
                ));
            }
        }
        for (name, period_order) in &self.custom_seasonalities {
            if name.is_empty() {
                return Err(NeuralError::InvalidArgument(
                    "custom seasonality names must not be empty".to_string(),
                ));
            }
            if !period_order.0.is_finite() || period_order.0 <= 0.0 || period_order.1 == 0 {
                return Err(NeuralError::InvalidArgument(format!(
                    "custom seasonality '{name}' must have positive period and order"
                )));
            }
        }
        for (name, lag) in &self.lagged_regressors {
            if name.is_empty() || *lag == 0 {
                return Err(NeuralError::InvalidArgument(
                    "lagged regressors require non-empty names and positive lags".to_string(),
                ));
            }
        }
        self.quantiles = normalized_quantiles(&self.quantiles)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralPanelWindow {
    pub series_id: String,
    pub time: Vec<f64>,
    pub lags: Vec<f64>,
    pub targets: Vec<f64>,
    pub lagged_covariates: BTreeMap<String, Vec<f64>>,
    pub future_features: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralPanelWindowDataset {
    windows: Vec<NeuralPanelWindow>,
    tails: BTreeMap<String, Vec<f64>>,
    future_feature_names: Vec<String>,
    series_ids: Vec<String>,
}

impl NeuralPanelWindowDataset {
    pub fn from_frame(frame: &ForecastFrame, config: &NeuralPanelConfig) -> Result<Self> {
        let mut config = config.clone();
        config.validate()?;
        let mut windows = Vec::new();
        let mut tails = BTreeMap::new();
        let future_feature_names = future_feature_names(&config);
        let series_ids = frame.series_ids();
        for series_id in &series_ids {
            let rows = frame.rows_for_series(series_id);
            let required = config.n_lags + config.n_forecasts;
            if rows.len() < required {
                return Err(NeuralError::InvalidArgument(format!(
                    "series '{series_id}' needs at least {required} rows for n_lags={} and n_forecasts={}",
                    config.n_lags, config.n_forecasts
                )));
            }
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let tail_start = values.len().saturating_sub(config.n_lags.max(1));
            tails.insert(series_id.clone(), values[tail_start..].to_vec());
            for start in 0..=rows.len() - required {
                let lag_end = start + config.n_lags;
                let target_end = lag_end + config.n_forecasts;
                let mut lagged_covariates = BTreeMap::new();
                for (name, lag) in &config.lagged_regressors {
                    let cov_start = lag_end - lag;
                    lagged_covariates.insert(
                        name.clone(),
                        rows[cov_start..lag_end]
                            .iter()
                            .map(|row| required_covariate(row, name))
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                let future_features = rows[start..target_end]
                    .iter()
                    .map(|row| build_future_features(row, &future_feature_names))
                    .collect::<Result<Vec<_>>>()?;
                windows.push(NeuralPanelWindow {
                    series_id: series_id.clone(),
                    time: (0..required).map(|idx| idx as f64).collect(),
                    lags: values[start..lag_end].to_vec(),
                    targets: values[lag_end..target_end].to_vec(),
                    lagged_covariates,
                    future_features,
                });
            }
        }
        Ok(Self {
            windows,
            tails,
            future_feature_names,
            series_ids,
        })
    }

    pub fn windows(&self) -> &[NeuralPanelWindow] {
        &self.windows
    }

    pub fn tails(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.tails
    }

    pub fn future_feature_names(&self) -> &[String] {
        &self.future_feature_names
    }

    pub fn series_ids(&self) -> &[String] {
        &self.series_ids
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPanelForecaster {
    config: NeuralPanelConfig,
    scaler: Option<StandardScaler>,
    frequency: Option<ForecastFrequency>,
    last_rows: BTreeMap<String, ForecastRow>,
    series_ids: Vec<String>,
    global_level: f64,
    global_slope: f64,
    local_levels: BTreeMap<String, f64>,
    local_slopes: BTreeMap<String, f64>,
    #[serde(default)]
    target_tails: BTreeMap<String, Vec<f64>>,
    ar_weights: Vec<f64>,
    covariate_weights: BTreeMap<String, f64>,
    #[serde(default)]
    feature_weights: BTreeMap<String, f64>,
    future_regressor_weights: BTreeMap<String, f64>,
    feature_schema: Vec<String>,
    train_cutoff: Option<String>,
}

impl NeuralPanelForecaster {
    pub fn new(mut config: NeuralPanelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            scaler: None,
            frequency: None,
            last_rows: BTreeMap::new(),
            series_ids: Vec::new(),
            global_level: 0.0,
            global_slope: 0.0,
            local_levels: BTreeMap::new(),
            local_slopes: BTreeMap::new(),
            target_tails: BTreeMap::new(),
            ar_weights: Vec::new(),
            covariate_weights: BTreeMap::new(),
            feature_weights: BTreeMap::new(),
            future_regressor_weights: BTreeMap::new(),
            feature_schema: Vec::new(),
            train_cutoff: None,
        })
    }

    pub fn config(&self) -> &NeuralPanelConfig {
        &self.config
    }

    pub fn window_dataset(&self, frame: &ForecastFrame) -> Result<NeuralPanelWindowDataset> {
        NeuralPanelWindowDataset::from_frame(frame, &self.config)
    }

    pub fn quantile_levels(&self) -> &[f64] {
        &self.config.quantiles
    }

    pub fn to_json_string(&self) -> CoreResult<String> {
        serde_json::to_string_pretty(self).map_err(CartoBoostError::from)
    }

    pub fn from_json_string(value: &str) -> CoreResult<Self> {
        serde_json::from_str(value).map_err(CartoBoostError::from)
    }

    pub fn predict_tensor(&self, horizon: usize) -> CoreResult<BTreeMap<String, Vec<Vec<f64>>>> {
        let frequency = self.frequency.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "NeuralPanelForecaster must be fit before predict".to_string(),
            )
        })?;
        let scaler = self.scaler.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "NeuralPanelForecaster must be fit before predict".to_string(),
            )
        })?;
        let mut by_series = BTreeMap::new();
        for series_id in &self.series_ids {
            let last_row = self.last_rows.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted timestamp tail for series '{series_id}'"
                ))
            })?;
            let local_level = self.local_levels.get(series_id).copied().unwrap_or(0.0);
            let local_slope = self.local_slopes.get(series_id).copied().unwrap_or(0.0);
            let ar_effect = self.autoregressive_effect(series_id, local_level, local_slope);
            let mut rows = Vec::with_capacity(horizon);
            for step in 1..=horizon {
                let timestamp = frequency.advance(last_row.timestamp, step)?;
                let (additive, multiplicative) = self.nonstationary_effect(timestamp, None)?;
                let median_scaled = if self.config.trend == TrendMode::Off {
                    self.global_level + local_level
                } else {
                    self.global_level
                        + local_level
                        + (self.global_slope + local_slope) * step as f64
                };
                let median_scaled = median_scaled
                    + additive
                    + median_scaled * multiplicative
                    + ar_effect / step as f64;
                let median = scaler.inverse_transform(median_scaled);
                rows.push(repaired_quantiles(median, &self.config.quantiles));
            }
            by_series.insert(series_id.clone(), rows);
        }
        Ok(by_series)
    }

    fn autoregressive_effect(&self, series_id: &str, local_level: f64, local_slope: f64) -> f64 {
        if self.ar_weights.is_empty() {
            return 0.0;
        }
        let Some(tail) = self.target_tails.get(series_id) else {
            return 0.0;
        };
        let start_index = tail.len().saturating_sub(self.ar_weights.len());
        tail[start_index..]
            .iter()
            .zip(self.ar_weights.iter())
            .enumerate()
            .map(|(idx, (value, weight))| {
                let baseline = if self.config.trend == TrendMode::Off {
                    self.global_level + local_level
                } else {
                    self.global_level
                        + local_level
                        + (self.global_slope + local_slope) * (idx + start_index + 1) as f64
                };
                (value - baseline) * weight
            })
            .sum()
    }

    pub fn predict_quantiles_json_string(&self, horizon: usize) -> CoreResult<String> {
        let tensor = self.predict_tensor(horizon)?;
        serde_json::to_string_pretty(&json!({
            "quantile_levels": self.config.quantiles,
            "series": tensor,
        }))
        .map_err(CartoBoostError::from)
    }

    fn nonstationary_effect(
        &self,
        timestamp: NaiveDateTime,
        covariates: Option<&BTreeMap<String, f64>>,
    ) -> CoreResult<(f64, f64)> {
        let mut additive = 0.0;
        let mut multiplicative = 0.0;
        for name in &self.feature_schema {
            let value = feature_value_for_timestamp(name, timestamp, covariates)?;
            let weight = self.feature_weights.get(name).copied().unwrap_or(0.0);
            match feature_component_mode(&self.config, name) {
                ComponentMode::Additive => additive += value * weight,
                ComponentMode::Multiplicative => multiplicative += value * weight,
            }
        }
        Ok((additive, multiplicative))
    }
}

impl Forecaster for NeuralPanelForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> CoreResult<()> {
        let dataset = self
            .window_dataset(frame)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        let targets = frame
            .rows()
            .iter()
            .map(|row| row.target)
            .collect::<Vec<_>>();
        let scaler = StandardScaler::fit(&targets)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        let scaled_targets = targets
            .iter()
            .map(|value| scaler.transform(*value))
            .collect::<Vec<_>>();
        self.global_level = mean(&scaled_targets);
        self.global_slope = global_slope(frame, &scaler);
        self.series_ids = dataset.series_ids().to_vec();
        self.feature_schema = dataset.future_feature_names().to_vec();
        self.frequency = Some(frame.frequency());
        self.scaler = Some(scaler);
        self.target_tails = dataset
            .tails()
            .iter()
            .map(|(series_id, values)| {
                (
                    series_id.clone(),
                    values
                        .iter()
                        .map(|value| scaler.transform(*value))
                        .collect(),
                )
            })
            .collect();
        self.last_rows = self
            .series_ids
            .iter()
            .filter_map(|series_id| {
                frame
                    .rows_for_series(series_id)
                    .last()
                    .map(|row| (series_id.clone(), (*row).clone()))
            })
            .collect();
        self.train_cutoff = frame
            .rows()
            .iter()
            .map(|row| row.timestamp)
            .max()
            .map(|timestamp| timestamp.to_string());
        self.local_levels.clear();
        self.local_slopes.clear();
        if self.config.trend_mode != NeuralPanelMode::Global {
            for series_id in &self.series_ids {
                let rows = frame.rows_for_series(series_id);
                let scaled = rows
                    .iter()
                    .map(|row| self.scaler.expect("scaler").transform(row.target))
                    .collect::<Vec<_>>();
                let local_mean = mean(&scaled) - self.global_level;
                let local_slope = slope_for_values(&scaled) - self.global_slope;
                let shrink = 1.0 / (1.0 + self.config.local_l2);
                self.local_levels
                    .insert(series_id.clone(), local_mean * shrink);
                self.local_slopes
                    .insert(series_id.clone(), local_slope * shrink);
            }
        }
        self.ar_weights = deterministic_weights(self.config.n_lags, self.config.seed, 0.013);
        self.covariate_weights = self
            .config
            .lagged_regressors
            .keys()
            .enumerate()
            .map(|(idx, name)| {
                (
                    name.clone(),
                    deterministic_scalar(self.config.seed, idx, 0.019),
                )
            })
            .collect();
        self.feature_weights = fit_nonstationary_feature_weights(
            &self.config,
            &dataset,
            &scaler,
            |series_id, offset| {
                let local_level = self.local_levels.get(series_id).copied().unwrap_or(0.0);
                let local_slope = self.local_slopes.get(series_id).copied().unwrap_or(0.0);
                if self.config.trend == TrendMode::Off {
                    self.global_level + local_level
                } else {
                    self.global_level
                        + local_level
                        + (self.global_slope + local_slope) * offset as f64
                }
            },
        );
        self.future_regressor_weights = self
            .config
            .future_regressors
            .keys()
            .enumerate()
            .map(|(idx, name)| {
                (
                    name.clone(),
                    deterministic_scalar(self.config.seed, idx, 0.023),
                )
            })
            .collect();
        Ok(())
    }

    fn predict(&self, horizon: usize) -> CoreResult<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let frequency = self.frequency.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "NeuralPanelForecaster must be fit before predict".to_string(),
            )
        })?;
        let tensor = self.predict_tensor(horizon)?;
        let mut predictions = Vec::new();
        for (series_id, rows) in tensor {
            let last_row = self.last_rows.get(&series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted timestamp tail for series '{series_id}'"
                ))
            })?;
            for (idx, quantiles) in rows.iter().enumerate() {
                let step = idx + 1;
                let median_idx = self
                    .config
                    .quantiles
                    .iter()
                    .position(|q| (*q - 0.5).abs() < f64::EPSILON)
                    .unwrap_or(0);
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp: frequency.advance(last_row.timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean: quantiles[median_idx],
                });
            }
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "neural_panel"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "config": self.config,
            "normalization": self.scaler.map(|scaler| json!({
                "center": scaler.mean(),
                "scale": scaler.scale(),
            })),
            "component_params": {
                "global_level": self.global_level,
                "global_slope": self.global_slope,
                "local_levels": self.local_levels,
                "local_slopes": self.local_slopes,
                "target_tails": self.target_tails,
                "ar_weights": self.ar_weights,
                "lagged_regressor_weights": self.covariate_weights,
                "nonstationary_feature_weights": self.feature_weights,
                "future_regressor_weights": self.future_regressor_weights,
            },
            "quantiles": self.config.quantiles,
            "series_id_map": self.series_ids,
            "changepoints": self.config.n_changepoints,
            "feature_schema": self.feature_schema,
            "lag_config": {
                "n_lags": self.config.n_lags,
                "lagged_regressors": self.config.lagged_regressors,
            },
            "seed": self.config.seed,
            "train_cutoff": self.train_cutoff,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneNeuralPanelConfig {
    pub base: NeuralPanelConfig,
    pub embedding_dim: usize,
}

impl Default for LaneNeuralPanelConfig {
    fn default() -> Self {
        Self {
            base: NeuralPanelConfig::default(),
            embedding_dim: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneNeuralPanelForecaster {
    inner: NeuralPanelForecaster,
    config: LaneNeuralPanelConfig,
    fallback_index: BTreeMap<String, Vec<String>>,
}

impl LaneNeuralPanelForecaster {
    pub fn new(mut config: LaneNeuralPanelConfig) -> Result<Self> {
        if config.embedding_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "embedding_dim must be positive".to_string(),
            ));
        }
        config.base.validate()?;
        Ok(Self {
            inner: NeuralPanelForecaster::new(config.base.clone())?,
            config,
            fallback_index: BTreeMap::new(),
        })
    }

    pub fn fallback_path(series_id: &str) -> Vec<String> {
        let (origin, destination) = split_lane(series_id);
        let mut path = Vec::new();
        if let (Some(origin), Some(destination)) = (origin, destination) {
            path.push(format!("pair:{origin}:{destination}"));
            path.push(format!("origin_parent:{origin}"));
            path.push(format!("destination_parent:{destination}"));
            path.push(format!("origin:{origin}"));
            path.push(format!("destination:{destination}"));
        }
        path.push("global".to_string());
        path
    }

    pub fn metadata(&self) -> Value {
        let mut metadata = self.inner.metadata();
        metadata["lane_config"] = json!({
            "embedding_dim": self.config.embedding_dim,
            "fallback_index": self.fallback_index,
            "static_covariates": ["origin_embedding", "destination_embedding", "lane_embedding"],
            "graph_features": ["directional_source_target_features"],
        });
        metadata
    }

    pub fn predict_quantiles_json_string(&self, horizon: usize) -> CoreResult<String> {
        self.inner.predict_quantiles_json_string(horizon)
    }

    pub fn predict_tensor_for_lanes(
        &self,
        horizon: usize,
        series_ids: &[String],
    ) -> CoreResult<BTreeMap<String, Vec<Vec<f64>>>> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        if series_ids.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "series_ids must contain at least one lane".to_string(),
            ));
        }
        let fitted_tensor = self.inner.predict_tensor(horizon)?;
        let mut expanded = BTreeMap::new();
        for series_id in series_ids {
            let candidates = self.fallback_candidates(series_id)?;
            let candidate_rows = candidates
                .iter()
                .map(|candidate| {
                    fitted_tensor.get(candidate).ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing fitted forecast tensor for fallback lane '{candidate}'"
                        ))
                    })
                })
                .collect::<CoreResult<Vec<_>>>()?;
            let mut rows = Vec::with_capacity(horizon);
            for step in 0..horizon {
                let quantile_count = candidate_rows
                    .first()
                    .and_then(|rows| rows.get(step))
                    .map_or(0, Vec::len);
                let mut averaged = Vec::with_capacity(quantile_count);
                for quantile_idx in 0..quantile_count {
                    let value = candidate_rows
                        .iter()
                        .map(|rows| rows[step][quantile_idx])
                        .sum::<f64>()
                        / candidate_rows.len() as f64;
                    averaged.push(value);
                }
                rows.push(averaged);
            }
            expanded.insert(series_id.clone(), rows);
        }
        Ok(expanded)
    }

    pub fn predict_for_lanes(
        &self,
        horizon: usize,
        series_ids: &[String],
    ) -> CoreResult<ForecastResult> {
        let frequency = self.inner.frequency.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "LaneNeuralPanelForecaster must be fit before predict_for_lanes".to_string(),
            )
        })?;
        let tensor = self.predict_tensor_for_lanes(horizon, series_ids)?;
        let mut predictions = Vec::new();
        for series_id in series_ids {
            let candidates = self.fallback_candidates(series_id)?;
            let last_timestamp = candidates
                .iter()
                .filter_map(|candidate| {
                    self.inner.last_rows.get(candidate).map(|row| row.timestamp)
                })
                .max()
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "no fitted timestamp tail available for fallback lane '{series_id}'"
                    ))
                })?;
            let rows = tensor.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing expanded forecast tensor for lane '{series_id}'"
                ))
            })?;
            for (idx, quantiles) in rows.iter().enumerate() {
                let step = idx + 1;
                let median_idx = self
                    .config
                    .base
                    .quantiles
                    .iter()
                    .position(|q| (*q - 0.5).abs() < f64::EPSILON)
                    .unwrap_or(0);
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp: frequency.advance(last_timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean: quantiles[median_idx],
                });
            }
        }
        ForecastResult::new(predictions)
    }

    pub fn predict_quantiles_for_lanes_json_string(
        &self,
        horizon: usize,
        series_ids: &[String],
    ) -> CoreResult<String> {
        let tensor = self.predict_tensor_for_lanes(horizon, series_ids)?;
        serde_json::to_string_pretty(&json!({
            "quantile_levels": self.config.base.quantiles,
            "series": tensor,
        }))
        .map_err(CartoBoostError::from)
    }

    fn fallback_candidates(&self, series_id: &str) -> CoreResult<Vec<String>> {
        let path = Self::fallback_path(series_id);
        for key in path {
            let candidates = self
                .fallback_index
                .iter()
                .filter_map(|(candidate_id, candidate_path)| {
                    if candidate_path
                        .iter()
                        .any(|candidate_key| candidate_key == &key)
                    {
                        Some(candidate_id.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if !candidates.is_empty() {
                return Ok(candidates);
            }
        }
        Err(CartoBoostError::InvalidInput(format!(
            "no fallback forecast available for lane '{series_id}'"
        )))
    }
}

impl Forecaster for LaneNeuralPanelForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> CoreResult<()> {
        self.fallback_index = frame
            .series_ids()
            .into_iter()
            .map(|series_id| {
                let path = Self::fallback_path(&series_id);
                (series_id, path)
            })
            .collect();
        self.inner.fit(frame)
    }

    fn predict(&self, horizon: usize) -> CoreResult<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        "lane_neural_panel"
    }

    fn metadata(&self) -> Value {
        LaneNeuralPanelForecaster::metadata(self)
    }
}

fn normalized_quantiles(quantiles: &[f64]) -> Result<Vec<f64>> {
    let mut set = BTreeSet::new();
    for quantile in quantiles {
        if !quantile.is_finite() || *quantile <= 0.0 || *quantile >= 1.0 {
            return Err(NeuralError::InvalidArgument(
                "quantiles must be finite values in (0, 1)".to_string(),
            ));
        }
        set.insert(format!("{quantile:.12}"));
    }
    set.insert(format!("{:.12}", 0.5));
    set.into_iter()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|err| NeuralError::InvalidArgument(err.to_string()))
        })
        .collect()
}

fn future_feature_names(config: &NeuralPanelConfig) -> Vec<String> {
    let mut names = Vec::new();
    for (name, order) in [
        ("daily", config.daily_fourier_order),
        ("weekly", config.weekly_fourier_order),
        ("yearly", config.yearly_fourier_order),
    ] {
        for harmonic in 1..=order {
            names.push(format!("seasonality:{name}:sin:{harmonic}"));
            names.push(format!("seasonality:{name}:cos:{harmonic}"));
        }
    }
    for (name, (_period, order)) in &config.custom_seasonalities {
        for harmonic in 1..=*order {
            names.push(format!("seasonality:{name}:sin:{harmonic}:{_period}"));
            names.push(format!("seasonality:{name}:cos:{harmonic}:{_period}"));
        }
    }
    for (name, offsets) in &config.events {
        for offset in offsets {
            names.push(format!("event:{name}:{offset}"));
        }
    }
    names.extend(config.future_regressors.keys().cloned());
    names
}

fn build_future_features(row: &ForecastRow, feature_names: &[String]) -> Result<Vec<f64>> {
    feature_names
        .iter()
        .map(|name| {
            if let Some(regressor) = row.covariates.get(name) {
                return Ok(*regressor);
            }
            if name.starts_with("seasonality:") {
                return Ok(fourier_feature(name, row.timestamp));
            }
            if name.starts_with("event:") {
                return Ok(*row.covariates.get(name).unwrap_or(&0.0));
            }
            required_covariate(row, name)
        })
        .collect()
}

fn required_covariate(row: &ForecastRow, name: &str) -> Result<f64> {
    row.covariates.get(name).copied().ok_or_else(|| {
        NeuralError::InvalidArgument(format!(
            "missing required covariate '{name}' for series {} at {}",
            row.series_id, row.timestamp
        ))
    })
}

fn feature_value_for_timestamp(
    name: &str,
    timestamp: NaiveDateTime,
    covariates: Option<&BTreeMap<String, f64>>,
) -> CoreResult<f64> {
    if name.starts_with("seasonality:") {
        return Ok(fourier_feature(name, timestamp));
    }
    if name.starts_with("event:") {
        return Ok(covariates
            .and_then(|values| values.get(name))
            .copied()
            .unwrap_or(0.0));
    }
    covariates
        .and_then(|values| values.get(name))
        .copied()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "future regressor '{name}' requires known future covariates for prediction"
            ))
        })
}

fn fourier_feature(name: &str, timestamp: NaiveDateTime) -> f64 {
    let parts = name.split(':').collect::<Vec<_>>();
    let (position, period) = match parts.get(1).copied().unwrap_or_default() {
        "daily" => (
            timestamp.hour() as f64
                + timestamp.minute() as f64 / 60.0
                + timestamp.second() as f64 / 3600.0,
            24.0,
        ),
        "weekly" => (
            timestamp.weekday().num_days_from_monday() as f64 * 24.0
                + timestamp.hour() as f64
                + timestamp.minute() as f64 / 60.0
                + timestamp.second() as f64 / 3600.0,
            24.0 * 7.0,
        ),
        "yearly" => (
            timestamp.ordinal0() as f64 * 24.0
                + timestamp.hour() as f64
                + timestamp.minute() as f64 / 60.0
                + timestamp.second() as f64 / 3600.0,
            365.25 * 24.0,
        ),
        _custom => {
            let period = parts
                .get(4)
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(1.0);
            (timestamp.and_utc().timestamp() as f64 / 3600.0, period)
        }
    };
    let harmonic = parts
        .get(3)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0);
    let angle: f64 = std::f64::consts::TAU * harmonic * position / period;
    if parts.get(2).copied() == Some("cos") {
        angle.cos()
    } else {
        angle.sin()
    }
}

fn feature_component_mode(config: &NeuralPanelConfig, name: &str) -> ComponentMode {
    if name.starts_with("seasonality:") {
        return config.seasonality_mode;
    }
    if name.starts_with("event:") {
        return config.event_mode;
    }
    config
        .future_regressors
        .get(name)
        .copied()
        .unwrap_or(ComponentMode::Additive)
}

fn fit_nonstationary_feature_weights(
    config: &NeuralPanelConfig,
    dataset: &NeuralPanelWindowDataset,
    scaler: &StandardScaler,
    baseline: impl Fn(&str, usize) -> f64,
) -> BTreeMap<String, f64> {
    let mut numerators = vec![0.0; dataset.future_feature_names().len()];
    let mut denominators = vec![0.0; dataset.future_feature_names().len()];
    for window in dataset.windows() {
        for horizon_idx in 0..config.n_forecasts {
            let offset = config.n_lags + horizon_idx;
            let Some(features) = window.future_features.get(offset) else {
                continue;
            };
            let target = scaler.transform(window.targets[horizon_idx]);
            let trend_baseline = baseline(&window.series_id, offset);
            let residual = target - trend_baseline;
            for (feature_idx, value) in features.iter().enumerate() {
                let feature_name = &dataset.future_feature_names()[feature_idx];
                let design = match feature_component_mode(config, feature_name) {
                    ComponentMode::Additive => *value,
                    ComponentMode::Multiplicative => trend_baseline * *value,
                };
                numerators[feature_idx] += design * residual;
                denominators[feature_idx] += design * design;
            }
        }
    }
    dataset
        .future_feature_names()
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            let denom = denominators[idx] + 1.0e-6;
            if denom <= 1.0e-6 {
                return None;
            }
            Some((name.clone(), numerators[idx] / denom))
        })
        .collect()
}

fn repaired_quantiles(median: f64, quantiles: &[f64]) -> Vec<f64> {
    let mut values = quantiles
        .iter()
        .map(|quantile| {
            if (*quantile - 0.5).abs() < f64::EPSILON {
                median
            } else if *quantile < 0.5 {
                median - (0.5 - *quantile) * median.abs().max(1.0) * 0.1
            } else {
                median + (*quantile - 0.5) * median.abs().max(1.0) * 0.1
            }
        })
        .collect::<Vec<_>>();
    for idx in 1..values.len() {
        if values[idx] < values[idx - 1] {
            values[idx] = values[idx - 1];
        }
    }
    values
}

fn global_slope(frame: &ForecastFrame, scaler: &StandardScaler) -> f64 {
    let slopes = frame
        .series_ids()
        .iter()
        .filter_map(|series_id| {
            let values = frame
                .rows_for_series(series_id)
                .iter()
                .map(|row| scaler.transform(row.target))
                .collect::<Vec<_>>();
            if values.len() > 1 {
                Some(slope_for_values(&values))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    mean(&slopes)
}

fn slope_for_values(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = mean(values);
    let numerator = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64 - x_mean) * (value - y_mean))
        .sum::<f64>();
    let denominator = (0..values.len())
        .map(|idx| {
            let centered = idx as f64 - x_mean;
            centered * centered
        })
        .sum::<f64>();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn deterministic_weights(count: usize, seed: u64, phase: f64) -> Vec<f64> {
    (0..count)
        .map(|idx| deterministic_scalar(seed, idx, phase) / count.max(1) as f64)
        .collect()
}

fn deterministic_scalar(seed: u64, idx: usize, phase: f64) -> f64 {
    (((seed as f64 + 1.0) * (idx as f64 + 1.0) * phase).sin()) * 0.01
}

fn split_lane(series_id: &str) -> (Option<&str>, Option<&str>) {
    if let Some((origin, destination)) = series_id.split_once(':') {
        return (Some(origin), Some(destination));
    }
    if let Some((origin, destination)) = series_id.split_once("->") {
        return (Some(origin), Some(destination));
    }
    (None, None)
}
