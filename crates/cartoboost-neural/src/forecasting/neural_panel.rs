use cartoboost_core::forecasting::{
    ForecastFrame, ForecastFrequency, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use cartoboost_core::{CartoBoostError, Result as CoreResult};
use chrono::{Datelike, NaiveDateTime, Timelike};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

use super::scaler::StandardScaler;
use crate::{backend_dense_layer_f32, BackendSelection, NeuralError, Result};

type KnownFutureCovariates = BTreeMap<(String, NaiveDateTime), BTreeMap<String, f64>>;
type KnownFutureCovariateIndex<'a> =
    BTreeMap<&'a str, BTreeMap<NaiveDateTime, &'a BTreeMap<String, f64>>>;
type FeatureComponentBreakdown = (f64, f64, BTreeMap<String, f64>, BTreeMap<String, f64>);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuralPanelLoss {
    SmoothL1,
    Mse,
    Mae,
    Pinball,
}

fn default_neural_panel_global_mode() -> NeuralPanelMode {
    NeuralPanelMode::Global
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
    pub custom_seasonality_conditions: BTreeMap<String, Option<String>>,
    pub seasonality_mode: ComponentMode,
    pub events: BTreeMap<String, Vec<i32>>,
    pub event_mode: ComponentMode,
    pub future_regressors: BTreeMap<String, ComponentMode>,
    pub lagged_regressors: BTreeMap<String, usize>,
    pub ar_layers: Vec<usize>,
    pub lagged_reg_layers: Vec<usize>,
    pub trend_mode: NeuralPanelMode,
    pub seasonality_global_local: NeuralPanelMode,
    #[serde(default = "default_neural_panel_global_mode")]
    pub event_global_local: NeuralPanelMode,
    #[serde(default = "default_neural_panel_global_mode")]
    pub regressor_global_local: NeuralPanelMode,
    pub local_l2: f64,
    pub seed: u64,
    pub loss: NeuralPanelLoss,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub newer_sample_weight: bool,
    #[serde(default)]
    pub backend: BackendSelection,
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
            custom_seasonality_conditions: BTreeMap::new(),
            seasonality_mode: ComponentMode::Additive,
            events: BTreeMap::new(),
            event_mode: ComponentMode::Additive,
            future_regressors: BTreeMap::new(),
            lagged_regressors: BTreeMap::new(),
            ar_layers: Vec::new(),
            lagged_reg_layers: Vec::new(),
            trend_mode: NeuralPanelMode::Global,
            seasonality_global_local: NeuralPanelMode::Global,
            event_global_local: NeuralPanelMode::Global,
            regressor_global_local: NeuralPanelMode::Global,
            local_l2: 0.0,
            seed: 0,
            loss: NeuralPanelLoss::SmoothL1,
            epochs: 80,
            learning_rate: 0.01,
            weight_decay: 0.0,
            newer_sample_weight: false,
            backend: BackendSelection::default(),
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
        if self.epochs == 0 {
            return Err(NeuralError::InvalidArgument(
                "epochs must be positive".to_string(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "learning_rate must be finite and positive".to_string(),
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(NeuralError::InvalidArgument(
                "weight_decay must be finite and non-negative".to_string(),
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
        for (name, condition_name) in &self.custom_seasonality_conditions {
            if name.is_empty() {
                return Err(NeuralError::InvalidArgument(
                    "custom seasonality condition names must not be empty".to_string(),
                ));
            }
            match condition_name {
                Some(condition_name) if condition_name.is_empty() => {
                    return Err(NeuralError::InvalidArgument(format!(
                        "custom seasonality '{name}' condition name must not be empty"
                    )));
                }
                _ => {}
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
pub(crate) enum SeasonalityBasis {
    Daily,
    Weekly,
    Yearly,
    Custom { period: f64 },
}

impl SeasonalityBasis {
    fn evaluate(&self, timestamp: NaiveDateTime) -> (f64, f64) {
        match self {
            Self::Daily => (
                timestamp.hour() as f64
                    + timestamp.minute() as f64 / 60.0
                    + timestamp.second() as f64 / 3600.0,
                24.0,
            ),
            Self::Weekly => (
                timestamp.weekday().num_days_from_monday() as f64 * 24.0
                    + timestamp.hour() as f64
                    + timestamp.minute() as f64 / 60.0
                    + timestamp.second() as f64 / 3600.0,
                24.0 * 7.0,
            ),
            Self::Yearly => (
                timestamp.ordinal0() as f64 * 24.0
                    + timestamp.hour() as f64
                    + timestamp.minute() as f64 / 60.0
                    + timestamp.second() as f64 / 3600.0,
                365.25 * 24.0,
            ),
            Self::Custom { period } => (timestamp.and_utc().timestamp() as f64 / 3600.0, *period),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) enum FutureFeatureSpec {
    Seasonality {
        name: String,
        basis: SeasonalityBasis,
        harmonic: usize,
        is_cosine: bool,
        component_mode: ComponentMode,
        global_local_mode: NeuralPanelMode,
        condition_name: Option<String>,
    },
    Event {
        name: String,
        component_mode: ComponentMode,
        global_local_mode: NeuralPanelMode,
    },
    Regressor {
        name: String,
        component_mode: ComponentMode,
        global_local_mode: NeuralPanelMode,
    },
}

impl FutureFeatureSpec {
    fn name(&self) -> &str {
        match self {
            Self::Seasonality { name, .. }
            | Self::Event { name, .. }
            | Self::Regressor { name, .. } => name,
        }
    }

    fn component_mode(&self) -> ComponentMode {
        match self {
            Self::Seasonality { component_mode, .. }
            | Self::Event { component_mode, .. }
            | Self::Regressor { component_mode, .. } => *component_mode,
        }
    }

    fn global_local_mode(&self) -> NeuralPanelMode {
        match self {
            Self::Seasonality {
                global_local_mode, ..
            }
            | Self::Event {
                global_local_mode, ..
            }
            | Self::Regressor {
                global_local_mode, ..
            } => *global_local_mode,
        }
    }

    fn value_for_row(&self, row: &ForecastRow, config: &NeuralPanelConfig) -> CoreResult<f64> {
        match self {
            Self::Seasonality {
                name,
                basis,
                harmonic,
                is_cosine,
                condition_name,
                ..
            } => {
                let (position, period) = basis.evaluate(row.timestamp);
                let angle = std::f64::consts::TAU * *harmonic as f64 * position / period;
                let value = if *is_cosine { angle.cos() } else { angle.sin() };
                apply_custom_seasonality_condition(
                    name,
                    value,
                    Some(&row.covariates),
                    None,
                    config,
                    condition_name.as_deref(),
                )
            }
            Self::Event { name, .. } => Ok(*row.covariates.get(name).unwrap_or(&0.0)),
            Self::Regressor { name, .. } => required_covariate(row, name)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string())),
        }
    }

    fn value_for_timestamp(
        &self,
        timestamp: NaiveDateTime,
        covariates: Option<&BTreeMap<String, f64>>,
        static_covariates: Option<&BTreeMap<String, f64>>,
        config: &NeuralPanelConfig,
    ) -> CoreResult<f64> {
        match self {
            Self::Seasonality {
                name,
                basis,
                harmonic,
                is_cosine,
                condition_name,
                ..
            } => {
                let (position, period) = basis.evaluate(timestamp);
                let angle = std::f64::consts::TAU * *harmonic as f64 * position / period;
                let value = if *is_cosine { angle.cos() } else { angle.sin() };
                apply_custom_seasonality_condition(
                    name,
                    value,
                    covariates,
                    static_covariates,
                    config,
                    condition_name.as_deref(),
                )
            }
            Self::Event { name, .. } => Ok(covariates
                .and_then(|values| values.get(name))
                .or_else(|| static_covariates.and_then(|values| values.get(name)))
                .copied()
                .unwrap_or(0.0)),
            Self::Regressor { name, .. } => covariates
                .and_then(|values| values.get(name))
                .or_else(|| static_covariates.and_then(|values| values.get(name)))
                .copied()
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "future regressor '{name}' requires known future covariates for prediction"
                    ))
                }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralPanelWindowDataset {
    windows: Vec<NeuralPanelWindow>,
    tails: BTreeMap<String, Vec<f64>>,
    future_feature_names: Vec<String>,
    future_feature_specs: Vec<FutureFeatureSpec>,
    series_ids: Vec<String>,
}

impl NeuralPanelWindowDataset {
    pub fn from_frame(frame: &ForecastFrame, config: &NeuralPanelConfig) -> Result<Self> {
        let mut config = config.clone();
        config.validate()?;
        let mut windows = Vec::new();
        let mut tails = BTreeMap::new();
        let future_feature_specs = future_feature_specs(&config);
        let future_feature_names = future_feature_specs
            .iter()
            .map(|spec| spec.name().to_string())
            .collect::<Vec<_>>();
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
                    .map(|row| build_future_features(row, &future_feature_specs, &config))
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
            future_feature_specs,
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

    pub(crate) fn future_feature_specs(&self) -> &[FutureFeatureSpec] {
        &self.future_feature_specs
    }

    pub fn series_ids(&self) -> &[String] {
        &self.series_ids
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpState {
    input_width: usize,
    output_width: usize,
    hidden_layers: Vec<usize>,
    layers: Vec<DenseLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPanelForecaster {
    config: NeuralPanelConfig,
    scaler: Option<StandardScaler>,
    frequency: Option<ForecastFrequency>,
    last_rows: BTreeMap<String, ForecastRow>,
    #[serde(default)]
    fitted_rows: BTreeMap<String, Vec<ForecastRow>>,
    series_ids: Vec<String>,
    global_level: f64,
    global_slope: f64,
    local_levels: BTreeMap<String, f64>,
    local_slopes: BTreeMap<String, f64>,
    #[serde(default)]
    series_lengths: BTreeMap<String, usize>,
    #[serde(default)]
    trend_changepoints: Vec<f64>,
    #[serde(default)]
    global_trend_coefficients: Vec<f64>,
    #[serde(default)]
    local_trend_coefficients: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    target_tails: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    lagged_covariate_tails: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
    ar_weights: Vec<f64>,
    covariate_weights: BTreeMap<String, f64>,
    #[serde(default)]
    feature_weights: BTreeMap<String, f64>,
    #[serde(default)]
    local_feature_weights: BTreeMap<String, BTreeMap<String, f64>>,
    #[serde(default)]
    ar_net: Option<MlpState>,
    #[serde(default)]
    covar_net: Option<MlpState>,
    #[serde(default)]
    quantile_output_order: Vec<f64>,
    #[serde(default)]
    quantile_residual_diffs: Vec<f64>,
    future_regressor_weights: BTreeMap<String, f64>,
    feature_schema: Vec<String>,
    #[serde(default)]
    future_feature_specs: Vec<FutureFeatureSpec>,
    #[serde(default)]
    feature_weight_values: Vec<f64>,
    #[serde(default)]
    local_feature_weight_values: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    static_future_covariates: BTreeMap<String, BTreeMap<String, f64>>,
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
            fitted_rows: BTreeMap::new(),
            series_ids: Vec::new(),
            global_level: 0.0,
            global_slope: 0.0,
            local_levels: BTreeMap::new(),
            local_slopes: BTreeMap::new(),
            series_lengths: BTreeMap::new(),
            trend_changepoints: Vec::new(),
            global_trend_coefficients: Vec::new(),
            local_trend_coefficients: BTreeMap::new(),
            target_tails: BTreeMap::new(),
            lagged_covariate_tails: BTreeMap::new(),
            ar_weights: Vec::new(),
            covariate_weights: BTreeMap::new(),
            feature_weights: BTreeMap::new(),
            local_feature_weights: BTreeMap::new(),
            ar_net: None,
            covar_net: None,
            quantile_output_order: Vec::new(),
            quantile_residual_diffs: Vec::new(),
            future_regressor_weights: BTreeMap::new(),
            feature_schema: Vec::new(),
            future_feature_specs: Vec::new(),
            feature_weight_values: Vec::new(),
            local_feature_weight_values: BTreeMap::new(),
            static_future_covariates: BTreeMap::new(),
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
        self.predict_tensor_with_covariates(horizon, None)
    }

    pub fn predict_tensor_with_known_future_covariates(
        &self,
        horizon: usize,
        known_future_covariates: &KnownFutureCovariates,
    ) -> CoreResult<BTreeMap<String, Vec<Vec<f64>>>> {
        self.predict_tensor_with_covariates(horizon, Some(known_future_covariates))
    }

    fn predict_tensor_with_covariates(
        &self,
        horizon: usize,
        known_future_covariates: Option<&KnownFutureCovariates>,
    ) -> CoreResult<BTreeMap<String, Vec<Vec<f64>>>> {
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
        let known_future_index = known_future_covariates.map(index_known_future_covariates);
        let mut by_series = BTreeMap::new();
        for series_id in &self.series_ids {
            let last_row = self.last_rows.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted timestamp tail for series '{series_id}'"
                ))
            })?;
            let stationary_effects = self.stationary_network_effects(series_id)?;
            let mut rows = Vec::with_capacity(horizon);
            for step in 1..=horizon {
                let timestamp = frequency.advance(last_row.timestamp, step)?;
                let covariates = known_future_index
                    .as_ref()
                    .and_then(|values| values.get(series_id.as_str()))
                    .and_then(|values| values.get(&timestamp).copied());
                let (additive, multiplicative) =
                    self.nonstationary_effect(series_id, timestamp, covariates)?;
                let median_scaled = self.trend_baseline(series_id, step);
                let median_scaled = median_scaled
                    + additive
                    + median_scaled * multiplicative
                    + stationary_effects
                        .get(step - 1)
                        .copied()
                        .or_else(|| stationary_effects.last().copied())
                        .unwrap_or(0.0);
                let median = scaler.inverse_transform(median_scaled);
                rows.push(repaired_quantiles(
                    median,
                    &self.config.quantiles,
                    &self.quantile_output_order,
                    &self.quantile_residual_diffs,
                    scaler.scale(),
                ));
            }
            by_series.insert(series_id.clone(), rows);
        }
        Ok(by_series)
    }

    pub fn predict_with_known_future_covariates(
        &self,
        horizon: usize,
        known_future_covariates: &KnownFutureCovariates,
    ) -> CoreResult<ForecastResult> {
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
        let tensor =
            self.predict_tensor_with_known_future_covariates(horizon, known_future_covariates)?;
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

    fn stationary_network_effects(&self, series_id: &str) -> CoreResult<Vec<f64>> {
        let mut output = vec![0.0; self.config.n_forecasts];
        if let (Some(net), Some(tail)) = (&self.ar_net, self.target_tails.get(series_id)) {
            let start_index = tail.len().saturating_sub(self.config.n_lags);
            let input = tail[start_index..]
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    let baseline = self.trend_baseline(series_id, idx + start_index + 1);
                    value - baseline
                })
                .collect::<Vec<_>>();
            let forward = self.forward_net(net, &input)?;
            add_median_outputs(&mut output, &forward, &self.config.quantiles);
        }
        if let (Some(net), Some(tails)) =
            (&self.covar_net, self.lagged_covariate_tails.get(series_id))
        {
            let input = lagged_covariate_input(&self.config, tails);
            let forward = self.forward_net(net, &input)?;
            add_median_outputs(&mut output, &forward, &self.config.quantiles);
        }
        Ok(output)
    }

    fn forward_net(&self, net: &MlpState, input: &[f64]) -> CoreResult<Vec<f64>> {
        net.forward(input, &self.config.backend)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
    }

    pub fn predict_quantiles_json_string(&self, horizon: usize) -> CoreResult<String> {
        let tensor = self.predict_tensor(horizon)?;
        serde_json::to_string_pretty(&json!({
            "quantile_levels": self.config.quantiles,
            "series": tensor,
        }))
        .map_err(CartoBoostError::from)
    }

    pub fn predict_components_json_value(&self, horizon: usize) -> CoreResult<Value> {
        self.predict_components_json_value_with_known_future_covariates(horizon, None)
    }

    pub fn predict_components_json_string(&self, horizon: usize) -> CoreResult<String> {
        serde_json::to_string_pretty(&self.predict_components_json_value(horizon)?)
            .map_err(CartoBoostError::from)
    }

    pub fn history_components_json_value(&self) -> CoreResult<Value> {
        if self.frequency.is_none() {
            return Err(CartoBoostError::InvalidInput(
                "NeuralPanelForecaster must be fit before predict".to_string(),
            ));
        }
        let scaler = self.scaler.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "NeuralPanelForecaster must be fit before predict".to_string(),
            )
        })?;
        let quantile_index = self
            .config
            .quantiles
            .iter()
            .position(|q| (*q - 0.5).abs() < f64::EPSILON)
            .unwrap_or(0);
        let mut series = BTreeMap::new();
        for series_id in &self.series_ids {
            let rows = self.fitted_rows.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted rows for series '{series_id}'"
                ))
            })?;
            let mut records = Vec::new();
            for (idx, row) in rows.iter().enumerate() {
                let trend = self.trend_baseline(series_id, idx);
                let static_covariates = self.static_future_covariates.get(series_id);
                let (additive, multiplicative, additive_components, multiplicative_components) =
                    self.feature_effects_for_timestamp(
                        series_id,
                        row.timestamp,
                        Some(&row.covariates),
                        static_covariates,
                        trend,
                        scaler.scale(),
                    )?;
                let (ar_component, lagged_regressor_component) = if idx >= self.config.n_lags {
                    let lag_start = idx - self.config.n_lags;
                    let ar_component = if let Some(net) = &self.ar_net {
                        let input = rows[lag_start..idx]
                            .iter()
                            .enumerate()
                            .map(|(lag_idx, lag_row)| {
                                let baseline = self.trend_baseline(series_id, lag_start + lag_idx);
                                scaler.transform(lag_row.target) - baseline
                            })
                            .collect::<Vec<_>>();
                        self.forward_net(net, &input)?
                            .get(quantile_index)
                            .copied()
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let lagged_regressor_component = if let Some(net) = &self.covar_net {
                        let mut input = Vec::new();
                        for (name, lag) in &self.config.lagged_regressors {
                            let start = idx.saturating_sub(*lag);
                            input.extend(rows[start..idx].iter().map(|history_row| {
                                history_row.covariates.get(name).copied().unwrap_or(0.0)
                            }));
                        }
                        self.forward_net(net, &input)?
                            .get(quantile_index)
                            .copied()
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    (ar_component, lagged_regressor_component)
                } else {
                    (0.0, 0.0)
                };
                let median_scaled = trend
                    + additive
                    + trend * multiplicative
                    + ar_component
                    + lagged_regressor_component;
                let median = scaler.inverse_transform(median_scaled);
                let quantiles = repaired_quantiles(
                    median,
                    &self.config.quantiles,
                    &self.quantile_output_order,
                    &self.quantile_residual_diffs,
                    scaler.scale(),
                );
                records.push(json!({
                    "series_id": series_id,
                    "timestamp": row.timestamp,
                    "index": idx,
                    "actual": row.target,
                    "fitted": median,
                    "residual": row.target - median,
                    "trend": scaler.inverse_transform(trend),
                    "feature_contributions": {
                        "additive": additive_components,
                        "multiplicative": multiplicative_components,
                    },
                    "additive_total": additive * scaler.scale(),
                    "multiplicative_total": trend * multiplicative * scaler.scale(),
                    "ar_component": ar_component * scaler.scale(),
                    "lagged_regressor_component": lagged_regressor_component * scaler.scale(),
                    "prediction": median,
                    "quantiles": quantiles,
                }));
            }
            series.insert(series_id.clone(), records);
        }
        Ok(json!({
            "quantile_levels": self.config.quantiles,
            "series": series,
        }))
    }

    pub fn history_components_json_string(&self) -> CoreResult<String> {
        serde_json::to_string_pretty(&self.history_components_json_value()?)
            .map_err(CartoBoostError::from)
    }

    pub fn predict_components_json_value_with_known_future_covariates(
        &self,
        horizon: usize,
        known_future_covariates: Option<&KnownFutureCovariates>,
    ) -> CoreResult<Value> {
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
        let scaler = self.scaler.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "NeuralPanelForecaster must be fit before predict".to_string(),
            )
        })?;
        let mut series = BTreeMap::new();
        let quantile_index = self
            .config
            .quantiles
            .iter()
            .position(|q| (*q - 0.5).abs() < f64::EPSILON)
            .unwrap_or(0);
        let quantile_count = self.config.quantiles.len();
        let output_width = self.config.n_forecasts * quantile_count;
        let known_future_index = known_future_covariates.map(index_known_future_covariates);
        for series_id in &self.series_ids {
            let last_row = self.last_rows.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted timestamp tail for series '{series_id}'"
                ))
            })?;
            let mut rows = Vec::with_capacity(horizon);
            let ar_output = if let Some(net) = &self.ar_net {
                let tail = self.target_tails.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing fitted target tail for series '{series_id}'"
                    ))
                })?;
                let start_index = tail.len().saturating_sub(self.config.n_lags);
                let input = tail[start_index..]
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| {
                        let baseline = self.trend_baseline(series_id, idx + start_index + 1);
                        value - baseline
                    })
                    .collect::<Vec<_>>();
                self.forward_net(net, &input)?
            } else {
                vec![0.0; output_width]
            };
            let covar_output = if let Some(net) = &self.covar_net {
                let tails = self.lagged_covariate_tails.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing fitted lagged covariate tail for series '{series_id}'"
                    ))
                })?;
                self.forward_net(net, &lagged_covariate_input(&self.config, tails))?
            } else {
                vec![0.0; output_width]
            };
            for step in 1..=horizon {
                let timestamp = frequency.advance(last_row.timestamp, step)?;
                let static_covariates = self.static_future_covariates.get(series_id);
                let covariates = known_future_index
                    .as_ref()
                    .and_then(|values| values.get(series_id.as_str()))
                    .and_then(|values| values.get(&timestamp).copied());
                let trend = self.trend_baseline(series_id, step);
                let (additive, multiplicative, additive_components, multiplicative_components) =
                    self.feature_effects_for_timestamp(
                        series_id,
                        timestamp,
                        covariates,
                        static_covariates,
                        trend,
                        scaler.scale(),
                    )?;
                let output_idx = (step - 1) * quantile_count + quantile_index;
                let ar_component = ar_output.get(output_idx).copied().unwrap_or(0.0);
                let lagged_regressor_component =
                    covar_output.get(output_idx).copied().unwrap_or(0.0);
                let median_scaled = trend
                    + additive
                    + trend * multiplicative
                    + ar_component
                    + lagged_regressor_component;
                let median = scaler.inverse_transform(median_scaled);
                let quantiles = repaired_quantiles(
                    median,
                    &self.config.quantiles,
                    &self.quantile_output_order,
                    &self.quantile_residual_diffs,
                    scaler.scale(),
                );
                rows.push(json!({
                    "timestamp": timestamp,
                    "horizon": step,
                    "trend": scaler.inverse_transform(trend),
                    "feature_contributions": {
                        "additive": additive_components,
                        "multiplicative": multiplicative_components,
                    },
                    "additive_total": additive * scaler.scale(),
                    "multiplicative_total": trend * multiplicative * scaler.scale(),
                    "ar_component": ar_component * scaler.scale(),
                    "lagged_regressor_component": lagged_regressor_component * scaler.scale(),
                    "median_scaled": median_scaled,
                    "prediction": median,
                    "quantiles": quantiles,
                }));
            }
            series.insert(series_id.clone(), rows);
        }
        Ok(json!({
            "quantile_levels": self.config.quantiles,
            "series": series,
        }))
    }

    fn nonstationary_effect(
        &self,
        series_id: &str,
        timestamp: NaiveDateTime,
        covariates: Option<&BTreeMap<String, f64>>,
    ) -> CoreResult<(f64, f64)> {
        self.feature_effect_totals_for_timestamp(series_id, timestamp, covariates)
    }

    fn feature_effects_for_timestamp(
        &self,
        series_id: &str,
        timestamp: NaiveDateTime,
        covariates: Option<&BTreeMap<String, f64>>,
        static_covariates: Option<&BTreeMap<String, f64>>,
        trend: f64,
        component_scale: f64,
    ) -> CoreResult<FeatureComponentBreakdown> {
        let mut additive = 0.0;
        let mut multiplicative = 0.0;
        let mut additive_components = BTreeMap::new();
        let mut multiplicative_components = BTreeMap::new();
        let local_weights = self.local_feature_weight_values.get(series_id);
        for (idx, spec) in self.future_feature_specs.iter().enumerate() {
            let value = feature_value_for_timestamp(
                spec,
                timestamp,
                covariates,
                static_covariates,
                &self.config,
            )?;
            let global_weight = self
                .feature_weight_values
                .get(idx)
                .copied()
                .or_else(|| self.feature_weights.get(spec.name()).copied())
                .unwrap_or(0.0);
            let local_weight = local_weights
                .and_then(|weights| weights.get(idx))
                .copied()
                .or_else(|| {
                    self.local_feature_weights
                        .get(series_id)
                        .and_then(|weights| weights.get(spec.name()))
                        .copied()
                })
                .unwrap_or(0.0);
            let weight = match spec.global_local_mode() {
                NeuralPanelMode::Global => global_weight,
                NeuralPanelMode::Local => local_weight,
                NeuralPanelMode::Glocal => global_weight + local_weight,
            };
            match spec.component_mode() {
                ComponentMode::Additive => {
                    let contribution = value * weight;
                    additive += contribution;
                    additive_components
                        .insert(spec.name().to_string(), contribution * component_scale);
                }
                ComponentMode::Multiplicative => {
                    let contribution = trend * value * weight;
                    multiplicative += contribution;
                    multiplicative_components
                        .insert(spec.name().to_string(), contribution * component_scale);
                }
            };
        }
        Ok((
            additive,
            multiplicative,
            additive_components,
            multiplicative_components,
        ))
    }

    fn feature_effect_totals_for_timestamp(
        &self,
        series_id: &str,
        timestamp: NaiveDateTime,
        covariates: Option<&BTreeMap<String, f64>>,
    ) -> CoreResult<(f64, f64)> {
        let static_covariates = self.static_future_covariates.get(series_id);
        let local_weights = self.local_feature_weight_values.get(series_id);
        let mut additive = 0.0;
        let mut multiplicative = 0.0;
        for (idx, spec) in self.future_feature_specs.iter().enumerate() {
            let value =
                spec.value_for_timestamp(timestamp, covariates, static_covariates, &self.config)?;
            let global_weight = self
                .feature_weight_values
                .get(idx)
                .copied()
                .or_else(|| self.feature_weights.get(spec.name()).copied())
                .unwrap_or(0.0);
            let local_weight = local_weights
                .and_then(|weights| weights.get(idx))
                .copied()
                .or_else(|| {
                    self.local_feature_weights
                        .get(series_id)
                        .and_then(|weights| weights.get(spec.name()))
                        .copied()
                })
                .unwrap_or(0.0);
            let weight = match spec.global_local_mode() {
                NeuralPanelMode::Global => global_weight,
                NeuralPanelMode::Local => local_weight,
                NeuralPanelMode::Glocal => global_weight + local_weight,
            };
            match spec.component_mode() {
                ComponentMode::Additive => additive += value * weight,
                ComponentMode::Multiplicative => multiplicative += value * weight,
            }
        }
        Ok((additive, multiplicative))
    }

    fn fit_quantile_residual_diffs(
        &self,
        dataset: &NeuralPanelWindowDataset,
        scaler: &StandardScaler,
    ) -> Vec<f64> {
        let mut residuals = Vec::new();
        let output_width = self.config.n_forecasts * self.config.quantiles.len();
        for window in dataset.windows() {
            let ar_output = self
                .ar_net
                .as_ref()
                .map(|net| {
                    let input = window
                        .lags
                        .iter()
                        .enumerate()
                        .map(|(idx, value)| {
                            let baseline = self.trend_baseline(&window.series_id, idx);
                            scaler.transform(*value) - baseline
                        })
                        .collect::<Vec<_>>();
                    net.forward_cpu(&input)
                })
                .unwrap_or_else(|| vec![0.0; output_width]);
            let covar_output = self
                .covar_net
                .as_ref()
                .map(|net| {
                    net.forward_cpu(&lagged_covariate_input(
                        &self.config,
                        &window.lagged_covariates,
                    ))
                })
                .unwrap_or_else(|| vec![0.0; output_width]);
            for horizon_idx in 0..self.config.n_forecasts {
                let offset = window.lags.len() + horizon_idx;
                let mut median_scaled = self.trend_baseline(&window.series_id, offset);
                let (additive, multiplicative) = self.fitted_feature_effects(
                    &window.series_id,
                    median_scaled,
                    window.future_features.get(offset).map(Vec::as_slice),
                );
                let output_idx = horizon_idx * self.config.quantiles.len();
                median_scaled += additive
                    + median_scaled * multiplicative
                    + ar_output.get(output_idx).copied().unwrap_or(0.0)
                    + covar_output.get(output_idx).copied().unwrap_or(0.0);
                residuals.push(scaler.transform(window.targets[horizon_idx]) - median_scaled);
            }
        }
        learned_quantile_diffs(&quantile_output_order(&self.config.quantiles), residuals)
    }

    fn trend_baseline(&self, series_id: &str, offset: usize) -> f64 {
        if self.config.trend == TrendMode::Off {
            let mut value = self
                .global_trend_coefficients
                .first()
                .copied()
                .unwrap_or(self.global_level);
            if self.config.trend_mode != NeuralPanelMode::Global {
                value += self
                    .local_trend_coefficients
                    .get(series_id)
                    .and_then(|coefficients| coefficients.first().copied())
                    .unwrap_or(0.0);
            }
            return value;
        }
        let coeffs = self
            .local_trend_coefficients
            .get(series_id)
            .filter(|coefficients| !coefficients.is_empty())
            .map(|coefficients| {
                let mut combined = self.global_trend_coefficients.clone();
                if combined.len() < coefficients.len() {
                    combined.resize(coefficients.len(), 0.0);
                }
                for (idx, value) in coefficients.iter().enumerate() {
                    combined[idx] += *value;
                }
                combined
            })
            .unwrap_or_else(|| self.global_trend_coefficients.clone());
        let position = self.series_position(series_id, offset);
        let basis = trend_basis(position, &self.trend_changepoints);
        dot(&coeffs, &basis)
    }

    fn series_position(&self, series_id: &str, offset: usize) -> f64 {
        let series_length = self
            .series_lengths
            .get(series_id)
            .copied()
            .unwrap_or(1)
            .max(1);
        series_position_from_length(offset, series_length)
    }

    fn fitted_feature_effects(
        &self,
        series_id: &str,
        _baseline: f64,
        features: Option<&[f64]>,
    ) -> (f64, f64) {
        let mut additive = 0.0;
        let mut multiplicative = 0.0;
        let Some(features) = features else {
            return (additive, multiplicative);
        };
        let local_weights = self.local_feature_weight_values.get(series_id);
        for (idx, (spec, value)) in self
            .future_feature_specs
            .iter()
            .zip(features.iter())
            .enumerate()
        {
            let global_weight = self
                .feature_weight_values
                .get(idx)
                .copied()
                .or_else(|| self.feature_weights.get(spec.name()).copied())
                .unwrap_or(0.0);
            let local_weight = local_weights
                .and_then(|weights| weights.get(idx))
                .copied()
                .or_else(|| {
                    self.local_feature_weights
                        .get(series_id)
                        .and_then(|weights| weights.get(spec.name()))
                        .copied()
                })
                .unwrap_or(0.0);
            let weight = match spec.global_local_mode() {
                NeuralPanelMode::Global => global_weight,
                NeuralPanelMode::Local => local_weight,
                NeuralPanelMode::Glocal => global_weight + local_weight,
            };
            match spec.component_mode() {
                ComponentMode::Additive => additive += value * weight,
                ComponentMode::Multiplicative => multiplicative += value * weight,
            }
        }
        (additive, multiplicative)
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
        self.series_lengths = self
            .series_ids
            .iter()
            .map(|series_id| {
                (
                    series_id.clone(),
                    frame.rows_for_series(series_id).len().max(1),
                )
            })
            .collect();
        self.future_feature_specs = dataset.future_feature_specs().to_vec();
        self.feature_schema = self
            .future_feature_specs
            .iter()
            .map(|spec| spec.name().to_string())
            .collect();
        self.frequency = Some(frame.frequency());
        self.scaler = Some(scaler);
        self.static_future_covariates = collect_static_future_covariates(frame, &self.config);
        self.trend_changepoints =
            trend_changepoints(self.config.n_changepoints, self.config.changepoints_range);
        let (global_trend_coefficients, local_trend_coefficients) = fit_piecewise_trend(
            frame,
            &scaler,
            &self.series_lengths,
            &self.trend_changepoints,
            &self.config,
        );
        self.global_trend_coefficients = global_trend_coefficients;
        self.local_trend_coefficients = local_trend_coefficients;
        self.global_level = self
            .global_trend_coefficients
            .first()
            .copied()
            .unwrap_or(self.global_level);
        self.global_slope = self
            .global_trend_coefficients
            .get(1)
            .copied()
            .unwrap_or(self.global_slope);
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
        self.lagged_covariate_tails = self
            .series_ids
            .iter()
            .map(|series_id| {
                let rows = frame.rows_for_series(series_id);
                let covariates = self
                    .config
                    .lagged_regressors
                    .iter()
                    .map(|(name, lag)| {
                        let start = rows.len().saturating_sub(*lag);
                        let values = rows[start..]
                            .iter()
                            .map(|row| row.covariates.get(name).copied().unwrap_or(0.0))
                            .collect::<Vec<_>>();
                        (name.clone(), values)
                    })
                    .collect::<BTreeMap<_, _>>();
                (series_id.clone(), covariates)
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
        self.fitted_rows = self
            .series_ids
            .iter()
            .map(|series_id| {
                (
                    series_id.clone(),
                    frame
                        .rows_for_series(series_id)
                        .into_iter()
                        .cloned()
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        self.local_levels.clear();
        self.local_slopes.clear();
        if self.config.trend_mode != NeuralPanelMode::Global {
            for series_id in &self.series_ids {
                let coefficients = self
                    .local_trend_coefficients
                    .get(series_id)
                    .cloned()
                    .unwrap_or_default();
                self.local_levels.insert(
                    series_id.clone(),
                    coefficients.first().copied().unwrap_or(0.0),
                );
                self.local_slopes.insert(
                    series_id.clone(),
                    coefficients.get(1).copied().unwrap_or(0.0),
                );
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
        self.feature_weights =
            fit_nonstationary_feature_weights(&dataset, &scaler, |series_id, offset| {
                self.trend_baseline(series_id, offset)
            });
        self.local_feature_weights = fit_local_feature_weights(
            &self.config,
            &dataset,
            &scaler,
            &self.feature_weights,
            |series_id, offset| self.trend_baseline(series_id, offset),
        );
        self.feature_weight_values = self
            .future_feature_specs
            .iter()
            .map(|spec| {
                self.feature_weights
                    .get(spec.name())
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        self.local_feature_weight_values = self
            .local_feature_weights
            .iter()
            .map(|(series_id, weights)| {
                (
                    series_id.clone(),
                    self.future_feature_specs
                        .iter()
                        .map(|spec| weights.get(spec.name()).copied().unwrap_or(0.0))
                        .collect(),
                )
            })
            .collect();
        let output_width = self.config.n_forecasts * self.config.quantiles.len();
        self.ar_net = if self.config.n_lags > 0 {
            Some(train_mlp(
                self.config.n_lags,
                output_width,
                &self.config.ar_layers,
                ar_training_examples(&self.config, &dataset, &scaler, |series_id, offset| {
                    self.trend_baseline(series_id, offset)
                }),
                &self.config,
                self.config.seed ^ 0xA71,
            ))
        } else {
            None
        };
        let covar_input_width = self.config.lagged_regressors.values().sum::<usize>();
        self.covar_net = if covar_input_width > 0 {
            Some(train_mlp(
                covar_input_width,
                output_width,
                &self.config.lagged_reg_layers,
                covar_training_examples(
                    &self.config,
                    &dataset,
                    &scaler,
                    &self.ar_net,
                    |series_id, offset| self.trend_baseline(series_id, offset),
                ),
                &self.config,
                self.config.seed ^ 0xC09A,
            ))
        } else {
            None
        };
        self.quantile_output_order = quantile_output_order(&self.config.quantiles);
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
        self.quantile_residual_diffs = self.fit_quantile_residual_diffs(&dataset, &scaler);
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
                "series_lengths": self.series_lengths,
                "trend_changepoints": self.trend_changepoints,
                "global_trend_coefficients": self.global_trend_coefficients,
                "local_trend_coefficients": self.local_trend_coefficients,
                "target_tails": self.target_tails,
                "ar_weights": self.ar_weights,
                "lagged_regressor_weights": self.covariate_weights,
                "nonstationary_feature_weights": self.feature_weights,
                "local_nonstationary_feature_weights": self.local_feature_weights,
                "ar_net": self.ar_net,
                "covar_net": self.covar_net,
                "quantile_output_order": self.quantile_output_order,
                "quantile_residual_diffs": self.quantile_residual_diffs,
                "future_regressor_weights": self.future_regressor_weights,
            },
            "quantiles": self.config.quantiles,
            "series_id_map": self.series_ids,
            "changepoints": self.config.n_changepoints,
            "feature_schema": self.feature_schema,
            "static_future_covariates": self.static_future_covariates,
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
    #[serde(default)]
    origin_embeddings: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    destination_embeddings: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    lane_embeddings: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    lane_biases: BTreeMap<String, f64>,
    #[serde(default)]
    graph_directional_features: BTreeMap<String, BTreeMap<String, f64>>,
    #[serde(default)]
    generated_lane_feature_names: Vec<String>,
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
            origin_embeddings: BTreeMap::new(),
            destination_embeddings: BTreeMap::new(),
            lane_embeddings: BTreeMap::new(),
            lane_biases: BTreeMap::new(),
            graph_directional_features: BTreeMap::new(),
            generated_lane_feature_names: Vec::new(),
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
            "origin_embeddings": self.origin_embeddings,
            "destination_embeddings": self.destination_embeddings,
            "lane_embeddings": self.lane_embeddings,
            "lane_biases": self.lane_biases,
            "graph_directional_features": self.graph_directional_features,
            "generated_lane_feature_names": self.generated_lane_feature_names,
            "static_covariates": self.generated_lane_feature_names,
            "graph_features": ["origin_hash", "destination_hash", "direction_hash_delta", "observed_lane_mean"],
        });
        metadata
    }

    pub fn predict_quantiles_json_string(&self, horizon: usize) -> CoreResult<String> {
        self.inner.predict_quantiles_json_string(horizon)
    }

    pub fn predict_components_json_value(&self, horizon: usize) -> CoreResult<Value> {
        self.inner.predict_components_json_value(horizon)
    }

    pub fn predict_components_json_string(&self, horizon: usize) -> CoreResult<String> {
        self.inner.predict_components_json_string(horizon)
    }

    pub fn history_components_json_value(&self) -> CoreResult<Value> {
        self.inner.history_components_json_value()
    }

    pub fn history_components_json_string(&self) -> CoreResult<String> {
        self.inner.history_components_json_string()
    }

    pub fn predict_components_json_value_with_known_future_covariates(
        &self,
        horizon: usize,
        known_future_covariates: &KnownFutureCovariates,
    ) -> CoreResult<Value> {
        self.inner
            .predict_components_json_value_with_known_future_covariates(
                horizon,
                Some(known_future_covariates),
            )
    }

    pub fn predict_with_known_future_covariates(
        &self,
        horizon: usize,
        known_future_covariates: &KnownFutureCovariates,
    ) -> CoreResult<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let frequency = self.inner.frequency.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "LaneNeuralPanelForecaster must be fit before predict".to_string(),
            )
        })?;
        let tensor = self
            .inner
            .predict_tensor_with_known_future_covariates(horizon, known_future_covariates)?;
        let mut predictions = Vec::new();
        for (series_id, rows) in tensor {
            let last_row = self.inner.last_rows.get(&series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted timestamp tail for series '{series_id}'"
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
                    timestamp: frequency.advance(last_row.timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean: quantiles[median_idx],
                });
            }
        }
        ForecastResult::new(predictions)
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
                    averaged.push(value + self.lane_bias(series_id));
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

    fn fit_lane_state(&mut self, frame: &ForecastFrame) {
        self.origin_embeddings.clear();
        self.destination_embeddings.clear();
        self.lane_embeddings.clear();
        self.lane_biases.clear();
        self.graph_directional_features.clear();
        self.generated_lane_feature_names = generated_lane_feature_names(self.config.embedding_dim);
        let global_mean = mean(
            &frame
                .rows()
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>(),
        );
        for series_id in frame.series_ids() {
            let rows = frame.rows_for_series(&series_id);
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let lane_mean = mean(&values);
            let lane_bias = (lane_mean - global_mean) / (1.0 + self.config.base.local_l2);
            self.lane_biases.insert(series_id.clone(), lane_bias * 0.05);
            let (origin, destination) = split_lane(&series_id);
            if let Some(origin) = origin {
                self.origin_embeddings
                    .entry(origin.to_string())
                    .or_insert_with(|| {
                        deterministic_embedding(
                            origin,
                            self.config.embedding_dim,
                            self.config.base.seed,
                        )
                    });
            }
            if let Some(destination) = destination {
                self.destination_embeddings
                    .entry(destination.to_string())
                    .or_insert_with(|| {
                        deterministic_embedding(
                            destination,
                            self.config.embedding_dim,
                            self.config.base.seed ^ 0xD057,
                        )
                    });
            }
            self.lane_embeddings
                .entry(series_id.clone())
                .or_insert_with(|| {
                    deterministic_embedding(
                        &series_id,
                        self.config.embedding_dim,
                        self.config.base.seed ^ 0x1A9E,
                    )
                });
            if let (Some(origin), Some(destination)) = (origin, destination) {
                self.graph_directional_features.insert(
                    series_id.clone(),
                    BTreeMap::from([
                        ("origin_hash".to_string(), stable_unit_hash(origin)),
                        (
                            "destination_hash".to_string(),
                            stable_unit_hash(destination),
                        ),
                        (
                            "direction_hash_delta".to_string(),
                            stable_unit_hash(origin) - stable_unit_hash(destination),
                        ),
                        ("observed_lane_mean".to_string(), lane_mean),
                    ]),
                );
            }
        }
    }

    fn augmented_lane_training_frame(&self, frame: &ForecastFrame) -> CoreResult<ForecastFrame> {
        let mut rows = Vec::with_capacity(frame.rows().len());
        for row in frame.rows() {
            let mut augmented = row.clone();
            for (name, value) in self.lane_feature_values(&row.series_id)? {
                augmented.covariates.insert(name, value);
            }
            rows.push(augmented);
        }
        ForecastFrame::with_metadata(rows, frame.frequency(), frame.metadata().clone())
    }

    fn augmented_lane_config(&self) -> NeuralPanelConfig {
        let mut config = self.config.base.clone();
        for name in &self.generated_lane_feature_names {
            config
                .future_regressors
                .entry(name.clone())
                .or_insert(ComponentMode::Additive);
        }
        config
    }

    fn lane_feature_values(&self, series_id: &str) -> CoreResult<BTreeMap<String, f64>> {
        let (origin, destination) = split_lane(series_id);
        let (Some(origin), Some(destination)) = (origin, destination) else {
            return Err(CartoBoostError::InvalidInput(format!(
                "lane neural panel expects series_id='origin:destination', got '{series_id}'"
            )));
        };
        let origin_embedding = self.origin_embeddings.get(origin).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!("missing origin embedding for '{origin}'"))
        })?;
        let destination_embedding =
            self.destination_embeddings
                .get(destination)
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing destination embedding for '{destination}'"
                    ))
                })?;
        let lane_embedding = self.lane_embeddings.get(series_id).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!("missing lane embedding for '{series_id}'"))
        })?;
        let graph = self
            .graph_directional_features
            .get(series_id)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing graph directional features for '{series_id}'"
                ))
            })?;
        let mut values = BTreeMap::new();
        for idx in 0..self.config.embedding_dim {
            values.insert(
                format!("lane_origin_embedding_{idx}"),
                origin_embedding.get(idx).copied().unwrap_or(0.0),
            );
            values.insert(
                format!("lane_destination_embedding_{idx}"),
                destination_embedding.get(idx).copied().unwrap_or(0.0),
            );
            values.insert(
                format!("lane_embedding_{idx}"),
                lane_embedding.get(idx).copied().unwrap_or(0.0),
            );
        }
        for name in [
            "origin_hash",
            "destination_hash",
            "direction_hash_delta",
            "observed_lane_mean",
        ] {
            values.insert(
                format!("lane_graph_{name}"),
                graph.get(name).copied().unwrap_or(0.0),
            );
        }
        Ok(values)
    }

    fn lane_bias(&self, series_id: &str) -> f64 {
        if let Some(value) = self.lane_biases.get(series_id) {
            return *value;
        }
        let (origin, destination) = split_lane(series_id);
        let mut values = Vec::new();
        if let Some(origin) = origin {
            values.extend(self.lane_biases.iter().filter_map(|(lane, value)| {
                let (candidate_origin, _) = split_lane(lane);
                (candidate_origin == Some(origin)).then_some(*value)
            }));
        }
        if let Some(destination) = destination {
            values.extend(self.lane_biases.iter().filter_map(|(lane, value)| {
                let (_, candidate_destination) = split_lane(lane);
                (candidate_destination == Some(destination)).then_some(*value)
            }));
        }
        if values.is_empty() {
            mean(&self.lane_biases.values().copied().collect::<Vec<_>>())
        } else {
            mean(&values)
        }
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
        self.fit_lane_state(frame);
        let augmented_frame = self.augmented_lane_training_frame(frame)?;
        self.inner = NeuralPanelForecaster::new(self.augmented_lane_config())
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        self.inner.fit(&augmented_frame)
    }

    fn predict(&self, horizon: usize) -> CoreResult<ForecastResult> {
        let result = self.inner.predict(horizon)?;
        let mut predictions = result.predictions().to_vec();
        for prediction in &mut predictions {
            prediction.model = self.model_name().to_string();
            prediction.mean += self.lane_bias(&prediction.series_id);
        }
        ForecastResult::new(predictions)
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

fn quantile_output_order(quantiles: &[f64]) -> Vec<f64> {
    let mut output = Vec::with_capacity(quantiles.len());
    output.push(0.5);
    output.extend(
        quantiles
            .iter()
            .copied()
            .filter(|quantile| (*quantile - 0.5).abs() >= f64::EPSILON),
    );
    output
}

fn index_known_future_covariates(
    covariates: &KnownFutureCovariates,
) -> KnownFutureCovariateIndex<'_> {
    let mut index: KnownFutureCovariateIndex<'_> = BTreeMap::new();
    for ((series_id, timestamp), values) in covariates {
        index
            .entry(series_id.as_str())
            .or_default()
            .insert(*timestamp, values);
    }
    index
}

fn future_feature_specs(config: &NeuralPanelConfig) -> Vec<FutureFeatureSpec> {
    let mut specs = Vec::new();
    let seasonality_basis = |name: &str| match name {
        "daily" => SeasonalityBasis::Daily,
        "weekly" => SeasonalityBasis::Weekly,
        "yearly" => SeasonalityBasis::Yearly,
        _ => unreachable!(),
    };
    for (name, order) in [
        ("daily", config.daily_fourier_order),
        ("weekly", config.weekly_fourier_order),
        ("yearly", config.yearly_fourier_order),
    ] {
        for harmonic in 1..=order {
            specs.push(FutureFeatureSpec::Seasonality {
                name: format!("seasonality:{name}:sin:{harmonic}"),
                basis: seasonality_basis(name),
                harmonic,
                is_cosine: false,
                component_mode: config.seasonality_mode,
                global_local_mode: config.seasonality_global_local,
                condition_name: None,
            });
            specs.push(FutureFeatureSpec::Seasonality {
                name: format!("seasonality:{name}:cos:{harmonic}"),
                basis: seasonality_basis(name),
                harmonic,
                is_cosine: true,
                component_mode: config.seasonality_mode,
                global_local_mode: config.seasonality_global_local,
                condition_name: None,
            });
        }
    }
    for (name, (period, order)) in &config.custom_seasonalities {
        for harmonic in 1..=*order {
            let condition_name = config
                .custom_seasonality_conditions
                .get(name)
                .and_then(|value| value.clone());
            let basis = SeasonalityBasis::Custom { period: *period };
            specs.push(FutureFeatureSpec::Seasonality {
                name: format!("seasonality:{name}:sin:{harmonic}"),
                basis: basis.clone(),
                harmonic,
                is_cosine: false,
                component_mode: config.seasonality_mode,
                global_local_mode: config.seasonality_global_local,
                condition_name: condition_name.clone(),
            });
            specs.push(FutureFeatureSpec::Seasonality {
                name: format!("seasonality:{name}:cos:{harmonic}"),
                basis,
                harmonic,
                is_cosine: true,
                component_mode: config.seasonality_mode,
                global_local_mode: config.seasonality_global_local,
                condition_name,
            });
        }
    }
    for (name, offsets) in &config.events {
        for offset in offsets {
            specs.push(FutureFeatureSpec::Event {
                name: format!("event:{name}:{offset}"),
                component_mode: config.event_mode,
                global_local_mode: config.event_global_local,
            });
        }
    }
    for (name, mode) in &config.future_regressors {
        specs.push(FutureFeatureSpec::Regressor {
            name: name.clone(),
            component_mode: *mode,
            global_local_mode: config.regressor_global_local,
        });
    }
    specs
}

fn build_future_features(
    row: &ForecastRow,
    feature_specs: &[FutureFeatureSpec],
    config: &NeuralPanelConfig,
) -> Result<Vec<f64>> {
    feature_specs
        .iter()
        .map(|spec| {
            spec.value_for_row(row, config)
                .map_err(|err| NeuralError::InvalidArgument(err.to_string()))
        })
        .collect()
}

fn collect_static_future_covariates(
    frame: &ForecastFrame,
    config: &NeuralPanelConfig,
) -> BTreeMap<String, BTreeMap<String, f64>> {
    let mut by_series = BTreeMap::new();
    for series_id in frame.series_ids() {
        let rows = frame.rows_for_series(&series_id);
        let mut values = BTreeMap::new();
        for name in config.future_regressors.keys().chain(
            config
                .custom_seasonality_conditions
                .values()
                .filter_map(|value| value.as_ref()),
        ) {
            let mut distinct = rows
                .iter()
                .filter_map(|row| row.covariates.get(name).copied())
                .collect::<Vec<_>>();
            if distinct.len() != rows.len() {
                continue;
            }
            distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let Some(first) = distinct.first().copied() else {
                continue;
            };
            let Some(last) = distinct.last().copied() else {
                continue;
            };
            if (first - last).abs() <= 1.0e-12 {
                values.insert(name.clone(), first);
            }
        }
        if !values.is_empty() {
            by_series.insert(series_id, values);
        }
    }
    by_series
}

fn fit_piecewise_trend(
    frame: &ForecastFrame,
    scaler: &StandardScaler,
    series_lengths: &BTreeMap<String, usize>,
    changepoints: &[f64],
    config: &NeuralPanelConfig,
) -> (Vec<f64>, BTreeMap<String, Vec<f64>>) {
    let basis_width = 2 + changepoints.len();
    let mut global_features = Vec::new();
    let mut global_targets = Vec::new();
    let mut global_weights = Vec::new();
    for series_id in frame.series_ids() {
        let rows = frame.rows_for_series(&series_id);
        let series_len = series_lengths
            .get(&series_id)
            .copied()
            .unwrap_or(rows.len())
            .max(1);
        for (idx, row) in rows.iter().enumerate() {
            let position = series_position_from_length(idx, series_len);
            global_features.push(trend_basis(position, changepoints));
            global_targets.push(scaler.transform(row.target));
            global_weights.push(recency_weight(idx, rows.len(), config));
        }
    }
    let mut global_trend_coefficients =
        ridge_fit(&global_features, &global_targets, &global_weights, 1.0e-6);
    if global_trend_coefficients.len() != basis_width {
        global_trend_coefficients.resize(basis_width, 0.0);
    }
    let mut local_trend_coefficients = BTreeMap::new();
    if config.trend_mode != NeuralPanelMode::Global {
        for series_id in frame.series_ids() {
            let rows = frame.rows_for_series(&series_id);
            let series_len = series_lengths
                .get(&series_id)
                .copied()
                .unwrap_or(rows.len())
                .max(1);
            let mut local_features = Vec::new();
            let mut local_targets = Vec::new();
            let mut local_weights = Vec::new();
            for (idx, row) in rows.iter().enumerate() {
                let position = series_position_from_length(idx, series_len);
                let basis = trend_basis(position, changepoints);
                local_features.push(basis.clone());
                local_targets
                    .push(scaler.transform(row.target) - dot(&global_trend_coefficients, &basis));
                local_weights.push(recency_weight(idx, rows.len(), config));
            }
            let coefficients = ridge_fit(
                &local_features,
                &local_targets,
                &local_weights,
                config.local_l2.max(1.0e-6),
            );
            if coefficients.iter().any(|value| value.abs() > 0.0) {
                local_trend_coefficients.insert(series_id, coefficients);
            }
        }
    }
    (global_trend_coefficients, local_trend_coefficients)
}

fn trend_changepoints(n_changepoints: usize, changepoints_range: f64) -> Vec<f64> {
    if n_changepoints == 0 {
        return Vec::new();
    }
    let range = changepoints_range.clamp(0.0, 1.0);
    (0..n_changepoints)
        .map(|idx| range * (idx as f64 + 1.0) / (n_changepoints as f64 + 1.0))
        .collect()
}

fn trend_basis(position: f64, changepoints: &[f64]) -> Vec<f64> {
    let mut basis = vec![1.0, position];
    basis.extend(
        changepoints
            .iter()
            .map(|changepoint| (position - changepoint).max(0.0)),
    );
    basis
}

fn series_position_from_length(index: usize, series_length: usize) -> f64 {
    if series_length <= 1 {
        0.0
    } else {
        index as f64 / (series_length - 1) as f64
    }
}

#[allow(clippy::needless_range_loop)]
fn ridge_fit(features: &[Vec<f64>], targets: &[f64], weights: &[f64], ridge: f64) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }
    let width = features[0].len();
    let mut xtwx = vec![vec![0.0; width]; width];
    let mut xtwy = vec![0.0; width];
    for ((row, target), weight) in features.iter().zip(targets).zip(weights) {
        let w = (*weight).max(0.0);
        for i in 0..width {
            xtwy[i] += w * row[i] * target;
            for j in i..width {
                xtwx[i][j] += w * row[i] * row[j];
            }
        }
    }
    for i in 0..width {
        xtwx[i][i] += ridge;
        for j in 0..i {
            xtwx[i][j] = xtwx[j][i];
        }
    }
    solve_linear_system(xtwx, xtwy).unwrap_or_else(|| vec![0.0; width])
}

#[allow(clippy::needless_range_loop)]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let pivot = (i..n)
            .max_by(|&left, &right| {
                a[left][i]
                    .abs()
                    .partial_cmp(&a[right][i].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(i);
        if a[pivot][i].abs() < 1.0e-12 {
            return None;
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let pivot_value = a[i][i];
        for j in i..n {
            a[i][j] /= pivot_value;
        }
        b[i] /= pivot_value;
        for row in 0..n {
            if row == i {
                continue;
            }
            let factor = a[row][i];
            if factor.abs() < 1.0e-12 {
                continue;
            }
            for col in i..n {
                a[row][col] -= factor * a[i][col];
            }
            b[row] -= factor * b[i];
        }
    }
    Some(b)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
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
    spec: &FutureFeatureSpec,
    timestamp: NaiveDateTime,
    covariates: Option<&BTreeMap<String, f64>>,
    static_covariates: Option<&BTreeMap<String, f64>>,
    config: &NeuralPanelConfig,
) -> CoreResult<f64> {
    spec.value_for_timestamp(timestamp, covariates, static_covariates, config)
}

fn apply_custom_seasonality_condition(
    feature_name: &str,
    value: f64,
    covariates: Option<&BTreeMap<String, f64>>,
    static_covariates: Option<&BTreeMap<String, f64>>,
    config: &NeuralPanelConfig,
    condition_name: Option<&str>,
) -> CoreResult<f64> {
    let Some(condition_name) = condition_name.or_else(|| {
        config
            .custom_seasonality_conditions
            .get(feature_name.split(':').nth(1).unwrap_or_default())
            .and_then(|value| value.as_deref())
    }) else {
        return Ok(value);
    };
    let condition = covariates
        .and_then(|values| values.get(condition_name).copied())
        .or_else(|| static_covariates.and_then(|values| values.get(condition_name).copied()))
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "conditional seasonality '{feature_name}' requires known covariate '{condition_name}'"
            ))
        })?;
    Ok(value * condition)
}

fn fit_nonstationary_feature_weights(
    dataset: &NeuralPanelWindowDataset,
    scaler: &StandardScaler,
    baseline: impl Fn(&str, usize) -> f64,
) -> BTreeMap<String, f64> {
    let feature_specs = dataset.future_feature_specs();
    let mut numerators = vec![0.0; feature_specs.len()];
    let mut denominators = vec![0.0; feature_specs.len()];
    for window in dataset.windows() {
        for horizon_idx in 0..window.targets.len() {
            let offset = window.lags.len() + horizon_idx;
            let Some(features) = window.future_features.get(offset) else {
                continue;
            };
            let target = scaler.transform(window.targets[horizon_idx]);
            let trend_baseline = baseline(&window.series_id, offset);
            let residual = target - trend_baseline;
            for (feature_idx, value) in features.iter().enumerate() {
                let design = match feature_specs[feature_idx].component_mode() {
                    ComponentMode::Additive => *value,
                    ComponentMode::Multiplicative => trend_baseline * *value,
                };
                numerators[feature_idx] += design * residual;
                denominators[feature_idx] += design * design;
            }
        }
    }
    feature_specs
        .iter()
        .enumerate()
        .filter_map(|(idx, spec)| {
            let denom = denominators[idx] + 1.0e-6;
            if denom <= 1.0e-6 {
                return None;
            }
            Some((spec.name().to_string(), numerators[idx] / denom))
        })
        .collect()
}

#[derive(Debug, Clone)]
struct TrainingExample {
    input: Vec<f64>,
    target: Vec<f64>,
    weight: f64,
}

impl MlpState {
    fn initialized(
        input_width: usize,
        output_width: usize,
        hidden_layers: &[usize],
        seed: u64,
    ) -> Self {
        let mut widths = Vec::with_capacity(hidden_layers.len() + 2);
        widths.push(input_width);
        widths.extend_from_slice(hidden_layers);
        widths.push(output_width);
        let layers = widths
            .windows(2)
            .enumerate()
            .map(|(layer_idx, pair)| {
                let fan_in = pair[0].max(1) as f64;
                let scale = (2.0 / fan_in).sqrt();
                let weights = (0..pair[1])
                    .map(|out_idx| {
                        (0..pair[0])
                            .map(|in_idx| {
                                deterministic_scalar(
                                    seed ^ ((layer_idx as u64 + 1) << 32),
                                    out_idx * pair[0] + in_idx,
                                    0.071,
                                ) * scale
                                    * 100.0
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                DenseLayer {
                    weights,
                    biases: vec![0.0; pair[1]],
                }
            })
            .collect();
        Self {
            input_width,
            output_width,
            hidden_layers: hidden_layers.to_vec(),
            layers,
        }
    }

    fn forward(&self, input: &[f64], backend: &BackendSelection) -> Result<Vec<f64>> {
        let mut activation = padded_input(input, self.input_width);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let is_last = layer_idx + 1 == self.layers.len();
            let features = vec![activation
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>()];
            let weights = (0..activation.len())
                .flat_map(|input_idx| layer.weights.iter().map(move |row| row[input_idx] as f32))
                .collect::<Vec<_>>();
            let biases = layer
                .biases
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let rows = backend_dense_layer_f32(backend, &features, &weights, &biases)?;
            activation = rows
                .into_iter()
                .next()
                .expect("backend dense layer returns one row")
                .into_iter()
                .map(|value| {
                    let value = f64::from(value);
                    if is_last {
                        value
                    } else {
                        value.max(0.0)
                    }
                })
                .collect();
        }
        Ok(activation)
    }

    fn forward_cpu(&self, input: &[f64]) -> Vec<f64> {
        let mut activation = padded_input(input, self.input_width);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let is_last = layer_idx + 1 == self.layers.len();
            let mut next = Vec::with_capacity(layer.biases.len());
            for (row, bias) in layer.weights.iter().zip(&layer.biases) {
                let value = row
                    .iter()
                    .zip(&activation)
                    .map(|(weight, input)| weight * input)
                    .sum::<f64>()
                    + bias;
                next.push(if is_last { value } else { value.max(0.0) });
            }
            activation = next;
        }
        activation
    }
}

fn train_mlp(
    input_width: usize,
    output_width: usize,
    hidden_layers: &[usize],
    examples: Vec<TrainingExample>,
    config: &NeuralPanelConfig,
    seed: u64,
) -> MlpState {
    let mut net = MlpState::initialized(input_width, output_width, hidden_layers, seed);
    if examples.is_empty() || input_width == 0 || output_width == 0 {
        return net;
    }
    let mut m = clone_zero_layers(&net.layers);
    let mut v = clone_zero_layers(&net.layers);
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1.0e-8;
    let mut step = 0.0_f64;
    for _epoch in 0..config.epochs {
        for example in &examples {
            step += 1.0;
            let (activations, preactivations) = forward_trace(&net, &example.input);
            let prediction = activations.last().cloned().unwrap_or_default();
            let mut delta = prediction
                .iter()
                .zip(&example.target)
                .map(|(pred, target)| {
                    example.weight * loss_gradient(*pred - *target, config.loss)
                        / output_width.max(1) as f64
                })
                .collect::<Vec<_>>();
            for layer_idx in (0..net.layers.len()).rev() {
                let previous = &activations[layer_idx];
                let layer = &mut net.layers[layer_idx];
                let mut previous_delta = vec![0.0; previous.len()];
                for (out_idx, delta_value) in
                    delta.iter().copied().enumerate().take(layer.biases.len())
                {
                    for in_idx in 0..layer.weights[out_idx].len() {
                        let grad = delta_value * previous[in_idx]
                            + config.weight_decay * layer.weights[out_idx][in_idx];
                        adamw_update(
                            &mut layer.weights[out_idx][in_idx],
                            &mut m[layer_idx].weights[out_idx][in_idx],
                            &mut v[layer_idx].weights[out_idx][in_idx],
                            grad,
                            config.learning_rate,
                            step,
                            beta1,
                            beta2,
                            eps,
                        );
                        previous_delta[in_idx] += delta_value * layer.weights[out_idx][in_idx];
                    }
                    adamw_update(
                        &mut layer.biases[out_idx],
                        &mut m[layer_idx].biases[out_idx],
                        &mut v[layer_idx].biases[out_idx],
                        delta_value,
                        config.learning_rate,
                        step,
                        beta1,
                        beta2,
                        eps,
                    );
                }
                if layer_idx > 0 {
                    delta = previous_delta
                        .into_iter()
                        .zip(&preactivations[layer_idx - 1])
                        .map(|(grad, preactivation)| if *preactivation > 0.0 { grad } else { 0.0 })
                        .collect();
                }
            }
        }
    }
    net
}

fn ar_training_examples(
    config: &NeuralPanelConfig,
    dataset: &NeuralPanelWindowDataset,
    scaler: &StandardScaler,
    baseline: impl Fn(&str, usize) -> f64,
) -> Vec<TrainingExample> {
    let output_width = config.n_forecasts * config.quantiles.len();
    dataset
        .windows()
        .iter()
        .enumerate()
        .map(|(example_idx, window)| {
            let input = window
                .lags
                .iter()
                .enumerate()
                .map(|(idx, value)| scaler.transform(*value) - baseline(&window.series_id, idx))
                .collect::<Vec<_>>();
            let mut target = vec![0.0; output_width];
            for horizon_idx in 0..config.n_forecasts {
                let offset = config.n_lags + horizon_idx;
                target[horizon_idx * config.quantiles.len()] = scaler
                    .transform(window.targets[horizon_idx])
                    - baseline(&window.series_id, offset);
            }
            TrainingExample {
                input,
                target,
                weight: recency_weight(example_idx, dataset.windows().len(), config),
            }
        })
        .collect()
}

fn covar_training_examples(
    config: &NeuralPanelConfig,
    dataset: &NeuralPanelWindowDataset,
    scaler: &StandardScaler,
    ar_net: &Option<MlpState>,
    baseline: impl Fn(&str, usize) -> f64,
) -> Vec<TrainingExample> {
    let output_width = config.n_forecasts * config.quantiles.len();
    dataset
        .windows()
        .iter()
        .enumerate()
        .map(|(example_idx, window)| {
            let input = lagged_covariate_input(config, &window.lagged_covariates);
            let ar_output = ar_net
                .as_ref()
                .map(|net| {
                    let ar_input = window
                        .lags
                        .iter()
                        .enumerate()
                        .map(|(idx, value)| {
                            scaler.transform(*value) - baseline(&window.series_id, idx)
                        })
                        .collect::<Vec<_>>();
                    net.forward_cpu(&ar_input)
                })
                .unwrap_or_else(|| vec![0.0; output_width]);
            let mut target = vec![0.0; output_width];
            for horizon_idx in 0..config.n_forecasts {
                let output_idx = horizon_idx * config.quantiles.len();
                let offset = config.n_lags + horizon_idx;
                target[output_idx] = scaler.transform(window.targets[horizon_idx])
                    - baseline(&window.series_id, offset)
                    - ar_output.get(output_idx).copied().unwrap_or(0.0);
            }
            TrainingExample {
                input,
                target,
                weight: recency_weight(example_idx, dataset.windows().len(), config),
            }
        })
        .collect()
}

fn fit_local_feature_weights(
    config: &NeuralPanelConfig,
    dataset: &NeuralPanelWindowDataset,
    scaler: &StandardScaler,
    global_weights: &BTreeMap<String, f64>,
    baseline: impl Fn(&str, usize) -> f64,
) -> BTreeMap<String, BTreeMap<String, f64>> {
    let feature_specs = dataset.future_feature_specs();
    if feature_specs
        .iter()
        .all(|spec| spec.global_local_mode() == NeuralPanelMode::Global)
    {
        return BTreeMap::new();
    }
    let mut numerators: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut denominators: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for window in dataset.windows() {
        let numerator = numerators
            .entry(window.series_id.clone())
            .or_insert_with(|| vec![0.0; feature_specs.len()]);
        let denominator = denominators
            .entry(window.series_id.clone())
            .or_insert_with(|| vec![0.0; feature_specs.len()]);
        for horizon_idx in 0..window.targets.len() {
            let offset = window.lags.len() + horizon_idx;
            let Some(features) = window.future_features.get(offset) else {
                continue;
            };
            let target = scaler.transform(window.targets[horizon_idx]);
            let trend_baseline = baseline(&window.series_id, offset);
            let global_component = features
                .iter()
                .enumerate()
                .map(|(feature_idx, value)| {
                    let design = match feature_specs[feature_idx].component_mode() {
                        ComponentMode::Additive => *value,
                        ComponentMode::Multiplicative => trend_baseline * *value,
                    };
                    design
                        * global_weights
                            .get(feature_specs[feature_idx].name())
                            .copied()
                            .unwrap_or(0.0)
                })
                .sum::<f64>();
            let residual = target - trend_baseline - global_component;
            for (feature_idx, value) in features.iter().enumerate() {
                let design = match feature_specs[feature_idx].component_mode() {
                    ComponentMode::Additive => *value,
                    ComponentMode::Multiplicative => trend_baseline * *value,
                };
                numerator[feature_idx] += design * residual;
                denominator[feature_idx] += design * design;
            }
        }
    }
    numerators
        .into_iter()
        .map(|(series_id, numerator)| {
            let denominator = denominators.remove(&series_id).unwrap_or_default();
            let weights = dataset
                .future_feature_specs()
                .iter()
                .enumerate()
                .filter_map(|(idx, spec)| {
                    let denom =
                        denominator.get(idx).copied().unwrap_or(0.0) + config.local_l2 + 1.0e-6;
                    if denom <= 1.0e-6 {
                        return None;
                    }
                    Some((spec.name().to_string(), numerator[idx] / denom))
                })
                .collect();
            (series_id, weights)
        })
        .collect()
}

fn lagged_covariate_input(
    config: &NeuralPanelConfig,
    covariates: &BTreeMap<String, Vec<f64>>,
) -> Vec<f64> {
    let mut input = Vec::new();
    for (name, lag) in &config.lagged_regressors {
        let values = covariates.get(name).cloned().unwrap_or_default();
        let start = values.len().saturating_sub(*lag);
        input.extend(values[start..].iter().copied());
    }
    input
}

fn add_median_outputs(output: &mut [f64], raw: &[f64], quantiles: &[f64]) {
    let quantile_count = quantiles.len();
    for (horizon_idx, value) in output.iter_mut().enumerate() {
        *value += raw
            .get(horizon_idx * quantile_count)
            .copied()
            .unwrap_or(0.0);
    }
}

fn recency_weight(idx: usize, total: usize, config: &NeuralPanelConfig) -> f64 {
    if !config.newer_sample_weight || total <= 1 {
        return 1.0;
    }
    let phase = idx as f64 / (total - 1) as f64;
    0.5 - 0.5 * (std::f64::consts::PI * (1.0 - phase)).cos()
}

fn loss_gradient(residual: f64, loss: NeuralPanelLoss) -> f64 {
    match loss {
        NeuralPanelLoss::SmoothL1 => {
            if residual.abs() < 1.0 {
                residual
            } else {
                residual.signum()
            }
        }
        NeuralPanelLoss::Mse => 2.0 * residual,
        NeuralPanelLoss::Mae => residual.signum(),
        NeuralPanelLoss::Pinball => {
            if residual >= 0.0 {
                0.5
            } else {
                -0.5
            }
        }
    }
}

fn padded_input(input: &[f64], width: usize) -> Vec<f64> {
    let mut values = input.iter().copied().take(width).collect::<Vec<_>>();
    values.resize(width, 0.0);
    values
}

fn forward_trace(net: &MlpState, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut activations = vec![padded_input(input, net.input_width)];
    let mut preactivations = Vec::new();
    for (layer_idx, layer) in net.layers.iter().enumerate() {
        let previous = activations.last().expect("activation");
        let is_last = layer_idx + 1 == net.layers.len();
        let preactivation = layer
            .weights
            .iter()
            .zip(&layer.biases)
            .map(|(row, bias)| {
                row.iter()
                    .zip(previous)
                    .map(|(weight, input)| weight * input)
                    .sum::<f64>()
                    + bias
            })
            .collect::<Vec<_>>();
        let activation = preactivation
            .iter()
            .map(|value| if is_last { *value } else { value.max(0.0) })
            .collect::<Vec<_>>();
        preactivations.push(preactivation);
        activations.push(activation);
    }
    (activations, preactivations)
}

fn clone_zero_layers(layers: &[DenseLayer]) -> Vec<DenseLayer> {
    layers
        .iter()
        .map(|layer| DenseLayer {
            weights: layer
                .weights
                .iter()
                .map(|row| vec![0.0; row.len()])
                .collect(),
            biases: vec![0.0; layer.biases.len()],
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn adamw_update(
    value: &mut f64,
    m: &mut f64,
    v: &mut f64,
    grad: f64,
    learning_rate: f64,
    step: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) {
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;
    let m_hat = *m / (1.0 - beta1.powf(step));
    let v_hat = *v / (1.0 - beta2.powf(step));
    *value -= learning_rate * m_hat / (v_hat.sqrt() + eps);
}

fn learned_quantile_diffs(quantiles: &[f64], mut residuals: Vec<f64>) -> Vec<f64> {
    residuals.retain(|value| value.is_finite());
    if residuals.is_empty() {
        return vec![0.0; quantiles.len()];
    }
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_residual = empirical_quantile(&residuals, 0.5);
    quantiles
        .iter()
        .map(|quantile| {
            if (*quantile - 0.5).abs() < f64::EPSILON {
                0.0
            } else {
                let diff = empirical_quantile(&residuals, *quantile) - median_residual;
                if *quantile < 0.5 {
                    diff.min(0.0)
                } else {
                    diff.max(0.0)
                }
            }
        })
        .collect()
}

fn empirical_quantile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let position = quantile.clamp(0.0, 1.0) * (sorted_values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let fraction = position - lower as f64;
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    }
}

fn repaired_quantiles(
    median: f64,
    quantiles: &[f64],
    quantile_output_order: &[f64],
    residual_diffs: &[f64],
    scale: f64,
) -> Vec<f64> {
    let mut values = quantiles
        .iter()
        .map(|quantile| {
            if (*quantile - 0.5).abs() < f64::EPSILON {
                median
            } else if let Some(diff) =
                residual_diff_for_quantile(*quantile, quantile_output_order, residual_diffs)
                    .filter(|v| v.is_finite())
            {
                median + diff * scale
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

fn residual_diff_for_quantile(
    quantile: f64,
    quantile_output_order: &[f64],
    residual_diffs: &[f64],
) -> Option<f64> {
    quantile_output_order
        .iter()
        .position(|candidate| (*candidate - quantile).abs() < 1.0e-12)
        .and_then(|idx| residual_diffs.get(idx).copied())
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

fn deterministic_embedding(key: &str, dim: usize, seed: u64) -> Vec<f64> {
    let base = key.bytes().fold(seed, |state, byte| {
        state.wrapping_mul(16777619) ^ byte as u64
    });
    (0..dim)
        .map(|idx| deterministic_scalar(base, idx, 0.037) * 100.0)
        .collect()
}

fn generated_lane_feature_names(embedding_dim: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(embedding_dim * 3 + 4);
    for idx in 0..embedding_dim {
        names.push(format!("lane_origin_embedding_{idx}"));
        names.push(format!("lane_destination_embedding_{idx}"));
        names.push(format!("lane_embedding_{idx}"));
    }
    names.extend([
        "lane_graph_origin_hash".to_string(),
        "lane_graph_destination_hash".to_string(),
        "lane_graph_direction_hash_delta".to_string(),
        "lane_graph_observed_lane_mean".to_string(),
    ]);
    names
}

fn stable_unit_hash(key: &str) -> f64 {
    let hash = key.bytes().fold(1469598103934665603_u64, |state, byte| {
        (state ^ byte as u64).wrapping_mul(1099511628211)
    });
    (hash % 10_000) as f64 / 10_000.0
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
