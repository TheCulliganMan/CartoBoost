use crate::forecasting::{ForecastFrame, ForecastRow};
use crate::{CartoBoostError, Result};
use chrono::{Datelike, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalendarFeature {
    DayOfWeek,
    DayOfWeekSin,
    DayOfWeekCos,
    Month,
    MonthSin,
    MonthCos,
    Day,
    DaySin,
    DayCos,
    MonthStart,
    MonthMiddle,
    MonthEnd,
    DayOfYear,
    ElapsedIndex,
    ElapsedPhase(usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LagFeatureConfig {
    pub lags: Vec<usize>,
    pub rolling_mean_windows: Vec<usize>,
    #[serde(default)]
    pub partial_rolling_mean_windows: Vec<usize>,
    #[serde(default)]
    pub rolling_std_windows: Vec<usize>,
    #[serde(default)]
    pub rolling_min_windows: Vec<usize>,
    #[serde(default)]
    pub rolling_max_windows: Vec<usize>,
    #[serde(default)]
    pub ewm_alpha_percents: Vec<u8>,
    pub calendar_features: Vec<CalendarFeature>,
    #[serde(default)]
    pub difference_lags: Vec<usize>,
    #[serde(default)]
    pub rolling_trend_windows: Vec<usize>,
    #[serde(default)]
    pub covariate_features: Vec<String>,
    #[serde(default)]
    pub covariate_indicator_values: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    pub covariate_calendar_interactions: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LagFeatureRow {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub target: f64,
    pub features: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LagFeatureBuilder {
    config: LagFeatureConfig,
    feature_names: Vec<String>,
}

impl Default for LagFeatureConfig {
    fn default() -> Self {
        Self {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
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
        }
    }
}

impl LagFeatureBuilder {
    pub fn new(config: LagFeatureConfig) -> Result<Self> {
        validate_config(&config)?;
        let mut feature_names = Vec::with_capacity(
            config.lags.len()
                + config.rolling_mean_windows.len()
                + config.partial_rolling_mean_windows.len()
                + config.rolling_std_windows.len()
                + config.rolling_min_windows.len()
                + config.rolling_max_windows.len()
                + config.ewm_alpha_percents.len()
                + config.difference_lags.len()
                + config.rolling_trend_windows.len()
                + config.calendar_features.len()
                + config.covariate_features.len()
                + covariate_indicator_count(&config)
                + if config.covariate_calendar_interactions {
                    (config.covariate_features.len() + covariate_indicator_count(&config))
                        * config.calendar_features.len()
                } else {
                    0
                },
        );
        feature_names.extend(config.lags.iter().map(|lag| format!("target_lag_{lag}")));
        feature_names.extend(
            config
                .rolling_mean_windows
                .iter()
                .map(|window| format!("target_roll_mean_{window}")),
        );
        feature_names.extend(
            config
                .partial_rolling_mean_windows
                .iter()
                .map(|window| format!("target_partial_roll_mean_{window}")),
        );
        feature_names.extend(
            config
                .rolling_std_windows
                .iter()
                .map(|window| format!("target_roll_std_{window}")),
        );
        feature_names.extend(
            config
                .rolling_min_windows
                .iter()
                .map(|window| format!("target_roll_min_{window}")),
        );
        feature_names.extend(
            config
                .rolling_max_windows
                .iter()
                .map(|window| format!("target_roll_max_{window}")),
        );
        feature_names.extend(
            config
                .ewm_alpha_percents
                .iter()
                .map(|alpha| format!("target_ewm_alpha_{alpha:03}")),
        );
        feature_names.extend(
            config
                .difference_lags
                .iter()
                .map(|lag| format!("target_delta_lag_{lag}")),
        );
        feature_names.extend(
            config
                .rolling_trend_windows
                .iter()
                .map(|window| format!("target_roll_trend_{window}")),
        );
        feature_names.extend(config.calendar_features.iter().map(calendar_feature_name));
        feature_names.extend(
            config
                .covariate_features
                .iter()
                .map(|name| format!("covariate_{name}")),
        );
        let indicator_feature_names = covariate_indicator_feature_names(&config);
        feature_names.extend(indicator_feature_names.iter().cloned());
        if config.covariate_calendar_interactions {
            for covariate in &config.covariate_features {
                for calendar in config
                    .calendar_features
                    .iter()
                    .filter(|feature| calendar_feature_allows_covariate_interaction(feature))
                {
                    feature_names.push(format!(
                        "covariate_{covariate}_x_{}",
                        calendar_feature_name(calendar)
                    ));
                }
            }
            for indicator in &indicator_feature_names {
                for calendar in config
                    .calendar_features
                    .iter()
                    .filter(|feature| calendar_feature_allows_covariate_interaction(feature))
                {
                    feature_names
                        .push(format!("{indicator}_x_{}", calendar_feature_name(calendar)));
                }
            }
        }
        Ok(Self {
            config,
            feature_names,
        })
    }

    pub fn config(&self) -> &LagFeatureConfig {
        &self.config
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn transform_frame(&self, frame: &ForecastFrame) -> Result<Vec<LagFeatureRow>> {
        let mut rows = Vec::new();
        for (series_id, history) in history_by_series(frame.rows()) {
            let cache = SeriesFeatureCache::new(&history, &self.config);
            for row_idx in 0..history.len() {
                if let Some(features) =
                    self.features_for_position_cached(&series_id, &history, &cache, row_idx)?
                {
                    rows.push(LagFeatureRow {
                        series_id: series_id.clone(),
                        timestamp: history[row_idx].timestamp,
                        target: history[row_idx].target,
                        features,
                    });
                }
            }
        }
        Ok(rows)
    }

    pub fn transform_next(
        &self,
        series_id: &str,
        history: &[ForecastRow],
        timestamp: NaiveDateTime,
    ) -> Result<Vec<f64>> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has no history for lag features"
            )));
        }
        if history.iter().any(|row| row.series_id != series_id) {
            return Err(CartoBoostError::InvalidInput(format!(
                "history for series {series_id} contains another series"
            )));
        }
        let mut sorted = history.to_vec();
        sorted.sort_by_key(|row| row.timestamp);
        if sorted
            .windows(2)
            .any(|pair| pair[0].timestamp >= pair[1].timestamp)
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "history for series {series_id} contains duplicate timestamps"
            )));
        }
        let prior = sorted
            .into_iter()
            .filter(|row| row.timestamp < timestamp)
            .collect::<Vec<_>>();
        self.features_from_prior(series_id, &prior, timestamp, None)?
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "series {series_id} does not have enough prior history for lag features"
                ))
            })
    }

    pub(crate) fn transform_next_sorted_prior(
        &self,
        series_id: &str,
        history: &[ForecastRow],
        timestamp: NaiveDateTime,
    ) -> Result<Vec<f64>> {
        self.transform_next_sorted_prior_with_covariates(series_id, history, timestamp, None)
    }

    pub(crate) fn transform_next_sorted_prior_with_covariates(
        &self,
        series_id: &str,
        history: &[ForecastRow],
        timestamp: NaiveDateTime,
        covariates: Option<&BTreeMap<String, f64>>,
    ) -> Result<Vec<f64>> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has no history for lag features"
            )));
        }
        if history.iter().any(|row| row.series_id != series_id) {
            return Err(CartoBoostError::InvalidInput(format!(
                "history for series {series_id} contains another series"
            )));
        }
        if history
            .windows(2)
            .any(|pair| pair[0].timestamp >= pair[1].timestamp)
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "history for series {series_id} must be strictly sorted by timestamp"
            )));
        }
        let prior_end = history.partition_point(|row| row.timestamp < timestamp);
        let covariate_source = covariates.map(|values| ForecastRow {
            series_id: series_id.to_string(),
            timestamp,
            target: 0.0,
            covariates: values.clone(),
        });
        self.features_from_prior(
            series_id,
            &history[..prior_end],
            timestamp,
            covariate_source.as_ref(),
        )?
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "series {series_id} does not have enough prior history for lag features"
            ))
        })
    }

    #[cfg(test)]
    fn features_for_position(
        &self,
        history: &[ForecastRow],
        row_idx: usize,
    ) -> Result<Option<Vec<f64>>> {
        let series_id = history
            .get(row_idx)
            .map(|row| row.series_id.as_str())
            .unwrap_or("<unknown>");
        self.features_from_prior(
            series_id,
            &history[..row_idx],
            history[row_idx].timestamp,
            Some(&history[row_idx]),
        )
    }

    fn features_for_position_cached(
        &self,
        series_id: &str,
        history: &[ForecastRow],
        cache: &SeriesFeatureCache,
        row_idx: usize,
    ) -> Result<Option<Vec<f64>>> {
        let prior_len = row_idx;
        let timestamp = history[row_idx].timestamp;
        let mut features = Vec::with_capacity(self.feature_names.len());
        for lag in &self.config.lags {
            if prior_len < *lag {
                return Ok(None);
            }
            features.push(cache.targets[prior_len - *lag]);
        }
        for window in &self.config.rolling_mean_windows {
            if prior_len < *window {
                return Ok(None);
            }
            let sum = cache.window_sum(prior_len, *window);
            let mean = sum / *window as f64;
            if !mean.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling mean for series {series_id} is not finite"
                )));
            }
            features.push(mean);
        }
        for window in &self.config.partial_rolling_mean_windows {
            if prior_len == 0 {
                return Ok(None);
            }
            let effective_window = (*window).min(prior_len);
            let sum = cache.window_sum(prior_len, effective_window);
            let mean = sum / effective_window as f64;
            if !mean.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "partial rolling mean for series {series_id} is not finite"
                )));
            }
            features.push(mean);
        }
        for window in &self.config.rolling_std_windows {
            if prior_len < *window {
                return Ok(None);
            }
            let start = prior_len - *window;
            let values = &cache.targets[start..prior_len];
            let mean = values.iter().sum::<f64>() / *window as f64;
            let variance = values
                .iter()
                .map(|target| {
                    let delta = *target - mean;
                    delta * delta
                })
                .sum::<f64>()
                / *window as f64;
            let std = variance.sqrt();
            if !std.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling standard deviation for series {series_id} is not finite"
                )));
            }
            features.push(std);
        }
        for window in &self.config.rolling_min_windows {
            if prior_len < *window {
                return Ok(None);
            }
            let start = prior_len - *window;
            let min = cache.targets[start..prior_len]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            if !min.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling minimum for series {series_id} is not finite"
                )));
            }
            features.push(min);
        }
        for window in &self.config.rolling_max_windows {
            if prior_len < *window {
                return Ok(None);
            }
            let start = prior_len - *window;
            let max = cache.targets[start..prior_len]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if !max.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling maximum for series {series_id} is not finite"
                )));
            }
            features.push(max);
        }
        for (alpha_idx, _alpha_percent) in self.config.ewm_alpha_percents.iter().enumerate() {
            if prior_len == 0 {
                return Ok(None);
            }
            let ewm = cache.ewm_prior_by_alpha[alpha_idx][prior_len];
            if !ewm.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "exponentially weighted mean for series {series_id} is not finite"
                )));
            }
            features.push(ewm);
        }
        for lag in &self.config.difference_lags {
            if prior_len <= *lag {
                return Ok(None);
            }
            let delta = cache.targets[prior_len - 1] - cache.targets[prior_len - 1 - *lag];
            if !delta.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "lag delta for series {series_id} is not finite"
                )));
            }
            features.push(delta);
        }
        for window in &self.config.rolling_trend_windows {
            if *window < 2 || prior_len < *window {
                return Ok(None);
            }
            let first = cache.targets[prior_len - *window];
            let last = cache.targets[prior_len - 1];
            let trend = (last - first) / (*window - 1) as f64;
            if !trend.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling trend for series {series_id} is not finite"
                )));
            }
            features.push(trend);
        }
        features.extend(
            self.config
                .calendar_features
                .iter()
                .map(|feature| calendar_feature_value(feature, timestamp, prior_len)),
        );
        let calendar_values = if self.config.covariate_calendar_interactions {
            self.config
                .calendar_features
                .iter()
                .filter(|feature| calendar_feature_allows_covariate_interaction(feature))
                .map(|feature| calendar_feature_value(feature, timestamp, prior_len))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut covariate_values = Vec::with_capacity(self.config.covariate_features.len());
        let mut indicator_values = Vec::with_capacity(covariate_indicator_count(&self.config));
        let mut indicator_calendar_values = Vec::new();
        if self.config.covariate_calendar_interactions {
            indicator_calendar_values = self
                .config
                .calendar_features
                .iter()
                .filter(|feature| calendar_feature_allows_covariate_interaction(feature))
                .map(|feature| calendar_feature_value(feature, timestamp, prior_len))
                .collect::<Vec<_>>();
        }
        for name in &self.config.covariate_features {
            let value = training_covariate_value(history, row_idx, name).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing covariate {name:?} for series {series_id}"
                ))
            })?;
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "covariate {name:?} for series {series_id} is not finite"
                )));
            }
            features.push(value);
            covariate_values.push(value);
        }
        for (name, values) in &self.config.covariate_indicator_values {
            let value = training_covariate_value(history, row_idx, name).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing covariate {name:?} for series {series_id}"
                ))
            })?;
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "covariate {name:?} for series {series_id} is not finite"
                )));
            }
            for indicator_value in values {
                let indicator = covariate_indicator_value(value, *indicator_value);
                features.push(indicator);
                indicator_values.push(indicator);
            }
        }
        if self.config.covariate_calendar_interactions {
            for covariate in &covariate_values {
                for calendar in &calendar_values {
                    features.push(covariate * calendar);
                }
            }
            for indicator in &indicator_values {
                for calendar in &indicator_calendar_values {
                    features.push(indicator * calendar);
                }
            }
        }
        Ok(Some(features))
    }

    fn features_from_prior(
        &self,
        series_id: &str,
        prior: &[ForecastRow],
        timestamp: NaiveDateTime,
        covariate_source: Option<&ForecastRow>,
    ) -> Result<Option<Vec<f64>>> {
        let mut features = Vec::with_capacity(self.feature_names.len());
        for lag in &self.config.lags {
            if prior.len() < *lag {
                return Ok(None);
            }
            let row = &prior[prior.len() - *lag];
            features.push(row.target);
        }
        for window in &self.config.rolling_mean_windows {
            if prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let sum = prior[start..].iter().map(|row| row.target).sum::<f64>();
            let mean = sum / *window as f64;
            if !mean.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling mean for series {series_id} is not finite"
                )));
            }
            features.push(mean);
        }
        for window in &self.config.partial_rolling_mean_windows {
            if prior.is_empty() {
                return Ok(None);
            }
            let effective_window = (*window).min(prior.len());
            let start = prior.len() - effective_window;
            let sum = prior[start..].iter().map(|row| row.target).sum::<f64>();
            let mean = sum / effective_window as f64;
            if !mean.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "partial rolling mean for series {series_id} is not finite"
                )));
            }
            features.push(mean);
        }
        for window in &self.config.rolling_std_windows {
            if prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let values = &prior[start..];
            let mean = values.iter().map(|row| row.target).sum::<f64>() / *window as f64;
            let variance = values
                .iter()
                .map(|row| {
                    let delta = row.target - mean;
                    delta * delta
                })
                .sum::<f64>()
                / *window as f64;
            let std = variance.sqrt();
            if !std.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling standard deviation for series {series_id} is not finite"
                )));
            }
            features.push(std);
        }
        for window in &self.config.rolling_min_windows {
            if prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let min = prior[start..]
                .iter()
                .map(|row| row.target)
                .fold(f64::INFINITY, f64::min);
            if !min.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling minimum for series {series_id} is not finite"
                )));
            }
            features.push(min);
        }
        for window in &self.config.rolling_max_windows {
            if prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let max = prior[start..]
                .iter()
                .map(|row| row.target)
                .fold(f64::NEG_INFINITY, f64::max);
            if !max.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling maximum for series {series_id} is not finite"
                )));
            }
            features.push(max);
        }
        for alpha_percent in &self.config.ewm_alpha_percents {
            if prior.is_empty() {
                return Ok(None);
            }
            let alpha = f64::from(*alpha_percent) / 100.0;
            let mut ewm = prior[0].target;
            for row in &prior[1..] {
                ewm = alpha * row.target + (1.0 - alpha) * ewm;
            }
            if !ewm.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "exponentially weighted mean for series {series_id} is not finite"
                )));
            }
            features.push(ewm);
        }
        for lag in &self.config.difference_lags {
            if prior.len() <= *lag {
                return Ok(None);
            }
            let last = prior
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty prior history".to_string()))?;
            let row = &prior[prior.len() - 1 - *lag];
            let delta = last.target - row.target;
            if !delta.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "lag delta for series {series_id} is not finite"
                )));
            }
            features.push(delta);
        }
        for window in &self.config.rolling_trend_windows {
            if *window < 2 || prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let first = prior[start].target;
            let last = prior
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty prior history".to_string()))?
                .target;
            let trend = (last - first) / (*window - 1) as f64;
            if !trend.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling trend for series {series_id} is not finite"
                )));
            }
            features.push(trend);
        }
        features.extend(
            self.config
                .calendar_features
                .iter()
                .map(|feature| calendar_feature_value(feature, timestamp, prior.len())),
        );
        let calendar_values = if self.config.covariate_calendar_interactions {
            self.config
                .calendar_features
                .iter()
                .filter(|feature| calendar_feature_allows_covariate_interaction(feature))
                .map(|feature| calendar_feature_value(feature, timestamp, prior.len()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let covariate_source = covariate_source.or_else(|| prior.last());
        let mut covariate_values = Vec::with_capacity(self.config.covariate_features.len());
        let mut indicator_values = Vec::with_capacity(covariate_indicator_count(&self.config));
        let mut indicator_calendar_values = Vec::new();
        if self.config.covariate_calendar_interactions {
            indicator_calendar_values = self
                .config
                .calendar_features
                .iter()
                .filter(|feature| calendar_feature_allows_covariate_interaction(feature))
                .map(|feature| calendar_feature_value(feature, timestamp, prior.len()))
                .collect::<Vec<_>>();
        }
        for name in &self.config.covariate_features {
            let value = covariate_source
                .and_then(|row| row.covariates.get(name))
                .copied()
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing covariate {name:?} for series {series_id}"
                    ))
                })?;
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "covariate {name:?} for series {series_id} is not finite"
                )));
            }
            features.push(value);
            covariate_values.push(value);
        }
        for (name, values) in &self.config.covariate_indicator_values {
            let value = covariate_source
                .and_then(|row| row.covariates.get(name))
                .copied()
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing covariate {name:?} for series {series_id}"
                    ))
                })?;
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "covariate {name:?} for series {series_id} is not finite"
                )));
            }
            for indicator_value in values {
                let indicator = covariate_indicator_value(value, *indicator_value);
                features.push(indicator);
                indicator_values.push(indicator);
            }
        }
        if self.config.covariate_calendar_interactions {
            for covariate in &covariate_values {
                for calendar in &calendar_values {
                    features.push(covariate * calendar);
                }
            }
            for indicator in &indicator_values {
                for calendar in &indicator_calendar_values {
                    features.push(indicator * calendar);
                }
            }
        }
        Ok(Some(features))
    }
}

pub(crate) fn validate_lag_config_supported_by_prior(
    config: &LagFeatureConfig,
    max_prior_len: usize,
    model_name: &str,
) -> Result<()> {
    let required_prior_len = minimum_prior_len(config);
    if required_prior_len > max_prior_len {
        return Err(CartoBoostError::InvalidInput(format!(
            "{model_name} lag configuration requires at least {required_prior_len} prior observations per training row, but the shortest series can provide at most {max_prior_len}"
        )));
    }
    Ok(())
}

pub(crate) fn minimum_prior_len(config: &LagFeatureConfig) -> usize {
    config
        .lags
        .iter()
        .chain(&config.rolling_mean_windows)
        .chain(&config.rolling_std_windows)
        .chain(&config.rolling_min_windows)
        .chain(&config.rolling_max_windows)
        .chain(&config.rolling_trend_windows)
        .copied()
        .chain(
            config
                .difference_lags
                .iter()
                .map(|lag| lag.saturating_add(1)),
        )
        .max()
        .unwrap_or(0)
        .max(
            if config.partial_rolling_mean_windows.is_empty()
                && config.ewm_alpha_percents.is_empty()
            {
                0
            } else {
                1
            },
        )
}

pub(crate) fn lag_config_supported_by_prior(
    config: &LagFeatureConfig,
    max_prior_len: usize,
) -> LagFeatureConfig {
    let mut supported = config.clone();
    supported
        .lags
        .retain(|window| *window > 0 && *window <= max_prior_len);
    supported
        .rolling_mean_windows
        .retain(|window| *window > 0 && *window <= max_prior_len);
    supported
        .partial_rolling_mean_windows
        .retain(|window| *window > 0 && max_prior_len > 0);
    supported
        .rolling_std_windows
        .retain(|window| *window > 1 && *window <= max_prior_len);
    supported
        .rolling_min_windows
        .retain(|window| *window > 0 && *window <= max_prior_len);
    supported
        .rolling_max_windows
        .retain(|window| *window > 0 && *window <= max_prior_len);
    supported
        .difference_lags
        .retain(|window| *window > 0 && *window < max_prior_len);
    supported
        .rolling_trend_windows
        .retain(|window| *window > 1 && *window <= max_prior_len);
    if lacks_target_history_feature(&supported)
        && supported.calendar_features.is_empty()
        && supported.covariate_features.is_empty()
        && supported.covariate_indicator_values.is_empty()
        && max_prior_len >= 1
    {
        supported.lags.push(1);
    }
    sort_dedup_lag_config(&mut supported);
    supported
}

fn lacks_target_history_feature(config: &LagFeatureConfig) -> bool {
    config.lags.is_empty()
        && config.rolling_mean_windows.is_empty()
        && config.partial_rolling_mean_windows.is_empty()
        && config.rolling_std_windows.is_empty()
        && config.rolling_min_windows.is_empty()
        && config.rolling_max_windows.is_empty()
        && config.ewm_alpha_percents.is_empty()
        && config.difference_lags.is_empty()
        && config.rolling_trend_windows.is_empty()
}

fn sort_dedup_lag_config(config: &mut LagFeatureConfig) {
    sort_dedup(&mut config.lags);
    sort_dedup(&mut config.rolling_mean_windows);
    sort_dedup(&mut config.partial_rolling_mean_windows);
    sort_dedup(&mut config.rolling_std_windows);
    sort_dedup(&mut config.rolling_min_windows);
    sort_dedup(&mut config.rolling_max_windows);
    sort_dedup_u8(&mut config.ewm_alpha_percents);
    sort_dedup(&mut config.difference_lags);
    sort_dedup(&mut config.rolling_trend_windows);
}

fn sort_dedup(values: &mut Vec<usize>) {
    values.sort_unstable();
    values.dedup();
}

fn sort_dedup_u8(values: &mut Vec<u8>) {
    values.sort_unstable();
    values.dedup();
}

struct SeriesFeatureCache {
    targets: Vec<f64>,
    prefix_sum: Vec<f64>,
    ewm_prior_by_alpha: Vec<Vec<f64>>,
}

impl SeriesFeatureCache {
    fn new(history: &[ForecastRow], config: &LagFeatureConfig) -> Self {
        let targets = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let mut prefix_sum = Vec::with_capacity(targets.len() + 1);
        prefix_sum.push(0.0);
        for target in &targets {
            prefix_sum.push(prefix_sum.last().copied().unwrap_or(0.0) + *target);
        }
        let ewm_prior_by_alpha = config
            .ewm_alpha_percents
            .iter()
            .map(|alpha_percent| {
                let alpha = f64::from(*alpha_percent) / 100.0;
                let mut values = vec![f64::NAN; targets.len() + 1];
                if let Some(first) = targets.first() {
                    let mut ewm = *first;
                    values[1] = ewm;
                    for idx in 2..=targets.len() {
                        ewm = alpha * targets[idx - 1] + (1.0 - alpha) * ewm;
                        values[idx] = ewm;
                    }
                }
                values
            })
            .collect();
        Self {
            targets,
            prefix_sum,
            ewm_prior_by_alpha,
        }
    }

    fn window_sum(&self, prior_len: usize, window: usize) -> f64 {
        self.prefix_sum[prior_len] - self.prefix_sum[prior_len - window]
    }
}

pub(crate) fn history_by_series(rows: &[ForecastRow]) -> BTreeMap<String, Vec<ForecastRow>> {
    let mut history_by_series: BTreeMap<String, Vec<ForecastRow>> = BTreeMap::new();
    for row in rows {
        history_by_series
            .entry(row.series_id.clone())
            .or_default()
            .push(row.clone());
    }
    history_by_series
}

fn validate_config(config: &LagFeatureConfig) -> Result<()> {
    if config.lags.is_empty()
        && config.rolling_mean_windows.is_empty()
        && config.partial_rolling_mean_windows.is_empty()
        && config.rolling_std_windows.is_empty()
        && config.rolling_min_windows.is_empty()
        && config.rolling_max_windows.is_empty()
        && config.ewm_alpha_percents.is_empty()
        && config.difference_lags.is_empty()
        && config.rolling_trend_windows.is_empty()
        && config.calendar_features.is_empty()
        && config.covariate_features.is_empty()
        && config.covariate_indicator_values.is_empty()
    {
        return Err(CartoBoostError::InvalidInput(
            "lag feature config must contain at least one feature".to_string(),
        ));
    }
    if config.lags.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "target lags must be positive".to_string(),
        ));
    }
    if config.rolling_mean_windows.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "rolling mean windows must be positive".to_string(),
        ));
    }
    if config.partial_rolling_mean_windows.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "partial rolling mean windows must be positive".to_string(),
        ));
    }
    if config.rolling_std_windows.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "rolling standard deviation windows must be positive".to_string(),
        ));
    }
    if config.rolling_min_windows.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "rolling minimum windows must be positive".to_string(),
        ));
    }
    if config.rolling_max_windows.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "rolling maximum windows must be positive".to_string(),
        ));
    }
    if config
        .ewm_alpha_percents
        .iter()
        .any(|alpha| *alpha == 0 || *alpha > 100)
    {
        return Err(CartoBoostError::InvalidInput(
            "EWM alpha percents must be in 1..=100".to_string(),
        ));
    }
    if config.difference_lags.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "difference lags must be positive".to_string(),
        ));
    }
    if config
        .rolling_trend_windows
        .iter()
        .any(|window| *window < 2)
    {
        return Err(CartoBoostError::InvalidInput(
            "rolling trend windows must be at least 2".to_string(),
        ));
    }
    let mut elapsed_phase_count = 0;
    for feature in &config.calendar_features {
        if let CalendarFeature::ElapsedPhase(period) = feature {
            elapsed_phase_count += 1;
            if *period < 2 {
                return Err(CartoBoostError::InvalidInput(
                    "elapsed calendar phase periods must be at least 2".to_string(),
                ));
            }
        }
    }
    if elapsed_phase_count > 1 {
        return Err(CartoBoostError::InvalidInput(
            "at most one elapsed calendar phase period is supported".to_string(),
        ));
    }
    let mut covariate_names = std::collections::HashSet::new();
    for name in &config.covariate_features {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "covariate feature names must not be empty".to_string(),
            ));
        }
        if !covariate_names.insert(name) {
            return Err(CartoBoostError::InvalidInput(
                "covariate feature names must be unique".to_string(),
            ));
        }
    }
    for (name, values) in &config.covariate_indicator_values {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "covariate indicator names must not be empty".to_string(),
            ));
        }
        if values.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "covariate indicator {name:?} must include at least one value"
            )));
        }
        let mut seen = Vec::new();
        for value in values {
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "covariate indicator {name:?} values must be finite"
                )));
            }
            if seen
                .iter()
                .any(|old: &f64| (*old - *value).abs() <= COVARIATE_INDICATOR_TOLERANCE)
            {
                return Err(CartoBoostError::InvalidInput(format!(
                    "covariate indicator {name:?} values must be unique"
                )));
            }
            seen.push(*value);
        }
    }
    Ok(())
}

const COVARIATE_INDICATOR_TOLERANCE: f64 = 1.0e-12;

fn covariate_indicator_count(config: &LagFeatureConfig) -> usize {
    config
        .covariate_indicator_values
        .values()
        .map(Vec::len)
        .sum()
}

fn covariate_indicator_feature_names(config: &LagFeatureConfig) -> Vec<String> {
    config
        .covariate_indicator_values
        .iter()
        .flat_map(|(name, values)| {
            values
                .iter()
                .map(move |value| format!("covariate_{name}_is_{}", format_indicator_value(*value)))
        })
        .collect()
}

fn format_indicator_value(value: f64) -> String {
    let formatted = if value.fract().abs() <= COVARIATE_INDICATOR_TOLERANCE {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    };
    formatted.replace('-', "neg").replace('.', "p")
}

fn covariate_indicator_value(observed: f64, expected: f64) -> f64 {
    if (observed - expected).abs() <= COVARIATE_INDICATOR_TOLERANCE {
        1.0
    } else {
        0.0
    }
}

fn training_covariate_value(history: &[ForecastRow], row_idx: usize, name: &str) -> Option<f64> {
    history[row_idx]
        .covariates
        .get(name)
        .or_else(|| {
            row_idx
                .checked_sub(1)
                .and_then(|idx| history[idx].covariates.get(name))
        })
        .copied()
}

fn calendar_feature_name(feature: &CalendarFeature) -> String {
    match feature {
        CalendarFeature::DayOfWeek => "calendar_day_of_week".to_string(),
        CalendarFeature::DayOfWeekSin => "calendar_day_of_week_sin".to_string(),
        CalendarFeature::DayOfWeekCos => "calendar_day_of_week_cos".to_string(),
        CalendarFeature::Month => "calendar_month".to_string(),
        CalendarFeature::MonthSin => "calendar_month_sin".to_string(),
        CalendarFeature::MonthCos => "calendar_month_cos".to_string(),
        CalendarFeature::Day => "calendar_day".to_string(),
        CalendarFeature::DaySin => "calendar_day_sin".to_string(),
        CalendarFeature::DayCos => "calendar_day_cos".to_string(),
        CalendarFeature::MonthStart => "calendar_month_start".to_string(),
        CalendarFeature::MonthMiddle => "calendar_month_middle".to_string(),
        CalendarFeature::MonthEnd => "calendar_month_end".to_string(),
        CalendarFeature::DayOfYear => "calendar_day_of_year".to_string(),
        CalendarFeature::ElapsedIndex => "calendar_elapsed_index".to_string(),
        CalendarFeature::ElapsedPhase(_) => "calendar_elapsed_phase".to_string(),
    }
}

fn calendar_feature_allows_covariate_interaction(feature: &CalendarFeature) -> bool {
    match feature {
        CalendarFeature::DayOfWeek
        | CalendarFeature::DayOfWeekSin
        | CalendarFeature::DayOfWeekCos
        | CalendarFeature::Month
        | CalendarFeature::MonthSin
        | CalendarFeature::MonthCos
        | CalendarFeature::Day
        | CalendarFeature::DaySin
        | CalendarFeature::DayCos
        | CalendarFeature::MonthStart
        | CalendarFeature::MonthMiddle
        | CalendarFeature::MonthEnd
        | CalendarFeature::DayOfYear
        | CalendarFeature::ElapsedIndex
        | CalendarFeature::ElapsedPhase(_) => true,
    }
}

fn calendar_feature_value(
    feature: &CalendarFeature,
    timestamp: NaiveDateTime,
    prior_len: usize,
) -> f64 {
    match feature {
        CalendarFeature::DayOfWeek => f64::from(timestamp.weekday().num_days_from_monday()),
        CalendarFeature::DayOfWeekSin => {
            cyclic_sin(f64::from(timestamp.weekday().num_days_from_monday()), 7.0)
        }
        CalendarFeature::DayOfWeekCos => {
            cyclic_cos(f64::from(timestamp.weekday().num_days_from_monday()), 7.0)
        }
        CalendarFeature::Month => f64::from(timestamp.month()),
        CalendarFeature::MonthSin => cyclic_sin(f64::from(timestamp.month0()), 12.0),
        CalendarFeature::MonthCos => cyclic_cos(f64::from(timestamp.month0()), 12.0),
        CalendarFeature::Day => f64::from(timestamp.day()),
        CalendarFeature::DaySin => cyclic_sin(f64::from(timestamp.day0()), 31.0),
        CalendarFeature::DayCos => cyclic_cos(f64::from(timestamp.day0()), 31.0),
        CalendarFeature::MonthStart => {
            if timestamp.day() <= 3 {
                1.0
            } else {
                0.0
            }
        }
        CalendarFeature::MonthMiddle => {
            if (14..=16).contains(&timestamp.day()) {
                1.0
            } else {
                0.0
            }
        }
        CalendarFeature::MonthEnd => {
            let Some(days_in_month) = days_in_month(timestamp) else {
                return 0.0;
            };
            if timestamp.day() + 2 >= days_in_month {
                1.0
            } else {
                0.0
            }
        }
        CalendarFeature::DayOfYear => f64::from(timestamp.ordinal()),
        CalendarFeature::ElapsedIndex => prior_len as f64,
        CalendarFeature::ElapsedPhase(period) => (prior_len % *period) as f64,
    }
}

fn cyclic_sin(position: f64, period: f64) -> f64 {
    (std::f64::consts::TAU * position / period).sin()
}

fn cyclic_cos(position: f64, period: f64) -> f64 {
    (std::f64::consts::TAU * position / period).cos()
}

fn days_in_month(timestamp: NaiveDateTime) -> Option<u32> {
    let (next_year, next_month) = if timestamp.month() == 12 {
        (timestamp.year() + 1, 1)
    } else {
        (timestamp.year(), timestamp.month() + 1)
    };
    chrono::NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .and_then(|first_next| first_next.pred_opt())
        .map(|last_this| last_this.day())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::parse_forecast_timestamp;

    fn ts(day: u32) -> NaiveDateTime {
        parse_forecast_timestamp(&format!("2026-01-{day:02}")).expect("timestamp")
    }

    #[test]
    fn delta_and_trend_features_use_only_prior_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
                ForecastRow::single(ts(3), 15.0),
                ForecastRow::single(ts(4), 19.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![3],
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![50],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let rows = builder.transform_frame(&frame).expect("features");
        let last = rows.last().expect("last row");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "target_roll_mean_2".to_string(),
                "target_roll_std_3".to_string(),
                "target_ewm_alpha_050".to_string(),
                "target_delta_lag_2".to_string(),
                "target_roll_trend_3".to_string(),
            ]
        );
        assert_eq!(last.timestamp, ts(4));
        assert_eq!(last.features[0], 15.0);
        assert_eq!(last.features[1], 13.5);
        assert!((last.features[2] - 2.0548046676563256).abs() < 1e-12);
        assert_eq!(last.features[3], 13.0);
        assert_eq!(last.features[4], 5.0);
        assert_eq!(last.features[5], 2.5);

        let next = builder
            .transform_next(&rows[0].series_id, frame.rows(), ts(5))
            .expect("next features");
        assert_eq!(next[3], 16.0);
    }

    #[test]
    fn partial_rolling_mean_uses_available_prior_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 16.0),
                ForecastRow::single(ts(3), 22.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: vec![5],
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
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &["target_partial_roll_mean_5".to_string()]
        );
        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].timestamp, ts(2));
        assert_eq!(rows[0].features, vec![10.0]);
        assert_eq!(rows[1].timestamp, ts(3));
        assert_eq!(rows[1].features, vec![13.0]);

        let next = builder
            .transform_next(&rows[0].series_id, frame.rows(), ts(4))
            .expect("next features");
        assert_eq!(next, vec![16.0]);
    }

    #[test]
    fn trend_and_ewm_feature_windows_are_validated() {
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![0],
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: vec![0],
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
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![0],
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: vec![0],
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: vec![0],
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: vec![1],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![0],
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![101],
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
    }

    #[test]
    fn sorted_prior_next_features_match_public_sorted_transform() {
        let history = vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 12.0),
            ForecastRow::single(ts(3), 16.0),
            ForecastRow::single(ts(4), 20.0),
        ];
        let unsorted_history = vec![
            history[2].clone(),
            history[0].clone(),
            history[3].clone(),
            history[1].clone(),
        ];
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![50],
            calendar_features: vec![CalendarFeature::ElapsedIndex],
            difference_lags: vec![2],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let public = builder
            .transform_next(&history[0].series_id, &unsorted_history, ts(5))
            .expect("public transform");
        let sorted = builder
            .transform_next_sorted_prior(&history[0].series_id, &history, ts(5))
            .expect("sorted transform");

        assert_eq!(sorted, public);
        assert!(builder
            .transform_next_sorted_prior(&history[0].series_id, &unsorted_history, ts(5))
            .is_err());
    }

    #[test]
    fn sorted_prior_next_features_use_known_future_covariates() {
        let mut rows = Vec::new();
        for day in 1..=4 {
            let mut covariates = BTreeMap::new();
            covariates.insert("promo".to_string(), f64::from(day));
            rows.push(ForecastRow::with_covariates(
                "store_item",
                ts(day),
                f64::from(day * 10),
                covariates,
            ));
        }
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["promo".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");
        let mut future_covariates = BTreeMap::new();
        future_covariates.insert("promo".to_string(), 99.0);

        let stale = builder
            .transform_next_sorted_prior("store_item", &rows, ts(5))
            .expect("stale features");
        let known = builder
            .transform_next_sorted_prior_with_covariates(
                "store_item",
                &rows,
                ts(5),
                Some(&future_covariates),
            )
            .expect("known future features");

        assert_eq!(
            builder.feature_names(),
            &["target_lag_1", "covariate_promo"]
        );
        assert_eq!(stale, vec![40.0, 4.0]);
        assert_eq!(known, vec![40.0, 99.0]);
    }

    #[test]
    fn cached_training_features_match_position_builder() {
        let mut rows = Vec::new();
        for day in 1..=8 {
            let mut covariates = BTreeMap::new();
            covariates.insert("distance_miles".to_string(), f64::from(day) * 1.5);
            rows.push(ForecastRow::with_covariates(
                "lane_a",
                ts(day),
                f64::from(day * day + 3),
                covariates,
            ));
        }
        let frame = ForecastFrame::new(rows.clone(), crate::forecasting::ForecastFrequency::Daily)
            .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1, 2, 4],
            rolling_mean_windows: vec![2, 4],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![3],
            rolling_min_windows: vec![3],
            rolling_max_windows: vec![3],
            ewm_alpha_percents: vec![25, 90],
            calendar_features: vec![
                CalendarFeature::DayOfYear,
                CalendarFeature::ElapsedIndex,
                CalendarFeature::ElapsedPhase(14),
            ],
            difference_lags: vec![2],
            rolling_trend_windows: vec![4],
            covariate_features: vec!["distance_miles".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let cached_rows = builder.transform_frame(&frame).expect("cached");
        let manual_rows = (0..rows.len())
            .filter_map(|idx| {
                builder
                    .features_for_position(&rows, idx)
                    .expect("manual features")
                    .map(|features| LagFeatureRow {
                        series_id: "lane_a".to_string(),
                        timestamp: rows[idx].timestamp,
                        target: rows[idx].target,
                        features,
                    })
            })
            .collect::<Vec<_>>();

        assert_eq!(cached_rows.len(), manual_rows.len());
        for (cached, manual) in cached_rows.iter().zip(manual_rows.iter()) {
            assert_eq!(cached.series_id, manual.series_id);
            assert_eq!(cached.timestamp, manual.timestamp);
            assert_eq!(cached.target, manual.target);
            assert_eq!(cached.features.len(), manual.features.len());
            for (left, right) in cached.features.iter().zip(manual.features.iter()) {
                assert!((left - right).abs() < 1e-10, "{left} != {right}");
            }
        }
    }

    #[test]
    fn covariate_features_use_current_row_for_training_and_latest_for_prediction() {
        let mut first_covariates = BTreeMap::new();
        first_covariates.insert("distance_miles".to_string(), 2.5);
        let mut second_covariates = BTreeMap::new();
        second_covariates.insert("distance_miles".to_string(), 2.5);
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::with_covariates("lane_a", ts(1), 10.0, first_covariates),
                ForecastRow::with_covariates("lane_a", ts(2), 12.0, second_covariates),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![
                CalendarFeature::DayOfYear,
                CalendarFeature::ElapsedIndex,
                CalendarFeature::ElapsedPhase(14),
            ],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["distance_miles".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day_of_year".to_string(),
                "calendar_elapsed_index".to_string(),
                "calendar_elapsed_phase".to_string(),
                "covariate_distance_miles".to_string(),
            ]
        );
        assert_eq!(rows[0].features, vec![10.0, 2.0, 1.0, 1.0, 2.5]);

        let next = builder
            .transform_next("lane_a", frame.rows(), ts(3))
            .expect("next");
        assert_eq!(next, vec![12.0, 3.0, 2.0, 2.0, 2.5]);
    }

    #[test]
    fn elapsed_phase_calendar_feature_uses_configured_period() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("lane_a", ts(1), 10.0),
                ForecastRow::new("lane_a", ts(2), 12.0),
                ForecastRow::new("lane_a", ts(3), 14.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::ElapsedPhase(7)],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_elapsed_phase".to_string(),
            ]
        );
        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(rows[0].features, vec![10.0, 1.0]);
        let next = builder
            .transform_next("lane_a", frame.rows(), ts(4))
            .expect("next");
        assert_eq!(next, vec![14.0, 3.0]);

        let invalid = LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::ElapsedPhase(1)],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        });
        assert!(invalid.is_err());
    }

    #[test]
    fn covariate_calendar_interactions_are_leakage_safe() {
        let mut first_covariates = BTreeMap::new();
        first_covariates.insert("airport_lane".to_string(), 1.0);
        let mut second_covariates = BTreeMap::new();
        second_covariates.insert("airport_lane".to_string(), 1.0);
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::with_covariates("lane_a", ts(1), 10.0, first_covariates),
                ForecastRow::with_covariates("lane_a", ts(2), 12.0, second_covariates),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::Day, CalendarFeature::ElapsedPhase(14)],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["airport_lane".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: true,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day".to_string(),
                "calendar_elapsed_phase".to_string(),
                "covariate_airport_lane".to_string(),
                "covariate_airport_lane_x_calendar_day".to_string(),
                "covariate_airport_lane_x_calendar_elapsed_phase".to_string(),
            ]
        );
        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(rows[0].features, vec![10.0, 2.0, 1.0, 1.0, 2.0, 1.0]);

        let next = builder
            .transform_next("lane_a", frame.rows(), ts(3))
            .expect("next");
        assert_eq!(next, vec![12.0, 3.0, 2.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn covariate_indicators_encode_low_cardinality_context() {
        let mut first_covariates = BTreeMap::new();
        first_covariates.insert("pickup_borough_code".to_string(), 3.0);
        let mut second_covariates = BTreeMap::new();
        second_covariates.insert("pickup_borough_code".to_string(), 3.0);
        let history = vec![
            ForecastRow::with_covariates("lane_a", ts(1), 10.0, first_covariates),
            ForecastRow::with_covariates("lane_a", ts(2), 12.0, second_covariates),
        ];
        let mut indicator_values = BTreeMap::new();
        indicator_values.insert("pickup_borough_code".to_string(), vec![1.0, 3.0]);
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::Day],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: indicator_values,
            covariate_calendar_interactions: true,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day".to_string(),
                "covariate_pickup_borough_code_is_1".to_string(),
                "covariate_pickup_borough_code_is_3".to_string(),
                "covariate_pickup_borough_code_is_1_x_calendar_day".to_string(),
                "covariate_pickup_borough_code_is_3_x_calendar_day".to_string(),
            ]
        );
        let features = builder
            .features_for_position(&history, 1)
            .expect("features")
            .expect("enough history");
        assert_eq!(features, vec![10.0, 2.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn calendar_fourier_features_encode_cycles_without_extra_history() {
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![
                CalendarFeature::DayOfWeekSin,
                CalendarFeature::DayOfWeekCos,
                CalendarFeature::MonthSin,
                CalendarFeature::MonthCos,
                CalendarFeature::DaySin,
                CalendarFeature::DayCos,
                CalendarFeature::MonthStart,
                CalendarFeature::MonthMiddle,
                CalendarFeature::MonthEnd,
            ],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "calendar_day_of_week_sin".to_string(),
                "calendar_day_of_week_cos".to_string(),
                "calendar_month_sin".to_string(),
                "calendar_month_cos".to_string(),
                "calendar_day_sin".to_string(),
                "calendar_day_cos".to_string(),
                "calendar_month_start".to_string(),
                "calendar_month_middle".to_string(),
                "calendar_month_end".to_string(),
            ]
        );
        let monday_in_january = parse_forecast_timestamp("2026-01-05").expect("timestamp");
        let history = [ForecastRow::single(ts(4), 10.0)];
        let features = builder
            .transform_next("__single__", &history, monday_in_january)
            .expect("features");
        assert!((features[0] - 0.0).abs() < 1e-12);
        assert!((features[1] - 1.0).abs() < 1e-12);
        assert!((features[2] - 0.0).abs() < 1e-12);
        assert!((features[3] - 1.0).abs() < 1e-12);
        assert!((features[4] - cyclic_sin(4.0, 31.0)).abs() < 1e-12);
        assert!((features[5] - cyclic_cos(4.0, 31.0)).abs() < 1e-12);
        assert_eq!(features[6], 0.0);
        assert_eq!(features[7], 0.0);
        assert_eq!(features[8], 0.0);

        let month_end = parse_forecast_timestamp("2026-01-31").expect("timestamp");
        let month_end_features = builder
            .transform_next("__single__", &history, month_end)
            .expect("features");
        assert_eq!(month_end_features[6], 0.0);
        assert_eq!(month_end_features[7], 0.0);
        assert_eq!(month_end_features[8], 1.0);
    }

    #[test]
    fn calendar_event_flags_expand_covariate_interactions() {
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![
                CalendarFeature::DayOfWeek,
                CalendarFeature::MonthStart,
                CalendarFeature::MonthEnd,
            ],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["airport_lane".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: true,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day_of_week".to_string(),
                "calendar_month_start".to_string(),
                "calendar_month_end".to_string(),
                "covariate_airport_lane".to_string(),
                "covariate_airport_lane_x_calendar_day_of_week".to_string(),
                "covariate_airport_lane_x_calendar_month_start".to_string(),
                "covariate_airport_lane_x_calendar_month_end".to_string(),
            ]
        );
    }
}
