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

