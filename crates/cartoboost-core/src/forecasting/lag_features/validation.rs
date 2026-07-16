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

