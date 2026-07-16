fn validate_theta_params(theta: f64, alpha: f64) -> Result<()> {
    if !theta.is_finite() || theta <= 0.0 || !(1.0 - 1.0 / theta).is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "theta must be a positive finite value with a finite reciprocal".to_string(),
        ));
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "alpha must be finite and in (0, 1]".to_string(),
        ));
    }
    Ok(())
}

fn validate_piecewise_linear_seasonal_config(config: &PiecewiseLinearSeasonalConfig) -> Result<()> {
    if !config.changepoint_range.is_finite()
        || config.changepoint_range <= 0.0
        || config.changepoint_range > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_range must be finite and in (0, 1]".to_string(),
        ));
    }
    if !config.changepoint_l2_regularization.is_finite()
        || config.changepoint_l2_regularization < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.changepoint_l1_regularization.is_finite()
        || config.changepoint_l1_regularization < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_l1_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.huber_delta.is_finite() || config.huber_delta <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "huber_delta must be a finite positive value".to_string(),
        ));
    }
    if !config.seasonality_l2_regularization.is_finite()
        || config.seasonality_l2_regularization < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "seasonality_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    validate_optional_nonnegative(config.yearly_l2_regularization, "yearly_l2_regularization")?;
    validate_optional_nonnegative(config.weekly_l2_regularization, "weekly_l2_regularization")?;
    validate_optional_nonnegative(config.daily_l2_regularization, "daily_l2_regularization")?;
    if !config.event_l2_regularization.is_finite() || config.event_l2_regularization < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "event_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.regressor_l2_regularization.is_finite() || config.regressor_l2_regularization < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "regressor_l2_regularization must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.trend_uncertainty_scale.is_finite() || config.trend_uncertainty_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "trend_uncertainty_scale must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.coefficient_uncertainty_scale.is_finite()
        || config.coefficient_uncertainty_scale < 0.0
    {
        return Err(CartoBoostError::InvalidInput(
            "coefficient_uncertainty_scale must be a finite nonnegative value".to_string(),
        ));
    }
    let mut changepoint_timestamps = config.changepoint_timestamps.clone();
    changepoint_timestamps.sort();
    if changepoint_timestamps
        .windows(2)
        .any(|window| window[0] == window[1])
    {
        return Err(CartoBoostError::InvalidInput(
            "changepoint_timestamps must be unique".to_string(),
        ));
    }
    let mut seasonality_names = BTreeSet::new();
    for seasonality in &config.custom_seasonalities {
        if seasonality.name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "custom seasonality names must not be empty".to_string(),
            ));
        }
        if !seasonality_names.insert(seasonality.name.clone()) {
            return Err(CartoBoostError::InvalidInput(
                "custom seasonality names must be unique".to_string(),
            ));
        }
        if !seasonality.period_days.is_finite() || seasonality.period_days <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "custom seasonality {:?} period_days must be a positive finite value",
                seasonality.name
            )));
        }
        if seasonality.fourier_order == 0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "custom seasonality {:?} fourier_order must be positive",
                seasonality.name
            )));
        }
        if let Some(condition_name) = &seasonality.condition_name {
            if condition_name.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "custom seasonality {:?} condition_name must not be empty",
                    seasonality.name
                )));
            }
        }
        if let Some(value) = seasonality.l2_regularization {
            if !value.is_finite() || value < 0.0 {
                return Err(CartoBoostError::InvalidInput(format!(
                    "custom seasonality {:?} l2_regularization must be a finite nonnegative value",
                    seasonality.name
                )));
            }
        }
    }
    for event in &config.events {
        if event.name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "piecewise seasonal event names must not be empty".to_string(),
            ));
        }
        if event.lower_window > event.upper_window {
            return Err(CartoBoostError::InvalidInput(format!(
                "piecewise seasonal event {:?} lower_window must be <= upper_window",
                event.name
            )));
        }
    }
    let mut regressors = config.extra_regressors.clone();
    regressors.sort();
    for name in &regressors {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "extra regressor names must not be empty".to_string(),
            ));
        }
    }
    if regressors.windows(2).any(|window| window[0] == window[1]) {
        return Err(CartoBoostError::InvalidInput(
            "extra regressor names must be unique".to_string(),
        ));
    }
    for name in config.regressor_modes.keys() {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "regressor mode names must not be empty".to_string(),
            ));
        }
        if !regressors.iter().any(|regressor| regressor == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "regressor mode {name:?} does not match an extra regressor"
            )));
        }
    }
    for (name, direction) in &config.extra_regressor_monotonic_constraints {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "extra regressor monotonic constraint names must not be empty".to_string(),
            ));
        }
        if !regressors.iter().any(|regressor| regressor == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "extra regressor monotonic constraint {name:?} does not match an extra regressor"
            )));
        }
        if !matches!(*direction, -1..=1) {
            return Err(CartoBoostError::InvalidInput(
                "extra regressor monotonic constraints must be -1, 0, or 1".to_string(),
            ));
        }
    }
    for (name, values) in &config.future_regressors {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "future regressor names must not be empty".to_string(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "future regressor {name:?} values must be finite"
            )));
        }
    }
    for (series_id, regressors_by_name) in &config.future_regressors_by_series {
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "future regressor series ids must not be empty".to_string(),
            ));
        }
        for (name, values) in regressors_by_name {
            if name.is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "per-series future regressor names must not be empty".to_string(),
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "per-series future regressor {name:?} for series {series_id:?} values must be finite"
                )));
            }
        }
    }
    validate_piecewise_trend_adjustments(&config.trend_adjustments, "trend_adjustments")?;
    for (series_id, adjustments) in &config.trend_adjustments_by_series {
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "trend adjustment series ids must not be empty".to_string(),
            ));
        }
        validate_piecewise_trend_adjustments(
            adjustments,
            &format!("trend_adjustments_by_series[{series_id:?}]"),
        )?;
    }
    if !config.residual_shock_scale.is_finite() || config.residual_shock_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "residual_shock_scale must be a finite nonnegative value".to_string(),
        ));
    }
    if !config.residual_shock_decay.is_finite()
        || config.residual_shock_decay < 0.0
        || config.residual_shock_decay > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "residual_shock_decay must be finite and in [0, 1]".to_string(),
        ));
    }
    let event_names = piecewise_event_names(config);
    for (name, value) in &config.event_l2_regularization_by_name {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "event l2 regularization names must not be empty".to_string(),
            ));
        }
        if !event_names.iter().any(|event_name| event_name == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "event l2 regularization {name:?} does not match a configured event"
            )));
        }
        if !value.is_finite() || *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "event l2 regularization {name:?} must be a finite nonnegative value"
            )));
        }
    }
    for (name, value) in &config.regressor_l2_regularization_by_name {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "regressor l2 regularization names must not be empty".to_string(),
            ));
        }
        if !regressors.iter().any(|regressor| regressor == name) {
            return Err(CartoBoostError::InvalidInput(format!(
                "regressor l2 regularization {name:?} does not match an extra regressor"
            )));
        }
        if !value.is_finite() || *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "regressor l2 regularization {name:?} must be a finite nonnegative value"
            )));
        }
    }
    for level in &config.interval_levels {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(CartoBoostError::InvalidInput(
                "prediction interval levels must be finite and in (0, 1)".to_string(),
            ));
        }
    }
    let mut interval_levels = config.interval_levels.clone();
    interval_levels.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("interval levels are finite after validation")
    });
    if interval_levels
        .windows(2)
        .any(|window| (window[0] - window[1]).abs() < 1.0e-12)
    {
        return Err(CartoBoostError::InvalidInput(
            "prediction interval levels must be unique".to_string(),
        ));
    }
    validate_piecewise_quantile_levels(&config.quantile_levels)?;
    if !config.floor.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal floor must be finite".to_string(),
        ));
    }
    if let Some(name) = &config.cap_regressor {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "cap_regressor must not be empty".to_string(),
            ));
        }
    }
    if let Some(name) = &config.floor_regressor {
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "floor_regressor must not be empty".to_string(),
            ));
        }
    }
    if let PiecewiseLinearGrowth::Logistic = config.growth {
        if config.cap.is_none() && config.cap_regressor.is_none() {
            return Err(CartoBoostError::InvalidInput(
                "logistic piecewise seasonal growth requires cap or cap_regressor".to_string(),
            ));
        }
        if let Some(cap) = config.cap {
            if !cap.is_finite() || cap <= config.floor {
                return Err(CartoBoostError::InvalidInput(
                    "logistic piecewise seasonal cap must be finite and greater than floor"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_piecewise_trend_adjustments(
    adjustments: &BTreeMap<usize, f64>,
    name: &str,
) -> Result<()> {
    for (horizon, multiplier) in adjustments {
        if *horizon == 0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} horizon keys must be positive"
            )));
        }
        if !multiplier.is_finite() || *multiplier <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} multipliers must be positive finite values"
            )));
        }
    }
    Ok(())
}

fn validate_optional_nonnegative(value: Option<f64>, name: &str) -> Result<()> {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} must be a finite nonnegative value"
            )));
        }
    }
    Ok(())
}

fn validate_piecewise_quantile_levels(levels: &[f64]) -> Result<()> {
    for level in levels {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(CartoBoostError::InvalidInput(
                "quantile levels must be finite and in (0, 1)".to_string(),
            ));
        }
    }
    let mut sorted = levels.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted
        .windows(2)
        .any(|window| (window[0] - window[1]).abs() < 1.0e-12)
    {
        return Err(CartoBoostError::InvalidInput(
            "quantile levels must be unique".to_string(),
        ));
    }
    Ok(())
}

fn resolve_piecewise_auto_seasonalities(
    frame: &ForecastFrame,
    config: &PiecewiseLinearSeasonalConfig,
) -> PiecewiseLinearSeasonalConfig {
    let mut resolved = config.clone();
    let Some((start, end)) = forecast_frame_timestamp_span(frame) else {
        return resolved;
    };
    let span_days = (end - start).num_seconds() as f64 / 86_400.0;
    if resolved.auto_yearly_seasonality && resolved.yearly_fourier_order == 0 && span_days >= 730.0
    {
        resolved.yearly_fourier_order = 10;
    }
    if resolved.auto_weekly_seasonality
        && resolved.weekly_fourier_order == 0
        && matches!(
            frame.frequency(),
            ForecastFrequency::Daily | ForecastFrequency::Hourly
        )
        && span_days >= 14.0
    {
        resolved.weekly_fourier_order = 3;
    }
    if resolved.auto_daily_seasonality
        && resolved.daily_fourier_order == 0
        && frame.frequency() == ForecastFrequency::Hourly
        && span_days >= 2.0
    {
        resolved.daily_fourier_order = 4;
    }
    resolved
}

fn forecast_frame_timestamp_span(
    frame: &ForecastFrame,
) -> Option<(chrono::NaiveDateTime, chrono::NaiveDateTime)> {
    let mut iter = frame.rows().iter().map(|row| row.timestamp);
    let first = iter.next()?;
    let (min_timestamp, max_timestamp) = iter.fold(
        (first, first),
        |(min_timestamp, max_timestamp), timestamp| {
            (min_timestamp.min(timestamp), max_timestamp.max(timestamp))
        },
    );
    Some((min_timestamp, max_timestamp))
}

fn elapsed_days(start_timestamp: chrono::NaiveDateTime, timestamp: chrono::NaiveDateTime) -> f64 {
    (timestamp - start_timestamp).num_seconds() as f64 / 86_400.0
}

fn select_piecewise_changepoints(
    series_id: &str,
    start_timestamp: chrono::NaiveDateTime,
    last_timestamp: chrono::NaiveDateTime,
    elapsed: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Vec<f64>> {
    if !config.changepoint_timestamps.is_empty() {
        let mut changepoints = config
            .changepoint_timestamps
            .iter()
            .map(|&timestamp| {
                if timestamp <= start_timestamp || timestamp >= last_timestamp {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "changepoint timestamp {timestamp} must be inside the training range for series {series_id:?}"
                    )));
                }
                Ok(elapsed_days(start_timestamp, timestamp))
            })
            .collect::<Result<Vec<_>>>()?;
        changepoints.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("explicit changepoint elapsed values are finite")
        });
        return Ok(changepoints);
    }
    Ok(select_even_changepoints(
        elapsed,
        config.changepoints,
        config.changepoint_range,
    ))
}

fn select_even_changepoints(elapsed: &[f64], requested: usize, changepoint_range: f64) -> Vec<f64> {
    if requested == 0 || elapsed.len() <= 2 {
        return Vec::new();
    }
    let last_idx = elapsed.len() - 1;
    let cutoff_idx = ((last_idx as f64) * changepoint_range).floor().max(1.0) as usize;
    let cutoff_idx = cutoff_idx.min(last_idx.saturating_sub(1));
    let count = requested.min(cutoff_idx.saturating_sub(1).max(1));
    (1..=count)
        .map(|idx| {
            let position = (idx * cutoff_idx) / (count + 1);
            elapsed[position.max(1).min(cutoff_idx)]
        })
        .collect()
}

fn fit_piecewise_trend_coefficients(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Vec<f64>> {
    if !piecewise_uses_multiplicative_components(config) {
        return Ok(Vec::new());
    }
    let feature_count = 2 + changepoints.len();
    let mut xtx = vec![vec![0.0; feature_count]; feature_count];
    let mut xty = vec![0.0; feature_count];
    let mut features = vec![0.0; feature_count];
    for (row, &t) in history.iter().zip(elapsed.iter()) {
        let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
        let target = transform_piecewise_target(row.target, bounds, config)?;
        fill_piecewise_trend_features(&mut features, t, changepoints, config);
        for i in 0..feature_count {
            xty[i] += features[i] * target;
            for j in i..feature_count {
                xtx[i][j] += features[i] * features[j];
            }
        }
    }
    for i in 0..feature_count {
        let (previous_rows, current_and_after) = xtx.split_at_mut(i);
        let current_row = &mut current_and_after[0];
        for (j, previous_row) in previous_rows.iter().enumerate() {
            current_row[j] = previous_row[i];
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        let _ = idx;
        row[idx] += config.changepoint_l2_regularization;
    }
    solve_piecewise_linear_coefficients(xtx, xty, changepoints.len(), config).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "could not solve piecewise linear seasonal trend prefit normal equations".to_string(),
        )
    })
}

fn piecewise_uses_multiplicative_components(config: &PiecewiseLinearSeasonalConfig) -> bool {
    config.component_mode == PiecewiseLinearComponentMode::Multiplicative
        || config.custom_seasonalities.iter().any(|seasonality| {
            seasonality.mode == Some(PiecewiseLinearComponentMode::Multiplicative)
        })
        || config.event_mode == Some(PiecewiseLinearComponentMode::Multiplicative)
        || config
            .regressor_modes
            .values()
            .any(|mode| *mode == PiecewiseLinearComponentMode::Multiplicative)
}

fn fill_piecewise_trend_features(
    features: &mut [f64],
    elapsed_days: f64,
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) {
    features.fill(0.0);
    features[0] = 1.0;
    if config.growth == PiecewiseLinearGrowth::Flat {
        return;
    }
    features[1] = elapsed_days;
    for (idx, &changepoint) in changepoints.iter().enumerate() {
        features[2 + idx] = (elapsed_days - changepoint).max(0.0);
    }
}

fn fit_component_multiplier(
    elapsed_days: f64,
    coefficients: &[f64],
    changepoints: &[f64],
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    match config.component_mode {
        PiecewiseLinearComponentMode::Additive => 1.0,
        PiecewiseLinearComponentMode::Multiplicative => piecewise_component_multiplier_from_trend(
            piecewise_trend_value(elapsed_days, coefficients, changepoints, config),
            bounds,
            config,
        ),
    }
}

fn piecewise_component_multiplier_from_trend(
    trend_linear: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    let trend = match config.growth {
        PiecewiseLinearGrowth::Logistic => inverse_piecewise_target(trend_linear, bounds, config),
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => trend_linear,
    };
    trend.abs().max(1.0e-9)
}

fn piecewise_trend_value(
    elapsed_days: f64,
    coefficients: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    let mut value = coefficients.first().copied().unwrap_or(0.0)
        + if config.growth == PiecewiseLinearGrowth::Flat {
            0.0
        } else {
            coefficients.get(1).copied().unwrap_or(0.0) * elapsed_days
        };
    if config.growth == PiecewiseLinearGrowth::Flat {
        return value;
    }
    for (idx, &changepoint) in changepoints.iter().enumerate() {
        value += coefficients.get(2 + idx).copied().unwrap_or(0.0)
            * (elapsed_days - changepoint).max(0.0);
    }
    value
}

fn piecewise_linear_seasonal_feature_count(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> usize {
    2 + changepoint_count
        + 2 * config.yearly_fourier_order
        + 2 * config.weekly_fourier_order
        + 2 * config.daily_fourier_order
        + 2 * config
            .custom_seasonalities
            .iter()
            .map(|seasonality| seasonality.fourier_order)
            .sum::<usize>()
        + piecewise_event_terms(config).len()
        + config.extra_regressors.len()
}

fn fill_piecewise_linear_seasonal_features(
    features: &mut [f64],
    elapsed_days: f64,
    context: &PiecewiseLinearFeatureContext<'_>,
) -> Result<()> {
    features.fill(0.0);
    features[0] = 1.0;
    features[1] = elapsed_days;
    let mut col = 2;
    if context.config.growth == PiecewiseLinearGrowth::Flat {
        features[1] = 0.0;
    }
    for &changepoint in context.changepoints {
        features[col] = if context.config.growth == PiecewiseLinearGrowth::Flat {
            0.0
        } else {
            (elapsed_days - changepoint).max(0.0)
        };
        col += 1;
    }
    let seasonal_start = col;
    col = fill_fourier_terms(
        features,
        col,
        elapsed_days,
        365.25,
        context.config.yearly_fourier_order,
    );
    col = fill_fourier_terms(
        features,
        col,
        elapsed_days,
        7.0,
        context.config.weekly_fourier_order,
    );
    col = fill_fourier_terms(
        features,
        col,
        elapsed_days,
        1.0,
        context.config.daily_fourier_order,
    );
    apply_component_mode_to_features(
        features,
        seasonal_start,
        col,
        context.component_multiplier,
        context.config.component_mode,
    );
    for seasonality in &context.config.custom_seasonalities {
        let seasonality_start = col;
        col = fill_fourier_terms(
            features,
            col,
            elapsed_days,
            seasonality.period_days,
            seasonality.fourier_order,
        );
        apply_component_mode_to_features(
            features,
            seasonality_start,
            col,
            context.component_multiplier,
            seasonality.mode.unwrap_or(context.config.component_mode),
        );
        if !piecewise_seasonality_condition_is_active(seasonality, context)? {
            for feature in features.iter_mut().take(col).skip(seasonality_start) {
                *feature = 0.0;
            }
        }
    }
    for term in piecewise_event_terms(context.config) {
        features[col] = if event_term_is_active(context.timestamp, &term, &context.config.events) {
            component_multiplier_for_mode(
                context
                    .config
                    .event_mode
                    .unwrap_or(context.config.component_mode),
                context.component_multiplier,
            )
        } else {
            0.0
        };
        col += 1;
    }
    for name in &context.config.extra_regressors {
        let mode = context
            .config
            .regressor_modes
            .get(name)
            .copied()
            .unwrap_or(context.config.component_mode);
        features[col] = component_multiplier_for_mode(mode, context.component_multiplier)
            * piecewise_extra_regressor_value(
                name,
                context.series_id,
                context.covariates,
                context.horizon_step,
                context.config,
                context.regressor_stats,
            )?;
        col += 1;
    }
    Ok(())
}

fn apply_component_mode_to_features(
    features: &mut [f64],
    start: usize,
    end: usize,
    component_multiplier: f64,
    mode: PiecewiseLinearComponentMode,
) {
    let multiplier = component_multiplier_for_mode(mode, component_multiplier);
    for feature in features.iter_mut().take(end).skip(start) {
        *feature *= multiplier;
    }
}

fn component_multiplier_for_mode(mode: PiecewiseLinearComponentMode, multiplier: f64) -> f64 {
    match mode {
        PiecewiseLinearComponentMode::Additive => 1.0,
        PiecewiseLinearComponentMode::Multiplicative => multiplier,
    }
}

fn fill_fourier_terms(
    features: &mut [f64],
    mut col: usize,
    elapsed_days: f64,
    period: f64,
    order: usize,
) -> usize {
    for harmonic in 1..=order {
        let angle = std::f64::consts::TAU * harmonic as f64 * elapsed_days / period;
        features[col] = angle.sin();
        features[col + 1] = angle.cos();
        col += 2;
    }
    col
}

fn apply_piecewise_linear_ridge(
    xtx: &mut [Vec<f64>],
    changepoint_count: usize,
    config: &PiecewiseLinearSeasonalConfig,
) {
    let penalties = piecewise_linear_l2_penalties(config, changepoint_count);
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        let penalty = penalties.get(idx).copied().unwrap_or(0.0);
        row[idx] += penalty;
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_piecewise_linear_weighted_coefficients(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: &BTreeMap<String, PiecewiseLinearRegressorStats>,
    trend_coefficients: &[f64],
    weights: Option<&[f64]>,
    compute_covariance: bool,
) -> Result<Option<PiecewiseLinearFitResult>> {
    if let Some(weights) = weights {
        if weights.len() != history.len() {
            return Err(CartoBoostError::InvalidInput(
                "piecewise robust fit received mismatched weight count".to_string(),
            ));
        }
    }
    let feature_count = piecewise_linear_seasonal_feature_count(config, changepoints.len());
    let mut xtx = vec![vec![0.0; feature_count]; feature_count];
    let mut xty = vec![0.0; feature_count];
    let mut features = vec![0.0; feature_count];
    for (idx, (row, &t)) in history.iter().zip(elapsed.iter()).enumerate() {
        let weight = weights.map(|values| values[idx]).unwrap_or(1.0);
        if !weight.is_finite() || weight < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "piecewise robust fit weights must be finite nonnegative values".to_string(),
            ));
        }
        if weight <= 0.0 {
            continue;
        }
        let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
        let target = transform_piecewise_target(row.target, bounds, config)?;
        fill_piecewise_linear_seasonal_features(
            &mut features,
            t,
            &PiecewiseLinearFeatureContext {
                series_id: None,
                timestamp: row.timestamp,
                covariates: Some(&row.covariates),
                horizon_step: None,
                component_multiplier: fit_component_multiplier(
                    t,
                    trend_coefficients,
                    changepoints,
                    bounds,
                    config,
                ),
                changepoints,
                config,
                regressor_stats: Some(regressor_stats),
            },
        )?;
        for i in 0..feature_count {
            xty[i] += weight * features[i] * target;
            for j in i..feature_count {
                xtx[i][j] += weight * features[i] * features[j];
            }
        }
    }
    for i in 0..feature_count {
        let (previous_rows, current_and_after) = xtx.split_at_mut(i);
        let current_row = &mut current_and_after[0];
        for (j, previous_row) in previous_rows.iter().enumerate() {
            current_row[j] = previous_row[i];
        }
    }
    apply_piecewise_linear_ridge(&mut xtx, changepoints.len(), config);
    let Some(coefficients) =
        solve_piecewise_linear_coefficients(xtx.clone(), xty, changepoints.len(), config)
    else {
        return Ok(None);
    };
    let coefficient_covariance = if compute_covariance {
        invert_piecewise_linear_normal_matrix(&xtx).ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "could not invert piecewise linear covariance normal matrix".to_string(),
            )
        })?
    } else {
        Vec::new()
    };
    Ok(Some(PiecewiseLinearFitResult {
        coefficients,
        coefficient_covariance,
    }))
}

fn piecewise_needs_coefficient_covariance(config: &PiecewiseLinearSeasonalConfig) -> bool {
    config.coefficient_uncertainty_scale > 0.0
        && (!config.interval_levels.is_empty()
            || !config.quantile_levels.is_empty()
            || config.uncertainty_samples > 0)
}

fn invert_piecewise_linear_normal_matrix(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut left = matrix.to_vec();
    let mut inverse = vec![vec![0.0; n]; n];
    for (idx, row) in inverse.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if left[row][pivot_idx].abs() > left[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if left[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        left.swap(pivot_idx, pivot_row);
        inverse.swap(pivot_idx, pivot_row);

        let pivot = left[pivot_idx][pivot_idx];
        for cell in left[pivot_idx].iter_mut() {
            *cell /= pivot;
        }
        for cell in inverse[pivot_idx].iter_mut() {
            *cell /= pivot;
        }
        let pivot_left = left[pivot_idx].clone();
        let pivot_inverse = inverse[pivot_idx].clone();

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = left[row][pivot_idx];
            if factor == 0.0 {
                continue;
            }
            for (cell, pivot_cell) in left[row].iter_mut().zip(pivot_left.iter()) {
                *cell -= factor * pivot_cell;
            }
            for (cell, pivot_cell) in inverse[row].iter_mut().zip(pivot_inverse.iter()) {
                *cell -= factor * pivot_cell;
            }
        }
    }
    Some(inverse)
}

fn piecewise_transformed_residual_scale(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: &BTreeMap<String, PiecewiseLinearRegressorStats>,
    coefficients: &[f64],
) -> Result<f64> {
    let residuals = piecewise_transformed_residuals(
        history,
        elapsed,
        changepoints,
        config,
        regressor_stats,
        coefficients,
        coefficients,
    )?;
    Ok(piecewise_residual_rmse_scale(&residuals).unwrap_or(0.0))
}

fn piecewise_transformed_residuals(
    history: &[ForecastRow],
    elapsed: &[f64],
    changepoints: &[f64],
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: &BTreeMap<String, PiecewiseLinearRegressorStats>,
    trend_coefficients: &[f64],
    coefficients: &[f64],
) -> Result<Vec<f64>> {
    let mut features =
        vec![0.0; piecewise_linear_seasonal_feature_count(config, changepoints.len())];
    history
        .iter()
        .zip(elapsed.iter())
        .map(|(row, &t)| {
            let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
            let target = transform_piecewise_target(row.target, bounds, config)?;
            fill_piecewise_linear_seasonal_features(
                &mut features,
                t,
                &PiecewiseLinearFeatureContext {
                    series_id: None,
                    timestamp: row.timestamp,
                    covariates: Some(&row.covariates),
                    horizon_step: None,
                    component_multiplier: fit_component_multiplier(
                        t,
                        trend_coefficients,
                        changepoints,
                        bounds,
                        config,
                    ),
                    changepoints,
                    config,
                    regressor_stats: Some(regressor_stats),
                },
            )?;
            let fitted = features
                .iter()
                .zip(coefficients.iter())
                .map(|(feature, coefficient)| feature * coefficient)
                .sum::<f64>();
            Ok(target - fitted)
        })
        .collect()
}

fn piecewise_residual_rmse_scale(residuals: &[f64]) -> Option<f64> {
    let mut count = 0usize;
    let mut sum_squared = 0.0;
    for residual in residuals.iter().copied().filter(|value| value.is_finite()) {
        count += 1;
        sum_squared += residual * residual;
    }
    (count > 0 && sum_squared > 0.0).then_some((sum_squared / count as f64).sqrt())
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn piecewise_linear_l2_penalties(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> Vec<f64> {
    let feature_count = piecewise_linear_seasonal_feature_count(config, changepoint_count);
    let mut penalties = vec![0.0; feature_count];
    let mut col = 2;
    penalties[1] = config.changepoint_l2_regularization;
    for _ in 0..changepoint_count {
        penalties[col] = config.changepoint_l2_regularization;
        col += 1;
    }
    col = fill_penalty_range(
        &mut penalties,
        col,
        2 * config.yearly_fourier_order,
        config
            .yearly_l2_regularization
            .unwrap_or(config.seasonality_l2_regularization),
    );
    col = fill_penalty_range(
        &mut penalties,
        col,
        2 * config.weekly_fourier_order,
        config
            .weekly_l2_regularization
            .unwrap_or(config.seasonality_l2_regularization),
    );
    col = fill_penalty_range(
        &mut penalties,
        col,
        2 * config.daily_fourier_order,
        config
            .daily_l2_regularization
            .unwrap_or(config.seasonality_l2_regularization),
    );
    for seasonality in &config.custom_seasonalities {
        col = fill_penalty_range(
            &mut penalties,
            col,
            2 * seasonality.fourier_order,
            seasonality
                .l2_regularization
                .unwrap_or(config.seasonality_l2_regularization),
        );
    }
    for term in piecewise_event_terms(config) {
        penalties[col] = config
            .event_l2_regularization_by_name
            .get(&term.name)
            .copied()
            .unwrap_or(config.event_l2_regularization);
        col += 1;
    }
    for name in &config.extra_regressors {
        penalties[col] = config
            .regressor_l2_regularization_by_name
            .get(name)
            .copied()
            .unwrap_or(config.regressor_l2_regularization);
        col += 1;
    }
    debug_assert_eq!(col, feature_count);
    penalties
}

fn fill_penalty_range(penalties: &mut [f64], start: usize, len: usize, penalty: f64) -> usize {
    for value in penalties.iter_mut().skip(start).take(len) {
        *value = penalty;
    }
    start + len
}

fn solve_piecewise_linear_coefficients(
    xtx: Vec<Vec<f64>>,
    xty: Vec<f64>,
    changepoint_count: usize,
    config: &PiecewiseLinearSeasonalConfig,
) -> Option<Vec<f64>> {
    let mut coefficients = solve_linear_system(xtx.clone(), xty.clone())?;
    let monotonic_constraints =
        piecewise_linear_coefficient_monotonic_constraints(config, changepoint_count);
    let has_monotonic_constraints = monotonic_constraints
        .iter()
        .any(|direction| *direction != 0);
    if (config.changepoint_l1_regularization <= 0.0 || changepoint_count == 0)
        && !has_monotonic_constraints
    {
        return Some(coefficients);
    }
    let penalized_start = 2;
    let penalized_end = 2 + changepoint_count;
    for _ in 0..100 {
        let mut max_delta = 0.0_f64;
        for j in 0..coefficients.len() {
            let diagonal = xtx[j][j];
            if diagonal.abs() <= 1.0e-12 || !diagonal.is_finite() {
                return None;
            }
            let without_j = xty[j]
                - xtx[j]
                    .iter()
                    .zip(coefficients.iter())
                    .enumerate()
                    .filter(|(idx, _)| *idx != j)
                    .map(|(_, (x, coefficient))| x * coefficient)
                    .sum::<f64>();
            let mut updated = if (penalized_start..penalized_end).contains(&j) {
                soft_threshold(without_j, config.changepoint_l1_regularization) / diagonal
            } else {
                without_j / diagonal
            };
            updated = match monotonic_constraints.get(j).copied().unwrap_or(0) {
                1 => updated.max(0.0),
                -1 => updated.min(0.0),
                _ => updated,
            };
            max_delta = max_delta.max((updated - coefficients[j]).abs());
            coefficients[j] = updated;
        }
        if max_delta < 1.0e-10 {
            break;
        }
    }
    Some(coefficients)
}

fn piecewise_linear_coefficient_monotonic_constraints(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> Vec<i8> {
    let feature_count = piecewise_linear_seasonal_feature_count(config, changepoint_count);
    let mut constraints = vec![0; feature_count];
    let start = piecewise_extra_regressor_start_col(config, changepoint_count);
    for (offset, name) in config.extra_regressors.iter().enumerate() {
        constraints[start + offset] = config
            .extra_regressor_monotonic_constraints
            .get(name)
            .copied()
            .unwrap_or(0);
    }
    constraints
}

fn piecewise_extra_regressor_start_col(
    config: &PiecewiseLinearSeasonalConfig,
    changepoint_count: usize,
) -> usize {
    2 + changepoint_count
        + 2 * config.yearly_fourier_order
        + 2 * config.weekly_fourier_order
        + 2 * config.daily_fourier_order
        + 2 * config
            .custom_seasonalities
            .iter()
            .map(|seasonality| seasonality.fourier_order)
            .sum::<usize>()
        + piecewise_event_terms(config).len()
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

fn piecewise_trend_delta_scale(coefficients: &[f64], changepoint_count: usize) -> f64 {
    if changepoint_count == 0 {
        return 0.0;
    }
    let mut deltas = coefficients
        .iter()
        .skip(2)
        .take(changepoint_count)
        .map(|value| value.abs())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if deltas.is_empty() {
        return 0.0;
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    quantile_from_sorted(&deltas, 0.5)
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn deterministic_standard_normal(seed: u64, step: u64, sample: u64) -> f64 {
    let state = splitmix64(
        seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ sample.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    );
    let uniform = uniform_open_0_1(state);
    inverse_standard_normal_cdf(uniform)
}

fn deterministic_trend_uncertainty_draw(
    seed: u64,
    step: u64,
    sample: u64,
    policy: PiecewiseLinearTrendUncertaintyPolicy,
) -> f64 {
    let state = splitmix64(
        seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ sample.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    );
    let uniform = uniform_open_0_1(state);
    match policy {
        PiecewiseLinearTrendUncertaintyPolicy::Normal => inverse_standard_normal_cdf(uniform),
        PiecewiseLinearTrendUncertaintyPolicy::Laplace => {
            if uniform < 0.5 {
                (2.0 * uniform).ln()
            } else {
                -(2.0 * (1.0 - uniform)).ln()
            }
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn uniform_open_0_1(value: u64) -> f64 {
    let mantissa = value >> 11;
    ((mantissa as f64) + 0.5) / ((1_u64 << 53) as f64)
}

fn predict_piecewise_linear_value(
    elapsed_days: f64,
    coefficients: &[f64],
    context: &PiecewiseLinearFeatureContext<'_>,
) -> Result<f64> {
    let mut features =
        vec![
            0.0;
            piecewise_linear_seasonal_feature_count(context.config, context.changepoints.len())
        ];
    fill_piecewise_linear_seasonal_features(&mut features, elapsed_days, context)?;
    Ok(features
        .iter()
        .zip(coefficients.iter())
        .map(|(feature, coefficient)| feature * coefficient)
        .sum())
}

fn piecewise_component_contributions(
    features: &[f64],
    coefficients: &[f64],
    changepoint_count: usize,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Value> {
    let expected = piecewise_linear_seasonal_feature_count(config, changepoint_count);
    if features.len() != expected || coefficients.len() != expected {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal component decomposition received mismatched feature dimensions"
                .to_string(),
        ));
    }
    let mut col = 2 + changepoint_count;
    let mut components = serde_json::Map::new();
    let trend = dot_range(features, coefficients, 0, col);
    components.insert("trend_linear".to_string(), json!(trend));
    components.insert(
        "changepoint_delta".to_string(),
        json!(dot_range(features, coefficients, 2, 2 + changepoint_count)),
    );
    col = insert_fourier_component(
        &mut components,
        "yearly",
        features,
        coefficients,
        col,
        config.yearly_fourier_order,
    );
    col = insert_fourier_component(
        &mut components,
        "weekly",
        features,
        coefficients,
        col,
        config.weekly_fourier_order,
    );
    col = insert_fourier_component(
        &mut components,
        "daily",
        features,
        coefficients,
        col,
        config.daily_fourier_order,
    );
    for seasonality in &config.custom_seasonalities {
        col = insert_fourier_component(
            &mut components,
            &seasonality.name,
            features,
            coefficients,
            col,
            seasonality.fourier_order,
        );
    }
    let mut event_components = serde_json::Map::new();
    let mut event_offset_components = serde_json::Map::new();
    for term in piecewise_event_terms(config) {
        let contribution = features[col] * coefficients[col];
        let event_total = event_components
            .get(&term.name)
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            + contribution;
        event_components.insert(term.name.clone(), json!(event_total));
        event_offset_components.insert(term.label(), json!(contribution));
        col += 1;
    }
    components.insert("events".to_string(), Value::Object(event_components));
    components.insert(
        "event_window_offsets".to_string(),
        Value::Object(event_offset_components),
    );
    let mut regressor_components = serde_json::Map::new();
    for name in &config.extra_regressors {
        regressor_components.insert(name.clone(), json!(features[col] * coefficients[col]));
        col += 1;
    }
    components.insert(
        "regressors".to_string(),
        Value::Object(regressor_components),
    );
    let seasonal_total = components
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "trend_linear"
                    | "changepoint_delta"
                    | "events"
                    | "event_window_offsets"
                    | "regressors"
            )
        })
        .filter_map(|(_, value)| value.as_f64())
        .sum::<f64>();
    let event_total = components["events"]
        .as_object()
        .map(|events| events.values().filter_map(Value::as_f64).sum::<f64>())
        .unwrap_or(0.0);
    let regressor_total = components["regressors"]
        .as_object()
        .map(|regressors| regressors.values().filter_map(Value::as_f64).sum::<f64>())
        .unwrap_or(0.0);
    components.insert("seasonal_total".to_string(), json!(seasonal_total));
    components.insert("event_total".to_string(), json!(event_total));
    components.insert("regressor_total".to_string(), json!(regressor_total));
    components.insert(
        "non_trend_total".to_string(),
        json!(seasonal_total + event_total + regressor_total),
    );
    debug_assert_eq!(col, expected);
    Ok(Value::Object(components))
}

fn insert_fourier_component(
    components: &mut serde_json::Map<String, Value>,
    name: &str,
    features: &[f64],
    coefficients: &[f64],
    col: usize,
    order: usize,
) -> usize {
    let len = 2 * order;
    components.insert(
        name.to_string(),
        json!(dot_range(features, coefficients, col, col + len)),
    );
    col + len
}

fn dot_range(features: &[f64], coefficients: &[f64], start: usize, end: usize) -> f64 {
    features
        .iter()
        .zip(coefficients.iter())
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|(feature, coefficient)| feature * coefficient)
        .sum()
}

fn piecewise_event_names(config: &PiecewiseLinearSeasonalConfig) -> Vec<String> {
    let mut names = config
        .events
        .iter()
        .map(|event| event.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn piecewise_event_terms(config: &PiecewiseLinearSeasonalConfig) -> Vec<PiecewiseEventTerm> {
    let mut terms = config
        .events
        .iter()
        .flat_map(|event| {
            (event.lower_window..=event.upper_window).map(|offset| PiecewiseEventTerm {
                name: event.name.clone(),
                offset,
            })
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn event_term_is_active(
    timestamp: chrono::NaiveDateTime,
    term: &PiecewiseEventTerm,
    events: &[PiecewiseLinearEvent],
) -> bool {
    events
        .iter()
        .filter(|event| event.name == term.name)
        .any(|event| {
            let days = (timestamp.date() - event.timestamp.date()).num_days();
            days == i64::from(term.offset)
                && days >= i64::from(event.lower_window)
                && days <= i64::from(event.upper_window)
        })
}

fn piecewise_regressor_stats(
    history: &[ForecastRow],
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<BTreeMap<String, PiecewiseLinearRegressorStats>> {
    config
        .extra_regressors
        .iter()
        .map(|name| {
            let values = history
                .iter()
                .map(|row| {
                    piecewise_named_value(
                        name,
                        None,
                        Some(&row.covariates),
                        None,
                        config,
                        "extra regressor",
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((
                name.clone(),
                piecewise_regressor_stat(&values, config.regressor_standardization),
            ))
        })
        .collect()
}

fn piecewise_regressor_stat(
    values: &[f64],
    standardization: PiecewiseLinearRegressorStandardization,
) -> PiecewiseLinearRegressorStats {
    if standardization == PiecewiseLinearRegressorStandardization::None || values.is_empty() {
        return PiecewiseLinearRegressorStats {
            mean: 0.0,
            scale: 1.0,
            standardized: false,
        };
    }
    let is_binary = values
        .iter()
        .all(|value| (*value - 0.0).abs() < 1.0e-12 || (*value - 1.0).abs() < 1.0e-12);
    if is_binary {
        return PiecewiseLinearRegressorStats {
            mean: 0.0,
            scale: 1.0,
            standardized: false,
        };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    let scale = variance.sqrt();
    if !scale.is_finite() || scale <= 1.0e-12 {
        PiecewiseLinearRegressorStats {
            mean: 0.0,
            scale: 1.0,
            standardized: false,
        }
    } else {
        PiecewiseLinearRegressorStats {
            mean,
            scale,
            standardized: true,
        }
    }
}

fn piecewise_regressor_value(
    name: &str,
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<f64> {
    piecewise_named_value(
        name,
        series_id,
        covariates,
        horizon_step,
        config,
        "extra regressor",
    )
}

fn piecewise_extra_regressor_value(
    name: &str,
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
    regressor_stats: Option<&BTreeMap<String, PiecewiseLinearRegressorStats>>,
) -> Result<f64> {
    let value = piecewise_regressor_value(name, series_id, covariates, horizon_step, config)?;
    Ok(match regressor_stats.and_then(|stats| stats.get(name)) {
        Some(stats) if stats.standardized => (value - stats.mean) / stats.scale,
        _ => value,
    })
}

fn piecewise_seasonality_condition_is_active(
    seasonality: &PiecewiseLinearSeasonality,
    context: &PiecewiseLinearFeatureContext<'_>,
) -> Result<bool> {
    let Some(condition_name) = &seasonality.condition_name else {
        return Ok(true);
    };
    let value = piecewise_named_value(
        condition_name,
        context.series_id,
        context.covariates,
        context.horizon_step,
        context.config,
        "seasonality condition",
    )?;
    Ok(value > 0.0)
}

fn piecewise_named_value(
    name: &str,
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
    role: &str,
) -> Result<f64> {
    if let Some(covariates) = covariates {
        let value = covariates.get(name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "piecewise seasonal {role} {name:?} missing from training covariates"
            ))
        })?;
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "piecewise seasonal {role} {name:?} training values must be finite"
            )));
        }
        return Ok(value);
    }
    let step = horizon_step.ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "piecewise seasonal prediction requires a future regressor horizon step".to_string(),
        )
    })?;
    let values = series_id
        .and_then(|series_id| config.future_regressors_by_series.get(series_id))
        .and_then(|values_by_name| values_by_name.get(name))
        .or_else(|| config.future_regressors.get(name))
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "piecewise seasonal future {role} {name:?} is missing"
            ))
        })?;
    let value = values.get(step - 1).copied().ok_or_else(|| {
        CartoBoostError::InvalidInput(format!(
            "piecewise seasonal future {role} {name:?} has fewer than {step} values"
        ))
    })?;
    if !value.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "piecewise seasonal future {role} {name:?} values must be finite"
        )));
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy)]
struct PiecewiseBounds {
    floor: f64,
    cap: Option<f64>,
}

fn piecewise_bounds(
    series_id: Option<&str>,
    covariates: Option<&BTreeMap<String, f64>>,
    horizon_step: Option<usize>,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<PiecewiseBounds> {
    let floor = match &config.floor_regressor {
        Some(name) => piecewise_named_value(
            name,
            series_id,
            covariates,
            horizon_step,
            config,
            "floor_regressor",
        )?,
        None => config.floor,
    };
    let cap = match &config.cap_regressor {
        Some(name) => Some(piecewise_named_value(
            name,
            series_id,
            covariates,
            horizon_step,
            config,
            "cap_regressor",
        )?),
        None => config.cap,
    };
    if !floor.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal floor must be finite".to_string(),
        ));
    }
    if let Some(cap) = cap {
        if !cap.is_finite() || cap <= floor {
            return Err(CartoBoostError::InvalidInput(
                "piecewise seasonal cap must be finite and greater than floor".to_string(),
            ));
        }
    }
    Ok(PiecewiseBounds { floor, cap })
}

fn transform_piecewise_target(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<f64> {
    match config.growth {
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => Ok(value),
        PiecewiseLinearGrowth::Logistic => {
            let cap = bounds.cap.expect("validated logistic cap");
            if value <= bounds.floor || value >= cap {
                return Err(CartoBoostError::InvalidInput(
                    "logistic piecewise seasonal targets must be strictly between floor and cap"
                        .to_string(),
                ));
            }
            let scaled = (value - bounds.floor) / (cap - bounds.floor);
            Ok((scaled / (1.0 - scaled)).ln())
        }
    }
}

fn inverse_piecewise_target(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    match config.growth {
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => value,
        PiecewiseLinearGrowth::Logistic => {
            let cap = bounds.cap.expect("validated logistic cap");
            let scaled = if value >= 0.0 {
                let z = (-value).exp();
                1.0 / (1.0 + z)
            } else {
                let z = value.exp();
                z / (1.0 + z)
            };
            bounds.floor + (cap - bounds.floor) * scaled
        }
    }
}

fn inverse_piecewise_target_derivative(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    match config.growth {
        PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => 1.0,
        PiecewiseLinearGrowth::Logistic => {
            let cap = bounds.cap.expect("validated logistic cap");
            let scaled = if value >= 0.0 {
                let z = (-value).exp();
                1.0 / (1.0 + z)
            } else {
                let z = value.exp();
                z / (1.0 + z)
            };
            (cap - bounds.floor) * scaled * (1.0 - scaled)
        }
    }
}

fn quadratic_form(vector: &[f64], matrix: &[Vec<f64>]) -> f64 {
    if matrix.len() != vector.len() || matrix.iter().any(|row| row.len() != vector.len()) {
        return 0.0;
    }
    vector
        .iter()
        .enumerate()
        .map(|(i, left)| {
            let row_dot = matrix[i]
                .iter()
                .zip(vector.iter())
                .map(|(matrix_value, right)| matrix_value * right)
                .sum::<f64>();
            left * row_dot
        })
        .sum()
}

fn validate_kalman_grid(name: &str, values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "auto_kalman {name} must not be empty"
        )));
    }
    for &value in values {
        if !value.is_finite() || value <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "auto_kalman {name} values must be positive finite numbers"
            )));
        }
    }
    Ok(())
}

fn score_kalman_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    params: KalmanParameterSet,
    validation_window: Option<usize>,
) -> Result<(f64, f64)> {
    let config = LocalLinearKalmanConfig::new(
        params.level_process_variance,
        params.trend_process_variance,
        params.observation_variance,
    )?;
    let (sum_squared_error, sum_negative_log_likelihood, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            let window = resolve_kalman_validation_window(
                series_id,
                history.len(),
                2,
                validation_window,
                "auto_kalman",
            )?;
            let train_len = history.len() - window;
            let train = history[..train_len]
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>();
            let result = fit_local_linear_kalman(&train, config)
                .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            if window == 0 {
                return Ok((
                    result.residual_summary.mse,
                    (-result.log_likelihood).max(0.0),
                    1,
                ));
            }
            let distribution = local_linear_kalman_forecast_distribution(
                result.final_state,
                result.final_covariance,
                config,
                window,
                0.0,
            )
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            let mut squared_error = 0.0;
            let mut negative_log_likelihood = 0.0;
            for (row, point) in history[train_len..].iter().zip(distribution) {
                let residual = row.target - point.mean;
                squared_error += residual * residual;
                negative_log_likelihood +=
                    normal_negative_log_likelihood(residual, point.variance)?;
            }
            Ok((squared_error, negative_log_likelihood, window))
        })
        .reduce(
            || Ok((0.0, 0.0, 0usize)),
            |left: Result<(f64, f64, usize)>, right: Result<(f64, f64, usize)>| {
                let (left_squared, left_nll, left_count) = left?;
                let (right_squared, right_nll, right_count) = right?;
                Ok((
                    left_squared + right_squared,
                    left_nll + right_nll,
                    left_count + right_count,
                ))
            },
        )?;
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto_kalman validation requires at least one held-out observation".to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    let negative_log_likelihood = sum_negative_log_likelihood / count as f64;
    if !mse.is_finite() || !negative_log_likelihood.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_kalman validation score must be finite".to_string(),
        ));
    }
    Ok((mse, negative_log_likelihood))
}

fn score_arima_order(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    p: usize,
    d: usize,
    q: usize,
    validation_window: usize,
) -> Result<(f64, bool, bool)> {
    if validation_window == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto_arima validation_window must be positive".to_string(),
        ));
    }
    let (sum_squared_error, count, ar_stable, ma_invertible) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            if history.len() <= validation_window {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} does not have enough rows for auto_arima validation"
                )));
            }
            let train_len = history.len() - validation_window;
            let fitted = FittedArimaSeries::fit(series_id, &history[..train_len], p, d, q)?;
            let forecasts = fitted.forecast_values(validation_window);
            if forecasts.len() != validation_window {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} ARIMA({p},{d},{q}) returned an incomplete validation forecast"
                )));
            }
            let squared_error = forecasts
                .iter()
                .zip(&history[train_len..])
                .map(|(forecast, actual)| {
                    let residual = forecast - actual.target;
                    residual * residual
                })
                .sum::<f64>();
            Ok((
                squared_error,
                validation_window,
                ar_recursion_is_stable(&fitted.ar_coefficients),
                ma_recursion_is_invertible(&fitted.ma_coefficients),
            ))
        })
        .reduce(
            || Ok((0.0, 0usize, true, true)),
            |left: Result<(f64, usize, bool, bool)>,
             right: Result<(f64, usize, bool, bool)>| {
                let (left_squared, left_count, left_ar, left_ma) = left?;
                let (right_squared, right_count, right_ar, right_ma) = right?;
                Ok((
                    left_squared + right_squared,
                    left_count + right_count,
                    left_ar && right_ar,
                    left_ma && right_ma,
                ))
            },
        )?;
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto_arima validation produced no held-out forecasts".to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    if !mse.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_arima validation MSE must be finite".to_string(),
        ));
    }
    Ok((mse, ar_stable, ma_invertible))
}

fn automatic_model_validation_window(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    minimum_train_len: usize,
    model_name: &str,
) -> Result<usize> {
    let minimum_history = history_by_series
        .values()
        .map(Vec::len)
        .min()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "{model_name} requires at least one series for validation"
            ))
        })?;
    let maximum_window = minimum_history.checked_sub(minimum_train_len).ok_or_else(|| {
        CartoBoostError::InvalidInput(format!(
            "{model_name} requires more than {minimum_train_len} rows per series for a real holdout; minimum history is {minimum_history}"
        ))
    })?;
    if maximum_window == 0 {
        return Ok(0);
    }
    Ok((minimum_history / 5).clamp(1, 8).min(maximum_window))
}

fn score_theta_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    theta: f64,
    alpha: f64,
    seasonality: Option<ThetaSeasonality>,
    validation_window: usize,
) -> Result<f64> {
    let (sum_squared_error, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            if validation_window == 0 {
                let fitted = FittedThetaSeries::fit(series_id, history, theta, alpha, seasonality)?;
                let (sum, count) = fitted
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    });
                return Ok((sum, count));
            }
            let train_len = history
                .len()
                .checked_sub(validation_window)
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                    "series {series_id} does not have enough rows for optimized_theta validation"
                ))
                })?;
            let fitted = FittedThetaSeries::fit(
                series_id,
                &history[..train_len],
                theta,
                alpha,
                seasonality,
            )?;
            let forecasts = fitted.forecast_values(validation_window, seasonality)?;
            squared_holdout_error(
                series_id,
                "optimized_theta",
                &forecasts,
                &history[train_len..],
            )
        })
        .reduce(
            || Ok((0.0, 0usize)),
            |left: Result<(f64, usize)>, right: Result<(f64, usize)>| {
                let (left_sum, left_count) = left?;
                let (right_sum, right_count) = right?;
                Ok((left_sum + right_sum, left_count + right_count))
            },
        )?;
    checked_validation_mse(sum_squared_error, count, "optimized_theta")
}

#[allow(clippy::too_many_arguments)]
fn score_ets_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    alpha: f64,
    beta: f64,
    gamma: Option<f64>,
    season_length: Option<usize>,
    damping_phi: f64,
    validation_window: usize,
) -> Result<f64> {
    let (sum_squared_error, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            if validation_window == 0 {
                let fitted = FittedETSSeries::fit(
                    series_id,
                    history,
                    alpha,
                    beta,
                    gamma,
                    season_length,
                    damping_phi,
                )?;
                let (sum, count) = fitted
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    });
                return Ok((sum, count));
            }
            let train_len = history
                .len()
                .checked_sub(validation_window)
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "series {series_id} does not have enough rows for auto_ets validation"
                    ))
                })?;
            let fitted = FittedETSSeries::fit(
                series_id,
                &history[..train_len],
                alpha,
                beta,
                gamma,
                season_length,
                damping_phi,
            )?;
            let forecasts = fitted.forecast_values(validation_window);
            squared_holdout_error(series_id, "auto_ets", &forecasts, &history[train_len..])
        })
        .reduce(
            || Ok((0.0, 0usize)),
            |left: Result<(f64, usize)>, right: Result<(f64, usize)>| {
                let (left_sum, left_count) = left?;
                let (right_sum, right_count) = right?;
                Ok((left_sum + right_sum, left_count + right_count))
            },
        )?;
    checked_validation_mse(sum_squared_error, count, "auto_ets")
}

fn squared_holdout_error(
    series_id: &str,
    model_name: &str,
    forecasts: &[f64],
    actuals: &[ForecastRow],
) -> Result<(f64, usize)> {
    if forecasts.len() != actuals.len() || forecasts.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} {model_name} validation forecast length does not match its non-empty holdout"
        )));
    }
    let sum = forecasts
        .iter()
        .zip(actuals)
        .map(|(forecast, actual)| {
            let residual = forecast - actual.target;
            residual * residual
        })
        .sum::<f64>();
    if !sum.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} {model_name} validation error must be finite"
        )));
    }
    Ok((sum, actuals.len()))
}

fn checked_validation_mse(sum_squared_error: f64, count: usize, model_name: &str) -> Result<f64> {
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{model_name} validation produced no held-out forecasts"
        )));
    }
    let mse = sum_squared_error / count as f64;
    if !mse.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{model_name} validation MSE must be finite"
        )));
    }
    Ok(mse)
}

fn score_local_level_kalman_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    params: LocalLevelKalmanParameterSet,
    validation_window: Option<usize>,
) -> Result<(f64, f64)> {
    let config =
        LocalLevelKalmanConfig::new(params.level_process_variance, params.observation_variance)?;
    let (sum_squared_error, sum_negative_log_likelihood, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            let window = resolve_kalman_validation_window(
                series_id,
                history.len(),
                1,
                validation_window,
                "auto_local_level_kalman",
            )?;
            let train_len = history.len() - window;
            let train = history[..train_len]
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>();
            let result = fit_local_level_kalman(&train, config)
                .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            if window == 0 {
                return Ok((
                    result.residual_summary.mse,
                    (-result.log_likelihood).max(0.0),
                    1,
                ));
            }
            let distribution = local_level_kalman_forecast_distribution(
                result.final_level,
                result.final_variance,
                config,
                window,
                0.0,
            )
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            let mut squared_error = 0.0;
            let mut negative_log_likelihood = 0.0;
            for (row, point) in history[train_len..].iter().zip(distribution) {
                let residual = row.target - point.mean;
                squared_error += residual * residual;
                negative_log_likelihood +=
                    normal_negative_log_likelihood(residual, point.variance)?;
            }
            Ok((squared_error, negative_log_likelihood, window))
        })
        .reduce(
            || Ok((0.0, 0.0, 0usize)),
            |left: Result<(f64, f64, usize)>, right: Result<(f64, f64, usize)>| {
                let (left_squared, left_nll, left_count) = left?;
                let (right_squared, right_nll, right_count) = right?;
                Ok((
                    left_squared + right_squared,
                    left_nll + right_nll,
                    left_count + right_count,
                ))
            },
        )?;
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto_local_level_kalman validation requires at least one held-out observation"
                .to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    let negative_log_likelihood = sum_negative_log_likelihood / count as f64;
    if !mse.is_finite() || !negative_log_likelihood.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_local_level_kalman validation score must be finite".to_string(),
        ));
    }
    Ok((mse, negative_log_likelihood))
}

fn resolve_kalman_validation_window(
    series_id: &str,
    history_len: usize,
    minimum_train_len: usize,
    requested: Option<usize>,
    model_name: &str,
) -> Result<usize> {
    let minimum_total = minimum_train_len + 1;
    if history_len < minimum_total {
        if requested.is_none() && history_len > 0 {
            return Ok(0);
        }
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} has {history_len} rows; {model_name} requires at least {minimum_total} rows for {minimum_train_len} training observations and a real holdout"
        )));
    }
    let maximum_window = history_len - minimum_train_len;
    match requested {
        Some(window) if window > maximum_window => Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} has {history_len} rows, so {model_name} validation_window must be <= {maximum_window}; got {window}"
        ))),
        Some(window) => Ok(window),
        None => Ok((history_len / 5).clamp(1, 12).min(maximum_window)),
    }
}

fn normal_negative_log_likelihood(residual: f64, variance: f64) -> Result<f64> {
    if !residual.is_finite() || !variance.is_finite() || variance <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman validation residual and forecast variance must be finite with positive variance"
                .to_string(),
        ));
    }
    Ok(0.5 * ((2.0 * std::f64::consts::PI).ln() + variance.ln() + residual * residual / variance))
}

fn validate_ets_params(
    alpha: f64,
    beta: f64,
    gamma: Option<f64>,
    season_length: Option<usize>,
    damping_phi: f64,
) -> Result<()> {
    validate_unit_interval("alpha", alpha, false)?;
    validate_unit_interval("beta", beta, true)?;
    validate_unit_interval("damping_phi", damping_phi, false)?;
    if let Some(gamma) = gamma {
        validate_unit_interval("gamma", gamma, true)?;
    }
    match (gamma, season_length) {
        (Some(_), Some(length)) if length > 1 => Ok(()),
        (None, None) => Ok(()),
        (Some(_), None) => Err(CartoBoostError::InvalidInput(
            "ETS gamma requires season_length".to_string(),
        )),
        (None, Some(_)) => Err(CartoBoostError::InvalidInput(
            "ETS season_length requires gamma".to_string(),
        )),
        (Some(_), Some(_)) => Err(CartoBoostError::InvalidInput(
            "ETS season_length must be greater than 1".to_string(),
        )),
    }
}

fn damped_trend_multiplier(damping_phi: f64, step: usize) -> f64 {
    if (damping_phi - 1.0).abs() <= f64::EPSILON {
        step as f64
    } else {
        damping_phi * (1.0 - damping_phi.powi(step as i32)) / (1.0 - damping_phi)
    }
}

fn validate_unit_interval(name: &str, value: f64, allow_zero: bool) -> Result<()> {
    let lower_ok = if allow_zero {
        value >= 0.0
    } else {
        value > 0.0
    };
    if !value.is_finite() || !lower_ok || value > 1.0 {
        let range = if allow_zero { "[0, 1]" } else { "(0, 1]" };
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite and in {range}"
        )));
    }
    Ok(())
}

fn validate_arima_order(p: usize, d: usize, q: usize) -> Result<()> {
    if p > MAX_ARIMA_ORDER {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA p must be <= 8".to_string(),
        ));
    }
    if d > 2 {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        ));
    }
    if q > MAX_ARIMA_ORDER {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA q must be <= 8".to_string(),
        ));
    }
    Ok(())
}

fn arima_order_supported_by_history(
    history_len: usize,
    p: usize,
    d: usize,
    q: usize,
) -> (usize, usize, usize) {
    let effective_d = d.min(2).min(history_len.saturating_sub(1));
    let differenced_len = history_len.saturating_sub(effective_d);
    let max_lag_order = differenced_len.saturating_sub(1).min(MAX_ARIMA_ORDER);
    (p.min(max_lag_order), effective_d, q.min(max_lag_order))
}

fn initial_trend(values: &[f64], seasonals: Option<&[f64]>) -> f64 {
    match seasonals {
        Some(seasonals) if values.len() > seasonals.len() => {
            let length = seasonals.len();
            let mut sum = 0.0;
            for idx in 0..length {
                sum += (values[idx + length] - values[idx]) / length as f64;
            }
            sum / length as f64
        }
        Some(seasonals) => {
            (values[1] - seasonals[1 % seasonals.len()]) - (values[0] - seasonals[0])
        }
        None => values[1] - values[0],
    }
}

fn difference_series(values: &[f64], d: usize) -> Result<Vec<f64>> {
    if values.len() <= d {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA differencing order leaves no observations".to_string(),
        ));
    }
    match d {
        0 => Ok(values.to_vec()),
        1 => {
            let mut differences = Vec::with_capacity(values.len() - 1);
            for idx in 1..values.len() {
                differences.push(values[idx] - values[idx - 1]);
            }
            Ok(differences)
        }
        2 => {
            let mut differences = Vec::with_capacity(values.len() - 2);
            for idx in 2..values.len() {
                differences.push(values[idx] - 2.0 * values[idx - 1] + values[idx - 2]);
            }
            Ok(differences)
        }
        _ => Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        )),
    }
}

fn last_differences(values: &[f64], d: usize) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA requires at least one observation".to_string(),
        ));
    }
    if values.len() <= d {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA differencing order leaves no observations".to_string(),
        ));
    }
    let last = values[values.len() - 1];
    match d {
        0 => Ok(vec![last]),
        1 => Ok(vec![last, last - values[values.len() - 2]]),
        2 => {
            let n = values.len();
            Ok(vec![
                last,
                last - values[n - 2],
                last - 2.0 * values[n - 2] + values[n - 3],
            ])
        }
        _ => Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        )),
    }
}

fn tail_values(values: &[f64], length: usize) -> Vec<f64> {
    if length == 0 {
        Vec::new()
    } else {
        values[values.len().saturating_sub(length)..].to_vec()
    }
}

fn push_tail(values: &mut Vec<f64>, max_len: usize, value: f64) {
    if max_len == 0 {
        return;
    }
    if values.len() == max_len {
        values.rotate_left(1);
        if let Some(last) = values.last_mut() {
            *last = value;
        }
    } else {
        values.push(value);
    }
}

fn arima_feature(values: &[f64], residuals: &[f64], idx: usize, p: usize, col: usize) -> f64 {
    if col == 0 {
        1.0
    } else if col <= p {
        values[idx - col]
    } else {
        residuals[idx - (col - p)]
    }
}

fn solve_arima_normal_equations(
    mut matrix: [[f64; MAX_ARIMA_COLUMNS]; MAX_ARIMA_COLUMNS],
    mut rhs: [f64; MAX_ARIMA_COLUMNS],
    n: usize,
) -> Option<Vec<f64>> {
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if matrix[row][pivot_idx].abs() > matrix[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if matrix[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_idx, pivot_row);
        rhs.swap(pivot_idx, pivot_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for value in matrix[pivot_idx].iter_mut().take(n).skip(pivot_idx) {
            *value /= pivot;
        }
        rhs[pivot_idx] /= pivot;
        let pivot_tail = matrix[pivot_idx];
        let pivot_rhs = rhs[pivot_idx];

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = matrix[row][pivot_idx];
            for (col, pivot_cell) in pivot_tail.iter().enumerate().take(n).skip(pivot_idx) {
                matrix[row][col] -= factor * pivot_cell;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    Some(rhs[..n].to_vec())
}

fn fit_arima_components(values: &[f64], p: usize, q: usize) -> Result<ArimaComponents> {
    let mut residuals = vec![0.0; values.len()];
    let mut intercept = values.iter().sum::<f64>() / values.len() as f64;
    let mut ar_coefficients = vec![0.0; p];
    let mut ma_coefficients = vec![0.0; q];
    let iterations = if q == 0 { 1 } else { 6 };
    for _ in 0..iterations {
        let (next_intercept, next_ar, next_ma) =
            fit_arima_coefficients_once(values, &residuals, p, q)?;
        intercept = next_intercept;
        ar_coefficients = next_ar;
        ma_coefficients = next_ma;
        let (fitted, next_residuals) = fitted_arima_values(
            values,
            intercept,
            &ar_coefficients,
            &ma_coefficients,
            &residuals,
        );
        let _ = fitted;
        residuals = next_residuals;
    }
    let (fitted, residuals) = fitted_arima_values(
        values,
        intercept,
        &ar_coefficients,
        &ma_coefficients,
        &residuals,
    );
    Ok((
        intercept,
        ar_coefficients,
        ma_coefficients,
        fitted,
        residuals,
    ))
}

fn fit_arima_coefficients_once(
    values: &[f64],
    residuals: &[f64],
    p: usize,
    q: usize,
) -> Result<(f64, Vec<f64>, Vec<f64>)> {
    if p == 0 && q == 0 {
        return Ok((
            values.iter().sum::<f64>() / values.len() as f64,
            Vec::new(),
            Vec::new(),
        ));
    }
    let cols = p + q + 1;
    let mut xtx = [[0.0; MAX_ARIMA_COLUMNS]; MAX_ARIMA_COLUMNS];
    let mut xty = [0.0; MAX_ARIMA_COLUMNS];
    let start = p.max(q);
    for idx in start..values.len() {
        for row in 0..cols {
            let row_value = arima_feature(values, residuals, idx, p, row);
            xty[row] += row_value * values[idx];
            for (col, cell) in xtx[row].iter_mut().enumerate().take(cols) {
                *cell += row_value * arima_feature(values, residuals, idx, p, col);
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().take(cols) {
        row[idx] += 1.0e-8;
    }
    let solution = solve_arima_normal_equations(xtx, xty, cols).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "could not solve ARIMA normal equations for coefficient fit".to_string(),
        )
    })?;
    Ok((
        solution[0],
        solution[1..=p].to_vec(),
        solution[(p + 1)..].to_vec(),
    ))
}

fn fitted_arima_values(
    values: &[f64],
    intercept: f64,
    ar_coefficients: &[f64],
    ma_coefficients: &[f64],
    residual_history: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let p = ar_coefficients.len();
    let q = ma_coefficients.len();
    let mut fitted = Vec::with_capacity(values.len());
    let mut residuals = vec![0.0; values.len()];
    for idx in 0..values.len() {
        let start = p.max(q);
        let mean = if idx < start {
            values[idx]
        } else {
            let mut mean = intercept;
            for (coef_idx, coef) in ar_coefficients.iter().enumerate() {
                mean += coef * values[idx - coef_idx - 1];
            }
            for (coef_idx, coef) in ma_coefficients.iter().enumerate() {
                mean += coef * residual_history[idx - coef_idx - 1];
            }
            mean
        };
        fitted.push(mean);
        if idx >= start {
            residuals[idx] = values[idx] - mean;
        }
    }
    (fitted, residuals)
}

fn forecast_arima_next(
    history: &[f64],
    residuals: &[f64],
    intercept: f64,
    ar_coefficients: &[f64],
    ma_coefficients: &[f64],
) -> f64 {
    let mut forecast = intercept;
    for (idx, coef) in ar_coefficients.iter().enumerate() {
        forecast += coef * history[history.len() - idx - 1];
    }
    for (idx, coef) in ma_coefficients.iter().enumerate() {
        forecast += coef * residuals[residuals.len() - idx - 1];
    }
    forecast
}

fn ar_recursion_is_stable(ar_coefficients: &[f64]) -> bool {
    if ar_coefficients
        .iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        return false;
    }
    let mut coefficients = ar_coefficients.to_vec();
    while let Some(&reflection) = coefficients.last() {
        if reflection.abs() >= 1.0 - 64.0 * f64::EPSILON {
            return false;
        }
        let denominator = 1.0 - reflection * reflection;
        let order = coefficients.len();
        let reduced = (0..order.saturating_sub(1))
            .map(|idx| {
                (coefficients[idx] + reflection * coefficients[order - idx - 2]) / denominator
            })
            .collect::<Vec<_>>();
        if reduced.iter().any(|coefficient| !coefficient.is_finite()) {
            return false;
        }
        coefficients = reduced;
    }
    true
}

fn ma_recursion_is_invertible(ma_coefficients: &[f64]) -> bool {
    let equivalent_ar = ma_coefficients
        .iter()
        .map(|coefficient| -*coefficient)
        .collect::<Vec<_>>();
    ar_recursion_is_stable(&equivalent_ar)
}

fn undifference_fitted_values(values: &[f64], fitted_diff: &[f64], d: usize) -> Vec<f64> {
    match d {
        0 => fitted_diff.to_vec(),
        1 => {
            let mut fitted = vec![values[0]];
            for idx in 1..values.len() {
                fitted.push(values[idx - 1] + fitted_diff[idx - 1]);
            }
            fitted
        }
        2 => {
            let first_diff = values
                .windows(2)
                .map(|window| window[1] - window[0])
                .collect::<Vec<_>>();
            let mut fitted = vec![values[0], values[1]];
            for idx in 2..values.len() {
                fitted.push(values[idx - 1] + first_diff[idx - 2] + fitted_diff[idx - 2]);
            }
            fitted
        }
        _ => values.to_vec(),
    }
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if matrix[row][pivot_idx].abs() > matrix[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if matrix[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_idx, pivot_row);
        rhs.swap(pivot_idx, pivot_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for cell in matrix[pivot_idx].iter_mut().take(n).skip(pivot_idx) {
            *cell /= pivot;
        }
        rhs[pivot_idx] /= pivot;
        let pivot_tail = matrix[pivot_idx][pivot_idx..n].to_vec();

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = matrix[row][pivot_idx];
            for (cell, pivot_cell) in matrix[row]
                .iter_mut()
                .take(n)
                .skip(pivot_idx)
                .zip(pivot_tail.iter())
            {
                *cell -= factor * pivot_cell;
            }
            rhs[row] -= factor * rhs[pivot_idx];
        }
    }
    Some(rhs)
}

fn deseasonalize(
    series_id: &str,
    values: &[f64],
    seasonality: Option<ThetaSeasonality>,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let Some(seasonality) = seasonality else {
        return Ok((values.to_vec(), None));
    };
    if values.len() < seasonality.season_length * 2 {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} requires at least two full seasonal cycles for theta seasonality"
        )));
    }
    if seasonality.kind == ThetaSeasonalityKind::Multiplicative
        && values.iter().any(|value| *value <= 0.0)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} uses multiplicative seasonality but contains non-positive values"
        )));
    }

    let mut pattern = vec![0.0; seasonality.season_length];
    let mut counts = vec![0usize; seasonality.season_length];
    for (idx, value) in values.iter().enumerate() {
        let season_idx = idx % seasonality.season_length;
        pattern[season_idx] += *value;
        counts[season_idx] += 1;
    }
    for (slot, count) in pattern.iter_mut().zip(counts) {
        *slot /= count as f64;
    }

    match seasonality.kind {
        ThetaSeasonalityKind::Additive => {
            let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
            for slot in &mut pattern {
                *slot -= mean;
            }
            let adjusted = values
                .iter()
                .enumerate()
                .map(|(idx, value)| value - pattern[idx % pattern.len()])
                .collect();
            Ok((adjusted, Some(pattern)))
        }
        ThetaSeasonalityKind::Multiplicative => {
            let series_mean = values.iter().sum::<f64>() / values.len() as f64;
            for slot in &mut pattern {
                *slot /= series_mean;
            }
            let pattern_mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
            for slot in &mut pattern {
                *slot /= pattern_mean;
            }
            let adjusted = values
                .iter()
                .enumerate()
                .map(|(idx, value)| value / pattern[idx % pattern.len()])
                .collect();
            Ok((adjusted, Some(pattern)))
        }
    }
}

fn fit_theta_component(values: &[f64], theta: f64, alpha: f64) -> ThetaComponent {
    let slope = linear_slope(values);
    let levels = ses_one_step_levels(values, alpha);
    let last_level = alpha * values[values.len() - 1] + (1.0 - alpha) * levels[values.len() - 1];
    ThetaComponent {
        last_level,
        slope,
        theta,
        alpha,
        n_obs: values.len(),
    }
}

fn fitted_theta_values(values: &[f64], theta: f64, alpha: f64) -> Vec<f64> {
    let levels = ses_one_step_levels(values, alpha);
    levels
        .iter()
        .enumerate()
        .map(|(idx, level)| {
            if idx == 0 {
                values[0]
            } else {
                let slope = linear_slope(&values[..idx]);
                let adjustment = 1.0 / alpha - (1.0 - alpha).powf(idx as f64) / alpha;
                level + (1.0 - 1.0 / theta) * slope * adjustment
            }
        })
        .collect()
}

fn ses_one_step_levels(values: &[f64], alpha: f64) -> Vec<f64> {
    let mut levels = Vec::with_capacity(values.len());
    levels.push(values[0]);
    for idx in 1..values.len() {
        levels.push(alpha * values[idx - 1] + (1.0 - alpha) * levels[idx - 1]);
    }
    levels
}

fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let x_mean = (values.len() - 1) as f64 / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (idx, value) in values.iter().enumerate() {
        let x_delta = idx as f64 - x_mean;
        numerator += x_delta * (value - y_mean);
        denominator += x_delta * x_delta;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn forecast_theta_component(component: &ThetaComponent, step: usize) -> f64 {
    let offset = (step - 1) as f64 + 1.0 / component.alpha
        - (1.0 - component.alpha).powf(component.n_obs as f64) / component.alpha;
    let drift = (1.0 - 1.0 / component.theta) * component.slope * offset;
    component.last_level + drift
}

fn reseasonalize_value(
    value: f64,
    position: usize,
    seasonality: Option<ThetaSeasonality>,
    pattern: Option<&[f64]>,
) -> Result<f64> {
    let Some(seasonality) = seasonality else {
        return Ok(value);
    };
    let pattern = pattern.ok_or_else(|| {
        CartoBoostError::InvalidInput("theta seasonal pattern is missing".to_string())
    })?;
    let seasonal = pattern[position % seasonality.season_length];
    match seasonality.kind {
        ThetaSeasonalityKind::Additive => Ok(value + seasonal),
        ThetaSeasonalityKind::Multiplicative => Ok(value * seasonal),
    }
}

fn validate_spatial_piecewise_kriging_config(config: &SpatialPiecewiseKrigingConfig) -> Result<()> {
    config.kriging_config.validate()?;
    if config.coordinates.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "spatial piecewise kriging coordinates must not be empty".to_string(),
        ));
    }
    for (series_id, (x, y)) in &config.coordinates {
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "spatial piecewise kriging series ids must not be empty".to_string(),
            ));
        }
        if !x.is_finite() || !y.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "spatial piecewise kriging coordinate for series {series_id} must be finite"
            )));
        }
    }
    if !config.residual_shrinkage.is_finite()
        || config.residual_shrinkage < 0.0
        || config.residual_shrinkage > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "residual_shrinkage must be finite and in [0, 1]".to_string(),
        ));
    }
    if config.uses_kriged_regressors() && config.spatial_regressors.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriged_regressors and hybrid modes require at least one spatial_regressor".to_string(),
        ));
    }
    let spatial_regressors = config
        .spatial_regressors
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if spatial_regressors.len() != config.spatial_regressors.len() {
        return Err(CartoBoostError::InvalidInput(
            "spatial_regressors must not contain duplicates".to_string(),
        ));
    }
    for regressor in &config.spatial_regressors {
        if regressor.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "spatial_regressors must not contain empty names".to_string(),
            ));
        }
        if config
            .piecewise_config
            .future_regressors
            .contains_key(regressor)
            || config
                .piecewise_config
                .future_regressors_by_series
                .values()
                .any(|values| values.contains_key(regressor))
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "future spatial regressor {regressor:?} would leak future observations"
            )));
        }
    }
    validate_piecewise_linear_seasonal_config(&config.piecewise_config)?;
    Ok(())
}

fn validate_spatial_piecewise_frame(
    frame: &ForecastFrame,
    config: &SpatialPiecewiseKrigingConfig,
) -> Result<()> {
    validate_common_spatial_cutoff(frame, "spatial_piecewise_kriging")?;
    for series_id in frame.series_ids() {
        if !config.coordinates.contains_key(&series_id) {
            return Err(CartoBoostError::InvalidInput(format!(
                "missing spatial piecewise kriging coordinate for series {series_id}"
            )));
        }
    }
    Ok(())
}

fn validate_common_spatial_cutoff(frame: &ForecastFrame, model_name: &str) -> Result<()> {
    let mut expected = None;
    for series_id in frame.series_ids() {
        let cutoff = frame
            .rows_for_series(&series_id)
            .last()
            .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
            .timestamp;
        match expected {
            None => expected = Some(cutoff),
            Some(expected_cutoff) if cutoff != expected_cutoff => {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{model_name} requires a common panel cutoff timestamp; series {series_id} ends at {cutoff}, expected {expected_cutoff}"
                )));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

fn spatial_piecewise_base_config(
    config: &SpatialPiecewiseKrigingConfig,
) -> PiecewiseLinearSeasonalConfig {
    let mut piecewise_config = config.piecewise_config.clone();
    if config.uses_kriged_regressors() {
        for regressor in &config.spatial_regressors {
            if !piecewise_config.extra_regressors.contains(regressor) {
                piecewise_config.extra_regressors.push(regressor.clone());
            }
        }
    }
    piecewise_config
}

fn kriged_regressor_frame(
    frame: &ForecastFrame,
    config: &SpatialPiecewiseKrigingConfig,
    backend: &BackendSelection,
) -> Result<ForecastFrame> {
    let mut rows = frame.rows().to_vec();
    let mut rows_by_timestamp: BTreeMap<chrono::NaiveDateTime, Vec<usize>> = BTreeMap::new();
    for (idx, row) in rows.iter().enumerate() {
        rows_by_timestamp
            .entry(row.timestamp)
            .or_default()
            .push(idx);
    }
    for regressor in &config.spatial_regressors {
        for indices in rows_by_timestamp.values() {
            let observations = indices
                .iter()
                .filter_map(|idx| {
                    let row = &rows[*idx];
                    row.covariates.get(regressor).map(|value| (row, *value))
                })
                .map(|(row, value)| {
                    let coord = config.coordinates.get(&row.series_id).ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing spatial piecewise kriging coordinate for series {}",
                            row.series_id
                        ))
                    })?;
                    Ok(KrigingObservation {
                        x: coord.0,
                        y: coord.1,
                        value,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if observations.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "spatial regressor {regressor:?} has no observed cutoff-safe values"
                )));
            }
            let targets = indices
                .iter()
                .map(|idx| {
                    let row = &rows[*idx];
                    config
                        .coordinates
                        .get(&row.series_id)
                        .copied()
                        .ok_or_else(|| {
                            CartoBoostError::InvalidInput(format!(
                                "missing spatial piecewise kriging coordinate for series {}",
                                row.series_id
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let predictions = ordinary_kriging_predict_many_with_backend(
                &observations,
                &targets,
                config.kriging_config,
                Some(&backend.selected),
            )?;
            for (idx, prediction) in indices.iter().zip(predictions) {
                rows[*idx]
                    .covariates
                    .insert(regressor.clone(), prediction.mean);
            }
        }
    }
    let mut metadata = frame.metadata().clone();
    for regressor in &config.spatial_regressors {
        if !metadata.known_future_covariates.contains(regressor) {
            metadata.known_future_covariates.push(regressor.clone());
        }
    }
    ForecastFrame::with_metadata(rows, frame.frequency(), metadata)
}

fn kriging_config_metadata(config: OrdinaryKrigingConfig) -> Value {
    json!({
        "range": config.range,
        "nugget": config.nugget,
        "sill": config.sill,
        "variogram_model": format!("{:?}", config.variogram_model).to_lowercase(),
        "drift": format!("{:?}", config.drift).to_lowercase(),
        "anisotropy_angle_degrees": config.anisotropy_angle_degrees,
        "anisotropy_scaling": config.anisotropy_scaling,
        "max_neighbors": config.max_neighbors,
        "min_neighbors": config.min_neighbors,
        "max_distance": config.max_distance,
    })
}

fn is_neighbor_rule_error(error: &CartoBoostError) -> bool {
    error.to_string().contains("neighbors")
}

fn residual_level_variance(levels: &BTreeMap<String, f64>) -> f64 {
    if levels.is_empty() {
        return 0.0;
    }
    let mean = levels.values().sum::<f64>() / levels.len() as f64;
    levels
        .values()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / levels.len() as f64
}

fn prediction_lookup_key(
    series_id: &str,
    timestamp: chrono::NaiveDateTime,
    horizon: usize,
) -> String {
    format!(
        "{}\x1f{}\x1f{}",
        series_id,
        timestamp.format("%Y-%m-%dT%H:%M:%S"),
        horizon
    )
}

fn component_record_key(record: &Value) -> String {
    let series_id = record
        .get("series_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let timestamp = record
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let horizon = record.get("horizon").and_then(Value::as_u64).unwrap_or(0);
    format!("{series_id}\x1f{timestamp}\x1f{horizon}")
}

fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_local_history_requirement(
    fitted: &FittedLocalState,
    required: usize,
    model_name: &str,
    requirement: &str,
) -> Result<()> {
    for (series_id, history) in &fitted.history_by_series {
        if history.len() < required {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has {} rows; {model_name} requires at least {required} rows for {requirement}",
                history.len()
            )));
        }
    }
    Ok(())
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

