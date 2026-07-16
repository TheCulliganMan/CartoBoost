fn build_forecaster(
    model: &str,
    options: &BrowserForecastOptions,
    frame: &ForecastFrame,
    horizon: usize,
) -> Result<Box<dyn Forecaster>> {
    let normalized = model.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "naive" => Ok(Box::new(NaiveForecaster::new())),
        "seasonal_naive" => Ok(Box::new(SeasonalNaiveForecaster::new(
            options.season_length.unwrap_or(7),
        )?)),
        "window_average" => Ok(Box::new(WindowAverageForecaster::new(
            options.window_size.unwrap_or(7),
        )?)),
        "seasonal_window_average" => Ok(Box::new(SeasonalWindowAverageForecaster::new(
            options.season_length.unwrap_or(7),
            options.window_count.unwrap_or(3),
        )?)),
        "theta" => Ok(Box::new(ThetaForecaster::with_seasonality(
            options.theta.unwrap_or(2.0),
            options.alpha.unwrap_or(0.2),
            theta_seasonality(options)?,
        )?)),
        "optimized_theta" => Ok(Box::new(OptimizedThetaForecaster::with_seasonality(
            options
                .theta_grid
                .clone()
                .unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
            options
                .alpha_grid
                .clone()
                .unwrap_or_else(|| vec![0.1, 0.2, 0.3, 0.5, 0.8]),
            theta_seasonality(options)?,
        )?)),
        "ets" => Ok(Box::new(ETSForecaster::with_additive_damped_trend(
            options.alpha.unwrap_or(0.3),
            options.beta.unwrap_or(0.1),
            options.gamma,
            None,
            options.damping_phi.unwrap_or(1.0),
        )?)),
        "seasonal_ets" => Ok(Box::new(ETSForecaster::with_additive_damped_trend(
            options.alpha.unwrap_or(0.3),
            options.beta.unwrap_or(0.1),
            Some(options.gamma.unwrap_or(0.1)),
            Some(options.season_length.unwrap_or(7)),
            options.damping_phi.unwrap_or(1.0),
        )?)),
        "auto_ets" => Ok(Box::new(AutoETSForecaster::new(options.season_length)?)),
        "arima" => Ok(Box::new(
            cartoboost_core::forecasting::ArimaForecaster::new(
                options.max_p.unwrap_or(1),
                options.max_d.unwrap_or(1),
                options.max_q.unwrap_or(0),
            )?,
        )),
        "auto_arima" => Ok(Box::new(AutoARIMAForecaster::with_max_order(
            options.max_p.unwrap_or(2),
            options.max_d.unwrap_or(1),
            options.max_q.unwrap_or(1),
        )?)),
        "kalman" => Ok(Box::new(KalmanForecaster::new(
            options.level_process_variance.unwrap_or(0.05),
            options.trend_process_variance.unwrap_or(0.005),
            options.observation_variance.unwrap_or(1.0),
        )?)),
        "local_level_kalman" => Ok(Box::new(LocalLevelKalmanForecaster::new(
            options.level_process_variance.unwrap_or(0.05),
            options.observation_variance.unwrap_or(1.0),
        )?)),
        "auto_kalman" => Ok(Box::new(AutoKalmanForecaster::new()?)),
        "auto_local_level_kalman" => Ok(Box::new(AutoLocalLevelKalmanForecaster::new()?)),
        "kriging" => Ok(Box::new(KrigingForecaster::new(
            coordinates_from_frame(frame, options)?,
            options.kriging_range.unwrap_or(1.0),
            options.kriging_nugget.unwrap_or(1e-6),
        )?)),
        "spatial_piecewise_kriging" => Ok(Box::new(SpatialPiecewiseKrigingForecaster::new(
            SpatialPiecewiseKrigingConfig {
                coordinates: coordinates_from_frame(frame, options)?,
                mode: spatial_piecewise_kriging_mode(options)?,
                piecewise_config: piecewise_linear_seasonal_config(options)?,
                kriging_config: cartoboost_core::utilities::OrdinaryKrigingConfig::new(
                    options.kriging_range.unwrap_or(1.0),
                    options.kriging_nugget.unwrap_or(1e-6),
                )?,
                spatial_regressors: options.spatial_regressors.clone().unwrap_or_default(),
                residual_shrinkage: options.residual_shrinkage.unwrap_or(1.0),
                allow_neighbor_fallback: options.allow_neighbor_fallback.unwrap_or(false),
            },
        )?)),
        "piecewise_linear_seasonal" => Ok(Box::new(PiecewiseLinearSeasonalForecaster::new(
            piecewise_linear_seasonal_config(options)?,
        )?)),
        "neural_panel" => Ok(Box::new(
            NeuralPanelForecaster::new(neural_panel_config(options, horizon)?)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )),
        "nbeats" => Ok(Box::new(
            NBeatsForecaster::new(nbeats_config(options))
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )),
        "nhits" => Ok(Box::new(
            NHiTSForecaster::new(nhits_config(options))
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )),
        "intermittent_demand" => {
            let config = IntermittentDemandConfig {
                alpha: options.alpha.unwrap_or(0.2),
                beta: options.beta.unwrap_or(0.2),
                validation_window: options.validation_window,
                ..IntermittentDemandConfig::default()
            };
            Ok(Box::new(IntermittentDemandForecaster::new(config)?))
        }
        "croston" => Ok(Box::new(CrostonForecaster::new(
            options.alpha.unwrap_or(0.2),
        )?)),
        "sba" => Ok(Box::new(SbaForecaster::new(options.alpha.unwrap_or(0.2))?)),
        "tsb" => Ok(Box::new(TsbForecaster::new(
            options.alpha.unwrap_or(0.2),
            options.beta.unwrap_or(0.2),
        )?)),
        "classical_expert_bank" => Ok(Box::new(ClassicalExpertBank::default_for_season_length(
            options.season_length.unwrap_or(7),
        )?)),
        "autostats_bank" => Ok(Box::new(AutoStatsBank::with_validation_window(
            options.season_length.unwrap_or(7),
            options.validation_window,
        )?)),
        "stl_cartoboost" => Ok(Box::new(STLCartoBoostForecaster::new(
            options.season_length.unwrap_or(7),
        )?)),
        "mstl_cartoboost" => Ok(Box::new(MSTLCartoBoostForecaster::new(
            options
                .mstl_season_lengths
                .clone()
                .unwrap_or_else(|| vec![options.season_length.unwrap_or(7)]),
        )?)),
        "cartoboost_lag" => Ok(Box::new(CartoBoostLagForecaster::new(
            lag_config(options),
            booster_config(options),
        )?)),
        "cartoboost_direct" => Ok(Box::new(BrowserDirectForecaster::new(
            lag_config(options),
            booster_config(options),
            horizon,
        )?)),
        "rectified_recursive" => Ok(Box::new(BrowserRectifiedRecursiveForecaster::new(
            lag_config(options),
            booster_config(options),
            horizon,
        )?)),
        "lag_plus" => Ok(Box::new(LagPlusForecaster::new(LagPlusConfig::new(
            lag_config(options),
            booster_config(options),
        ))?)),
        "auto_forecast" => {
            let mut config = AutoForecastConfig {
                lag_config: lag_config(options),
                booster_config: booster_config(options),
                ..AutoForecastConfig::default()
            };
            if let Some(season_length) = options.season_length {
                config.season_length = season_length;
            }
            if let Some(validation_window) = options.validation_window {
                config.validation_window = Some(validation_window);
            }
            config.max_candidate_count = options.max_auto_candidate_count;
            config.max_direct_horizon = options.max_direct_horizon.unwrap_or(horizon);
            Ok(Box::new(AutoForecastModel::new(config)?))
        }
        "scaled_cartoboost_lag" => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new(
                lag_config(options),
                booster_config(options),
            )?),
            1e-6,
            "scaled_cartoboost_lag",
        )?)),
        "log1p_cartoboost_lag" => Ok(Box::new(Log1pForecaster::new(
            Box::new(CartoBoostLagForecaster::new(
                lag_config(options),
                booster_config(options),
            )?),
            "log1p_cartoboost_lag",
        ))),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported browser forecast model {other:?}"
        ))),
    }
}

struct BrowserDirectForecaster {
    inner: CartoBoostDirectForecaster,
    fit_horizon: usize,
}

impl BrowserDirectForecaster {
    fn new(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        fit_horizon: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: CartoBoostDirectForecaster::new(lag_config, booster_config)?,
            fit_horizon,
        })
    }
}

impl Forecaster for BrowserDirectForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.inner.fit_horizon(frame, self.fit_horizon)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn metadata(&self) -> Value {
        self.inner.metadata()
    }
}

struct BrowserRectifiedRecursiveForecaster {
    inner: RectifiedRecursiveForecaster,
    fit_horizon: usize,
}

impl BrowserRectifiedRecursiveForecaster {
    fn new(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        fit_horizon: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: RectifiedRecursiveForecaster::new(lag_config, booster_config)?,
            fit_horizon,
        })
    }
}

impl Forecaster for BrowserRectifiedRecursiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.inner.fit_horizon(frame, self.fit_horizon)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn metadata(&self) -> Value {
        self.inner.metadata()
    }
}

fn theta_seasonality(options: &BrowserForecastOptions) -> Result<Option<ThetaSeasonality>> {
    let Some(kind) = options.theta_seasonality.as_deref() else {
        return Ok(None);
    };
    let season_length = options.season_length.unwrap_or(7);
    match kind.trim().to_ascii_lowercase().as_str() {
        "" | "none" => Ok(None),
        "additive" => ThetaSeasonality::additive(season_length).map(Some),
        "multiplicative" => ThetaSeasonality::multiplicative(season_length).map(Some),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported theta seasonality {other:?}"
        ))),
    }
}

fn is_piecewise_linear_seasonal_model(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().replace('-', "_").as_str(),
        "piecewise_linear_seasonal"
    )
}

fn spatial_piecewise_kriging_mode(
    options: &BrowserForecastOptions,
) -> Result<SpatialPiecewiseKrigingMode> {
    let value = options
        .spatial_kriging_mode
        .as_deref()
        .unwrap_or("residual_kriging")
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_");
    match value.as_str() {
        "kriged_regressors" | "regressors" => Ok(SpatialPiecewiseKrigingMode::KrigedRegressors),
        "residual_kriging" | "residual" => Ok(SpatialPiecewiseKrigingMode::ResidualKriging),
        "hybrid" => Ok(SpatialPiecewiseKrigingMode::Hybrid),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported spatial_piecewise_kriging mode {other:?}"
        ))),
    }
}

fn piecewise_linear_seasonal_config(
    options: &BrowserForecastOptions,
) -> Result<PiecewiseLinearSeasonalConfig> {
    let mut config = PiecewiseLinearSeasonalConfig::default();
    if options.mcmc_samples.unwrap_or(0) > 0 {
        return Err(CartoBoostError::InvalidInput(
            "mcmc_samples are not supported by the Rust-native piecewise seasonal model; use uncertainty_samples for deterministic native intervals".to_string(),
        ));
    }
    if let Some(growth) = options.growth.as_deref() {
        config.growth = match growth.trim().to_ascii_lowercase().as_str() {
            "" | "linear" => PiecewiseLinearGrowth::Linear,
            "flat" => PiecewiseLinearGrowth::Flat,
            "logistic" => PiecewiseLinearGrowth::Logistic,
            other => {
                return Err(CartoBoostError::InvalidInput(format!(
                    "unsupported piecewise seasonal growth {other:?}"
                )))
            }
        };
    }
    if let Some(mode) = options
        .seasonality_mode
        .as_deref()
        .or(options.component_mode.as_deref())
    {
        config.component_mode = piecewise_component_mode(mode)?;
    }
    if let Some(loss) = options.fit_loss.as_deref() {
        config.fit_loss = piecewise_fit_loss(loss)?;
    }
    if let Some(delta) = options.huber_delta {
        config.huber_delta = delta;
    }
    if let Some(iterations) = options.irls_iterations {
        config.irls_iterations = iterations;
    }
    if let Some(changepoints) = options.n_changepoints.or(options.changepoints) {
        config.changepoints = changepoints;
    }
    if let Some(changepoint_range) = options.changepoint_range {
        config.changepoint_range = changepoint_range;
    }
    if let Some(timestamps) = &options.changepoint_timestamps {
        config.changepoint_timestamps = timestamps
            .iter()
            .map(|timestamp| cartoboost_core::forecasting::parse_forecast_timestamp(timestamp))
            .collect::<Result<Vec<_>>>()?;
    }
    if let Some(order) = options.yearly_fourier_order {
        config.yearly_fourier_order = order;
    }
    if let Some(order) = options.weekly_fourier_order {
        config.weekly_fourier_order = order;
    }
    if let Some(order) = options.daily_fourier_order {
        config.daily_fourier_order = order;
    }
    if let Some(value) = options.auto_yearly_seasonality {
        config.auto_yearly_seasonality = value;
    }
    if let Some(value) = options.auto_weekly_seasonality {
        config.auto_weekly_seasonality = value;
    }
    if let Some(value) = options.auto_daily_seasonality {
        config.auto_daily_seasonality = value;
    }
    if let Some(seasonalities) = &options.custom_seasonalities {
        config.custom_seasonalities = seasonalities
            .iter()
            .map(|seasonality| {
                Ok(PiecewiseLinearSeasonality {
                    name: seasonality.name.clone(),
                    period_days: seasonality.period_days,
                    fourier_order: seasonality.fourier_order,
                    mode: seasonality
                        .mode
                        .as_deref()
                        .map(piecewise_component_mode)
                        .transpose()?,
                    condition_name: seasonality.condition_name.clone(),
                    l2_regularization: seasonality.l2_regularization,
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    if let Some(value) = options.changepoint_l2_regularization {
        config.changepoint_l2_regularization = value;
    }
    if let Some(value) = options.changepoint_l1_regularization {
        config.changepoint_l1_regularization = value;
    }
    if let Some(value) = options.changepoint_prior_scale {
        config.changepoint_l1_regularization = piecewise_prior_scale_to_l1(value)?;
    }
    if let Some(value) = options.seasonality_l2_regularization {
        config.seasonality_l2_regularization = value;
    }
    if let Some(value) = options.seasonality_prior_scale {
        config.seasonality_l2_regularization = piecewise_prior_scale_to_l2(value)?;
    }
    if let Some(value) = options.yearly_l2_regularization {
        config.yearly_l2_regularization = Some(value);
    }
    if let Some(value) = options.weekly_l2_regularization {
        config.weekly_l2_regularization = Some(value);
    }
    if let Some(value) = options.daily_l2_regularization {
        config.daily_l2_regularization = Some(value);
    }
    if let Some(value) = options.event_l2_regularization {
        config.event_l2_regularization = value;
    }
    if let Some(value) = options.holidays_prior_scale {
        config.event_l2_regularization = piecewise_prior_scale_to_l2(value)?;
    }
    if let Some(value) = options.regressor_l2_regularization {
        config.regressor_l2_regularization = value;
    }
    if let Some(values) = &options.event_l2_regularization_by_name {
        config.event_l2_regularization_by_name = values.clone();
    }
    if let Some(values) = &options.regressor_l2_regularization_by_name {
        config.regressor_l2_regularization_by_name = values.clone();
    }
    if let Some(events) = &options.events {
        config.events = events
            .iter()
            .map(|event| {
                Ok(PiecewiseLinearEvent {
                    name: event.name.clone(),
                    timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(
                        &event.timestamp,
                    )?,
                    lower_window: event.lower_window.unwrap_or(0),
                    upper_window: event.upper_window.unwrap_or(0),
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    if let Some(holidays) = &options.holidays {
        let mut holiday_events = Vec::with_capacity(holidays.len());
        for holiday in holidays {
            let timestamp = cartoboost_core::forecasting::parse_forecast_timestamp(&holiday.ds)?;
            holiday_events.push(PiecewiseLinearEvent {
                name: holiday.holiday.clone(),
                timestamp,
                lower_window: holiday.lower_window.unwrap_or(0),
                upper_window: holiday.upper_window.unwrap_or(0),
            });
            if let Some(scale) = holiday.prior_scale {
                config
                    .event_l2_regularization_by_name
                    .insert(holiday.holiday.clone(), piecewise_prior_scale_to_l2(scale)?);
            }
        }
        config.events.extend(holiday_events);
    }
    if let Some(mode) = options
        .holidays_mode
        .as_deref()
        .or(options.event_mode.as_deref())
    {
        config.event_mode = Some(piecewise_component_mode(mode)?);
    }
    if let Some(regressors) = &options.extra_regressors {
        config.extra_regressors = regressors.clone();
    }
    if let Some(regressor_modes) = &options.regressor_modes {
        config.regressor_modes = regressor_modes
            .iter()
            .map(|(name, mode)| Ok((name.clone(), piecewise_component_mode(mode)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
    }
    if let Some(constraints) = &options.extra_regressor_monotonic_constraints {
        config.extra_regressor_monotonic_constraints = constraints.clone();
    }
    if let Some(value) = options.regressor_standardization.as_deref() {
        config.regressor_standardization = piecewise_regressor_standardization(value)?;
    }
    if let Some(future_regressors) = &options.future_regressors {
        config.future_regressors = future_regressors.clone();
    }
    if let Some(future_regressors_by_series) = &options.future_regressors_by_series {
        config.future_regressors_by_series = future_regressors_by_series.clone();
    }
    if let Some(trend_adjustments) = &options.trend_adjustments {
        config.trend_adjustments = trend_adjustments.clone();
    }
    if let Some(trend_adjustments_by_series) = &options.trend_adjustments_by_series {
        config.trend_adjustments_by_series = trend_adjustments_by_series.clone();
    }
    if let Some(value) = options.residual_shock_window {
        config.residual_shock_window = value;
    }
    if let Some(value) = options.residual_shock_scale {
        config.residual_shock_scale = value;
    }
    if let Some(value) = options.residual_shock_decay {
        config.residual_shock_decay = value;
    }
    if let Some(levels) = &options.interval_levels {
        config.interval_levels = levels.clone();
    }
    if let Some(width) = options.interval_width {
        if !(0.0..=1.0).contains(&width) || width == 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "interval_width must be in (0, 1]".to_string(),
            ));
        }
        config.interval_levels = vec![width];
    }
    if let Some(levels) = &options.quantile_levels {
        config.quantile_levels = levels.clone();
    }
    if let Some(value) = options.uncertainty_samples {
        config.uncertainty_samples = value;
    }
    if let Some(value) = options.trend_uncertainty_policy.as_deref() {
        config.trend_uncertainty_policy = piecewise_trend_uncertainty_policy(value)?;
    }
    if let Some(value) = options.trend_uncertainty_scale {
        config.trend_uncertainty_scale = value;
    }
    if let Some(value) = options.coefficient_uncertainty_scale {
        config.coefficient_uncertainty_scale = value;
    }
    if let Some(value) = options.uncertainty_seed {
        config.uncertainty_seed = value;
    }
    if let Some(cap) = options.cap {
        config.cap = Some(cap);
    }
    if let Some(floor) = options.floor {
        config.floor = floor;
    }
    if let Some(name) = &options.cap_regressor {
        config.cap_regressor = Some(name.clone());
    }
    if let Some(name) = &options.floor_regressor {
        config.floor_regressor = Some(name.clone());
    }
    Ok(config)
}

fn neural_panel_config(
    options: &BrowserForecastOptions,
    horizon: usize,
) -> Result<NeuralPanelConfig> {
    let mut config = NeuralPanelConfig {
        n_lags: options
            .n_lags
            .or_else(|| {
                options
                    .lags
                    .as_ref()
                    .and_then(|lags| lags.iter().copied().max())
            })
            .unwrap_or(8),
        n_forecasts: options.n_forecasts.unwrap_or(horizon.max(1)),
        quantiles: options.quantile_levels.clone().unwrap_or_else(|| vec![0.5]),
        trend: options
            .growth
            .as_deref()
            .map(neural_panel_trend_mode)
            .transpose()?
            .unwrap_or(NeuralTrendMode::PiecewiseLinear),
        n_changepoints: options
            .n_changepoints
            .or(options.changepoints)
            .unwrap_or(10),
        changepoints_range: options.changepoint_range.unwrap_or(0.8),
        daily_fourier_order: options.daily_fourier_order.unwrap_or(0),
        weekly_fourier_order: options.weekly_fourier_order.unwrap_or(0),
        yearly_fourier_order: options.yearly_fourier_order.unwrap_or(0),
        custom_seasonalities: BTreeMap::new(),
        custom_seasonality_conditions: BTreeMap::new(),
        seasonality_mode: options
            .seasonality_mode
            .as_deref()
            .or(options.component_mode.as_deref())
            .map(neural_panel_component_mode)
            .transpose()?
            .unwrap_or(NeuralComponentMode::Additive),
        events: BTreeMap::new(),
        event_mode: options
            .event_mode
            .as_deref()
            .or(options.holidays_mode.as_deref())
            .map(neural_panel_component_mode)
            .transpose()?
            .unwrap_or(NeuralComponentMode::Additive),
        future_regressors: BTreeMap::new(),
        lagged_regressors: options.lagged_regressors.clone().unwrap_or_default(),
        ar_layers: options.ar_layers.clone().unwrap_or_default(),
        lagged_reg_layers: options.lagged_reg_layers.clone().unwrap_or_default(),
        trend_mode: options
            .trend_mode
            .as_deref()
            .map(neural_panel_global_local_mode)
            .transpose()?
            .unwrap_or(NeuralPanelMode::Global),
        seasonality_global_local: NeuralPanelMode::Global,
        event_global_local: NeuralPanelMode::Global,
        regressor_global_local: NeuralPanelMode::Global,
        local_l2: options.local_l2.unwrap_or(0.0),
        seed: options.uncertainty_seed.unwrap_or(0),
        loss: NeuralPanelLoss::SmoothL1,
        epochs: 80,
        learning_rate: 0.01,
        weight_decay: 0.0,
        newer_sample_weight: false,
        backend: BackendSelection::default(),
    };
    if let Some(seasonalities) = &options.custom_seasonalities {
        config.custom_seasonalities = seasonalities
            .iter()
            .map(|seasonality| {
                (
                    seasonality.name.clone(),
                    (seasonality.period_days * 24.0, seasonality.fourier_order),
                )
            })
            .collect();
        config.custom_seasonality_conditions = seasonalities
            .iter()
            .map(|seasonality| (seasonality.name.clone(), seasonality.condition_name.clone()))
            .collect();
    }
    if let Some(events) = &options.events {
        for event in events {
            let lower = event.lower_window.unwrap_or(0);
            let upper = event.upper_window.unwrap_or(0);
            config
                .events
                .entry(event.name.clone())
                .or_default()
                .extend(lower..=upper);
        }
    }
    if let Some(holidays) = &options.holidays {
        for holiday in holidays {
            let lower = holiday.lower_window.unwrap_or(0);
            let upper = holiday.upper_window.unwrap_or(0);
            config
                .events
                .entry(holiday.holiday.clone())
                .or_default()
                .extend(lower..=upper);
        }
    }
    if let Some(regressors) = &options.extra_regressors {
        for name in regressors {
            let mode = options
                .regressor_modes
                .as_ref()
                .and_then(|modes| modes.get(name))
                .map(|value| neural_panel_component_mode(value))
                .transpose()?
                .unwrap_or(NeuralComponentMode::Additive);
            config.future_regressors.insert(name.clone(), mode);
        }
    }
    config
        .validate()
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
    Ok(config)
}

fn nbeats_config(options: &BrowserForecastOptions) -> NBeatsConfig {
    NBeatsConfig {
        input_size: options.input_size.unwrap_or(8),
        hidden_size: options.hidden_size.unwrap_or(16),
        epochs: options.epochs.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.01),
        ..NBeatsConfig::default()
    }
}

fn nhits_config(options: &BrowserForecastOptions) -> NHiTSConfig {
    NHiTSConfig {
        input_size: options.input_size.unwrap_or(12),
        hidden_size: options.hidden_size.unwrap_or(16),
        epochs: options.epochs.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.01),
        pooling_size: options.pooling_size.unwrap_or(2),
        ..NHiTSConfig::default()
    }
}

fn neural_panel_trend_mode(value: &str) -> Result<NeuralTrendMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "linear" | "piecewise_linear" => Ok(NeuralTrendMode::PiecewiseLinear),
        "off" | "none" | "flat" => Ok(NeuralTrendMode::Off),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported neural_panel trend mode {other:?}"
        ))),
    }
}

fn neural_panel_component_mode(value: &str) -> Result<NeuralComponentMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "additive" => Ok(NeuralComponentMode::Additive),
        "multiplicative" => Ok(NeuralComponentMode::Multiplicative),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported neural_panel component mode {other:?}"
        ))),
    }
}

fn neural_panel_global_local_mode(value: &str) -> Result<NeuralPanelMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "global" => Ok(NeuralPanelMode::Global),
        "local" => Ok(NeuralPanelMode::Local),
        "glocal" => Ok(NeuralPanelMode::Glocal),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported neural_panel global/local mode {other:?}"
        ))),
    }
}

fn piecewise_component_mode(value: &str) -> Result<PiecewiseLinearComponentMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "additive" => Ok(PiecewiseLinearComponentMode::Additive),
        "multiplicative" => Ok(PiecewiseLinearComponentMode::Multiplicative),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal component mode {other:?}"
        ))),
    }
}

fn piecewise_fit_loss(value: &str) -> Result<PiecewiseLinearFitLoss> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "squared" | "l2" | "least_squares" => Ok(PiecewiseLinearFitLoss::Squared),
        "huber" | "robust" => Ok(PiecewiseLinearFitLoss::Huber),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal fit loss {other:?}"
        ))),
    }
}

fn piecewise_prior_scale_to_l2(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "prior scale values must be positive finite numbers".to_string(),
        ));
    }
    Ok(1.0 / (value * value))
}

fn piecewise_prior_scale_to_l1(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "prior scale values must be positive finite numbers".to_string(),
        ));
    }
    Ok(1.0 / value)
}

fn piecewise_regressor_standardization(
    value: &str,
) -> Result<cartoboost_core::forecasting::PiecewiseLinearRegressorStandardization> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "auto" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearRegressorStandardization::Auto)
        }
        "none" | "off" | "false" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearRegressorStandardization::None)
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal regressor standardization {other:?}"
        ))),
    }
}

fn piecewise_trend_uncertainty_policy(
    value: &str,
) -> Result<cartoboost_core::forecasting::PiecewiseLinearTrendUncertaintyPolicy> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "laplace" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearTrendUncertaintyPolicy::Laplace)
        }
        "normal" | "gaussian" => {
            Ok(cartoboost_core::forecasting::PiecewiseLinearTrendUncertaintyPolicy::Normal)
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported piecewise seasonal trend uncertainty policy {other:?}"
        ))),
    }
}

fn default_model() -> String {
    "auto_forecast".to_string()
}

fn booster_config(options: &BrowserForecastOptions) -> BoosterConfig {
    let mut config = BoosterConfig::default();
    if let Some(n_estimators) = options.n_estimators {
        config.n_estimators = n_estimators;
    }
    if let Some(learning_rate) = options.learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(max_depth) = options.max_depth {
        config.max_depth = max_depth;
    }
    if let Some(min_samples_leaf) = options.min_samples_leaf {
        config.min_samples_leaf = min_samples_leaf;
    }
    config
}

fn lag_config(options: &BrowserForecastOptions) -> LagFeatureConfig {
    LagFeatureConfig {
        lags: options
            .lags
            .clone()
            .unwrap_or_else(|| vec![1, 2, 3, options.season_length.unwrap_or(7)]),
        rolling_mean_windows: options
            .rolling_mean_windows
            .clone()
            .unwrap_or_else(|| vec![options.season_length.unwrap_or(7)]),
        rolling_std_windows: options.rolling_std_windows.clone().unwrap_or_default(),
        rolling_min_windows: options.rolling_min_windows.clone().unwrap_or_default(),
        rolling_max_windows: options.rolling_max_windows.clone().unwrap_or_default(),
        difference_lags: options.difference_lags.clone().unwrap_or_default(),
        rolling_trend_windows: options.rolling_trend_windows.clone().unwrap_or_default(),
        calendar_features: calendar_features(options),
        ..LagFeatureConfig::default()
    }
}

fn calendar_features(options: &BrowserForecastOptions) -> Vec<CalendarFeature> {
    let Some(features) = &options.calendar_features else {
        return vec![CalendarFeature::DayOfWeek, CalendarFeature::Month];
    };
    features
        .iter()
        .filter_map(
            |feature| match feature.trim().to_ascii_lowercase().as_str() {
                "day_of_week" | "dow" => Some(CalendarFeature::DayOfWeek),
                "day_of_week_sin" | "dow_sin" => Some(CalendarFeature::DayOfWeekSin),
                "day_of_week_cos" | "dow_cos" => Some(CalendarFeature::DayOfWeekCos),
                "month" => Some(CalendarFeature::Month),
                "month_sin" => Some(CalendarFeature::MonthSin),
                "month_cos" => Some(CalendarFeature::MonthCos),
                "day" => Some(CalendarFeature::Day),
                "day_sin" => Some(CalendarFeature::DaySin),
                "day_cos" => Some(CalendarFeature::DayCos),
                "day_of_year" | "doy" => Some(CalendarFeature::DayOfYear),
                "elapsed_index" => Some(CalendarFeature::ElapsedIndex),
                "elapsed_phase" => Some(CalendarFeature::ElapsedPhase(
                    options.season_length.unwrap_or(7).max(2),
                )),
                _ => None,
            },
        )
        .collect()
}

fn coordinates_from_frame(
    frame: &ForecastFrame,
    options: &BrowserForecastOptions,
) -> Result<BTreeMap<String, (f64, f64)>> {
    let x_name = options
        .coordinate_x
        .as_deref()
        .unwrap_or_else(|| infer_covariate(frame, &["longitude", "lon", "lng", "x"]).unwrap_or(""));
    let y_name = options
        .coordinate_y
        .as_deref()
        .unwrap_or_else(|| infer_covariate(frame, &["latitude", "lat", "y"]).unwrap_or(""));
    if x_name.is_empty() || y_name.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging requires coordinate covariates such as longitude/latitude or x/y".to_string(),
        ));
    }
    let mut coordinates = BTreeMap::new();
    for row in frame.rows() {
        if coordinates.contains_key(&row.series_id) {
            continue;
        }
        let x = row.covariates.get(x_name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing kriging x coordinate covariate {x_name:?}"
            ))
        })?;
        let y = row.covariates.get(y_name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing kriging y coordinate covariate {y_name:?}"
            ))
        })?;
        coordinates.insert(row.series_id.clone(), (x, y));
    }
    Ok(coordinates)
}

fn infer_covariate<'a>(frame: &'a ForecastFrame, names: &[&str]) -> Option<&'a str> {
    let first = frame.rows().first()?;
    for candidate in names {
        if let Some((name, _)) = first
            .covariates
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(candidate))
        {
            return Some(name.as_str());
        }
    }
    None
}

#[allow(dead_code)]
fn _assert_forecast_result_is_serializable(result: &ForecastResult) -> Value {
    result.to_json_value()
}

