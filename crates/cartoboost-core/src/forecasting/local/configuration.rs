impl PiecewiseEventTerm {
    fn label(&self) -> String {
        format!("{}[{:+}]", self.name, self.offset)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PiecewiseLinearSeasonalArtifact {
    kind: String,
    schema_version: u32,
    model: PiecewiseLinearSeasonalForecaster,
}

#[derive(Debug, Clone)]
struct FittedThetaSeries {
    last_timestamp: chrono::NaiveDateTime,
    n_obs: usize,
    component: ThetaComponent,
    seasonal_pattern: Option<Vec<f64>>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ThetaComponent {
    last_level: f64,
    slope: f64,
    theta: f64,
    alpha: f64,
    n_obs: usize,
}

#[derive(Debug, Clone)]
pub struct ThetaValidationScore {
    pub theta: f64,
    pub alpha: f64,
    pub mse: f64,
}

#[derive(Debug, Clone)]
pub struct ArimaValidationScore {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub mse: f64,
    pub ar_stable: bool,
    pub ma_invertible: bool,
    pub stable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ETSParameterSet {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: Option<f64>,
    pub damping_phi: f64,
}

#[derive(Debug, Clone)]
pub struct ETSValidationScore {
    pub params: ETSParameterSet,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanParameterSet {
    pub level_process_variance: f64,
    pub trend_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone)]
pub struct KalmanValidationScore {
    pub params: KalmanParameterSet,
    pub mse: f64,
    pub negative_log_likelihood: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelKalmanParameterSet {
    pub level_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone)]
pub struct LocalLevelKalmanValidationScore {
    pub params: LocalLevelKalmanParameterSet,
    pub mse: f64,
    pub negative_log_likelihood: f64,
}

impl Default for PiecewiseLinearSeasonalConfig {
    fn default() -> Self {
        Self {
            growth: PiecewiseLinearGrowth::Linear,
            component_mode: PiecewiseLinearComponentMode::Additive,
            fit_loss: PiecewiseLinearFitLoss::Squared,
            huber_delta: 1.345,
            irls_iterations: 5,
            changepoints: 25,
            changepoint_range: 1.0,
            changepoint_timestamps: Vec::new(),
            yearly_fourier_order: 0,
            weekly_fourier_order: 3,
            daily_fourier_order: 0,
            auto_yearly_seasonality: true,
            auto_weekly_seasonality: true,
            auto_daily_seasonality: true,
            custom_seasonalities: Vec::new(),
            changepoint_l2_regularization: 0.05,
            changepoint_l1_regularization: 0.0,
            seasonality_l2_regularization: 0.01,
            yearly_l2_regularization: None,
            weekly_l2_regularization: None,
            daily_l2_regularization: None,
            event_l2_regularization: 0.01,
            regressor_l2_regularization: 0.01,
            event_l2_regularization_by_name: BTreeMap::new(),
            regressor_l2_regularization_by_name: BTreeMap::new(),
            events: Vec::new(),
            event_mode: None,
            extra_regressors: Vec::new(),
            regressor_modes: BTreeMap::new(),
            extra_regressor_monotonic_constraints: BTreeMap::new(),
            regressor_standardization: PiecewiseLinearRegressorStandardization::Auto,
            future_regressors: BTreeMap::new(),
            future_regressors_by_series: BTreeMap::new(),
            trend_adjustments: BTreeMap::new(),
            trend_adjustments_by_series: BTreeMap::new(),
            residual_shock_window: 0,
            residual_shock_scale: 0.0,
            residual_shock_decay: 1.0,
            interval_levels: Vec::new(),
            quantile_levels: Vec::new(),
            uncertainty_samples: 0,
            trend_uncertainty_policy: PiecewiseLinearTrendUncertaintyPolicy::Laplace,
            trend_uncertainty_scale: 1.0,
            coefficient_uncertainty_scale: 1.0,
            uncertainty_seed: 0xC4B0_0575_A11C_E123,
            cap: None,
            floor: 0.0,
            cap_regressor: None,
            floor_regressor: None,
        }
    }
}

impl NaiveForecaster {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PiecewiseLinearSeasonalForecaster {
    pub fn new(config: PiecewiseLinearSeasonalConfig) -> Result<Self> {
        validate_piecewise_linear_seasonal_config(&config)?;
        Ok(Self {
            config,
            fitted: None,
        })
    }

    pub fn config(&self) -> &PiecewiseLinearSeasonalConfig {
        &self.config
    }

    pub fn update_config<F>(&mut self, update: F) -> Result<()>
    where
        F: FnOnce(&mut PiecewiseLinearSeasonalConfig),
    {
        let mut config = self.config.clone();
        update(&mut config);
        validate_piecewise_linear_seasonal_config(&config)?;
        self.config = config;
        Ok(())
    }

    pub fn to_json_string(&self) -> Result<String> {
        let artifact = PiecewiseLinearSeasonalArtifact {
            kind: PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND.to_string(),
            schema_version: PIECEWISE_LINEAR_SEASONAL_ARTIFACT_VERSION,
            model: self.clone(),
        };
        serde_json::to_string(&artifact).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal artifact: {err}"
            ))
        })
    }

    pub fn from_json_string(payload: &str) -> Result<Self> {
        let artifact =
            serde_json::from_str::<PiecewiseLinearSeasonalArtifact>(payload).map_err(|err| {
                CartoBoostError::InvalidInput(format!(
                    "failed to parse piecewise linear seasonal artifact: {err}"
                ))
            })?;
        if artifact.kind != PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported piecewise linear seasonal artifact kind {:?}",
                artifact.kind
            )));
        }
        if artifact.schema_version != PIECEWISE_LINEAR_SEASONAL_ARTIFACT_VERSION {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported piecewise linear seasonal artifact schema_version {}",
                artifact.schema_version
            )));
        }
        validate_piecewise_linear_seasonal_config(&artifact.model.config)?;
        Ok(artifact.model)
    }

    pub fn fitted_series_ids(&self) -> Result<Vec<String>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        Ok(fitted.series.keys().cloned().collect())
    }

    pub fn predict_components_json_value(&self, horizon: usize) -> Result<Value> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let records = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                let anchor_timestamp = fitted
                    .anchor_timestamp_by_series
                    .get(series_id)
                    .copied()
                    .unwrap_or(series.last_timestamp);
                (1..=horizon)
                    .map(|step| {
                        let timestamp = fitted.frame.frequency().advance(anchor_timestamp, step)?;
                        let elapsed = elapsed_days(series.start_timestamp, timestamp);
                        series.predict_component_record(
                            series_id,
                            elapsed,
                            timestamp,
                            step,
                            &self.config,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(json!({
            "model": self.model_name(),
            "columns": [
                "series_id",
                "timestamp",
                "horizon",
                "prediction",
                "trend",
                "adjusted_trend",
                "trend_adjustment_multiplier",
                "trend_adjustment",
                "residual_shock",
                "linear_predictor",
                "components",
            ],
            "records": records,
        }))
    }

    pub fn history_components_json_value(&self) -> Result<Value> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let history_frame = fitted.history_frame.as_ref().unwrap_or(&fitted.frame);
        let mut history_by_series: BTreeMap<String, Vec<&ForecastRow>> = BTreeMap::new();
        for row in history_frame.rows() {
            history_by_series
                .entry(row.series_id.clone())
                .or_default()
                .push(row);
        }
        let records = history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let series = fitted.series.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing fitted piecewise linear seasonal state for series {series_id:?}"
                    ))
                })?;
                series.history_component_records(series_id, history, &self.config)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(json!({
            "model": self.model_name(),
            "columns": [
                "series_id",
                "timestamp",
                "index",
                "actual",
                "fitted",
                "residual",
                "trend",
                "adjusted_trend",
                "trend_adjustment_multiplier",
                "trend_adjustment",
                "trend_movement",
                "fitted_movement",
                "linear_predictor",
                "components",
            ],
            "records": records,
        }))
    }

    pub fn history_components_json_string(&self) -> Result<String> {
        serde_json::to_string(&self.history_components_json_value()?).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal history components: {err}"
            ))
        })
    }

    pub fn predict_components_json_string(&self, horizon: usize) -> Result<String> {
        serde_json::to_string(&self.predict_components_json_value(horizon)?).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal components: {err}"
            ))
        })
    }

    pub fn predict_samples_json_value(&self, horizon: usize) -> Result<Value> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let sample_count = self.config.uncertainty_samples;
        let records = if sample_count == 0 {
            Vec::new()
        } else {
            fitted
                .series
                .iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(series_id, series)| {
                    let residual_scale = series.residual_scale();
                    (1..=horizon)
                        .map(|step| {
                            let timestamp = fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?;
                            let elapsed = elapsed_days(series.start_timestamp, timestamp);
                            let bounds =
                                piecewise_bounds(Some(series_id), None, Some(step), &self.config)?;
                            let terms = series.prediction_terms_at(
                                series_id,
                                elapsed,
                                timestamp,
                                step,
                                bounds,
                                &self.config,
                            )?;
                            let trend_offsets = series.trend_uncertainty_offsets(
                                series_id,
                                elapsed,
                                timestamp,
                                step,
                                &self.config,
                            )?;
                            let linear_trend_offsets = series.trend_uncertainty_linear_offsets(
                                series_id,
                                elapsed,
                                step,
                                &self.config,
                            );
                            Ok((0..sample_count)
                                .map(|sample| {
                                    series.predictive_sample_record(
                                        series_id,
                                        timestamp,
                                        step,
                                        sample,
                                        terms.mean,
                                        terms.linear_predictor,
                                        bounds,
                                        residual_scale,
                                        terms.coefficient_scale,
                                        terms.linear_coefficient_scale,
                                        trend_offsets.get(sample).copied().unwrap_or(0.0),
                                        linear_trend_offsets.get(sample).copied().unwrap_or(0.0),
                                        &self.config,
                                    )
                                })
                                .collect::<Vec<_>>())
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>()
        };
        Ok(json!({
            "model": self.model_name(),
            "sample_count": sample_count,
            "columns": [
                "series_id",
                "timestamp",
                "horizon",
                "sample",
                "prediction",
                "mean",
                "residual_draw",
                "coefficient_draw",
                "trend_draw",
            ],
            "records": records,
        }))
    }

    pub fn predict_samples_json_string(&self, horizon: usize) -> Result<String> {
        serde_json::to_string(&self.predict_samples_json_value(horizon)?).map_err(|err| {
            CartoBoostError::InvalidInput(format!(
                "failed to serialize piecewise linear seasonal posterior samples: {err}"
            ))
        })
    }

    pub fn predict_at_timestamps(
        &self,
        timestamps_by_series: BTreeMap<String, Vec<chrono::NaiveDateTime>>,
    ) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let schedule = self.validate_prediction_schedule(timestamps_by_series)?;
        self.predict_with_schedule(fitted, &schedule)
    }

    fn horizon_schedule(
        &self,
        fitted: &FittedPiecewiseLinearSeasonalState,
        horizon: usize,
    ) -> Result<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
        validate_horizon(horizon)?;
        fitted
            .series
            .iter()
            .map(|(series_id, series)| {
                let anchor_timestamp = fitted
                    .anchor_timestamp_by_series
                    .get(series_id)
                    .copied()
                    .unwrap_or(series.last_timestamp);
                let timestamps = (1..=horizon)
                    .map(|step| fitted.frame.frequency().advance(anchor_timestamp, step))
                    .collect::<Result<Vec<_>>>()?;
                Ok((series_id.clone(), timestamps))
            })
            .collect()
    }

    fn validate_prediction_schedule(
        &self,
        timestamps_by_series: BTreeMap<String, Vec<chrono::NaiveDateTime>>,
    ) -> Result<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        if timestamps_by_series.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "future_timestamps must contain at least one timestamp".to_string(),
            ));
        }
        for series_id in fitted.series.keys() {
            let timestamps = timestamps_by_series.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "future_timestamps_by_series is missing fitted series {series_id:?}"
                ))
            })?;
            if timestamps.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "future_timestamps for series {series_id:?} must not be empty"
                )));
            }
            let series = fitted.series.get(series_id).expect("series key exists");
            let mut previous = series.last_timestamp;
            for timestamp in timestamps {
                if *timestamp <= previous {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "future_timestamps for series {series_id:?} must be strictly after the last training timestamp and increasing"
                    )));
                }
                previous = *timestamp;
            }
        }
        for series_id in timestamps_by_series.keys() {
            if !fitted.series.contains_key(series_id) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "future_timestamps_by_series contains unknown series {series_id:?}"
                )));
            }
        }
        Ok(timestamps_by_series)
    }

    fn predict_with_schedule(
        &self,
        fitted: &FittedPiecewiseLinearSeasonalState,
        schedule: &BTreeMap<String, Vec<chrono::NaiveDateTime>>,
    ) -> Result<ForecastResult> {
        let per_series = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                let emit_intervals = !self.config.interval_levels.is_empty();
                let residual_scale = if emit_intervals {
                    series.residual_scale()
                } else {
                    0.0
                };
                let timestamps = schedule.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing prediction timestamps for series {series_id:?}"
                    ))
                })?;
                timestamps
                    .iter()
                    .enumerate()
                    .map(|(idx, &timestamp)| {
                        let step = idx + 1;
                        let elapsed = elapsed_days(series.start_timestamp, timestamp);
                        let bounds =
                            piecewise_bounds(Some(series_id), None, Some(step), &self.config)?;
                        let terms = series.prediction_terms_at(
                            series_id,
                            elapsed,
                            timestamp,
                            step,
                            bounds,
                            &self.config,
                        )?;
                        let prediction = ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean: terms.mean,
                        };
                        let intervals = if emit_intervals {
                            piecewise_prediction_intervals(
                                &prediction,
                                residual_scale,
                                terms.coefficient_scale,
                                if series.transformed_residual_scale > 0.0 {
                                    series.transformed_residual_scale
                                } else {
                                    residual_scale
                                },
                                terms.linear_predictor,
                                terms.linear_coefficient_scale,
                                series.trend_uncertainty_offsets(
                                    series_id,
                                    elapsed,
                                    timestamp,
                                    step,
                                    &self.config,
                                )?,
                                series.trend_uncertainty_linear_offsets(
                                    series_id,
                                    elapsed,
                                    step,
                                    &self.config,
                                ),
                                &self.config.interval_levels,
                                bounds,
                                &self.config,
                            )?
                        } else {
                            Vec::new()
                        };
                        Ok((prediction, intervals))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut predictions = Vec::with_capacity(per_series.len());
        let mut intervals = Vec::new();
        for (prediction, prediction_intervals) in per_series {
            predictions.push(prediction);
            intervals.extend(prediction_intervals);
        }
        ForecastResult::new_with_intervals(predictions, intervals)
    }

    pub fn predict_quantiles_json_value(
        &self,
        horizon: usize,
        quantile_levels: Option<Vec<f64>>,
    ) -> Result<Value> {
        validate_horizon(horizon)?;
        let levels = quantile_levels.unwrap_or_else(|| self.config.quantile_levels.clone());
        validate_piecewise_quantile_levels(&levels)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let records = if levels.is_empty() {
            Vec::new()
        } else {
            fitted
                .series
                .iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(series_id, series)| {
                    let residual_scale = series.residual_scale();
                    (1..=horizon)
                        .map(|step| {
                            let timestamp = fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?;
                            let elapsed = elapsed_days(series.start_timestamp, timestamp);
                            let bounds =
                                piecewise_bounds(Some(series_id), None, Some(step), &self.config)?;
                            let terms = series.prediction_terms_at(
                                series_id,
                                elapsed,
                                timestamp,
                                step,
                                bounds,
                                &self.config,
                            )?;
                            let prediction = ForecastPrediction {
                                series_id: series_id.clone(),
                                timestamp,
                                horizon: step,
                                model: self.model_name().to_string(),
                                mean: terms.mean,
                            };
                            Ok(piecewise_prediction_quantiles(
                                &prediction,
                                residual_scale,
                                terms.coefficient_scale,
                                if series.transformed_residual_scale > 0.0 {
                                    series.transformed_residual_scale
                                } else {
                                    residual_scale
                                },
                                terms.linear_predictor,
                                terms.linear_coefficient_scale,
                                series.trend_uncertainty_offsets(
                                    series_id,
                                    elapsed,
                                    timestamp,
                                    step,
                                    &self.config,
                                )?,
                                series.trend_uncertainty_linear_offsets(
                                    series_id,
                                    elapsed,
                                    step,
                                    &self.config,
                                ),
                                &levels,
                                bounds,
                                &self.config,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>()
        };
        Ok(json!({
            "model": self.model_name(),
            "quantile_levels": levels,
            "columns": [
                "series_id",
                "timestamp",
                "horizon",
                "quantile",
                "prediction",
                "mean",
            ],
            "records": records,
        }))
    }

    pub fn predict_quantiles_json_string(
        &self,
        horizon: usize,
        quantile_levels: Option<Vec<f64>>,
    ) -> Result<String> {
        serde_json::to_string(&self.predict_quantiles_json_value(horizon, quantile_levels)?)
            .map_err(|err| {
                CartoBoostError::InvalidInput(format!(
                    "failed to serialize piecewise linear seasonal quantiles: {err}"
                ))
            })
    }
}

impl PiecewiseLinearGrowth {
    fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Flat => "flat",
            Self::Logistic => "logistic",
        }
    }
}

impl PiecewiseLinearComponentMode {
    fn name(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::Multiplicative => "multiplicative",
        }
    }
}

impl SpatialPiecewiseKrigingMode {
    fn name(self) -> &'static str {
        match self {
            Self::KrigedRegressors => "kriged_regressors",
            Self::ResidualKriging => "residual_kriging",
            Self::Hybrid => "hybrid",
        }
    }
}

impl SpatialPiecewiseKrigingConfig {
    fn uses_kriged_regressors(&self) -> bool {
        matches!(
            self.mode,
            SpatialPiecewiseKrigingMode::KrigedRegressors | SpatialPiecewiseKrigingMode::Hybrid
        )
    }

    fn uses_residual_kriging(&self) -> bool {
        matches!(
            self.mode,
            SpatialPiecewiseKrigingMode::ResidualKriging | SpatialPiecewiseKrigingMode::Hybrid
        )
    }

    fn piecewise_config_metadata(&self) -> Value {
        let piecewise_config = spatial_piecewise_base_config(self);
        json!({
            "growth": piecewise_config.growth.name(),
            "component_mode": piecewise_config.component_mode.name(),
            "changepoints": piecewise_config.changepoints,
            "weekly_fourier_order": piecewise_config.weekly_fourier_order,
            "daily_fourier_order": piecewise_config.daily_fourier_order,
            "yearly_fourier_order": piecewise_config.yearly_fourier_order,
            "extra_regressors": piecewise_config.extra_regressors,
        })
    }
}

impl PiecewiseLinearFitLoss {
    fn name(self) -> &'static str {
        match self {
            Self::Squared => "squared",
            Self::Huber => "huber",
        }
    }
}

impl PiecewiseLinearRegressorStandardization {
    fn name(self) -> &'static str {
        match self {
            PiecewiseLinearRegressorStandardization::None => "none",
            PiecewiseLinearRegressorStandardization::Auto => "auto",
        }
    }
}

impl PiecewiseLinearTrendUncertaintyPolicy {
    fn name(self) -> &'static str {
        match self {
            PiecewiseLinearTrendUncertaintyPolicy::Normal => "normal",
            PiecewiseLinearTrendUncertaintyPolicy::Laplace => "laplace",
        }
    }
}

impl SeasonalNaiveForecaster {
    pub fn new(season_length: usize) -> Result<Self> {
        if season_length == 0 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be positive".to_string(),
            ));
        }
        Ok(Self {
            season_length,
            fitted: None,
        })
    }
}

impl WindowAverageForecaster {
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(CartoBoostError::InvalidInput(
                "window_size must be positive".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            fitted: None,
        })
    }
}

impl SeasonalWindowAverageForecaster {
    pub fn new(season_length: usize, window_count: usize) -> Result<Self> {
        if season_length == 0 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be positive".to_string(),
            ));
        }
        if window_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "seasonal window_count must be positive".to_string(),
            ));
        }
        Ok(Self {
            season_length,
            window_count,
            fitted: None,
        })
    }
}

impl ThetaSeasonality {
    pub fn additive(season_length: usize) -> Result<Self> {
        Self::new(ThetaSeasonalityKind::Additive, season_length)
    }

    pub fn multiplicative(season_length: usize) -> Result<Self> {
        Self::new(ThetaSeasonalityKind::Multiplicative, season_length)
    }

    fn new(kind: ThetaSeasonalityKind, season_length: usize) -> Result<Self> {
        if season_length <= 1 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be greater than 1 for theta seasonality".to_string(),
            ));
        }
        Ok(Self {
            kind,
            season_length,
        })
    }

    fn name(self) -> &'static str {
        match self.kind {
            ThetaSeasonalityKind::Additive => "additive",
            ThetaSeasonalityKind::Multiplicative => "multiplicative",
        }
    }
}

impl ThetaForecaster {
    pub fn new(theta: f64, alpha: f64) -> Result<Self> {
        Self::with_seasonality(theta, alpha, None)
    }

    pub fn with_seasonality(
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        validate_theta_params(theta, alpha)?;
        Ok(Self {
            theta,
            alpha,
            seasonality,
            fitted: None,
        })
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }
}

impl OptimizedThetaForecaster {
    pub fn new(theta_grid: Vec<f64>, alpha_grid: Vec<f64>) -> Result<Self> {
        Self::with_seasonality(theta_grid, alpha_grid, None)
    }

    pub fn with_seasonality(
        theta_grid: Vec<f64>,
        alpha_grid: Vec<f64>,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        if theta_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "theta_grid must not be empty".to_string(),
            ));
        }
        if alpha_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "alpha_grid must not be empty".to_string(),
            ));
        }
        for &theta in &theta_grid {
            validate_theta_params(theta, 0.5)?;
        }
        for &alpha in &alpha_grid {
            validate_theta_params(1.0, alpha)?;
        }
        Ok(Self {
            theta_grid,
            alpha_grid,
            seasonality,
            selected_theta: None,
            selected_alpha: None,
            validation_window: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_theta(&self) -> Option<f64> {
        self.selected_theta
    }

    pub fn selected_alpha(&self) -> Option<f64> {
        self.selected_alpha
    }

    pub fn validation_scores(&self) -> &[ThetaValidationScore] {
        &self.validation_scores
    }
}

impl ETSForecaster {
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        Self::with_additive_seasonality(alpha, beta, None, None)
    }

    pub fn with_additive_seasonality(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        Self::with_additive_damped_trend(alpha, beta, gamma, season_length, 1.0)
    }

    pub fn with_additive_damped_trend(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
        damping_phi: f64,
    ) -> Result<Self> {
        validate_ets_params(alpha, beta, gamma, season_length, damping_phi)?;
        Ok(Self {
            alpha,
            beta,
            gamma,
            season_length,
            damping_phi,
            fitted: None,
        })
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }

    pub fn level_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.level_values.as_slice())
    }

    pub fn trend_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.trend_values.as_slice())
    }

    pub fn seasonal_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.seasonal_values.as_slice())
    }
}

impl AutoETSForecaster {
    pub fn new(season_length: Option<usize>) -> Result<Self> {
        Self::with_grids(
            vec![0.1, 0.2, 0.3, 0.5, 0.8, 0.95],
            vec![0.0, 0.05, 0.1, 0.2, 0.4],
            match season_length {
                Some(_) => vec![
                    Some(0.0),
                    Some(0.05),
                    Some(0.1),
                    Some(0.2),
                    Some(0.3),
                    Some(0.5),
                ],
                None => vec![None],
            },
            vec![0.8, 0.9, 0.95, 0.98, 1.0],
            season_length,
        )
    }

    pub fn with_grids(
        alpha_grid: Vec<f64>,
        beta_grid: Vec<f64>,
        gamma_grid: Vec<Option<f64>>,
        damping_phi_grid: Vec<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        if alpha_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets alpha_grid must not be empty".to_string(),
            ));
        }
        if beta_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets beta_grid must not be empty".to_string(),
            ));
        }
        if gamma_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets gamma_grid must not be empty".to_string(),
            ));
        }
        if damping_phi_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "auto_ets damping_phi_grid must not be empty".to_string(),
            ));
        }
        for &alpha in &alpha_grid {
            validate_ets_params(alpha, 0.0, None, None, 1.0)?;
        }
        for &beta in &beta_grid {
            validate_ets_params(0.5, beta, None, None, 1.0)?;
        }
        for &gamma in &gamma_grid {
            validate_ets_params(0.5, 0.1, gamma, season_length, 1.0)?;
        }
        for &damping_phi in &damping_phi_grid {
            validate_ets_params(0.5, 0.1, None, None, damping_phi)?;
        }
        Ok(Self {
            alpha_grid,
            beta_grid,
            gamma_grid,
            damping_phi_grid,
            season_length,
            selected_params: None,
            validation_window: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<ETSParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[ETSValidationScore] {
        &self.validation_scores
    }
}

impl ArimaForecaster {
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        validate_arima_order(p, d, q)?;
        Ok(Self {
            p,
            d,
            q,
            fitted: None,
        })
    }

    pub fn order(&self) -> (usize, usize, usize) {
        (self.p, self.d, self.q)
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }
}

impl AutoARIMAForecaster {
    pub fn new(max_p: usize, max_d: usize) -> Result<Self> {
        Self::with_max_order(max_p, max_d, 2)
    }

    pub fn with_max_order(max_p: usize, max_d: usize, max_q: usize) -> Result<Self> {
        if max_p > 8 {
            return Err(CartoBoostError::InvalidInput(
                "max_p must be <= 8 for auto_arima".to_string(),
            ));
        }
        if max_d > 2 {
            return Err(CartoBoostError::InvalidInput(
                "max_d must be <= 2 for auto_arima".to_string(),
            ));
        }
        if max_q > 8 {
            return Err(CartoBoostError::InvalidInput(
                "max_q must be <= 8 for auto_arima".to_string(),
            ));
        }
        Ok(Self {
            max_p,
            max_d,
            max_q,
            selected_order: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_order(&self) -> Option<(usize, usize, usize)> {
        self.selected_order
    }

    pub fn validation_scores(&self) -> &[ArimaValidationScore] {
        &self.validation_scores
    }
}

impl KalmanForecaster {
    pub fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        LocalLinearKalmanConfig::new(
            level_process_variance,
            trend_process_variance,
            observation_variance,
        )?;
        Ok(Self {
            level_process_variance,
            trend_process_variance,
            observation_variance,
            fitted: None,
        })
    }
}

impl LocalLevelKalmanForecaster {
    pub fn new(level_process_variance: f64, observation_variance: f64) -> Result<Self> {
        LocalLevelKalmanConfig::new(level_process_variance, observation_variance)?;
        Ok(Self {
            level_process_variance,
            observation_variance,
            fitted: None,
        })
    }
}

impl AutoKalmanForecaster {
    pub fn new() -> Result<Self> {
        Self::with_grids(
            vec![0.001, 0.01, 0.05, 0.1],
            vec![0.0001, 0.001, 0.005, 0.01],
            vec![0.1, 0.5, 1.0, 2.0],
            None,
        )
    }

    pub fn with_grids(
        level_process_variance_grid: Vec<f64>,
        trend_process_variance_grid: Vec<f64>,
        observation_variance_grid: Vec<f64>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        validate_kalman_grid("level_process_variance_grid", &level_process_variance_grid)?;
        validate_kalman_grid("trend_process_variance_grid", &trend_process_variance_grid)?;
        validate_kalman_grid("observation_variance_grid", &observation_variance_grid)?;
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "auto_kalman validation_window must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            level_process_variance_grid,
            trend_process_variance_grid,
            observation_variance_grid,
            validation_window,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<KalmanParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[KalmanValidationScore] {
        &self.validation_scores
    }
}

impl AutoLocalLevelKalmanForecaster {
    pub fn new() -> Result<Self> {
        Self::with_grids(vec![0.001, 0.01, 0.05, 0.1], vec![0.1, 0.5, 1.0, 2.0], None)
    }

    pub fn with_grids(
        level_process_variance_grid: Vec<f64>,
        observation_variance_grid: Vec<f64>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        validate_kalman_grid("level_process_variance_grid", &level_process_variance_grid)?;
        validate_kalman_grid("observation_variance_grid", &observation_variance_grid)?;
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "auto_local_level_kalman validation_window must be positive when provided"
                    .to_string(),
            ));
        }
        Ok(Self {
            level_process_variance_grid,
            observation_variance_grid,
            validation_window,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<LocalLevelKalmanParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[LocalLevelKalmanValidationScore] {
        &self.validation_scores
    }
}

impl Default for AutoLocalLevelKalmanForecaster {
    fn default() -> Self {
        Self::new().expect("default auto_local_level_kalman grid is valid")
    }
}

impl Default for AutoKalmanForecaster {
    fn default() -> Self {
        Self::new().expect("default auto_kalman grid is valid")
    }
}

impl Default for KalmanForecaster {
    fn default() -> Self {
        Self {
            level_process_variance: 0.05,
            trend_process_variance: 0.005,
            observation_variance: 1.0,
            fitted: None,
        }
    }
}

impl Default for LocalLevelKalmanForecaster {
    fn default() -> Self {
        Self {
            level_process_variance: 0.05,
            observation_variance: 1.0,
            fitted: None,
        }
    }
}

impl KrigingForecaster {
    pub fn new(coordinates: BTreeMap<String, (f64, f64)>, range: f64, nugget: f64) -> Result<Self> {
        Self::with_config(coordinates, OrdinaryKrigingConfig::new(range, nugget)?)
    }

    pub fn with_config(
        coordinates: BTreeMap<String, (f64, f64)>,
        config: OrdinaryKrigingConfig,
    ) -> Result<Self> {
        Self::with_config_and_backend(coordinates, config, Some("cpu"))
    }

    pub fn with_config_and_backend(
        coordinates: BTreeMap<String, (f64, f64)>,
        config: OrdinaryKrigingConfig,
        backend: Option<&str>,
    ) -> Result<Self> {
        let config = config.validate()?;
        let backend = select_backend_for(backend, BackendOperation::PairwiseDistance)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        if coordinates.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "kriging coordinates must not be empty".to_string(),
            ));
        }
        for (series_id, (x, y)) in &coordinates {
            if !x.is_finite() || !y.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "kriging coordinate for series {series_id} must be finite"
                )));
            }
        }
        Ok(Self {
            coordinates,
            config,
            backend,
            fitted: None,
        })
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }
}

impl SpatialPiecewiseKrigingForecaster {
    pub fn new(config: SpatialPiecewiseKrigingConfig) -> Result<Self> {
        Self::new_with_backend(config, Some("cpu"))
    }

    pub fn new_with_backend(
        config: SpatialPiecewiseKrigingConfig,
        backend: Option<&str>,
    ) -> Result<Self> {
        validate_spatial_piecewise_kriging_config(&config)?;
        let backend = select_backend_for(backend, BackendOperation::PairwiseDistance)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(Self {
            config,
            backend,
            fitted: None,
        })
    }

    pub fn config(&self) -> &SpatialPiecewiseKrigingConfig {
        &self.config
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }
}

