fn boxed_forecaster_from_py(py: Python<'_>, model: &Py<PyAny>) -> PyResult<Box<dyn Forecaster>> {
    let model = model.bind(py);
    if let Ok(model) = model.extract::<PyRef<'_, NativeNaiveForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeSeasonalNaiveForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeThetaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeOptimizedThetaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativePiecewiseLinearSeasonalForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeETSForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeArimaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeAutoARIMAForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(_model) = model.extract::<PyRef<'_, NativeAutoStatsBank>>() {
        return Err(PyValueError::new_err(
            "AutoStatsBank cannot be cloned into WeightedEnsembleForecaster",
        ));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeKalmanForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeCartoBoostLagForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    Err(PyValueError::new_err(
        "WeightedEnsembleForecaster members must be native forecasting models",
    ))
}

fn forecast_to_py(
    result: cartoboost_core::Result<CoreForecastResult>,
) -> PyResult<NativeForecastResult> {
    Ok(NativeForecastResult {
        result: result.map_err(to_py_value_error)?,
    })
}

fn fit_forecaster_py<M: Forecaster>(
    py: Python<'_>,
    model: &mut M,
    frame: &NativeForecastFrame,
) -> PyResult<()> {
    py.detach(|| model.fit(&frame.frame))
        .map_err(to_py_value_error)
}

fn predict_forecaster_py<M: Forecaster>(
    py: Python<'_>,
    model: &M,
    horizon: usize,
) -> PyResult<NativeForecastResult> {
    forecast_to_py(py.detach(|| model.predict(horizon)))
}

#[allow(dead_code)]
fn ets_diagnostic_values(
    values: Option<&[f64]>,
    series_id: &str,
    name: &str,
) -> PyResult<Vec<f64>> {
    values.map(|values| values.to_vec()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "ETS {name} are unavailable for series {series_id:?}; fit the model and check the series id"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_kriging_config(
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<OrdinaryKrigingConfig> {
    let variogram_model = parse_kriging_variogram_model(variogram_model)?;
    let drift = parse_kriging_drift(drift)?;
    OrdinaryKrigingConfig::new(range, nugget)
        .and_then(|config| config.with_sill(sill))
        .and_then(|config| config.with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling))
        .and_then(|config| config.with_neighbor_limits(max_neighbors, min_neighbors, max_distance))
        .map(|config| {
            config
                .with_variogram_model(variogram_model)
                .with_drift(drift)
        })
        .map_err(to_py_value_error)
}

fn parse_kriging_variogram_model(value: &str) -> PyResult<KrigingVariogramModel> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "exponential" | "exp" => Ok(KrigingVariogramModel::Exponential),
        "gaussian" | "gauss" => Ok(KrigingVariogramModel::Gaussian),
        "spherical" | "sphere" => Ok(KrigingVariogramModel::Spherical),
        "linear" => Ok(KrigingVariogramModel::Linear),
        other => Err(PyValueError::new_err(format!(
            "unsupported kriging variogram_model {other:?}; expected exponential, gaussian, spherical, or linear"
        ))),
    }
}

fn parse_kriging_drift(value: &str) -> PyResult<KrigingDrift> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "ordinary" | "constant" | "none" => Ok(KrigingDrift::Ordinary),
        "linear" | "universal_linear" | "universal" => Ok(KrigingDrift::Linear),
        other => Err(PyValueError::new_err(format!(
            "unsupported kriging drift {other:?}; expected ordinary or linear"
        ))),
    }
}

fn parse_spatial_piecewise_kriging_mode(value: &str) -> PyResult<SpatialPiecewiseKrigingMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "kriged_regressors" | "regressors" => {
            Ok(SpatialPiecewiseKrigingMode::KrigedRegressors)
        }
        "residual_kriging" | "residual" => Ok(SpatialPiecewiseKrigingMode::ResidualKriging),
        "hybrid" => Ok(SpatialPiecewiseKrigingMode::Hybrid),
        other => Err(PyValueError::new_err(format!(
            "unsupported spatial piecewise kriging mode {other:?}; expected kriged_regressors, residual_kriging, or hybrid"
        ))),
    }
}

fn parse_classical_validation_objective(
    value: &str,
    season_length: usize,
) -> PyResult<ClassicalExpertValidationObjective> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "mse" | "mean_squared_error" => Ok(ClassicalExpertValidationObjective::MeanSquaredError),
        "smape_mase_average" | "owa_proxy" => Ok(
            ClassicalExpertValidationObjective::SmapeMaseAverage {
                seasonality: season_length.max(1),
            },
        ),
        other => Err(PyValueError::new_err(format!(
            "unsupported validation_objective {other:?}; expected mean_squared_error or smape_mase_average"
        ))),
    }
}

fn kriging_config_json(config: OrdinaryKrigingConfig) -> Value {
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

fn backtest_to_py(
    result: cartoboost_core::Result<CoreBacktestResult>,
) -> PyResult<NativeBacktestResult> {
    Ok(NativeBacktestResult {
        result: result.map_err(to_py_value_error)?,
    })
}

fn parse_forecast_window(value: &str) -> PyResult<ForecastWindow> {
    match value {
        "expanding" => Ok(ForecastWindow::Expanding),
        "sliding" => Ok(ForecastWindow::Sliding),
        _ => Err(PyValueError::new_err(
            "forecast window must be 'expanding' or 'sliding'",
        )),
    }
}

fn forecast_window_name(window: &ForecastWindow) -> &'static str {
    match window {
        ForecastWindow::Expanding => "expanding",
        ForecastWindow::Sliding => "sliding",
    }
}

fn parse_forecast_actuals(
    actuals: Vec<(String, String, usize, f64)>,
) -> PyResult<Vec<ForecastActual>> {
    actuals
        .into_iter()
        .map(|(series_id, timestamp, horizon, actual)| {
            Ok(ForecastActual {
                series_id,
                timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                    .map_err(to_py_value_error)?,
                horizon,
                actual,
            })
        })
        .collect()
}

fn forecast_prediction_tuple(
    prediction: &ForecastPrediction,
) -> (String, String, usize, String, f64) {
    (
        prediction.series_id.clone(),
        format_forecast_timestamp(prediction.timestamp),
        prediction.horizon,
        prediction.model.clone(),
        prediction.mean,
    )
}

fn format_forecast_timestamp(timestamp: impl std::fmt::Display) -> String {
    timestamp.to_string().replace(' ', "T")
}

fn validate_interval_levels(levels: Option<&[f64]>) -> PyResult<()> {
    for level in levels.unwrap_or(&[]) {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(PyValueError::new_err(
                "prediction interval levels must be finite values between 0 and 1",
            ));
        }
    }
    Ok(())
}

fn calendar_feature_config(
    enabled: bool,
    rich: bool,
    elapsed_only: bool,
    elapsed_periods: Option<&[usize]>,
) -> Vec<CalendarFeature> {
    if !enabled {
        return Vec::new();
    }
    if elapsed_only {
        let mut features = vec![CalendarFeature::ElapsedIndex];
        push_elapsed_calendar_periods(&mut features, elapsed_periods);
        return features;
    }
    let mut features = vec![
        CalendarFeature::DayOfWeek,
        CalendarFeature::Month,
        CalendarFeature::Day,
    ];
    if rich {
        features.push(CalendarFeature::DayOfWeekSin);
        features.push(CalendarFeature::DayOfWeekCos);
        features.push(CalendarFeature::MonthSin);
        features.push(CalendarFeature::MonthCos);
        features.push(CalendarFeature::DaySin);
        features.push(CalendarFeature::DayCos);
        features.push(CalendarFeature::MonthStart);
        features.push(CalendarFeature::MonthMiddle);
        features.push(CalendarFeature::MonthEnd);
        features.push(CalendarFeature::DayOfYear);
        features.push(CalendarFeature::ElapsedIndex);
        push_elapsed_calendar_periods(&mut features, elapsed_periods);
    }
    features
}

fn push_elapsed_calendar_periods(
    features: &mut Vec<CalendarFeature>,
    elapsed_periods: Option<&[usize]>,
) {
    let mut periods = BTreeSet::new();
    for period in elapsed_periods.unwrap_or(&[]) {
        if *period >= 2 && periods.insert(*period) {
            features.push(CalendarFeature::ElapsedPhase(*period));
        }
    }
}

fn parse_theta_seasonality(
    season_length: Option<usize>,
    seasonality: Option<String>,
) -> PyResult<Option<ThetaSeasonality>> {
    let Some(mode) = seasonality else {
        return Ok(None);
    };
    let season_length = season_length.ok_or_else(|| {
        PyValueError::new_err("season_length is required when seasonality is set")
    })?;
    match mode.as_str() {
        "additive" => ThetaSeasonality::additive(season_length)
            .map(Some)
            .map_err(to_py_value_error),
        "multiplicative" => ThetaSeasonality::multiplicative(season_length)
            .map(Some)
            .map_err(to_py_value_error),
        _ => Err(PyValueError::new_err(
            "seasonality must be 'additive' or 'multiplicative'",
        )),
    }
}

fn parse_piecewise_growth(value: &str) -> PyResult<PiecewiseLinearGrowth> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "linear" => Ok(PiecewiseLinearGrowth::Linear),
        "flat" => Ok(PiecewiseLinearGrowth::Flat),
        "logistic" => Ok(PiecewiseLinearGrowth::Logistic),
        other => Err(PyValueError::new_err(format!(
            "growth must be 'linear', 'flat', or 'logistic', got {other:?}"
        ))),
    }
}

fn parse_piecewise_component_mode(value: &str) -> PyResult<PiecewiseLinearComponentMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "additive" => Ok(PiecewiseLinearComponentMode::Additive),
        "multiplicative" => Ok(PiecewiseLinearComponentMode::Multiplicative),
        other => Err(PyValueError::new_err(format!(
            "component_mode must be 'additive' or 'multiplicative', got {other:?}"
        ))),
    }
}

fn parse_piecewise_fit_loss(value: &str) -> PyResult<PiecewiseLinearFitLoss> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "squared" | "l2" | "least_squares" => Ok(PiecewiseLinearFitLoss::Squared),
        "huber" | "robust" => Ok(PiecewiseLinearFitLoss::Huber),
        other => Err(PyValueError::new_err(format!(
            "fit_loss must be 'squared' or 'huber', got {other:?}"
        ))),
    }
}

fn parse_piecewise_regressor_standardization(
    value: &str,
) -> PyResult<PiecewiseLinearRegressorStandardization> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "auto" => Ok(PiecewiseLinearRegressorStandardization::Auto),
        "none" | "off" | "false" => Ok(PiecewiseLinearRegressorStandardization::None),
        other => Err(PyValueError::new_err(format!(
            "regressor_standardization must be 'auto' or 'none', got {other:?}"
        ))),
    }
}

fn parse_piecewise_trend_uncertainty_policy(
    value: &str,
) -> PyResult<PiecewiseLinearTrendUncertaintyPolicy> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "laplace" => Ok(PiecewiseLinearTrendUncertaintyPolicy::Laplace),
        "normal" | "gaussian" => Ok(PiecewiseLinearTrendUncertaintyPolicy::Normal),
        other => Err(PyValueError::new_err(format!(
            "trend_uncertainty_policy must be 'laplace' or 'normal', got {other:?}"
        ))),
    }
}

fn parse_optional_piecewise_component_mode(
    value: Option<String>,
) -> PyResult<Option<PiecewiseLinearComponentMode>> {
    value
        .as_deref()
        .map(parse_piecewise_component_mode)
        .transpose()
}

fn parse_piecewise_regressor_modes(
    values: Option<BTreeMap<String, String>>,
) -> PyResult<BTreeMap<String, PiecewiseLinearComponentMode>> {
    values
        .unwrap_or_default()
        .into_iter()
        .map(|(name, mode)| Ok((name, parse_piecewise_component_mode(&mode)?)))
        .collect()
}

fn parse_piecewise_changepoint_timestamps(
    timestamps: Option<Vec<String>>,
) -> PyResult<Vec<chrono::NaiveDateTime>> {
    timestamps
        .unwrap_or_default()
        .into_iter()
        .map(|timestamp| {
            cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                .map_err(to_py_value_error)
        })
        .collect()
}

fn parse_piecewise_events(
    events: Option<Vec<PyPiecewiseEvent>>,
) -> PyResult<Vec<PiecewiseLinearEvent>> {
    events
        .unwrap_or_default()
        .into_iter()
        .map(|(name, timestamp, lower_window, upper_window)| {
            Ok(PiecewiseLinearEvent {
                name,
                timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                    .map_err(to_py_value_error)?,
                lower_window: lower_window.unwrap_or(0),
                upper_window: upper_window.unwrap_or(0),
            })
        })
        .collect()
}

fn parse_piecewise_seasonalities(
    seasonalities: Option<Vec<PyPiecewiseSeasonality>>,
) -> PyResult<Vec<PiecewiseLinearSeasonality>> {
    seasonalities
        .unwrap_or_default()
        .into_iter()
        .map(
            |(name, period_days, fourier_order, mode, condition_name, l2_regularization)| {
                Ok(PiecewiseLinearSeasonality {
                    name,
                    period_days,
                    fourier_order,
                    mode: parse_optional_piecewise_component_mode(mode)?,
                    condition_name,
                    l2_regularization,
                })
            },
        )
        .collect()
}

