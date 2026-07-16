#[pyclass(name = "PiecewiseLinearSeasonalForecaster")]
#[derive(Clone, Debug)]
struct NativePiecewiseLinearSeasonalForecaster {
    model: CorePiecewiseLinearSeasonalForecaster,
}

#[pymethods]
impl NativePiecewiseLinearSeasonalForecaster {
    #[new]
    #[pyo3(signature = (
        growth="linear",
        component_mode="additive",
        changepoints=25,
        changepoint_range=1.0,
        changepoint_timestamps=None,
        yearly_fourier_order=0,
        weekly_fourier_order=3,
        daily_fourier_order=0,
        auto_yearly_seasonality=true,
        auto_weekly_seasonality=true,
        auto_daily_seasonality=true,
        custom_seasonalities=None,
        changepoint_l2_regularization=0.05,
        changepoint_l1_regularization=0.0,
        seasonality_l2_regularization=0.01,
        yearly_l2_regularization=None,
        weekly_l2_regularization=None,
        daily_l2_regularization=None,
        event_l2_regularization=0.01,
        regressor_l2_regularization=0.01,
        event_l2_regularization_by_name=None,
        regressor_l2_regularization_by_name=None,
        events=None,
        event_mode=None,
        extra_regressors=None,
        regressor_modes=None,
        extra_regressor_monotonic_constraints=None,
        regressor_standardization="auto",
        future_regressors=None,
        future_regressors_by_series=None,
        trend_adjustments=None,
        trend_adjustments_by_series=None,
        residual_shock_window=0,
        residual_shock_scale=0.0,
        residual_shock_decay=1.0,
        prediction_interval_levels=None,
        quantile_levels=None,
        uncertainty_samples=0,
        trend_uncertainty_policy="laplace",
        trend_uncertainty_scale=1.0,
        coefficient_uncertainty_scale=1.0,
        uncertainty_seed=14172296343723622691,
        cap=None,
        floor=0.0,
        cap_regressor=None,
        floor_regressor=None,
        fit_loss="squared",
        huber_delta=1.345,
        irls_iterations=5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        growth: &str,
        component_mode: &str,
        changepoints: usize,
        changepoint_range: f64,
        changepoint_timestamps: Option<Vec<String>>,
        yearly_fourier_order: usize,
        weekly_fourier_order: usize,
        daily_fourier_order: usize,
        auto_yearly_seasonality: bool,
        auto_weekly_seasonality: bool,
        auto_daily_seasonality: bool,
        custom_seasonalities: Option<Vec<PyPiecewiseSeasonality>>,
        changepoint_l2_regularization: f64,
        changepoint_l1_regularization: f64,
        seasonality_l2_regularization: f64,
        yearly_l2_regularization: Option<f64>,
        weekly_l2_regularization: Option<f64>,
        daily_l2_regularization: Option<f64>,
        event_l2_regularization: f64,
        regressor_l2_regularization: f64,
        event_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
        regressor_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
        events: Option<Vec<PyPiecewiseEvent>>,
        event_mode: Option<String>,
        extra_regressors: Option<Vec<String>>,
        regressor_modes: Option<BTreeMap<String, String>>,
        extra_regressor_monotonic_constraints: Option<BTreeMap<String, i8>>,
        regressor_standardization: &str,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
        residual_shock_window: usize,
        residual_shock_scale: f64,
        residual_shock_decay: f64,
        prediction_interval_levels: Option<Vec<f64>>,
        quantile_levels: Option<Vec<f64>>,
        uncertainty_samples: usize,
        trend_uncertainty_policy: &str,
        trend_uncertainty_scale: f64,
        coefficient_uncertainty_scale: f64,
        uncertainty_seed: u64,
        cap: Option<f64>,
        floor: f64,
        cap_regressor: Option<String>,
        floor_regressor: Option<String>,
        fit_loss: &str,
        huber_delta: f64,
        irls_iterations: usize,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        validate_interval_levels(quantile_levels.as_deref())?;
        let config = CorePiecewiseLinearSeasonalConfig {
            growth: parse_piecewise_growth(growth)?,
            component_mode: parse_piecewise_component_mode(component_mode)?,
            fit_loss: parse_piecewise_fit_loss(fit_loss)?,
            huber_delta,
            irls_iterations,
            changepoints,
            changepoint_range,
            changepoint_timestamps: parse_piecewise_changepoint_timestamps(changepoint_timestamps)?,
            yearly_fourier_order,
            weekly_fourier_order,
            daily_fourier_order,
            auto_yearly_seasonality,
            auto_weekly_seasonality,
            auto_daily_seasonality,
            custom_seasonalities: parse_piecewise_seasonalities(custom_seasonalities)?,
            changepoint_l2_regularization,
            changepoint_l1_regularization,
            seasonality_l2_regularization,
            yearly_l2_regularization,
            weekly_l2_regularization,
            daily_l2_regularization,
            event_l2_regularization,
            regressor_l2_regularization,
            event_l2_regularization_by_name: event_l2_regularization_by_name.unwrap_or_default(),
            regressor_l2_regularization_by_name: regressor_l2_regularization_by_name
                .unwrap_or_default(),
            events: parse_piecewise_events(events)?,
            event_mode: parse_optional_piecewise_component_mode(event_mode)?,
            extra_regressors: extra_regressors.unwrap_or_default(),
            regressor_modes: parse_piecewise_regressor_modes(regressor_modes)?,
            extra_regressor_monotonic_constraints: extra_regressor_monotonic_constraints
                .unwrap_or_default(),
            regressor_standardization: parse_piecewise_regressor_standardization(
                regressor_standardization,
            )?,
            future_regressors: future_regressors.unwrap_or_default(),
            future_regressors_by_series: future_regressors_by_series.unwrap_or_default(),
            trend_adjustments: trend_adjustments.unwrap_or_default(),
            trend_adjustments_by_series: trend_adjustments_by_series.unwrap_or_default(),
            residual_shock_window,
            residual_shock_scale,
            residual_shock_decay,
            interval_levels: prediction_interval_levels.unwrap_or_default(),
            quantile_levels: quantile_levels.unwrap_or_default(),
            uncertainty_samples,
            trend_uncertainty_policy: parse_piecewise_trend_uncertainty_policy(
                trend_uncertainty_policy,
            )?,
            trend_uncertainty_scale,
            coefficient_uncertainty_scale,
            uncertainty_seed,
            cap,
            floor,
            cap_regressor,
            floor_regressor,
        };
        Ok(Self {
            model: CorePiecewiseLinearSeasonalForecaster::new(config).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    #[pyo3(signature = (horizon, future_regressors=None, future_regressors_by_series=None, prediction_interval_levels=None, uncertainty_samples=None, trend_adjustments=None, trend_adjustments_by_series=None, future_timestamps=None, future_timestamps_by_series=None))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        prediction_interval_levels: Option<Vec<f64>>,
        uncertainty_samples: Option<usize>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
        future_timestamps: Option<Vec<String>>,
        future_timestamps_by_series: Option<BTreeMap<String, Vec<String>>>,
    ) -> PyResult<NativeForecastResult> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            prediction_interval_levels,
            None,
            uncertainty_samples,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        match (future_timestamps, future_timestamps_by_series) {
            (None, None) => predict_forecaster_py(py, &model, horizon),
            (Some(timestamps), None) => {
                let schedule = piecewise_shared_future_timestamps(&model, timestamps, horizon)?;
                forecast_to_py(py.detach(|| model.predict_at_timestamps(schedule)))
            }
            (None, Some(timestamps_by_series)) => {
                let schedule =
                    piecewise_future_timestamps_by_series(timestamps_by_series, horizon)?;
                forecast_to_py(py.detach(|| model.predict_at_timestamps(schedule)))
            }
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "pass either future_timestamps or future_timestamps_by_series, not both",
            )),
        }
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.model.to_json_string())
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (horizon, future_regressors=None, future_regressors_by_series=None, trend_adjustments=None, trend_adjustments_by_series=None))]
    fn components_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    ) -> PyResult<String> {
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            None,
            None,
            None,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        py.detach(|| model.predict_components_json_string(horizon))
            .map_err(to_py_value_error)
    }

    fn history_components_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.model.history_components_json_string())
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (horizon, future_regressors=None, future_regressors_by_series=None, uncertainty_samples=None, trend_adjustments=None, trend_adjustments_by_series=None))]
    #[allow(clippy::too_many_arguments)]
    fn samples_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        uncertainty_samples: Option<usize>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    ) -> PyResult<String> {
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            None,
            None,
            uncertainty_samples,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        py.detach(|| model.predict_samples_json_string(horizon))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (horizon, quantile_levels=None, future_regressors=None, future_regressors_by_series=None, uncertainty_samples=None, trend_adjustments=None, trend_adjustments_by_series=None))]
    #[allow(clippy::too_many_arguments)]
    fn quantiles_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        quantile_levels: Option<Vec<f64>>,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        uncertainty_samples: Option<usize>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    ) -> PyResult<String> {
        validate_interval_levels(quantile_levels.as_deref())?;
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            None,
            quantile_levels.clone(),
            uncertainty_samples,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        py.detach(|| model.predict_quantiles_json_string(horizon, quantile_levels))
            .map_err(to_py_value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let model = py
            .detach(|| CorePiecewiseLinearSeasonalForecaster::from_json_string(value))
            .map_err(to_py_value_error)?;
        Ok(Self { model })
    }
}

#[allow(clippy::too_many_arguments)]
fn piecewise_model_with_prediction_overrides(
    model: &CorePiecewiseLinearSeasonalForecaster,
    future_regressors: Option<BTreeMap<String, Vec<f64>>>,
    future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
    interval_levels: Option<Vec<f64>>,
    quantile_levels: Option<Vec<f64>>,
    uncertainty_samples: Option<usize>,
    trend_adjustments: Option<BTreeMap<usize, f64>>,
    trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
) -> PyResult<CorePiecewiseLinearSeasonalForecaster> {
    let mut model = model.clone();
    model
        .update_config(|config| {
            if let Some(future_regressors) = future_regressors {
                config.future_regressors = future_regressors;
            }
            if let Some(future_regressors_by_series) = future_regressors_by_series {
                config.future_regressors_by_series = future_regressors_by_series;
            }
            if let Some(interval_levels) = interval_levels {
                config.interval_levels = interval_levels;
            }
            if let Some(quantile_levels) = quantile_levels {
                config.quantile_levels = quantile_levels;
            }
            if let Some(uncertainty_samples) = uncertainty_samples {
                config.uncertainty_samples = uncertainty_samples;
            }
            if let Some(trend_adjustments) = trend_adjustments {
                config.trend_adjustments = trend_adjustments;
            }
            if let Some(trend_adjustments_by_series) = trend_adjustments_by_series {
                config.trend_adjustments_by_series = trend_adjustments_by_series;
            }
        })
        .map_err(to_py_value_error)?;
    Ok(model)
}

fn piecewise_shared_future_timestamps(
    model: &CorePiecewiseLinearSeasonalForecaster,
    timestamps: Vec<String>,
    horizon: usize,
) -> PyResult<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
    let parsed = parse_future_timestamps(timestamps)?;
    validate_future_timestamp_count(parsed.len(), horizon)?;
    let series_ids = model.fitted_series_ids().map_err(to_py_value_error)?;
    Ok(series_ids
        .into_iter()
        .map(|series_id| (series_id, parsed.clone()))
        .collect())
}

fn piecewise_future_timestamps_by_series(
    timestamps_by_series: BTreeMap<String, Vec<String>>,
    horizon: usize,
) -> PyResult<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
    timestamps_by_series
        .into_iter()
        .map(|(series_id, timestamps)| {
            let parsed = parse_future_timestamps(timestamps)?;
            validate_future_timestamp_count(parsed.len(), horizon)?;
            Ok((series_id, parsed))
        })
        .collect()
}

fn parse_future_timestamps(timestamps: Vec<String>) -> PyResult<Vec<chrono::NaiveDateTime>> {
    timestamps
        .into_iter()
        .map(|timestamp| parse_forecast_timestamp(&timestamp).map_err(to_py_value_error))
        .collect()
}

fn validate_future_timestamp_count(count: usize, horizon: usize) -> PyResult<()> {
    if count != horizon {
        return Err(PyValueError::new_err(format!(
            "future_timestamps length must match horizon; got {count} timestamps for horizon {horizon}"
        )));
    }
    Ok(())
}

