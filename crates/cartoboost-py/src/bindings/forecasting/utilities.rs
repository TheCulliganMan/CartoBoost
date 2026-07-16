#[pyfunction]
#[pyo3(signature = (values, level_process_variance=0.05, trend_process_variance=0.005, observation_variance=1.0, horizon=0, interval_z=1.959963984540054))]
fn utility_kalman_filter(
    py: Python<'_>,
    values: Vec<f64>,
    level_process_variance: f64,
    trend_process_variance: f64,
    observation_variance: f64,
    horizon: usize,
    interval_z: f64,
) -> PyResult<String> {
    let config = LocalLinearKalmanConfig::new(
        level_process_variance,
        trend_process_variance,
        observation_variance,
    )
    .map_err(to_py_value_error)?;
    let (result, forecast, forecast_distribution) = py
        .detach(|| {
            let result = fit_local_linear_kalman(&values, config)?;
            let forecast = if horizon == 0 {
                Vec::new()
            } else {
                local_linear_kalman_forecast(result.final_state, horizon)?
            };
            let forecast_distribution = if horizon == 0 {
                Vec::new()
            } else {
                local_linear_kalman_forecast_distribution(
                    result.final_state,
                    result.final_covariance,
                    config,
                    horizon,
                    interval_z,
                )?
            };
            Ok((result, forecast, forecast_distribution))
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "final_state": {
            "level": result.final_state.level,
            "trend": result.final_state.trend,
            "covariance": result.final_covariance,
        },
        "estimates": result.estimates.iter().map(|estimate| {
            json!({
                "step": estimate.step,
                "observed": estimate.observed,
                "prior_level": estimate.prior_level,
                "prior_trend": estimate.prior_trend,
                "prior_level_variance": estimate.prior_level_variance,
                "prior_trend_variance": estimate.prior_trend_variance,
                "prior_covariance": estimate.prior_covariance,
                "level": estimate.level,
                "trend": estimate.trend,
                "level_variance": estimate.level_variance,
                "trend_variance": estimate.trend_variance,
                "covariance": estimate.covariance,
                "fitted": estimate.prior_level,
                "residual": estimate.innovation,
                "innovation": estimate.innovation,
                "innovation_variance": estimate.innovation_variance,
                "standardized_innovation": estimate.innovation / estimate.innovation_variance.sqrt(),
                "level_gain": estimate.level_gain,
                "trend_gain": estimate.trend_gain,
                "log_likelihood": estimate.log_likelihood,
            })
        }).collect::<Vec<_>>(),
        "smoothed_states": result.smoothed_states.iter().map(|state| {
            json!({
                "step": state.step,
                "level": state.level,
                "trend": state.trend,
                "covariance": state.covariance,
            })
        }).collect::<Vec<_>>(),
        "forecast": forecast,
        "forecast_distribution": forecast_distribution.iter().map(|point| {
            json!({
                "step": point.step,
                "mean": point.mean,
                "variance": point.variance,
                "lower": point.lower,
                "upper": point.upper,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "log_likelihood": result.log_likelihood,
            "interval_z": interval_z,
            "observation_count": result.residual_summary.observation_count,
            "fitted_count": result.residual_summary.fitted_count,
            "aic": result.residual_summary.aic,
            "bic": result.residual_summary.bic,
            "mse": result.residual_summary.mse,
            "rmse": result.residual_summary.rmse,
            "mae": result.residual_summary.mae,
            "mean_standardized_innovation": result.residual_summary.mean_standardized_innovation,
            "max_abs_standardized_innovation": result.residual_summary.max_abs_standardized_innovation,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (values, level_process_variance=0.05, observation_variance=1.0, horizon=0, interval_z=1.959963984540054))]
fn utility_local_level_kalman_filter(
    py: Python<'_>,
    values: Vec<f64>,
    level_process_variance: f64,
    observation_variance: f64,
    horizon: usize,
    interval_z: f64,
) -> PyResult<String> {
    let config = LocalLevelKalmanConfig::new(level_process_variance, observation_variance)
        .map_err(to_py_value_error)?;
    let (result, forecast, forecast_distribution) = py
        .detach(|| {
            let result = fit_local_level_kalman(&values, config)?;
            let forecast = if horizon == 0 {
                Vec::new()
            } else {
                local_level_kalman_forecast(result.final_level, horizon)?
            };
            let forecast_distribution = if horizon == 0 {
                Vec::new()
            } else {
                local_level_kalman_forecast_distribution(
                    result.final_level,
                    result.final_variance,
                    config,
                    horizon,
                    interval_z,
                )?
            };
            Ok((result, forecast, forecast_distribution))
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "final_state": {
            "level": result.final_level,
            "variance": result.final_variance,
        },
        "estimates": result.estimates.iter().map(|estimate| {
            json!({
                "step": estimate.step,
                "observed": estimate.observed,
                "prior_level": estimate.prior_level,
                "prior_variance": estimate.prior_variance,
                "level": estimate.level,
                "variance": estimate.variance,
                "fitted": estimate.prior_level,
                "residual": estimate.innovation,
                "innovation": estimate.innovation,
                "innovation_variance": estimate.innovation_variance,
                "standardized_innovation": estimate.innovation / estimate.innovation_variance.sqrt(),
                "gain": estimate.gain,
                "log_likelihood": estimate.log_likelihood,
            })
        }).collect::<Vec<_>>(),
        "smoothed_states": result.smoothed_states.iter().map(|state| {
            json!({
                "step": state.step,
                "level": state.level,
                "variance": state.variance,
            })
        }).collect::<Vec<_>>(),
        "forecast": forecast,
        "forecast_distribution": forecast_distribution.iter().map(|point| {
            json!({
                "step": point.step,
                "mean": point.mean,
                "variance": point.variance,
                "lower": point.lower,
                "upper": point.upper,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "log_likelihood": result.log_likelihood,
            "interval_z": interval_z,
            "observation_count": result.residual_summary.observation_count,
            "fitted_count": result.residual_summary.fitted_count,
            "aic": result.residual_summary.aic,
            "bic": result.residual_summary.bic,
            "mse": result.residual_summary.mse,
            "rmse": result.residual_summary.rmse,
            "mae": result.residual_summary.mae,
            "mean_standardized_innovation": result.residual_summary.mean_standardized_innovation,
            "max_abs_standardized_innovation": result.residual_summary.max_abs_standardized_innovation,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (values, horizon, method="croston", alpha=0.1, beta=0.1))]
fn utility_intermittent_demand_forecast(
    py: Python<'_>,
    values: Vec<f64>,
    horizon: usize,
    method: &str,
    alpha: f64,
    beta: f64,
) -> PyResult<Vec<f64>> {
    let method = match method {
        "croston" => IntermittentDemandMethod::Croston,
        "sba" => IntermittentDemandMethod::Sba,
        "tsb" => IntermittentDemandMethod::Tsb,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported intermittent demand method {other:?}"
            )));
        }
    };
    py.detach(|| intermittent_demand_forecast(&values, horizon, alpha, beta, method))
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (observations, targets, range=1.0, nugget=1.0e-6))]
fn utility_ordinary_kriging_predict(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    targets: Vec<(f64, f64)>,
    range: f64,
    nugget: f64,
) -> PyResult<Vec<PyKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = OrdinaryKrigingConfig::new(range, nugget).map_err(to_py_value_error)?;
    let predictions = py
        .detach(|| ordinary_kriging_predict_many(&observations, &targets, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.weights,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    targets,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_predict_detailed(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    targets: Vec<(f64, f64)>,
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
) -> PyResult<Vec<PyDetailedKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let predictions = py
        .detach(|| ordinary_kriging_predict_many(&observations, &targets, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.variance,
                prediction.weights,
                prediction.neighbor_indices,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_leave_one_out(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
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
) -> PyResult<Vec<PyDetailedKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let predictions = py
        .detach(|| ordinary_kriging_leave_one_out(&observations, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.variance,
                prediction.weights,
                prediction.neighbor_indices,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    bin_count=10,
    max_distance=None,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0
))]
fn utility_empirical_variogram(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let bins = py
        .detach(|| {
            empirical_variogram(
                &observations,
                bin_count,
                max_distance,
                anisotropy_angle_degrees,
                anisotropy_scaling,
            )
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "bins": bins.iter().map(|bin| {
            json!({
                "lag_min": bin.lag_min,
                "lag_max": bin.lag_max,
                "lag_center": bin.lag_center,
                "mean_distance": bin.mean_distance,
                "semivariance": bin.semivariance,
                "pair_count": bin.pair_count,
            })
        }).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    variogram_models=None,
    range_candidates=None,
    nugget_candidates=None,
    sill_candidates=None,
    bin_count=10,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0
))]
#[allow(clippy::too_many_arguments)]
fn utility_fit_ordinary_kriging_variogram(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    variogram_models: Option<Vec<String>>,
    range_candidates: Option<Vec<f64>>,
    nugget_candidates: Option<Vec<f64>>,
    sill_candidates: Option<Vec<f64>>,
    bin_count: usize,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let models = variogram_models
        .unwrap_or_default()
        .iter()
        .map(|model| parse_kriging_variogram_model(model))
        .collect::<PyResult<Vec<_>>>()?;
    let ranges = range_candidates.unwrap_or_default();
    let nuggets = nugget_candidates.unwrap_or_default();
    let sills = sill_candidates.unwrap_or_default();
    let fit = py
        .detach(|| {
            fit_ordinary_kriging_variogram(
                &observations,
                &models,
                &ranges,
                &nuggets,
                &sills,
                bin_count,
                anisotropy_angle_degrees,
                anisotropy_scaling,
            )
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "config": kriging_config_json(fit.config),
        "weighted_sse": fit.weighted_sse,
        "bins": fit.bins.iter().map(|bin| {
            json!({
                "lag_min": bin.lag_min,
                "lag_max": bin.lag_max,
                "lag_center": bin.lag_center,
                "mean_distance": bin.mean_distance,
                "semivariance": bin.semivariance,
                "pair_count": bin.pair_count,
            })
        }).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_leave_one_out_diagnostics(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
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
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let (predictions, diagnostics) = py
        .detach(|| ordinary_kriging_leave_one_out_diagnostics(&observations, config))
        .map_err(to_py_value_error)?;
    let payload = json!({
        "predictions": predictions.iter().map(|prediction| {
            json!({
                "x": prediction.x,
                "y": prediction.y,
                "mean": prediction.mean,
                "variance": prediction.variance,
                "weights": prediction.weights,
                "neighbor_indices": prediction.neighbor_indices,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "observation_count": diagnostics.observation_count,
            "mean_error": diagnostics.mean_error,
            "mae": diagnostics.mae,
            "rmse": diagnostics.rmse,
            "mean_standardized_error": diagnostics.mean_standardized_error,
            "rmse_standardized_error": diagnostics.rmse_standardized_error,
            "max_abs_standardized_error": diagnostics.max_abs_standardized_error,
            "interval_coverage_95": diagnostics.interval_coverage_95,
            "average_variance": diagnostics.average_variance,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (model, values, horizon, params_json=None))]
fn utility_series_forecast(
    py: Python<'_>,
    model: &str,
    values: Vec<f64>,
    horizon: usize,
    params_json: Option<&str>,
) -> PyResult<Vec<f64>> {
    let params = match params_json {
        Some(raw) => serde_json::from_str::<Value>(raw).map_err(|err| {
            PyValueError::new_err(format!("params_json must be valid JSON: {err}"))
        })?,
        None => json!({}),
    };
    let frame = utility_frame_from_values(values).map_err(to_py_value_error)?;
    let mut forecaster = utility_forecaster(model, &params).map_err(to_py_value_error)?;
    let result = py
        .detach(|| {
            forecaster.fit(&frame)?;
            forecaster.predict(horizon)
        })
        .map_err(to_py_value_error)?;
    Ok(result
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect())
}

fn utility_frame_from_values(values: Vec<f64>) -> cartoboost_core::Result<CoreForecastFrame> {
    let frequency = ForecastFrequency::Daily;
    let start = cartoboost_core::forecasting::parse_forecast_timestamp("1970-01-01")?;
    let rows = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            Ok(CoreForecastRow::single(
                frequency.advance(start, index)?,
                value,
            ))
        })
        .collect::<cartoboost_core::Result<Vec<_>>>()?;
    CoreForecastFrame::new(rows, frequency)
}

fn utility_forecaster(model: &str, params: &Value) -> cartoboost_core::Result<Box<dyn Forecaster>> {
    match model {
        "naive" => Ok(Box::new(CoreNaiveForecaster::new())),
        "seasonal_naive" | "seasonal-naive" => {
            let season_length = utility_usize_param(params, "season_length")?.unwrap_or(1);
            Ok(Box::new(CoreSeasonalNaiveForecaster::new(season_length)?))
        }
        "theta" => {
            let theta = utility_f64_param(params, "theta")?.unwrap_or(2.0);
            let alpha = utility_f64_param(params, "alpha")?.unwrap_or(0.5);
            Ok(Box::new(CoreThetaForecaster::new(theta, alpha)?))
        }
        "optimized_theta" | "optimized-theta" => {
            let theta_grid =
                utility_f64_vec_param(params, "theta_grid")?.unwrap_or_else(|| vec![1.0, 2.0]);
            let alpha_grid =
                utility_f64_vec_param(params, "alpha_grid")?.unwrap_or_else(|| vec![0.2, 0.5, 0.8]);
            Ok(Box::new(CoreOptimizedThetaForecaster::new(
                theta_grid, alpha_grid,
            )?))
        }
        "ets" => {
            let alpha = utility_f64_param(params, "alpha")?.unwrap_or(0.5);
            let beta = utility_f64_param(params, "beta")?.unwrap_or(0.1);
            let gamma = utility_f64_param(params, "gamma")?;
            let season_length = utility_usize_param(params, "season_length")?;
            Ok(Box::new(CoreETSForecaster::with_additive_seasonality(
                alpha,
                beta,
                gamma,
                season_length,
            )?))
        }
        "arima" => {
            let p = utility_usize_param(params, "p")?.unwrap_or(1);
            let d = utility_usize_param(params, "d")?.unwrap_or(0);
            let q = utility_usize_param(params, "q")?.unwrap_or(0);
            Ok(Box::new(CoreArimaForecaster::new(p, d, q)?))
        }
        "auto_arima" | "auto-arima" => {
            let max_p = utility_usize_param(params, "max_p")?.unwrap_or(3);
            let max_d = utility_usize_param(params, "max_d")?.unwrap_or(1);
            let max_q = utility_usize_param(params, "max_q")?.unwrap_or(2);
            Ok(Box::new(CoreAutoARIMAForecaster::with_max_order(
                max_p, max_d, max_q,
            )?))
        }
        "kalman" | "local_linear_trend_kalman" | "local-linear-trend-kalman" => {
            let level_process_variance =
                utility_f64_param(params, "level_process_variance")?.unwrap_or(0.05);
            let trend_process_variance =
                utility_f64_param(params, "trend_process_variance")?.unwrap_or(0.005);
            let observation_variance =
                utility_f64_param(params, "observation_variance")?.unwrap_or(1.0);
            Ok(Box::new(CoreKalmanForecaster::new(
                level_process_variance,
                trend_process_variance,
                observation_variance,
            )?))
        }
        "auto_kalman" | "self_tuning_kalman" | "self-tuning-kalman" => {
            let level_process_variance_grid =
                utility_f64_vec_param(params, "level_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
            let trend_process_variance_grid =
                utility_f64_vec_param(params, "trend_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]);
            let observation_variance_grid =
                utility_f64_vec_param(params, "observation_variance_grid")?
                    .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
            let validation_window = utility_usize_param(params, "validation_window")?;
            Ok(Box::new(CoreAutoKalmanForecaster::with_grids(
                level_process_variance_grid,
                trend_process_variance_grid,
                observation_variance_grid,
                validation_window,
            )?))
        }
        "local_level_kalman" | "local-level-kalman" => {
            let level_process_variance =
                utility_f64_param(params, "level_process_variance")?.unwrap_or(0.05);
            let observation_variance =
                utility_f64_param(params, "observation_variance")?.unwrap_or(1.0);
            Ok(Box::new(CoreLocalLevelKalmanForecaster::new(
                level_process_variance,
                observation_variance,
            )?))
        }
        "auto_local_level_kalman" | "auto-local-level-kalman" => {
            let level_process_variance_grid =
                utility_f64_vec_param(params, "level_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
            let observation_variance_grid =
                utility_f64_vec_param(params, "observation_variance_grid")?
                    .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
            let validation_window = utility_usize_param(params, "validation_window")?;
            Ok(Box::new(CoreAutoLocalLevelKalmanForecaster::with_grids(
                level_process_variance_grid,
                observation_variance_grid,
                validation_window,
            )?))
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown utility series forecast model {other:?}"
        ))),
    }
}

fn utility_f64_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<f64>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!("parameter {name} must be numeric"))
            })
            .map(Some),
    }
}

fn utility_usize_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<usize>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => {
            let raw = value.as_u64().ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "parameter {name} must be a nonnegative integer"
                ))
            })?;
            usize::try_from(raw)
                .map_err(|_| {
                    CartoBoostError::InvalidInput(format!("parameter {name} is too large"))
                })
                .map(Some)
        }
    }
}

fn utility_f64_vec_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<Vec<f64>>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "parameter {name} must contain only numbers"
                    ))
                })
            })
            .collect::<cartoboost_core::Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(CartoBoostError::InvalidInput(format!(
            "parameter {name} must be a numeric array"
        ))),
    }
}

