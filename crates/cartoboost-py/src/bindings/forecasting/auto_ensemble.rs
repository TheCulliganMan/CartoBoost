#[pyclass(name = "CartoBoostLagForecaster")]
#[derive(Clone, Debug)]
struct NativeCartoBoostLagForecaster {
    model: CoreCartoBoostLagForecaster,
}

#[pyclass(name = "AutoForecastModel", unsendable)]
#[derive(Clone)]
struct NativeAutoForecastModel {
    model: CoreAutoForecastModel,
}

#[pymethods]
impl NativeAutoForecastModel {
    #[new]
    #[pyo3(signature = (lags=None, rolling_windows=None, partial_rolling_mean_windows=None, rolling_std_windows=None, rolling_min_windows=None, rolling_max_windows=None, ewm_alpha_percents=None, difference_lags=None, rolling_trend_windows=None, covariate_features=None, covariate_indicator_values=None, covariate_calendar_interactions=false, calendar_features=true, rich_calendar_features=false, elapsed_calendar_features=false, elapsed_calendar_periods=None, season_length=7, validation_window=None, validation_origin_count=2, objective="rmse_wape", baseline_displacement_gain=0.03, hard_winner_relative_gain=0.05, min_blend_weight=0.15, max_blend_weight=0.85, max_direct_horizon=28, max_candidate_count=None, recursive=true, n_estimators=None, learning_rate=None, max_depth=None, min_samples_leaf=None, min_gain=None, splitters=None, trend_features=true, target_mode="level"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lags: Option<Vec<usize>>,
        rolling_windows: Option<Vec<usize>>,
        partial_rolling_mean_windows: Option<Vec<usize>>,
        rolling_std_windows: Option<Vec<usize>>,
        rolling_min_windows: Option<Vec<usize>>,
        rolling_max_windows: Option<Vec<usize>>,
        ewm_alpha_percents: Option<Vec<u8>>,
        difference_lags: Option<Vec<usize>>,
        rolling_trend_windows: Option<Vec<usize>>,
        covariate_features: Option<Vec<String>>,
        covariate_indicator_values: Option<BTreeMap<String, Vec<f64>>>,
        covariate_calendar_interactions: bool,
        calendar_features: bool,
        rich_calendar_features: bool,
        elapsed_calendar_features: bool,
        elapsed_calendar_periods: Option<Vec<usize>>,
        season_length: usize,
        validation_window: Option<usize>,
        validation_origin_count: usize,
        objective: &str,
        baseline_displacement_gain: f64,
        hard_winner_relative_gain: f64,
        min_blend_weight: f64,
        max_blend_weight: f64,
        max_direct_horizon: usize,
        max_candidate_count: Option<usize>,
        recursive: bool,
        n_estimators: Option<usize>,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        min_gain: Option<f64>,
        splitters: Option<Vec<String>>,
        trend_features: bool,
        target_mode: &str,
    ) -> PyResult<Self> {
        if !recursive {
            return Err(PyValueError::new_err(
                "AutoForecastModel currently supports recursive=true only",
            ));
        }
        let lags = lags.unwrap_or_else(|| vec![1, 2, 3, 7, 14, 28]);
        let rolling_mean_windows = rolling_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let rolling_std_windows = rolling_std_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let rolling_min_windows = rolling_min_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let rolling_max_windows = rolling_max_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let difference_lags = match difference_lags {
            Some(values) => values,
            None if trend_features => lags.iter().copied().filter(|lag| *lag > 1).collect(),
            None => Vec::new(),
        };
        let rolling_trend_windows = match rolling_trend_windows {
            Some(values) => values,
            None if trend_features => rolling_mean_windows
                .iter()
                .copied()
                .filter(|window| *window > 1)
                .collect(),
            None => Vec::new(),
        };
        let lag_config = LagFeatureConfig {
            difference_lags,
            rolling_trend_windows,
            lags,
            rolling_mean_windows,
            partial_rolling_mean_windows: partial_rolling_mean_windows.unwrap_or_default(),
            rolling_std_windows,
            rolling_min_windows,
            rolling_max_windows,
            ewm_alpha_percents: ewm_alpha_percents.unwrap_or_default(),
            calendar_features: calendar_feature_config(
                calendar_features,
                rich_calendar_features,
                elapsed_calendar_features,
                elapsed_calendar_periods.as_deref(),
            ),
            covariate_features: covariate_features.unwrap_or_default(),
            covariate_indicator_values: covariate_indicator_values.unwrap_or_default(),
            covariate_calendar_interactions,
        };
        let mut booster_config = BoosterConfig::default();
        if let Some(value) = n_estimators {
            booster_config.n_estimators = value;
        }
        if let Some(value) = learning_rate {
            booster_config.learning_rate = value;
        }
        if let Some(value) = max_depth {
            booster_config.max_depth = value;
        }
        if let Some(value) = min_samples_leaf {
            booster_config.min_samples_leaf = value;
        }
        if let Some(value) = min_gain {
            booster_config.min_gain = value;
        }
        if let Some(values) = splitters {
            booster_config.splitters = parse_splitters(&values)?;
        }
        validate_params(
            booster_config.n_estimators,
            booster_config.learning_rate,
            booster_config.max_depth,
            booster_config.min_samples_leaf,
            booster_config.min_gain,
            booster_config.linear_lambda_l2,
            booster_config.constant_lambda_l2,
            booster_config.fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        let target_mode = parse_global_target_mode(target_mode)?;
        let objective = CoreForecastObjective::parse(objective).map_err(to_py_value_error)?;
        Ok(Self {
            model: CoreAutoForecastModel::new(CoreAutoForecastConfig {
                lag_config,
                booster_config,
                target_mode,
                season_length,
                validation_window,
                validation_origin_count,
                objective,
                baseline_displacement_gain,
                hard_winner_relative_gain,
                min_blend_weight,
                max_blend_weight,
                max_direct_horizon,
                max_candidate_count,
            })
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata()).map_err(|err| {
            PyValueError::new_err(format!("failed to serialize forecaster metadata: {err}"))
        })
    }
}

#[pymethods]
impl NativeCartoBoostLagForecaster {
    #[new]
    #[pyo3(signature = (lags=None, rolling_windows=None, partial_rolling_mean_windows=None, rolling_std_windows=None, rolling_min_windows=None, rolling_max_windows=None, ewm_alpha_percents=None, difference_lags=None, rolling_trend_windows=None, covariate_features=None, covariate_indicator_values=None, covariate_calendar_interactions=false, calendar_features=true, rich_calendar_features=false, elapsed_calendar_features=false, elapsed_calendar_periods=None, recursive=true, prediction_interval_levels=None, n_estimators=None, learning_rate=None, max_depth=None, min_samples_leaf=None, min_gain=None, splitters=None, trend_features=true, target_mode="level"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lags: Option<Vec<usize>>,
        rolling_windows: Option<Vec<usize>>,
        partial_rolling_mean_windows: Option<Vec<usize>>,
        rolling_std_windows: Option<Vec<usize>>,
        rolling_min_windows: Option<Vec<usize>>,
        rolling_max_windows: Option<Vec<usize>>,
        ewm_alpha_percents: Option<Vec<u8>>,
        difference_lags: Option<Vec<usize>>,
        rolling_trend_windows: Option<Vec<usize>>,
        covariate_features: Option<Vec<String>>,
        covariate_indicator_values: Option<BTreeMap<String, Vec<f64>>>,
        covariate_calendar_interactions: bool,
        calendar_features: bool,
        rich_calendar_features: bool,
        elapsed_calendar_features: bool,
        elapsed_calendar_periods: Option<Vec<usize>>,
        recursive: bool,
        prediction_interval_levels: Option<Vec<f64>>,
        n_estimators: Option<usize>,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        min_gain: Option<f64>,
        splitters: Option<Vec<String>>,
        trend_features: bool,
        target_mode: &str,
    ) -> PyResult<Self> {
        if !recursive {
            return Err(PyValueError::new_err(
                "CartoBoostLagForecaster currently supports recursive=true only",
            ));
        }
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let lags = lags.unwrap_or_else(|| vec![1, 7, 14]);
        let rolling_mean_windows = rolling_windows.unwrap_or_else(|| vec![7, 28]);
        let rolling_std_windows = rolling_std_windows.unwrap_or_else(|| vec![7, 28]);
        let rolling_min_windows = rolling_min_windows.unwrap_or_else(|| vec![7, 28]);
        let rolling_max_windows = rolling_max_windows.unwrap_or_else(|| vec![7, 28]);
        let difference_lags = match difference_lags {
            Some(values) => values,
            None if trend_features => lags.iter().copied().filter(|lag| *lag > 1).collect(),
            None => Vec::new(),
        };
        let rolling_trend_windows = match rolling_trend_windows {
            Some(values) => values,
            None if trend_features => rolling_mean_windows
                .iter()
                .copied()
                .filter(|window| *window > 1)
                .collect(),
            None => Vec::new(),
        };
        let config = LagFeatureConfig {
            difference_lags,
            rolling_trend_windows,
            lags,
            rolling_mean_windows,
            partial_rolling_mean_windows: partial_rolling_mean_windows.unwrap_or_default(),
            rolling_std_windows,
            rolling_min_windows,
            rolling_max_windows,
            ewm_alpha_percents: ewm_alpha_percents.unwrap_or_default(),
            calendar_features: calendar_feature_config(
                calendar_features,
                rich_calendar_features,
                elapsed_calendar_features,
                elapsed_calendar_periods.as_deref(),
            ),
            covariate_features: covariate_features.unwrap_or_default(),
            covariate_indicator_values: covariate_indicator_values.unwrap_or_default(),
            covariate_calendar_interactions,
        };
        let mut booster_config = BoosterConfig::default();
        if let Some(value) = n_estimators {
            booster_config.n_estimators = value;
        }
        if let Some(value) = learning_rate {
            booster_config.learning_rate = value;
        }
        if let Some(value) = max_depth {
            booster_config.max_depth = value;
        }
        if let Some(value) = min_samples_leaf {
            booster_config.min_samples_leaf = value;
        }
        if let Some(value) = min_gain {
            booster_config.min_gain = value;
        }
        if let Some(values) = splitters {
            booster_config.splitters = parse_splitters(&values)?;
        }
        validate_params(
            booster_config.n_estimators,
            booster_config.learning_rate,
            booster_config.max_depth,
            booster_config.min_samples_leaf,
            booster_config.min_gain,
            booster_config.linear_lambda_l2,
            booster_config.constant_lambda_l2,
            booster_config.fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        let target_mode = parse_global_target_mode(target_mode)?;
        Ok(Self {
            model: CoreCartoBoostLagForecaster::new_with_target_mode(
                config,
                booster_config,
                target_mode,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn predict_with_known_future(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeForecastResult> {
        let mut covariates = BTreeMap::new();
        for row in frame.frame.rows() {
            covariates.insert(
                (row.series_id.clone(), row.timestamp),
                row.covariates.clone(),
            );
        }
        forecast_to_py(py.detach(|| {
            self.model
                .predict_with_known_future_covariates(horizon, &covariates)
        }))
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata()).map_err(|err| {
            PyValueError::new_err(format!("failed to serialize forecaster metadata: {err}"))
        })
    }
}

#[pyclass(name = "WeightedEnsembleForecaster", unsendable)]
struct NativeWeightedEnsembleForecaster {
    model: CoreWeightedEnsembleForecaster,
}

#[pymethods]
impl NativeWeightedEnsembleForecaster {
    #[new]
    fn new(py: Python<'_>, members: Vec<(String, Py<PyAny>, f64)>) -> PyResult<Self> {
        let members = members
            .iter()
            .map(|(name, model, weight)| {
                Ok((name.clone(), boxed_forecaster_from_py(py, model)?, *weight))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            model: CoreWeightedEnsembleForecaster::new(members).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

