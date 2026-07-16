#[pyclass(name = "ETSForecaster")]
#[derive(Clone, Debug)]
struct NativeETSForecaster {
    model: CoreETSForecaster,
}

#[pymethods]
impl NativeETSForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.5, beta=0.1, gamma=None, season_length=None))]
    fn new(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreETSForecaster::with_additive_seasonality(alpha, beta, gamma, season_length)
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
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn fitted_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.fitted_values(series_id),
            series_id,
            "fitted values",
        )
    }

    fn residuals(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(self.model.residuals(series_id), series_id, "residuals")
    }

    fn level_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.level_values(series_id),
            series_id,
            "level values",
        )
    }

    fn trend_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.trend_values(series_id),
            series_id,
            "trend values",
        )
    }

    fn seasonal_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.seasonal_values(series_id),
            series_id,
            "seasonal values",
        )
    }
}

#[pyclass(name = "ArimaForecaster")]
#[derive(Clone, Debug)]
struct NativeArimaForecaster {
    model: CoreArimaForecaster,
}

#[pymethods]
impl NativeArimaForecaster {
    #[new]
    #[pyo3(signature = (p=1, d=0, q=0))]
    fn new(p: usize, d: usize, q: usize) -> PyResult<Self> {
        Ok(Self {
            model: CoreArimaForecaster::new(p, d, q).map_err(to_py_value_error)?,
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

#[pyclass(name = "AutoARIMAForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoARIMAForecaster {
    model: CoreAutoARIMAForecaster,
}

#[pymethods]
impl NativeAutoARIMAForecaster {
    #[new]
    #[pyo3(signature = (max_p=3, max_d=1, max_q=2))]
    fn new(max_p: usize, max_d: usize, max_q: usize) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoARIMAForecaster::with_max_order(max_p, max_d, max_q)
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
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoStatsBank")]
struct NativeAutoStatsBank {
    model: CoreAutoStatsBank,
}

#[pymethods]
impl NativeAutoStatsBank {
    #[new]
    #[pyo3(signature = (season_length, validation_window=None, validation_objective="mean_squared_error"))]
    fn new(
        season_length: usize,
        validation_window: Option<usize>,
        validation_objective: &str,
    ) -> PyResult<Self> {
        let validation_objective =
            parse_classical_validation_objective(validation_objective, season_length)?;
        Ok(Self {
            model: CoreAutoStatsBank::with_validation_objective(
                season_length,
                validation_window,
                validation_objective,
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

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "CrostonForecaster")]
#[derive(Clone, Debug)]
struct NativeCrostonForecaster {
    model: CoreCrostonForecaster,
}

#[pymethods]
impl NativeCrostonForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.2))]
    fn new(alpha: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreCrostonForecaster::new(alpha).map_err(to_py_value_error)?,
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

#[pyclass(name = "SbaForecaster")]
#[derive(Clone, Debug)]
struct NativeSbaForecaster {
    model: CoreSbaForecaster,
}

#[pymethods]
impl NativeSbaForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.2))]
    fn new(alpha: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreSbaForecaster::new(alpha).map_err(to_py_value_error)?,
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

#[pyclass(name = "TsbForecaster")]
#[derive(Clone, Debug)]
struct NativeTsbForecaster {
    model: CoreTsbForecaster,
}

#[pymethods]
impl NativeTsbForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.2, beta=0.2))]
    fn new(alpha: f64, beta: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreTsbForecaster::new(alpha, beta).map_err(to_py_value_error)?,
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

#[pyclass(name = "KalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeKalmanForecaster {
    model: CoreKalmanForecaster,
}

#[pymethods]
impl NativeKalmanForecaster {
    #[new]
    #[pyo3(signature = (level_process_variance=0.05, trend_process_variance=0.005, observation_variance=1.0))]
    fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreKalmanForecaster::new(
                level_process_variance,
                trend_process_variance,
                observation_variance,
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

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "LocalLevelKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeLocalLevelKalmanForecaster {
    model: CoreLocalLevelKalmanForecaster,
}

#[pymethods]
impl NativeLocalLevelKalmanForecaster {
    #[new]
    #[pyo3(signature = (level_process_variance=0.05, observation_variance=1.0))]
    fn new(level_process_variance: f64, observation_variance: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreLocalLevelKalmanForecaster::new(
                level_process_variance,
                observation_variance,
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

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoKalmanForecaster {
    model: CoreAutoKalmanForecaster,
}

#[pymethods]
impl NativeAutoKalmanForecaster {
    #[new]
    #[pyo3(signature = (
        level_process_variance_grid=None,
        trend_process_variance_grid=None,
        observation_variance_grid=None,
        validation_window=None
    ))]
    fn new(
        level_process_variance_grid: Option<Vec<f64>>,
        trend_process_variance_grid: Option<Vec<f64>>,
        observation_variance_grid: Option<Vec<f64>>,
        validation_window: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoKalmanForecaster::with_grids(
                level_process_variance_grid.unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]),
                trend_process_variance_grid.unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]),
                observation_variance_grid.unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]),
                validation_window,
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

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoLocalLevelKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoLocalLevelKalmanForecaster {
    model: CoreAutoLocalLevelKalmanForecaster,
}

#[pymethods]
impl NativeAutoLocalLevelKalmanForecaster {
    #[new]
    #[pyo3(signature = (
        level_process_variance_grid=None,
        observation_variance_grid=None,
        validation_window=None
    ))]
    fn new(
        level_process_variance_grid: Option<Vec<f64>>,
        observation_variance_grid: Option<Vec<f64>>,
        validation_window: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoLocalLevelKalmanForecaster::with_grids(
                level_process_variance_grid.unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]),
                observation_variance_grid.unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]),
                validation_window,
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

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

