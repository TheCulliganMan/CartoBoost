#[pyclass(name = "NaiveForecaster")]
#[derive(Clone, Debug)]
struct NativeNaiveForecaster {
    model: CoreNaiveForecaster,
}

#[pymethods]
impl NativeNaiveForecaster {
    #[new]
    #[pyo3(signature = (prediction_interval_levels=None))]
    fn new(prediction_interval_levels: Option<Vec<f64>>) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        Ok(Self {
            model: CoreNaiveForecaster::new(),
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

    fn save(&self, path: PathBuf) -> PyResult<()> {
        let payload = serde_json::to_string(&self.model).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to serialize NaiveForecaster: {err}"))
        })?;
        std::fs::write(path, payload).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to write NaiveForecaster artifact: {err}"))
        })
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        let payload = std::fs::read_to_string(path).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to read NaiveForecaster artifact: {err}"))
        })?;
        let model = serde_json::from_str(&payload).map_err(|err| {
            PyValueError::new_err(format!("failed to parse NaiveForecaster artifact: {err}"))
        })?;
        Ok(Self { model })
    }
}

#[pyclass(name = "SeasonalNaiveForecaster")]
#[derive(Clone, Debug)]
struct NativeSeasonalNaiveForecaster {
    model: CoreSeasonalNaiveForecaster,
}

#[pymethods]
impl NativeSeasonalNaiveForecaster {
    #[new]
    #[pyo3(signature = (season_length, prediction_interval_levels=None))]
    fn new(season_length: usize, prediction_interval_levels: Option<Vec<f64>>) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        Ok(Self {
            model: CoreSeasonalNaiveForecaster::new(season_length).map_err(to_py_value_error)?,
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

    fn save(&self, path: PathBuf) -> PyResult<()> {
        let payload = serde_json::to_string(&self.model).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "failed to serialize SeasonalNaiveForecaster: {err}"
            ))
        })?;
        std::fs::write(path, payload).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "failed to write SeasonalNaiveForecaster artifact: {err}"
            ))
        })
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        let payload = std::fs::read_to_string(path).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "failed to read SeasonalNaiveForecaster artifact: {err}"
            ))
        })?;
        let model = serde_json::from_str(&payload).map_err(|err| {
            PyValueError::new_err(format!(
                "failed to parse SeasonalNaiveForecaster artifact: {err}"
            ))
        })?;
        Ok(Self { model })
    }
}

#[pyclass(name = "ThetaForecaster")]
#[derive(Clone, Debug)]
struct NativeThetaForecaster {
    model: CoreThetaForecaster,
}

#[pymethods]
impl NativeThetaForecaster {
    #[new]
    #[pyo3(signature = (theta=2.0, alpha=0.2, season_length=None, seasonality=None, prediction_interval_levels=None))]
    fn new(
        theta: f64,
        alpha: f64,
        season_length: Option<usize>,
        seasonality: Option<String>,
        prediction_interval_levels: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let seasonality = parse_theta_seasonality(season_length, seasonality)?;
        Ok(Self {
            model: CoreThetaForecaster::with_seasonality(theta, alpha, seasonality)
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

#[pyclass(name = "OptimizedThetaForecaster")]
#[derive(Clone, Debug)]
struct NativeOptimizedThetaForecaster {
    model: CoreOptimizedThetaForecaster,
}

#[pymethods]
impl NativeOptimizedThetaForecaster {
    #[new]
    #[pyo3(signature = (theta_grid=None, alpha_grid=None, season_length=None, seasonality=None, prediction_interval_levels=None))]
    fn new(
        theta_grid: Option<Vec<f64>>,
        alpha_grid: Option<Vec<f64>>,
        season_length: Option<usize>,
        seasonality: Option<String>,
        prediction_interval_levels: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let seasonality = parse_theta_seasonality(season_length, seasonality)?;
        Ok(Self {
            model: CoreOptimizedThetaForecaster::with_seasonality(
                theta_grid.unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
                alpha_grid.unwrap_or_else(|| vec![0.1, 0.2, 0.4, 0.6, 0.8]),
                seasonality,
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

