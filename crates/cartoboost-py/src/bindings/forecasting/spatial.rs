#[pyclass(name = "KrigingForecaster")]
#[derive(Clone, Debug)]
struct NativeKrigingForecaster {
    model: CoreKrigingForecaster,
}

#[pymethods]
impl NativeKrigingForecaster {
    #[new]
    #[pyo3(signature = (
        coordinates,
        range=1.0,
        nugget=1.0e-9,
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
    fn new(
        coordinates: Vec<(String, f64, f64)>,
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
    ) -> PyResult<Self> {
        let coordinates = coordinates
            .into_iter()
            .map(|(series_id, x, y)| (series_id, (x, y)))
            .collect::<BTreeMap<_, _>>();
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
        Ok(Self {
            model: CoreKrigingForecaster::with_config(coordinates, config)
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

#[pyclass(name = "SpatialPiecewiseKrigingForecaster")]
#[derive(Clone, Debug)]
struct NativeSpatialPiecewiseKrigingForecaster {
    model: CoreSpatialPiecewiseKrigingForecaster,
}

#[pymethods]
impl NativeSpatialPiecewiseKrigingForecaster {
    #[new]
    #[pyo3(signature = (
        coordinates,
        mode="residual_kriging",
        spatial_regressors=None,
        range=1.0,
        nugget=1.0e-6,
        sill=1.0,
        variogram_model="exponential",
        drift="ordinary",
        anisotropy_angle_degrees=0.0,
        anisotropy_scaling=1.0,
        max_neighbors=None,
        min_neighbors=1,
        max_distance=None,
        residual_shrinkage=1.0,
        allow_neighbor_fallback=false,
        piecewise_config_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coordinates: Vec<(String, f64, f64)>,
        mode: &str,
        spatial_regressors: Option<Vec<String>>,
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
        residual_shrinkage: f64,
        allow_neighbor_fallback: bool,
        piecewise_config_json: Option<String>,
    ) -> PyResult<Self> {
        let coordinates = coordinates
            .into_iter()
            .map(|(series_id, x, y)| (series_id, (x, y)))
            .collect::<BTreeMap<_, _>>();
        let kriging_config = build_kriging_config(
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
        let piecewise_config = match piecewise_config_json {
            Some(payload) => CorePiecewiseLinearSeasonalForecaster::from_json_string(&payload)
                .map_err(to_py_value_error)?
                .config()
                .clone(),
            None => CorePiecewiseLinearSeasonalConfig::default(),
        };
        let config = CoreSpatialPiecewiseKrigingConfig {
            coordinates,
            mode: parse_spatial_piecewise_kriging_mode(mode)?,
            piecewise_config,
            kriging_config,
            spatial_regressors: spatial_regressors.unwrap_or_default(),
            residual_shrinkage,
            allow_neighbor_fallback,
        };
        Ok(Self {
            model: CoreSpatialPiecewiseKrigingForecaster::new(config).map_err(to_py_value_error)?,
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

