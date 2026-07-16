#[pyclass(name = "NBeatsForecaster")]
struct NativeNBeatsForecaster {
    model: CoreNBeatsForecaster,
}

#[pyclass(name = "GraphTemporalFrame")]
#[derive(Clone, Debug)]
struct NativeGraphTemporalFrame {
    frame: CoreGraphTemporalFrame,
}

#[pyclass(name = "MarketPanelFrame")]
#[derive(Clone, Debug)]
struct NativeMarketPanelFrame {
    frame: CoreMarketPanelFrame,
}

#[pymethods]
impl NativeMarketPanelFrame {
    #[new]
    #[pyo3(signature = (lane_ids, timestamps, target_names, primary, secondary, origin_ids, destination_ids, coordinates, calendar, hierarchy_groups=None, mix=None, expert_priors_json="[]", expert_labels_json="[]", horizon=1, frequency="daily"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lane_ids: Vec<String>,
        timestamps: Vec<i64>,
        target_names: Vec<String>,
        primary: Vec<Vec<f64>>,
        secondary: Vec<Vec<f64>>,
        origin_ids: Vec<String>,
        destination_ids: Vec<String>,
        coordinates: Vec<Vec<f64>>,
        calendar: Vec<Vec<f64>>,
        hierarchy_groups: Option<Vec<Vec<String>>>,
        mix: Option<Vec<Vec<Vec<f64>>>>,
        expert_priors_json: &str,
        expert_labels_json: &str,
        horizon: usize,
        frequency: &str,
    ) -> PyResult<Self> {
        let coordinates = coordinates.into_iter().map(|point| {
            if point.len() != 4 { return Err(PyValueError::new_err("each coordinate row must contain origin_x, origin_y, destination_x, destination_y")); }
            Ok([point[0], point[1], point[2], point[3]])
        }).collect::<PyResult<Vec<_>>>()?;
        let expert_priors: Vec<CoreExpertRelationshipPrior> =
            serde_json::from_str(expert_priors_json).map_err(|err| {
                PyValueError::new_err(format!("invalid expert priors JSON: {err}"))
            })?;
        let expert_labels: Vec<CoreExpertEventLabel> = serde_json::from_str(expert_labels_json)
            .map_err(|err| PyValueError::new_err(format!("invalid expert labels JSON: {err}")))?;
        Ok(Self {
            frame: CoreMarketPanelFrame::new(
                lane_ids,
                timestamps,
                target_names,
                primary,
                secondary,
                origin_ids,
                destination_ids,
                hierarchy_groups.unwrap_or_else(|| vec![Vec::new(); coordinates.len()]),
                coordinates,
                calendar,
                mix,
                expert_priors,
                expert_labels,
                horizon,
                frequency.to_string(),
            )
            .map_err(to_py_geo_st_error)?,
        })
    }

    #[getter]
    fn lane_ids(&self) -> Vec<String> {
        self.frame.lane_ids.clone()
    }
    #[getter]
    fn target_names(&self) -> Vec<String> {
        self.frame.target_names.clone()
    }
}

#[pyclass(name = "MarketStructureForecaster")]
#[derive(Clone, Debug)]
struct NativeMarketStructureForecaster {
    model: CoreMarketStructureForecaster,
}

#[pymethods]
impl NativeMarketStructureForecaster {
    #[new]
    #[pyo3(signature = (top_k=8, neural_hidden_dim=16, neural_epochs=20, head_epochs=80, head_learning_rate=0.02, huber_delta=1.0, quantile_levels=None, graph_strength=0.55, local_strength=0.35, correlation_floor=0.10, shift_zscore=2.0, calibrate_intervals=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        top_k: usize,
        neural_hidden_dim: usize,
        neural_epochs: usize,
        head_epochs: usize,
        head_learning_rate: f64,
        huber_delta: f64,
        quantile_levels: Option<Vec<f64>>,
        graph_strength: f64,
        local_strength: f64,
        correlation_floor: f64,
        shift_zscore: f64,
        calibrate_intervals: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreMarketStructureForecaster::new(CoreMarketStructureConfig {
                top_k,
                neural_hidden_dim,
                neural_epochs,
                head_epochs,
                head_learning_rate,
                huber_delta,
                quantile_levels: quantile_levels.unwrap_or_else(|| vec![0.1, 0.5, 0.9]),
                graph_strength,
                local_strength,
                correlation_floor,
                shift_zscore,
                calibrate_intervals,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }
    fn fit(&mut self, py: Python<'_>, frame: &NativeMarketPanelFrame) -> PyResult<()> {
        py.detach(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }
    fn predict_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_calendar: Option<Vec<Vec<f64>>>,
    ) -> PyResult<String> {
        let rows = py
            .detach(|| self.model.predict(horizon, future_calendar.as_deref()))
            .map_err(to_py_geo_st_error)?;
        serde_json::to_string(&rows).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
    fn weekly_rollups_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_calendar: Option<Vec<Vec<f64>>>,
    ) -> PyResult<String> {
        let rows = py
            .detach(|| {
                self.model
                    .weekly_rollups(horizon, future_calendar.as_deref())
            })
            .map_err(to_py_geo_st_error)?;
        serde_json::to_string(&rows).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
    fn nowcast_json(&self, py: Python<'_>) -> PyResult<String> {
        let rows = py
            .detach(|| self.model.nowcast())
            .map_err(to_py_geo_st_error)?;
        serde_json::to_string(&rows).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
    fn relationships_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.relationships().map_err(to_py_geo_st_error)?)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
    fn explorer_json(&self, py: Python<'_>, horizon: usize) -> PyResult<String> {
        let payload = py
            .detach(|| self.model.explorer_payload(horizon))
            .map_err(to_py_geo_st_error)?;
        serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreMarketStructureForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }
    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreMarketStructureForecaster::from_json_string(value)
                .map_err(to_py_geo_st_error)?,
        })
    }
}

#[pymethods]
impl NativeGraphTemporalFrame {
    #[new]
    #[pyo3(signature = (node_ids, timestamps, target, indptr, indices, data, horizon, frequency, covariates=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        node_ids: Vec<String>,
        timestamps: Vec<i64>,
        target: Vec<Vec<f64>>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        horizon: usize,
        frequency: String,
        covariates: Option<Vec<Vec<Vec<f64>>>>,
    ) -> PyResult<Self> {
        let adjacency = CoreStCsrAdjacency::new(indptr, indices, data, node_ids.len())
            .map_err(to_py_geo_st_error)?;
        Ok(Self {
            frame: CoreGraphTemporalFrame::new(
                node_ids, timestamps, target, covariates, adjacency, horizon, frequency,
            )
            .map_err(to_py_geo_st_error)?,
        })
    }

    #[getter]
    fn node_ids(&self) -> Vec<String> {
        self.frame.node_ids.clone()
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.frame.horizon
    }

    #[getter]
    fn frequency(&self) -> String {
        self.frame.frequency.clone()
    }
}

#[pyclass(name = "DCRNNForecaster")]
#[derive(Clone, Debug)]
struct NativeDcrnnForecaster {
    model: CoreDcrnnForecaster,
}

#[pymethods]
impl NativeDcrnnForecaster {
    #[new]
    #[pyo3(signature = (
        diffusion_steps=2,
        hidden_size=8,
        epochs=160,
        learning_rate=0.03,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.2,
        ridge=0.0001,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        diffusion_steps: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        teacher_forcing_start: f64,
        teacher_forcing_end: f64,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreDcrnnForecaster::new(CoreDcrnnConfig {
                diffusion_steps,
                hidden_size,
                epochs,
                learning_rate,
                teacher_forcing_start,
                teacher_forcing_end,
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.detach(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn backtest(
        &self,
        py: Python<'_>,
        frame: &NativeGraphTemporalFrame,
        train_size: usize,
    ) -> PyResult<String> {
        let metrics = py
            .detach(|| self.model.backtest(&frame.frame, train_size))
            .map_err(to_py_geo_st_error)?;
        serde_json::to_string(&metrics).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreDcrnnForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.config.backend.selected.clone())
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreDcrnnForecaster::from_json_string(value).map_err(to_py_geo_st_error)?,
        })
    }
}

#[pyclass(name = "STAEformerForecaster")]
#[derive(Clone, Debug)]
struct NativeSTAEformerForecaster {
    model: CoreSTAEformerForecaster,
}

#[pyclass(name = "GraphWaveNetForecaster")]
#[derive(Clone, Debug)]
struct NativeGraphWaveNetForecaster {
    model: CoreGraphWaveNetForecaster,
}

#[pyclass(name = "PropagationDelayGraphForecaster")]
#[derive(Clone, Debug)]
struct NativePropagationDelayGraphForecaster {
    model: CoreDelayAwareGraphTransformer,
}

#[pyclass(name = "PaperGraphTransformerForecaster")]
#[derive(Clone, Debug)]
struct NativePaperGraphTransformerForecaster {
    model: CorePaperGraphTransformerForecaster,
}

#[pymethods]
impl NativeSTAEformerForecaster {
    #[new]
    #[pyo3(signature = (
        lookback=8,
        attention_heads=4,
        hidden_size=8,
        epochs=120,
        learning_rate=0.02,
        ridge=0.0001,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lookback: usize,
        attention_heads: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreSTAEformerForecaster::new(CoreSTAEformerConfig {
                lookback,
                attention_heads,
                hidden_size,
                epochs,
                learning_rate,
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.detach(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.detach(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreSTAEformerForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreSTAEformerForecaster::from_json_string(value).map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }
}

#[pymethods]
impl NativeGraphWaveNetForecaster {
    #[new]
    #[pyo3(signature = (
        lookback=8,
        dilation_depth=3,
        hidden_size=8,
        epochs=120,
        learning_rate=0.02,
        ridge=0.0001,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lookback: usize,
        dilation_depth: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreGraphWaveNetForecaster::new(CoreGraphWaveNetConfig {
                lookback,
                dilation_depth,
                hidden_size,
                epochs,
                learning_rate,
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.detach(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.detach(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreGraphWaveNetForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreGraphWaveNetForecaster::from_json_string(value)
                .map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }
}

#[pymethods]
impl NativePropagationDelayGraphForecaster {
    #[new]
    #[pyo3(signature = (horizon=1, edge_delay_prior=None, ridge=0.000001, backend=None))]
    fn new(
        horizon: usize,
        edge_delay_prior: Option<Vec<usize>>,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreDelayAwareGraphTransformer::new(CoreDelayAwareGraphConfig {
                horizon,
                edge_delay_prior: edge_delay_prior.unwrap_or_default(),
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.detach(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.detach(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn edge_delay_sensitivity(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.edge_delay_sensitivity())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreDelayAwareGraphTransformer::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreDelayAwareGraphTransformer::from_json_string(value)
                .map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }
}

#[pymethods]
impl NativePaperGraphTransformerForecaster {
    #[new]
    #[pyo3(signature = (profile, lookback=12, hidden_size=16, attention_heads=4, graph_order=2, experts=4, periodicity=24, recent_window=12, epochs=80, learning_rate=0.01, weight_decay=0.00001, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        profile: &str,
        lookback: usize,
        hidden_size: usize,
        attention_heads: usize,
        graph_order: usize,
        experts: usize,
        periodicity: usize,
        recent_window: usize,
        epochs: usize,
        learning_rate: f64,
        weight_decay: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CorePaperGraphTransformerForecaster::new(CorePaperGraphTransformerConfig {
                profile: parse_graph_transformer_profile(profile)?,
                lookback,
                hidden_size,
                attention_heads,
                graph_order,
                experts,
                periodicity,
                recent_window,
                epochs,
                learning_rate,
                weight_decay,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    #[pyo3(signature = (frame, checkpoint_path=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        frame: &NativeGraphTemporalFrame,
        checkpoint_path: Option<PathBuf>,
    ) -> PyResult<()> {
        py.detach(|| match checkpoint_path {
            Some(path) => self.model.fit_checkpointed(&frame.frame, path),
            None => self.model.fit(&frame.frame),
        })
        .map_err(to_py_geo_st_error)
    }

    fn fit_checkpointed(
        &mut self,
        py: Python<'_>,
        frame: &NativeGraphTemporalFrame,
        checkpoint_path: PathBuf,
    ) -> PyResult<()> {
        py.detach(|| self.model.fit_checkpointed(&frame.frame, checkpoint_path))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.detach(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.detach(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CorePaperGraphTransformerForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CorePaperGraphTransformerForecaster::from_json_string(value)
                .map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }

    fn architecture_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.architecture_report())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pymethods]
impl NativeNBeatsForecaster {
    #[new]
    #[pyo3(signature = (input_size=8, hidden_size=16, epochs=80, learning_rate=0.01, backend=None))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreNBeatsForecaster::new(CoreNBeatsConfig {
                input_size,
                hidden_size,
                epochs,
                learning_rate,
                backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
            })
            .map_err(to_py_neural_error)?,
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

#[pyclass(name = "NHiTSForecaster")]
struct NativeNHiTSForecaster {
    model: CoreNHiTSForecaster,
}

#[pymethods]
impl NativeNHiTSForecaster {
    #[new]
    #[pyo3(signature = (input_size=12, hidden_size=16, epochs=80, learning_rate=0.01, pooling_size=2, backend=None))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        pooling_size: usize,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreNHiTSForecaster::new(CoreNHiTSConfig {
                input_size,
                hidden_size,
                epochs,
                learning_rate,
                pooling_size,
                backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
            })
            .map_err(to_py_neural_error)?,
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

#[pyclass(name = "NeuralPanelForecaster")]
struct NativeNeuralPanelForecaster {
    model: CoreNeuralPanelForecaster,
}

#[pymethods]
impl NativeNeuralPanelForecaster {
    #[new]
    #[pyo3(signature = (
        n_lags=8,
        n_forecasts=1,
        quantiles=None,
        trend="piecewise_linear",
        n_changepoints=10,
        changepoints_range=0.8,
        daily_fourier_order=0,
        weekly_fourier_order=0,
        yearly_fourier_order=0,
        custom_seasonalities=None,
        seasonality_mode="additive",
        events=None,
        event_mode="additive",
        future_regressors=None,
        lagged_regressors=None,
        ar_layers=None,
        lagged_reg_layers=None,
        trend_mode="global",
        seasonality_global_local="global",
        event_global_local="global",
        regressor_global_local="global",
        local_l2=0.0,
        seed=0,
        loss="smooth_l1",
        epochs=80,
        learning_rate=0.01,
        weight_decay=0.0,
        newer_sample_weight=false,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_lags: usize,
        n_forecasts: usize,
        quantiles: Option<Vec<f64>>,
        trend: &str,
        n_changepoints: usize,
        changepoints_range: f64,
        daily_fourier_order: usize,
        weekly_fourier_order: usize,
        yearly_fourier_order: usize,
        custom_seasonalities: Option<Vec<CustomSeasonalitySpec>>,
        seasonality_mode: &str,
        events: Option<BTreeMap<String, Vec<i32>>>,
        event_mode: &str,
        future_regressors: Option<BTreeMap<String, String>>,
        lagged_regressors: Option<BTreeMap<String, usize>>,
        ar_layers: Option<Vec<usize>>,
        lagged_reg_layers: Option<Vec<usize>>,
        trend_mode: &str,
        seasonality_global_local: &str,
        event_global_local: &str,
        regressor_global_local: &str,
        local_l2: f64,
        seed: u64,
        loss: &str,
        epochs: usize,
        learning_rate: f64,
        weight_decay: f64,
        newer_sample_weight: bool,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = neural_panel_config_from_parts(
            n_lags,
            n_forecasts,
            quantiles,
            trend,
            n_changepoints,
            changepoints_range,
            daily_fourier_order,
            weekly_fourier_order,
            yearly_fourier_order,
            custom_seasonalities,
            seasonality_mode,
            events,
            event_mode,
            future_regressors,
            lagged_regressors,
            ar_layers,
            lagged_reg_layers,
            trend_mode,
            seasonality_global_local,
            event_global_local,
            regressor_global_local,
            local_l2,
            seed,
            loss,
            epochs,
            learning_rate,
            weight_decay,
            newer_sample_weight,
            backend,
        )?;
        Ok(Self {
            model: CoreNeuralPanelForecaster::new(config).map_err(to_py_neural_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    #[pyo3(signature = (horizon, frame=None))]
    fn components_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: Option<&NativeForecastFrame>,
    ) -> PyResult<String> {
        let value = if let Some(frame) = frame {
            let mut covariates = BTreeMap::new();
            for row in frame.frame.rows() {
                covariates.insert(
                    (row.series_id.clone(), row.timestamp),
                    row.covariates.clone(),
                );
            }
            py.detach(|| {
                self.model
                    .predict_components_json_value_with_known_future_covariates(
                        horizon,
                        Some(&covariates),
                    )
            })
        } else {
            py.detach(|| self.model.predict_components_json_value(horizon))
        };
        value.map_err(to_py_value_error).and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }

    fn history_components_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.model.history_components_json_string())
            .map_err(to_py_value_error)
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

    fn quantiles_json(&self, py: Python<'_>, horizon: usize) -> PyResult<String> {
        py.detach(|| self.model.predict_quantiles_json_string(horizon))
            .map_err(to_py_value_error)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.model.to_json_string())
            .map_err(to_py_value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let model = py
            .detach(|| CoreNeuralPanelForecaster::from_json_string(value))
            .map_err(to_py_value_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "LaneNeuralPanelForecaster")]
struct NativeLaneNeuralPanelForecaster {
    model: CoreLaneNeuralPanelForecaster,
}

#[pymethods]
impl NativeLaneNeuralPanelForecaster {
    #[new]
    #[pyo3(signature = (
        n_lags=8,
        n_forecasts=1,
        quantiles=None,
        trend="piecewise_linear",
        n_changepoints=10,
        changepoints_range=0.8,
        daily_fourier_order=0,
        weekly_fourier_order=0,
        yearly_fourier_order=0,
        custom_seasonalities=None,
        seasonality_mode="additive",
        events=None,
        event_mode="additive",
        future_regressors=None,
        lagged_regressors=None,
        ar_layers=None,
        lagged_reg_layers=None,
        trend_mode="global",
        seasonality_global_local="global",
        event_global_local="global",
        regressor_global_local="global",
        local_l2=0.0,
        seed=0,
        loss="smooth_l1",
        epochs=80,
        learning_rate=0.01,
        weight_decay=0.0,
        newer_sample_weight=false,
        backend=None,
        embedding_dim=8
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_lags: usize,
        n_forecasts: usize,
        quantiles: Option<Vec<f64>>,
        trend: &str,
        n_changepoints: usize,
        changepoints_range: f64,
        daily_fourier_order: usize,
        weekly_fourier_order: usize,
        yearly_fourier_order: usize,
        custom_seasonalities: Option<Vec<CustomSeasonalitySpec>>,
        seasonality_mode: &str,
        events: Option<BTreeMap<String, Vec<i32>>>,
        event_mode: &str,
        future_regressors: Option<BTreeMap<String, String>>,
        lagged_regressors: Option<BTreeMap<String, usize>>,
        ar_layers: Option<Vec<usize>>,
        lagged_reg_layers: Option<Vec<usize>>,
        trend_mode: &str,
        seasonality_global_local: &str,
        event_global_local: &str,
        regressor_global_local: &str,
        local_l2: f64,
        seed: u64,
        loss: &str,
        epochs: usize,
        learning_rate: f64,
        weight_decay: f64,
        newer_sample_weight: bool,
        backend: Option<&str>,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let base = neural_panel_config_from_parts(
            n_lags,
            n_forecasts,
            quantiles,
            trend,
            n_changepoints,
            changepoints_range,
            daily_fourier_order,
            weekly_fourier_order,
            yearly_fourier_order,
            custom_seasonalities,
            seasonality_mode,
            events,
            event_mode,
            future_regressors,
            lagged_regressors,
            ar_layers,
            lagged_reg_layers,
            trend_mode,
            seasonality_global_local,
            event_global_local,
            regressor_global_local,
            local_l2,
            seed,
            loss,
            epochs,
            learning_rate,
            weight_decay,
            newer_sample_weight,
            backend,
        )?;
        Ok(Self {
            model: CoreLaneNeuralPanelForecaster::new(CoreLaneNeuralPanelConfig {
                base,
                embedding_dim,
            })
            .map_err(to_py_neural_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    #[pyo3(signature = (horizon, frame=None))]
    fn components_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: Option<&NativeForecastFrame>,
    ) -> PyResult<String> {
        let value = if let Some(frame) = frame {
            let mut covariates = BTreeMap::new();
            for row in frame.frame.rows() {
                covariates.insert(
                    (row.series_id.clone(), row.timestamp),
                    row.covariates.clone(),
                );
            }
            py.detach(|| {
                self.model
                    .predict_components_json_value_with_known_future_covariates(
                        horizon,
                        &covariates,
                    )
            })
        } else {
            py.detach(|| self.model.predict_components_json_value(horizon))
        };
        value.map_err(to_py_value_error).and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }

    fn history_components_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.model.history_components_json_string())
            .map_err(to_py_value_error)
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

    fn predict_for_lanes(
        &self,
        py: Python<'_>,
        horizon: usize,
        series_ids: Vec<String>,
    ) -> PyResult<NativeForecastResult> {
        forecast_to_py(py.detach(|| self.model.predict_for_lanes(horizon, &series_ids)))
    }

    fn quantiles_json(&self, py: Python<'_>, horizon: usize) -> PyResult<String> {
        py.detach(|| self.model.predict_quantiles_json_string(horizon))
            .map_err(to_py_value_error)
    }

    fn quantiles_json_for_lanes(
        &self,
        py: Python<'_>,
        horizon: usize,
        series_ids: Vec<String>,
    ) -> PyResult<String> {
        py.detach(|| {
            self.model
                .predict_quantiles_for_lanes_json_string(horizon, &series_ids)
        })
        .map_err(to_py_value_error)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

