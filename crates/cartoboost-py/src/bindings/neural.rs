#[pyclass(name = "NeuralEmbeddingFeatures")]
#[derive(Clone)]
struct NativeNeuralEmbeddingFeatures {
    dim: usize,
    fallback: ArtifactFallbackKind,
    random_state: Option<i64>,
    parent_resolution: Option<u8>,
    support_prior_strength: f64,
    table: Option<EmbeddingTable>,
}

#[pymethods]
impl NativeNeuralEmbeddingFeatures {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, parent_resolution=None, support_prior_strength=1.0))]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<i64>,
        parent_resolution: Option<u8>,
        support_prior_strength: f64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be positive"));
        }
        if !support_prior_strength.is_finite() || support_prior_strength <= 0.0 {
            return Err(PyValueError::new_err(
                "support_prior_strength must be positive and finite",
            ));
        }

        let fallback = parse_embedding_fallback(fallback, parent_resolution)?;

        Ok(Self {
            dim,
            fallback,
            random_state,
            parent_resolution,
            support_prior_strength,
            table: None,
        })
    }

    #[pyo3(signature = (ids, target))]
    fn fit(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let ids = ids.as_slice()?.to_vec();
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);

        let table = py
            .detach(|| {
                fit_embedding_table_with_options(
                    self.dim,
                    &ids,
                    &target,
                    self.fallback.clone(),
                    random_state,
                    self.support_prior_strength,
                )
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        Ok(())
    }

    #[pyo3(signature = (ids, target))]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let ids = ids.as_slice()?.to_vec();
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);
        let (table, block) = py
            .detach(|| {
                let table = fit_embedding_table_with_options(
                    self.dim,
                    &ids,
                    &target,
                    self.fallback.clone(),
                    random_state,
                    self.support_prior_strength,
                )?;
                let block = table.encode_ids(&ids, "neural_embedding")?;
                Ok::<_, cartoboost_neural::NeuralError>((table, block))
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (ids))]
    fn transform(&self, py: Python<'_>, ids: PyReadonlyArray1<'_, u64>) -> PyResult<Vec<Vec<f32>>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("transform called before fit or load"))?;

        let ids = ids.as_slice()?.to_vec();
        let block = py
            .detach(|| table.encode_ids(&ids, "neural_embedding"))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (path))]
    fn export(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("export called before fit or load"))?;

        py.detach(|| {
            let artifact = build_embedding_table_artifact(
                self.dim,
                table.rows().to_vec(),
                table.artifact_metadata().fallback.clone(),
            )?;
            write_embedding_table_artifact(path, &artifact)
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[classmethod]
    fn from_artifact(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        let table = py
            .detach(|| EmbeddingTable::load(path))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let metadata = table.artifact_metadata().clone();
        let parent_resolution = match metadata.fallback {
            ArtifactFallbackKind::ParentCell { parent_resolution } => Some(parent_resolution),
            _ => None,
        };

        Ok(Self {
            dim: metadata.dim,
            fallback: metadata.fallback,
            random_state: None,
            parent_resolution,
            support_prior_strength: 1.0,
            table: Some(table),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn fallback(&self) -> String {
        artifact_fallback_name(&self.fallback).to_string()
    }

    #[getter]
    fn random_state(&self) -> Option<i64> {
        self.random_state
    }

    #[getter]
    fn parent_resolution(&self) -> Option<u8> {
        self.parent_resolution
    }

    #[getter]
    fn support_prior_strength(&self) -> f64 {
        self.support_prior_strength
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.table.is_some()
    }

    fn artifact_rows(&self) -> PyResult<Vec<(u64, Vec<f32>)>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("artifact_rows called before fit or load"))?;
        Ok(table
            .rows()
            .iter()
            .map(|row| (row.id, row.values.clone()))
            .collect())
    }
}

fn parse_embedding_fallback(
    value: &str,
    parent_resolution: Option<u8>,
) -> PyResult<ArtifactFallbackKind> {
    match value {
        "zero_vector" => Ok(ArtifactFallbackKind::ZeroVector),
        "global_mean_vector" => Ok(ArtifactFallbackKind::GlobalMeanVector),
        "parent_cell" => parent_resolution
            .map(|parent_resolution| ArtifactFallbackKind::ParentCell { parent_resolution })
            .ok_or_else(|| PyValueError::new_err("parent_resolution is required for parent_cell")),
        _ => Err(PyValueError::new_err(
            "fallback must be one of zero_vector, global_mean_vector, parent_cell",
        )),
    }
}

#[pyclass(name = "GraphSageEncoder")]
#[derive(Clone)]
struct NativeGraphSageEncoder {
    config: GraphSageConfig,
    encoder: GraphSageEncoder,
}

#[pymethods]
impl NativeGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };

        let encoder =
            GraphSageEncoder::new(config.clone(), input_dim).map_err(to_py_neural_error)?;

        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let mut model = GraphSageEncoder::new(self.config.clone(), self.encoder.input_dim())
            .map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .detach(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .detach(|| GraphSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_add_self_loop(&self) -> bool {
        self.config.add_self_loop
    }

    #[getter]
    fn config_l2_regularization(&self) -> f32 {
        self.config.l2_regularization
    }

    #[getter]
    fn hidden_dims(&self) -> Vec<usize> {
        self.config.hidden_dims.clone()
    }
}

#[pyclass(name = "Node2VecEncoder")]
#[derive(Clone)]
struct NativeNode2VecEncoder {
    config: Node2VecConfig,
    encoder: Node2VecEncoder,
}

#[pymethods]
impl NativeNode2VecEncoder {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let encoder = Node2VecEncoder::new(config.clone()).map_err(to_py_neural_error)?;
        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let mut model = Node2VecEncoder::new(self.config.clone()).map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| model.fit(node_count, &edges, edge_weights.as_deref()))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .detach(|| self.encoder.encode())
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .detach(|| Node2VecEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.encoder.node_count()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_p(&self) -> f32 {
        self.config.p
    }

    #[getter]
    fn config_q(&self) -> f32 {
        self.config.q
    }
}

#[pyclass(name = "StandaloneNeuralEmbeddingRegressor")]
#[derive(Clone)]
struct NativeStandaloneNeuralEmbeddingRegressor {
    model: StandaloneNeuralEmbeddingRegressor,
}

#[pymethods]
impl NativeStandaloneNeuralEmbeddingRegressor {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, support_prior_strength=1.0, n_estimators=80, learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<u64>,
        support_prior_strength: f64,
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let fallback = parse_embedding_fallback(fallback, None)?;
        let model = StandaloneNeuralEmbeddingRegressor::new(
            dim,
            fallback,
            random_state,
            support_prior_strength,
            standalone_booster_config(
                n_estimators,
                learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (ids, y, dense=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        ids: Vec<u64>,
        y: Vec<f64>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.detach(|| self.model.fit(&ids, &y, dense.as_deref()))
            .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (ids, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        ids: Vec<u64>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.model.predict(&ids, dense.as_deref()))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| StandaloneNeuralEmbeddingRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneNode2VecRegressor")]
#[derive(Clone)]
struct NativeStandaloneNode2VecRegressor {
    model: Node2VecRegressor,
}

#[pymethods]
impl NativeStandaloneNode2VecRegressor {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let model = Node2VecRegressor::new(
            config,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_count, edges, row_nodes, y, row_targets=None, dense=None, edge_weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.model.fit(
                node_count,
                &edges,
                edge_weights.as_deref(),
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.model
                .predict(&row_nodes, row_targets.as_deref(), dense.as_deref())
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| Node2VecRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneGraphSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneGraphSageRegressor {
    model: GraphSageRegressor,
}

#[pymethods]
impl NativeStandaloneGraphSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = GraphSageRegressor::new(
            config,
            input_dim,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.model.fit(
                &node_features,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| GraphSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHeteroGraphSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneHeteroGraphSageRegressor {
    model: HeteroGraphSageRegressor,
}

#[pymethods]
impl NativeStandaloneHeteroGraphSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = HeteroGraphSageRegressor::new(
            config,
            input_dim,
            relation_count,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.model.fit(
                &node_features,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| HeteroGraphSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHinSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneHinSageRegressor {
    model: HinSageRegressor,
}

#[pymethods]
impl NativeStandaloneHinSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = HinSageRegressor::new(
            config,
            input_dim,
            node_type_count,
            edge_type_triples,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, node_types, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.model.fit(
                &node_features,
                &node_types,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| HinSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneNode2VecLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneNode2VecLinkPredictor {
    model: Node2VecLinkPredictor,
}

#[pymethods]
impl NativeStandaloneNode2VecLinkPredictor {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let model = Node2VecLinkPredictor::new(config).map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<()> {
        py.detach(|| self.model.fit(node_count, &edges, edge_weights.as_deref()))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(&self, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Vec<f64>> {
        py.detach(|| self.model.predict_scores(&pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| Node2VecLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneGraphSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneGraphSageLinkPredictor {
    model: GraphSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneGraphSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = GraphSageLinkPredictor::new(config, input_dim).map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        py.detach(|| self.model.fit(&node_features, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| GraphSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHeteroGraphSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneHeteroGraphSageLinkPredictor {
    model: HeteroGraphSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneHeteroGraphSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = HeteroGraphSageLinkPredictor::new(config, input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize, usize)>,
    ) -> PyResult<()> {
        py.detach(|| self.model.fit(&node_features, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| HeteroGraphSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHinSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneHinSageLinkPredictor {
    model: HinSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneHinSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model =
            HinSageLinkPredictor::new(config, input_dim, node_type_count, edge_type_triples)
                .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
    ) -> PyResult<()> {
        py.detach(|| self.model.fit(&node_features, &node_types, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .detach(|| HinSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "HeteroGraphSageEncoder")]
#[derive(Clone)]
struct NativeHeteroGraphSageEncoder {
    config: HeteroGraphSageConfig,
    relation_count: usize,
    encoder: HeteroGraphSageEncoder,
}

#[pymethods]
impl NativeHeteroGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let encoder = HeteroGraphSageEncoder::new(config.clone(), input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            relation_count,
            encoder,
        })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let mut model = HeteroGraphSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.relation_count,
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .detach(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .detach(|| HeteroGraphSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            relation_count: encoder.relation_count(),
            config,
            encoder,
        })
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.relation_count
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyclass(name = "HinSageEncoder")]
#[derive(Clone)]
struct NativeHinSageEncoder {
    config: HinSageConfig,
    node_type_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    encoder: HinSageEncoder,
}

#[pymethods]
impl NativeHinSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let encoder = HinSageEncoder::new(
            config.clone(),
            input_dim,
            node_type_count,
            edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            node_type_count,
            edge_type_triples,
            encoder,
        })
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let mut model = HinSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.node_type_count,
            self.edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .detach(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .detach(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn link_embeddings(
        &self,
        py: Python<'_>,
        embeddings: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.encoder.link_embeddings(&embeddings, &pairs))
            .map_err(to_py_neural_error)
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .detach(|| HinSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            node_type_count: encoder.node_type_count(),
            edge_type_triples: encoder.edge_type_triples().to_vec(),
            config,
            encoder,
        })
    }

    #[getter]
    fn node_type_count(&self) -> usize {
        self.node_type_count
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.edge_type_triples.len()
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn edge_type_triples(&self) -> Vec<(usize, usize, usize)> {
        self.edge_type_triples.clone()
    }

    #[getter]
    fn neighbor_samples(&self) -> Vec<usize> {
        self.config.neighbor_samples.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

