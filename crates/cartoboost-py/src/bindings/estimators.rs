#[pyclass(name = "NearestNeighborGPRegressor")]
struct NativeNearestNeighborGPRegressor {
    model: CoreNearestNeighborGPRegressor,
}

#[pymethods]
impl NativeNearestNeighborGPRegressor {
    #[new]
    #[pyo3(signature = (kernel="exponential", range=1.0, sill=1.0, nugget=1.0e-6, n_neighbors=16, anisotropy_angle_degrees=0.0, anisotropy_scaling=1.0, brute_force_threshold=2048, duplicate_tolerance=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel: &str,
        range: f64,
        sill: f64,
        nugget: f64,
        n_neighbors: usize,
        anisotropy_angle_degrees: f64,
        anisotropy_scaling: f64,
        brute_force_threshold: usize,
        duplicate_tolerance: f64,
    ) -> PyResult<Self> {
        let config = CoreNngpConfig {
            kernel: CoreCovarianceKernel::parse(kernel).map_err(to_py_geostats_error)?,
            range,
            sill,
            nugget,
            anisotropy: CoreGeostatsAnisotropy {
                angle_degrees: anisotropy_angle_degrees,
                scaling: anisotropy_scaling,
            },
            n_neighbors,
            brute_force_threshold,
            duplicate_tolerance,
        };
        Ok(Self {
            model: CoreNearestNeighborGPRegressor::new(config).map_err(to_py_geostats_error)?,
        })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        coords: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let coords = coords_from_array(coords)?;
        let targets = y.as_slice()?.to_vec();
        py.detach(|| self.model.fit(&coords, &targets))
            .map_err(to_py_geostats_error)
    }

    fn predict(
        &self,
        py: Python<'_>,
        coords: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyNngpPrediction> {
        let coords = coords_from_array(coords)?;
        let predictions = py
            .detach(|| self.model.predict(&coords))
            .map_err(to_py_geostats_error)?;
        let means = predictions
            .iter()
            .map(|prediction| prediction.mean)
            .collect();
        let variances = predictions
            .iter()
            .map(|prediction| prediction.variance)
            .collect();
        let neighbors = predictions
            .into_iter()
            .map(|prediction| prediction.neighbor_indices)
            .collect();
        Ok((means, variances, neighbors))
    }

    fn config_json(&self) -> PyResult<String> {
        let config = self.model.config();
        serde_json::to_string(&json!({
            "kernel": config.kernel.as_str(),
            "range": config.range,
            "sill": config.sill,
            "nugget": config.nugget,
            "anisotropy_angle_degrees": config.anisotropy.angle_degrees,
            "anisotropy_scaling": config.anisotropy.scaling,
            "n_neighbors": config.n_neighbors,
            "brute_force_threshold": config.brute_force_threshold,
            "duplicate_tolerance": config.duplicate_tolerance,
        }))
        .map_err(to_py_json_error)
    }
}

#[pyclass(name = "CartoBoostRegressor")]
#[derive(Clone, Debug)]
struct NativeCartoBoostRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    loss: String,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    monotonic_constraints: Vec<i8>,
    model: Option<Model>,
    flat_axis_predictor: Option<FlatAxisPredictor>,
}

#[pymethods]
impl NativeCartoBoostRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_leaf=20, min_gain=1e-8, loss="l2", quantile_alpha=0.5, huber_delta=1.0, log_offset=1.0, splitters=None, leaf_predictor="constant", linear_leaf_features=None, l2_regularization=1.0, constant_l2_regularization=0.0, fuzzy=false, fuzzy_bandwidth=0.0, fuzzy_kernel="linear", n_threads=None, monotonic_constraints=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        loss: &str,
        quantile_alpha: f64,
        huber_delta: f64,
        log_offset: f64,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
        monotonic_constraints: Option<Vec<i8>>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            quantile_alpha,
            huber_delta,
            log_offset,
        )?;
        parse_loss(loss, quantile_alpha, huber_delta, log_offset)?;
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss: loss.to_string(),
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            monotonic_constraints: monotonic_constraints.unwrap_or_default(),
            model: None,
            flat_axis_predictor: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_sets=None, feature_schema_json=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = self.booster_config(splitters, leaf_predictor)?;
        let n_threads = self.n_threads;
        let model = py
            .detach(move || {
                run_with_optional_threads(n_threads, || {
                    Booster::new(config).fit(&dataset, &y, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.set_model(model);
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_offsets=None, sparse_ids=None, feature_schema_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = self.booster_config(splitters, leaf_predictor)?;
        let n_threads = self.n_threads;
        let model = py
            .detach(move || {
                run_with_optional_threads(n_threads, || {
                    Booster::new(config).fit(&dataset, &targets, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.set_model(model);
        Ok(())
    }

    #[pyo3(signature = (x, y, sparse_sets, feature_schema_json=None, sample_weight=None))]
    fn fit_mixed(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.fit(
            py,
            x,
            y,
            sample_weight,
            Some(sparse_sets),
            feature_schema_json,
        )
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.try_predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let n_threads = self.n_threads;
        let predictions = py
            .detach(|| {
                run_with_optional_threads(n_threads, || {
                    // Sparse inputs may be supplied by a caller even when
                    // the fitted forest never selected a sparse split.  In
                    // that case they do not affect routing and should not
                    // disable the dense flat predictor fast path.
                    if !model.requires_sparse_sets() {
                        if let Some(predictor) = &self.flat_axis_predictor {
                            model.validate_dense_flat_prediction_inputs(rows, cols, values)?;
                            Ok(predictor.predict_flat(rows, cols, values))
                        } else {
                            model.try_predict_flat(rows, cols, values, &offsets, &ids)
                        }
                    } else {
                        model.try_predict_flat(rows, cols, values, &offsets, &ids)
                    }
                })
            })
            .map_err(to_py_value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    #[pyo3(signature = (x, sparse_sets))]
    fn predict_mixed(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        self.predict(py, x, Some(sparse_sets))
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict_additive(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.try_predict_additive(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_additive_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let n_threads = self.n_threads;
        py.detach(|| {
            run_with_optional_threads(n_threads, || {
                model.try_predict_additive_flat(rows, cols, values, &offsets, &ids)
            })
        })
        .map_err(to_py_value_error)
    }

    /// Serialize the exact axis-tree ensemble accepted by SHAP's TreeExplainer.
    fn tree_shap_ensemble_json(&self) -> PyResult<String> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let ensemble = model.tree_shap_ensemble().map_err(to_py_value_error)?;
        serde_json::to_string(&ensemble).map_err(to_py_json_error)
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        py.detach(|| model.save(path)).map_err(to_py_error)
    }

    fn save_weights(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        py.detach(|| model.save_weights(path)).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py.detach(|| Model::load(path)).map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[staticmethod]
    fn load_weights(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .detach(|| Model::load_weights(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn loss(&self) -> String {
        self.loss.clone()
    }

    #[getter]
    fn quantile_alpha(&self) -> f64 {
        self.quantile_alpha
    }

    #[getter]
    fn huber_delta(&self) -> f64 {
        self.huber_delta
    }

    #[getter]
    fn log_offset(&self) -> f64 {
        self.log_offset
    }

    #[getter]
    fn leaf_predictor(&self) -> String {
        self.leaf_predictor.clone()
    }

    #[getter]
    fn linear_leaf_features(&self) -> Vec<usize> {
        self.linear_leaf_features.clone()
    }

    #[getter]
    fn l2_regularization(&self) -> f64 {
        self.l2_regularization
    }

    #[getter]
    fn constant_l2_regularization(&self) -> f64 {
        self.constant_l2_regularization
    }

    #[getter]
    fn fuzzy(&self) -> bool {
        self.fuzzy
    }

    #[getter]
    fn fuzzy_bandwidth(&self) -> f64 {
        self.fuzzy_bandwidth
    }

    #[getter]
    fn fuzzy_kernel(&self) -> String {
        self.fuzzy_kernel.clone()
    }

    #[getter]
    fn n_threads(&self) -> Option<usize> {
        self.n_threads
    }

    #[getter]
    fn monotonic_constraints(&self) -> Vec<i8> {
        self.monotonic_constraints.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(Model::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_history_json(&self) -> PyResult<String> {
        self.model
            .as_ref()
            .map(|model| serde_json::to_string(&model.training_history))
            .transpose()
            .map(|value| value.unwrap_or_else(|| "[]".to_string()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostRegressor {
    fn from_model(model: Model) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            monotonic_constraints,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                loss_name(&config.loss).to_string(),
                quantile_alpha(&config.loss),
                huber_delta(&config.loss),
                log_offset(&config.loss),
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                config.monotonic_constraints,
            )
        } else {
            (
                1,
                1,
                0.0,
                "l2".to_string(),
                0.5,
                1.0,
                1.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                Vec::new(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            monotonic_constraints,
            model: Some(model),
            flat_axis_predictor: None,
        })
        .map(|mut regressor| {
            regressor.refresh_prediction_cache();
            regressor
        })
    }

    fn booster_config(
        &self,
        splitters: Vec<SplitterKind>,
        leaf_predictor: LeafPredictorKind,
    ) -> PyResult<BoosterConfig> {
        Ok(BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            loss: parse_loss(
                &self.loss,
                self.quantile_alpha,
                self.huber_delta,
                self.log_offset,
            )?,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            monotonic_constraints: self.monotonic_constraints.clone(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
            graph_leaf_smoothing: None,
        })
    }

    fn set_model(&mut self, model: Model) {
        self.model = Some(model);
        self.refresh_prediction_cache();
    }

    fn refresh_prediction_cache(&mut self) {
        self.flat_axis_predictor = self.model.as_ref().and_then(Model::flat_axis_predictor);
    }
}

#[pyclass(name = "CartoBoostClassifier")]
#[derive(Clone, Debug)]
struct NativeCartoBoostClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    objective: String,
    class_count: usize,
    class_weights: Vec<f64>,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    model: Option<ClassifierModel>,
}

#[pymethods]
impl NativeCartoBoostClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        min_gain=1e-8,
        objective="auto",
        class_count=2,
        class_weights=None,
        splitters=None,
        leaf_predictor="constant",
        linear_leaf_features=None,
        l2_regularization=1.0,
        constant_l2_regularization=0.0,
        fuzzy=false,
        fuzzy_bandwidth=0.0,
        fuzzy_kernel="linear",
        n_threads=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        objective: &str,
        class_count: usize,
        class_weights: Option<Vec<f64>>,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        if class_count < 2 {
            return Err(PyValueError::new_err("class_count must be at least 2"));
        }
        parse_classification_objective(objective, class_count)?;
        let class_weights = class_weights.unwrap_or_default();
        if !class_weights.is_empty() && class_weights.len() != class_count {
            return Err(PyValueError::new_err(format!(
                "class_weights has length {}, but class_count is {class_count}",
                class_weights.len()
            )));
        }
        if class_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(PyValueError::new_err(
                "class_weights must be finite and non-negative",
            ));
        }
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective: objective.to_string(),
            class_count,
            class_weights,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            model: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_sets=None, feature_schema_json=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let config = self.classifier_config()?;
        let n_threads = self.n_threads;
        let model = py
            .detach(move || {
                run_with_optional_threads(n_threads, || {
                    Classifier::new(config).fit(&dataset, &y, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (
        x,
        y,
        sample_weight=None,
        sparse_offsets=None,
        sparse_ids=None,
        feature_schema_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let config = self.classifier_config()?;
        let n_threads = self.n_threads;
        let model = py
            .detach(move || {
                run_with_optional_threads(n_threads, || {
                    Classifier::new(config).fit(&dataset, &targets, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict_proba(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.predict_proba(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_proba_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.predict_proba(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn decision_function(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.decision_function(&dataset)))
            .map_err(to_py_value_error)
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        py.detach(|| model.save(path)).map_err(to_py_error)
    }

    fn save_weights(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        py.detach(|| model.save(path)).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .detach(|| ClassifierModel::load(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[staticmethod]
    fn load_weights(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Self::load(py, path)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn objective(&self) -> String {
        self.objective.clone()
    }

    #[getter]
    fn class_count(&self) -> usize {
        self.class_count
    }

    #[getter]
    fn class_weights(&self) -> Vec<f64> {
        self.class_weights.clone()
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(ClassifierModel::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn class_values(&self) -> Vec<f64> {
        self.model
            .as_ref()
            .map(|model| model.class_values.clone())
            .unwrap_or_default()
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_history_json(&self) -> PyResult<String> {
        self.model
            .as_ref()
            .map(|model| serde_json::to_string(&model.training_history))
            .transpose()
            .map(|value| value.unwrap_or_else(|| "[]".to_string()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostClassifier {
    fn classifier_config(&self) -> PyResult<ClassifierConfig> {
        Ok(ClassifierConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters: parse_splitters(&self.splitters)?,
            leaf_predictor: parse_leaf_predictor(&self.leaf_predictor)?,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            objective: parse_classification_objective(&self.objective, self.class_count)?,
            class_count: self.class_count,
            class_weights: self.class_weights.clone(),
        })
    }

    fn from_model(model: ClassifierModel) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            class_weights,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                config.class_weights,
            )
        } else {
            (
                1,
                1,
                0.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                Vec::new(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective: classification_objective_name(model.objective).to_string(),
            class_count: model.class_values.len(),
            class_weights,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            model: Some(model),
        })
    }
}

#[pyclass(name = "CartoBoostRanker")]
#[derive(Clone, Debug)]
struct NativeCartoBoostRanker {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    objective: String,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    model: Option<RankerModel>,
}

#[pymethods]
impl NativeCartoBoostRanker {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        min_gain=1e-8,
        objective="lambdarank",
        splitters=None,
        leaf_predictor="constant",
        linear_leaf_features=None,
        l2_regularization=1.0,
        constant_l2_regularization=0.0,
        fuzzy=false,
        fuzzy_bandwidth=0.0,
        fuzzy_kernel="linear",
        n_threads=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        objective: &str,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        parse_ranking_objective(objective)?;
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective: objective.to_string(),
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            model: None,
        })
    }

    #[pyo3(signature = (
        x,
        y,
        groups,
        sample_weight=None,
        sparse_sets=None,
        feature_schema_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        groups: Vec<usize>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let config = self.ranker_config()?;
        let n_threads = self.n_threads;
        let model = py
            .detach(move || {
                run_with_optional_threads(n_threads, || {
                    Ranker::new(config).fit(&dataset, &y, &groups, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (
        x,
        y,
        groups,
        sample_weight=None,
        sparse_offsets=None,
        sparse_ids=None,
        feature_schema_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        groups: Vec<usize>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let config = self.ranker_config()?;
        let n_threads = self.n_threads;
        let model = py
            .detach(move || {
                run_with_optional_threads(n_threads, || {
                    Ranker::new(config).fit(&dataset, &targets, &groups, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let n_threads = self.n_threads;
        py.detach(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, y, groups, sparse_sets=None))]
    fn metrics(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        groups: Vec<usize>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<BTreeMap<String, f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        let metrics = py
            .detach(|| {
                run_with_optional_threads(n_threads, || model.metrics(&dataset, &y, &groups))
            })
            .map_err(to_py_value_error)?;
        Ok(BTreeMap::from([
            ("ndcg".to_string(), metrics.ndcg),
            ("map".to_string(), metrics.map),
            ("mrr".to_string(), metrics.mrr),
        ]))
    }

    #[pyo3(signature = (x, y, groups, sparse_offsets=None, sparse_ids=None))]
    fn metrics_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        groups: Vec<usize>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<BTreeMap<String, f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let targets = y.as_slice()?.to_vec();
        let n_threads = self.n_threads;
        let metrics = py
            .detach(|| {
                run_with_optional_threads(n_threads, || model.metrics(&dataset, &targets, &groups))
            })
            .map_err(to_py_value_error)?;
        Ok(BTreeMap::from([
            ("ndcg".to_string(), metrics.ndcg),
            ("map".to_string(), metrics.map),
            ("mrr".to_string(), metrics.mrr),
        ]))
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        py.detach(|| model.save(path)).map_err(to_py_error)
    }

    fn save_weights(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        py.detach(|| model.save(path)).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py.detach(|| RankerModel::load(path)).map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[staticmethod]
    fn load_weights(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Self::load(py, path)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn objective(&self) -> String {
        self.objective.clone()
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(RankerModel::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_history_json(&self) -> PyResult<String> {
        self.model
            .as_ref()
            .map(|model| serde_json::to_string(&model.training_history))
            .transpose()
            .map(|value| value.unwrap_or_else(|| "[]".to_string()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostRanker {
    fn ranker_config(&self) -> PyResult<RankerConfig> {
        Ok(RankerConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters: parse_splitters(&self.splitters)?,
            leaf_predictor: parse_leaf_predictor(&self.leaf_predictor)?,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            objective: parse_ranking_objective(&self.objective)?,
        })
    }

    fn from_model(model: RankerModel) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            objective,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                ranking_objective_name(config.objective).to_string(),
            )
        } else {
            (
                1,
                1,
                0.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                ranking_objective_name(model.objective).to_string(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            model: Some(model),
        })
    }
}

fn parse_classification_objective(
    name: &str,
    class_count: usize,
) -> PyResult<ClassificationObjective> {
    match name {
        "auto" if class_count == 2 => Ok(ClassificationObjective::BinaryLogLoss),
        "auto" => Ok(ClassificationObjective::MulticlassLogLoss),
        "binary_logloss" | "logloss" | "binary" => Ok(ClassificationObjective::BinaryLogLoss),
        "multiclass_logloss" | "multi_logloss" | "multiclass" => {
            Ok(ClassificationObjective::MulticlassLogLoss)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown classification objective {name:?}; expected 'auto', 'binary_logloss', \
             or 'multiclass_logloss'"
        ))),
    }
}

fn classification_objective_name(objective: ClassificationObjective) -> &'static str {
    match objective {
        ClassificationObjective::BinaryLogLoss => "binary_logloss",
        ClassificationObjective::MulticlassLogLoss => "multiclass_logloss",
    }
}

fn parse_ranking_objective(name: &str) -> PyResult<RankingObjective> {
    match name {
        "pairwise_logit" | "pairwise" => Ok(RankingObjective::PairwiseLogit),
        "lambdarank" | "lambda_rank" => Ok(RankingObjective::LambdaRank),
        _ => Err(PyValueError::new_err(format!(
            "unknown ranking objective {name:?}; expected 'pairwise_logit' or 'lambdarank'"
        ))),
    }
}

fn ranking_objective_name(objective: RankingObjective) -> &'static str {
    match objective {
        RankingObjective::PairwiseLogit => "pairwise_logit",
        RankingObjective::LambdaRank => "lambdarank",
    }
}

#[pyfunction]
#[pyo3(signature = (
    rows,
    targets,
    feature_schema_json=None,
    sample_weight=None,
    low_cardinality_threshold=16,
    smoothing=10.0
))]
fn categorical_fit_transform(
    rows: Vec<Vec<String>>,
    targets: Vec<f64>,
    feature_schema_json: Option<String>,
    sample_weight: Option<Vec<f64>>,
    low_cardinality_threshold: usize,
    smoothing: f64,
) -> PyResult<(Vec<Vec<f64>>, String)> {
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let (dataset, encoder) = CategoricalEncoder::fit_transform_rows(
        &rows,
        &targets,
        schema.as_ref(),
        sample_weight.as_deref(),
        CategoricalEncodingConfig {
            low_cardinality_threshold,
            smoothing,
        },
    )
    .map_err(to_py_value_error)?;
    let encoder_json =
        serde_json::to_string(&encoder).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((dataset_to_rows(&dataset), encoder_json))
}

#[pyfunction]
fn categorical_transform(rows: Vec<Vec<String>>, encoder_json: String) -> PyResult<Vec<Vec<f64>>> {
    let encoder: CategoricalEncoder = serde_json::from_str(&encoder_json)
        .map_err(|err| PyValueError::new_err(format!("invalid categorical encoder: {err}")))?;
    let dataset = encoder.transform_rows(&rows).map_err(to_py_value_error)?;
    Ok(dataset_to_rows(&dataset))
}

/// Validate a serialized feature schema using the Rust core contract.
///
/// Python wrappers normalize ergonomic schema declarations, but the final
/// payload is always checked here before it crosses into dataset/model code.
/// Keeping this validation at the native boundary prevents custom Python
/// schema providers from bypassing duplicate-name, length, or periodic-field
/// checks implemented by `cartoboost-core`.
#[pyfunction]
fn validate_feature_schema_json(payload: &str) -> PyResult<()> {
    let schema: FeatureSchema = serde_json::from_str(payload)
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema JSON: {err}")))?;
    schema.validate().map_err(to_py_value_error)
}

#[pyfunction]
fn model_manifest_json() -> &'static str {
    core_model_manifest_json()
}

fn dataset_to_rows(dataset: &Dataset) -> Vec<Vec<f64>> {
    (0..dataset.n_rows())
        .map(|row| {
            (0..dataset.n_cols())
                .map(|col| dataset.get(row, col))
                .collect()
        })
        .collect()
}

fn parse_splitters(names: &[String]) -> PyResult<Vec<SplitterKind>> {
    let mut splitters = Vec::with_capacity(names.len());
    for name in names {
        let splitter = match name.as_str() {
            "auto" => SplitterKind::Auto,
            "axis" => SplitterKind::Axis,
            "axis_histogram" | "axis_hist" | "histogram" => {
                SplitterKind::AxisHistogram { bins: 64 }
            }
            "diagonal_2d" | "diagonal2d" => SplitterKind::Diagonal2D,
            "gaussian_2d" | "gaussian2d" | "radial" => SplitterKind::Gaussian2D,
            "periodic_time" | "periodic_24" => SplitterKind::Periodic { period: 24.0 },
            "sparse_set" | "sparse" => SplitterKind::SparseSet,
            _ => {
                if let Some(bins) = name
                    .strip_prefix("axis_histogram:")
                    .or_else(|| name.strip_prefix("axis_hist:"))
                    .and_then(|bins| bins.parse::<usize>().ok())
                    .filter(|bins| *bins >= 2)
                {
                    SplitterKind::AxisHistogram { bins }
                } else if let Some(period) = name
                    .strip_prefix("periodic:")
                    .and_then(|period| period.parse::<f64>().ok())
                    .filter(|period| period.is_finite() && *period > 0.0)
                {
                    SplitterKind::Periodic { period }
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unknown splitter {name:?}; expected one of 'auto', 'axis', 'axis_histogram', \
                         'diagonal_2d', 'gaussian_2d', 'periodic_time', or 'sparse_set'"
                    )));
                }
            }
        };
        splitters.push(splitter);
    }
    if splitters.is_empty() {
        Ok(vec![SplitterKind::Auto])
    } else {
        Ok(splitters)
    }
}

fn parse_global_target_mode(name: &str) -> PyResult<GlobalForecastTargetMode> {
    match name {
        "level" => Ok(GlobalForecastTargetMode::Level),
        "delta_from_last" | "delta" => Ok(GlobalForecastTargetMode::DeltaFromLast),
        seasonal if seasonal.starts_with("seasonal_delta_") => {
            parse_seasonal_delta_target_mode(&seasonal["seasonal_delta_".len()..])
        }
        seasonal if seasonal.starts_with("seasonal_delta:") => {
            parse_seasonal_delta_target_mode(&seasonal["seasonal_delta:".len()..])
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown CartoBoostLagForecaster target_mode {name:?}; expected 'level' or \
             'delta_from_last' or 'seasonal_delta_<positive season length>'"
        ))),
    }
}

fn parse_seasonal_delta_target_mode(value: &str) -> PyResult<GlobalForecastTargetMode> {
    let season_length = value.parse::<usize>().map_err(|_| {
        PyValueError::new_err(format!(
            "seasonal_delta target_mode requires a positive integer season length, got {value:?}"
        ))
    })?;
    if season_length == 0 {
        return Err(PyValueError::new_err(
            "seasonal_delta target_mode requires a positive season length",
        ));
    }
    Ok(GlobalForecastTargetMode::SeasonalDelta { season_length })
}

