#[pyclass(name = "CoordinateMatrix")]
#[derive(Clone, Debug)]
struct NativeCoordinateMatrix {
    inner: CoreCoordinateMatrix,
}

#[pymethods]
impl NativeCoordinateMatrix {
    #[new]
    #[pyo3(signature = (x, y, crs=None, id_col=None))]
    fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        crs: Option<String>,
        id_col: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCoordinateMatrix::new(x, y, crs, id_col).map_err(to_py_geo_core_error)?,
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCoordinateMatrix::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "TimeIndex")]
#[derive(Clone, Debug)]
struct NativeTimeIndex {
    inner: CoreTimeIndex,
}

#[pymethods]
impl NativeTimeIndex {
    #[new]
    #[pyo3(signature = (timestamps, frequency=None, timezone=None))]
    fn new(
        timestamps: Vec<String>,
        frequency: Option<String>,
        timezone: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreTimeIndex::new(timestamps, frequency, timezone)
                .map_err(to_py_geo_core_error)?,
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn timestamps(&self) -> Vec<String> {
        self.inner.iso_strings()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreTimeIndex::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "PanelIndex")]
#[derive(Clone, Debug)]
struct NativePanelIndex {
    inner: CorePanelIndex,
}

#[pymethods]
impl NativePanelIndex {
    #[new]
    #[pyo3(signature = (entity_ids, time=None))]
    fn new(entity_ids: Vec<String>, time: Option<&NativeTimeIndex>) -> PyResult<Self> {
        Ok(Self {
            inner: CorePanelIndex::new(entity_ids, time.map(|value| value.inner.clone()))
                .map_err(to_py_geo_core_error)?,
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CorePanelIndex::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "GeoSpatialWeights")]
#[derive(Clone, Debug)]
struct NativeGeoSpatialWeights {
    inner: CoreGeoSpatialWeights,
}

#[pymethods]
impl NativeGeoSpatialWeights {
    #[new]
    #[pyo3(signature = (n_nodes, indptr, indices, data, node_ids=None, row_normalized=false))]
    fn new(
        n_nodes: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        node_ids: Option<Vec<String>>,
        row_normalized: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGeoSpatialWeights::new(
                n_nodes,
                indptr,
                indices,
                data,
                node_ids,
                row_normalized,
            )
            .map_err(to_py_geo_core_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n_nodes, edges, symmetric=false))]
    fn from_edges(
        n_nodes: usize,
        edges: Vec<(usize, usize, f64)>,
        symmetric: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGeoSpatialWeights::from_edges(n_nodes, edges, symmetric)
                .map_err(to_py_geo_core_error)?,
        })
    }

    fn row_normalize(&self) -> Self {
        Self {
            inner: self.inner.row_normalize(),
        }
    }

    fn is_symmetric(&self, tolerance: f64) -> bool {
        self.inner.is_symmetric(tolerance)
    }

    fn isolated_nodes(&self) -> Vec<usize> {
        self.inner.isolated_nodes()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGeoSpatialWeights::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "SplitManifest")]
#[derive(Clone, Debug)]
struct NativeSplitManifest {
    inner: CoreSplitManifest,
}

#[pymethods]
impl NativeSplitManifest {
    fn hash(&self) -> PyResult<String> {
        self.inner.hash().map_err(to_py_geo_core_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    fn folds(&self) -> Vec<(String, Vec<usize>, Vec<usize>)> {
        self.inner
            .folds
            .iter()
            .map(|fold| {
                (
                    fold.fold_id.clone(),
                    fold.train_indices.clone(),
                    fold.test_indices.clone(),
                )
            })
            .collect()
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreSplitManifest::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (coords, n_folds, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="spatial_block_cv"))]
#[allow(clippy::too_many_arguments)]
fn geo_spatial_block_cv(
    coords: &NativeCoordinateMatrix,
    n_folds: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        coords.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_spatial_block_cv(&coords.inner, n_folds, meta, split_id.to_string())
            .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (coords, n_folds, buffer_distance, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="buffered_spatial_cv"))]
#[allow(clippy::too_many_arguments)]
fn geo_buffered_spatial_cv(
    coords: &NativeCoordinateMatrix,
    n_folds: usize,
    buffer_distance: f64,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        coords.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_buffered_spatial_cv(
            &coords.inner,
            n_folds,
            buffer_distance,
            meta,
            split_id.to_string(),
        )
        .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (groups, n_folds, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="group_spatial_cv"))]
#[allow(clippy::too_many_arguments)]
fn geo_group_spatial_cv(
    groups: Vec<String>,
    n_folds: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        groups.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_group_spatial_cv(groups, n_folds, meta, split_id.to_string())
            .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (panel, min_train_size, horizon, step, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="rolling_origin_panel_split"))]
#[allow(clippy::too_many_arguments)]
fn geo_rolling_origin_panel_split(
    panel: &NativePanelIndex,
    min_train_size: usize,
    horizon: usize,
    step: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        panel.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_rolling_origin_panel_split(
            &panel.inner,
            min_train_size,
            horizon,
            step,
            meta,
            split_id.to_string(),
        )
        .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (coords, time, n_spatial_folds, min_train_time, horizon, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="spatial_temporal_blocked_split"))]
#[allow(clippy::too_many_arguments)]
fn geo_spatial_temporal_blocked_split(
    coords: &NativeCoordinateMatrix,
    time: &NativeTimeIndex,
    n_spatial_folds: usize,
    min_train_time: usize,
    horizon: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        coords.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_spatial_temporal_blocked_split(
            &coords.inner,
            &time.inner,
            n_spatial_folds,
            min_train_time,
            horizon,
            meta,
            split_id.to_string(),
        )
        .map_err(to_py_geo_core_error)?,
    })
}

fn geo_meta(
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    row_count: usize,
    split_id: Option<String>,
) -> PyResult<CoreGeoFrameMeta> {
    CoreGeoFrameMeta::new(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        row_count,
        split_id,
    )
    .map_err(to_py_geo_core_error)
}

#[pyclass(name = "SpatialWeights")]
#[derive(Clone, Debug)]
struct NativeSpatialWeights {
    weights: SpatialWeights,
}

#[pymethods]
impl NativeSpatialWeights {
    #[new]
    #[pyo3(signature = (n_rows, n_cols, rows, cols, values, row_standardize=true))]
    fn new(
        n_rows: usize,
        n_cols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        row_standardize: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            weights: spatial_weights_from_coo(n_rows, n_cols, rows, cols, values, row_standardize)
                .map_err(to_py_spatial_error)?,
        })
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.weights.n_nodes
    }

    fn isolated_rows(&self) -> Vec<usize> {
        self.weights.isolated_nodes()
    }
}

macro_rules! native_spatial_regressor {
    ($name:ident, $py_name:literal, $kind:expr) => {
        #[pyclass(name = $py_name)]
        #[derive(Clone, Debug)]
        struct $name {
            model: Option<SpatialRegressionModel>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self { model: None }
            }

            fn fit(
                &mut self,
                py: Python<'_>,
                x: Vec<Vec<f64>>,
                y: Vec<f64>,
                spatial_weights: &NativeSpatialWeights,
            ) -> PyResult<()> {
                let weights = spatial_weights.weights.clone();
                let model = py
                    .detach(|| SpatialRegressionModel::fit($kind, x, y, &weights))
                    .map_err(to_py_spatial_error)?;
                self.model = Some(model);
                Ok(())
            }

            fn predict(
                &self,
                py: Python<'_>,
                x: Vec<Vec<f64>>,
                spatial_weights: &NativeSpatialWeights,
            ) -> PyResult<Vec<f64>> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                let weights = spatial_weights.weights.clone();
                py.detach(|| model.predict(x, &weights))
                    .map_err(to_py_spatial_error)
            }

            fn diagnostics_json(&self) -> PyResult<String> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                serde_json::to_string(model.diagnostics()).map_err(|err| {
                    PyRuntimeError::new_err(format!("failed to serialize diagnostics: {err}"))
                })
            }

            fn coefficients(&self) -> PyResult<Vec<f64>> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                Ok(model.coefficients().to_vec())
            }

            fn durbin_coefficients(&self) -> PyResult<Vec<f64>> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                Ok(model.durbin_coefficients().to_vec())
            }

            fn intercept(&self) -> PyResult<f64> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                Ok(model.intercept())
            }

            fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                py.detach(|| model.save(path)).map_err(to_py_spatial_error)
            }

            #[classmethod]
            fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: PathBuf) -> PyResult<Self> {
                let model = py
                    .detach(|| SpatialRegressionModel::load(path))
                    .map_err(to_py_spatial_error)?;
                if model.kind() != $kind {
                    return Err(PyValueError::new_err(format!(
                        "artifact contains {:?}, but {} requires {:?}",
                        model.kind(),
                        $py_name,
                        $kind
                    )));
                }
                Ok(Self { model: Some(model) })
            }
        }
    };
}

native_spatial_regressor!(
    NativeSpatialLagRegressor,
    "SpatialLagRegressor",
    SpatialModelKind::SpatialLag
);
native_spatial_regressor!(
    NativeSpatialErrorRegressor,
    "SpatialErrorRegressor",
    SpatialModelKind::SpatialError
);
native_spatial_regressor!(
    NativeSpatialDurbinRegressor,
    "SpatialDurbinRegressor",
    SpatialModelKind::SpatialDurbin
);
native_spatial_regressor!(
    NativeSpatialTwoStageLeastSquares,
    "SpatialTwoStageLeastSquares",
    SpatialModelKind::SpatialTwoStageLeastSquares
);

