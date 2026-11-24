//! NumPy conversions shared by neural and graph bindings.
use super::*;

pub(super) fn rows<T: Element + Clone>(array: PyReadonlyArray2<'_, T>) -> PyResult<Vec<Vec<T>>> {
    let shape = array.shape();
    if shape[1] == 0 {
        return Ok(vec![Vec::new(); shape[0]]);
    }
    let values = numpy_row_major(&array)?;
    Ok(values.chunks_exact(shape[1]).map(<[T]>::to_vec).collect())
}

pub(super) fn pairs(array: PyReadonlyArray2<'_, usize>) -> PyResult<Vec<(usize, usize)>> {
    if array.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "pairs or edges must have shape (n, 2)",
        ));
    }
    let values = numpy_row_major(&array)?;
    Ok(values.chunks_exact(2).map(|row| (row[0], row[1])).collect())
}

pub(super) fn triples(array: PyReadonlyArray2<'_, usize>) -> PyResult<Vec<(usize, usize, usize)>> {
    if array.shape()[1] != 3 {
        return Err(PyValueError::new_err("typed edges must have shape (n, 3)"));
    }
    let values = numpy_row_major(&array)?;
    Ok(values
        .chunks_exact(3)
        .map(|row| (row[0], row[1], row[2]))
        .collect())
}

pub(super) fn matrix<'py, T: Element>(
    py: Python<'py>,
    rows: Vec<Vec<T>>,
    columns: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    if rows.iter().any(|row| row.len() != columns) {
        return Err(PyRuntimeError::new_err(
            "native neural output rows have inconsistent widths",
        ));
    }
    let shape = (rows.len(), columns);
    Array2::from_shape_vec(shape, rows.into_iter().flatten().collect())
        .map(|array| array.into_pyarray(py))
        .map_err(|error| {
            PyRuntimeError::new_err(format!("invalid native neural output shape: {error}"))
        })
}

pub(super) fn flat_matrix<'py, T: Element>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    values: Vec<T>,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    Array2::from_shape_vec((rows, columns), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|error| {
            PyRuntimeError::new_err(format!("invalid native neural output shape: {error}"))
        })
}

#[pyfunction]
#[pyo3(signature = (node_count, edges, embeddings, edge_weights=None, edge_timestamps=None, feature_prefix="graph", requested_features=None, backend=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn graph_compute_directional_features_arrays<'py>(
    py: Python<'py>,
    node_count: usize,
    edges: PyReadonlyArray2<'_, usize>,
    embeddings: PyReadonlyArray2<'_, f32>,
    edge_weights: Option<PyReadonlyArray1<'_, f32>>,
    edge_timestamps: Option<PyReadonlyArray1<'_, f32>>,
    feature_prefix: &str,
    requested_features: Option<Vec<String>>,
    backend: Option<&str>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Vec<String>)> {
    let edges = pairs(edges)?;
    let embeddings = rows(embeddings)?;
    let edge_weights = edge_weights.as_ref().map(numpy_row_major).transpose()?;
    let edge_timestamps = edge_timestamps.as_ref().map(numpy_row_major).transpose()?;
    let requested_features = requested_features.unwrap_or_default();
    let block = py
        .detach(|| {
            compute_directional_features_with_backend(
                node_count,
                &edges,
                &embeddings,
                edge_weights.as_deref(),
                edge_timestamps.as_deref(),
                feature_prefix,
                &requested_features,
                backend,
            )
        })
        .map_err(to_py_neural_error)?;
    Ok((
        matrix(py, block.values, block.feature_names.len())?,
        block.feature_names,
    ))
}
