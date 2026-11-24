use pyo3::prelude::*;

#[pyfunction]
fn r_squared_arrays(
    py: Python<'_>,
    actual: numpy::PyReadonlyArray1<'_, f64>,
    predicted: numpy::PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let actual = super::numpy_row_major(&actual)?;
    let predicted = super::numpy_row_major(&predicted)?;
    py.detach(|| cartoboost_core::metrics::r_squared(&actual, &predicted))
        .map_err(super::to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (labels, scores, sample_weight=None))]
fn binary_roc_auc_value(
    py: Python<'_>,
    labels: Vec<f64>,
    scores: Vec<f64>,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<f64> {
    py.detach(|| {
        cartoboost_core::metrics::binary_roc_auc(&labels, &scores, sample_weight.as_deref())
    })
    .map_err(super::to_py_value_error)
}

#[pyfunction]
fn graph_mean_reciprocal_rank_value(
    py: Python<'_>,
    labels: Vec<f64>,
    scores: Vec<f64>,
    group_ids: Vec<usize>,
) -> PyResult<f64> {
    py.detach(|| cartoboost_core::metrics::graph_mean_reciprocal_rank(&labels, &scores, &group_ids))
        .map_err(super::to_py_value_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(r_squared_arrays, module)?)?;
    module.add_function(wrap_pyfunction!(binary_roc_auc_value, module)?)?;
    module.add_function(wrap_pyfunction!(graph_mean_reciprocal_rank_value, module)?)?;
    Ok(())
}
