//! Typed input transfer for causal representation reports.
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (features, outcomes, regions, heldout_region, backend=None))]
fn geo_causal_representation_report_arrays(
    py: Python<'_>,
    features: PyReadonlyArray2<'_, f64>,
    outcomes: PyReadonlyArray1<'_, f64>,
    regions: Vec<String>,
    heldout_region: &str,
    backend: Option<&str>,
) -> PyResult<String> {
    let columns = features.shape()[1];
    if columns == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "features must have nonempty rows",
        ));
    }
    let features = super::numpy_row_major(&features)?;
    let outcomes = super::numpy_row_major(&outcomes)?;
    py.detach(|| {
        cartoboost_geo_causal::causal_representation_report_json_with_backend(
            &features
                .chunks_exact(columns)
                .map(<[f64]>::to_vec)
                .collect::<Vec<_>>(),
            &outcomes,
            &regions,
            heldout_region,
            backend,
        )
    })
    .map_err(super::to_py_geo_causal_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        geo_causal_representation_report_arrays,
        module
    )?)?;
    Ok(())
}
