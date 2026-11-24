use cartoboost_neural::regime::RegimeMoEModel;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

#[pyclass(name = "RegimeMoEModel")]
struct PyRegimeMoEModel {
    model: RegimeMoEModel,
}

#[pymethods]
impl PyRegimeMoEModel {
    #[staticmethod]
    fn fit_arrays(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        router: PyReadonlyArray2<'_, f64>,
        widths: Vec<usize>,
        ridge: f64,
        backend: &str,
    ) -> PyResult<Self> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        if router.shape()
            != [
                rows,
                widths
                    .iter()
                    .try_fold(0_usize, |sum, width| sum.checked_add(*width))
                    .ok_or_else(|| value_error("router width overflow"))?,
            ]
        {
            return Err(value_error("invalid router shape"));
        }
        let x = super::numpy_row_major(&x)?;
        let y = super::numpy_row_major(&y)?;
        let router = super::numpy_row_major(&router)?;
        py.detach(|| RegimeMoEModel::fit(&x, rows, cols, &y, &router, &widths, ridge, backend))
            .map(|model| Self { model })
            .map_err(value_error)
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, raw: &str) -> PyResult<Self> {
        py.detach(|| RegimeMoEModel::from_json(raw))
            .map(|model| Self { model })
            .map_err(value_error)
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.model.to_json()).map_err(value_error)
    }

    fn predict_arrays<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        router: PyReadonlyArray2<'_, f64>,
        widths: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        if router.shape()
            != [
                rows,
                widths
                    .iter()
                    .try_fold(0_usize, |sum, width| sum.checked_add(*width))
                    .ok_or_else(|| value_error("router width overflow"))?,
            ]
        {
            return Err(value_error("invalid router shape"));
        }
        let x = super::numpy_row_major(&x)?;
        let router = super::numpy_row_major(&router)?;
        let result = py
            .detach(|| self.model.predict(&x, rows, cols, &router, &widths, false))
            .map_err(value_error)?;
        Ok(result.combined.into_pyarray(py))
    }

    #[allow(clippy::type_complexity)]
    fn components_arrays<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        router: PyReadonlyArray2<'_, f64>,
        widths: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        if router.shape()
            != [
                rows,
                widths
                    .iter()
                    .try_fold(0_usize, |sum, width| sum.checked_add(*width))
                    .ok_or_else(|| value_error("router width overflow"))?,
            ]
        {
            return Err(value_error("invalid router shape"));
        }
        let x = super::numpy_row_major(&x)?;
        let router = super::numpy_row_major(&router)?;
        let result = py
            .detach(|| self.model.predict(&x, rows, cols, &router, &widths, true))
            .map_err(value_error)?;
        Ok((
            numpy::ndarray::Array2::from_shape_vec((rows, 6), result.weights)
                .map_err(value_error)?
                .into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((rows, 6), result.experts)
                .map_err(value_error)?
                .into_pyarray(py),
            result.combined.into_pyarray(py),
        ))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRegimeMoEModel>()?;
    Ok(())
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}
