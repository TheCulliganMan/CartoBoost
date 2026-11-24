//! Array and record transfer endpoints for the public deep-model facades.
use super::*;
use numpy::ndarray::Array3;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};
use serde::{de::DeserializeOwned, Serialize};

macro_rules! append_record {
    ($py:expr, $list:expr, $row:ident, $($field:ident),+ $(,)?) => {{
        let record = PyDict::new($py);
        $(record.set_item(stringify!($field), $row.$field)?;)+
        $list.append(record)?;
    }};
}

fn records<T: DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    serde_json::from_value(python_value(value)?).map_err(to_py_json_error)
}

fn python_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    python_value_at_depth(value, 0)
}

fn python_value_at_depth(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if depth >= 1024 {
        return Err(PyValueError::new_err(
            "record nesting exceeds the supported depth or contains a cycle",
        ));
    }
    if value.is_none() {
        return Ok(Value::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Value::String(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(number) = value.extract::<i64>() {
            return Ok(number.into());
        }
        if let Ok(number) = value.extract::<u64>() {
            return Ok(number.into());
        }
        let number: f64 = value
            .extract()
            .map_err(|_| PyValueError::new_err("integer is outside the supported JSON range"))?;
        return serde_json::Number::from_f64(number)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("integer is outside the supported JSON range"));
    }
    if value.is_instance_of::<PyFloat>() {
        let number: f64 = value.extract()?;
        return serde_json::Number::from_f64(number)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("record numbers must be finite"));
    }
    if let Ok(mapping) = value.cast::<PyDict>() {
        let mut result = serde_json::Map::new();
        for (key, value) in mapping.iter() {
            let name = if key.is_instance_of::<PyString>() {
                key.extract::<String>()?
            } else if key.is_none() {
                "null".to_owned()
            } else if key.is_instance_of::<PyBool>() {
                if key.extract::<bool>()? {
                    "true"
                } else {
                    "false"
                }
                .to_owned()
            } else if key.is_instance_of::<PyInt>() {
                key.str()?.to_str()?.to_owned()
            } else if key.is_instance_of::<PyFloat>() {
                let number: f64 = key.extract()?;
                if number.is_nan() {
                    "NaN".to_owned()
                } else if number == f64::INFINITY {
                    "Infinity".to_owned()
                } else if number == f64::NEG_INFINITY {
                    "-Infinity".to_owned()
                } else {
                    key.str()?.to_str()?.to_owned()
                }
            } else {
                return Err(PyValueError::new_err(
                    "record mapping keys must be strings, numbers, bool or None",
                ));
            };
            result.insert(name, python_value_at_depth(&value, depth + 1)?);
        }
        return Ok(Value::Object(result));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        return value
            .try_iter()?
            .map(|value| python_value_at_depth(&value?, depth + 1))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    Err(PyValueError::new_err(
        "records must contain JSON-compatible values",
    ))
}

fn object(py: Python<'_>, value: impl Serialize) -> PyResult<Py<PyAny>> {
    json_value_to_python(py, serde_json::to_value(value).map_err(to_py_json_error)?)
}

fn matrix_output<'py>(
    py: Python<'py>,
    values: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = values.len();
    let cols = values.first().map_or(0, Vec::len);
    Array2::from_shape_vec((rows, cols), values.into_iter().flatten().collect())
        .map_err(|err| PyValueError::new_err(err.to_string()))
        .map(|array| array.into_pyarray(py))
}

#[pyfunction(signature=(hidden,residuals,quantiles,sample_count,backend=None))]
fn prob_conditional_flow_fit_arrays(
    py: Python<'_>,
    hidden: PyReadonlyArray2<'_, f64>,
    residuals: PyReadonlyArray1<'_, f64>,
    quantiles: Vec<f64>,
    sample_count: usize,
    backend: Option<&str>,
) -> PyResult<String> {
    let hidden = rows_from_numpy_2d(hidden, "hidden")?;
    let residuals = numpy_row_major(&residuals)?;
    py.detach(|| {
        core_prob_conditional_flow_fit_json(&hidden, &residuals, &quantiles, sample_count, backend)
    })
    .map_err(to_py_value_error)
}

#[pyfunction(signature=(artifact_json,hidden,actual=None))]
fn prob_conditional_flow_predict_arrays<'py>(
    py: Python<'py>,
    artifact_json: &str,
    hidden: PyReadonlyArray2<'_, f64>,
    actual: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let model: cartoboost_prob::ConditionalFlowDistributionHead =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let hidden = rows_from_numpy_2d(hidden, "hidden")?;
    let actual = actual.as_ref().map(numpy_row_major).transpose()?;
    let result = py
        .detach(|| model.predict(&hidden, actual.as_deref()))
        .map_err(to_py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("samples", matrix_output(py, result.samples)?)?;
    out.set_item(
        "joint_scenario_paths",
        matrix_output(py, result.joint_scenario_paths)?,
    )?;
    out.set_item(
        "marginal_quantiles",
        matrix_output(py, result.marginal_quantiles)?,
    )?;
    out.set_item("log_likelihood", result.log_likelihood.into_pyarray(py))?;
    out.set_item("tail_risk_metrics", result.tail_risk_metrics)?;
    out.set_item("metrics", result.metrics)?;
    Ok(out)
}

#[pyfunction(signature=(point_forecast,edges,scenario_count,diffusion_steps,shock_scale,backend=None))]
fn prob_diffusion_scenario_generate_arrays<'py>(
    py: Python<'py>,
    point_forecast: PyReadonlyArray2<'_, f64>,
    edges: Vec<(usize, usize, f64)>,
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
    backend: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let panel = rows_from_numpy_2d(point_forecast, "point_forecast")?;
    let edges: Vec<_> = edges
        .into_iter()
        .map(|(source, target, weight)| CoreDiffusionEdge {
            source,
            target,
            weight,
        })
        .collect();
    let result = py
        .detach(|| {
            cartoboost_prob::GeoTemporalDiffusionScenarioModel::new_with_backend(
                scenario_count,
                diffusion_steps,
                shock_scale,
                backend,
            )?
            .generate(&panel, &edges)
        })
        .map_err(to_py_value_error)?;
    let shape = (result.scenarios.len(), panel.len(), panel[0].len());
    let samples = Array3::from_shape_vec(
        shape,
        result.scenarios.into_iter().flatten().flatten().collect(),
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("scenarios", samples.into_pyarray(py))?;
    out.set_item("scenario_mean", matrix_output(py, result.scenario_mean)?)?;
    out.set_item(
        "scenario_variance",
        matrix_output(py, result.scenario_variance)?,
    )?;
    out.set_item("spatial_correlation", result.spatial_correlation)?;
    out.set_item(
        "point_forecast_comparison",
        result.point_forecast_comparison,
    )?;
    out.set_item("metadata", result.metadata)?;
    Ok(out)
}

#[pyfunction(signature=(field_values,coordinates,edges,exogenous_fields,smoothing,coordinate_scale,backend=None))]
#[allow(clippy::too_many_arguments)]
fn deep_graph_neural_operator_predict_arrays<'py>(
    py: Python<'py>,
    field_values: PyReadonlyArray2<'_, f64>,
    coordinates: PyReadonlyArray2<'_, f64>,
    edges: Vec<(usize, usize, f64)>,
    exogenous_fields: PyReadonlyArray2<'_, f64>,
    smoothing: f64,
    coordinate_scale: f64,
    backend: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = rows_from_numpy_2d(field_values, "field_values")?;
    let coordinates = rows_from_numpy_2d(coordinates, "coordinates")?;
    let exogenous = if exogenous_fields.len() == 0 {
        Vec::new()
    } else {
        rows_from_numpy_2d(exogenous_fields, "exogenous_fields")?
    };
    let edges: Vec<_> = edges
        .into_iter()
        .map(|(source, target, weight)| CoreSpatialOperatorEdge {
            source,
            target,
            weight,
        })
        .collect();
    let result = py
        .detach(|| {
            cartoboost_neural::operator::GraphNeuralOperator::new_with_backend(
                smoothing,
                coordinate_scale,
                backend,
            )?
            .predict(&fields, &coordinates, &edges, &exogenous)
        })
        .map_err(to_py_neural_error)?;
    let out = PyDict::new(py);
    out.set_item("future_field", matrix_output(py, result.future_field)?)?;
    out.set_item("residual_field", matrix_output(py, result.residual_field)?)?;
    out.set_item(
        "uncertainty_field",
        matrix_output(py, result.uncertainty_field)?,
    )?;
    out.set_item("metadata", result.metadata)?;
    Ok(out)
}

#[pyfunction(signature=(rows,response_type,monotone=None,backend=None))]
fn deep_response_curve_fit_records(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    response_type: &str,
    monotone: Option<&str>,
    backend: Option<&str>,
) -> PyResult<String> {
    let rows: Vec<DeepResponseRow> = records(rows)?;
    let result = py
        .detach(|| core_deep_response_curve_fit(&rows, response_type, monotone, backend))
        .map_err(to_py_neural_error)?;
    serde_json::to_string(&result).map_err(to_py_json_error)
}
#[pyfunction]
fn deep_response_curve_predict_records(
    py: Python<'_>,
    artifact_json: &str,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let model: DeepResponseArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepResponseRow> = records(rows)?;
    let result = py
        .detach(|| core_deep_response_curve_predict(&model, &rows))
        .map_err(to_py_neural_error)?;
    let output = PyList::empty(py);
    for row in result {
        append_record!(
            py,
            output,
            row,
            group_id,
            candidate_id,
            candidate_value,
            response_score,
            response_probability,
            calibrated_probability
        );
    }
    Ok(output.into_any().unbind())
}
#[pyfunction(signature=(artifact_json,rows,probability=false))]
fn deep_response_curve_predict_arrays<'py>(
    py: Python<'py>,
    artifact_json: &str,
    rows: &Bound<'_, PyAny>,
    probability: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let model: DeepResponseArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepResponseRow> = records(rows)?;
    let values = py
        .detach(|| {
            core_deep_response_curve_predict(&model, &rows).map(|rows| {
                rows.into_iter()
                    .map(|r| {
                        if probability {
                            r.response_probability.unwrap_or(r.response_score)
                        } else {
                            r.response_score
                        }
                    })
                    .collect::<Vec<_>>()
            })
        })
        .map_err(to_py_neural_error)?;
    Ok(values.into_pyarray(py))
}

#[pyfunction(signature=(rows,backend=None))]
fn deep_service_residual_fit_records(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    backend: Option<&str>,
) -> PyResult<String> {
    let rows: Vec<DeepServiceResidualRow> = records(rows)?;
    let result = py
        .detach(|| core_deep_service_residual_fit(&rows, backend))
        .map_err(to_py_neural_error)?;
    serde_json::to_string(&result).map_err(to_py_json_error)
}
#[pyfunction]
fn deep_service_residual_predict_records(
    py: Python<'_>,
    artifact_json: &str,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let model: DeepServiceResidualArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepServiceResidualRow> = records(rows)?;
    let result = py
        .detach(|| core_deep_service_residual_predict(&model, &rows))
        .map_err(to_py_neural_error)?;
    let output = PyList::empty(py);
    for row in result {
        append_record!(
            py,
            output,
            row,
            prediction,
            residual_mean,
            lower_quantile,
            upper_quantile
        );
    }
    Ok(output.into_any().unbind())
}
#[pyfunction]
fn deep_service_residual_predict_arrays<'py>(
    py: Python<'py>,
    artifact_json: &str,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let model: DeepServiceResidualArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepServiceResidualRow> = records(rows)?;
    let values = py
        .detach(|| {
            core_deep_service_residual_predict(&model, &rows)
                .map(|rows| rows.into_iter().map(|r| r.prediction).collect::<Vec<_>>())
        })
        .map_err(to_py_neural_error)?;
    Ok(values.into_pyarray(py))
}

#[pyfunction(signature=(rows,options=None,backend=None))]
fn deep_directional_pair_fit_records(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
    backend: Option<&str>,
) -> PyResult<String> {
    let rows: Vec<DeepDirectionalPairRow> = records(rows)?;
    let options = options.map(records).transpose()?.unwrap_or_default();
    let result = py
        .detach(|| {
            cartoboost_neural::directional_pair_fit_with_options_and_backend(
                &rows, &options, backend,
            )
        })
        .map_err(to_py_neural_error)?;
    serde_json::to_string(&result).map_err(to_py_json_error)
}
#[pyfunction]
fn deep_directional_pair_predict_arrays<'py>(
    py: Python<'py>,
    artifact_json: &str,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let model: DeepDirectionalPairArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepDirectionalPairRow> = records(rows)?;
    let values = py
        .detach(|| core_deep_directional_pair_predict(&model, &rows))
        .map_err(to_py_neural_error)?;
    Ok(values.into_pyarray(py))
}

#[pyfunction(signature=(candidates,temperature=1.0,monotone_candidate_value=None,backend=None))]
fn deep_choice_set_transformer_report_records(
    py: Python<'_>,
    candidates: &Bound<'_, PyAny>,
    temperature: f64,
    monotone_candidate_value: Option<&str>,
    backend: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let rows: Vec<BTreeMap<String, Value>> = records(candidates)?;
    let result = py
        .detach(|| {
            cartoboost_neural::choice_set_transformer_report_with_backend(
                &rows,
                temperature,
                monotone_candidate_value,
                backend,
            )
        })
        .map_err(to_py_neural_error)?;
    let predictions = PyList::empty(py);
    for row in result.predictions {
        append_record!(
            py,
            predictions,
            row,
            decision_id,
            candidate_id,
            candidate_value,
            utility,
            choice_probability,
            nested_probability
        );
    }
    let best = PyList::empty(py);
    for row in result.counterfactual_best {
        append_record!(
            py,
            best,
            row,
            decision_id,
            candidate_id,
            candidate_value,
            utility,
            choice_probability
        );
    }
    let output = PyDict::new(py);
    output.set_item("predictions", predictions)?;
    output.set_item("counterfactual_best", best)?;
    output.set_item("calibration", result.calibration)?;
    output.set_item("benchmark", result.benchmark)?;
    output.set_item("metadata", result.metadata)?;
    Ok(output.into_any().unbind())
}
#[pyfunction(signature=(candidates,objective,constraints,fallback,risk_aversion=0.0))]
fn deep_constrained_decision_select_records(
    py: Python<'_>,
    candidates: &Bound<'_, PyAny>,
    objective: &str,
    constraints: &Bound<'_, PyAny>,
    fallback: &str,
    risk_aversion: f64,
) -> PyResult<Py<PyAny>> {
    let rows: Vec<BTreeMap<String, Value>> = records(candidates)?;
    let constraints: BTreeMap<String, f64> = records(constraints)?;
    let result = py
        .detach(|| {
            core_deep_constrained_decision_select(
                &rows,
                objective,
                &constraints,
                fallback,
                risk_aversion,
            )
        })
        .map_err(to_py_neural_error)?;
    let output = PyList::empty(py);
    for row in result {
        append_record!(
            py,
            output,
            row,
            decision_id,
            candidate_id,
            candidate_value,
            score,
            reason_code
        );
    }
    Ok(output.into_any().unbind())
}
#[pyfunction(signature=(y,lookback,horizon,backend=None))]
fn deep_temporal_entity_fit_arrays(
    py: Python<'_>,
    y: PyReadonlyArray2<'_, f64>,
    lookback: usize,
    horizon: usize,
    backend: Option<&str>,
) -> PyResult<String> {
    let y = rows_from_numpy_2d(y, "y")?;
    let model = py
        .detach(|| core_deep_temporal_entity_fit(&y, lookback, horizon, backend))
        .map_err(to_py_neural_error)?;
    serde_json::to_string(&model).map_err(to_py_json_error)
}
#[pyfunction]
fn deep_temporal_entity_predict_arrays<'py>(
    py: Python<'py>,
    artifact_json: &str,
    horizon: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let model: DeepTemporalEntityArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    matrix_output(
        py,
        py.detach(|| core_deep_temporal_entity_predict(&model, horizon))
            .map_err(to_py_neural_error)?,
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prob_conditional_flow_fit_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(prob_conditional_flow_predict_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(
        prob_diffusion_scenario_generate_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deep_graph_neural_operator_predict_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_fit_records, m)?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_predict_records, m)?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_predict_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_fit_records, m)?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_predict_records, m)?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_predict_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_directional_pair_fit_records, m)?)?;
    m.add_function(wrap_pyfunction!(deep_directional_pair_predict_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_choice_set_transformer_report_records,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deep_constrained_decision_select_records,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_fit_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_predict_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_inverted_temporal_fit_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_inverted_temporal_predict_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_quantiles_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_backtest_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_temporal_residual_quantiles_arrays,
        m
    )?)?;
    Ok(())
}

#[pyfunction]
fn deep_inverted_temporal_fit_arrays<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'_, f64>,
    lookback: usize,
    backend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (rows, cols) = (y.shape()[0], y.shape()[1]);
    let y = numpy_row_major(&y)?;
    let model = py
        .detach(|| cartoboost_neural::inverted_temporal::fit(&y, rows, cols, lookback, backend))
        .map_err(to_py_neural_error)?;
    let out = PyDict::new(py);
    out.set_item("last_values", model.last_values.into_pyarray(py))?;
    out.set_item("local_trend", model.local_trend.into_pyarray(py))?;
    out.set_item(
        "cross_entity_delta",
        model.cross_entity_delta.into_pyarray(py),
    )?;
    out.set_item(
        "delta_coef",
        matrix_output(
            py,
            model.delta_coef.into_iter().map(|r| r.to_vec()).collect(),
        )?,
    )?;
    out.set_item(
        "attention_weights",
        matrix_output(py, model.attention_weights)?,
    )?;
    out.set_item("backend", object(py, model.backend)?)?;
    out.set_item("accelerated", model.accelerated)?;
    out.set_item("residual_scales", model.residual_scales.into_pyarray(py))?;
    Ok(out)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn deep_inverted_temporal_predict_arrays<'py>(
    py: Python<'py>,
    last: PyReadonlyArray1<'_, f64>,
    local: PyReadonlyArray1<'_, f64>,
    coefficients: PyReadonlyArray2<'_, f64>,
    attention: PyReadonlyArray2<'_, f64>,
    horizon: usize,
    cross_entity: bool,
    backend: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = last.len();
    if coefficients.shape() != [n, 3] || attention.shape() != [n, n] {
        return Err(PyValueError::new_err(
            "invalid inverted temporal parameter shapes",
        ));
    }
    let last = numpy_row_major(&last)?;
    let local = numpy_row_major(&local)?;
    let coefficients = numpy_row_major(&coefficients)?;
    let attention = numpy_row_major(&attention)?;
    matrix_output(
        py,
        py.detach(|| {
            cartoboost_neural::inverted_temporal::predict(
                &last,
                &local,
                &coefficients,
                &attention,
                horizon,
                cross_entity,
                backend,
            )
        })
        .map_err(to_py_neural_error)?,
    )
}
#[pyfunction]
fn deep_temporal_entity_quantiles_arrays<'py>(
    py: Python<'py>,
    artifact_json: &str,
    horizon: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let model: DeepTemporalEntityArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let values = py
        .detach(|| cartoboost_neural::inverted_temporal::temporal_quantiles(&model, horizon))
        .map_err(to_py_neural_error)?;
    let out = PyDict::new(py);
    for (q, rows) in [0.1, 0.5, 0.9].into_iter().zip(values) {
        out.set_item(q, matrix_output(py, rows)?)?;
    }
    Ok(out)
}
#[pyfunction]
fn deep_temporal_entity_backtest_arrays(
    py: Python<'_>,
    y: PyReadonlyArray2<'_, f64>,
    folds: Vec<(Vec<usize>, Vec<usize>)>,
    lookback: usize,
    horizon: usize,
    backend: &str,
    inverted: bool,
) -> PyResult<Py<PyAny>> {
    let y = rows_from_numpy_2d(y, "y")?;
    object(
        py,
        py.detach(|| {
            cartoboost_neural::inverted_temporal::temporal_backtest(
                &y, &folds, lookback, horizon, backend, inverted,
            )
        })
        .map_err(to_py_neural_error)?,
    )
}
#[pyfunction]
fn deep_temporal_residual_quantiles_arrays<'py>(
    py: Python<'py>,
    point: PyReadonlyArray2<'_, f64>,
    scales: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let point = rows_from_numpy_2d(point, "point")?;
    let scales = numpy_row_major(&scales)?;
    let values = py
        .detach(|| cartoboost_neural::inverted_temporal::residual_quantiles(&point, &scales))
        .map_err(to_py_neural_error)?;
    let out = PyDict::new(py);
    for (q, rows) in [0.1, 0.5, 0.9].into_iter().zip(values) {
        out.set_item(q, matrix_output(py, rows)?)?;
    }
    Ok(out)
}
