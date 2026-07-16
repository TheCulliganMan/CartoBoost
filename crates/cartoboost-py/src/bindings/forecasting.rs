#[pyclass(name = "ForecastFrame")]
#[derive(Clone, Debug)]
struct NativeForecastFrame {
    frame: CoreForecastFrame,
}

#[pymethods]
impl NativeForecastFrame {
    #[new]
    #[pyo3(signature = (rows, frequency, timestamp_col=None, target_col=None, series_id_col=None, static_covariates=None, known_future_covariates=None, historical_covariates=None, row_covariates=None, sample_weights=None, sample_weight_col=None, allow_irregular=false, allow_missing_targets=false, allow_missing_covariates=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        rows: Vec<(String, String, f64)>,
        frequency: &str,
        timestamp_col: Option<String>,
        target_col: Option<String>,
        series_id_col: Option<String>,
        static_covariates: Option<Vec<String>>,
        known_future_covariates: Option<Vec<String>>,
        historical_covariates: Option<Vec<String>>,
        row_covariates: Option<Vec<BTreeMap<String, f64>>>,
        sample_weights: Option<Vec<f64>>,
        sample_weight_col: Option<String>,
        allow_irregular: bool,
        allow_missing_targets: bool,
        allow_missing_covariates: bool,
    ) -> PyResult<Self> {
        let frequency = ForecastFrequency::parse(frequency).map_err(to_py_value_error)?;
        let frequency_name = frequency.as_str().to_string();
        let metadata = ForecastFrameMetadata {
            timestamp_col,
            target_col,
            series_id_col,
            static_covariates: static_covariates.unwrap_or_default(),
            known_future_covariates: known_future_covariates.unwrap_or_default(),
            historical_covariates: historical_covariates.unwrap_or_default(),
            allow_irregular,
            allow_missing_targets,
            allow_missing_covariates,
        };
        let frame = py
            .detach(|| {
                let frequency = ForecastFrequency::parse(&frequency_name)?;
                match row_covariates {
                    Some(covariates) => {
                        if covariates.len() != rows.len() {
                            return Err(cartoboost_core::CartoBoostError::InvalidInput(
                                "row_covariates length must match rows length".to_string(),
                            ));
                        }
                        let rows = rows
                            .into_iter()
                            .zip(covariates)
                            .map(|((series_id, timestamp, target), covariates)| {
                                (series_id, timestamp, target, covariates)
                            })
                            .collect();
                        match sample_weights {
                            Some(weights) => {
                                CoreForecastFrame::from_string_rows_with_covariates_and_weights(
                                    rows,
                                    weights,
                                    sample_weight_col,
                                    frequency,
                                    metadata,
                                )
                            }
                            None => CoreForecastFrame::from_string_rows_with_covariates(
                                rows, frequency, metadata,
                            ),
                        }
                    }
                    None => {
                        if sample_weights.is_some() {
                            return Err(cartoboost_core::CartoBoostError::InvalidInput(
                                "sample_weights require row_covariates".to_string(),
                            ));
                        }
                        CoreForecastFrame::from_string_rows(rows, frequency, metadata)
                    }
                }
            })
            .map_err(to_py_value_error)?;
        Ok(Self { frame })
    }

    fn row_count(&self) -> usize {
        self.frame.rows().len()
    }

    fn frequency(&self) -> String {
        self.frame.frequency().as_str().to_string()
    }

    fn series_ids(&self) -> Vec<String> {
        self.frame.series_ids()
    }

    fn metadata_json(&self) -> PyResult<String> {
        self.frame.metadata_json_string().map_err(to_py_value_error)
    }

    fn rows(&self) -> Vec<(String, String, f64)> {
        self.frame
            .rows()
            .iter()
            .map(|row| {
                (
                    row.series_id.clone(),
                    row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    row.target,
                )
            })
            .collect()
    }

    fn row_covariates(&self) -> Vec<BTreeMap<String, f64>> {
        self.frame
            .rows()
            .iter()
            .map(|row| row.covariates.clone())
            .collect()
    }
}

#[pyclass(name = "ForecastResult")]
#[derive(Clone, Debug)]
struct NativeForecastResult {
    result: CoreForecastResult,
}

#[pymethods]
impl NativeForecastResult {
    #[new]
    fn new(
        py: Python<'_>,
        predictions: Vec<(String, String, usize, String, f64)>,
    ) -> PyResult<Self> {
        let result = py
            .detach(|| {
                let predictions = predictions
                    .into_iter()
                    .map(|(series_id, timestamp, horizon, model, mean)| {
                        Ok(ForecastPrediction {
                            series_id,
                            timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(
                                &timestamp,
                            )?,
                            horizon,
                            model,
                            mean,
                        })
                    })
                    .collect::<cartoboost_core::Result<Vec<_>>>()?;
                CoreForecastResult::new(predictions)
            })
            .map_err(to_py_value_error)?;
        Ok(Self { result })
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_string();
        let result = py
            .detach(|| CoreForecastResult::from_json_string(&value))
            .map_err(to_py_value_error)?;
        Ok(Self { result })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.result.to_json_string())
            .map_err(to_py_value_error)
    }

    fn columns(&self) -> Vec<String> {
        self.result.result_columns()
    }

    fn predictions(&self) -> Vec<(String, String, usize, String, f64)> {
        self.result
            .predictions()
            .iter()
            .map(|prediction| {
                (
                    prediction.series_id.clone(),
                    prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    prediction.horizon,
                    prediction.model.clone(),
                    prediction.mean,
                )
            })
            .collect()
    }
}

#[pyclass(name = "ForecastFold")]
#[derive(Clone, Debug)]
struct NativeForecastFold {
    fold: CoreForecastFold,
}

#[pymethods]
impl NativeForecastFold {
    #[getter]
    fn fold_id(&self) -> String {
        self.fold.fold_id.clone()
    }

    #[getter]
    fn train_indices(&self) -> Vec<usize> {
        self.fold.train_indices.clone()
    }

    #[getter]
    fn validation_indices(&self) -> Vec<usize> {
        self.fold.validation_indices.clone()
    }

    #[getter]
    fn train_start(&self) -> String {
        format_forecast_timestamp(self.fold.train_start)
    }

    #[getter]
    fn train_end(&self) -> String {
        format_forecast_timestamp(self.fold.train_end)
    }

    #[getter]
    fn validation_start(&self) -> String {
        format_forecast_timestamp(self.fold.validation_start)
    }

    #[getter]
    fn validation_end(&self) -> String {
        format_forecast_timestamp(self.fold.validation_end)
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.fold.horizon
    }

    #[getter]
    fn step(&self) -> usize {
        self.fold.step
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fold.metadata)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fold).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "RollingOriginSplitter")]
#[derive(Clone, Debug)]
struct NativeRollingOriginSplitter {
    splitter: CoreRollingOriginSplitter,
}

#[pymethods]
impl NativeRollingOriginSplitter {
    #[new]
    #[pyo3(signature = (horizon, step=1, min_train_size=1, max_train_size=None, n_splits=None, window="expanding"))]
    fn new(
        horizon: usize,
        step: usize,
        min_train_size: usize,
        max_train_size: Option<usize>,
        n_splits: Option<usize>,
        window: &str,
    ) -> PyResult<Self> {
        let window = parse_forecast_window(window)?;
        Ok(Self {
            splitter: CoreRollingOriginSplitter::new(
                horizon,
                step,
                min_train_size,
                max_train_size,
                n_splits,
                window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    #[staticmethod]
    fn expanding(horizon: usize, min_train_size: usize) -> PyResult<Self> {
        Ok(Self {
            splitter: CoreRollingOriginSplitter::expanding(horizon, min_train_size)
                .map_err(to_py_value_error)?,
        })
    }

    #[staticmethod]
    fn sliding(horizon: usize, min_train_size: usize, max_train_size: usize) -> PyResult<Self> {
        Ok(Self {
            splitter: CoreRollingOriginSplitter::sliding(horizon, min_train_size, max_train_size)
                .map_err(to_py_value_error)?,
        })
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.splitter.horizon
    }

    #[getter]
    fn step(&self) -> usize {
        self.splitter.step
    }

    #[getter]
    fn min_train_size(&self) -> usize {
        self.splitter.min_train_size
    }

    #[getter]
    fn max_train_size(&self) -> Option<usize> {
        self.splitter.max_train_size
    }

    #[getter]
    fn n_splits(&self) -> Option<usize> {
        self.splitter.n_splits
    }

    #[getter]
    fn window(&self) -> &'static str {
        forecast_window_name(&self.splitter.window)
    }

    fn split(
        &self,
        py: Python<'_>,
        frame: &NativeForecastFrame,
    ) -> PyResult<Vec<NativeForecastFold>> {
        Ok(py
            .detach(|| self.splitter.split(&frame.frame))
            .map_err(to_py_value_error)?
            .into_iter()
            .map(|fold| NativeForecastFold { fold })
            .collect())
    }

    fn n_splits_for_frame(&self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<usize> {
        Ok(self.split(py, frame)?.len())
    }
}

#[pyclass(name = "ForecastMetricSet")]
#[derive(Clone, Debug)]
struct NativeForecastMetricSet {
    metrics: CoreForecastMetricSet,
}

#[pymethods]
impl NativeForecastMetricSet {
    #[new]
    #[pyo3(signature = (mae=0.0, rmse=0.0, normalized_rmse=0.0, wape=0.0, smape=0.0, bias=0.0, mase=None))]
    fn new(
        mae: f64,
        rmse: f64,
        normalized_rmse: f64,
        wape: f64,
        smape: f64,
        bias: f64,
        mase: Option<f64>,
    ) -> Self {
        Self {
            metrics: CoreForecastMetricSet {
                mae,
                rmse,
                normalized_rmse,
                wape,
                smape,
                bias,
                mase,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (forecast, actuals, training_actuals=None, mase_seasonality=None))]
    fn evaluate(
        py: Python<'_>,
        forecast: &NativeForecastResult,
        actuals: Vec<(String, String, usize, f64)>,
        training_actuals: Option<Vec<(String, String, usize, f64)>>,
        mase_seasonality: Option<usize>,
    ) -> PyResult<Self> {
        let actuals = parse_forecast_actuals(actuals)?;
        let training_actuals = parse_forecast_actuals(training_actuals.unwrap_or_default())?;
        let metrics = py
            .detach(|| {
                cartoboost_core::forecasting::evaluate_forecast_with_training(
                    &forecast.result,
                    &actuals,
                    &training_actuals,
                    mase_seasonality,
                )
            })
            .map_err(to_py_value_error)?;
        Ok(Self { metrics })
    }

    #[getter]
    fn mae(&self) -> f64 {
        self.metrics.mae
    }

    #[getter]
    fn rmse(&self) -> f64 {
        self.metrics.rmse
    }

    #[getter]
    fn normalized_rmse(&self) -> f64 {
        self.metrics.normalized_rmse
    }

    #[getter]
    fn wape(&self) -> f64 {
        self.metrics.wape
    }

    #[getter]
    fn smape(&self) -> f64 {
        self.metrics.smape
    }

    #[getter]
    fn bias(&self) -> f64 {
        self.metrics.bias
    }

    #[getter]
    fn mase(&self) -> Option<f64> {
        self.metrics.mase
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.metrics).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyfunction]
#[pyo3(signature = (forecast, actuals, training_actuals=None, mase_seasonality=None))]
fn forecast_evaluate_metrics(
    py: Python<'_>,
    forecast: &NativeForecastResult,
    actuals: Vec<(String, String, usize, f64)>,
    training_actuals: Option<Vec<(String, String, usize, f64)>>,
    mase_seasonality: Option<usize>,
) -> PyResult<NativeForecastMetricSet> {
    NativeForecastMetricSet::evaluate(py, forecast, actuals, training_actuals, mase_seasonality)
}

#[pyclass(name = "BacktestFoldResult")]
#[derive(Clone, Debug)]
struct NativeBacktestFoldResult {
    result: CoreBacktestFoldResult,
}

#[pymethods]
impl NativeBacktestFoldResult {
    #[getter]
    fn fold(&self) -> NativeForecastFold {
        NativeForecastFold {
            fold: self.result.fold.clone(),
        }
    }

    #[getter]
    fn metrics(&self) -> NativeForecastMetricSet {
        NativeForecastMetricSet {
            metrics: self.result.metrics.clone(),
        }
    }

    #[getter]
    fn predictions(&self) -> Vec<(String, String, usize, String, f64)> {
        self.result
            .predictions
            .iter()
            .map(forecast_prediction_tuple)
            .collect()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "BacktestResult")]
#[derive(Clone, Debug)]
struct NativeBacktestResult {
    result: CoreBacktestResult,
}

#[pymethods]
impl NativeBacktestResult {
    #[getter]
    fn folds(&self) -> Vec<NativeBacktestFoldResult> {
        self.result
            .folds
            .iter()
            .cloned()
            .map(|result| NativeBacktestFoldResult { result })
            .collect()
    }

    #[getter]
    fn metrics(&self) -> Option<NativeForecastMetricSet> {
        self.result
            .metrics
            .clone()
            .map(|metrics| NativeForecastMetricSet { metrics })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "RollingOriginBacktester")]
#[derive(Clone, Debug)]
struct NativeRollingOriginBacktester {
    backtester: CoreRollingOriginBacktester,
}

#[pymethods]
impl NativeRollingOriginBacktester {
    #[new]
    #[pyo3(signature = (splitter, mase_seasonality=None))]
    fn new(
        splitter: &NativeRollingOriginSplitter,
        mase_seasonality: Option<usize>,
    ) -> PyResult<Self> {
        let mut backtester = CoreRollingOriginBacktester::new(splitter.splitter.clone());
        if let Some(seasonality) = mase_seasonality {
            backtester = backtester
                .with_mase_seasonality(seasonality)
                .map_err(to_py_value_error)?;
        }
        Ok(Self { backtester })
    }

    #[getter]
    fn splitter(&self) -> NativeRollingOriginSplitter {
        NativeRollingOriginSplitter {
            splitter: self.backtester.splitter.clone(),
        }
    }

    #[getter]
    fn mase_seasonality(&self) -> Option<usize> {
        self.backtester.mase_seasonality
    }

    fn run_naive(
        &self,
        py: Python<'_>,
        model: &NativeNaiveForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_seasonal_naive(
        &self,
        py: Python<'_>,
        model: &NativeSeasonalNaiveForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_theta(
        &self,
        py: Python<'_>,
        model: &NativeThetaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_optimized_theta(
        &self,
        py: Python<'_>,
        model: &NativeOptimizedThetaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_ets(
        &self,
        py: Python<'_>,
        model: &NativeETSForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_arima(
        &self,
        py: Python<'_>,
        model: &NativeArimaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_auto_arima(
        &self,
        py: Python<'_>,
        model: &NativeAutoARIMAForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_auto_forecast(
        &self,
        py: Python<'_>,
        model: &NativeAutoForecastModel,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_cartoboost_lag(
        &self,
        py: Python<'_>,
        model: &NativeCartoBoostLagForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.detach(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }
}

#[pyfunction]
fn forecast_parse_frequency(value: &str) -> PyResult<String> {
    Ok(ForecastFrequency::parse(value)
        .map_err(to_py_value_error)?
        .as_str()
        .to_string())
}


// Forecast binding families share the parent binding namespace.
include!("forecasting/utilities.rs");
include!("forecasting/classical.rs");
include!("forecasting/piecewise.rs");
include!("forecasting/statistical.rs");
include!("forecasting/spatial.rs");
include!("forecasting/neural_graph.rs");
include!("forecasting/auto_ensemble.rs");
include!("forecasting/helpers.rs");
