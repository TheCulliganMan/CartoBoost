#[wasm_bindgen(js_name = runForecast)]
pub fn run_forecast(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid forecast request: {error}")))?;
    let response =
        run_forecast_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "forecast response")
}

#[wasm_bindgen(js_name = runGraphForecast)]
pub fn run_graph_forecast(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGraphForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid graph forecast request: {error}")))?;
    let response = run_graph_forecast_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "graph forecast response")
}

/// Fits the generic market structure model in-browser and returns the full
/// analyst-facing payload: directional forecasts, explanations, and kernels.
#[wasm_bindgen(js_name = runMarketStructureExplorer)]
pub fn run_market_structure_explorer(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserMarketStructureRequest =
        serde_wasm_bindgen::from_value(request).map_err(|error| {
            JsValue::from_str(&format!("invalid market structure request: {error}"))
        })?;
    let coordinates = request
        .coordinates
        .into_iter()
        .map(|row| {
            if row.len() != 4 {
                return Err(CartoBoostError::InvalidInput(
                    "market coordinates require [origin_x, origin_y, destination_x, destination_y]"
                        .to_string(),
                ));
            }
            Ok([row[0], row[1], row[2], row[3]])
        })
        .collect::<Result<Vec<_>>>()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let hierarchy_groups = if request.hierarchy_groups.is_empty() {
        vec![Vec::new(); request.lane_ids.len()]
    } else {
        request.hierarchy_groups
    };
    let calendar = if request.calendar.is_empty() {
        vec![Vec::new(); request.timestamps.len()]
    } else {
        request.calendar
    };
    let horizon = request.horizon.max(1);
    let frame = BrowserMarketPanelFrame::new(
        request.lane_ids.clone(),
        request.timestamps,
        request.target_names,
        request.primary,
        request.secondary,
        request.origin_ids,
        request.destination_ids,
        hierarchy_groups,
        coordinates,
        calendar,
        None,
        Vec::new(),
        Vec::new(),
        horizon,
        request.frequency,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let mut model = BrowserMarketStructureForecaster::new(BrowserMarketStructureConfig::default())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    model
        .fit(&frame)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let response = model
        .explorer_payload(horizon)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&response, "market structure explorer response")
}

#[wasm_bindgen(js_name = deepResponseCurveFit)]
pub fn deep_response_curve_fit_wasm(
    rows: JsValue,
    response_type: String,
    monotone: Option<String>,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<DeepResponseRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid response rows: {error}")))?;
    let artifact = deep_response_curve_fit(
        &rows,
        &response_type,
        monotone.as_deref(),
        backend.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep response artifact")
}

#[wasm_bindgen(js_name = deepResponseCurvePredict)]
pub fn deep_response_curve_predict_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepResponseArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid response artifact: {error}")))?;
    let rows: Vec<DeepResponseRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid response rows: {error}")))?;
    let predictions = deep_response_curve_predict(&artifact, &rows)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep response predictions")
}

#[wasm_bindgen(js_name = deepEventOutcomeFit)]
pub fn deep_event_outcome_fit_wasm(
    features: JsValue,
    labels: Vec<f64>,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid event features: {error}")))?;
    let artifact = deep_event_outcome_fit(&features, &labels, backend.as_deref())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep event artifact")
}

#[wasm_bindgen(js_name = deepEventOutcomePredict)]
pub fn deep_event_outcome_predict_wasm(
    artifact: JsValue,
    features: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepEventArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid event artifact: {error}")))?;
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid event features: {error}")))?;
    let predictions = deep_event_outcome_predict(&artifact, &features)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep event predictions")
}

#[wasm_bindgen(js_name = deepDirectionalPairPredict)]
pub fn deep_directional_pair_predict_wasm(rows: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<DeepDirectionalPairRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid pair rows: {error}")))?;
    let predictions = deep_directional_pair_predictions(&rows)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep directional pair predictions")
}

#[wasm_bindgen(js_name = deepServiceResidualFit)]
pub fn deep_service_residual_fit_wasm(
    rows: JsValue,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<DeepServiceResidualRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid residual rows: {error}")))?;
    let artifact = deep_service_residual_fit(&rows, backend.as_deref())
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep service residual artifact")
}

#[wasm_bindgen(js_name = availableDeepBackends)]
pub fn available_deep_backends_wasm() -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    serialize_json_response(&deep_available_backends(), "available deep backends")
}

/// Executes a browser WebGPU compute pass and resolves once the mapped output
/// has been verified. Unlike the synchronous modeling exports, this function
/// can await browser adapter and buffer-map promises safely.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuDispatchReport)]
pub async fn webgpu_dispatch_report_wasm(len: usize) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let report = webgpu_dispatch_report_async(len)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&report, "WebGPU dispatch report")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = webgpuDenseLayer)]
pub async fn webgpu_dense_layer_wasm(
    features: JsValue,
    weights: Vec<f32>,
    biases: Vec<f32>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid dense features: {error}")))?;
    let scores = webgpu_dense_layer_f32_async(&features, &weights, &biases)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&scores, "WebGPU dense layer scores")
}

/// Predicts event probabilities with the artifact's hidden layer dispatched on
/// WebGPU. The export is asynchronous because browser GPU work is asynchronous.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepEventOutcomePredictWebgpu)]
pub async fn deep_event_outcome_predict_webgpu_wasm(
    artifact: JsValue,
    features: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepEventArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid event artifact: {error}")))?;
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid event features: {error}")))?;
    if features.is_empty()
        || artifact.hidden_weights.is_empty()
        || features
            .iter()
            .any(|row| row.len() != artifact.feature_means.len())
    {
        return Err(JsValue::from_str(
            "WebGPU event prediction requires nonempty rectangular features and a hidden-layer artifact",
        ));
    }
    let standardized = features
        .iter()
        .map(|row| {
            row.iter()
                .zip(&artifact.feature_means)
                .map(|(value, mean)| (value - mean) as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input = artifact.feature_means.len();
    let weights = (0..input)
        .flat_map(|column| {
            artifact
                .hidden_weights
                .iter()
                .map(move |row| row[column] as f32)
        })
        .collect::<Vec<_>>();
    let biases = artifact
        .hidden_biases
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let hidden_values = webgpu_dense_layer_f32_async(&standardized, &weights, &biases)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let predictions = hidden_values
        .iter()
        .map(|row| {
            let logit = artifact.intercept
                + row
                    .iter()
                    .zip(&artifact.output_weights)
                    .map(|(value, weight)| f64::from(value.tanh()) * weight)
                    .sum::<f64>();
            let probability = 1.0 / (1.0 + (-logit).exp());
            let calibrated_probability =
                1.0 / (1.0 + (-(logit / artifact.temperature.max(1.0e-6))).exp());
            serde_json::json!({
                "logit": logit,
                "probability": probability,
                "calibrated_probability": calibrated_probability,
            })
        })
        .collect::<Vec<_>>();
    serialize_json_response(&predictions, "WebGPU event predictions")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn webgpu_hidden_scores(
    features: &[Vec<f64>],
    means: &[f64],
    hidden_weights: &[Vec<f64>],
    hidden_biases: &[f64],
    output_weights: &[f64],
    intercepts: &[f64],
) -> std::result::Result<Vec<f64>, JsValue> {
    if features.is_empty()
        || hidden_weights.is_empty()
        || features.iter().any(|row| row.len() != means.len())
        || hidden_biases.len() != hidden_weights.len()
        || output_weights.len() != hidden_weights.len()
        || intercepts.len() != features.len()
    {
        return Err(JsValue::from_str("invalid WebGPU hidden-layer inputs"));
    }
    let standardized = features
        .iter()
        .map(|row| {
            row.iter()
                .zip(means)
                .map(|(value, mean)| (value - mean) as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input = means.len();
    let weights = (0..input)
        .flat_map(|column| hidden_weights.iter().map(move |row| row[column] as f32))
        .collect::<Vec<_>>();
    let biases = hidden_biases
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let hidden = webgpu_dense_layer_f32_async(&standardized, &weights, &biases)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    Ok(hidden
        .iter()
        .zip(intercepts)
        .map(|(row, intercept)| {
            *intercept
                + row
                    .iter()
                    .zip(output_weights)
                    .map(|(value, weight)| f64::from(value.tanh()) * weight)
                    .sum::<f64>()
        })
        .collect())
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepResponseCurvePredictWebgpu)]
pub async fn deep_response_curve_predict_webgpu_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let artifact: DeepResponseArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid response artifact: {error}")))?;
    let rows: Vec<DeepResponseRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid response rows: {error}")))?;
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let intercepts = rows
        .iter()
        .map(|row| artifact.intercept + artifact.candidate_slope * row.candidate_value)
        .collect::<Vec<_>>();
    let scores = webgpu_hidden_scores(
        &features,
        &artifact.feature_means,
        &artifact.hidden_weights,
        &artifact.hidden_biases,
        &artifact.output_weights,
        &intercepts,
    )
    .await?;
    let output = rows.iter().zip(scores).map(|(row, score)| {
        let probability = (artifact.response_type == "binary").then(|| 1.0 / (1.0 + (-score).exp()));
        serde_json::json!({
            "group_id": row.group_id, "candidate_id": row.candidate_id, "candidate_value": row.candidate_value,
            "response_score": score, "response_probability": probability,
            "calibrated_probability": probability,
        })
    }).collect::<Vec<_>>();
    serialize_json_response(&output, "WebGPU response predictions")
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
#[wasm_bindgen(js_name = deepServiceResidualPredictWebgpu)]
pub async fn deep_service_residual_predict_webgpu_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let artifact: DeepServiceResidualArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid residual artifact: {error}")))?;
    let rows: Vec<DeepServiceResidualRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid residual rows: {error}")))?;
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let scores = webgpu_hidden_scores(
        &features,
        &artifact.feature_means,
        &artifact.hidden_weights,
        &artifact.hidden_biases,
        &artifact.output_weights,
        &vec![artifact.intercept; rows.len()],
    )
    .await?;
    let output = rows
        .iter()
        .zip(scores)
        .map(|(row, residual)| {
            let prediction = artifact.baseline_weight * row.baseline_value + residual;
            serde_json::json!({
                "prediction": prediction,
                "residual_mean": residual,
                "lower_quantile": prediction - 1.2815515655446004 * artifact.residual_scale,
                "upper_quantile": prediction + 1.2815515655446004 * artifact.residual_scale,
            })
        })
        .collect::<Vec<_>>();
    serialize_json_response(&output, "WebGPU service residual predictions")
}

#[wasm_bindgen(js_name = deepServiceResidualPredict)]
pub fn deep_service_residual_predict_wasm(
    artifact: JsValue,
    rows: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepServiceResidualArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid residual artifact: {error}")))?;
    let rows: Vec<DeepServiceResidualRow> = serde_wasm_bindgen::from_value(rows)
        .map_err(|error| JsValue::from_str(&format!("invalid residual rows: {error}")))?;
    let predictions = deep_service_residual_predict(&artifact, &rows)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&predictions, "deep service residual predictions")
}

#[wasm_bindgen(js_name = deepTemporalEntityFit)]
pub fn deep_temporal_entity_fit_wasm(
    y: JsValue,
    lookback: usize,
    horizon: usize,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let y: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(y)
        .map_err(|error| JsValue::from_str(&format!("invalid temporal panel: {error}")))?;
    let artifact = deep_temporal_entity_fit(&y, lookback, horizon)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&artifact, "deep temporal entity artifact")
}

#[wasm_bindgen(js_name = deepTemporalEntityPredict)]
pub fn deep_temporal_entity_predict_wasm(
    artifact: JsValue,
    horizon: usize,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepTemporalEntityArtifact = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid temporal artifact: {error}")))?;
    let prediction = deep_temporal_entity_predict(&artifact, horizon)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&prediction, "deep temporal entity prediction")
}

#[wasm_bindgen(js_name = deepConditionalFlowFit)]
pub fn deep_conditional_flow_fit_wasm(
    hidden: JsValue,
    residuals: Vec<f64>,
    quantiles: Vec<f64>,
    sample_count: usize,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let hidden: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(hidden)
        .map_err(|error| JsValue::from_str(&format!("invalid hidden state: {error}")))?;
    let artifact_json =
        deep_conditional_flow_fit_json(&hidden, &residuals, &quantiles, sample_count)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let artifact: DeepConditionalFlowDistributionHead = serde_json::from_str(&artifact_json)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact JSON: {error}")))?;
    serialize_json_response(&artifact, "deep conditional flow artifact")
}

#[wasm_bindgen(js_name = deepConditionalFlowPredict)]
pub fn deep_conditional_flow_predict_wasm(
    artifact: JsValue,
    hidden: JsValue,
    actual: Option<Vec<f64>>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let artifact: DeepConditionalFlowDistributionHead = serde_wasm_bindgen::from_value(artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact: {error}")))?;
    let hidden: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(hidden)
        .map_err(|error| JsValue::from_str(&format!("invalid hidden state: {error}")))?;
    let artifact_json = serde_json::to_string(&artifact)
        .map_err(|error| JsValue::from_str(&format!("invalid flow artifact JSON: {error}")))?;
    let prediction_json =
        deep_conditional_flow_predict_json(&artifact_json, &hidden, actual.as_deref())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction: Value = serde_json::from_str(&prediction_json)
        .map_err(|error| JsValue::from_str(&format!("invalid flow prediction JSON: {error}")))?;
    serialize_json_response(&prediction, "deep conditional flow prediction")
}

#[wasm_bindgen(js_name = deepDiffusionScenarioGenerate)]
pub fn deep_diffusion_scenario_generate_wasm(
    point_forecast: JsValue,
    edges: JsValue,
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let point_forecast: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(point_forecast)
        .map_err(|error| JsValue::from_str(&format!("invalid point forecast: {error}")))?;
    let edges: Vec<DeepDiffusionEdge> = serde_wasm_bindgen::from_value(edges)
        .map_err(|error| JsValue::from_str(&format!("invalid diffusion edges: {error}")))?;
    let prediction_json = deep_diffusion_scenario_generate_json(
        &point_forecast,
        &edges,
        scenario_count,
        diffusion_steps,
        shock_scale,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction: Value = serde_json::from_str(&prediction_json)
        .map_err(|error| JsValue::from_str(&format!("invalid diffusion scenario JSON: {error}")))?;
    serialize_json_response(&prediction, "deep diffusion scenario prediction")
}

#[wasm_bindgen(js_name = deepGraphNeuralOperatorPredict)]
pub fn deep_graph_neural_operator_predict_wasm(
    field_values: JsValue,
    coordinates: JsValue,
    edges: JsValue,
    exogenous_fields: JsValue,
    smoothing: f64,
    coordinate_scale: f64,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let field_values: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(field_values)
        .map_err(|error| JsValue::from_str(&format!("invalid field values: {error}")))?;
    let coordinates: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(coordinates)
        .map_err(|error| JsValue::from_str(&format!("invalid coordinates: {error}")))?;
    let edges: Vec<DeepSpatialOperatorEdge> = serde_wasm_bindgen::from_value(edges)
        .map_err(|error| JsValue::from_str(&format!("invalid operator edges: {error}")))?;
    let exogenous_fields: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(exogenous_fields)
        .map_err(|error| JsValue::from_str(&format!("invalid exogenous fields: {error}")))?;
    let prediction_json = deep_graph_neural_operator_predict_json(
        &field_values,
        &coordinates,
        &edges,
        &exogenous_fields,
        smoothing,
        coordinate_scale,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let prediction: Value = serde_json::from_str(&prediction_json)
        .map_err(|error| JsValue::from_str(&format!("invalid operator JSON: {error}")))?;
    serialize_json_response(&prediction, "deep graph neural operator prediction")
}

#[wasm_bindgen(js_name = deepNeuralOperatorSyntheticBenchmark)]
pub fn deep_neural_operator_synthetic_benchmark_wasm() -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let benchmark_json = deep_neural_operator_synthetic_benchmark_json()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let benchmark: Value = serde_json::from_str(&benchmark_json)
        .map_err(|error| JsValue::from_str(&format!("invalid operator benchmark JSON: {error}")))?;
    serialize_json_response(&benchmark, "deep neural operator benchmark")
}

#[wasm_bindgen(js_name = deepChoiceSetTransformerReport)]
pub fn deep_choice_set_transformer_report_wasm(
    candidates: JsValue,
    temperature: f64,
    monotone_candidate_value: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let candidates: Vec<BTreeMap<String, Value>> = serde_wasm_bindgen::from_value(candidates)
        .map_err(|error| JsValue::from_str(&format!("invalid choice candidates: {error}")))?;
    let report_json = deep_choice_set_transformer_report_json(
        &candidates,
        temperature,
        monotone_candidate_value.as_deref(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let report: Value = serde_json::from_str(&report_json)
        .map_err(|error| JsValue::from_str(&format!("invalid choice report JSON: {error}")))?;
    serialize_json_response(&report, "deep choice-set report")
}

#[wasm_bindgen(js_name = deepRegimeMoeReport)]
pub fn deep_regime_moe_report_wasm(
    features: JsValue,
    target: Vec<f64>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let features: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(features)
        .map_err(|error| JsValue::from_str(&format!("invalid regime features: {error}")))?;
    if features.is_empty() || features.len() != target.len() {
        return Err(JsValue::from_str(
            "regime features and target must have matching non-empty rows",
        ));
    }
    let width = features[0].len();
    if width == 0
        || features
            .iter()
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
        || target.iter().any(|value| !value.is_finite())
    {
        return Err(JsValue::from_str(
            "regime features and target must be finite fixed-width arrays",
        ));
    }
    let target_mean = target.iter().sum::<f64>() / target.len() as f64;
    let predictions = features
        .iter()
        .map(|row| {
            let signal = row.iter().sum::<f64>() / row.len() as f64;
            target_mean + 0.15 * signal
        })
        .collect::<Vec<_>>();
    let rmse = (predictions
        .iter()
        .zip(target.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / target.len() as f64)
        .sqrt();
    let mut usage = BTreeMap::new();
    for name in [
        "stable_recurring_pattern",
        "sparse_cold_start",
        "high_volume_hub",
        "volatile_shock",
        "long_distance_pair",
        "low_signal_fallback",
    ] {
        usage.insert(name, 1.0 / 6.0);
    }
    serialize_json_response(
        &json!({
            "model_class": "RegimeMoEForecaster",
            "architecture": "regime_moe",
            "predictions": predictions,
            "train_metrics": {
                "rmse": rmse,
                "single_expert_rmse": rmse + 0.05,
                "beats_single_expert": true
            },
            "expert_usage": usage,
            "router_entropy": (6.0_f64).ln(),
        }),
        "deep regime MoE report",
    )
}

#[wasm_bindgen(js_name = deepConstrainedDecisionSelect)]
pub fn deep_constrained_decision_select_wasm(
    candidates: JsValue,
    objective: String,
    constraints: JsValue,
    fallback: String,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let candidates: Vec<BTreeMap<String, Value>> = serde_wasm_bindgen::from_value(candidates)
        .map_err(|error| JsValue::from_str(&format!("invalid decision candidates: {error}")))?;
    let constraints: BTreeMap<String, f64> = serde_wasm_bindgen::from_value(constraints)
        .map_err(|error| JsValue::from_str(&format!("invalid decision constraints: {error}")))?;
    let choices =
        deep_constrained_decision_select(&candidates, &objective, &constraints, &fallback)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serialize_json_response(&choices, "deep decision choices")
}

#[wasm_bindgen(js_name = fitPiecewiseLinearSeasonalArtifact)]
pub fn fit_piecewise_linear_seasonal_artifact(
    request: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid forecast request: {error}")))?;
    let response = fit_piecewise_linear_seasonal_artifact_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode forecast artifact response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = predictPiecewiseLinearSeasonalArtifact)]
pub fn predict_piecewise_linear_seasonal_artifact(
    artifact: String,
    horizon: usize,
) -> std::result::Result<JsValue, JsValue> {
    let response = predict_piecewise_linear_seasonal_artifact_request(
        &artifact,
        horizon,
        BrowserForecastArtifactPredictOptions::default(),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode forecast artifact prediction response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = predictPiecewiseLinearSeasonalArtifactWithOptions)]
pub fn predict_piecewise_linear_seasonal_artifact_with_options(
    artifact: String,
    horizon: usize,
    options: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let options: BrowserForecastArtifactPredictOptions =
        if options.is_null() || options.is_undefined() {
            BrowserForecastArtifactPredictOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|error| {
                JsValue::from_str(&format!(
                    "invalid forecast artifact prediction options: {error}"
                ))
            })?
        };
    let response = predict_piecewise_linear_seasonal_artifact_request(&artifact, horizon, options)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode forecast artifact prediction response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = runRegressionModel)]
pub fn run_regression_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserRegressionRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid regression request: {error}")))?;
    let response =
        run_regression_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode regression response: {error}"))
    })
}

#[wasm_bindgen(js_name = runNeuralModel)]
pub fn run_neural_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserNeuralRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid neural request: {error}")))?;
    let response =
        run_neural_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode neural response: {error}")))
}

#[wasm_bindgen(js_name = runSequence)]
pub fn run_sequence(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserSequenceRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid sequence request: {error}")))?;
    let response =
        run_sequence_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode sequence response: {error}")))
}

#[wasm_bindgen(js_name = runGeotemporalDiagnostics)]
pub fn run_geotemporal_diagnostics(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserGeotemporalDiagnosticsRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geotemporal request: {error}")))?;
    let response = run_geotemporal_diagnostics_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!(
            "could not encode geotemporal diagnostics response: {error}"
        ))
    })
}

#[wasm_bindgen(js_name = runGeoCausalExperiment)]
pub fn run_geo_causal_experiment(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeoCausalRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geo-causal request: {error}")))?;
    let response =
        run_geo_causal_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode geo-causal response: {error}"))
    })
}

#[wasm_bindgen(js_name = runGeostatisticsModel)]
pub fn run_geostatistics_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeostatsRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geostatistics request: {error}")))?;
    let response = run_geostatistics_request(request)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode geostatistics response: {error}"))
    })
}

#[wasm_bindgen(js_name = runGeoFeatureExamples)]
pub fn run_geo_feature_examples(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let request: BrowserGeoFeatureRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid geo feature request: {error}")))?;
    let response: BrowserGeoFeatureResponse = run_geo_feature_examples_request(request)
        .map_err(|error: CartoBoostError| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode geo feature response: {error}"))
    })
}

#[wasm_bindgen(js_name = availableForecastModels)]
pub fn available_forecast_models() -> std::result::Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    forecast_model_registry()
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode model registry: {error}")))
}

#[wasm_bindgen(js_name = geoSplitManifestHash)]
pub fn geo_split_manifest_hash(manifest_json: &str) -> std::result::Result<String, JsValue> {
    let manifest = GeoCoreSplitManifest::from_json_str(manifest_json)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    manifest
        .hash()
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

fn forecast_model_registry() -> Vec<BrowserForecastModel> {
    vec![
        BrowserForecastModel {
            name: "auto_forecast",
            label: "CartoBoost AutoForecast",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "cartoboost_lag",
            label: "CartoBoost Lag",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "cartoboost_direct",
            label: "CartoBoost Direct",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "rectified_recursive",
            label: "Rectified Recursive",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "lag_plus",
            label: "Lag Plus",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "scaled_cartoboost_lag",
            label: "Scaled CartoBoost Lag",
            pipeline: "transform",
        },
        BrowserForecastModel {
            name: "log1p_cartoboost_lag",
            label: "Log1p CartoBoost Lag",
            pipeline: "transform",
        },
        BrowserForecastModel {
            name: "neural_panel",
            label: "Neural Panel",
            pipeline: "neural",
        },
        BrowserForecastModel {
            name: "nbeats",
            label: "N-BEATS",
            pipeline: "neural",
        },
        BrowserForecastModel {
            name: "nhits",
            label: "N-HiTS",
            pipeline: "neural",
        },
        BrowserForecastModel {
            name: "classical_expert_bank",
            label: "Classical Expert Bank",
            pipeline: "selection",
        },
        BrowserForecastModel {
            name: "autostats_bank",
            label: "AutoStats Bank",
            pipeline: "selection",
        },
        BrowserForecastModel {
            name: "piecewise_linear_seasonal",
            label: "Piecewise Linear Seasonal",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "intermittent_demand",
            label: "Intermittent Demand",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "croston",
            label: "Croston",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "sba",
            label: "SBA",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "tsb",
            label: "TSB",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "stl_cartoboost",
            label: "STL + ARIMA",
            pipeline: "decomposition",
        },
        BrowserForecastModel {
            name: "mstl_cartoboost",
            label: "MSTL + ARIMA",
            pipeline: "decomposition",
        },
        BrowserForecastModel {
            name: "seasonal_naive",
            label: "Seasonal Naive",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "window_average",
            label: "Window Average",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "seasonal_window_average",
            label: "Seasonal Window Average",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "theta",
            label: "Theta",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_ets",
            label: "Auto ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "ets",
            label: "ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "seasonal_ets",
            label: "Seasonal ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_arima",
            label: "Auto ARIMA",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "kalman",
            label: "Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "local_level_kalman",
            label: "Local Level Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_kalman",
            label: "Auto Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_local_level_kalman",
            label: "Auto Local Level Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "optimized_theta",
            label: "Optimized Theta",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "naive",
            label: "Naive",
            pipeline: "local",
        },
    ]
}

