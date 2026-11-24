//! Browser entry points for the same entity-token model used by Python.
use cartoboost_neural::inverted_temporal::{fit, predict, InvertedFit};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = deepInvertedTemporalFit)]
pub fn deep_inverted_temporal_fit_wasm(
    y: JsValue,
    lookback: usize,
    backend: Option<String>,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let rows: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(y)
        .map_err(|error| JsValue::from_str(&format!("invalid inverted temporal panel: {error}")))?;
    let cols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != cols) {
        return Err(JsValue::from_str("panel rows must have equal widths"));
    }
    let values: Vec<f64> = rows.iter().flatten().copied().collect();
    let model = fit(
        &values,
        rows.len(),
        cols,
        lookback,
        backend.as_deref().unwrap_or("cpu"),
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    super::serialize_json_response(&model, "inverted temporal artifact")
}

#[wasm_bindgen(js_name = deepInvertedTemporalPredict)]
pub fn deep_inverted_temporal_predict_wasm(
    artifact: JsValue,
    horizon: usize,
    cross_entity: bool,
) -> std::result::Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let model: InvertedFit = serde_wasm_bindgen::from_value(artifact).map_err(|error| {
        JsValue::from_str(&format!("invalid inverted temporal artifact: {error}"))
    })?;
    let entities = model.last_values.len();
    if model.attention_weights.len() != entities
        || model
            .attention_weights
            .iter()
            .any(|row| row.len() != entities)
    {
        return Err(JsValue::from_str(
            "attention weights must match the entity count",
        ));
    }
    let coefficients: Vec<f64> = model.delta_coef.iter().flatten().copied().collect();
    let attention: Vec<f64> = model.attention_weights.iter().flatten().copied().collect();
    let result = predict(
        &model.last_values,
        &model.local_trend,
        &coefficients,
        &attention,
        horizon,
        cross_entity,
        &model.backend.requested,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    super::serialize_json_response(&result, "inverted temporal prediction")
}
