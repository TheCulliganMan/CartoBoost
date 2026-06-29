use crate::{backend_affine_scores, select_backend, BackendSelection, NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepResponseRow {
    pub features: Vec<f64>,
    pub candidate_value: f64,
    pub response: Option<f64>,
    pub group_id: Option<String>,
    pub candidate_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepResponseArtifact {
    pub model_class: String,
    pub model_version: String,
    pub response_type: String,
    pub monotone: Option<String>,
    pub feature_means: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub intercept: f64,
    pub candidate_slope: f64,
    pub calibration: BTreeMap<String, f64>,
    pub schema_hash: String,
    #[serde(default)]
    pub backend: BackendSelection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepResponsePrediction {
    pub group_id: Option<String>,
    pub candidate_id: Option<String>,
    pub candidate_value: f64,
    pub response_score: f64,
    pub response_probability: Option<f64>,
    pub calibrated_probability: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepEventArtifact {
    pub model_class: String,
    pub model_version: String,
    pub feature_means: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub intercept: f64,
    pub temperature: f64,
    pub calibration: BTreeMap<String, f64>,
    pub schema_hash: String,
    #[serde(default)]
    pub backend: BackendSelection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepEventPrediction {
    pub logit: f64,
    pub probability: f64,
    pub calibrated_probability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepDirectionalPairRow {
    pub source_id: String,
    pub target_id: String,
    pub timestamp: Option<i64>,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepServiceResidualRow {
    pub baseline_value: f64,
    pub actual_value: Option<f64>,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepServiceResidualArtifact {
    pub model_class: String,
    pub model_version: String,
    pub feature_means: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub intercept: f64,
    pub baseline_weight: f64,
    pub residual_scale: f64,
    pub schema_hash: String,
    #[serde(default)]
    pub backend: BackendSelection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepServiceResidualPrediction {
    pub prediction: f64,
    pub residual_mean: f64,
    pub lower_quantile: f64,
    pub upper_quantile: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepDecisionChoice {
    pub decision_id: String,
    pub candidate_id: String,
    pub candidate_value: f64,
    pub score: f64,
    pub reason_code: String,
}

pub fn response_curve_fit(
    rows: &[DeepResponseRow],
    response_type: &str,
    monotone: Option<&str>,
) -> Result<DeepResponseArtifact> {
    response_curve_fit_with_backend(rows, response_type, monotone, None)
}

pub fn response_curve_fit_with_backend(
    rows: &[DeepResponseRow],
    response_type: &str,
    monotone: Option<&str>,
    backend: Option<&str>,
) -> Result<DeepResponseArtifact> {
    let backend = select_backend(backend)?;
    validate_response_rows(rows, true)?;
    let dim = rows[0].features.len();
    let means = feature_means(rows.iter().map(|row| row.features.as_slice()), dim)?;
    let mut weights = vec![0.0; dim];
    let y_mean = rows.iter().filter_map(|row| row.response).sum::<f64>() / rows.len() as f64;
    for idx in 0..dim {
        let mut num = 0.0;
        let mut den = 0.0;
        for row in rows {
            let x = row.features[idx] - means[idx];
            let y = row.response.unwrap_or(0.0) - y_mean;
            num += x * y;
            den += x * x;
        }
        weights[idx] = if den > 0.0 { num / (den + 1e-9) } else { 0.0 };
    }
    let c_mean = rows.iter().map(|row| row.candidate_value).sum::<f64>() / rows.len() as f64;
    let mut c_num = 0.0;
    let mut c_den = 0.0;
    for row in rows {
        let x = row.candidate_value - c_mean;
        let y = row.response.unwrap_or(0.0) - y_mean;
        c_num += x * y;
        c_den += x * x;
    }
    let mut slope = if c_den > 0.0 {
        c_num / (c_den + 1e-9)
    } else {
        0.0
    };
    match monotone {
        Some("increasing") => slope = slope.abs().max(1e-9),
        Some("decreasing") => slope = -slope.abs().min(-1e-9),
        Some(other) => return invalid(format!("unknown monotone mode {other:?}")),
        None => {}
    }
    let intercept = y_mean - slope * c_mean;
    let mut calibration = BTreeMap::new();
    if response_type == "binary" {
        calibration.insert("positive_rate".to_string(), y_mean.clamp(0.0, 1.0));
    }
    Ok(DeepResponseArtifact {
        model_class: "ResponseCurveModel".to_string(),
        model_version: "1".to_string(),
        response_type: response_type.to_string(),
        monotone: monotone.map(str::to_string),
        feature_means: means,
        feature_weights: weights,
        intercept,
        candidate_slope: slope,
        calibration,
        schema_hash: schema_hash(dim, response_type),
        backend,
    })
}

pub fn response_curve_predict(
    artifact: &DeepResponseArtifact,
    rows: &[DeepResponseRow],
) -> Result<Vec<DeepResponsePrediction>> {
    validate_response_rows(rows, false)?;
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let intercepts = rows
        .iter()
        .map(|row| artifact.intercept + artifact.candidate_slope * row.candidate_value)
        .collect::<Vec<_>>();
    let scores = backend_affine_scores(
        &artifact.backend,
        &features,
        &artifact.feature_means,
        &artifact.feature_weights,
        &intercepts,
    )?;
    rows.iter()
        .zip(scores)
        .map(|(row, score)| {
            let probability = (artifact.response_type == "binary").then(|| sigmoid(score));
            Ok(DeepResponsePrediction {
                group_id: row.group_id.clone(),
                candidate_id: row.candidate_id.clone(),
                candidate_value: row.candidate_value,
                response_score: score,
                response_probability: probability,
                calibrated_probability: probability,
            })
        })
        .collect()
}

pub fn event_outcome_fit(features: &[Vec<f64>], labels: &[f64]) -> Result<DeepEventArtifact> {
    event_outcome_fit_with_backend(features, labels, None)
}

pub fn event_outcome_fit_with_backend(
    features: &[Vec<f64>],
    labels: &[f64],
    backend: Option<&str>,
) -> Result<DeepEventArtifact> {
    let backend = select_backend(backend)?;
    validate_matrix(features)?;
    if labels.len() != features.len() || labels.is_empty() {
        return invalid("labels must have the same nonzero row count as features");
    }
    if labels
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return invalid("event labels must be finite values in [0, 1]");
    }
    let dim = features[0].len();
    let means = feature_means(features.iter().map(Vec::as_slice), dim)?;
    let y_mean = labels.iter().sum::<f64>() / labels.len() as f64;
    let mut weights = vec![0.0; dim];
    for idx in 0..dim {
        let mut num = 0.0;
        let mut den = 0.0;
        for (row, &label) in features.iter().zip(labels) {
            let x = row[idx] - means[idx];
            num += x * (label - y_mean);
            den += x * x;
        }
        weights[idx] = if den > 0.0 { num / (den + 1e-9) } else { 0.0 };
    }
    let intercept = logit(y_mean.clamp(1e-6, 1.0 - 1e-6));
    let mut calibration = BTreeMap::new();
    calibration.insert("positive_rate".to_string(), y_mean);
    Ok(DeepEventArtifact {
        model_class: "EventOutcomeModel".to_string(),
        model_version: "1".to_string(),
        feature_means: means,
        feature_weights: weights,
        intercept,
        temperature: 1.0,
        calibration,
        schema_hash: schema_hash(dim, "event"),
        backend,
    })
}

pub fn event_outcome_predict(
    artifact: &DeepEventArtifact,
    features: &[Vec<f64>],
) -> Result<Vec<DeepEventPrediction>> {
    validate_matrix(features)?;
    let intercepts = vec![artifact.intercept; features.len()];
    let logits = backend_affine_scores(
        &artifact.backend,
        features,
        &artifact.feature_means,
        &artifact.feature_weights,
        &intercepts,
    )?;
    features
        .iter()
        .zip(logits)
        .map(|(_row, logit)| {
            let calibrated = sigmoid(logit / artifact.temperature.max(1e-6));
            Ok(DeepEventPrediction {
                logit,
                probability: sigmoid(logit),
                calibrated_probability: calibrated,
            })
        })
        .collect()
}

pub fn directional_pair_predictions(rows: &[DeepDirectionalPairRow]) -> Result<Vec<f64>> {
    if rows.is_empty() {
        return invalid("directional pair rows cannot be empty");
    }
    Ok(rows
        .iter()
        .map(|row| {
            let signed = stable_hash(&format!("{}->{}", row.source_id, row.target_id)) as f64;
            let reverse = stable_hash(&format!("{}->{}", row.target_id, row.source_id)) as f64;
            let feature_sum = row.features.iter().sum::<f64>();
            ((signed % 997.0) - (reverse % 991.0)) / 997.0 + feature_sum
        })
        .collect())
}

pub fn service_residual_fit(
    rows: &[DeepServiceResidualRow],
) -> Result<DeepServiceResidualArtifact> {
    service_residual_fit_with_backend(rows, None)
}

pub fn service_residual_fit_with_backend(
    rows: &[DeepServiceResidualRow],
    backend: Option<&str>,
) -> Result<DeepServiceResidualArtifact> {
    let backend = select_backend(backend)?;
    if rows.is_empty() {
        return invalid("service residual rows cannot be empty");
    }
    let dim = rows[0].features.len();
    if rows.iter().any(|row| row.features.len() != dim) {
        return invalid("all service residual feature rows must have the same width");
    }
    let means = feature_means(rows.iter().map(|row| row.features.as_slice()), dim)?;
    let mut weights = vec![0.0; dim];
    let residuals = rows
        .iter()
        .map(|row| row.actual_value.unwrap_or(row.baseline_value) - row.baseline_value)
        .collect::<Vec<_>>();
    let residual_mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    for idx in 0..dim {
        let mut num = 0.0;
        let mut den = 0.0;
        for (row, residual) in rows.iter().zip(&residuals) {
            let x = row.features[idx] - means[idx];
            num += x * (residual - residual_mean);
            den += x * x;
        }
        weights[idx] = if den > 0.0 { num / (den + 1e-9) } else { 0.0 };
    }
    let residual_scale = (residuals
        .iter()
        .map(|v| (v - residual_mean).powi(2))
        .sum::<f64>()
        / residuals.len() as f64)
        .sqrt();
    Ok(DeepServiceResidualArtifact {
        model_class: "ServiceTimeResidualModel".to_string(),
        model_version: "1".to_string(),
        feature_means: means,
        feature_weights: weights,
        intercept: residual_mean,
        baseline_weight: 1.0,
        residual_scale,
        schema_hash: schema_hash(dim, "service_residual"),
        backend,
    })
}

pub fn service_residual_predict(
    artifact: &DeepServiceResidualArtifact,
    rows: &[DeepServiceResidualRow],
) -> Result<Vec<DeepServiceResidualPrediction>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let intercepts = vec![artifact.intercept; rows.len()];
    let residuals = backend_affine_scores(
        &artifact.backend,
        &features,
        &artifact.feature_means,
        &artifact.feature_weights,
        &intercepts,
    )?;
    rows.iter()
        .zip(residuals)
        .map(|(row, residual)| {
            let prediction = artifact.baseline_weight * row.baseline_value + residual;
            Ok(DeepServiceResidualPrediction {
                prediction,
                residual_mean: residual,
                lower_quantile: prediction - 1.2815515655446004 * artifact.residual_scale,
                upper_quantile: prediction + 1.2815515655446004 * artifact.residual_scale,
            })
        })
        .collect()
}

pub fn constrained_decision_select(
    candidates: &[BTreeMap<String, serde_json::Value>],
    objective: &str,
    constraints: &BTreeMap<String, f64>,
    fallback: &str,
) -> Result<Vec<DeepDecisionChoice>> {
    let mut groups: BTreeMap<String, Vec<&BTreeMap<String, serde_json::Value>>> = BTreeMap::new();
    for row in candidates {
        let decision_id = json_str(row, "decision_id")?;
        groups.entry(decision_id).or_default().push(row);
    }
    let mut choices = Vec::new();
    for (decision_id, rows) in groups {
        let mut feasible = rows
            .iter()
            .copied()
            .filter(|row| constraints_hold(row, constraints))
            .collect::<Vec<_>>();
        let reason_code = if feasible.is_empty() {
            if fallback == "raise" {
                return invalid(format!("no feasible candidates for decision {decision_id}"));
            }
            feasible = rows.clone();
            format!("fallback:{fallback}")
        } else {
            "constraints_satisfied".to_string()
        };
        feasible.sort_by(|left, right| {
            let l_score = objective_score(left, objective);
            let r_score = objective_score(right, objective);
            r_score.total_cmp(&l_score).then_with(|| {
                json_str(left, "candidate_id")
                    .unwrap_or_default()
                    .cmp(&json_str(right, "candidate_id").unwrap_or_default())
            })
        });
        let best = feasible[0];
        choices.push(DeepDecisionChoice {
            decision_id,
            candidate_id: json_str(best, "candidate_id")?,
            candidate_value: json_f64(best, "candidate_value").unwrap_or(0.0),
            score: objective_score(best, objective),
            reason_code,
        });
    }
    Ok(choices)
}

fn constraints_hold(
    row: &BTreeMap<String, serde_json::Value>,
    constraints: &BTreeMap<String, f64>,
) -> bool {
    for (name, &limit) in constraints {
        let ok = match name.as_str() {
            "min_response_probability" => {
                json_f64(row, "response_probability").unwrap_or(f64::NEG_INFINITY) >= limit
            }
            "max_risk_score" | "max_tail_risk" => {
                json_f64(row, "risk_score").unwrap_or(f64::INFINITY) <= limit
            }
            "min_candidate_value" => {
                json_f64(row, "candidate_value").unwrap_or(f64::NEG_INFINITY) >= limit
            }
            "max_candidate_value" => {
                json_f64(row, "candidate_value").unwrap_or(f64::INFINITY) <= limit
            }
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}

fn objective_score(row: &BTreeMap<String, serde_json::Value>, objective: &str) -> f64 {
    match objective {
        "max_response_probability" => json_f64(row, "response_probability").unwrap_or(0.0),
        "min_expected_value" => -json_f64(row, "expected_value").unwrap_or(f64::INFINITY),
        "risk_adjusted_utility" => {
            json_f64(row, "expected_utility").unwrap_or(0.0)
                - json_f64(row, "risk_score").unwrap_or(0.0)
        }
        _ => json_f64(row, "expected_utility").unwrap_or(0.0),
    }
}

fn validate_response_rows(rows: &[DeepResponseRow], require_response: bool) -> Result<()> {
    if rows.is_empty() {
        return invalid("response rows cannot be empty");
    }
    let dim = rows[0].features.len();
    if dim == 0 {
        return invalid("response rows must contain at least one feature");
    }
    for row in rows {
        if row.features.len() != dim || row.features.iter().any(|value| !value.is_finite()) {
            return invalid("all response feature rows must be finite and have the same width");
        }
        if !row.candidate_value.is_finite() {
            return invalid("candidate values must be finite");
        }
        if require_response && row.response.is_none() {
            return invalid("fit rows must include response values");
        }
    }
    Ok(())
}

fn validate_matrix(features: &[Vec<f64>]) -> Result<()> {
    if features.is_empty() || features[0].is_empty() {
        return invalid("feature matrix must be non-empty");
    }
    let dim = features[0].len();
    if features
        .iter()
        .any(|row| row.len() != dim || row.iter().any(|value| !value.is_finite()))
    {
        return invalid("feature matrix rows must be finite and have the same width");
    }
    Ok(())
}

fn feature_means<'a>(rows: impl Iterator<Item = &'a [f64]>, dim: usize) -> Result<Vec<f64>> {
    let mut means = vec![0.0; dim];
    let mut count = 0usize;
    for row in rows {
        if row.len() != dim {
            return invalid("feature rows must have the same width");
        }
        for (idx, value) in row.iter().enumerate() {
            means[idx] += *value;
        }
        count += 1;
    }
    if count == 0 {
        return invalid("feature rows cannot be empty");
    }
    for value in &mut means {
        *value /= count as f64;
    }
    Ok(means)
}

fn json_f64(row: &BTreeMap<String, serde_json::Value>, key: &str) -> Option<f64> {
    row.get(key).and_then(serde_json::Value::as_f64)
}

fn json_str(row: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<String> {
    row.get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| NeuralError::InvalidArgument(format!("{key} must be a string")))
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn logit(value: f64) -> f64 {
    (value / (1.0 - value)).ln()
}

fn schema_hash(dim: usize, label: &str) -> String {
    format!("{label}:{dim}:{}", stable_hash(label))
}

fn stable_hash(value: &str) -> u64 {
    let mut seen = BTreeSet::new();
    let mut hash = 1469598103934665603u64;
    for byte in value.as_bytes() {
        seen.insert(*byte);
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash ^ seen.len() as u64
}

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(NeuralError::InvalidArgument(message.into()))
}

#[cfg(test)]
mod tests {
    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    use super::*;

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn response_curve_predict_uses_metal_affine_path_with_cpu_parity() {
        if !crate::available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let rows = vec![
            DeepResponseRow {
                features: vec![1.0, 0.0],
                candidate_value: 1.0,
                response: Some(1.0),
                group_id: Some("g".to_string()),
                candidate_id: Some("a".to_string()),
            },
            DeepResponseRow {
                features: vec![0.0, 1.0],
                candidate_value: 2.0,
                response: Some(0.0),
                group_id: Some("g".to_string()),
                candidate_id: Some("b".to_string()),
            },
        ];
        let cpu = response_curve_fit_with_backend(&rows, "binary", None, Some("cpu")).unwrap();
        let metal = response_curve_fit_with_backend(&rows, "binary", None, Some("metal")).unwrap();
        let cpu_scores = response_curve_predict(&cpu, &rows).unwrap();
        let metal_scores = response_curve_predict(&metal, &rows).unwrap();
        assert_eq!(metal.backend.selected, "metal");
        for (left, right) in cpu_scores.iter().zip(&metal_scores) {
            assert!((left.response_score - right.response_score).abs() < 1.0e-4);
        }
    }
}
