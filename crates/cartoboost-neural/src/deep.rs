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
    #[serde(default)]
    pub hidden_weights: Vec<Vec<f64>>,
    #[serde(default)]
    pub hidden_biases: Vec<f64>,
    #[serde(default)]
    pub output_weights: Vec<f64>,
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
    #[serde(default)]
    pub hidden_weights: Vec<Vec<f64>>,
    #[serde(default)]
    pub hidden_biases: Vec<f64>,
    #[serde(default)]
    pub output_weights: Vec<f64>,
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
    pub target: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepDirectionalPairArtifact {
    pub model_class: String,
    pub model_version: String,
    pub pair_weights: BTreeMap<String, f64>,
    pub source_weights: BTreeMap<String, f64>,
    pub target_weights: BTreeMap<String, f64>,
    pub feature_means: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub intercept: f64,
    pub global_mean: f64,
    pub schema_hash: String,
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
    #[serde(default)]
    pub hidden_weights: Vec<Vec<f64>>,
    #[serde(default)]
    pub hidden_biases: Vec<f64>,
    #[serde(default)]
    pub output_weights: Vec<f64>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepTemporalEntityArtifact {
    pub model_class: String,
    pub model_version: String,
    pub lookback: usize,
    pub horizon: usize,
    pub entity_count: usize,
    pub attention_heads: usize,
    pub attention_queries: Vec<Vec<f64>>,
    pub attention_keys: Vec<Vec<f64>>,
    pub decoder_weights: Vec<Vec<Vec<f64>>>,
    pub intercepts: Vec<Vec<f64>>,
    pub residual_scales: Vec<f64>,
    pub last_window: Vec<Vec<f64>>,
    pub schema_hash: String,
}

struct TanhFit {
    hidden_weights: Vec<Vec<f64>>,
    hidden_biases: Vec<f64>,
    output_weights: Vec<f64>,
    intercept: f64,
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
    let y_mean = rows.iter().filter_map(|row| row.response).sum::<f64>() / rows.len() as f64;
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
        Some("decreasing") => slope = -slope.abs().max(1e-9),
        Some(other) => return invalid(format!("unknown monotone mode {other:?}")),
        None => {}
    }
    let fit = fit_tanh_response_network(rows, &means, slope, response_type, y_mean)?;
    let intercept = fit.intercept - slope * c_mean;
    let hidden_weights = fit.hidden_weights;
    let hidden_biases = fit.hidden_biases;
    let output_weights = fit.output_weights;
    let weights = linearized_feature_weights(&hidden_weights, &output_weights, dim);
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
        hidden_weights,
        hidden_biases,
        output_weights,
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
    let scores = if artifact.hidden_weights.is_empty() {
        backend_affine_scores(
            &artifact.backend,
            &features,
            &artifact.feature_means,
            &artifact.feature_weights,
            &intercepts,
        )?
    } else {
        rows.iter()
            .zip(intercepts)
            .map(|(row, intercept)| {
                neural_score(
                    &row.features,
                    &artifact.feature_means,
                    &artifact.hidden_weights,
                    &artifact.hidden_biases,
                    &artifact.output_weights,
                    intercept,
                )
            })
            .collect()
    };
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
    let fit = fit_tanh_binary_network(features, labels, &means, y_mean)?;
    let hidden_weights = fit.hidden_weights;
    let hidden_biases = fit.hidden_biases;
    let output_weights = fit.output_weights;
    let intercept = fit.intercept;
    let weights = linearized_feature_weights(&hidden_weights, &output_weights, dim);
    let mut calibration = BTreeMap::new();
    calibration.insert("positive_rate".to_string(), y_mean);
    Ok(DeepEventArtifact {
        model_class: "EventOutcomeModel".to_string(),
        model_version: "1".to_string(),
        feature_means: means,
        feature_weights: weights,
        intercept,
        hidden_weights,
        hidden_biases,
        output_weights,
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
    let logits = if artifact.hidden_weights.is_empty() {
        backend_affine_scores(
            &artifact.backend,
            features,
            &artifact.feature_means,
            &artifact.feature_weights,
            &intercepts,
        )?
    } else {
        features
            .iter()
            .map(|row| {
                neural_score(
                    row,
                    &artifact.feature_means,
                    &artifact.hidden_weights,
                    &artifact.hidden_biases,
                    &artifact.output_weights,
                    artifact.intercept,
                )
            })
            .collect()
    };
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

pub fn directional_pair_fit(
    rows: &[DeepDirectionalPairRow],
) -> Result<DeepDirectionalPairArtifact> {
    if rows.is_empty() {
        return invalid("directional pair rows cannot be empty");
    }
    if rows.iter().any(|row| row.target.is_none()) {
        return invalid("directional pair fit rows must include target values");
    }
    let dim = rows[0].features.len();
    if rows.iter().any(|row| row.features.len() != dim) {
        return invalid("all directional pair feature rows must have the same width");
    }
    let means = feature_means(rows.iter().map(|row| row.features.as_slice()), dim)?;
    let y = rows
        .iter()
        .map(|row| row.target.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let global_mean = y.iter().sum::<f64>() / y.len() as f64;
    let mut pair_sum: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut source_sum: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut target_sum: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for (row, value) in rows.iter().zip(&y) {
        let centered = *value - global_mean;
        add_group_sum(
            &mut pair_sum,
            &format!("{}->{}", row.source_id, row.target_id),
            centered,
        );
        add_group_sum(&mut source_sum, &row.source_id, centered);
        add_group_sum(&mut target_sum, &row.target_id, centered);
    }
    let pair_weights = shrink_group_means(pair_sum, 2.0);
    let source_weights = shrink_group_means(source_sum, 4.0);
    let target_weights = shrink_group_means(target_sum, 4.0);
    let residuals = rows
        .iter()
        .zip(&y)
        .map(|(row, value)| {
            value
                - global_mean
                - pair_weights
                    .get(&format!("{}->{}", row.source_id, row.target_id))
                    .copied()
                    .unwrap_or(0.0)
                - source_weights.get(&row.source_id).copied().unwrap_or(0.0)
                - target_weights.get(&row.target_id).copied().unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    let feature_weights = fit_linear_weights(
        rows.iter()
            .map(|row| row.features.as_slice())
            .collect::<Vec<_>>()
            .as_slice(),
        &residuals,
        &means,
    );
    Ok(DeepDirectionalPairArtifact {
        model_class: "DirectionalPairForecaster".to_string(),
        model_version: "1".to_string(),
        pair_weights,
        source_weights,
        target_weights,
        feature_means: means,
        feature_weights,
        intercept: global_mean,
        global_mean,
        schema_hash: schema_hash(dim, "directional_pair"),
    })
}

pub fn directional_pair_predict(
    artifact: &DeepDirectionalPairArtifact,
    rows: &[DeepDirectionalPairRow],
) -> Result<Vec<f64>> {
    if rows.is_empty() {
        return invalid("directional pair rows cannot be empty");
    }
    Ok(rows
        .iter()
        .map(|row| {
            let pair = format!("{}->{}", row.source_id, row.target_id);
            let mut score = artifact.intercept
                + artifact.pair_weights.get(&pair).copied().unwrap_or(0.0)
                + artifact
                    .source_weights
                    .get(&row.source_id)
                    .copied()
                    .unwrap_or(0.0)
                + artifact
                    .target_weights
                    .get(&row.target_id)
                    .copied()
                    .unwrap_or(0.0);
            for (idx, value) in row.features.iter().enumerate() {
                score += artifact.feature_weights.get(idx).copied().unwrap_or(0.0)
                    * (value - artifact.feature_means.get(idx).copied().unwrap_or(0.0));
            }
            score
        })
        .collect())
}

pub fn directional_pair_predictions(rows: &[DeepDirectionalPairRow]) -> Result<Vec<f64>> {
    let artifact = directional_pair_fit(rows)?;
    directional_pair_predict(&artifact, rows)
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
    let residuals = rows
        .iter()
        .map(|row| row.actual_value.unwrap_or(row.baseline_value) - row.baseline_value)
        .collect::<Vec<_>>();
    let residual_mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let features = rows
        .iter()
        .map(|row| row.features.clone())
        .collect::<Vec<_>>();
    let fit = fit_tanh_regression_network(&features, &residuals, &means, residual_mean)?;
    let hidden_weights = fit.hidden_weights;
    let hidden_biases = fit.hidden_biases;
    let output_weights = fit.output_weights;
    let intercept = fit.intercept;
    let weights = linearized_feature_weights(&hidden_weights, &output_weights, dim);
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
        intercept,
        baseline_weight: 1.0,
        hidden_weights,
        hidden_biases,
        output_weights,
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
    let residuals = if artifact.hidden_weights.is_empty() {
        backend_affine_scores(
            &artifact.backend,
            &features,
            &artifact.feature_means,
            &artifact.feature_weights,
            &intercepts,
        )?
    } else {
        features
            .iter()
            .map(|row| {
                neural_score(
                    row,
                    &artifact.feature_means,
                    &artifact.hidden_weights,
                    &artifact.hidden_biases,
                    &artifact.output_weights,
                    artifact.intercept,
                )
            })
            .collect()
    };
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
    constrained_decision_select_with_options(candidates, objective, constraints, fallback, 0.0)
}

pub fn constrained_decision_select_with_options(
    candidates: &[BTreeMap<String, serde_json::Value>],
    objective: &str,
    constraints: &BTreeMap<String, f64>,
    fallback: &str,
    risk_aversion: f64,
) -> Result<Vec<DeepDecisionChoice>> {
    if !risk_aversion.is_finite() || risk_aversion < 0.0 {
        return invalid("risk_aversion must be a nonnegative finite value");
    }
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
            let l_score = objective_score(left, objective, risk_aversion)
                - constraint_penalty(left, constraints);
            let r_score = objective_score(right, objective, risk_aversion)
                - constraint_penalty(right, constraints);
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
            score: objective_score(best, objective, risk_aversion)
                - constraint_penalty(best, constraints),
            reason_code,
        });
    }
    Ok(choices)
}

pub fn temporal_entity_fit(
    y: &[Vec<f64>],
    lookback: usize,
    horizon: usize,
) -> Result<DeepTemporalEntityArtifact> {
    validate_panel(y)?;
    if lookback == 0 || horizon == 0 {
        return invalid("lookback and horizon must be positive");
    }
    if y.len() <= lookback {
        return invalid("panel history must exceed lookback");
    }
    let entity_count = y[0].len();
    if y.len() <= lookback + horizon {
        return invalid("panel history must exceed lookback plus horizon");
    }
    let attention_heads = lookback.clamp(2, 6);
    let attention_queries =
        deterministic_weight_matrix(attention_heads, lookback, 0x17c9_aa7f_9e37_79b9);
    let attention_keys =
        deterministic_weight_matrix(attention_heads, lookback, 0x6a09_e667_f3bc_c909);
    let feature_len = temporal_entity_feature_len(attention_heads);
    let mut decoder_weights = vec![vec![vec![0.0; feature_len]; entity_count]; horizon];
    let mut intercepts = vec![vec![0.0; entity_count]; horizon];
    let mut residual_scales = vec![0.0; entity_count];

    let samples = y.len() - lookback - horizon + 1;
    for h in 0..horizon {
        for entity in 0..entity_count {
            let mut xtx = vec![vec![0.0; feature_len]; feature_len];
            let mut xty = vec![0.0; feature_len];
            let mut target_sum = 0.0;
            for sample in 0..samples {
                let cutoff = sample + lookback;
                let features = temporal_entity_features(
                    &y[sample..cutoff],
                    entity,
                    &attention_queries,
                    &attention_keys,
                );
                let target = y[cutoff + h][entity];
                target_sum += target;
                for row in 0..feature_len {
                    xty[row] += features[row] * target;
                    for col in 0..feature_len {
                        xtx[row][col] += features[row] * features[col];
                    }
                }
            }
            for (idx, row) in xtx.iter_mut().enumerate() {
                row[idx] += 1.0e-4;
            }
            decoder_weights[h][entity] = solve_dense_system(xtx, xty);
            intercepts[h][entity] = target_sum / samples as f64;
        }
    }
    for entity in 0..entity_count {
        let mut rss = 0.0;
        let mut count = 0usize;
        for sample in 0..samples {
            let cutoff = sample + lookback;
            let features = temporal_entity_features(
                &y[sample..cutoff],
                entity,
                &attention_queries,
                &attention_keys,
            );
            let pred = dot(&decoder_weights[0][entity], &features);
            rss += (y[cutoff][entity] - pred).powi(2);
            count += 1;
        }
        residual_scales[entity] = (rss / count.max(1) as f64).sqrt();
    }
    Ok(DeepTemporalEntityArtifact {
        model_class: "TemporalEntityTransformer".to_string(),
        model_version: "1".to_string(),
        lookback,
        horizon,
        entity_count,
        attention_heads,
        attention_queries,
        attention_keys,
        decoder_weights,
        intercepts,
        residual_scales,
        last_window: y[y.len() - lookback..].to_vec(),
        schema_hash: schema_hash(entity_count, "temporal_entity"),
    })
}

pub fn temporal_entity_predict(
    artifact: &DeepTemporalEntityArtifact,
    horizon: usize,
) -> Result<Vec<Vec<f64>>> {
    if artifact.last_window.len() != artifact.lookback {
        return invalid("temporal entity artifact last window does not match lookback");
    }
    let horizon = if horizon == 0 {
        artifact.horizon
    } else {
        horizon
    };
    let mut history = artifact.last_window.clone();
    let mut out = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let mut row = vec![0.0; artifact.entity_count];
        for (entity, value) in row.iter_mut().enumerate() {
            let h = out.len().min(artifact.decoder_weights.len() - 1);
            let start = history.len() - artifact.lookback;
            let features = temporal_entity_features(
                &history[start..],
                entity,
                &artifact.attention_queries,
                &artifact.attention_keys,
            );
            *value = dot(&artifact.decoder_weights[h][entity], &features);
        }
        history.push(row.clone());
        out.push(row);
    }
    Ok(out)
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

fn fit_tanh_response_network(
    rows: &[DeepResponseRow],
    means: &[f64],
    candidate_slope: f64,
    response_type: &str,
    y_mean: f64,
) -> Result<TanhFit> {
    let features = rows
        .iter()
        .map(|row| {
            let mut values = row.features.clone();
            values.push(row.candidate_value);
            values
        })
        .collect::<Vec<_>>();
    let mut expanded_means = means.to_vec();
    expanded_means
        .push(rows.iter().map(|row| row.candidate_value).sum::<f64>() / rows.len() as f64);
    let labels = rows
        .iter()
        .map(|row| row.response.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mut fit = if response_type == "binary" {
        fit_tanh_binary_network(&features, &labels, &expanded_means, y_mean)?
    } else {
        fit_tanh_regression_network(&features, &labels, &expanded_means, y_mean)?
    };
    for hidden in &mut fit.hidden_weights {
        if let Some(last) = hidden.last_mut() {
            *last += candidate_slope;
        }
        hidden.truncate(means.len());
    }
    Ok(fit)
}

fn fit_tanh_binary_network(
    features: &[Vec<f64>],
    labels: &[f64],
    means: &[f64],
    y_mean: f64,
) -> Result<TanhFit> {
    fit_tanh_network(
        features,
        labels,
        means,
        logit(y_mean.clamp(1e-6, 1.0 - 1e-6)),
        true,
    )
}

fn fit_tanh_regression_network(
    features: &[Vec<f64>],
    labels: &[f64],
    means: &[f64],
    y_mean: f64,
) -> Result<TanhFit> {
    fit_tanh_network(features, labels, means, y_mean, false)
}

fn fit_tanh_network(
    features: &[Vec<f64>],
    labels: &[f64],
    means: &[f64],
    initial_intercept: f64,
    logistic: bool,
) -> Result<TanhFit> {
    validate_matrix(features)?;
    if labels.len() != features.len() {
        return invalid("labels must match feature row count");
    }
    let dim = features[0].len();
    let hidden = dim.clamp(2, 8);
    let mut hidden_weights = vec![vec![0.0; dim]; hidden];
    let mut hidden_biases = vec![0.0; hidden];
    let mut output_weights = vec![0.0; hidden];
    for (h, weights) in hidden_weights.iter_mut().enumerate() {
        output_weights[h] = if h % 2 == 0 { 0.05 } else { -0.05 };
        for (j, weight) in weights.iter_mut().enumerate() {
            *weight = ((((h + 1) * (j + 3)) % 7) as f64 - 3.0) * 0.03;
        }
    }
    let mut intercept = initial_intercept;
    let lr = if logistic { 0.05 } else { 0.02 };
    for _ in 0..240 {
        for (row, &label) in features.iter().zip(labels) {
            let centered = row
                .iter()
                .enumerate()
                .map(|(idx, value)| value - means[idx])
                .collect::<Vec<_>>();
            let mut acts = vec![0.0; hidden];
            let mut raw = intercept;
            for h in 0..hidden {
                let z = hidden_biases[h]
                    + hidden_weights[h]
                        .iter()
                        .zip(&centered)
                        .map(|(w, x)| w * x)
                        .sum::<f64>();
                acts[h] = z.tanh();
                raw += output_weights[h] * acts[h];
            }
            let pred = if logistic { sigmoid(raw) } else { raw };
            let grad_raw = if logistic {
                pred - label
            } else {
                2.0 * (pred - label)
            };
            intercept -= lr * grad_raw;
            for h in 0..hidden {
                let old_out = output_weights[h];
                output_weights[h] -= lr * (grad_raw * acts[h] + 1e-4 * output_weights[h]);
                let grad_z = grad_raw * old_out * (1.0 - acts[h] * acts[h]);
                hidden_biases[h] -= lr * grad_z;
                for (j, x) in centered.iter().enumerate() {
                    hidden_weights[h][j] -= lr * (grad_z * x + 1e-4 * hidden_weights[h][j]);
                }
            }
        }
    }
    Ok(TanhFit {
        hidden_weights,
        hidden_biases,
        output_weights,
        intercept,
    })
}

fn neural_score(
    row: &[f64],
    means: &[f64],
    hidden_weights: &[Vec<f64>],
    hidden_biases: &[f64],
    output_weights: &[f64],
    intercept: f64,
) -> f64 {
    let mut score = intercept;
    for (h, weights) in hidden_weights.iter().enumerate() {
        let z = hidden_biases.get(h).copied().unwrap_or(0.0)
            + weights
                .iter()
                .enumerate()
                .map(|(idx, weight)| {
                    weight
                        * (row.get(idx).copied().unwrap_or(0.0)
                            - means.get(idx).copied().unwrap_or(0.0))
                })
                .sum::<f64>();
        score += output_weights.get(h).copied().unwrap_or(0.0) * z.tanh();
    }
    score
}

fn linearized_feature_weights(
    hidden_weights: &[Vec<f64>],
    output_weights: &[f64],
    dim: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; dim];
    for (h, hidden) in hidden_weights.iter().enumerate() {
        let out = output_weights.get(h).copied().unwrap_or(0.0);
        for (idx, weight) in hidden.iter().take(dim).enumerate() {
            weights[idx] += out * weight;
        }
    }
    weights
}

fn fit_linear_weights(rows: &[&[f64]], labels: &[f64], means: &[f64]) -> Vec<f64> {
    let dim = means.len();
    let y_mean = labels.iter().sum::<f64>() / labels.len().max(1) as f64;
    let mut weights = vec![0.0; dim];
    for idx in 0..dim {
        let mut num = 0.0;
        let mut den = 0.0;
        for (row, label) in rows.iter().zip(labels) {
            let x = row[idx] - means[idx];
            num += x * (label - y_mean);
            den += x * x;
        }
        weights[idx] = if den > 0.0 { num / (den + 1e-9) } else { 0.0 };
    }
    weights
}

fn add_group_sum(groups: &mut BTreeMap<String, (f64, usize)>, key: &str, value: f64) {
    let entry = groups.entry(key.to_string()).or_insert((0.0, 0));
    entry.0 += value;
    entry.1 += 1;
}

fn shrink_group_means(
    groups: BTreeMap<String, (f64, usize)>,
    shrinkage: f64,
) -> BTreeMap<String, f64> {
    groups
        .into_iter()
        .map(|(key, (sum, count))| (key, sum / (count as f64 + shrinkage)))
        .collect()
}

fn temporal_entity_feature_len(attention_heads: usize) -> usize {
    3 + attention_heads * 2
}

fn temporal_entity_features(
    window: &[Vec<f64>],
    entity: usize,
    attention_queries: &[Vec<f64>],
    attention_keys: &[Vec<f64>],
) -> Vec<f64> {
    let lookback = window.len();
    let entity_count = window[0].len();
    let attention_heads = attention_queries.len();
    let mut features = vec![0.0; temporal_entity_feature_len(attention_heads)];
    let own = window.iter().map(|row| row[entity]).collect::<Vec<_>>();
    let panel_mean = window
        .iter()
        .map(|row| row.iter().sum::<f64>() / entity_count as f64)
        .collect::<Vec<_>>();
    features[0] = own[lookback - 1];
    features[1] = panel_mean[lookback - 1];
    features[2] = 1.0;
    for head in 0..attention_heads {
        features[3 + head * 2] =
            temporal_attention_pool(&own, &attention_queries[head], &attention_keys[head]);
        features[3 + head * 2 + 1] =
            temporal_attention_pool(&panel_mean, &attention_queries[head], &attention_keys[head]);
    }
    features
}

fn temporal_attention_pool(values: &[f64], query: &[f64], key: &[f64]) -> f64 {
    let width = values.len().min(query.len()).min(key.len());
    let scores = (0..width)
        .map(|idx| (query[idx] * values[idx] + key[idx]).tanh())
        .collect::<Vec<_>>();
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .collect::<Vec<_>>();
    let denom = weights.iter().sum::<f64>().max(1.0e-12);
    weights
        .iter()
        .zip(values)
        .take(width)
        .map(|(weight, value)| weight / denom * value)
        .sum()
}

fn deterministic_weight_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let scale = (cols.max(1) as f64).sqrt().recip();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                    (unit * 2.0 - 1.0) * scale
                })
                .collect()
        })
        .collect()
}

fn dot(weights: &[f64], values: &[f64]) -> f64 {
    weights.iter().zip(values).map(|(w, v)| w * v).sum()
}

fn solve_dense_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if best != pivot {
            matrix.swap(pivot, best);
            rhs.swap(pivot, best);
        }
        let diag = matrix[pivot][pivot];
        if diag.abs() < 1.0e-12 {
            continue;
        }
        for value in matrix[pivot].iter_mut().take(n).skip(pivot) {
            *value /= diag;
        }
        rhs[pivot] /= diag;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor.abs() < 1.0e-12 {
                continue;
            }
            for (value, pivot_value) in matrix[row]
                .iter_mut()
                .zip(pivot_row.iter())
                .take(n)
                .skip(pivot)
            {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    rhs
}

fn constraint_penalty(
    row: &BTreeMap<String, serde_json::Value>,
    constraints: &BTreeMap<String, f64>,
) -> f64 {
    let mut penalty = 0.0;
    for (name, &limit) in constraints {
        penalty += match name.as_str() {
            "soft_min_response_probability" => {
                (limit - json_f64(row, "response_probability").unwrap_or(0.0)).max(0.0)
            }
            "soft_max_risk_score" | "soft_max_tail_risk" => {
                (json_f64(row, "risk_score").unwrap_or(limit) - limit).max(0.0)
            }
            "soft_min_candidate_value" => {
                (limit - json_f64(row, "candidate_value").unwrap_or(0.0)).max(0.0)
            }
            "soft_max_candidate_value" => {
                (json_f64(row, "candidate_value").unwrap_or(limit) - limit).max(0.0)
            }
            _ => 0.0,
        };
    }
    penalty * 1_000.0
}

fn objective_score(
    row: &BTreeMap<String, serde_json::Value>,
    objective: &str,
    risk_aversion: f64,
) -> f64 {
    let risk = json_f64(row, "risk_score")
        .or_else(|| json_f64(row, "tail_risk"))
        .unwrap_or(0.0);
    let utility = json_f64(row, "expected_utility")
        .or_else(|| json_f64(row, "utility"))
        .unwrap_or(0.0);
    let probability = json_f64(row, "response_probability")
        .or_else(|| json_f64(row, "calibrated_probability"))
        .unwrap_or(0.0);
    let expected_value = json_f64(row, "expected_value")
        .or_else(|| json_f64(row, "candidate_value"))
        .unwrap_or(0.0);
    match objective {
        "max_response_probability" => probability - risk_aversion * risk,
        "min_expected_value" => -expected_value - risk_aversion * risk,
        "risk_adjusted_utility" => utility - (1.0 + risk_aversion) * risk,
        "expected_value" | "max_expected_value" => expected_value - risk_aversion * risk,
        "probability_weighted_value" => probability * expected_value - risk_aversion * risk,
        "max_score" => json_f64(row, "score").unwrap_or(utility) - risk_aversion * risk,
        _ => utility + probability * expected_value - risk_aversion * risk,
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

fn validate_panel(y: &[Vec<f64>]) -> Result<()> {
    if y.is_empty() || y[0].is_empty() {
        return invalid("panel matrix must be non-empty");
    }
    let width = y[0].len();
    if y.iter()
        .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
    {
        return invalid("panel rows must be finite and have the same width");
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
    use super::*;

    #[test]
    fn directional_pair_fit_learns_ordered_pair_effects() {
        let rows = vec![
            DeepDirectionalPairRow {
                source_id: "A".to_string(),
                target_id: "B".to_string(),
                timestamp: None,
                features: vec![0.0],
                target: Some(1.0),
            },
            DeepDirectionalPairRow {
                source_id: "B".to_string(),
                target_id: "A".to_string(),
                timestamp: None,
                features: vec![0.0],
                target: Some(4.0),
            },
            DeepDirectionalPairRow {
                source_id: "A".to_string(),
                target_id: "B".to_string(),
                timestamp: None,
                features: vec![1.0],
                target: Some(2.0),
            },
            DeepDirectionalPairRow {
                source_id: "B".to_string(),
                target_id: "A".to_string(),
                timestamp: None,
                features: vec![1.0],
                target: Some(5.0),
            },
        ];
        let artifact = directional_pair_fit(&rows).unwrap();
        let preds = directional_pair_predict(&artifact, &rows).unwrap();
        assert!(preds[1] > preds[0]);
        assert!(preds[3] > preds[2]);
        assert!(artifact.pair_weights.contains_key("A->B"));
        assert!(artifact.pair_weights.contains_key("B->A"));
    }

    #[test]
    fn response_event_and_residual_artifacts_have_learned_hidden_heads() {
        let response_rows = vec![
            DeepResponseRow {
                features: vec![0.0, 0.0],
                candidate_value: 1.0,
                response: Some(0.0),
                group_id: None,
                candidate_id: None,
            },
            DeepResponseRow {
                features: vec![1.0, 0.0],
                candidate_value: 2.0,
                response: Some(1.0),
                group_id: None,
                candidate_id: None,
            },
            DeepResponseRow {
                features: vec![0.0, 1.0],
                candidate_value: 3.0,
                response: Some(1.0),
                group_id: None,
                candidate_id: None,
            },
        ];
        let response = response_curve_fit(&response_rows, "binary", Some("increasing")).unwrap();
        assert!(!response.hidden_weights.is_empty());
        let response_pred = response_curve_predict(&response, &response_rows).unwrap();
        assert!(response_pred[2].response_score > response_pred[0].response_score);

        let features = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let event = event_outcome_fit(&features, &labels).unwrap();
        assert!(!event.hidden_weights.is_empty());
        let event_pred = event_outcome_predict(&event, &features).unwrap();
        assert!(event_pred[3].calibrated_probability > event_pred[0].calibrated_probability);

        let residual_rows = vec![
            DeepServiceResidualRow {
                baseline_value: 10.0,
                actual_value: Some(10.0),
                features: vec![0.0],
            },
            DeepServiceResidualRow {
                baseline_value: 10.0,
                actual_value: Some(13.0),
                features: vec![2.0],
            },
        ];
        let residual = service_residual_fit(&residual_rows).unwrap();
        assert!(!residual.hidden_weights.is_empty());
        let residual_pred = service_residual_predict(&residual, &residual_rows).unwrap();
        assert!(residual_pred[1].prediction > residual_pred[0].prediction);
    }

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
