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
    #[serde(default = "default_directional_pair_architecture")]
    pub architecture: String,
    pub pair_weights: BTreeMap<String, f64>,
    pub source_weights: BTreeMap<String, f64>,
    pub target_weights: BTreeMap<String, f64>,
    pub feature_means: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub intercept: f64,
    pub global_mean: f64,
    pub schema_hash: String,
    #[serde(default)]
    pub loss: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub source_id_map: BTreeMap<String, usize>,
    #[serde(default)]
    pub target_id_map: BTreeMap<String, usize>,
    #[serde(default)]
    pub pair_bucket_count: usize,
    #[serde(default)]
    pub pair_global_bucket: usize,
    #[serde(default)]
    pub embedding_dim: usize,
    #[serde(default)]
    pub source_embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub target_embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub pair_bucket_embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub dense_projection: Vec<Vec<f64>>,
    #[serde(default)]
    pub hidden_weights: Vec<Vec<f64>>,
    #[serde(default)]
    pub hidden_biases: Vec<f64>,
    #[serde(default)]
    pub output_weights: Vec<f64>,
    #[serde(default)]
    pub train_metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectionalPairFitOptions {
    #[serde(default = "default_directional_pair_architecture")]
    pub architecture: String,
    #[serde(default = "default_pair_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default = "default_pair_bucket_count")]
    pub pair_bucket_count: usize,
    #[serde(default = "default_pair_hidden_dim")]
    pub hidden_dim: usize,
    #[serde(default = "default_pair_epochs")]
    pub epochs: usize,
    #[serde(default = "default_pair_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_pair_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_pair_gradient_clip")]
    pub gradient_clip: f64,
    #[serde(default = "default_pair_patience")]
    pub early_stopping_rounds: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_pair_loss")]
    pub loss: String,
}

fn default_directional_pair_architecture() -> String {
    "shrinkage_effects".to_string()
}
fn default_pair_embedding_dim() -> usize {
    4
}
fn default_pair_bucket_count() -> usize {
    64
}
fn default_pair_hidden_dim() -> usize {
    12
}
fn default_pair_epochs() -> usize {
    700
}
fn default_pair_learning_rate() -> f64 {
    0.018
}
fn default_pair_weight_decay() -> f64 {
    1e-4
}
fn default_pair_gradient_clip() -> f64 {
    1.0
}
fn default_pair_patience() -> usize {
    80
}
fn default_pair_loss() -> String {
    "squared_error".to_string()
}

impl Default for DirectionalPairFitOptions {
    fn default() -> Self {
        Self {
            architecture: default_directional_pair_architecture(),
            embedding_dim: default_pair_embedding_dim(),
            pair_bucket_count: default_pair_bucket_count(),
            hidden_dim: default_pair_hidden_dim(),
            epochs: default_pair_epochs(),
            learning_rate: default_pair_learning_rate(),
            weight_decay: default_pair_weight_decay(),
            gradient_clip: default_pair_gradient_clip(),
            early_stopping_rounds: default_pair_patience(),
            seed: 0,
            loss: default_pair_loss(),
        }
    }
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
pub struct DeepChoiceSetPrediction {
    pub decision_id: String,
    pub candidate_id: String,
    pub candidate_value: f64,
    pub utility: f64,
    pub choice_probability: f64,
    pub nested_probability: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepCounterfactualCandidate {
    pub decision_id: String,
    pub candidate_id: String,
    pub candidate_value: f64,
    pub utility: f64,
    pub choice_probability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepChoiceSetReport {
    pub predictions: Vec<DeepChoiceSetPrediction>,
    pub counterfactual_best: Vec<DeepCounterfactualCandidate>,
    pub calibration: BTreeMap<String, f64>,
    pub benchmark: BTreeMap<String, f64>,
    pub metadata: BTreeMap<String, String>,
}

pub type ChoiceSetTransformer = DeepChoiceSetReport;
pub type UtilityNet = DeepChoiceSetReport;
pub type NestedChoiceHead = DeepChoiceSetReport;
pub type CounterfactualCandidateScorer = DeepChoiceSetReport;

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
    directional_pair_fit_with_options(rows, &DirectionalPairFitOptions::default())
}

pub fn directional_pair_fit_with_options(
    rows: &[DeepDirectionalPairRow],
    options: &DirectionalPairFitOptions,
) -> Result<DeepDirectionalPairArtifact> {
    match options.architecture.as_str() {
        "shrinkage_effects" => directional_pair_fit_shrinkage(rows, options),
        "pair_embedding_mlp" => directional_pair_fit_embedding_mlp(rows, options),
        "pair_temporal_ssm" | "pair_regime_moe" => {
            directional_pair_fit_expanded_embedding(rows, options)
        }
        other => invalid(format!("unknown directional pair architecture {other:?}")),
    }
}

fn directional_pair_fit_expanded_embedding(
    rows: &[DeepDirectionalPairRow],
    options: &DirectionalPairFitOptions,
) -> Result<DeepDirectionalPairArtifact> {
    let expanded = expand_pair_architecture_rows(rows, &options.architecture);
    let mut inner = options.clone();
    let requested_architecture = options.architecture.clone();
    inner.architecture = "pair_embedding_mlp".to_string();
    let mut artifact = directional_pair_fit_embedding_mlp(&expanded, &inner)?;
    artifact.architecture = requested_architecture.clone();
    artifact.schema_hash = schema_hash(expanded[0].features.len(), &requested_architecture);
    artifact.train_metrics.insert(
        "expanded_feature_count".to_string(),
        expanded[0].features.len() as f64,
    );
    Ok(artifact)
}

fn directional_pair_fit_shrinkage(
    rows: &[DeepDirectionalPairRow],
    options: &DirectionalPairFitOptions,
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
        architecture: "shrinkage_effects".to_string(),
        pair_weights,
        source_weights,
        target_weights,
        feature_means: means,
        feature_weights,
        intercept: global_mean,
        global_mean,
        schema_hash: schema_hash(dim, "directional_pair"),
        loss: options.loss.clone(),
        seed: options.seed,
        source_id_map: BTreeMap::new(),
        target_id_map: BTreeMap::new(),
        pair_bucket_count: 0,
        pair_global_bucket: 0,
        embedding_dim: 0,
        source_embeddings: Vec::new(),
        target_embeddings: Vec::new(),
        pair_bucket_embeddings: Vec::new(),
        dense_projection: Vec::new(),
        hidden_weights: Vec::new(),
        hidden_biases: Vec::new(),
        output_weights: Vec::new(),
        train_metrics: BTreeMap::new(),
    })
}

fn directional_pair_fit_embedding_mlp(
    rows: &[DeepDirectionalPairRow],
    options: &DirectionalPairFitOptions,
) -> Result<DeepDirectionalPairArtifact> {
    if rows.is_empty() {
        return invalid("directional pair rows cannot be empty");
    }
    if rows.iter().any(|row| row.target.is_none()) {
        return invalid("directional pair fit rows must include target values");
    }
    let dim = rows[0].features.len();
    if rows
        .iter()
        .any(|row| row.features.len() != dim || row.features.iter().any(|v| !v.is_finite()))
    {
        return invalid("all directional pair feature rows must be finite and have the same width");
    }
    if options.embedding_dim == 0 || options.pair_bucket_count < 2 || options.hidden_dim == 0 {
        return invalid("embedding_dim, hidden_dim, and pair_bucket_count must be positive");
    }
    let means = feature_means(rows.iter().map(|row| row.features.as_slice()), dim)?;
    let targets = rows
        .iter()
        .map(|row| row.target.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let global_mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let mut source_id_map = BTreeMap::new();
    let mut target_id_map = BTreeMap::new();
    source_id_map.insert("__unknown__".to_string(), 0);
    target_id_map.insert("__unknown__".to_string(), 0);
    for row in rows {
        if !source_id_map.contains_key(&row.source_id) {
            source_id_map.insert(row.source_id.clone(), source_id_map.len());
        }
        if !target_id_map.contains_key(&row.target_id) {
            target_id_map.insert(row.target_id.clone(), target_id_map.len());
        }
    }
    let embed = options.embedding_dim;
    let dense_dim = if dim == 0 { 0 } else { embed };
    let input_dim = embed * 5 + dense_dim;
    let hidden = options.hidden_dim;
    let seed = options.seed;
    let mut source_embeddings = init_matrix(source_id_map.len(), embed, seed ^ 0x51);
    let mut target_embeddings = init_matrix(target_id_map.len(), embed, seed ^ 0x71);
    let mut pair_bucket_embeddings = init_matrix(options.pair_bucket_count, embed, seed ^ 0x91);
    let mut dense_projection = init_matrix(dim, dense_dim, seed ^ 0xB1);
    let mut hidden_weights = init_matrix(hidden, input_dim, seed ^ 0xD1);
    let mut hidden_biases = vec![0.0; hidden];
    let mut output_weights = init_vec(hidden, seed ^ 0xF1);
    let mut intercept = global_mean;
    let mut opt = AdamWState::new(
        source_embeddings.len() * embed
            + target_embeddings.len() * embed
            + pair_bucket_embeddings.len() * embed
            + dim * dense_dim
            + hidden * input_dim
            + hidden
            + hidden
            + 1,
    );
    let split = if rows.len() >= 12 {
        rows.len() * 4 / 5
    } else {
        rows.len()
    };
    let train_idx = (0..split).collect::<Vec<_>>();
    let valid_idx = (split..rows.len()).collect::<Vec<_>>();
    let mut best_loss = f64::INFINITY;
    let mut stale = 0usize;
    for epoch in 0..options.epochs.max(1) {
        let mut order = train_idx.clone();
        deterministic_shuffle(&mut order, seed ^ epoch as u64);
        for idx in order {
            let row = &rows[idx];
            let src = source_id_map[&row.source_id];
            let dst = target_id_map[&row.target_id];
            let bucket = pair_bucket(&row.source_id, &row.target_id, options.pair_bucket_count);
            let (input, acts, pred) = pair_mlp_forward(
                row,
                &means,
                src,
                dst,
                bucket,
                &source_embeddings,
                &target_embeddings,
                &pair_bucket_embeddings,
                &dense_projection,
                &hidden_weights,
                &hidden_biases,
                &output_weights,
                intercept,
            );
            let mut grad = 2.0 * (pred - row.target.unwrap_or(0.0));
            grad = grad.clamp(-options.gradient_clip, options.gradient_clip);
            intercept =
                opt.update_scalar(intercept, grad, options.learning_rate, options.weight_decay);
            let old_out = output_weights.clone();
            for h in 0..hidden {
                output_weights[h] = opt.update_scalar(
                    output_weights[h],
                    grad * acts[h],
                    options.learning_rate,
                    options.weight_decay,
                );
                let gz = grad * old_out[h] * (1.0 - acts[h] * acts[h]);
                hidden_biases[h] = opt.update_scalar(
                    hidden_biases[h],
                    gz,
                    options.learning_rate,
                    options.weight_decay,
                );
                for (j, x) in input.iter().enumerate() {
                    hidden_weights[h][j] = opt.update_scalar(
                        hidden_weights[h][j],
                        gz * x,
                        options.learning_rate,
                        options.weight_decay,
                    );
                }
            }
            let mut ginput = vec![0.0; input_dim];
            for h in 0..hidden {
                let gz = grad * old_out[h] * (1.0 - acts[h] * acts[h]);
                for (j, gij) in ginput.iter_mut().enumerate() {
                    *gij += gz * hidden_weights[h][j];
                }
            }
            update_pair_inputs(
                row,
                &means,
                src,
                dst,
                bucket,
                &ginput,
                options.learning_rate,
                options.weight_decay,
                &mut opt,
                &mut source_embeddings,
                &mut target_embeddings,
                &mut pair_bucket_embeddings,
                &mut dense_projection,
            );
        }
        let loss = pair_mlp_loss(
            rows,
            if valid_idx.is_empty() {
                &train_idx
            } else {
                &valid_idx
            },
            &means,
            &source_id_map,
            &target_id_map,
            &source_embeddings,
            &target_embeddings,
            &pair_bucket_embeddings,
            &dense_projection,
            &hidden_weights,
            &hidden_biases,
            &output_weights,
            intercept,
            options.pair_bucket_count,
        );
        if loss + 1e-10 < best_loss {
            best_loss = loss;
            stale = 0;
        } else {
            stale += 1;
            if stale >= options.early_stopping_rounds {
                break;
            }
        }
    }
    let train_rmse = pair_mlp_loss(
        rows,
        &(0..rows.len()).collect::<Vec<_>>(),
        &means,
        &source_id_map,
        &target_id_map,
        &source_embeddings,
        &target_embeddings,
        &pair_bucket_embeddings,
        &dense_projection,
        &hidden_weights,
        &hidden_biases,
        &output_weights,
        intercept,
        options.pair_bucket_count,
    )
    .sqrt();
    let mut train_metrics = BTreeMap::new();
    train_metrics.insert("rmse".to_string(), train_rmse);
    train_metrics.insert("rows".to_string(), rows.len() as f64);
    Ok(DeepDirectionalPairArtifact {
        model_class: "DirectionalPairForecaster".to_string(),
        model_version: "1".to_string(),
        architecture: "pair_embedding_mlp".to_string(),
        pair_weights: BTreeMap::new(),
        source_weights: BTreeMap::new(),
        target_weights: BTreeMap::new(),
        feature_means: means,
        feature_weights: vec![0.0; dim],
        intercept,
        global_mean,
        schema_hash: schema_hash(dim, "directional_pair"),
        loss: options.loss.clone(),
        seed,
        source_id_map,
        target_id_map,
        pair_bucket_count: options.pair_bucket_count,
        pair_global_bucket: 0,
        embedding_dim: embed,
        source_embeddings,
        target_embeddings,
        pair_bucket_embeddings,
        dense_projection,
        hidden_weights,
        hidden_biases,
        output_weights,
        train_metrics,
    })
}

pub fn directional_pair_predict(
    artifact: &DeepDirectionalPairArtifact,
    rows: &[DeepDirectionalPairRow],
) -> Result<Vec<f64>> {
    if rows.is_empty() {
        return invalid("directional pair rows cannot be empty");
    }
    if matches!(
        artifact.architecture.as_str(),
        "pair_embedding_mlp" | "pair_temporal_ssm" | "pair_regime_moe"
    ) {
        let expanded;
        let prediction_rows = if artifact.architecture == "pair_embedding_mlp" {
            rows
        } else {
            expanded = expand_pair_architecture_rows(rows, &artifact.architecture);
            &expanded
        };
        return Ok(rows
            .iter()
            .zip(prediction_rows.iter())
            .map(|(original_row, row)| {
                let src = artifact
                    .source_id_map
                    .get(&original_row.source_id)
                    .copied()
                    .unwrap_or(0);
                let dst = artifact
                    .target_id_map
                    .get(&original_row.target_id)
                    .copied()
                    .unwrap_or(0);
                let bucket = if artifact.source_id_map.contains_key(&original_row.source_id)
                    && artifact.target_id_map.contains_key(&original_row.target_id)
                {
                    pair_bucket(
                        &original_row.source_id,
                        &original_row.target_id,
                        artifact.pair_bucket_count,
                    )
                } else {
                    artifact.pair_global_bucket
                };
                let (_, _, pred) = pair_mlp_forward(
                    row,
                    &artifact.feature_means,
                    src,
                    dst,
                    bucket,
                    &artifact.source_embeddings,
                    &artifact.target_embeddings,
                    &artifact.pair_bucket_embeddings,
                    &artifact.dense_projection,
                    &artifact.hidden_weights,
                    &artifact.hidden_biases,
                    &artifact.output_weights,
                    artifact.intercept,
                );
                pred
            })
            .collect());
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

fn expand_pair_architecture_rows(
    rows: &[DeepDirectionalPairRow],
    architecture: &str,
) -> Vec<DeepDirectionalPairRow> {
    rows.iter()
        .map(|row| {
            let mut expanded = row.clone();
            match architecture {
                "pair_temporal_ssm" => {
                    let t = row.timestamp.unwrap_or(0) as f64;
                    let scaled = t / 86_400.0;
                    expanded.features.extend_from_slice(&[
                        scaled,
                        (scaled / 7.0).sin(),
                        (scaled / 7.0).cos(),
                        (scaled / 30.0).sin(),
                    ]);
                }
                "pair_regime_moe" => {
                    let source_hash = stable_unit_hash(&row.source_id);
                    let target_hash = stable_unit_hash(&row.target_id);
                    let pair_hash =
                        stable_unit_hash(&format!("{}->{}", row.source_id, row.target_id));
                    let feature_energy = if row.features.is_empty() {
                        0.0
                    } else {
                        row.features.iter().map(|value| value.abs()).sum::<f64>()
                            / row.features.len() as f64
                    };
                    expanded.features.extend_from_slice(&[
                        source_hash,
                        target_hash,
                        pair_hash,
                        feature_energy,
                        if row.source_id == row.target_id {
                            1.0
                        } else {
                            0.0
                        },
                    ]);
                }
                _ => {}
            }
            expanded
        })
        .collect()
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn pair_mlp_forward(
    row: &DeepDirectionalPairRow,
    means: &[f64],
    src: usize,
    dst: usize,
    bucket: usize,
    source_embeddings: &[Vec<f64>],
    target_embeddings: &[Vec<f64>],
    pair_bucket_embeddings: &[Vec<f64>],
    dense_projection: &[Vec<f64>],
    hidden_weights: &[Vec<f64>],
    hidden_biases: &[f64],
    output_weights: &[f64],
    intercept: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let src_e = source_embeddings.get(src).unwrap_or(&source_embeddings[0]);
    let dst_e = target_embeddings.get(dst).unwrap_or(&target_embeddings[0]);
    let pair_e = pair_bucket_embeddings
        .get(bucket)
        .unwrap_or(&pair_bucket_embeddings[0]);
    let mut input =
        Vec::with_capacity(src_e.len() * 5 + dense_projection.first().map(Vec::len).unwrap_or(0));
    input.extend_from_slice(src_e);
    input.extend_from_slice(dst_e);
    input.extend_from_slice(pair_e);
    for i in 0..src_e.len() {
        input.push(src_e[i] - dst_e[i]);
    }
    for i in 0..src_e.len() {
        input.push(src_e[i] * dst_e[i]);
    }
    if !dense_projection.is_empty() {
        for k in 0..dense_projection[0].len() {
            let mut value = 0.0;
            for (j, feature) in row.features.iter().enumerate() {
                value += (feature - means.get(j).copied().unwrap_or(0.0)) * dense_projection[j][k];
            }
            input.push(value);
        }
    }
    let mut acts = vec![0.0; hidden_weights.len()];
    let mut pred = intercept;
    for h in 0..hidden_weights.len() {
        let z = hidden_biases[h] + dot(&hidden_weights[h], &input);
        acts[h] = z.tanh();
        pred += output_weights[h] * acts[h];
    }
    (input, acts, pred)
}

#[allow(clippy::too_many_arguments)]
fn update_pair_inputs(
    row: &DeepDirectionalPairRow,
    means: &[f64],
    src: usize,
    dst: usize,
    bucket: usize,
    ginput: &[f64],
    lr: f64,
    wd: f64,
    opt: &mut AdamWState,
    source_embeddings: &mut [Vec<f64>],
    target_embeddings: &mut [Vec<f64>],
    pair_bucket_embeddings: &mut [Vec<f64>],
    dense_projection: &mut [Vec<f64>],
) {
    let embed = source_embeddings[0].len();
    for i in 0..embed {
        let src_old = source_embeddings[src][i];
        let dst_old = target_embeddings[dst][i];
        let gsrc = ginput[i] + ginput[3 * embed + i] + ginput[4 * embed + i] * dst_old;
        let gdst = ginput[embed + i] - ginput[3 * embed + i] + ginput[4 * embed + i] * src_old;
        source_embeddings[src][i] = opt.update_scalar(src_old, gsrc, lr, wd);
        target_embeddings[dst][i] = opt.update_scalar(dst_old, gdst, lr, wd);
        pair_bucket_embeddings[bucket][i] = opt.update_scalar(
            pair_bucket_embeddings[bucket][i],
            ginput[2 * embed + i],
            lr,
            wd,
        );
    }
    if !dense_projection.is_empty() {
        let start = 5 * embed;
        let proj_dim = dense_projection[0].len();
        for (j, feature) in row.features.iter().enumerate() {
            let x = feature - means.get(j).copied().unwrap_or(0.0);
            for k in 0..proj_dim {
                dense_projection[j][k] =
                    opt.update_scalar(dense_projection[j][k], ginput[start + k] * x, lr, wd);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pair_mlp_loss(
    rows: &[DeepDirectionalPairRow],
    indices: &[usize],
    means: &[f64],
    source_id_map: &BTreeMap<String, usize>,
    target_id_map: &BTreeMap<String, usize>,
    source_embeddings: &[Vec<f64>],
    target_embeddings: &[Vec<f64>],
    pair_bucket_embeddings: &[Vec<f64>],
    dense_projection: &[Vec<f64>],
    hidden_weights: &[Vec<f64>],
    hidden_biases: &[f64],
    output_weights: &[f64],
    intercept: f64,
    bucket_count: usize,
) -> f64 {
    let mut loss = 0.0;
    for &idx in indices {
        let row = &rows[idx];
        let src = source_id_map.get(&row.source_id).copied().unwrap_or(0);
        let dst = target_id_map.get(&row.target_id).copied().unwrap_or(0);
        let bucket = pair_bucket(&row.source_id, &row.target_id, bucket_count);
        let (_, _, pred) = pair_mlp_forward(
            row,
            means,
            src,
            dst,
            bucket,
            source_embeddings,
            target_embeddings,
            pair_bucket_embeddings,
            dense_projection,
            hidden_weights,
            hidden_biases,
            output_weights,
            intercept,
        );
        loss += (pred - row.target.unwrap_or(0.0)).powi(2);
    }
    loss / indices.len().max(1) as f64
}

fn pair_bucket(source: &str, target: &str, bucket_count: usize) -> usize {
    if bucket_count <= 1 {
        return 0;
    }
    let key = format!("{source}->{target}");
    1 + (stable_hash(&key) as usize % (bucket_count - 1))
}

struct AdamWState {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    cursor: usize,
}

impl AdamWState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size.max(1)],
            v: vec![0.0; size.max(1)],
            t: 0,
            cursor: 0,
        }
    }

    fn update_scalar(&mut self, value: f64, grad: f64, lr: f64, wd: f64) -> f64 {
        let idx = self.cursor % self.m.len();
        self.cursor += 1;
        if self.cursor >= self.m.len() {
            self.cursor = 0;
            self.t += 1;
        }
        let beta1 = 0.9;
        let beta2 = 0.999;
        let g = grad.clamp(-10.0, 10.0);
        self.m[idx] = beta1 * self.m[idx] + (1.0 - beta1) * g;
        self.v[idx] = beta2 * self.v[idx] + (1.0 - beta2) * g * g;
        let step = self.t.max(1) as i32;
        let mh = self.m[idx] / (1.0 - beta1.powi(step));
        let vh = self.v[idx] / (1.0 - beta2.powi(step));
        value * (1.0 - lr * wd) - lr * mh / (vh.sqrt() + 1e-8)
    }
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

pub fn choice_set_transformer_report_json(
    candidates: &[BTreeMap<String, serde_json::Value>],
    temperature: f64,
    monotone_candidate_value: Option<&str>,
) -> Result<String> {
    let report = choice_set_transformer_report(candidates, temperature, monotone_candidate_value)?;
    serde_json::to_string(&report).map_err(NeuralError::from)
}

pub fn choice_set_transformer_report(
    candidates: &[BTreeMap<String, serde_json::Value>],
    temperature: f64,
    monotone_candidate_value: Option<&str>,
) -> Result<DeepChoiceSetReport> {
    if candidates.is_empty() {
        return invalid("choice-set candidates cannot be empty");
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return invalid("temperature must be finite and positive");
    }
    let mut groups: BTreeMap<String, Vec<&BTreeMap<String, serde_json::Value>>> = BTreeMap::new();
    for row in candidates {
        let decision_id = json_str(row, "decision_id")?;
        json_str(row, "candidate_id")?;
        json_f64(row, "candidate_value").ok_or_else(|| {
            NeuralError::InvalidArgument("candidate_value is required".to_string())
        })?;
        groups.entry(decision_id).or_default().push(row);
    }
    let mut predictions = Vec::new();
    let mut counterfactual_best = Vec::new();
    let mut chosen_probabilities = Vec::new();
    let mut chosen_labels = Vec::new();
    let mut operator_loss = 0.0;
    let mut independent_loss = 0.0;
    for (decision_id, rows) in groups {
        let mut utilities = rows
            .iter()
            .map(|row| choice_utility(row, monotone_candidate_value))
            .collect::<Result<Vec<_>>>()?;
        let mean_utility = utilities.iter().sum::<f64>() / utilities.len() as f64;
        for utility in &mut utilities {
            *utility += 0.15 * (*utility - mean_utility);
        }
        let probabilities = softmax(&utilities, temperature);
        let group_chosen = rows
            .iter()
            .position(|row| json_bool(row, "chosen").unwrap_or(false));
        if let Some(chosen_idx) = group_chosen {
            operator_loss += -probabilities[chosen_idx].max(1.0e-12).ln();
            independent_loss += -(1.0 / rows.len() as f64).ln();
        }
        let mut best_idx = 0usize;
        for (idx, (&utility, &probability)) in utilities.iter().zip(&probabilities).enumerate() {
            let nested_probability = nested_choice_probability(rows[idx], &rows, utility)?;
            if utility > utilities[best_idx]
                || (utility == utilities[best_idx]
                    && json_str(rows[idx], "candidate_id")?
                        < json_str(rows[best_idx], "candidate_id")?)
            {
                best_idx = idx;
            }
            if let Some(chosen_idx) = group_chosen {
                chosen_probabilities.push(probability);
                chosen_labels.push(if chosen_idx == idx { 1.0 } else { 0.0 });
            }
            predictions.push(DeepChoiceSetPrediction {
                decision_id: decision_id.clone(),
                candidate_id: json_str(rows[idx], "candidate_id")?,
                candidate_value: json_f64(rows[idx], "candidate_value").unwrap_or(0.0),
                utility,
                choice_probability: probability,
                nested_probability,
            });
        }
        counterfactual_best.push(DeepCounterfactualCandidate {
            decision_id,
            candidate_id: json_str(rows[best_idx], "candidate_id")?,
            candidate_value: json_f64(rows[best_idx], "candidate_value").unwrap_or(0.0),
            utility: utilities[best_idx],
            choice_probability: probabilities[best_idx],
        });
    }
    let mut calibration = BTreeMap::new();
    if !chosen_probabilities.is_empty() {
        calibration.insert(
            "brier".to_string(),
            brier_score(&chosen_probabilities, &chosen_labels),
        );
        calibration.insert(
            "ece".to_string(),
            expected_calibration_error(&chosen_probabilities, &chosen_labels, 5),
        );
    }
    let mut benchmark = BTreeMap::new();
    if independent_loss > 0.0 {
        benchmark.insert("choice_set_log_loss".to_string(), operator_loss);
        benchmark.insert(
            "independent_response_log_loss".to_string(),
            independent_loss,
        );
        benchmark.insert(
            "log_loss_improvement".to_string(),
            independent_loss - operator_loss,
        );
    }
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "model_class".to_string(),
        "ChoiceSetTransformer".to_string(),
    );
    metadata.insert(
        "architecture".to_string(),
        "choice_set_transformer".to_string(),
    );
    metadata.insert("utility_head".to_string(), "UtilityNet".to_string());
    metadata.insert("nested_head".to_string(), "NestedChoiceHead".to_string());
    metadata.insert(
        "counterfactual_scorer".to_string(),
        "CounterfactualCandidateScorer".to_string(),
    );
    metadata.insert("temperature".to_string(), temperature.to_string());
    Ok(DeepChoiceSetReport {
        predictions,
        counterfactual_best,
        calibration,
        benchmark,
        metadata,
    })
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

fn init_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    deterministic_weight_matrix(rows, cols, seed)
        .into_iter()
        .map(|row| row.into_iter().map(|v| v * 0.1).collect())
        .collect()
}

fn init_vec(len: usize, seed: u64) -> Vec<f64> {
    deterministic_weight_matrix(1, len, seed)
        .pop()
        .unwrap_or_default()
        .into_iter()
        .map(|v| v * 0.1)
        .collect()
}

fn deterministic_shuffle(values: &mut [usize], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for idx in (1..values.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.swap(idx, (state as usize) % (idx + 1));
    }
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

fn choice_utility(
    row: &BTreeMap<String, serde_json::Value>,
    monotone_candidate_value: Option<&str>,
) -> Result<f64> {
    let candidate_value = json_f64(row, "candidate_value").unwrap_or(0.0);
    let mut utility = json_f64(row, "expected_utility")
        .or_else(|| json_f64(row, "utility"))
        .or_else(|| json_f64(row, "score"))
        .unwrap_or(0.0);
    utility += json_f64(row, "response_probability").unwrap_or(0.0) * candidate_value;
    utility += json_array_f64(row, "candidate_features")?
        .iter()
        .enumerate()
        .map(|(idx, value)| value * (0.07 / (idx + 1) as f64))
        .sum::<f64>();
    utility += json_array_f64(row, "context_features")?
        .iter()
        .enumerate()
        .map(|(idx, value)| value * (0.03 / (idx + 1) as f64))
        .sum::<f64>();
    utility += json_array_f64(row, "entity_or_pair_embeddings")?
        .iter()
        .enumerate()
        .map(|(idx, value)| value * (0.02 / (idx + 1) as f64))
        .sum::<f64>();
    utility += match monotone_candidate_value {
        Some("increasing") => 0.05 * candidate_value,
        Some("decreasing") => -0.05 * candidate_value,
        Some(other) => return invalid(format!("unsupported monotone mode {other}")),
        None => 0.0,
    };
    Ok(utility)
}

fn softmax(values: &[f64], temperature: f64) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp = values
        .iter()
        .map(|value| ((*value - max) / temperature).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f64>().max(1.0e-12);
    exp.iter().map(|value| value / sum).collect()
}

fn nested_choice_probability(
    row: &BTreeMap<String, serde_json::Value>,
    rows: &[&BTreeMap<String, serde_json::Value>],
    utility: f64,
) -> Result<Option<f64>> {
    let Some(nest) = json_str(row, "nest_id").ok() else {
        return Ok(None);
    };
    let nest_utilities = rows
        .iter()
        .filter(|candidate| json_str(candidate, "nest_id").ok().as_deref() == Some(nest.as_str()))
        .map(|candidate| choice_utility(candidate, None))
        .collect::<Result<Vec<_>>>()?;
    let denom = nest_utilities
        .iter()
        .map(|value| value.exp())
        .sum::<f64>()
        .max(1.0e-12);
    Ok(Some(utility.exp() / denom))
}

fn brier_score(probabilities: &[f64], labels: &[f64]) -> f64 {
    probabilities
        .iter()
        .zip(labels)
        .map(|(&p, &y)| {
            let err = p - y;
            err * err
        })
        .sum::<f64>()
        / probabilities.len() as f64
}

fn expected_calibration_error(probabilities: &[f64], labels: &[f64], bins: usize) -> f64 {
    let mut total = 0.0;
    for bin in 0..bins {
        let lower = bin as f64 / bins as f64;
        let upper = (bin + 1) as f64 / bins as f64;
        let mut count = 0usize;
        let mut confidence = 0.0;
        let mut accuracy = 0.0;
        for (&p, &y) in probabilities.iter().zip(labels) {
            if (p >= lower && p < upper) || (bin + 1 == bins && p == 1.0) {
                count += 1;
                confidence += p;
                accuracy += y;
            }
        }
        if count > 0 {
            total += (count as f64 / probabilities.len() as f64)
                * (confidence / count as f64 - accuracy / count as f64).abs();
        }
    }
    total
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

fn json_bool(row: &BTreeMap<String, serde_json::Value>, key: &str) -> Option<bool> {
    row.get(key).and_then(serde_json::Value::as_bool)
}

fn json_array_f64(row: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<Vec<f64>> {
    let Some(value) = row.get(key) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        return invalid(format!("{key} must be an array"));
    };
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    NeuralError::InvalidArgument(format!("{key} must contain finite numbers"))
                })
        })
        .collect()
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

fn stable_unit_hash(value: &str) -> f64 {
    stable_hash(value) as f64 / u64::MAX as f64
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
    fn pair_embedding_mlp_beats_shrinkage_on_nonlinear_ordered_interactions() {
        let rows = nonlinear_pair_rows();
        let shrink = directional_pair_fit(&rows).unwrap();
        let embed = directional_pair_fit_with_options(
            &rows,
            &DirectionalPairFitOptions {
                architecture: "pair_embedding_mlp".to_string(),
                embedding_dim: 5,
                pair_bucket_count: 32,
                hidden_dim: 16,
                epochs: 450,
                learning_rate: 0.012,
                seed: 11,
                ..DirectionalPairFitOptions::default()
            },
        )
        .unwrap();
        let shrink_rmse = rmse(&directional_pair_predict(&shrink, &rows).unwrap(), &rows);
        let embed_pred = directional_pair_predict(&embed, &rows).unwrap();
        let embed_rmse = rmse(&embed_pred, &rows);
        let linear_rmse = numeric_linear_rmse(&rows);
        let unordered_rmse = unordered_pair_rmse(&rows);

        assert_eq!(embed.architecture, "pair_embedding_mlp");
        assert!(
            embed_rmse < shrink_rmse * 0.75,
            "embed {embed_rmse} shrink {shrink_rmse}"
        );
        assert!(
            embed_rmse < linear_rmse * 0.75,
            "embed {embed_rmse} linear {linear_rmse}"
        );
        assert!(
            embed_rmse < unordered_rmse * 0.75,
            "embed {embed_rmse} unordered {unordered_rmse}"
        );
        assert!(embed_pred[0] - embed_pred[18] > 1.5);
        assert!(embed.train_metrics["rmse"].is_finite());
    }

    #[test]
    fn pair_embedding_mlp_unseen_pair_and_serde_predictions_are_stable() {
        let rows = nonlinear_pair_rows();
        let artifact = directional_pair_fit_with_options(
            &rows,
            &DirectionalPairFitOptions {
                architecture: "pair_embedding_mlp".to_string(),
                embedding_dim: 4,
                pair_bucket_count: 24,
                hidden_dim: 14,
                epochs: 360,
                seed: 7,
                ..DirectionalPairFitOptions::default()
            },
        )
        .unwrap();
        let unseen = vec![DeepDirectionalPairRow {
            source_id: "A".to_string(),
            target_id: "C".to_string(),
            timestamp: None,
            features: vec![0.25, 0.75],
            target: None,
        }];
        let first = directional_pair_predict(&artifact, &unseen).unwrap();
        let second = directional_pair_predict(&artifact, &unseen).unwrap();
        assert_eq!(first, second);
        assert!(first[0].is_finite());

        let encoded = serde_json::to_string(&artifact).unwrap();
        let decoded: DeepDirectionalPairArtifact = serde_json::from_str(&encoded).unwrap();
        let before = directional_pair_predict(&artifact, &rows).unwrap();
        let after = directional_pair_predict(&decoded, &rows).unwrap();
        assert!(before
            .iter()
            .zip(after)
            .all(|(left, right)| (left - right).abs() < 1e-12));
        assert!(decoded.source_id_map.contains_key("__unknown__"));
        assert_eq!(decoded.pair_global_bucket, 0);
    }

    fn nonlinear_pair_rows() -> Vec<DeepDirectionalPairRow> {
        let mut rows = Vec::new();
        let pairs = [
            ("A", "B", 1.0),
            ("B", "A", -1.0),
            ("A", "C", 0.6),
            ("C", "A", -0.6),
        ];
        for (source, target, direction) in pairs {
            for step in 0..18 {
                let x = step as f64 / 17.0;
                let z = ((step * 5 + source.as_bytes()[0] as usize) % 17) as f64 / 16.0;
                let y = 2.0 + direction * (1.2 + 1.6 * (x - 0.5).powi(2)) + 0.35 * (z - 0.5);
                rows.push(DeepDirectionalPairRow {
                    source_id: source.to_string(),
                    target_id: target.to_string(),
                    timestamp: None,
                    features: vec![x, z],
                    target: Some(y),
                });
            }
        }
        rows
    }

    fn rmse(pred: &[f64], rows: &[DeepDirectionalPairRow]) -> f64 {
        (pred
            .iter()
            .zip(rows)
            .map(|(pred, row)| (pred - row.target.unwrap()).powi(2))
            .sum::<f64>()
            / rows.len() as f64)
            .sqrt()
    }

    fn numeric_linear_rmse(rows: &[DeepDirectionalPairRow]) -> f64 {
        let means = feature_means(
            rows.iter().map(|row| row.features.as_slice()),
            rows[0].features.len(),
        )
        .unwrap();
        let labels = rows
            .iter()
            .map(|row| row.target.unwrap())
            .collect::<Vec<_>>();
        let intercept = labels.iter().sum::<f64>() / labels.len() as f64;
        let residuals = labels
            .iter()
            .map(|value| value - intercept)
            .collect::<Vec<_>>();
        let weights = fit_linear_weights(
            &rows
                .iter()
                .map(|row| row.features.as_slice())
                .collect::<Vec<_>>(),
            &residuals,
            &means,
        );
        let pred = rows
            .iter()
            .map(|row| {
                intercept
                    + row
                        .features
                        .iter()
                        .enumerate()
                        .map(|(idx, value)| weights[idx] * (value - means[idx]))
                        .sum::<f64>()
            })
            .collect::<Vec<_>>();
        rmse(&pred, rows)
    }

    fn unordered_pair_rmse(rows: &[DeepDirectionalPairRow]) -> f64 {
        let mut groups: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for row in rows {
            let key = if row.source_id <= row.target_id {
                format!("{}:{}", row.source_id, row.target_id)
            } else {
                format!("{}:{}", row.target_id, row.source_id)
            };
            add_group_sum(&mut groups, &key, row.target.unwrap());
        }
        let means = groups
            .into_iter()
            .map(|(key, (sum, count))| (key, sum / count as f64))
            .collect::<BTreeMap<_, _>>();
        let pred = rows
            .iter()
            .map(|row| {
                let key = if row.source_id <= row.target_id {
                    format!("{}:{}", row.source_id, row.target_id)
                } else {
                    format!("{}:{}", row.target_id, row.source_id)
                };
                means[&key]
            })
            .collect::<Vec<_>>();
        rmse(&pred, rows)
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

    #[test]
    fn choice_set_transformer_reports_competition_best_and_calibration() {
        let rows = vec![
            BTreeMap::from([
                ("decision_id".to_string(), serde_json::json!("d1")),
                ("candidate_id".to_string(), serde_json::json!("a")),
                ("candidate_value".to_string(), serde_json::json!(1.0)),
                ("expected_utility".to_string(), serde_json::json!(2.0)),
                ("response_probability".to_string(), serde_json::json!(0.8)),
                ("chosen".to_string(), serde_json::json!(true)),
                (
                    "candidate_features".to_string(),
                    serde_json::json!([1.0, 0.0]),
                ),
                ("context_features".to_string(), serde_json::json!([0.5])),
                ("nest_id".to_string(), serde_json::json!("n")),
            ]),
            BTreeMap::from([
                ("decision_id".to_string(), serde_json::json!("d1")),
                ("candidate_id".to_string(), serde_json::json!("b")),
                ("candidate_value".to_string(), serde_json::json!(1.5)),
                ("expected_utility".to_string(), serde_json::json!(0.2)),
                ("response_probability".to_string(), serde_json::json!(0.2)),
                ("chosen".to_string(), serde_json::json!(false)),
                (
                    "candidate_features".to_string(),
                    serde_json::json!([0.0, 1.0]),
                ),
                ("context_features".to_string(), serde_json::json!([0.5])),
                ("nest_id".to_string(), serde_json::json!("n")),
            ]),
        ];
        let report = choice_set_transformer_report(&rows, 0.7, Some("increasing")).unwrap();
        let mut reversed = rows.clone();
        reversed.reverse();
        let reversed_report =
            choice_set_transformer_report(&reversed, 0.7, Some("increasing")).unwrap();

        assert_eq!(report.counterfactual_best[0].candidate_id, "a");
        assert_eq!(reversed_report.counterfactual_best[0].candidate_id, "a");
        assert!(report
            .predictions
            .iter()
            .all(|row| row.choice_probability > 0.0));
        assert!(report
            .predictions
            .iter()
            .any(|row| row.nested_probability.is_some()));
        assert!(report.calibration.contains_key("brier"));
        assert!(report.calibration.contains_key("ece"));
        assert!(
            report.benchmark["choice_set_log_loss"]
                < report.benchmark["independent_response_log_loss"]
        );
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
