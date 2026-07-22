#[cfg(all(feature = "cuda", target_os = "linux"))]
use cartoboost_neural::CudaTensorArena;
use cartoboost_neural::{
    backend_adamw_step_f32, backend_affine_scores, backend_csr_diffusion_f32,
    backend_csr_row_softmax_f32, backend_dense_layer_f32, backend_layer_norm_f32,
    backend_scalar_graph_f32, backend_scalar_graph_train_step_f32, select_backend_for_operations,
    BackendOperation, BackendSelection,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

const GEO_ST_AFFINE_DISPATCH_MIN_OPS: usize = 16_384;
#[cfg(not(test))]
const GEO_ST_CSR_DISPATCH_MIN_OPS: usize = 16_384;
#[cfg(test)]
const GEO_ST_CSR_DISPATCH_MIN_OPS: usize = 2_048;
const GEO_ST_DENSE_DISPATCH_MIN_OPS: usize = 16_384;
const GEO_ST_ADAMW_DISPATCH_MIN_VALUES: usize = 16_384;

pub mod market;
pub use market::{
    ExpertEventLabel, ExpertRelationshipPrior, MarketExplanation, MarketPanelFrame,
    MarketPrediction, MarketRelationship, MarketShiftKind, MarketStructureConfig,
    MarketStructureForecaster, MarketSupportKind, RelationshipKind, WeeklyMarketPrediction,
};

pub type Result<T> = std::result::Result<T, GeoStError>;

fn unit_scale() -> f64 {
    1.0
}

type GraphForwardOutput = (
    AutodiffTape,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
);
type AcceleratorGraphArrays = (Vec<f32>, Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>);

struct GraphForwardContext<'a> {
    profile: &'a GraphTransformerProfile,
    window: &'a [Vec<f64>],
    adjacency: &'a CsrAdjacency,
    excluded_expert: Option<usize>,
    phase_offset: usize,
    long_context_is_pooled: bool,
    lsttn_frozen_patches: Option<&'a [Vec<Vec<f32>>]>,
    lsttn_time_features: Option<&'a [Vec<f64>]>,
    deferred: bool,
    training: bool,
}

pub fn available_compute_backends() -> Vec<String> {
    cartoboost_neural::available_backends()
}

pub fn select_compute_backend(requested: Option<&str>) -> Result<ComputeBackendSelection> {
    // Keep graph/spatiotemporal models on the same parser, aliases, runtime
    // probes, and auto-selection policy as every other accelerated model.
    let neural = cartoboost_neural::select_backend(requested)
        .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
    Ok(ComputeBackendSelection {
        requested: neural.requested,
        selected: neural.selected,
        available: neural.available,
    })
}

pub fn select_compute_backend_for_operations(
    requested: Option<&str>,
    operations: &[BackendOperation],
) -> Result<ComputeBackendSelection> {
    let requested_name = requested.unwrap_or("cpu");
    let neural = if requested_name.eq_ignore_ascii_case("cpu") {
        cartoboost_neural::select_backend(Some("cpu"))
    } else {
        select_backend_for_operations(Some(requested_name), operations).or_else(|error| {
            if requested_name.eq_ignore_ascii_case("auto") {
                cartoboost_neural::select_backend(Some("cpu"))
            } else {
                Err(error)
            }
        })
    }
    .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
    Ok(ComputeBackendSelection {
        requested: neural.requested,
        selected: neural.selected,
        available: neural.available,
    })
}

fn profitable_affine_backend(
    backend: BackendSelection,
    row_count: usize,
    feature_count: usize,
) -> Result<BackendSelection> {
    if backend.selected != "cpu"
        && row_count.saturating_mul(feature_count) < GEO_ST_AFFINE_DISPATCH_MIN_OPS
    {
        return cartoboost_neural::select_backend_for(Some("cpu"), BackendOperation::Affine)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()));
    }
    Ok(backend)
}

#[derive(Debug, thiserror::Error)]
pub enum GeoStError {
    #[error("invalid graph temporal frame: {0}")]
    InvalidFrame(String),
    #[error("model must be fit before prediction")]
    NotFit,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid compute backend: {0}")]
    InvalidBackend(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComputeBackendSelection {
    pub requested: String,
    pub selected: String,
    pub available: Vec<String>,
}

impl Default for ComputeBackendSelection {
    fn default() -> Self {
        select_compute_backend(None).expect("CPU backend is always available")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CsrAdjacency {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl CsrAdjacency {
    pub fn new(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        node_count: usize,
    ) -> Result<Self> {
        if indptr.len() != node_count + 1 {
            return Err(GeoStError::InvalidFrame(
                "CSR indptr length must equal node_count + 1".to_string(),
            ));
        }
        if indices.len() != data.len() {
            return Err(GeoStError::InvalidFrame(
                "CSR indices and weights must have the same length".to_string(),
            ));
        }
        if indptr.first() != Some(&0) || indptr.last() != Some(&indices.len()) {
            return Err(GeoStError::InvalidFrame(
                "CSR indptr must start at 0 and end at the edge count".to_string(),
            ));
        }
        if indptr.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(GeoStError::InvalidFrame(
                "CSR indptr must be nondecreasing".to_string(),
            ));
        }
        if indices.iter().any(|&idx| idx >= node_count) {
            return Err(GeoStError::InvalidFrame(
                "CSR edge index exceeds node count".to_string(),
            ));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(GeoStError::InvalidFrame(
                "CSR weights must be finite".to_string(),
            ));
        }
        Ok(Self {
            indptr,
            indices,
            data,
        })
    }

    pub fn row_normalized(&self) -> Self {
        let mut data = self.data.clone();
        for row in 0..self.indptr.len() - 1 {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let sum: f64 = data[start..end].iter().map(|v| v.abs()).sum();
            if sum > 0.0 {
                for value in &mut data[start..end] {
                    *value /= sum;
                }
            }
        }
        Self {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data,
        }
    }

    pub fn transpose(&self, node_count: usize) -> Self {
        let mut counts = vec![0usize; node_count];
        for &col in &self.indices {
            counts[col] += 1;
        }
        let mut indptr = vec![0usize; node_count + 1];
        for idx in 0..node_count {
            indptr[idx + 1] = indptr[idx] + counts[idx];
        }
        let mut next = indptr.clone();
        let mut indices = vec![0usize; self.indices.len()];
        let mut data = vec![0.0; self.data.len()];
        for row in 0..node_count {
            for edge in self.indptr[row]..self.indptr[row + 1] {
                let col = self.indices[edge];
                let slot = next[col];
                indices[slot] = row;
                data[slot] = self.data[edge];
                next[col] += 1;
            }
        }
        Self {
            indptr,
            indices,
            data,
        }
        .row_normalized()
    }

    /// Adds at most one self candidate per row for learned adaptive graph
    /// attention. This is deliberately separate from structural diffusion:
    /// forward/backward CSR propagation retains its explicit zero behavior
    /// for isolated nodes, while adaptive softmax always has a differentiable
    /// local candidate without allocating a dense adjacency matrix.
    fn with_adaptive_self_candidates(&self, node_count: usize) -> Self {
        let mut indptr = Vec::with_capacity(node_count + 1);
        let mut indices = Vec::with_capacity(self.indices.len() + node_count);
        let mut data = Vec::with_capacity(self.data.len() + node_count);
        indptr.push(0);
        for row in 0..node_count {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let has_self = self.indices[start..end].contains(&row);
            indices.extend_from_slice(&self.indices[start..end]);
            data.extend_from_slice(&self.data[start..end]);
            if !has_self {
                indices.push(row);
                data.push(1.0);
            }
            indptr.push(indices.len());
        }
        Self {
            indptr,
            indices,
            data,
        }
    }

    fn matvec(&self, input: &[f64], output: &mut [f64]) {
        output.fill(0.0);
        for (row, value) in output.iter_mut().enumerate() {
            for edge in self.indptr[row]..self.indptr[row + 1] {
                *value += self.data[edge] * input[self.indices[edge]];
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphTemporalFrame {
    pub node_ids: Vec<String>,
    pub timestamps: Vec<i64>,
    pub target: Vec<Vec<f64>>,
    pub covariates: Option<Vec<Vec<Vec<f64>>>>,
    pub adjacency: CsrAdjacency,
    pub horizon: usize,
    pub frequency: String,
    #[serde(default)]
    pub owner_mask: Option<Vec<bool>>,
    /// Explicit observation status; `target` values are transport values and
    /// may legitimately be zero when this mask is true.
    #[serde(default)]
    pub target_mask: Option<Vec<Vec<bool>>>,
    /// Values filled by an upstream imputer. These remain available as model
    /// inputs, but do not become supervised labels.
    #[serde(default)]
    pub imputed_mask: Option<Vec<Vec<bool>>>,
    /// Optional per-label reliability weights, aligned with `target`.
    #[serde(default)]
    pub target_weights: Option<Vec<Vec<f64>>>,
    /// Declared covariate channel roles. LSTTN requires a `time_of_day` role
    /// whenever covariates are supplied.
    #[serde(default)]
    pub covariate_roles: Option<Vec<String>>,
}

impl GraphTemporalFrame {
    pub fn new(
        node_ids: Vec<String>,
        timestamps: Vec<i64>,
        target: Vec<Vec<f64>>,
        covariates: Option<Vec<Vec<Vec<f64>>>>,
        adjacency: CsrAdjacency,
        horizon: usize,
        frequency: String,
    ) -> Result<Self> {
        Self::new_with_training_metadata(
            node_ids, timestamps, target, covariates, adjacency, horizon, frequency, None, None,
            None, None, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_training_metadata(
        node_ids: Vec<String>,
        timestamps: Vec<i64>,
        target: Vec<Vec<f64>>,
        covariates: Option<Vec<Vec<Vec<f64>>>>,
        adjacency: CsrAdjacency,
        horizon: usize,
        frequency: String,
        owner_mask: Option<Vec<bool>>,
        target_mask: Option<Vec<Vec<bool>>>,
        imputed_mask: Option<Vec<Vec<bool>>>,
        target_weights: Option<Vec<Vec<f64>>>,
        covariate_roles: Option<Vec<String>>,
    ) -> Result<Self> {
        let frame = Self {
            node_ids,
            timestamps,
            target,
            covariates,
            adjacency,
            horizon,
            frequency,
            owner_mask,
            target_mask,
            imputed_mask,
            target_weights,
            covariate_roles,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn with_owner_mask(mut self, owner_mask: Option<Vec<bool>>) -> Result<Self> {
        self.owner_mask = owner_mask;
        self.validate()?;
        Ok(self)
    }

    pub fn with_training_metadata(
        mut self,
        target_mask: Option<Vec<Vec<bool>>>,
        imputed_mask: Option<Vec<Vec<bool>>>,
        target_weights: Option<Vec<Vec<f64>>>,
        covariate_roles: Option<Vec<String>>,
    ) -> Result<Self> {
        self.target_mask = target_mask;
        self.imputed_mask = imputed_mask;
        self.target_weights = target_weights;
        self.covariate_roles = covariate_roles;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        let nodes = self.node_ids.len();
        if nodes == 0 {
            return Err(GeoStError::InvalidFrame(
                "node ids cannot be empty".to_string(),
            ));
        }
        if self
            .owner_mask
            .as_ref()
            .is_some_and(|mask| mask.len() != nodes || !mask.iter().any(|value| *value))
        {
            return Err(GeoStError::InvalidFrame(
                "owner mask must match node ids and select at least one owner".to_string(),
            ));
        }
        if self.timestamps.len() != self.target.len() {
            return Err(GeoStError::InvalidFrame(
                "timestamps and target must have the same length".to_string(),
            ));
        }
        if self.horizon == 0 || self.target.len() <= self.horizon {
            return Err(GeoStError::InvalidFrame(
                "target length must exceed a positive horizon".to_string(),
            ));
        }
        let validate_mask = |name: &str, mask: &Option<Vec<Vec<bool>>>| -> Result<()> {
            if mask.as_ref().is_some_and(|rows| {
                rows.len() != self.target.len() || rows.iter().any(|row| row.len() != nodes)
            }) {
                return Err(GeoStError::InvalidFrame(format!(
                    "{name} must have shape [time, node]"
                )));
            }
            Ok(())
        };
        validate_mask("target mask", &self.target_mask)?;
        validate_mask("imputed mask", &self.imputed_mask)?;
        if self.target_weights.as_ref().is_some_and(|rows| {
            rows.len() != self.target.len()
                || rows.iter().any(|row| {
                    row.len() != nodes || row.iter().any(|value| !value.is_finite() || *value < 0.0)
                })
        }) {
            return Err(GeoStError::InvalidFrame(
                "target weights must be finite, non-negative, and have shape [time, node]"
                    .to_string(),
            ));
        }
        for (time, row) in self.target.iter().enumerate() {
            if row.len() != nodes
                || row.iter().enumerate().any(|(node, value)| {
                    self.target_mask
                        .as_ref()
                        .is_none_or(|mask| mask[time][node])
                        && !value.is_finite()
                })
            {
                return Err(GeoStError::InvalidFrame(
                    "active target values must be finite with shape [time, node]".to_string(),
                ));
            }
        }
        CsrAdjacency::new(
            self.adjacency.indptr.clone(),
            self.adjacency.indices.clone(),
            self.adjacency.data.clone(),
            nodes,
        )?;
        if let Some(covariates) = &self.covariates {
            if covariates.len() != self.target.len()
                || covariates.iter().any(|time_row| time_row.len() != nodes)
            {
                return Err(GeoStError::InvalidFrame(
                    "covariates must have shape [time, node, feature]".to_string(),
                ));
            }
            let feature_count = covariates[0][0].len();
            if feature_count == 0
                || covariates.iter().flatten().any(|features| {
                    features.len() != feature_count
                        || features.iter().any(|value| !value.is_finite())
                })
            {
                return Err(GeoStError::InvalidFrame(
                    "covariates must have a non-empty, finite, consistent feature axis".to_string(),
                ));
            }
            if let Some(roles) = &self.covariate_roles {
                if roles.len() != feature_count || roles.iter().any(|role| role.trim().is_empty()) {
                    return Err(GeoStError::InvalidFrame(
                        "covariate roles must name every covariate channel".to_string(),
                    ));
                }
            }
        } else if self.covariate_roles.is_some() {
            return Err(GeoStError::InvalidFrame(
                "covariate roles require covariates".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DcrnnConfig {
    pub diffusion_steps: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub teacher_forcing_start: f64,
    pub teacher_forcing_end: f64,
    pub ridge: f64,
    #[serde(default)]
    pub backend: ComputeBackendSelection,
}

impl Default for DcrnnConfig {
    fn default() -> Self {
        Self {
            diffusion_steps: 2,
            hidden_size: 8,
            epochs: 160,
            learning_rate: 0.03,
            teacher_forcing_start: 1.0,
            teacher_forcing_end: 0.2,
            ridge: 1e-4,
            backend: select_compute_backend(None).expect("default CPU backend is always available"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HorizonMetric {
    pub horizon: usize,
    pub mae: f64,
    pub rmse: f64,
    pub wape: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeMetric {
    pub node_id: String,
    pub mae: f64,
    pub rmse: f64,
    pub wape: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphDistanceResidual {
    pub distance: usize,
    pub mean_abs_residual: f64,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphForecastMetrics {
    pub by_horizon: Vec<HorizonMetric>,
    pub by_node: Vec<NodeMetric>,
    pub graph_distance_residuals: Vec<GraphDistanceResidual>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DcrnnForecaster {
    pub config: DcrnnConfig,
    node_ids: Vec<String>,
    frequency: String,
    horizon: usize,
    adjacency: Option<CsrAdjacency>,
    reverse_adjacency: Option<CsrAdjacency>,
    weights: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    encoder_weights: Vec<Vec<f64>>,
    recurrent_weights: Vec<Vec<f64>>,
    history: Vec<Vec<f64>>,
    target_mean: f64,
    target_scale: f64,
}

impl DcrnnForecaster {
    pub fn new(config: DcrnnConfig) -> Result<Self> {
        if config.diffusion_steps == 0 || config.hidden_size == 0 || config.epochs == 0 {
            return Err(GeoStError::InvalidFrame(
                "diffusion_steps, hidden_size, and epochs must be positive".to_string(),
            ));
        }
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err(GeoStError::InvalidFrame(
                "learning_rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            config,
            node_ids: Vec::new(),
            frequency: String::new(),
            horizon: 0,
            adjacency: None,
            reverse_adjacency: None,
            weights: Vec::new(),
            intercepts: Vec::new(),
            encoder_weights: Vec::new(),
            recurrent_weights: Vec::new(),
            history: Vec::new(),
            target_mean: 0.0,
            target_scale: 1.0,
        })
    }

    pub fn fit(&mut self, frame: &GraphTemporalFrame) -> Result<()> {
        frame.validate()?;
        let nodes = frame.node_ids.len();
        let diffusion_feature_len = self.diffusion_feature_len();
        let decoder_feature_len = self.decoder_feature_len();
        let (target_mean, target_scale) = target_center_scale(&frame.target);
        let normalized_target: Vec<Vec<f64>> = frame
            .target
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| (value - target_mean) / target_scale)
                    .collect()
            })
            .collect();
        self.encoder_weights = deterministic_weight_matrix(
            self.config.hidden_size,
            diffusion_feature_len,
            0x9e37_79b9_7f4a_7c15,
        );
        self.recurrent_weights = deterministic_weight_matrix(
            self.config.hidden_size,
            self.config.hidden_size,
            0xbf58_476d_1ce4_e5b9,
        );
        let forward = frame.adjacency.row_normalized();
        let reverse = forward.transpose(nodes);
        let samples = frame.target.len() - frame.horizon;
        let mut hidden_by_cutoff = Vec::with_capacity(samples);
        let mut hidden = vec![vec![0.0; self.config.hidden_size]; nodes];
        for row in normalized_target.iter().take(samples) {
            hidden = self.recurrent_hidden(&hidden, row, &forward, &reverse)?;
            hidden_by_cutoff.push(hidden.clone());
        }
        let teacher_ratio = self.average_teacher_forcing_ratio();

        self.weights = (0..frame.horizon)
            .into_par_iter()
            .map(|h| -> Result<Vec<f64>> {
                let mut xtx = vec![vec![0.0; decoder_feature_len]; decoder_feature_len];
                let mut xty = vec![0.0; decoder_feature_len];
                for t in 0..samples {
                    let mut decoder_input = normalized_target[t].clone();
                    let mut decoder_hidden = hidden_by_cutoff[t].clone();
                    let mut prior_prediction = decoder_input.clone();
                    for step in 0..=h {
                        let decoder_features = self.decoder_features(
                            &decoder_input,
                            &decoder_hidden,
                            &forward,
                            &reverse,
                        )?;
                        if step == h {
                            for node in 0..nodes {
                                let actual = normalized_target[t + h + 1][node];
                                let x = &decoder_features
                                    [node * decoder_feature_len..(node + 1) * decoder_feature_len];
                                for row in 0..decoder_feature_len {
                                    xty[row] += x[row] * actual;
                                    for col in 0..decoder_feature_len {
                                        xtx[row][col] += x[row] * x[col];
                                    }
                                }
                            }
                            break;
                        }
                        let next_actual = &normalized_target[t + step + 1];
                        prior_prediction =
                            blend_rows(next_actual, &prior_prediction, teacher_ratio);
                        decoder_hidden = self.recurrent_hidden(
                            &decoder_hidden,
                            &prior_prediction,
                            &forward,
                            &reverse,
                        )?;
                        decoder_input = prior_prediction.clone();
                    }
                }
                for (idx, row) in xtx.iter_mut().enumerate() {
                    row[idx] += self.config.ridge.max(1.0e-8);
                }
                Ok(solve_linear_system(xtx, xty))
            })
            .collect::<Result<Vec<_>>>()?;
        self.intercepts = vec![0.0; frame.horizon];

        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.horizon = frame.horizon;
        self.adjacency = Some(forward);
        self.reverse_adjacency = Some(reverse);
        self.history = normalized_target;
        self.target_mean = target_mean;
        self.target_scale = target_scale;
        Ok(())
    }

    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "prediction horizon must be positive".to_string(),
            ));
        }
        if self.weights.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let forward = self.adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        let reverse = self.reverse_adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        let mut state = self.history.last().cloned().ok_or(GeoStError::NotFit)?;
        let normalized_min = self
            .history
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let normalized_max = self
            .history
            .iter()
            .flatten()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut hidden = self.encode_history(forward, reverse)?;
        let mut predictions = Vec::with_capacity(horizon);
        for step in 0..horizon {
            let h = step.min(self.weights.len() - 1);
            let features = self.decoder_features(&state, &hidden, forward, reverse)?;
            let feature_len = self.weights[h].len();
            let rows = features
                .chunks(feature_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let means = vec![0.0; feature_len];
            let intercepts = vec![self.intercepts[h]; state.len()];
            let execution_backend = profitable_affine_backend(
                self.neural_backend_selection(),
                rows.len(),
                feature_len,
            )?;
            let next = backend_affine_scores(
                &execution_backend,
                &rows,
                &means,
                &self.weights[h],
                &intercepts,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?
            .into_iter()
            .map(|value| value.clamp(normalized_min, normalized_max))
            .collect::<Vec<_>>();
            hidden = self.recurrent_hidden(&hidden, &next, forward, reverse)?;
            state = next.clone();
            predictions.push(
                next.into_iter()
                    .map(|value| value * self.target_scale + self.target_mean)
                    .collect(),
            );
        }
        Ok(predictions)
    }

    pub fn backtest(
        &self,
        frame: &GraphTemporalFrame,
        train_size: usize,
    ) -> Result<GraphForecastMetrics> {
        if train_size == 0 || train_size + frame.horizon > frame.target.len() {
            return Err(GeoStError::InvalidFrame(
                "train_size must leave a full forecast horizon".to_string(),
            ));
        }
        let mut train = frame.clone();
        train.timestamps.truncate(train_size);
        train.target.truncate(train_size);
        let mut model = Self::new(self.config.clone())?;
        model.fit(&train)?;
        let predictions = model.predict(frame.horizon)?;
        let actual = frame.target[train_size..train_size + frame.horizon].to_vec();
        Ok(graph_metrics(
            &predictions,
            &actual,
            &frame.node_ids,
            &frame.adjacency,
        ))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json_string(value: &str) -> Result<Self> {
        Ok(serde_json::from_str(value)?)
    }

    fn diffusion_feature_len(&self) -> usize {
        2 * self.config.diffusion_steps + 2
    }

    fn decoder_feature_len(&self) -> usize {
        self.diffusion_feature_len() + self.config.hidden_size
    }

    fn diffusion_features(
        &self,
        state: &[f64],
        forward: &CsrAdjacency,
        reverse: &CsrAdjacency,
    ) -> Result<Vec<f64>> {
        let nodes = state.len();
        let feature_len = self.diffusion_feature_len();
        let mut vectors = Vec::with_capacity(feature_len - 1);
        vectors.push(state.to_vec());
        let backend = self.neural_backend_selection();
        let accelerate = backend.selected != "cpu"
            && nodes.saturating_mul(self.config.diffusion_steps) >= GEO_ST_CSR_DISPATCH_MIN_OPS;
        let forward_indptr = forward
            .indptr
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>();
        let forward_indices = forward
            .indices
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>();
        let forward_weights = forward
            .data
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let reverse_indptr = reverse
            .indptr
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>();
        let reverse_indices = reverse
            .indices
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>();
        let reverse_weights = reverse
            .data
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let mut current = state.to_vec();
        for _ in 0..self.config.diffusion_steps {
            let next = if accelerate {
                backend_csr_diffusion_f32(
                    &backend,
                    &forward_indptr,
                    &forward_indices,
                    &forward_weights,
                    1,
                    &current
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?
                .into_iter()
                .map(f64::from)
                .collect()
            } else {
                let mut next = vec![0.0; nodes];
                forward.matvec(&current, &mut next);
                next
            };
            vectors.push(next.clone());
            current = next;
        }
        current = state.to_vec();
        for _ in 0..self.config.diffusion_steps {
            let next = if accelerate {
                backend_csr_diffusion_f32(
                    &backend,
                    &reverse_indptr,
                    &reverse_indices,
                    &reverse_weights,
                    1,
                    &current
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?
                .into_iter()
                .map(f64::from)
                .collect()
            } else {
                let mut next = vec![0.0; nodes];
                reverse.matvec(&current, &mut next);
                next
            };
            vectors.push(next.clone());
            current = next;
        }
        let mut out = vec![0.0; nodes * feature_len];
        for node in 0..nodes {
            let offset = node * feature_len;
            for (idx, vector) in vectors.iter().enumerate() {
                out[offset + idx] = vector[node];
            }
            out[offset + feature_len - 1] = 1.0;
        }
        Ok(out)
    }

    fn decoder_features(
        &self,
        state: &[f64],
        hidden: &[Vec<f64>],
        forward: &CsrAdjacency,
        reverse: &CsrAdjacency,
    ) -> Result<Vec<f64>> {
        let nodes = state.len();
        let diffusion_len = self.diffusion_feature_len();
        let decoder_len = self.decoder_feature_len();
        let diffusion = self.diffusion_features(state, forward, reverse)?;
        let mut out = vec![0.0; nodes * decoder_len];
        for (node, hidden_row) in hidden.iter().enumerate().take(nodes) {
            let src = node * diffusion_len;
            let dst = node * decoder_len;
            out[dst..dst + diffusion_len].copy_from_slice(&diffusion[src..src + diffusion_len]);
            out[dst + diffusion_len..dst + decoder_len].copy_from_slice(hidden_row);
        }
        Ok(out)
    }

    fn recurrent_hidden(
        &self,
        previous_hidden: &[Vec<f64>],
        state: &[f64],
        forward: &CsrAdjacency,
        reverse: &CsrAdjacency,
    ) -> Result<Vec<Vec<f64>>> {
        let diffusion_len = self.diffusion_feature_len();
        let diffusion = self.diffusion_features(state, forward, reverse)?;
        let diffusion_rows = diffusion
            .chunks(diffusion_len)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let transpose = |weights: &[Vec<f64>], input_width: usize, output_width: usize| {
            (0..input_width)
                .flat_map(|input| (0..output_width).map(move |output| weights[output][input]))
                .collect::<Vec<_>>()
        };
        let backend = self.neural_backend_selection();
        let input_terms = dense_matrix_product(
            &diffusion_rows,
            &transpose(
                &self.encoder_weights,
                diffusion_len,
                self.config.hidden_size,
            ),
            &vec![0.0; self.config.hidden_size],
            diffusion_len,
            self.config.hidden_size,
            Some(&backend),
        )?;
        let recurrent_terms = dense_matrix_product(
            previous_hidden,
            &transpose(
                &self.recurrent_weights,
                self.config.hidden_size,
                self.config.hidden_size,
            ),
            &vec![0.0; self.config.hidden_size],
            self.config.hidden_size,
            self.config.hidden_size,
            Some(&backend),
        )?;
        Ok(input_terms
            .into_iter()
            .zip(recurrent_terms)
            .map(|(input, recurrent)| {
                input
                    .into_iter()
                    .zip(recurrent)
                    .map(|(input, recurrent)| (input + 0.35 * recurrent).tanh())
                    .collect()
            })
            .collect())
    }

    fn encode_history(
        &self,
        forward: &CsrAdjacency,
        reverse: &CsrAdjacency,
    ) -> Result<Vec<Vec<f64>>> {
        let nodes = self.node_ids.len();
        if nodes == 0 || self.encoder_weights.is_empty() || self.recurrent_weights.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let mut hidden = vec![vec![0.0; self.config.hidden_size]; nodes];
        for row in &self.history {
            hidden = self.recurrent_hidden(&hidden, row, forward, reverse)?;
        }
        Ok(hidden)
    }

    fn teacher_forcing_ratio(&self, epoch: usize) -> f64 {
        if self.config.epochs <= 1 {
            return self.config.teacher_forcing_end;
        }
        let progress = epoch as f64 / (self.config.epochs - 1) as f64;
        self.config.teacher_forcing_start
            + progress * (self.config.teacher_forcing_end - self.config.teacher_forcing_start)
    }

    fn average_teacher_forcing_ratio(&self) -> f64 {
        let total: f64 = (0..self.config.epochs)
            .map(|epoch| self.teacher_forcing_ratio(epoch))
            .sum();
        (total / self.config.epochs as f64).clamp(0.0, 1.0)
    }

    fn neural_backend_selection(&self) -> BackendSelection {
        BackendSelection {
            requested: self.config.backend.requested.clone(),
            selected: self.config.backend.selected.clone(),
            available: self.config.backend.available.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct STAEformerConfig {
    pub lookback: usize,
    pub attention_heads: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub ridge: f64,
    #[serde(default)]
    pub backend: ComputeBackendSelection,
}

impl Default for STAEformerConfig {
    fn default() -> Self {
        Self {
            lookback: 8,
            attention_heads: 4,
            hidden_size: 8,
            epochs: 120,
            learning_rate: 0.02,
            ridge: 1e-4,
            backend: select_compute_backend(None).expect("default CPU backend is always available"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct STAEformerForecaster {
    pub config: STAEformerConfig,
    node_ids: Vec<String>,
    frequency: String,
    horizon: usize,
    adjacency: Option<CsrAdjacency>,
    weights: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    temporal_queries: Vec<Vec<f64>>,
    temporal_keys: Vec<Vec<f64>>,
    spatial_weights: Vec<f64>,
    history: Vec<Vec<f64>>,
    target_mean: f64,
    target_scale: f64,
}

/// Native graph-transformer profiles. The public Python names map to these
/// generic architectural behaviors rather than embedding a benchmark in the
/// shared Rust API.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphTransformerProfile {
    HeterogeneousMoE,
    EfficientHighOrder,
    LongShortFusion,
    GatedGraphTemporal,
    SpatialShiftGraphonMoE,
}

const fn default_lsttn_batch_size() -> usize {
    32
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperGraphTransformerConfig {
    pub profile: GraphTransformerProfile,
    pub lookback: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub graph_order: usize,
    pub experts: usize,
    pub periodicity: usize,
    pub recent_window: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    #[serde(default = "default_lsttn_batch_size")]
    pub batch_size: usize,
    #[serde(default)]
    pub backend: ComputeBackendSelection,
}

impl Default for PaperGraphTransformerConfig {
    fn default() -> Self {
        Self {
            profile: GraphTransformerProfile::HeterogeneousMoE,
            lookback: 12,
            hidden_size: 16,
            attention_heads: 4,
            graph_order: 2,
            experts: 4,
            periodicity: 24,
            recent_window: 12,
            epochs: 80,
            learning_rate: 0.01,
            weight_decay: 1e-5,
            batch_size: 32,
            backend: select_compute_backend(None).expect("default CPU backend is always available"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperGraphTransformerForecaster {
    pub config: PaperGraphTransformerConfig,
    node_ids: Vec<String>,
    frequency: String,
    horizon: usize,
    adjacency: Option<CsrAdjacency>,
    #[serde(default)]
    trainable_state: Option<TrainableGraphTransformerState>,
    history: Vec<Vec<f64>>,
    #[serde(default)]
    history_time_features: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    covariate_roles: Option<Vec<String>>,
    owner_mask: Vec<bool>,
    fit_activation_tape_bytes: usize,
    fit_frozen_patch_cache_bytes: usize,
    target_mean: f64,
    target_scale: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnTrainingCheckpoint {
    version: u32,
    architecture_fingerprint: u64,
    backbone_fingerprint: u64,
    shard_data_fingerprint: u64,
    epoch_fingerprint: u64,
    config_json: String,
    state: TrainableGraphTransformerState,
    pretraining_completed: usize,
    supervised_batches_completed: usize,
    complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnSharedSegment {
    name: String,
    parameters: Vec<f64>,
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnSharedState {
    version: u32,
    architecture_fingerprint: u64,
    backbone_fingerprint: u64,
    hidden: usize,
    attention_heads: usize,
    periodicity: usize,
    recent_window: usize,
    context_window: usize,
    horizons: usize,
    experts: usize,
    graph_order: usize,
    steps: u64,
    target_mean: f64,
    target_scale: f64,
    segments: Vec<LsttnSharedSegment>,
}

/// Commit record for an LSTTN shared/local checkpoint pair. The manifest is
/// written last, so a loader can reject any pair interrupted between files.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnShardPairManifest {
    version: u32,
    local_hash: u64,
    shared_hash: u64,
    architecture_fingerprint: u64,
    backbone_fingerprint: u64,
    shared_steps: u64,
    local_steps: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnShardIdentity {
    backbone_id: String,
    equipment_category: String,
    target_unit: String,
    node_registry_fingerprint: String,
    graph_cutoff: String,
    shard_policy_fingerprint: String,
    epoch: usize,
}

/// Immutable output of one independently trained shard round.  The base hash
/// prevents applying a delta to a different frozen shared state.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnSharedRound {
    version: u32,
    shard_id: String,
    objective_weight: f64,
    base_shared_hash: u64,
    shared: LsttnSharedState,
}

#[derive(Clone, Copy, Debug, Serialize)]
enum LsttnTrainingPhase {
    Both,
    Pretrain,
    Supervised,
    LocalAdaptation,
}

impl LsttnTrainingPhase {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "both" => Ok(Self::Both),
            "pretrain" => Ok(Self::Pretrain),
            "supervised" => Ok(Self::Supervised),
            "local_adaptation" => Ok(Self::LocalAdaptation),
            _ => Err(GeoStError::InvalidFrame(format!(
                "unknown LSTTN shard training phase {value:?}"
            ))),
        }
    }

    fn runs_pretraining(self) -> bool {
        matches!(self, Self::Both | Self::Pretrain)
    }

    fn runs_supervised(self) -> bool {
        !matches!(self, Self::Pretrain)
    }

    fn freezes_shared(self) -> bool {
        matches!(self, Self::LocalAdaptation)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PaperGraphTransformerArchitectureReport {
    pub profile: GraphTransformerProfile,
    pub components: Vec<String>,
    pub graphon_expert_count: usize,
    pub direct_multi_horizon: bool,
    pub trainable_forecast_head: bool,
}

/// Auditable ownership of a persisted LSTTN parameter segment. Segments are
/// deliberately reported at serialization boundaries so shared-state size can
/// be checked independently of the shard's node count.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LsttnParameterInventoryEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub byte_size: usize,
    pub ownership: String,
    pub optimizer_ownership: String,
    pub hash: String,
    pub node_count_dependent: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LsttnMemoryTelemetry {
    pub frame_storage_bytes: usize,
    pub parameter_bytes: usize,
    pub optimizer_moment_bytes: usize,
    pub activation_tape_bytes: usize,
    pub frozen_patch_cache_bytes: usize,
    pub graph_diffusion_workspace_bytes: usize,
    pub prediction_buffer_bytes: usize,
    pub total_accounted_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LsttnEdgeDiagnostic {
    pub source: usize,
    pub target: usize,
    pub structural_prior: f64,
    pub forward_diffusion: f64,
    pub backward_diffusion: f64,
    pub learned_adaptive_contribution: f64,
    pub normalized_attention: f64,
    pub horizon_sensitivity: Vec<f64>,
}

/// Noncrossing conformal intervals around the native LSTTN median decoder.
/// Calibration observations and their median forecasts are supplied explicitly
/// so callers can enforce a raw held-out calibration split.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LsttnConformalForecast {
    pub lower: Vec<Vec<f64>>,
    pub median: Vec<Vec<f64>>,
    pub upper: Vec<Vec<f64>>,
    pub radius_by_horizon: Vec<f64>,
    pub calibration_coverage_by_horizon: Vec<f64>,
    pub calibration_count_by_horizon: Vec<usize>,
}

pub fn lsttn_conformal_forecast(
    calibration_actual: &[Vec<Vec<f64>>],
    calibration_median: &[Vec<Vec<f64>>],
    forecast_median: &[Vec<f64>],
    alpha: f64,
) -> Result<LsttnConformalForecast> {
    if !(0.0 < alpha && alpha < 1.0)
        || calibration_actual.is_empty()
        || calibration_actual.len() != calibration_median.len()
        || forecast_median.is_empty()
    {
        return Err(GeoStError::InvalidFrame(
            "conformal calibration requires matching non-empty held-out actual and median arrays with alpha in (0, 1)"
                .to_string(),
        ));
    }
    let horizons = forecast_median.len();
    let nodes = forecast_median[0].len();
    if nodes == 0
        || forecast_median
            .iter()
            .any(|row| row.len() != nodes || row.iter().any(|value| !value.is_finite()))
    {
        return Err(GeoStError::InvalidFrame(
            "forecast median must be a finite non-empty horizon by node matrix".to_string(),
        ));
    }
    let mut radii = Vec::with_capacity(horizons);
    let mut coverage = Vec::with_capacity(horizons);
    let mut counts = Vec::with_capacity(horizons);
    for horizon in 0..horizons {
        let mut residuals = Vec::new();
        for (actual_origin, median_origin) in calibration_actual.iter().zip(calibration_median) {
            if actual_origin.len() != horizons
                || median_origin.len() != horizons
                || actual_origin.iter().any(|row| row.len() != nodes)
                || median_origin.iter().any(|row| row.len() != nodes)
            {
                return Err(GeoStError::InvalidFrame(
                    "calibration arrays must have shape [origins, forecast_horizon, nodes]"
                        .to_string(),
                ));
            }
            for node in 0..nodes {
                let actual = actual_origin[horizon][node];
                let median = median_origin[horizon][node];
                if !actual.is_finite() || !median.is_finite() {
                    return Err(GeoStError::InvalidFrame(
                        "calibration actual and median values must be finite".to_string(),
                    ));
                }
                residuals.push((actual - median).abs());
            }
        }
        residuals.sort_by(f64::total_cmp);
        let count = residuals.len();
        let rank = (((count + 1) as f64 * (1.0 - alpha)).ceil() as usize)
            .saturating_sub(1)
            .min(count - 1);
        let radius = residuals[rank];
        let covered = residuals.iter().filter(|value| **value <= radius).count();
        radii.push(radius);
        coverage.push(covered as f64 / count as f64);
        counts.push(count);
    }
    let lower = forecast_median
        .iter()
        .enumerate()
        .map(|(horizon, row)| row.iter().map(|value| value - radii[horizon]).collect())
        .collect();
    let upper = forecast_median
        .iter()
        .enumerate()
        .map(|(horizon, row)| row.iter().map(|value| value + radii[horizon]).collect())
        .collect();
    Ok(LsttnConformalForecast {
        lower,
        median: forecast_median.to_vec(),
        upper,
        radius_by_horizon: radii,
        calibration_coverage_by_horizon: coverage,
        calibration_count_by_horizon: counts,
    })
}

/// Serializable native parameters for the graph-transformer profiles.
///
/// The state deliberately owns every learned projection and its optimizer
/// moments.  This avoids a Python or fixed-feature training path and makes
/// attention, routing, gated fusion, and graphon embeddings part of the
/// fitted artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainableGraphTransformerState {
    parameters: Vec<f64>,
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
    steps: u64,
    #[serde(default)]
    shared_steps: u64,
    #[serde(default)]
    local_steps: u64,
    #[serde(default)]
    architecture_fingerprint: u64,
    #[serde(default)]
    backbone_fingerprint: u64,
    #[serde(default)]
    shard_data_fingerprint: u64,
    #[serde(default)]
    warm_start_policy_fingerprint: u64,
    nodes: usize,
    hidden: usize,
    attention_heads: usize,
    periodicity: usize,
    recent_window: usize,
    context_window: usize,
    horizons: usize,
    experts: usize,
    graph_order: usize,
    /// Native cadence for the first periodic branch. Zero preserves legacy
    /// non-weekly checkpoints, where it is derived from `periodicity`.
    #[serde(default)]
    periodic_short_lag: usize,
    #[serde(default = "unit_scale")]
    target_scale: f64,
    #[serde(default)]
    normalized_zero: f64,
    /// Present only in compact shard-local artifacts. Each pair addresses a
    /// range in the architecture's full parameter vector; loading expands
    /// these values before a shared backbone is imported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    local_parameter_ranges: Option<Vec<(usize, usize)>>,
    #[serde(skip)]
    shard_training: bool,
    #[serde(skip)]
    freeze_shared: bool,
}

/// CUDA-owned, portable-state bridge for LSTTN.  It deliberately owns no
/// serialized driver object: parameters and Adam moments are uploaded from
/// `TrainableGraphTransformerState` on construction and copied back only at a
/// checkpoint boundary.  CSR topology and structural weights remain resident
/// for every pretraining/supervised batch.
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(dead_code)]
struct CudaLsttnTensorExecutor {
    arena: CudaTensorArena,
    nodes: usize,
    node_tile_size: usize,
    forward_edges: usize,
    reverse_edges: usize,
    adaptive_edges: usize,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(dead_code)]
impl CudaLsttnTensorExecutor {
    const PARAMETERS: usize = 0;
    const FIRST_MOMENT: usize = 1;
    const SECOND_MOMENT: usize = 2;
    const FORWARD_WEIGHTS: usize = 3;
    const REVERSE_WEIGHTS: usize = 4;
    const ADAPTIVE_WEIGHTS: usize = 5;
    const SUPERVISED_INPUT: usize = 6;
    const SUPERVISED_TARGET: usize = 7;
    const DIRECT_OUTPUT: usize = 8;
    const PATCH_EMBEDDING: usize = 9;
    const PATCH_WITH_POSITION: usize = 10;
    const ATTENTION_SEQUENCE: usize = 11;
    const ENCODER_Q: usize = 12;
    const ENCODER_K: usize = 13;
    const ENCODER_V: usize = 14;
    const ENCODER_ATTENTION: usize = 15;
    const ENCODER_PROJECTED: usize = 16;
    const ENCODER_RESIDUAL: usize = 17;
    const ENCODER_NORMALIZED: usize = 18;
    const ENCODER_FFN_EXPANDED: usize = 19;
    const ENCODER_FFN_ACTIVATED: usize = 20;
    const ENCODER_FFN_CONTRACTED: usize = 21;
    const ENCODER_FFN_RESIDUAL: usize = 22;
    const ADAPTIVE_LOGITS: usize = 23;
    const LONG_TEMPORAL: usize = 24;
    const LONG_STAGE_A: usize = 25;
    const LONG_STAGE_B: usize = 26;
    const SHORT_INPUT: usize = 27;
    const SHORT_FILTER: usize = 28;
    const SHORT_GATE: usize = 29;
    const SHORT_GATED: usize = 30;
    const SHORT_SKIP_PROJECTION: usize = 31;
    const SHORT_FORWARD_ONE: usize = 32;
    const SHORT_FORWARD_TWO: usize = 33;
    const SHORT_BACKWARD_ONE: usize = 34;
    const SHORT_BACKWARD_TWO: usize = 35;
    const SHORT_ADAPTIVE_ONE: usize = 36;
    const SHORT_ADAPTIVE_TWO: usize = 37;
    const SHORT_CONCAT_A: usize = 38;
    const SHORT_CONCAT_B: usize = 39;
    const SHORT_CONCAT_C: usize = 40;
    const SHORT_CONCAT_D: usize = 41;
    const SHORT_CONCAT_E: usize = 42;
    const SHORT_CONCAT_F: usize = 43;
    const SHORT_GRAPH: usize = 44;
    const SHORT_RESIDUAL: usize = 45;
    const SHORT_SKIP_A: usize = 46;
    const SHORT_NORMALIZED: usize = 47;
    const SHORT_SKIP_B: usize = 48;
    const SHORT_BATCH_STATS: usize = 49;
    const PARAMETER_GRADIENT: usize = 50;
    const GRADIENT_NORM: usize = 51;
    const SUPERVISED_LOSS: usize = 52;
    const PERIODIC_SHORT: usize = 53;
    const PERIODIC_SEASONAL: usize = 54;
    const FUSION_A: usize = 55;
    const FUSION_B: usize = 56;
    const FUSION_C: usize = 57;
    const FUSION_D: usize = 58;
    const FUSION_OUTPUT: usize = 59;
    const DIRECT_NODE_MAJOR: usize = 60;
    const FUSED_DIRECT_OUTPUT: usize = 61;
    const DIRECT_OUTPUT_GRADIENT: usize = 62;
    const DIRECT_NODE_MAJOR_GRADIENT: usize = 63;
    const FUSION_REPRESENTATION_GRADIENT: usize = 64;
    const FUSION_RELU_GRADIENT: usize = 65;
    const FUSION_CONCAT_GRADIENT: usize = 66;
    const SHORT_GRADIENT: usize = 67;
    const TREND_GRADIENT: usize = 68;
    const FUSION_SECOND_GRADIENT: usize = 69;
    const FUSION_FIRST_RELU_GRADIENT: usize = 70;
    const FUSION_INPUT_GRADIENT: usize = 71;
    const LONG_GRADIENT: usize = 72;
    const PERIODIC_PAIR_GRADIENT: usize = 73;
    const PERIODIC_SHORT_GRADIENT: usize = 74;
    const PERIODIC_SEASONAL_GRADIENT: usize = 75;
    const PERIODIC_SHORT_INPUT: usize = 76;
    const PERIODIC_SEASONAL_INPUT: usize = 77;
    const PERIODIC_SHORT_INPUT_GRADIENT: usize = 78;
    const PERIODIC_SEASONAL_INPUT_GRADIENT: usize = 79;
    const PERIODIC_GRAD_IDENTITY: usize = 80;
    const PERIODIC_GRAD_FORWARD_ONE: usize = 81;
    const PERIODIC_GRAD_FORWARD_TWO: usize = 82;
    const PERIODIC_GRAD_REVERSE_ONE: usize = 83;
    const PERIODIC_GRAD_REVERSE_TWO: usize = 84;
    const PERIODIC_GRAD_ADAPTIVE_ONE: usize = 85;
    const PERIODIC_GRAD_ADAPTIVE_TWO: usize = 86;
    const PERIODIC_TEMP_A: usize = 87;
    const PERIODIC_TEMP_B: usize = 88;
    const PERIODIC_TEMP_C: usize = 89;
    const PERIODIC_BASE_GRADIENT: usize = 90;
    const PERIODIC_FORWARD_BASE_GRADIENT: usize = 91;
    const PERIODIC_REVERSE_BASE_GRADIENT: usize = 92;
    const PERIODIC_ADAPTIVE_BASE_GRADIENT: usize = 93;
    const PERIODIC_ADAPTIVE_EDGE_GRADIENT: usize = 94;
    const PERIODIC_ADAPTIVE_LOGIT_GRADIENT: usize = 95;
    const LONG_BACKWARD_A: usize = 96;
    const LONG_BACKWARD_B: usize = 97;
    const SHORT_REVERSE_INPUT_GRADIENT_A: usize = 98;
    const SHORT_REVERSE_INPUT_GRADIENT_B: usize = 99;
    const SHORT_REVERSE_SKIP_GRADIENT_A: usize = 100;
    const SHORT_REVERSE_SKIP_GRADIENT_B: usize = 101;
    const SHORT_REVERSE_GRAPH_GRADIENT: usize = 102;
    const SHORT_REVERSE_CONCAT_GRADIENT: usize = 103;
    const SHORT_REVERSE_SPLIT_A: usize = 104;
    const SHORT_REVERSE_SPLIT_B: usize = 105;
    const SHORT_REVERSE_SPLIT_C: usize = 106;
    const SHORT_REVERSE_SPLIT_D: usize = 107;
    const SHORT_REVERSE_SPLIT_E: usize = 108;
    const SHORT_REVERSE_SPLIT_F: usize = 109;
    const SHORT_REVERSE_GATED_GRADIENT: usize = 110;
    const SHORT_REVERSE_FILTER_GRADIENT: usize = 111;
    const SHORT_REVERSE_GATE_GRADIENT: usize = 112;
    const SHORT_REVERSE_EDGE_GRADIENT: usize = 113;
    const SHORT_REVERSE_LOGIT_GRADIENT: usize = 114;
    const SHORT_GRAPH_GRAD_IDENTITY: usize = 115;
    const SHORT_GRAPH_GRAD_FORWARD_ONE: usize = 116;
    const SHORT_GRAPH_GRAD_FORWARD_TWO: usize = 117;
    const SHORT_GRAPH_GRAD_REVERSE_ONE: usize = 118;
    const SHORT_GRAPH_GRAD_REVERSE_TWO: usize = 119;
    const SHORT_GRAPH_GRAD_ADAPTIVE_ONE: usize = 120;
    const SHORT_GRAPH_GRAD_ADAPTIVE_TWO: usize = 121;
    const PRETRAIN_VISIBLE_TOKENS: usize = 122;
    const PRETRAIN_DECODER_INPUT: usize = 123;
    const PRETRAIN_DECODER_Q: usize = 124;
    const PRETRAIN_DECODER_K: usize = 125;
    const PRETRAIN_DECODER_V: usize = 126;
    const PRETRAIN_DECODER_ATTENTION: usize = 127;
    const PRETRAIN_DECODER_PROJECTED: usize = 128;
    const PRETRAIN_DECODER_RESIDUAL: usize = 129;
    const PRETRAIN_DECODER_NORMALIZED: usize = 130;
    const PRETRAIN_DECODER_FFN_EXPANDED: usize = 131;
    const PRETRAIN_DECODER_FFN_ACTIVATED: usize = 132;
    const PRETRAIN_DECODER_FFN_CONTRACTED: usize = 133;
    const PRETRAIN_DECODER_FFN_RESIDUAL: usize = 134;
    const PRETRAIN_CONTEXT_GRADIENT: usize = 135;
    const PRETRAIN_DECODER_INPUT_GRADIENT: usize = 136;
    const PRETRAIN_VISIBLE_GRADIENT: usize = 137;
    const PRETRAIN_ENCODER_DECODER_GRADIENT: usize = 138;
    const PRETRAIN_ENCODER_OUTPUT_GRADIENT: usize = 139;
    const PRETRAIN_SEQUENCE_GRADIENT: usize = 140;
    const PRETRAIN_PATCH_LAYOUT_GRADIENT: usize = 141;
    const PRETRAIN_PATCH_EMBEDDING_GRADIENT: usize = 142;
    const PRETRAIN_TEMP_A: usize = 143;
    const PRETRAIN_TEMP_B: usize = 144;
    const PRETRAIN_TEMP_C: usize = 145;
    const PRETRAIN_TEMP_D: usize = 146;
    const PRETRAIN_TEMP_E: usize = 147;
    const PRETRAIN_TEMP_F: usize = 148;
    const PRETRAIN_LOSS: usize = 149;
    const PRETRAIN_LAYER_A: usize = 150;
    const PRETRAIN_LAYER_B: usize = 151;
    const PRETRAIN_LAYER_GRAD_A: usize = 152;
    const PRETRAIN_LAYER_GRAD_B: usize = 153;
    const PRETRAIN_DECODER_OUTPUT: usize = 154;
    const PRETRAIN_POSITION_GRADIENT: usize = 155;
    const SUPERVISED_WEIGHT: usize = 156;
    const FORWARD_INDPTR: usize = 0;
    const FORWARD_INDICES: usize = 1;
    const REVERSE_INDPTR: usize = 2;
    const REVERSE_INDICES: usize = 3;
    const ADAPTIVE_INDPTR: usize = 4;
    const ADAPTIVE_INDICES: usize = 5;
    const VISIBLE_PATCH_INDICES: usize = 6;
    const MASKED_PATCH_INDICES: usize = 7;

    fn new(state: &TrainableGraphTransformerState, adjacency: &CsrAdjacency) -> Result<Self> {
        let nodes = state.nodes;
        let reverse = adjacency.transpose(nodes).row_normalized();
        let adaptive = adjacency.with_adaptive_self_candidates(nodes);
        let u32_values = |values: &[usize], label: &str| -> Result<Vec<u32>> {
            values
                .iter()
                .map(|value| {
                    u32::try_from(*value).map_err(|_| {
                        GeoStError::InvalidFrame(format!(
                            "LSTTN CUDA {label} exceeds the u32 CSR address range"
                        ))
                    })
                })
                .collect()
        };
        let to_f32 = |values: &[f64], label: &str| -> Result<Vec<f32>> {
            values
                .iter()
                .map(|value| {
                    let value = *value as f32;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(GeoStError::InvalidFrame(format!(
                            "LSTTN CUDA {label} contains a non-finite f32 value"
                        )))
                    }
                })
                .collect()
        };
        let mut arena = CudaTensorArena::new(157)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .upload_f32(Self::PARAMETERS, &to_f32(&state.parameters, "parameters")?)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .upload_f32(
                Self::FIRST_MOMENT,
                &to_f32(&state.first_moment, "first moments")?,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .upload_f32(
                Self::SECOND_MOMENT,
                &to_f32(&state.second_moment, "second moments")?,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .fill_f32(Self::PARAMETER_GRADIENT, state.parameters.len(), 0.0)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let upload_graph = |arena: &mut CudaTensorArena,
                            indptr_slot: usize,
                            indices_slot: usize,
                            weights_slot: usize,
                            graph: &CsrAdjacency,
                            label: &str|
         -> Result<()> {
            arena
                .upload_u32(indptr_slot, &u32_values(&graph.indptr, label)?)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            arena
                .upload_u32(indices_slot, &u32_values(&graph.indices, label)?)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            arena
                .upload_f32(weights_slot, &to_f32(&graph.data, label)?)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
        };
        upload_graph(
            &mut arena,
            Self::FORWARD_INDPTR,
            Self::FORWARD_INDICES,
            Self::FORWARD_WEIGHTS,
            adjacency,
            "forward CSR",
        )?;
        upload_graph(
            &mut arena,
            Self::REVERSE_INDPTR,
            Self::REVERSE_INDICES,
            Self::REVERSE_WEIGHTS,
            &reverse,
            "reverse CSR",
        )?;
        upload_graph(
            &mut arena,
            Self::ADAPTIVE_INDPTR,
            Self::ADAPTIVE_INDICES,
            Self::ADAPTIVE_WEIGHTS,
            &adaptive,
            "adaptive CSR",
        )?;
        Ok(Self {
            arena,
            nodes,
            // A 1,024-node physical tile keeps a `[32, 52, tile, 32]`
            // hidden activation below 208 MiB. Logical batches remain 32 and
            // gradients are reduced across every tile before AdamW.
            node_tile_size: 1_024.min(nodes).max(1),
            forward_edges: adjacency.indices.len(),
            reverse_edges: reverse.indices.len(),
            adaptive_edges: adaptive.indices.len(),
        })
    }

    fn allocation_count(&self) -> usize {
        self.arena.allocation_count()
    }

    fn node_tiles(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.nodes)
            .step_by(self.node_tile_size)
            .map(|start| start..(start + self.node_tile_size).min(self.nodes))
    }

    fn affine_projection(
        &mut self,
        input_slot: usize,
        output_slot: usize,
        parameter_offset: usize,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        self.arena
            .affine_parameter_slice_f32(
                input_slot,
                Self::PARAMETERS,
                parameter_offset,
                parameter_offset + input_width * output_width,
                output_slot,
                rows,
                input_width,
                output_width,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn patch_embedding(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        times: usize,
        tile_nodes: usize,
        channels: usize,
        patch_width: usize,
        hidden: usize,
    ) -> Result<()> {
        self.arena
            .patch_embedding_f32(
                Self::SUPERVISED_INPUT,
                Self::PARAMETERS,
                layout.lsttn_patch_embedding,
                layout.lsttn_patch_embedding + patch_width * hidden,
                Self::PATCH_EMBEDDING,
                batches,
                times,
                tile_nodes,
                channels,
                patch_width,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn add_patch_positions(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
    ) -> Result<()> {
        self.arena
            .add_patch_positions_f32(
                Self::PATCH_EMBEDDING,
                Self::PARAMETERS,
                layout.pretrain_position,
                Self::PATCH_WITH_POSITION,
                batches,
                patches,
                tile_nodes,
                hidden,
                (hidden as f32).sqrt(),
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn patch_attention_layout(
        &mut self,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
    ) -> Result<()> {
        self.arena
            .patches_to_attention_sequences_f32(
                Self::PATCH_WITH_POSITION,
                Self::ATTENTION_SEQUENCE,
                batches,
                patches,
                tile_nodes,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn encoder_attention(
        &mut self,
        layout: GraphParameterLayout,
        layer: usize,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        let projection = hidden * (hidden + 1);
        let rows = batches * patches * tile_nodes;
        self.affine_projection(
            Self::ATTENTION_SEQUENCE,
            Self::ENCODER_Q,
            layout.temporal_q + layer * projection,
            rows,
            hidden,
            hidden,
        )?;
        self.affine_projection(
            Self::ATTENTION_SEQUENCE,
            Self::ENCODER_K,
            layout.temporal_k + layer * projection,
            rows,
            hidden,
            hidden,
        )?;
        self.affine_projection(
            Self::ATTENTION_SEQUENCE,
            Self::ENCODER_V,
            layout.temporal_v + layer * projection,
            rows,
            hidden,
            hidden,
        )?;
        self.arena
            .attention_f32(
                Self::ENCODER_Q,
                Self::ENCODER_K,
                Self::ENCODER_V,
                Self::ENCODER_ATTENTION,
                batches * tile_nodes,
                patches,
                heads,
                hidden / heads,
                false,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Runs all four frozen MST encoder layers on contiguous device tensors.
    /// The input and final representation are both `[batch * nodes, patches,
    /// hidden]`; no activation crosses PCIe between blocks.  These parameters
    /// are frozen during supervised fitting, but the device computation is
    /// still the representation consumed by the CUDA LSTTN branches.
    fn frozen_encoder(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        if hidden == 0 || heads == 0 || hidden % heads != 0 {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN attention hidden size must divide evenly across heads".to_string(),
            ));
        }
        let rows = batches * patches * tile_nodes;
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;
        for layer in 0..4 {
            self.encoder_attention(layout, layer, batches, patches, tile_nodes, hidden, heads)?;
            self.affine_projection(
                Self::ENCODER_ATTENTION,
                Self::ENCODER_PROJECTED,
                layout.lsttn_transformer_out + layer * projection,
                rows,
                hidden,
                hidden,
            )?;
            self.arena
                .add_f32(
                    Self::ATTENTION_SEQUENCE,
                    Self::ENCODER_PROJECTED,
                    Self::ENCODER_RESIDUAL,
                    rows * hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            let norm = layout.lsttn_transformer_norm + layer * 4 * hidden;
            self.arena
                .layer_norm_parameter_slice_f32(
                    Self::ENCODER_RESIDUAL,
                    Self::PARAMETERS,
                    norm,
                    norm + hidden,
                    Self::ENCODER_NORMALIZED,
                    rows,
                    hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.affine_projection(
                Self::ENCODER_NORMALIZED,
                Self::ENCODER_FFN_EXPANDED,
                layout.lsttn_transformer_ffn + layer * ffn_width,
                rows,
                hidden,
                4 * hidden,
            )?;
            self.arena
                .relu_f32(
                    Self::ENCODER_FFN_EXPANDED,
                    Self::ENCODER_FFN_ACTIVATED,
                    rows * 4 * hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.affine_projection(
                Self::ENCODER_FFN_ACTIVATED,
                Self::ENCODER_FFN_CONTRACTED,
                layout.lsttn_transformer_ffn + layer * ffn_width + (hidden + 1) * 4 * hidden,
                rows,
                4 * hidden,
                hidden,
            )?;
            self.arena
                .add_f32(
                    Self::ENCODER_NORMALIZED,
                    Self::ENCODER_FFN_CONTRACTED,
                    Self::ENCODER_FFN_RESIDUAL,
                    rows * hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.arena
                .layer_norm_parameter_slice_f32(
                    Self::ENCODER_FFN_RESIDUAL,
                    Self::PARAMETERS,
                    norm + 2 * hidden,
                    norm + 3 * hidden,
                    Self::ATTENTION_SEQUENCE,
                    rows,
                    hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_forward_slots(
        &mut self,
        input: usize,
        output: usize,
        q: usize,
        k: usize,
        v: usize,
        attention: usize,
        projected: usize,
        residual: usize,
        normalized: usize,
        ffn_expanded: usize,
        ffn_activated: usize,
        ffn_contracted: usize,
        ffn_residual: usize,
        q_offset: usize,
        k_offset: usize,
        v_offset: usize,
        out_offset: usize,
        ffn_offset: usize,
        norm_offset: usize,
        sequences: usize,
        tokens: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        let rows = sequences * tokens;
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;
        self.affine_projection(input, q, q_offset, rows, hidden, hidden)?;
        self.affine_projection(input, k, k_offset, rows, hidden, hidden)?;
        self.affine_projection(input, v, v_offset, rows, hidden, hidden)?;
        self.arena
            .attention_f32(
                q,
                k,
                v,
                attention,
                sequences,
                tokens,
                heads,
                hidden / heads,
                false,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.affine_projection(attention, projected, out_offset, rows, hidden, hidden)?;
        self.arena
            .add_f32(input, projected, residual, rows * hidden)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .layer_norm_parameter_slice_f32(
                residual,
                Self::PARAMETERS,
                norm_offset,
                norm_offset + hidden,
                normalized,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.affine_projection(
            normalized,
            ffn_expanded,
            ffn_offset,
            rows,
            hidden,
            4 * hidden,
        )?;
        self.arena
            .relu_f32(ffn_expanded, ffn_activated, rows * 4 * hidden)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.affine_projection(
            ffn_activated,
            ffn_contracted,
            ffn_offset + (hidden + 1) * 4 * hidden,
            rows,
            4 * hidden,
            hidden,
        )?;
        self.arena
            .add_f32(normalized, ffn_contracted, ffn_residual, rows * hidden)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .layer_norm_parameter_slice_f32(
                ffn_residual,
                Self::PARAMETERS,
                norm_offset + 2 * hidden,
                norm_offset + 3 * hidden,
                output,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let _ = (projection, ffn_width);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_backward_slots(
        &mut self,
        input: usize,
        output_gradient: usize,
        input_gradient: usize,
        q: usize,
        k: usize,
        v: usize,
        attention: usize,
        projected: usize,
        residual: usize,
        normalized: usize,
        ffn_expanded: usize,
        ffn_activated: usize,
        ffn_contracted: usize,
        ffn_residual: usize,
        q_offset: usize,
        k_offset: usize,
        v_offset: usize,
        out_offset: usize,
        ffn_offset: usize,
        norm_offset: usize,
        sequences: usize,
        tokens: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        let rows = sequences * tokens;
        self.transformer_layer_forward_slots(
            input,
            Self::PRETRAIN_TEMP_F,
            q,
            k,
            v,
            attention,
            projected,
            residual,
            normalized,
            ffn_expanded,
            ffn_activated,
            ffn_contracted,
            ffn_residual,
            q_offset,
            k_offset,
            v_offset,
            out_offset,
            ffn_offset,
            norm_offset,
            sequences,
            tokens,
            hidden,
            heads,
        )?;
        self.arena
            .layer_norm_parameter_slice_backward_f32(
                ffn_residual,
                Self::PARAMETERS,
                norm_offset + 2 * hidden,
                norm_offset + 3 * hidden,
                output_gradient,
                Self::PRETRAIN_TEMP_A,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                ffn_activated,
                Self::PARAMETERS,
                ffn_offset + (hidden + 1) * 4 * hidden,
                ffn_offset + (hidden + 1) * 4 * hidden + 4 * hidden * hidden,
                Self::PRETRAIN_TEMP_A,
                Self::PRETRAIN_TEMP_B,
                Self::PARAMETER_GRADIENT,
                rows,
                4 * hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .relu_backward_f32(
                ffn_expanded,
                Self::PRETRAIN_TEMP_B,
                Self::PRETRAIN_TEMP_C,
                rows * 4 * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                normalized,
                Self::PARAMETERS,
                ffn_offset,
                ffn_offset + hidden * 4 * hidden,
                Self::PRETRAIN_TEMP_C,
                Self::PRETRAIN_TEMP_B,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                4 * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                Self::PRETRAIN_TEMP_A,
                Self::PRETRAIN_TEMP_B,
                Self::PRETRAIN_TEMP_C,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .layer_norm_parameter_slice_backward_f32(
                residual,
                Self::PARAMETERS,
                norm_offset,
                norm_offset + hidden,
                Self::PRETRAIN_TEMP_C,
                Self::PRETRAIN_TEMP_A,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                attention,
                Self::PARAMETERS,
                out_offset,
                out_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_A,
                Self::PRETRAIN_TEMP_B,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .attention_backward_f32(
                q,
                k,
                v,
                Self::PRETRAIN_TEMP_B,
                Self::PRETRAIN_TEMP_C,
                Self::PRETRAIN_TEMP_D,
                Self::PRETRAIN_TEMP_E,
                sequences,
                tokens,
                heads,
                hidden / heads,
                false,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                q_offset,
                q_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_C,
                input_gradient,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                k_offset,
                k_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_D,
                Self::PRETRAIN_TEMP_C,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                input_gradient,
                Self::PRETRAIN_TEMP_C,
                input_gradient,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                v_offset,
                v_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_E,
                Self::PRETRAIN_TEMP_C,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                input_gradient,
                Self::PRETRAIN_TEMP_C,
                input_gradient,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                input_gradient,
                Self::PRETRAIN_TEMP_A,
                input_gradient,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Materializes the learned directed adaptive CSR weights on-device. The
    /// row softmax writes directly into the graph-weight slot consumed by the
    /// two-hop diffusion stages, keeping the O(edges) adaptive graph sparse.
    fn adaptive_weights(&mut self, source_offset: usize, target_offset: usize) -> Result<()> {
        self.arena
            .csr_adaptive_logits_parameter_slice_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::PARAMETERS,
                source_offset,
                target_offset,
                Self::ADAPTIVE_LOGITS,
                self.nodes,
                self.adaptive_edges,
                10,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .csr_row_softmax_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_LOGITS,
                Self::ADAPTIVE_WEIGHTS,
                self.nodes,
                self.adaptive_edges,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Executes the four-stage exponentially dilated long-trend stack from
    /// the frozen patch encoder. The final tensor is `[batch, 1, tile_nodes,
    /// hidden]` for the normal LSTTN context sizes and remains resident for
    /// the daily/weekly/short fusion stage.
    fn long_branch(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
    ) -> Result<usize> {
        self.arena
            .transpose_node_time_f32(
                Self::ATTENTION_SEQUENCE,
                Self::LONG_TEMPORAL,
                batches,
                tile_nodes,
                patches,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut input = Self::LONG_TEMPORAL;
        let mut output = Self::LONG_STAGE_A;
        let mut times = patches;
        for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate() {
            let offset = layout.lsttn_dilated_convolution + layer * (3 * hidden * hidden + hidden);
            self.arena
                .lsttn_long_conv_pool_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    offset,
                    offset + 3 * hidden * hidden,
                    output,
                    batches,
                    times,
                    tile_nodes,
                    hidden,
                    dilation,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            times = times.div_ceil(2).div_ceil(2);
            input = output;
            output = if output == Self::LONG_STAGE_A {
                Self::LONG_STAGE_B
            } else {
                Self::LONG_STAGE_A
            };
        }
        debug_assert_eq!(times, 1);
        Ok(input)
    }

    fn short_input_projection(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        lookback: usize,
        tile_nodes: usize,
        channels: usize,
        recent_window: usize,
        hidden: usize,
        phase_offset: usize,
        periodicity: usize,
    ) -> Result<usize> {
        self.arena
            .lsttn_short_input_projection_parameter_slice_f32(
                Self::SUPERVISED_INPUT,
                Self::PARAMETERS,
                layout.lsttn_short_wave,
                layout.lsttn_short_wave + 2 * hidden,
                Self::SHORT_INPUT,
                batches,
                lookback,
                tile_nodes,
                channels,
                recent_window,
                hidden,
                phase_offset,
                periodicity,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn short_wave_layer_forward(
        &mut self,
        input: usize,
        skip: usize,
        next_skip: usize,
        layer: usize,
        dilation: usize,
        layout: GraphParameterLayout,
        batches: usize,
        times: usize,
        hidden: usize,
        training: bool,
        step: u64,
    ) -> Result<usize> {
        let nodes = self.nodes;
        let layer_width = 12 * hidden * hidden + 6 * hidden;
        let layers_start = layout.lsttn_short_wave + 3 * hidden;
        let base = layers_start + layer * layer_width;
        let filter = base;
        let gate = filter + 2 * hidden * hidden;
        let filter_bias = gate + 2 * hidden * hidden;
        let gate_bias = filter_bias + hidden;
        let graph_projection = gate_bias + hidden;
        let skip_projection = graph_projection + 7 * hidden * hidden + hidden;
        let norm = skip_projection + hidden * hidden + hidden;
        let out_times = times - dilation;
        let seq_batches = batches * out_times;
        let seq_len = seq_batches * nodes * hidden;
        self.arena
            .causal_conv2_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                filter,
                filter_bias,
                Self::SHORT_FILTER,
                batches,
                times,
                nodes,
                hidden,
                hidden,
                dilation,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .causal_conv2_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                gate,
                gate_bias,
                Self::SHORT_GATE,
                batches,
                times,
                nodes,
                hidden,
                hidden,
                dilation,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .gated_tanh_sigmoid_f32(
                Self::SHORT_FILTER,
                Self::SHORT_GATE,
                Self::SHORT_GATED,
                seq_len,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_GATED,
            Self::SHORT_SKIP_PROJECTION,
            skip_projection,
            seq_batches * nodes,
            hidden,
            hidden,
        )?;
        self.arena
            .add_tail_time_f32(
                skip,
                Self::SHORT_SKIP_PROJECTION,
                next_skip,
                batches,
                times,
                out_times,
                nodes,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_GATED,
                Self::SHORT_FORWARD_ONE,
                seq_batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_FORWARD_TWO,
                seq_batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_GATED,
                Self::SHORT_BACKWARD_ONE,
                seq_batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_BACKWARD_TWO,
                seq_batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_GATED,
                Self::SHORT_ADAPTIVE_ONE,
                seq_batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_ADAPTIVE_TWO,
                seq_batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_GATED,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_CONCAT_A,
                seq_batches * nodes,
                hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_A,
                Self::SHORT_FORWARD_TWO,
                Self::SHORT_CONCAT_B,
                seq_batches * nodes,
                2 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_B,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_CONCAT_C,
                seq_batches * nodes,
                3 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_C,
                Self::SHORT_BACKWARD_TWO,
                Self::SHORT_CONCAT_D,
                seq_batches * nodes,
                4 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_D,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_CONCAT_E,
                seq_batches * nodes,
                5 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_E,
                Self::SHORT_ADAPTIVE_TWO,
                Self::SHORT_CONCAT_F,
                seq_batches * nodes,
                6 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_CONCAT_F,
            Self::SHORT_GRAPH,
            graph_projection,
            seq_batches * nodes,
            7 * hidden,
            hidden,
        )?;
        self.arena
            .add_tail_time_f32(
                input,
                Self::SHORT_GRAPH,
                Self::SHORT_RESIDUAL,
                batches,
                times,
                out_times,
                nodes,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .deterministic_dropout_f32(
                Self::SHORT_RESIDUAL,
                Self::SHORT_INPUT,
                seq_len,
                step ^ ((layer as u64) << 48),
                0,
                training,
                0.3,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .batch_norm_channels_parameter_slice_f32(
                Self::SHORT_INPUT,
                Self::PARAMETERS,
                norm,
                norm + hidden,
                Self::SHORT_BATCH_STATS,
                Self::SHORT_NORMALIZED,
                batches,
                out_times,
                nodes,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        Ok(out_times)
    }

    /// Full-node Graph WaveNet executor. CSR diffusion is global by design;
    /// callers must not use this after a node-tile upload.
    #[allow(clippy::too_many_arguments)]
    fn short_branch(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        lookback: usize,
        channels: usize,
        recent_window: usize,
        hidden: usize,
        phase_offset: usize,
        periodicity: usize,
        training: bool,
        step: u64,
    ) -> Result<usize> {
        let nodes = self.nodes;
        let mut times = self.short_input_projection(
            layout,
            batches,
            lookback,
            nodes,
            channels,
            recent_window,
            hidden,
            phase_offset,
            periodicity,
        )?;
        self.adaptive_weights(
            layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_target,
        )?;
        self.arena
            .fill_f32(Self::SHORT_SKIP_A, batches * times * nodes * hidden, 0.0)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let mut input = Self::SHORT_INPUT;
        let mut skip = Self::SHORT_SKIP_A;
        let layer_width = 12 * hidden * hidden + 6 * hidden;
        let layers_start = layout.lsttn_short_wave + 3 * hidden;
        for (layer, dilation) in [1usize, 2, 1, 2, 1, 2, 1, 2].into_iter().enumerate() {
            let base = layers_start + layer * layer_width;
            let filter = base;
            let gate = filter + 2 * hidden * hidden;
            let filter_bias = gate + 2 * hidden * hidden;
            let gate_bias = filter_bias + hidden;
            let graph_projection = gate_bias + hidden;
            let skip_projection = graph_projection + 7 * hidden * hidden + hidden;
            let norm = skip_projection + hidden * hidden + hidden;
            let out_times = times - dilation;
            let seq_batches = batches * out_times;
            let seq_len = seq_batches * nodes * hidden;
            self.arena
                .causal_conv2_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    filter,
                    filter_bias,
                    Self::SHORT_FILTER,
                    batches,
                    times,
                    nodes,
                    hidden,
                    hidden,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .causal_conv2_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    gate,
                    gate_bias,
                    Self::SHORT_GATE,
                    batches,
                    times,
                    nodes,
                    hidden,
                    hidden,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .gated_tanh_sigmoid_f32(
                    Self::SHORT_FILTER,
                    Self::SHORT_GATE,
                    Self::SHORT_GATED,
                    seq_len,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.affine_projection(
                Self::SHORT_GATED,
                Self::SHORT_SKIP_PROJECTION,
                skip_projection,
                seq_batches * nodes,
                hidden,
                hidden,
            )?;
            let next_skip = if skip == Self::SHORT_SKIP_A {
                Self::SHORT_SKIP_B
            } else {
                Self::SHORT_SKIP_A
            };
            self.arena
                .add_tail_time_f32(
                    skip,
                    Self::SHORT_SKIP_PROJECTION,
                    next_skip,
                    batches,
                    times,
                    out_times,
                    nodes,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_FORWARD_ONE,
                    seq_batches,
                    nodes,
                    hidden,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_FORWARD_ONE,
                    Self::SHORT_FORWARD_TWO,
                    seq_batches,
                    nodes,
                    hidden,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_BACKWARD_ONE,
                    seq_batches,
                    nodes,
                    hidden,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_BACKWARD_ONE,
                    Self::SHORT_BACKWARD_TWO,
                    seq_batches,
                    nodes,
                    hidden,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_ADAPTIVE_ONE,
                    seq_batches,
                    nodes,
                    hidden,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_ADAPTIVE_ONE,
                    Self::SHORT_ADAPTIVE_TWO,
                    seq_batches,
                    nodes,
                    hidden,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_GATED,
                    Self::SHORT_FORWARD_ONE,
                    Self::SHORT_CONCAT_A,
                    seq_batches * nodes,
                    hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_A,
                    Self::SHORT_FORWARD_TWO,
                    Self::SHORT_CONCAT_B,
                    seq_batches * nodes,
                    2 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_B,
                    Self::SHORT_BACKWARD_ONE,
                    Self::SHORT_CONCAT_C,
                    seq_batches * nodes,
                    3 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_C,
                    Self::SHORT_BACKWARD_TWO,
                    Self::SHORT_CONCAT_D,
                    seq_batches * nodes,
                    4 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_D,
                    Self::SHORT_ADAPTIVE_ONE,
                    Self::SHORT_CONCAT_E,
                    seq_batches * nodes,
                    5 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_E,
                    Self::SHORT_ADAPTIVE_TWO,
                    Self::SHORT_CONCAT_F,
                    seq_batches * nodes,
                    6 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.affine_projection(
                Self::SHORT_CONCAT_F,
                Self::SHORT_GRAPH,
                graph_projection,
                seq_batches * nodes,
                7 * hidden,
                hidden,
            )?;
            self.arena
                .add_tail_time_f32(
                    input,
                    Self::SHORT_GRAPH,
                    Self::SHORT_RESIDUAL,
                    batches,
                    times,
                    out_times,
                    nodes,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .deterministic_dropout_f32(
                    Self::SHORT_RESIDUAL,
                    Self::SHORT_INPUT,
                    seq_len,
                    step ^ ((layer as u64) << 48),
                    0,
                    training,
                    0.3,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .batch_norm_channels_parameter_slice_f32(
                    Self::SHORT_INPUT,
                    Self::PARAMETERS,
                    norm,
                    norm + hidden,
                    Self::SHORT_BATCH_STATS,
                    Self::SHORT_NORMALIZED,
                    batches,
                    out_times,
                    nodes,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            input = Self::SHORT_NORMALIZED;
            skip = next_skip;
            times = out_times;
        }
        let end_one = layers_start + 8 * layer_width;
        let end_two = end_one + hidden * hidden + hidden;
        let rows = batches * times * nodes;
        self.arena
            .relu_f32(skip, Self::SHORT_FILTER, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_FILTER,
            Self::SHORT_GATE,
            end_one,
            rows,
            hidden,
            hidden,
        )?;
        self.arena
            .relu_f32(Self::SHORT_GATE, Self::SHORT_GATED, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_GATED,
            Self::SHORT_SKIP_PROJECTION,
            end_two,
            rows,
            hidden,
            hidden,
        )?;
        Ok(Self::SHORT_SKIP_PROJECTION)
    }

    fn periodic_feature(
        &mut self,
        layout: GraphParameterLayout,
        output: usize,
        batches: usize,
        patches: usize,
        hidden: usize,
        lag_patches: usize,
        seasonal: bool,
    ) -> Result<()> {
        if lag_patches >= patches {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN periodic lag exceeds frozen patch context".to_string(),
            ));
        }
        let nodes = self.nodes;
        self.arena
            .select_node_major_time_f32(
                Self::ATTENTION_SEQUENCE,
                Self::SHORT_FILTER,
                batches,
                nodes,
                patches,
                hidden,
                patches - lag_patches - 1,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let (source, target, projection) = if seasonal {
            (
                layout.lsttn_weekly_adaptive_source,
                layout.lsttn_weekly_adaptive_target,
                layout.lsttn_periodic_projection + (7 * hidden * hidden + hidden),
            )
        } else {
            (
                layout.lsttn_adaptive_source,
                layout.lsttn_adaptive_target,
                layout.lsttn_periodic_projection,
            )
        };
        self.adaptive_weights(source, target)?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FILTER,
                Self::SHORT_FORWARD_ONE,
                batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_FORWARD_TWO,
                batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::SHORT_BACKWARD_ONE,
                batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_BACKWARD_TWO,
                batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::SHORT_ADAPTIVE_ONE,
                batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_ADAPTIVE_TWO,
                batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_FILTER,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_CONCAT_A,
                batches * nodes,
                hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_A,
                Self::SHORT_FORWARD_TWO,
                Self::SHORT_CONCAT_B,
                batches * nodes,
                2 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_B,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_CONCAT_C,
                batches * nodes,
                3 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_C,
                Self::SHORT_BACKWARD_TWO,
                Self::SHORT_CONCAT_D,
                batches * nodes,
                4 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_D,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_CONCAT_E,
                batches * nodes,
                5 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_E,
                Self::SHORT_ADAPTIVE_TWO,
                Self::SHORT_CONCAT_F,
                batches * nodes,
                6 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let saved_input = if seasonal {
            Self::PERIODIC_SEASONAL_INPUT
        } else {
            Self::PERIODIC_SHORT_INPUT
        };
        self.arena
            .deterministic_dropout_f32(
                Self::SHORT_CONCAT_F,
                saved_input,
                batches * nodes * 7 * hidden,
                0,
                0,
                false,
                0.0,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            saved_input,
            output,
            projection,
            batches * nodes,
            7 * hidden,
            hidden,
        )
    }

    fn fuse_and_direct_output(
        &mut self,
        layout: GraphParameterLayout,
        long: usize,
        short: usize,
        batches: usize,
        hidden: usize,
        horizons: usize,
    ) -> Result<usize> {
        let rows = batches * self.nodes;
        self.arena
            .concat_channels_f32(
                long,
                Self::PERIODIC_SHORT,
                Self::FUSION_A,
                rows,
                hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::FUSION_A,
                Self::PERIODIC_SEASONAL,
                Self::FUSION_B,
                rows,
                2 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_B,
            Self::FUSION_C,
            layout.lsttn_fusion,
            rows,
            3 * hidden,
            hidden,
        )?;
        self.arena
            .relu_f32(Self::FUSION_C, Self::FUSION_D, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_D,
            Self::FUSION_C,
            layout.lsttn_fusion + (3 * hidden + 1) * hidden,
            rows,
            hidden,
            hidden,
        )?;
        self.arena
            .concat_channels_f32(short, Self::FUSION_C, Self::FUSION_A, rows, hidden, hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_A,
            Self::FUSION_OUTPUT,
            layout.lsttn_fusion + (4 * hidden * hidden + 2 * hidden),
            rows,
            2 * hidden,
            hidden,
        )?;
        self.arena
            .relu_f32(Self::FUSION_OUTPUT, Self::FUSION_OUTPUT, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_OUTPUT,
            Self::DIRECT_NODE_MAJOR,
            layout.output,
            rows,
            hidden,
            horizons,
        )?;
        self.arena
            .node_major_horizons_to_output_f32(
                Self::DIRECT_NODE_MAJOR,
                Self::FUSED_DIRECT_OUTPUT,
                batches,
                self.nodes,
                horizons,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        Ok(Self::FUSED_DIRECT_OUTPUT)
    }

    fn supervised_forward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        channels: usize,
        phase_offset: usize,
        training: bool,
    ) -> Result<usize> {
        if batches == 0 || batches > 32 {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN batches must be 1..=32".to_string(),
            ));
        }
        let layout = state.layout();
        let patch_width = (state.periodicity / 24).max(1);
        let patches = state.context_window / patch_width;
        let hidden = state.hidden;
        self.patch_embedding(
            layout,
            batches,
            state.context_window,
            self.nodes,
            channels,
            patch_width,
            hidden,
        )?;
        self.add_patch_positions(layout, batches, patches, self.nodes, hidden)?;
        self.patch_attention_layout(batches, patches, self.nodes, hidden)?;
        self.frozen_encoder(
            layout,
            batches,
            patches,
            self.nodes,
            hidden,
            state.attention_heads,
        )?;
        let long = self.long_branch(layout, batches, patches, self.nodes, hidden)?;
        let short = self.short_branch(
            layout,
            batches,
            state.context_window,
            channels,
            state.recent_window,
            hidden,
            phase_offset,
            state.periodicity,
            training,
            state.steps,
        )?;
        let short_lag = (if state.periodic_short_lag == 0 {
            state.periodicity
        } else {
            state.periodic_short_lag
        } / patch_width)
            .max(1);
        let seasonal_lag = (state.periodicity / patch_width).max(1);
        self.periodic_feature(
            layout,
            Self::PERIODIC_SHORT,
            batches,
            patches,
            hidden,
            short_lag,
            false,
        )?;
        self.periodic_feature(
            layout,
            Self::PERIODIC_SEASONAL,
            batches,
            patches,
            hidden,
            seasonal_lag,
            true,
        )?;
        self.fuse_and_direct_output(layout, long, short, batches, hidden, state.horizons)
    }

    fn direct_head_loss_and_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
    ) -> Result<()> {
        let layout = state.layout();
        let len = batches * state.horizons * self.nodes;
        self.begin_supervised_gradient()?;
        self.arena
            .weighted_inverse_scale_mae_loss_backward_f32(
                Self::FUSED_DIRECT_OUTPUT,
                Self::SUPERVISED_TARGET,
                Self::SUPERVISED_WEIGHT,
                Self::DIRECT_OUTPUT_GRADIENT,
                Self::SUPERVISED_LOSS,
                len,
                state.target_scale as f32,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .output_to_node_major_horizons_f32(
                Self::DIRECT_OUTPUT_GRADIENT,
                Self::DIRECT_NODE_MAJOR_GRADIENT,
                batches,
                self.nodes,
                state.horizons,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_OUTPUT,
                Self::PARAMETERS,
                layout.output,
                layout.output + state.hidden * state.horizons,
                Self::DIRECT_NODE_MAJOR_GRADIENT,
                Self::FUSION_REPRESENTATION_GRADIENT,
                Self::PARAMETER_GRADIENT,
                batches * self.nodes,
                state.hidden,
                state.horizons,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        Ok(())
    }

    fn supervised_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        channels: usize,
        phase_offset: usize,
        training: bool,
    ) -> Result<()> {
        self.direct_head_loss_and_backward(state, batches)?;
        self.fusion_backward(state, batches)?;
        self.short_branch_backward(state, batches, channels, phase_offset, training)?;
        self.long_branch_backward(state, batches)?;
        self.periodic_projection_backward(state, batches, false)?;
        self.periodic_projection_backward(state, batches, true)?;
        let patch_width = (state.periodicity / 24).max(1);
        let short_lag = (if state.periodic_short_lag == 0 {
            state.periodicity
        } else {
            state.periodic_short_lag
        } / patch_width)
            .max(1);
        let seasonal_lag = (state.periodicity / patch_width).max(1);
        self.periodic_graph_backward(state, batches, short_lag, false)?;
        self.periodic_graph_backward(state, batches, seasonal_lag, true)
    }

    fn fusion_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
    ) -> Result<()> {
        let layout = state.layout();
        let rows = batches * self.nodes;
        let h = state.hidden;
        self.arena
            .relu_backward_f32(
                Self::FUSION_OUTPUT,
                Self::FUSION_REPRESENTATION_GRADIENT,
                Self::FUSION_RELU_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_A,
                Self::PARAMETERS,
                layout.lsttn_fusion + (4 * h * h + 2 * h),
                layout.lsttn_fusion + (4 * h * h + 2 * h) + 2 * h * h,
                Self::FUSION_RELU_GRADIENT,
                Self::FUSION_CONCAT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                2 * h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::FUSION_CONCAT_GRADIENT,
                Self::SHORT_GRADIENT,
                Self::TREND_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_D,
                Self::PARAMETERS,
                layout.lsttn_fusion + (3 * h + 1) * h,
                layout.lsttn_fusion + (3 * h + 1) * h + h * h,
                Self::TREND_GRADIENT,
                Self::FUSION_SECOND_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .relu_backward_f32(
                Self::FUSION_D,
                Self::FUSION_SECOND_GRADIENT,
                Self::FUSION_FIRST_RELU_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_B,
                Self::PARAMETERS,
                layout.lsttn_fusion,
                layout.lsttn_fusion + 3 * h * h,
                Self::FUSION_FIRST_RELU_GRADIENT,
                Self::FUSION_INPUT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                3 * h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::FUSION_INPUT_GRADIENT,
                Self::LONG_GRADIENT,
                Self::PERIODIC_PAIR_GRADIENT,
                rows,
                h,
                2 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_PAIR_GRADIENT,
                Self::PERIODIC_SHORT_GRADIENT,
                Self::PERIODIC_SEASONAL_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))
    }

    fn periodic_projection_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        seasonal: bool,
    ) -> Result<()> {
        let layout = state.layout();
        let h = state.hidden;
        let projection =
            layout.lsttn_periodic_projection + if seasonal { 7 * h * h + h } else { 0 };
        let (input, output_gradient, input_gradient) = if seasonal {
            (
                Self::PERIODIC_SEASONAL_INPUT,
                Self::PERIODIC_SEASONAL_GRADIENT,
                Self::PERIODIC_SEASONAL_INPUT_GRADIENT,
            )
        } else {
            (
                Self::PERIODIC_SHORT_INPUT,
                Self::PERIODIC_SHORT_GRADIENT,
                Self::PERIODIC_SHORT_INPUT_GRADIENT,
            )
        };
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                projection,
                projection + 7 * h * h,
                output_gradient,
                input_gradient,
                Self::PARAMETER_GRADIENT,
                batches * self.nodes,
                7 * h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))
    }

    fn periodic_graph_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        lag_patches: usize,
        seasonal: bool,
    ) -> Result<()> {
        let layout = state.layout();
        let h = state.hidden;
        let patches = state.context_window / (state.periodicity / 24).max(1);
        let rows = batches * self.nodes;
        let (source, target, output, input_gradient) = if seasonal {
            (
                layout.lsttn_weekly_adaptive_source,
                layout.lsttn_weekly_adaptive_target,
                Self::PERIODIC_SEASONAL,
                Self::PERIODIC_SEASONAL_INPUT_GRADIENT,
            )
        } else {
            (
                layout.lsttn_adaptive_source,
                layout.lsttn_adaptive_target,
                Self::PERIODIC_SHORT,
                Self::PERIODIC_SHORT_INPUT_GRADIENT,
            )
        };
        self.periodic_feature(layout, output, batches, patches, h, lag_patches, seasonal)?;
        self.arena
            .split_channels_f32(
                input_gradient,
                Self::PERIODIC_GRAD_IDENTITY,
                Self::PERIODIC_TEMP_A,
                rows,
                h,
                6 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_GRAD_FORWARD_ONE,
                Self::PERIODIC_TEMP_B,
                rows,
                h,
                5 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_B,
                Self::PERIODIC_GRAD_FORWARD_TWO,
                Self::PERIODIC_TEMP_A,
                rows,
                h,
                4 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_GRAD_REVERSE_ONE,
                Self::PERIODIC_TEMP_B,
                rows,
                h,
                3 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_B,
                Self::PERIODIC_GRAD_REVERSE_TWO,
                Self::PERIODIC_TEMP_A,
                rows,
                h,
                2 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_GRAD_ADAPTIVE_ONE,
                Self::PERIODIC_GRAD_ADAPTIVE_TWO,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .csr_diffuse_backward_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FORWARD_ONE,
                Self::PERIODIC_GRAD_FORWARD_TWO,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_FORWARD_ONE,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_C,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_backward_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FILTER,
                Self::PERIODIC_TEMP_C,
                Self::PERIODIC_FORWARD_BASE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .csr_diffuse_backward_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_BACKWARD_ONE,
                Self::PERIODIC_GRAD_REVERSE_TWO,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_REVERSE_ONE,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_C,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_backward_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::PERIODIC_TEMP_C,
                Self::PERIODIC_REVERSE_BASE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .csr_diffuse_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_ADAPTIVE_ONE,
                Self::PERIODIC_GRAD_ADAPTIVE_TWO,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                batches,
                self.nodes,
                h,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_ADAPTIVE_ONE,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_C,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::PERIODIC_TEMP_C,
                Self::PERIODIC_ADAPTIVE_BASE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_row_softmax_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_WEIGHTS,
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                Self::PERIODIC_ADAPTIVE_LOGIT_GRADIENT,
                self.nodes,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_adaptive_logits_parameter_slice_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::PARAMETERS,
                source,
                target,
                Self::PERIODIC_ADAPTIVE_LOGIT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                self.nodes,
                self.adaptive_edges,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_IDENTITY,
                Self::PERIODIC_FORWARD_BASE_GRADIENT,
                Self::PERIODIC_BASE_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_BASE_GRADIENT,
                Self::PERIODIC_REVERSE_BASE_GRADIENT,
                Self::PERIODIC_BASE_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_BASE_GRADIENT,
                Self::PERIODIC_ADAPTIVE_BASE_GRADIENT,
                Self::PERIODIC_BASE_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))
    }

    fn recompute_long_input_for_layer(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        hidden: usize,
        layer_limit: usize,
    ) -> Result<(usize, usize)> {
        self.arena
            .transpose_node_time_f32(
                Self::ATTENTION_SEQUENCE,
                Self::LONG_TEMPORAL,
                batches,
                self.nodes,
                patches,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut input = Self::LONG_TEMPORAL;
        let mut output = Self::LONG_STAGE_A;
        let mut times = patches;
        for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate().take(layer_limit) {
            let offset = layout.lsttn_dilated_convolution + layer * (3 * hidden * hidden + hidden);
            self.arena
                .lsttn_long_conv_pool_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    offset,
                    offset + 3 * hidden * hidden,
                    output,
                    batches,
                    times,
                    self.nodes,
                    hidden,
                    dilation,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            times = times.div_ceil(2).div_ceil(2);
            input = output;
            output = if output == Self::LONG_STAGE_A {
                Self::LONG_STAGE_B
            } else {
                Self::LONG_STAGE_A
            };
        }
        Ok((input, times))
    }

    fn long_branch_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
    ) -> Result<()> {
        let layout = state.layout();
        let patch_width = (state.periodicity / 24).max(1);
        let patches = state.context_window / patch_width;
        let hidden = state.hidden;
        let mut output_gradient = Self::LONG_GRADIENT;
        for layer in (0..4).rev() {
            let (input, times) =
                self.recompute_long_input_for_layer(layout, batches, patches, hidden, layer)?;
            let input_gradient = if output_gradient == Self::LONG_BACKWARD_A {
                Self::LONG_BACKWARD_B
            } else {
                Self::LONG_BACKWARD_A
            };
            let offset = layout.lsttn_dilated_convolution + layer * (3 * hidden * hidden + hidden);
            let dilation = [1usize, 2, 4, 8][layer];
            self.arena
                .lsttn_long_conv_pool_parameter_slice_backward_f32(
                    input,
                    Self::PARAMETERS,
                    offset,
                    offset + 3 * hidden * hidden,
                    output_gradient,
                    input_gradient,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    times,
                    self.nodes,
                    hidden,
                    dilation,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            output_gradient = input_gradient;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn recompute_short_prefix(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        lookback: usize,
        channels: usize,
        recent_window: usize,
        hidden: usize,
        phase_offset: usize,
        periodicity: usize,
        training: bool,
        step: u64,
        layer_limit: usize,
    ) -> Result<(usize, usize, usize)> {
        let mut times = self.short_input_projection(
            layout,
            batches,
            lookback,
            self.nodes,
            channels,
            recent_window,
            hidden,
            phase_offset,
            periodicity,
        )?;
        self.adaptive_weights(
            layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_target,
        )?;
        self.arena
            .fill_f32(
                Self::SHORT_SKIP_A,
                batches * times * self.nodes * hidden,
                0.0,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let mut input = Self::SHORT_INPUT;
        let mut skip = Self::SHORT_SKIP_A;
        for (layer, dilation) in [1usize, 2, 1, 2, 1, 2, 1, 2]
            .into_iter()
            .enumerate()
            .take(layer_limit)
        {
            let next_skip = if skip == Self::SHORT_SKIP_A {
                Self::SHORT_SKIP_B
            } else {
                Self::SHORT_SKIP_A
            };
            times = self.short_wave_layer_forward(
                input, skip, next_skip, layer, dilation, layout, batches, times, hidden, training,
                step,
            )?;
            input = Self::SHORT_NORMALIZED;
            skip = next_skip;
        }
        Ok((input, skip, times))
    }

    #[allow(clippy::too_many_arguments)]
    fn short_branch_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        channels: usize,
        phase_offset: usize,
        training: bool,
    ) -> Result<()> {
        let layout = state.layout();
        let h = state.hidden;
        let nodes = self.nodes;
        let layer_width = 12 * h * h + 6 * h;
        let layers_start = layout.lsttn_short_wave + 3 * h;
        let (mut input, mut skip, mut times) = self.recompute_short_prefix(
            layout,
            batches,
            state.context_window,
            channels,
            state.recent_window,
            h,
            phase_offset,
            state.periodicity,
            training,
            state.steps,
            8,
        )?;
        let rows = batches * times * nodes;
        let end_one = layers_start + 8 * layer_width;
        let end_two = end_one + h * h + h;
        self.arena
            .relu_f32(skip, Self::SHORT_FILTER, rows * h)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(Self::SHORT_FILTER, Self::SHORT_GATE, end_one, rows, h, h)?;
        self.arena
            .relu_f32(Self::SHORT_GATE, Self::SHORT_GATED, rows * h)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_GATED,
            Self::SHORT_SKIP_PROJECTION,
            end_two,
            rows,
            h,
            h,
        )?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::SHORT_GATED,
                Self::PARAMETERS,
                end_two,
                end_two + h * h,
                Self::SHORT_GRADIENT,
                Self::SHORT_REVERSE_GATED_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .relu_backward_f32(
                Self::SHORT_GATE,
                Self::SHORT_REVERSE_GATED_GRADIENT,
                Self::SHORT_REVERSE_GRAPH_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::SHORT_FILTER,
                Self::PARAMETERS,
                end_one,
                end_one + h * h,
                Self::SHORT_REVERSE_GRAPH_GRADIENT,
                Self::SHORT_REVERSE_SKIP_GRADIENT_A,
                Self::PARAMETER_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .relu_backward_f32(
                skip,
                Self::SHORT_REVERSE_SKIP_GRADIENT_A,
                Self::SHORT_REVERSE_SKIP_GRADIENT_B,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let mut skip_gradient = Self::SHORT_REVERSE_SKIP_GRADIENT_B;
        let mut output_gradient = Self::SHORT_REVERSE_INPUT_GRADIENT_A;
        self.arena
            .fill_f32(output_gradient, batches * times * nodes * h, 0.0)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        for layer in (0..8).rev() {
            let dilation = [1usize, 2, 1, 2, 1, 2, 1, 2][layer];
            let (layer_input, layer_skip, layer_times) = self.recompute_short_prefix(
                layout,
                batches,
                state.context_window,
                channels,
                state.recent_window,
                h,
                phase_offset,
                state.periodicity,
                training,
                state.steps,
                layer,
            )?;
            let layer_next_skip = if layer_skip == Self::SHORT_SKIP_A {
                Self::SHORT_SKIP_B
            } else {
                Self::SHORT_SKIP_A
            };
            let out_times = self.short_wave_layer_forward(
                layer_input,
                layer_skip,
                layer_next_skip,
                layer,
                dilation,
                layout,
                batches,
                layer_times,
                h,
                training,
                state.steps,
            )?;
            debug_assert_eq!(out_times, times);
            let base = layers_start + layer * layer_width;
            let filter = base;
            let gate = filter + 2 * h * h;
            let filter_bias = gate + 2 * h * h;
            let gate_bias = filter_bias + h;
            let graph_projection = gate_bias + h;
            let skip_projection = graph_projection + 7 * h * h + h;
            let norm = skip_projection + h * h + h;
            let seq_rows = batches * out_times * nodes;
            self.arena
                .batch_norm_channels_parameter_slice_backward_f32(
                    Self::SHORT_INPUT,
                    Self::PARAMETERS,
                    norm,
                    norm + h,
                    Self::SHORT_BATCH_STATS,
                    output_gradient,
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    out_times,
                    nodes,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .deterministic_dropout_f32(
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_REVERSE_SPLIT_B,
                    batches * out_times * nodes * h,
                    state.steps ^ ((layer as u64) << 48),
                    0,
                    training,
                    0.3,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_tail_time_backward_f32(
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    Self::SHORT_REVERSE_GRAPH_GRADIENT,
                    batches,
                    layer_times,
                    out_times,
                    nodes,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .affine_backward_parameter_slice_f32(
                    Self::SHORT_CONCAT_F,
                    Self::PARAMETERS,
                    graph_projection,
                    graph_projection + 7 * h * h,
                    Self::SHORT_REVERSE_GRAPH_GRADIENT,
                    Self::SHORT_REVERSE_CONCAT_GRADIENT,
                    Self::PARAMETER_GRADIENT,
                    seq_rows,
                    7 * h,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_CONCAT_GRADIENT,
                    Self::SHORT_GRAPH_GRAD_IDENTITY,
                    Self::SHORT_REVERSE_SPLIT_A,
                    seq_rows,
                    h,
                    6 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_GRAPH_GRAD_FORWARD_ONE,
                    Self::SHORT_REVERSE_SPLIT_C,
                    seq_rows,
                    h,
                    5 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::SHORT_GRAPH_GRAD_FORWARD_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    seq_rows,
                    h,
                    4 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_GRAPH_GRAD_REVERSE_ONE,
                    Self::SHORT_REVERSE_SPLIT_A,
                    seq_rows,
                    h,
                    3 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_GRAPH_GRAD_REVERSE_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    seq_rows,
                    h,
                    2 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_ONE,
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_TWO,
                    seq_rows,
                    h,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .deterministic_dropout_f32(
                    Self::SHORT_GRAPH_GRAD_IDENTITY,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                    0,
                    0,
                    false,
                    0.0,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_FORWARD_ONE,
                    Self::SHORT_GRAPH_GRAD_FORWARD_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_GRAPH_GRAD_FORWARD_ONE,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_SPLIT_B,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_BACKWARD_ONE,
                    Self::SHORT_GRAPH_GRAD_REVERSE_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_GRAPH_GRAD_REVERSE_ONE,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_SPLIT_F,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_REVERSE_SPLIT_F,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_ADAPTIVE_ONE,
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_ONE,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_SPLIT_A,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_row_softmax_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    nodes,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_adaptive_logits_parameter_slice_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::PARAMETERS,
                    layout.lsttn_short_adaptive_source,
                    layout.lsttn_short_adaptive_target,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    Self::PARAMETER_GRADIENT,
                    nodes,
                    self.adaptive_edges,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .affine_backward_parameter_slice_f32(
                    Self::SHORT_GATED,
                    Self::PARAMETERS,
                    skip_projection,
                    skip_projection + h * h,
                    skip_gradient,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::PARAMETER_GRADIENT,
                    seq_rows,
                    h,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_tail_time_backward_f32(
                    skip_gradient,
                    Self::SHORT_REVERSE_SKIP_GRADIENT_A,
                    Self::SHORT_REVERSE_SPLIT_C,
                    batches,
                    layer_times,
                    out_times,
                    nodes,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .gated_tanh_sigmoid_backward_f32(
                    Self::SHORT_FILTER,
                    Self::SHORT_GATE,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_FILTER_GRADIENT,
                    Self::SHORT_REVERSE_GATE_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .causal_conv2_parameter_slice_backward_f32(
                    layer_input,
                    Self::PARAMETERS,
                    filter,
                    filter_bias,
                    Self::SHORT_REVERSE_FILTER_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    layer_times,
                    nodes,
                    h,
                    h,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .causal_conv2_parameter_slice_backward_f32(
                    layer_input,
                    Self::PARAMETERS,
                    gate,
                    gate_bias,
                    Self::SHORT_REVERSE_GATE_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    layer_times,
                    nodes,
                    h,
                    h,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    batches * layer_times * nodes * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    batches * layer_times * nodes * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            output_gradient = Self::SHORT_REVERSE_INPUT_GRADIENT_B;
            skip_gradient = Self::SHORT_REVERSE_SKIP_GRADIENT_A;
            input = layer_input;
            skip = layer_skip;
            times = layer_times;
        }
        let _ = (input, skip, times);
        Ok(())
    }

    /// Starts a deterministic reduced-batch gradient accumulation and applies
    /// its single AdamW update. Every caller contributes into the same
    /// resident vector between these calls; checkpoints later copy only the
    /// portable parameter/moment vectors back to Rust.
    fn begin_supervised_gradient(&mut self) -> Result<()> {
        let len = self
            .arena
            .capacity_f32(Self::PARAMETERS)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .fill_f32(Self::PARAMETER_GRADIENT, len, 0.0)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn adamw_supervised_step(
        &mut self,
        step: u64,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let len = self
            .arena
            .capacity_f32(Self::PARAMETERS)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .clip_gradient_l2_f32(Self::PARAMETER_GRADIENT, Self::GRADIENT_NORM, len, 5.0)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .adamw_step_f32(
                Self::PARAMETERS,
                Self::FIRST_MOMENT,
                Self::SECOND_MOMENT,
                Self::PARAMETER_GRADIENT,
                len,
                step,
                learning_rate as f32,
                weight_decay as f32,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn portable_adamw_step(
        &mut self,
        state: &mut TrainableGraphTransformerState,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let mut gradients = vec![0.0_f32; state.parameters.len()];
        self.arena
            .download_f32(Self::PARAMETER_GRADIENT, &mut gradients)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut gradients = gradients.into_iter().map(f64::from).collect::<Vec<_>>();
        clip_gradient_norm(&mut gradients, 5.0);
        state.adamw_step(&gradients, learning_rate, weight_decay, None)?;
        let to_f32 = |values: &[f64], label: &str| -> Result<Vec<f32>> {
            values
                .iter()
                .map(|value| {
                    let value = *value as f32;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(GeoStError::InvalidFrame(format!(
                            "LSTTN CUDA {label} contains a non-finite f32 value"
                        )))
                    }
                })
                .collect()
        };
        self.arena
            .upload_f32(Self::PARAMETERS, &to_f32(&state.parameters, "parameters")?)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .upload_f32(
                Self::FIRST_MOMENT,
                &to_f32(&state.first_moment, "first moments")?,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .upload_f32(
                Self::SECOND_MOMENT,
                &to_f32(&state.second_moment, "second moments")?,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Packs the real supervised contract without per-node host dispatch:
    /// inputs are row-major `[batch, lookback, nodes, channels]`, targets
    /// are `[batch, horizon, nodes]`. `windows` and `targets` retain the
    /// native graph-frame row representation at the API boundary only.
    fn upload_supervised_batch(
        &mut self,
        windows: &[&[Vec<f64>]],
        targets: &[&[Vec<f64>]],
        weights: &[&[Vec<f64>]],
        channels: usize,
    ) -> Result<()> {
        self.upload_supervised_node_tile(windows, targets, weights, channels, 0..self.nodes)
    }

    /// Physical node-tile upload for the global logical batch.  Rows remain
    /// ordered as the original global node axis; the range is merely a
    /// checkpointing boundary and never changes graph/node identity.
    fn upload_supervised_node_tile(
        &mut self,
        windows: &[&[Vec<f64>]],
        targets: &[&[Vec<f64>]],
        weights: &[&[Vec<f64>]],
        channels: usize,
        node_range: std::ops::Range<usize>,
    ) -> Result<()> {
        if windows.is_empty()
            || windows.len() > 32
            || windows.len() != targets.len()
            || targets.len() != weights.len()
            || channels == 0
            || node_range.start >= node_range.end
            || node_range.end > self.nodes
        {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN supervised batches require 1..=32 matching windows and targets"
                    .to_string(),
            ));
        }
        let lookback = windows[0].len();
        let horizon = targets[0].len();
        if lookback == 0 || horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN supervised windows and targets must be non-empty".to_string(),
            ));
        }
        let nodes = windows[0].first().map_or(0, |row| row.len() / channels);
        if nodes != self.nodes {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN input channels do not match the resident graph nodes".to_string(),
            ));
        }
        let finite_f32 = |value: f64| -> Result<f32> {
            let value = value as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(GeoStError::InvalidFrame(
                    "CUDA LSTTN batch contains a non-finite f32 value".to_string(),
                ))
            }
        };
        let tile_nodes = node_range.len();
        let mut input = Vec::with_capacity(windows.len() * lookback * tile_nodes * channels);
        let mut target = Vec::with_capacity(targets.len() * horizon * tile_nodes);
        let mut weight = Vec::with_capacity(targets.len() * horizon * tile_nodes);
        for ((window, target_rows), weight_rows) in windows.iter().zip(targets).zip(weights) {
            if window.len() != lookback
                || target_rows.len() != horizon
                || window.iter().any(|row| row.len() != nodes * channels)
                || target_rows.iter().any(|row| row.len() != nodes)
                || weight_rows.len() != horizon
                || weight_rows.iter().any(|row| row.len() != nodes)
            {
                return Err(GeoStError::InvalidFrame(
                    "CUDA LSTTN batch has inconsistent tensor rows".to_string(),
                ));
            }
            for row in window.iter() {
                for node in node_range.clone() {
                    for channel in 0..channels {
                        input.push(finite_f32(row[node * channels + channel])?);
                    }
                }
            }
            for row in target_rows.iter() {
                for node in node_range.clone() {
                    target.push(finite_f32(row[node])?);
                }
            }
            for row in weight_rows.iter() {
                for node in node_range.clone() {
                    weight.push(finite_f32(row[node])?);
                }
            }
        }
        self.arena
            .upload_f32(Self::SUPERVISED_INPUT, &input)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .upload_f32(Self::SUPERVISED_TARGET, &target)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .upload_f32(Self::SUPERVISED_WEIGHT, &weight)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .reserve_f32(Self::DIRECT_OUTPUT, target.len())
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn upload_pretraining_window(&mut self, window: &[Vec<f64>], channels: usize) -> Result<()> {
        if window.is_empty() || channels == 0 || window.iter().any(|row| row.len() != self.nodes) {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN pretraining window must match the resident graph".to_string(),
            ));
        }
        let mut input = Vec::with_capacity(window.len() * self.nodes * channels);
        for row in window {
            for value in row {
                let value = *value as f32;
                if !value.is_finite() {
                    return Err(GeoStError::InvalidFrame(
                        "CUDA LSTTN pretraining window contains a non-finite f32 value".to_string(),
                    ));
                }
                input.push(value);
            }
        }
        self.arena
            .upload_f32(Self::SUPERVISED_INPUT, &input)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn pretraining_encoder_prefix(
        &mut self,
        layout: GraphParameterLayout,
        layers: usize,
        visible: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<usize> {
        if layers == 0 {
            return Ok(Self::PRETRAIN_VISIBLE_TOKENS);
        }
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;
        let mut input = Self::PRETRAIN_VISIBLE_TOKENS;
        let mut output = Self::PRETRAIN_LAYER_A;
        for layer in 0..layers {
            self.transformer_layer_forward_slots(
                input,
                output,
                Self::ENCODER_Q,
                Self::ENCODER_K,
                Self::ENCODER_V,
                Self::ENCODER_ATTENTION,
                Self::ENCODER_PROJECTED,
                Self::ENCODER_RESIDUAL,
                Self::ENCODER_NORMALIZED,
                Self::ENCODER_FFN_EXPANDED,
                Self::ENCODER_FFN_ACTIVATED,
                Self::ENCODER_FFN_CONTRACTED,
                Self::ENCODER_FFN_RESIDUAL,
                layout.temporal_q + layer * projection,
                layout.temporal_k + layer * projection,
                layout.temporal_v + layer * projection,
                layout.lsttn_transformer_out + layer * projection,
                layout.lsttn_transformer_ffn + layer * ffn_width,
                layout.lsttn_transformer_norm + layer * 4 * hidden,
                self.nodes,
                visible,
                hidden,
                heads,
            )?;
            input = output;
            output = if output == Self::PRETRAIN_LAYER_A {
                Self::PRETRAIN_LAYER_B
            } else {
                Self::PRETRAIN_LAYER_A
            };
        }
        Ok(input)
    }

    fn cuda_train_masked_subseries_reconstruction(
        &mut self,
        state: &mut TrainableGraphTransformerState,
        window: &[Vec<f64>],
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<f64> {
        let patch_width = (state.periodicity / 24).max(1);
        if !window.len().is_multiple_of(patch_width) {
            return Err(GeoStError::InvalidFrame(format!(
                "LSTTN long-history window length {} must be divisible by patch width {}",
                window.len(),
                patch_width
            )));
        }
        let patches = window.len() / patch_width;
        if patches < 2 {
            return Err(GeoStError::InvalidFrame(
                "LSTTN masked-subseries pretraining requires at least two patches".to_string(),
            ));
        }
        let masked = masked_patch_indices(patches, state.steps);
        let visible = (0..patches)
            .filter(|patch| !masked.contains(patch))
            .collect::<Vec<_>>();
        let masked_u32 = masked
            .iter()
            .map(|patch| u32::try_from(*patch))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                GeoStError::InvalidFrame("CUDA LSTTN patch index exceeds u32".to_string())
            })?;
        let visible_u32 = visible
            .iter()
            .map(|patch| u32::try_from(*patch))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                GeoStError::InvalidFrame("CUDA LSTTN patch index exceeds u32".to_string())
            })?;
        let layout = state.layout();
        let hidden = state.hidden;
        let heads = state.attention_heads;
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / hidden;
        let scale = (hidden as f32).sqrt();
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;

        self.upload_pretraining_window(window, 1)?;
        self.arena
            .upload_u32(Self::VISIBLE_PATCH_INDICES, &visible_u32)
            .and_then(|_| {
                self.arena
                    .upload_u32(Self::MASKED_PATCH_INDICES, &masked_u32)
            })
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.begin_supervised_gradient()?;
        self.patch_embedding(layout, 1, window.len(), self.nodes, 1, patch_width, hidden)?;
        self.add_patch_positions(layout, 1, patches, self.nodes, hidden)?;
        self.patch_attention_layout(1, patches, self.nodes, hidden)?;
        self.arena
            .gather_patch_tokens_f32(
                Self::ATTENTION_SEQUENCE,
                Self::VISIBLE_PATCH_INDICES,
                Self::PRETRAIN_VISIBLE_TOKENS,
                1,
                self.nodes,
                patches,
                visible.len(),
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let encoded = self.pretraining_encoder_prefix(layout, 4, visible.len(), hidden, heads)?;
        self.affine_projection(
            encoded,
            Self::PRETRAIN_LAYER_GRAD_A,
            layout.lsttn_encoder_decoder,
            self.nodes * visible.len(),
            hidden,
            hidden,
        )?;
        self.arena
            .assemble_masked_decoder_tokens_f32(
                Self::PRETRAIN_LAYER_GRAD_A,
                Self::MASKED_PATCH_INDICES,
                Self::PARAMETERS,
                layout.pretrain_mask_token,
                layout.pretrain_position,
                Self::PRETRAIN_DECODER_INPUT,
                1,
                self.nodes,
                visible.len(),
                masked.len(),
                hidden,
                position_count,
                scale,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.transformer_layer_forward_slots(
            Self::PRETRAIN_DECODER_INPUT,
            Self::PRETRAIN_DECODER_OUTPUT,
            Self::PRETRAIN_DECODER_Q,
            Self::PRETRAIN_DECODER_K,
            Self::PRETRAIN_DECODER_V,
            Self::PRETRAIN_DECODER_ATTENTION,
            Self::PRETRAIN_DECODER_PROJECTED,
            Self::PRETRAIN_DECODER_RESIDUAL,
            Self::PRETRAIN_DECODER_NORMALIZED,
            Self::PRETRAIN_DECODER_FFN_EXPANDED,
            Self::PRETRAIN_DECODER_FFN_ACTIVATED,
            Self::PRETRAIN_DECODER_FFN_CONTRACTED,
            Self::PRETRAIN_DECODER_FFN_RESIDUAL,
            layout.lsttn_decoder_q,
            layout.lsttn_decoder_k,
            layout.lsttn_decoder_v,
            layout.lsttn_decoder_out,
            layout.lsttn_decoder_ffn,
            layout.lsttn_decoder_norm,
            self.nodes,
            visible.len() + masked.len(),
            hidden,
            heads,
        )?;
        self.arena
            .masked_patch_reconstruction_loss_backward_f32(
                Self::PRETRAIN_DECODER_OUTPUT,
                Self::SUPERVISED_INPUT,
                Self::MASKED_PATCH_INDICES,
                Self::PARAMETERS,
                layout.pretrain_decoder,
                layout.pretrain_decoder + patch_width * hidden,
                Self::PRETRAIN_CONTEXT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                Self::PRETRAIN_LOSS,
                1,
                window.len(),
                self.nodes,
                1,
                visible.len(),
                masked.len(),
                patch_width,
                hidden,
                state.normalized_zero as f32,
                state.target_scale as f32,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.transformer_layer_backward_slots(
            Self::PRETRAIN_DECODER_INPUT,
            Self::PRETRAIN_CONTEXT_GRADIENT,
            Self::PRETRAIN_DECODER_INPUT_GRADIENT,
            Self::PRETRAIN_DECODER_Q,
            Self::PRETRAIN_DECODER_K,
            Self::PRETRAIN_DECODER_V,
            Self::PRETRAIN_DECODER_ATTENTION,
            Self::PRETRAIN_DECODER_PROJECTED,
            Self::PRETRAIN_DECODER_RESIDUAL,
            Self::PRETRAIN_DECODER_NORMALIZED,
            Self::PRETRAIN_DECODER_FFN_EXPANDED,
            Self::PRETRAIN_DECODER_FFN_ACTIVATED,
            Self::PRETRAIN_DECODER_FFN_CONTRACTED,
            Self::PRETRAIN_DECODER_FFN_RESIDUAL,
            layout.lsttn_decoder_q,
            layout.lsttn_decoder_k,
            layout.lsttn_decoder_v,
            layout.lsttn_decoder_out,
            layout.lsttn_decoder_ffn,
            layout.lsttn_decoder_norm,
            self.nodes,
            visible.len() + masked.len(),
            hidden,
            heads,
        )?;
        self.arena
            .assemble_masked_decoder_tokens_backward_f32(
                Self::PRETRAIN_DECODER_INPUT_GRADIENT,
                Self::MASKED_PATCH_INDICES,
                Self::PARAMETER_GRADIENT,
                layout.pretrain_mask_token,
                layout.pretrain_position,
                Self::PRETRAIN_VISIBLE_GRADIENT,
                1,
                self.nodes,
                visible.len(),
                masked.len(),
                hidden,
                position_count,
                scale,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                encoded,
                Self::PARAMETERS,
                layout.lsttn_encoder_decoder,
                layout.lsttn_encoder_decoder + hidden * hidden,
                Self::PRETRAIN_VISIBLE_GRADIENT,
                Self::PRETRAIN_ENCODER_OUTPUT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                self.nodes * visible.len(),
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut output_gradient = Self::PRETRAIN_ENCODER_OUTPUT_GRADIENT;
        for layer in (0..4).rev() {
            let input =
                self.pretraining_encoder_prefix(layout, layer, visible.len(), hidden, heads)?;
            let input_gradient = if output_gradient == Self::PRETRAIN_LAYER_GRAD_A {
                Self::PRETRAIN_LAYER_GRAD_B
            } else {
                Self::PRETRAIN_LAYER_GRAD_A
            };
            self.transformer_layer_backward_slots(
                input,
                output_gradient,
                input_gradient,
                Self::ENCODER_Q,
                Self::ENCODER_K,
                Self::ENCODER_V,
                Self::ENCODER_ATTENTION,
                Self::ENCODER_PROJECTED,
                Self::ENCODER_RESIDUAL,
                Self::ENCODER_NORMALIZED,
                Self::ENCODER_FFN_EXPANDED,
                Self::ENCODER_FFN_ACTIVATED,
                Self::ENCODER_FFN_CONTRACTED,
                Self::ENCODER_FFN_RESIDUAL,
                layout.temporal_q + layer * projection,
                layout.temporal_k + layer * projection,
                layout.temporal_v + layer * projection,
                layout.lsttn_transformer_out + layer * projection,
                layout.lsttn_transformer_ffn + layer * ffn_width,
                layout.lsttn_transformer_norm + layer * 4 * hidden,
                self.nodes,
                visible.len(),
                hidden,
                heads,
            )?;
            output_gradient = input_gradient;
        }
        self.arena
            .gather_patch_tokens_backward_f32(
                output_gradient,
                Self::VISIBLE_PATCH_INDICES,
                Self::PRETRAIN_SEQUENCE_GRADIENT,
                1,
                self.nodes,
                patches,
                visible.len(),
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .attention_sequences_to_patches_f32(
                Self::PRETRAIN_SEQUENCE_GRADIENT,
                Self::PRETRAIN_PATCH_LAYOUT_GRADIENT,
                1,
                patches,
                self.nodes,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_patch_positions_backward_f32(
                Self::PRETRAIN_PATCH_LAYOUT_GRADIENT,
                Self::PRETRAIN_PATCH_EMBEDDING_GRADIENT,
                Self::PARAMETER_GRADIENT,
                layout.pretrain_position,
                1,
                patches,
                self.nodes,
                hidden,
                scale,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .patch_embedding_parameter_slice_backward_f32(
                Self::SUPERVISED_INPUT,
                Self::PRETRAIN_PATCH_EMBEDDING_GRADIENT,
                Self::PARAMETER_GRADIENT,
                layout.lsttn_patch_embedding,
                layout.lsttn_patch_embedding + patch_width * hidden,
                1,
                window.len(),
                self.nodes,
                1,
                patch_width,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.portable_adamw_step(state, learning_rate, weight_decay)?;
        let mut loss = [0.0_f32; 2];
        self.arena
            .download_f32(Self::PRETRAIN_LOSS, &mut loss)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        Ok(f64::from(loss[0]))
    }

    /// Copies only portable f32 optimizer/model state back into the native
    /// checkpoint representation. CUDA buffers and contexts remain owned by
    /// the executor and are recreated from this state on resume.
    fn synchronize_portable_state(&self, state: &mut TrainableGraphTransformerState) -> Result<()> {
        if state.parameters.len() != state.first_moment.len()
            || state.parameters.len() != state.second_moment.len()
        {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN checkpoint state has inconsistent optimizer lengths".to_string(),
            ));
        }
        let len = state.parameters.len();
        let mut parameters = vec![0.0_f32; len];
        let mut first = vec![0.0_f32; len];
        let mut second = vec![0.0_f32; len];
        self.arena
            .download_f32(Self::PARAMETERS, &mut parameters)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .download_f32(Self::FIRST_MOMENT, &mut first)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .download_f32(Self::SECOND_MOMENT, &mut second)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        state.parameters = parameters.into_iter().map(f64::from).collect();
        state.first_moment = first.into_iter().map(f64::from).collect();
        state.second_moment = second.into_iter().map(f64::from).collect();
        Ok(())
    }

    /// LSTTN pretrains its masked-subseries transformer, then keeps that
    /// encoder frozen during direct forecasting.  The CUDA executor owns the
    /// gradient buffer, so enforce the same parameter contract there before
    /// AdamW rather than allowing a CUDA-only fine-tuning variant.
    fn freeze_pretrained_transformer_gradients(
        &mut self,
        state: &TrainableGraphTransformerState,
    ) -> Result<()> {
        let layout = state.layout();
        let mut gradients = vec![0.0_f32; state.parameters.len()];
        self.arena
            .download_f32(Self::PARAMETER_GRADIENT, &mut gradients)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        gradients[layout.input..layout.spatial_q].fill(0.0);
        gradients[layout.pretrain_mask_token..layout.total].fill(0.0);
        self.arena
            .upload_f32(Self::PARAMETER_GRADIENT, &gradients)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn mean_supervised_loss(&self) -> Result<f64> {
        // The CUDA loss kernel reduces all examples, horizons, and nodes into
        // one MAE scalar (plus an internal valid-value count).
        let mut losses = [0.0_f32; 2];
        self.arena
            .download_f32(Self::SUPERVISED_LOSS, &mut losses)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        Ok(f64::from(losses[0]))
    }
}

#[derive(Clone, Copy)]
struct GraphParameterLayout {
    input: usize,
    time2vec_frequency: usize,
    time2vec_phase: usize,
    in_degree_embedding: usize,
    out_degree_embedding: usize,
    temporal_q: usize,
    temporal_k: usize,
    temporal_v: usize,
    spatial_q: usize,
    spatial_k: usize,
    spatial_v: usize,
    shortest_path_bias: usize,
    router: usize,
    spatial_router: usize,
    spatial_expert_heads: usize,
    expert_heads: usize,
    recurrence: usize,
    lsttn_dilated_convolution: usize,
    lsttn_short_wave: usize,
    stgformer_pointwise: usize,
    lsttn_adaptive_source: usize,
    lsttn_adaptive_target: usize,
    lsttn_weekly_adaptive_source: usize,
    lsttn_weekly_adaptive_target: usize,
    lsttn_short_adaptive_source: usize,
    lsttn_short_adaptive_target: usize,
    lsttn_periodic_projection: usize,
    lsttn_fusion: usize,
    graphon_nodes: usize,
    graphon_time: usize,
    output: usize,
    pretrain_mask_token: usize,
    pretrain_position: usize,
    pretrain_decoder: usize,
    lsttn_patch_embedding: usize,
    lsttn_transformer_ffn: usize,
    lsttn_transformer_norm: usize,
    lsttn_transformer_out: usize,
    lsttn_encoder_decoder: usize,
    lsttn_decoder_q: usize,
    lsttn_decoder_k: usize,
    lsttn_decoder_v: usize,
    lsttn_decoder_out: usize,
    lsttn_decoder_ffn: usize,
    lsttn_decoder_norm: usize,
    total: usize,
}

impl GraphParameterLayout {
    fn new(
        nodes: usize,
        hidden: usize,
        horizons: usize,
        experts: usize,
        graph_order: usize,
        periodicity: usize,
        context_window: usize,
    ) -> Self {
        let input = 0;
        // Seven numeric inputs (local signal, graph signal, learned Time2Vec,
        // daily and weekly phase, and independent in/out-degree encodings),
        // followed by a learned bias per hidden channel.
        let time2vec_frequency = input + hidden * 8;
        let time2vec_phase = time2vec_frequency + hidden;
        let in_degree_embedding = time2vec_phase + hidden;
        let out_degree_embedding = in_degree_embedding + (nodes + 1) * hidden;
        let temporal_q = out_degree_embedding + (nodes + 1) * hidden;
        // `tape_linear` stores a full matrix plus one bias per output.  Keep
        // every Q/K/V range disjoint: sharing a trailing bias with the next
        // projection would silently couple attention parameters.
        let projection = hidden * (hidden + 1);
        // STGormer uses three temporal and three spatial transformer blocks.
        // Reserve one independent Q/K/V projection per block; the other
        // profiles use the first stage of these generic attention ranges.
        let transformer_blocks = 4;
        let temporal_k = temporal_q + transformer_blocks * projection;
        let temporal_v = temporal_k + transformer_blocks * projection;
        let spatial_q = temporal_v + transformer_blocks * projection;
        let spatial_k = spatial_q + transformer_blocks * projection;
        let spatial_v = spatial_k + transformer_blocks * projection;
        let shortest_path_bias = spatial_v + transformer_blocks * projection;
        let router = shortest_path_bias + nodes + 1;
        // STGormer keeps independent temporal and spatial routers and
        // expert FNNs.  `router`/`expert_heads` name the temporal path for
        // backwards readability; the adjacent ranges are the spatial path.
        let spatial_router = router + experts * (hidden + 1);
        let spatial_expert_heads = spatial_router + experts * (hidden + 1);
        let expert_heads = spatial_expert_heads + experts * horizons * (hidden + 1);
        // Long/short fusion, recurrent gates, and high-order propagation
        // gates share this learned block; each profile uses the appropriate
        // subset in its forward graph.
        let recurrence = expert_heads + experts * horizons * (hidden + 1);
        let lsttn_dilated_convolution = recurrence + (graph_order + 6) * hidden;
        // Two gated temporal-convolution blocks for LSTTN's short-term
        // Graph WaveNet branch.  Each owns a filter and gate convolution,
        // their biases, and a post-adaptive-graph channel projection.
        let lsttn_short_wave = lsttn_dilated_convolution + 4 * (3 * hidden * hidden + hidden);
        let lsttn_short_layer = 12 * hidden * hidden + 6 * hidden;
        let stgformer_pointwise = lsttn_short_wave
            + 2 * hidden
            + hidden
            + 8 * lsttn_short_layer
            + 2 * (hidden * hidden + hidden);
        let lsttn_adaptive_source = stgformer_pointwise + graph_order * hidden * (hidden + 1);
        let lsttn_adaptive_target = lsttn_adaptive_source + nodes * 10;
        let lsttn_weekly_adaptive_source = lsttn_adaptive_target + nodes * 10;
        let lsttn_weekly_adaptive_target = lsttn_weekly_adaptive_source + nodes * 10;
        let lsttn_short_adaptive_source = lsttn_weekly_adaptive_target + nodes * 10;
        let lsttn_short_adaptive_target = lsttn_short_adaptive_source + nodes * 10;
        let lsttn_periodic_projection = lsttn_short_adaptive_target + nodes * 10;
        // LSTTN has three explicit MLP stages: long-trend/day/week fusion,
        // a second trend-seasonality projection, and short/long fusion.
        let lsttn_fusion = lsttn_periodic_projection + 2 * (7 * hidden * hidden + hidden);
        let graphon_nodes = lsttn_fusion + (6 * hidden * hidden + 3 * hidden);
        let graphon_time = graphon_nodes + experts * nodes;
        let output = graphon_time + experts * hidden;
        let pretrain_mask_token = output + horizons * (hidden + 1);
        // Allocate an independent learned position for every patch in the
        // configured long context. This keeps long-horizon positions distinct
        // instead of aliasing later weeks onto an old fixed-size table.
        let patch_width = (periodicity / 24).max(1);
        let pretrain_positions = context_window.div_ceil(patch_width).max(1);
        let pretrain_position = pretrain_mask_token + hidden;
        let pretrain_decoder = pretrain_position + pretrain_positions * hidden;
        let lsttn_patch_embedding = pretrain_decoder + patch_width * (hidden + 1);
        let lsttn_transformer_ffn = lsttn_patch_embedding + patch_width * hidden + hidden;
        let transformer_ffn = 8 * hidden * hidden + 5 * hidden;
        let lsttn_transformer_norm = lsttn_transformer_ffn + transformer_blocks * transformer_ffn;
        let lsttn_transformer_out = lsttn_transformer_norm + transformer_blocks * 4 * hidden;
        let lsttn_encoder_decoder = lsttn_transformer_out + transformer_blocks * projection;
        let lsttn_decoder_q = lsttn_encoder_decoder + projection;
        let lsttn_decoder_k = lsttn_decoder_q + projection;
        let lsttn_decoder_v = lsttn_decoder_k + projection;
        let lsttn_decoder_out = lsttn_decoder_v + projection;
        let lsttn_decoder_ffn = lsttn_decoder_out + projection;
        let lsttn_decoder_norm = lsttn_decoder_ffn + transformer_ffn;
        let total = lsttn_decoder_norm + 4 * hidden;
        Self {
            input,
            time2vec_frequency,
            time2vec_phase,
            in_degree_embedding,
            out_degree_embedding,
            temporal_q,
            temporal_k,
            temporal_v,
            spatial_q,
            spatial_k,
            spatial_v,
            shortest_path_bias,
            router,
            spatial_router,
            spatial_expert_heads,
            expert_heads,
            recurrence,
            lsttn_dilated_convolution,
            lsttn_short_wave,
            stgformer_pointwise,
            lsttn_adaptive_source,
            lsttn_adaptive_target,
            lsttn_weekly_adaptive_source,
            lsttn_weekly_adaptive_target,
            lsttn_short_adaptive_source,
            lsttn_short_adaptive_target,
            lsttn_periodic_projection,
            lsttn_fusion,
            graphon_nodes,
            graphon_time,
            output,
            pretrain_mask_token,
            pretrain_position,
            pretrain_decoder,
            lsttn_patch_embedding,
            lsttn_transformer_ffn,
            lsttn_transformer_norm,
            lsttn_transformer_out,
            lsttn_encoder_decoder,
            lsttn_decoder_q,
            lsttn_decoder_k,
            lsttn_decoder_v,
            lsttn_decoder_out,
            lsttn_decoder_ffn,
            lsttn_decoder_norm,
            total,
        }
    }
}

impl TrainableGraphTransformerState {
    #[allow(clippy::too_many_arguments)]
    fn initialized(
        nodes: usize,
        hidden: usize,
        attention_heads: usize,
        periodicity: usize,
        recent_window: usize,
        context_window: usize,
        horizons: usize,
        experts: usize,
        graph_order: usize,
        seed: u64,
    ) -> Self {
        let layout = GraphParameterLayout::new(
            nodes,
            hidden,
            horizons,
            experts,
            graph_order,
            periodicity,
            context_window,
        );
        let mut state = seed;
        let mut parameters = (0..layout.total)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) - 0.5) * 0.08
            })
            .collect::<Vec<_>>();
        for layer in 0..4 {
            let norm = layout.lsttn_transformer_norm + layer * 4 * hidden;
            parameters[norm..norm + hidden].fill(1.0);
            parameters[norm + 2 * hidden..norm + 3 * hidden].fill(1.0);
        }
        parameters[layout.lsttn_decoder_norm..layout.lsttn_decoder_norm + hidden].fill(1.0);
        parameters[layout.lsttn_decoder_norm + 2 * hidden..layout.lsttn_decoder_norm + 3 * hidden]
            .fill(1.0);
        let short_layer_width = 12 * hidden * hidden + 6 * hidden;
        let short_layers = layout.lsttn_short_wave + 2 * hidden + hidden;
        for layer in 0..8 {
            let gamma =
                short_layers + layer * short_layer_width + 12 * hidden * hidden + 4 * hidden;
            parameters[gamma..gamma + hidden].fill(1.0);
        }
        Self {
            first_moment: vec![0.0; layout.total],
            second_moment: vec![0.0; layout.total],
            parameters,
            steps: 0,
            shared_steps: 0,
            local_steps: 0,
            architecture_fingerprint: 0,
            backbone_fingerprint: 0,
            shard_data_fingerprint: 0,
            warm_start_policy_fingerprint: 0,
            nodes,
            hidden,
            attention_heads,
            periodicity,
            recent_window,
            context_window,
            horizons,
            experts,
            graph_order,
            periodic_short_lag: 0,
            target_scale: 1.0,
            normalized_zero: 0.0,
            local_parameter_ranges: None,
            shard_training: false,
            freeze_shared: false,
        }
    }

    fn layout(&self) -> GraphParameterLayout {
        GraphParameterLayout::new(
            self.nodes,
            self.hidden,
            self.horizons,
            self.experts,
            self.graph_order,
            self.periodicity,
            self.context_window,
        )
    }

    fn shared_ranges(&self) -> Vec<(&'static str, std::ops::Range<usize>)> {
        let layout = self.layout();
        vec![
            ("input_time", layout.input..layout.in_degree_embedding),
            ("attention", layout.temporal_q..layout.shortest_path_bias),
            (
                "routing_temporal",
                layout.router..layout.lsttn_adaptive_source,
            ),
            (
                "periodic_fusion",
                layout.lsttn_periodic_projection..layout.graphon_nodes,
            ),
            ("decoder", layout.graphon_time..layout.total),
        ]
    }

    fn export_shared(&self, target_mean: f64, target_scale: f64) -> LsttnSharedState {
        let segments = self
            .shared_ranges()
            .into_iter()
            .map(|(name, range)| LsttnSharedSegment {
                name: name.to_string(),
                parameters: self.parameters[range.clone()].to_vec(),
                first_moment: self.first_moment[range.clone()].to_vec(),
                second_moment: self.second_moment[range].to_vec(),
            })
            .collect();
        LsttnSharedState {
            version: 2,
            architecture_fingerprint: self.architecture_fingerprint,
            backbone_fingerprint: self.backbone_fingerprint,
            hidden: self.hidden,
            attention_heads: self.attention_heads,
            periodicity: self.periodicity,
            recent_window: self.recent_window,
            context_window: self.context_window,
            horizons: self.horizons,
            experts: self.experts,
            graph_order: self.graph_order,
            steps: self.shared_steps,
            target_mean,
            target_scale,
            segments,
        }
    }

    fn import_shared(&mut self, shared: &LsttnSharedState) -> Result<()> {
        if shared.version != 2
            || shared.architecture_fingerprint != self.architecture_fingerprint
            || shared.backbone_fingerprint != self.backbone_fingerprint
            || shared.hidden != self.hidden
            || shared.attention_heads != self.attention_heads
            || shared.periodicity != self.periodicity
            || shared.recent_window != self.recent_window
            || shared.context_window != self.context_window
            || shared.horizons != self.horizons
            || shared.experts != self.experts
            || shared.graph_order != self.graph_order
        {
            return Err(GeoStError::InvalidFrame(
                "shared LSTTN state is incompatible with the shard architecture".to_string(),
            ));
        }
        let ranges = self.shared_ranges();
        if shared.segments.len() != ranges.len() {
            return Err(GeoStError::InvalidFrame(
                "shared LSTTN state has an invalid segment count".to_string(),
            ));
        }
        for (segment, (name, range)) in shared.segments.iter().zip(ranges) {
            let length = range.len();
            if segment.name != name
                || segment.parameters.len() != length
                || segment.first_moment.len() != length
                || segment.second_moment.len() != length
            {
                return Err(GeoStError::InvalidFrame(format!(
                    "shared LSTTN segment {name} is incompatible"
                )));
            }
            self.parameters[range.clone()].copy_from_slice(&segment.parameters);
            self.first_moment[range.clone()].copy_from_slice(&segment.first_moment);
            self.second_moment[range].copy_from_slice(&segment.second_moment);
        }
        self.shared_steps = shared.steps;
        self.steps = self.steps.max(self.shared_steps).max(self.local_steps);
        Ok(())
    }

    fn freeze_shared_gradients(&self, gradients: &mut [f64]) {
        for (_, range) in self.shared_ranges() {
            gradients[range].fill(0.0);
        }
    }

    fn local_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut cursor = 0;
        for (_, shared) in self.shared_ranges() {
            if cursor < shared.start {
                ranges.push(cursor..shared.start);
            }
            cursor = shared.end;
        }
        if cursor < self.layout().total {
            ranges.push(cursor..self.layout().total);
        }
        ranges
    }

    fn compact_to_local_state(&mut self) {
        let ranges = self.local_ranges();
        self.parameters = ranges
            .iter()
            .flat_map(|range| self.parameters[range.clone()].iter().copied())
            .collect();
        self.first_moment = ranges
            .iter()
            .flat_map(|range| self.first_moment[range.clone()].iter().copied())
            .collect();
        self.second_moment = ranges
            .iter()
            .flat_map(|range| self.second_moment[range.clone()].iter().copied())
            .collect();
        self.local_parameter_ranges = Some(
            ranges
                .into_iter()
                .map(|range| (range.start, range.end))
                .collect(),
        );
    }

    fn expand_local_state(&mut self) -> Result<()> {
        let Some(ranges) = self.local_parameter_ranges.take() else {
            return Ok(());
        };
        let compact_length: usize = ranges.iter().map(|(start, end)| end - start).sum();
        if compact_length != self.parameters.len()
            || compact_length != self.first_moment.len()
            || compact_length != self.second_moment.len()
            || ranges
                .iter()
                .any(|(start, end)| start >= end || *end > self.layout().total)
        {
            return Err(GeoStError::InvalidFrame(
                "compact shard-local LSTTN state is incompatible".to_string(),
            ));
        }
        let mut parameters = vec![0.0; self.layout().total];
        let mut first_moment = vec![0.0; self.layout().total];
        let mut second_moment = vec![0.0; self.layout().total];
        let mut offset = 0;
        for (start, end) in ranges {
            let length = end - start;
            parameters[start..end].copy_from_slice(&self.parameters[offset..offset + length]);
            first_moment[start..end].copy_from_slice(&self.first_moment[offset..offset + length]);
            second_moment[start..end].copy_from_slice(&self.second_moment[offset..offset + length]);
            offset += length;
        }
        self.parameters = parameters;
        self.first_moment = first_moment;
        self.second_moment = second_moment;
        Ok(())
    }

    fn frozen_lsttn_patch_representations(
        &self,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        phase_offset: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        self.frozen_lsttn_patch_representations_with_backend(window, adjacency, phase_offset, None)
            .expect("CPU frozen LSTTN transformer is infallible")
    }

    fn frozen_lsttn_patch_representations_with_backend(
        &self,
        window: &[Vec<f64>],
        _adjacency: &CsrAdjacency,
        _phase_offset: usize,
        backend: Option<&BackendSelection>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let layout = self.layout();
        let patch_width = (self.periodicity / 24).max(1);
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / self.hidden;
        let patch_count = window.len() / patch_width;
        let projection = self.hidden * (self.hidden + 1);
        let ffn_width = 8 * self.hidden * self.hidden + 5 * self.hidden;
        let by_node = (0..self.nodes)
            .into_par_iter()
            .map(|node| -> Result<Vec<Vec<f64>>> {
                let mut sequence = (0..patch_count)
                    .map(|patch| {
                        (0..self.hidden)
                            .map(|channel| {
                                let mut value = self.parameters[layout.lsttn_patch_embedding
                                    + patch_width * self.hidden
                                    + channel];
                                for offset in 0..patch_width {
                                    value += self.parameters[layout.lsttn_patch_embedding
                                        + offset * self.hidden
                                        + channel]
                                        * window[patch * patch_width + offset][node];
                                }
                                (value
                                    + self.parameters[layout.pretrain_position
                                        + (patch % position_count) * self.hidden
                                        + channel])
                                    * (self.hidden as f64).sqrt()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for layer in 0..4 {
                    sequence = numeric_transformer_encoder_layer(
                        &self.parameters,
                        &sequence,
                        layout.temporal_q + layer * projection,
                        layout.temporal_k + layer * projection,
                        layout.temporal_v + layer * projection,
                        layout.lsttn_transformer_out + layer * projection,
                        layout.lsttn_transformer_ffn + layer * ffn_width,
                        layout.lsttn_transformer_norm + layer * 4 * self.hidden,
                        self.hidden,
                        self.attention_heads,
                        backend,
                    )?;
                }
                Ok(sequence)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((0..patch_count)
            .map(|patch| {
                (0..self.nodes)
                    .map(|node| {
                        by_node[node][patch]
                            .iter()
                            .map(|value| *value as f32)
                            .collect()
                    })
                    .collect()
            })
            .collect())
    }

    fn adamw_step(
        &mut self,
        gradients: &[f64],
        learning_rate: f64,
        weight_decay: f64,
        backend: Option<&BackendSelection>,
    ) -> Result<()> {
        if self.shard_training {
            self.adamw_shard_step(gradients, learning_rate, weight_decay);
            return Ok(());
        }
        let next_step = self.steps + 1;
        if let Some(selection) = backend.filter(|selection| {
            selection.selected != "cpu" && self.parameters.len() >= GEO_ST_ADAMW_DISPATCH_MIN_VALUES
        }) {
            let mut parameters = self
                .parameters
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let mut first_moment = self
                .first_moment
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let mut second_moment = self
                .second_moment
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let gradients = gradients
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            backend_adamw_step_f32(
                selection,
                &mut parameters,
                &mut first_moment,
                &mut second_moment,
                &gradients,
                next_step,
                learning_rate as f32,
                weight_decay as f32,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.parameters = parameters.into_iter().map(f64::from).collect();
            self.first_moment = first_moment.into_iter().map(f64::from).collect();
            self.second_moment = second_moment.into_iter().map(f64::from).collect();
            self.steps = next_step;
            return Ok(());
        }
        self.steps = next_step;
        let step = next_step as f64;
        for (index, gradient) in gradients.iter().copied().enumerate() {
            let gradient = gradient + weight_decay * self.parameters[index];
            self.first_moment[index] = 0.9 * self.first_moment[index] + 0.1 * gradient;
            self.second_moment[index] =
                0.999 * self.second_moment[index] + 0.001 * gradient * gradient;
            let corrected_first = self.first_moment[index] / (1.0 - 0.9_f64.powf(step));
            let corrected_second = self.second_moment[index] / (1.0 - 0.999_f64.powf(step));
            self.parameters[index] -=
                learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8);
        }
        Ok(())
    }

    fn adamw_shard_step(&mut self, gradients: &[f64], learning_rate: f64, weight_decay: f64) {
        let shared_ranges = self.shared_ranges();
        let mut is_shared = vec![false; self.parameters.len()];
        for (_, range) in shared_ranges {
            is_shared[range].fill(true);
        }
        let next_shared_step = self.shared_steps + u64::from(!self.freeze_shared);
        let next_local_step = self.local_steps + 1;
        for (index, gradient) in gradients.iter().copied().enumerate() {
            let shared = is_shared[index];
            if (shared && self.freeze_shared) || gradient == 0.0 {
                continue;
            }
            let step = if shared {
                next_shared_step
            } else {
                next_local_step
            } as f64;
            let gradient = gradient + weight_decay * self.parameters[index];
            self.first_moment[index] = 0.9 * self.first_moment[index] + 0.1 * gradient;
            self.second_moment[index] =
                0.999 * self.second_moment[index] + 0.001 * gradient * gradient;
            let corrected_first = self.first_moment[index] / (1.0 - 0.9_f64.powf(step));
            let corrected_second = self.second_moment[index] / (1.0 - 0.999_f64.powf(step));
            self.parameters[index] -=
                learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8);
        }
        if !self.freeze_shared {
            self.shared_steps = next_shared_step;
        }
        self.local_steps = next_local_step;
        self.steps = self.steps.max(self.shared_steps).max(self.local_steps);
    }

    /// The paper pretrains MST and freezes it before fitting LSTTN.  These
    /// ranges own the patch projection, learned patch positions, and the
    /// temporal Q/K/V projections used to contextualize patch embeddings.
    fn freeze_lsttn_transformer_gradients(
        &self,
        layout: GraphParameterLayout,
        gradients: &mut [f64],
    ) {
        gradients[layout.input..layout.spatial_q].fill(0.0);
        gradients[layout.pretrain_mask_token..layout.total].fill(0.0);
    }

    /// Build one paper-style LSTTN training example without mutating model
    /// state.  This is deliberately separate from Adam so a 32-window batch
    /// can evaluate on Rayon workers and reduce its gradients deterministically
    /// before taking one optimizer step, just like the reference mini-batch
    /// trainer.
    #[allow(clippy::too_many_arguments)]
    fn lsttn_example_loss_and_gradients(
        &self,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        targets: &[Vec<f64>],
        phase_offset: usize,
        frozen_patches: Option<&[Vec<Vec<f32>>]>,
        time_features: Option<&[Vec<f64>]>,
        target_mask: Option<&[Vec<bool>]>,
        imputed_mask: Option<&[Vec<bool>]>,
        target_weights: Option<&[Vec<f64>]>,
        owner_mask: Option<&[bool]>,
    ) -> (f64, Vec<f64>) {
        let owned_frozen = frozen_patches
            .is_none()
            .then(|| self.frozen_lsttn_patch_representations(window, adjacency, phase_offset));
        let frozen_patches = frozen_patches.or(owned_frozen.as_deref());
        let (tape, outputs, _, _) = self.forward(GraphForwardContext {
            profile: &GraphTransformerProfile::LongShortFusion,
            window,
            adjacency,
            excluded_expert: None,
            phase_offset,
            long_context_is_pooled: false,
            lsttn_frozen_patches: frozen_patches,
            lsttn_time_features: time_features,
            deferred: false,
            training: true,
        });
        let mut loss = tape.constant(0.0);
        let active_weight = |horizon: usize, node: usize| {
            let observed = target_mask.is_none_or(|mask| mask[horizon][node]);
            let imputed = imputed_mask.is_some_and(|mask| mask[horizon][node]);
            let owner = owner_mask.is_none_or(|mask| mask[node]);
            if observed && !imputed && owner {
                target_weights.map_or(1.0, |weights| weights[horizon][node])
            } else {
                0.0
            }
        };
        let total_weight: f64 = (0..self.horizons)
            .flat_map(|horizon| (0..self.nodes).map(move |node| active_weight(horizon, node)))
            .sum();
        let scale = tape.constant(1.0 / total_weight.max(1.0));
        for (node, node_outputs) in outputs.iter().enumerate().take(self.nodes) {
            for (horizon, horizon_targets) in targets.iter().enumerate().take(self.horizons) {
                let weight = active_weight(horizon, node);
                if weight == 0.0 {
                    continue;
                }
                let residual =
                    tape.add(node_outputs[horizon], tape.constant(-horizon_targets[node]));
                let residual = tape.mul(residual, tape.constant(self.target_scale));
                let mae = tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)));
                loss = tape.add(loss, tape.mul(mae, tape.mul(scale, tape.constant(weight))));
            }
        }
        (tape.value(loss), tape.backward(loss, self.parameters.len()))
    }

    #[allow(clippy::too_many_arguments)]
    fn train_example_with_context(
        &mut self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        targets: &[Vec<f64>],
        excluded_expert: Option<usize>,
        learning_rate: f64,
        weight_decay: f64,
        phase_offset: usize,
        long_context_is_pooled: bool,
        backend: Option<&BackendSelection>,
    ) -> Result<f64> {
        // LSTTN freezes its pretrained masked-subseries Transformer during
        // supervised fitting.  Keep this path on the native tape so the
        // frozen ranges are enforced identically on every supported host.
        let accelerated = *profile != GraphTransformerProfile::LongShortFusion
            && backend.is_some_and(|selection| selection.selected != "cpu");
        let frozen_lsttn = if *profile == GraphTransformerProfile::LongShortFusion {
            Some(self.frozen_lsttn_patch_representations_with_backend(
                window,
                adjacency,
                phase_offset,
                backend,
            )?)
        } else {
            None
        };
        let (tape, outputs, router_weights, _) = self.forward(GraphForwardContext {
            profile,
            window,
            adjacency,
            excluded_expert,
            phase_offset,
            long_context_is_pooled,
            lsttn_frozen_patches: frozen_lsttn.as_deref(),
            lsttn_time_features: None,
            deferred: accelerated,
            training: true,
        });
        let mut loss = tape.constant(0.0);
        let valid = targets
            .iter()
            .flatten()
            .filter(|target| {
                *profile != GraphTransformerProfile::LongShortFusion
                    || (**target - self.normalized_zero).abs() > 1e-12
            })
            .count();
        let scale = tape.constant(1.0 / valid.max(1) as f64);
        for (node, node_outputs) in outputs.iter().enumerate().take(self.nodes) {
            for (horizon, horizon_targets) in targets.iter().enumerate().take(self.horizons) {
                if *profile == GraphTransformerProfile::LongShortFusion
                    && (horizon_targets[node] - self.normalized_zero).abs() <= 1e-12
                {
                    continue;
                }
                let residual =
                    tape.add(node_outputs[horizon], tape.constant(-horizon_targets[node]));
                let point_loss = if *profile == GraphTransformerProfile::LongShortFusion {
                    let residual = tape.mul(residual, tape.constant(self.target_scale));
                    // The reference LSTTN trains its forecast stage with
                    // masked MAE.  The infinitesimal smoothing keeps the
                    // derivative defined at zero for the native tape while
                    // preserving the MAE value at reporting precision.
                    tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)))
                } else {
                    tape.mul(residual, residual)
                };
                loss = tape.add(loss, tape.mul(point_loss, scale));
            }
        }
        if *profile == GraphTransformerProfile::HeterogeneousMoE && !router_weights.is_empty() {
            // STGormer's auxiliary router objective penalizes concentrated
            // expert probability mass.  The mean routing probability per
            // expert is differentiable, so this keeps experts available
            // rather than allowing one feed-forward path to collapse.
            let count = tape.constant(router_weights.len() as f64);
            let expert_count = tape.constant(self.experts as f64);
            let coefficient = tape.constant(0.01);
            for expert in 0..self.experts {
                let mass = router_weights
                    .iter()
                    .fold(tape.constant(0.0), |sum, weights| {
                        tape.add(sum, weights[expert])
                    });
                let mean_mass = tape.div(mass, count);
                loss = tape.add(
                    loss,
                    tape.mul(
                        coefficient,
                        tape.mul(expert_count, tape.mul(mean_mass, mean_mass)),
                    ),
                );
            }
        }
        if accelerated {
            let next_step = self.steps + 1;
            let value = tape.accelerated_train_step(
                backend.expect("non-CPU backend is present"),
                loss,
                &mut self.parameters,
                &mut self.first_moment,
                &mut self.second_moment,
                next_step,
                learning_rate,
                weight_decay,
            )?;
            self.steps = next_step;
            Ok(value)
        } else {
            let value = tape.value(loss);
            let mut gradients = tape.backward(loss, self.parameters.len());
            if *profile == GraphTransformerProfile::LongShortFusion {
                self.freeze_lsttn_transformer_gradients(self.layout(), &mut gradients);
            }
            self.adamw_step(&gradients, learning_rate, weight_decay, backend)?;
            Ok(value)
        }
    }

    /// LSTTN's self-supervised stage: encode the unmasked equal-length
    /// subseries, insert learned mask tokens at the withheld positions, and
    /// decode only those patches.  The shared input/Q/K/V projections are the
    /// same ones used by the forecasting path, so pretraining transfers a
    /// contextual long-history representation instead of training a detached
    /// auxiliary model.
    #[allow(clippy::needless_range_loop)]
    fn train_masked_subseries_reconstruction(
        &mut self,
        window: &[Vec<f64>],
        learning_rate: f64,
        weight_decay: f64,
        backend: Option<&BackendSelection>,
    ) -> Result<f64> {
        let patch_width = (self.periodicity / 24).max(1);
        if !window.len().is_multiple_of(patch_width) {
            return Err(GeoStError::InvalidFrame(format!(
                "LSTTN long-history window length {} must be divisible by patch width {}",
                window.len(),
                patch_width
            )));
        }
        let patches = window.len() / patch_width;
        if patches < 2 {
            return Err(GeoStError::InvalidFrame(
                "LSTTN masked-subseries pretraining requires at least two patches".to_string(),
            ));
        }
        let masked = masked_patch_indices(patches, self.steps);
        let visible = (0..patches)
            .filter(|patch| !masked.contains(patch))
            .collect::<Vec<_>>();
        let layout = self.layout();
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / self.hidden;
        let mut total_loss = 0.0;
        let mut gradients = vec![0.0; self.parameters.len()];
        let valid_reconstruction_values = masked
            .iter()
            .flat_map(|patch| {
                (0..self.nodes).flat_map(move |node| {
                    (0..patch_width).map(move |offset| window[patch * patch_width + offset][node])
                })
            })
            .filter(|target| (*target - self.normalized_zero).abs() > 1e-12)
            .count();
        let accelerated = backend.is_some_and(|selection| selection.selected != "cpu");
        // Self-attention over the visible long-history patches is independent
        // per H3 node.  Keep a bounded tape for both CPU and accelerator
        // execution, while accumulating the exact full-batch gradient before
        // taking the optimizer step.  This prevents a city-scale panel from
        // multiplying `nodes × patches × hidden` tape state at once.
        const CPU_PRETRAIN_NODE_BATCH: usize = 16;
        const ACCELERATOR_PRETRAIN_NODE_BATCH: usize = 32;
        let node_batch_size = if accelerated {
            ACCELERATOR_PRETRAIN_NODE_BATCH
        } else {
            CPU_PRETRAIN_NODE_BATCH
        }
        .min(self.nodes)
        .max(1);
        // One tape now contains every randomly masked patch for a bounded
        // node batch.  Constructing a tape per patch used to repeat the
        // visible-context encoder and issue hundreds of tiny accelerator
        // launches for a long daily history.  Keeping the node dimension
        // bounded preserves the city-scale memory ceiling while making each
        // reconstruction pass a single batched objective.
        for node_start in (0..self.nodes).step_by(node_batch_size) {
            let node_end = (node_start + node_batch_size).min(self.nodes);
            let tape = if accelerated {
                AutodiffTape::deferred()
            } else {
                AutodiffTape::new()
            };
            let parameter_nodes = self
                .parameters
                .iter()
                .enumerate()
                .map(|(index, value)| tape.parameter(index, *value))
                .collect::<Vec<_>>();
            let parameter = |index: usize| parameter_nodes[index];
            let mut loss = tape.constant(0.0);
            let scale = tape.constant(1.0 / valid_reconstruction_values.max(1) as f64);
            for node in node_start..node_end {
                let mut encoded = visible
                    .iter()
                    .map(|patch| {
                        (0..self.hidden)
                            .map(|channel| {
                                let mut value = parameter(
                                    layout.lsttn_patch_embedding
                                        + patch_width * self.hidden
                                        + channel,
                                );
                                for offset in 0..patch_width {
                                    value = tape.add(
                                        value,
                                        tape.mul(
                                            parameter(
                                                layout.lsttn_patch_embedding
                                                    + offset * self.hidden
                                                    + channel,
                                            ),
                                            tape.constant(
                                                window[patch * patch_width + offset][node],
                                            ),
                                        ),
                                    );
                                }
                                tape_deterministic_dropout(
                                    &tape,
                                    tape.mul(
                                        tape.add(
                                            value,
                                            parameter(
                                                layout.pretrain_position
                                                    + (patch % position_count) * self.hidden
                                                    + channel,
                                            ),
                                        ),
                                        tape.constant((self.hidden as f64).sqrt()),
                                    ),
                                    self.steps ^ node as u64,
                                    patch * self.hidden + channel,
                                    true,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let projection = self.hidden * (self.hidden + 1);
                let ffn_width = 8 * self.hidden * self.hidden + 5 * self.hidden;
                for layer in 0..4 {
                    encoded = tape_transformer_encoder_layer(
                        &tape,
                        &parameter_nodes,
                        &encoded,
                        layout.temporal_q + layer * projection,
                        layout.temporal_k + layer * projection,
                        layout.temporal_v + layer * projection,
                        layout.lsttn_transformer_out + layer * projection,
                        layout.lsttn_transformer_ffn + layer * ffn_width,
                        layout.lsttn_transformer_norm + layer * 4 * self.hidden,
                        self.hidden,
                        self.attention_heads,
                        self.steps ^ ((node as u64) << 16) ^ layer as u64,
                        true,
                    );
                }
                let decoder_scale = tape.constant((self.hidden as f64).sqrt());
                let mut decoder_input = encoded
                    .iter()
                    .map(|token| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.lsttn_encoder_decoder,
                            token,
                            self.hidden,
                            self.hidden,
                        )
                        .into_iter()
                        .map(|value| tape.mul(value, decoder_scale))
                        .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                decoder_input.extend(masked.iter().map(|patch| {
                    (0..self.hidden)
                        .map(|channel| {
                            tape.mul(
                                tape_deterministic_dropout(
                                    &tape,
                                    tape.add(
                                        parameter(layout.pretrain_mask_token + channel),
                                        parameter(
                                            layout.pretrain_position
                                                + (patch % position_count) * self.hidden
                                                + channel,
                                        ),
                                    ),
                                    self.steps ^ ((node as u64) << 32),
                                    patch * self.hidden + channel,
                                    true,
                                ),
                                decoder_scale,
                            )
                        })
                        .collect::<Vec<_>>()
                }));
                let decoded = tape_transformer_encoder_layer(
                    &tape,
                    &parameter_nodes,
                    &decoder_input,
                    layout.lsttn_decoder_q,
                    layout.lsttn_decoder_k,
                    layout.lsttn_decoder_v,
                    layout.lsttn_decoder_out,
                    layout.lsttn_decoder_ffn,
                    layout.lsttn_decoder_norm,
                    self.hidden,
                    self.attention_heads,
                    self.steps ^ ((node as u64) << 48),
                    true,
                );
                for (masked_index, patch) in masked.iter().enumerate() {
                    let context = &decoded[visible.len() + masked_index];
                    for offset in 0..patch_width {
                        let target = window[patch * patch_width + offset][node];
                        if (target - self.normalized_zero).abs() <= 1e-12 {
                            continue;
                        }
                        let mut prediction =
                            parameter(layout.pretrain_decoder + patch_width * self.hidden + offset);
                        for (channel, context_value) in context.iter().enumerate() {
                            prediction = tape.add(
                                prediction,
                                tape.mul(
                                    parameter(
                                        layout.pretrain_decoder + offset * self.hidden + channel,
                                    ),
                                    *context_value,
                                ),
                            );
                        }
                        let residual = tape.add(prediction, tape.constant(-target));
                        let residual = tape.mul(residual, tape.constant(self.target_scale));
                        let mae =
                            tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)));
                        loss = tape.add(loss, tape.mul(scale, mae));
                    }
                }
            }
            if accelerated {
                let next_step = self.steps + 1;
                total_loss += tape.accelerated_train_step(
                    backend.expect("non-CPU backend is present"),
                    loss,
                    &mut self.parameters,
                    &mut self.first_moment,
                    &mut self.second_moment,
                    next_step,
                    learning_rate,
                    weight_decay,
                )?;
                self.steps = next_step;
            } else {
                total_loss += tape.value(loss);
                for (total, gradient) in gradients
                    .iter_mut()
                    .zip(tape.backward(loss, self.parameters.len()))
                {
                    *total += gradient;
                }
            }
        }
        if !accelerated {
            self.adamw_step(&gradients, learning_rate, weight_decay, backend)?;
        }
        Ok(total_loss)
    }

    #[cfg(test)]
    fn predict_window(
        &self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
    ) -> Vec<Vec<f64>> {
        self.predict_window_with_context(profile, window, adjacency, 0, false, None, None)
            .expect("CPU graph transformer prediction")
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_window_with_context(
        &self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        phase_offset: usize,
        long_context_is_pooled: bool,
        backend: Option<&BackendSelection>,
        time_features: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<f64>>> {
        let accelerated = *profile == GraphTransformerProfile::LongShortFusion
            && backend.is_some_and(|selection| selection.selected != "cpu");
        let frozen_lsttn = if *profile == GraphTransformerProfile::LongShortFusion {
            Some(self.frozen_lsttn_patch_representations_with_backend(
                window,
                adjacency,
                phase_offset,
                backend,
            )?)
        } else {
            None
        };
        let (tape, outputs, _, _) = self.forward(GraphForwardContext {
            profile,
            window,
            adjacency,
            excluded_expert: None,
            phase_offset,
            long_context_is_pooled,
            lsttn_frozen_patches: frozen_lsttn.as_deref(),
            lsttn_time_features: time_features,
            deferred: accelerated,
            training: false,
        });
        if accelerated {
            let selection = backend.expect("non-CPU backend is present");
            let values = tape.accelerated_values(selection)?;
            return Ok((0..self.horizons)
                .map(|horizon| {
                    (0..self.nodes)
                        .map(|node| values[outputs[node][horizon]] as f64)
                        .collect()
                })
                .collect());
        }
        Ok((0..self.horizons)
            .map(|horizon| {
                (0..self.nodes)
                    .map(|node| tape.value(outputs[node][horizon]))
                    .collect()
            })
            .collect())
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(&self, context: GraphForwardContext<'_>) -> GraphForwardOutput {
        let GraphForwardContext {
            profile,
            window,
            adjacency,
            excluded_expert,
            phase_offset,
            long_context_is_pooled,
            lsttn_frozen_patches,
            lsttn_time_features,
            deferred,
            training,
        } = context;
        let layout = self.layout();
        let tape = if deferred {
            AutodiffTape::deferred()
        } else {
            AutodiffTape::new()
        };
        let parameter_nodes = self
            .parameters
            .iter()
            .enumerate()
            .map(|(index, value)| tape.parameter(index, *value))
            .collect::<Vec<_>>();
        let parameter = |_tape: &AutodiffTape, index: usize| parameter_nodes[index];
        let nodes = self.nodes;
        let hidden = self.hidden;
        let times = window.len();
        let native_patch_width = (self.periodicity / 24).max(1);
        let time_scale = if long_context_is_pooled {
            native_patch_width
        } else {
            1
        };
        let effective_periodicity = (self.periodicity / time_scale).max(1);
        let effective_short_periodicity = (if self.periodic_short_lag == 0 {
            self.periodicity
        } else {
            self.periodic_short_lag
        } / time_scale)
            .max(1);
        // LSTTN's periodic graph convolution uses both directed structural
        // diffusions as well as its learned adaptive diffusion.  Preserve a
        // normalized reverse graph rather than treating the supplied road
        // graph as undirected.
        let reverse_adjacency = adjacency.transpose(nodes).row_normalized();
        let adaptive_adjacency = adjacency.with_adaptive_self_candidates(nodes);
        let observed_values = window
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| tape.constant(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let adjacency_weights = adjacency
            .data
            .iter()
            .map(|weight| tape.constant(*weight))
            .collect::<Vec<_>>();
        let reverse_adjacency_weights = reverse_adjacency
            .data
            .iter()
            .map(|weight| tape.constant(*weight))
            .collect::<Vec<_>>();
        let mut required_embedding_times = vec![true; times];
        if *profile == GraphTransformerProfile::LongShortFusion && lsttn_frozen_patches.is_some() {
            required_embedding_times.fill(false);
        }
        let mut graph_values = vec![vec![0usize; nodes]; times];
        for time in 0..times {
            if !required_embedding_times[time] {
                continue;
            }
            for target in 0..nodes {
                graph_values[time][target] = (adjacency.indptr[target]
                    ..adjacency.indptr[target + 1])
                    .fold(tape.constant(0.0), |sum, edge| {
                        tape.add(
                            sum,
                            tape.mul(
                                adjacency_weights[edge],
                                observed_values[time][adjacency.indices[edge]],
                            ),
                        )
                    });
            }
        }
        let degrees = graph_in_degrees(adjacency, nodes);
        let out_degrees = graph_out_degrees(adjacency, nodes);
        let positions = (0..times)
            .map(|time| tape.constant((time + 1) as f64 / times as f64))
            .collect::<Vec<_>>();
        let periodic_features = (0..times)
            .map(|time| {
                [
                    tape.constant(periodic_phase(
                        phase_offset + time + 1,
                        effective_short_periodicity,
                    )),
                    tape.constant(periodic_phase(
                        phase_offset + time + 1,
                        effective_periodicity,
                    )),
                ]
            })
            .collect::<Vec<_>>();
        let degree_features = (0..nodes)
            .map(|node| {
                [
                    tape.constant(degrees[node] / nodes.max(1) as f64),
                    tape.constant(out_degrees[node] / nodes.max(1) as f64),
                ]
            })
            .collect::<Vec<_>>();
        let mut embedding = vec![vec![vec![0usize; hidden]; nodes]; times];
        for time in 0..times {
            if !required_embedding_times[time] {
                continue;
            }
            let position = positions[time];
            for node in 0..nodes {
                for channel in 0..hidden {
                    let time2vec = tape.sin(tape.add(
                        tape.mul(
                            parameter(&tape, layout.time2vec_frequency + channel),
                            position,
                        ),
                        parameter(&tape, layout.time2vec_phase + channel),
                    ));
                    let inputs = [
                        observed_values[time][node],
                        graph_values[time][node],
                        time2vec,
                        periodic_features[time][0],
                        periodic_features[time][1],
                        degree_features[node][0],
                        degree_features[node][1],
                    ];
                    let mut value = parameter(&tape, layout.input + 7 * hidden + channel);
                    for (input, input_value) in inputs.iter().enumerate() {
                        value = tape.add(
                            value,
                            tape.mul(
                                parameter(&tape, layout.input + input * hidden + channel),
                                *input_value,
                            ),
                        );
                    }
                    // STGormer uses independent learned embeddings indexed by
                    // in- and out-degree, rather than treating centrality as
                    // a single scalar feature.
                    let in_degree = degrees[node].round().clamp(0.0, nodes as f64) as usize;
                    let out_degree = out_degrees[node].round().clamp(0.0, nodes as f64) as usize;
                    value = tape.add(
                        value,
                        parameter(
                            &tape,
                            layout.in_degree_embedding + in_degree * hidden + channel,
                        ),
                    );
                    value = tape.add(
                        value,
                        parameter(
                            &tape,
                            layout.out_degree_embedding + out_degree * hidden + channel,
                        ),
                    );
                    embedding[time][node][channel] = tape.tanh(value);
                }
            }
        }

        let mut temporal = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::LongShortFusion {
            // LSTTN keeps the latest native-resolution embedding available to
            // its profile path. Its long branch builds learned
            // patch states and one-query contextual attention below;
            // materializing generic all-pairs attention here would make the
            // long-context profile quadratic in history length.
            temporal.clone_from(&embedding[times - 1]);
        } else {
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.temporal_q,
                    &embedding[times - 1][node],
                    hidden,
                    hidden,
                );
                let mut keys = Vec::with_capacity(times);
                let mut values = Vec::with_capacity(times);
                for time in 0..times {
                    let key = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_k,
                        &embedding[time][node],
                        hidden,
                        hidden,
                    );
                    let value = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_v,
                        &embedding[time][node],
                        hidden,
                        hidden,
                    );
                    keys.push(key);
                    values.push(value);
                }
                for head in 0..self.attention_heads {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    if *profile == GraphTransformerProfile::EfficientHighOrder {
                        // The official STGformer implementation L2-normalizes
                        // Q and K, then applies the efficient-attention
                        // rearrangement: Q(K^T V) + N V over Q(sum K) + N.
                        // This is not a generic kernel/feature-map attention.
                        temporal[node][start..end].copy_from_slice(&tape_stgformer_fast_attention(
                            &tape,
                            &query[start..end],
                            &tape_stgformer_attention_summary(
                                &tape,
                                &keys
                                    .iter()
                                    .map(|key| key[start..end].to_vec())
                                    .collect::<Vec<_>>(),
                                &values
                                    .iter()
                                    .map(|value| value[start..end].to_vec())
                                    .collect::<Vec<_>>(),
                            ),
                            &values[times - 1][start..end],
                        ));
                    } else {
                        let scores = keys
                            .iter()
                            .map(|key| tape_dot(&tape, &query[start..end], &key[start..end]))
                            .collect::<Vec<_>>();
                        let weights = tape_softmax(&tape, &scores);
                        let head_values = values
                            .iter()
                            .map(|value| value[start..end].to_vec())
                            .collect::<Vec<_>>();
                        temporal[node][start..end].copy_from_slice(&tape_weighted_sum(
                            &tape,
                            &weights,
                            &head_values,
                            end - start,
                        ));
                    }
                }
            }
        }

        // LongShortFusion owns its sparse forward, backward, and adaptive
        // graph diffusions below. It does not use Graphormer shortest-path
        // attention, so allocating an all-pairs distance matrix would be
        // quadratic and unnecessary for global graphs.
        let distances = (*profile != GraphTransformerProfile::LongShortFusion)
            .then(|| graph_distances(adjacency, nodes));
        let mut spatial = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::LongShortFusion {
            // LSTTN's periodic branch owns its forward, backward, and
            // adaptive graph diffusions, so no generic spatial-attention
            // state is needed before long/short fusion.
        } else if *profile == GraphTransformerProfile::EfficientHighOrder {
            // STGformer uses one QKV projection for its spatial and temporal
            // paths.  The efficient K^T V statistic is shared by all query
            // nodes within each head.
            let keys = (0..nodes)
                .map(|node| {
                    tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_k,
                        &temporal[node],
                        hidden,
                        hidden,
                    )
                })
                .collect::<Vec<_>>();
            let values = (0..nodes)
                .map(|node| {
                    tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_v,
                        &temporal[node],
                        hidden,
                        hidden,
                    )
                })
                .collect::<Vec<_>>();
            let summaries = (0..self.attention_heads)
                .map(|head| {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    tape_stgformer_attention_summary(
                        &tape,
                        &keys
                            .iter()
                            .map(|key| key[start..end].to_vec())
                            .collect::<Vec<_>>(),
                        &values
                            .iter()
                            .map(|value| value[start..end].to_vec())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.temporal_q,
                    &temporal[node],
                    hidden,
                    hidden,
                );
                for head in 0..self.attention_heads {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    spatial[node][start..end].copy_from_slice(&tape_stgformer_fast_attention(
                        &tape,
                        &query[start..end],
                        &summaries[head],
                        &values[node][start..end],
                    ));
                }
            }
        } else {
            let distances = distances
                .as_ref()
                .expect("non-LSTTN spatial attention requires graph distances");
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.spatial_q,
                    &temporal[node],
                    hidden,
                    hidden,
                );
                let mut keys = Vec::with_capacity(nodes);
                let mut values = Vec::with_capacity(nodes);
                for other in 0..nodes {
                    let key = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.spatial_k,
                        &temporal[other],
                        hidden,
                        hidden,
                    );
                    let value = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.spatial_v,
                        &temporal[other],
                        hidden,
                        hidden,
                    );
                    let distance = distances[node][other].min(nodes) as f64;
                    // Graphormer-style learnable scalar embedding for each
                    // shortest-path distance, including the disconnected cap.
                    let bias = parameter(&tape, layout.shortest_path_bias + distance as usize);
                    keys.push((key, bias));
                    values.push(value);
                }
                for head in 0..self.attention_heads {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    let scores = keys
                        .iter()
                        .map(|(key, bias)| {
                            tape.add(tape_dot(&tape, &query[start..end], &key[start..end]), *bias)
                        })
                        .collect::<Vec<_>>();
                    let weights = tape_softmax(&tape, &scores);
                    let head_values = values
                        .iter()
                        .map(|value| value[start..end].to_vec())
                        .collect::<Vec<_>>();
                    spatial[node][start..end].copy_from_slice(&tape_weighted_sum(
                        &tape,
                        &weights,
                        &head_values,
                        end - start,
                    ));
                }
            }
        }

        if *profile == GraphTransformerProfile::HeterogeneousMoE {
            let distances = distances
                .as_ref()
                .expect("heterogeneous graph attention requires graph distances");
            // STGormer stacks three causal temporal-attention and spatial-
            // attention blocks.  Each stage owns an independent Q/K/V set;
            // spatial output becomes the representation consumed by the next
            // temporal block, preserving both axes at every depth.
            let projection = hidden * (hidden + 1);
            let mut states = embedding.clone();
            let mut final_temporal = vec![vec![tape.constant(0.0); hidden]; nodes];
            let mut final_spatial = vec![vec![tape.constant(0.0); hidden]; nodes];
            for block in 0..3 {
                let temporal_q = layout.temporal_q + block * projection;
                let temporal_k = layout.temporal_k + block * projection;
                let temporal_v = layout.temporal_v + block * projection;
                let spatial_q = layout.spatial_q + block * projection;
                let spatial_k = layout.spatial_k + block * projection;
                let spatial_v = layout.spatial_v + block * projection;
                let mut block_temporal = vec![vec![vec![tape.constant(0.0); hidden]; nodes]; times];
                for time in 0..times {
                    for node in 0..nodes {
                        let query = tape_linear(
                            &tape,
                            &parameter_nodes,
                            temporal_q,
                            &states[time][node],
                            hidden,
                            hidden,
                        );
                        let keys = (0..=time)
                            .map(|past| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    temporal_k,
                                    &states[past][node],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        let values = (0..=time)
                            .map(|past| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    temporal_v,
                                    &states[past][node],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        for head in 0..self.attention_heads {
                            let start = head * hidden / self.attention_heads;
                            let end = (head + 1) * hidden / self.attention_heads;
                            let scores = keys
                                .iter()
                                .map(|key| tape_dot(&tape, &query[start..end], &key[start..end]))
                                .collect::<Vec<_>>();
                            let weights = tape_softmax(&tape, &scores);
                            let values = values
                                .iter()
                                .map(|value| value[start..end].to_vec())
                                .collect::<Vec<_>>();
                            block_temporal[time][node][start..end].copy_from_slice(
                                &tape_weighted_sum(&tape, &weights, &values, end - start),
                            );
                        }
                    }
                }
                let mut block_spatial = vec![vec![vec![tape.constant(0.0); hidden]; nodes]; times];
                for time in 0..times {
                    for node in 0..nodes {
                        let query = tape_linear(
                            &tape,
                            &parameter_nodes,
                            spatial_q,
                            &block_temporal[time][node],
                            hidden,
                            hidden,
                        );
                        let keys = (0..nodes)
                            .map(|other| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    spatial_k,
                                    &block_temporal[time][other],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        let values = (0..nodes)
                            .map(|other| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    spatial_v,
                                    &block_temporal[time][other],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        for head in 0..self.attention_heads {
                            let start = head * hidden / self.attention_heads;
                            let end = (head + 1) * hidden / self.attention_heads;
                            let scores = (0..nodes)
                                .map(|other| {
                                    let distance = distances[node][other].min(nodes);
                                    tape.add(
                                        tape_dot(
                                            &tape,
                                            &query[start..end],
                                            &keys[other][start..end],
                                        ),
                                        parameter(&tape, layout.shortest_path_bias + distance),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let weights = tape_softmax(&tape, &scores);
                            let values = values
                                .iter()
                                .map(|value| value[start..end].to_vec())
                                .collect::<Vec<_>>();
                            block_spatial[time][node][start..end].copy_from_slice(
                                &tape_weighted_sum(&tape, &weights, &values, end - start),
                            );
                        }
                    }
                }
                states = (0..times)
                    .map(|time| {
                        (0..nodes)
                            .map(|node| {
                                tape_add_vectors(
                                    &tape,
                                    &block_temporal[time][node],
                                    &block_spatial[time][node],
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                final_temporal = block_temporal[times - 1].clone();
                final_spatial = block_spatial[times - 1].clone();
            }
            temporal = final_temporal;
            spatial = final_spatial;
        }

        // The gated graph-temporal profile keeps an explicit normalized graph
        // convolution alongside spatial attention.  The convolution starts
        // from causal temporal states and passes through a learned projection
        // before the GRU-style gates consume it.
        let mut graph_convolution = vec![vec![tape.constant(0.0); hidden]; nodes];
        if *profile == GraphTransformerProfile::GatedGraphTemporal {
            for node in 0..nodes {
                let mut aggregated = vec![tape.constant(0.0); hidden];
                for edge in adjacency.indptr[node]..adjacency.indptr[node + 1] {
                    let neighbor = adjacency.indices[edge];
                    for channel in 0..hidden {
                        aggregated[channel] = tape.add(
                            aggregated[channel],
                            tape.mul(adjacency_weights[edge], temporal[neighbor][channel]),
                        );
                    }
                }
                graph_convolution[node] = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.spatial_v,
                    &aggregated,
                    hidden,
                    hidden,
                );
            }
        }

        // STGformer retains every graph propagation order, applies the same
        // efficient QKV attention block to each order, and recursively
        // interacts the resulting attention with a learned pointwise map of
        // the prior order.  `temporal` is fused into each order's input so the
        // efficient spatial operation receives a spatiotemporal state.
        let mut stgformer_representation = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::EfficientHighOrder {
            let mut propagated = embedding[times - 1].clone();
            let mut previous = propagated.clone();
            stgformer_representation = propagated.clone();
            for order in 0..self.graph_order {
                let order_input = (0..nodes)
                    .map(|node| tape_add_vectors(&tape, &propagated[node], &temporal[node]))
                    .collect::<Vec<_>>();
                let queries = order_input
                    .iter()
                    .map(|input| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.temporal_q,
                            input,
                            hidden,
                            hidden,
                        )
                    })
                    .collect::<Vec<_>>();
                let keys = order_input
                    .iter()
                    .map(|input| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.temporal_k,
                            input,
                            hidden,
                            hidden,
                        )
                    })
                    .collect::<Vec<_>>();
                let values = order_input
                    .iter()
                    .map(|input| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.temporal_v,
                            input,
                            hidden,
                            hidden,
                        )
                    })
                    .collect::<Vec<_>>();
                let summaries = (0..self.attention_heads)
                    .map(|head| {
                        let start = head * hidden / self.attention_heads;
                        let end = (head + 1) * hidden / self.attention_heads;
                        tape_stgformer_attention_summary(
                            &tape,
                            &keys
                                .iter()
                                .map(|key| key[start..end].to_vec())
                                .collect::<Vec<_>>(),
                            &values
                                .iter()
                                .map(|value| value[start..end].to_vec())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut attention = vec![vec![tape.constant(0.0); hidden]; nodes];
                for node in 0..nodes {
                    for head in 0..self.attention_heads {
                        let start = head * hidden / self.attention_heads;
                        let end = (head + 1) * hidden / self.attention_heads;
                        attention[node][start..end].copy_from_slice(
                            &tape_stgformer_fast_attention(
                                &tape,
                                &queries[node][start..end],
                                &summaries[head],
                                &values[node][start..end],
                            ),
                        );
                    }
                }
                let scale = tape.constant(match order {
                    0 => 1.0,
                    1 => 0.01,
                    _ => 0.001,
                });
                for node in 0..nodes {
                    let pointwise = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.stgformer_pointwise + order * hidden * (hidden + 1),
                        &previous[node],
                        hidden,
                        hidden,
                    );
                    for channel in 0..hidden {
                        stgformer_representation[node][channel] = tape.add(
                            stgformer_representation[node][channel],
                            tape.mul(
                                scale,
                                tape.mul(attention[node][channel], pointwise[channel]),
                            ),
                        );
                    }
                }
                previous = attention;
                let mut next = vec![vec![tape.constant(0.0); hidden]; nodes];
                for target in 0..nodes {
                    for edge in adjacency.indptr[target]..adjacency.indptr[target + 1] {
                        let source = adjacency.indices[edge];
                        for channel in 0..hidden {
                            next[target][channel] = tape.add(
                                next[target][channel],
                                tape.mul(adjacency_weights[edge], propagated[source][channel]),
                            );
                        }
                    }
                }
                propagated = next;
            }
        }

        // Periodic patch embeddings are shared by every destination node in
        // LSTTN's graph diffusion.  Building them inside the node loop would
        // duplicate the same tape subgraph `nodes` times, which turns one
        // periodic feature extraction into quadratic memory use on METR-LA.
        let lsttn_period_embeddings = if *profile == GraphTransformerProfile::LongShortFusion {
            let patch_width = if long_context_is_pooled {
                1
            } else {
                native_patch_width
            };
            let patch_count = lsttn_frozen_patches
                .map_or_else(|| embedding.chunks(patch_width).len(), <[_]>::len);
            [effective_short_periodicity, effective_periodicity]
                .into_iter()
                .filter_map(|period| {
                    let period_patches = (period / patch_width).max(1);
                    (patch_count > period_patches).then(|| {
                        let patch_index = patch_count - period_patches - 1;
                        if let Some(cached) = lsttn_frozen_patches {
                            cached[patch_index]
                                .iter()
                                .map(|node_values| {
                                    node_values
                                        .iter()
                                        .map(|value| tape.constant(*value as f64))
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        } else {
                            let patch_start = patch_index * patch_width;
                            let patch =
                                &embedding[patch_start..(patch_start + patch_width).min(times)];
                            (0..nodes)
                                .map(|period_node| {
                                    (0..hidden)
                                        .map(|channel| {
                                            let sum = patch
                                                .iter()
                                                .fold(tape.constant(0.0), |sum, row| {
                                                    tape.add(sum, row[period_node][channel])
                                                });
                                            tape.div(sum, tape.constant(patch.len() as f64))
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let lsttn_periodic_features = if *profile == GraphTransformerProfile::LongShortFusion {
            lsttn_period_embeddings
                .iter()
                .enumerate()
                .map(|(period, values)| {
                    let (adaptive_source, adaptive_target) = if period == 0 {
                        (layout.lsttn_adaptive_source, layout.lsttn_adaptive_target)
                    } else {
                        (
                            layout.lsttn_weekly_adaptive_source,
                            layout.lsttn_weekly_adaptive_target,
                        )
                    };
                    // Adaptive adjacency is learned on the directed CSR
                    // support, not over every possible node pair. The old
                    // dense [nodes, nodes] softmax made a 36k-node LSTTN
                    // infeasible before either CPU or CUDA could reach the
                    // actual diffusion kernels. Each output row retains a
                    // row-normalized learned distribution over its existing
                    // directed neighbors plus one sparse self candidate.
                    let adaptive_weights = (0..nodes)
                        .flat_map(|target| {
                            let range = adaptive_adjacency.indptr[target]
                                ..adaptive_adjacency.indptr[target + 1];
                            let logits = range
                                .map(|edge| {
                                    let source = adaptive_adjacency.indices[edge];
                                    let score = (0..10).fold(tape.constant(0.0), |sum, latent| {
                                        tape.add(
                                            sum,
                                            tape.mul(
                                                parameter(
                                                    &tape,
                                                    adaptive_source + target * 10 + latent,
                                                ),
                                                parameter(
                                                    &tape,
                                                    adaptive_target + source * 10 + latent,
                                                ),
                                            ),
                                        )
                                    });
                                    tape.max(score, tape.constant(0.0))
                                })
                                .collect::<Vec<_>>();
                            if logits.is_empty() {
                                Vec::new().into_iter()
                            } else {
                                tape_softmax(&tape, &logits).into_iter()
                            }
                        })
                        .collect::<Vec<_>>();
                    let forward_one =
                        tape_csr_diffuse(&tape, adjacency, &adjacency_weights, values, hidden);
                    let forward_two = tape_csr_diffuse(
                        &tape,
                        adjacency,
                        &adjacency_weights,
                        &forward_one,
                        hidden,
                    );
                    let backward_one = tape_csr_diffuse(
                        &tape,
                        &reverse_adjacency,
                        &reverse_adjacency_weights,
                        values,
                        hidden,
                    );
                    let backward_two = tape_csr_diffuse(
                        &tape,
                        &reverse_adjacency,
                        &reverse_adjacency_weights,
                        &backward_one,
                        hidden,
                    );
                    let adaptive_one = tape_csr_diffuse(
                        &tape,
                        &adaptive_adjacency,
                        &adaptive_weights,
                        values,
                        hidden,
                    );
                    let adaptive_two = tape_csr_diffuse(
                        &tape,
                        &adaptive_adjacency,
                        &adaptive_weights,
                        &adaptive_one,
                        hidden,
                    );
                    (0..nodes)
                        .map(|node| {
                            let mut concatenated = values[node].clone();
                            concatenated.extend(&forward_one[node]);
                            concatenated.extend(&forward_two[node]);
                            concatenated.extend(&backward_one[node]);
                            concatenated.extend(&backward_two[node]);
                            concatenated.extend(&adaptive_one[node]);
                            concatenated.extend(&adaptive_two[node]);
                            tape_linear(
                                &tape,
                                &parameter_nodes,
                                layout.lsttn_periodic_projection
                                    + period * (7 * hidden * hidden + hidden),
                                &concatenated,
                                7 * hidden,
                                hidden,
                            )
                            .into_iter()
                            .enumerate()
                            .map(|(channel, value)| {
                                tape_deterministic_dropout_rate(
                                    &tape,
                                    value,
                                    self.steps ^ ((period as u64) << 40),
                                    node * hidden + channel,
                                    training,
                                    0.3,
                                )
                            })
                            .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut graphon_expert_states =
            vec![vec![vec![tape.constant(0.0); hidden]; self.experts]; nodes];
        let mut lsttn_short_sequence = None;
        let mut representation = vec![vec![0usize; hidden]; nodes];
        for node in 0..nodes {
            representation[node] = match profile {
                GraphTransformerProfile::HeterogeneousMoE => {
                    tape_add_vectors(&tape, &temporal[node], &spatial[node])
                }
                GraphTransformerProfile::EfficientHighOrder => {
                    stgformer_representation[node].clone()
                }
                GraphTransformerProfile::LongShortFusion => {
                    // Four stacked three-tap dilated convolutions (dilations
                    // 1, 2, 4, and 8) provide LSTTN's exponentially growing
                    // long-term receptive field.  Each tap mixes hidden
                    // channels through its own learned convolution kernel.
                    // The convolution consumes the masked-subseries encoder's
                    // patch representations, not raw timestamps; this is the
                    // level at which LSTTN extracts long trend and periodic
                    // features.
                    let patch_width = if long_context_is_pooled {
                        1
                    } else {
                        native_patch_width
                    };
                    let position_count =
                        (layout.pretrain_decoder - layout.pretrain_position) / hidden;
                    let subseries = if let Some(cached) = lsttn_frozen_patches {
                        cached
                            .iter()
                            .map(|patch| {
                                patch[node]
                                    .iter()
                                    .map(|value| tape.constant(*value as f64))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    } else {
                        embedding
                            .chunks(patch_width)
                            .enumerate()
                            .map(|(patch_index, patch)| {
                                (0..hidden)
                                    .map(|channel| {
                                        let sum =
                                            patch.iter().fold(tape.constant(0.0), |sum, row| {
                                                tape.add(sum, row[node][channel])
                                            });
                                        let pooled =
                                            tape.div(sum, tape.constant(patch.len() as f64));
                                        tape.tanh(tape.add(
                                            pooled,
                                            parameter(
                                                &tape,
                                                layout.pretrain_position
                                                    + (patch_index % position_count) * hidden
                                                    + channel,
                                            ),
                                        ))
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    };
                    let mut long_sequence = subseries.clone();
                    for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate() {
                        let layer_offset = layout.lsttn_dilated_convolution
                            + layer * (3 * hidden * hidden + hidden);
                        let convolution_times = long_sequence.len().div_ceil(2);
                        let mut convolved =
                            vec![vec![tape.constant(0.0); hidden]; convolution_times];
                        for output_time in 0..convolution_times {
                            for output_channel in 0..hidden {
                                let mut value = parameter(
                                    &tape,
                                    layer_offset + 3 * hidden * hidden + output_channel,
                                );
                                for tap in 0..3 {
                                    let centered = output_time * 2;
                                    let source_time = match tap {
                                        0 => centered.checked_sub(dilation),
                                        1 => Some(centered),
                                        _ => centered
                                            .checked_add(dilation)
                                            .filter(|time| *time < long_sequence.len()),
                                    };
                                    if let Some(source_time) = source_time {
                                        for input_channel in 0..hidden {
                                            value = tape.add(
                                                value,
                                                tape.mul(
                                                    parameter(
                                                        &tape,
                                                        layer_offset
                                                            + tap * hidden * hidden
                                                            + input_channel * hidden
                                                            + output_channel,
                                                    ),
                                                    long_sequence[source_time][input_channel],
                                                ),
                                            );
                                        }
                                    }
                                }
                                convolved[output_time][output_channel] = tape_gelu(&tape, value);
                            }
                        }
                        let pooled_times = convolved.len().div_ceil(2);
                        long_sequence = (0..pooled_times)
                            .map(|output_time| {
                                (0..hidden)
                                    .map(|channel| {
                                        let center = output_time * 2;
                                        [center.checked_sub(1), Some(center), Some(center + 1)]
                                            .into_iter()
                                            .flatten()
                                            .filter(|time| *time < convolved.len())
                                            .map(|time| convolved[time][channel])
                                            .reduce(|left, right| tape.max(left, right))
                                            .expect("nonempty LSTTN max-pool window")
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                    }
                    let long = long_sequence[long_sequence.len() - 1].clone();
                    let periodic_components = lsttn_periodic_features
                        .iter()
                        .map(|period| period[node].clone())
                        .collect::<Vec<_>>();
                    // The short branch is a Graph WaveNet-style stack rather
                    // than a reuse of the generic transformer attention.  A
                    // causal gated temporal convolution learns local traffic
                    // changes, then an input-conditioned adaptive adjacency
                    // propagates those changes across nodes.  Keeping this
                    // separate from the long dilation stack makes the
                    // long/short fusion an actual architectural distinction.
                    if lsttn_short_sequence.is_none() {
                        let short_start = times.saturating_sub(self.recent_window);
                        let start_projection = layout.lsttn_short_wave;
                        // The reference Graph WaveNet consumes the first two
                        // traffic-frame channels: the normalized signal and
                        // normalized time-of-day.  Left padding brings a
                        // 12-step short history to its 13-step receptive field.
                        let receptive_field = 13usize;
                        let raw_short_len = times - short_start;
                        let padded_len = (raw_short_len + 1).max(receptive_field);
                        let left_padding = padded_len - raw_short_len;
                        let mut short_sequence =
                            vec![vec![vec![tape.constant(0.0); hidden]; nodes]; padded_len];
                        for (local_time, absolute_time) in (short_start..times).enumerate() {
                            for current_node in 0..nodes {
                                let time_of_day = lsttn_time_features
                                    .map(|features| features[absolute_time][current_node])
                                    .unwrap_or_else(|| {
                                        ((phase_offset + absolute_time)
                                            % effective_periodicity.max(1))
                                            as f64
                                            / effective_periodicity.max(1) as f64
                                    });
                                let inputs = [
                                    observed_values[absolute_time][current_node],
                                    tape.constant(time_of_day),
                                ];
                                short_sequence[left_padding + local_time][current_node] =
                                    tape_linear(
                                        &tape,
                                        &parameter_nodes,
                                        start_projection,
                                        &inputs,
                                        2,
                                        hidden,
                                    );
                            }
                        }
                        let mut skip = vec![
                            vec![vec![tape.constant(0.0); hidden]; nodes];
                            short_sequence.len()
                        ];
                        let short_adaptive = (0..nodes)
                            .flat_map(|target| {
                                let logits = (adaptive_adjacency.indptr[target]
                                    ..adaptive_adjacency.indptr[target + 1])
                                    .map(|edge| {
                                        let source = adaptive_adjacency.indices[edge];
                                        let score =
                                            (0..10).fold(tape.constant(0.0), |sum, latent| {
                                                tape.add(
                                                    sum,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            layout.lsttn_short_adaptive_source
                                                                + target * 10
                                                                + latent,
                                                        ),
                                                        parameter(
                                                            &tape,
                                                            layout.lsttn_short_adaptive_target
                                                                + source * 10
                                                                + latent,
                                                        ),
                                                    ),
                                                )
                                            });
                                        tape.max(score, tape.constant(0.0))
                                    })
                                    .collect::<Vec<_>>();
                                if logits.is_empty() {
                                    Vec::new().into_iter()
                                } else {
                                    tape_softmax(&tape, &logits).into_iter()
                                }
                            })
                            .collect::<Vec<_>>();
                        let layer_width = 12 * hidden * hidden + 6 * hidden;
                        let layers_start = start_projection + 2 * hidden + hidden;
                        for (layer, dilation) in
                            [1usize, 2, 1, 2, 1, 2, 1, 2].into_iter().enumerate()
                        {
                            let layer_offset = layers_start + layer * layer_width;
                            let filter_offset = layer_offset;
                            let gate_offset = filter_offset + 2 * hidden * hidden;
                            let filter_bias = gate_offset + 2 * hidden * hidden;
                            let gate_bias = filter_bias + hidden;
                            let graph_projection = gate_bias + hidden;
                            let skip_projection = graph_projection + 7 * hidden * hidden + hidden;
                            let norm = skip_projection + hidden * hidden + hidden;
                            let output_times = short_sequence.len() - dilation;
                            let mut gated =
                                vec![vec![vec![tape.constant(0.0); hidden]; nodes]; output_times];
                            let skip_offset = skip.len() - output_times;
                            let mut cropped_skip =
                                vec![vec![vec![tape.constant(0.0); hidden]; nodes]; output_times];
                            for time in 0..output_times {
                                for current_node in 0..nodes {
                                    for output_channel in 0..hidden {
                                        let mut filter =
                                            parameter(&tape, filter_bias + output_channel);
                                        let mut gate = parameter(&tape, gate_bias + output_channel);
                                        for tap in 0..2 {
                                            let source_time = time + tap * dilation;
                                            for input_channel in 0..hidden {
                                                let source = short_sequence[source_time]
                                                    [current_node][input_channel];
                                                filter = tape.add(
                                                    filter,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            filter_offset
                                                                + tap * hidden * hidden
                                                                + input_channel * hidden
                                                                + output_channel,
                                                        ),
                                                        source,
                                                    ),
                                                );
                                                gate = tape.add(
                                                    gate,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            gate_offset
                                                                + tap * hidden * hidden
                                                                + input_channel * hidden
                                                                + output_channel,
                                                        ),
                                                        source,
                                                    ),
                                                );
                                            }
                                        }
                                        gated[time][current_node][output_channel] =
                                            tape.mul(tape.tanh(filter), tape.sigmoid(gate));
                                    }
                                    let projected_skip = tape_linear(
                                        &tape,
                                        &parameter_nodes,
                                        skip_projection,
                                        &gated[time][current_node],
                                        hidden,
                                        hidden,
                                    );
                                    for channel in 0..hidden {
                                        cropped_skip[time][current_node][channel] = tape.add(
                                            skip[skip_offset + time][current_node][channel],
                                            projected_skip[channel],
                                        );
                                    }
                                }
                            }
                            skip = cropped_skip;
                            let mut next = Vec::with_capacity(output_times);
                            for time in 0..output_times {
                                let forward_one = tape_csr_diffuse(
                                    &tape,
                                    adjacency,
                                    &adjacency_weights,
                                    &gated[time],
                                    hidden,
                                );
                                let forward_two = tape_csr_diffuse(
                                    &tape,
                                    adjacency,
                                    &adjacency_weights,
                                    &forward_one,
                                    hidden,
                                );
                                let backward_one = tape_csr_diffuse(
                                    &tape,
                                    &reverse_adjacency,
                                    &reverse_adjacency_weights,
                                    &gated[time],
                                    hidden,
                                );
                                let backward_two = tape_csr_diffuse(
                                    &tape,
                                    &reverse_adjacency,
                                    &reverse_adjacency_weights,
                                    &backward_one,
                                    hidden,
                                );
                                let adaptive_one = tape_csr_diffuse(
                                    &tape,
                                    &adaptive_adjacency,
                                    &short_adaptive,
                                    &gated[time],
                                    hidden,
                                );
                                let adaptive_two = tape_csr_diffuse(
                                    &tape,
                                    &adaptive_adjacency,
                                    &short_adaptive,
                                    &adaptive_one,
                                    hidden,
                                );
                                next.push(
                                    (0..nodes)
                                        .map(|current_node| {
                                            let mut features = gated[time][current_node].clone();
                                            features.extend(&forward_one[current_node]);
                                            features.extend(&forward_two[current_node]);
                                            features.extend(&backward_one[current_node]);
                                            features.extend(&backward_two[current_node]);
                                            features.extend(&adaptive_one[current_node]);
                                            features.extend(&adaptive_two[current_node]);
                                            let graph = tape_linear(
                                                &tape,
                                                &parameter_nodes,
                                                graph_projection,
                                                &features,
                                                7 * hidden,
                                                hidden,
                                            );
                                            graph
                                                .iter()
                                                .zip(&short_sequence[time + dilation][current_node])
                                                .enumerate()
                                                .map(|(channel, (value, residual))| {
                                                    let value = tape_deterministic_dropout_rate(
                                                        &tape,
                                                        *value,
                                                        self.steps ^ ((layer as u64) << 48),
                                                        (time * nodes + current_node) * hidden
                                                            + channel,
                                                        training,
                                                        0.3,
                                                    );
                                                    tape.add(value, *residual)
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .collect::<Vec<_>>(),
                                );
                            }
                            for channel in 0..hidden {
                                let count = (next.len() * nodes) as f64;
                                let mean = next.iter().flatten().fold(
                                    tape.constant(0.0),
                                    |sum, values| {
                                        tape.add(
                                            sum,
                                            tape.mul(values[channel], tape.constant(1.0 / count)),
                                        )
                                    },
                                );
                                let variance = next.iter().flatten().fold(
                                    tape.constant(0.0),
                                    |sum, values| {
                                        let centered = tape.add(
                                            values[channel],
                                            tape.mul(mean, tape.constant(-1.0)),
                                        );
                                        tape.add(
                                            sum,
                                            tape.mul(
                                                tape.mul(centered, centered),
                                                tape.constant(1.0 / count),
                                            ),
                                        )
                                    },
                                );
                                let denominator =
                                    tape.sqrt(tape.add(variance, tape.constant(1e-5)));
                                for time in &mut next {
                                    for values in time {
                                        values[channel] = tape.add(
                                            tape.mul(
                                                tape.div(
                                                    tape.add(
                                                        values[channel],
                                                        tape.mul(mean, tape.constant(-1.0)),
                                                    ),
                                                    denominator,
                                                ),
                                                parameter(&tape, norm + channel),
                                            ),
                                            parameter(&tape, norm + hidden + channel),
                                        );
                                    }
                                }
                            }
                            short_sequence = next;
                        }
                        let end_one = layers_start + 8 * layer_width;
                        let end_two = end_one + hidden * hidden + hidden;
                        let final_short = (0..nodes)
                            .map(|current_node| {
                                let activated = skip[skip.len() - 1][current_node]
                                    .iter()
                                    .map(|value| tape.max(*value, tape.constant(0.0)))
                                    .collect::<Vec<_>>();
                                let hidden_values = tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    end_one,
                                    &activated,
                                    hidden,
                                    hidden,
                                )
                                .into_iter()
                                .map(|value| tape.max(value, tape.constant(0.0)))
                                .collect::<Vec<_>>();
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    end_two,
                                    &hidden_values,
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        lsttn_short_sequence = Some(vec![final_short]);
                    }
                    let short_sequence = lsttn_short_sequence
                        .as_ref()
                        .expect("LSTTN short branch is initialized");
                    // The paper concatenates the long-trend, weekly, and
                    // daily graph features, sends them through a two-layer
                    // trend-seasonality MLP, then concatenates that result
                    // with the Graph WaveNet short-term state for a final
                    // MLP.  A learned gate is not equivalent to this path.
                    let zero = tape.constant(0.0);
                    let daily = periodic_components
                        .first()
                        .cloned()
                        .unwrap_or_else(|| vec![zero; hidden]);
                    let weekly = periodic_components
                        .get(1)
                        .cloned()
                        .unwrap_or_else(|| vec![zero; hidden]);
                    let mut trend_seasonality_input = long;
                    trend_seasonality_input.extend(daily);
                    trend_seasonality_input.extend(weekly);
                    let first_trend_seasonality = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.lsttn_fusion,
                        &trend_seasonality_input,
                        3 * hidden,
                        hidden,
                    )
                    .into_iter()
                    .map(|value| tape.max(value, zero))
                    .collect::<Vec<_>>();
                    let trend_seasonality = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.lsttn_fusion + (3 * hidden + 1) * hidden,
                        &first_trend_seasonality,
                        hidden,
                        hidden,
                    );
                    let mut final_input = short_sequence[short_sequence.len() - 1][node].clone();
                    final_input.extend(trend_seasonality);
                    tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.lsttn_fusion + (4 * hidden * hidden + 2 * hidden),
                        &final_input,
                        2 * hidden,
                        hidden,
                    )
                    .into_iter()
                    .map(|value| tape.max(value, zero))
                    .collect()
                }
                GraphTransformerProfile::GatedGraphTemporal => {
                    let mut gated = vec![0usize; hidden];
                    for channel in 0..hidden {
                        let reset = tape.sigmoid(tape.add(
                            temporal[node][channel],
                            tape.mul(
                                parameter(&tape, layout.recurrence + 1 + channel),
                                graph_convolution[node][channel],
                            ),
                        ));
                        let update = tape.sigmoid(tape.add(
                            graph_convolution[node][channel],
                            parameter(&tape, layout.recurrence + 1 + hidden + channel),
                        ));
                        let candidate = tape.tanh(tape.add(
                            temporal[node][channel],
                            tape.mul(reset, graph_convolution[node][channel]),
                        ));
                        gated[channel] = tape.add(
                            tape.mul(update, temporal[node][channel]),
                            tape.mul(
                                tape.add(tape.constant(1.0), tape.mul(tape.constant(-1.0), update)),
                                candidate,
                            ),
                        );
                    }
                    gated
                }
                GraphTransformerProfile::SpatialShiftGraphonMoE => {
                    for expert in 0..self.experts {
                        let source_embedding =
                            parameter(&tape, layout.graphon_nodes + expert * nodes + node);
                        for other in 0..nodes {
                            let target_embedding =
                                parameter(&tape, layout.graphon_nodes + expert * nodes + other);
                            for channel in 0..hidden {
                                let temporal_embedding = parameter(
                                    &tape,
                                    layout.graphon_time + expert * hidden + channel,
                                );
                                let edge_logit = tape.add(
                                    tape.add(
                                        tape.mul(source_embedding, target_embedding),
                                        temporal_embedding,
                                    ),
                                    temporal[node][channel],
                                );
                                // A binary Gumbel-Softmax relaxation samples
                                // a graph from each expert graphon while
                                // retaining a pathwise gradient.  `steps` is
                                // advanced by every optimizer update, so
                                // training sees fresh samples; a fitted model
                                // uses its final serialized step and therefore
                                // makes deterministic predictions.
                                let sample_noise = tape.constant(graphon_gumbel_logistic_noise(
                                    self.steps, expert, node, other, channel,
                                ));
                                let edge_probability =
                                    tape.sigmoid(tape.mul(
                                        tape.constant(2.0),
                                        tape.add(edge_logit, sample_noise),
                                    ));
                                graphon_expert_states[node][expert][channel] = tape.add(
                                    graphon_expert_states[node][expert][channel],
                                    tape.mul(edge_probability, spatial[other][channel]),
                                );
                            }
                        }
                    }
                    // Each expert remains separate through its router and
                    // forecast head; `representation` is only a placeholder
                    // for the common branch below.
                    temporal[node].clone()
                }
            };
        }
        let mut outputs = vec![vec![0usize; self.horizons]; nodes];
        let mut router_weights = Vec::with_capacity(nodes);
        for node in 0..nodes {
            let routed = matches!(
                profile,
                GraphTransformerProfile::HeterogeneousMoE
                    | GraphTransformerProfile::SpatialShiftGraphonMoE
            );
            if routed {
                if *profile == GraphTransformerProfile::HeterogeneousMoE {
                    // STGormer routes the temporal and spatial transformer
                    // outputs independently.  They must not share either a
                    // gate or expert FNN: those are the mechanisms that let
                    // a road/time regime select different specialists along
                    // the two axes.
                    let temporal_logits = (0..self.experts)
                        .map(|expert| {
                            let mut logit =
                                parameter(&tape, layout.router + expert * (hidden + 1) + hidden);
                            for channel in 0..hidden {
                                logit = tape.add(
                                    logit,
                                    tape.mul(
                                        parameter(
                                            &tape,
                                            layout.router + expert * (hidden + 1) + channel,
                                        ),
                                        temporal[node][channel],
                                    ),
                                );
                            }
                            logit
                        })
                        .collect::<Vec<_>>();
                    let spatial_logits = (0..self.experts)
                        .map(|expert| {
                            let mut logit = parameter(
                                &tape,
                                layout.spatial_router + expert * (hidden + 1) + hidden,
                            );
                            for channel in 0..hidden {
                                logit = tape.add(
                                    logit,
                                    tape.mul(
                                        parameter(
                                            &tape,
                                            layout.spatial_router + expert * (hidden + 1) + channel,
                                        ),
                                        spatial[node][channel],
                                    ),
                                );
                            }
                            logit
                        })
                        .collect::<Vec<_>>();
                    let temporal_weights = tape_softmax(&tape, &temporal_logits);
                    let spatial_weights = tape_softmax(&tape, &spatial_logits);
                    router_weights.push(temporal_weights.clone());
                    router_weights.push(spatial_weights.clone());
                    for horizon in 0..self.horizons {
                        let mut temporal_result = tape.constant(0.0);
                        let mut spatial_result = tape.constant(0.0);
                        for expert in 0..self.experts {
                            let temporal_offset = layout.expert_heads
                                + expert * self.horizons * (hidden + 1)
                                + horizon * (hidden + 1);
                            let spatial_offset = layout.spatial_expert_heads
                                + expert * self.horizons * (hidden + 1)
                                + horizon * (hidden + 1);
                            let mut temporal_head = parameter(&tape, temporal_offset + hidden);
                            let mut spatial_head = parameter(&tape, spatial_offset + hidden);
                            for channel in 0..hidden {
                                temporal_head = tape.add(
                                    temporal_head,
                                    tape.mul(
                                        parameter(&tape, temporal_offset + channel),
                                        temporal[node][channel],
                                    ),
                                );
                                spatial_head = tape.add(
                                    spatial_head,
                                    tape.mul(
                                        parameter(&tape, spatial_offset + channel),
                                        spatial[node][channel],
                                    ),
                                );
                            }
                            temporal_result = tape.add(
                                temporal_result,
                                tape.mul(temporal_weights[expert], temporal_head),
                            );
                            spatial_result = tape.add(
                                spatial_result,
                                tape.mul(spatial_weights[expert], spatial_head),
                            );
                        }
                        outputs[node][horizon] = tape.mul(
                            tape.constant(0.5),
                            tape.add(temporal_result, spatial_result),
                        );
                    }
                } else {
                    let mut logits = Vec::with_capacity(self.experts);
                    for expert in 0..self.experts {
                        let expert_representation =
                            if *profile == GraphTransformerProfile::SpatialShiftGraphonMoE {
                                let graphon_state = if excluded_expert.is_some() {
                                    tape_detach_vectors(&tape, &graphon_expert_states[node][expert])
                                } else {
                                    graphon_expert_states[node][expert].clone()
                                };
                                tape_add_vectors(&tape, &temporal[node], &graphon_state)
                            } else {
                                representation[node].clone()
                            };
                        let mut logit =
                            parameter(&tape, layout.router + expert * (hidden + 1) + hidden);
                        for channel in 0..hidden {
                            logit = tape.add(
                                logit,
                                tape.mul(
                                    parameter(
                                        &tape,
                                        layout.router + expert * (hidden + 1) + channel,
                                    ),
                                    expert_representation[channel],
                                ),
                            );
                        }
                        if excluded_expert == Some(expert) {
                            logit = tape.add(logit, tape.constant(-30.0));
                        }
                        logits.push(logit);
                    }
                    let weights = tape_softmax(&tape, &logits);
                    for horizon in 0..self.horizons {
                        let mut result = tape.constant(0.0);
                        for expert in 0..self.experts {
                            let expert_representation = if *profile
                                == GraphTransformerProfile::SpatialShiftGraphonMoE
                            {
                                let graphon_state = if excluded_expert.is_some() {
                                    tape_detach_vectors(&tape, &graphon_expert_states[node][expert])
                                } else {
                                    graphon_expert_states[node][expert].clone()
                                };
                                tape_add_vectors(&tape, &temporal[node], &graphon_state)
                            } else {
                                representation[node].clone()
                            };
                            let offset = layout.expert_heads
                                + expert * self.horizons * (hidden + 1)
                                + horizon * (hidden + 1);
                            let mut head = parameter(&tape, offset + hidden);
                            for channel in 0..hidden {
                                head = tape.add(
                                    head,
                                    tape.mul(
                                        parameter(&tape, offset + channel),
                                        expert_representation[channel],
                                    ),
                                );
                            }
                            result = tape.add(result, tape.mul(weights[expert], head));
                        }
                        outputs[node][horizon] = result;
                    }
                }
            } else {
                for horizon in 0..self.horizons {
                    let offset = layout.output + horizon * (hidden + 1);
                    let mut head = parameter(&tape, offset + hidden);
                    for channel in 0..hidden {
                        head = tape.add(
                            head,
                            tape.mul(
                                parameter(&tape, offset + channel),
                                representation[node][channel],
                            ),
                        );
                    }
                    outputs[node][horizon] = head;
                }
            }
        }
        (tape, outputs, router_weights, representation)
    }
}

fn tape_linear(
    tape: &AutodiffTape,
    parameter_nodes: &[usize],
    offset: usize,
    input: &[usize],
    input_width: usize,
    output_width: usize,
) -> Vec<usize> {
    (0..output_width)
        .map(|output| {
            let mut value = parameter_nodes[offset + input_width * output_width + output];
            for (index, input_value) in input.iter().enumerate().take(input_width) {
                value = tape.add(
                    value,
                    tape.mul(
                        parameter_nodes[offset + index * output_width + output],
                        *input_value,
                    ),
                );
            }
            value
        })
        .collect()
}

fn numeric_linear_batch(
    parameters: &[f64],
    offset: usize,
    inputs: &[Vec<f64>],
    input_width: usize,
    output_width: usize,
    backend: Option<&BackendSelection>,
) -> Result<Vec<Vec<f64>>> {
    let weight_count = input_width * output_width;
    dense_matrix_product(
        inputs,
        &parameters[offset..offset + weight_count],
        &parameters[offset + weight_count..offset + weight_count + output_width],
        input_width,
        output_width,
        backend,
    )
}

fn dense_matrix_product(
    inputs: &[Vec<f64>],
    weights: &[f64],
    biases: &[f64],
    input_width: usize,
    output_width: usize,
    backend: Option<&BackendSelection>,
) -> Result<Vec<Vec<f64>>> {
    let operations = inputs
        .len()
        .saturating_mul(input_width)
        .saturating_mul(output_width);
    if let Some(selection) = backend.filter(|selection| {
        selection.selected != "cpu" && operations >= GEO_ST_DENSE_DISPATCH_MIN_OPS
    }) {
        let features = inputs
            .iter()
            .map(|row| row.iter().map(|value| *value as f32).collect())
            .collect::<Vec<Vec<f32>>>();
        let weights = weights
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let biases = biases.iter().map(|value| *value as f32).collect::<Vec<_>>();
        return backend_dense_layer_f32(selection, &features, &weights, &biases)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.into_iter().map(f64::from).collect())
                    .collect()
            })
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()));
    }
    Ok(inputs
        .iter()
        .map(|input| {
            (0..output_width)
                .map(|output| {
                    input
                        .iter()
                        .enumerate()
                        .take(input_width)
                        .fold(biases[output], |sum, (index, value)| {
                            sum + weights[index * output_width + output] * value
                        })
                })
                .collect()
        })
        .collect())
}

fn numeric_layer_norm(parameters: &[f64], offset: usize, values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let denominator = (variance + 1e-5).sqrt();
    values
        .iter()
        .enumerate()
        .map(|(channel, value)| {
            (value - mean) / denominator * parameters[offset + channel]
                + parameters[offset + values.len() + channel]
        })
        .collect()
}

fn numeric_layer_norm_batch(
    parameters: &[f64],
    offset: usize,
    values: &[Vec<f64>],
    backend: Option<&BackendSelection>,
) -> Result<Vec<Vec<f64>>> {
    let rows = values.len();
    let width = values.first().map_or(0, Vec::len);
    let operations = rows.saturating_mul(width);
    if let Some(selection) = backend.filter(|selection| {
        selection.selected != "cpu" && operations >= GEO_ST_DENSE_DISPATCH_MIN_OPS
    }) {
        let flat_values = values
            .iter()
            .flatten()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let gamma = parameters[offset..offset + width]
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let beta = parameters[offset + width..offset + 2 * width]
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let normalized =
            backend_layer_norm_f32(selection, &flat_values, rows, width, &gamma, &beta)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        return Ok(normalized
            .chunks(width)
            .map(|row| row.iter().copied().map(f64::from).collect())
            .collect());
    }
    Ok(values
        .iter()
        .map(|row| numeric_layer_norm(parameters, offset, row))
        .collect())
}

fn attention_row_softmax(
    logits: &mut [Vec<f64>],
    scale: f64,
    backend: Option<&BackendSelection>,
) -> Result<()> {
    let rows = logits.len();
    let width = logits.first().map_or(0, Vec::len);
    let operations = rows.saturating_mul(width);
    if let Some(selection) = backend.filter(|selection| {
        selection.selected != "cpu" && operations >= GEO_ST_DENSE_DISPATCH_MIN_OPS
    }) {
        let indptr = (0..=rows)
            .map(|row| u32::try_from(row * width))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                GeoStError::InvalidFrame("attention matrix exceeds u32 CSR limits".into())
            })?;
        let values = logits
            .iter()
            .flatten()
            .map(|value| (*value * scale) as f32)
            .collect::<Vec<_>>();
        let weights = backend_csr_row_softmax_f32(selection, &indptr, &values)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        for (target, weight) in logits.iter_mut().flatten().zip(weights) {
            *target = f64::from(weight);
        }
        return Ok(());
    }
    for weights in logits {
        let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max) * scale;
        for weight in weights.iter_mut() {
            *weight = (*weight * scale - max).exp();
        }
        let denominator = weights.iter().sum::<f64>().max(1e-12);
        for weight in weights {
            *weight /= denominator;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn numeric_transformer_encoder_layer(
    parameters: &[f64],
    sequence: &[Vec<f64>],
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    out_offset: usize,
    ffn_offset: usize,
    norm_offset: usize,
    hidden: usize,
    heads: usize,
    backend: Option<&BackendSelection>,
) -> Result<Vec<Vec<f64>>> {
    let queries = numeric_linear_batch(parameters, q_offset, sequence, hidden, hidden, backend)?;
    let keys = numeric_linear_batch(parameters, k_offset, sequence, hidden, hidden, backend)?;
    let values = numeric_linear_batch(parameters, v_offset, sequence, hidden, hidden, backend)?;
    let tokens = sequence.len();
    let mut attended = vec![vec![0.0; hidden]; tokens];
    for head in 0..heads {
        let start = head * hidden / heads;
        let end = (head + 1) * hidden / heads;
        let head_width = end - start;
        let scale = 1.0 / (head_width as f64).sqrt();
        let head_queries = queries
            .iter()
            .map(|query| query[start..end].to_vec())
            .collect::<Vec<_>>();
        let transposed_keys = (start..end)
            .flat_map(|channel| keys.iter().map(move |key| key[channel]))
            .collect::<Vec<_>>();
        let mut attention_weights = dense_matrix_product(
            &head_queries,
            &transposed_keys,
            &vec![0.0; tokens],
            head_width,
            tokens,
            backend,
        )?;
        attention_row_softmax(&mut attention_weights, scale, backend)?;
        let mut value_weights = Vec::with_capacity(tokens * head_width);
        for value in &values {
            value_weights.extend_from_slice(&value[start..end]);
        }
        let head_values = dense_matrix_product(
            &attention_weights,
            &value_weights,
            &vec![0.0; head_width],
            tokens,
            head_width,
            backend,
        )?;
        for (attended, head_values) in attended.iter_mut().zip(head_values) {
            attended[start..end].copy_from_slice(&head_values);
        }
    }
    let projected =
        numeric_linear_batch(parameters, out_offset, &attended, hidden, hidden, backend)?;
    let first_residual = sequence
        .iter()
        .zip(projected)
        .map(|(residual, projected)| {
            residual
                .iter()
                .zip(projected)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let normalized = numeric_layer_norm_batch(parameters, norm_offset, &first_residual, backend)?;
    let mut expanded = numeric_linear_batch(
        parameters,
        ffn_offset,
        &normalized,
        hidden,
        4 * hidden,
        backend,
    )?;
    expanded
        .iter_mut()
        .flatten()
        .for_each(|value| *value = value.max(0.0));
    let contracted = numeric_linear_batch(
        parameters,
        ffn_offset + (hidden + 1) * 4 * hidden,
        &expanded,
        4 * hidden,
        hidden,
        backend,
    )?;
    let second_residual = normalized
        .iter()
        .zip(contracted)
        .map(|(normalized, contracted)| {
            normalized
                .iter()
                .zip(contracted)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    numeric_layer_norm_batch(
        parameters,
        norm_offset + 2 * hidden,
        &second_residual,
        backend,
    )
}

fn tape_layer_norm(
    tape: &AutodiffTape,
    parameters: &[usize],
    offset: usize,
    values: &[usize],
) -> Vec<usize> {
    let width = values.len();
    let inverse = tape.constant(1.0 / width as f64);
    let mean = values.iter().fold(tape.constant(0.0), |sum, value| {
        tape.add(sum, tape.mul(*value, inverse))
    });
    let variance = values.iter().fold(tape.constant(0.0), |sum, value| {
        let centered = tape.add(*value, tape.mul(mean, tape.constant(-1.0)));
        tape.add(sum, tape.mul(tape.mul(centered, centered), inverse))
    });
    let denominator = tape.sqrt(tape.add(variance, tape.constant(1e-5)));
    values
        .iter()
        .enumerate()
        .map(|(channel, value)| {
            let centered = tape.add(*value, tape.mul(mean, tape.constant(-1.0)));
            tape.add(
                tape.mul(
                    tape.div(centered, denominator),
                    parameters[offset + channel],
                ),
                parameters[offset + width + channel],
            )
        })
        .collect()
}

fn tape_deterministic_dropout(
    tape: &AutodiffTape,
    value: usize,
    seed: u64,
    index: usize,
    enabled: bool,
) -> usize {
    tape_deterministic_dropout_rate(tape, value, seed, index, enabled, 0.1)
}

fn tape_deterministic_dropout_rate(
    tape: &AutodiffTape,
    value: usize,
    seed: u64,
    index: usize,
    enabled: bool,
    probability: f64,
) -> usize {
    if !enabled {
        return value;
    }
    let mut state = seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let threshold = (probability * 10_000.0).round() as u64;
    if state % 10_000 < threshold {
        tape.constant(0.0)
    } else {
        tape.mul(value, tape.constant(1.0 / (1.0 - probability)))
    }
}

fn tape_gelu(tape: &AutodiffTape, value: usize) -> usize {
    let cube = tape.mul(tape.mul(value, value), value);
    let inner = tape.mul(
        tape.constant((2.0 / std::f64::consts::PI).sqrt()),
        tape.add(value, tape.mul(tape.constant(0.044715), cube)),
    );
    tape.mul(
        tape.constant(0.5),
        tape.mul(value, tape.add(tape.constant(1.0), tape.tanh(inner))),
    )
}

#[allow(clippy::too_many_arguments)]
fn tape_transformer_encoder_layer(
    tape: &AutodiffTape,
    parameters: &[usize],
    sequence: &[Vec<usize>],
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    out_offset: usize,
    ffn_offset: usize,
    norm_offset: usize,
    hidden: usize,
    heads: usize,
    dropout_seed: u64,
    dropout: bool,
) -> Vec<Vec<usize>> {
    let queries = sequence
        .iter()
        .map(|token| tape_linear(tape, parameters, q_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let keys = sequence
        .iter()
        .map(|token| tape_linear(tape, parameters, k_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let values = sequence
        .iter()
        .map(|token| tape_linear(tape, parameters, v_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    sequence
        .iter()
        .enumerate()
        .map(|(token_index, residual)| {
            let mut attended = vec![tape.constant(0.0); hidden];
            for head in 0..heads {
                let start = head * hidden / heads;
                let end = (head + 1) * hidden / heads;
                let scale = tape.constant(1.0 / ((end - start) as f64).sqrt());
                let logits = keys
                    .iter()
                    .map(|key| {
                        tape.mul(
                            scale,
                            tape_dot(tape, &queries[token_index][start..end], &key[start..end]),
                        )
                    })
                    .collect::<Vec<_>>();
                let weights = tape_softmax(tape, &logits)
                    .into_iter()
                    .enumerate()
                    .map(|(key_index, weight)| {
                        tape_deterministic_dropout(
                            tape,
                            weight,
                            dropout_seed ^ 0x6a09_e667_f3bc_c909,
                            (token_index * sequence.len() + key_index) * heads + head,
                            dropout,
                        )
                    })
                    .collect::<Vec<_>>();
                let head_values = values
                    .iter()
                    .map(|value| value[start..end].to_vec())
                    .collect::<Vec<_>>();
                attended[start..end].copy_from_slice(&tape_weighted_sum(
                    tape,
                    &weights,
                    &head_values,
                    end - start,
                ));
            }
            let projected = tape_linear(tape, parameters, out_offset, &attended, hidden, hidden);
            let first_residual = residual
                .iter()
                .zip(projected)
                .enumerate()
                .map(|(channel, (skip, value))| {
                    tape.add(
                        *skip,
                        tape_deterministic_dropout(
                            tape,
                            value,
                            dropout_seed,
                            token_index * hidden + channel,
                            dropout,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let normalized = tape_layer_norm(tape, parameters, norm_offset, &first_residual);
            let expanded = tape_linear(
                tape,
                parameters,
                ffn_offset,
                &normalized,
                hidden,
                4 * hidden,
            )
            .into_iter()
            .map(|value| tape.max(value, tape.constant(0.0)))
            .collect::<Vec<_>>();
            let contracted = tape_linear(
                tape,
                parameters,
                ffn_offset + (hidden + 1) * 4 * hidden,
                &expanded,
                4 * hidden,
                hidden,
            );
            let second_residual = normalized
                .iter()
                .zip(contracted)
                .enumerate()
                .map(|(channel, (skip, value))| {
                    tape.add(
                        *skip,
                        tape_deterministic_dropout(
                            tape,
                            value,
                            dropout_seed ^ 0xa5a5_a5a5_a5a5_a5a5,
                            token_index * hidden + channel,
                            dropout,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            tape_layer_norm(tape, parameters, norm_offset + 2 * hidden, &second_residual)
        })
        .collect()
}

fn periodic_phase(absolute_step: usize, period: usize) -> f64 {
    (absolute_step as f64 * std::f64::consts::TAU / period.max(1) as f64).sin()
}

fn tape_dot(tape: &AutodiffTape, left: &[usize], right: &[usize]) -> usize {
    left.iter()
        .zip(right)
        .fold(tape.constant(0.0), |sum, (a, b)| {
            tape.add(sum, tape.mul(*a, *b))
        })
}

fn clip_gradient_norm(gradients: &mut [f64], maximum_norm: f64) {
    let norm = gradients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > maximum_norm {
        let scale = maximum_norm / norm;
        for gradient in gradients {
            *gradient *= scale;
        }
    }
}

/// Logistic noise obtained from the difference of two Gumbel variates.  With
/// the sigmoid relaxation in the graphon branch, this is the binary form of
/// Gumbel-Softmax.  It is deterministic for a serialized optimizer step so a
/// saved model cannot change predictions merely by being reloaded.
fn graphon_gumbel_logistic_noise(
    step: u64,
    expert: usize,
    target: usize,
    source: usize,
    channel: usize,
) -> f64 {
    let mut value = step
        ^ (expert as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (target as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (source as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ (channel as u64).wrapping_mul(0xD6E8_FD50_6A6A_5A93);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let uniform = ((value >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64);
    (uniform / (1.0 - uniform)).ln()
}

/// Summary shared by every query in one STGformer attention head.  Construct
/// this once rather than materializing an N-by-N attention matrix.
fn tape_stgformer_attention_summary(
    tape: &AutodiffTape,
    keys: &[Vec<usize>],
    values: &[Vec<usize>],
) -> (Vec<usize>, Vec<Vec<usize>>, usize) {
    let keys = keys
        .iter()
        .map(|key| tape_l2_normalize(tape, key))
        .collect::<Vec<_>>();
    let width = keys.first().map_or(0, Vec::len);
    let key_sum = (0..width)
        .map(|channel| {
            keys.iter()
                .fold(tape.constant(0.0), |sum, key| tape.add(sum, key[channel]))
        })
        .collect::<Vec<_>>();
    let key_value = (0..key_sum.len())
        .map(|key_channel| {
            (0..key_sum.len())
                .map(|value_channel| {
                    keys.iter()
                        .zip(values)
                        .fold(tape.constant(0.0), |sum, (key, value)| {
                            tape.add(sum, tape.mul(key[key_channel], value[value_channel]))
                        })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (key_sum, key_value, keys.len())
}

/// STGformer's scaling-normalized efficient attention.  The official layer
/// computes `Q(K^T V) + N V` over `Q(sum K) + N`; the residual V belongs to
/// the current query position, not an arbitrary key/value position.
fn tape_stgformer_fast_attention(
    tape: &AutodiffTape,
    query: &[usize],
    summary: &(Vec<usize>, Vec<Vec<usize>>, usize),
    residual_value: &[usize],
) -> Vec<usize> {
    let query = tape_l2_normalize(tape, query);
    let (key_sum, key_value, count) = summary;
    let positions = tape.constant(*count as f64);
    let denominator = tape.add(tape_dot(tape, &query, key_sum), positions);
    (0..query.len())
        .map(|output| {
            let mut numerator = tape.mul(positions, residual_value[output]);
            for channel in 0..query.len() {
                numerator = tape.add(
                    numerator,
                    tape.mul(query[channel], key_value[channel][output]),
                );
            }
            tape.div(numerator, denominator)
        })
        .collect()
}

fn tape_l2_normalize(tape: &AutodiffTape, values: &[usize]) -> Vec<usize> {
    let squared_norm = values.iter().fold(tape.constant(1e-12), |sum, value| {
        tape.add(sum, tape.mul(*value, *value))
    });
    let norm = tape.sqrt(squared_norm);
    values.iter().map(|value| tape.div(*value, norm)).collect()
}

#[cfg(test)]
fn adaptive_neighbor_indices<'a>(
    adjacency: &'a CsrAdjacency,
    node: usize,
    fallback: &'a [usize; 1],
) -> &'a [usize] {
    let neighbors = &adjacency.indices[adjacency.indptr[node]..adjacency.indptr[node + 1]];
    if neighbors.is_empty() {
        fallback
    } else {
        neighbors
    }
}

fn tape_softmax(tape: &AutodiffTape, logits: &[usize]) -> Vec<usize> {
    let max_logit = logits
        .iter()
        .copied()
        .reduce(|left, right| tape.max(left, right))
        .expect("softmax requires at least one logit");
    let shift = tape.mul(tape.constant(-1.0), tape.stop_gradient(max_logit));
    let exponentials = logits
        .iter()
        // Attention and MoE routing use the actual Transformer softmax.  The
        // detached max shift is algebraically invariant and avoids overflow.
        .map(|value| tape.exp(tape.add(*value, shift)))
        .collect::<Vec<_>>();
    let denominator = exponentials
        .iter()
        .fold(tape.constant(0.0), |sum, value| tape.add(sum, *value));
    exponentials
        .into_iter()
        .map(|value| tape.div(value, denominator))
        .collect()
}
fn tape_weighted_sum(
    tape: &AutodiffTape,
    weights: &[usize],
    values: &[Vec<usize>],
    width: usize,
) -> Vec<usize> {
    (0..width)
        .map(|channel| {
            weights
                .iter()
                .zip(values)
                .fold(tape.constant(0.0), |sum, (weight, value)| {
                    tape.add(sum, tape.mul(*weight, value[channel]))
                })
        })
        .collect()
}

fn tape_csr_diffuse(
    tape: &AutodiffTape,
    adjacency: &CsrAdjacency,
    weights: &[usize],
    values: &[Vec<usize>],
    hidden: usize,
) -> Vec<Vec<usize>> {
    (0..values.len())
        .map(|target| {
            (0..hidden)
                .map(|channel| {
                    (adjacency.indptr[target]..adjacency.indptr[target + 1]).fold(
                        tape.constant(0.0),
                        |sum, edge| {
                            tape.add(
                                sum,
                                tape.mul(weights[edge], values[adjacency.indices[edge]][channel]),
                            )
                        },
                    )
                })
                .collect()
        })
        .collect()
}

fn tape_add_vectors(tape: &AutodiffTape, left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .zip(right)
        .map(|(a, b)| tape.add(*a, *b))
        .collect()
}

/// Preserve a graphon value for episodic mixup while intentionally stopping
/// its gradient.  The episode then learns only how to combine independently
/// trained expert graphons for the held-out environment.
fn tape_detach_vectors(tape: &AutodiffTape, values: &[usize]) -> Vec<usize> {
    values
        .iter()
        .map(|value| tape.stop_gradient(*value))
        .collect()
}

#[derive(Clone, Copy)]
enum TapeOp {
    Constant,
    Parameter(usize),
    Add(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Tanh(usize),
    Exp(usize),
    Sqrt(usize),
    Sin(usize),
    Sigmoid(usize),
    Max(usize, usize),
    StopGradient(usize),
}

#[derive(Clone, Copy)]
struct TapeNode {
    value: f64,
    op: TapeOp,
}

struct AutodiffTape {
    nodes: RefCell<Vec<TapeNode>>,
    deferred: bool,
}

impl AutodiffTape {
    fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
            deferred: false,
        }
    }
    fn deferred() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
            deferred: true,
        }
    }
    fn constant(&self, value: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Constant,
        });
        index
    }
    fn parameter(&self, parameter: usize, value: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Parameter(parameter),
        });
        index
    }
    fn add(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value + nodes[right].value
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Add(left, right),
        });
        index
    }
    fn mul(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value * nodes[right].value
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Mul(left, right),
        });
        index
    }
    fn div(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value / nodes[right].value.max(1e-12)
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Div(left, right),
        });
        index
    }
    fn tanh(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.tanh()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Tanh(input),
        });
        index
    }
    fn exp(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.exp()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Exp(input),
        });
        index
    }
    fn sqrt(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.max(1e-12).sqrt()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sqrt(input),
        });
        index
    }
    fn sin(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value.sin()
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sin(input),
        });
        index
    }
    fn sigmoid(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            sigmoid(nodes[input].value)
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sigmoid(input),
        });
        index
    }
    fn max(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[left].value.max(nodes[right].value)
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::Max(left, right),
        });
        index
    }
    fn stop_gradient(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = if self.deferred {
            0.0
        } else {
            nodes[input].value
        };
        nodes.push(TapeNode {
            value,
            op: TapeOp::StopGradient(input),
        });
        index
    }
    fn accelerated_values(&self, selection: &BackendSelection) -> Result<Vec<f32>> {
        let (initial_values, opcodes, left, right, _) = self.accelerator_arrays();
        backend_scalar_graph_f32(selection, &initial_values, &opcodes, &left, &right)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }
    fn accelerator_arrays(&self) -> AcceleratorGraphArrays {
        let nodes = self.nodes.borrow();
        let mut initial_values = Vec::with_capacity(nodes.len());
        let mut opcodes = Vec::with_capacity(nodes.len());
        let mut left = Vec::with_capacity(nodes.len());
        let mut right = Vec::with_capacity(nodes.len());
        let mut parameter_ids = Vec::with_capacity(nodes.len());
        for node in nodes.iter() {
            initial_values.push(node.value as f32);
            let (opcode, lhs, rhs, parameter) = match node.op {
                TapeOp::Constant => (0, 0, 0, u32::MAX),
                TapeOp::Parameter(parameter) => (1, 0, 0, parameter as u32),
                TapeOp::Add(lhs, rhs) => (2, lhs, rhs, u32::MAX),
                TapeOp::Mul(lhs, rhs) => (3, lhs, rhs, u32::MAX),
                TapeOp::Div(lhs, rhs) => (4, lhs, rhs, u32::MAX),
                TapeOp::Tanh(input) => (5, input, 0, u32::MAX),
                TapeOp::Exp(input) => (6, input, 0, u32::MAX),
                TapeOp::Sqrt(input) => (7, input, 0, u32::MAX),
                TapeOp::Sin(input) => (8, input, 0, u32::MAX),
                TapeOp::Sigmoid(input) => (9, input, 0, u32::MAX),
                TapeOp::Max(lhs, rhs) => (10, lhs, rhs, u32::MAX),
                TapeOp::StopGradient(input) => (11, input, 0, u32::MAX),
            };
            opcodes.push(opcode);
            left.push(lhs as u32);
            right.push(rhs as u32);
            parameter_ids.push(parameter);
        }
        (initial_values, opcodes, left, right, parameter_ids)
    }
    #[allow(clippy::too_many_arguments)]
    fn accelerated_train_step(
        &self,
        selection: &BackendSelection,
        loss: usize,
        parameters: &mut [f64],
        first_moment: &mut [f64],
        second_moment: &mut [f64],
        step: u64,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<f64> {
        let (initial_values, opcodes, left, right, parameter_ids) = self.accelerator_arrays();
        let mut parameters_f32 = parameters
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let mut first_f32 = first_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let mut second_f32 = second_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let value = backend_scalar_graph_train_step_f32(
            selection,
            &initial_values,
            &opcodes,
            &left,
            &right,
            &parameter_ids,
            loss,
            &mut parameters_f32,
            &mut first_f32,
            &mut second_f32,
            step,
            learning_rate as f32,
            weight_decay as f32,
        )
        .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        for (output, value) in parameters.iter_mut().zip(parameters_f32) {
            *output = value as f64;
        }
        for (output, value) in first_moment.iter_mut().zip(first_f32) {
            *output = value as f64;
        }
        for (output, value) in second_moment.iter_mut().zip(second_f32) {
            *output = value as f64;
        }
        Ok(value as f64)
    }
    fn value(&self, value: usize) -> f64 {
        self.nodes.borrow()[value].value
    }
    fn backward(&self, loss: usize, parameter_count: usize) -> Vec<f64> {
        let nodes = self.nodes.borrow();
        let mut gradients = vec![0.0; nodes.len()];
        let mut parameter_gradients = vec![0.0; parameter_count];
        gradients[loss] = 1.0;
        for index in (0..nodes.len()).rev() {
            let gradient = gradients[index];
            match nodes[index].op {
                TapeOp::Constant => {}
                TapeOp::Parameter(parameter) => parameter_gradients[parameter] += gradient,
                TapeOp::Add(left, right) => {
                    gradients[left] += gradient;
                    gradients[right] += gradient;
                }
                TapeOp::Mul(left, right) => {
                    gradients[left] += gradient * nodes[right].value;
                    gradients[right] += gradient * nodes[left].value;
                }
                TapeOp::Div(left, right) => {
                    let denominator = nodes[right].value.max(1e-12);
                    gradients[left] += gradient / denominator;
                    gradients[right] -= gradient * nodes[left].value / denominator.powi(2);
                }
                TapeOp::Tanh(input) => {
                    gradients[input] += gradient * (1.0 - nodes[index].value.powi(2))
                }
                TapeOp::Exp(input) => gradients[input] += gradient * nodes[index].value,
                TapeOp::Sqrt(input) => {
                    gradients[input] += gradient / (2.0 * nodes[index].value.max(1e-12))
                }
                TapeOp::Sin(input) => gradients[input] += gradient * nodes[input].value.cos(),
                TapeOp::Sigmoid(input) => {
                    gradients[input] += gradient * nodes[index].value * (1.0 - nodes[index].value)
                }
                TapeOp::Max(left, right) => {
                    if nodes[left].value >= nodes[right].value {
                        gradients[left] += gradient;
                    } else {
                        gradients[right] += gradient;
                    }
                }
                TapeOp::StopGradient(_) => {}
            }
        }
        parameter_gradients
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphWaveNetConfig {
    pub lookback: usize,
    pub dilation_depth: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub ridge: f64,
    #[serde(default)]
    pub backend: ComputeBackendSelection,
}

impl Default for GraphWaveNetConfig {
    fn default() -> Self {
        Self {
            lookback: 8,
            dilation_depth: 3,
            hidden_size: 8,
            epochs: 120,
            learning_rate: 0.02,
            ridge: 1e-4,
            backend: select_compute_backend(None).expect("default CPU backend is always available"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphWaveNetForecaster {
    pub config: GraphWaveNetConfig,
    node_ids: Vec<String>,
    frequency: String,
    horizon: usize,
    adjacency: Option<CsrAdjacency>,
    weights: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    history: Vec<Vec<f64>>,
    target_mean: f64,
    target_scale: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DelayAwareGraphConfig {
    pub horizon: usize,
    pub edge_delay_prior: Vec<usize>,
    pub ridge: f64,
    pub backend: ComputeBackendSelection,
}

impl Default for DelayAwareGraphConfig {
    fn default() -> Self {
        Self {
            horizon: 1,
            edge_delay_prior: Vec::new(),
            ridge: 1.0e-6,
            backend: ComputeBackendSelection::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeDelaySensitivity {
    pub graph_signal_coefficient: f64,
    pub delay_counts: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DelayAwareGraphTransformer {
    pub config: DelayAwareGraphConfig,
    node_ids: Vec<String>,
    frequency: String,
    edges: Vec<(usize, usize)>,
    edge_weights: Vec<f64>,
    coefficients: Vec<f64>,
    history: Vec<Vec<f64>>,
    target_mean: f64,
    target_scale: f64,
}

impl DelayAwareGraphTransformer {
    pub fn new(config: DelayAwareGraphConfig) -> Result<Self> {
        if config.horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "horizon must be positive".to_string(),
            ));
        }
        if !config.ridge.is_finite() || config.ridge < 0.0 {
            return Err(GeoStError::InvalidFrame(
                "ridge must be finite and non-negative".to_string(),
            ));
        }
        if config.edge_delay_prior.contains(&0) {
            return Err(GeoStError::InvalidFrame(
                "edge_delay_prior values must be positive".to_string(),
            ));
        }
        Ok(Self {
            config,
            node_ids: Vec::new(),
            frequency: String::new(),
            edges: Vec::new(),
            edge_weights: Vec::new(),
            coefficients: Vec::new(),
            history: Vec::new(),
            target_mean: 0.0,
            target_scale: 1.0,
        })
    }

    pub fn fit(&mut self, frame: &GraphTemporalFrame) -> Result<()> {
        frame.validate()?;
        let nodes = frame.node_ids.len();
        let edges = csr_edges(&frame.adjacency, nodes);
        if edges.is_empty() {
            return Err(GeoStError::InvalidFrame(
                "delay-aware graph transformer requires at least one directed edge".to_string(),
            ));
        }
        let delays = self.resolved_delays(edges.len())?;
        let max_delay = delays.iter().copied().max().unwrap_or(1);
        if frame.target.len() <= max_delay + self.config.horizon {
            return Err(GeoStError::InvalidFrame(
                "target length must exceed maximum edge delay plus horizon".to_string(),
            ));
        }
        let (target_mean, target_scale) = target_center_scale(&frame.target);
        let normalized_target = frame
            .target
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| (value - target_mean) / target_scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut xtx = vec![vec![0.0; 3]; 3];
        let mut xty = vec![0.0; 3];
        for time_idx in max_delay - 1..normalized_target.len() - 1 {
            let signal = delayed_graph_signal_backend(
                &normalized_target,
                &edges,
                &frame.adjacency.data,
                &delays,
                time_idx,
                &self.config.backend,
            )?;
            for (node, signal_value) in signal.iter().enumerate().take(nodes) {
                let x = [1.0, normalized_target[time_idx][node], *signal_value];
                let actual = normalized_target[time_idx + 1][node];
                for row in 0..3 {
                    xty[row] += x[row] * actual;
                    for col in 0..3 {
                        xtx[row][col] += x[row] * x[col];
                    }
                }
            }
        }
        for (idx, row) in xtx.iter_mut().enumerate() {
            if idx > 0 {
                row[idx] += self.config.ridge.max(1.0e-12);
            }
        }
        self.coefficients = solve_linear_system(xtx, xty);
        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.edges = edges;
        self.edge_weights = frame.adjacency.data.clone();
        self.history = normalized_target;
        self.target_mean = target_mean;
        self.target_scale = target_scale;
        Ok(())
    }

    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "prediction horizon must be positive".to_string(),
            ));
        }
        if self.coefficients.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let delays = self.resolved_delays(self.edges.len())?;
        let mut history = self.history.clone();
        let mut predictions = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let time_idx = history.len() - 1;
            let signal = delayed_graph_signal_backend(
                &history,
                &self.edges,
                &self.edge_weights,
                &delays,
                time_idx,
                &self.config.backend,
            )?;
            let features = history[time_idx]
                .iter()
                .zip(&signal)
                .map(|(&current, &graph)| vec![current, graph])
                .collect::<Vec<_>>();
            let execution_backend = profitable_affine_backend(
                BackendSelection {
                    requested: self.config.backend.requested.clone(),
                    selected: self.config.backend.selected.clone(),
                    available: self.config.backend.available.clone(),
                },
                features.len(),
                2,
            )?;
            let next = backend_affine_scores(
                &execution_backend,
                &features,
                &[0.0, 0.0],
                &self.coefficients[1..3],
                &vec![self.coefficients[0]; features.len()],
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            history.push(next.clone());
            predictions.push(
                next.into_iter()
                    .map(|value| quantize_prediction(value * self.target_scale + self.target_mean))
                    .collect(),
            );
        }
        Ok(predictions)
    }

    pub fn score(&self, actual: &[Vec<f64>]) -> Result<f64> {
        if actual.is_empty() {
            return Err(GeoStError::InvalidFrame(
                "actual must contain at least one row".to_string(),
            ));
        }
        let predictions = self.predict(actual.len())?;
        let mut sum = 0.0;
        let mut count = 0.0;
        for (pred_row, actual_row) in predictions.iter().zip(actual) {
            if pred_row.len() != actual_row.len() {
                return Err(GeoStError::InvalidFrame(
                    "prediction and actual row widths must match".to_string(),
                ));
            }
            for (&pred, &actual) in pred_row.iter().zip(actual_row) {
                let error = actual - pred;
                sum += error * error;
                count += 1.0;
            }
        }
        Ok((sum / count).sqrt())
    }

    pub fn edge_delay_sensitivity(&self) -> EdgeDelaySensitivity {
        let mut counts = std::collections::BTreeMap::<usize, usize>::new();
        if let Ok(delays) = self.resolved_delays(self.edges.len()) {
            for delay in delays {
                *counts.entry(delay).or_insert(0) += 1;
            }
        }
        EdgeDelaySensitivity {
            graph_signal_coefficient: self.coefficients.get(2).copied().unwrap_or(0.0),
            delay_counts: counts.into_iter().collect(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json_string(value: &str) -> Result<Self> {
        Ok(serde_json::from_str(value)?)
    }

    pub fn backend(&self) -> String {
        self.config.backend.selected.clone()
    }

    fn resolved_delays(&self, edge_count: usize) -> Result<Vec<usize>> {
        if self.config.edge_delay_prior.is_empty() {
            return Ok(vec![1; edge_count]);
        }
        if self.config.edge_delay_prior.len() != edge_count {
            return Err(GeoStError::InvalidFrame(
                "edge_delay_prior must match edge count".to_string(),
            ));
        }
        Ok(self.config.edge_delay_prior.clone())
    }
}

impl GraphWaveNetForecaster {
    pub fn new(config: GraphWaveNetConfig) -> Result<Self> {
        if config.lookback == 0
            || config.dilation_depth == 0
            || config.hidden_size == 0
            || config.epochs == 0
        {
            return Err(GeoStError::InvalidFrame(
                "lookback, dilation_depth, hidden_size, and epochs must be positive".to_string(),
            ));
        }
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err(GeoStError::InvalidFrame(
                "learning_rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            config,
            node_ids: Vec::new(),
            frequency: String::new(),
            horizon: 0,
            adjacency: None,
            weights: Vec::new(),
            intercepts: Vec::new(),
            history: Vec::new(),
            target_mean: 0.0,
            target_scale: 1.0,
        })
    }

    pub fn fit(&mut self, frame: &GraphTemporalFrame) -> Result<()> {
        frame.validate()?;
        if frame.target.len() <= self.config.lookback + frame.horizon {
            return Err(GeoStError::InvalidFrame(
                "target length must exceed lookback plus horizon".to_string(),
            ));
        }
        let nodes = frame.node_ids.len();
        let feature_len = self.feature_len();
        let (target_mean, target_scale) = target_center_scale(&frame.target);
        let normalized_target = frame
            .target
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| (value - target_mean) / target_scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let adjacency = frame.adjacency.row_normalized();
        let samples = frame.target.len() - self.config.lookback - frame.horizon + 1;
        self.weights = (0..frame.horizon)
            .into_par_iter()
            .map(|h| -> Result<Vec<f64>> {
                let mut xtx = vec![vec![0.0; feature_len]; feature_len];
                let mut xty = vec![0.0; feature_len];
                for sample in 0..samples {
                    let cutoff = sample + self.config.lookback;
                    let features =
                        self.wave_features(&normalized_target[sample..cutoff], &adjacency)?;
                    let actual = &normalized_target[cutoff + h];
                    for node in 0..nodes {
                        let x = &features[node * feature_len..(node + 1) * feature_len];
                        for row in 0..feature_len {
                            xty[row] += x[row] * actual[node];
                            for col in 0..feature_len {
                                xtx[row][col] += x[row] * x[col];
                            }
                        }
                    }
                }
                for (idx, row) in xtx.iter_mut().enumerate() {
                    row[idx] += self.config.ridge.max(1.0e-8);
                }
                Ok(solve_linear_system(xtx, xty))
            })
            .collect::<Result<Vec<_>>>()?;
        self.intercepts = vec![0.0; frame.horizon];
        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.horizon = frame.horizon;
        self.adjacency = Some(adjacency);
        self.history = normalized_target;
        self.target_mean = target_mean;
        self.target_scale = target_scale;
        Ok(())
    }

    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "prediction horizon must be positive".to_string(),
            ));
        }
        if self.weights.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let adjacency = self.adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        let mut history = self.history.clone();
        let mut predictions = Vec::with_capacity(horizon);
        for step in 0..horizon {
            let h = step.min(self.weights.len() - 1);
            let start = history.len() - self.config.lookback;
            let features = self.wave_features(&history[start..], adjacency)?;
            let feature_len = self.weights[h].len();
            let rows = features
                .chunks(feature_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let means = vec![0.0; feature_len];
            let intercepts = vec![self.intercepts[h]; self.node_ids.len()];
            let execution_backend = profitable_affine_backend(
                self.neural_backend_selection(),
                rows.len(),
                feature_len,
            )?;
            let next = backend_affine_scores(
                &execution_backend,
                &rows,
                &means,
                &self.weights[h],
                &intercepts,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            history.push(next.clone());
            predictions.push(
                next.into_iter()
                    .map(|value| value * self.target_scale + self.target_mean)
                    .collect(),
            );
        }
        Ok(predictions)
    }

    pub fn score(&self, actual: &[Vec<f64>]) -> Result<f64> {
        let predictions = self.predict(actual.len())?;
        let mut sum = 0.0;
        let mut count = 0usize;
        for (pred_row, actual_row) in predictions.iter().zip(actual) {
            if pred_row.len() != actual_row.len() {
                return Err(GeoStError::InvalidFrame(
                    "prediction and actual rows must have the same width".to_string(),
                ));
            }
            for (pred, actual) in pred_row.iter().zip(actual_row) {
                sum += (pred - actual).powi(2);
                count += 1;
            }
        }
        Ok((sum / count.max(1) as f64).sqrt())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json_string(value: &str) -> Result<Self> {
        Ok(serde_json::from_str(value)?)
    }

    pub fn backend(&self) -> String {
        self.config.backend.selected.clone()
    }

    fn feature_len(&self) -> usize {
        3 + self.config.dilation_depth * 4
    }

    fn wave_features(&self, window: &[Vec<f64>], adjacency: &CsrAdjacency) -> Result<Vec<f64>> {
        let nodes = window[0].len();
        let feature_len = self.feature_len();
        let last = window.last().expect("non-empty lookback window");
        let mut source_vectors = Vec::with_capacity(self.config.dilation_depth + 1);
        source_vectors.push(last.as_slice());
        for depth in 0..self.config.dilation_depth {
            let lag = 2usize.pow(depth as u32).min(window.len());
            source_vectors.push(window[window.len() - lag].as_slice());
        }
        let channels = source_vectors.len();
        let backend = self.neural_backend_selection();
        let accelerated = backend.selected != "cpu"
            && nodes.saturating_mul(channels) >= GEO_ST_CSR_DISPATCH_MIN_OPS;
        let neighbor_vectors = if accelerated {
            let values = (0..nodes)
                .flat_map(|node| source_vectors.iter().map(move |values| values[node] as f32))
                .collect::<Vec<_>>();
            let indptr = adjacency
                .indptr
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>();
            let indices = adjacency
                .indices
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>();
            let weights = adjacency
                .data
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let output =
                backend_csr_diffusion_f32(&backend, &indptr, &indices, &weights, channels, &values)
                    .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            (0..channels)
                .map(|channel| {
                    (0..nodes)
                        .map(|node| f64::from(output[node * channels + channel]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            source_vectors
                .iter()
                .map(|values| {
                    let mut neighbor = vec![0.0; nodes];
                    adjacency.matvec(values, &mut neighbor);
                    neighbor
                })
                .collect::<Vec<_>>()
        };
        let neighbor_last = &neighbor_vectors[0];
        let mut out = vec![0.0; nodes * feature_len];
        for node in 0..nodes {
            let offset = node * feature_len;
            out[offset] = last[node];
            out[offset + 1] = neighbor_last[node];
            out[offset + 2] = 1.0;
            for depth in 0..self.config.dilation_depth {
                let lag = 2usize.pow(depth as u32).min(window.len());
                let current = &window[window.len() - lag];
                let previous = &window[window.len().saturating_sub(lag + 1)];
                let neighbor = &neighbor_vectors[depth + 1];
                let gated = (current[node] + neighbor[node]).tanh();
                let filter = sigmoid(current[node] - previous[node]);
                let base = offset + 3 + depth * 4;
                out[base] = gated * filter;
                out[base + 1] = current[node];
                out[base + 2] = neighbor[node];
                out[base + 3] = current[node] - previous[node];
            }
        }
        Ok(out)
    }

    fn neural_backend_selection(&self) -> BackendSelection {
        BackendSelection {
            requested: self.config.backend.requested.clone(),
            selected: self.config.backend.selected.clone(),
            available: self.config.backend.available.clone(),
        }
    }
}

impl Default for STAEformerForecaster {
    fn default() -> Self {
        Self::new(STAEformerConfig::default()).expect("default STAEformer config is valid")
    }
}

impl STAEformerForecaster {
    pub fn new(config: STAEformerConfig) -> Result<Self> {
        if config.lookback == 0
            || config.attention_heads == 0
            || config.hidden_size == 0
            || config.epochs == 0
        {
            return Err(GeoStError::InvalidFrame(
                "lookback, attention_heads, hidden_size, and epochs must be positive".to_string(),
            ));
        }
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err(GeoStError::InvalidFrame(
                "learning_rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            config,
            node_ids: Vec::new(),
            frequency: String::new(),
            horizon: 0,
            adjacency: None,
            weights: Vec::new(),
            intercepts: Vec::new(),
            temporal_queries: Vec::new(),
            temporal_keys: Vec::new(),
            spatial_weights: Vec::new(),
            history: Vec::new(),
            target_mean: 0.0,
            target_scale: 1.0,
        })
    }

    pub fn fit(&mut self, frame: &GraphTemporalFrame) -> Result<()> {
        frame.validate()?;
        if frame.target.len() <= self.config.lookback + frame.horizon {
            return Err(GeoStError::InvalidFrame(
                "target length must exceed lookback plus horizon".to_string(),
            ));
        }
        let nodes = frame.node_ids.len();
        let feature_len = self.feature_len();
        let (target_mean, target_scale) = target_center_scale(&frame.target);
        let normalized_target = frame
            .target
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| (value - target_mean) / target_scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let adjacency = frame.adjacency.row_normalized();
        self.temporal_queries = deterministic_weight_matrix(
            self.config.attention_heads,
            self.config.lookback,
            0x94d0_49bb_1331_11eb,
        );
        self.temporal_keys = deterministic_weight_matrix(
            self.config.attention_heads,
            self.config.lookback,
            0x2545_f491_4f6c_dd1d,
        );
        self.spatial_weights = (0..self.config.attention_heads)
            .map(|idx| 0.5 + idx as f64 / self.config.attention_heads as f64)
            .collect();
        let samples = frame.target.len() - self.config.lookback - frame.horizon + 1;
        self.weights = (0..frame.horizon)
            .into_par_iter()
            .map(|h| -> Result<Vec<f64>> {
                let mut xtx = vec![vec![0.0; feature_len]; feature_len];
                let mut xty = vec![0.0; feature_len];
                for sample in 0..samples {
                    let cutoff = sample + self.config.lookback;
                    let features =
                        self.attention_features(&normalized_target[sample..cutoff], &adjacency)?;
                    let actual = &normalized_target[cutoff + h];
                    for node in 0..nodes {
                        let x = &features[node * feature_len..(node + 1) * feature_len];
                        for row in 0..feature_len {
                            xty[row] += x[row] * actual[node];
                            for col in 0..feature_len {
                                xtx[row][col] += x[row] * x[col];
                            }
                        }
                    }
                }
                for (idx, row) in xtx.iter_mut().enumerate() {
                    row[idx] += self.config.ridge.max(1.0e-8);
                }
                Ok(solve_linear_system(xtx, xty))
            })
            .collect::<Result<Vec<_>>>()?;
        self.intercepts = vec![0.0; frame.horizon];
        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.horizon = frame.horizon;
        self.adjacency = Some(adjacency);
        self.history = normalized_target;
        self.target_mean = target_mean;
        self.target_scale = target_scale;
        Ok(())
    }

    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "prediction horizon must be positive".to_string(),
            ));
        }
        if self.weights.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let adjacency = self.adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        let mut history = self.history.clone();
        let mut predictions = Vec::with_capacity(horizon);
        for step in 0..horizon {
            let h = step.min(self.weights.len() - 1);
            let start = history.len() - self.config.lookback;
            let features = self.attention_features(&history[start..], adjacency)?;
            let feature_len = self.weights[h].len();
            let rows = features
                .chunks(feature_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let means = vec![0.0; feature_len];
            let intercepts = vec![self.intercepts[h]; self.node_ids.len()];
            let execution_backend = profitable_affine_backend(
                self.neural_backend_selection(),
                rows.len(),
                feature_len,
            )?;
            let next = backend_affine_scores(
                &execution_backend,
                &rows,
                &means,
                &self.weights[h],
                &intercepts,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            history.push(next.clone());
            predictions.push(
                next.into_iter()
                    .map(|value| value * self.target_scale + self.target_mean)
                    .collect(),
            );
        }
        Ok(predictions)
    }

    pub fn score(&self, actual: &[Vec<f64>]) -> Result<f64> {
        if actual.is_empty() {
            return Err(GeoStError::InvalidFrame(
                "actual horizon cannot be empty".to_string(),
            ));
        }
        let predictions = self.predict(actual.len())?;
        let mut sum = 0.0;
        let mut count = 0usize;
        for (pred_row, actual_row) in predictions.iter().zip(actual) {
            if pred_row.len() != actual_row.len() {
                return Err(GeoStError::InvalidFrame(
                    "prediction and actual rows must have the same width".to_string(),
                ));
            }
            for (pred, actual) in pred_row.iter().zip(actual_row) {
                sum += (pred - actual).powi(2);
                count += 1;
            }
        }
        Ok((sum / count.max(1) as f64).sqrt())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json_string(value: &str) -> Result<Self> {
        Ok(serde_json::from_str(value)?)
    }

    pub fn backend(&self) -> String {
        self.config.backend.selected.clone()
    }

    fn feature_len(&self) -> usize {
        3 + self.config.attention_heads * 2
    }

    fn attention_features(
        &self,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
    ) -> Result<Vec<f64>> {
        let nodes = window[0].len();
        let feature_len = self.feature_len();
        let mut out = vec![0.0; nodes * feature_len];
        let last = window.last().expect("non-empty lookback window");
        let channels = window.len();
        let backend = self.neural_backend_selection();
        let accelerated = backend.selected != "cpu"
            && nodes.saturating_mul(channels) >= GEO_ST_CSR_DISPATCH_MIN_OPS;
        let neighbor_window = if accelerated {
            let values = (0..nodes)
                .flat_map(|node| window.iter().map(move |row| row[node] as f32))
                .collect::<Vec<_>>();
            let indptr = adjacency
                .indptr
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>();
            let indices = adjacency
                .indices
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>();
            let weights = adjacency
                .data
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let smoothed =
                backend_csr_diffusion_f32(&backend, &indptr, &indices, &weights, channels, &values)
                    .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            (0..channels)
                .map(|channel| {
                    (0..nodes)
                        .map(|node| f64::from(smoothed[node * channels + channel]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            window
                .iter()
                .map(|row| {
                    let mut smoothed = vec![0.0; nodes];
                    adjacency.matvec(row, &mut smoothed);
                    smoothed
                })
                .collect::<Vec<_>>()
        };
        let neighbor_last = neighbor_window
            .last()
            .expect("non-empty neighbor lookback window");
        for node in 0..nodes {
            let offset = node * feature_len;
            out[offset] = last[node];
            out[offset + 1] = neighbor_last[node];
            out[offset + 2] = 1.0;
            let series = window.iter().map(|row| row[node]).collect::<Vec<_>>();
            let neighbor_series = neighbor_window
                .iter()
                .map(|row| row[node])
                .collect::<Vec<_>>();
            for head in 0..self.config.attention_heads {
                let temporal = attention_pool(
                    &series,
                    &self.temporal_queries[head],
                    &self.temporal_keys[head],
                );
                let spatial = attention_pool(
                    &neighbor_series,
                    &self.temporal_queries[head],
                    &self.temporal_keys[head],
                ) * self.spatial_weights[head];
                out[offset + 3 + head * 2] = temporal;
                out[offset + 3 + head * 2 + 1] = spatial;
            }
        }
        Ok(out)
    }

    fn neural_backend_selection(&self) -> BackendSelection {
        BackendSelection {
            requested: self.config.backend.requested.clone(),
            selected: self.config.backend.selected.clone(),
            available: self.config.backend.available.clone(),
        }
    }
}

impl PaperGraphTransformerForecaster {
    pub fn new(config: PaperGraphTransformerConfig) -> Result<Self> {
        if config.lookback < 2
            || config.hidden_size == 0
            || config.attention_heads == 0
            || config.attention_heads > config.hidden_size
            || !config.hidden_size.is_multiple_of(config.attention_heads)
            || config.graph_order == 0
            || config.experts == 0
            || config.periodicity == 0
            || config.recent_window == 0
            || (config.profile == GraphTransformerProfile::LongShortFusion
                && config.recent_window > config.lookback)
            || config.epochs == 0
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.weight_decay.is_finite()
            || config.weight_decay < 0.0
            || config.batch_size == 0
            || config.batch_size > 32
        {
            return Err(GeoStError::InvalidFrame(
                "invalid paper graph transformer configuration".to_string(),
            ));
        }
        Ok(Self {
            config,
            node_ids: Vec::new(),
            frequency: String::new(),
            horizon: 0,
            adjacency: None,
            trainable_state: None,
            history: Vec::new(),
            history_time_features: None,
            covariate_roles: None,
            owner_mask: Vec::new(),
            fit_activation_tape_bytes: 0,
            fit_frozen_patch_cache_bytes: 0,
            target_mean: 0.0,
            target_scale: 1.0,
        })
    }

    pub fn fit(&mut self, frame: &GraphTemporalFrame) -> Result<()> {
        self.fit_internal(
            frame,
            None,
            None,
            None,
            LsttnTrainingPhase::Both,
            None,
            None,
        )
    }

    pub fn fit_checkpointed(
        &mut self,
        frame: &GraphTemporalFrame,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.fit_internal(
            frame,
            Some(checkpoint_path.as_ref()),
            None,
            None,
            LsttnTrainingPhase::Both,
            None,
            None,
        )
    }

    /// Convert a compatible shard artifact into a fresh-cutoff warm start.
    /// Parameters remain, while every cutoff-bound optimizer/normalization
    /// field is reset. Exact resume continues to use the checkpoint path.
    pub fn prepare_shard_warm_start(&mut self, identity_json: &str) -> Result<()> {
        let identity: LsttnShardIdentity =
            serde_json::from_str(identity_json).map_err(|error| {
                GeoStError::InvalidFrame(format!("invalid LSTTN shard identity: {error}"))
            })?;
        let policy = stable_fingerprint(&serde_json::json!({
            "equipment_category": identity.equipment_category,
            "target_unit": identity.target_unit,
            "node_registry": identity.node_registry_fingerprint,
            "shard_policy": identity.shard_policy_fingerprint,
        }))?;
        let state = self.trainable_state.as_mut().ok_or(GeoStError::NotFit)?;
        if state.warm_start_policy_fingerprint == 0 || state.warm_start_policy_fingerprint != policy
        {
            return Err(GeoStError::InvalidFrame(
                "LSTTN warm start rejects an objective or node-policy mismatch".to_string(),
            ));
        }
        state.first_moment.fill(0.0);
        state.second_moment.fill(0.0);
        state.steps = 0;
        state.shared_steps = 0;
        state.local_steps = 0;
        state.shard_data_fingerprint = 0;
        state.target_scale = 1.0;
        state.normalized_zero = 0.0;
        Ok(())
    }

    /// Fit against a frozen shared snapshot and return an objective-local
    /// proposal. This method never writes shared state, so shard workers can
    /// fan out safely; only the deterministic reducer may publish a new
    /// snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_shard_round(
        &mut self,
        frame: &GraphTemporalFrame,
        shared_state_path: impl AsRef<Path>,
        checkpoint_path: impl AsRef<Path>,
        phase: &str,
        normalization: Option<(f64, f64)>,
        identity_json: &str,
        objective_weight: f64,
    ) -> Result<String> {
        if !objective_weight.is_finite() || objective_weight <= 0.0 {
            return Err(GeoStError::InvalidFrame(
                "LSTTN shard round objective weight must be finite and positive".to_string(),
            ));
        }
        let phase = LsttnTrainingPhase::parse(phase)?;
        let identity: LsttnShardIdentity =
            serde_json::from_str(identity_json).map_err(|error| {
                GeoStError::InvalidFrame(format!("invalid LSTTN shard identity: {error}"))
            })?;
        let shared_path = shared_state_path.as_ref();
        let shared_bytes = shared_path
            .is_file()
            .then(|| fs::read(shared_path))
            .transpose()?;
        let base_shared_hash = shared_bytes
            .as_ref()
            .map_or(0, |bytes| bytes_fingerprint(bytes));
        let shared = shared_bytes
            .as_ref()
            .map(|bytes| serde_json::from_slice(bytes))
            .transpose()?;
        self.fit_internal(
            frame,
            Some(checkpoint_path.as_ref()),
            self.trainable_state.clone(),
            shared,
            phase,
            normalization,
            Some(identity.clone()),
        )?;
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        serde_json::to_string(&LsttnSharedRound {
            version: 1,
            shard_id: identity.node_registry_fingerprint,
            objective_weight,
            base_shared_hash,
            shared: state.export_shared(self.target_mean, self.target_scale),
        })
        .map_err(Into::into)
    }

    /// Reduce independently emitted rounds in stable shard order. A round can
    /// only be reduced once against the exact frozen shared snapshot it used.
    pub fn reduce_shard_rounds(rounds_json: &[String], expected_base_hash: u64) -> Result<String> {
        if rounds_json.is_empty() {
            return Err(GeoStError::InvalidFrame(
                "LSTTN shared round reduction requires at least one round".to_string(),
            ));
        }
        let mut rounds = rounds_json
            .iter()
            .map(|json| serde_json::from_str::<LsttnSharedRound>(json))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if rounds.iter().any(|round| {
            round.version != 1
                || round.base_shared_hash != expected_base_hash
                || !round.objective_weight.is_finite()
                || round.objective_weight <= 0.0
        }) {
            return Err(GeoStError::InvalidFrame(
                "LSTTN shared rounds do not share a valid frozen base".to_string(),
            ));
        }
        rounds.sort_by(|left, right| left.shard_id.cmp(&right.shard_id));
        if rounds
            .windows(2)
            .any(|pair| pair[0].shard_id == pair[1].shard_id)
        {
            return Err(GeoStError::InvalidFrame(
                "LSTTN shared round contains duplicate shard identities".to_string(),
            ));
        }
        let mut reduced = rounds[0].shared.clone();
        for round in &rounds[1..] {
            if round.shared.version != reduced.version
                || round.shared.architecture_fingerprint != reduced.architecture_fingerprint
                || round.shared.backbone_fingerprint != reduced.backbone_fingerprint
                || round.shared.segments.len() != reduced.segments.len()
            {
                return Err(GeoStError::InvalidFrame(
                    "LSTTN shared rounds are architecturally incompatible".to_string(),
                ));
            }
        }
        let total = rounds
            .iter()
            .map(|round| round.objective_weight)
            .sum::<f64>();
        for segment_index in 0..reduced.segments.len() {
            let output = &mut reduced.segments[segment_index];
            for index in 0..output.parameters.len() {
                output.parameters[index] = rounds
                    .iter()
                    .map(|round| {
                        round.shared.segments[segment_index].parameters[index]
                            * round.objective_weight
                            / total
                    })
                    .sum();
            }
            for index in 0..output.first_moment.len() {
                output.first_moment[index] = rounds
                    .iter()
                    .map(|round| {
                        round.shared.segments[segment_index].first_moment[index]
                            * round.objective_weight
                            / total
                    })
                    .sum();
            }
            for index in 0..output.second_moment.len() {
                output.second_moment[index] = rounds
                    .iter()
                    .map(|round| {
                        round.shared.segments[segment_index].second_moment[index]
                            * round.objective_weight
                            / total
                    })
                    .sum();
            }
        }
        serde_json::to_string(&reduced).map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_internal(
        &mut self,
        frame: &GraphTemporalFrame,
        checkpoint_path: Option<&Path>,
        seed_state: Option<TrainableGraphTransformerState>,
        shared_state: Option<LsttnSharedState>,
        training_phase: LsttnTrainingPhase,
        normalization: Option<(f64, f64)>,
        shard_identity: Option<LsttnShardIdentity>,
    ) -> Result<()> {
        frame.validate()?;
        if self.config.profile == GraphTransformerProfile::LongShortFusion {
            let patch_width = (self.config.periodicity / 24).max(1);
            let patch_count = self.config.lookback / patch_width;
            let seasonal_lag = if frame.frequency.eq_ignore_ascii_case("weekly") {
                // Weekly DAT has no seven-times-faster daily samples. Its two
                // periodic paths are one weekly cadence and the configured
                // seasonal cadence (for example 13 weeks), not a fictitious
                // 91-week `13 * 7` requirement.
                (self.config.periodicity / patch_width).max(1)
            } else {
                (self.config.periodicity / patch_width).max(1) * 7
            };
            if !self.config.lookback.is_multiple_of(patch_width) || patch_count <= seasonal_lag {
                return Err(GeoStError::InvalidFrame(format!(
                    "LSTTN lookback must contain complete patches and exceed its seasonal lag: lookback={}, patch_width={}, seasonal_lag_patches={seasonal_lag}",
                    self.config.lookback, patch_width
                )));
            }
        }
        if frame.target.len() <= self.config.lookback + frame.horizon {
            return Err(GeoStError::InvalidFrame(
                "target length must exceed lookback plus horizon".to_string(),
            ));
        }
        let (mean, scale) = if let Some(shared) = &shared_state {
            (shared.target_mean, shared.target_scale)
        } else if let Some((mean, scale)) = normalization {
            if !mean.is_finite() || !scale.is_finite() || scale <= 0.0 {
                return Err(GeoStError::InvalidFrame(
                    "global LSTTN normalization must be finite with positive scale".to_string(),
                ));
            }
            (mean, scale)
        } else {
            target_center_scale_observed(
                &frame.target,
                frame.target_mask.as_deref(),
                frame.imputed_mask.as_deref(),
            )
        };
        let normalized = frame
            .target
            .iter()
            .enumerate()
            .map(|(time, row)| {
                row.iter()
                    .enumerate()
                    .map(|(node, value)| {
                        let observed = frame
                            .target_mask
                            .as_ref()
                            .is_none_or(|mask| mask[time][node]);
                        let imputed = frame
                            .imputed_mask
                            .as_ref()
                            .is_some_and(|mask| mask[time][node]);
                        if observed && !imputed {
                            (value - mean) / scale
                        } else {
                            -mean / scale
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut training_normalized = normalized.clone();
        if let Some(owner_mask) = &frame.owner_mask {
            let normalized_zero = -mean / scale;
            for row in &mut training_normalized {
                for (node, is_owner) in owner_mask.iter().enumerate() {
                    if !is_owner {
                        row[node] = normalized_zero;
                    }
                }
            }
        }
        let lsttn_time_features = if let Some(covariates) = &frame.covariates {
            let role_index = frame
                .covariate_roles
                .as_ref()
                .and_then(|roles| roles.iter().position(|role| role == "time_of_day"))
                .ok_or_else(|| GeoStError::InvalidFrame(
                    "LSTTN covariates require a declared 'time_of_day' role; positional channel 0 is no longer supported".to_string(),
                ))?;
            Some(
                covariates
                    .iter()
                    .map(|time| time.iter().map(|node| node[role_index]).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        self.covariate_roles = frame.covariate_roles.clone();
        let adjacency = frame.adjacency.row_normalized();
        if !matches!(training_phase, LsttnTrainingPhase::Both) && shard_identity.is_none() {
            return Err(GeoStError::InvalidFrame(
                "shared LSTTN shard training requires an explicit backbone/shard identity"
                    .to_string(),
            ));
        }
        let backend = if training_phase.freezes_shared() {
            BackendSelection {
                requested: "cpu".to_string(),
                selected: "cpu".to_string(),
                available: self.config.backend.available.clone(),
            }
        } else {
            BackendSelection {
                requested: self.config.backend.requested.clone(),
                selected: self.config.backend.selected.clone(),
                available: self.config.backend.available.clone(),
            }
        };
        let frame_fingerprint = graph_temporal_training_fingerprint(frame);
        let architecture_fingerprint = stable_fingerprint(&serde_json::json!({
            "profile": self.config.profile,
            "lookback": self.config.lookback,
            "hidden_size": self.config.hidden_size,
            "attention_heads": self.config.attention_heads,
            "graph_order": self.config.graph_order,
            "experts": self.config.experts,
            "periodicity": self.config.periodicity,
            "recent_window": self.config.recent_window,
            "horizon": frame.horizon,
            "frequency": frame.frequency,
        }))?;
        let backbone_fingerprint = shard_identity
            .as_ref()
            .map(|identity| {
                stable_fingerprint(&serde_json::json!({
                    "backbone_id": identity.backbone_id,
                    "equipment_category": identity.equipment_category,
                    "target_unit": identity.target_unit,
                }))
            })
            .transpose()?
            .unwrap_or(0);
        let shard_data_fingerprint = shard_identity
            .as_ref()
            .map(|identity| {
                stable_fingerprint(&serde_json::json!({
                    "frame": frame_fingerprint,
                    "node_registry": identity.node_registry_fingerprint,
                    "graph_cutoff": identity.graph_cutoff,
                    "shard_policy": identity.shard_policy_fingerprint,
                }))
            })
            .transpose()?
            .unwrap_or(frame_fingerprint);
        let shared_fingerprint = shared_state.as_ref().map(|shared| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            serde_json::to_vec(shared)
                .expect("serializable shared LSTTN state")
                .hash(&mut hasher);
            hasher.finish()
        });
        let epoch_fingerprint = stable_fingerprint(&serde_json::json!({
            "phase": training_phase,
            "epoch": shard_identity.as_ref().map(|identity| identity.epoch),
            "epochs_per_pass": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "batch_size": self.config.batch_size,
            "normalization": [mean, scale],
            "shared_fingerprint": shared_fingerprint,
        }))?;
        let config_json = serde_json::to_string(&serde_json::json!({
            "config": &self.config,
            "identity": &shard_identity,
            "architecture_fingerprint": architecture_fingerprint,
            "backbone_fingerprint": backbone_fingerprint,
            "shard_data_fingerprint": shard_data_fingerprint,
            "epoch_fingerprint": epoch_fingerprint,
            "owner_mask": &frame.owner_mask,
            "normalization": [mean, scale],
            "shared_fingerprint": shared_fingerprint,
        }))?;
        let resumed = checkpoint_path
            .filter(|path| path.is_file())
            .map(|path| -> Result<LsttnTrainingCheckpoint> {
                let checkpoint: LsttnTrainingCheckpoint =
                    serde_json::from_str(&fs::read_to_string(path)?)?;
                if checkpoint.version != 3
                    || checkpoint.architecture_fingerprint != architecture_fingerprint
                    || checkpoint.backbone_fingerprint != backbone_fingerprint
                    || checkpoint.shard_data_fingerprint != shard_data_fingerprint
                    || checkpoint.epoch_fingerprint != epoch_fingerprint
                {
                    return Err(GeoStError::InvalidFrame(format!(
                        "LSTTN checkpoint {} has an incompatible architecture, backbone, shard data, or epoch identity",
                        path.display()
                    )));
                }
                Ok(checkpoint)
            })
            .transpose()?;
        let mut state = if let Some(checkpoint) = &resumed {
            checkpoint.state.clone()
        } else if let Some(seed) = seed_state {
            seed
        } else {
            TrainableGraphTransformerState::initialized(
                frame.node_ids.len(),
                self.config.hidden_size,
                self.config.attention_heads,
                self.config.periodicity,
                self.config.recent_window,
                self.config.lookback,
                frame.horizon,
                self.config.experts,
                self.config.graph_order,
                0x5354_474d_4f45,
            )
        };
        if state.nodes != frame.node_ids.len()
            || state.hidden != self.config.hidden_size
            || state.horizons != frame.horizon
            || (state.architecture_fingerprint != 0
                && state.architecture_fingerprint != architecture_fingerprint)
            || (state.backbone_fingerprint != 0
                && state.backbone_fingerprint != backbone_fingerprint)
            || (state.shard_data_fingerprint != 0
                && state.shard_data_fingerprint != shard_data_fingerprint)
        {
            return Err(GeoStError::InvalidFrame(
                "local LSTTN state is incompatible with the architecture, backbone, node registry, cutoff, or shard policy"
                    .to_string(),
            ));
        }
        state.architecture_fingerprint = architecture_fingerprint;
        state.backbone_fingerprint = backbone_fingerprint;
        state.shard_data_fingerprint = shard_data_fingerprint;
        state.warm_start_policy_fingerprint = shard_identity
            .as_ref()
            .map(|identity| {
                stable_fingerprint(&serde_json::json!({
                    "equipment_category": identity.equipment_category,
                    "target_unit": identity.target_unit,
                    "node_registry": identity.node_registry_fingerprint,
                    "shard_policy": identity.shard_policy_fingerprint,
                }))
                .expect("serializable LSTTN warm-start policy")
            })
            .unwrap_or(0);
        if resumed.is_none() {
            if let Some(shared) = &shared_state {
                state.import_shared(shared)?;
            }
        }
        state.shard_training = !matches!(training_phase, LsttnTrainingPhase::Both);
        state.freeze_shared = training_phase.freezes_shared();
        state.target_scale = scale;
        state.normalized_zero = -mean / scale;
        state.periodic_short_lag = if frame.frequency.eq_ignore_ascii_case("weekly") {
            1
        } else {
            0
        };
        let mut pretraining_completed = resumed
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.pretraining_completed);
        let mut supervised_batches_completed = resumed
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.supervised_batches_completed);
        let sample_count = normalized.len() - self.config.lookback - frame.horizon + 1;
        let spatial_shift_environments =
            if self.config.profile == GraphTransformerProfile::SpatialShiftGraphonMoE {
                Some(maximum_spatiotemporal_graph_division(
                    &normalized,
                    self.config.periodicity,
                    self.config.experts,
                )?)
            } else {
                None
            };
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let mut cuda_executor = if self.config.profile == GraphTransformerProfile::LongShortFusion
            && backend.selected == "cuda"
        {
            Some(CudaLsttnTensorExecutor::new(&state, &adjacency)?)
        } else {
            None
        };
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let uses_cuda_tensor_executor = cuda_executor.is_some();
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let uses_cuda_tensor_executor = false;
        if self.config.profile == GraphTransformerProfile::LongShortFusion
            && training_phase.runs_pretraining()
        {
            // LSTTN first reconstructs randomly withheld whole patches from
            // the unmasked long-history context, then fine-tunes those shared
            // representations for direct multi-horizon forecasting. Cover
            // the complete training history with bounded long-context
            // windows. Adjacent supervised windows differ by one timestamp,
            // so replaying reconstruction for every origin heavily
            // overweights the same observations. Non-overlapping contexts,
            // plus a final tail-aligned context, retain full data coverage
            // without restoring that quadratic amount of repeated work.
            let final_start = normalized.len() - self.config.lookback;
            let mut pretraining_starts = (0..=final_start)
                .step_by(self.config.lookback)
                .collect::<Vec<_>>();
            if pretraining_starts.last().copied() != Some(final_start) {
                pretraining_starts.push(final_start);
            }
            let pretraining_epochs = (self.config.epochs / 4).max(1);
            let total_pretraining = pretraining_epochs * pretraining_starts.len();
            for epoch in 0..pretraining_epochs {
                for (window_index, start) in pretraining_starts.iter().enumerate() {
                    let task_index = epoch * pretraining_starts.len() + window_index;
                    if task_index < pretraining_completed {
                        continue;
                    }
                    #[cfg(all(feature = "cuda", target_os = "linux"))]
                    if let Some(executor) = cuda_executor.as_mut() {
                        executor.cuda_train_masked_subseries_reconstruction(
                            &mut state,
                            &normalized[*start..*start + self.config.lookback],
                            self.config.learning_rate,
                            self.config.weight_decay,
                        )?;
                    } else {
                        state.train_masked_subseries_reconstruction(
                            &normalized[*start..*start + self.config.lookback],
                            self.config.learning_rate,
                            self.config.weight_decay,
                            Some(&backend),
                        )?;
                    }
                    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
                    state.train_masked_subseries_reconstruction(
                        &normalized[*start..*start + self.config.lookback],
                        self.config.learning_rate,
                        self.config.weight_decay,
                        Some(&backend),
                    )?;
                    pretraining_completed = task_index + 1;
                    eprintln!(
                        "LSTTN pretrain epoch {}/{} window {}/{} ({}/{})",
                        epoch + 1,
                        pretraining_epochs,
                        window_index + 1,
                        pretraining_starts.len(),
                        pretraining_completed,
                        total_pretraining
                    );
                    if let Some(path) = checkpoint_path {
                        write_lsttn_checkpoint(
                            path,
                            &LsttnTrainingCheckpoint {
                                version: 3,
                                architecture_fingerprint,
                                backbone_fingerprint,
                                shard_data_fingerprint,
                                epoch_fingerprint,
                                config_json: config_json.clone(),
                                state: state.clone(),
                                pretraining_completed,
                                supervised_batches_completed,
                                complete: false,
                            },
                        )?;
                    }
                }
            }
        }
        let supervised_starts = (0..sample_count).collect::<Vec<_>>();
        let frozen_lsttn_cache = if self.config.profile == GraphTransformerProfile::LongShortFusion
            && !uses_cuda_tensor_executor
            && training_phase.runs_supervised()
        {
            let patch_width = (self.config.periodicity / 24).max(1);
            let patches = self.config.lookback / patch_width;
            let estimated_bytes = supervised_starts
                .len()
                .saturating_mul(patches)
                .saturating_mul(frame.node_ids.len())
                .saturating_mul(self.config.hidden_size)
                .saturating_mul(std::mem::size_of::<f32>());
            const MAX_FROZEN_CACHE_BYTES: usize = 512 * 1024 * 1024;
            if estimated_bytes <= MAX_FROZEN_CACHE_BYTES {
                eprintln!(
                    "LSTTN caching frozen MST patch representations for {} supervised windows ({:.1} MiB)",
                    supervised_starts.len(),
                    estimated_bytes as f64 / (1024.0 * 1024.0)
                );
                Some(
                    supervised_starts
                        .par_iter()
                        .map(|start| {
                            let cutoff = *start + self.config.lookback;
                            state.frozen_lsttn_patch_representations(
                                &normalized[*start..cutoff],
                                &adjacency,
                                *start,
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                eprintln!(
                    "LSTTN frozen MST cache would require {:.1} GiB; using bounded per-window reconstruction",
                    estimated_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                None
            }
        } else {
            None
        };
        if training_phase.runs_supervised() {
            if self.config.profile == GraphTransformerProfile::LongShortFusion {
                // The reference LSTTN trainer uses 32-example mini-batches. Each
                // example is independent up to Adam's update, so evaluate all
                // scalar tapes on Rayon workers, average their gradients in
                // stable start-order, and take one serialized Adam step.
                let lsttn_batch_size = self.config.batch_size;
                let batches_per_epoch = supervised_starts.len().div_ceil(lsttn_batch_size);
                let total_batches = batches_per_epoch * self.config.epochs;
                for epoch in 0..self.config.epochs {
                    for (batch_index, starts) in
                        supervised_starts.chunks(lsttn_batch_size).enumerate()
                    {
                        let task_index = epoch * batches_per_epoch + batch_index;
                        if task_index < supervised_batches_completed {
                            continue;
                        }
                        let scheduler_steps = [1usize, 18, 36, 54, 72]
                            .into_iter()
                            .filter(|milestone| *milestone <= epoch)
                            .count();
                        let epoch_learning_rate =
                            self.config.learning_rate * 0.5_f64.powi(scheduler_steps as i32);
                        #[cfg(all(feature = "cuda", target_os = "linux"))]
                        let cuda_batch_loss = if let Some(executor) = cuda_executor.as_mut() {
                            let windows = starts
                                .iter()
                                .map(|start| {
                                    let cutoff = *start + self.config.lookback;
                                    &normalized[*start..cutoff]
                                })
                                .collect::<Vec<_>>();
                            let targets = starts
                                .iter()
                                .map(|start| {
                                    let cutoff = *start + self.config.lookback;
                                    &training_normalized[cutoff..cutoff + frame.horizon]
                                })
                                .collect::<Vec<_>>();
                            let objective_weights = starts
                                .iter()
                                .map(|start| {
                                    let cutoff = *start + self.config.lookback;
                                    (0..frame.horizon)
                                        .map(|horizon| {
                                            (0..frame.node_ids.len())
                                                .map(|node| {
                                                    let observed =
                                                        frame.target_mask.as_ref().is_none_or(
                                                            |mask| mask[cutoff + horizon][node],
                                                        );
                                                    let imputed =
                                                        frame.imputed_mask.as_ref().is_some_and(
                                                            |mask| mask[cutoff + horizon][node],
                                                        );
                                                    let owned = frame
                                                        .owner_mask
                                                        .as_ref()
                                                        .is_none_or(|owners| owners[node]);
                                                    if observed && !imputed && owned {
                                                        frame.target_weights.as_ref().map_or(
                                                            1.0,
                                                            |weights| {
                                                                weights[cutoff + horizon][node]
                                                            },
                                                        )
                                                    } else {
                                                        0.0
                                                    }
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            let weights = objective_weights
                                .iter()
                                .map(Vec::as_slice)
                                .collect::<Vec<_>>();
                            executor.upload_supervised_batch(&windows, &targets, &weights, 1)?;
                            executor.supervised_forward(
                                &state,
                                starts.len(),
                                1,
                                starts[0],
                                true,
                            )?;
                            executor.supervised_backward(
                                &state,
                                starts.len(),
                                1,
                                starts[0],
                                true,
                            )?;
                            executor.freeze_pretrained_transformer_gradients(&state)?;
                            executor.portable_adamw_step(
                                &mut state,
                                epoch_learning_rate,
                                self.config.weight_decay,
                            )?;
                            let loss = executor.mean_supervised_loss()?;
                            Some(loss)
                        } else {
                            None
                        };
                        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
                        let cuda_batch_loss: Option<f64> = None;
                        let mean_batch_loss = if let Some(loss) = cuda_batch_loss {
                            loss
                        } else {
                            let examples = starts
                                .par_iter()
                                .map(|start| {
                                    let cutoff = *start + self.config.lookback;
                                    let cache_index = supervised_starts
                                        .binary_search(start)
                                        .expect("supervised start has a frozen MST cache entry");
                                    state.lsttn_example_loss_and_gradients(
                                        &normalized[*start..cutoff],
                                        &adjacency,
                                        &training_normalized[cutoff..cutoff + frame.horizon],
                                        *start,
                                        frozen_lsttn_cache
                                            .as_ref()
                                            .map(|cache| cache[cache_index].as_slice()),
                                        lsttn_time_features
                                            .as_ref()
                                            .map(|features| &features[*start..cutoff]),
                                        frame
                                            .target_mask
                                            .as_ref()
                                            .map(|mask| &mask[cutoff..cutoff + frame.horizon]),
                                        frame
                                            .imputed_mask
                                            .as_ref()
                                            .map(|mask| &mask[cutoff..cutoff + frame.horizon]),
                                        frame.target_weights.as_ref().map(|weights| {
                                            &weights[cutoff..cutoff + frame.horizon]
                                        }),
                                        frame.owner_mask.as_deref(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let mut gradients = vec![0.0; state.parameters.len()];
                            let mut loss = 0.0;
                            for (example_loss, example_gradients) in examples {
                                loss += example_loss;
                                for (total, gradient) in gradients.iter_mut().zip(example_gradients)
                                {
                                    *total += gradient;
                                }
                            }
                            let batch_scale = 1.0 / starts.len() as f64;
                            for gradient in &mut gradients {
                                *gradient *= batch_scale;
                            }
                            state
                                .freeze_lsttn_transformer_gradients(state.layout(), &mut gradients);
                            if training_phase.freezes_shared() {
                                state.freeze_shared_gradients(&mut gradients);
                            }
                            clip_gradient_norm(&mut gradients, 3.0);
                            state.adamw_step(
                                &gradients,
                                epoch_learning_rate,
                                self.config.weight_decay,
                                Some(&backend),
                            )?;
                            loss * batch_scale
                        };
                        supervised_batches_completed = task_index + 1;
                        eprintln!(
                            "LSTTN supervised epoch {}/{} batch {}/{} ({}/{}) loss={:.8}",
                            epoch + 1,
                            self.config.epochs,
                            batch_index + 1,
                            batches_per_epoch,
                            supervised_batches_completed,
                            total_batches,
                            mean_batch_loss
                        );
                        if let Some(path) = checkpoint_path {
                            write_lsttn_checkpoint(
                                path,
                                &LsttnTrainingCheckpoint {
                                    version: 3,
                                    architecture_fingerprint,
                                    backbone_fingerprint,
                                    shard_data_fingerprint,
                                    epoch_fingerprint,
                                    config_json: config_json.clone(),
                                    state: state.clone(),
                                    pretraining_completed,
                                    supervised_batches_completed,
                                    complete: false,
                                },
                            )?;
                        }
                    }
                }
            } else {
                for _ in 0..self.config.epochs {
                    for start in supervised_starts.iter().copied() {
                        let cutoff = start + self.config.lookback;
                        // Keep the frame's native resolution through the input
                        // projection. The long branch forms learned patches inside
                        // `forward`, while the short branch consumes the configured
                        // number of recent rows instead of pre-averaged values.
                        let long_context_is_pooled = false;
                        let model_window = &normalized[start..cutoff];
                        // LSTTN's decoder predicts the traffic-flow value at every
                        // forecast step directly.  Do not turn this profile into a
                        // residual forecaster: that changes both the paper's output
                        // equation and the meaning of its multi-step MAE objective.
                        // Other graph-transformer profiles retain their established
                        // displacement heads for backward-compatible behaviour.
                        let targets: Vec<Vec<f64>> =
                            if self.config.profile == GraphTransformerProfile::LongShortFusion {
                                training_normalized[cutoff..cutoff + frame.horizon].to_vec()
                            } else {
                                let baseline = &normalized[cutoff - 1];
                                training_normalized[cutoff..cutoff + frame.horizon]
                                    .iter()
                                    .map(|row| {
                                        row.iter()
                                            .zip(baseline)
                                            .map(|(value, base)| value - base)
                                            .collect()
                                    })
                                    .collect()
                            };
                        if let Some(environments) = &spatial_shift_environments {
                            // First learn each expert graphon on the observed traffic
                            // environment.  The episodic pass then hides the
                            // environment's designated expert, forcing the router to
                            // mix the remaining graphons for that shifted relation.
                            state.train_example_with_context(
                                &self.config.profile,
                                model_window,
                                &adjacency,
                                &targets,
                                None,
                                self.config.learning_rate * 0.5,
                                self.config.weight_decay,
                                start,
                                long_context_is_pooled,
                                Some(&backend),
                            )?;
                            let environment = environments[cutoff % environments.len()];
                            state.train_example_with_context(
                                &self.config.profile,
                                model_window,
                                &adjacency,
                                &targets,
                                Some(environment),
                                self.config.learning_rate * 0.5,
                                self.config.weight_decay,
                                start,
                                long_context_is_pooled,
                                Some(&backend),
                            )?;
                        } else {
                            state.train_example_with_context(
                                &self.config.profile,
                                model_window,
                                &adjacency,
                                &targets,
                                None,
                                self.config.learning_rate,
                                self.config.weight_decay,
                                start,
                                long_context_is_pooled,
                                Some(&backend),
                            )?;
                        }
                    }
                }
            }
        }
        self.trainable_state = Some(state);
        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.horizon = frame.horizon;
        self.adjacency = Some(adjacency);
        self.history = normalized;
        self.history_time_features = lsttn_time_features;
        self.owner_mask = frame
            .owner_mask
            .clone()
            .unwrap_or_else(|| vec![true; frame.node_ids.len()]);
        self.fit_frozen_patch_cache_bytes = frozen_lsttn_cache.as_ref().map_or(0, |cache| {
            cache
                .iter()
                .flat_map(|window| window.iter())
                .flat_map(|patch| patch.iter())
                .map(Vec::len)
                .sum::<usize>()
                * std::mem::size_of::<f32>()
        });
        // A scalar tape retains forward values and gradients for the active
        // mini-batch.  This is an allocation upper bound at the supervised
        // checkpoint, not a process-wide RSS estimate.
        self.fit_activation_tape_bytes = self
            .config
            .batch_size
            .saturating_mul(self.config.lookback)
            .saturating_mul(frame.node_ids.len())
            .saturating_mul(self.config.hidden_size)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<f64>());
        self.target_mean = mean;
        self.target_scale = scale;
        if let Some(path) = checkpoint_path {
            write_lsttn_checkpoint(
                path,
                &LsttnTrainingCheckpoint {
                    version: 3,
                    architecture_fingerprint,
                    backbone_fingerprint,
                    shard_data_fingerprint,
                    epoch_fingerprint,
                    config_json,
                    state: self
                        .trainable_state
                        .as_ref()
                        .expect("fitted trainable state")
                        .clone(),
                    pretraining_completed,
                    supervised_batches_completed,
                    complete: true,
                },
            )?;
        }
        Ok(())
    }

    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "prediction horizon must be positive".to_string(),
            ));
        }
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        let adjacency = self.adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        let backend = BackendSelection {
            requested: self.config.backend.requested.clone(),
            selected: self.config.backend.selected.clone(),
            available: self.config.backend.available.clone(),
        };
        let mut history = self.history.clone();
        let mut history_time_features = self.history_time_features.clone();
        let mut output = Vec::with_capacity(horizon);
        while output.len() < horizon {
            let start = history.len() - self.config.lookback;
            // Spatial-shift inference dynamically recomputes every expert
            // graphon and its router weight from this observed window.  The
            // fitted parameters remain fixed, matching the paper's testing
            // policy and preventing hidden test-time optimization.
            let long_context_is_pooled = false;
            let model_window = &history[start..];
            let rows = state.predict_window_with_context(
                &self.config.profile,
                model_window,
                adjacency,
                start,
                long_context_is_pooled,
                Some(&backend),
                history_time_features
                    .as_ref()
                    .map(|features| &features[start..]),
            )?;
            // The LSTTN decoder is a direct multi-horizon traffic-flow head,
            // as in the paper.  The other profiles preserve their residual
            // decoder contract.
            let baseline = history
                .last()
                .expect("forecast history is non-empty")
                .clone();
            for row in rows.iter().take(self.horizon.min(horizon - output.len())) {
                let next = if self.config.profile == GraphTransformerProfile::LongShortFusion {
                    row.clone()
                } else {
                    row.iter()
                        .zip(&baseline)
                        .map(|(delta, value)| delta + value)
                        .collect::<Vec<_>>()
                };
                output.push(next.clone());
                history.push(next);
                if let Some(features) = &mut history_time_features {
                    let increment = 1.0 / self.config.periodicity as f64;
                    let next_time = features
                        .last()
                        .expect("fitted LSTTN time features are non-empty")
                        .iter()
                        .map(|value| (value + increment).rem_euclid(1.0))
                        .collect();
                    features.push(next_time);
                }
            }
        }
        Ok(output
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|v| v * self.target_scale + self.target_mean)
                    .collect()
            })
            .collect())
    }

    /// The deterministic direct decoder is the model's native median surface.
    pub fn predict_median(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        self.predict(horizon)
    }

    pub fn predict_conformal_from_calibration(
        &self,
        horizon: usize,
        calibration_actual: &[Vec<Vec<f64>>],
        calibration_median: &[Vec<Vec<f64>>],
        alpha: f64,
    ) -> Result<LsttnConformalForecast> {
        let median = self.predict_median(horizon)?;
        lsttn_conformal_forecast(calibration_actual, calibration_median, &median, alpha)
    }

    pub fn historical_fits(&self) -> Result<(usize, Vec<Vec<f64>>)> {
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        let adjacency = self.adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        if self.history.len() <= self.config.lookback {
            return Ok((self.config.lookback, Vec::new()));
        }
        let backend = BackendSelection {
            requested: self.config.backend.requested.clone(),
            selected: self.config.backend.selected.clone(),
            available: self.config.backend.available.clone(),
        };
        let mut fitted = Vec::with_capacity(self.history.len() - self.config.lookback);
        for cutoff in self.config.lookback..self.history.len() {
            let start = cutoff - self.config.lookback;
            let window = &self.history[start..cutoff];
            let predictions = state.predict_window_with_context(
                &self.config.profile,
                window,
                adjacency,
                start,
                false,
                Some(&backend),
                self.history_time_features
                    .as_ref()
                    .map(|features| &features[start..cutoff]),
            )?;
            let first = predictions.first().ok_or_else(|| {
                GeoStError::InvalidFrame("LSTTN historical fit returned no horizon".to_string())
            })?;
            let baseline = window.last().expect("historical fit window is non-empty");
            fitted.push(
                first
                    .iter()
                    .zip(baseline)
                    .map(|(prediction, previous)| {
                        let normalized =
                            if self.config.profile == GraphTransformerProfile::LongShortFusion {
                                *prediction
                            } else {
                                prediction + previous
                            };
                        normalized * self.target_scale + self.target_mean
                    })
                    .collect(),
            );
        }
        Ok((self.config.lookback, fitted))
    }

    /// Return forecasts only for this shard's owner nodes.  Halo nodes remain
    /// available internally for graph message passing but are never emitted
    /// from this owner-scoped surface.
    pub fn predict_owned(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        let predictions = self.predict(horizon)?;
        if self.owner_mask.len() != self.node_ids.len() {
            return Err(GeoStError::InvalidFrame(
                "artifact is missing the required owner mask; refit this model".to_string(),
            ));
        }
        Ok(predictions
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(&self.owner_mask)
                    .filter_map(|(value, owner)| (*owner).then_some(value))
                    .collect()
            })
            .collect())
    }

    pub fn score(&self, actual: &[Vec<f64>]) -> Result<f64> {
        if actual.is_empty() {
            return Err(GeoStError::InvalidFrame(
                "actual horizon cannot be empty".to_string(),
            ));
        }
        let predicted = self.predict(actual.len())?;
        let mut squared = 0.0;
        let mut count = 0usize;
        for (prediction, observation) in predicted.iter().zip(actual) {
            if prediction.len() != observation.len() {
                return Err(GeoStError::InvalidFrame(
                    "prediction and actual rows must have the same width".to_string(),
                ));
            }
            for (p, y) in prediction.iter().zip(observation) {
                squared += (p - y).powi(2);
                count += 1;
            }
        }
        Ok((squared / count as f64).sqrt())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_artifact()?;
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
    pub fn save_local(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_artifact()?;
        let mut local = self.clone();
        local
            .trainable_state
            .as_mut()
            .ok_or(GeoStError::NotFit)?
            .compact_to_local_state();
        fs::write(path, serde_json::to_string_pretty(&local)?)?;
        Ok(())
    }

    /// Persist local state, shared state, and a commit manifest. The manifest
    /// is the transaction boundary: readers must use `load_shard_pair` and
    /// will reject a kill-at-any-write-boundary partial update.
    pub fn save_shard_pair(
        &self,
        local_path: impl AsRef<Path>,
        shared_path: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.validate_artifact()?;
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        let shared = state.export_shared(self.target_mean, self.target_scale);
        let mut local = self.clone();
        local
            .trainable_state
            .as_mut()
            .ok_or(GeoStError::NotFit)?
            .compact_to_local_state();
        let local_bytes = serde_json::to_vec(&local)?;
        let shared_bytes = serde_json::to_vec(&shared)?;
        let manifest = LsttnShardPairManifest {
            version: 1,
            local_hash: bytes_fingerprint(&local_bytes),
            shared_hash: bytes_fingerprint(&shared_bytes),
            architecture_fingerprint: state.architecture_fingerprint,
            backbone_fingerprint: state.backbone_fingerprint,
            shared_steps: state.shared_steps,
            local_steps: state.local_steps,
        };
        atomic_write_bytes(local_path.as_ref(), &local_bytes)?;
        atomic_write_bytes(shared_path.as_ref(), &shared_bytes)?;
        atomic_write_bytes(manifest_path.as_ref(), &serde_json::to_vec(&manifest)?)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut model: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        if let Some(state) = &mut model.trainable_state {
            state.expand_local_state()?;
        }
        model.validate_artifact()?;
        Ok(model)
    }
    pub fn load_shard(
        local_path: impl AsRef<Path>,
        shared_state_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut model = Self::load(local_path)?;
        let shared: LsttnSharedState =
            serde_json::from_str(&fs::read_to_string(shared_state_path)?)?;
        let state = model.trainable_state.as_mut().ok_or(GeoStError::NotFit)?;
        state.import_shared(&shared)?;
        model.target_mean = shared.target_mean;
        model.target_scale = shared.target_scale;
        state.target_scale = shared.target_scale;
        state.normalized_zero = -shared.target_mean / shared.target_scale;
        Ok(model)
    }

    pub fn load_shard_pair(
        local_path: impl AsRef<Path>,
        shared_path: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let manifest: LsttnShardPairManifest =
            serde_json::from_slice(&fs::read(manifest_path.as_ref())?)?;
        let local = fs::read(local_path.as_ref())?;
        let shared = fs::read(shared_path.as_ref())?;
        if manifest.version != 1
            || manifest.local_hash != bytes_fingerprint(&local)
            || manifest.shared_hash != bytes_fingerprint(&shared)
        {
            return Err(GeoStError::InvalidFrame(
                "LSTTN shard pair is incomplete or does not match its commit manifest".to_string(),
            ));
        }
        let model = Self::load_shard(local_path, shared_path)?;
        let state = model.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        if state.architecture_fingerprint != manifest.architecture_fingerprint
            || state.backbone_fingerprint != manifest.backbone_fingerprint
            || state.shared_steps != manifest.shared_steps
            || state.local_steps != manifest.local_steps
        {
            return Err(GeoStError::InvalidFrame(
                "LSTTN shard pair state does not match its commit manifest".to_string(),
            ));
        }
        Ok(model)
    }
    pub fn to_json_string(&self) -> Result<String> {
        self.validate_artifact()?;
        Ok(serde_json::to_string(self)?)
    }
    pub fn from_json_string(value: &str) -> Result<Self> {
        let mut model: Self = serde_json::from_str(value)?;
        if let Some(state) = &mut model.trainable_state {
            state.expand_local_state()?;
        }
        model.validate_artifact()?;
        Ok(model)
    }
    pub fn backend(&self) -> String {
        self.config.backend.selected.clone()
    }

    fn validate_artifact(&self) -> Result<()> {
        let Some(state) = &self.trainable_state else {
            return Ok(());
        };
        let expected_parameters = state.layout().total;
        let valid = state.parameters.len() == expected_parameters
            && state.first_moment.len() == expected_parameters
            && state.second_moment.len() == expected_parameters
            && state.nodes == self.node_ids.len()
            && state.hidden == self.config.hidden_size
            && state.attention_heads == self.config.attention_heads
            && state.periodicity == self.config.periodicity
            && state.recent_window == self.config.recent_window
            && state.context_window == self.config.lookback
            && state.horizons == self.horizon
            && self.history.iter().all(|row| row.len() == state.nodes)
            && self.owner_mask.len() == state.nodes
            && self.history_time_features.as_ref().is_none_or(|features| {
                features.len() == self.history.len()
                    && features.iter().all(|row| row.len() == state.nodes)
            });
        if self.history_time_features.is_some()
            && self
                .covariate_roles
                .as_ref()
                .is_none_or(|roles| !roles.iter().any(|role| role == "time_of_day"))
        {
            return Err(GeoStError::InvalidFrame(
                "artifact contains LSTTN covariates without a declared time_of_day role; refit with this CartoBoost version".to_string(),
            ));
        }
        if !valid {
            return Err(GeoStError::InvalidFrame(
                "paper graph-transformer artifact is incompatible with the current native architecture; refit and save the model with this CartoBoost version"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn architecture_report(&self) -> PaperGraphTransformerArchitectureReport {
        let mut components = match self.config.profile {
            GraphTransformerProfile::HeterogeneousMoE => vec![
                "time2vec_temporal_encoding",
                "learned_in_out_degree_embeddings",
                "three_stage_causal_temporal_spatial_attention",
                "shortest_path_distance_attention_bias_embeddings",
                "independent_spatial_temporal_routed_moe",
                "moe_load_balancing_loss",
            ],
            GraphTransformerProfile::EfficientHighOrder => vec![
                "daily_weekly_position_encoding",
                "retained_graph_propagation_orders",
                "orderwise_shared_qkv_scaling_normalized_linear_attention",
                "recursive_pointwise_high_order_interaction",
            ],
            GraphTransformerProfile::LongShortFusion => vec![
                "seventy_five_percent_masked_patch_pretraining",
                "learned_patch_convolution_and_temporal_positions",
                "four_layer_multihead_transformer_encoder",
                "one_layer_mask_token_transformer_decoder",
                "frozen_masked_subseries_transformer",
                "four_stage_dilated_long_trend_convolution",
                "previous_day_and_week_transformer_states",
                "independent_forward_backward_adaptive_periodic_graph_convolutions",
                "eight_layer_causal_graph_wavenet_short_branch",
                "signal_and_time_of_day_short_term_channels",
                "long_periodic_short_feature_fusion",
                "zero_masked_direct_multi_horizon_mae",
                "all_origin_thirty_two_window_supervision",
            ],
            GraphTransformerProfile::GatedGraphTemporal => vec![
                "normalized_graph_convolution",
                "causal_temporal_attention",
                "gru_reset_update_gates",
            ],
            GraphTransformerProfile::SpatialShiftGraphonMoE => vec![
                "maximum_spatiotemporal_graph_division",
                "input_conditioned_expert_graphons",
                "binary_gumbel_softmax_graphon_sampling",
                "softmax_graphon_mixup",
                "episodic_held_out_environment_router_training",
                "stop_gradient_expert_graphons",
                "graphon_mixture_forecast_head",
            ],
        }
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        if self.config.profile == GraphTransformerProfile::LongShortFusion
            && self.config.backend.selected != "cpu"
        {
            components.push(if self.config.backend.selected == "metal" {
                "metal_full_graph_training_and_inference".to_string()
            } else {
                format!(
                    "{}_accelerated_graph_training_and_inference",
                    self.config.backend.selected
                )
            });
        }
        PaperGraphTransformerArchitectureReport {
            profile: self.config.profile.clone(),
            components,
            graphon_expert_count: if self.config.profile
                == GraphTransformerProfile::SpatialShiftGraphonMoE
            {
                self.config.experts
            } else {
                0
            },
            direct_multi_horizon: true,
            trainable_forecast_head: self.trainable_state.is_some(),
        }
    }

    pub fn parameter_inventory(&self) -> Result<Vec<LsttnParameterInventoryEntry>> {
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        let mut result = Vec::new();
        let alternate_state = TrainableGraphTransformerState {
            nodes: state.nodes.saturating_add(1),
            hidden: state.hidden,
            horizons: state.horizons,
            experts: state.experts,
            graph_order: state.graph_order,
            periodicity: state.periodicity,
            context_window: state.context_window,
            ..state.clone()
        };
        for ((name, range), (alternate_name, alternate_range)) in state
            .shared_ranges()
            .into_iter()
            .zip(alternate_state.shared_ranges())
        {
            if name != alternate_name
                || range.len() != alternate_range.len()
                || range.end > state.parameters.len()
            {
                return Err(GeoStError::InvalidFrame(format!(
                    "shared LSTTN parameter segment {name} is node-count-dependent or out of bounds"
                )));
            }
            result.push(parameter_inventory_entry(
                name,
                &state.parameters[range],
                "shared",
                false,
            ));
        }
        for (index, range) in state.local_ranges().into_iter().enumerate() {
            result.push(parameter_inventory_entry(
                &format!("local_{index}"),
                &state.parameters[range],
                "local",
                true,
            ));
        }
        Ok(result)
    }

    /// Native component accounting at the most recent fit checkpoint.
    pub fn memory_telemetry(&self) -> Result<LsttnMemoryTelemetry> {
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        let scalar = std::mem::size_of::<f64>();
        let frame_storage_bytes = self.history.iter().map(Vec::len).sum::<usize>() * scalar
            + self
                .history_time_features
                .as_ref()
                .map_or(0, |rows| rows.iter().map(Vec::len).sum::<usize>() * scalar)
            + self.adjacency.as_ref().map_or(0, |adjacency| {
                adjacency.data.len() * scalar
                    + adjacency.indices.len() * std::mem::size_of::<usize>()
                    + adjacency.indptr.len() * std::mem::size_of::<usize>()
            });
        let parameter_bytes = state.parameters.len() * scalar;
        let optimizer_moment_bytes =
            (state.first_moment.len() + state.second_moment.len()) * scalar;
        let graph_diffusion_workspace_bytes = self.adjacency.as_ref().map_or(0, |adjacency| {
            adjacency.indices.len() * std::mem::size_of::<usize>()
        });
        let telemetry = LsttnMemoryTelemetry {
            frame_storage_bytes,
            parameter_bytes,
            optimizer_moment_bytes,
            activation_tape_bytes: self.fit_activation_tape_bytes,
            frozen_patch_cache_bytes: self.fit_frozen_patch_cache_bytes,
            graph_diffusion_workspace_bytes,
            prediction_buffer_bytes: self
                .horizon
                .saturating_mul(state.nodes)
                .saturating_mul(scalar),
            total_accounted_bytes: 0,
        };
        Ok(LsttnMemoryTelemetry {
            total_accounted_bytes: telemetry.frame_storage_bytes
                + telemetry.parameter_bytes
                + telemetry.optimizer_moment_bytes
                + telemetry.activation_tape_bytes
                + telemetry.frozen_patch_cache_bytes
                + telemetry.graph_diffusion_workspace_bytes
                + telemetry.prediction_buffer_bytes,
            ..telemetry
        })
    }

    pub fn edge_diagnostics(&self) -> Result<Vec<LsttnEdgeDiagnostic>> {
        let state = self.trainable_state.as_ref().ok_or(GeoStError::NotFit)?;
        let adjacency = self.adjacency.as_ref().ok_or(GeoStError::NotFit)?;
        let layout = state.layout();
        let mut incoming_sum = vec![0.0; state.nodes];
        for (target, value) in adjacency.indices.iter().zip(&adjacency.data) {
            incoming_sum[*target] += value.abs();
        }
        let mut diagnostics = Vec::with_capacity(adjacency.indices.len());
        for target in 0..state.nodes {
            if !self.owner_mask[target] {
                continue;
            }
            let range = adjacency.indptr[target]..adjacency.indptr[target + 1];
            let row_sum = adjacency.data[range.clone()]
                .iter()
                .map(|v| v.abs())
                .sum::<f64>();
            let logits = range
                .clone()
                .map(|edge| {
                    let source = adjacency.indices[edge];
                    (0..10)
                        .map(|latent| {
                            state.parameters[layout.lsttn_adaptive_source + target * 10 + latent]
                                * state.parameters
                                    [layout.lsttn_adaptive_target + source * 10 + latent]
                        })
                        .sum::<f64>()
                        .max(0.0)
                })
                .collect::<Vec<_>>();
            let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator = logits
                .iter()
                .map(|value| (*value - max_logit).exp())
                .sum::<f64>();
            for (offset, edge) in range.enumerate() {
                let source = adjacency.indices[edge];
                let attention = (logits[offset] - max_logit).exp() / denominator.max(1e-12);
                let horizon_sensitivity = (0..state.horizons)
                    .map(|horizon| {
                        state.parameters[layout.output + horizon * (state.hidden + 1)
                            ..layout.output + (horizon + 1) * (state.hidden + 1)]
                            .iter()
                            .map(|value| value.abs())
                            .sum::<f64>()
                    })
                    .collect();
                diagnostics.push(LsttnEdgeDiagnostic {
                    source,
                    target,
                    structural_prior: adjacency.data[edge],
                    forward_diffusion: adjacency.data[edge] / row_sum.max(1e-12),
                    backward_diffusion: adjacency.data[edge] / incoming_sum[source].max(1e-12),
                    learned_adaptive_contribution: logits[offset],
                    normalized_attention: attention,
                    horizon_sensitivity,
                });
            }
        }
        Ok(diagnostics)
    }
}

fn parameter_inventory_entry(
    name: &str,
    values: &[f64],
    ownership: &str,
    node_count_dependent: bool,
) -> LsttnParameterInventoryEntry {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in values {
        value.to_bits().hash(&mut hasher);
    }
    LsttnParameterInventoryEntry {
        name: name.to_string(),
        shape: vec![values.len()],
        byte_size: std::mem::size_of_val(values),
        ownership: ownership.to_string(),
        optimizer_ownership: ownership.to_string(),
        hash: format!("{:016x}", hasher.finish()),
        node_count_dependent,
    }
}

fn graph_in_degrees(adjacency: &CsrAdjacency, nodes: usize) -> Vec<f64> {
    let mut result = vec![0.0; nodes];
    for target in &adjacency.indices {
        result[*target] += 1.0;
    }
    result
}

fn graph_out_degrees(adjacency: &CsrAdjacency, nodes: usize) -> Vec<f64> {
    (0..nodes)
        .map(|node| (adjacency.indptr[node + 1] - adjacency.indptr[node]) as f64)
        .collect()
}

/// Draw a reproducible 75% patch mask.  Randomized patch selection mirrors
/// LSTTN's pretraining policy without making fitted artifacts nondeterministic.
fn masked_patch_indices(patches: usize, step: u64) -> Vec<usize> {
    let mut indices = (0..patches).collect::<Vec<_>>();
    let mut state = step ^ 0x9e37_79b9_7f4a_7c15;
    for index in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        indices.swap(index, (state as usize) % (index + 1));
    }
    let masked_count = (patches * 3).div_ceil(4).min(patches.saturating_sub(1));
    indices.truncate(masked_count);
    indices.sort_unstable();
    indices
}

fn graph_temporal_training_fingerprint(frame: &GraphTemporalFrame) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    frame.node_ids.hash(&mut hasher);
    frame.timestamps.hash(&mut hasher);
    frame.horizon.hash(&mut hasher);
    frame.frequency.hash(&mut hasher);
    frame.adjacency.indptr.hash(&mut hasher);
    frame.adjacency.indices.hash(&mut hasher);
    for value in frame
        .target
        .iter()
        .flatten()
        .chain(frame.adjacency.data.iter())
        .chain(frame.covariates.iter().flatten().flatten().flatten())
    {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn stable_fingerprint(value: &impl Serialize) -> Result<u64> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_vec(value)?.hash(&mut hasher);
    Ok(hasher.finish())
}

fn bytes_fingerprint(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("part");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_lsttn_checkpoint(path: &Path, checkpoint: &LsttnTrainingCheckpoint) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.part");
    fs::write(&temporary, serde_json::to_vec(checkpoint)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Divide one recurring traffic cycle into contiguous graph environments.  We
/// first estimate each phase's graph relation from the rank ordering of node
/// signals across all observed cycles, then use dynamic programming to choose
/// the contiguous partition with maximal within-environment Kendall coherence.
/// The resulting labels are used as the known source environments during the
/// spatial-shift paper's episodic expert training policy.
fn maximum_spatiotemporal_graph_division(
    values: &[Vec<f64>],
    periodicity: usize,
    experts: usize,
) -> Result<Vec<usize>> {
    if values.len() < periodicity {
        return Err(GeoStError::InvalidFrame(format!(
            "spatial-shift graph division requires at least one complete period ({periodicity} observations)"
        )));
    }
    let nodes = values
        .first()
        .ok_or_else(|| {
            GeoStError::InvalidFrame("graph division requires observations".to_string())
        })?
        .len();
    let signatures = (0..periodicity)
        .map(|phase| {
            let rows = values
                .iter()
                .skip(phase)
                .step_by(periodicity)
                .collect::<Vec<_>>();
            let mean = (0..nodes)
                .map(|node| rows.iter().map(|row| row[node]).sum::<f64>() / rows.len() as f64)
                .collect::<Vec<_>>();
            rank_signature(&mean)
        })
        .collect::<Vec<_>>();
    let groups = experts.min(periodicity).max(1);
    let mut score = vec![vec![0.0; periodicity + 1]; periodicity];
    for (start, score_row) in score.iter_mut().enumerate().take(periodicity) {
        for (end, segment_score) in score_row
            .iter_mut()
            .enumerate()
            .take(periodicity + 1)
            .skip(start + 1)
        {
            let mut total = 0.0;
            let mut count = 0usize;
            for left in start..end {
                for right in left + 1..end {
                    total += kendall_tau(&signatures[left], &signatures[right]);
                    count += 1;
                }
            }
            *segment_score = if count == 0 {
                0.0
            } else {
                total / count as f64
            };
        }
    }
    let mut dp = vec![vec![f64::NEG_INFINITY; periodicity + 1]; groups + 1];
    let mut parent = vec![vec![0usize; periodicity + 1]; groups + 1];
    dp[0][0] = 0.0;
    for group in 1..=groups {
        for end in group..=periodicity {
            for (start, score_row) in score.iter().enumerate().take(end).skip(group - 1) {
                let candidate = dp[group - 1][start] + score_row[end];
                if candidate > dp[group][end] {
                    dp[group][end] = candidate;
                    parent[group][end] = start;
                }
            }
        }
    }
    let mut boundaries = Vec::with_capacity(groups + 1);
    let mut end = periodicity;
    boundaries.push(end);
    for group in (1..=groups).rev() {
        end = parent[group][end];
        boundaries.push(end);
    }
    boundaries.reverse();
    let mut labels = vec![0usize; periodicity];
    for group in 0..groups {
        for label in labels
            .iter_mut()
            .take(boundaries[group + 1])
            .skip(boundaries[group])
        {
            *label = group;
        }
    }
    Ok(labels)
}

fn rank_signature(values: &[f64]) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0usize; values.len()];
    for (rank, node) in order.into_iter().enumerate() {
        ranks[node] = rank;
    }
    ranks
}

fn kendall_tau(left: &[usize], right: &[usize]) -> f64 {
    if left.len() < 2 || left.len() != right.len() {
        return 0.0;
    }
    let mut concordant = 0isize;
    let mut discordant = 0isize;
    for first in 0..left.len() {
        for second in first + 1..left.len() {
            let left_order = left[first].cmp(&left[second]);
            let right_order = right[first].cmp(&right[second]);
            if left_order == right_order {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    (concordant - discordant) as f64 / (left.len() * (left.len() - 1) / 2) as f64
}

pub fn graph_metrics(
    predictions: &[Vec<f64>],
    actual: &[Vec<f64>],
    node_ids: &[String],
    adjacency: &CsrAdjacency,
) -> GraphForecastMetrics {
    let horizons = predictions.len().min(actual.len());
    let nodes = node_ids.len();
    let by_horizon = (0..horizons)
        .map(|h| {
            let (mae, rmse, wape) = metric_values(&predictions[h], &actual[h]);
            HorizonMetric {
                horizon: h + 1,
                mae,
                rmse,
                wape,
            }
        })
        .collect();
    let by_node = (0..nodes)
        .map(|node| {
            let pred: Vec<f64> = (0..horizons).map(|h| predictions[h][node]).collect();
            let obs: Vec<f64> = (0..horizons).map(|h| actual[h][node]).collect();
            let (mae, rmse, wape) = metric_values(&pred, &obs);
            NodeMetric {
                node_id: node_ids[node].clone(),
                mae,
                rmse,
                wape,
            }
        })
        .collect();
    GraphForecastMetrics {
        by_horizon,
        by_node,
        graph_distance_residuals: distance_residuals(predictions, actual, adjacency, nodes),
    }
}

pub fn synthetic_graph_diffusion_frame() -> GraphTemporalFrame {
    let nodes = 4;
    let adjacency = CsrAdjacency::new(vec![0, 1, 2, 3, 4], vec![1, 2, 3, 0], vec![1.0; 4], nodes)
        .expect("fixture adjacency");
    let mut target = Vec::new();
    for t in 0..80 {
        let mut row = Vec::with_capacity(nodes);
        for node in 0..nodes {
            let phase = (t as f64 - node as f64) * 0.45;
            let upstream_phase = (t as f64 - (node + 1) as f64) * 0.45;
            row.push(12.0 + 2.4 * phase.sin() + 1.1 * upstream_phase.cos());
        }
        target.push(row);
    }
    GraphTemporalFrame::new(
        (0..nodes).map(|idx| format!("zone_{idx}")).collect(),
        (0..80).map(i64::from).collect(),
        target,
        None,
        adjacency,
        3,
        "hourly".to_string(),
    )
    .expect("fixture frame")
}

pub fn traffic_style_fixture_frame() -> GraphTemporalFrame {
    let adjacency = CsrAdjacency::new(
        vec![0, 2, 3, 4],
        vec![1, 2, 2, 0],
        vec![0.7, 0.3, 1.0, 1.0],
        3,
    )
    .expect("traffic adjacency");
    let mut target = Vec::new();
    for t in 0..48 {
        let hour = t as f64;
        target.push(vec![
            18.0 + (hour / 24.0 * std::f64::consts::TAU).sin() * 4.0,
            16.0 + ((hour - 1.0) / 24.0 * std::f64::consts::TAU).sin() * 3.5,
            14.0 + ((hour - 2.0) / 24.0 * std::f64::consts::TAU).sin() * 3.0,
        ]);
    }
    GraphTemporalFrame::new(
        vec!["sensor_a".into(), "sensor_b".into(), "sensor_c".into()],
        (0..48).map(i64::from).collect(),
        target,
        None,
        adjacency,
        4,
        "hourly".to_string(),
    )
    .expect("traffic frame")
}

fn attention_pool(values: &[f64], query: &[f64], key: &[f64]) -> f64 {
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

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn blend_rows(actual: &[f64], predicted: &[f64], teacher_ratio: f64) -> Vec<f64> {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(&teacher, &model)| teacher_ratio * teacher + (1.0 - teacher_ratio) * model)
        .collect()
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

fn csr_edges(adjacency: &CsrAdjacency, node_count: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(adjacency.indices.len());
    for source in 0..node_count {
        for edge_idx in adjacency.indptr[source]..adjacency.indptr[source + 1] {
            edges.push((source, adjacency.indices[edge_idx]));
        }
    }
    edges
}

fn delayed_graph_signal(
    target: &[Vec<f64>],
    edges: &[(usize, usize)],
    weights: &[f64],
    delays: &[usize],
    time_idx: usize,
) -> Vec<f64> {
    let nodes = target.first().map_or(0, Vec::len);
    let mut signal = vec![0.0; nodes];
    let mut weight_sum = vec![0.0; nodes];
    for (edge_idx, &(source, target_node)) in edges.iter().enumerate() {
        let delay = delays[edge_idx];
        let lag_idx = (time_idx + 1).saturating_sub(delay);
        let weight = weights[edge_idx];
        signal[target_node] += weight * target[lag_idx][source];
        weight_sum[target_node] += weight.abs();
    }
    for node in 0..nodes {
        if weight_sum[node] > 1.0e-12 {
            signal[node] /= weight_sum[node];
        }
    }
    signal
}

fn delayed_graph_signal_backend(
    target: &[Vec<f64>],
    edges: &[(usize, usize)],
    weights: &[f64],
    delays: &[usize],
    time_idx: usize,
    backend: &ComputeBackendSelection,
) -> Result<Vec<f64>> {
    const DELAYED_GRAPH_DISPATCH_MIN_EDGES: usize = 16_384;
    if backend.selected == "cpu" || edges.len() < DELAYED_GRAPH_DISPATCH_MIN_EDGES {
        return Ok(delayed_graph_signal(
            target, edges, weights, delays, time_idx,
        ));
    }
    let nodes = target.first().map_or(0, Vec::len);
    let mut normalization = vec![0.0_f64; nodes];
    for (edge_idx, &(_, target_node)) in edges.iter().enumerate() {
        normalization[target_node] += weights[edge_idx].abs();
    }
    let mut unique_delays = delays.to_vec();
    unique_delays.sort_unstable();
    unique_delays.dedup();
    let selection = BackendSelection {
        requested: backend.requested.clone(),
        selected: backend.selected.clone(),
        available: backend.available.clone(),
    };
    let mut signal = vec![0.0_f32; nodes];
    for delay in unique_delays {
        let mut rows = vec![Vec::<(u32, f32)>::new(); nodes];
        for (edge_idx, &(source, target_node)) in edges.iter().enumerate() {
            if delays[edge_idx] == delay && normalization[target_node] > 1.0e-12 {
                rows[target_node].push((
                    source as u32,
                    (weights[edge_idx] / normalization[target_node]) as f32,
                ));
            }
        }
        let mut indptr = Vec::with_capacity(nodes + 1);
        let mut indices = Vec::new();
        let mut edge_weights = Vec::new();
        indptr.push(0_u32);
        for row in rows {
            for (source, weight) in row {
                indices.push(source);
                edge_weights.push(weight);
            }
            indptr.push(indices.len() as u32);
        }
        let lag_idx = (time_idx + 1).saturating_sub(delay);
        let values = target[lag_idx]
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let contribution =
            backend_csr_diffusion_f32(&selection, &indptr, &indices, &edge_weights, 1, &values)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        for (value, contribution) in signal.iter_mut().zip(contribution) {
            *value += contribution;
        }
    }
    Ok(signal.into_iter().map(f64::from).collect())
}

fn quantize_prediction(value: f64) -> f64 {
    if value.is_finite() {
        (value * 1.0e12).round() / 1.0e12
    } else {
        value
    }
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
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
            if factor == 0.0 {
                continue;
            }
            for (col, pivot_value) in pivot_row.iter().enumerate().take(n).skip(pivot) {
                matrix[row][col] -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    rhs
}

fn target_center_scale(target: &[Vec<f64>]) -> (f64, f64) {
    let mut count = 0.0;
    let mut sum = 0.0;
    for row in target {
        for value in row {
            count += 1.0;
            sum += value;
        }
    }
    let mean = if count > 0.0 { sum / count } else { 0.0 };
    let mut variance = 0.0;
    for row in target {
        for value in row {
            let centered = value - mean;
            variance += centered * centered;
        }
    }
    let scale = if count > 0.0 {
        (variance / count).sqrt().max(1.0e-6)
    } else {
        1.0
    };
    (mean, scale)
}

fn target_center_scale_observed(
    target: &[Vec<f64>],
    target_mask: Option<&[Vec<bool>]>,
    imputed_mask: Option<&[Vec<bool>]>,
) -> (f64, f64) {
    let values = target.iter().enumerate().flat_map(|(time, row)| {
        row.iter().enumerate().filter_map(move |(node, value)| {
            let observed = target_mask.is_none_or(|mask| mask[time][node]);
            let imputed = imputed_mask.is_some_and(|mask| mask[time][node]);
            (observed && !imputed).then_some(*value)
        })
    });
    let values = values.collect::<Vec<_>>();
    target_center_scale(&[values])
}

fn metric_values(predictions: &[f64], actual: &[f64]) -> (f64, f64, f64) {
    let n = predictions.len().max(1) as f64;
    let mut abs = 0.0;
    let mut squared = 0.0;
    let mut denom = 0.0;
    for (&pred, &obs) in predictions.iter().zip(actual.iter()) {
        let err = pred - obs;
        abs += err.abs();
        squared += err * err;
        denom += obs.abs();
    }
    (
        abs / n,
        (squared / n).sqrt(),
        if denom > 0.0 { abs / denom } else { 0.0 },
    )
}

fn distance_residuals(
    predictions: &[Vec<f64>],
    actual: &[Vec<f64>],
    adjacency: &CsrAdjacency,
    nodes: usize,
) -> Vec<GraphDistanceResidual> {
    let distances = graph_distances(adjacency, nodes);
    let mut sums = vec![0.0; nodes];
    let mut counts = vec![0usize; nodes];
    for distance_row in distances.iter().take(nodes) {
        for (target, distance) in distance_row.iter().enumerate().take(nodes) {
            let distance = *distance;
            if distance < nodes {
                for h in 0..predictions.len().min(actual.len()) {
                    sums[distance] += (predictions[h][target] - actual[h][target]).abs();
                    counts[distance] += 1;
                }
            }
        }
    }
    sums.into_iter()
        .zip(counts)
        .enumerate()
        .filter_map(|(distance, (sum, count))| {
            (count > 0).then_some(GraphDistanceResidual {
                distance,
                mean_abs_residual: sum / count as f64,
                count,
            })
        })
        .collect()
}

fn graph_distances(adjacency: &CsrAdjacency, nodes: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![usize::MAX / 4; nodes]; nodes];
    for (source, distances) in all.iter_mut().enumerate().take(nodes) {
        let mut queue = VecDeque::new();
        distances[source] = 0;
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            let next_distance = distances[node] + 1;
            for edge in adjacency.indptr[node]..adjacency.indptr[node + 1] {
                let next = adjacency.indices[edge];
                if distances[next] > next_distance {
                    distances[next] = next_distance;
                    queue.push_back(next);
                }
            }
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_compute_backend_selects_an_available_backend() {
        let selection = select_compute_backend(Some("auto")).unwrap();
        assert_eq!(selection.requested, "auto");
        assert!(selection.available.contains(&selection.selected));

        let default_selection = select_compute_backend(None).unwrap();
        assert_eq!(default_selection.requested, "auto");
        assert!(default_selection
            .available
            .contains(&default_selection.selected));
    }

    #[test]
    fn graph_affine_inference_avoids_small_device_launches() {
        for backend_name in available_compute_backends() {
            let backend = cartoboost_neural::select_backend_for(
                Some(&backend_name),
                BackendOperation::Affine,
            )
            .unwrap();
            let small = profitable_affine_backend(backend.clone(), 1, 4).unwrap();
            assert_eq!(small.selected, "cpu");
            let large = profitable_affine_backend(backend, 4_096, 4).unwrap();
            assert_eq!(
                large.selected,
                if backend_name == "cpu" {
                    "cpu"
                } else {
                    backend_name.as_str()
                }
            );
        }
    }

    #[test]
    fn dcrnn_recurrent_stage_matches_cpu_on_available_backends() {
        let nodes = 512;
        let indptr = (0..=nodes).collect::<Vec<_>>();
        let indices = (0..nodes)
            .map(|node| (node + 1) % nodes)
            .collect::<Vec<_>>();
        let adjacency = CsrAdjacency::new(indptr, indices, vec![1.0; nodes], nodes).unwrap();
        let reverse = adjacency.transpose(nodes);
        let state = (0..nodes)
            .map(|node| (node as f64 * 0.007).sin())
            .collect::<Vec<_>>();
        let previous = (0..nodes)
            .map(|node| {
                (0..4)
                    .map(|unit| ((node * 4 + unit) as f64 * 0.011).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let operations = [
            BackendOperation::Affine,
            BackendOperation::CsrDiffusion,
            BackendOperation::Dense,
        ];
        let build = |backend| {
            let mut model = DcrnnForecaster::new(DcrnnConfig {
                diffusion_steps: 4,
                hidden_size: 4,
                epochs: 1,
                backend,
                ..DcrnnConfig::default()
            })
            .unwrap();
            model.encoder_weights = deterministic_weight_matrix(4, 10, 17);
            model.recurrent_weights = deterministic_weight_matrix(4, 4, 29);
            model
        };
        let cpu_backend = select_compute_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected = build(cpu_backend)
            .recurrent_hidden(&previous, &state, &adjacency, &reverse)
            .unwrap();

        for backend_name in available_compute_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                select_compute_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual = build(backend)
                .recurrent_hidden(&previous, &state, &adjacency, &reverse)
                .unwrap_or_else(|error| panic!("{backend_name} DCRNN recurrent stage: {error}"));
            for (actual_row, expected_row) in actual.iter().zip(&expected) {
                for (actual, expected) in actual_row.iter().zip(expected_row) {
                    assert!(
                        (actual - expected).abs() < 3.0e-4,
                        "{backend_name} DCRNN mismatch: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn graph_wavenet_csr_features_match_cpu_on_available_backends() {
        let nodes = 512;
        let adjacency = CsrAdjacency::new(
            (0..=nodes).collect(),
            (0..nodes).map(|node| (node + 1) % nodes).collect(),
            vec![1.0; nodes],
            nodes,
        )
        .unwrap();
        let window = (0..8)
            .map(|time| {
                (0..nodes)
                    .map(|node| ((time * nodes + node) as f64 * 0.009).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let operations = [BackendOperation::Affine, BackendOperation::CsrDiffusion];
        let build = |backend| {
            GraphWaveNetForecaster::new(GraphWaveNetConfig {
                lookback: 8,
                dilation_depth: 4,
                hidden_size: 4,
                epochs: 1,
                backend,
                ..GraphWaveNetConfig::default()
            })
            .unwrap()
        };
        let cpu = select_compute_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected = build(cpu).wave_features(&window, &adjacency).unwrap();
        for backend_name in available_compute_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                select_compute_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual = build(backend)
                .wave_features(&window, &adjacency)
                .unwrap_or_else(|error| panic!("{backend_name} Graph WaveNet CSR: {error}"));
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 2.0e-4,
                    "{backend_name} Graph WaveNet mismatch: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn staeformer_spatial_attention_matches_cpu_on_available_backends() {
        let nodes = 512;
        let adjacency = CsrAdjacency::new(
            (0..=nodes).collect(),
            (0..nodes).map(|node| (node + 3) % nodes).collect(),
            vec![1.0; nodes],
            nodes,
        )
        .unwrap();
        let window = (0..8)
            .map(|time| {
                (0..nodes)
                    .map(|node| ((time * nodes + node) as f64 * 0.006).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let operations = [BackendOperation::Affine, BackendOperation::CsrDiffusion];
        let build = |backend| {
            let mut model = STAEformerForecaster::new(STAEformerConfig {
                lookback: 8,
                attention_heads: 4,
                hidden_size: 8,
                epochs: 1,
                backend,
                ..STAEformerConfig::default()
            })
            .unwrap();
            model.temporal_queries = deterministic_weight_matrix(4, 8, 31);
            model.temporal_keys = deterministic_weight_matrix(4, 8, 43);
            model.spatial_weights = vec![0.5, 0.75, 1.0, 1.25];
            model
        };
        let cpu = select_compute_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected = build(cpu).attention_features(&window, &adjacency).unwrap();
        for backend_name in available_compute_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                select_compute_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual = build(backend)
                .attention_features(&window, &adjacency)
                .unwrap_or_else(|error| panic!("{backend_name} STAEformer CSR: {error}"));
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 2.0e-4,
                    "{backend_name} STAEformer mismatch: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn delay_aware_decode_matches_cpu_on_available_backends() {
        let nodes = 8_192;
        let operations = [BackendOperation::CsrDiffusion, BackendOperation::Affine];
        let build = |backend| DelayAwareGraphTransformer {
            config: DelayAwareGraphConfig {
                horizon: 1,
                edge_delay_prior: vec![1; nodes],
                ridge: 1.0e-6,
                backend,
            },
            node_ids: (0..nodes).map(|node| format!("node_{node}")).collect(),
            frequency: "hourly".to_string(),
            edges: (0..nodes).map(|node| (node, (node + 1) % nodes)).collect(),
            edge_weights: vec![1.0; nodes],
            coefficients: vec![0.03, 0.8, 0.17],
            history: vec![(0..nodes).map(|node| (node as f64 * 0.004).sin()).collect()],
            target_mean: 0.0,
            target_scale: 1.0,
        };
        let cpu = select_compute_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected = build(cpu).predict(1).unwrap();
        for backend_name in available_compute_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                select_compute_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual = build(backend)
                .predict(1)
                .unwrap_or_else(|error| panic!("{backend_name} delay decode: {error}"));
            for (actual, expected) in actual[0].iter().zip(&expected[0]) {
                assert!(
                    (actual - expected).abs() < 2.0e-4,
                    "{backend_name} delay decode mismatch: {actual} vs {expected}"
                );
            }
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_lsttn_fit_uses_tensor_executor_without_scalar_fallback() {
        let backend = match select_compute_backend(Some("cuda")) {
            Ok(backend) => backend,
            Err(_) => return,
        };
        let mut model = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend,
        })
        .unwrap();
        model.fit(&traffic_style_fixture_frame()).unwrap();
        assert!(model
            .trainable_state
            .as_ref()
            .is_some_and(|state| state.steps > 0));
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_lsttn_pretraining_updates_mst_parameters_on_device() {
        if select_compute_backend(Some("cuda")).is_err() {
            return;
        }
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 17);
        state.target_scale = 1.0;
        state.normalized_zero = 0.0;
        let before = state.parameters.clone();
        let mut executor = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        let loss = executor
            .cuda_train_masked_subseries_reconstruction(&mut state, &frame.target[..8], 0.001, 0.0)
            .unwrap();
        assert!(loss.is_finite(), "{loss}");
        assert_eq!(state.steps, 1);
        let layout = state.layout();
        let changed_pretrain = (layout.pretrain_mask_token..layout.total)
            .any(|idx| (state.parameters[idx] - before[idx]).abs() > 1.0e-9);
        assert!(
            changed_pretrain,
            "CUDA pretraining must update MST/pretraining parameters"
        );
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_lsttn_executor_keeps_model_and_all_three_graphs_resident() {
        if select_compute_backend(Some("cuda")).is_err() {
            return;
        }
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 17);
        let layout = state.layout();
        for offset in [
            layout.lsttn_adaptive_source,
            layout.lsttn_adaptive_target,
            layout.lsttn_weekly_adaptive_source,
            layout.lsttn_weekly_adaptive_target,
            layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_target,
        ] {
            for (idx, value) in state.parameters[offset..offset + state.nodes * state.hidden]
                .iter_mut()
                .enumerate()
            {
                *value = 0.05 + (idx % state.hidden) as f64 * 0.01;
            }
        }
        let mut executor = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        let unit_weights = vec![vec![vec![1.0; state.nodes]; 4]];
        assert_eq!(executor.forward_edges, adjacency.indices.len());
        assert_eq!(executor.reverse_edges, adjacency.indices.len());
        assert!(executor.adaptive_edges >= adjacency.indices.len());
        assert_eq!(executor.node_tiles().collect::<Vec<_>>(), vec![0..3]);
        // Parameters, moments, and six CSR buffers were allocated during the
        // one-time construction; no batch activation has been allocated yet.
        assert!(executor.allocation_count() >= 9);
        executor
            .upload_supervised_batch(
                &[&frame.target[..8]],
                &[&frame.target[8..12]],
                &[&unit_weights[0]],
                1,
            )
            .unwrap();
        assert_eq!(
            executor
                .short_input_projection(state.layout(), 1, 8, 3, 1, 4, 4, 0, 1)
                .unwrap(),
            13
        );
        let short = executor
            .short_branch(state.layout(), 1, 8, 1, 4, 4, 0, 1, true, state.steps)
            .unwrap();
        assert!(
            executor.arena.capacity_f32(short).unwrap() >= 12,
            "eight dilated short layers must retain [batch, one step, nodes, hidden]"
        );
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::SUPERVISED_INPUT)
                .unwrap(),
            24
        );
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::SUPERVISED_TARGET)
                .unwrap(),
            12
        );
        executor
            .patch_embedding(state.layout(), 1, 8, 3, 1, 1, 4)
            .unwrap();
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::PATCH_EMBEDDING)
                .unwrap(),
            96
        );
        executor
            .add_patch_positions(state.layout(), 1, 8, 3, 4)
            .unwrap();
        executor.patch_attention_layout(1, 8, 3, 4).unwrap();
        executor
            .frozen_encoder(state.layout(), 1, 8, 3, 4, 2)
            .unwrap();
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::ATTENTION_SEQUENCE)
                .unwrap(),
            96
        );
        let mut cuda_encoder = vec![0.0_f32; 96];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::ATTENTION_SEQUENCE,
                &mut cuda_encoder,
            )
            .unwrap();
        let cpu_encoder =
            state.frozen_lsttn_patch_representations(&frame.target[..8], &adjacency, 0);
        for node in 0..3 {
            for patch in 0..8 {
                for channel in 0..4 {
                    let device = cuda_encoder[(node * 8 + patch) * 4 + channel];
                    let host = cpu_encoder[patch][node][channel];
                    assert!(
                        (device - host).abs() < 2e-4,
                        "encoder mismatch node={node} patch={patch} channel={channel}: {device} vs {host}"
                    );
                }
            }
        }
        executor
            .adaptive_weights(
                layout.lsttn_short_adaptive_source,
                layout.lsttn_short_adaptive_target,
            )
            .unwrap();
        let mut adaptive_weights = vec![0.0_f32; executor.adaptive_edges];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::ADAPTIVE_WEIGHTS,
                &mut adaptive_weights,
            )
            .unwrap();
        let adaptive = adjacency.with_adaptive_self_candidates(3);
        for row in 0..3 {
            let sum = adaptive_weights[adaptive.indptr[row]..adaptive.indptr[row + 1]]
                .iter()
                .sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-5);
        }
        let long = executor.long_branch(layout, 1, 8, 3, 4).unwrap();
        assert!(
            executor.arena.capacity_f32(long).unwrap() >= 12,
            "long branch must retain [batch, final_time, nodes, hidden]"
        );
        executor
            .periodic_feature(
                layout,
                CudaLsttnTensorExecutor::PERIODIC_SHORT,
                1,
                8,
                4,
                1,
                false,
            )
            .unwrap();
        executor
            .periodic_feature(
                layout,
                CudaLsttnTensorExecutor::PERIODIC_SEASONAL,
                1,
                8,
                4,
                2,
                true,
            )
            .unwrap();
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::PERIODIC_SHORT)
                .unwrap(),
            12
        );
        let direct = executor
            .fuse_and_direct_output(layout, long, short, 1, 4, 4)
            .unwrap();
        assert_eq!(executor.arena.capacity_f32(direct).unwrap(), 12);
        let direct = executor.supervised_forward(&state, 1, 1, 0, true).unwrap();
        assert_eq!(executor.arena.capacity_f32(direct).unwrap(), 12);
        executor.direct_head_loss_and_backward(&state, 1).unwrap();
        executor.fusion_backward(&state, 1).unwrap();
        executor
            .short_branch_backward(&state, 1, 1, 0, true)
            .unwrap();
        let mut gradient_after_short = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_after_short,
            )
            .unwrap();
        let short_width =
            3 * state.hidden + 8 * (12 * state.hidden * state.hidden + 6 * state.hidden);
        assert!(
            gradient_after_short[layout.lsttn_short_wave..layout.lsttn_short_wave + short_width]
                .iter()
                .any(|value| value.abs() > 1.0e-8),
            "Graph WaveNet short-branch reverse must accumulate short-wave gradients"
        );
        executor.long_branch_backward(&state, 1).unwrap();
        let mut gradient_after_long = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_after_long,
            )
            .unwrap();
        assert!(
            gradient_after_long[layout.lsttn_dilated_convolution
                ..layout.lsttn_dilated_convolution
                    + 4 * (3 * state.hidden * state.hidden + state.hidden)]
                .iter()
                .any(|value| value.abs() > 1.0e-8),
            "long branch reverse must accumulate dilated-convolution gradients"
        );
        executor
            .periodic_projection_backward(&state, 1, false)
            .unwrap();
        executor
            .periodic_projection_backward(&state, 1, true)
            .unwrap();
        let mut gradient_before_periodic_graph = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_before_periodic_graph,
            )
            .unwrap();
        executor
            .arena
            .deterministic_dropout_f32(
                CudaLsttnTensorExecutor::PERIODIC_SHORT_INPUT,
                CudaLsttnTensorExecutor::PERIODIC_SHORT_INPUT_GRADIENT,
                state.nodes * 7 * state.hidden,
                0,
                0,
                false,
                0.0,
            )
            .unwrap();
        executor
            .arena
            .deterministic_dropout_f32(
                CudaLsttnTensorExecutor::PERIODIC_SEASONAL_INPUT,
                CudaLsttnTensorExecutor::PERIODIC_SEASONAL_INPUT_GRADIENT,
                state.nodes * 7 * state.hidden,
                0,
                0,
                false,
                0.0,
            )
            .unwrap();
        executor
            .periodic_graph_backward(&state, 1, 1, false)
            .unwrap();
        executor
            .periodic_graph_backward(&state, 1, 1, true)
            .unwrap();
        let mut gradient_after_periodic_graph = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_after_periodic_graph,
            )
            .unwrap();
        let adaptive_gradient_changed = [
            layout.lsttn_adaptive_source,
            layout.lsttn_adaptive_target,
            layout.lsttn_weekly_adaptive_source,
            layout.lsttn_weekly_adaptive_target,
        ]
        .into_iter()
        .any(|offset| {
            gradient_after_periodic_graph[offset..offset + state.nodes * state.hidden]
                .iter()
                .zip(&gradient_before_periodic_graph[offset..offset + state.nodes * state.hidden])
                .any(|(after, before)| (after - before).abs() > 1.0e-8)
        });
        assert!(
            adaptive_gradient_changed,
            "periodic graph reverse must backpropagate through adaptive CSR logits"
        );
        let mut loss = [0.0_f32; 2];
        executor
            .arena
            .download_f32(CudaLsttnTensorExecutor::SUPERVISED_LOSS, &mut loss)
            .unwrap();
        assert!(loss[0].is_finite());
        let expected = state
            .parameters
            .iter()
            .map(|value| *value as f32 as f64)
            .collect::<Vec<_>>();
        let mut checkpoint_state = state.clone();
        checkpoint_state.parameters.fill(0.0);
        checkpoint_state.first_moment.fill(0.0);
        checkpoint_state.second_moment.fill(0.0);
        executor
            .synchronize_portable_state(&mut checkpoint_state)
            .unwrap();
        assert_eq!(checkpoint_state.parameters, expected);

        let mut supervised = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        supervised
            .upload_supervised_batch(
                &[&frame.target[..8]],
                &[&frame.target[8..12]],
                &[&unit_weights[0]],
                1,
            )
            .unwrap();
        supervised
            .supervised_forward(&state, 1, 1, 0, true)
            .unwrap();
        supervised
            .supervised_backward(&state, 1, 1, 0, true)
            .unwrap();
        let mut supervised_gradient = vec![0.0_f32; state.parameters.len()];
        supervised
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut supervised_gradient,
            )
            .unwrap();
        assert!(
            supervised_gradient.iter().any(|value| value.abs() > 1.0e-8),
            "consolidated CUDA supervised backward must accumulate model gradients"
        );

        let mut tiled = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        tiled
            .upload_supervised_node_tile(&[&frame.target[..8]], &[&frame.target[8..12]], 1, 1..3)
            .unwrap();
        let mut input_tile = vec![0.0_f32; 16];
        let mut target_tile = vec![0.0_f32; 8];
        tiled
            .arena
            .download_f32(CudaLsttnTensorExecutor::SUPERVISED_INPUT, &mut input_tile)
            .unwrap();
        tiled
            .arena
            .download_f32(CudaLsttnTensorExecutor::SUPERVISED_TARGET, &mut target_tile)
            .unwrap();
        let expected_input = frame.target[..8]
            .iter()
            .flat_map(|row| row[1..3].iter().map(|value| *value as f32))
            .collect::<Vec<_>>();
        let expected_target = frame.target[8..12]
            .iter()
            .flat_map(|row| row[1..3].iter().map(|value| *value as f32))
            .collect::<Vec<_>>();
        assert_eq!(input_tile, expected_input);
        assert_eq!(target_tile, expected_target);
    }

    #[test]
    fn hip_alias_uses_the_shared_rocm_backend_contract() {
        match select_compute_backend(Some("hip")) {
            Ok(selection) => {
                assert_eq!(selection.requested, "rocm");
                assert_eq!(selection.selected, "rocm");
            }
            Err(error) => assert!(error.to_string().contains("not available")),
        }
    }

    #[test]
    fn paper_graph_transformer_profiles_fit_predict_and_round_trip() {
        let frame = traffic_style_fixture_frame();
        for profile in [
            GraphTransformerProfile::HeterogeneousMoE,
            GraphTransformerProfile::EfficientHighOrder,
            GraphTransformerProfile::LongShortFusion,
            GraphTransformerProfile::GatedGraphTemporal,
            GraphTransformerProfile::SpatialShiftGraphonMoE,
        ] {
            let mut model = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
                profile: profile.clone(),
                lookback: 8,
                hidden_size: 8,
                attention_heads: 2,
                graph_order: 2,
                experts: 3,
                periodicity: if profile == GraphTransformerProfile::LongShortFusion {
                    1
                } else {
                    6
                },
                recent_window: 4,
                epochs: 8,
                learning_rate: 0.01,
                weight_decay: 0.0,
                batch_size: 32,
                backend: select_compute_backend(Some("cpu")).unwrap(),
            })
            .unwrap();
            model.fit(&frame).unwrap();
            let prediction = model.predict(4).unwrap();
            assert_eq!(prediction.len(), 4);
            assert!(prediction.iter().flatten().all(|value| value.is_finite()));
            let restored =
                PaperGraphTransformerForecaster::from_json_string(&model.to_json_string().unwrap())
                    .unwrap();
            for (actual, expected) in restored.predict(4).unwrap().iter().zip(&prediction) {
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            if profile == GraphTransformerProfile::SpatialShiftGraphonMoE {
                assert_eq!(model.architecture_report().graphon_expert_count, 3);
            }
            if profile == GraphTransformerProfile::LongShortFusion {
                let supervised_examples = 48usize - 8 - 4 + 1;
                let supervised_steps = supervised_examples.div_ceil(32) * 8;
                let pretraining_windows = (48usize - 8).div_ceil(8) + 1;
                let pretraining_steps = pretraining_windows * (8 / 4);
                assert_eq!(
                    model.trainable_state.as_ref().unwrap().steps as usize,
                    supervised_steps + pretraining_steps
                );
            }
            let report = model.architecture_report();
            let required_component = match profile {
                GraphTransformerProfile::HeterogeneousMoE => "moe_load_balancing_loss",
                GraphTransformerProfile::EfficientHighOrder => {
                    "recursive_pointwise_high_order_interaction"
                }
                GraphTransformerProfile::LongShortFusion => {
                    "seventy_five_percent_masked_patch_pretraining"
                }
                GraphTransformerProfile::GatedGraphTemporal => "normalized_graph_convolution",
                GraphTransformerProfile::SpatialShiftGraphonMoE => {
                    "maximum_spatiotemporal_graph_division"
                }
            };
            assert!(report
                .components
                .iter()
                .any(|component| component == required_component));
        }
    }

    #[test]
    fn lsttn_checkpoint_is_resumable_and_fingerprint_guarded() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 2,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("lsttn-checkpoint.json");
        let mut first = PaperGraphTransformerForecaster::new(config.clone()).unwrap();
        first.fit_checkpointed(&frame, &checkpoint).unwrap();
        let first_prediction = first.predict(4).unwrap();
        let first_steps = first.trainable_state.as_ref().unwrap().steps;

        let mut resumed = PaperGraphTransformerForecaster::new(config).unwrap();
        resumed.fit_checkpointed(&frame, &checkpoint).unwrap();
        assert_eq!(resumed.trainable_state.as_ref().unwrap().steps, first_steps);
        assert_eq!(resumed.predict(4).unwrap(), first_prediction);

        let mut changed = frame.clone();
        changed.target[0][0] += 1.0;
        assert!(resumed.fit_checkpointed(&changed, checkpoint).is_err());
    }

    #[test]
    fn lsttn_shard_pair_manifest_is_atomic_and_round_trips_state() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let expected = model.predict(2).unwrap();
        let expected_state = model.trainable_state.as_ref().unwrap().clone();
        let directory = tempfile::tempdir().unwrap();
        let local = directory.path().join("local.json");
        let shared = directory.path().join("shared.json");
        let manifest = directory.path().join("pair.json");
        model.save_shard_pair(&local, &shared, &manifest).unwrap();
        let restored =
            PaperGraphTransformerForecaster::load_shard_pair(&local, &shared, &manifest).unwrap();
        assert_eq!(restored.predict(2).unwrap(), expected);
        let restored_state = restored.trainable_state.as_ref().unwrap();
        assert_eq!(restored_state.steps, expected_state.steps);
        assert_eq!(restored_state.shared_steps, expected_state.shared_steps);
        assert_eq!(restored_state.local_steps, expected_state.local_steps);

        fs::write(&local, b"interrupted local write").unwrap();
        assert!(
            PaperGraphTransformerForecaster::load_shard_pair(&local, &shared, &manifest).is_err()
        );
    }

    #[test]
    fn lsttn_shared_state_moves_only_node_independent_segments() {
        let mut source = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 41);
        for (index, value) in source.parameters.iter_mut().enumerate() {
            *value = index as f64 / 1000.0;
        }
        source.steps = 17;
        source.shared_steps = 17;
        let shared = source.export_shared(1.25, 0.5);
        let mut target = TrainableGraphTransformerState::initialized(5, 4, 2, 1, 4, 8, 4, 2, 2, 99);
        let local_before = target.parameters.clone();
        let local_ranges = {
            let layout = target.layout();
            vec![
                layout.in_degree_embedding..layout.temporal_q,
                layout.shortest_path_bias..layout.router,
                layout.lsttn_adaptive_source..layout.lsttn_periodic_projection,
                layout.graphon_nodes..layout.graphon_time,
            ]
        };
        target.import_shared(&shared).unwrap();
        assert_eq!(target.steps, 17);
        assert_eq!(target.shared_steps, 17);
        for (_, range) in target.shared_ranges() {
            assert_ne!(target.parameters[range.clone()], local_before[range]);
        }
        for range in local_ranges {
            assert_eq!(target.parameters[range.clone()], local_before[range]);
        }
        let source_shared_bytes = shared
            .segments
            .iter()
            .map(|segment| {
                (segment.parameters.len()
                    + segment.first_moment.len()
                    + segment.second_moment.len())
                    * std::mem::size_of::<f64>()
            })
            .sum::<usize>();
        let target_shared_bytes = target
            .export_shared(1.25, 0.5)
            .segments
            .iter()
            .map(|segment| {
                (segment.parameters.len()
                    + segment.first_moment.len()
                    + segment.second_moment.len())
                    * std::mem::size_of::<f64>()
            })
            .sum::<usize>();
        assert_eq!(source_shared_bytes, target_shared_bytes);
        let source_local_bytes = source
            .local_ranges()
            .into_iter()
            .map(|range| range.len() * 3 * std::mem::size_of::<f64>())
            .sum::<usize>();
        let target_local_bytes = target
            .local_ranges()
            .into_iter()
            .map(|range| range.len() * 3 * std::mem::size_of::<f64>())
            .sum::<usize>();
        assert!(target_local_bytes > source_local_bytes);
    }

    #[test]
    fn lsttn_shared_round_reducer_is_order_independent_and_rejects_wrong_base() {
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 41);
        let mut left = state.export_shared(0.0, 1.0);
        let mut right = state.export_shared(0.0, 1.0);
        left.segments[0].parameters[0] = 2.0;
        right.segments[0].parameters[0] = 8.0;
        let round = |shard_id: &str, objective_weight: f64, shared: LsttnSharedState| {
            serde_json::to_string(&LsttnSharedRound {
                version: 1,
                shard_id: shard_id.to_string(),
                objective_weight,
                base_shared_hash: 77,
                shared,
            })
            .unwrap()
        };
        let left = round("left", 1.0, left);
        let right = round("right", 3.0, right);
        let forward = PaperGraphTransformerForecaster::reduce_shard_rounds(
            &[left.clone(), right.clone()],
            77,
        )
        .unwrap();
        let reverse =
            PaperGraphTransformerForecaster::reduce_shard_rounds(&[right, left], 77).unwrap();
        assert_eq!(forward, reverse);
        let reduced: LsttnSharedState = serde_json::from_str(&forward).unwrap();
        assert!((reduced.segments[0].parameters[0] - 6.5).abs() < 1e-12);
        assert!(PaperGraphTransformerForecaster::reduce_shard_rounds(&[forward], 77).is_err());
    }

    #[test]
    fn lsttn_warm_start_keeps_parameters_but_resets_cutoff_bound_state() {
        let identity = serde_json::json!({
            "backbone_id": "backbone", "equipment_category": "objective", "target_unit": "unit",
            "node_registry_fingerprint": "nodes", "graph_cutoff": "2026-01-01",
            "shard_policy_fingerprint": "policy", "epoch": 0,
        });
        let policy = stable_fingerprint(&serde_json::json!({
            "equipment_category": "objective", "target_unit": "unit", "node_registry": "nodes", "shard_policy": "policy",
        })).unwrap();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 41);
        state.parameters[0] = 9.0;
        state.first_moment.fill(2.0);
        state.second_moment.fill(3.0);
        state.steps = 7;
        state.shared_steps = 6;
        state.local_steps = 5;
        state.shard_data_fingerprint = 99;
        state.warm_start_policy_fingerprint = policy;
        let mut model =
            PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig::default()).unwrap();
        model.trainable_state = Some(state);
        model
            .prepare_shard_warm_start(&identity.to_string())
            .unwrap();
        let reset = model.trainable_state.unwrap();
        assert_eq!(reset.parameters[0], 9.0);
        assert!(reset.first_moment.iter().all(|value| *value == 0.0));
        assert!(reset.second_moment.iter().all(|value| *value == 0.0));
        assert_eq!(
            (
                reset.steps,
                reset.shared_steps,
                reset.local_steps,
                reset.shard_data_fingerprint
            ),
            (0, 0, 0, 0)
        );
        let incompatible = serde_json::json!({
            "backbone_id": "backbone", "equipment_category": "other", "target_unit": "unit",
            "node_registry_fingerprint": "nodes", "graph_cutoff": "2026-01-08",
            "shard_policy_fingerprint": "policy", "epoch": 0,
        });
        let mut rejected =
            PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig::default()).unwrap();
        rejected.trainable_state = Some(reset);
        assert!(rejected
            .prepare_shard_warm_start(&incompatible.to_string())
            .is_err());
    }

    #[test]
    fn lsttn_local_adaptation_freezes_shared_adam_state() {
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 41);
        state.shard_training = true;
        let gradients = vec![0.25; state.parameters.len()];
        state.adamw_step(&gradients, 0.01, 0.00001, None).unwrap();
        assert_eq!(state.shared_steps, 1);
        assert_eq!(state.local_steps, 1);
        let shared_before = state
            .shared_ranges()
            .into_iter()
            .map(|(_, range)| {
                (
                    state.parameters[range.clone()].to_vec(),
                    state.first_moment[range.clone()].to_vec(),
                    state.second_moment[range].to_vec(),
                )
            })
            .collect::<Vec<_>>();
        let local_before = state.local_ranges()[0].clone();
        let local_parameters = state.parameters[local_before.clone()].to_vec();
        state.freeze_shared = true;
        state.adamw_step(&gradients, 0.01, 0.00001, None).unwrap();
        assert_eq!(state.shared_steps, 1);
        assert_eq!(state.local_steps, 2);
        for ((parameters, first, second), (_, range)) in
            shared_before.iter().zip(state.shared_ranges())
        {
            assert_eq!(&state.parameters[range.clone()], parameters);
            assert_eq!(&state.first_moment[range.clone()], first);
            assert_eq!(&state.second_moment[range], second);
        }
        assert_ne!(state.parameters[local_before], local_parameters);
    }

    #[test]
    fn graph_temporal_owner_mask_is_validated() {
        let frame = traffic_style_fixture_frame();
        assert!(frame
            .clone()
            .with_owner_mask(Some(vec![true, false, false]))
            .is_ok());
        assert!(frame.clone().with_owner_mask(Some(vec![false; 3])).is_err());
        assert!(frame.with_owner_mask(Some(vec![true])).is_err());
    }

    #[test]
    fn graph_temporal_target_mask_distinguishes_missing_transport_values() {
        let base = traffic_style_fixture_frame();
        let mut target = base.target.clone();
        target[0][0] = f64::NAN;
        assert!(GraphTemporalFrame::new_with_training_metadata(
            base.node_ids.clone(),
            base.timestamps.clone(),
            target.clone(),
            base.covariates.clone(),
            base.adjacency.clone(),
            base.horizon,
            base.frequency.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .is_err());
        let mut mask = vec![vec![true; base.node_ids.len()]; target.len()];
        mask[0][0] = false;
        assert!(GraphTemporalFrame::new_with_training_metadata(
            base.node_ids,
            base.timestamps,
            target,
            base.covariates,
            base.adjacency,
            base.horizon,
            base.frequency,
            None,
            Some(mask),
            None,
            None,
            None,
        )
        .is_ok());
    }

    #[test]
    fn observed_target_normalization_ignores_masked_and_imputed_transport_values() {
        let target = vec![vec![0.0, 10.0], vec![2.0, 5000.0]];
        let target_mask = vec![vec![true, true], vec![true, false]];
        let imputed_mask = vec![vec![false, false], vec![false, true]];
        let (mean, scale) =
            target_center_scale_observed(&target, Some(&target_mask), Some(&imputed_mask));
        assert!((mean - 4.0).abs() < 1e-12);
        assert!(scale.is_finite() && scale < 5.0);
    }

    #[test]
    fn lsttn_native_conformal_intervals_are_noncrossing_and_horizon_specific() {
        let actual = vec![
            vec![vec![1.0, 10.0], vec![2.0, 20.0]],
            vec![vec![3.0, 30.0], vec![4.0, 40.0]],
        ];
        let median = vec![
            vec![vec![0.0, 11.0], vec![2.5, 18.0]],
            vec![vec![4.0, 28.0], vec![3.0, 43.0]],
        ];
        let forecast = vec![vec![5.0, 50.0], vec![6.0, 60.0]];
        let intervals = lsttn_conformal_forecast(&actual, &median, &forecast, 0.25).unwrap();
        assert_eq!(intervals.median, forecast);
        assert_eq!(intervals.calibration_count_by_horizon, vec![4, 4]);
        assert_ne!(
            intervals.radius_by_horizon[0],
            intervals.radius_by_horizon[1]
        );
        for ((lower, median), upper) in intervals
            .lower
            .iter()
            .zip(&intervals.median)
            .zip(&intervals.upper)
        {
            for ((lower, median), upper) in lower.iter().zip(median).zip(upper) {
                assert!(lower <= median && median <= upper);
            }
        }
    }

    #[test]
    fn lsttn_target_weights_normalize_and_masks_dominate_loss_gradients() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 91);
        let targets = &frame.target[8..12];
        let (_, unweighted_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            targets,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let uniform = vec![vec![1.0; 3]; 4];
        let (uniform_loss, uniform_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            targets,
            0,
            None,
            None,
            None,
            None,
            Some(&uniform),
            None,
        );
        let (unweighted_loss, _) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            targets,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!((uniform_loss - unweighted_loss).abs() < 1e-12);
        assert_eq!(uniform_gradients, unweighted_gradients);

        let mut changed = targets.to_vec();
        changed[0][0] += 1000.0;
        let mut zero_weight = uniform.clone();
        zero_weight[0][0] = 0.0;
        let (zero_loss, zero_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            targets,
            0,
            None,
            None,
            None,
            None,
            Some(&zero_weight),
            None,
        );
        let (changed_zero_loss, changed_zero_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &changed,
            0,
            None,
            None,
            None,
            None,
            Some(&zero_weight),
            None,
        );
        assert!((zero_loss - changed_zero_loss).abs() < 1e-12);
        assert_eq!(zero_gradients, changed_zero_gradients);

        let mut masks = vec![vec![true; 3]; 4];
        masks[0][1] = false;
        let mut masked_changed = targets.to_vec();
        masked_changed[0][1] += 1000.0;
        let mut high_weight = uniform;
        high_weight[0][1] = 1_000_000.0;
        let (masked_loss, masked_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            targets,
            0,
            None,
            None,
            Some(&masks),
            None,
            Some(&high_weight),
            Some(&[true, true, false]),
        );
        let (masked_changed_loss, masked_changed_gradients) = state
            .lsttn_example_loss_and_gradients(
                &frame.target[..8],
                &adjacency,
                &masked_changed,
                0,
                None,
                None,
                Some(&masks),
                None,
                Some(&high_weight),
                Some(&[true, true, false]),
            );
        assert!((masked_loss - masked_changed_loss).abs() < 1e-12);
        assert_eq!(masked_gradients, masked_changed_gradients);
    }

    #[test]
    fn frozen_lsttn_patch_cache_preserves_trainable_gradients() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 8, 4, 2, 2, 41);
        let cache = state.frozen_lsttn_patch_representations(&frame.target[..8], &adjacency, 0);
        let (uncached_loss, uncached) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let (cached_loss, cached) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            Some(&cache),
            None,
            None,
            None,
            None,
            None,
        );
        assert!((cached_loss - uncached_loss).abs() < 1e-5);
        let layout = state.layout();
        for (actual, expected) in cached[layout.spatial_q..layout.pretrain_position]
            .iter()
            .zip(&uncached[layout.spatial_q..layout.pretrain_position])
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn graph_transformer_tape_backpropagates_through_attention_parameters() {
        let tape = AutodiffTape::new();
        let parameter = tape.parameter(0, 3.0);
        let squared = tape.mul(parameter, parameter);
        let gradients = tape.backward(squared, 1);
        assert!((gradients[0] - 6.0).abs() < 1e-12);

        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
            ..PaperGraphTransformerConfig::default()
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for range in [
            layout.temporal_q..layout.temporal_k,
            layout.spatial_q..layout.spatial_k,
            layout.router..layout.expert_heads,
        ] {
            assert!(trained.parameters[range.clone()]
                .iter()
                .zip(&initial.parameters[range])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn attention_projection_ranges_include_independent_biases() {
        let layout = GraphParameterLayout::new(3, 4, 2, 2, 1, 6, 8);
        let width = 4 * (4 + 1);
        let block_width = 4 * width;
        assert_eq!(layout.temporal_k, layout.temporal_q + block_width);
        assert_eq!(layout.temporal_v, layout.temporal_k + block_width);
        assert_eq!(layout.spatial_q, layout.temporal_v + block_width);
        assert_eq!(layout.spatial_k, layout.spatial_q + block_width);
        assert_eq!(layout.spatial_v, layout.spatial_k + block_width);
        assert_eq!(layout.shortest_path_bias, layout.spatial_v + block_width);
        assert_eq!((layout.pretrain_decoder - layout.pretrain_position) / 4, 8);
    }

    #[test]
    fn periodic_phase_is_stable_across_sliding_window_origins() {
        let absolute_step = 37;
        let first_window = periodic_phase(12 + (absolute_step - 12), 24);
        let second_window = periodic_phase(29 + (absolute_step - 29), 24);

        assert!((first_window - second_window).abs() < 1e-12);
    }

    #[test]
    fn efficient_high_order_profile_uses_stgformer_scaling_normalized_attention() {
        let frame = traffic_style_fixture_frame();
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 8, 2, 2, 2, 19);
        let adjacency = frame.adjacency.row_normalized();
        let window = &frame.target[..8];
        let linear = state.predict_window(
            &GraphTransformerProfile::EfficientHighOrder,
            window,
            &adjacency,
        );
        let softmax = state.predict_window(
            &GraphTransformerProfile::HeterogeneousMoE,
            window,
            &adjacency,
        );
        assert!(linear.iter().flatten().all(|value| value.is_finite()));
        assert!(softmax.iter().flatten().all(|value| value.is_finite()));
        assert!(linear
            .iter()
            .flatten()
            .zip(softmax.iter().flatten())
            .any(|(left, right)| (left - right).abs() > 1e-12));
    }

    #[test]
    fn stgformer_trains_each_order_pointwise_interaction() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::EfficientHighOrder,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for order in 0..2 {
            let start = layout.stgformer_pointwise + order * 4 * 5;
            assert!(trained.parameters[start..start + 20]
                .iter()
                .zip(&initial.parameters[start..start + 20])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn stgformer_fast_attention_reuses_summary_and_keeps_query_value_residual() {
        let tape = AutodiffTape::new();
        let keys = vec![
            vec![tape.constant(1.0), tape.constant(0.0)],
            vec![tape.constant(0.0), tape.constant(1.0)],
        ];
        let values = vec![
            vec![tape.constant(2.0), tape.constant(3.0)],
            vec![tape.constant(5.0), tape.constant(7.0)],
        ];
        let summary = tape_stgformer_attention_summary(&tape, &keys, &values);
        let output = tape_stgformer_fast_attention(
            &tape,
            &[tape.constant(1.0), tape.constant(0.0)],
            &summary,
            &values[1],
        );
        // Q(K^T V) + N * V_query over Q(sum K) + N, for N = 2.
        assert!((tape.value(output[0]) - 4.0).abs() < 1e-12);
        assert!((tape.value(output[1]) - 17.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn attention_and_router_softmax_preserve_true_logit_ratios() {
        let tape = AutodiffTape::new();
        let weights = tape_softmax(&tape, &[tape.constant(0.0), tape.constant(2.0)]);
        let ratio = tape.value(weights[1]) / tape.value(weights[0]);
        assert!((ratio - 2.0_f64.exp()).abs() < 1e-12);
    }

    #[test]
    fn training_tape_interns_each_model_parameter_once() {
        let tape = AutodiffTape::new();
        let parameters = [0.25, -0.5];
        let parameter_nodes = parameters
            .iter()
            .enumerate()
            .map(|(index, value)| tape.parameter(index, *value))
            .collect::<Vec<_>>();

        let first_use = tape.mul(parameter_nodes[0], tape.constant(2.0));
        let second_use = tape.add(parameter_nodes[0], parameter_nodes[1]);
        let loss = tape.add(first_use, second_use);
        let gradients = tape.backward(loss, parameters.len());

        assert_eq!(parameter_nodes.len(), parameters.len());
        assert_eq!(tape.nodes.borrow().len(), 6);
        assert_eq!(gradients, vec![3.0, 1.0]);
    }

    #[test]
    fn adaptive_diffusion_uses_csr_neighbors_with_isolated_node_fallback() {
        let adjacency =
            CsrAdjacency::new(vec![0, 2, 3, 3], vec![1, 2, 0], vec![1.0; 3], 3).unwrap();
        let first_fallback = [0];
        let isolated_fallback = [2];

        assert_eq!(
            adaptive_neighbor_indices(&adjacency, 0, &first_fallback),
            &[1, 2]
        );
        assert_eq!(
            adaptive_neighbor_indices(&adjacency, 2, &isolated_fallback),
            &[2]
        );

        let adaptive = adjacency.with_adaptive_self_candidates(3);
        assert_eq!(adaptive.indptr, vec![0, 3, 5, 6]);
        assert_eq!(adaptive.indices, vec![1, 2, 0, 0, 1, 2]);
        // The learned adaptive support remains O(E + N), not N².
        assert_eq!(adaptive.indices.len(), adjacency.indices.len() + 3);
    }

    #[test]
    fn maximum_graph_division_groups_contiguous_rank_stable_environments() {
        let values = [
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 2.0, 1.0],
            vec![6.0, 4.0, 2.0],
        ]
        .into_iter()
        .cycle()
        .take(16)
        .collect::<Vec<_>>();
        assert_eq!(
            maximum_spatiotemporal_graph_division(&values, 4, 2).unwrap(),
            vec![0, 0, 1, 1]
        );
    }

    #[test]
    fn maximum_graph_division_rejects_an_incomplete_cycle() {
        let error = maximum_spatiotemporal_graph_division(&[vec![1.0, 2.0]], 2, 2)
            .expect_err("one observation cannot define a two-step graph cycle");
        assert!(error.to_string().contains("complete period"));
    }

    #[test]
    fn episodic_graphon_detach_stops_expert_gradient() {
        let tape = AutodiffTape::new();
        let expert = tape.parameter(0, 3.0);
        let detached = tape_detach_vectors(&tape, &[expert]);
        let loss = tape.mul(detached[0], detached[0]);
        assert_eq!(tape.backward(loss, 1), vec![0.0]);
    }

    #[test]
    fn stgormer_trains_degree_and_shortest_path_embedding_tables() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::HeterogeneousMoE,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for range in [
            layout.in_degree_embedding..layout.out_degree_embedding,
            layout.temporal_q..layout.temporal_k,
            layout.temporal_k..layout.temporal_v,
            layout.temporal_v..layout.spatial_q,
            layout.spatial_q..layout.spatial_k,
            layout.spatial_k..layout.spatial_v,
            layout.spatial_v..layout.shortest_path_bias,
            layout.shortest_path_bias..layout.router,
            layout.router..layout.spatial_router,
            layout.spatial_router..layout.spatial_expert_heads,
            layout.spatial_expert_heads..layout.expert_heads,
        ] {
            assert!(trained.parameters[range.clone()]
                .iter()
                .zip(&initial.parameters[range])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn long_short_profile_trains_dynamic_periodic_graph_parameters() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for range in [
            layout.lsttn_adaptive_source..layout.lsttn_adaptive_target,
            layout.lsttn_adaptive_target..layout.lsttn_weekly_adaptive_source,
            layout.lsttn_weekly_adaptive_source..layout.lsttn_weekly_adaptive_target,
            layout.lsttn_weekly_adaptive_target..layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_source..layout.lsttn_short_adaptive_target,
            layout.lsttn_short_adaptive_target..layout.lsttn_periodic_projection,
        ] {
            assert!(trained.parameters[range.clone()]
                .iter()
                .zip(&initial.parameters[range])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
        assert!(trained.parameters[layout.pretrain_mask_token..layout.total]
            .iter()
            .zip(&initial.parameters[layout.pretrain_mask_token..layout.total])
            .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        assert!(
            trained.parameters[layout.lsttn_fusion..layout.graphon_nodes]
                .iter()
                .zip(&initial.parameters[layout.lsttn_fusion..layout.graphon_nodes])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
        assert!(
            trained.parameters[layout.lsttn_dilated_convolution..layout.stgformer_pointwise]
                .iter()
                .zip(
                    &initial.parameters
                        [layout.lsttn_dilated_convolution..layout.stgformer_pointwise],
                )
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
        assert!(
            trained.parameters[layout.lsttn_short_wave..layout.stgformer_pointwise]
                .iter()
                .zip(&initial.parameters[layout.lsttn_short_wave..layout.stgformer_pointwise],)
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
    }

    #[test]
    fn long_short_profile_rejects_recent_context_larger_than_history() {
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 24,
            recent_window: 25,
            ..PaperGraphTransformerConfig::default()
        };

        assert!(PaperGraphTransformerForecaster::new(config).is_err());
    }

    #[test]
    fn long_short_fit_requires_a_real_weekly_transformer_state() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            periodicity: 2,
            recent_window: 4,
            epochs: 1,
            ..PaperGraphTransformerConfig::default()
        };
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        let error = model.fit(&frame).unwrap_err();
        assert!(error.to_string().contains("exceed its seasonal lag"));
    }

    #[test]
    fn long_short_report_exposes_every_paper_spatial_temporal_branch() {
        let model = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 24 * 14,
            periodicity: 24,
            recent_window: 12,
            ..PaperGraphTransformerConfig::default()
        })
        .unwrap();
        let report = model.architecture_report();
        for component in [
            "seventy_five_percent_masked_patch_pretraining",
            "four_layer_multihead_transformer_encoder",
            "four_stage_dilated_long_trend_convolution",
            "previous_day_and_week_transformer_states",
            "independent_forward_backward_adaptive_periodic_graph_convolutions",
            "eight_layer_causal_graph_wavenet_short_branch",
            "signal_and_time_of_day_short_term_channels",
            "long_periodic_short_feature_fusion",
            "all_origin_thirty_two_window_supervision",
        ] {
            assert!(report.components.iter().any(|actual| actual == component));
        }
    }

    #[test]
    fn long_short_graph_wavenet_consumes_supplied_time_of_day_channel() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let state = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            1,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let first = vec![vec![0.0; 3]; 8];
        let second = (0..8)
            .map(|time| vec![time as f64 / 8.0; 3])
            .collect::<Vec<_>>();
        let (first_loss, _) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            Some(&first),
            None,
            None,
            None,
            None,
        );
        let (second_loss, second_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            Some(&second),
            None,
            None,
            None,
            None,
        );
        let layout = state.layout();
        assert!(second_gradients
            [layout.lsttn_short_wave + state.hidden..layout.lsttn_short_wave + 2 * state.hidden]
            .iter()
            .any(|gradient| gradient.abs() > 0.0));
        assert!(first_loss.is_finite() && second_loss.is_finite());
    }

    #[test]
    fn long_short_supervised_path_freezes_pretrained_context_encoder() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 8, 4, 2, 2, 29);
        let initial = state.parameters.clone();
        let layout = state.layout();

        state
            .train_example_with_context(
                &GraphTransformerProfile::LongShortFusion,
                &frame.target[..8],
                &adjacency,
                &frame.target[8..12],
                None,
                0.01,
                0.0,
                0,
                false,
                None,
            )
            .unwrap();

        for range in [
            layout.temporal_q..layout.temporal_k,
            layout.temporal_k..layout.temporal_v,
            layout.temporal_v..layout.spatial_q,
            layout.pretrain_position..layout.pretrain_decoder,
        ] {
            assert!(state.parameters[range.clone()]
                .iter()
                .zip(&initial[range])
                .all(|(trained, initial)| (trained - initial).abs() < 1e-12));
        }
    }

    #[test]
    fn long_short_direct_horizon_decoder_does_not_rebase_predictions() {
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 2,
            hidden_size: 1,
            attention_heads: 1,
            graph_order: 1,
            experts: 1,
            periodicity: 1,
            recent_window: 2,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let mut state = TrainableGraphTransformerState::initialized(1, 1, 1, 1, 2, 2, 3, 1, 1, 31);
        state.parameters.fill(0.0);
        let layout = state.layout();
        for (horizon, delta) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            state.parameters[layout.output + horizon * 2 + 1] = delta;
        }
        let model = PaperGraphTransformerForecaster {
            config,
            node_ids: vec!["lane".into()],
            frequency: "hourly".into(),
            horizon: 3,
            adjacency: Some(CsrAdjacency::new(vec![0, 0], vec![], vec![], 1).unwrap()),
            trainable_state: Some(state),
            history: vec![vec![9.0], vec![10.0]],
            history_time_features: None,
            covariate_roles: None,
            owner_mask: vec![true],
            fit_activation_tape_bytes: 0,
            fit_frozen_patch_cache_bytes: 0,
            target_mean: 0.0,
            target_scale: 1.0,
        };

        assert_eq!(
            model.predict(3).unwrap(),
            vec![vec![1.0], vec![2.0], vec![3.0]]
        );
        let mut artifact: serde_json::Value =
            serde_json::from_str(&model.to_json_string().unwrap()).unwrap();
        artifact["trainable_state"]["parameters"]
            .as_array_mut()
            .unwrap()
            .pop();
        let error = PaperGraphTransformerForecaster::from_json_string(
            &serde_json::to_string(&artifact).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("incompatible"));
    }

    #[test]
    fn lsttn_dense_attention_matches_cpu_across_available_backends() {
        let hidden = 32;
        let projection = hidden * (hidden + 1);
        let q_offset = 0;
        let k_offset = q_offset + projection;
        let v_offset = k_offset + projection;
        let out_offset = v_offset + projection;
        let ffn_offset = out_offset + projection;
        let ffn_width = (hidden + 1) * 4 * hidden + (4 * hidden + 1) * hidden;
        let norm_offset = ffn_offset + ffn_width;
        let mut parameters = (0..norm_offset + 4 * hidden)
            .map(|index| ((index as f64 * 0.013).sin()) * 0.03)
            .collect::<Vec<_>>();
        parameters[norm_offset..norm_offset + hidden].fill(1.0);
        parameters[norm_offset + 2 * hidden..norm_offset + 3 * hidden].fill(1.0);
        let sequence = (0..64)
            .map(|row| {
                (0..hidden)
                    .map(|column| ((row * hidden + column) as f64 * 0.017).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = numeric_transformer_encoder_layer(
            &parameters,
            &sequence,
            q_offset,
            k_offset,
            v_offset,
            out_offset,
            ffn_offset,
            norm_offset,
            hidden,
            4,
            None,
        )
        .unwrap();

        for backend in available_compute_backends()
            .into_iter()
            .filter(|backend| backend != "cpu")
        {
            let selection =
                cartoboost_neural::select_backend_for(Some(&backend), BackendOperation::Dense)
                    .unwrap();
            let actual = numeric_transformer_encoder_layer(
                &parameters,
                &sequence,
                q_offset,
                k_offset,
                v_offset,
                out_offset,
                ffn_offset,
                norm_offset,
                hidden,
                4,
                Some(&selection),
            )
            .unwrap_or_else(|error| panic!("{backend} LSTTN dense projection failed: {error}"));
            for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
                assert!(
                    (actual - expected).abs() < 2.0e-3,
                    "{backend}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn lsttn_adamw_matches_cpu_across_available_backends() {
        let initial =
            TrainableGraphTransformerState::initialized(8, 32, 4, 24, 8, 96, 4, 4, 2, 0xada0);
        assert!(initial.parameters.len() >= GEO_ST_ADAMW_DISPATCH_MIN_VALUES);
        let gradients = (0..initial.parameters.len())
            .map(|index| (index as f64 * 0.019).sin() * 0.01)
            .collect::<Vec<_>>();
        let mut expected = initial.clone();
        expected
            .adamw_step(&gradients, 0.001, 1.0e-5, None)
            .unwrap();

        for backend in available_compute_backends()
            .into_iter()
            .filter(|backend| backend != "cpu")
        {
            let selection =
                cartoboost_neural::select_backend_for(Some(&backend), BackendOperation::AdamW)
                    .unwrap();
            let mut actual = initial.clone();
            actual
                .adamw_step(&gradients, 0.001, 1.0e-5, Some(&selection))
                .unwrap_or_else(|error| panic!("{backend} LSTTN AdamW failed: {error}"));
            assert_eq!(actual.steps, 1);
            for (actual, expected) in actual.parameters.iter().zip(&expected.parameters) {
                assert!(
                    (actual - expected).abs() < 2.0e-5,
                    "{backend}: {actual} != {expected}"
                );
            }
            for (actual, expected) in actual.first_moment.iter().zip(&expected.first_moment) {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            for (actual, expected) in actual.second_moment.iter().zip(&expected.second_moment) {
                assert!((actual - expected).abs() < 1.0e-7);
            }
        }
    }

    #[test]
    fn lsttn_layer_norm_matches_cpu_across_available_backends() {
        let rows = 128;
        let width = 128;
        let mut parameters = vec![0.0; 2 * width];
        for channel in 0..width {
            parameters[channel] = 0.75 + channel as f64 * 0.001;
            parameters[width + channel] = (channel as f64 * 0.023).sin() * 0.01;
        }
        let values = (0..rows)
            .map(|row| {
                (0..width)
                    .map(|column| ((row * width + column) as f64 * 0.007).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = numeric_layer_norm_batch(&parameters, 0, &values, None).unwrap();

        for backend in available_compute_backends()
            .into_iter()
            .filter(|backend| backend != "cpu")
        {
            let selection =
                cartoboost_neural::select_backend_for(Some(&backend), BackendOperation::LayerNorm)
                    .unwrap();
            let actual = numeric_layer_norm_batch(&parameters, 0, &values, Some(&selection))
                .unwrap_or_else(|error| panic!("{backend} LSTTN layer norm failed: {error}"));
            for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
                assert!(
                    (actual - expected).abs() < 2.0e-4,
                    "{backend}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn long_short_full_metal_inference_matches_cpu_predictions() {
        let frame = traffic_style_fixture_frame();
        let mut cpu = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        })
        .unwrap();
        cpu.fit(&frame).unwrap();
        let expected = cpu.predict(4).unwrap();

        let mut metal = cpu.clone();
        metal.config.backend = select_compute_backend(Some("metal")).unwrap();
        let actual = metal.predict(4).unwrap();

        for (actual_row, expected_row) in actual.iter().zip(expected) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-4);
            }
        }
        assert!(metal
            .architecture_report()
            .components
            .iter()
            .any(|component| component == "metal_full_graph_training_and_inference"));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn long_short_metal_trains_and_predicts_complete_graph() {
        let mut frame = traffic_style_fixture_frame();
        frame.timestamps.truncate(13);
        frame.target.truncate(13);
        let config = |backend| PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some(backend)).unwrap(),
        };
        let mut cpu = PaperGraphTransformerForecaster::new(config("cpu")).unwrap();
        let mut metal = PaperGraphTransformerForecaster::new(config("metal")).unwrap();
        cpu.fit(&frame).unwrap();
        metal.fit(&frame).unwrap();
        let expected = cpu.predict(4).unwrap();
        let actual = metal.predict(4).unwrap();

        for (actual_row, expected_row) in actual.iter().zip(expected) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-1, "{actual} != {expected}");
            }
        }
        let state = metal.trainable_state.as_ref().unwrap();
        assert!(state.steps > 0);
        assert!(state.first_moment.iter().any(|value| value.abs() > 0.0));
        assert!(metal
            .architecture_report()
            .components
            .iter()
            .any(|component| component == "metal_full_graph_training_and_inference"));
    }

    #[test]
    fn spatial_shift_profile_trains_each_graphon_expert_before_router_mixup() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::SpatialShiftGraphonMoE,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 2,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for expert in 0..2 {
            let start = layout.graphon_nodes + expert * 3;
            assert!(trained.parameters[start..start + 3]
                .iter()
                .zip(&initial.parameters[start..start + 3])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn spatial_shift_prediction_reuses_fitted_experts_without_test_time_updates() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::SpatialShiftGraphonMoE,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let before = model.to_json_string().unwrap();
        let prediction = model.predict(3).unwrap();
        assert!(prediction.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(model.to_json_string().unwrap(), before);
    }

    #[test]
    fn graphon_gumbel_sampling_is_step_seeded_and_finite() {
        let first = graphon_gumbel_logistic_noise(7, 1, 2, 3, 4);
        let repeated = graphon_gumbel_logistic_noise(7, 1, 2, 3, 4);
        let next_step = graphon_gumbel_logistic_noise(8, 1, 2, 3, 4);
        assert!(first.is_finite());
        assert_eq!(first, repeated);
        assert_ne!(first, next_step);
    }

    #[test]
    fn gated_profile_trains_the_graph_convolution_projection() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::GatedGraphTemporal,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 32,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        assert!(trained.parameters[layout.spatial_v..layout.router]
            .iter()
            .zip(&initial.parameters[layout.spatial_v..layout.router])
            .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
    }

    #[test]
    fn reconstruction_masks_are_patchwise_reproducible_and_seventy_five_percent() {
        let first = masked_patch_indices(8, 41);
        assert_eq!(first.len(), 6);
        assert_eq!(first, masked_patch_indices(8, 41));
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(first.iter().all(|index| *index < 8));
        assert_eq!(masked_patch_indices(2, 41).len(), 1);
        assert_eq!(masked_patch_indices(3, 41).len(), 2);
    }
}
