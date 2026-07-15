use cartoboost_neural::{
    backend_affine_scores, backend_scalar_graph_f32, backend_scalar_graph_train_step_f32,
    BackendSelection,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

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
    let requested = requested.unwrap_or("auto").to_ascii_lowercase();
    if !matches!(
        requested.as_str(),
        "auto" | "cpu" | "cuda" | "rocm" | "metal"
    ) {
        return Err(GeoStError::InvalidBackend(format!(
            "unknown backend {requested:?}; expected auto, cpu, cuda, rocm, or metal"
        )));
    }
    let available = available_compute_backends();
    let selected = if requested == "auto" {
        "cpu".to_string()
    } else if available.iter().any(|name| name == &requested) {
        requested.clone()
    } else {
        return Err(GeoStError::InvalidBackend(format!(
            "requested backend {requested:?} is not available in this build; available backends: {}",
            available.join(", ")
        )));
    };
    Ok(ComputeBackendSelection {
        requested,
        selected,
        available,
    })
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
        Self {
            requested: "auto".to_string(),
            selected: "cpu".to_string(),
            available: available_compute_backends(),
        }
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
        let frame = Self {
            node_ids,
            timestamps,
            target,
            covariates,
            adjacency,
            horizon,
            frequency,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<()> {
        let nodes = self.node_ids.len();
        if nodes == 0 {
            return Err(GeoStError::InvalidFrame(
                "node ids cannot be empty".to_string(),
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
        for row in &self.target {
            if row.len() != nodes || row.iter().any(|value| !value.is_finite()) {
                return Err(GeoStError::InvalidFrame(
                    "target must be finite with shape [time, node]".to_string(),
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
        self.weights = vec![vec![0.0; decoder_feature_len]; frame.horizon];
        self.intercepts = vec![0.0; frame.horizon];
        let forward = frame.adjacency.row_normalized();
        let reverse = forward.transpose(nodes);
        let samples = frame.target.len() - frame.horizon;
        let mut hidden_by_cutoff = Vec::with_capacity(samples);
        let mut hidden = vec![vec![0.0; self.config.hidden_size]; nodes];
        for row in normalized_target.iter().take(samples) {
            hidden = self.recurrent_hidden(&hidden, row, &forward, &reverse);
            hidden_by_cutoff.push(hidden.clone());
        }
        let teacher_ratio = self.average_teacher_forcing_ratio();

        for h in 0..frame.horizon {
            let mut xtx = vec![vec![0.0; decoder_feature_len]; decoder_feature_len];
            let mut xty = vec![0.0; decoder_feature_len];
            for t in 0..samples {
                let mut decoder_input = normalized_target[t].clone();
                let mut decoder_hidden = hidden_by_cutoff[t].clone();
                let mut prior_prediction = decoder_input.clone();
                for step in 0..=h {
                    let decoder_features =
                        self.decoder_features(&decoder_input, &decoder_hidden, &forward, &reverse);
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
                    prior_prediction = blend_rows(next_actual, &prior_prediction, teacher_ratio);
                    decoder_hidden = self.recurrent_hidden(
                        &decoder_hidden,
                        &prior_prediction,
                        &forward,
                        &reverse,
                    );
                    decoder_input = prior_prediction.clone();
                }
            }
            for (idx, row) in xtx.iter_mut().enumerate() {
                row[idx] += self.config.ridge.max(1.0e-8);
            }
            let solved = solve_linear_system(xtx, xty);
            self.weights[h] = solved;
            self.intercepts[h] = 0.0;
        }

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
            let features = self.decoder_features(&state, &hidden, forward, reverse);
            let feature_len = self.weights[h].len();
            let rows = features
                .chunks(feature_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let means = vec![0.0; feature_len];
            let intercepts = vec![self.intercepts[h]; state.len()];
            let next = backend_affine_scores(
                &self.neural_backend_selection(),
                &rows,
                &means,
                &self.weights[h],
                &intercepts,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?
            .into_iter()
            .map(|value| value.clamp(normalized_min, normalized_max))
            .collect::<Vec<_>>();
            hidden = self.recurrent_hidden(&hidden, &next, forward, reverse);
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
    ) -> Vec<f64> {
        let nodes = state.len();
        let feature_len = self.diffusion_feature_len();
        let mut vectors = Vec::with_capacity(feature_len - 1);
        vectors.push(state.to_vec());
        let mut current = state.to_vec();
        for _ in 0..self.config.diffusion_steps {
            let mut next = vec![0.0; nodes];
            forward.matvec(&current, &mut next);
            vectors.push(next.clone());
            current = next;
        }
        current = state.to_vec();
        for _ in 0..self.config.diffusion_steps {
            let mut next = vec![0.0; nodes];
            reverse.matvec(&current, &mut next);
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
        out
    }

    fn decoder_features(
        &self,
        state: &[f64],
        hidden: &[Vec<f64>],
        forward: &CsrAdjacency,
        reverse: &CsrAdjacency,
    ) -> Vec<f64> {
        let nodes = state.len();
        let diffusion_len = self.diffusion_feature_len();
        let decoder_len = self.decoder_feature_len();
        let diffusion = self.diffusion_features(state, forward, reverse);
        let mut out = vec![0.0; nodes * decoder_len];
        for (node, hidden_row) in hidden.iter().enumerate().take(nodes) {
            let src = node * diffusion_len;
            let dst = node * decoder_len;
            out[dst..dst + diffusion_len].copy_from_slice(&diffusion[src..src + diffusion_len]);
            out[dst + diffusion_len..dst + decoder_len].copy_from_slice(hidden_row);
        }
        out
    }

    fn recurrent_hidden(
        &self,
        previous_hidden: &[Vec<f64>],
        state: &[f64],
        forward: &CsrAdjacency,
        reverse: &CsrAdjacency,
    ) -> Vec<Vec<f64>> {
        let nodes = state.len();
        let diffusion_len = self.diffusion_feature_len();
        let diffusion = self.diffusion_features(state, forward, reverse);
        let mut next = vec![vec![0.0; self.config.hidden_size]; nodes];
        for (node, next_row) in next.iter_mut().enumerate().take(nodes) {
            let x = &diffusion[node * diffusion_len..(node + 1) * diffusion_len];
            for (unit, value) in next_row
                .iter_mut()
                .enumerate()
                .take(self.config.hidden_size)
            {
                let input_term = dot(&self.encoder_weights[unit], x);
                let recurrent_term = dot(&self.recurrent_weights[unit], &previous_hidden[node]);
                *value = (input_term + 0.35 * recurrent_term).tanh();
            }
        }
        next
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
            hidden = self.recurrent_hidden(&hidden, row, forward, reverse);
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
    target_mean: f64,
    target_scale: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnTrainingCheckpoint {
    version: u32,
    data_fingerprint: u64,
    config_json: String,
    state: TrainableGraphTransformerState,
    pretraining_completed: usize,
    supervised_batches_completed: usize,
    complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PaperGraphTransformerArchitectureReport {
    pub profile: GraphTransformerProfile,
    pub components: Vec<String>,
    pub graphon_expert_count: usize,
    pub direct_multi_horizon: bool,
    pub trainable_forecast_head: bool,
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
    nodes: usize,
    hidden: usize,
    attention_heads: usize,
    periodicity: usize,
    recent_window: usize,
    context_window: usize,
    horizons: usize,
    experts: usize,
    graph_order: usize,
    #[serde(default = "unit_scale")]
    target_scale: f64,
    #[serde(default)]
    normalized_zero: f64,
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
            nodes,
            hidden,
            attention_heads,
            periodicity,
            recent_window,
            context_window,
            horizons,
            experts,
            graph_order,
            target_scale: 1.0,
            normalized_zero: 0.0,
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

    fn frozen_lsttn_patch_representations(
        &self,
        window: &[Vec<f64>],
        _adjacency: &CsrAdjacency,
        _phase_offset: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let layout = self.layout();
        let patch_width = (self.periodicity / 24).max(1);
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / self.hidden;
        let patch_count = window.len() / patch_width;
        let projection = self.hidden * (self.hidden + 1);
        let ffn_width = 8 * self.hidden * self.hidden + 5 * self.hidden;
        let by_node = (0..self.nodes)
            .into_par_iter()
            .map(|node| {
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
                    );
                }
                sequence
            })
            .collect::<Vec<_>>();
        (0..patch_count)
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
            .collect()
    }

    fn adamw_step(&mut self, gradients: &[f64], learning_rate: f64, weight_decay: f64) {
        self.steps += 1;
        let step = self.steps as f64;
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
    fn lsttn_example_loss_and_gradients(
        &self,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        targets: &[Vec<f64>],
        phase_offset: usize,
        frozen_patches: Option<&[Vec<Vec<f32>>]>,
        time_features: Option<&[Vec<f64>]>,
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
        let valid = targets
            .iter()
            .flatten()
            .filter(|target| (**target - self.normalized_zero).abs() > 1e-12)
            .count();
        let scale = tape.constant(1.0 / valid.max(1) as f64);
        for node in 0..self.nodes {
            for horizon in 0..self.horizons {
                if (targets[horizon][node] - self.normalized_zero).abs() <= 1e-12 {
                    continue;
                }
                let residual = tape.add(
                    outputs[node][horizon],
                    tape.constant(-targets[horizon][node]),
                );
                let residual = tape.mul(residual, tape.constant(self.target_scale));
                let mae = tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)));
                loss = tape.add(loss, tape.mul(mae, scale));
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
        let frozen_lsttn = (*profile == GraphTransformerProfile::LongShortFusion)
            .then(|| self.frozen_lsttn_patch_representations(window, adjacency, phase_offset));
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
        for node in 0..self.nodes {
            for horizon in 0..self.horizons {
                if *profile == GraphTransformerProfile::LongShortFusion
                    && (targets[horizon][node] - self.normalized_zero).abs() <= 1e-12
                {
                    continue;
                }
                let residual = tape.add(
                    outputs[node][horizon],
                    tape.constant(-targets[horizon][node]),
                );
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
            self.adamw_step(&gradients, learning_rate, weight_decay);
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
            self.adamw_step(&gradients, learning_rate, weight_decay);
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
        let frozen_lsttn = (*profile == GraphTransformerProfile::LongShortFusion)
            .then(|| self.frozen_lsttn_patch_representations(window, adjacency, phase_offset));
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
        // LSTTN's periodic graph convolution uses both directed structural
        // diffusions as well as its learned adaptive diffusion.  Preserve a
        // normalized reverse graph rather than treating the supplied road
        // graph as undirected.
        let reverse_adjacency = adjacency.transpose(nodes).row_normalized();
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
                        effective_periodicity,
                    )),
                    tape.constant(periodic_phase(
                        phase_offset + time + 1,
                        effective_periodicity * 7,
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
            [effective_periodicity, effective_periodicity * 7]
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
                    let adaptive_weights = (0..nodes)
                        .map(|source| {
                            let logits = (0..nodes)
                                .map(|target| {
                                    let score = (0..10).fold(tape.constant(0.0), |sum, latent| {
                                        tape.add(
                                            sum,
                                            tape.mul(
                                                parameter(
                                                    &tape,
                                                    adaptive_source + source * 10 + latent,
                                                ),
                                                parameter(
                                                    &tape,
                                                    adaptive_target + target * 10 + latent,
                                                ),
                                            ),
                                        )
                                    });
                                    tape.max(score, tape.constant(0.0))
                                })
                                .collect::<Vec<_>>();
                            tape_softmax(&tape, &logits)
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
                    let adaptive_one = tape_dense_diffuse(&tape, &adaptive_weights, values, hidden);
                    let adaptive_two =
                        tape_dense_diffuse(&tape, &adaptive_weights, &adaptive_one, hidden);
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
                            .map(|source| {
                                let logits = (0..nodes)
                                    .map(|target| {
                                        let score =
                                            (0..10).fold(tape.constant(0.0), |sum, latent| {
                                                tape.add(
                                                    sum,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            layout.lsttn_short_adaptive_source
                                                                + source * 10
                                                                + latent,
                                                        ),
                                                        parameter(
                                                            &tape,
                                                            layout.lsttn_short_adaptive_target
                                                                + target * 10
                                                                + latent,
                                                        ),
                                                    ),
                                                )
                                            });
                                        tape.max(score, tape.constant(0.0))
                                    })
                                    .collect::<Vec<_>>();
                                tape_softmax(&tape, &logits)
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
                                let adaptive_one = tape_dense_diffuse(
                                    &tape,
                                    &short_adaptive,
                                    &gated[time],
                                    hidden,
                                );
                                let adaptive_two = tape_dense_diffuse(
                                    &tape,
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

fn numeric_linear(
    parameters: &[f64],
    offset: usize,
    input: &[f64],
    input_width: usize,
    output_width: usize,
) -> Vec<f64> {
    (0..output_width)
        .map(|output| {
            input.iter().enumerate().take(input_width).fold(
                parameters[offset + input_width * output_width + output],
                |sum, (index, value)| {
                    sum + parameters[offset + index * output_width + output] * value
                },
            )
        })
        .collect()
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
) -> Vec<Vec<f64>> {
    let queries = sequence
        .iter()
        .map(|token| numeric_linear(parameters, q_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let keys = sequence
        .iter()
        .map(|token| numeric_linear(parameters, k_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let values = sequence
        .iter()
        .map(|token| numeric_linear(parameters, v_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    sequence
        .iter()
        .enumerate()
        .map(|(token_index, residual)| {
            let mut attended = vec![0.0; hidden];
            for head in 0..heads {
                let start = head * hidden / heads;
                let end = (head + 1) * hidden / heads;
                let scale = 1.0 / ((end - start) as f64).sqrt();
                let logits = keys
                    .iter()
                    .map(|key| {
                        queries[token_index][start..end]
                            .iter()
                            .zip(&key[start..end])
                            .map(|(left, right)| left * right)
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut weights = logits
                    .iter()
                    .map(|value| (value - max).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f64>().max(1e-12);
                for weight in &mut weights {
                    *weight /= denominator;
                }
                for channel in start..end {
                    attended[channel] = weights
                        .iter()
                        .zip(&values)
                        .map(|(weight, value)| weight * value[channel])
                        .sum();
                }
            }
            let projected = numeric_linear(parameters, out_offset, &attended, hidden, hidden);
            let first = residual
                .iter()
                .zip(projected)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>();
            let normalized = numeric_layer_norm(parameters, norm_offset, &first);
            let expanded = numeric_linear(parameters, ffn_offset, &normalized, hidden, 4 * hidden)
                .into_iter()
                .map(|value| value.max(0.0))
                .collect::<Vec<_>>();
            let contracted = numeric_linear(
                parameters,
                ffn_offset + (hidden + 1) * 4 * hidden,
                &expanded,
                4 * hidden,
                hidden,
            );
            numeric_layer_norm(
                parameters,
                norm_offset + 2 * hidden,
                &normalized
                    .iter()
                    .zip(contracted)
                    .map(|(left, right)| left + right)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
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

fn tape_dense_diffuse(
    tape: &AutodiffTape,
    weights: &[Vec<usize>],
    values: &[Vec<usize>],
    hidden: usize,
) -> Vec<Vec<usize>> {
    weights
        .iter()
        .map(|row| {
            (0..hidden)
                .map(|channel| {
                    row.iter()
                        .zip(values)
                        .fold(tape.constant(0.0), |sum, (weight, source)| {
                            tape.add(sum, tape.mul(*weight, source[channel]))
                        })
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
            let signal = delayed_graph_signal(
                &normalized_target,
                &edges,
                &frame.adjacency.data,
                &delays,
                time_idx,
            );
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
            let signal =
                delayed_graph_signal(&history, &self.edges, &self.edge_weights, &delays, time_idx);
            let mut next = vec![0.0; self.node_ids.len()];
            for node in 0..self.node_ids.len() {
                next[node] = self.coefficients[0]
                    + self.coefficients[1] * history[time_idx][node]
                    + self.coefficients[2] * signal[node];
            }
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
        self.weights = vec![vec![0.0; feature_len]; frame.horizon];
        self.intercepts = vec![0.0; frame.horizon];
        let samples = frame.target.len() - self.config.lookback - frame.horizon + 1;
        for h in 0..frame.horizon {
            let mut xtx = vec![vec![0.0; feature_len]; feature_len];
            let mut xty = vec![0.0; feature_len];
            for sample in 0..samples {
                let cutoff = sample + self.config.lookback;
                let features = self.wave_features(&normalized_target[sample..cutoff], &adjacency);
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
            self.weights[h] = solve_linear_system(xtx, xty);
        }
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
            let features = self.wave_features(&history[start..], adjacency);
            let feature_len = self.weights[h].len();
            let rows = features
                .chunks(feature_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let means = vec![0.0; feature_len];
            let intercepts = vec![self.intercepts[h]; self.node_ids.len()];
            let next = backend_affine_scores(
                &self.neural_backend_selection(),
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

    fn wave_features(&self, window: &[Vec<f64>], adjacency: &CsrAdjacency) -> Vec<f64> {
        let nodes = window[0].len();
        let feature_len = self.feature_len();
        let last = window.last().expect("non-empty lookback window");
        let mut neighbor_last = vec![0.0; nodes];
        adjacency.matvec(last, &mut neighbor_last);
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
                let mut neighbor = vec![0.0; nodes];
                adjacency.matvec(current, &mut neighbor);
                let gated = (current[node] + neighbor[node]).tanh();
                let filter = sigmoid(current[node] - previous[node]);
                let base = offset + 3 + depth * 4;
                out[base] = gated * filter;
                out[base + 1] = current[node];
                out[base + 2] = neighbor[node];
                out[base + 3] = current[node] - previous[node];
            }
        }
        out
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
        self.weights = vec![vec![0.0; feature_len]; frame.horizon];
        self.intercepts = vec![0.0; frame.horizon];
        let samples = frame.target.len() - self.config.lookback - frame.horizon + 1;
        for h in 0..frame.horizon {
            let mut xtx = vec![vec![0.0; feature_len]; feature_len];
            let mut xty = vec![0.0; feature_len];
            for sample in 0..samples {
                let cutoff = sample + self.config.lookback;
                let features =
                    self.attention_features(&normalized_target[sample..cutoff], &adjacency);
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
            self.weights[h] = solve_linear_system(xtx, xty);
        }
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
            let features = self.attention_features(&history[start..], adjacency);
            let feature_len = self.weights[h].len();
            let rows = features
                .chunks(feature_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let means = vec![0.0; feature_len];
            let intercepts = vec![self.intercepts[h]; self.node_ids.len()];
            let next = backend_affine_scores(
                &self.neural_backend_selection(),
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

    fn attention_features(&self, window: &[Vec<f64>], adjacency: &CsrAdjacency) -> Vec<f64> {
        let nodes = window[0].len();
        let feature_len = self.feature_len();
        let mut out = vec![0.0; nodes * feature_len];
        let last = window.last().expect("non-empty lookback window");
        let mut neighbor_last = vec![0.0; nodes];
        adjacency.matvec(last, &mut neighbor_last);
        for node in 0..nodes {
            let offset = node * feature_len;
            out[offset] = last[node];
            out[offset + 1] = neighbor_last[node];
            out[offset + 2] = 1.0;
            let series = window.iter().map(|row| row[node]).collect::<Vec<_>>();
            let neighbor_series = window
                .iter()
                .map(|row| {
                    let mut smoothed = vec![0.0; nodes];
                    adjacency.matvec(row, &mut smoothed);
                    smoothed[node]
                })
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
        out
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
            target_mean: 0.0,
            target_scale: 1.0,
        })
    }

    pub fn fit(&mut self, frame: &GraphTemporalFrame) -> Result<()> {
        self.fit_internal(frame, None)
    }

    pub fn fit_checkpointed(
        &mut self,
        frame: &GraphTemporalFrame,
        checkpoint_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.fit_internal(frame, Some(checkpoint_path.as_ref()))
    }

    fn fit_internal(
        &mut self,
        frame: &GraphTemporalFrame,
        checkpoint_path: Option<&Path>,
    ) -> Result<()> {
        frame.validate()?;
        if self.config.profile == GraphTransformerProfile::LongShortFusion {
            let patch_width = (self.config.periodicity / 24).max(1);
            let patch_count = self.config.lookback / patch_width;
            let weekly_lag = (self.config.periodicity / patch_width).max(1) * 7;
            if !self.config.lookback.is_multiple_of(patch_width) || patch_count <= weekly_lag {
                return Err(GeoStError::InvalidFrame(format!(
                    "LSTTN lookback must contain complete patches and exceed one weekly lag: lookback={}, patch_width={}, weekly_lag_patches={weekly_lag}",
                    self.config.lookback, patch_width
                )));
            }
        }
        if frame.target.len() <= self.config.lookback + frame.horizon {
            return Err(GeoStError::InvalidFrame(
                "target length must exceed lookback plus horizon".to_string(),
            ));
        }
        let (mean, scale) = target_center_scale(&frame.target);
        let normalized = frame
            .target
            .iter()
            .map(|row| row.iter().map(|v| (v - mean) / scale).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let lsttn_time_features = frame.covariates.as_ref().map(|covariates| {
            covariates
                .iter()
                .map(|time| time.iter().map(|node| node[0]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        });
        let adjacency = frame.adjacency.row_normalized();
        let backend = BackendSelection {
            requested: self.config.backend.requested.clone(),
            selected: self.config.backend.selected.clone(),
            available: self.config.backend.available.clone(),
        };
        let data_fingerprint = graph_temporal_training_fingerprint(frame);
        let config_json = serde_json::to_string(&self.config)?;
        let resumed = checkpoint_path
            .filter(|path| path.is_file())
            .map(|path| -> Result<LsttnTrainingCheckpoint> {
                let checkpoint: LsttnTrainingCheckpoint =
                    serde_json::from_str(&fs::read_to_string(path)?)?;
                if checkpoint.version != 2
                    || checkpoint.data_fingerprint != data_fingerprint
                    || checkpoint.config_json != config_json
                {
                    return Err(GeoStError::InvalidFrame(format!(
                        "LSTTN checkpoint {} does not match this frame and configuration",
                        path.display()
                    )));
                }
                Ok(checkpoint)
            })
            .transpose()?;
        let mut state = resumed
            .as_ref()
            .map(|checkpoint| checkpoint.state.clone())
            .unwrap_or_else(|| {
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
            });
        state.target_scale = scale;
        state.normalized_zero = -mean / scale;
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
        if self.config.profile == GraphTransformerProfile::LongShortFusion {
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
                                version: 2,
                                data_fingerprint,
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
        if self.config.profile == GraphTransformerProfile::LongShortFusion {
            // The reference LSTTN trainer uses 32-example mini-batches. Each
            // example is independent up to Adam's update, so evaluate all
            // scalar tapes on Rayon workers, average their gradients in
            // stable start-order, and take one serialized Adam step.
            const LSTTN_BATCH_SIZE: usize = 32;
            let batches_per_epoch = supervised_starts.len().div_ceil(LSTTN_BATCH_SIZE);
            let total_batches = batches_per_epoch * self.config.epochs;
            for epoch in 0..self.config.epochs {
                for (batch_index, starts) in supervised_starts.chunks(LSTTN_BATCH_SIZE).enumerate()
                {
                    let task_index = epoch * batches_per_epoch + batch_index;
                    if task_index < supervised_batches_completed {
                        continue;
                    }
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
                                &normalized[cutoff..cutoff + frame.horizon],
                                *start,
                                frozen_lsttn_cache
                                    .as_ref()
                                    .map(|cache| cache[cache_index].as_slice()),
                                lsttn_time_features
                                    .as_ref()
                                    .map(|features| &features[*start..cutoff]),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut gradients = vec![0.0; state.parameters.len()];
                    let mut loss = 0.0;
                    for (example_loss, example_gradients) in examples {
                        loss += example_loss;
                        for (total, gradient) in gradients.iter_mut().zip(example_gradients) {
                            *total += gradient;
                        }
                    }
                    let batch_scale = 1.0 / starts.len() as f64;
                    for gradient in &mut gradients {
                        *gradient *= batch_scale;
                    }
                    state.freeze_lsttn_transformer_gradients(state.layout(), &mut gradients);
                    clip_gradient_norm(&mut gradients, 3.0);
                    let scheduler_steps = [1usize, 18, 36, 54, 72]
                        .into_iter()
                        .filter(|milestone| *milestone <= epoch)
                        .count();
                    let epoch_learning_rate =
                        self.config.learning_rate * 0.5_f64.powi(scheduler_steps as i32);
                    state.adamw_step(&gradients, epoch_learning_rate, self.config.weight_decay);
                    let mean_batch_loss = loss * batch_scale;
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
                                version: 2,
                                data_fingerprint,
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
                            normalized[cutoff..cutoff + frame.horizon].to_vec()
                        } else {
                            let baseline = &normalized[cutoff - 1];
                            normalized[cutoff..cutoff + frame.horizon]
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
        self.trainable_state = Some(state);
        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.horizon = frame.horizon;
        self.adjacency = Some(adjacency);
        self.history = normalized;
        self.history_time_features = lsttn_time_features;
        self.target_mean = mean;
        self.target_scale = scale;
        if let Some(path) = checkpoint_path {
            write_lsttn_checkpoint(
                path,
                &LsttnTrainingCheckpoint {
                    version: 2,
                    data_fingerprint,
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
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        model.validate_artifact()?;
        Ok(model)
    }
    pub fn to_json_string(&self) -> Result<String> {
        self.validate_artifact()?;
        Ok(serde_json::to_string(self)?)
    }
    pub fn from_json_string(value: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(value)?;
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
            && self.history_time_features.as_ref().is_none_or(|features| {
                features.len() == self.history.len()
                    && features.iter().all(|row| row.len() == state.nodes)
            });
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
            components.push(format!(
                "{}_full_graph_training_and_inference",
                self.config.backend.selected
            ));
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
            for start in group - 1..end {
                let candidate = dp[group - 1][start] + score[start][end];
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

fn dot(weights: &[f64], values: &[f64]) -> f64 {
    weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
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
    fn auto_compute_backend_selects_cpu() {
        let selection = select_compute_backend(Some("auto")).unwrap();
        assert_eq!(selection.requested, "auto");
        assert_eq!(selection.selected, "cpu");

        let default_selection = select_compute_backend(None).unwrap();
        assert_eq!(default_selection.requested, "auto");
        assert_eq!(default_selection.selected, "cpu");
    }

    #[test]
    fn webgpu_compute_backend_is_not_selectable() {
        let err = select_compute_backend(Some("webgpu")).unwrap_err();
        assert!(err
            .to_string()
            .contains("expected auto, cpu, cuda, rocm, or metal"));
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
        );
        let (cached_loss, cached) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            Some(&cache),
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
        assert!(error.to_string().contains("exceed one weekly lag"));
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
        );
        let (second_loss, second_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            Some(&second),
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
