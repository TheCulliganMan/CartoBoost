use cartoboost_neural::{backend_affine_scores, BackendSelection};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub mod market;
pub use market::{
    ExpertEventLabel, ExpertRelationshipPrior, MarketExplanation, MarketPanelFrame,
    MarketPrediction, MarketRelationship, MarketShiftKind, MarketStructureConfig,
    MarketStructureForecaster, MarketSupportKind, RelationshipKind, WeeklyMarketPrediction,
};

pub type Result<T> = std::result::Result<T, GeoStError>;

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
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
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
    target_mean: f64,
    target_scale: f64,
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
    horizons: usize,
    experts: usize,
    graph_order: usize,
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
    graphon_nodes: usize,
    graphon_time: usize,
    output: usize,
    pretrain_mask_token: usize,
    pretrain_position: usize,
    pretrain_decoder: usize,
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
        let transformer_blocks = 3;
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
        let stgformer_pointwise = lsttn_short_wave + 2 * (7 * hidden * hidden + 3 * hidden);
        let lsttn_adaptive_source = stgformer_pointwise + graph_order * hidden * (hidden + 1);
        let lsttn_adaptive_target = lsttn_adaptive_source + nodes * hidden;
        let graphon_nodes = lsttn_adaptive_target + nodes * hidden;
        let graphon_time = graphon_nodes + experts * nodes;
        let output = graphon_time + experts * hidden;
        let pretrain_mask_token = output + horizons * (hidden + 1);
        // The reference LSTTN configuration has 336 twelve-step patches in a
        // two-week history.  Keep a learnable table for at least that many;
        // longer user contexts wrap by patch index rather than silently
        // dropping positional information.
        let pretrain_positions = (periodicity * 14 / (periodicity / 24).max(1)).max(336);
        let pretrain_position = pretrain_mask_token + hidden;
        let pretrain_decoder = pretrain_position + pretrain_positions * hidden;
        let patch_width = (periodicity / 24).max(1);
        let total = pretrain_decoder + patch_width * (hidden + 1);
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
            graphon_nodes,
            graphon_time,
            output,
            pretrain_mask_token,
            pretrain_position,
            pretrain_decoder,
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
        horizons: usize,
        experts: usize,
        graph_order: usize,
        seed: u64,
    ) -> Self {
        let layout =
            GraphParameterLayout::new(nodes, hidden, horizons, experts, graph_order, periodicity);
        let mut state = seed;
        let parameters = (0..layout.total)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) - 0.5) * 0.08
            })
            .collect::<Vec<_>>();
        Self {
            first_moment: vec![0.0; layout.total],
            second_moment: vec![0.0; layout.total],
            parameters,
            steps: 0,
            nodes,
            hidden,
            attention_heads,
            periodicity,
            horizons,
            experts,
            graph_order,
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
        )
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

    #[allow(clippy::too_many_arguments)]
    fn train_example(
        &mut self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        targets: &[Vec<f64>],
        excluded_expert: Option<usize>,
        learning_rate: f64,
        weight_decay: f64,
    ) -> f64 {
        let (tape, outputs, router_weights) =
            self.forward(profile, window, adjacency, excluded_expert);
        let mut loss = tape.constant(0.0);
        let scale = tape.constant(1.0 / (self.nodes * self.horizons) as f64);
        for node in 0..self.nodes {
            for horizon in 0..self.horizons {
                let residual = tape.add(
                    outputs[node][horizon],
                    tape.constant(-targets[horizon][node]),
                );
                loss = tape.add(loss, tape.mul(tape.mul(residual, residual), scale));
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
        let value = tape.value(loss);
        self.adamw_step(
            &tape.backward(loss, self.parameters.len()),
            learning_rate,
            weight_decay,
        );
        value
    }

    /// LSTTN's self-supervised stage: encode the unmasked equal-length
    /// subseries, insert learned mask tokens at the withheld positions, and
    /// decode only those patches.  The shared input/Q/K/V projections are the
    /// same ones used by the forecasting path, so pretraining transfers a
    /// contextual long-history representation instead of training a detached
    /// auxiliary model.
    fn train_masked_subseries_reconstruction(
        &mut self,
        window: &[Vec<f64>],
        learning_rate: f64,
        weight_decay: f64,
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
        let tape = AutodiffTape::new();
        let parameter =
            |tape: &AutodiffTape, index: usize| tape.parameter(index, self.parameters[index]);
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / self.hidden;
        let mut loss = tape.constant(0.0);
        let scale = tape.constant(1.0 / (masked.len() * self.nodes * patch_width) as f64);
        for node in 0..self.nodes {
            let representations = visible
                .iter()
                .map(|patch| {
                    let mean = window[patch * patch_width..(patch + 1) * patch_width]
                        .iter()
                        .map(|row| row[node])
                        .sum::<f64>()
                        / patch_width as f64;
                    (0..self.hidden)
                        .map(|channel| {
                            let projected = tape.add(
                                parameter(&tape, layout.input + 7 * self.hidden + channel),
                                tape.mul(
                                    parameter(&tape, layout.input + channel),
                                    tape.constant(mean),
                                ),
                            );
                            tape.tanh(tape.add(
                                projected,
                                parameter(
                                    &tape,
                                    layout.pretrain_position
                                        + (patch % position_count) * self.hidden
                                        + channel,
                                ),
                            ))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let keys = representations
                .iter()
                .map(|representation| {
                    tape_linear(
                        &tape,
                        &self.parameters,
                        layout.temporal_k,
                        representation,
                        self.hidden,
                        self.hidden,
                    )
                })
                .collect::<Vec<_>>();
            let values = representations
                .iter()
                .map(|representation| {
                    tape_linear(
                        &tape,
                        &self.parameters,
                        layout.temporal_v,
                        representation,
                        self.hidden,
                        self.hidden,
                    )
                })
                .collect::<Vec<_>>();
            for patch in &masked {
                let mask_representation = (0..self.hidden)
                    .map(|channel| {
                        tape.add(
                            parameter(&tape, layout.pretrain_mask_token + channel),
                            parameter(
                                &tape,
                                layout.pretrain_position
                                    + (patch % position_count) * self.hidden
                                    + channel,
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                let query = tape_linear(
                    &tape,
                    &self.parameters,
                    layout.temporal_q,
                    &mask_representation,
                    self.hidden,
                    self.hidden,
                );
                let weights = tape_softmax(
                    &tape,
                    &keys
                        .iter()
                        .map(|key| tape_dot(&tape, &query, key))
                        .collect::<Vec<_>>(),
                );
                let context = tape_weighted_sum(&tape, &weights, &values, self.hidden);
                for offset in 0..patch_width {
                    let mut prediction = parameter(
                        &tape,
                        layout.pretrain_decoder + patch_width * self.hidden + offset,
                    );
                    for (channel, context_value) in context.iter().enumerate().take(self.hidden) {
                        prediction = tape.add(
                            prediction,
                            tape.mul(
                                parameter(
                                    &tape,
                                    layout.pretrain_decoder + offset * self.hidden + channel,
                                ),
                                *context_value,
                            ),
                        );
                    }
                    let residual = tape.add(
                        prediction,
                        tape.constant(-window[patch * patch_width + offset][node]),
                    );
                    loss = tape.add(loss, tape.mul(scale, tape.mul(residual, residual)));
                }
            }
        }
        let value = tape.value(loss);
        self.adamw_step(
            &tape.backward(loss, self.parameters.len()),
            learning_rate,
            weight_decay,
        );
        Ok(value)
    }

    fn predict_window(
        &self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
    ) -> Vec<Vec<f64>> {
        let (tape, outputs, _) = self.forward(profile, window, adjacency, None);
        (0..self.horizons)
            .map(|horizon| {
                (0..self.nodes)
                    .map(|node| tape.value(outputs[node][horizon]))
                    .collect()
            })
            .collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(
        &self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        excluded_expert: Option<usize>,
    ) -> (AutodiffTape, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let layout = self.layout();
        let tape = AutodiffTape::new();
        let parameter =
            |tape: &AutodiffTape, index: usize| tape.parameter(index, self.parameters[index]);
        let nodes = self.nodes;
        let hidden = self.hidden;
        let times = window.len();
        // LSTTN's periodic graph convolution uses both directed structural
        // diffusions as well as its learned adaptive diffusion.  Preserve a
        // normalized reverse graph rather than treating the supplied road
        // graph as undirected.
        let reverse_adjacency = adjacency.transpose(nodes);
        let mut graph_values = vec![vec![0.0; nodes]; times];
        for (time, row) in window.iter().enumerate() {
            adjacency.matvec(row, &mut graph_values[time]);
        }
        let degrees = graph_in_degrees(adjacency, nodes);
        let out_degrees = graph_out_degrees(adjacency, nodes);
        let mut embedding = vec![vec![vec![0usize; hidden]; nodes]; times];
        for time in 0..times {
            let position = tape.constant((time + 1) as f64 / times as f64);
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
                        tape.constant(window[time][node]),
                        tape.constant(graph_values[time][node]),
                        time2vec,
                        tape.constant(
                            ((time + 1) as f64 * std::f64::consts::TAU / self.periodicity as f64)
                                .sin(),
                        ),
                        tape.constant(
                            ((time + 1) as f64 * std::f64::consts::TAU
                                / (self.periodicity * 7) as f64)
                                .sin(),
                        ),
                        tape.constant(degrees[node] / nodes.max(1) as f64),
                        tape.constant(out_degrees[node] / nodes.max(1) as f64),
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
        for node in 0..nodes {
            let query = tape_linear(
                &tape,
                &self.parameters,
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
                    &self.parameters,
                    layout.temporal_k,
                    &embedding[time][node],
                    hidden,
                    hidden,
                );
                let value = tape_linear(
                    &tape,
                    &self.parameters,
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

        let distances = graph_distances(adjacency, nodes);
        let mut spatial = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::EfficientHighOrder {
            // STGformer uses one QKV projection for its spatial and temporal
            // paths.  The efficient K^T V statistic is shared by all query
            // nodes within each head.
            let keys = (0..nodes)
                .map(|node| {
                    tape_linear(
                        &tape,
                        &self.parameters,
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
                        &self.parameters,
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
                    &self.parameters,
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
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &self.parameters,
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
                        &self.parameters,
                        layout.spatial_k,
                        &temporal[other],
                        hidden,
                        hidden,
                    );
                    let value = tape_linear(
                        &tape,
                        &self.parameters,
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
                            &self.parameters,
                            temporal_q,
                            &states[time][node],
                            hidden,
                            hidden,
                        );
                        let keys = (0..=time)
                            .map(|past| {
                                tape_linear(
                                    &tape,
                                    &self.parameters,
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
                                    &self.parameters,
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
                            &self.parameters,
                            spatial_q,
                            &block_temporal[time][node],
                            hidden,
                            hidden,
                        );
                        let keys = (0..nodes)
                            .map(|other| {
                                tape_linear(
                                    &tape,
                                    &self.parameters,
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
                                    &self.parameters,
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
                            tape.mul(
                                tape.constant(adjacency.data[edge]),
                                temporal[neighbor][channel],
                            ),
                        );
                    }
                }
                graph_convolution[node] = tape_linear(
                    &tape,
                    &self.parameters,
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
                            &self.parameters,
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
                            &self.parameters,
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
                            &self.parameters,
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
                        &self.parameters,
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
                                tape.mul(
                                    tape.constant(adjacency.data[edge]),
                                    propagated[source][channel],
                                ),
                            );
                        }
                    }
                }
                propagated = next;
            }
        }

        let mut graphon_expert_states =
            vec![vec![vec![tape.constant(0.0); hidden]; self.experts]; nodes];
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
                    let patch_width = (self.periodicity / 24).max(1);
                    let subseries = embedding
                        .chunks(patch_width)
                        .map(|patch| {
                            (0..hidden)
                                .map(|channel| {
                                    let sum = patch.iter().fold(tape.constant(0.0), |sum, row| {
                                        tape.add(sum, row[node][channel])
                                    });
                                    tape.div(sum, tape.constant(patch.len() as f64))
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut long_sequence = subseries.clone();
                    for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate() {
                        let layer_offset = layout.lsttn_dilated_convolution
                            + layer * (3 * hidden * hidden + hidden);
                        let long_times = long_sequence.len();
                        let mut next = vec![vec![tape.constant(0.0); hidden]; long_times];
                        for time in 0..long_times {
                            for output_channel in 0..hidden {
                                let mut value = parameter(
                                    &tape,
                                    layer_offset + 3 * hidden * hidden + output_channel,
                                );
                                for tap in 0..3 {
                                    if let Some(source_time) = time.checked_sub(tap * dilation) {
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
                                next[time][output_channel] = tape.tanh(value);
                            }
                        }
                        // LSTTN pairs each dilated layer with temporal max
                        // pooling to compact the patch sequence.  Selecting
                        // the winning tape node gives max-pool's standard
                        // gradient path without introducing a Python kernel.
                        long_sequence = if next.len() < 2 {
                            next
                        } else {
                            next.chunks(2)
                                .map(|pool| {
                                    (0..hidden)
                                        .map(|channel| {
                                            pool.iter()
                                                .map(|row| row[channel])
                                                .max_by(|left, right| {
                                                    tape.value(*left)
                                                        .partial_cmp(&tape.value(*right))
                                                        .expect("finite convolution activations")
                                                })
                                                .expect("nonempty pooling window")
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        };
                    }
                    let mut long = long_sequence[long_sequence.len() - 1].clone();
                    // LSTTN's periodic branch uses a separately learned,
                    // input-conditioned adjacency for the available daily and
                    // weekly lags.  It is deliberately distinct from the
                    // supplied road graph: the node embeddings learn a
                    // dynamic periodic affinity while the current temporal
                    // state conditions that affinity at each forecast cutoff.
                    let mut periodic = vec![tape.constant(0.0); hidden];
                    let mut periodic_count = 0usize;
                    for period in [self.periodicity, self.periodicity * 7] {
                        let period_patches = (period / patch_width).max(1);
                        if subseries.len() > period_patches {
                            periodic_count += 1;
                            let patch_index = subseries.len() - period_patches - 1;
                            let patch_start = patch_index * patch_width;
                            let period_embedding = (0..nodes)
                                .map(|period_node| {
                                    (0..hidden)
                                        .map(|channel| {
                                            let patch = &embedding[patch_start
                                                ..(patch_start + patch_width).min(times)];
                                            let sum = patch
                                                .iter()
                                                .fold(tape.constant(0.0), |sum, row| {
                                                    tape.add(sum, row[period_node][channel])
                                                });
                                            tape.div(sum, tape.constant(patch.len() as f64))
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            // Forward and backward graph diffusions from the
                            // observed road topology.  The adaptive branch
                            // below is intentionally an additional third
                            // diffusion, not a replacement for these paths.
                            for edge in adjacency.indptr[node]..adjacency.indptr[node + 1] {
                                let source = adjacency.indices[edge];
                                for channel in 0..hidden {
                                    periodic[channel] = tape.add(
                                        periodic[channel],
                                        tape.mul(
                                            tape.constant(adjacency.data[edge]),
                                            period_embedding[source][channel],
                                        ),
                                    );
                                }
                            }
                            for edge in
                                reverse_adjacency.indptr[node]..reverse_adjacency.indptr[node + 1]
                            {
                                let source = reverse_adjacency.indices[edge];
                                for channel in 0..hidden {
                                    periodic[channel] = tape.add(
                                        periodic[channel],
                                        tape.mul(
                                            tape.constant(reverse_adjacency.data[edge]),
                                            period_embedding[source][channel],
                                        ),
                                    );
                                }
                            }
                            // Adaptive diffusion from separate source and
                            // target node embeddings.  The current temporal
                            // state is added to the source embedding before
                            // the dot product, making the graph conditioned on
                            // the input window; softmax supplies the required
                            // row-normalized hidden transition matrix.
                            let logits = (0..nodes)
                                .map(|other| {
                                    (0..hidden).fold(tape.constant(0.0), |score, latent| {
                                        let source = tape.add(
                                            parameter(
                                                &tape,
                                                layout.lsttn_adaptive_source
                                                    + node * hidden
                                                    + latent,
                                            ),
                                            temporal[node][latent],
                                        );
                                        let target = parameter(
                                            &tape,
                                            layout.lsttn_adaptive_target + other * hidden + latent,
                                        );
                                        tape.add(score, tape.mul(source, target))
                                    })
                                })
                                .collect::<Vec<_>>();
                            let adaptive = tape_softmax(&tape, &logits);
                            for other in 0..nodes {
                                for channel in 0..hidden {
                                    periodic[channel] = tape.add(
                                        periodic[channel],
                                        tape.mul(adaptive[other], period_embedding[other][channel]),
                                    );
                                }
                            }
                        }
                    }
                    if periodic_count > 0 {
                        let denominator = tape.constant((periodic_count * (nodes + 2)) as f64);
                        for channel in 0..hidden {
                            long[channel] =
                                tape.add(long[channel], tape.div(periodic[channel], denominator));
                        }
                    }
                    // The short branch is a Graph WaveNet-style stack rather
                    // than a reuse of the generic transformer attention.  A
                    // causal gated temporal convolution learns local traffic
                    // changes, then an input-conditioned adaptive adjacency
                    // propagates those changes across nodes.  Keeping this
                    // separate from the long dilation stack makes the
                    // long/short fusion an actual architectural distinction.
                    let short_start = times.saturating_sub(12);
                    let mut short_sequence = embedding[short_start..].to_vec();
                    for (layer, dilation) in [1usize, 2].into_iter().enumerate() {
                        let layer_width = 7 * hidden * hidden + 3 * hidden;
                        let layer_offset = layout.lsttn_short_wave + layer * layer_width;
                        let filter_offset = layer_offset;
                        let gate_offset = filter_offset + 3 * hidden * hidden;
                        let filter_bias = gate_offset + 3 * hidden * hidden;
                        let gate_bias = filter_bias + hidden;
                        let graph_projection = gate_bias + hidden;
                        let graph_bias = graph_projection + hidden * hidden;
                        let short_times = short_sequence.len();
                        let mut temporal_next =
                            vec![vec![vec![tape.constant(0.0); hidden]; nodes]; short_times];
                        for time in 0..short_times {
                            for current_node in 0..nodes {
                                for output_channel in 0..hidden {
                                    let mut filter = parameter(&tape, filter_bias + output_channel);
                                    let mut gate = parameter(&tape, gate_bias + output_channel);
                                    for tap in 0..3 {
                                        if let Some(source_time) = time.checked_sub(tap * dilation)
                                        {
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
                                    }
                                    temporal_next[time][current_node][output_channel] =
                                        tape.mul(tape.tanh(filter), tape.sigmoid(gate));
                                }
                            }
                        }
                        let mut next =
                            vec![vec![vec![tape.constant(0.0); hidden]; nodes]; short_times];
                        for time in 0..short_times {
                            for target in 0..nodes {
                                let source_embedding =
                                    parameter(&tape, layout.graphon_nodes + target);
                                let mut adaptive = vec![tape.constant(0.0); hidden];
                                for channel in 0..hidden {
                                    let logits = (0..nodes)
                                        .map(|source| {
                                            tape.add(
                                                tape.mul(
                                                    source_embedding,
                                                    parameter(&tape, layout.graphon_nodes + source),
                                                ),
                                                temporal_next[time][source][channel],
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    let weights = tape_softmax(&tape, &logits);
                                    for source in 0..nodes {
                                        adaptive[channel] = tape.add(
                                            adaptive[channel],
                                            tape.mul(
                                                weights[source],
                                                temporal_next[time][source][channel],
                                            ),
                                        );
                                    }
                                }
                                for output_channel in 0..hidden {
                                    let mut value = parameter(&tape, graph_bias + output_channel);
                                    for input_channel in 0..hidden {
                                        value = tape.add(
                                            value,
                                            tape.mul(
                                                parameter(
                                                    &tape,
                                                    graph_projection
                                                        + input_channel * hidden
                                                        + output_channel,
                                                ),
                                                adaptive[input_channel],
                                            ),
                                        );
                                    }
                                    next[time][target][output_channel] = tape.tanh(value);
                                }
                            }
                        }
                        short_sequence = next;
                    }
                    let mut fused = vec![0usize; hidden];
                    for channel in 0..hidden {
                        let gate = tape.sigmoid(parameter(&tape, layout.recurrence + 1 + channel));
                        let short = short_sequence[short_sequence.len() - 1][node][channel];
                        fused[channel] = tape.add(
                            tape.mul(gate, long[channel]),
                            tape.mul(
                                tape.add(tape.constant(1.0), tape.mul(tape.constant(-1.0), gate)),
                                short,
                            ),
                        );
                    }
                    fused
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
        (tape, outputs, router_weights)
    }
}

fn tape_linear(
    tape: &AutodiffTape,
    parameters: &[f64],
    offset: usize,
    input: &[usize],
    input_width: usize,
    output_width: usize,
) -> Vec<usize> {
    (0..output_width)
        .map(|output| {
            let mut value = tape.parameter(
                offset + input_width * output_width + output,
                parameters[offset + input_width * output_width + output],
            );
            for (index, input_value) in input.iter().enumerate().take(input_width) {
                value = tape.add(
                    value,
                    tape.mul(
                        tape.parameter(
                            offset + index * output_width + output,
                            parameters[offset + index * output_width + output],
                        ),
                        *input_value,
                    ),
                );
            }
            value
        })
        .collect()
}
fn tape_dot(tape: &AutodiffTape, left: &[usize], right: &[usize]) -> usize {
    left.iter()
        .zip(right)
        .fold(tape.constant(0.0), |sum, (a, b)| {
            tape.add(sum, tape.mul(*a, *b))
        })
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

fn tape_softmax(tape: &AutodiffTape, logits: &[usize]) -> Vec<usize> {
    let max_logit = logits
        .iter()
        .map(|value| tape.value(*value))
        .fold(f64::NEG_INFINITY, f64::max);
    let shift = tape.constant(-max_logit);
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
        .map(|value| tape.constant(tape.value(*value)))
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
}

#[derive(Clone, Copy)]
struct TapeNode {
    value: f64,
    op: TapeOp,
}

struct AutodiffTape {
    nodes: RefCell<Vec<TapeNode>>,
}

impl AutodiffTape {
    fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
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
        let value = nodes[left].value + nodes[right].value;
        nodes.push(TapeNode {
            value,
            op: TapeOp::Add(left, right),
        });
        index
    }
    fn mul(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = nodes[left].value * nodes[right].value;
        nodes.push(TapeNode {
            value,
            op: TapeOp::Mul(left, right),
        });
        index
    }
    fn div(&self, left: usize, right: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = nodes[left].value / nodes[right].value.max(1e-12);
        nodes.push(TapeNode {
            value,
            op: TapeOp::Div(left, right),
        });
        index
    }
    fn tanh(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = nodes[input].value.tanh();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Tanh(input),
        });
        index
    }
    fn exp(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = nodes[input].value.exp();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Exp(input),
        });
        index
    }
    fn sqrt(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = nodes[input].value.max(1e-12).sqrt();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sqrt(input),
        });
        index
    }
    fn sin(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = nodes[input].value.sin();
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sin(input),
        });
        index
    }
    fn sigmoid(&self, input: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        let value = sigmoid(nodes[input].value);
        nodes.push(TapeNode {
            value,
            op: TapeOp::Sigmoid(input),
        });
        index
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
            || config.graph_order == 0
            || config.experts == 0
            || config.periodicity == 0
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
        let (mean, scale) = target_center_scale(&frame.target);
        let normalized = frame
            .target
            .iter()
            .map(|row| row.iter().map(|v| (v - mean) / scale).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(
            frame.node_ids.len(),
            self.config.hidden_size,
            self.config.attention_heads,
            self.config.periodicity,
            frame.horizon,
            self.config.experts,
            self.config.graph_order,
            0x5354_474d_4f45,
        );
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
            // representations for direct multi-horizon forecasting.
            for _ in 0..(self.config.epochs / 4).max(1) {
                for start in 0..sample_count {
                    let cutoff = start + self.config.lookback;
                    state.train_masked_subseries_reconstruction(
                        &normalized[start..cutoff],
                        self.config.learning_rate,
                        self.config.weight_decay,
                    )?;
                }
            }
        }
        for _ in 0..self.config.epochs {
            for start in 0..sample_count {
                let cutoff = start + self.config.lookback;
                let baseline = &normalized[cutoff - 1];
                let targets: Vec<Vec<f64>> = normalized[cutoff..cutoff + frame.horizon]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(baseline)
                            .map(|(value, base)| value - base)
                            .collect()
                    })
                    .collect();
                if let Some(environments) = &spatial_shift_environments {
                    // First learn each expert graphon on the observed traffic
                    // environment.  The episodic pass then hides the
                    // environment's designated expert, forcing the router to
                    // mix the remaining graphons for that shifted relation.
                    state.train_example(
                        &self.config.profile,
                        &normalized[start..cutoff],
                        &adjacency,
                        &targets,
                        None,
                        self.config.learning_rate * 0.5,
                        self.config.weight_decay,
                    );
                    let environment = environments[cutoff % environments.len()];
                    state.train_example(
                        &self.config.profile,
                        &normalized[start..cutoff],
                        &adjacency,
                        &targets,
                        Some(environment),
                        self.config.learning_rate * 0.5,
                        self.config.weight_decay,
                    );
                } else {
                    state.train_example(
                        &self.config.profile,
                        &normalized[start..cutoff],
                        &adjacency,
                        &targets,
                        None,
                        self.config.learning_rate,
                        self.config.weight_decay,
                    );
                }
            }
        }
        self.trainable_state = Some(state);
        self.node_ids = frame.node_ids.clone();
        self.frequency = frame.frequency.clone();
        self.horizon = frame.horizon;
        self.adjacency = Some(adjacency);
        self.history = normalized;
        self.target_mean = mean;
        self.target_scale = scale;
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
        let mut history = self.history.clone();
        let mut output = Vec::with_capacity(horizon);
        while output.len() < horizon {
            let start = history.len() - self.config.lookback;
            // Spatial-shift inference dynamically recomputes every expert
            // graphon and its router weight from this observed window.  The
            // fitted parameters remain fixed, matching the paper's testing
            // policy and preventing hidden test-time optimization.
            let rows = state.predict_window(&self.config.profile, &history[start..], adjacency);
            for row in rows.iter().take(self.horizon.min(horizon - output.len())) {
                let baseline = history.last().expect("forecast history is non-empty");
                let next = row
                    .iter()
                    .zip(baseline)
                    .map(|(delta, value)| delta + value)
                    .collect::<Vec<_>>();
                output.push(next.clone());
                history.push(next);
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

    pub fn architecture_report(&self) -> PaperGraphTransformerArchitectureReport {
        let components = match self.config.profile {
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
                "masked_patch_encoder_decoder_pretraining",
                "dilated_long_trend_extractor",
                "forward_backward_adaptive_graph_day_week_periodicity",
                "gated_temporal_adaptive_graph_short_term_branch",
                "short_term_graph_forecaster_fusion",
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
        .collect();
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
    indices.truncate((patches * 3).div_ceil(4));
    indices.sort_unstable();
    indices
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
                periodicity: 6,
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
            let report = model.architecture_report();
            let required_component = match profile {
                GraphTransformerProfile::HeterogeneousMoE => "moe_load_balancing_loss",
                GraphTransformerProfile::EfficientHighOrder => {
                    "recursive_pointwise_high_order_interaction"
                }
                GraphTransformerProfile::LongShortFusion => {
                    "masked_patch_encoder_decoder_pretraining"
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
            periodicity: 6,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
            ..PaperGraphTransformerConfig::default()
        };
        let initial =
            TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 2, 2, 0x5354_474d_4f45);
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
        let layout = GraphParameterLayout::new(3, 4, 2, 2, 1, 6);
        let width = 4 * (4 + 1);
        let block_width = 3 * width;
        assert_eq!(layout.temporal_k, layout.temporal_q + block_width);
        assert_eq!(layout.temporal_v, layout.temporal_k + block_width);
        assert_eq!(layout.spatial_q, layout.temporal_v + block_width);
        assert_eq!(layout.spatial_k, layout.spatial_q + block_width);
        assert_eq!(layout.spatial_v, layout.spatial_k + block_width);
        assert_eq!(layout.shortest_path_bias, layout.spatial_v + block_width);
    }

    #[test]
    fn efficient_high_order_profile_uses_stgformer_scaling_normalized_attention() {
        let frame = traffic_style_fixture_frame();
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 2, 2, 2, 19);
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
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial =
            TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 2, 2, 0x5354_474d_4f45);
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
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial =
            TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 2, 2, 0x5354_474d_4f45);
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
            periodicity: 6,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial =
            TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 2, 2, 0x5354_474d_4f45);
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        assert!(
            trained.parameters[layout.graphon_nodes..layout.graphon_time]
                .iter()
                .zip(&initial.parameters[layout.graphon_nodes..layout.graphon_time])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
        for range in [
            layout.lsttn_adaptive_source..layout.lsttn_adaptive_target,
            layout.lsttn_adaptive_target..layout.graphon_nodes,
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
            epochs: 2,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial =
            TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 2, 2, 0x5354_474d_4f45);
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
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial =
            TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 2, 2, 0x5354_474d_4f45);
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
    }
}
