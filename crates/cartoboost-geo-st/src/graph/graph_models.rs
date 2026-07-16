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
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        let mut cuda_executor = if self.config.profile == GraphTransformerProfile::LongShortFusion
            && backend.selected == "cuda"
        {
            Some(CudaLsttnTensorExecutor::new(&state, &adjacency)?)
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
                    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
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
                    #[cfg(not(all(
                        feature = "cuda",
                        any(target_os = "linux", target_os = "windows")
                    )))]
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
            && backend.selected != "cuda"
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
                    let scheduler_steps = [1usize, 18, 36, 54, 72]
                        .into_iter()
                        .filter(|milestone| *milestone <= epoch)
                        .count();
                    let epoch_learning_rate =
                        self.config.learning_rate * 0.5_f64.powi(scheduler_steps as i32);
                    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
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
                                &normalized[cutoff..cutoff + frame.horizon]
                            })
                            .collect::<Vec<_>>();
                        executor.upload_supervised_batch(&windows, &targets, 1)?;
                        executor.supervised_forward(&state, starts.len(), 1, starts[0], true)?;
                        executor.supervised_backward(&state, starts.len(), 1, starts[0], true)?;
                        executor.freeze_pretrained_transformer_gradients(&state)?;
                        executor.adamw_supervised_step(
                            state.steps + 1,
                            epoch_learning_rate,
                            self.config.weight_decay,
                        )?;
                        state.steps += 1;
                        let loss = executor.mean_supervised_loss()?;
                        executor.synchronize_portable_state(&mut state)?;
                        Some(loss)
                    } else {
                        None
                    };
                    #[cfg(not(all(
                        feature = "cuda",
                        any(target_os = "linux", target_os = "windows")
                    )))]
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
                        state.adamw_step(&gradients, epoch_learning_rate, self.config.weight_decay);
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
            && self.config.backend.selected == "metal"
        {
            components.push(format!(
                "{}_full_graph_training_and_inference",
                self.config.backend.selected
            ));
        }
        if self.config.profile == GraphTransformerProfile::LongShortFusion
            && self.config.backend.selected == "cuda"
        {
            components.push("cuda_tensor_training_pending".to_string());
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

