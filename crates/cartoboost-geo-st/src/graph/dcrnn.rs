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

