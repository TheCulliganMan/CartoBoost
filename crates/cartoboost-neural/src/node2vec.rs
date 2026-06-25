use crate::error::{NeuralError, Result};
use crate::graphsage::GraphSageEmbedding;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const NODE2VEC_ARTIFACT_TYPE: &str = "cartoboost.neural.node2vec_encoder";
pub const NODE2VEC_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node2VecEncoderArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub node_count: usize,
    pub output_dim: usize,
    pub config: Node2VecConfig,
    pub embeddings: Vec<Vec<f32>>,
    pub loss_curve: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node2VecConfig {
    pub dim: usize,
    pub walk_length: usize,
    pub walks_per_node: usize,
    pub window_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub min_learning_rate: f32,
    pub negative_samples: usize,
    pub p: f32,
    pub q: f32,
    pub seed: u64,
    pub l2_regularization: f32,
    pub normalize: bool,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Self {
            dim: 16,
            walk_length: 16,
            walks_per_node: 8,
            window_size: 5,
            epochs: 3,
            learning_rate: 0.025,
            min_learning_rate: 0.0001,
            negative_samples: 5,
            p: 1.0,
            q: 1.0,
            seed: 0xA2B2_C2D2_E2F2_1234,
            l2_regularization: 0.0,
            normalize: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node2VecLoss {
    epoch_losses: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AliasSampler {
    probabilities: Vec<f32>,
    aliases: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RandomWalkGenerator {
    pub config: Node2VecConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node2VecTrainer {
    pub config: Node2VecConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgeEmbeddingFeatures {
    pub values: Vec<Vec<f32>>,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgeEmbeddingModel {
    pub node_count: usize,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingFeatureTransformer {
    pub model: EdgeEmbeddingModel,
}

impl Node2VecLoss {
    pub fn values(&self) -> &[f32] {
        &self.epoch_losses
    }
}

impl AliasSampler {
    pub fn new(weights: &[f32]) -> Result<Self> {
        if weights.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "alias sampler requires at least one weight".to_string(),
            ));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(NeuralError::InvalidArgument(
                "alias sampler weights must be finite and non-negative".to_string(),
            ));
        }
        let total = weights.iter().sum::<f32>();
        if total <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "alias sampler weights must have positive total mass".to_string(),
            ));
        }
        let n = weights.len();
        let mut scaled = weights
            .iter()
            .map(|weight| *weight * n as f32 / total)
            .collect::<Vec<_>>();
        let mut probabilities = vec![0.0; n];
        let mut aliases = (0..n).collect::<Vec<_>>();
        let mut small = Vec::new();
        let mut large = Vec::new();
        for (idx, &value) in scaled.iter().enumerate() {
            if value < 1.0 {
                small.push(idx);
            } else {
                large.push(idx);
            }
        }
        while let (Some(less), Some(more)) = (small.pop(), large.pop()) {
            probabilities[less] = scaled[less];
            aliases[less] = more;
            scaled[more] = scaled[more] + scaled[less] - 1.0;
            if scaled[more] < 1.0 {
                small.push(more);
            } else {
                large.push(more);
            }
        }
        for idx in large.into_iter().chain(small) {
            probabilities[idx] = 1.0;
            aliases[idx] = idx;
        }
        Ok(Self {
            probabilities,
            aliases,
        })
    }

    pub fn sample_many(&self, seed: u64, count: usize) -> Vec<usize> {
        let mut rng = SplitMix64::from_seed(seed);
        (0..count).map(|_| self.sample_with_rng(&mut rng)).collect()
    }

    fn sample_with_rng(&self, rng: &mut SplitMix64) -> usize {
        let idx = rng.next_usize(self.probabilities.len());
        if rng.next_unit() < self.probabilities[idx] {
            idx
        } else {
            self.aliases[idx]
        }
    }
}

impl RandomWalkGenerator {
    pub fn new(config: Node2VecConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    pub fn generate(
        &self,
        node_count: usize,
        edges: &[(usize, usize)],
        edge_weights: Option<&[f32]>,
    ) -> Result<Vec<Vec<usize>>> {
        validate_node_count(node_count)?;
        let weights = validate_weights(edges.len(), edge_weights)?;
        let graph = WeightedGraph::from_edges(node_count, edges, &weights)?;
        Ok(generate_walks(&self.config, &graph))
    }
}

impl Node2VecTrainer {
    pub fn new(config: Node2VecConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    pub fn fit(
        &self,
        node_count: usize,
        edges: &[(usize, usize)],
        edge_weights: Option<&[f32]>,
    ) -> Result<EdgeEmbeddingModel> {
        let mut encoder = Node2VecEncoder::new(self.config.clone())?;
        let embeddings = encoder.fit(node_count, edges, edge_weights)?;
        EdgeEmbeddingModel::new(embeddings.into_inner())
    }
}

impl EdgeEmbeddingModel {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Result<Self> {
        if embeddings.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "edge embedding model requires at least one node".to_string(),
            ));
        }
        let width = embeddings[0].len();
        if width == 0 {
            return Err(NeuralError::InvalidArgument(
                "edge embedding vectors must have positive width".to_string(),
            ));
        }
        if embeddings
            .iter()
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
        {
            return Err(NeuralError::InvalidArgument(
                "edge embedding rows must have consistent finite width".to_string(),
            ));
        }
        Ok(Self {
            node_count: embeddings.len(),
            embeddings,
        })
    }

    pub fn from_encoder(encoder: &Node2VecEncoder) -> Result<Self> {
        Self::new(encoder.encode()?.into_inner())
    }

    pub fn transform_edges(
        &self,
        edges: &[(usize, usize)],
        graph_edges: &[(usize, usize)],
    ) -> Result<EdgeEmbeddingFeatures> {
        EmbeddingFeatureTransformer::new(self.clone()).transform_edges(edges, graph_edges)
    }
}

impl EmbeddingFeatureTransformer {
    pub fn new(model: EdgeEmbeddingModel) -> Self {
        Self { model }
    }

    pub fn transform_edges(
        &self,
        edges: &[(usize, usize)],
        graph_edges: &[(usize, usize)],
    ) -> Result<EdgeEmbeddingFeatures> {
        let degrees = neighborhood_degrees(self.model.node_count, graph_edges)?;
        let values = edges
            .iter()
            .map(|&(origin, destination)| {
                if origin >= self.model.node_count || destination >= self.model.node_count {
                    return Err(NeuralError::InvalidArgument(
                        "edge feature endpoint must be in [0, node_count)".to_string(),
                    ));
                }
                let origin_vec = &self.model.embeddings[origin];
                let destination_vec = &self.model.embeddings[destination];
                let reverse_similarity = cosine(origin_vec, destination_vec);
                let dot_product = dot(origin_vec, destination_vec);
                let origin_density = degrees[origin] as f32 / self.model.node_count.max(1) as f32;
                let destination_density =
                    degrees[destination] as f32 / self.model.node_count.max(1) as f32;
                let local_context = 0.5 * (origin_density + destination_density);
                let mut row = Vec::with_capacity(origin_vec.len() + destination_vec.len() + 5);
                row.extend(origin_vec.iter().copied());
                row.extend(destination_vec.iter().copied());
                row.push(reverse_similarity);
                row.push(dot_product);
                row.push(origin_density);
                row.push(destination_density);
                row.push(local_context);
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut feature_names = Vec::with_capacity(self.model.embeddings[0].len() * 2 + 5);
        for dim in 0..self.model.embeddings[0].len() {
            feature_names.push(format!("origin_embedding_{dim}"));
        }
        for dim in 0..self.model.embeddings[0].len() {
            feature_names.push(format!("destination_embedding_{dim}"));
        }
        feature_names.extend([
            "edge_reverse_similarity".to_string(),
            "edge_dot_product".to_string(),
            "origin_neighborhood_density".to_string(),
            "destination_neighborhood_density".to_string(),
            "local_graph_context".to_string(),
        ]);
        Ok(EdgeEmbeddingFeatures {
            values,
            feature_names,
        })
    }

    pub fn append_to_dense(
        &self,
        dense_rows: &[Vec<f32>],
        edges: &[(usize, usize)],
        graph_edges: &[(usize, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        if dense_rows.len() != edges.len() {
            return Err(NeuralError::InvalidArgument(
                "dense row count must match edge row count".to_string(),
            ));
        }
        let features = self.transform_edges(edges, graph_edges)?;
        Ok(dense_rows
            .iter()
            .zip(features.values)
            .map(|(dense, extra)| {
                let mut row = dense.clone();
                row.extend(extra);
                row
            })
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct Node2VecEncoder {
    config: Node2VecConfig,
    embeddings: Vec<Vec<f32>>,
    context_embeddings: Vec<Vec<f32>>,
    losses: Vec<f32>,
}

impl Node2VecEncoder {
    pub fn new(config: Node2VecConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            embeddings: Vec::new(),
            context_embeddings: Vec::new(),
            losses: Vec::new(),
        })
    }

    pub fn fit(
        &mut self,
        node_count: usize,
        edges: &[(usize, usize)],
        edge_weights: Option<&[f32]>,
    ) -> Result<GraphSageEmbedding> {
        validate_node_count(node_count)?;
        let weights = validate_weights(edges.len(), edge_weights)?;
        let adjacency = WeightedGraph::from_edges(node_count, edges, &weights)?;
        let walks = generate_walks(&self.config, &adjacency);
        let mut model = SkipGramState::new(node_count, &self.config);
        self.losses = model.fit(&self.config, &walks, &adjacency);
        self.embeddings = finalize_embeddings(model.embeddings, self.config.normalize);
        self.context_embeddings = model.context_embeddings;
        Ok(GraphSageEmbedding::new(self.embeddings.clone()))
    }

    pub fn encode(&self) -> Result<GraphSageEmbedding> {
        if self.embeddings.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "Node2VecEncoder must be fitted before encode".to_string(),
            ));
        }
        Ok(GraphSageEmbedding::new(self.embeddings.clone()))
    }

    pub fn output_dim(&self) -> usize {
        self.config.dim
    }

    pub fn node_count(&self) -> usize {
        self.embeddings.len()
    }

    pub fn config(&self) -> Node2VecConfig {
        self.config.clone()
    }

    pub fn loss_curve(&self) -> Node2VecLoss {
        Node2VecLoss {
            epoch_losses: self.losses.clone(),
        }
    }

    pub fn to_artifact(&self) -> Node2VecEncoderArtifact {
        Node2VecEncoderArtifact {
            artifact_type: NODE2VEC_ARTIFACT_TYPE.to_string(),
            artifact_version: NODE2VEC_ARTIFACT_VERSION,
            node_count: self.embeddings.len(),
            output_dim: self.config.dim,
            config: self.config.clone(),
            embeddings: self.embeddings.clone(),
            loss_curve: self.losses.clone(),
        }
    }

    pub fn to_artifact_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.to_artifact())?)
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_artifact_json()?)?;
        Ok(())
    }

    pub fn from_artifact(artifact: Node2VecEncoderArtifact) -> Result<Self> {
        validate_artifact(&artifact)?;
        Ok(Self {
            config: artifact.config,
            embeddings: artifact.embeddings,
            context_embeddings: Vec::new(),
            losses: artifact.loss_curve,
        })
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }
}

#[derive(Debug, Clone)]
struct WeightedGraph {
    outgoing: Vec<Vec<(usize, f32)>>,
    connectivity: Vec<HashSet<usize>>,
}

impl WeightedGraph {
    fn from_edges(node_count: usize, edges: &[(usize, usize)], weights: &[f32]) -> Result<Self> {
        let mut outgoing_maps = vec![Vec::<(usize, f32)>::new(); node_count];
        let mut connectivity = vec![HashSet::new(); node_count];
        for (index, &(source, target)) in edges.iter().enumerate() {
            if source >= node_count || target >= node_count {
                return Err(NeuralError::InvalidArgument(
                    "edge endpoint must be in [0, node_count)".to_string(),
                ));
            }
            let weight = weights[index];
            if weight == 0.0 {
                continue;
            }
            outgoing_maps[source].push((target, weight));
            connectivity[source].insert(target);
            connectivity[target].insert(source);
        }
        Ok(Self {
            outgoing: outgoing_maps,
            connectivity,
        })
    }

    fn weighted_degree(&self, node: usize) -> f32 {
        self.outgoing[node]
            .iter()
            .map(|(_target, weight)| *weight)
            .sum()
    }

    fn has_edge_or_reverse(&self, left: usize, right: usize) -> bool {
        self.connectivity[left].contains(&right)
    }
}

#[derive(Debug)]
struct SkipGramState {
    embeddings: Vec<Vec<f32>>,
    context_embeddings: Vec<Vec<f32>>,
    rng: SplitMix64,
}

impl SkipGramState {
    fn new(node_count: usize, config: &Node2VecConfig) -> Self {
        let mut rng = SplitMix64::from_seed(config.seed ^ 0x5EED_5EED);
        let scale = 0.5 / config.dim.max(1) as f32;
        let mut embeddings = vec![vec![0.0; config.dim]; node_count];
        for row in &mut embeddings {
            for value in row {
                *value = (rng.next_unit() - 0.5) * 2.0 * scale;
            }
        }
        Self {
            embeddings,
            context_embeddings: vec![vec![0.0; config.dim]; node_count],
            rng,
        }
    }

    fn fit(
        &mut self,
        config: &Node2VecConfig,
        walks: &[Vec<usize>],
        graph: &WeightedGraph,
    ) -> Vec<f32> {
        let pairs = context_pairs(walks, config.window_size);
        if pairs.is_empty() {
            return vec![0.0; config.epochs];
        }
        let negative_distribution = AliasSampler::new(&negative_distribution(graph))
            .expect("negative distribution is positive");
        let total_steps = (pairs.len() * config.epochs).max(1);
        let mut step = 0usize;
        let mut losses = Vec::with_capacity(config.epochs);
        let mut order = (0..pairs.len()).collect::<Vec<_>>();

        for _ in 0..config.epochs {
            shuffle(&mut order, &mut self.rng);
            let mut epoch_loss = 0.0;
            for &pair_index in &order {
                let (source, target) = pairs[pair_index];
                let progress = step as f32 / total_steps as f32;
                let learning_rate = config
                    .min_learning_rate
                    .max(config.learning_rate * (1.0 - progress));
                epoch_loss += self.update_pair(source, target, 1.0, learning_rate, config);
                for _ in 0..config.negative_samples {
                    let negative = negative_distribution.sample_with_rng(&mut self.rng);
                    if negative == target {
                        continue;
                    }
                    epoch_loss += self.update_pair(source, negative, 0.0, learning_rate, config);
                }
                step += 1;
            }
            losses.push(epoch_loss / pairs.len().max(1) as f32);
        }
        losses
    }

    fn update_pair(
        &mut self,
        source: usize,
        target: usize,
        label: f32,
        learning_rate: f32,
        config: &Node2VecConfig,
    ) -> f32 {
        let source_vec = self.embeddings[source].clone();
        let target_vec = self.context_embeddings[target].clone();
        let score = sigmoid(dot(&source_vec, &target_vec));
        let gradient = learning_rate * (label - score);
        for index in 0..config.dim {
            self.embeddings[source][index] += gradient * target_vec[index];
            self.context_embeddings[target][index] += gradient * source_vec[index];
        }
        if config.l2_regularization > 0.0 {
            let shrink = 1.0 - learning_rate * config.l2_regularization;
            for index in 0..config.dim {
                self.embeddings[source][index] *= shrink;
                self.context_embeddings[target][index] *= shrink;
            }
        }
        if label == 1.0 {
            -score.max(f32::EPSILON).ln()
        } else {
            -(1.0 - score).max(f32::EPSILON).ln()
        }
    }
}

fn generate_walks(config: &Node2VecConfig, graph: &WeightedGraph) -> Vec<Vec<usize>> {
    let node_count = graph.outgoing.len();
    let total = node_count * config.walks_per_node;
    (0..total)
        .into_par_iter()
        .map(|walk_index| {
            let start = walk_index % node_count;
            let mut rng = SplitMix64::from_seed(
                config.seed
                    ^ 0xC0FF_EE77_D15A_5EED
                    ^ (walk_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            generate_walk(config, graph, start, &mut rng)
        })
        .collect()
}

fn generate_walk(
    config: &Node2VecConfig,
    graph: &WeightedGraph,
    start: usize,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let mut walk = vec![start];
    while walk.len() < config.walk_length {
        let current = *walk.last().expect("walk contains start");
        if graph.outgoing[current].is_empty() {
            break;
        }
        let previous = if walk.len() > 1 {
            Some(walk[walk.len() - 2])
        } else {
            None
        };
        walk.push(sample_next(config, graph, previous, current, rng));
    }
    walk
}

fn sample_next(
    config: &Node2VecConfig,
    graph: &WeightedGraph,
    previous: Option<usize>,
    current: usize,
    rng: &mut SplitMix64,
) -> usize {
    let neighbors = &graph.outgoing[current];
    let weights = neighbors
        .iter()
        .map(|&(candidate, weight)| {
            let bias = previous.map_or(1.0, |prev| transition_bias(config, graph, prev, candidate));
            weight * bias
        })
        .collect::<Vec<_>>();
    if weights.iter().sum::<f32>() <= 0.0 {
        return neighbors[rng.next_usize(neighbors.len())].0;
    }
    let sampler = AliasSampler::new(&weights).expect("positive transition weights");
    neighbors[sampler.sample_with_rng(rng)].0
}

fn transition_bias(
    config: &Node2VecConfig,
    graph: &WeightedGraph,
    previous: usize,
    candidate: usize,
) -> f32 {
    if candidate == previous {
        1.0 / config.p
    } else if graph.has_edge_or_reverse(previous, candidate) {
        1.0
    } else {
        1.0 / config.q
    }
}

fn context_pairs(walks: &[Vec<usize>], window_size: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for walk in walks {
        for (index, &source) in walk.iter().enumerate() {
            let left = index.saturating_sub(window_size);
            let right = (index + window_size + 1).min(walk.len());
            for (context_index, &target) in walk.iter().enumerate().take(right).skip(left) {
                if context_index != index {
                    pairs.push((source, target));
                }
            }
        }
    }
    pairs
}

fn negative_distribution(graph: &WeightedGraph) -> Vec<f32> {
    let mut values = graph
        .outgoing
        .iter()
        .enumerate()
        .map(|(node, _)| graph.weighted_degree(node).max(1.0).powf(0.75))
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    if total == 0.0 {
        let uniform = 1.0 / values.len().max(1) as f32;
        values.fill(uniform);
        return values;
    }
    for value in &mut values {
        *value /= total;
    }
    values
}

fn finalize_embeddings(mut embeddings: Vec<Vec<f32>>, normalize: bool) -> Vec<Vec<f32>> {
    if !normalize {
        return embeddings;
    }
    for row in &mut embeddings {
        let norm = dot(row, row).sqrt();
        if norm > 0.0 {
            for value in row {
                *value /= norm;
            }
        }
    }
    embeddings
}

fn validate_config(config: &Node2VecConfig) -> Result<()> {
    if config.dim == 0 {
        return Err(NeuralError::InvalidArgument(
            "dim must be positive".to_string(),
        ));
    }
    if config.walk_length == 0 || config.walks_per_node == 0 || config.window_size == 0 {
        return Err(NeuralError::InvalidArgument(
            "walk_length, walks_per_node, and window_size must be positive".to_string(),
        ));
    }
    if config.epochs == 0 || config.negative_samples == 0 {
        return Err(NeuralError::InvalidArgument(
            "epochs and negative_samples must be positive".to_string(),
        ));
    }
    if config.learning_rate <= 0.0 || config.min_learning_rate < 0.0 {
        return Err(NeuralError::InvalidArgument(
            "learning rates must be positive/non-negative".to_string(),
        ));
    }
    if config.p <= 0.0 || config.q <= 0.0 {
        return Err(NeuralError::InvalidArgument(
            "p and q must be positive".to_string(),
        ));
    }
    if config.l2_regularization < 0.0 {
        return Err(NeuralError::InvalidArgument(
            "l2_regularization must be non-negative".to_string(),
        ));
    }
    Ok(())
}

fn validate_node_count(node_count: usize) -> Result<()> {
    if node_count == 0 {
        return Err(NeuralError::InvalidArgument(
            "node_count must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_weights(edge_count: usize, edge_weights: Option<&[f32]>) -> Result<Vec<f32>> {
    match edge_weights {
        Some(weights) if weights.len() == edge_count => {
            if weights
                .iter()
                .any(|weight| *weight < 0.0 || !weight.is_finite())
            {
                return Err(NeuralError::InvalidArgument(
                    "edge_weights must be finite and non-negative".to_string(),
                ));
            }
            Ok(weights.to_vec())
        }
        Some(_) => Err(NeuralError::InvalidArgument(
            "edge_weights length must match edge count".to_string(),
        )),
        None => Ok(vec![1.0; edge_count]),
    }
}

fn validate_artifact(artifact: &Node2VecEncoderArtifact) -> Result<()> {
    if artifact.artifact_type != NODE2VEC_ARTIFACT_TYPE {
        return Err(NeuralError::InvalidArgument(format!(
            "unexpected node2vec artifact type: {}",
            artifact.artifact_type
        )));
    }
    if artifact.artifact_version != NODE2VEC_ARTIFACT_VERSION {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported node2vec artifact version: {}",
            artifact.artifact_version
        )));
    }
    validate_config(&artifact.config)?;
    if artifact.output_dim != artifact.config.dim {
        return Err(NeuralError::InvalidArgument(
            "node2vec artifact output_dim must match config.dim".to_string(),
        ));
    }
    if artifact.node_count != artifact.embeddings.len() {
        return Err(NeuralError::InvalidArgument(
            "node2vec artifact node_count must match embeddings length".to_string(),
        ));
    }
    for row in &artifact.embeddings {
        if row.len() != artifact.output_dim {
            return Err(NeuralError::InvalidArgument(
                "node2vec artifact embedding row width mismatch".to_string(),
            ));
        }
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let left_norm = dot(left, left).sqrt();
    let right_norm = dot(right, right).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot(left, right) / (left_norm * right_norm)
    }
}

fn neighborhood_degrees(node_count: usize, edges: &[(usize, usize)]) -> Result<Vec<usize>> {
    let mut degrees = vec![0usize; node_count];
    for &(source, target) in edges {
        if source >= node_count || target >= node_count {
            return Err(NeuralError::InvalidArgument(
                "graph edge endpoint must be in [0, node_count)".to_string(),
            ));
        }
        degrees[source] += 1;
        degrees[target] += 1;
    }
    Ok(degrees)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn shuffle(values: &mut [usize], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_index = rng.next_usize(index + 1);
        values.swap(index, swap_index);
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn next_unit(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((u64::MAX as f64) + 1.0) as f32;
        self.next_u64() as f32 * SCALE
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() % max as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> Node2VecConfig {
        Node2VecConfig {
            dim: 4,
            walk_length: 5,
            walks_per_node: 3,
            window_size: 2,
            epochs: 2,
            negative_samples: 2,
            seed: 17,
            ..Node2VecConfig::default()
        }
    }

    #[test]
    fn node2vec_fits_directed_weighted_embeddings() {
        let mut encoder = Node2VecEncoder::new(small_config()).unwrap();

        let embeddings = encoder
            .fit(
                4,
                &[(0, 1), (1, 2), (2, 0), (2, 3)],
                Some(&[1.0, 2.0, 1.0, 3.0]),
            )
            .unwrap();

        assert_eq!(embeddings.node_count(), 4);
        assert_eq!(embeddings.dim(), 4);
        assert_eq!(encoder.loss_curve().values().len(), 2);
        assert!(encoder.loss_curve().values()[0].is_finite());
    }

    #[test]
    fn alias_sampler_is_seed_deterministic_and_weighted() {
        let sampler = AliasSampler::new(&[0.0, 1.0, 3.0]).unwrap();

        let left = sampler.sample_many(42, 16);
        let right = sampler.sample_many(42, 16);

        assert_eq!(left, right);
        assert!(left.iter().all(|&idx| idx == 1 || idx == 2));
        assert!(left.contains(&2));
    }

    #[test]
    fn deterministic_seed_reproduces_embeddings_and_loss() {
        let mut left = Node2VecEncoder::new(small_config()).unwrap();
        let mut right = Node2VecEncoder::new(small_config()).unwrap();

        let left_embeddings = left
            .fit(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], None)
            .unwrap();
        let right_embeddings = right
            .fit(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], None)
            .unwrap();

        assert_eq!(left_embeddings, right_embeddings);
        assert_eq!(left.loss_curve(), right.loss_curve());
    }

    #[test]
    fn random_walk_generator_is_seed_deterministic() {
        let generator = RandomWalkGenerator::new(small_config()).unwrap();
        let left = generator
            .generate(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], None)
            .unwrap();
        let right = generator
            .generate(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], None)
            .unwrap();

        assert_eq!(left, right);
    }

    #[test]
    fn random_walk_generator_parallel_order_is_stable() {
        let generator = RandomWalkGenerator::new(Node2VecConfig {
            walk_length: 3,
            walks_per_node: 2,
            ..small_config()
        })
        .unwrap();
        let walks = generator
            .generate(3, &[(0, 1), (1, 2), (2, 0)], None)
            .unwrap();

        assert_eq!(walks.len(), 6);
        assert_eq!(
            walks.iter().map(|walk| walk[0]).collect::<Vec<_>>(),
            vec![0, 1, 2, 0, 1, 2]
        );
    }

    #[test]
    fn node2vec_trainer_builds_edge_embedding_model() {
        let trainer = Node2VecTrainer::new(Node2VecConfig {
            dim: 3,
            ..small_config()
        })
        .unwrap();
        let model = trainer
            .fit(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], None)
            .unwrap();

        assert_eq!(model.node_count, 4);
        assert_eq!(model.embeddings[0].len(), 3);
    }

    #[test]
    fn different_seed_changes_embeddings() {
        let mut left = Node2VecEncoder::new(Node2VecConfig {
            seed: 1,
            ..small_config()
        })
        .unwrap();
        let mut right = Node2VecEncoder::new(Node2VecConfig {
            seed: 2,
            ..small_config()
        })
        .unwrap();

        let left_embeddings = left.fit(4, &[(0, 1), (1, 2), (2, 0)], None).unwrap();
        let right_embeddings = right.fit(4, &[(0, 1), (1, 2), (2, 0)], None).unwrap();

        assert_ne!(left_embeddings, right_embeddings);
    }

    #[test]
    fn normalized_embeddings_have_unit_norm() {
        let mut encoder = Node2VecEncoder::new(Node2VecConfig {
            normalize: true,
            ..small_config()
        })
        .unwrap();
        let embeddings = encoder.fit(3, &[(0, 1), (1, 2), (2, 0)], None).unwrap();

        for row in embeddings.vectors() {
            let norm = dot(row, row).sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn non_normalized_embeddings_keep_raw_scale() {
        let mut encoder = Node2VecEncoder::new(Node2VecConfig {
            normalize: false,
            ..small_config()
        })
        .unwrap();
        let embeddings = encoder.fit(3, &[(0, 1), (1, 2), (2, 0)], None).unwrap();

        assert!(embeddings
            .vectors()
            .iter()
            .any(|row| (dot(row, row).sqrt() - 1.0).abs() > 1e-3));
    }

    #[test]
    fn empty_edge_graph_returns_stable_initial_embeddings_and_zero_loss() {
        let mut encoder = Node2VecEncoder::new(small_config()).unwrap();
        let embeddings = encoder.fit(3, &[], None).unwrap();

        assert_eq!(embeddings.node_count(), 3);
        assert_eq!(embeddings.dim(), 4);
        assert_eq!(encoder.loss_curve().values(), &[0.0, 0.0]);
    }

    #[test]
    fn encode_before_fit_is_rejected() {
        let encoder = Node2VecEncoder::new(small_config()).unwrap();

        assert!(encoder.encode().is_err());
    }

    #[test]
    fn validates_config_parameters() {
        let invalid_configs = [
            Node2VecConfig {
                dim: 0,
                ..small_config()
            },
            Node2VecConfig {
                walk_length: 0,
                ..small_config()
            },
            Node2VecConfig {
                walks_per_node: 0,
                ..small_config()
            },
            Node2VecConfig {
                window_size: 0,
                ..small_config()
            },
            Node2VecConfig {
                epochs: 0,
                ..small_config()
            },
            Node2VecConfig {
                negative_samples: 0,
                ..small_config()
            },
            Node2VecConfig {
                learning_rate: 0.0,
                ..small_config()
            },
            Node2VecConfig {
                min_learning_rate: -1.0,
                ..small_config()
            },
            Node2VecConfig {
                p: 0.0,
                ..small_config()
            },
            Node2VecConfig {
                q: 0.0,
                ..small_config()
            },
            Node2VecConfig {
                l2_regularization: -1.0,
                ..small_config()
            },
        ];

        for config in invalid_configs {
            assert!(Node2VecEncoder::new(config).is_err());
        }
    }

    #[test]
    fn validates_fit_inputs() {
        let mut encoder = Node2VecEncoder::new(small_config()).unwrap();

        assert!(encoder.fit(0, &[], None).is_err());
        assert!(encoder.fit(2, &[(0, 2)], None).is_err());
        assert!(encoder.fit(2, &[(0, 1)], Some(&[])).is_err());
        assert!(encoder.fit(2, &[(0, 1)], Some(&[-1.0])).is_err());
        assert!(encoder.fit(2, &[(0, 1)], Some(&[f32::NAN])).is_err());
    }

    #[test]
    fn weighted_graph_accumulates_duplicate_edges_and_drops_zero_weights() {
        let graph =
            WeightedGraph::from_edges(3, &[(0, 1), (0, 1), (0, 2)], &[1.0, 2.5, 0.0]).unwrap();

        assert_eq!(graph.outgoing[0], vec![(1, 1.0), (1, 2.5)]);
        assert_eq!(graph.weighted_degree(0), 3.5);
        assert!(!graph.has_edge_or_reverse(0, 2));
    }

    #[test]
    fn transition_bias_matches_node2vec_p_q_cases() {
        let config = Node2VecConfig {
            p: 2.0,
            q: 4.0,
            ..small_config()
        };
        let graph =
            WeightedGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 0), (1, 3)], &[1.0; 5])
                .unwrap();

        assert_eq!(transition_bias(&config, &graph, 0, 0), 0.5);
        assert_eq!(transition_bias(&config, &graph, 0, 2), 1.0);
        assert_eq!(transition_bias(&config, &graph, 0, 3), 0.25);
    }

    #[test]
    fn walk_generation_respects_outgoing_direction() {
        let config = Node2VecConfig {
            walk_length: 4,
            walks_per_node: 1,
            ..small_config()
        };
        let graph = WeightedGraph::from_edges(3, &[(0, 1)], &[1.0]).unwrap();
        let walks = generate_walks(&config, &graph);

        let walk_from_one = walks
            .iter()
            .find(|walk| walk.first().copied() == Some(1))
            .unwrap();
        assert_eq!(walk_from_one, &vec![1]);
    }

    #[test]
    fn sample_next_uses_edge_weights() {
        let config = small_config();
        let graph = WeightedGraph::from_edges(3, &[(0, 1), (0, 2)], &[0.0, 10.0]).unwrap();
        let mut rng = SplitMix64::from_seed(3);

        for _ in 0..10 {
            assert_eq!(sample_next(&config, &graph, None, 0, &mut rng), 2);
        }
    }

    #[test]
    fn context_pairs_use_symmetric_window_without_self_pairs() {
        let pairs = context_pairs(&[vec![0, 1, 2, 3]], 1);

        assert_eq!(pairs, vec![(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]);
    }

    #[test]
    fn negative_distribution_is_degree_weighted_and_sums_to_one() {
        let graph =
            WeightedGraph::from_edges(3, &[(0, 1), (0, 2), (1, 2)], &[4.0, 4.0, 1.0]).unwrap();
        let probabilities = negative_distribution(&graph);

        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] >= probabilities[2]);
    }

    #[test]
    fn node2vec_artifact_round_trips() {
        let mut encoder = Node2VecEncoder::new(Node2VecConfig {
            dim: 2,
            walk_length: 4,
            walks_per_node: 2,
            window_size: 1,
            epochs: 1,
            negative_samples: 1,
            ..Node2VecConfig::default()
        })
        .unwrap();
        encoder.fit(3, &[(0, 1), (1, 2)], None).unwrap();

        let artifact = encoder.to_artifact();
        let restored = Node2VecEncoder::from_artifact(artifact).unwrap();

        assert_eq!(restored.node_count(), 3);
        assert_eq!(restored.output_dim(), 2);
        assert_eq!(restored.encode().unwrap().dim(), 2);
    }

    #[test]
    fn artifact_validation_rejects_corrupt_metadata() {
        let mut encoder = Node2VecEncoder::new(Node2VecConfig {
            dim: 2,
            walk_length: 4,
            walks_per_node: 2,
            window_size: 1,
            epochs: 1,
            negative_samples: 1,
            ..Node2VecConfig::default()
        })
        .unwrap();
        encoder.fit(3, &[(0, 1), (1, 2)], None).unwrap();

        let mut bad_type = encoder.to_artifact();
        bad_type.artifact_type = "wrong".to_string();
        assert!(Node2VecEncoder::from_artifact(bad_type).is_err());

        let mut bad_width = encoder.to_artifact();
        bad_width.embeddings[0].push(1.0);
        assert!(Node2VecEncoder::from_artifact(bad_width).is_err());

        let mut bad_count = encoder.to_artifact();
        bad_count.node_count += 1;
        assert!(Node2VecEncoder::from_artifact(bad_count).is_err());
    }

    #[test]
    fn artifact_json_file_round_trips() {
        let mut encoder = Node2VecEncoder::new(Node2VecConfig {
            dim: 2,
            walk_length: 4,
            walks_per_node: 2,
            window_size: 1,
            epochs: 1,
            negative_samples: 1,
            ..Node2VecConfig::default()
        })
        .unwrap();
        encoder.fit(3, &[(0, 1), (1, 2)], None).unwrap();

        let file = tempfile::NamedTempFile::new().unwrap();
        encoder.save_artifact_json(file.path()).unwrap();
        let restored = Node2VecEncoder::load_artifact_json(file.path()).unwrap();

        assert_eq!(restored.encode().unwrap(), encoder.encode().unwrap());
    }

    #[test]
    fn edge_embedding_transformer_emits_context_features_and_appends_rows() {
        let model =
            EdgeEmbeddingModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        let transformer = EmbeddingFeatureTransformer::new(model);
        let features = transformer
            .transform_edges(&[(0, 1), (1, 2)], &[(0, 1), (1, 0), (1, 2)])
            .unwrap();

        assert_eq!(
            features.feature_names,
            vec![
                "origin_embedding_0",
                "origin_embedding_1",
                "destination_embedding_0",
                "destination_embedding_1",
                "edge_reverse_similarity",
                "edge_dot_product",
                "origin_neighborhood_density",
                "destination_neighborhood_density",
                "local_graph_context",
            ]
        );
        assert_eq!(features.values.len(), 2);
        assert_eq!(features.values[0].len(), 9);
        assert_eq!(features.values[0][5], 0.0);

        let appended = transformer
            .append_to_dense(
                &[vec![10.0], vec![20.0]],
                &[(0, 1), (1, 2)],
                &[(0, 1), (1, 0), (1, 2)],
            )
            .unwrap();
        assert_eq!(appended[0].len(), 10);
        assert_eq!(appended[0][0], 10.0);
    }

    #[test]
    fn edge_embedding_transformer_rejects_mismatched_or_invalid_rows() {
        let model =
            EdgeEmbeddingModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        let transformer = EmbeddingFeatureTransformer::new(model);

        assert!(transformer
            .transform_edges(&[(0, 3)], &[(0, 1), (1, 2)])
            .is_err());
        assert!(transformer.transform_edges(&[(0, 1)], &[(0, 3)]).is_err());
        assert!(transformer
            .append_to_dense(&[vec![1.0], vec![2.0]], &[(0, 1)], &[(0, 1)])
            .is_err());
    }

    #[test]
    fn embeddings_reload_and_append_to_booster_matrices() {
        let mut encoder = Node2VecEncoder::new(Node2VecConfig {
            dim: 2,
            walk_length: 4,
            walks_per_node: 2,
            window_size: 1,
            epochs: 1,
            negative_samples: 1,
            seed: 99,
            ..Node2VecConfig::default()
        })
        .unwrap();
        encoder.fit(3, &[(0, 1), (1, 2), (2, 0)], None).unwrap();
        let artifact = encoder.to_artifact();
        let restored = Node2VecEncoder::from_artifact(artifact).unwrap();
        let model = EdgeEmbeddingModel::from_encoder(&restored).unwrap();
        let appended = EmbeddingFeatureTransformer::new(model)
            .append_to_dense(&[vec![1.0]], &[(0, 1)], &[(0, 1), (1, 2), (2, 0)])
            .unwrap();

        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0].len(), 1 + 2 + 2 + 5);
        assert!(appended[0].iter().all(|value| value.is_finite()));
    }
}
