use crate::backend::{
    backend_csr_diffusion_backward_f32, backend_csr_diffusion_f32, backend_dense_layer_f32,
    backend_supports_operation, select_backend_for, BackendOperation, BackendSelection,
};
use crate::error::{NeuralError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const GRAPH_SAGE_ARTIFACT_TYPE: &str = "cartoboost.neural.graphsage_encoder";
const GRAPH_SAGE_CSR_DISPATCH_MIN_OPS: usize = 16_384;
const GRAPH_SAGE_DENSE_DISPATCH_MIN_OPS: usize = 16_384;
pub const GRAPH_SAGE_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageEncoderArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub model: GraphSageModelArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum GraphSageModelArtifact {
    Homogeneous(HomogeneousGraphSageEncoderArtifact),
    Hetero(HeteroGraphSageEncoderArtifact),
    HinSage(HinSageEncoderArtifact),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HomogeneousGraphSageEncoderArtifact {
    pub input_dim: usize,
    pub output_dim: usize,
    pub config: GraphSageConfig,
    pub layers: Vec<GraphSageLayer>,
    pub loss_curve: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeteroGraphSageEncoderArtifact {
    pub input_dim: usize,
    pub output_dim: usize,
    pub relation_count: usize,
    pub config: HeteroGraphSageConfig,
    pub layers: Vec<HeteroGraphSageLayer>,
    pub loss_curve: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HinSageEncoderArtifact {
    pub input_dim: usize,
    pub output_dim: usize,
    pub node_type_count: usize,
    pub relation_count: usize,
    pub edge_type_triples: Vec<(usize, usize, usize)>,
    pub neighbor_samples: Vec<usize>,
    pub config: HinSageConfig,
    pub inner: HeteroGraphSageEncoderArtifact,
}

/// Hyper-parameters for homogeneous GraphSAGE-style layers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageConfig {
    pub hidden_dims: Vec<usize>,
    pub epochs: usize,
    pub learning_rate: f32,
    pub negative_samples: usize,
    pub seed: u64,
    pub add_self_loop: bool,
    pub l2_regularization: f32,
    #[serde(default)]
    pub backend: BackendSelection,
}

impl Default for GraphSageConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![16],
            epochs: 20,
            learning_rate: 0.05,
            negative_samples: 4,
            seed: 0x5A17_9A4E_7F33_C0DE,
            add_self_loop: true,
            l2_regularization: 1e-5,
            backend: BackendSelection::default(),
        }
    }
}

/// Hyper-parameters for hetero-typed GraphSAGE-style layers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeteroGraphSageConfig {
    pub hidden_dims: Vec<usize>,
    pub epochs: usize,
    pub learning_rate: f32,
    pub negative_samples: usize,
    pub seed: u64,
    pub l2_regularization: f32,
    #[serde(default)]
    pub backend: BackendSelection,
}

impl Default for HeteroGraphSageConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![16],
            epochs: 20,
            learning_rate: 0.05,
            negative_samples: 4,
            seed: 0x0D1A_2A3B_4C5D_6E7F,
            l2_regularization: 1e-5,
            backend: BackendSelection::default(),
        }
    }
}

/// Hyper-parameters and schema controls for HinSAGE-style typed sampling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HinSageConfig {
    pub hidden_dims: Vec<usize>,
    pub epochs: usize,
    pub learning_rate: f32,
    pub negative_samples: usize,
    pub seed: u64,
    pub l2_regularization: f32,
    pub neighbor_samples: Vec<usize>,
    #[serde(default)]
    pub backend: BackendSelection,
}

impl Default for HinSageConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![16],
            epochs: 20,
            learning_rate: 0.05,
            negative_samples: 4,
            seed: 0xA11C_E5A6_5EED_1234,
            l2_regularization: 1e-5,
            neighbor_samples: Vec::new(),
            backend: BackendSelection::default(),
        }
    }
}

/// Per-epoch loss record returned after fit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageLoss {
    epoch_losses: Vec<f32>,
}

impl GraphSageLoss {
    pub fn values(&self) -> &[f32] {
        &self.epoch_losses
    }

    pub fn final_loss(&self) -> Option<f32> {
        self.epoch_losses.last().copied()
    }
}

/// Directed homogeneous graph with neighbor lists by source node.
#[derive(Debug, Clone)]
pub struct HomogeneousGraph {
    node_count: usize,
    neighbors: Vec<Vec<usize>>,
    edges: Vec<(usize, usize)>,
}

impl HomogeneousGraph {
    /// Builds a graph from explicit directed edges.
    ///
    /// `node_count` must be positive and every edge endpoint must be in-range.
    pub fn from_directed_edges(node_count: usize, edges: &[(usize, usize)]) -> Result<Self> {
        if node_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_count must be positive for a homogeneous graph".to_string(),
            ));
        }
        let neighbors = build_directed_neighbors(node_count, edges)?;
        Ok(Self {
            node_count,
            neighbors,
            edges: edges.to_vec(),
        })
    }

    pub fn from_undirected_edges(node_count: usize, edges: &[(usize, usize)]) -> Result<Self> {
        if node_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_count must be positive for a homogeneous graph".to_string(),
            ));
        }
        let mut undirected = Vec::with_capacity(edges.len() * 2);
        for &(source, target) in edges {
            undirected.push((source, target));
            undirected.push((target, source));
        }
        let neighbors = build_directed_neighbors(node_count, &undirected)?;
        Ok(Self {
            node_count,
            neighbors,
            edges: undirected,
        })
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    pub fn neighbors(&self) -> &[Vec<usize>] {
        &self.neighbors
    }
}

/// A typed edge for heterogeneous graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroTypedEdge {
    pub source: usize,
    pub target: usize,
    pub relation: usize,
}

/// Directed heterogeneous graph grouped by relation index.
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    node_count: usize,
    relation_count: usize,
    edges: Vec<HeteroTypedEdge>,
    neighbors: Vec<Vec<Vec<usize>>>,
}

/// Directed heterogeneous graph with explicit node-type and edge-type schemas.
#[derive(Debug, Clone)]
pub struct HinSageGraph {
    node_count: usize,
    node_type_count: usize,
    relation_count: usize,
    node_types: Vec<usize>,
    edge_type_triples: Vec<(usize, usize, usize)>,
    edges: Vec<HeteroTypedEdge>,
}

impl HinSageGraph {
    pub fn from_typed_schema(
        node_types: Vec<usize>,
        node_type_count: usize,
        relation_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        edges: Vec<HeteroTypedEdge>,
    ) -> Result<Self> {
        if node_types.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "node_types must be non-empty for a HinSAGE graph".to_string(),
            ));
        }
        if node_type_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_type_count must be positive for a HinSAGE graph".to_string(),
            ));
        }
        validate_relation_count(relation_count)?;
        if edge_type_triples.len() != relation_count {
            return Err(NeuralError::InvalidArgument(
                "edge_type_triples length must match relation_count".to_string(),
            ));
        }

        for &node_type in &node_types {
            if node_type >= node_type_count {
                return Err(NeuralError::InvalidArgument(format!(
                    "node type id {node_type} exceeds node_type_count {node_type_count}"
                )));
            }
        }

        for &(source_type, relation, target_type) in &edge_type_triples {
            if source_type >= node_type_count || target_type >= node_type_count {
                return Err(NeuralError::InvalidArgument(
                    "edge_type_triples contain out-of-range node type ids".to_string(),
                ));
            }
            validate_relation_index(relation, relation_count)?;
        }

        let node_count = node_types.len();
        for edge in &edges {
            validate_node_index(edge.source, node_count)?;
            validate_node_index(edge.target, node_count)?;
            validate_relation_index(edge.relation, relation_count)?;
            let expected = edge_type_triples[edge.relation];
            let actual = (
                node_types[edge.source],
                edge.relation,
                node_types[edge.target],
            );
            if actual != expected {
                return Err(NeuralError::InvalidArgument(format!(
                    "edge {edge:?} does not match relation type triple {expected:?}"
                )));
            }
        }

        Ok(Self {
            node_count,
            node_type_count,
            relation_count,
            node_types,
            edge_type_triples,
            edges,
        })
    }

    pub fn to_hetero_graph(&self, neighbor_samples: &[usize]) -> Result<HeteroGraph> {
        let sampled = sample_hinsage_edges(
            self.node_count,
            self.relation_count,
            &self.edges,
            neighbor_samples,
        )?;
        HeteroGraph::from_typed_edges(self.node_count, self.relation_count, &sampled)
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn node_type_count(&self) -> usize {
        self.node_type_count
    }

    pub fn relation_count(&self) -> usize {
        self.relation_count
    }

    pub fn node_types(&self) -> &[usize] {
        &self.node_types
    }

    pub fn edge_type_triples(&self) -> &[(usize, usize, usize)] {
        &self.edge_type_triples
    }
}

impl HeteroGraph {
    /// Builds a relation-typed graph from typed edges.
    ///
    /// `node_count` and `relation_count` must be positive. Every edge index must be
    /// in-range.
    pub fn from_typed_edges(
        node_count: usize,
        relation_count: usize,
        edges: &[HeteroTypedEdge],
    ) -> Result<Self> {
        if node_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_count must be positive for a heterogeneous graph".to_string(),
            ));
        }
        if relation_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "relation_count must be positive for a heterogeneous graph".to_string(),
            ));
        }

        let mut neighbors = vec![vec![Vec::new(); relation_count]; node_count];

        for edge in edges {
            validate_node_index(edge.source, node_count)?;
            validate_node_index(edge.target, node_count)?;
            validate_relation_index(edge.relation, relation_count)?;
            neighbors[edge.source][edge.relation].push(edge.target);
        }

        for row in &mut neighbors {
            for rel_neighbors in row {
                rel_neighbors.sort_unstable();
                rel_neighbors.dedup();
            }
        }

        Ok(Self {
            node_count,
            relation_count,
            edges: edges.to_vec(),
            neighbors,
        })
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn relation_count(&self) -> usize {
        self.relation_count
    }

    pub fn edges(&self) -> &[HeteroTypedEdge] {
        &self.edges
    }

    pub fn neighbors(&self) -> &[Vec<Vec<usize>>] {
        &self.neighbors
    }
}

#[derive(Debug, Clone)]
pub struct GraphSageEncoder {
    config: GraphSageConfig,
    layers: Vec<GraphSageLayer>,
    input_dim: usize,
    output_dim: usize,
    losses: Vec<f32>,
    fitted_neighbors: Option<Vec<Vec<usize>>>,
}

impl GraphSageEncoder {
    pub fn new(config: GraphSageConfig, input_dim: usize) -> Result<Self> {
        validate_input_dim(input_dim)?;
        validate_dimensions(&config.hidden_dims)?;
        validate_homogeneous_backend(&config.backend)?;

        let mut dims = Vec::with_capacity(config.hidden_dims.len() + 1);
        dims.push(input_dim);
        dims.extend(config.hidden_dims.iter().copied());

        let mut rng = SplitMix64::from_seed(config.seed);
        let mut layers = Vec::with_capacity(config.hidden_dims.len());
        for pair in dims.windows(2) {
            let in_dim = pair[0];
            let out_dim = pair[1];
            layers.push(GraphSageLayer::new(in_dim, out_dim, &mut rng));
        }

        let output_dim = dims.last().copied().unwrap_or(input_dim);

        Ok(Self {
            config,
            layers,
            input_dim,
            output_dim,
            losses: Vec::new(),
            fitted_neighbors: None,
        })
    }

    /// Serializes the full encoder state (hyperparameters and learned weights).
    pub fn to_artifact(&self) -> GraphSageEncoderArtifact {
        GraphSageEncoderArtifact {
            artifact_type: GRAPH_SAGE_ARTIFACT_TYPE.to_string(),
            artifact_version: GRAPH_SAGE_ARTIFACT_VERSION,
            model: GraphSageModelArtifact::Homogeneous(HomogeneousGraphSageEncoderArtifact {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                config: self.config.clone(),
                layers: self.layers.clone(),
                loss_curve: self.losses.clone(),
            }),
        }
    }

    /// Serializes encoder state as pretty JSON.
    pub fn to_artifact_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.to_artifact())?)
    }

    /// Writes encoder artifact JSON to `path`.
    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_artifact_json()?)?;
        Ok(())
    }

    /// Reconstructs an encoder from a previous artifact payload.
    pub fn from_artifact(artifact: GraphSageEncoderArtifact) -> Result<Self> {
        validate_graphsage_artifact(&artifact)?;
        let GraphSageModelArtifact::Homogeneous(payload) = artifact.model else {
            return Err(NeuralError::InvalidArgument(
                "artifact model kind is not homogeneous".to_string(),
            ));
        };
        Ok(Self {
            config: payload.config,
            layers: payload.layers,
            input_dim: payload.input_dim,
            output_dim: payload.output_dim,
            losses: payload.loss_curve,
            fitted_neighbors: None,
        })
    }

    /// Loads an encoder from an artifact JSON payload.
    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    /// Returns the encoder configuration copy.
    pub fn config(&self) -> GraphSageConfig {
        self.config.clone()
    }

    pub fn fit(
        &mut self,
        graph: &HomogeneousGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        validate_node_features(Some(graph.node_count()), self.input_dim, node_features)?;

        if self.layers.is_empty() {
            self.losses.clear();
            self.losses.resize(self.config.epochs, 0.0);
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }

        let node_count = graph.node_count();
        let neighbors = if self.config.add_self_loop {
            add_self_neighbors(graph.neighbors())
        } else {
            graph.neighbors().to_vec()
        };
        self.fitted_neighbors = Some(neighbors.clone());

        self.losses = Vec::with_capacity(self.config.epochs);
        let mut rng = SplitMix64::from_seed(self.config.seed);
        let effective_negative = if node_count > 1 {
            self.config.negative_samples
        } else {
            0
        };

        for _ in 0..self.config.epochs {
            let cache = forward_homogeneous(
                node_features,
                &self.layers,
                &neighbors,
                &self.config.backend,
            )?;
            let mut grad = vec![vec![0.0_f32; self.output_dim]; node_count];

            let loss = if graph.edges().is_empty() {
                0.0
            } else {
                compute_link_loss(
                    cache
                        .representations
                        .last()
                        .expect("cache must include final embeddings"),
                    graph.edges(),
                    effective_negative,
                    node_count,
                    &mut rng,
                    &mut grad,
                )
            };
            self.losses.push(loss);

            if !graph.edges().is_empty() {
                apply_homogeneous_backward(
                    &mut self.layers,
                    &cache,
                    &neighbors,
                    &grad,
                    self.config.learning_rate,
                    self.config.l2_regularization,
                    &self.config.backend,
                )?;
            }
        }

        Ok(GraphSageEmbedding::new(
            forward_homogeneous(
                node_features,
                &self.layers,
                &neighbors,
                &self.config.backend,
            )?
            .representations
            .pop()
            .expect("cache must include final embeddings"),
        ))
    }

    pub fn encode(&self, node_features: &[Vec<f32>]) -> Result<GraphSageEmbedding> {
        validate_node_features(None, self.input_dim, node_features)?;
        if self.layers.is_empty() {
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }
        let node_count = node_features.len();
        if node_count == 0 {
            return Ok(GraphSageEmbedding::new(Vec::new()));
        }
        let fallback_neighbors = vec![Vec::new(); node_count];
        let neighbors = self
            .fitted_neighbors
            .as_deref()
            .filter(|neighbors| neighbors.len() == node_count)
            .unwrap_or(&fallback_neighbors);
        Ok(GraphSageEmbedding::new(
            forward_homogeneous(node_features, &self.layers, neighbors, &self.config.backend)?
                .representations
                .last()
                .expect("cache must include final embeddings")
                .clone(),
        ))
    }

    pub fn encode_graph(
        &self,
        graph: &HomogeneousGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        validate_node_features(Some(graph.node_count()), self.input_dim, node_features)?;
        if self.layers.is_empty() {
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }
        let neighbors = if self.config.add_self_loop {
            add_self_neighbors(graph.neighbors())
        } else {
            graph.neighbors().to_vec()
        };
        Ok(GraphSageEmbedding::new(
            forward_homogeneous(
                node_features,
                &self.layers,
                &neighbors,
                &self.config.backend,
            )?
            .representations
            .last()
            .expect("cache must include final embeddings")
            .clone(),
        ))
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn loss_curve(&self) -> GraphSageLoss {
        GraphSageLoss {
            epoch_losses: self.losses.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeteroGraphSageEncoder {
    config: HeteroGraphSageConfig,
    layers: Vec<HeteroGraphSageLayer>,
    input_dim: usize,
    output_dim: usize,
    relation_count: usize,
    losses: Vec<f32>,
    fitted_neighbors: Option<Vec<Vec<Vec<usize>>>>,
}

impl HeteroGraphSageEncoder {
    pub fn new(
        config: HeteroGraphSageConfig,
        input_dim: usize,
        relation_count: usize,
    ) -> Result<Self> {
        validate_input_dim(input_dim)?;
        validate_relation_count(relation_count)?;
        validate_dimensions(&config.hidden_dims)?;
        validate_homogeneous_backend(&config.backend)?;

        let mut dims = Vec::with_capacity(config.hidden_dims.len() + 1);
        dims.push(input_dim);
        dims.extend(config.hidden_dims.iter().copied());

        let mut rng = SplitMix64::from_seed(config.seed);
        let mut layers = Vec::with_capacity(config.hidden_dims.len());
        for pair in dims.windows(2) {
            let in_dim = pair[0];
            let out_dim = pair[1];
            layers.push(HeteroGraphSageLayer::new(
                in_dim,
                out_dim,
                relation_count,
                &mut rng,
            ));
        }

        let output_dim = dims.last().copied().unwrap_or(input_dim);

        Ok(Self {
            config,
            layers,
            input_dim,
            output_dim,
            relation_count,
            losses: Vec::new(),
            fitted_neighbors: None,
        })
    }

    /// Serializes the full encoder state (hyperparameters and learned weights).
    pub fn to_artifact(&self) -> GraphSageEncoderArtifact {
        GraphSageEncoderArtifact {
            artifact_type: GRAPH_SAGE_ARTIFACT_TYPE.to_string(),
            artifact_version: GRAPH_SAGE_ARTIFACT_VERSION,
            model: GraphSageModelArtifact::Hetero(HeteroGraphSageEncoderArtifact {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                relation_count: self.relation_count,
                config: self.config.clone(),
                layers: self.layers.clone(),
                loss_curve: self.losses.clone(),
            }),
        }
    }

    /// Serializes encoder state as pretty JSON.
    pub fn to_artifact_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.to_artifact())?)
    }

    /// Writes encoder artifact JSON to `path`.
    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_artifact_json()?)?;
        Ok(())
    }

    /// Reconstructs an encoder from a previous artifact payload.
    pub fn from_artifact(artifact: GraphSageEncoderArtifact) -> Result<Self> {
        validate_graphsage_artifact(&artifact)?;
        let GraphSageModelArtifact::Hetero(payload) = artifact.model else {
            return Err(NeuralError::InvalidArgument(
                "artifact model kind is not heterogeneous".to_string(),
            ));
        };
        Ok(Self {
            config: payload.config,
            layers: payload.layers,
            input_dim: payload.input_dim,
            output_dim: payload.output_dim,
            relation_count: payload.relation_count,
            losses: payload.loss_curve,
            fitted_neighbors: None,
        })
    }

    /// Loads an encoder from an artifact JSON payload.
    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    /// Returns the encoder configuration copy.
    pub fn config(&self) -> HeteroGraphSageConfig {
        self.config.clone()
    }

    pub fn fit(
        &mut self,
        graph: &HeteroGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        validate_node_features(Some(graph.node_count()), self.input_dim, node_features)?;

        if self.layers.is_empty() {
            self.losses.clear();
            self.losses.resize(self.config.epochs, 0.0);
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }

        let node_count = graph.node_count();
        let neighbors = graph.neighbors().to_vec();
        self.fitted_neighbors = Some(neighbors.clone());
        self.losses = Vec::with_capacity(self.config.epochs);
        let mut rng = SplitMix64::from_seed(self.config.seed);
        let effective_negative = if node_count > 1 {
            self.config.negative_samples
        } else {
            0
        };

        for _ in 0..self.config.epochs {
            let cache = forward_hetero(
                node_features,
                &self.layers,
                &neighbors,
                &self.config.backend,
            )?;
            let mut grad = vec![vec![0.0_f32; self.output_dim]; node_count];

            let loss = if graph.edges().is_empty() {
                0.0
            } else {
                compute_link_loss_hetero(
                    cache
                        .representations
                        .last()
                        .expect("cache must include final embeddings"),
                    graph.edges(),
                    effective_negative,
                    node_count,
                    &mut rng,
                    &mut grad,
                )
            };
            self.losses.push(loss);

            if !graph.edges().is_empty() {
                apply_hetero_backward(
                    &mut self.layers,
                    &cache,
                    &neighbors,
                    &grad,
                    self.config.learning_rate,
                    self.config.l2_regularization,
                    &self.config.backend,
                )?;
            }
        }

        Ok(GraphSageEmbedding::new(
            forward_hetero(
                node_features,
                &self.layers,
                &neighbors,
                &self.config.backend,
            )?
            .representations
            .last()
            .expect("cache must include final embeddings")
            .clone(),
        ))
    }

    pub fn encode(&self, node_features: &[Vec<f32>]) -> Result<GraphSageEmbedding> {
        validate_node_features(None, self.input_dim, node_features)?;
        if self.layers.is_empty() {
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }
        let node_count = node_features.len();
        if node_count == 0 {
            return Ok(GraphSageEmbedding::new(Vec::new()));
        }
        let fallback_neighbors = vec![vec![Vec::new(); self.relation_count]; node_count];
        let neighbors = self
            .fitted_neighbors
            .as_deref()
            .filter(|neighbors| neighbors.len() == node_count)
            .unwrap_or(&fallback_neighbors);
        Ok(GraphSageEmbedding::new(
            forward_hetero(node_features, &self.layers, neighbors, &self.config.backend)?
                .representations
                .last()
                .expect("cache must include final embeddings")
                .clone(),
        ))
    }

    pub fn encode_graph(
        &self,
        graph: &HeteroGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        validate_node_features(Some(graph.node_count()), self.input_dim, node_features)?;
        if self.layers.is_empty() {
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }
        Ok(GraphSageEmbedding::new(
            forward_hetero(
                node_features,
                &self.layers,
                graph.neighbors(),
                &self.config.backend,
            )?
            .representations
            .last()
            .expect("cache must include final embeddings")
            .clone(),
        ))
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn relation_count(&self) -> usize {
        self.relation_count
    }

    pub fn loss_curve(&self) -> GraphSageLoss {
        GraphSageLoss {
            epoch_losses: self.losses.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HinSageEncoder {
    config: HinSageConfig,
    node_type_count: usize,
    relation_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    inner: HeteroGraphSageEncoder,
}

impl HinSageEncoder {
    pub fn new(
        config: HinSageConfig,
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
    ) -> Result<Self> {
        if node_type_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_type_count must be positive for a HinSAGE encoder".to_string(),
            ));
        }
        if edge_type_triples.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "edge_type_triples must be non-empty for a HinSAGE encoder".to_string(),
            ));
        }
        let relation_count = edge_type_triples.len();
        for (relation, &(source_type, relation_id, target_type)) in
            edge_type_triples.iter().enumerate()
        {
            if relation_id != relation {
                return Err(NeuralError::InvalidArgument(
                    "edge_type_triples relation ids must be zero-based and ordered".to_string(),
                ));
            }
            if source_type >= node_type_count || target_type >= node_type_count {
                return Err(NeuralError::InvalidArgument(
                    "edge_type_triples contain out-of-range node type ids".to_string(),
                ));
            }
        }
        validate_dimensions(&config.hidden_dims)?;
        validate_neighbor_samples(&config.neighbor_samples, relation_count)?;

        let inner_config = HeteroGraphSageConfig {
            hidden_dims: config.hidden_dims.clone(),
            epochs: config.epochs,
            learning_rate: config.learning_rate,
            negative_samples: config.negative_samples,
            seed: config.seed,
            l2_regularization: config.l2_regularization,
            backend: config.backend.clone(),
        };
        let inner = HeteroGraphSageEncoder::new(inner_config, input_dim, relation_count)?;
        Ok(Self {
            config,
            node_type_count,
            relation_count,
            edge_type_triples,
            inner,
        })
    }

    pub fn fit(
        &mut self,
        graph: &HinSageGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        self.validate_graph_schema(graph)?;
        let hetero_graph = graph.to_hetero_graph(&self.config.neighbor_samples)?;
        self.inner.fit(&hetero_graph, node_features)
    }

    pub fn encode(&self, node_features: &[Vec<f32>]) -> Result<GraphSageEmbedding> {
        self.inner.encode(node_features)
    }

    pub fn encode_graph(
        &self,
        graph: &HinSageGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        self.validate_graph_schema(graph)?;
        let hetero_graph = graph.to_hetero_graph(&self.config.neighbor_samples)?;
        self.inner.encode_graph(&hetero_graph, node_features)
    }

    pub fn link_embeddings(
        &self,
        embeddings: &[Vec<f32>],
        pairs: &[(usize, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        build_link_embeddings(embeddings, pairs)
    }

    pub fn to_artifact(&self) -> GraphSageEncoderArtifact {
        let GraphSageModelArtifact::Hetero(inner) = self.inner.to_artifact().model else {
            unreachable!("HinSAGE inner encoder is always hetero");
        };
        GraphSageEncoderArtifact {
            artifact_type: GRAPH_SAGE_ARTIFACT_TYPE.to_string(),
            artifact_version: GRAPH_SAGE_ARTIFACT_VERSION,
            model: GraphSageModelArtifact::HinSage(HinSageEncoderArtifact {
                input_dim: self.input_dim(),
                output_dim: self.output_dim(),
                node_type_count: self.node_type_count,
                relation_count: self.relation_count,
                edge_type_triples: self.edge_type_triples.clone(),
                neighbor_samples: self.config.neighbor_samples.clone(),
                config: self.config.clone(),
                inner,
            }),
        }
    }

    pub fn from_artifact(artifact: GraphSageEncoderArtifact) -> Result<Self> {
        validate_graphsage_artifact(&artifact)?;
        let GraphSageModelArtifact::HinSage(payload) = artifact.model else {
            return Err(NeuralError::InvalidArgument(
                "artifact model kind is not hinsage".to_string(),
            ));
        };
        let inner = HeteroGraphSageEncoder::from_artifact(GraphSageEncoderArtifact {
            artifact_type: GRAPH_SAGE_ARTIFACT_TYPE.to_string(),
            artifact_version: GRAPH_SAGE_ARTIFACT_VERSION,
            model: GraphSageModelArtifact::Hetero(payload.inner),
        })?;
        Ok(Self {
            config: payload.config,
            node_type_count: payload.node_type_count,
            relation_count: payload.relation_count,
            edge_type_triples: payload.edge_type_triples,
            inner,
        })
    }

    pub fn to_artifact_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.to_artifact())?)
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_artifact_json()?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn config(&self) -> HinSageConfig {
        self.config.clone()
    }

    pub fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }

    pub fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    pub fn node_type_count(&self) -> usize {
        self.node_type_count
    }

    pub fn relation_count(&self) -> usize {
        self.relation_count
    }

    pub fn edge_type_triples(&self) -> &[(usize, usize, usize)] {
        &self.edge_type_triples
    }

    pub fn loss_curve(&self) -> GraphSageLoss {
        self.inner.loss_curve()
    }

    fn validate_graph_schema(&self, graph: &HinSageGraph) -> Result<()> {
        if graph.node_type_count() != self.node_type_count {
            return Err(NeuralError::InvalidArgument(
                "HinSAGE graph node_type_count does not match encoder".to_string(),
            ));
        }
        if graph.relation_count() != self.relation_count {
            return Err(NeuralError::InvalidArgument(
                "HinSAGE graph relation_count does not match encoder".to_string(),
            ));
        }
        if graph.edge_type_triples() != self.edge_type_triples() {
            return Err(NeuralError::InvalidArgument(
                "HinSAGE graph edge_type_triples do not match encoder".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphSageEmbedding {
    vectors: Vec<Vec<f32>>,
}

impl GraphSageEmbedding {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        Self { vectors }
    }

    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    pub fn into_inner(self) -> Vec<Vec<f32>> {
        self.vectors
    }

    pub fn dim(&self) -> usize {
        self.vectors.first().map_or(0, |row| row.len())
    }

    pub fn node_count(&self) -> usize {
        self.vectors.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageLayer {
    in_dim: usize,
    out_dim: usize,
    self_weight: Vec<f32>,
    neigh_weight: Vec<f32>,
    bias: Vec<f32>,
}

impl GraphSageLayer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut SplitMix64) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let mut self_weight = Vec::with_capacity(in_dim * out_dim);
        let mut neigh_weight = Vec::with_capacity(in_dim * out_dim);
        for _ in 0..(in_dim * out_dim) {
            self_weight.push((rng.next_unit() * 2.0 - 1.0) * scale * 0.1);
            neigh_weight.push((rng.next_unit() * 2.0 - 1.0) * scale * 0.1);
        }

        Self {
            in_dim,
            out_dim,
            self_weight,
            neigh_weight,
            bias: vec![0.0; out_dim],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeteroGraphSageLayer {
    in_dim: usize,
    out_dim: usize,
    self_weight: Vec<f32>,
    relation_weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl HeteroGraphSageLayer {
    fn new(in_dim: usize, out_dim: usize, relation_count: usize, rng: &mut SplitMix64) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let mut relation_weights = Vec::with_capacity(relation_count);
        for _ in 0..relation_count {
            let mut relation_weight = Vec::with_capacity(in_dim * out_dim);
            for _ in 0..(in_dim * out_dim) {
                relation_weight.push((rng.next_unit() * 2.0 - 1.0) * scale * 0.1);
            }
            relation_weights.push(relation_weight);
        }

        Self {
            in_dim,
            out_dim,
            self_weight: (0..in_dim * out_dim)
                .map(|_| (rng.next_unit() * 2.0 - 1.0) * scale * 0.1)
                .collect(),
            relation_weights,
            bias: vec![0.0; out_dim],
        }
    }
}

#[derive(Debug)]
struct HomogeneousLayerCache {
    preactivations: Vec<Vec<f32>>,
    neighborhood_means: Vec<Vec<f32>>,
    representations: Vec<Vec<f32>>,
}

#[derive(Debug)]
struct HeteroLayerCache {
    preactivations: Vec<Vec<f32>>,
    relation_means: Vec<Vec<Vec<f32>>>,
    representations: Vec<Vec<f32>>,
}

#[derive(Debug)]
struct GraphSageForwardCache {
    representations: Vec<Vec<Vec<f32>>>,
    layers: Vec<HomogeneousLayerCache>,
}

#[derive(Debug)]
struct HeteroForwardCache {
    representations: Vec<Vec<Vec<f32>>>,
    layers: Vec<HeteroLayerCache>,
}

fn forward_homogeneous(
    node_features: &[Vec<f32>],
    layers: &[GraphSageLayer],
    neighbors: &[Vec<usize>],
    backend: &BackendSelection,
) -> Result<GraphSageForwardCache> {
    if node_features.is_empty() {
        return Ok(GraphSageForwardCache {
            representations: vec![Vec::new()],
            layers: Vec::new(),
        });
    }

    let mut representations = Vec::with_capacity(layers.len() + 1);
    representations.push(node_features.to_vec());
    let mut cache_layers = Vec::with_capacity(layers.len());

    for layer in layers {
        let current = representations
            .last()
            .expect("layer activation should exist before running forward");
        let means = homogeneous_neighbor_means(current, neighbors, layer.in_dim, backend)?;

        let combined = current
            .iter()
            .zip(&means)
            .map(|(self_row, mean_row)| {
                let mut row = Vec::with_capacity(layer.in_dim * 2);
                row.extend_from_slice(self_row);
                row.extend_from_slice(mean_row);
                row
            })
            .collect::<Vec<_>>();
        let mut weights = Vec::with_capacity(layer.in_dim * layer.out_dim * 2);
        weights.extend_from_slice(&layer.self_weight);
        weights.extend_from_slice(&layer.neigh_weight);
        let cpu_backend = graph_dense_cpu_fallback(
            backend,
            combined.len(),
            layer.in_dim.saturating_mul(2),
            layer.out_dim,
        )?;
        let preactivations = backend_dense_layer_f32(
            cpu_backend.as_ref().unwrap_or(backend),
            &combined,
            &weights,
            &layer.bias,
        )?;
        let next = preactivations
            .par_iter()
            .map(|row| row.iter().map(|value| value.max(0.0)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let current = current.to_vec();
        representations.push(next);
        cache_layers.push(HomogeneousLayerCache {
            preactivations,
            neighborhood_means: means,
            representations: current,
        });
    }

    Ok(GraphSageForwardCache {
        representations,
        layers: cache_layers,
    })
}

fn homogeneous_neighbor_means(
    current: &[Vec<f32>],
    neighbors: &[Vec<usize>],
    width: usize,
    backend: &BackendSelection,
) -> Result<Vec<Vec<f32>>> {
    let operations = current.len().saturating_mul(width);
    if backend.selected != "cpu" && operations >= GRAPH_SAGE_CSR_DISPATCH_MIN_OPS {
        let mut indptr = Vec::with_capacity(neighbors.len() + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        indptr.push(0_u32);
        for neighbor_ids in neighbors {
            let weight = if neighbor_ids.is_empty() {
                0.0
            } else {
                1.0 / neighbor_ids.len() as f32
            };
            indices.extend(neighbor_ids.iter().map(|neighbor| *neighbor as u32));
            weights.extend(std::iter::repeat_n(weight, neighbor_ids.len()));
            indptr.push(indices.len() as u32);
        }
        let values = current.iter().flatten().copied().collect::<Vec<_>>();
        let output =
            backend_csr_diffusion_f32(backend, &indptr, &indices, &weights, width, &values)?;
        return Ok(output.chunks(width).map(|row| row.to_vec()).collect());
    }
    Ok(neighbors
        .par_iter()
        .map(|neighbor_ids| {
            let mut mean = vec![0.0_f32; width];
            if neighbor_ids.is_empty() {
                return mean;
            }
            let inv = 1.0 / (neighbor_ids.len() as f32);
            for &neighbor in neighbor_ids {
                for (slot, value) in mean.iter_mut().zip(current[neighbor].iter()) {
                    *slot += *value * inv;
                }
            }
            mean
        })
        .collect())
}

fn forward_hetero(
    node_features: &[Vec<f32>],
    layers: &[HeteroGraphSageLayer],
    neighbors: &[Vec<Vec<usize>>],
    backend: &BackendSelection,
) -> Result<HeteroForwardCache> {
    if node_features.is_empty() {
        return Ok(HeteroForwardCache {
            representations: vec![Vec::new()],
            layers: Vec::new(),
        });
    }

    if layers.first().is_some_and(|layer| {
        layer.relation_weights.len() != neighbors.first().map_or(0, |row| row.len())
    }) {
        return Err(NeuralError::InvalidArgument(
            "relation count must match hetero neighbor tensor".to_string(),
        ));
    }

    let mut representations = Vec::with_capacity(layers.len() + 1);
    representations.push(node_features.to_vec());
    let mut cache_layers = Vec::with_capacity(layers.len());

    for layer in layers {
        let current = representations
            .last()
            .expect("layer activation should exist before running forward");
        let relation_count = neighbors.first().map_or(0, |row| row.len());
        let relation_means = heterogeneous_neighbor_means(
            current,
            neighbors,
            relation_count,
            layer.in_dim,
            backend,
        )?;

        let combined = current
            .iter()
            .zip(&relation_means)
            .map(|(self_row, relation_rows)| {
                let mut row = Vec::with_capacity(layer.in_dim * (relation_count + 1));
                row.extend_from_slice(self_row);
                for relation_row in relation_rows.iter().take(relation_count) {
                    row.extend_from_slice(relation_row);
                }
                row
            })
            .collect::<Vec<_>>();
        let mut weights = Vec::with_capacity(layer.in_dim * layer.out_dim * (relation_count + 1));
        weights.extend_from_slice(&layer.self_weight);
        for relation_weight in layer.relation_weights.iter().take(relation_count) {
            weights.extend_from_slice(relation_weight);
        }
        let cpu_backend = graph_dense_cpu_fallback(
            backend,
            combined.len(),
            layer.in_dim.saturating_mul(relation_count + 1),
            layer.out_dim,
        )?;
        let preactivations = backend_dense_layer_f32(
            cpu_backend.as_ref().unwrap_or(backend),
            &combined,
            &weights,
            &layer.bias,
        )?;
        let next = preactivations
            .par_iter()
            .map(|row| row.iter().map(|value| value.max(0.0)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let current = current.to_vec();
        representations.push(next);
        cache_layers.push(HeteroLayerCache {
            preactivations,
            relation_means,
            representations: current,
        });
    }

    Ok(HeteroForwardCache {
        representations,
        layers: cache_layers,
    })
}

fn heterogeneous_neighbor_means(
    current: &[Vec<f32>],
    neighbors: &[Vec<Vec<usize>>],
    relation_count: usize,
    width: usize,
    backend: &BackendSelection,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let operations = current
        .len()
        .saturating_mul(relation_count)
        .saturating_mul(width);
    if backend.selected != "cpu" && operations >= GRAPH_SAGE_CSR_DISPATCH_MIN_OPS {
        let row_count = current.len().saturating_mul(relation_count);
        let mut indptr = Vec::with_capacity(row_count + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        indptr.push(0_u32);
        for relation in 0..relation_count {
            let relation_offset = relation.saturating_mul(current.len());
            for node_neighbors in neighbors.iter().take(current.len()) {
                let neighbor_ids = &node_neighbors[relation];
                let weight = if neighbor_ids.is_empty() {
                    0.0
                } else {
                    1.0 / neighbor_ids.len() as f32
                };
                indices.extend(
                    neighbor_ids
                        .iter()
                        .map(|neighbor| (relation_offset + *neighbor) as u32),
                );
                weights.extend(std::iter::repeat_n(weight, neighbor_ids.len()));
                indptr.push(indices.len() as u32);
            }
        }
        let values = (0..relation_count)
            .flat_map(|_| current.iter().flatten().copied())
            .collect::<Vec<_>>();
        let output =
            backend_csr_diffusion_f32(backend, &indptr, &indices, &weights, width, &values)?;
        let mut relation_means = vec![vec![vec![0.0_f32; width]; relation_count]; current.len()];
        for (relation, relation_output) in output
            .chunks_exact(current.len() * width)
            .take(relation_count)
            .enumerate()
        {
            for (node, row) in relation_output.chunks_exact(width).enumerate() {
                relation_means[node][relation].copy_from_slice(row);
            }
        }
        return Ok(relation_means);
    }

    let mut relation_means = vec![vec![vec![0.0_f32; width]; relation_count]; current.len()];
    relation_means
        .par_iter_mut()
        .zip(neighbors.par_iter())
        .for_each(|(relation_slots, node_neighbors)| {
            for (relation_mean, neighbor_ids) in relation_slots
                .iter_mut()
                .zip(node_neighbors.iter().take(relation_count))
            {
                if neighbor_ids.is_empty() {
                    continue;
                }
                let inv = 1.0 / neighbor_ids.len() as f32;
                for &neighbor in neighbor_ids {
                    for (slot, input_value) in relation_mean.iter_mut().zip(&current[neighbor]) {
                        *slot += *input_value * inv;
                    }
                }
            }
        });
    Ok(relation_means)
}

fn homogeneous_neighbor_mean_backward(
    current: &[Vec<f32>],
    neighbors: &[Vec<usize>],
    width: usize,
    output_grad: &[Vec<f32>],
    backend: &BackendSelection,
) -> Result<Vec<Vec<f32>>> {
    let mut indptr = Vec::with_capacity(neighbors.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indptr.push(0_u32);
    for neighbor_ids in neighbors {
        let weight = if neighbor_ids.is_empty() {
            0.0
        } else {
            1.0 / neighbor_ids.len() as f32
        };
        indices.extend(neighbor_ids.iter().map(|neighbor| *neighbor as u32));
        weights.extend(std::iter::repeat_n(weight, neighbor_ids.len()));
        indptr.push(indices.len() as u32);
    }
    let values = current.iter().flatten().copied().collect::<Vec<_>>();
    let output_grad = output_grad.iter().flatten().copied().collect::<Vec<_>>();
    let backward = backend_csr_diffusion_backward_f32(
        backend,
        &indptr,
        &indices,
        &weights,
        width,
        &values,
        &output_grad,
    )?;
    Ok(backward
        .input_grad
        .chunks_exact(width)
        .map(|row| row.to_vec())
        .collect())
}

fn heterogeneous_neighbor_mean_backward(
    current: &[Vec<f32>],
    neighbors: &[Vec<Vec<usize>>],
    relation_count: usize,
    width: usize,
    output_grad: &[Vec<Vec<f32>>],
    backend: &BackendSelection,
) -> Result<Vec<Vec<f32>>> {
    let row_count = current.len().saturating_mul(relation_count);
    let mut indptr = Vec::with_capacity(row_count + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indptr.push(0_u32);
    for relation in 0..relation_count {
        let relation_offset = relation.saturating_mul(current.len());
        for node_neighbors in neighbors.iter().take(current.len()) {
            let neighbor_ids = &node_neighbors[relation];
            let weight = if neighbor_ids.is_empty() {
                0.0
            } else {
                1.0 / neighbor_ids.len() as f32
            };
            indices.extend(
                neighbor_ids
                    .iter()
                    .map(|neighbor| (relation_offset + *neighbor) as u32),
            );
            weights.extend(std::iter::repeat_n(weight, neighbor_ids.len()));
            indptr.push(indices.len() as u32);
        }
    }
    let values = (0..relation_count)
        .flat_map(|_| current.iter().flatten().copied())
        .collect::<Vec<_>>();
    let output_grad = (0..relation_count)
        .flat_map(|relation| {
            output_grad
                .iter()
                .flat_map(move |node_grad| node_grad[relation].iter().copied())
        })
        .collect::<Vec<_>>();
    let backward = backend_csr_diffusion_backward_f32(
        backend,
        &indptr,
        &indices,
        &weights,
        width,
        &values,
        &output_grad,
    )?;
    let mut input_grad = vec![vec![0.0_f32; width]; current.len()];
    for relation in 0..relation_count {
        let start = relation * current.len() * width;
        for (node, row) in backward.input_grad[start..start + current.len() * width]
            .chunks_exact(width)
            .enumerate()
        {
            for (target, value) in input_grad[node].iter_mut().zip(row) {
                *target += *value;
            }
        }
    }
    Ok(input_grad)
}

#[allow(clippy::needless_range_loop)]
fn apply_homogeneous_backward(
    layers: &mut [GraphSageLayer],
    cache: &GraphSageForwardCache,
    neighbors: &[Vec<usize>],
    grad_output: &[Vec<f32>],
    learning_rate: f32,
    l2_regularization: f32,
    backend: &BackendSelection,
) -> Result<()> {
    if cache.layers.is_empty() {
        return Ok(());
    }

    let mut upstream_grad = grad_output.to_vec();

    for layer_index in (0..layers.len()).rev() {
        let layer = &mut layers[layer_index];
        let layer_cache = &cache.layers[layer_index];
        let input = &layer_cache.representations;
        let pre = &layer_cache.preactivations;
        let means = &layer_cache.neighborhood_means;
        let out_dim = layer.out_dim;
        let in_dim = layer.in_dim;

        let mut next_grad = vec![vec![0.0_f32; in_dim]; input.len()];
        let mut self_grad = vec![0.0_f32; layer.self_weight.len()];
        let mut neigh_grad = vec![0.0_f32; layer.neigh_weight.len()];
        let mut bias_grad = vec![0.0_f32; layer.bias.len()];
        let use_device_csr = backend.selected != "cpu"
            && input.len().saturating_mul(in_dim) >= GRAPH_SAGE_CSR_DISPATCH_MIN_OPS;
        let mut mean_output_grad = use_device_csr.then(|| vec![vec![0.0_f32; in_dim]; input.len()]);

        for node in 0..input.len() {
            for out in 0..out_dim {
                let grad = upstream_grad[node][out];
                if pre[node][out] <= 0.0 {
                    continue;
                }

                bias_grad[out] += grad;

                for index in 0..in_dim {
                    let idx = index * out_dim + out;
                    let self_value = input[node][index];
                    let neigh_value = means[node][index];
                    let g = grad;

                    self_grad[idx] += self_value * g;
                    neigh_grad[idx] += neigh_value * g;
                    next_grad[node][index] += layer.self_weight[idx] * g;
                    if let Some(mean_grad) = mean_output_grad.as_mut() {
                        mean_grad[node][index] += layer.neigh_weight[idx] * g;
                    }
                }

                if !use_device_csr && !neighbors[node].is_empty() {
                    let inv = 1.0 / neighbors[node].len() as f32;
                    for &neighbor in &neighbors[node] {
                        for index in 0..in_dim {
                            let idx = index * out_dim + out;
                            next_grad[neighbor][index] += layer.neigh_weight[idx] * grad * inv;
                        }
                    }
                }
            }
        }

        if let Some(mean_grad) = mean_output_grad {
            let message_grad =
                homogeneous_neighbor_mean_backward(input, neighbors, in_dim, &mean_grad, backend)?;
            for (next_row, message_row) in next_grad.iter_mut().zip(message_grad) {
                for (next, message) in next_row.iter_mut().zip(message_row) {
                    *next += message;
                }
            }
        }

        for (index, weight) in layer.self_weight.iter_mut().enumerate() {
            *weight -= learning_rate * (self_grad[index] + l2_regularization * *weight);
        }
        for (index, weight) in layer.neigh_weight.iter_mut().enumerate() {
            *weight -= learning_rate * (neigh_grad[index] + l2_regularization * *weight);
        }
        for (index, value) in layer.bias.iter_mut().enumerate() {
            *value -= learning_rate * bias_grad[index];
        }

        upstream_grad = next_grad;
    }

    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn apply_hetero_backward(
    layers: &mut [HeteroGraphSageLayer],
    cache: &HeteroForwardCache,
    neighbors: &[Vec<Vec<usize>>],
    grad_output: &[Vec<f32>],
    learning_rate: f32,
    l2_regularization: f32,
    backend: &BackendSelection,
) -> Result<()> {
    if cache.layers.is_empty() {
        return Ok(());
    }

    let relation_count = neighbors.first().map_or(0, |row| row.len());
    let mut upstream_grad = grad_output.to_vec();

    for layer_index in (0..layers.len()).rev() {
        let layer = &mut layers[layer_index];
        let layer_cache = &cache.layers[layer_index];
        let input = &layer_cache.representations;
        let pre = &layer_cache.preactivations;
        let means = &layer_cache.relation_means;

        let out_dim = layer.out_dim;
        let in_dim = layer.in_dim;
        let mut next_grad = vec![vec![0.0_f32; in_dim]; input.len()];
        let mut self_grad = vec![0.0_f32; layer.self_weight.len()];
        let mut relation_grad = vec![vec![0.0_f32; layer.in_dim * out_dim]; relation_count];
        let mut bias_grad = vec![0.0_f32; layer.bias.len()];
        let use_device_csr = backend.selected != "cpu"
            && input
                .len()
                .saturating_mul(relation_count)
                .saturating_mul(in_dim)
                >= GRAPH_SAGE_CSR_DISPATCH_MIN_OPS;
        let mut relation_output_grad =
            use_device_csr.then(|| vec![vec![vec![0.0_f32; in_dim]; relation_count]; input.len()]);

        for node in 0..input.len() {
            for out in 0..out_dim {
                let grad = upstream_grad[node][out];
                if pre[node][out] <= 0.0 {
                    continue;
                }

                bias_grad[out] += grad;

                for index in 0..in_dim {
                    let weight_index = index * out_dim + out;
                    self_grad[weight_index] += input[node][index] * grad;
                    next_grad[node][index] += layer.self_weight[weight_index] * grad;
                }

                for (relation, neighbors_for_relation) in
                    neighbors[node].iter().enumerate().take(relation_count)
                {
                    if neighbors_for_relation.is_empty() {
                        continue;
                    }
                    let inv = 1.0 / (neighbors_for_relation.len() as f32);
                    #[allow(clippy::needless_range_loop)]
                    for index in 0..in_dim {
                        let weight_index = index * out_dim + out;
                        relation_grad[relation][weight_index] +=
                            means[node][relation][index] * grad;
                        let relation_weight = layer.relation_weights[relation][weight_index];
                        if let Some(mean_grad) = relation_output_grad.as_mut() {
                            mean_grad[node][relation][index] += relation_weight * grad;
                        } else {
                            for &neighbor in neighbors_for_relation {
                                next_grad[neighbor][index] += relation_weight * grad * inv;
                            }
                        }
                    }
                }
            }
        }

        if let Some(mean_grad) = relation_output_grad {
            let message_grad = heterogeneous_neighbor_mean_backward(
                input,
                neighbors,
                relation_count,
                in_dim,
                &mean_grad,
                backend,
            )?;
            for (next_row, message_row) in next_grad.iter_mut().zip(message_grad) {
                for (next, message) in next_row.iter_mut().zip(message_row) {
                    *next += message;
                }
            }
        }

        for (index, weight) in layer.self_weight.iter_mut().enumerate() {
            *weight -= learning_rate * (self_grad[index] + l2_regularization * *weight);
        }
        for (relation_grad_row, relation_weights) in relation_grad
            .iter()
            .zip(layer.relation_weights.iter_mut())
            .take(relation_count)
        {
            for (index, weight) in relation_weights.iter_mut().enumerate() {
                *weight -= learning_rate * (relation_grad_row[index] + l2_regularization * *weight);
            }
        }
        for (index, value) in layer.bias.iter_mut().enumerate() {
            *value -= learning_rate * bias_grad[index];
        }

        upstream_grad = next_grad;
    }

    Ok(())
}

fn compute_link_loss(
    embeddings: &[Vec<f32>],
    edges: &[(usize, usize)],
    negative_samples: usize,
    node_count: usize,
    rng: &mut SplitMix64,
    grad: &mut [Vec<f32>],
) -> f32 {
    if edges.is_empty() || node_count == 0 {
        return 0.0;
    }
    let observed_targets = source_observed_targets(node_count, edges.iter().copied());
    let mut loss = 0.0_f32;
    let scale = if edges.is_empty() {
        1.0
    } else {
        1.0 / ((edges.len() * (1 + negative_samples).max(1)) as f32)
    };

    for &(left, right) in edges {
        let mut pos_score = 0.0_f32;
        for (left_value, right_value) in embeddings[left].iter().zip(embeddings[right].iter()) {
            pos_score += left_value * right_value;
        }
        let pos_prob = sigmoid(pos_score);
        let safe_pos = safe_prob(pos_prob);
        loss += -safe_pos.ln();
        let pos_grad = (pos_prob - 1.0) * scale;

        for index in 0..grad[left].len() {
            grad[left][index] += pos_grad * embeddings[right][index];
            grad[right][index] += pos_grad * embeddings[left][index];
        }

        if negative_samples == 0 {
            continue;
        }

        for _ in 0..negative_samples {
            let Some(negative) = sample_negative_node(rng, node_count, &observed_targets[left])
            else {
                break;
            };
            let mut neg_score = 0.0_f32;
            for (left_value, neg_value) in embeddings[left].iter().zip(embeddings[negative].iter())
            {
                neg_score += left_value * neg_value;
            }
            let neg_prob = sigmoid(neg_score);
            let safe_neg = (1.0 - neg_prob).max(f32::EPSILON);
            loss += -safe_neg.ln();
            let neg_grad = neg_prob * scale;

            for index in 0..grad[left].len() {
                grad[left][index] += neg_grad * embeddings[negative][index];
                grad[negative][index] += neg_grad * embeddings[left][index];
            }
        }
    }

    loss
}

fn compute_link_loss_hetero(
    embeddings: &[Vec<f32>],
    edges: &[HeteroTypedEdge],
    negative_samples: usize,
    node_count: usize,
    rng: &mut SplitMix64,
    grad: &mut [Vec<f32>],
) -> f32 {
    if edges.is_empty() || node_count == 0 {
        return 0.0;
    }

    let observed_targets = source_observed_targets(
        node_count,
        edges.iter().map(|edge| (edge.source, edge.target)),
    );
    let mut loss = 0.0_f32;
    let scale = if edges.is_empty() {
        1.0
    } else {
        1.0 / ((edges.len() * (1 + negative_samples).max(1)) as f32)
    };
    for edge in edges {
        let left = edge.source;
        let right = edge.target;
        let mut pos_score = 0.0_f32;
        for (left_value, right_value) in embeddings[left].iter().zip(embeddings[right].iter()) {
            pos_score += left_value * right_value;
        }
        let pos_prob = sigmoid(pos_score);
        let safe_pos = safe_prob(pos_prob);
        loss += -safe_pos.ln();
        let pos_grad = (pos_prob - 1.0) * scale;

        for index in 0..grad[left].len() {
            grad[left][index] += pos_grad * embeddings[right][index];
            grad[right][index] += pos_grad * embeddings[left][index];
        }

        if negative_samples == 0 {
            continue;
        }

        for _ in 0..negative_samples {
            let Some(negative) = sample_negative_node(rng, node_count, &observed_targets[left])
            else {
                break;
            };
            let mut neg_score = 0.0_f32;
            for (left_value, neg_value) in embeddings[left].iter().zip(embeddings[negative].iter())
            {
                neg_score += left_value * neg_value;
            }
            let neg_prob = sigmoid(neg_score);
            let safe_neg = (1.0 - neg_prob).max(f32::EPSILON);
            loss += -safe_neg.ln();
            let neg_grad = neg_prob * scale;

            for index in 0..grad[left].len() {
                grad[left][index] += neg_grad * embeddings[negative][index];
                grad[negative][index] += neg_grad * embeddings[left][index];
            }
        }
    }

    loss
}

fn safe_prob(probability: f32) -> f32 {
    probability.clamp(f32::EPSILON, 1.0 - f32::EPSILON)
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

fn source_observed_targets<I>(node_count: usize, edges: I) -> Vec<HashSet<usize>>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut observed = vec![HashSet::new(); node_count];
    for (source, target) in edges {
        observed[source].insert(target);
    }

    observed
}

fn sample_negative_node(
    rng: &mut SplitMix64,
    node_count: usize,
    observed: &HashSet<usize>,
) -> Option<usize> {
    if observed.len() >= node_count {
        return None;
    }
    for _ in 0..32 {
        let candidate = rng.next_usize(node_count);
        if !observed.contains(&candidate) {
            return Some(candidate);
        }
    }
    (0..node_count).find(|candidate| !observed.contains(candidate))
}

fn build_directed_neighbors(
    node_count: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>> {
    let mut neighbors = vec![Vec::new(); node_count];
    for &(source, target) in edges {
        validate_node_index(source, node_count)?;
        validate_node_index(target, node_count)?;
        neighbors[source].push(target);
    }

    for row in &mut neighbors {
        row.sort_unstable();
        row.dedup();
    }

    Ok(neighbors)
}

fn add_self_neighbors(neighbors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut with_self = Vec::with_capacity(neighbors.len());
    for (node, source_neighbors) in neighbors.iter().enumerate() {
        let mut neighbors = source_neighbors.clone();
        if neighbors.binary_search(&node).is_err() {
            neighbors.push(node);
            neighbors.sort_unstable();
        }
        with_self.push(neighbors);
    }
    with_self
}

fn validate_node_features(
    expected_node_count: Option<usize>,
    input_dim: usize,
    features: &[Vec<f32>],
) -> Result<()> {
    if let Some(expected_nodes) = expected_node_count {
        if expected_nodes != features.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "expected {expected_nodes} rows of features, got {}",
                features.len(),
            )));
        }
    }

    if let Some((index, row)) = features
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != input_dim)
    {
        return Err(NeuralError::InvalidArgument(format!(
            "row {index} has width {}, expected {}",
            row.len(),
            input_dim,
        )));
    }

    Ok(())
}

fn validate_input_dim(input_dim: usize) -> Result<()> {
    if input_dim == 0 {
        return Err(NeuralError::InvalidArgument(
            "input feature dimension must be positive".to_string(),
        ));
    }
    Ok(())
}

fn graph_dense_cpu_fallback(
    backend: &BackendSelection,
    row_count: usize,
    input_width: usize,
    output_width: usize,
) -> Result<Option<BackendSelection>> {
    if backend.selected != "cpu"
        && row_count
            .saturating_mul(input_width)
            .saturating_mul(output_width)
            < GRAPH_SAGE_DENSE_DISPATCH_MIN_OPS
    {
        return Ok(Some(select_backend_for(
            Some("cpu"),
            BackendOperation::Dense,
        )?));
    }
    Ok(None)
}

fn validate_dense_backend(backend: &BackendSelection) -> Result<()> {
    if backend_supports_operation(&backend.selected, BackendOperation::Dense) {
        Ok(())
    } else {
        Err(NeuralError::InvalidArgument(format!(
            "backend {:?} does not implement GraphSAGE dense propagation",
            backend.selected
        )))
    }
}

fn validate_homogeneous_backend(backend: &BackendSelection) -> Result<()> {
    validate_dense_backend(backend)?;
    if backend_supports_operation(&backend.selected, BackendOperation::CsrDiffusion)
        && backend_supports_operation(&backend.selected, BackendOperation::CsrDiffusionBackward)
    {
        Ok(())
    } else {
        Err(NeuralError::InvalidArgument(format!(
            "backend {:?} does not implement GraphSAGE CSR aggregation and backward propagation",
            backend.selected
        )))
    }
}

fn validate_dimensions(hidden_dims: &[usize]) -> Result<()> {
    if hidden_dims.contains(&0) {
        return Err(NeuralError::InvalidArgument(
            "hidden_dims must contain only positive values".to_string(),
        ));
    }

    Ok(())
}

fn validate_node_index(node: usize, node_count: usize) -> Result<()> {
    if node >= node_count {
        return Err(NeuralError::InvalidArgument(format!(
            "node id {node} exceeds graph size {node_count}",
        )));
    }
    Ok(())
}

fn validate_relation_index(relation: usize, relation_count: usize) -> Result<()> {
    if relation >= relation_count {
        return Err(NeuralError::InvalidArgument(format!(
            "relation id {relation} exceeds relation count {relation_count}"
        )));
    }
    Ok(())
}

fn validate_relation_count(relation_count: usize) -> Result<()> {
    if relation_count == 0 {
        return Err(NeuralError::InvalidArgument(
            "relation_count must be positive for a hetero model".to_string(),
        ));
    }
    Ok(())
}

fn validate_neighbor_samples(neighbor_samples: &[usize], relation_count: usize) -> Result<()> {
    if neighbor_samples.is_empty() || neighbor_samples.len() == relation_count {
        return Ok(());
    }
    Err(NeuralError::InvalidArgument(
        "neighbor_samples must be empty or have one entry per relation".to_string(),
    ))
}

fn sample_hinsage_edges(
    node_count: usize,
    relation_count: usize,
    edges: &[HeteroTypedEdge],
    neighbor_samples: &[usize],
) -> Result<Vec<HeteroTypedEdge>> {
    if neighbor_samples.is_empty() {
        return Ok(edges.to_vec());
    }
    validate_neighbor_samples(neighbor_samples, relation_count)?;
    let mut grouped = vec![vec![Vec::<usize>::new(); relation_count]; node_count];
    for edge in edges {
        validate_node_index(edge.source, node_count)?;
        validate_node_index(edge.target, node_count)?;
        validate_relation_index(edge.relation, relation_count)?;
        grouped[edge.source][edge.relation].push(edge.target);
    }

    let mut sampled = Vec::new();
    for (source, by_relation) in grouped.iter_mut().enumerate() {
        for (relation, targets) in by_relation.iter_mut().enumerate() {
            targets.sort_unstable();
            targets.dedup();
            let limit = neighbor_samples[relation];
            let take_count = if limit == 0 {
                targets.len()
            } else {
                targets.len().min(limit)
            };
            for &target in targets.iter().take(take_count) {
                sampled.push(HeteroTypedEdge {
                    source,
                    target,
                    relation,
                });
            }
        }
    }
    Ok(sampled)
}

fn build_link_embeddings(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<Vec<f32>>> {
    let width = embeddings.first().map_or(0, Vec::len);
    if width == 0 {
        return Err(NeuralError::InvalidArgument(
            "embeddings must be non-empty with positive width".to_string(),
        ));
    }
    if embeddings.iter().any(|row| row.len() != width) {
        return Err(NeuralError::InvalidArgument(
            "embedding rows must have consistent width".to_string(),
        ));
    }
    pairs
        .par_iter()
        .map(|&(source, target)| {
            validate_node_index(source, embeddings.len())?;
            validate_node_index(target, embeddings.len())?;
            let left = &embeddings[source];
            let right = &embeddings[target];
            let mut row = Vec::with_capacity(width * 4);
            row.extend(left);
            row.extend(right);
            row.extend(left.iter().zip(right).map(|(l, r)| (l - r).abs()));
            row.extend(left.iter().zip(right).map(|(l, r)| l * r));
            Ok(row)
        })
        .collect()
}

fn validate_graphsage_artifact(artifact: &GraphSageEncoderArtifact) -> Result<()> {
    if artifact.artifact_type != GRAPH_SAGE_ARTIFACT_TYPE {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact type {}",
            artifact.artifact_type,
        )));
    }

    if artifact.artifact_version != GRAPH_SAGE_ARTIFACT_VERSION {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact version {}",
            artifact.artifact_version,
        )));
    }

    Ok(())
}

#[derive(Debug)]
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
        (self.next_u64() as f64 % (max as f64)) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphsage_dense_dispatch_avoids_small_device_launches() {
        for backend_name in crate::available_backends() {
            let backend = select_backend_for(Some(&backend_name), BackendOperation::Dense).unwrap();
            assert_eq!(
                graph_dense_cpu_fallback(&backend, 1, 8, 8)
                    .unwrap()
                    .is_some(),
                backend_name != "cpu"
            );
            assert!(
                graph_dense_cpu_fallback(&backend, 256, 8, 8)
                    .unwrap()
                    .is_none(),
                "{backend_name}"
            );
        }
    }

    #[test]
    fn graphsage_csr_means_match_cpu_on_available_backends() {
        let nodes = 1_024;
        let width = 16;
        let current = (0..nodes)
            .map(|node| {
                (0..width)
                    .map(|feature| ((node * width + feature) as f32 * 0.007).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let neighbors = (0..nodes)
            .map(|node| vec![(node + 1) % nodes, (node + 7) % nodes])
            .collect::<Vec<_>>();
        let operations = [BackendOperation::Dense, BackendOperation::CsrDiffusion];
        let cpu = crate::select_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected = homogeneous_neighbor_means(&current, &neighbors, width, &cpu).unwrap();
        for backend_name in crate::available_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                crate::select_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual = homogeneous_neighbor_means(&current, &neighbors, width, &backend)
                .unwrap_or_else(|error| panic!("{backend_name} GraphSAGE CSR: {error}"));
            for (actual_row, expected_row) in actual.iter().zip(&expected) {
                for (actual, expected) in actual_row.iter().zip(expected_row) {
                    assert!(
                        (actual - expected).abs() < 2.0e-4,
                        "{backend_name} GraphSAGE mean mismatch: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn hetero_graphsage_csr_means_match_cpu_on_available_backends() {
        let nodes = 512;
        let relations = 2;
        let width = 16;
        let current = (0..nodes)
            .map(|node| {
                (0..width)
                    .map(|feature| ((node * width + feature) as f32 * 0.011).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let neighbors = (0..nodes)
            .map(|node| {
                vec![
                    vec![(node + 1) % nodes, (node + 5) % nodes],
                    vec![(node + 11) % nodes],
                ]
            })
            .collect::<Vec<_>>();
        let operations = [BackendOperation::Dense, BackendOperation::CsrDiffusion];
        let cpu = crate::select_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected =
            heterogeneous_neighbor_means(&current, &neighbors, relations, width, &cpu).unwrap();
        for backend_name in crate::available_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                crate::select_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual =
                heterogeneous_neighbor_means(&current, &neighbors, relations, width, &backend)
                    .unwrap_or_else(|error| panic!("{backend_name} hetero GraphSAGE CSR: {error}"));
            for (actual_node, expected_node) in actual.iter().zip(&expected) {
                for (actual_relation, expected_relation) in actual_node.iter().zip(expected_node) {
                    for (actual, expected) in actual_relation.iter().zip(expected_relation) {
                        assert!(
                            (actual - expected).abs() < 2.0e-4,
                            "{backend_name} hetero GraphSAGE mean mismatch: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn graphsage_csr_backward_matches_cpu_on_available_backends() {
        let nodes = 512;
        let relations = 2;
        let width = 16;
        let current = (0..nodes)
            .map(|node| {
                (0..width)
                    .map(|feature| ((node * width + feature) as f32 * 0.013).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let homogeneous_neighbors = (0..nodes)
            .map(|node| vec![(node + 1) % nodes, (node + 9) % nodes])
            .collect::<Vec<_>>();
        let hetero_neighbors = (0..nodes)
            .map(|node| {
                vec![
                    vec![(node + 1) % nodes, (node + 9) % nodes],
                    vec![(node + 17) % nodes],
                ]
            })
            .collect::<Vec<_>>();
        let homogeneous_grad = (0..nodes)
            .map(|node| {
                (0..width)
                    .map(|feature| ((node + feature) as f32 * 0.017).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let hetero_grad = (0..nodes)
            .map(|node| {
                (0..relations)
                    .map(|relation| {
                        (0..width)
                            .map(|feature| {
                                ((node + relation * width + feature) as f32 * 0.019).sin()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let operations = [
            BackendOperation::Dense,
            BackendOperation::CsrDiffusion,
            BackendOperation::CsrDiffusionBackward,
        ];
        let cpu = crate::select_backend_for_operations(Some("cpu"), &operations).unwrap();
        let expected_homogeneous = homogeneous_neighbor_mean_backward(
            &current,
            &homogeneous_neighbors,
            width,
            &homogeneous_grad,
            &cpu,
        )
        .unwrap();
        let expected_hetero = heterogeneous_neighbor_mean_backward(
            &current,
            &hetero_neighbors,
            relations,
            width,
            &hetero_grad,
            &cpu,
        )
        .unwrap();
        for backend_name in crate::available_backends() {
            if backend_name == "cpu" {
                continue;
            }
            let Ok(backend) =
                crate::select_backend_for_operations(Some(&backend_name), &operations)
            else {
                continue;
            };
            let actual_homogeneous = homogeneous_neighbor_mean_backward(
                &current,
                &homogeneous_neighbors,
                width,
                &homogeneous_grad,
                &backend,
            )
            .unwrap_or_else(|error| panic!("{backend_name} GraphSAGE CSR backward: {error}"));
            let actual_hetero = heterogeneous_neighbor_mean_backward(
                &current,
                &hetero_neighbors,
                relations,
                width,
                &hetero_grad,
                &backend,
            )
            .unwrap_or_else(|error| {
                panic!("{backend_name} hetero GraphSAGE CSR backward: {error}")
            });
            for (actual_rows, expected_rows) in [
                (&actual_homogeneous, &expected_homogeneous),
                (&actual_hetero, &expected_hetero),
            ] {
                for (actual_row, expected_row) in actual_rows.iter().zip(expected_rows) {
                    for (actual, expected) in actual_row.iter().zip(expected_row) {
                        assert!(
                            (actual - expected).abs() < 2.0e-4,
                            "{backend_name} GraphSAGE backward mismatch: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn observed_targets_capture_edges_without_quadratic_negative_cache() {
        let observed = source_observed_targets(4, [(0, 1), (0, 2), (1, 3)]);

        assert!(observed[0].contains(&1));
        assert!(observed[0].contains(&2));
        assert!(!observed[0].contains(&0));
        assert!(observed[1].contains(&3));
    }

    #[test]
    fn observed_targets_can_cover_a_dense_source() {
        let observed = source_observed_targets(3, [(0, 0), (0, 1), (0, 2)]);

        assert_eq!(observed[0].len(), 3);
    }
}
