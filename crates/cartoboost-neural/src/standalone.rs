use crate::artifact::{
    build_embedding_table_artifact, ArtifactFallbackKind, EmbeddingTable, EmbeddingTableArtifact,
};
use crate::error::{NeuralError, Result};
use crate::graphsage::{
    GraphSageConfig, GraphSageEncoder, HeteroGraph, HeteroGraphSageConfig, HeteroGraphSageEncoder,
    HeteroTypedEdge, HinSageConfig, HinSageEncoder, HinSageGraph, HomogeneousGraph,
};
use crate::node2vec::{Node2VecConfig, Node2VecEncoder};
use crate::{
    backend_pair_sigmoid_scores_f32, backend_supports_operation,
    fit_embedding_table_with_options_and_backend, select_backend_for,
    select_backend_for_operations, BackendOperation, BackendSelection, GraphSageEncoderArtifact,
    Node2VecEncoderArtifact,
};
use cartoboost_core::loss::LossConfig;
use cartoboost_core::tree::{FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::{Booster, BoosterConfig, Dataset, Model};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const NEURAL_REGRESSOR_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_embedding_regressor";
const NODE2VEC_REGRESSOR_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_node2vec_regressor";
const GRAPHSAGE_REGRESSOR_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_graphsage_regressor";
const NODE2VEC_LINK_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_node2vec_link_predictor";
const GRAPHSAGE_LINK_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_graphsage_link_predictor";
const HETERO_GRAPHSAGE_REGRESSOR_ARTIFACT_TYPE: &str =
    "cartoboost.neural.standalone_hetero_graphsage_regressor";
const HINSAGE_REGRESSOR_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_hinsage_regressor";
const LINK_PAIR_DISPATCH_MIN_VALUES: usize = 16_384;
const HETERO_GRAPHSAGE_LINK_ARTIFACT_TYPE: &str =
    "cartoboost.neural.standalone_hetero_graphsage_link_predictor";
const HINSAGE_LINK_ARTIFACT_TYPE: &str = "cartoboost.neural.standalone_hinsage_link_predictor";
pub const STANDALONE_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StandaloneBoosterConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    #[serde(default = "default_standalone_backend")]
    pub backend: String,
}

impl Default for StandaloneBoosterConfig {
    fn default() -> Self {
        Self {
            n_estimators: 80,
            learning_rate: 0.07,
            max_depth: 4,
            min_samples_leaf: 2,
            min_gain: 0.0,
            backend: default_standalone_backend(),
        }
    }
}

fn default_standalone_backend() -> String {
    "cpu".to_string()
}

impl StandaloneBoosterConfig {
    fn to_booster_config(&self) -> BoosterConfig {
        BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters: vec![SplitterKind::AxisHistogram { bins: 256 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
            graph_leaf_smoothing: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphRegressionMode {
    Node,
    Pair,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEmbeddingRegressorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub dim: usize,
    pub booster_config: StandaloneBoosterConfig,
    pub dense_width: usize,
    pub table: EmbeddingTableArtifact,
    pub model: Model,
}

#[derive(Clone)]
pub struct NeuralEmbeddingRegressor {
    dim: usize,
    fallback: ArtifactFallbackKind,
    random_state: Option<u64>,
    support_prior_strength: f64,
    booster_config: StandaloneBoosterConfig,
    table: Option<EmbeddingTable>,
    model: Option<Model>,
    dense_width: usize,
}

impl NeuralEmbeddingRegressor {
    pub fn new(
        dim: usize,
        fallback: ArtifactFallbackKind,
        random_state: Option<u64>,
        support_prior_strength: f64,
        booster_config: StandaloneBoosterConfig,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "dim must be positive".to_string(),
            ));
        }
        validate_booster_config(&booster_config)?;
        Ok(Self {
            dim,
            fallback,
            random_state,
            support_prior_strength,
            booster_config,
            table: None,
            model: None,
            dense_width: 0,
        })
    }

    pub fn fit(&mut self, ids: &[u64], target: &[f64], dense: Option<&[Vec<f64>]>) -> Result<()> {
        validate_targets(target)?;
        if ids.len() != target.len() {
            return Err(NeuralError::InvalidArgument(
                "ids and target must have the same length".to_string(),
            ));
        }
        let target_f32 = target.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let table = fit_embedding_table_with_options_and_backend(
            self.dim,
            ids,
            &target_f32,
            self.fallback.clone(),
            self.random_state,
            self.support_prior_strength,
            Some(&self.booster_config.backend),
        )?;
        let features = embedding_rows_with_dense(&table, ids, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        let model = Booster::new_with_backend(
            self.booster_config.to_booster_config(),
            Some(&self.booster_config.backend),
        )
        .map_err(core_to_neural)?
        .fit(&dataset, target, None)
        .map_err(core_to_neural)?;
        self.dense_width = dense_width(dense)?;
        self.table = Some(table);
        self.model = Some(model);
        Ok(())
    }

    pub fn backend(&self) -> &str {
        &self.booster_config.backend
    }

    pub fn predict(&self, ids: &[u64], dense: Option<&[Vec<f64>]>) -> Result<Vec<f64>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        if dense_width(dense)? != self.dense_width {
            return Err(NeuralError::InvalidArgument(
                "dense feature width does not match fitted model".to_string(),
            ));
        }
        let features = embedding_rows_with_dense(table, ids, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        model.try_predict(&dataset).map_err(core_to_neural)
    }

    pub fn to_artifact(&self) -> Result<NeuralEmbeddingRegressorArtifact> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        Ok(NeuralEmbeddingRegressorArtifact {
            artifact_type: NEURAL_REGRESSOR_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            dim: self.dim,
            booster_config: self.booster_config.clone(),
            dense_width: self.dense_width,
            table: table_to_artifact(table)?,
            model: model.clone(),
        })
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_artifact()?)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: NeuralEmbeddingRegressorArtifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: NeuralEmbeddingRegressorArtifact) -> Result<Self> {
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            NEURAL_REGRESSOR_ARTIFACT_TYPE,
        )?;
        let table = EmbeddingTable::from_artifact(artifact.table)?;
        Ok(Self {
            dim: artifact.dim,
            fallback: table.artifact_metadata().fallback.clone(),
            random_state: None,
            support_prior_strength: 1.0,
            booster_config: artifact.booster_config,
            dense_width: artifact.dense_width,
            table: Some(table),
            model: Some(artifact.model),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node2VecRegressorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub mode: GraphRegressionMode,
    pub dense_width: usize,
    pub booster_config: StandaloneBoosterConfig,
    pub encoder: Node2VecEncoderArtifact,
    pub model: Model,
}

#[derive(Clone)]
pub struct Node2VecRegressor {
    config: Node2VecConfig,
    booster_config: StandaloneBoosterConfig,
    mode: GraphRegressionMode,
    dense_width: usize,
    encoder: Node2VecEncoder,
    model: Option<Model>,
}

impl Node2VecRegressor {
    pub fn new(config: Node2VecConfig, booster_config: StandaloneBoosterConfig) -> Result<Self> {
        validate_booster_config(&booster_config)?;
        let encoder =
            Node2VecEncoder::new_with_backend(config.clone(), Some(&booster_config.backend))?;
        Ok(Self {
            config,
            booster_config,
            mode: GraphRegressionMode::Node,
            dense_width: 0,
            encoder,
            model: None,
        })
    }

    pub fn backend(&self) -> &str {
        &self.booster_config.backend
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        node_count: usize,
        edges: &[(usize, usize)],
        edge_weights: Option<&[f32]>,
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
        target: &[f64],
    ) -> Result<()> {
        validate_targets(target)?;
        validate_row_count(row_nodes.len(), target.len(), "row_nodes", "target")?;
        if let Some(row_targets) = row_targets {
            validate_row_count(row_targets.len(), target.len(), "row_targets", "target")?;
        }
        self.encoder = Node2VecEncoder::new_with_backend(
            self.config.clone(),
            Some(&self.booster_config.backend),
        )?;
        self.encoder.fit(node_count, edges, edge_weights)?;
        self.fit_with_fitted_encoder(row_nodes, row_targets, dense, target)
    }

    pub fn fit_with_encoder(
        &mut self,
        encoder: Node2VecEncoder,
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
        target: &[f64],
    ) -> Result<()> {
        if encoder.output_dim() != self.config.dim {
            return Err(NeuralError::InvalidArgument(
                "pretrained Node2Vec encoder width does not match regressor config".to_string(),
            ));
        }
        self.encoder = encoder;
        self.fit_with_fitted_encoder(row_nodes, row_targets, dense, target)
    }

    fn fit_with_fitted_encoder(
        &mut self,
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
        target: &[f64],
    ) -> Result<()> {
        validate_targets(target)?;
        validate_row_count(row_nodes.len(), target.len(), "row_nodes", "target")?;
        if let Some(row_targets) = row_targets {
            validate_row_count(row_targets.len(), target.len(), "row_targets", "target")?;
        }
        let embeddings = self.encoder.encode()?.into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        self.model = Some(
            Booster::new_with_backend(
                self.booster_config.to_booster_config(),
                Some(&self.booster_config.backend),
            )
            .map_err(core_to_neural)?
            .fit(&dataset, target, None)
            .map_err(core_to_neural)?,
        );
        self.mode = if row_targets.is_some() {
            GraphRegressionMode::Pair
        } else {
            GraphRegressionMode::Node
        };
        self.dense_width = dense_width(dense)?;
        Ok(())
    }

    pub fn predict(
        &self,
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
    ) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        validate_prediction_mode(self.mode, row_targets)?;
        if dense_width(dense)? != self.dense_width {
            return Err(NeuralError::InvalidArgument(
                "dense feature width does not match fitted model".to_string(),
            ));
        }
        let embeddings = self.encoder.encode()?.into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        model.try_predict(&dataset).map_err(core_to_neural)
    }

    pub fn to_artifact(&self) -> Result<Node2VecRegressorArtifact> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        Ok(Node2VecRegressorArtifact {
            artifact_type: NODE2VEC_REGRESSOR_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            mode: self.mode,
            dense_width: self.dense_width,
            booster_config: self.booster_config.clone(),
            encoder: self.encoder.to_artifact(),
            model: model.clone(),
        })
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_artifact()?)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: Node2VecRegressorArtifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: Node2VecRegressorArtifact) -> Result<Self> {
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            NODE2VEC_REGRESSOR_ARTIFACT_TYPE,
        )?;
        let encoder = Node2VecEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            booster_config: artifact.booster_config,
            mode: artifact.mode,
            dense_width: artifact.dense_width,
            encoder,
            model: Some(artifact.model),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSageRegressorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub mode: GraphRegressionMode,
    pub dense_width: usize,
    pub input_dim: usize,
    pub booster_config: StandaloneBoosterConfig,
    pub encoder: GraphSageEncoderArtifact,
    pub model: Model,
    pub edges: Vec<(usize, usize)>,
}

#[derive(Clone)]
pub struct GraphSageRegressor {
    config: GraphSageConfig,
    input_dim: usize,
    booster_config: StandaloneBoosterConfig,
    mode: GraphRegressionMode,
    dense_width: usize,
    encoder: GraphSageEncoder,
    model: Option<Model>,
    edges: Vec<(usize, usize)>,
}

impl GraphSageRegressor {
    pub fn new(
        config: GraphSageConfig,
        input_dim: usize,
        booster_config: StandaloneBoosterConfig,
    ) -> Result<Self> {
        validate_booster_config(&booster_config)?;
        let encoder = GraphSageEncoder::new(config.clone(), input_dim)?;
        Ok(Self {
            config,
            input_dim,
            booster_config,
            mode: GraphRegressionMode::Node,
            dense_width: 0,
            encoder,
            model: None,
            edges: Vec::new(),
        })
    }

    pub fn backend(&self) -> &str {
        &self.booster_config.backend
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        node_features: &[Vec<f32>],
        edges: &[(usize, usize)],
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
        target: &[f64],
    ) -> Result<()> {
        validate_targets(target)?;
        validate_row_count(row_nodes.len(), target.len(), "row_nodes", "target")?;
        if let Some(row_targets) = row_targets {
            validate_row_count(row_targets.len(), target.len(), "row_targets", "target")?;
        }
        let graph = HomogeneousGraph::from_directed_edges(node_features.len(), edges)?;
        self.encoder = GraphSageEncoder::new(self.config.clone(), self.input_dim)?;
        let embeddings = self.encoder.fit(&graph, node_features)?.into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        self.model = Some(
            Booster::new_with_backend(
                self.booster_config.to_booster_config(),
                Some(&self.booster_config.backend),
            )
            .map_err(core_to_neural)?
            .fit(&dataset, target, None)
            .map_err(core_to_neural)?,
        );
        self.mode = if row_targets.is_some() {
            GraphRegressionMode::Pair
        } else {
            GraphRegressionMode::Node
        };
        self.dense_width = dense_width(dense)?;
        self.edges = edges.to_vec();
        Ok(())
    }

    pub fn predict(
        &self,
        node_features: &[Vec<f32>],
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
    ) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        validate_prediction_mode(self.mode, row_targets)?;
        if dense_width(dense)? != self.dense_width {
            return Err(NeuralError::InvalidArgument(
                "dense feature width does not match fitted model".to_string(),
            ));
        }
        let graph = HomogeneousGraph::from_directed_edges(node_features.len(), &self.edges)?;
        let embeddings = self
            .encoder
            .encode_graph(&graph, node_features)?
            .into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        model.try_predict(&dataset).map_err(core_to_neural)
    }

    pub fn to_artifact(&self) -> Result<GraphSageRegressorArtifact> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        Ok(GraphSageRegressorArtifact {
            artifact_type: GRAPHSAGE_REGRESSOR_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            mode: self.mode,
            dense_width: self.dense_width,
            input_dim: self.input_dim,
            booster_config: self.booster_config.clone(),
            encoder: self.encoder.to_artifact(),
            model: model.clone(),
            edges: self.edges.clone(),
        })
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_artifact()?)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: GraphSageRegressorArtifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: GraphSageRegressorArtifact) -> Result<Self> {
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            GRAPHSAGE_REGRESSOR_ARTIFACT_TYPE,
        )?;
        let encoder = GraphSageEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            input_dim: artifact.input_dim,
            booster_config: artifact.booster_config,
            mode: artifact.mode,
            dense_width: artifact.dense_width,
            encoder,
            model: Some(artifact.model),
            edges: artifact.edges,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroGraphSageRegressorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub mode: GraphRegressionMode,
    pub dense_width: usize,
    pub input_dim: usize,
    pub relation_count: usize,
    pub booster_config: StandaloneBoosterConfig,
    pub encoder: GraphSageEncoderArtifact,
    pub model: Model,
    pub edges: Vec<(usize, usize, usize)>,
}

#[derive(Clone)]
pub struct HeteroGraphSageRegressor {
    config: HeteroGraphSageConfig,
    input_dim: usize,
    relation_count: usize,
    booster_config: StandaloneBoosterConfig,
    mode: GraphRegressionMode,
    dense_width: usize,
    encoder: HeteroGraphSageEncoder,
    model: Option<Model>,
    edges: Vec<(usize, usize, usize)>,
}

impl HeteroGraphSageRegressor {
    pub fn new(
        config: HeteroGraphSageConfig,
        input_dim: usize,
        relation_count: usize,
        booster_config: StandaloneBoosterConfig,
    ) -> Result<Self> {
        validate_booster_config(&booster_config)?;
        let encoder = HeteroGraphSageEncoder::new(config.clone(), input_dim, relation_count)?;
        Ok(Self {
            config,
            input_dim,
            relation_count,
            booster_config,
            mode: GraphRegressionMode::Node,
            dense_width: 0,
            encoder,
            model: None,
            edges: Vec::new(),
        })
    }

    pub fn backend(&self) -> &str {
        &self.booster_config.backend
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        node_features: &[Vec<f32>],
        edges: &[(usize, usize, usize)],
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
        target: &[f64],
    ) -> Result<()> {
        validate_targets(target)?;
        validate_row_count(row_nodes.len(), target.len(), "row_nodes", "target")?;
        if let Some(row_targets) = row_targets {
            validate_row_count(row_targets.len(), target.len(), "row_targets", "target")?;
        }
        let typed_edges = typed_edges(edges);
        let graph =
            HeteroGraph::from_typed_edges(node_features.len(), self.relation_count, &typed_edges)?;
        self.encoder =
            HeteroGraphSageEncoder::new(self.config.clone(), self.input_dim, self.relation_count)?;
        let embeddings = self.encoder.fit(&graph, node_features)?.into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        self.model = Some(
            Booster::new_with_backend(
                self.booster_config.to_booster_config(),
                Some(&self.booster_config.backend),
            )
            .map_err(core_to_neural)?
            .fit(&dataset, target, None)
            .map_err(core_to_neural)?,
        );
        self.mode = if row_targets.is_some() {
            GraphRegressionMode::Pair
        } else {
            GraphRegressionMode::Node
        };
        self.dense_width = dense_width(dense)?;
        self.edges = edges.to_vec();
        Ok(())
    }

    pub fn predict(
        &self,
        node_features: &[Vec<f32>],
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
    ) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        validate_prediction_mode(self.mode, row_targets)?;
        if dense_width(dense)? != self.dense_width {
            return Err(NeuralError::InvalidArgument(
                "dense feature width does not match fitted model".to_string(),
            ));
        }
        let typed_edges = typed_edges(&self.edges);
        let graph =
            HeteroGraph::from_typed_edges(node_features.len(), self.relation_count, &typed_edges)?;
        let embeddings = self
            .encoder
            .encode_graph(&graph, node_features)?
            .into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        model.try_predict(&dataset).map_err(core_to_neural)
    }

    pub fn to_artifact(&self) -> Result<HeteroGraphSageRegressorArtifact> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        Ok(HeteroGraphSageRegressorArtifact {
            artifact_type: HETERO_GRAPHSAGE_REGRESSOR_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            mode: self.mode,
            dense_width: self.dense_width,
            input_dim: self.input_dim,
            relation_count: self.relation_count,
            booster_config: self.booster_config.clone(),
            encoder: self.encoder.to_artifact(),
            model: model.clone(),
            edges: self.edges.clone(),
        })
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_artifact()?)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: HeteroGraphSageRegressorArtifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: HeteroGraphSageRegressorArtifact) -> Result<Self> {
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            HETERO_GRAPHSAGE_REGRESSOR_ARTIFACT_TYPE,
        )?;
        let encoder = HeteroGraphSageEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            input_dim: artifact.input_dim,
            relation_count: artifact.relation_count,
            booster_config: artifact.booster_config,
            mode: artifact.mode,
            dense_width: artifact.dense_width,
            encoder,
            model: Some(artifact.model),
            edges: artifact.edges,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HinSageRegressorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub mode: GraphRegressionMode,
    pub dense_width: usize,
    pub input_dim: usize,
    pub node_type_count: usize,
    pub edge_type_triples: Vec<(usize, usize, usize)>,
    pub booster_config: StandaloneBoosterConfig,
    pub encoder: GraphSageEncoderArtifact,
    pub model: Model,
    pub node_types: Vec<usize>,
    pub edges: Vec<(usize, usize, usize)>,
}

#[derive(Clone)]
pub struct HinSageRegressor {
    config: HinSageConfig,
    input_dim: usize,
    node_type_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    booster_config: StandaloneBoosterConfig,
    mode: GraphRegressionMode,
    dense_width: usize,
    encoder: HinSageEncoder,
    model: Option<Model>,
    node_types: Vec<usize>,
    edges: Vec<(usize, usize, usize)>,
}

impl HinSageRegressor {
    pub fn new(
        config: HinSageConfig,
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        booster_config: StandaloneBoosterConfig,
    ) -> Result<Self> {
        validate_booster_config(&booster_config)?;
        let encoder = HinSageEncoder::new(
            config.clone(),
            input_dim,
            node_type_count,
            edge_type_triples.clone(),
        )?;
        Ok(Self {
            config,
            input_dim,
            node_type_count,
            edge_type_triples,
            booster_config,
            mode: GraphRegressionMode::Node,
            dense_width: 0,
            encoder,
            model: None,
            node_types: Vec::new(),
            edges: Vec::new(),
        })
    }

    pub fn backend(&self) -> &str {
        &self.booster_config.backend
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        node_features: &[Vec<f32>],
        node_types: &[usize],
        edges: &[(usize, usize, usize)],
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
        target: &[f64],
    ) -> Result<()> {
        validate_targets(target)?;
        validate_row_count(row_nodes.len(), target.len(), "row_nodes", "target")?;
        if let Some(row_targets) = row_targets {
            validate_row_count(row_targets.len(), target.len(), "row_targets", "target")?;
        }
        let graph = HinSageGraph::from_typed_schema(
            node_types.to_vec(),
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges(edges),
        )?;
        self.encoder = HinSageEncoder::new(
            self.config.clone(),
            self.input_dim,
            self.node_type_count,
            self.edge_type_triples.clone(),
        )?;
        let embeddings = self.encoder.fit(&graph, node_features)?.into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        self.model = Some(
            Booster::new_with_backend(
                self.booster_config.to_booster_config(),
                Some(&self.booster_config.backend),
            )
            .map_err(core_to_neural)?
            .fit(&dataset, target, None)
            .map_err(core_to_neural)?,
        );
        self.mode = if row_targets.is_some() {
            GraphRegressionMode::Pair
        } else {
            GraphRegressionMode::Node
        };
        self.dense_width = dense_width(dense)?;
        self.node_types = node_types.to_vec();
        self.edges = edges.to_vec();
        Ok(())
    }

    pub fn predict(
        &self,
        node_features: &[Vec<f32>],
        row_nodes: &[usize],
        row_targets: Option<&[usize]>,
        dense: Option<&[Vec<f64>]>,
    ) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        validate_prediction_mode(self.mode, row_targets)?;
        if dense_width(dense)? != self.dense_width {
            return Err(NeuralError::InvalidArgument(
                "dense feature width does not match fitted model".to_string(),
            ));
        }
        let graph = HinSageGraph::from_typed_schema(
            self.node_types.clone(),
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges(&self.edges),
        )?;
        let embeddings = self
            .encoder
            .encode_graph(&graph, node_features)?
            .into_inner();
        let features = graph_rows_with_dense(&embeddings, row_nodes, row_targets, dense)?;
        let dataset = Dataset::from_rows(features).map_err(core_to_neural)?;
        model.try_predict(&dataset).map_err(core_to_neural)
    }

    pub fn to_artifact(&self) -> Result<HinSageRegressorArtifact> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidArgument("model is not fitted".to_string()))?;
        Ok(HinSageRegressorArtifact {
            artifact_type: HINSAGE_REGRESSOR_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            mode: self.mode,
            dense_width: self.dense_width,
            input_dim: self.input_dim,
            node_type_count: self.node_type_count,
            edge_type_triples: self.edge_type_triples.clone(),
            booster_config: self.booster_config.clone(),
            encoder: self.encoder.to_artifact(),
            model: model.clone(),
            node_types: self.node_types.clone(),
            edges: self.edges.clone(),
        })
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(&self.to_artifact()?)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: HinSageRegressorArtifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: HinSageRegressorArtifact) -> Result<Self> {
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            HINSAGE_REGRESSOR_ARTIFACT_TYPE,
        )?;
        let encoder = HinSageEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            input_dim: artifact.input_dim,
            node_type_count: artifact.node_type_count,
            edge_type_triples: artifact.edge_type_triples,
            booster_config: artifact.booster_config,
            mode: artifact.mode,
            dense_width: artifact.dense_width,
            encoder,
            model: Some(artifact.model),
            node_types: artifact.node_types,
            edges: artifact.edges,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node2VecLinkPredictorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub encoder: Node2VecEncoderArtifact,
    #[serde(default)]
    pub backend: BackendSelection,
}

#[derive(Clone)]
pub struct Node2VecLinkPredictor {
    config: Node2VecConfig,
    encoder: Node2VecEncoder,
    backend: BackendSelection,
}

impl Node2VecLinkPredictor {
    pub fn new(config: Node2VecConfig) -> Result<Self> {
        Self::new_with_backend(config, None)
    }

    pub fn new_with_backend(config: Node2VecConfig, backend: Option<&str>) -> Result<Self> {
        let backend = select_backend_for_operations(
            backend,
            &[
                BackendOperation::PairScoring,
                BackendOperation::ScalarGraphTraining,
            ],
        )?;
        Ok(Self {
            encoder: Node2VecEncoder::new_with_backend(
                config.clone(),
                Some(backend.selected.as_str()),
            )?,
            config,
            backend,
        })
    }

    pub fn fit(
        &mut self,
        node_count: usize,
        edges: &[(usize, usize)],
        edge_weights: Option<&[f32]>,
    ) -> Result<()> {
        self.encoder = Node2VecEncoder::new_with_backend(
            self.config.clone(),
            Some(self.backend.selected.as_str()),
        )?;
        self.encoder.fit(node_count, edges, edge_weights)?;
        Ok(())
    }

    pub fn predict_scores(&self, pairs: &[(usize, usize)]) -> Result<Vec<f64>> {
        let embeddings = self.encoder.encode()?.into_inner();
        link_scores_with_backend(&embeddings, pairs, &self.backend)
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let artifact = Node2VecLinkPredictorArtifact {
            artifact_type: NODE2VEC_LINK_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            encoder: self.encoder.to_artifact(),
            backend: self.backend.clone(),
        };
        fs::write(path, serde_json::to_string_pretty(&artifact)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: Node2VecLinkPredictorArtifact = serde_json::from_str(&text)?;
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            NODE2VEC_LINK_ARTIFACT_TYPE,
        )?;
        let encoder = Node2VecEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        validate_pair_backend(&artifact.backend)?;
        Ok(Self {
            config,
            encoder,
            backend: artifact.backend,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSageLinkPredictorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub input_dim: usize,
    pub encoder: GraphSageEncoderArtifact,
    pub edges: Vec<(usize, usize)>,
}

#[derive(Clone)]
pub struct GraphSageLinkPredictor {
    config: GraphSageConfig,
    input_dim: usize,
    encoder: GraphSageEncoder,
    edges: Vec<(usize, usize)>,
}

impl GraphSageLinkPredictor {
    pub fn new(config: GraphSageConfig, input_dim: usize) -> Result<Self> {
        validate_pair_backend(&config.backend)?;
        Ok(Self {
            encoder: GraphSageEncoder::new(config.clone(), input_dim)?,
            config,
            input_dim,
            edges: Vec::new(),
        })
    }

    pub fn fit(&mut self, node_features: &[Vec<f32>], edges: &[(usize, usize)]) -> Result<()> {
        let graph = HomogeneousGraph::from_directed_edges(node_features.len(), edges)?;
        self.encoder = GraphSageEncoder::new(self.config.clone(), self.input_dim)?;
        self.encoder.fit(&graph, node_features)?;
        self.edges = edges.to_vec();
        Ok(())
    }

    pub fn predict_scores(
        &self,
        node_features: &[Vec<f32>],
        pairs: &[(usize, usize)],
    ) -> Result<Vec<f64>> {
        let graph = HomogeneousGraph::from_directed_edges(node_features.len(), &self.edges)?;
        let embeddings = self
            .encoder
            .encode_graph(&graph, node_features)?
            .into_inner();
        link_scores_with_backend(&embeddings, pairs, &self.config.backend)
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.config.backend
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let artifact = GraphSageLinkPredictorArtifact {
            artifact_type: GRAPHSAGE_LINK_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            input_dim: self.input_dim,
            encoder: self.encoder.to_artifact(),
            edges: self.edges.clone(),
        };
        fs::write(path, serde_json::to_string_pretty(&artifact)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: GraphSageLinkPredictorArtifact = serde_json::from_str(&text)?;
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            GRAPHSAGE_LINK_ARTIFACT_TYPE,
        )?;
        let encoder = GraphSageEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            input_dim: artifact.input_dim,
            encoder,
            edges: artifact.edges,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroGraphSageLinkPredictorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub input_dim: usize,
    pub relation_count: usize,
    pub encoder: GraphSageEncoderArtifact,
    pub edges: Vec<(usize, usize, usize)>,
}

#[derive(Clone)]
pub struct HeteroGraphSageLinkPredictor {
    config: HeteroGraphSageConfig,
    input_dim: usize,
    relation_count: usize,
    encoder: HeteroGraphSageEncoder,
    edges: Vec<(usize, usize, usize)>,
}

impl HeteroGraphSageLinkPredictor {
    pub fn new(
        config: HeteroGraphSageConfig,
        input_dim: usize,
        relation_count: usize,
    ) -> Result<Self> {
        validate_pair_backend(&config.backend)?;
        Ok(Self {
            encoder: HeteroGraphSageEncoder::new(config.clone(), input_dim, relation_count)?,
            config,
            input_dim,
            relation_count,
            edges: Vec::new(),
        })
    }

    pub fn fit(
        &mut self,
        node_features: &[Vec<f32>],
        edges: &[(usize, usize, usize)],
    ) -> Result<()> {
        let graph = HeteroGraph::from_typed_edges(
            node_features.len(),
            self.relation_count,
            &typed_edges(edges),
        )?;
        self.encoder =
            HeteroGraphSageEncoder::new(self.config.clone(), self.input_dim, self.relation_count)?;
        self.encoder.fit(&graph, node_features)?;
        self.edges = edges.to_vec();
        Ok(())
    }

    pub fn predict_scores(
        &self,
        node_features: &[Vec<f32>],
        pairs: &[(usize, usize)],
    ) -> Result<Vec<f64>> {
        let graph = HeteroGraph::from_typed_edges(
            node_features.len(),
            self.relation_count,
            &typed_edges(&self.edges),
        )?;
        let embeddings = self
            .encoder
            .encode_graph(&graph, node_features)?
            .into_inner();
        link_scores_with_backend(&embeddings, pairs, &self.config.backend)
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.config.backend
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let artifact = HeteroGraphSageLinkPredictorArtifact {
            artifact_type: HETERO_GRAPHSAGE_LINK_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            input_dim: self.input_dim,
            relation_count: self.relation_count,
            encoder: self.encoder.to_artifact(),
            edges: self.edges.clone(),
        };
        fs::write(path, serde_json::to_string_pretty(&artifact)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: HeteroGraphSageLinkPredictorArtifact = serde_json::from_str(&text)?;
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            HETERO_GRAPHSAGE_LINK_ARTIFACT_TYPE,
        )?;
        let encoder = HeteroGraphSageEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            input_dim: artifact.input_dim,
            relation_count: artifact.relation_count,
            encoder,
            edges: artifact.edges,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HinSageLinkPredictorArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub input_dim: usize,
    pub node_type_count: usize,
    pub edge_type_triples: Vec<(usize, usize, usize)>,
    pub node_types: Vec<usize>,
    pub encoder: GraphSageEncoderArtifact,
    pub edges: Vec<(usize, usize, usize)>,
}

#[derive(Clone)]
pub struct HinSageLinkPredictor {
    config: HinSageConfig,
    input_dim: usize,
    node_type_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    node_types: Vec<usize>,
    encoder: HinSageEncoder,
    edges: Vec<(usize, usize, usize)>,
}

impl HinSageLinkPredictor {
    pub fn new(
        config: HinSageConfig,
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
    ) -> Result<Self> {
        validate_pair_backend(&config.backend)?;
        Ok(Self {
            encoder: HinSageEncoder::new(
                config.clone(),
                input_dim,
                node_type_count,
                edge_type_triples.clone(),
            )?,
            config,
            input_dim,
            node_type_count,
            edge_type_triples,
            node_types: Vec::new(),
            edges: Vec::new(),
        })
    }

    pub fn fit(
        &mut self,
        node_features: &[Vec<f32>],
        node_types: &[usize],
        edges: &[(usize, usize, usize)],
    ) -> Result<()> {
        let graph = HinSageGraph::from_typed_schema(
            node_types.to_vec(),
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges(edges),
        )?;
        self.encoder = HinSageEncoder::new(
            self.config.clone(),
            self.input_dim,
            self.node_type_count,
            self.edge_type_triples.clone(),
        )?;
        self.encoder.fit(&graph, node_features)?;
        self.node_types = node_types.to_vec();
        self.edges = edges.to_vec();
        Ok(())
    }

    pub fn predict_scores(
        &self,
        node_features: &[Vec<f32>],
        pairs: &[(usize, usize)],
    ) -> Result<Vec<f64>> {
        let graph = HinSageGraph::from_typed_schema(
            self.node_types.clone(),
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges(&self.edges),
        )?;
        let embeddings = self
            .encoder
            .encode_graph(&graph, node_features)?
            .into_inner();
        link_scores_with_backend(&embeddings, pairs, &self.config.backend)
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.config.backend
    }

    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let artifact = HinSageLinkPredictorArtifact {
            artifact_type: HINSAGE_LINK_ARTIFACT_TYPE.to_string(),
            artifact_version: STANDALONE_ARTIFACT_VERSION,
            input_dim: self.input_dim,
            node_type_count: self.node_type_count,
            edge_type_triples: self.edge_type_triples.clone(),
            node_types: self.node_types.clone(),
            encoder: self.encoder.to_artifact(),
            edges: self.edges.clone(),
        };
        fs::write(path, serde_json::to_string_pretty(&artifact)?)?;
        Ok(())
    }

    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: HinSageLinkPredictorArtifact = serde_json::from_str(&text)?;
        validate_artifact_header(
            &artifact.artifact_type,
            artifact.artifact_version,
            HINSAGE_LINK_ARTIFACT_TYPE,
        )?;
        let encoder = HinSageEncoder::from_artifact(artifact.encoder)?;
        let config = encoder.config();
        Ok(Self {
            config,
            input_dim: artifact.input_dim,
            node_type_count: artifact.node_type_count,
            edge_type_triples: artifact.edge_type_triples,
            node_types: artifact.node_types,
            encoder,
            edges: artifact.edges,
        })
    }
}

fn embedding_rows_with_dense(
    table: &EmbeddingTable,
    ids: &[u64],
    dense: Option<&[Vec<f64>]>,
) -> Result<Vec<Vec<f64>>> {
    if let Some(dense) = dense {
        validate_row_count(dense.len(), ids.len(), "dense", "ids")?;
    }
    let block = table.encode_ids(ids, "standalone_embedding")?;
    let values = block.values;
    Ok((0..ids.len())
        .into_par_iter()
        .map(|row_index| {
            let start = row_index * table.dim();
            let mut row = dense
                .and_then(|dense| dense.get(row_index).cloned())
                .unwrap_or_default();
            row.extend(
                values[start..start + table.dim()]
                    .iter()
                    .map(|value| f64::from(*value)),
            );
            row
        })
        .collect())
}

fn graph_rows_with_dense(
    embeddings: &[Vec<f32>],
    row_nodes: &[usize],
    row_targets: Option<&[usize]>,
    dense: Option<&[Vec<f64>]>,
) -> Result<Vec<Vec<f64>>> {
    if let Some(dense) = dense {
        validate_row_count(dense.len(), row_nodes.len(), "dense", "row_nodes")?;
    }
    if let Some(row_targets) = row_targets {
        validate_row_count(
            row_targets.len(),
            row_nodes.len(),
            "row_targets",
            "row_nodes",
        )?;
    }
    row_nodes
        .par_iter()
        .enumerate()
        .map(|(row_index, &source)| {
            let source_vec = embeddings.get(source).ok_or_else(|| {
                NeuralError::InvalidArgument("row node id exceeds fitted node count".to_string())
            })?;
            let mut row = dense
                .and_then(|dense| dense.get(row_index).cloned())
                .unwrap_or_default();
            if let Some(targets) = row_targets {
                let target_vec = embeddings.get(targets[row_index]).ok_or_else(|| {
                    NeuralError::InvalidArgument(
                        "row target id exceeds fitted node count".to_string(),
                    )
                })?;
                append_pair_features(&mut row, source_vec, target_vec);
            } else {
                row.extend(source_vec.iter().map(|value| f64::from(*value)));
            }
            Ok(row)
        })
        .collect()
}

fn append_pair_features(row: &mut Vec<f64>, source: &[f32], target: &[f32]) {
    row.extend(source.iter().map(|value| f64::from(*value)));
    row.extend(target.iter().map(|value| f64::from(*value)));
    row.extend(
        source
            .iter()
            .zip(target.iter())
            .map(|(left, right)| f64::from((*left - *right).abs())),
    );
    row.extend(
        source
            .iter()
            .zip(target.iter())
            .map(|(left, right)| f64::from(*left * *right)),
    );
}

fn link_scores_with_backend(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
    backend: &BackendSelection,
) -> Result<Vec<f64>> {
    let cpu_backend = if !should_accelerate_link_scores(
        backend,
        pairs.len(),
        embeddings.first().map_or(0, Vec::len),
    ) {
        Some(select_backend_for(
            Some("cpu"),
            BackendOperation::PairScoring,
        )?)
    } else {
        None
    };
    backend_pair_sigmoid_scores_f32(cpu_backend.as_ref().unwrap_or(backend), embeddings, pairs)
        .map_err(|err| match err {
            cartoboost_accelerator::AcceleratorError::InvalidArgument(message)
                if message.contains("pair scoring node ids must be within") =>
            {
                NeuralError::InvalidArgument(
                    "source or target node id exceeds node count".to_string(),
                )
            }
            other => other.into(),
        })
}

fn should_accelerate_link_scores(
    backend: &BackendSelection,
    pair_count: usize,
    embedding_width: usize,
) -> bool {
    backend.selected != "cpu"
        && pair_count.saturating_mul(embedding_width) >= LINK_PAIR_DISPATCH_MIN_VALUES
}

fn validate_pair_backend(backend: &BackendSelection) -> Result<()> {
    if backend_supports_operation(&backend.selected, BackendOperation::PairScoring) {
        Ok(())
    } else {
        Err(NeuralError::InvalidArgument(format!(
            "backend {:?} does not implement link-prediction pair scoring",
            backend.selected
        )))
    }
}

fn typed_edges(edges: &[(usize, usize, usize)]) -> Vec<HeteroTypedEdge> {
    edges
        .iter()
        .map(|&(source, target, relation)| HeteroTypedEdge {
            source,
            target,
            relation,
        })
        .collect()
}

fn table_to_artifact(table: &EmbeddingTable) -> Result<EmbeddingTableArtifact> {
    build_embedding_table_artifact(
        table.dim(),
        table.rows().to_vec(),
        table.artifact_metadata().fallback.clone(),
    )
}

fn validate_booster_config(config: &StandaloneBoosterConfig) -> Result<()> {
    if config.n_estimators == 0 {
        return Err(NeuralError::InvalidArgument(
            "n_estimators must be positive".to_string(),
        ));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(NeuralError::InvalidArgument(
            "learning_rate must be positive and finite".to_string(),
        ));
    }
    if config.max_depth == 0 {
        return Err(NeuralError::InvalidArgument(
            "max_depth must be positive".to_string(),
        ));
    }
    if config.min_samples_leaf == 0 {
        return Err(NeuralError::InvalidArgument(
            "min_samples_leaf must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_targets(target: &[f64]) -> Result<()> {
    if target.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "target must not be empty".to_string(),
        ));
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(NeuralError::InvalidArgument(
            "target values must be finite".to_string(),
        ));
    }
    Ok(())
}

fn validate_row_count(
    actual: usize,
    expected: usize,
    actual_name: &str,
    expected_name: &str,
) -> Result<()> {
    if actual != expected {
        return Err(NeuralError::InvalidArgument(format!(
            "{actual_name} length {actual} does not match {expected_name} length {expected}"
        )));
    }
    Ok(())
}

fn validate_prediction_mode(
    mode: GraphRegressionMode,
    row_targets: Option<&[usize]>,
) -> Result<()> {
    match (mode, row_targets.is_some()) {
        (GraphRegressionMode::Node, false) | (GraphRegressionMode::Pair, true) => Ok(()),
        (GraphRegressionMode::Node, true) => Err(NeuralError::InvalidArgument(
            "model was fitted for node regression but pair targets were provided".to_string(),
        )),
        (GraphRegressionMode::Pair, false) => Err(NeuralError::InvalidArgument(
            "model was fitted for pair regression but row_targets were not provided".to_string(),
        )),
    }
}

fn dense_width(dense: Option<&[Vec<f64>]>) -> Result<usize> {
    let Some(rows) = dense else {
        return Ok(0);
    };
    let width = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != width) {
        return Err(NeuralError::InvalidArgument(
            "dense rows must have a consistent width".to_string(),
        ));
    }
    Ok(width)
}

fn validate_artifact_header(actual_type: &str, version: u32, expected_type: &str) -> Result<()> {
    if actual_type != expected_type {
        return Err(NeuralError::InvalidArgument(format!(
            "unexpected artifact type: {actual_type}"
        )));
    }
    if version != STANDALONE_ARTIFACT_VERSION {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact version: {version}"
        )));
    }
    Ok(())
}

fn core_to_neural(err: cartoboost_core::CartoBoostError) -> NeuralError {
    NeuralError::InvalidArgument(err.to_string())
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn link_inference_avoids_small_device_launches() {
        for backend_name in crate::available_backends() {
            let backend =
                select_backend_for(Some(&backend_name), BackendOperation::PairScoring).unwrap();
            assert!(!should_accelerate_link_scores(&backend, 1, 16));
            assert_eq!(
                should_accelerate_link_scores(&backend, 1_024, 16),
                backend_name != "cpu"
            );
        }
    }
}
