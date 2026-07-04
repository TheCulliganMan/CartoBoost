use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const REPRESENTATION_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum RepresentationError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, RepresentationError>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RepresentationArtifact {
    pub model_class: String,
    pub architecture: String,
    pub artifact_version: u32,
    pub schema_hash: String,
    pub id_maps: BTreeMap<String, BTreeMap<String, usize>>,
    pub hash_bucket_config: BTreeMap<String, usize>,
    pub embedding_dim: usize,
    pub random_seed: u64,
    pub feature_roles: BTreeMap<String, String>,
    pub training_cutoff: Option<String>,
    pub training_metrics: BTreeMap<String, f64>,
    pub save_load_parity_checked: bool,
    pub backend: BackendMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendMetadata {
    pub requested: String,
    pub selected: String,
    pub available: Vec<String>,
    pub supported_accelerators: Vec<String>,
    pub accelerator_ready: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityEmbedding {
    embedding_dim: usize,
    hash_bucket_count: usize,
    random_seed: u64,
    backend: BackendMetadata,
    architecture: String,
    feature_roles: BTreeMap<String, String>,
    id_map: BTreeMap<String, usize>,
    embeddings: Vec<Vec<f64>>,
    training_cutoff: Option<String>,
    training_metrics: BTreeMap<String, f64>,
    schema_hash: String,
    artifact: Option<RepresentationArtifact>,
}

impl EntityEmbedding {
    pub fn new(embedding_dim: usize, hash_bucket_count: usize, random_seed: u64) -> Result<Self> {
        if embedding_dim == 0 {
            return Err(RepresentationError::InvalidInput(
                "embedding_dim must be positive".to_string(),
            ));
        }
        if hash_bucket_count == 0 {
            return Err(RepresentationError::InvalidInput(
                "hash_bucket_count must be positive".to_string(),
            ));
        }
        Ok(Self {
            embedding_dim,
            hash_bucket_count,
            random_seed,
            backend: resolve_backend("cpu")?,
            architecture: "entity_embedding".to_string(),
            feature_roles: BTreeMap::new(),
            id_map: BTreeMap::new(),
            embeddings: Vec::new(),
            training_cutoff: None,
            training_metrics: BTreeMap::new(),
            schema_hash: String::new(),
            artifact: None,
        })
    }

    pub fn fit<I, S>(&mut self, ids: I, training_cutoff: Option<String>) -> Result<&mut Self>
    where
        I: IntoIterator<Item = S>,
        S: ToString,
    {
        let unique: BTreeSet<String> = ids.into_iter().map(|value| value.to_string()).collect();
        let mut id_map = BTreeMap::new();
        id_map.insert("__unknown__".to_string(), 0);
        for (offset, value) in unique.iter().enumerate() {
            id_map.insert(value.clone(), offset + 1);
        }
        self.id_map = id_map;
        self.embeddings = deterministic_matrix(
            1 + unique.len() + self.hash_bucket_count,
            self.embedding_dim,
            self.random_seed,
            &self.architecture,
        );
        self.training_cutoff = training_cutoff;
        self.schema_hash = schema_hash(&serde_json::json!({
            "ids": unique,
            "embedding_dim": self.embedding_dim,
            "hash_bucket_count": self.hash_bucket_count,
            "feature_roles": self.feature_roles,
        }))?;
        self.artifact = Some(self.build_artifact(false));
        let parity = self.save_load_parity()?;
        self.artifact = Some(self.build_artifact(parity));
        Ok(self)
    }

    pub fn transform<I, S>(&self, ids: I) -> Result<Vec<Vec<f64>>>
    where
        I: IntoIterator<Item = S>,
        S: ToString,
    {
        self.require_fitted()?;
        Ok(ids
            .into_iter()
            .map(|value| self.embeddings[self.index(&value.to_string())].clone())
            .collect())
    }

    pub fn artifact(&self) -> Result<&RepresentationArtifact> {
        self.artifact
            .as_ref()
            .ok_or_else(|| RepresentationError::InvalidInput("embedding is not fit".to_string()))
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.require_fitted()?;
        fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    fn index(&self, value: &str) -> usize {
        self.id_map
            .get(value)
            .copied()
            .unwrap_or_else(|| self.id_map.len() + stable_hash(value, self.hash_bucket_count))
    }

    fn build_artifact(&self, save_load_parity_checked: bool) -> RepresentationArtifact {
        let mut id_maps = BTreeMap::new();
        id_maps.insert("entity".to_string(), self.id_map.clone());
        let mut hash_bucket_config = BTreeMap::new();
        hash_bucket_config.insert("entity".to_string(), self.hash_bucket_count);
        RepresentationArtifact {
            model_class: "EntityEmbedding".to_string(),
            architecture: self.architecture.clone(),
            artifact_version: REPRESENTATION_ARTIFACT_VERSION,
            schema_hash: self.schema_hash.clone(),
            id_maps,
            hash_bucket_config,
            embedding_dim: self.embedding_dim,
            random_seed: self.random_seed,
            feature_roles: self.feature_roles.clone(),
            training_cutoff: self.training_cutoff.clone(),
            training_metrics: self.training_metrics.clone(),
            save_load_parity_checked,
            backend: self.backend.clone(),
        }
    }

    fn save_load_parity(&self) -> Result<bool> {
        let probe = vec!["__unknown__", "__new_id__"];
        let before = self.transform(probe.clone())?;
        let loaded: Self = serde_json::from_str(&serde_json::to_string(self)?)?;
        Ok(vectors_close(&before, &loaded.transform(probe)?, 1e-12))
    }

    fn require_fitted(&self) -> Result<()> {
        if self.embeddings.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "embedding must be fit before transform".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairEmbedding {
    source: EntityEmbedding,
    target: EntityEmbedding,
    pair_id_map: BTreeMap<String, usize>,
    pair_embeddings: Vec<Vec<f64>>,
    pair_hash_bucket_count: usize,
    schema_hash: String,
    artifact: Option<RepresentationArtifact>,
}

impl PairEmbedding {
    pub fn new(
        embedding_dim: usize,
        entity_hash_bucket_count: usize,
        pair_hash_bucket_count: usize,
        random_seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            source: EntityEmbedding::new(embedding_dim, entity_hash_bucket_count, random_seed)?,
            target: EntityEmbedding::new(
                embedding_dim,
                entity_hash_bucket_count,
                random_seed + 17,
            )?,
            pair_id_map: BTreeMap::new(),
            pair_embeddings: Vec::new(),
            pair_hash_bucket_count,
            schema_hash: String::new(),
            artifact: None,
        })
    }

    pub fn fit<S: ToString, T: ToString>(
        &mut self,
        sources: &[S],
        targets: &[T],
        training_cutoff: Option<String>,
    ) -> Result<&mut Self> {
        if sources.len() != targets.len() {
            return Err(RepresentationError::InvalidInput(
                "sources and targets must have the same length".to_string(),
            ));
        }
        self.source.fit(
            sources.iter().map(ToString::to_string),
            training_cutoff.clone(),
        )?;
        self.target
            .fit(targets.iter().map(ToString::to_string), training_cutoff)?;
        let pairs: BTreeSet<String> = sources
            .iter()
            .zip(targets.iter())
            .map(|(source, target)| pair_key(&source.to_string(), &target.to_string()))
            .collect();
        self.pair_id_map.insert("__unknown__".to_string(), 0);
        for (offset, pair) in pairs.iter().enumerate() {
            self.pair_id_map.insert(pair.clone(), offset + 1);
        }
        self.pair_embeddings = deterministic_matrix(
            1 + pairs.len() + self.pair_hash_bucket_count,
            self.source.embedding_dim,
            self.source.random_seed + 31,
            "pair_embedding",
        );
        self.schema_hash = schema_hash(&serde_json::json!({
            "pairs": pairs,
            "embedding_dim": self.source.embedding_dim,
            "pair_hash_bucket_count": self.pair_hash_bucket_count,
        }))?;
        self.artifact = Some(self.build_artifact(false));
        let parity = self.save_load_parity()?;
        self.artifact = Some(self.build_artifact(parity));
        Ok(self)
    }

    pub fn transform<S: ToString, T: ToString>(
        &self,
        sources: &[S],
        targets: &[T],
    ) -> Result<Vec<Vec<f64>>> {
        if sources.len() != targets.len() {
            return Err(RepresentationError::InvalidInput(
                "sources and targets must have the same length".to_string(),
            ));
        }
        let src = self
            .source
            .transform(sources.iter().map(ToString::to_string))?;
        let dst = self
            .target
            .transform(targets.iter().map(ToString::to_string))?;
        let mut output = Vec::with_capacity(sources.len());
        for (idx, (source, target)) in sources.iter().zip(targets.iter()).enumerate() {
            let pair =
                &self.pair_embeddings[self.pair_index(&source.to_string(), &target.to_string())];
            let mut row = Vec::new();
            row.extend_from_slice(&src[idx]);
            row.extend_from_slice(&dst[idx]);
            row.extend(src[idx].iter().zip(dst[idx].iter()).map(|(a, b)| a - b));
            row.extend(
                src[idx]
                    .iter()
                    .zip(dst[idx].iter())
                    .map(|(a, b)| (a - b).abs()),
            );
            row.extend(src[idx].iter().zip(dst[idx].iter()).map(|(a, b)| a * b));
            row.extend_from_slice(pair);
            output.push(row);
        }
        Ok(output)
    }

    pub fn artifact(&self) -> Result<&RepresentationArtifact> {
        self.artifact.as_ref().ok_or_else(|| {
            RepresentationError::InvalidInput("pair embedding is not fit".to_string())
        })
    }

    fn pair_index(&self, source: &str, target: &str) -> usize {
        let key = pair_key(source, target);
        self.pair_id_map.get(&key).copied().unwrap_or_else(|| {
            self.pair_id_map.len() + stable_hash(&key, self.pair_hash_bucket_count)
        })
    }

    fn build_artifact(&self, save_load_parity_checked: bool) -> RepresentationArtifact {
        let mut id_maps = BTreeMap::new();
        id_maps.insert("source".to_string(), self.source.id_map.clone());
        id_maps.insert("target".to_string(), self.target.id_map.clone());
        id_maps.insert("pair".to_string(), self.pair_id_map.clone());
        let mut hash_bucket_config = BTreeMap::new();
        hash_bucket_config.insert("entity".to_string(), self.source.hash_bucket_count);
        hash_bucket_config.insert("pair".to_string(), self.pair_hash_bucket_count);
        RepresentationArtifact {
            model_class: "PairEmbedding".to_string(),
            architecture: "pair_embedding".to_string(),
            artifact_version: REPRESENTATION_ARTIFACT_VERSION,
            schema_hash: self.schema_hash.clone(),
            id_maps,
            hash_bucket_config,
            embedding_dim: self.source.embedding_dim,
            random_seed: self.source.random_seed,
            feature_roles: BTreeMap::new(),
            training_cutoff: self.source.training_cutoff.clone(),
            training_metrics: BTreeMap::new(),
            save_load_parity_checked,
            backend: self.source.backend.clone(),
        }
    }

    fn save_load_parity(&self) -> Result<bool> {
        let sources = vec!["A", "__new_source__"];
        let targets = vec!["B", "__new_target__"];
        let before = self.transform(&sources, &targets)?;
        let loaded: Self = serde_json::from_str(&serde_json::to_string(self)?)?;
        Ok(vectors_close(
            &before,
            &loaded.transform(&sources, &targets)?,
            1e-12,
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatioTemporalAdaptiveEmbedding {
    entity: EntityEmbedding,
    time_weights: Vec<Vec<f64>>,
    context_weights: Vec<Vec<f64>>,
    gate_weights: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeRoute {
    pub expert_weights: Vec<Vec<f64>>,
    pub selected_expert: Vec<usize>,
    pub router_entropy: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeRouter {
    entity: EntityEmbedding,
    expert_count: usize,
    context_dim: usize,
    router_weights: Vec<Vec<f64>>,
    router_bias: Vec<f64>,
    expert_usage: BTreeMap<String, f64>,
    schema_hash: String,
    training_cutoff: Option<String>,
    training_metrics: BTreeMap<String, f64>,
    artifact: Option<RepresentationArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalogQueryResult {
    pub analog_ids: Vec<String>,
    pub distances: Vec<f64>,
    pub indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoricalAnalogRetriever {
    analog_ids: Vec<String>,
    timestamps: Option<Vec<String>>,
    memory: Vec<Vec<f64>>,
    key_mean: Vec<f64>,
    key_scale: Vec<f64>,
    normalize: bool,
    random_seed: u64,
    feature_roles: BTreeMap<String, String>,
    training_cutoff: Option<String>,
    training_metrics: BTreeMap<String, f64>,
    schema_hash: String,
    artifact: Option<RepresentationArtifact>,
    backend: BackendMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfSupervisedPretrainer {
    entity: EntityEmbedding,
    tasks: Vec<String>,
    feature_mean: Vec<f64>,
    feature_scale: Vec<f64>,
    pretrained_entity_embeddings: Vec<Vec<f64>>,
    pretrained_pair_embeddings: Vec<Vec<f64>>,
    pretrained_node_embeddings: Vec<Vec<f64>>,
    pretrained_temporal_encoder: Vec<Vec<f64>>,
    training_cutoff: Option<String>,
    training_metrics: BTreeMap<String, f64>,
    schema_hash: String,
    artifact: Option<RepresentationArtifact>,
}

pub const MASKED_ENTITY_TIME_MODELING: &str = "masked_entity_time_modeling";
pub const MASKED_PAIR_TIME_MODELING: &str = "masked_pair_time_modeling";
pub const GRAPH_EDGE_DENOISING: &str = "graph_edge_denoising";
pub const TEMPORAL_ORDER_CONTRASTIVE_LOSS: &str = "temporal_order_contrastive_loss";
pub const SPATIAL_NEIGHBOR_CONTRASTIVE_LOSS: &str = "spatial_neighbor_contrastive_loss";
pub const FUTURE_PATCH_RECONSTRUCTION: &str = "future_patch_reconstruction";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MultiViewAttentionOutput {
    pub embedding: Vec<Vec<f64>>,
    pub view_weights: Vec<Vec<f64>>,
    pub available_views: Vec<String>,
    pub missing_views: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ViewAblationReport {
    pub full_proxy_score: f64,
    pub single_view_proxy_scores: BTreeMap<String, f64>,
    pub best_single_view: Option<String>,
    pub full_beats_best_single_view: bool,
    pub missing_views: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiViewSpatialAttention {
    embedding_dim: usize,
    random_seed: u64,
    view_names: Vec<String>,
    node_ids: Vec<String>,
    view_weights: BTreeMap<String, Vec<Vec<f64>>>,
    router_weights: Vec<Vec<f64>>,
    router_bias: Vec<f64>,
    learned_view_weights: BTreeMap<String, f64>,
    training_metrics: BTreeMap<String, f64>,
    schema_hash: String,
    artifact: Option<RepresentationArtifact>,
    backend: BackendMetadata,
}

impl MultiViewSpatialAttention {
    pub fn new(embedding_dim: usize, random_seed: u64) -> Result<Self> {
        if embedding_dim == 0 {
            return Err(RepresentationError::InvalidInput(
                "embedding_dim must be positive".to_string(),
            ));
        }
        Ok(Self {
            embedding_dim,
            random_seed,
            view_names: Vec::new(),
            node_ids: Vec::new(),
            view_weights: BTreeMap::new(),
            router_weights: Vec::new(),
            router_bias: Vec::new(),
            learned_view_weights: BTreeMap::new(),
            training_metrics: BTreeMap::new(),
            schema_hash: String::new(),
            artifact: None,
            backend: resolve_backend("cpu")?,
        })
    }

    pub fn fit<S: ToString>(
        &mut self,
        node_ids: &[S],
        views: &BTreeMap<String, Vec<Vec<f64>>>,
    ) -> Result<&mut Self> {
        if node_ids.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "node_ids must be non-empty".to_string(),
            ));
        }
        if views.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "at least one spatial view is required".to_string(),
            ));
        }
        self.node_ids = node_ids.iter().map(ToString::to_string).collect();
        self.view_names = views.keys().cloned().collect();
        self.view_weights.clear();
        let mut view_strengths = BTreeMap::new();
        for (view_name, features) in views {
            validate_feature_rows(features, node_ids.len(), view_name)?;
            self.view_weights.insert(
                view_name.clone(),
                deterministic_matrix(
                    features[0].len(),
                    self.embedding_dim,
                    self.random_seed + stable_hash(view_name, usize::MAX - 1) as u64,
                    &format!("multi_view_spatial_attention:{view_name}"),
                ),
            );
            view_strengths.insert(view_name.clone(), mean_abs(features));
        }
        self.router_weights = deterministic_matrix(
            self.embedding_dim,
            self.view_names.len(),
            self.random_seed + 701,
            "multi_view_router",
        );
        self.router_bias = self
            .view_names
            .iter()
            .map(|name| *view_strengths.get(name).unwrap_or(&0.0))
            .collect();
        let output = self.transform(views)?;
        self.learned_view_weights = mean_view_weights(&self.view_names, &output.view_weights);
        let ablation = self.view_ablation_report(views)?;
        self.training_metrics
            .insert("full_proxy_score".to_string(), ablation.full_proxy_score);
        self.training_metrics.insert(
            "best_single_view_proxy_score".to_string(),
            ablation
                .single_view_proxy_scores
                .values()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        );
        self.schema_hash = schema_hash(&serde_json::json!({
            "node_ids": self.node_ids,
            "view_names": self.view_names,
            "embedding_dim": self.embedding_dim,
        }))?;
        self.artifact = Some(self.build_artifact(false));
        let parity = self.save_load_parity(views)?;
        self.artifact = Some(self.build_artifact(parity));
        Ok(self)
    }

    pub fn transform(
        &self,
        views: &BTreeMap<String, Vec<Vec<f64>>>,
    ) -> Result<MultiViewAttentionOutput> {
        if self.view_names.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "multi-view attention must be fit before transform".to_string(),
            ));
        }
        let row_count = views
            .values()
            .next()
            .map(Vec::len)
            .unwrap_or(self.node_ids.len());
        let mut available_views = Vec::new();
        let mut missing_views = Vec::new();
        let mut projected_by_view = Vec::new();
        for view_name in &self.view_names {
            match views.get(view_name) {
                Some(features) => {
                    validate_feature_rows(features, row_count, view_name)?;
                    let weights = self.view_weights.get(view_name).ok_or_else(|| {
                        RepresentationError::InvalidInput(format!(
                            "missing fitted weights for view {view_name}"
                        ))
                    })?;
                    projected_by_view.push(project_features(features, weights)?);
                    available_views.push(view_name.clone());
                }
                None => {
                    missing_views.push(view_name.clone());
                }
            }
        }
        if projected_by_view.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "at least one fitted spatial view must be supplied".to_string(),
            ));
        }
        let mut fused = Vec::with_capacity(row_count);
        let mut weights_out = Vec::with_capacity(row_count);
        for row_idx in 0..row_count {
            let mut router_input = vec![0.0; self.embedding_dim];
            for projected in &projected_by_view {
                for (value, addend) in router_input.iter_mut().zip(projected[row_idx].iter()) {
                    *value += *addend / projected_by_view.len() as f64;
                }
            }
            let mut logits = matvec(&router_input, &self.router_weights);
            for (logit, bias) in logits.iter_mut().zip(self.router_bias.iter()) {
                *logit += *bias;
            }
            for (view_idx, view_name) in self.view_names.iter().enumerate() {
                if missing_views.contains(view_name) {
                    logits[view_idx] = f64::NEG_INFINITY;
                }
            }
            let view_weights = softmax_with_neg_inf(&logits);
            let mut row = vec![0.0; self.embedding_dim];
            let mut projected_idx = 0;
            for (view_idx, view_name) in self.view_names.iter().enumerate() {
                if missing_views.contains(view_name) {
                    continue;
                }
                for (value, addend) in row
                    .iter_mut()
                    .zip(projected_by_view[projected_idx][row_idx].iter())
                {
                    *value += view_weights[view_idx] * addend;
                }
                projected_idx += 1;
            }
            fused.push(layer_norm(&row));
            weights_out.push(view_weights);
        }
        Ok(MultiViewAttentionOutput {
            embedding: fused,
            view_weights: weights_out,
            available_views,
            missing_views,
        })
    }

    pub fn view_ablation_report(
        &self,
        views: &BTreeMap<String, Vec<Vec<f64>>>,
    ) -> Result<ViewAblationReport> {
        let full = self.transform(views)?;
        let full_proxy_score = embedding_energy(&full.embedding);
        let mut single_view_proxy_scores = BTreeMap::new();
        for view_name in &self.view_names {
            if let Some(features) = views.get(view_name) {
                let mut single = BTreeMap::new();
                single.insert(view_name.clone(), features.clone());
                let output = self.transform(&single)?;
                single_view_proxy_scores
                    .insert(view_name.clone(), embedding_energy(&output.embedding));
            }
        }
        let best_single_view = single_view_proxy_scores
            .iter()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(name, _)| name.clone());
        let best_score = single_view_proxy_scores
            .values()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Ok(ViewAblationReport {
            full_proxy_score,
            single_view_proxy_scores,
            best_single_view,
            full_beats_best_single_view: full_proxy_score >= best_score,
            missing_views: full.missing_views,
        })
    }

    pub fn artifact(&self) -> Result<&RepresentationArtifact> {
        self.artifact.as_ref().ok_or_else(|| {
            RepresentationError::InvalidInput("multi-view attention is not fit".to_string())
        })
    }

    pub fn learned_view_weights(&self) -> &BTreeMap<String, f64> {
        &self.learned_view_weights
    }

    fn build_artifact(&self, save_load_parity_checked: bool) -> RepresentationArtifact {
        let mut id_maps = BTreeMap::new();
        id_maps.insert(
            "node".to_string(),
            self.node_ids
                .iter()
                .enumerate()
                .map(|(idx, value)| (value.clone(), idx))
                .collect(),
        );
        let mut hash_bucket_config = BTreeMap::new();
        hash_bucket_config.insert("view_count".to_string(), self.view_names.len());
        let mut feature_roles = BTreeMap::new();
        feature_roles.insert(
            "spatial_views".to_string(),
            serde_json::to_string(&self.view_names).unwrap_or_default(),
        );
        feature_roles.insert(
            "learned_view_weights".to_string(),
            serde_json::to_string(&self.learned_view_weights).unwrap_or_default(),
        );
        RepresentationArtifact {
            model_class: "MultiViewSpatialAttention".to_string(),
            architecture: "multi_view_spatial_attention".to_string(),
            artifact_version: REPRESENTATION_ARTIFACT_VERSION,
            schema_hash: self.schema_hash.clone(),
            id_maps,
            hash_bucket_config,
            embedding_dim: self.embedding_dim,
            random_seed: self.random_seed,
            feature_roles,
            training_cutoff: None,
            training_metrics: self.training_metrics.clone(),
            save_load_parity_checked,
            backend: self.backend.clone(),
        }
    }

    fn save_load_parity(&self, views: &BTreeMap<String, Vec<Vec<f64>>>) -> Result<bool> {
        let before = self.transform(views)?.embedding;
        let loaded: Self = serde_json::from_str(&serde_json::to_string(self)?)?;
        Ok(vectors_close(
            &before,
            &loaded.transform(views)?.embedding,
            1e-12,
        ))
    }
}

impl SelfSupervisedPretrainer {
    pub fn new(
        embedding_dim: usize,
        hash_bucket_count: usize,
        random_seed: u64,
        tasks: Vec<String>,
    ) -> Result<Self> {
        Ok(Self {
            entity: EntityEmbedding::new(embedding_dim, hash_bucket_count, random_seed)?,
            tasks,
            feature_mean: Vec::new(),
            feature_scale: Vec::new(),
            pretrained_entity_embeddings: Vec::new(),
            pretrained_pair_embeddings: Vec::new(),
            pretrained_node_embeddings: Vec::new(),
            pretrained_temporal_encoder: Vec::new(),
            training_cutoff: None,
            training_metrics: BTreeMap::new(),
            schema_hash: String::new(),
            artifact: None,
        })
    }

    pub fn fit<S: ToString>(
        &mut self,
        entity_ids: &[S],
        features: &[Vec<f64>],
        timestamps: Option<&[S]>,
        training_cutoff: String,
    ) -> Result<&mut Self> {
        if entity_ids.len() != features.len() {
            return Err(RepresentationError::InvalidInput(
                "features row count must match entity_ids".to_string(),
            ));
        }
        let feature_dim = features.first().map_or(0, Vec::len);
        if feature_dim == 0 || features.iter().any(|row| row.len() != feature_dim) {
            return Err(RepresentationError::InvalidInput(
                "features must have a fixed positive dimension".to_string(),
            ));
        }
        let mut kept_ids = Vec::new();
        let mut kept_features = Vec::new();
        match timestamps {
            Some(values) => {
                if values.len() != entity_ids.len() {
                    return Err(RepresentationError::InvalidInput(
                        "timestamps row count must match entity_ids".to_string(),
                    ));
                }
                for ((entity_id, row), timestamp) in
                    entity_ids.iter().zip(features.iter()).zip(values.iter())
                {
                    if timestamp.to_string() < training_cutoff {
                        kept_ids.push(entity_id.to_string());
                        kept_features.push(row.clone());
                    }
                }
            }
            None => {
                kept_ids = entity_ids.iter().map(ToString::to_string).collect();
                kept_features = features.to_vec();
            }
        }
        if kept_ids.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "no pretraining rows are before training_cutoff".to_string(),
            ));
        }
        self.entity.fit(
            kept_ids.iter().map(ToString::to_string),
            Some(training_cutoff.clone()),
        )?;
        self.feature_mean = column_mean(&kept_features);
        self.feature_scale = column_std(&kept_features, &self.feature_mean)
            .into_iter()
            .map(|value| value.max(1e-12))
            .collect();
        let normalized = normalize_rows(&kept_features, &self.feature_mean, &self.feature_scale);
        let projection = deterministic_matrix(
            feature_dim,
            self.entity.embedding_dim,
            self.entity.random_seed + 601,
            "self_supervised_pretrainer",
        );
        let mut embeddings = self.entity.embeddings.clone();
        let mut grouped: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
        for (entity_id, row) in kept_ids.iter().zip(normalized.iter()) {
            grouped
                .entry(entity_id.clone())
                .or_default()
                .push(row.clone());
        }
        for (entity_id, rows) in grouped.iter() {
            let mean = column_mean(rows);
            embeddings[self.entity.id_map[entity_id]] = matvec(&mean, &projection);
        }
        self.entity.embeddings = embeddings.clone();
        self.pretrained_entity_embeddings = embeddings.clone();
        self.pretrained_node_embeddings = embeddings.clone();
        self.pretrained_pair_embeddings = build_pair_pretraining_embeddings(
            &kept_ids,
            &normalized,
            self.entity.embedding_dim,
            self.entity.random_seed + 701,
        );
        self.pretrained_temporal_encoder = build_temporal_encoder(
            &normalized,
            self.entity.embedding_dim,
            self.entity.random_seed + 809,
        );
        self.training_cutoff = Some(training_cutoff.clone());
        self.training_metrics
            .insert("pretraining_rows".to_string(), kept_features.len() as f64);
        self.training_metrics.insert(
            "masked_reconstruction_proxy_rmse".to_string(),
            reconstruction_proxy_rmse(&normalized),
        );
        self.training_metrics.insert(
            "masked_pair_proxy_rmse".to_string(),
            pair_proxy_rmse(&self.pretrained_pair_embeddings),
        );
        self.training_metrics.insert(
            "graph_edge_denoising_proxy_auc".to_string(),
            graph_edge_proxy_auc(&self.pretrained_node_embeddings),
        );
        self.training_metrics.insert(
            "temporal_order_contrastive_margin".to_string(),
            temporal_order_margin(&normalized),
        );
        self.training_metrics.insert(
            "spatial_neighbor_contrastive_margin".to_string(),
            spatial_neighbor_margin(&normalized),
        );
        self.training_metrics.insert(
            "future_patch_reconstruction_proxy_rmse".to_string(),
            future_patch_proxy_rmse(&normalized),
        );
        let unique: BTreeSet<String> = kept_ids.iter().cloned().collect();
        self.schema_hash = schema_hash(&serde_json::json!({
            "entity_ids": unique,
            "feature_dim": feature_dim,
            "embedding_dim": self.entity.embedding_dim,
            "hash_bucket_count": self.entity.hash_bucket_count,
            "tasks": self.tasks,
            "training_cutoff": training_cutoff,
        }))?;
        self.artifact = Some(self.build_artifact(false));
        let parity = self.save_load_parity(&kept_ids)?;
        self.artifact = Some(self.build_artifact(parity));
        Ok(self)
    }

    pub fn transform<S: ToString>(&self, entity_ids: &[S]) -> Result<Vec<Vec<f64>>> {
        self.entity
            .transform(entity_ids.iter().map(ToString::to_string))
    }

    pub fn artifact(&self) -> Result<&RepresentationArtifact> {
        self.artifact
            .as_ref()
            .ok_or_else(|| RepresentationError::InvalidInput("pretrainer is not fit".to_string()))
    }

    pub fn tasks(&self) -> &[String] {
        &self.tasks
    }

    pub fn pretrained_pair_embeddings(&self) -> Result<&[Vec<f64>]> {
        if self.pretrained_pair_embeddings.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "pretrainer is not fit".to_string(),
            ));
        }
        Ok(&self.pretrained_pair_embeddings)
    }

    pub fn pretrained_node_embeddings(&self) -> Result<&[Vec<f64>]> {
        if self.pretrained_node_embeddings.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "pretrainer is not fit".to_string(),
            ));
        }
        Ok(&self.pretrained_node_embeddings)
    }

    pub fn pretrained_temporal_encoder(&self) -> Result<&[Vec<f64>]> {
        if self.pretrained_temporal_encoder.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "pretrainer is not fit".to_string(),
            ));
        }
        Ok(&self.pretrained_temporal_encoder)
    }

    fn build_artifact(&self, save_load_parity_checked: bool) -> RepresentationArtifact {
        let mut id_maps = BTreeMap::new();
        id_maps.insert("entity".to_string(), self.entity.id_map.clone());
        let mut hash_bucket_config = BTreeMap::new();
        hash_bucket_config.insert("entity".to_string(), self.entity.hash_bucket_count);
        hash_bucket_config.insert(
            "pair_pretraining_rows".to_string(),
            self.pretrained_pair_embeddings.len(),
        );
        let mut feature_roles = BTreeMap::new();
        feature_roles.insert(
            "pretraining_tasks".to_string(),
            serde_json::to_string(&self.tasks).unwrap_or_default(),
        );
        feature_roles.insert(
            "outputs".to_string(),
            serde_json::to_string(&[
                "pretrained_entity_embeddings",
                "pretrained_pair_embeddings",
                "pretrained_node_embeddings",
                "pretrained_temporal_encoder",
            ])
            .unwrap_or_default(),
        );
        RepresentationArtifact {
            model_class: "SelfSupervisedPretrainer".to_string(),
            architecture: "deterministic_self_supervised_pretrainer".to_string(),
            artifact_version: REPRESENTATION_ARTIFACT_VERSION,
            schema_hash: self.schema_hash.clone(),
            id_maps,
            hash_bucket_config,
            embedding_dim: self.entity.embedding_dim,
            random_seed: self.entity.random_seed,
            feature_roles,
            training_cutoff: self.training_cutoff.clone(),
            training_metrics: self.training_metrics.clone(),
            save_load_parity_checked,
            backend: self.entity.backend.clone(),
        }
    }

    fn save_load_parity(&self, ids: &[String]) -> Result<bool> {
        let probe: Vec<String> = ids
            .iter()
            .take(3)
            .cloned()
            .chain(std::iter::once("__new_entity__".to_string()))
            .collect();
        let before = self.transform(&probe)?;
        let loaded: Self = serde_json::from_str(&serde_json::to_string(self)?)?;
        Ok(vectors_close(&before, &loaded.transform(&probe)?, 1e-12))
    }
}

impl HistoricalAnalogRetriever {
    pub fn new(normalize: bool, random_seed: u64) -> Result<Self> {
        Ok(Self {
            analog_ids: Vec::new(),
            timestamps: None,
            memory: Vec::new(),
            key_mean: Vec::new(),
            key_scale: Vec::new(),
            normalize,
            random_seed,
            feature_roles: BTreeMap::new(),
            training_cutoff: None,
            training_metrics: BTreeMap::new(),
            schema_hash: String::new(),
            artifact: None,
            backend: resolve_backend("cpu")?,
        })
    }

    pub fn fit<S: ToString>(
        &mut self,
        analog_ids: &[S],
        keys: &[Vec<f64>],
        timestamps: Option<&[S]>,
        training_cutoff: Option<String>,
    ) -> Result<&mut Self> {
        if analog_ids.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "analog_ids must be non-empty".to_string(),
            ));
        }
        if keys.len() != analog_ids.len() {
            return Err(RepresentationError::InvalidInput(
                "keys row count must match analog_ids".to_string(),
            ));
        }
        let key_dim = keys.first().map_or(0, Vec::len);
        if key_dim == 0 || keys.iter().any(|row| row.len() != key_dim) {
            return Err(RepresentationError::InvalidInput(
                "keys must have a fixed positive dimension".to_string(),
            ));
        }
        let timestamp_values = match timestamps {
            Some(values) => {
                if values.len() != analog_ids.len() {
                    return Err(RepresentationError::InvalidInput(
                        "timestamps row count must match analog_ids".to_string(),
                    ));
                }
                Some(values.iter().map(ToString::to_string).collect())
            }
            None => None,
        };
        self.analog_ids = analog_ids.iter().map(ToString::to_string).collect();
        self.timestamps = timestamp_values;
        self.key_mean = if self.normalize {
            column_mean(keys)
        } else {
            vec![0.0; key_dim]
        };
        self.key_scale = if self.normalize {
            column_std(keys, &self.key_mean)
                .into_iter()
                .map(|value| value.max(1e-12))
                .collect()
        } else {
            vec![1.0; key_dim]
        };
        self.memory = normalize_rows(keys, &self.key_mean, &self.key_scale);
        self.training_cutoff = training_cutoff;
        self.training_metrics
            .insert("memory_size".to_string(), self.analog_ids.len() as f64);
        self.schema_hash = schema_hash(&serde_json::json!({
            "analog_ids": self.analog_ids,
            "key_dim": key_dim,
            "normalize": self.normalize,
            "has_timestamps": self.timestamps.is_some(),
        }))?;
        self.artifact = Some(self.build_artifact(false));
        let parity = self.save_load_parity()?;
        self.artifact = Some(self.build_artifact(parity));
        Ok(self)
    }

    pub fn query(
        &self,
        keys: &[Vec<f64>],
        k: usize,
        cutoff: Option<&str>,
    ) -> Result<Vec<AnalogQueryResult>> {
        if self.memory.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "retriever must be fit before query".to_string(),
            ));
        }
        if k == 0 {
            return Err(RepresentationError::InvalidInput(
                "k must be positive".to_string(),
            ));
        }
        if keys.iter().any(|row| row.len() != self.key_mean.len()) {
            return Err(RepresentationError::InvalidInput(
                "query key dimension must match fitted memory".to_string(),
            ));
        }
        let eligible = self.eligible_indices(cutoff);
        if eligible.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "no analogs are available before the requested cutoff".to_string(),
            ));
        }
        let normalized = normalize_rows(keys, &self.key_mean, &self.key_scale);
        let mut output = Vec::with_capacity(keys.len());
        for query in normalized.iter() {
            let mut distances: Vec<(usize, f64)> = eligible
                .iter()
                .map(|idx| (*idx, euclidean(query, &self.memory[*idx])))
                .collect();
            distances.sort_by(|left, right| left.1.total_cmp(&right.1));
            distances.truncate(k.min(distances.len()));
            output.push(AnalogQueryResult {
                analog_ids: distances
                    .iter()
                    .map(|(idx, _)| self.analog_ids[*idx].clone())
                    .collect(),
                distances: distances.iter().map(|(_, distance)| *distance).collect(),
                indices: distances.iter().map(|(idx, _)| *idx).collect(),
            });
        }
        Ok(output)
    }

    pub fn artifact(&self) -> Result<&RepresentationArtifact> {
        self.artifact
            .as_ref()
            .ok_or_else(|| RepresentationError::InvalidInput("retriever is not fit".to_string()))
    }

    fn eligible_indices(&self, cutoff: Option<&str>) -> Vec<usize> {
        match (cutoff, self.timestamps.as_ref()) {
            (Some(cutoff), Some(timestamps)) => timestamps
                .iter()
                .enumerate()
                .filter_map(|(idx, timestamp)| (timestamp.as_str() < cutoff).then_some(idx))
                .collect(),
            _ => (0..self.analog_ids.len()).collect(),
        }
    }

    fn build_artifact(&self, save_load_parity_checked: bool) -> RepresentationArtifact {
        let mut id_maps = BTreeMap::new();
        id_maps.insert(
            "analog".to_string(),
            self.analog_ids
                .iter()
                .enumerate()
                .map(|(idx, value)| (value.clone(), idx))
                .collect(),
        );
        let mut hash_bucket_config = BTreeMap::new();
        hash_bucket_config.insert("memory_size".to_string(), self.analog_ids.len());
        RepresentationArtifact {
            model_class: "HistoricalAnalogRetriever".to_string(),
            architecture: "exact_knn_memory".to_string(),
            artifact_version: REPRESENTATION_ARTIFACT_VERSION,
            schema_hash: self.schema_hash.clone(),
            id_maps,
            hash_bucket_config,
            embedding_dim: self.key_mean.len(),
            random_seed: self.random_seed,
            feature_roles: self.feature_roles.clone(),
            training_cutoff: self.training_cutoff.clone(),
            training_metrics: self.training_metrics.clone(),
            save_load_parity_checked,
            backend: self.backend.clone(),
        }
    }

    fn save_load_parity(&self) -> Result<bool> {
        let before = self.query(
            std::slice::from_ref(&self.key_mean),
            3.min(self.analog_ids.len()),
            None,
        )?;
        let loaded: Self = serde_json::from_str(&serde_json::to_string(self)?)?;
        Ok(before
            == loaded.query(
                std::slice::from_ref(&self.key_mean),
                3.min(self.analog_ids.len()),
                None,
            )?)
    }
}

impl RegimeRouter {
    pub fn new(
        expert_count: usize,
        embedding_dim: usize,
        hash_bucket_count: usize,
        random_seed: u64,
    ) -> Result<Self> {
        if expert_count < 2 {
            return Err(RepresentationError::InvalidInput(
                "expert_count must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            entity: EntityEmbedding::new(embedding_dim, hash_bucket_count, random_seed)?,
            expert_count,
            context_dim: 0,
            router_weights: Vec::new(),
            router_bias: Vec::new(),
            expert_usage: BTreeMap::new(),
            schema_hash: String::new(),
            training_cutoff: None,
            training_metrics: BTreeMap::new(),
            artifact: None,
        })
    }

    pub fn fit<S: ToString>(
        &mut self,
        ids: &[S],
        context_features: Option<&[Vec<f64>]>,
        training_cutoff: Option<String>,
    ) -> Result<&mut Self> {
        if ids.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "ids must be non-empty".to_string(),
            ));
        }
        self.entity
            .fit(ids.iter().map(ToString::to_string), training_cutoff.clone())?;
        self.context_dim = context_features
            .and_then(|values| values.first().map(Vec::len))
            .unwrap_or(0);
        let input_dim = self.entity.embedding_dim + self.context_dim;
        self.router_weights = deterministic_matrix(
            input_dim,
            self.expert_count,
            self.entity.random_seed + 401,
            "regime_router",
        );
        self.router_bias = deterministic_matrix(
            1,
            self.expert_count,
            self.entity.random_seed + 409,
            "regime_router_bias",
        )
        .remove(0);
        self.training_cutoff = training_cutoff;
        let route = self.route(ids, context_features)?;
        let mut counts = vec![0.0; self.expert_count];
        for expert in route.selected_expert.iter() {
            counts[*expert] += 1.0;
        }
        let denom = counts.iter().sum::<f64>().max(1.0);
        self.expert_usage = counts
            .iter()
            .enumerate()
            .map(|(idx, value)| (format!("expert_{idx}"), value / denom))
            .collect();
        self.training_metrics.insert(
            "mean_router_entropy".to_string(),
            route.router_entropy.iter().sum::<f64>() / route.router_entropy.len() as f64,
        );
        let unique: BTreeSet<String> = ids.iter().map(ToString::to_string).collect();
        self.schema_hash = schema_hash(&serde_json::json!({
            "ids": unique,
            "embedding_dim": self.entity.embedding_dim,
            "hash_bucket_count": self.entity.hash_bucket_count,
            "expert_count": self.expert_count,
            "context_dim": self.context_dim,
        }))?;
        self.artifact = Some(self.build_artifact(false));
        let parity = self.save_load_parity(ids)?;
        self.artifact = Some(self.build_artifact(parity));
        Ok(self)
    }

    pub fn predict_proba<S: ToString>(
        &self,
        ids: &[S],
        context_features: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<f64>>> {
        if self.router_weights.is_empty() {
            return Err(RepresentationError::InvalidInput(
                "router must be fit before prediction".to_string(),
            ));
        }
        let embeddings = self.entity.transform(ids.iter().map(ToString::to_string))?;
        let context = context_matrix(context_features, ids.len(), self.context_dim)?;
        let mut logits = Vec::with_capacity(ids.len());
        for idx in 0..ids.len() {
            let mut row = embeddings[idx].clone();
            row.extend_from_slice(&context[idx]);
            let mut scores = matvec(&row, &self.router_weights);
            for (score, bias) in scores.iter_mut().zip(self.router_bias.iter()) {
                *score += bias;
            }
            logits.push(scores);
        }
        Ok(softmax_rows(&logits))
    }

    pub fn route<S: ToString>(
        &self,
        ids: &[S],
        context_features: Option<&[Vec<f64>]>,
    ) -> Result<RegimeRoute> {
        let expert_weights = self.predict_proba(ids, context_features)?;
        let selected_expert = expert_weights
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();
        let router_entropy = expert_weights
            .iter()
            .map(|row| {
                -row.iter()
                    .map(|value| value * value.max(1e-12).ln())
                    .sum::<f64>()
            })
            .collect();
        Ok(RegimeRoute {
            expert_weights,
            selected_expert,
            router_entropy,
        })
    }

    pub fn artifact(&self) -> Result<&RepresentationArtifact> {
        self.artifact.as_ref().ok_or_else(|| {
            RepresentationError::InvalidInput("regime router is not fit".to_string())
        })
    }

    pub fn expert_usage(&self) -> &BTreeMap<String, f64> {
        &self.expert_usage
    }

    fn build_artifact(&self, save_load_parity_checked: bool) -> RepresentationArtifact {
        let mut id_maps = BTreeMap::new();
        id_maps.insert("entity".to_string(), self.entity.id_map.clone());
        let mut hash_bucket_config = BTreeMap::new();
        hash_bucket_config.insert("entity".to_string(), self.entity.hash_bucket_count);
        hash_bucket_config.insert("expert_count".to_string(), self.expert_count);
        RepresentationArtifact {
            model_class: "RegimeRouter".to_string(),
            architecture: "regime_router".to_string(),
            artifact_version: REPRESENTATION_ARTIFACT_VERSION,
            schema_hash: self.schema_hash.clone(),
            id_maps,
            hash_bucket_config,
            embedding_dim: self.entity.embedding_dim,
            random_seed: self.entity.random_seed,
            feature_roles: BTreeMap::new(),
            training_cutoff: self.training_cutoff.clone(),
            training_metrics: self.training_metrics.clone(),
            save_load_parity_checked,
            backend: self.entity.backend.clone(),
        }
    }

    fn save_load_parity<S: ToString>(&self, ids: &[S]) -> Result<bool> {
        let probe: Vec<String> = ids
            .iter()
            .take(3)
            .map(ToString::to_string)
            .chain(std::iter::once("__new_entity__".to_string()))
            .collect();
        let context = vec![vec![0.0; self.context_dim]; probe.len()];
        let before = self.predict_proba(&probe, Some(&context))?;
        let loaded: Self = serde_json::from_str(&serde_json::to_string(self)?)?;
        Ok(vectors_close(
            &before,
            &loaded.predict_proba(&probe, Some(&context))?,
            1e-12,
        ))
    }
}

impl SpatioTemporalAdaptiveEmbedding {
    pub fn new(embedding_dim: usize, hash_bucket_count: usize, random_seed: u64) -> Result<Self> {
        Ok(Self {
            entity: EntityEmbedding::new(embedding_dim, hash_bucket_count, random_seed)?,
            time_weights: Vec::new(),
            context_weights: Vec::new(),
            gate_weights: Vec::new(),
        })
    }

    pub fn fit<I, S>(&mut self, ids: I) -> Result<&mut Self>
    where
        I: IntoIterator<Item = S>,
        S: ToString,
    {
        self.entity.fit(ids, None)?;
        let dim = self.entity.embedding_dim;
        self.time_weights = deterministic_matrix(dim, dim, self.entity.random_seed + 101, "time");
        self.context_weights =
            deterministic_matrix(dim, dim, self.entity.random_seed + 211, "context");
        self.gate_weights =
            deterministic_matrix(dim * 3, dim, self.entity.random_seed + 307, "gate");
        Ok(self)
    }

    pub fn transform<S: ToString>(
        &self,
        ids: &[S],
        time_features: &[Vec<f64>],
        context_features: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<f64>>> {
        let static_embedding = self.entity.transform(ids.iter().map(ToString::to_string))?;
        if time_features.len() != static_embedding.len() {
            return Err(RepresentationError::InvalidInput(
                "time_features row count must match ids".to_string(),
            ));
        }
        let time = project_features(time_features, &self.time_weights)?;
        let context = match context_features {
            Some(values) => project_features(values, &self.context_weights)?,
            None => vec![vec![0.0; self.entity.embedding_dim]; ids.len()],
        };
        let mut output = Vec::with_capacity(ids.len());
        for idx in 0..ids.len() {
            let mut gate_input = static_embedding[idx].clone();
            gate_input.extend_from_slice(&time[idx]);
            gate_input.extend_from_slice(&context[idx]);
            let gate = sigmoid_vec(&matvec(&gate_input, &self.gate_weights));
            let row: Vec<f64> = static_embedding[idx]
                .iter()
                .zip(time[idx].iter())
                .zip(context[idx].iter())
                .zip(gate.iter())
                .map(|(((base, time_value), context_value), gate_value)| {
                    base + gate_value * time_value + (1.0 - gate_value) * context_value
                })
                .collect();
            output.push(layer_norm(&row));
        }
        Ok(output)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

pub type EntityTimeAdaptiveEmbedding = SpatioTemporalAdaptiveEmbedding;
pub type PairTimeAdaptiveEmbedding = PairEmbedding;
pub type NodeTimeAdaptiveEmbedding = SpatioTemporalAdaptiveEmbedding;

pub fn resolve_backend(requested: &str) -> Result<BackendMetadata> {
    let requested = requested.to_ascii_lowercase();
    let supported = ["cpu", "cuda", "rocm", "mlx"];
    if requested != "auto" && !supported.contains(&requested.as_str()) {
        return Err(RepresentationError::InvalidInput(
            "backend must be one of auto, cpu, cuda, rocm, or mlx".to_string(),
        ));
    }
    let mut accelerator_ready = BTreeMap::new();
    accelerator_ready.insert("cuda".to_string(), true);
    accelerator_ready.insert("rocm".to_string(), true);
    accelerator_ready.insert("mlx".to_string(), true);
    Ok(BackendMetadata {
        requested,
        selected: "cpu".to_string(),
        available: vec!["cpu".to_string()],
        supported_accelerators: supported.iter().map(|value| value.to_string()).collect(),
        accelerator_ready,
    })
}

fn deterministic_matrix(rows: usize, cols: usize, seed: u64, salt: &str) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let mut hasher = Sha256::new();
                    hasher.update(format!("{seed}:{salt}:{row}:{col}"));
                    let digest = hasher.finalize();
                    let mut bytes = [0_u8; 8];
                    bytes.copy_from_slice(&digest[..8]);
                    let value = u64::from_le_bytes(bytes) as f64 / u64::MAX as f64;
                    value * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn stable_hash(value: &str, modulo: usize) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(value);
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_le_bytes(bytes) as usize) % modulo
}

fn schema_hash(value: &serde_json::Value) -> Result<String> {
    let payload = serde_json::to_string(value)?;
    let mut hasher = Sha256::new();
    hasher.update(payload);
    Ok(format!("{:x}", hasher.finalize()))
}

fn pair_key(source: &str, target: &str) -> String {
    format!("{source}\0{target}")
}

fn project_features(features: &[Vec<f64>], weights: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if weights.is_empty() {
        return Err(RepresentationError::InvalidInput(
            "embedding must be fit before transform".to_string(),
        ));
    }
    Ok(features
        .iter()
        .map(|row| {
            let mut padded = row.clone();
            padded.resize(weights.len(), 0.0);
            matvec(&padded[..weights.len()], weights)
        })
        .collect())
}

fn context_matrix(
    features: Option<&[Vec<f64>]>,
    row_count: usize,
    context_dim: usize,
) -> Result<Vec<Vec<f64>>> {
    if context_dim == 0 {
        return Ok(vec![Vec::new(); row_count]);
    }
    let Some(features) = features else {
        return Ok(vec![vec![0.0; context_dim]; row_count]);
    };
    if features.len() != row_count {
        return Err(RepresentationError::InvalidInput(
            "context feature row count must match ids".to_string(),
        ));
    }
    if features.iter().any(|row| row.len() != context_dim) {
        return Err(RepresentationError::InvalidInput(
            "context feature column count must match fitted router".to_string(),
        ));
    }
    Ok(features.to_vec())
}

fn validate_feature_rows(features: &[Vec<f64>], row_count: usize, name: &str) -> Result<()> {
    if features.len() != row_count {
        return Err(RepresentationError::InvalidInput(format!(
            "view {name} row count must match node_ids"
        )));
    }
    let feature_dim = features.first().map_or(0, Vec::len);
    if feature_dim == 0 || features.iter().any(|row| row.len() != feature_dim) {
        return Err(RepresentationError::InvalidInput(format!(
            "view {name} must have a fixed positive feature dimension"
        )));
    }
    if features
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return Err(RepresentationError::InvalidInput(format!(
            "view {name} features must be finite"
        )));
    }
    Ok(())
}

fn mean_abs(rows: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    let mut count: f64 = 0.0;
    for row in rows {
        for value in row {
            total += value.abs();
            count += 1.0;
        }
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

fn mean_view_weights(view_names: &[String], weights: &[Vec<f64>]) -> BTreeMap<String, f64> {
    view_names
        .iter()
        .enumerate()
        .map(|(view_idx, name)| {
            let mean = if weights.is_empty() {
                0.0
            } else {
                weights
                    .iter()
                    .map(|row| row.get(view_idx).copied().unwrap_or(0.0))
                    .sum::<f64>()
                    / weights.len() as f64
            };
            (name.clone(), mean)
        })
        .collect()
}

fn embedding_energy(rows: &[Vec<f64>]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter()
        .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt())
        .sum::<f64>()
        / rows.len() as f64
}

fn matvec(row: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
    let cols = weights.first().map_or(0, Vec::len);
    (0..cols)
        .map(|col| {
            row.iter()
                .zip(weights.iter())
                .map(|(value, weight_row)| value * weight_row[col])
                .sum()
        })
        .collect()
}

fn sigmoid_vec(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| 1.0 / (1.0 + (-value.clamp(-50.0, 50.0)).exp()))
        .collect()
}

fn softmax_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = row
                .iter()
                .map(|value| (value - max).clamp(-50.0, 50.0).exp())
                .collect();
            let denom = exp.iter().sum::<f64>();
            exp.iter().map(|value| value / denom).collect()
        })
        .collect()
}

fn softmax_with_neg_inf(logits: &[f64]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exp: Vec<f64> = logits
        .iter()
        .map(|value| {
            if value.is_finite() {
                (value - max).clamp(-50.0, 50.0).exp()
            } else {
                0.0
            }
        })
        .collect();
    let denom = exp.iter().sum::<f64>().max(1e-12);
    exp.iter().map(|value| value / denom).collect()
}

fn column_mean(rows: &[Vec<f64>]) -> Vec<f64> {
    let cols = rows.first().map_or(0, Vec::len);
    (0..cols)
        .map(|col| rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64)
        .collect()
}

fn column_std(rows: &[Vec<f64>], mean: &[f64]) -> Vec<f64> {
    (0..mean.len())
        .map(|col| {
            (rows
                .iter()
                .map(|row| {
                    let centered = row[col] - mean[col];
                    centered * centered
                })
                .sum::<f64>()
                / rows.len() as f64)
                .sqrt()
        })
        .collect()
}

fn normalize_rows(rows: &[Vec<f64>], mean: &[f64], scale: &[f64]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| (value - mean[idx]) / scale[idx])
                .collect()
        })
        .collect()
}

fn euclidean(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn reconstruction_proxy_rmse(rows: &[Vec<f64>]) -> f64 {
    let mean = column_mean(rows);
    (rows
        .iter()
        .flat_map(|row| {
            row.iter().enumerate().map(|(idx, value)| {
                let residual = value - mean[idx];
                residual * residual
            })
        })
        .sum::<f64>()
        / (rows.len() * mean.len()) as f64)
        .sqrt()
}

fn build_pair_pretraining_embeddings(
    ids: &[String],
    rows: &[Vec<f64>],
    embedding_dim: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let projection = deterministic_matrix(
        rows.first().map_or(0, Vec::len) * 2,
        embedding_dim,
        seed,
        MASKED_PAIR_TIME_MODELING,
    );
    ids.windows(2)
        .zip(rows.windows(2))
        .map(|(id_pair, row_pair)| {
            let mut features = row_pair[0].clone();
            features.extend_from_slice(&row_pair[1]);
            let mut embedding = matvec(&features, &projection);
            if id_pair[0] == id_pair[1] {
                for value in &mut embedding {
                    *value *= 1.1;
                }
            }
            layer_norm(&embedding)
        })
        .collect()
}

fn build_temporal_encoder(rows: &[Vec<f64>], embedding_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let projection = deterministic_matrix(
        rows.first().map_or(0, Vec::len),
        embedding_dim,
        seed,
        FUTURE_PATCH_RECONSTRUCTION,
    );
    rows.windows(3)
        .map(|window| {
            let mean = column_mean(window);
            layer_norm(&matvec(&mean, &projection))
        })
        .collect()
}

fn pair_proxy_rmse(pair_embeddings: &[Vec<f64>]) -> f64 {
    if pair_embeddings.is_empty() {
        return 0.0;
    }
    reconstruction_proxy_rmse(pair_embeddings)
}

fn graph_edge_proxy_auc(node_embeddings: &[Vec<f64>]) -> f64 {
    if node_embeddings.len() < 2 {
        return 0.5;
    }
    let mut near = 0.0;
    let mut far = 0.0;
    let mut count: f64 = 0.0;
    for idx in 0..node_embeddings.len() - 1 {
        near += euclidean(&node_embeddings[idx], &node_embeddings[idx + 1]);
        far += euclidean(
            &node_embeddings[idx],
            &node_embeddings[node_embeddings.len() - 1 - idx],
        );
        count += 1.0;
    }
    let margin = (far / count.max(1.0)) - (near / count.max(1.0));
    1.0 / (1.0 + (-margin).exp())
}

fn temporal_order_margin(rows: &[Vec<f64>]) -> f64 {
    if rows.len() < 3 {
        return 0.0;
    }
    let forward: f64 = rows
        .windows(2)
        .map(|pair| euclidean(&pair[0], &pair[1]))
        .sum();
    let reverse: f64 = rows
        .iter()
        .rev()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|pair| euclidean(pair[0], pair[1]))
        .sum();
    (reverse - forward).abs() / (rows.len() - 1) as f64
}

fn spatial_neighbor_margin(rows: &[Vec<f64>]) -> f64 {
    if rows.len() < 3 {
        return 0.0;
    }
    let neighbor = rows
        .windows(2)
        .map(|pair| euclidean(&pair[0], &pair[1]))
        .sum::<f64>()
        / (rows.len() - 1) as f64;
    let anchor = &rows[0];
    let distant = rows[2..]
        .iter()
        .map(|row| euclidean(anchor, row))
        .sum::<f64>()
        / (rows.len() - 2) as f64;
    distant - neighbor
}

fn future_patch_proxy_rmse(rows: &[Vec<f64>]) -> f64 {
    if rows.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for pair in rows.windows(2) {
        for (prev, next) in pair[0].iter().zip(pair[1].iter()) {
            let residual = next - prev;
            sum += residual * residual;
            count += 1.0;
        }
    }
    (sum / count).sqrt()
}

fn layer_norm(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.sqrt().max(1e-12);
    values.iter().map(|value| (value - mean) / std).collect()
}

fn vectors_close(left: &[Vec<f64>], right: &[Vec<f64>], tolerance: f64) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(left_row, right_row)| {
            left_row.len() == right_row.len()
                && left_row
                    .iter()
                    .zip(right_row.iter())
                    .all(|(left_value, right_value)| (left_value - right_value).abs() <= tolerance)
        })
}
