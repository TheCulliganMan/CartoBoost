#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cartoboost_neural::CudaTensorArena;
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
    let requested = if requested == "hip" {
        "rocm".to_string()
    } else {
        requested
    };
    if !matches!(
        requested.as_str(),
        "auto" | "cpu" | "cuda" | "rocm" | "metal" | "webgpu"
    ) {
        return Err(GeoStError::InvalidBackend(format!(
            "unknown backend {requested:?}; expected auto, cpu, cuda, hip/rocm, metal, or webgpu"
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

// Graph-temporal implementation families share the crate namespace.
include!("graph/dcrnn.rs");
include!("graph/transformers.rs");
include!("graph/transformer_parameters.rs");
include!("graph/transformer_ops.rs");
include!("graph/autodiff.rs");
include!("graph/graph_models.rs");
include!("graph/metrics.rs");
include!("graph/tests.rs");
