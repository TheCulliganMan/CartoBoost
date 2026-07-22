use crate::error::{NeuralError, Result};
use crate::features::NeuralFeatureBlock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;
use std::sync::Arc;

const ARTIFACT_TYPE: &str = "cartoboost.neural.embedding_table";
pub const EMBEDDING_TABLE_ARTIFACT_VERSION: u32 = 1;
pub type EmbeddingChecksum = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingIdType {
    U64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "strategy", content = "params", rename_all = "snake_case")]
pub enum ArtifactFallbackKind {
    ZeroVector,
    #[default]
    GlobalMeanVector,
    ParentCell {
        parent_resolution: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    ZeroVector,
    GlobalMeanVector,
    ParentCell { parent_resolution: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingTableMetadata {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub dim: usize,
    pub id_type: EmbeddingIdType,
    pub row_count: usize,
    pub checksum: EmbeddingChecksum,
    #[serde(default)]
    pub fallback: ArtifactFallbackKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRow {
    pub id: u64,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTableArtifact {
    pub metadata: EmbeddingTableMetadata,
    pub rows: Vec<EmbeddingRow>,
}

#[derive(Clone)]
pub struct EmbeddingTable {
    metadata: EmbeddingTableMetadata,
    rows: Vec<EmbeddingRow>,
    row_index: HashMap<u64, usize>,
    fallback: FallbackStrategy,
    parent_lookup: Option<Arc<dyn Fn(u64) -> Option<u64> + Send + Sync>>,
    global_mean: Vec<f32>,
    zero_vector: Vec<f32>,
}

impl EmbeddingTable {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact: EmbeddingTableArtifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: EmbeddingTableArtifact) -> Result<Self> {
        validate_artifact(&artifact)?;

        let metadata = artifact.metadata;
        let mut index = HashMap::with_capacity(metadata.row_count);
        let mut sorted_rows = artifact.rows;

        for row in &sorted_rows {
            if row.values.len() != metadata.dim {
                return Err(NeuralError::InvalidArgument(format!(
                    "row {} has {} values but embedding dimension is {}",
                    row.id,
                    row.values.len(),
                    metadata.dim
                )));
            }

            if index.insert(row.id, usize::MAX).is_some() {
                return Err(NeuralError::DuplicateId(row.id));
            }
        }

        sorted_rows.sort_by_key(|row| row.id);
        index.clear();

        for (position, row) in sorted_rows.iter().enumerate() {
            index.insert(row.id, position);
        }

        let expected_checksum = compute_checksum(&metadata, &sorted_rows);
        if metadata.checksum != expected_checksum {
            return Err(NeuralError::ChecksumMismatch {
                expected: metadata.checksum,
                actual: expected_checksum,
            });
        }

        let dim = metadata.dim;
        let zero_vector = vec![0.0_f32; dim];
        let global_mean = compute_global_mean(dim, &sorted_rows);
        let fallback = match metadata.fallback {
            ArtifactFallbackKind::ZeroVector => FallbackStrategy::ZeroVector,
            ArtifactFallbackKind::GlobalMeanVector => FallbackStrategy::GlobalMeanVector,
            ArtifactFallbackKind::ParentCell { parent_resolution } => {
                FallbackStrategy::ParentCell { parent_resolution }
            }
        };

        Ok(Self {
            metadata,
            rows: sorted_rows,
            row_index: index,
            fallback,
            parent_lookup: None,
            global_mean,
            zero_vector,
        })
    }

    pub fn dim(&self) -> usize {
        self.metadata.dim
    }

    pub fn row_count(&self) -> usize {
        self.metadata.row_count
    }

    pub fn rows(&self) -> &[EmbeddingRow] {
        &self.rows
    }

    pub fn id_type(&self) -> EmbeddingIdType {
        self.metadata.id_type
    }

    pub fn artifact_metadata(&self) -> &EmbeddingTableMetadata {
        &self.metadata
    }

    pub fn fallback_strategy(&self) -> FallbackStrategy {
        self.fallback
    }

    pub fn with_parent_id_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(u64) -> Option<u64> + Send + Sync + 'static,
    {
        self.parent_lookup = Some(Arc::new(callback));
        self
    }

    pub fn lookup(&self, id: u64) -> Option<&[f32]> {
        let row_index = self.row_index.get(&id)?;
        Some(&self.rows[*row_index].values)
    }

    pub fn encode_ids(&self, ids: &[u64], name: impl Into<String>) -> Result<NeuralFeatureBlock> {
        let values = ids
            .par_iter()
            .flat_map_iter(|&id| {
                self.lookup(id)
                    .or_else(|| self.fallback_embedding(id))
                    .unwrap_or(&[])
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        NeuralFeatureBlock::new(name, self.dim(), values)
    }

    fn fallback_embedding(&self, id: u64) -> Option<&[f32]> {
        match self.fallback {
            FallbackStrategy::ZeroVector => Some(&self.zero_vector),
            FallbackStrategy::GlobalMeanVector => Some(&self.global_mean),
            FallbackStrategy::ParentCell { .. } => self
                .parent_lookup
                .as_ref()
                .and_then(|callback| callback(id))
                .and_then(|parent_id| self.lookup(parent_id))
                .or_else(|| {
                    if self.global_mean.is_empty() {
                        Some(&self.zero_vector)
                    } else {
                        Some(&self.global_mean)
                    }
                }),
        }
    }
}

fn validate_artifact(artifact: &EmbeddingTableArtifact) -> Result<()> {
    if artifact.metadata.artifact_type != ARTIFACT_TYPE {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact type {}",
            artifact.metadata.artifact_type
        )));
    }

    if artifact.metadata.artifact_version != EMBEDDING_TABLE_ARTIFACT_VERSION {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact version {}",
            artifact.metadata.artifact_version
        )));
    }

    if artifact.metadata.dim == 0 {
        return Err(NeuralError::InvalidArgument(
            "embedding dimension must be greater than zero".to_string(),
        ));
    }

    if artifact.metadata.row_count != artifact.rows.len() {
        return Err(NeuralError::InvalidRowCount {
            expected: artifact.metadata.row_count,
            actual: artifact.rows.len(),
        });
    }

    Ok(())
}

pub fn build_embedding_table_artifact(
    dim: usize,
    rows: Vec<EmbeddingRow>,
    fallback: ArtifactFallbackKind,
) -> Result<EmbeddingTableArtifact> {
    let metadata = EmbeddingTableMetadata {
        artifact_type: ARTIFACT_TYPE.to_string(),
        artifact_version: EMBEDDING_TABLE_ARTIFACT_VERSION,
        dim,
        id_type: EmbeddingIdType::U64,
        row_count: rows.len(),
        checksum: String::new(),
        fallback,
    };
    let checksum = compute_checksum(&metadata, &rows);
    let mut metadata = metadata;
    metadata.checksum = checksum;

    Ok(EmbeddingTableArtifact { metadata, rows })
}

pub fn write_embedding_table_artifact(
    path: impl AsRef<Path>,
    artifact: &EmbeddingTableArtifact,
) -> Result<()> {
    let text = serde_json::to_string_pretty(artifact)?;
    fs::write(path, text)?;
    Ok(())
}

fn compute_global_mean(dim: usize, rows: &[EmbeddingRow]) -> Vec<f32> {
    if rows.is_empty() {
        return vec![0.0; dim];
    }

    let mut totals = vec![0.0_f64; dim];
    for row in rows {
        for (index, value) in row.values.iter().enumerate() {
            totals[index] += f64::from(*value);
        }
    }

    let row_count = rows.len() as f64;
    totals
        .into_iter()
        .map(|value| (value / row_count) as f32)
        .collect()
}

fn compute_checksum(metadata: &EmbeddingTableMetadata, rows: &[EmbeddingRow]) -> String {
    let mut rows = rows.to_vec();
    rows.sort_by_key(|row| row.id);
    let mut hasher = Sha256::new();

    hasher.update(metadata.artifact_type.as_bytes());
    hasher.update(metadata.artifact_version.to_le_bytes());
    hasher.update(u8::from(metadata.id_type).to_le_bytes());
    hasher.update(metadata.dim.to_le_bytes());
    hasher.update(metadata.row_count.to_le_bytes());
    match metadata.fallback {
        ArtifactFallbackKind::ZeroVector => hasher.update([0u8]),
        ArtifactFallbackKind::GlobalMeanVector => hasher.update([1u8]),
        ArtifactFallbackKind::ParentCell { parent_resolution } => {
            hasher.update([2u8]);
            hasher.update([parent_resolution]);
        }
    }

    for row in rows {
        hasher.update(row.id.to_le_bytes());
        for value in row.values {
            hasher.update(value.to_le_bytes());
        }
    }

    let digest = hasher.finalize();
    let mut checksum = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut checksum, "{byte:02x}");
    }

    checksum
}

impl From<EmbeddingIdType> for u8 {
    fn from(value: EmbeddingIdType) -> Self {
        match value {
            EmbeddingIdType::U64 => 0,
        }
    }
}
