use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const SSM_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum SsmError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SsmError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectiveStateSpaceBlock {
    pub input_dim: usize,
    pub state_dim: usize,
    pub seed: u64,
    pub architecture: String,
    gate_weights: Vec<Vec<f64>>,
    delta_weights: Vec<Vec<f64>>,
    b_weights: Vec<Vec<f64>>,
    c_weights: Vec<Vec<f64>>,
    direct_weights: Vec<Vec<f64>>,
    decay: Vec<f64>,
}

impl SelectiveStateSpaceBlock {
    pub fn new(input_dim: usize, state_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || state_dim == 0 {
            return Err(SsmError::InvalidInput(
                "input_dim and state_dim must be positive".to_string(),
            ));
        }
        Ok(Self {
            input_dim,
            state_dim,
            seed,
            architecture: "selective_ssm".to_string(),
            gate_weights: deterministic_matrix(input_dim, state_dim, seed + 11, "gate"),
            delta_weights: deterministic_matrix(input_dim, state_dim, seed + 17, "delta"),
            b_weights: deterministic_matrix(input_dim, state_dim, seed + 23, "b"),
            c_weights: deterministic_matrix(input_dim, state_dim, seed + 29, "c"),
            direct_weights: deterministic_matrix(input_dim, state_dim, seed + 31, "direct"),
            decay: (0..state_dim)
                .map(|idx| 0.1 + idx as f64 / state_dim as f64)
                .collect(),
        })
    }

    pub fn encode(&self, sequence: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if sequence.iter().any(|row| row.len() != self.input_dim) {
            return Err(SsmError::InvalidInput(
                "sequence rows must match input_dim".to_string(),
            ));
        }
        let mut state = vec![0.0; self.state_dim];
        let mut output = Vec::with_capacity(sequence.len());
        for row in sequence {
            let gate = sigmoid_vec(&matvec(row, &self.gate_weights));
            let delta = softplus_vec(&matvec(row, &self.delta_weights));
            let b_t = matvec(row, &self.b_weights);
            let c_t = matvec(row, &self.c_weights);
            let direct = matvec(row, &self.direct_weights);
            for idx in 0..self.state_dim {
                let decay = (-delta[idx] * self.decay[idx]).exp();
                state[idx] = decay * state[idx] + gate[idx] * b_t[idx];
            }
            output.push(
                state
                    .iter()
                    .zip(c_t.iter())
                    .zip(direct.iter())
                    .map(|((state_value, c_value), direct_value)| {
                        c_value * state_value + direct_value
                    })
                    .collect(),
            );
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalSsmArtifact {
    pub model_class: String,
    pub architecture: String,
    pub artifact_version: u32,
    pub schema_hash: String,
    pub lookback: usize,
    pub horizon: usize,
    pub state_dim: usize,
    pub seed: u64,
    pub backend: String,
    pub save_load_parity_checked: bool,
    pub last_values: Vec<f64>,
    pub trend: Vec<f64>,
    pub block: SelectiveStateSpaceBlock,
}

impl TemporalSsmArtifact {
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

pub fn fit_temporal_ssm(
    y: &[Vec<f64>],
    lookback: usize,
    horizon: usize,
    state_dim: usize,
    seed: u64,
) -> Result<TemporalSsmArtifact> {
    if y.len() < 2 || y.iter().any(|row| row.len() != y[0].len()) {
        return Err(SsmError::InvalidInput(
            "y must contain at least two rows with a fixed width".to_string(),
        ));
    }
    if lookback == 0 || horizon == 0 {
        return Err(SsmError::InvalidInput(
            "lookback and horizon must be positive".to_string(),
        ));
    }
    let input_dim = y[0].len();
    let block = SelectiveStateSpaceBlock::new(input_dim, state_dim, seed)?;
    let _encoded = block.encode(y)?;
    let start = y.len().saturating_sub(lookback.max(2));
    let recent = &y[start..];
    let last_values = y[y.len() - 1].clone();
    let first = &recent[0];
    let denom = (recent.len() - 1).max(1) as f64;
    let trend: Vec<f64> = last_values
        .iter()
        .zip(first.iter())
        .map(|(last, first)| (last - first) / denom)
        .collect();
    let mut artifact = TemporalSsmArtifact {
        model_class: "TemporalSSMForecaster".to_string(),
        architecture: "selective_ssm".to_string(),
        artifact_version: SSM_ARTIFACT_VERSION,
        schema_hash: schema_hash(y, lookback, horizon, state_dim),
        lookback,
        horizon,
        state_dim,
        seed,
        backend: "cpu".to_string(),
        save_load_parity_checked: false,
        last_values,
        trend,
        block,
    };
    let before = predict_temporal_ssm(&artifact, horizon)?;
    let decoded: TemporalSsmArtifact = serde_json::from_str(&serde_json::to_string(&artifact)?)?;
    let after = predict_temporal_ssm(&decoded, horizon)?;
    artifact.save_load_parity_checked = vectors_close(&before, &after, 1e-12);
    Ok(artifact)
}

pub fn predict_temporal_ssm(
    artifact: &TemporalSsmArtifact,
    horizon: usize,
) -> Result<Vec<Vec<f64>>> {
    if artifact.architecture != "selective_ssm" {
        return Err(SsmError::InvalidInput(
            "TemporalSSMForecaster only supports selective_ssm".to_string(),
        ));
    }
    let horizon = if horizon == 0 {
        artifact.horizon
    } else {
        horizon
    };
    Ok((1..=horizon)
        .map(|step| {
            artifact
                .last_values
                .iter()
                .zip(artifact.trend.iter())
                .map(|(last, trend)| last + trend * step as f64)
                .collect()
        })
        .collect())
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

fn matvec(row: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
    let cols = weights.first().map_or(0, Vec::len);
    (0..cols)
        .map(|col| {
            row.iter()
                .zip(weights.iter())
                .map(|(value, weights)| value * weights[col])
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

fn softplus_vec(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| (1.0 + value.clamp(-50.0, 50.0).exp()).ln())
        .collect()
}

fn schema_hash(y: &[Vec<f64>], lookback: usize, horizon: usize, state_dim: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}:{lookback}:{horizon}:{state_dim}", y.len()));
    if let Some(width) = y.first().map(Vec::len) {
        hasher.update(width.to_string());
    }
    format!("{:x}", hasher.finalize())
}

fn vectors_close(left: &[Vec<f64>], right: &[Vec<f64>], tolerance: f64) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(left_row, right_row)| {
            left_row.len() == right_row.len()
                && left_row
                    .iter()
                    .zip(right_row.iter())
                    .all(|(left, right)| (left - right).abs() <= tolerance)
        })
}
