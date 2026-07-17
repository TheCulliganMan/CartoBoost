use crate::tree::Model;
use crate::{CartoBoostError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[cfg(test)]
mod serialize_tests;

pub const WEIGHTS_ARTIFACT_VERSION: u32 = 1;
pub const WEIGHTS_ARTIFACT_TYPE: &str = "cartoboost.weights";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsArtifact {
    pub artifact_type: String,
    pub weights_artifact_version: u32,
    pub model_artifact_version: u32,
    pub backend: String,
    pub model: Model,
}

pub fn save_json<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, model)?;
    Ok(())
}

pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let reader = BufReader::new(File::open(path)?);
    Ok(serde_json::from_reader(reader)?)
}

pub fn save_weights_json(model: &Model, path: impl AsRef<Path>) -> Result<()> {
    let backend = model
        .training_config
        .as_ref()
        .and_then(|config| config.backend.as_ref())
        .map_or_else(
            || "rust".to_string(),
            |selection| selection.selected.clone(),
        );
    let artifact = WeightsArtifact {
        artifact_type: WEIGHTS_ARTIFACT_TYPE.to_string(),
        weights_artifact_version: WEIGHTS_ARTIFACT_VERSION,
        model_artifact_version: model.artifact_version,
        backend,
        model: model.clone(),
    };
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, &artifact)?;
    Ok(())
}

pub fn load_weights_json(path: impl AsRef<Path>) -> Result<Model> {
    let reader = BufReader::new(File::open(path)?);
    let value: serde_json::Value = serde_json::from_reader(reader)?;
    if value.get("artifact_version").is_some() && value.get("trees").is_some() {
        return Ok(serde_json::from_value(value)?);
    }

    let artifact_type = value
        .get("artifact_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput("missing weights artifact type".to_string())
        })?;
    if artifact_type != WEIGHTS_ARTIFACT_TYPE {
        return Err(CartoBoostError::InvalidInput(format!(
            "unsupported weights artifact type {artifact_type}"
        )));
    }
    let weights_artifact_version = value
        .get("weights_artifact_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput("missing weights artifact version".to_string())
        })?;
    if weights_artifact_version != u64::from(WEIGHTS_ARTIFACT_VERSION) {
        return Err(CartoBoostError::InvalidInput(format!(
            "unsupported weights artifact version {weights_artifact_version}"
        )));
    }

    let artifact: WeightsArtifact = serde_json::from_value(value)?;
    Ok(artifact.model)
}
