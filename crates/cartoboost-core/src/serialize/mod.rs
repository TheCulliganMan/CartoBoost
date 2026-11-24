use crate::data::{FeatureKind, FeatureSchema};
use crate::tree::{Model, Tree};
use crate::{CartoBoostError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
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
    write_json(model, File::create(path)?)
}

fn write_json<T: Serialize>(model: &T, writer: impl Write) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer_pretty(&mut writer, model)?;
    // BufWriter's destructor cannot report an I/O failure. A successful save
    // must include flushing the final buffered bytes to its destination.
    writer.flush()?;
    Ok(())
}

pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let reader = BufReader::new(File::open(path)?);
    Ok(serde_json::from_reader(reader)?)
}

pub(crate) fn validate_model_layout<'a>(
    feature_count: usize,
    schema: Option<&FeatureSchema>,
    trees: impl IntoIterator<Item = &'a Tree>,
) -> Result<()> {
    let sparse_count = if let Some(schema) = schema.filter(|schema| !schema.is_empty()) {
        schema.validate()?;
        if schema.len() < feature_count
            || schema.kinds[..feature_count]
                .iter()
                .any(|kind| matches!(kind, FeatureKind::SparseSet))
            || schema.kinds[feature_count..]
                .iter()
                .any(|kind| !matches!(kind, FeatureKind::SparseSet))
        {
            return Err(CartoBoostError::InvalidInput(
                "model feature schema does not match dense and sparse feature layout".to_string(),
            ));
        }
        Some(schema.len() - feature_count)
    } else {
        None
    };
    for tree in trees {
        tree.validate_model_layout(feature_count, sparse_count)?;
    }
    Ok(())
}

pub fn save_weights_json(model: &Model, path: impl AsRef<Path>) -> Result<()> {
    let backend = model
        .training_config
        .as_ref()
        .and_then(|config| config.backend.as_ref())
        .map_or("rust", |selection| selection.selected.as_str());
    // Keep the owned public artifact type for callers and deserialization,
    // while saving the same envelope without cloning the complete forest.
    #[derive(Serialize)]
    struct BorrowedWeightsArtifact<'a> {
        artifact_type: &'a str,
        weights_artifact_version: u32,
        model_artifact_version: u32,
        backend: &'a str,
        model: &'a Model,
    }
    let artifact = BorrowedWeightsArtifact {
        artifact_type: WEIGHTS_ARTIFACT_TYPE,
        weights_artifact_version: WEIGHTS_ARTIFACT_VERSION,
        model_artifact_version: model.artifact_version,
        backend,
        model,
    };
    save_json(&artifact, path)
}

pub fn load_weights_json(path: impl AsRef<Path>) -> Result<Model> {
    let reader = BufReader::new(File::open(path)?);
    let value: serde_json::Value = serde_json::from_reader(reader)?;
    if value.get("artifact_version").is_some() && value.get("trees").is_some() {
        let model: Model = serde_json::from_value(value)?;
        model.validate_loaded()?;
        return Ok(model);
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
    if artifact.model_artifact_version != artifact.model.artifact_version {
        return Err(CartoBoostError::InvalidInput(
            "weights envelope model version does not match embedded model".to_string(),
        ));
    }
    artifact.model.validate_loaded()?;
    Ok(artifact.model)
}
