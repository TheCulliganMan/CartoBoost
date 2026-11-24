use crate::forecasting::ForecastResult;
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};

pub const FORECAST_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MANIFEST_FILE: &str = "manifest.json";
pub const DEFAULT_FORECAST_FILE: &str = "forecast.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastArtifactManifest {
    pub schema_version: u32,
    pub model_name: String,
    pub horizon: usize,
    pub forecast_path: String,
    #[serde(default)]
    pub frequency: Option<String>,
    #[serde(default)]
    pub target_column: Option<String>,
    #[serde(default)]
    pub time_column: Option<String>,
    #[serde(default)]
    pub series_id_column: Option<String>,
    #[serde(default)]
    pub feature_config: Map<String, Value>,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub metrics: Map<String, Value>,
    #[serde(default)]
    pub interval_metadata: Map<String, Value>,
    #[serde(default)]
    pub ensemble_metadata: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastArtifact {
    pub manifest: ForecastArtifactManifest,
    pub forecast: ForecastResult,
}

impl ForecastArtifactManifest {
    pub fn new(
        model_name: impl Into<String>,
        horizon: usize,
        forecast_path: impl Into<String>,
    ) -> Result<Self> {
        let model_name = model_name.into();
        let forecast_path = forecast_path.into();
        if model_name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "artifact model_name must be non-empty".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "artifact horizon must be positive".to_string(),
            ));
        }
        if forecast_path.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "artifact forecast_path must be non-empty".to_string(),
            ));
        }
        let manifest = Self {
            schema_version: FORECAST_ARTIFACT_SCHEMA_VERSION,
            model_name,
            horizon,
            forecast_path,
            frequency: None,
            target_column: None,
            time_column: None,
            series_id_column: None,
            feature_config: Map::new(),
            params: Map::new(),
            metrics: Map::new(),
            interval_metadata: Map::new(),
            ensemble_metadata: Map::new(),
            metadata: Map::new(),
        };
        validate_manifest(&manifest)?;
        Ok(manifest)
    }
}

impl ForecastArtifact {
    pub fn new(manifest: ForecastArtifactManifest, forecast: ForecastResult) -> Result<Self> {
        validate_manifest(&manifest)?;
        validate_manifest_matches_forecast(&manifest, &forecast)?;
        Ok(Self { manifest, forecast })
    }

    pub fn save_json(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        validate_manifest(&self.manifest)?;
        validate_manifest_matches_forecast(&self.manifest, &self.forecast)?;
        fs::create_dir_all(directory)?;
        let manifest_path = directory.join(DEFAULT_MANIFEST_FILE);
        let forecast_path = resolve_child_path(directory, &self.manifest.forecast_path)?;
        if let Some(parent) = forecast_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut forecast_file = BufWriter::new(File::create(forecast_path)?);
        serde_json::to_writer_pretty(&mut forecast_file, &self.forecast)?;
        forecast_file.flush()?;
        let mut manifest_file = BufWriter::new(File::create(manifest_path)?);
        serde_json::to_writer_pretty(&mut manifest_file, &self.manifest)?;
        manifest_file.flush()?;
        Ok(())
    }

    pub fn load_json(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest_path = directory.join(DEFAULT_MANIFEST_FILE);
        let manifest_file = BufReader::new(File::open(manifest_path)?);
        let manifest: ForecastArtifactManifest = serde_json::from_reader(manifest_file)?;
        if manifest.schema_version != FORECAST_ARTIFACT_SCHEMA_VERSION {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported forecast artifact schema_version {}",
                manifest.schema_version
            )));
        }
        validate_manifest(&manifest)?;
        let forecast_path = resolve_child_path(directory, &manifest.forecast_path)?;
        let forecast_file = BufReader::new(File::open(forecast_path)?);
        let forecast: ForecastResult = serde_json::from_reader(forecast_file)?;
        let forecast = forecast.into_validated()?;
        Self::new(manifest, forecast)
    }
}

fn validate_manifest(manifest: &ForecastArtifactManifest) -> Result<()> {
    if manifest.schema_version != FORECAST_ARTIFACT_SCHEMA_VERSION {
        return Err(CartoBoostError::InvalidInput(format!(
            "unsupported forecast artifact schema_version {}",
            manifest.schema_version
        )));
    }
    if Path::new(&manifest.forecast_path) == Path::new(DEFAULT_MANIFEST_FILE) {
        return Err(CartoBoostError::InvalidInput(
            "artifact forecast_path must not overwrite manifest.json".to_string(),
        ));
    }
    if manifest.model_name.trim().is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "artifact model_name must be non-empty".to_string(),
        ));
    }
    if manifest.horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "artifact horizon must be positive".to_string(),
        ));
    }
    resolve_child_path(Path::new("."), &manifest.forecast_path)?;
    Ok(())
}

fn validate_manifest_matches_forecast(
    manifest: &ForecastArtifactManifest,
    forecast: &ForecastResult,
) -> Result<()> {
    if forecast
        .predictions()
        .iter()
        .any(|prediction| prediction.horizon > manifest.horizon)
    {
        return Err(CartoBoostError::InvalidInput(
            "artifact forecast contains predictions beyond manifest horizon".to_string(),
        ));
    }
    if forecast
        .predictions()
        .iter()
        .any(|prediction| prediction.model != manifest.model_name)
    {
        return Err(CartoBoostError::InvalidInput(
            "artifact forecast prediction models must match manifest model_name".to_string(),
        ));
    }
    Ok(())
}

fn resolve_child_path(directory: &Path, child: &str) -> Result<PathBuf> {
    let path = Path::new(child);
    if child.trim().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CartoBoostError::InvalidInput(
            "artifact child paths must be relative and stay inside the artifact directory"
                .to_string(),
        ));
    }
    Ok(directory.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{ForecastPrediction, ForecastResult};
    use chrono::NaiveDate;

    #[test]
    fn artifact_manifest_and_forecast_round_trip_as_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let forecast = ForecastResult::new(vec![ForecastPrediction {
            series_id: "PULocationID=100".to_string(),
            timestamp: ts(),
            horizon: 1,
            model: "naive".to_string(),
            mean: 42.0,
        }])
        .expect("forecast");
        let mut manifest =
            ForecastArtifactManifest::new("naive", 1, DEFAULT_FORECAST_FILE).expect("manifest");
        manifest.target_column = Some("pickup_demand".to_string());
        manifest
            .metadata
            .insert("split".to_string(), Value::from("taxi_holdout"));
        let artifact = ForecastArtifact::new(manifest, forecast).expect("artifact");

        artifact.save_json(dir.path()).expect("save");
        let restored = ForecastArtifact::load_json(dir.path()).expect("load");

        assert_eq!(restored, artifact);
    }

    #[test]
    fn artifact_rejects_parent_path_forecast_file() {
        let err = resolve_child_path(Path::new("/tmp/artifact"), "../forecast.json")
            .expect_err("invalid path");

        assert!(err.to_string().contains("relative"));
    }

    #[test]
    fn artifact_round_trip_supports_nested_forecast_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let forecast = ForecastResult::new(vec![ForecastPrediction {
            series_id: "PULocationID=101".to_string(),
            timestamp: ts(),
            horizon: 1,
            model: "theta".to_string(),
            mean: 12.0,
        }])
        .expect("forecast");
        let manifest =
            ForecastArtifactManifest::new("theta", 1, "forecasts/holdout.json").expect("manifest");
        let artifact = ForecastArtifact::new(manifest, forecast).expect("artifact");

        artifact.save_json(dir.path()).expect("save");
        let restored = ForecastArtifact::load_json(dir.path()).expect("load");

        assert_eq!(restored, artifact);
    }

    #[test]
    fn artifact_rejects_manifest_model_mismatch() {
        let forecast = ForecastResult::new(vec![ForecastPrediction {
            series_id: "PULocationID=100".to_string(),
            timestamp: ts(),
            horizon: 1,
            model: "naive".to_string(),
            mean: 42.0,
        }])
        .expect("forecast");
        let manifest = ForecastArtifactManifest::new("seasonal_naive", 1, DEFAULT_FORECAST_FILE)
            .expect("manifest");

        let err = ForecastArtifact::new(manifest, forecast).expect_err("model mismatch");

        assert!(err.to_string().contains("must match manifest model_name"));
    }

    #[test]
    fn artifact_rejects_manifest_path_collision_and_unsupported_schema() {
        assert!(ForecastArtifactManifest::new("naive", 1, DEFAULT_MANIFEST_FILE).is_err());
        let mut manifest =
            ForecastArtifactManifest::new("naive", 1, DEFAULT_FORECAST_FILE).expect("manifest");
        manifest.schema_version += 1;
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn artifact_load_validates_deserialized_prediction_contents() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest =
            ForecastArtifactManifest::new("naive", 1, DEFAULT_FORECAST_FILE).expect("manifest");
        fs::write(
            dir.path().join(DEFAULT_MANIFEST_FILE),
            serde_json::to_vec(&manifest).expect("manifest JSON"),
        )
        .expect("write manifest");
        for invalid_forecast in [
            serde_json::json!({"predictions": []}),
            serde_json::json!({"predictions": [{
                "series_id": "PULocationID=100",
                "timestamp": ts(),
                "horizon": 0,
                "model": "naive",
                "mean": 42.0
            }]}),
        ] {
            fs::write(
                dir.path().join(DEFAULT_FORECAST_FILE),
                serde_json::to_vec(&invalid_forecast).expect("forecast JSON"),
            )
            .expect("write forecast");
            assert!(ForecastArtifact::load_json(dir.path()).is_err());
        }
    }

    fn ts() -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, 1)
            .expect("date")
            .and_hms_opt(0, 0, 0)
            .expect("time")
    }
}
