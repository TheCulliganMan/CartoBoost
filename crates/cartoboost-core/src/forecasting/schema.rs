use crate::forecasting::{parse_forecast_timestamp, ForecastFrequency};
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

pub const SINGLE_SERIES_ID: &str = "__single__";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastRow {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub target: f64,
    #[serde(default)]
    pub covariates: BTreeMap<String, f64>,
}

impl ForecastRow {
    pub fn new(series_id: impl Into<String>, timestamp: NaiveDateTime, target: f64) -> Self {
        Self::with_covariates(series_id, timestamp, target, BTreeMap::new())
    }

    pub fn with_covariates(
        series_id: impl Into<String>,
        timestamp: NaiveDateTime,
        target: f64,
        covariates: BTreeMap<String, f64>,
    ) -> Self {
        let series_id = series_id.into();
        Self {
            series_id: if series_id.is_empty() {
                SINGLE_SERIES_ID.to_string()
            } else {
                series_id
            },
            timestamp,
            target,
            covariates,
        }
    }

    pub fn single(timestamp: NaiveDateTime, target: f64) -> Self {
        Self::new(SINGLE_SERIES_ID, timestamp, target)
    }

    pub fn from_timestamp_str(
        series_id: impl Into<String>,
        timestamp: &str,
        target: f64,
    ) -> Result<Self> {
        Ok(Self::with_covariates(
            series_id,
            parse_forecast_timestamp(timestamp)?,
            target,
            BTreeMap::new(),
        ))
    }

    pub fn from_timestamp_str_with_covariates(
        series_id: impl Into<String>,
        timestamp: &str,
        target: f64,
        covariates: BTreeMap<String, f64>,
    ) -> Result<Self> {
        Ok(Self::with_covariates(
            series_id,
            parse_forecast_timestamp(timestamp)?,
            target,
            covariates,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastFrame {
    rows: Vec<ForecastRow>,
    frequency: ForecastFrequency,
    metadata: ForecastFrameMetadata,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFrameMetadata {
    pub timestamp_col: Option<String>,
    pub target_col: Option<String>,
    pub series_id_col: Option<String>,
    pub static_covariates: Vec<String>,
    pub known_future_covariates: Vec<String>,
    pub historical_covariates: Vec<String>,
    #[serde(default)]
    pub allow_irregular: bool,
    #[serde(default)]
    pub allow_missing_targets: bool,
    #[serde(default)]
    pub allow_missing_covariates: bool,
}

impl ForecastFrame {
    pub fn new(rows: Vec<ForecastRow>, frequency: ForecastFrequency) -> Result<Self> {
        Self::with_metadata(rows, frequency, ForecastFrameMetadata::default())
    }

    pub fn with_metadata(
        mut rows: Vec<ForecastRow>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        if rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "forecast frame must contain at least one row".to_string(),
            ));
        }
        validate_metadata(&metadata)?;
        for row in &mut rows {
            if row.series_id.is_empty() {
                row.series_id = SINGLE_SERIES_ID.to_string();
            }
            if row.target.is_infinite() {
                return Err(CartoBoostError::InvalidInput(
                    "forecast targets must not be infinite".to_string(),
                ));
            }
            if !metadata.allow_missing_targets && row.target.is_nan() {
                return Err(CartoBoostError::InvalidInput(
                    "forecast targets must be finite unless allow_missing_targets is enabled"
                        .to_string(),
                ));
            }
            for (name, value) in &row.covariates {
                if name.is_empty() {
                    return Err(CartoBoostError::InvalidInput(
                        "forecast covariate names must not be empty".to_string(),
                    ));
                }
                if value.is_infinite() {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "forecast covariate {name:?} for series {} at {} must not be infinite",
                        row.series_id, row.timestamp
                    )));
                }
                if !metadata.allow_missing_covariates && value.is_nan() {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "forecast covariate {name:?} for series {} at {} must be finite unless allow_missing_covariates is enabled",
                        row.series_id, row.timestamp
                    )));
                }
            }
        }
        rows.sort_by(|a, b| {
            a.series_id
                .cmp(&b.series_id)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });

        let mut seen = HashSet::with_capacity(rows.len());
        for row in &rows {
            if !seen.insert((row.series_id.clone(), row.timestamp)) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate forecast timestamp {} for series {}",
                    row.timestamp, row.series_id
                )));
            }
        }
        if !metadata.allow_irregular {
            validate_regular_frequency(&rows, frequency)?;
        }
        Ok(Self {
            rows,
            frequency,
            metadata,
        })
    }

    pub fn from_string_rows(
        rows: Vec<(String, String, f64)>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        let parsed = rows
            .into_iter()
            .map(|(series_id, timestamp, target)| {
                ForecastRow::from_timestamp_str(series_id, &timestamp, target)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_metadata(parsed, frequency, metadata)
    }

    pub fn from_string_rows_with_covariates(
        rows: Vec<(String, String, f64, BTreeMap<String, f64>)>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        let parsed = rows
            .into_iter()
            .map(|(series_id, timestamp, target, covariates)| {
                ForecastRow::from_timestamp_str_with_covariates(
                    series_id, &timestamp, target, covariates,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_metadata(parsed, frequency, metadata)
    }

    pub fn from_string_rows_with_covariates_and_weights(
        rows: Vec<(String, String, f64, BTreeMap<String, f64>)>,
        sample_weights: Vec<f64>,
        sample_weight_col: Option<String>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        if rows.len() != sample_weights.len() {
            return Err(CartoBoostError::InvalidInput(
                "forecast sample weights length must match rows length".to_string(),
            ));
        }
        let parsed = rows
            .into_iter()
            .map(|(series_id, timestamp, target, covariates)| {
                ForecastRow::from_timestamp_str_with_covariates(
                    series_id, &timestamp, target, covariates,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let aggregated =
            aggregate_weighted_forecast_rows(parsed, sample_weights, sample_weight_col.as_deref())?;
        Self::with_metadata(aggregated, frequency, metadata)
    }

    pub fn rows(&self) -> &[ForecastRow] {
        &self.rows
    }

    pub fn frequency(&self) -> ForecastFrequency {
        self.frequency
    }

    pub fn series_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        let mut last: Option<&str> = None;
        for row in &self.rows {
            if last != Some(row.series_id.as_str()) {
                ids.push(row.series_id.clone());
                last = Some(row.series_id.as_str());
            }
        }
        ids
    }

    pub fn rows_for_series(&self, series_id: &str) -> Vec<&ForecastRow> {
        self.rows
            .iter()
            .filter(|row| row.series_id == series_id)
            .collect()
    }

    pub fn metadata(&self) -> &ForecastFrameMetadata {
        &self.metadata
    }

    pub fn metadata_value(&self) -> Value {
        json!({
            "timestamp_col": self.metadata.timestamp_col,
            "target_col": self.metadata.target_col,
            "series_id_col": self.metadata.series_id_col,
            "frequency": self.frequency.as_str(),
            "is_panel": self.is_panel(),
            "n_rows": self.rows.len(),
            "series_ids": self.series_ids(),
            "static_covariates": self.metadata.static_covariates,
            "known_future_covariates": self.metadata.known_future_covariates,
            "historical_covariates": self.metadata.historical_covariates,
            "allow_irregular": self.metadata.allow_irregular,
            "allow_missing_targets": self.metadata.allow_missing_targets,
            "allow_missing_covariates": self.metadata.allow_missing_covariates,
        })
    }

    pub fn allow_irregular(&self) -> bool {
        self.metadata.allow_irregular
    }

    pub fn is_irregular(&self) -> bool {
        self.allow_irregular()
    }

    pub fn allow_missing_targets(&self) -> bool {
        self.metadata.allow_missing_targets
    }

    pub fn has_missing_targets(&self) -> bool {
        self.rows.iter().any(|row| row.target.is_nan())
    }

    pub fn allow_missing_covariates(&self) -> bool {
        self.metadata.allow_missing_covariates
    }

    pub fn has_missing_covariates(&self) -> bool {
        self.rows
            .iter()
            .any(|row| row.covariates.values().any(|value| value.is_nan()))
    }

    pub fn require_regular_for_model(&self, model_name: &str) -> Result<()> {
        self.require_observed_targets_for_model(model_name)?;
        if self.allow_irregular() {
            return Err(CartoBoostError::InvalidInput(format!(
                "{model_name} requires a regular ForecastFrame; use a regularized frame or a model that supports irregular history"
            )));
        }
        Ok(())
    }

    pub fn require_observed_targets_for_model(&self, model_name: &str) -> Result<()> {
        if self.has_missing_targets() {
            return Err(CartoBoostError::InvalidInput(format!(
                "{model_name} requires observed finite targets; drop missing target rows or use a model that supports allow_missing_targets"
            )));
        }
        Ok(())
    }

    pub fn observed_target_frame(&self, model_name: &str) -> Result<Self> {
        if !self.has_missing_targets() {
            return Ok(self.clone());
        }
        let mut observed_by_series: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_by_series: BTreeMap<String, usize> = BTreeMap::new();
        let mut rows = Vec::new();
        for row in &self.rows {
            *total_by_series.entry(row.series_id.clone()).or_default() += 1;
            if row.target.is_finite() {
                *observed_by_series.entry(row.series_id.clone()).or_default() += 1;
                rows.push(row.clone());
            }
        }
        for series_id in total_by_series.keys() {
            if !observed_by_series.contains_key(series_id) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{model_name} requires at least one observed target row for series {series_id:?}"
                )));
            }
        }
        let mut metadata = self.metadata.clone();
        metadata.allow_irregular = true;
        metadata.allow_missing_targets = false;
        Self::with_metadata(rows, self.frequency, metadata)
    }

    pub fn metadata_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.metadata_value()).map_err(CartoBoostError::from)
    }

    pub fn is_panel(&self) -> bool {
        self.series_ids()
            .iter()
            .any(|series_id| series_id != SINGLE_SERIES_ID)
    }
}

#[derive(Debug, Default)]
struct WeightedForecastRowAccumulator {
    total_weight: f64,
    weighted_target: f64,
    weighted_covariates: BTreeMap<String, f64>,
}

fn aggregate_weighted_forecast_rows(
    rows: Vec<ForecastRow>,
    sample_weights: Vec<f64>,
    sample_weight_col: Option<&str>,
) -> Result<Vec<ForecastRow>> {
    let mut groups: BTreeMap<(String, NaiveDateTime), WeightedForecastRowAccumulator> =
        BTreeMap::new();
    for (row, weight) in rows.into_iter().zip(sample_weights) {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast sample weights must be positive finite values".to_string(),
            ));
        }
        let key = (row.series_id.clone(), row.timestamp);
        let entry = groups.entry(key).or_default();
        entry.total_weight += weight;
        entry.weighted_target += weight * row.target;
        for (name, value) in row.covariates {
            *entry.weighted_covariates.entry(name).or_insert(0.0) += weight * value;
        }
    }
    groups
        .into_iter()
        .map(|((series_id, timestamp), accumulator)| {
            if accumulator.total_weight <= 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "forecast sample weights must have positive grouped totals".to_string(),
                ));
            }
            let covariates = accumulator
                .weighted_covariates
                .into_iter()
                .map(|(name, value)| {
                    if sample_weight_col == Some(name.as_str()) {
                        (name, accumulator.total_weight)
                    } else {
                        (name, value / accumulator.total_weight)
                    }
                })
                .collect();
            Ok(ForecastRow::with_covariates(
                series_id,
                timestamp,
                accumulator.weighted_target / accumulator.total_weight,
                covariates,
            ))
        })
        .collect()
}

fn validate_metadata(metadata: &ForecastFrameMetadata) -> Result<()> {
    let mut seen = HashSet::new();
    for (label, values) in [
        ("static_covariates", &metadata.static_covariates),
        ("known_future_covariates", &metadata.known_future_covariates),
        ("historical_covariates", &metadata.historical_covariates),
    ] {
        let mut role_seen = HashSet::new();
        for value in values {
            if value.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{label} must not contain empty column names"
                )));
            }
            if !role_seen.insert(value) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{label} must not contain duplicate column names"
                )));
            }
            if !seen.insert(value) {
                return Err(CartoBoostError::InvalidInput(
                    "forecast covariate columns must belong to only one role".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_regular_frequency(rows: &[ForecastRow], frequency: ForecastFrequency) -> Result<()> {
    let expected = frequency.step();
    let mut previous: Option<&ForecastRow> = None;
    for row in rows {
        if let Some(prev) = previous {
            if prev.series_id == row.series_id {
                let observed = row.timestamp - prev.timestamp;
                if observed != expected {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "irregular forecast frequency for series {} between {} and {}; expected {:?}, observed {:?}",
                        row.series_id, prev.timestamp, row.timestamp, expected, observed
                    )));
                }
            }
        }
        previous = Some(row);
    }
    Ok(())
}
