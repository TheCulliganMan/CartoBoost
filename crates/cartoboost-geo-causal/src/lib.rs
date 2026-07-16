use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, thiserror::Error)]
pub enum GeoCausalError {
    #[error("{0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, GeoCausalError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCausalRow {
    pub unit_id: String,
    pub time: String,
    pub outcome: f64,
    pub treatment: bool,
    pub covariates: BTreeMap<String, f64>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub region_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialWeight {
    pub from_unit: String,
    pub to_unit: String,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCausalPanel {
    rows: Vec<GeoCausalRow>,
    spatial_weights: Vec<SpatialWeight>,
}

impl GeoCausalPanel {
    pub fn new(rows: Vec<GeoCausalRow>, spatial_weights: Vec<SpatialWeight>) -> Result<Self> {
        if rows.is_empty() {
            return Err(GeoCausalError::InvalidInput(
                "GeoCausalPanel requires at least one row".to_string(),
            ));
        }
        for (idx, row) in rows.iter().enumerate() {
            if row.unit_id.is_empty() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} has an empty unit_id"
                )));
            }
            if row.time.is_empty() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} has an empty time"
                )));
            }
            if !row.outcome.is_finite() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} outcome must be finite"
                )));
            }
            if row.latitude.is_some() != row.longitude.is_some() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} must provide both latitude and longitude or neither"
                )));
            }
        }
        let units: BTreeSet<_> = rows.iter().map(|row| row.unit_id.clone()).collect();
        for edge in &spatial_weights {
            if !units.contains(&edge.from_unit) || !units.contains(&edge.to_unit) {
                return Err(GeoCausalError::InvalidInput(format!(
                    "spatial weight references unknown units {} -> {}",
                    edge.from_unit, edge.to_unit
                )));
            }
            if !edge.weight.is_finite() || edge.weight < 0.0 {
                return Err(GeoCausalError::InvalidInput(
                    "spatial weights must be finite and non-negative".to_string(),
                ));
            }
        }
        Ok(Self {
            rows,
            spatial_weights,
        })
    }

    pub fn rows(&self) -> &[GeoCausalRow] {
        &self.rows
    }

    pub fn spatial_weights(&self) -> &[SpatialWeight] {
        &self.spatial_weights
    }

    pub fn unit_ids(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| row.unit_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn times(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| row.time.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

// Cohesive implementation families share the crate namespace.
include!("causal/synthetic_did.rs");
include!("causal/experiment.rs");
include!("causal/spillover.rs");
include!("causal/representation.rs");
include!("causal/helpers.rs");
include!("causal/tests.rs");
