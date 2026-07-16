pub use cartoboost_geo_core::SpatialWeights;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpatialEconError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("linear system is singular or ill-conditioned")]
    SingularSystem,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("geo core error: {0}")]
    GeoCore(#[from] cartoboost_geo_core::GeoCoreError),
}

pub type Result<T> = std::result::Result<T, SpatialEconError>;
type SpatialEffects = (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>);

pub fn spatial_weights_from_coo(
    n_rows: usize,
    n_cols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
    row_standardize: bool,
) -> Result<SpatialWeights> {
    if n_rows != n_cols {
        return Err(SpatialEconError::InvalidInput(
            "spatial weights must be square".to_string(),
        ));
    }
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err(SpatialEconError::InvalidInput(
            "weights rows, cols, and values must have the same length".to_string(),
        ));
    }
    let mut edges = Vec::with_capacity(values.len());
    for ((row, col), value) in rows.into_iter().zip(cols).zip(values) {
        if !value.is_finite() {
            return Err(SpatialEconError::InvalidInput(
                "spatial weights must contain only finite values".to_string(),
            ));
        }
        if row == col && value != 0.0 {
            return Err(SpatialEconError::InvalidInput(
                "spatial econometric weights must have a zero diagonal".to_string(),
            ));
        }
        if value != 0.0 {
            edges.push((row, col, value));
        }
    }
    let weights = SpatialWeights::from_edges(n_rows, edges, false)?;
    Ok(if row_standardize {
        weights.row_normalize()
    } else {
        weights
    })
}

// Cohesive implementation families share the crate namespace.
include!("spatial/models.rs");
include!("spatial/fitting.rs");
include!("spatial/effects.rs");
include!("spatial/linear_algebra.rs");
include!("spatial/tests.rs");
