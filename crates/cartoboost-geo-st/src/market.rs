//! Learned, auditable structure for directional market panels.
//!
//! This module deliberately accepts generic lane panels.  Domain ingestion and
//! aggregation belong in callers; the native model owns relationship learning,
//! smoothing, prediction, and explanation.

use crate::{GeoStError, Result};
use cartoboost_neural::{GraphSageConfig, GraphSageEncoder, HomogeneousGraph};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipKind {
    SharedOrigin,
    SharedDestination,
    ReverseLane,
    Geographic,
    ResidualCorrelation,
    NeuralKernel,
    Expert,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MarketShiftKind {
    Market,
    LocalOrMix,
    NoShift,
}

/// Provenance for the current lane state used in a nowcast explanation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MarketSupportKind {
    Lane,
    Hierarchy,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExpertRelationshipPrior {
    pub version: String,
    pub source_lane_id: String,
    pub target_lane_id: String,
    pub allowed: bool,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExpertEventLabel {
    pub lane_id: String,
    pub timestamp: i64,
    pub shift: MarketShiftKind,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketPanelFrame {
    pub lane_ids: Vec<String>,
    pub timestamps: Vec<i64>,
    /// Caller-selected target names. The first is the smoothed primary target;
    /// the second is modeled jointly as a supporting target.
    pub target_names: Vec<String>,
    /// Positive primary observations shaped `[time, lane]`; `NaN` denotes an
    /// unobserved lane-day and is not treated as a zero or filled value.
    pub primary: Vec<Vec<f64>>,
    /// Secondary target, shaped `[time, lane]`; nonnegative for log modeling.
    pub secondary: Vec<Vec<f64>>,
    pub origin_ids: Vec<String>,
    pub destination_ids: Vec<String>,
    /// Optional generic parent-group keys for each lane. Supply H3/S2 parent
    /// cells or another stable multi-resolution geography from the caller.
    pub hierarchy_groups: Vec<Vec<String>>,
    /// `[origin_x, origin_y, destination_x, destination_y]` for each lane.
    pub coordinates: Vec<[f64; 4]>,
    /// Known-at-cutoff daily calendar features, shaped `[time, feature]`.
    pub calendar: Vec<Vec<f64>>,
    /// Historical lane mix features, shaped `[time, lane, feature]`.
    pub mix: Option<Vec<Vec<Vec<f64>>>>,
    pub expert_priors: Vec<ExpertRelationshipPrior>,
    pub expert_labels: Vec<ExpertEventLabel>,
    pub horizon: usize,
    pub frequency: String,
}

// Market-model responsibilities share this module namespace.
include!("market/frame.rs");
include!("market/contracts.rs");
include!("market/forecaster.rs");
include!("market/relationships.rs");
include!("market/joint_heads.rs");
include!("market/calibration.rs");
include!("market/numerics.rs");
include!("market/tests.rs");
