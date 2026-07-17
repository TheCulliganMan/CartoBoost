//! Learned, auditable structure for directional market panels.
//!
//! This module deliberately accepts generic lane panels.  Domain ingestion and
//! aggregation belong in callers; the native model owns relationship learning,
//! smoothing, prediction, and explanation.

use crate::{GeoStError, Result};
use cartoboost_neural::{
    backend_dense_layer_f32, select_backend_for, BackendOperation, BackendSelection,
    GraphSageConfig, GraphSageEncoder, HomogeneousGraph,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const MARKET_HEAD_DENSE_DISPATCH_MIN_OPS: usize = 16_384;

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

impl MarketPanelFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lane_ids: Vec<String>,
        timestamps: Vec<i64>,
        target_names: Vec<String>,
        primary: Vec<Vec<f64>>,
        secondary: Vec<Vec<f64>>,
        origin_ids: Vec<String>,
        destination_ids: Vec<String>,
        hierarchy_groups: Vec<Vec<String>>,
        coordinates: Vec<[f64; 4]>,
        calendar: Vec<Vec<f64>>,
        mix: Option<Vec<Vec<Vec<f64>>>>,
        expert_priors: Vec<ExpertRelationshipPrior>,
        expert_labels: Vec<ExpertEventLabel>,
        horizon: usize,
        frequency: String,
    ) -> Result<Self> {
        let frame = Self {
            lane_ids,
            timestamps,
            target_names,
            primary,
            secondary,
            origin_ids,
            destination_ids,
            hierarchy_groups,
            coordinates,
            calendar,
            mix,
            expert_priors,
            expert_labels,
            horizon,
            frequency,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<()> {
        let lanes = self.lane_ids.len();
        let times = self.timestamps.len();
        if lanes == 0 || times <= self.horizon || self.horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "market frame requires lanes and more rows than a positive horizon".to_string(),
            ));
        }
        if self.target_names.len() != 2
            || self.target_names.iter().any(String::is_empty)
            || self.target_names[0] == self.target_names[1]
        {
            return Err(GeoStError::InvalidFrame(
                "market frame requires two distinct nonempty target names".to_string(),
            ));
        }
        if self.primary.len() != times
            || self.secondary.len() != times
            || self.calendar.len() != times
            || self.origin_ids.len() != lanes
            || self.destination_ids.len() != lanes
            || self.hierarchy_groups.len() != lanes
            || self.coordinates.len() != lanes
        {
            return Err(GeoStError::InvalidFrame(
                "market frame dimensions do not agree".to_string(),
            ));
        }
        if self.timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(GeoStError::InvalidFrame(
                "market timestamps must be strictly increasing".to_string(),
            ));
        }
        let calendar_width = self.calendar.first().map_or(0, Vec::len);
        for row in &self.calendar {
            if row.len() != calendar_width || row.iter().any(|x| !x.is_finite()) {
                return Err(GeoStError::InvalidFrame(
                    "calendar features must be finite and rectangular".to_string(),
                ));
            }
        }
        for (primary_row, secondary_row) in self.primary.iter().zip(&self.secondary) {
            if primary_row.len() != lanes
                || secondary_row.len() != lanes
                || primary_row
                    .iter()
                    .any(|x| x.is_infinite() || (!x.is_nan() && *x <= 0.0))
                || secondary_row.iter().any(|x| !x.is_finite() || *x < 0.0)
            {
                return Err(GeoStError::InvalidFrame("primary observations must be positive or NaN and secondary target nonnegative with shape [time, lane]".to_string()));
            }
        }
        for lane in 0..lanes {
            let observed = self
                .primary
                .iter()
                .filter(|row| !row[lane].is_nan())
                .count();
            if observed == 0 && self.hierarchy_groups[lane].is_empty() {
                return Err(GeoStError::InvalidFrame(format!(
                    "lane '{}' has no observed primary values and no hierarchy group for partial pooling",
                    self.lane_ids[lane]
                )));
            }
        }
        if self
            .coordinates
            .iter()
            .any(|point| point.iter().any(|x| !x.is_finite()))
        {
            return Err(GeoStError::InvalidFrame(
                "lane coordinates must be finite".to_string(),
            ));
        }
        if let Some(mix) = &self.mix {
            if mix.len() != times || mix.iter().any(|row| row.len() != lanes) {
                return Err(GeoStError::InvalidFrame(
                    "mix must have shape [time, lane, feature]".to_string(),
                ));
            }
            let width = mix.first().and_then(|row| row.first()).map_or(0, Vec::len);
            if width == 0
                || mix
                    .iter()
                    .flatten()
                    .any(|row| row.len() != width || row.iter().any(|x| !x.is_finite()))
            {
                return Err(GeoStError::InvalidFrame(
                    "mix features must be nonempty, finite, and rectangular".to_string(),
                ));
            }
        }
        let known: BTreeSet<_> = self.lane_ids.iter().collect();
        if self.expert_priors.iter().any(|prior| {
            !known.contains(&prior.source_lane_id)
                || !known.contains(&prior.target_lane_id)
                || !prior.weight.is_finite()
                || prior.weight < 0.0
                || prior.version.is_empty()
        }) {
            return Err(GeoStError::InvalidFrame("expert priors must reference known lanes with a nonnegative finite weight and version".to_string()));
        }
        if self.expert_labels.iter().any(|label| {
            !known.contains(&label.lane_id)
                || !self.timestamps.contains(&label.timestamp)
                || label.version.is_empty()
        }) {
            return Err(GeoStError::InvalidFrame(
                "expert labels must reference a known lane, observed timestamp, and version"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketStructureConfig {
    /// Accelerator used by the trainable graph kernel and persisted artifact.
    #[serde(default = "default_market_backend")]
    pub backend: String,
    pub top_k: usize,
    /// Width of the trainable graph kernel used to score candidate relationships.
    pub neural_hidden_dim: usize,
    /// Self-supervised graph-kernel optimization epochs.
    pub neural_epochs: usize,
    /// Joint supervised optimization epochs for the graph-embedding adapters.
    pub head_epochs: usize,
    pub head_learning_rate: f64,
    pub huber_delta: f64,
    /// Sorted, open-unit-interval levels trained with pinball loss.
    pub quantile_levels: Vec<f64>,
    pub graph_strength: f64,
    pub local_strength: f64,
    pub correlation_floor: f64,
    pub shift_zscore: f64,
    /// Estimate interval widening from a train-only trailing origin.
    pub calibrate_intervals: bool,
}

fn default_market_backend() -> String {
    "cpu".to_string()
}

impl Default for MarketStructureConfig {
    fn default() -> Self {
        Self {
            backend: default_market_backend(),
            top_k: 8,
            neural_hidden_dim: 16,
            neural_epochs: 20,
            head_epochs: 80,
            head_learning_rate: 0.02,
            huber_delta: 1.0,
            quantile_levels: vec![0.1, 0.5, 0.9],
            graph_strength: 0.55,
            local_strength: 0.35,
            correlation_floor: 0.10,
            shift_zscore: 2.0,
            calibrate_intervals: true,
        }
    }
}

impl MarketStructureConfig {
    fn validate(&self) -> Result<()> {
        select_backend_for(Some(&self.backend), BackendOperation::Dense).map_err(|error| {
            GeoStError::InvalidFrame(format!("invalid market accelerator backend: {error}"))
        })?;
        if self.top_k == 0
            || self.top_k > 8
            || self.neural_hidden_dim == 0
            || self.head_epochs == 0
            || !self.head_learning_rate.is_finite()
            || self.head_learning_rate <= 0.0
            || !self.huber_delta.is_finite()
            || self.huber_delta <= 0.0
            || !self.graph_strength.is_finite()
            || !self.local_strength.is_finite()
            || self.graph_strength < 0.0
            || self.local_strength < 0.0
            || self.graph_strength + self.local_strength > 1.0
            || !self.correlation_floor.is_finite()
            || !self.shift_zscore.is_finite()
            || self.shift_zscore <= 0.0
        {
            return Err(GeoStError::InvalidFrame(
                "invalid market structure configuration".to_string(),
            ));
        }
        if self.quantile_levels.is_empty()
            || self
                .quantile_levels
                .iter()
                .any(|level| !level.is_finite() || *level <= 0.0 || *level >= 1.0)
            || self
                .quantile_levels
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(GeoStError::InvalidFrame(
                "quantile levels must be strictly increasing values in (0, 1)".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketRelationship {
    pub source_lane_id: String,
    pub target_lane_id: String,
    pub weight: f64,
    /// Train-only periodic edge multipliers, indexed by `timestamp mod 7`.
    /// They make the sparse learned topology responsive to recurring market state.
    pub periodic_weights: Vec<f64>,
    pub kinds: Vec<RelationshipKind>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketPrediction {
    pub lane_id: String,
    pub timestamp: i64,
    pub horizon: usize,
    pub primary: f64,
    pub primary_lower: f64,
    pub primary_upper: f64,
    pub secondary: f64,
}

/// Calendar-week aggregation of daily native forecasts. Primary values are
/// averaged while the supporting target is summed, preserving their generic
/// caller-selected semantics without inventing an independently trained model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WeeklyMarketPrediction {
    pub lane_id: String,
    pub week_start_timestamp: i64,
    pub days: usize,
    pub primary: f64,
    pub primary_lower: f64,
    pub primary_upper: f64,
    pub secondary: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketExplanation {
    pub lane_id: String,
    pub timestamp: i64,
    pub observed_primary: Option<f64>,
    pub support: MarketSupportKind,
    pub smoothed_primary: f64,
    pub market_component: f64,
    pub local_or_mix_component: f64,
    pub seasonal_component: f64,
    pub residual_component: f64,
    pub uncertainty: f64,
    pub shift: MarketShiftKind,
    pub top_relationships: Vec<MarketRelationship>,
    pub expert_label: Option<ExpertEventLabel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketStructureForecaster {
    pub config: MarketStructureConfig,
    lane_ids: Vec<String>,
    #[serde(default)]
    origin_ids: Vec<String>,
    #[serde(default)]
    destination_ids: Vec<String>,
    #[serde(default)]
    coordinates: Vec<[f64; 4]>,
    hierarchy_groups: Vec<Vec<String>>,
    timestamps: Vec<i64>,
    frequency: String,
    relationships: Vec<Vec<MarketRelationship>>,
    target_names: Vec<String>,
    primary_means: Vec<f64>,
    secondary_means: Vec<f64>,
    primary_scales: Vec<f64>,
    /// Per-lane empirical 80% residual radii fitted only on observed history.
    primary_interval_radii: Vec<f64>,
    interval_calibration_multiplier: f64,
    secondary_scales: Vec<f64>,
    weekly_primary: Vec<Vec<f64>>,
    weekly_secondary: Vec<Vec<f64>>,
    primary_calendar_weights: Vec<f64>,
    secondary_calendar_weights: Vec<f64>,
    primary_history: Vec<Vec<f64>>,
    primary_observed: Vec<Vec<bool>>,
    secondary_history: Vec<Vec<f64>>,
    calendar_width: usize,
    last_calendar: Vec<f64>,
    mix_coefficients: Vec<f64>,
    cross_target_couplings: Vec<f64>,
    last_mix: Option<Vec<f64>>,
    /// Trainable GraphSAGE kernel embeddings of candidate lane relationships.
    neural_embeddings: Vec<Vec<f32>>,
    /// Frozen GraphSAGE encoder plus jointly-trained robust output adapters.
    /// The point adapters use Huber gradients; each quantile adapter uses its
    /// own pinball gradient on the same graph-aware lane state.
    joint_heads: Option<JointMarketHeads>,
    expert_shift_calibration: Option<ExpertShiftCalibration>,
    expert_labels: Vec<ExpertEventLabel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct JointMarketHeads {
    primary_huber: Vec<f64>,
    secondary_huber: Vec<f64>,
    primary_quantiles: Vec<Vec<f64>>,
    secondary_quantiles: Vec<Vec<f64>>,
}

/// Train-only class centroids for optional reviewer-labelled event calibration.
/// Recorded labels remain exposed independently on explanations; these values
/// only calibrate the model's classification boundary for later assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ExpertShiftCalibration {
    market: Option<[f64; 3]>,
    local_or_mix: Option<[f64; 3]>,
    no_shift: Option<[f64; 3]>,
}

impl MarketStructureForecaster {
    pub fn new(mut config: MarketStructureConfig) -> Result<Self> {
        config.validate()?;
        config.backend = select_backend_for(Some(&config.backend), BackendOperation::Dense)
            .map_err(|error| {
                GeoStError::InvalidFrame(format!("invalid market accelerator backend: {error}"))
            })?
            .selected;
        Ok(Self {
            config,
            lane_ids: Vec::new(),
            origin_ids: Vec::new(),
            destination_ids: Vec::new(),
            coordinates: Vec::new(),
            hierarchy_groups: Vec::new(),
            timestamps: Vec::new(),
            frequency: String::new(),
            relationships: Vec::new(),
            target_names: Vec::new(),
            primary_means: Vec::new(),
            secondary_means: Vec::new(),
            primary_scales: Vec::new(),
            primary_interval_radii: Vec::new(),
            interval_calibration_multiplier: 1.0,
            secondary_scales: Vec::new(),
            weekly_primary: Vec::new(),
            weekly_secondary: Vec::new(),
            primary_calendar_weights: Vec::new(),
            secondary_calendar_weights: Vec::new(),
            primary_history: Vec::new(),
            primary_observed: Vec::new(),
            secondary_history: Vec::new(),
            calendar_width: 0,
            last_calendar: Vec::new(),
            mix_coefficients: Vec::new(),
            cross_target_couplings: Vec::new(),
            last_mix: None,
            neural_embeddings: Vec::new(),
            joint_heads: None,
            expert_shift_calibration: None,
            expert_labels: Vec::new(),
        })
    }

    pub fn backend(&self) -> &str {
        &self.config.backend
    }

    pub fn fit(&mut self, frame: &MarketPanelFrame) -> Result<()> {
        frame.validate()?;
        self.config.validate()?;
        let lanes = frame.lane_ids.len();
        let (log_primary, primary_observed) = log_primary_with_missing(&frame.primary);
        let log_secondary = matrix_map(&frame.secondary, |x| (x + 1.0).ln());
        self.primary_means = primary_means_with_hierarchy(frame, &log_primary, &primary_observed)?;
        self.secondary_means = column_means(&log_secondary);
        let primary_residuals =
            centered_masked(&log_primary, &self.primary_means, &primary_observed);
        self.primary_scales = primary_scales_with_hierarchy(
            frame,
            &log_primary,
            &primary_observed,
            &self.primary_means,
        )?;
        self.primary_interval_radii =
            primary_interval_radii(&primary_residuals, &primary_observed, &self.primary_scales);
        self.interval_calibration_multiplier = if self.config.calibrate_intervals {
            interval_calibration_multiplier(frame, &self.config)?
        } else {
            1.0
        };
        self.secondary_scales = column_scales(&centered(&log_secondary, &self.secondary_means));
        self.weekly_primary =
            weekly_effects_masked(&primary_residuals, &primary_observed, &frame.timestamps);
        self.weekly_secondary = weekly_effects(
            &centered(&log_secondary, &self.secondary_means),
            &frame.timestamps,
        );
        self.primary_calendar_weights =
            calendar_weights_masked(&primary_residuals, &primary_observed, &frame.calendar);
        self.secondary_calendar_weights = calendar_weights(
            &centered(&log_secondary, &self.secondary_means),
            &frame.calendar,
        );
        self.mix_coefficients = mix_coefficients(&primary_residuals, frame.mix.as_ref());
        self.cross_target_couplings = cross_target_couplings(
            &primary_residuals,
            &centered(&log_secondary, &self.secondary_means),
        );
        let provisional = learn_relationships(
            frame,
            &primary_residuals,
            &primary_observed,
            self.config.top_k,
            self.config.correlation_floor,
            None,
        )?;
        self.neural_embeddings = if self.config.neural_epochs == 0 {
            Vec::new()
        } else {
            fit_graph_kernel(
                frame,
                &self.primary_means,
                &self.secondary_means,
                &provisional,
                self.config.neural_hidden_dim,
                self.config.neural_epochs,
                &self.config.backend,
            )?
        };
        self.relationships = learn_relationships(
            frame,
            &primary_residuals,
            &primary_observed,
            self.config.top_k,
            self.config.correlation_floor,
            (!self.neural_embeddings.is_empty()).then_some(&self.neural_embeddings),
        )?;
        self.joint_heads = Some(fit_joint_heads(
            frame,
            &log_primary,
            &primary_observed,
            &log_secondary,
            &self.primary_means,
            &self.secondary_means,
            &self.weekly_primary,
            &self.weekly_secondary,
            &self.primary_calendar_weights,
            &self.secondary_calendar_weights,
            &self.cross_target_couplings,
            &self.relationships,
            &self.neural_embeddings,
            &self.config,
        )?);
        self.expert_shift_calibration = fit_expert_shift_calibration(
            frame,
            &log_primary,
            &primary_observed,
            &self.primary_means,
            &self.primary_scales,
            &self.weekly_primary,
            &self.primary_calendar_weights,
            &self.mix_coefficients,
            &self.relationships,
            &self.config,
        );
        if self.relationships.len() != lanes {
            return Err(GeoStError::InvalidFrame(
                "failed to learn a relationship set for every lane".to_string(),
            ));
        }
        self.lane_ids = frame.lane_ids.clone();
        self.origin_ids = frame.origin_ids.clone();
        self.destination_ids = frame.destination_ids.clone();
        self.coordinates = frame.coordinates.clone();
        self.hierarchy_groups = frame.hierarchy_groups.clone();
        self.timestamps = frame.timestamps.clone();
        self.frequency = frame.frequency.clone();
        self.target_names = frame.target_names.clone();
        self.primary_history = log_primary;
        self.primary_observed = primary_observed;
        self.secondary_history = log_secondary;
        self.calendar_width = frame.calendar.first().map_or(0, Vec::len);
        self.last_calendar = frame.calendar.last().cloned().unwrap_or_default();
        self.last_mix = frame
            .mix
            .as_ref()
            .and_then(|rows| rows.last())
            .map(|lanes| lanes.iter().map(|features| features[0]).collect());
        self.expert_labels = frame.expert_labels.clone();
        Ok(())
    }

    pub fn predict(
        &self,
        horizon: usize,
        future_calendar: Option<&[Vec<f64>]>,
    ) -> Result<Vec<MarketPrediction>> {
        if self.lane_ids.is_empty() {
            return Err(GeoStError::NotFit);
        }
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "forecast horizon must be positive".to_string(),
            ));
        }
        if let Some(calendar) = future_calendar {
            if calendar.len() < horizon
                || calendar.iter().take(horizon).any(|row| {
                    row.len() != self.calendar_width || row.iter().any(|x| !x.is_finite())
                })
            {
                return Err(GeoStError::InvalidFrame(
                    "future calendar must provide finite features for every forecast step"
                        .to_string(),
                ));
            }
        } else if self.calendar_width > 0 {
            return Err(GeoStError::InvalidFrame(
                "future calendar is required because calendar features were fitted".to_string(),
            ));
        }
        let mut primary = last_observed_by_lane(
            &self.primary_history,
            &self.primary_observed,
            &self.primary_means,
        );
        let mut secondary = self
            .secondary_history
            .last()
            .cloned()
            .ok_or(GeoStError::NotFit)?;
        let last_timestamp = *self.timestamps.last().ok_or(GeoStError::NotFit)?;
        let head_backend = select_backend_for(Some(&self.config.backend), BackendOperation::Dense)
            .map_err(|error| {
                GeoStError::InvalidFrame(format!("invalid market head backend: {error}"))
            })?;
        let mut result = Vec::with_capacity(horizon * self.lane_ids.len());
        for step in 1..=horizon {
            let timestamp = last_timestamp + step as i64;
            let mut next_primary = vec![0.0; self.lane_ids.len()];
            let mut next_secondary = vec![0.0; self.lane_ids.len()];
            let calendar = future_calendar
                .map(|rows| rows[step - 1].as_slice())
                .unwrap_or(&[]);
            let head_features = self.joint_heads.as_ref().map(|_| {
                (0..self.lane_ids.len())
                    .map(|lane| {
                        forecast_head_features(
                            lane,
                            &primary,
                            &secondary,
                            &self.primary_means,
                            &self.secondary_means,
                            &self.relationships,
                            &self.lane_ids,
                            timestamp,
                            calendar,
                            self.last_mix.as_deref(),
                            &self.neural_embeddings,
                        )
                    })
                    .collect::<Vec<_>>()
            });
            let accelerated_heads = match (&self.joint_heads, &head_features) {
                (Some(heads), Some(features))
                    if head_backend.selected != "cpu"
                        && market_head_dense_ops(heads, features)
                            >= MARKET_HEAD_DENSE_DISPATCH_MIN_OPS =>
                {
                    Some(joint_head_outputs_with_backend(
                        heads,
                        features,
                        &head_backend,
                    )?)
                }
                _ => None,
            };
            for lane in 0..self.lane_ids.len() {
                let seasonal_primary =
                    self.weekly_primary[(timestamp.rem_euclid(7)) as usize][lane];
                let seasonal_secondary =
                    self.weekly_secondary[(timestamp.rem_euclid(7)) as usize][lane];
                let peer_primary = peer_value(
                    lane,
                    &primary,
                    &self.primary_means,
                    &self.relationships,
                    &self.lane_ids,
                    timestamp,
                );
                let peer_secondary = peer_value(
                    lane,
                    &secondary,
                    &self.secondary_means,
                    &self.relationships,
                    &self.lane_ids,
                    timestamp,
                );
                next_primary[lane] = self.primary_means[lane]
                    + seasonal_primary
                    + self.config.local_strength * (primary[lane] - self.primary_means[lane])
                    + self.config.graph_strength * peer_primary
                    + calendar_effect(&self.primary_calendar_weights, calendar);
                next_secondary[lane] = self.secondary_means[lane]
                    + seasonal_secondary
                    + self.config.local_strength * (secondary[lane] - self.secondary_means[lane])
                    + self.config.graph_strength * peer_secondary
                    + calendar_effect(&self.secondary_calendar_weights, calendar);
                // Preserve the primary smoother as the benchmark path. The
                // supporting target consumes its co-movement with the primary
                // target, but cannot push the primary benchmark away from its
                // direct graph and temporal evidence.
                next_secondary[lane] +=
                    self.cross_target_couplings[lane] * (primary[lane] - self.primary_means[lane]);
                if let Some(heads) = &self.joint_heads {
                    let features = &head_features.as_ref().expect("joint head features")[lane];
                    // The robust adapters predict residual corrections to the
                    // decomposed graph path, never a competing absolute level.
                    if let Some(outputs) = &accelerated_heads {
                        next_primary[lane] += f64::from(outputs[lane][0]);
                        next_secondary[lane] += f64::from(outputs[lane][1]);
                    } else {
                        next_primary[lane] += dot(&heads.primary_huber, features);
                        next_secondary[lane] += dot(&heads.secondary_huber, features);
                    }
                }
                let primary_value = next_primary[lane].exp();
                let spread =
                    self.primary_interval_radii[lane] * self.interval_calibration_multiplier;
                let (lower, upper) = self.joint_heads.as_ref().map_or_else(
                    || (next_primary[lane] - spread, next_primary[lane] + spread),
                    |heads| {
                        let features = &head_features.as_ref().expect("joint head features")[lane];
                        let values = if let Some(outputs) = &accelerated_heads {
                            outputs[lane][2..]
                                .iter()
                                .map(|value| next_primary[lane] + f64::from(*value))
                                .collect::<Vec<_>>()
                        } else {
                            heads
                                .primary_quantiles
                                .iter()
                                .map(|head| next_primary[lane] + dot(head, features))
                                .collect::<Vec<_>>()
                        };
                        quantile_interval(
                            &self.config.quantile_levels,
                            &values,
                            next_primary[lane],
                            spread,
                        )
                    },
                );
                result.push(MarketPrediction {
                    lane_id: self.lane_ids[lane].clone(),
                    timestamp,
                    horizon: step,
                    primary: primary_value,
                    primary_lower: lower.exp(),
                    primary_upper: upper.exp(),
                    secondary: (next_secondary[lane].exp() - 1.0).max(0.0),
                });
            }
            primary = next_primary;
            secondary = next_secondary;
        }
        Ok(result)
    }

    pub fn nowcast(&self) -> Result<Vec<MarketExplanation>> {
        if self.lane_ids.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let primary = last_observed_by_lane(
            &self.primary_history,
            &self.primary_observed,
            &self.primary_means,
        );
        // Remove the fitted lane-local mix contribution before passing a lane
        // state through the market graph. A known composition event therefore
        // remains local rather than becoming evidence for a neighbor alert.
        let primary_without_mix = primary
            .iter()
            .enumerate()
            .map(|(lane, value)| {
                *value
                    - self
                        .last_mix
                        .as_ref()
                        .map_or(0.0, |mix| self.mix_coefficients[lane] * mix[lane])
            })
            .collect::<Vec<_>>();
        let active_mix = self
            .last_mix
            .as_ref()
            .is_some_and(|mix| mix.iter().any(|value| value.abs() > 1e-12));
        let mut rows = Vec::with_capacity(self.lane_ids.len());
        for lane in 0..self.lane_ids.len() {
            let observed_index = last_observed_index(&self.primary_observed, lane);
            let timestamp = observed_index
                .map(|index| self.timestamps[index])
                .or_else(|| self.timestamps.last().copied())
                .ok_or(GeoStError::NotFit)?;
            let seasonal = self.weekly_primary[(timestamp.rem_euclid(7)) as usize][lane];
            let calendar = calendar_effect(&self.primary_calendar_weights, &self.last_calendar);
            let peer = peer_value(
                lane,
                &primary_without_mix,
                &self.primary_means,
                &self.relationships,
                &self.lane_ids,
                timestamp,
            );
            let market_log =
                self.primary_means[lane] + seasonal + calendar + self.config.graph_strength * peer;
            let observed = primary[lane];
            let local = observed - market_log;
            let mix_component = self
                .last_mix
                .as_ref()
                .map_or(0.0, |mix| self.mix_coefficients[lane] * mix[lane]);
            let unexplained_local = local - mix_component;
            let scale = self.primary_scales[lane].max(1e-8);
            let observed_delta =
                last_observed_delta(&self.primary_history, &self.primary_observed, lane);
            let heuristic_shift = if observed_index.is_none()
                || observed_index.is_some_and(|index| index + 1 != self.primary_history.len())
                || observed_delta.abs() / scale < 1.0
            {
                // A connected lane can move while this lane does not. Avoid
                // turning that relationship into a red-herring local alert.
                MarketShiftKind::NoShift
            } else if peer.abs() / scale >= self.config.shift_zscore
                && unexplained_local.abs() / scale < self.config.shift_zscore
                // A current caller-supplied mix event contaminates the graph
                // snapshot. Defer market classification until a clean cutoff
                // rather than propagating that local evidence to neighbors.
                && !active_mix
            {
                MarketShiftKind::Market
            } else if local.abs() / scale >= self.config.shift_zscore
                || mix_component.abs() / scale >= self.config.shift_zscore
            {
                MarketShiftKind::LocalOrMix
            } else {
                MarketShiftKind::NoShift
            };
            let shift = self
                .expert_shift_calibration
                .as_ref()
                .and_then(|calibration| {
                    calibrated_shift(
                        calibration,
                        [
                            peer.abs() / scale,
                            local.abs() / scale,
                            mix_component.abs() / scale,
                        ],
                    )
                })
                .unwrap_or(heuristic_shift);
            let label = self
                .expert_labels
                .iter()
                .find(|label| label.lane_id == self.lane_ids[lane] && label.timestamp == timestamp)
                .cloned();
            rows.push(MarketExplanation {
                lane_id: self.lane_ids[lane].clone(),
                timestamp,
                observed_primary: observed_index.map(|_| observed.exp()),
                support: if observed_index.is_some() {
                    MarketSupportKind::Lane
                } else {
                    MarketSupportKind::Hierarchy
                },
                smoothed_primary: market_log.exp(),
                market_component: (self.primary_means[lane] + seasonal + calendar).exp(),
                local_or_mix_component: local,
                seasonal_component: seasonal + calendar,
                residual_component: unexplained_local,
                uncertainty: scale,
                shift,
                top_relationships: self.relationships[lane].clone(),
                expert_label: label,
            });
        }
        Ok(rows)
    }

    pub fn weekly_rollups(
        &self,
        horizon: usize,
        future_calendar: Option<&[Vec<f64>]>,
    ) -> Result<Vec<WeeklyMarketPrediction>> {
        let daily = self.predict(horizon, future_calendar)?;
        let mut grouped = BTreeMap::<(String, i64), Vec<MarketPrediction>>::new();
        for row in daily {
            let week_start_timestamp = row.timestamp - row.timestamp.rem_euclid(7);
            grouped
                .entry((row.lane_id.clone(), week_start_timestamp))
                .or_default()
                .push(row);
        }
        Ok(grouped
            .into_iter()
            .map(|((lane_id, week_start_timestamp), rows)| {
                let days = rows.len();
                WeeklyMarketPrediction {
                    lane_id,
                    week_start_timestamp,
                    days,
                    primary: rows.iter().map(|row| row.primary).sum::<f64>() / days as f64,
                    primary_lower: rows.iter().map(|row| row.primary_lower).sum::<f64>()
                        / days as f64,
                    primary_upper: rows.iter().map(|row| row.primary_upper).sum::<f64>()
                        / days as f64,
                    secondary: rows.iter().map(|row| row.secondary).sum(),
                }
            })
            .collect())
    }

    pub fn relationships(&self) -> Result<Vec<MarketRelationship>> {
        if self.lane_ids.is_empty() {
            Err(GeoStError::NotFit)
        } else {
            Ok(self.relationships.iter().flatten().cloned().collect())
        }
    }

    /// A portable analyst payload for Python notebooks and browser/WASM views.
    /// It intentionally exposes model evidence instead of rendering policy.
    pub fn explorer_payload(&self, horizon: usize) -> Result<serde_json::Value> {
        if self.lane_ids.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let lanes = self
            .lane_ids
            .iter()
            .enumerate()
            .map(|(index, lane_id)| {
                let coordinate = self.coordinates.get(index).copied().unwrap_or([0.0; 4]);
                json!({
                    "lane_id": lane_id,
                    "origin_id": self.origin_ids.get(index),
                    "destination_id": self.destination_ids.get(index),
                    "origin_x": coordinate[0],
                    "origin_y": coordinate[1],
                    "destination_x": coordinate[2],
                    "destination_y": coordinate[3],
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "lanes": lanes,
            // Explorer callers request an immediately inspectable current-state
            // view. Reuse the final known calendar state here; explicit
            // `predict` calls still require caller-supplied future calendar
            // values when calendar features were fitted.
            "forecasts": self.predict(
                horizon,
                (!self.last_calendar.is_empty()).then(|| vec![self.last_calendar.clone(); horizon]).as_deref(),
            )?,
            "explanations": self.nowcast()?,
            "kernels": self.relationships()?,
            "target_names": self.target_names,
        }))
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_json_string()?).map_err(GeoStError::from)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_string(&fs::read_to_string(path).map_err(GeoStError::from)?)
    }
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(GeoStError::from)
    }
    pub fn from_json_string(value: &str) -> Result<Self> {
        serde_json::from_str(value).map_err(GeoStError::from)
    }
}

fn learn_relationships(
    frame: &MarketPanelFrame,
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    top_k: usize,
    floor: f64,
    neural_embeddings: Option<&[Vec<f32>]>,
) -> Result<Vec<Vec<MarketRelationship>>> {
    let mut index = BTreeMap::new();
    for (idx, lane) in frame.lane_ids.iter().enumerate() {
        index.insert(lane.as_str(), idx);
    }
    // A market panel may contain every observed origin-to-destination lane.
    // Comparing each lane with every other lane is quadratic and makes a
    // 40K+ lane city graph unusable.  Build the candidate topology from the
    // endpoint relationships that define this model instead: shared origin,
    // shared destination, reverse lanes, and caller-supplied expert edges.
    // Residual correlation and the learned kernel still rank those candidates
    // using train-only values; they are not used to manufacture a dense graph.
    let mut origins = BTreeMap::<&str, Vec<usize>>::new();
    let mut destinations = BTreeMap::<&str, Vec<usize>>::new();
    let mut endpoint_pairs = BTreeMap::<(&str, &str), Vec<usize>>::new();
    for lane in 0..frame.lane_ids.len() {
        origins
            .entry(frame.origin_ids[lane].as_str())
            .or_default()
            .push(lane);
        destinations
            .entry(frame.destination_ids[lane].as_str())
            .or_default()
            .push(lane);
        endpoint_pairs
            .entry((
                frame.origin_ids[lane].as_str(),
                frame.destination_ids[lane].as_str(),
            ))
            .or_default()
            .push(lane);
    }
    let mut priors = BTreeMap::<(usize, usize), &ExpertRelationshipPrior>::new();
    for prior in &frame.expert_priors {
        priors.insert(
            (
                *index
                    .get(prior.source_lane_id.as_str())
                    .ok_or_else(|| GeoStError::InvalidFrame("unknown expert source".to_string()))?,
                *index
                    .get(prior.target_lane_id.as_str())
                    .ok_or_else(|| GeoStError::InvalidFrame("unknown expert target".to_string()))?,
            ),
            prior,
        );
    }
    Ok((0..frame.lane_ids.len())
        .into_par_iter()
        .map(|source| {
            // Build and release one lane's candidate list at a time. Retaining a
            // set for every lane would itself duplicate a full city graph.
            let mut candidate_indices = Vec::new();
            if let Some(rows) = origins.get(frame.origin_ids[source].as_str()) {
                candidate_indices.extend(rows.iter().copied());
            }
            if let Some(rows) = destinations.get(frame.destination_ids[source].as_str()) {
                candidate_indices.extend(rows.iter().copied());
            }
            if let Some(rows) = endpoint_pairs.get(&(
                frame.destination_ids[source].as_str(),
                frame.origin_ids[source].as_str(),
            )) {
                candidate_indices.extend(rows.iter().copied());
            }
            candidate_indices.extend(
                priors.keys().filter_map(|&(prior_source, target)| {
                    (prior_source == source).then_some(target)
                }),
            );
            candidate_indices.sort_unstable();
            candidate_indices.dedup();
            let mut candidates = Vec::new();
            for target in candidate_indices {
                if source == target {
                    continue;
                }
                if let Some(prior) = priors.get(&(source, target)) {
                    if !prior.allowed {
                        continue;
                    }
                }
                let mut kinds = Vec::new();
                let mut score = 0.0;
                if frame.origin_ids[source] == frame.origin_ids[target] {
                    kinds.push(RelationshipKind::SharedOrigin);
                    score += 0.35;
                }
                if frame.destination_ids[source] == frame.destination_ids[target] {
                    kinds.push(RelationshipKind::SharedDestination);
                    score += 0.35;
                }
                if frame.origin_ids[source] == frame.destination_ids[target]
                    && frame.destination_ids[source] == frame.origin_ids[target]
                {
                    kinds.push(RelationshipKind::ReverseLane);
                    score += 0.45;
                }
                let distance =
                    endpoint_distance(frame.coordinates[source], frame.coordinates[target]);
                if distance < 2.0 {
                    kinds.push(RelationshipKind::Geographic);
                    score += 0.2 / (1.0 + distance);
                }
                let correlation = masked_correlation(residuals, observed, source, target);
                if correlation >= floor {
                    kinds.push(RelationshipKind::ResidualCorrelation);
                    score += 0.5 * correlation;
                }
                if let Some(embeddings) = neural_embeddings {
                    let similarity = cosine_similarity(&embeddings[source], &embeddings[target]);
                    if similarity > 0.0 {
                        // The kernel is learned from the candidate graph and static lane state;
                        // residual evidence remains the task-specific selection signal.
                        score += 0.2 * similarity;
                        kinds.push(RelationshipKind::NeuralKernel);
                    }
                }
                if let Some(prior) = priors.get(&(source, target)) {
                    kinds.push(RelationshipKind::Expert);
                    score += prior.weight.max(0.01);
                }
                if !kinds.is_empty() && score > 0.0 {
                    candidates.push((target, score, kinds));
                }
            }
            candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| frame.lane_ids[a.0].cmp(&frame.lane_ids[b.0]))
            });
            candidates.truncate(top_k);
            let total: f64 = candidates.iter().map(|row| row.1).sum();
            candidates
                .into_iter()
                .map(|(target, score, kinds)| MarketRelationship {
                    source_lane_id: frame.lane_ids[source].clone(),
                    target_lane_id: frame.lane_ids[target].clone(),
                    weight: score / total.max(1e-12),
                    periodic_weights: periodic_edge_weights(
                        source,
                        target,
                        residuals,
                        observed,
                        &frame.timestamps,
                    ),
                    kinds,
                })
                .collect()
        })
        .collect())
}

fn fit_graph_kernel(
    frame: &MarketPanelFrame,
    primary_means: &[f64],
    secondary_means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    hidden_dim: usize,
    epochs: usize,
    backend: &str,
) -> Result<Vec<Vec<f32>>> {
    let features = lane_kernel_features(frame, primary_means, secondary_means);
    let lane_index = frame
        .lane_ids
        .iter()
        .enumerate()
        .map(|(index, lane_id)| (lane_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let edges = relationships
        .iter()
        .flat_map(|rows| rows.iter())
        .filter_map(|edge| {
            let source = *lane_index.get(edge.source_lane_id.as_str())?;
            let target = *lane_index.get(edge.target_lane_id.as_str())?;
            Some((source, target))
        })
        .collect::<Vec<_>>();
    if edges.is_empty() {
        return Ok(features);
    }
    let graph = HomogeneousGraph::from_directed_edges(frame.lane_ids.len(), &edges)
        .map_err(|err| GeoStError::InvalidFrame(format!("invalid learned market graph: {err}")))?;
    let config = GraphSageConfig {
        hidden_dims: vec![hidden_dim],
        epochs,
        backend: select_backend_for(Some(backend), BackendOperation::Dense).map_err(|err| {
            GeoStError::InvalidFrame(format!("invalid market graph backend: {err}"))
        })?,
        ..GraphSageConfig::default()
    };
    GraphSageEncoder::new(config, features[0].len())
        .map_err(|err| GeoStError::InvalidFrame(format!("invalid market graph kernel: {err}")))?
        .fit(&graph, &features)
        .map(|embedding| embedding.into_inner())
        .map_err(|err| {
            GeoStError::InvalidFrame(format!("market graph kernel fitting failed: {err}"))
        })
}

fn lane_kernel_features(
    frame: &MarketPanelFrame,
    primary_means: &[f64],
    secondary_means: &[f64],
) -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(frame.lane_ids.len());
    for (idx, point) in frame.coordinates.iter().enumerate() {
        rows.push(vec![
            ((point[0] + point[2]) * 0.5) as f32,
            ((point[1] + point[3]) * 0.5) as f32,
            (point[2] - point[0]) as f32,
            (point[3] - point[1]) as f32,
            primary_means[idx] as f32,
            secondary_means[idx] as f32,
        ]);
    }
    standardize_kernel_features(&mut rows);
    rows
}

fn standardize_kernel_features(rows: &mut [Vec<f32>]) {
    for col in 0..rows[0].len() {
        let mean = rows.iter().map(|row| row[col] as f64).sum::<f64>() / rows.len() as f64;
        let scale = (rows
            .iter()
            .map(|row| (row[col] as f64 - mean).powi(2))
            .sum::<f64>()
            / rows.len() as f64)
            .sqrt()
            .max(1e-6);
        for row in rows.iter_mut() {
            row[col] = ((row[col] as f64 - mean) / scale) as f32;
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &b) in left.iter().zip(right) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm <= 1e-12 || right_norm <= 1e-12 {
        0.0
    } else {
        dot / (left_norm * right_norm).sqrt()
    }
}

/// Fit point and distributional heads together over frozen GraphSAGE lane
/// embeddings.  Freezing the encoder is intentional: relationships are
/// learned from the complete train-only panel first, then these heads consume
/// only one-step-ahead states available at each historical cutoff.  This is a
/// compact adapter design rather than a post-hoc residual quantile adjustment.
#[allow(clippy::too_many_arguments)]
fn fit_joint_heads(
    frame: &MarketPanelFrame,
    primary: &[Vec<f64>],
    observed: &[Vec<bool>],
    secondary: &[Vec<f64>],
    primary_means: &[f64],
    secondary_means: &[f64],
    weekly_primary: &[Vec<f64>],
    weekly_secondary: &[Vec<f64>],
    primary_calendar_weights: &[f64],
    secondary_calendar_weights: &[f64],
    cross_target_couplings: &[f64],
    relationships: &[Vec<MarketRelationship>],
    embeddings: &[Vec<f32>],
    config: &MarketStructureConfig,
) -> Result<JointMarketHeads> {
    let has_samples = (1..primary.len()).any(|time| {
        (0..frame.lane_ids.len()).any(|lane| observed[time][lane] && observed[time - 1][lane])
    });
    if !has_samples {
        return Err(GeoStError::InvalidFrame(
            "market joint heads require at least one observed one-step training target".to_string(),
        ));
    }
    let width =
        embeddings.first().map_or(0, Vec::len) + 6 + frame.calendar.first().map_or(0, Vec::len);
    let mut heads = JointMarketHeads {
        primary_huber: vec![0.0; width],
        secondary_huber: vec![0.0; width],
        primary_quantiles: vec![vec![0.0; width]; config.quantile_levels.len()],
        secondary_quantiles: vec![vec![0.0; width]; config.quantile_levels.len()],
    };
    // Stream examples through every epoch. Materializing a feature vector for
    // every lane and timestamp duplicates a full city panel in memory and
    // prevents all-observed-lane fits. The updates are identical to the
    // former sample-cache loop, while peak memory stays bounded.
    for _ in 0..config.head_epochs {
        for time in 1..primary.len() {
            let calendar = frame.calendar[time].as_slice();
            let mix = frame.mix.as_ref().map(|rows| {
                rows[time]
                    .iter()
                    .map(|features| features[0])
                    .collect::<Vec<_>>()
            });
            for lane in 0..frame.lane_ids.len() {
                if !observed[time][lane] || !observed[time - 1][lane] {
                    continue;
                }
                let features = head_features(
                    lane,
                    &primary[time - 1],
                    &secondary[time - 1],
                    primary_means,
                    secondary_means,
                    relationships,
                    &frame.lane_ids,
                    frame.timestamps[time],
                    calendar,
                    mix.as_deref(),
                    embeddings,
                );
                let timestamp = frame.timestamps[time];
                let primary_peer = peer_value(
                    lane,
                    &primary[time - 1],
                    primary_means,
                    relationships,
                    &frame.lane_ids,
                    timestamp,
                );
                let secondary_peer = peer_value(
                    lane,
                    &secondary[time - 1],
                    secondary_means,
                    relationships,
                    &frame.lane_ids,
                    timestamp,
                );
                let primary_base = primary_means[lane]
                    + weekly_primary[timestamp.rem_euclid(7) as usize][lane]
                    + config.local_strength * (primary[time - 1][lane] - primary_means[lane])
                    + config.graph_strength * primary_peer
                    + calendar_effect(primary_calendar_weights, calendar);
                let secondary_base = secondary_means[lane]
                    + weekly_secondary[timestamp.rem_euclid(7) as usize][lane]
                    + config.local_strength * (secondary[time - 1][lane] - secondary_means[lane])
                    + config.graph_strength * secondary_peer
                    + calendar_effect(secondary_calendar_weights, calendar)
                    + cross_target_couplings[lane]
                        * (primary[time - 1][lane] - primary_means[lane]);
                let primary_target = primary[time][lane] - primary_base;
                let secondary_target = secondary[time][lane] - secondary_base;
                huber_step(&mut heads.primary_huber, &features, primary_target, config);
                huber_step(
                    &mut heads.secondary_huber,
                    &features,
                    secondary_target,
                    config,
                );
                for (idx, level) in config.quantile_levels.iter().enumerate() {
                    pinball_step(
                        &mut heads.primary_quantiles[idx],
                        &features,
                        primary_target,
                        *level,
                        config,
                    );
                    pinball_step(
                        &mut heads.secondary_quantiles[idx],
                        &features,
                        secondary_target,
                        *level,
                        config,
                    );
                }
            }
        }
    }
    Ok(heads)
}

fn huber_step(head: &mut [f64], features: &[f64], target: f64, config: &MarketStructureConfig) {
    let residual = dot(head, features) - target;
    let gradient = residual.clamp(-config.huber_delta, config.huber_delta);
    for (weight, feature) in head.iter_mut().zip(features) {
        *weight -= config.head_learning_rate * (gradient * feature + 1e-5 * weight.signum());
    }
}

fn pinball_step(
    head: &mut [f64],
    features: &[f64],
    target: f64,
    level: f64,
    config: &MarketStructureConfig,
) {
    let prediction = dot(head, features);
    let gradient = if target >= prediction {
        -level
    } else {
        1.0 - level
    };
    for (weight, feature) in head.iter_mut().zip(features) {
        *weight -= config.head_learning_rate * (gradient * feature + 1e-5 * weight.signum());
    }
}

fn dot(weights: &[f64], features: &[f64]) -> f64 {
    weights
        .iter()
        .zip(features)
        .map(|(weight, feature)| weight * feature)
        .sum()
}

fn joint_head_outputs_with_backend(
    heads: &JointMarketHeads,
    features: &[Vec<f64>],
    backend: &BackendSelection,
) -> Result<Vec<Vec<f32>>> {
    let output_width = 2 + heads.primary_quantiles.len();
    let feature_width = heads.primary_huber.len();
    let mut weights = Vec::with_capacity(feature_width * output_width);
    for feature in 0..feature_width {
        weights.push(heads.primary_huber[feature] as f32);
        weights.push(heads.secondary_huber[feature] as f32);
        weights.extend(
            heads
                .primary_quantiles
                .iter()
                .map(|head| head[feature] as f32),
        );
    }
    let features = features
        .iter()
        .map(|row| row.iter().map(|value| *value as f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    backend_dense_layer_f32(backend, &features, &weights, &vec![0.0; output_width])
        .map_err(|error| GeoStError::InvalidFrame(format!("market head dispatch failed: {error}")))
}

fn market_head_dense_ops(heads: &JointMarketHeads, features: &[Vec<f64>]) -> usize {
    features
        .len()
        .saturating_mul(heads.primary_huber.len())
        .saturating_mul(2 + heads.primary_quantiles.len())
}

#[allow(clippy::too_many_arguments)]
fn head_features(
    lane: usize,
    primary: &[f64],
    secondary: &[f64],
    primary_means: &[f64],
    secondary_means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    lane_ids: &[String],
    timestamp: i64,
    calendar: &[f64],
    mix: Option<&[f64]>,
    embeddings: &[Vec<f32>],
) -> Vec<f64> {
    let mut values = vec![
        1.0,
        primary[lane] - primary_means[lane],
        peer_value(
            lane,
            primary,
            primary_means,
            relationships,
            lane_ids,
            timestamp,
        ),
        secondary[lane] - secondary_means[lane],
        peer_value(
            lane,
            secondary,
            secondary_means,
            relationships,
            lane_ids,
            timestamp,
        ),
        mix.map_or(0.0, |rows| rows[lane]),
    ];
    values.extend_from_slice(calendar);
    values.extend(
        embeddings
            .get(lane)
            .into_iter()
            .flatten()
            .map(|value| *value as f64),
    );
    values
}

#[allow(clippy::too_many_arguments)]
fn forecast_head_features(
    lane: usize,
    primary: &[f64],
    secondary: &[f64],
    primary_means: &[f64],
    secondary_means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    lane_ids: &[String],
    timestamp: i64,
    calendar: &[f64],
    mix: Option<&[f64]>,
    embeddings: &[Vec<f32>],
) -> Vec<f64> {
    head_features(
        lane,
        primary,
        secondary,
        primary_means,
        secondary_means,
        relationships,
        lane_ids,
        timestamp,
        calendar,
        mix,
        embeddings,
    )
}

fn quantile_interval(
    levels: &[f64],
    values: &[f64],
    point: f64,
    fallback_spread: f64,
) -> (f64, f64) {
    let lower = levels
        .iter()
        .position(|level| *level <= 0.1)
        .map(|idx| values[idx]);
    let upper = levels
        .iter()
        .rposition(|level| *level >= 0.9)
        .map(|idx| values[idx]);
    // Quantile heads determine asymmetric tails. The train-only residual
    // radius remains a calibration floor so a narrow fitted head cannot make
    // the advertised interval spuriously overconfident on a new cutoff.
    let lower = lower
        .unwrap_or(point - fallback_spread)
        .min(point - fallback_spread);
    let upper = upper
        .unwrap_or(point + fallback_spread)
        .max(point + fallback_spread);
    (lower, upper)
}

#[allow(clippy::too_many_arguments)]
fn fit_expert_shift_calibration(
    frame: &MarketPanelFrame,
    primary: &[Vec<f64>],
    observed: &[Vec<bool>],
    means: &[f64],
    scales: &[f64],
    weekly: &[Vec<f64>],
    calendar_weights: &[f64],
    mix_coefficients: &[f64],
    relationships: &[Vec<MarketRelationship>],
    config: &MarketStructureConfig,
) -> Option<ExpertShiftCalibration> {
    let mut groups = [Vec::<[f64; 3]>::new(), Vec::new(), Vec::new()];
    for label in &frame.expert_labels {
        let lane = frame.lane_ids.iter().position(|id| id == &label.lane_id)?;
        let time = frame
            .timestamps
            .iter()
            .position(|time| *time == label.timestamp)?;
        if !observed[time][lane] {
            continue;
        }
        let timestamp = frame.timestamps[time];
        let peer = peer_value(
            lane,
            &primary[time],
            means,
            relationships,
            &frame.lane_ids,
            timestamp,
        );
        let market = means[lane]
            + weekly[(timestamp.rem_euclid(7)) as usize][lane]
            + calendar_effect(calendar_weights, &frame.calendar[time])
            + config.graph_strength * peer;
        let local = primary[time][lane] - market;
        let mix = frame
            .mix
            .as_ref()
            .map_or(0.0, |rows| mix_coefficients[lane] * rows[time][lane][0]);
        let scale = scales[lane].max(1e-8);
        let metrics = [peer.abs() / scale, local.abs() / scale, mix.abs() / scale];
        let group = match label.shift {
            MarketShiftKind::Market => 0,
            MarketShiftKind::LocalOrMix => 1,
            MarketShiftKind::NoShift => 2,
        };
        groups[group].push(metrics);
    }
    let centroids = groups.map(|values| {
        (values.len() >= 2).then(|| {
            let mut centroid = [0.0; 3];
            for values in &values {
                for (idx, value) in values.iter().enumerate() {
                    centroid[idx] += value;
                }
            }
            for value in &mut centroid {
                *value /= values.len() as f64;
            }
            centroid
        })
    });
    let calibration = ExpertShiftCalibration {
        market: centroids[0],
        local_or_mix: centroids[1],
        no_shift: centroids[2],
    };
    let trained_classes = [
        calibration.market.is_some(),
        calibration.local_or_mix.is_some(),
        calibration.no_shift.is_some(),
    ]
    .into_iter()
    .filter(|trained| *trained)
    .count();
    (trained_classes >= 2).then_some(calibration)
}

fn calibrated_shift(
    calibration: &ExpertShiftCalibration,
    metrics: [f64; 3],
) -> Option<MarketShiftKind> {
    let mut candidates = Vec::new();
    if let Some(centroid) = calibration.market {
        candidates.push((MarketShiftKind::Market, squared_distance(metrics, centroid)));
    }
    if let Some(centroid) = calibration.local_or_mix {
        candidates.push((
            MarketShiftKind::LocalOrMix,
            squared_distance(metrics, centroid),
        ));
    }
    if let Some(centroid) = calibration.no_shift {
        candidates.push((
            MarketShiftKind::NoShift,
            squared_distance(metrics, centroid),
        ));
    }
    candidates
        .into_iter()
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(shift, _)| shift)
}

fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

fn peer_value(
    lane: usize,
    values: &[f64],
    means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    lane_ids: &[String],
    timestamp: i64,
) -> f64 {
    let mut total = 0.0;
    for edge in &relationships[lane] {
        if let Some(target) = lane_ids.iter().position(|id| id == &edge.target_lane_id) {
            let period = timestamp.rem_euclid(7) as usize;
            let periodic = edge.periodic_weights.get(period).copied().unwrap_or(1.0);
            total += edge.weight * periodic * (values[target] - means[target]);
        }
    }
    total
}

fn periodic_edge_weights(
    source: usize,
    target: usize,
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    timestamps: &[i64],
) -> Vec<f64> {
    let overall = masked_correlation(residuals, observed, source, target).max(0.0);
    (0..7)
        .map(|period| {
            let (left, right): (Vec<_>, Vec<_>) = residuals
                .iter()
                .zip(observed)
                .zip(timestamps)
                .filter_map(|((row, mask), time)| {
                    (mask[source] && mask[target] && time.rem_euclid(7) as usize == period)
                        .then_some((row[source], row[target]))
                })
                .unzip();
            let local = if left.len() >= 3 {
                correlation(&left, &right).max(0.0)
            } else {
                overall
            };
            // Shrink periodic estimates toward the full-history estimate to reject short-lived look-alikes.
            (0.5 + 0.5 * (0.7 * overall + 0.3 * local)).clamp(0.25, 1.0)
        })
        .collect()
}

fn endpoint_distance(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let ex = a[2] - b[2];
    let ey = a[3] - b[3];
    ((dx * dx + dy * dy + ex * ex + ey * ey) / 2.0).sqrt()
}
fn matrix_map(input: &[Vec<f64>], f: impl Fn(f64) -> f64) -> Vec<Vec<f64>> {
    input
        .iter()
        .map(|row| row.iter().copied().map(&f).collect())
        .collect()
}

fn log_primary_with_missing(values: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<bool>>) {
    let observed = values
        .iter()
        .map(|row| row.iter().map(|value| !value.is_nan()).collect())
        .collect::<Vec<Vec<bool>>>();
    let logged = values
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| if value.is_nan() { 0.0 } else { value.ln() })
                .collect()
        })
        .collect();
    (logged, observed)
}

/// Estimate lane means with explicit partial pooling through caller-provided
/// parent groups. Group order is significant: callers should put the most
/// specific stable parent first. A lane with no observations must resolve to a
/// parent with observations; it is never silently filled from a global mean.
fn primary_means_with_hierarchy(
    frame: &MarketPanelFrame,
    values: &[Vec<f64>],
    observed: &[Vec<bool>],
) -> Result<Vec<f64>> {
    let mut groups = BTreeMap::<&str, Vec<f64>>::new();
    for (row, mask) in values.iter().zip(observed) {
        for (lane, value) in row.iter().enumerate() {
            if mask[lane] {
                for group in &frame.hierarchy_groups[lane] {
                    groups.entry(group.as_str()).or_default().push(*value);
                }
            }
        }
    }
    (0..frame.lane_ids.len())
        .map(|lane| {
            let own = values
                .iter()
                .zip(observed)
                .filter_map(|(row, mask)| mask[lane].then_some(row[lane]))
                .collect::<Vec<_>>();
            let parent = frame.hierarchy_groups[lane]
                .iter()
                .find_map(|group| groups.get(group.as_str()))
                .filter(|values| !values.is_empty())
                .map(|values| values.iter().sum::<f64>() / values.len() as f64);
            match (own.is_empty(), parent) {
                (false, Some(parent_mean)) => {
                    // Eight pseudo-observations makes the parent a stabilizing
                    // prior for intermittent lanes without erasing local data.
                    let own_mean = own.iter().sum::<f64>() / own.len() as f64;
                    Ok(
                        (own_mean * own.len() as f64 + parent_mean * 8.0)
                            / (own.len() as f64 + 8.0),
                    )
                }
                (false, None) => Ok(own.iter().sum::<f64>() / own.len() as f64),
                (true, Some(parent_mean)) => Ok(parent_mean),
                (true, None) => Err(GeoStError::InvalidFrame(format!(
                    "lane '{}' has no observed primary values in any supplied hierarchy group",
                    frame.lane_ids[lane]
                ))),
            }
        })
        .collect()
}

fn primary_scales_with_hierarchy(
    frame: &MarketPanelFrame,
    values: &[Vec<f64>],
    observed: &[Vec<bool>],
    means: &[f64],
) -> Result<Vec<f64>> {
    let mut group_values = BTreeMap::<&str, Vec<f64>>::new();
    for (row, mask) in values.iter().zip(observed) {
        for (lane, value) in row.iter().enumerate() {
            if mask[lane] {
                for group in &frame.hierarchy_groups[lane] {
                    group_values.entry(group.as_str()).or_default().push(*value);
                }
            }
        }
    }
    (0..frame.lane_ids.len())
        .map(|lane| {
            let own = values
                .iter()
                .zip(observed)
                .filter_map(|(row, mask)| mask[lane].then_some(row[lane] - means[lane]))
                .collect::<Vec<_>>();
            if !own.is_empty() {
                return Ok(rms_scale(&own));
            }
            let parent_values = frame.hierarchy_groups[lane]
                .iter()
                .find_map(|group| group_values.get(group.as_str()))
                .ok_or_else(|| {
                    GeoStError::InvalidFrame(format!(
                        "lane '{}' has no hierarchy observations for uncertainty",
                        frame.lane_ids[lane]
                    ))
                })?;
            let parent_mean = parent_values.iter().sum::<f64>() / parent_values.len() as f64;
            Ok(rms_scale(
                &parent_values
                    .iter()
                    .map(|value| value - parent_mean)
                    .collect::<Vec<_>>(),
            ))
        })
        .collect()
}

fn rms_scale(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64)
        .sqrt()
        .max(1e-6)
}

fn primary_interval_radii(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    fallback_scales: &[f64],
) -> Vec<f64> {
    (0..residuals[0].len())
        .map(|lane| {
            let mut absolute = residuals
                .iter()
                .zip(observed)
                .filter_map(|(row, mask)| mask[lane].then_some(row[lane].abs()))
                .collect::<Vec<_>>();
            if absolute.is_empty() {
                return 1.28155 * fallback_scales[lane];
            }
            absolute.sort_by(|left, right| left.total_cmp(right));
            // Finite-sample empirical conformal rank for nominal 80% coverage.
            let rank = ((absolute.len() + 1) * 4).div_ceil(5).saturating_sub(1);
            absolute[rank.min(absolute.len() - 1)].max(1e-6)
        })
        .collect()
}

fn interval_calibration_multiplier(
    frame: &MarketPanelFrame,
    config: &MarketStructureConfig,
) -> Result<f64> {
    let calibration_horizon = frame.horizon.min(frame.timestamps.len() / 4).max(1);
    let minimum_prefix = calibration_horizon + 1;
    if frame.timestamps.len() <= minimum_prefix + calibration_horizon {
        return Ok(1.0);
    }
    let mut calibration_config = config.clone();
    calibration_config.calibrate_intervals = false;
    let mut ratios = Vec::new();
    let mut prefixes = (1..=3)
        .filter_map(|origin| {
            frame
                .timestamps
                .len()
                .checked_sub(origin * calibration_horizon)
        })
        .filter(|prefix| *prefix >= minimum_prefix)
        .collect::<Vec<_>>();
    prefixes.sort_unstable();
    prefixes.dedup();
    for prefix in prefixes {
        let mut calibration_frame = frame.clone();
        calibration_frame.timestamps.truncate(prefix);
        calibration_frame.primary.truncate(prefix);
        calibration_frame.secondary.truncate(prefix);
        calibration_frame.calendar.truncate(prefix);
        if let Some(mix) = &mut calibration_frame.mix {
            mix.truncate(prefix);
        }
        let calibration_timestamps = calibration_frame.timestamps.clone();
        calibration_frame
            .expert_labels
            .retain(|label| calibration_timestamps.contains(&label.timestamp));
        calibration_frame.horizon = calibration_horizon;
        calibration_frame.validate()?;
        let mut model = MarketStructureForecaster::new(calibration_config.clone())?;
        model.fit(&calibration_frame)?;
        let future_calendar = (!frame.calendar.is_empty())
            .then_some(&frame.calendar[prefix..prefix + calibration_horizon]);
        for prediction in model.predict(calibration_horizon, future_calendar)? {
            let lane = frame
                .lane_ids
                .iter()
                .position(|lane_id| lane_id == &prediction.lane_id)
                .ok_or(GeoStError::NotFit)?;
            let step = prediction.horizon - 1;
            let actual = frame.primary[prefix + step][lane];
            if !actual.is_nan() {
                let radius = model.primary_interval_radii[lane].max(1e-6);
                ratios.push((actual.ln() - prediction.primary.ln()).abs() / radius);
            }
        }
    }
    if ratios.len() < 8 {
        return Ok(1.0);
    }
    ratios.sort_by(|left, right| left.total_cmp(right));
    // Use a conservative 90th-percentile rolling-origin multiplier: the
    // underlying radius is 80%, while multi-step graph propagation adds tail
    // risk that is only visible in held-out origins.
    let rank = ((ratios.len() + 1) * 9).div_ceil(10).saturating_sub(1);
    Ok(ratios[rank.min(ratios.len() - 1)].max(1.0))
}

fn centered_masked(values: &[Vec<f64>], means: &[f64], observed: &[Vec<bool>]) -> Vec<Vec<f64>> {
    values
        .iter()
        .zip(observed)
        .map(|(row, mask)| {
            row.iter()
                .enumerate()
                .map(|(lane, value)| if mask[lane] { value - means[lane] } else { 0.0 })
                .collect()
        })
        .collect()
}

fn last_observed_index(observed: &[Vec<bool>], lane: usize) -> Option<usize> {
    observed.iter().rposition(|row| row[lane])
}

fn last_observed_by_lane(values: &[Vec<f64>], observed: &[Vec<bool>], prior: &[f64]) -> Vec<f64> {
    (0..values[0].len())
        .map(|lane| {
            last_observed_index(observed, lane).map_or(prior[lane], |index| values[index][lane])
        })
        .collect()
}

fn last_observed_delta(values: &[Vec<f64>], observed: &[Vec<bool>], lane: usize) -> f64 {
    let indices = observed
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row[lane].then_some(index))
        .collect::<Vec<_>>();
    match indices.as_slice() {
        [.., previous, current] => values[*current][lane] - values[*previous][lane],
        _ => 0.0,
    }
}
fn column_means(values: &[Vec<f64>]) -> Vec<f64> {
    (0..values[0].len())
        .map(|col| values.iter().map(|row| row[col]).sum::<f64>() / values.len() as f64)
        .collect()
}
fn centered(values: &[Vec<f64>], means: &[f64]) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| value - means[idx])
                .collect()
        })
        .collect()
}
fn column_scales(values: &[Vec<f64>]) -> Vec<f64> {
    (0..values[0].len())
        .map(|col| {
            (values.iter().map(|row| row[col] * row[col]).sum::<f64>() / values.len() as f64)
                .sqrt()
                .max(1e-6)
        })
        .collect()
}

fn calendar_weights(residuals: &[Vec<f64>], calendar: &[Vec<f64>]) -> Vec<f64> {
    let width = calendar.first().map_or(0, Vec::len);
    if width == 0 {
        return Vec::new();
    }
    let target = residuals
        .iter()
        .map(|row| row.iter().sum::<f64>() / row.len() as f64)
        .collect::<Vec<_>>();
    let target_mean = target.iter().sum::<f64>() / target.len() as f64;
    (0..width)
        .map(|feature| {
            let mean = calendar.iter().map(|row| row[feature]).sum::<f64>() / calendar.len() as f64;
            let (mut numerator, mut denominator) = (0.0, 0.0);
            for (row, &value) in calendar.iter().zip(&target) {
                let delta = row[feature] - mean;
                numerator += delta * (value - target_mean);
                denominator += delta * delta;
            }
            if denominator <= 1e-12 {
                0.0
            } else {
                (numerator / denominator).clamp(-0.25, 0.25)
            }
        })
        .collect()
}

fn calendar_weights_masked(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    calendar: &[Vec<f64>],
) -> Vec<f64> {
    let width = calendar.first().map_or(0, Vec::len);
    if width == 0 {
        return Vec::new();
    }
    let target = residuals
        .iter()
        .zip(observed)
        .map(|(row, mask)| {
            let rows = row
                .iter()
                .enumerate()
                .filter_map(|(lane, value)| mask[lane].then_some(*value))
                .collect::<Vec<_>>();
            if rows.is_empty() {
                None
            } else {
                Some(rows.iter().sum::<f64>() / rows.len() as f64)
            }
        })
        .collect::<Vec<_>>();
    (0..width)
        .map(|feature| {
            let paired = target
                .iter()
                .enumerate()
                .filter_map(|(time, value)| value.map(|target| (calendar[time][feature], target)))
                .collect::<Vec<_>>();
            if paired.len() < 2 {
                return 0.0;
            }
            let x_mean = paired.iter().map(|(x, _)| x).sum::<f64>() / paired.len() as f64;
            let y_mean = paired.iter().map(|(_, y)| y).sum::<f64>() / paired.len() as f64;
            let (numerator, denominator) = paired.iter().fold((0.0, 0.0), |(num, den), (x, y)| {
                let delta = x - x_mean;
                (num + delta * (y - y_mean), den + delta * delta)
            });
            if denominator <= 1e-12 {
                0.0
            } else {
                (numerator / denominator).clamp(-0.25, 0.25)
            }
        })
        .collect()
}

fn calendar_effect(weights: &[f64], calendar: &[f64]) -> f64 {
    weights
        .iter()
        .zip(calendar)
        .map(|(weight, value)| weight * value)
        .sum()
}
fn weekly_effects(residuals: &[Vec<f64>], timestamps: &[i64]) -> Vec<Vec<f64>> {
    let lanes = residuals[0].len();
    (0..7)
        .map(|day| {
            (0..lanes)
                .map(|lane| {
                    let rows: Vec<_> = residuals
                        .iter()
                        .zip(timestamps)
                        .filter_map(|(row, timestamp)| {
                            if timestamp.rem_euclid(7) as usize == day {
                                Some(row[lane])
                            } else {
                                None
                            }
                        })
                        .collect();
                    if rows.is_empty() {
                        0.0
                    } else {
                        rows.iter().sum::<f64>() / rows.len() as f64
                    }
                })
                .collect()
        })
        .collect()
}

fn weekly_effects_masked(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    timestamps: &[i64],
) -> Vec<Vec<f64>> {
    let lanes = residuals[0].len();
    (0..7)
        .map(|day| {
            (0..lanes)
                .map(|lane| {
                    let rows = residuals
                        .iter()
                        .zip(observed)
                        .zip(timestamps)
                        .filter_map(|((row, mask), timestamp)| {
                            (mask[lane] && timestamp.rem_euclid(7) as usize == day)
                                .then_some(row[lane])
                        })
                        .collect::<Vec<_>>();
                    if rows.is_empty() {
                        0.0
                    } else {
                        rows.iter().sum::<f64>() / rows.len() as f64
                    }
                })
                .collect()
        })
        .collect()
}
fn mix_coefficients(residuals: &[Vec<f64>], mix: Option<&Vec<Vec<Vec<f64>>>>) -> Vec<f64> {
    let lanes = residuals[0].len();
    match mix {
        None => vec![0.0; lanes],
        Some(mix) => (0..lanes)
            .map(|lane| {
                let (mut xy, mut xx) = (0.0, 0.0);
                for (time, row) in mix.iter().enumerate() {
                    let x = row[lane][0];
                    xy += x * residuals[time][lane];
                    xx += x * x;
                }
                if xx > 1e-12 {
                    xy / xx
                } else {
                    0.0
                }
            })
            .collect(),
    }
}

fn cross_target_couplings(primary: &[Vec<f64>], secondary: &[Vec<f64>]) -> Vec<f64> {
    (0..primary[0].len())
        .map(|lane| {
            let left = primary.iter().map(|row| row[lane]).collect::<Vec<_>>();
            let right = secondary.iter().map(|row| row[lane]).collect::<Vec<_>>();
            // Keep cross-target transfer conservative; the shared graph remains
            // the principal mechanism and this term only carries lane-local
            // co-movement between the caller-selected measures.
            (0.02 * correlation(&left, &right)).clamp(-0.02, 0.02)
        })
        .collect()
}
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let x = x - ma;
        let y = y - mb;
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa <= 1e-12 || bb <= 1e-12 {
        0.0
    } else {
        ab / (aa * bb).sqrt()
    }
}

fn masked_correlation(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    source: usize,
    target: usize,
) -> f64 {
    let (mut count, mut sum_left, mut sum_right, mut sum_sq_left, mut sum_sq_right, mut sum_xy) =
        (0usize, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (row, mask) in residuals.iter().zip(observed) {
        if mask[source] && mask[target] {
            let left = row[source];
            let right = row[target];
            count += 1;
            sum_left += left;
            sum_right += right;
            sum_sq_left += left * left;
            sum_sq_right += right * right;
            sum_xy += left * right;
        }
    }
    if count < 3 {
        0.0
    } else {
        let count = count as f64;
        let covariance = sum_xy - sum_left * sum_right / count;
        let left_variance = sum_sq_left - sum_left * sum_left / count;
        let right_variance = sum_sq_right - sum_right * sum_right / count;
        if left_variance <= 1e-12 || right_variance <= 1e-12 {
            0.0
        } else {
            covariance / (left_variance * right_variance).sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> MarketPanelFrame {
        let primary = (0..21)
            .map(|day| vec![10.0 + day as f64 * 0.1, 12.0 + day as f64 * 0.1, 8.0])
            .collect();
        let secondary = (0..21)
            .map(|day| vec![20.0 + day as f64, 18.0 + day as f64, 5.0])
            .collect();
        MarketPanelFrame::new(
            vec!["a:b".into(), "a:c".into(), "b:a".into()],
            (0..21).collect(),
            vec!["benchmark".into(), "volume".into()],
            primary,
            secondary,
            vec!["a".into(), "a".into(), "b".into()],
            vec!["b".into(), "c".into(), "a".into()],
            vec![vec![]; 3],
            vec![[0.0; 4]; 3],
            vec![vec![]; 21],
            None,
            vec![],
            vec![],
            2,
            "daily".into(),
        )
        .unwrap()
    }
    #[test]
    fn learns_sparse_directional_relationships() {
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&frame()).unwrap();
        let edges = model.relationships().unwrap();
        assert_eq!(model.neural_embeddings.len(), 3);
        assert_eq!(model.neural_embeddings[0].len(), 16);
        assert!(edges.len() <= 8 * 3);
        assert!(edges
            .iter()
            .any(|edge| edge.kinds.contains(&RelationshipKind::SharedOrigin)));
        assert!(edges
            .iter()
            .any(|edge| edge.kinds.contains(&RelationshipKind::ReverseLane)));
    }

    #[test]
    fn market_graph_kernel_runs_on_every_available_backend() {
        let mut cpu = MarketStructureForecaster::new(MarketStructureConfig {
            neural_hidden_dim: 4,
            neural_epochs: 1,
            head_epochs: 1,
            calibrate_intervals: false,
            ..MarketStructureConfig::default()
        })
        .unwrap();
        cpu.fit(&frame()).unwrap();
        let expected = cpu.predict(2, None).unwrap();
        for backend in cartoboost_neural::available_backends() {
            let mut model = MarketStructureForecaster::new(MarketStructureConfig {
                backend: backend.clone(),
                neural_hidden_dim: 4,
                neural_epochs: 1,
                head_epochs: 1,
                calibrate_intervals: false,
                ..MarketStructureConfig::default()
            })
            .unwrap_or_else(|error| panic!("{backend} market construction failed: {error}"));
            model
                .fit(&frame())
                .unwrap_or_else(|error| panic!("{backend} market fit failed: {error}"));
            assert_eq!(model.config.backend, backend);
            assert_eq!(model.neural_embeddings.len(), 3);
            assert_eq!(model.neural_embeddings[0].len(), 4);
            let actual = model.predict(2, None).unwrap();
            for (left, right) in actual.iter().zip(&expected) {
                assert!(
                    (left.primary - right.primary).abs() < 2.0e-3,
                    "{backend} primary mismatch: {} != {}",
                    left.primary,
                    right.primary
                );
                assert!(
                    (left.primary_lower - right.primary_lower).abs() < 2.0e-3,
                    "{backend} lower mismatch: {} != {}",
                    left.primary_lower,
                    right.primary_lower
                );
                assert!(
                    (left.primary_upper - right.primary_upper).abs() < 2.0e-3,
                    "{backend} upper mismatch: {} != {}",
                    left.primary_upper,
                    right.primary_upper
                );
            }
        }
    }

    #[test]
    fn top_k_is_enforced_per_source_lane() {
        let mut model = MarketStructureForecaster::new(MarketStructureConfig {
            top_k: 1,
            ..MarketStructureConfig::default()
        })
        .unwrap();
        model.fit(&frame()).unwrap();
        let edges = model.relationships().unwrap();
        for lane in ["a:b", "a:c", "b:a"] {
            assert!(
                edges
                    .iter()
                    .filter(|edge| edge.source_lane_id == lane)
                    .count()
                    <= 1
            );
        }
    }

    #[test]
    fn rejects_missing_geography_and_unavailable_label_cutoff() {
        let mut invalid_geo = frame();
        invalid_geo.coordinates[0][0] = f64::NAN;
        assert!(invalid_geo.validate().is_err());

        let mut invalid_label = frame();
        invalid_label.expert_labels.push(ExpertEventLabel {
            lane_id: "a:b".into(),
            timestamp: 999,
            shift: MarketShiftKind::Market,
            version: "review-1".into(),
        });
        assert!(invalid_label.validate().is_err());
    }
    #[test]
    fn predicts_and_explains() {
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&frame()).unwrap();
        assert_eq!(model.predict(2, None).unwrap().len(), 6);
        assert_eq!(model.nowcast().unwrap().len(), 3);
        let weekly = model.weekly_rollups(2, None).unwrap();
        assert_eq!(weekly.len(), 3);
        assert_eq!(weekly[0].days, 2);
        let explorer = model.explorer_payload(2).unwrap();
        assert_eq!(explorer["lanes"].as_array().unwrap().len(), 3);
        assert_eq!(explorer["forecasts"].as_array().unwrap().len(), 6);
        assert_eq!(explorer["explanations"].as_array().unwrap().len(), 3);
        assert!(explorer["kernels"].is_array());
    }
    #[test]
    fn rejects_unknown_expert_lane() {
        let mut input = frame();
        input.expert_priors.push(ExpertRelationshipPrior {
            version: "1".into(),
            source_lane_id: "missing".into(),
            target_lane_id: "a:b".into(),
            allowed: true,
            weight: 1.0,
        });
        assert!(input.validate().is_err());
    }

    #[test]
    fn expert_ban_and_artifact_round_trip_are_preserved() {
        let mut input = frame();
        input.expert_priors.push(ExpertRelationshipPrior {
            version: "review-1".into(),
            source_lane_id: "a:b".into(),
            target_lane_id: "a:c".into(),
            allowed: false,
            weight: 0.0,
        });
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert!(!model
            .relationships()
            .unwrap()
            .iter()
            .any(|edge| edge.source_lane_id == "a:b" && edge.target_lane_id == "a:c"));
        let restored =
            MarketStructureForecaster::from_json_string(&model.to_json_string().unwrap()).unwrap();
        assert_eq!(
            model.predict(1, None).unwrap(),
            restored.predict(1, None).unwrap()
        );
    }

    #[test]
    fn recorded_expert_label_is_preserved_without_overriding_model_assessment() {
        let mut input = frame();
        input.expert_labels.push(ExpertEventLabel {
            lane_id: "a:b".into(),
            timestamp: 20,
            shift: MarketShiftKind::LocalOrMix,
            version: "review-1".into(),
        });
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        let explanation = model.nowcast().unwrap().remove(0);
        assert_eq!(explanation.shift, MarketShiftKind::NoShift);
        assert_eq!(
            explanation.expert_label,
            Some(ExpertEventLabel {
                lane_id: "a:b".into(),
                timestamp: 20,
                shift: MarketShiftKind::LocalOrMix,
                version: "review-1".into(),
            })
        );
    }

    #[test]
    fn shared_shift_is_market_but_isolated_mix_shock_stays_local() {
        let mut input = frame();
        for time in 0..input.primary.len() {
            input.primary[time][0] = 10.0;
            input.primary[time][1] = 12.0;
        }
        let last = input.primary.len() - 1;
        input.primary[last][0] = 30.0;
        input.primary[last][1] = 36.0;
        let mut model = MarketStructureForecaster::new(MarketStructureConfig {
            graph_strength: 0.8,
            local_strength: 0.1,
            ..MarketStructureConfig::default()
        })
        .unwrap();
        model.fit(&input).unwrap();
        let shared = model.nowcast().unwrap();
        assert_eq!(shared[0].shift, MarketShiftKind::Market);

        input.primary[last][1] = 12.0;
        input.mix = Some(
            (0..input.primary.len())
                .map(|time| {
                    vec![
                        vec![if time == last { 1.0 } else { 0.0 }],
                        vec![0.0],
                        vec![0.0],
                    ]
                })
                .collect(),
        );
        let mut local = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        local.fit(&input).unwrap();
        let isolated = local.nowcast().unwrap();
        assert_eq!(isolated[0].shift, MarketShiftKind::LocalOrMix);
        assert_eq!(isolated[1].shift, MarketShiftKind::NoShift);
    }

    #[test]
    fn known_future_calendar_is_required_and_changes_forecast_path() {
        let mut input = frame();
        input.calendar = (0..input.timestamps.len())
            .map(|index| vec![if index % 2 == 0 { 1.0 } else { 0.0 }])
            .collect();
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert!(model.predict(1, None).is_err());
        assert!(model.explorer_payload(1).is_ok());
        let inactive = model.predict(1, Some(&[vec![0.0]])).unwrap();
        let active = model.predict(1, Some(&[vec![1.0]])).unwrap();
        assert_ne!(inactive[0].primary, active[0].primary);
    }

    #[test]
    fn interval_calibration_uses_a_train_only_trailing_origin() {
        let mut input = frame();
        for (time, row) in input.primary.iter_mut().enumerate() {
            row[0] += time as f64 * 0.05;
        }
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert!(model.interval_calibration_multiplier.is_finite());
        assert!(model.interval_calibration_multiplier >= 1.0);
        let prediction = model.predict(1, None).unwrap();
        assert!(prediction[0].primary_lower < prediction[0].primary_upper);
        assert!(model.joint_heads.is_some());
        assert_eq!(
            model.joint_heads.as_ref().unwrap().primary_quantiles.len(),
            model.config.quantile_levels.len()
        );
    }

    #[test]
    fn joint_heads_reject_invalid_quantile_levels() {
        assert!(MarketStructureForecaster::new(MarketStructureConfig {
            quantile_levels: vec![0.5, 0.1],
            ..MarketStructureConfig::default()
        })
        .is_err());
    }

    #[test]
    fn auto_backend_is_resolved_before_artifact_serialization() {
        let model = MarketStructureForecaster::new(MarketStructureConfig {
            backend: "auto".to_string(),
            ..MarketStructureConfig::default()
        })
        .unwrap();
        assert_ne!(model.backend(), "auto");
        let payload = serde_json::to_value(&model).unwrap();
        assert_eq!(payload["config"]["backend"], model.backend());
    }

    #[test]
    fn unobserved_lane_uses_explicit_hierarchy_without_a_filled_observation() {
        let mut input = frame();
        for row in &mut input.primary {
            row[2] = f64::NAN;
        }
        input.hierarchy_groups[2] = vec!["parent:a".into()];
        input.hierarchy_groups[0] = vec!["parent:a".into()];
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert_eq!(model.predict(1, None).unwrap().len(), 3);
        let explanation = model.nowcast().unwrap();
        assert_eq!(explanation[2].observed_primary, None);
        assert_eq!(explanation[2].support, MarketSupportKind::Hierarchy);
        assert_eq!(explanation[2].shift, MarketShiftKind::NoShift);
    }
}
