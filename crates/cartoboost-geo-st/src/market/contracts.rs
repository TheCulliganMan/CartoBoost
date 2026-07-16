#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketStructureConfig {
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

impl Default for MarketStructureConfig {
    fn default() -> Self {
        Self {
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

