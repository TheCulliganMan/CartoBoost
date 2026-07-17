#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct STAEformerConfig {
    pub lookback: usize,
    pub attention_heads: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub ridge: f64,
    #[serde(default)]
    pub backend: ComputeBackendSelection,
}

impl Default for STAEformerConfig {
    fn default() -> Self {
        Self {
            lookback: 8,
            attention_heads: 4,
            hidden_size: 8,
            epochs: 120,
            learning_rate: 0.02,
            ridge: 1e-4,
            backend: select_compute_backend(None).expect("default CPU backend is always available"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct STAEformerForecaster {
    pub config: STAEformerConfig,
    node_ids: Vec<String>,
    frequency: String,
    horizon: usize,
    adjacency: Option<CsrAdjacency>,
    weights: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    temporal_queries: Vec<Vec<f64>>,
    temporal_keys: Vec<Vec<f64>>,
    spatial_weights: Vec<f64>,
    history: Vec<Vec<f64>>,
    target_mean: f64,
    target_scale: f64,
}

/// Native graph-transformer profiles. The public Python names map to these
/// generic architectural behaviors rather than embedding a benchmark in the
/// shared Rust API.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphTransformerProfile {
    HeterogeneousMoE,
    EfficientHighOrder,
    LongShortFusion,
    GatedGraphTemporal,
    SpatialShiftGraphonMoE,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperGraphTransformerConfig {
    pub profile: GraphTransformerProfile,
    pub lookback: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub graph_order: usize,
    pub experts: usize,
    pub periodicity: usize,
    pub recent_window: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    #[serde(default)]
    pub backend: ComputeBackendSelection,
}

impl Default for PaperGraphTransformerConfig {
    fn default() -> Self {
        Self {
            profile: GraphTransformerProfile::HeterogeneousMoE,
            lookback: 12,
            hidden_size: 16,
            attention_heads: 4,
            graph_order: 2,
            experts: 4,
            periodicity: 24,
            recent_window: 12,
            epochs: 80,
            learning_rate: 0.01,
            weight_decay: 1e-5,
            backend: select_compute_backend(None).expect("default CPU backend is always available"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaperGraphTransformerForecaster {
    pub config: PaperGraphTransformerConfig,
    node_ids: Vec<String>,
    frequency: String,
    horizon: usize,
    adjacency: Option<CsrAdjacency>,
    #[serde(default)]
    trainable_state: Option<TrainableGraphTransformerState>,
    history: Vec<Vec<f64>>,
    #[serde(default)]
    history_time_features: Option<Vec<Vec<f64>>>,
    target_mean: f64,
    target_scale: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LsttnTrainingCheckpoint {
    version: u32,
    data_fingerprint: u64,
    config_json: String,
    state: TrainableGraphTransformerState,
    pretraining_completed: usize,
    supervised_batches_completed: usize,
    complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PaperGraphTransformerArchitectureReport {
    pub profile: GraphTransformerProfile,
    pub components: Vec<String>,
    pub graphon_expert_count: usize,
    pub direct_multi_horizon: bool,
    pub trainable_forecast_head: bool,
}

/// Serializable native parameters for the graph-transformer profiles.
///
/// The state deliberately owns every learned projection and its optimizer
/// moments.  This avoids a Python or fixed-feature training path and makes
/// attention, routing, gated fusion, and graphon embeddings part of the
/// fitted artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainableGraphTransformerState {
    parameters: Vec<f64>,
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
    steps: u64,
    nodes: usize,
    hidden: usize,
    attention_heads: usize,
    periodicity: usize,
    recent_window: usize,
    context_window: usize,
    horizons: usize,
    experts: usize,
    graph_order: usize,
    /// Native cadence for the first periodic branch. Zero preserves legacy
    /// non-weekly checkpoints, where it is derived from `periodicity`.
    #[serde(default)]
    periodic_short_lag: usize,
    #[serde(default = "unit_scale")]
    target_scale: f64,
    #[serde(default)]
    normalized_zero: f64,
}

/// CUDA-owned, portable-state bridge for LSTTN.  It deliberately owns no
/// serialized driver object: parameters and Adam moments are uploaded from
/// `TrainableGraphTransformerState` on construction and copied back only at a
/// checkpoint boundary.  CSR topology and structural weights remain resident
/// for every pretraining/supervised batch.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[allow(dead_code)]
struct CudaLsttnTensorExecutor {
    arena: CudaTensorArena,
    nodes: usize,
    node_tile_size: usize,
    forward_edges: usize,
    reverse_edges: usize,
    adaptive_edges: usize,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[allow(dead_code)]
impl CudaLsttnTensorExecutor {
    const PARAMETERS: usize = 0;
    const FIRST_MOMENT: usize = 1;
    const SECOND_MOMENT: usize = 2;
    const FORWARD_WEIGHTS: usize = 3;
    const REVERSE_WEIGHTS: usize = 4;
    const ADAPTIVE_WEIGHTS: usize = 5;
    const SUPERVISED_INPUT: usize = 6;
    const SUPERVISED_TARGET: usize = 7;
    const DIRECT_OUTPUT: usize = 8;
    const PATCH_EMBEDDING: usize = 9;
    const PATCH_WITH_POSITION: usize = 10;
    const ATTENTION_SEQUENCE: usize = 11;
    const ENCODER_Q: usize = 12;
    const ENCODER_K: usize = 13;
    const ENCODER_V: usize = 14;
    const ENCODER_ATTENTION: usize = 15;
    const ENCODER_PROJECTED: usize = 16;
    const ENCODER_RESIDUAL: usize = 17;
    const ENCODER_NORMALIZED: usize = 18;
    const ENCODER_FFN_EXPANDED: usize = 19;
    const ENCODER_FFN_ACTIVATED: usize = 20;
    const ENCODER_FFN_CONTRACTED: usize = 21;
    const ENCODER_FFN_RESIDUAL: usize = 22;
    const ADAPTIVE_LOGITS: usize = 23;
    const LONG_TEMPORAL: usize = 24;
    const LONG_STAGE_A: usize = 25;
    const LONG_STAGE_B: usize = 26;
    const SHORT_INPUT: usize = 27;
    const SHORT_FILTER: usize = 28;
    const SHORT_GATE: usize = 29;
    const SHORT_GATED: usize = 30;
    const SHORT_SKIP_PROJECTION: usize = 31;
    const SHORT_FORWARD_ONE: usize = 32;
    const SHORT_FORWARD_TWO: usize = 33;
    const SHORT_BACKWARD_ONE: usize = 34;
    const SHORT_BACKWARD_TWO: usize = 35;
    const SHORT_ADAPTIVE_ONE: usize = 36;
    const SHORT_ADAPTIVE_TWO: usize = 37;
    const SHORT_CONCAT_A: usize = 38;
    const SHORT_CONCAT_B: usize = 39;
    const SHORT_CONCAT_C: usize = 40;
    const SHORT_CONCAT_D: usize = 41;
    const SHORT_CONCAT_E: usize = 42;
    const SHORT_CONCAT_F: usize = 43;
    const SHORT_GRAPH: usize = 44;
    const SHORT_RESIDUAL: usize = 45;
    const SHORT_SKIP_A: usize = 46;
    const SHORT_NORMALIZED: usize = 47;
    const SHORT_SKIP_B: usize = 48;
    const SHORT_BATCH_STATS: usize = 49;
    const PARAMETER_GRADIENT: usize = 50;
    const GRADIENT_NORM: usize = 51;
    const SUPERVISED_LOSS: usize = 52;
    const PERIODIC_SHORT: usize = 53;
    const PERIODIC_SEASONAL: usize = 54;
    const FUSION_A: usize = 55;
    const FUSION_B: usize = 56;
    const FUSION_C: usize = 57;
    const FUSION_D: usize = 58;
    const FUSION_OUTPUT: usize = 59;
    const DIRECT_NODE_MAJOR: usize = 60;
    const FUSED_DIRECT_OUTPUT: usize = 61;
    const DIRECT_OUTPUT_GRADIENT: usize = 62;
    const DIRECT_NODE_MAJOR_GRADIENT: usize = 63;
    const FUSION_REPRESENTATION_GRADIENT: usize = 64;
    const FUSION_RELU_GRADIENT: usize = 65;
    const FUSION_CONCAT_GRADIENT: usize = 66;
    const SHORT_GRADIENT: usize = 67;
    const TREND_GRADIENT: usize = 68;
    const FUSION_SECOND_GRADIENT: usize = 69;
    const FUSION_FIRST_RELU_GRADIENT: usize = 70;
    const FUSION_INPUT_GRADIENT: usize = 71;
    const LONG_GRADIENT: usize = 72;
    const PERIODIC_PAIR_GRADIENT: usize = 73;
    const PERIODIC_SHORT_GRADIENT: usize = 74;
    const PERIODIC_SEASONAL_GRADIENT: usize = 75;
    const PERIODIC_SHORT_INPUT: usize = 76;
    const PERIODIC_SEASONAL_INPUT: usize = 77;
    const PERIODIC_SHORT_INPUT_GRADIENT: usize = 78;
    const PERIODIC_SEASONAL_INPUT_GRADIENT: usize = 79;
    const PERIODIC_GRAD_IDENTITY: usize = 80;
    const PERIODIC_GRAD_FORWARD_ONE: usize = 81;
    const PERIODIC_GRAD_FORWARD_TWO: usize = 82;
    const PERIODIC_GRAD_REVERSE_ONE: usize = 83;
    const PERIODIC_GRAD_REVERSE_TWO: usize = 84;
    const PERIODIC_GRAD_ADAPTIVE_ONE: usize = 85;
    const PERIODIC_GRAD_ADAPTIVE_TWO: usize = 86;
    const PERIODIC_TEMP_A: usize = 87;
    const PERIODIC_TEMP_B: usize = 88;
    const PERIODIC_TEMP_C: usize = 89;
    const PERIODIC_BASE_GRADIENT: usize = 90;
    const PERIODIC_FORWARD_BASE_GRADIENT: usize = 91;
    const PERIODIC_REVERSE_BASE_GRADIENT: usize = 92;
    const PERIODIC_ADAPTIVE_BASE_GRADIENT: usize = 93;
    const PERIODIC_ADAPTIVE_EDGE_GRADIENT: usize = 94;
    const PERIODIC_ADAPTIVE_LOGIT_GRADIENT: usize = 95;
    const LONG_BACKWARD_A: usize = 96;
    const LONG_BACKWARD_B: usize = 97;
    const SHORT_REVERSE_INPUT_GRADIENT_A: usize = 98;
    const SHORT_REVERSE_INPUT_GRADIENT_B: usize = 99;
    const SHORT_REVERSE_SKIP_GRADIENT_A: usize = 100;
    const SHORT_REVERSE_SKIP_GRADIENT_B: usize = 101;
    const SHORT_REVERSE_GRAPH_GRADIENT: usize = 102;
    const SHORT_REVERSE_CONCAT_GRADIENT: usize = 103;
    const SHORT_REVERSE_SPLIT_A: usize = 104;
    const SHORT_REVERSE_SPLIT_B: usize = 105;
    const SHORT_REVERSE_SPLIT_C: usize = 106;
    const SHORT_REVERSE_SPLIT_D: usize = 107;
    const SHORT_REVERSE_SPLIT_E: usize = 108;
    const SHORT_REVERSE_SPLIT_F: usize = 109;
    const SHORT_REVERSE_GATED_GRADIENT: usize = 110;
    const SHORT_REVERSE_FILTER_GRADIENT: usize = 111;
    const SHORT_REVERSE_GATE_GRADIENT: usize = 112;
    const SHORT_REVERSE_EDGE_GRADIENT: usize = 113;
    const SHORT_REVERSE_LOGIT_GRADIENT: usize = 114;
    const SHORT_GRAPH_GRAD_IDENTITY: usize = 115;
    const SHORT_GRAPH_GRAD_FORWARD_ONE: usize = 116;
    const SHORT_GRAPH_GRAD_FORWARD_TWO: usize = 117;
    const SHORT_GRAPH_GRAD_REVERSE_ONE: usize = 118;
    const SHORT_GRAPH_GRAD_REVERSE_TWO: usize = 119;
    const SHORT_GRAPH_GRAD_ADAPTIVE_ONE: usize = 120;
    const SHORT_GRAPH_GRAD_ADAPTIVE_TWO: usize = 121;
    const PRETRAIN_VISIBLE_TOKENS: usize = 122;
    const PRETRAIN_DECODER_INPUT: usize = 123;
    const PRETRAIN_DECODER_Q: usize = 124;
    const PRETRAIN_DECODER_K: usize = 125;
    const PRETRAIN_DECODER_V: usize = 126;
    const PRETRAIN_DECODER_ATTENTION: usize = 127;
    const PRETRAIN_DECODER_PROJECTED: usize = 128;
    const PRETRAIN_DECODER_RESIDUAL: usize = 129;
    const PRETRAIN_DECODER_NORMALIZED: usize = 130;
    const PRETRAIN_DECODER_FFN_EXPANDED: usize = 131;
    const PRETRAIN_DECODER_FFN_ACTIVATED: usize = 132;
    const PRETRAIN_DECODER_FFN_CONTRACTED: usize = 133;
    const PRETRAIN_DECODER_FFN_RESIDUAL: usize = 134;
    const PRETRAIN_CONTEXT_GRADIENT: usize = 135;
    const PRETRAIN_DECODER_INPUT_GRADIENT: usize = 136;
    const PRETRAIN_VISIBLE_GRADIENT: usize = 137;
    const PRETRAIN_ENCODER_DECODER_GRADIENT: usize = 138;
    const PRETRAIN_ENCODER_OUTPUT_GRADIENT: usize = 139;
    const PRETRAIN_SEQUENCE_GRADIENT: usize = 140;
    const PRETRAIN_PATCH_LAYOUT_GRADIENT: usize = 141;
    const PRETRAIN_PATCH_EMBEDDING_GRADIENT: usize = 142;
    const PRETRAIN_TEMP_A: usize = 143;
    const PRETRAIN_TEMP_B: usize = 144;
    const PRETRAIN_TEMP_C: usize = 145;
    const PRETRAIN_TEMP_D: usize = 146;
    const PRETRAIN_TEMP_E: usize = 147;
    const PRETRAIN_TEMP_F: usize = 148;
    const PRETRAIN_LOSS: usize = 149;
    const PRETRAIN_LAYER_A: usize = 150;
    const PRETRAIN_LAYER_B: usize = 151;
    const PRETRAIN_LAYER_GRAD_A: usize = 152;
    const PRETRAIN_LAYER_GRAD_B: usize = 153;
    const PRETRAIN_DECODER_OUTPUT: usize = 154;
    const PRETRAIN_POSITION_GRADIENT: usize = 155;
    const FORWARD_INDPTR: usize = 0;
    const FORWARD_INDICES: usize = 1;
    const REVERSE_INDPTR: usize = 2;
    const REVERSE_INDICES: usize = 3;
    const ADAPTIVE_INDPTR: usize = 4;
    const ADAPTIVE_INDICES: usize = 5;
    const VISIBLE_PATCH_INDICES: usize = 6;
    const MASKED_PATCH_INDICES: usize = 7;

    fn new(state: &TrainableGraphTransformerState, adjacency: &CsrAdjacency) -> Result<Self> {
        let nodes = state.nodes;
        let reverse = adjacency.transpose(nodes).row_normalized();
        let adaptive = adjacency.with_adaptive_self_candidates(nodes);
        let u32_values = |values: &[usize], label: &str| -> Result<Vec<u32>> {
            values
                .iter()
                .map(|value| {
                    u32::try_from(*value).map_err(|_| {
                        GeoStError::InvalidFrame(format!(
                            "LSTTN CUDA {label} exceeds the u32 CSR address range"
                        ))
                    })
                })
                .collect()
        };
        let to_f32 = |values: &[f64], label: &str| -> Result<Vec<f32>> {
            values
                .iter()
                .map(|value| {
                    let value = *value as f32;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(GeoStError::InvalidFrame(format!(
                            "LSTTN CUDA {label} contains a non-finite f32 value"
                        )))
                    }
                })
                .collect()
        };
        let mut arena = CudaTensorArena::new(156)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .upload_f32(Self::PARAMETERS, &to_f32(&state.parameters, "parameters")?)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .upload_f32(
                Self::FIRST_MOMENT,
                &to_f32(&state.first_moment, "first moments")?,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .upload_f32(
                Self::SECOND_MOMENT,
                &to_f32(&state.second_moment, "second moments")?,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        arena
            .fill_f32(Self::PARAMETER_GRADIENT, state.parameters.len(), 0.0)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let upload_graph = |arena: &mut CudaTensorArena,
                            indptr_slot: usize,
                            indices_slot: usize,
                            weights_slot: usize,
                            graph: &CsrAdjacency,
                            label: &str|
         -> Result<()> {
            arena
                .upload_u32(indptr_slot, &u32_values(&graph.indptr, label)?)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            arena
                .upload_u32(indices_slot, &u32_values(&graph.indices, label)?)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            arena
                .upload_f32(weights_slot, &to_f32(&graph.data, label)?)
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
        };
        upload_graph(
            &mut arena,
            Self::FORWARD_INDPTR,
            Self::FORWARD_INDICES,
            Self::FORWARD_WEIGHTS,
            adjacency,
            "forward CSR",
        )?;
        upload_graph(
            &mut arena,
            Self::REVERSE_INDPTR,
            Self::REVERSE_INDICES,
            Self::REVERSE_WEIGHTS,
            &reverse,
            "reverse CSR",
        )?;
        upload_graph(
            &mut arena,
            Self::ADAPTIVE_INDPTR,
            Self::ADAPTIVE_INDICES,
            Self::ADAPTIVE_WEIGHTS,
            &adaptive,
            "adaptive CSR",
        )?;
        Ok(Self {
            arena,
            nodes,
            // A 1,024-node physical tile keeps a `[32, 52, tile, 32]`
            // hidden activation below 208 MiB. Logical batches remain 32 and
            // gradients are reduced across every tile before AdamW.
            node_tile_size: 1_024.min(nodes).max(1),
            forward_edges: adjacency.indices.len(),
            reverse_edges: reverse.indices.len(),
            adaptive_edges: adaptive.indices.len(),
        })
    }

    fn allocation_count(&self) -> usize {
        self.arena.allocation_count()
    }

    fn node_tiles(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.nodes)
            .step_by(self.node_tile_size)
            .map(|start| start..(start + self.node_tile_size).min(self.nodes))
    }

}

// CUDA transformer execution stages share this graph-module namespace.
include!("transformers/tensor_ops.rs");
include!("transformers/graph_branches.rs");
include!("transformers/fusion.rs");
include!("transformers/backward.rs");
include!("transformers/training.rs");
include!("transformers/state_sync.rs");
