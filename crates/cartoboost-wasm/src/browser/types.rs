#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastRequest {
    rows: Vec<BrowserForecastRow>,
    frequency: String,
    horizon: usize,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    options: BrowserForecastOptions,
    #[serde(default)]
    metadata: BrowserForecastMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastRow {
    #[serde(default)]
    series_id: Option<String>,
    timestamp: String,
    target: f64,
    #[serde(default)]
    covariates: BTreeMap<String, f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastOptions {
    input_size: Option<usize>,
    hidden_size: Option<usize>,
    epochs: Option<usize>,
    pooling_size: Option<usize>,
    season_length: Option<usize>,
    theta: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    damping_phi: Option<f64>,
    theta_grid: Option<Vec<f64>>,
    alpha_grid: Option<Vec<f64>>,
    theta_seasonality: Option<String>,
    max_p: Option<usize>,
    max_d: Option<usize>,
    max_q: Option<usize>,
    level_process_variance: Option<f64>,
    trend_process_variance: Option<f64>,
    observation_variance: Option<f64>,
    window_size: Option<usize>,
    window_count: Option<usize>,
    n_lags: Option<usize>,
    n_forecasts: Option<usize>,
    validation_window: Option<usize>,
    max_direct_horizon: Option<usize>,
    max_auto_candidate_count: Option<usize>,
    include_components: Option<bool>,
    include_history_components: Option<bool>,
    include_samples: Option<bool>,
    include_quantiles: Option<bool>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    lags: Option<Vec<usize>>,
    rolling_mean_windows: Option<Vec<usize>>,
    rolling_std_windows: Option<Vec<usize>>,
    rolling_min_windows: Option<Vec<usize>>,
    rolling_max_windows: Option<Vec<usize>>,
    difference_lags: Option<Vec<usize>>,
    rolling_trend_windows: Option<Vec<usize>>,
    calendar_features: Option<Vec<String>>,
    mstl_season_lengths: Option<Vec<usize>>,
    coordinate_x: Option<String>,
    coordinate_y: Option<String>,
    kriging_range: Option<f64>,
    kriging_nugget: Option<f64>,
    spatial_kriging_mode: Option<String>,
    spatial_regressors: Option<Vec<String>>,
    residual_shrinkage: Option<f64>,
    allow_neighbor_fallback: Option<bool>,
    changepoints: Option<usize>,
    n_changepoints: Option<usize>,
    changepoint_range: Option<f64>,
    changepoint_timestamps: Option<Vec<String>>,
    yearly_fourier_order: Option<usize>,
    weekly_fourier_order: Option<usize>,
    daily_fourier_order: Option<usize>,
    auto_yearly_seasonality: Option<bool>,
    auto_weekly_seasonality: Option<bool>,
    auto_daily_seasonality: Option<bool>,
    custom_seasonalities: Option<Vec<BrowserForecastSeasonality>>,
    changepoint_l2_regularization: Option<f64>,
    changepoint_l1_regularization: Option<f64>,
    changepoint_prior_scale: Option<f64>,
    seasonality_l2_regularization: Option<f64>,
    seasonality_prior_scale: Option<f64>,
    yearly_l2_regularization: Option<f64>,
    weekly_l2_regularization: Option<f64>,
    daily_l2_regularization: Option<f64>,
    event_l2_regularization: Option<f64>,
    holidays_prior_scale: Option<f64>,
    regressor_l2_regularization: Option<f64>,
    event_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
    regressor_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
    events: Option<Vec<BrowserForecastEvent>>,
    holidays: Option<Vec<BrowserForecastHoliday>>,
    event_mode: Option<String>,
    holidays_mode: Option<String>,
    extra_regressors: Option<Vec<String>>,
    regressor_modes: Option<BTreeMap<String, String>>,
    lagged_regressors: Option<BTreeMap<String, usize>>,
    ar_layers: Option<Vec<usize>>,
    lagged_reg_layers: Option<Vec<usize>>,
    trend_mode: Option<String>,
    local_l2: Option<f64>,
    extra_regressor_monotonic_constraints: Option<BTreeMap<String, i8>>,
    regressor_standardization: Option<String>,
    future_regressors: Option<BTreeMap<String, Vec<f64>>>,
    future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
    trend_adjustments: Option<BTreeMap<usize, f64>>,
    trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    residual_shock_window: Option<usize>,
    residual_shock_scale: Option<f64>,
    residual_shock_decay: Option<f64>,
    interval_levels: Option<Vec<f64>>,
    interval_width: Option<f64>,
    quantile_levels: Option<Vec<f64>>,
    uncertainty_samples: Option<usize>,
    mcmc_samples: Option<usize>,
    trend_uncertainty_policy: Option<String>,
    trend_uncertainty_scale: Option<f64>,
    coefficient_uncertainty_scale: Option<f64>,
    uncertainty_seed: Option<u64>,
    growth: Option<String>,
    component_mode: Option<String>,
    seasonality_mode: Option<String>,
    fit_loss: Option<String>,
    huber_delta: Option<f64>,
    irls_iterations: Option<usize>,
    cap: Option<f64>,
    floor: Option<f64>,
    cap_regressor: Option<String>,
    floor_regressor: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeotemporalDiagnosticsRequest {
    quantiles: Option<BrowserQuantileDiagnosticsRequest>,
    residual_correction: Option<BrowserResidualCorrectionRequest>,
    regime: Option<BrowserRegimeDiagnosticsRequest>,
    calibration: Option<BrowserCalibrationRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoCausalRequest {
    rows: Vec<BrowserGeoCausalRow>,
    intervention_time: String,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    placebo_n: usize,
    #[serde(default)]
    spatial_weights: Vec<BrowserSpatialWeight>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoCausalRow {
    unit_id: String,
    time: String,
    outcome: f64,
    treatment: bool,
    #[serde(default)]
    covariates: BTreeMap<String, f64>,
    latitude: Option<f64>,
    longitude: Option<f64>,
    region_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSpatialWeight {
    from_unit: String,
    to_unit: String,
    weight: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserQuantileDiagnosticsRequest {
    values: Option<Vec<f64>>,
    actual: Option<Vec<f64>>,
    prediction: Option<Vec<f64>>,
    quantile: Option<f64>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    quantile_rows: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserResidualCorrectionRequest {
    process_variance: f64,
    observation_variance: f64,
    observations: Vec<BrowserResidualObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserResidualObservation {
    key: BrowserResidualStateKey,
    structural_prediction: f64,
    observed: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserResidualStateKey {
    origin: Option<String>,
    destination: Option<String>,
    corridor: Option<String>,
    segment: Option<String>,
    entity_family: Option<String>,
    target_family: Option<String>,
    time_bucket: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegimeDiagnosticsRequest {
    residuals: Vec<f64>,
    cusum: Option<CusumConfig>,
    page_hinkley: Option<PageHinkleyConfig>,
    ewma: Option<EwmaVolatilityConfig>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    policy: Option<RegimeIntervalPolicy>,
    rolling_window: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCalibrationRequest {
    scores: Option<Vec<f64>>,
    labels: Vec<f64>,
    probabilities: Option<Vec<f64>>,
    before_probabilities: Option<Vec<f64>>,
    method: Option<String>,
    bucket_count: Option<usize>,
    event: Option<BrowserCalibrationEventRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCalibrationEventRequest {
    kind: String,
    actual: Vec<f64>,
    prediction: Option<Vec<f64>>,
    threshold: Option<f64>,
    horizon: Option<f64>,
    warning_threshold: Option<f64>,
    critical_threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastEvent {
    name: String,
    timestamp: String,
    #[serde(default)]
    lower_window: Option<i32>,
    #[serde(default)]
    upper_window: Option<i32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastHoliday {
    holiday: String,
    ds: String,
    #[serde(default)]
    lower_window: Option<i32>,
    #[serde(default)]
    upper_window: Option<i32>,
    #[serde(default)]
    prior_scale: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastSeasonality {
    name: String,
    period_days: f64,
    fourier_order: usize,
    mode: Option<String>,
    condition_name: Option<String>,
    l2_regularization: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastMetadata {
    timestamp_col: Option<String>,
    target_col: Option<String>,
    series_id_col: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastArtifactPredictOptions {
    #[serde(default = "default_true")]
    include_components: bool,
    #[serde(default)]
    include_history_components: bool,
    #[serde(default = "default_true")]
    include_samples: bool,
    #[serde(default = "default_true")]
    include_quantiles: bool,
    future_regressors: Option<BTreeMap<String, Vec<f64>>>,
    future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
    trend_adjustments: Option<BTreeMap<usize, f64>>,
    trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    interval_levels: Option<Vec<f64>>,
    quantile_levels: Option<Vec<f64>>,
    uncertainty_samples: Option<usize>,
}

impl Default for BrowserForecastArtifactPredictOptions {
    fn default() -> Self {
        Self {
            include_components: true,
            include_history_components: false,
            include_samples: true,
            include_quantiles: true,
            future_regressors: None,
            future_regressors_by_series: None,
            trend_adjustments: None,
            trend_adjustments_by_series: None,
            interval_levels: None,
            quantile_levels: None,
            uncertainty_samples: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionRequest {
    rows: Vec<BrowserRegressionRow>,
    feature_names: Vec<String>,
    #[serde(default)]
    sparse_feature_names: Vec<String>,
    #[serde(default)]
    options: BrowserRegressionOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionRow {
    features: Vec<f64>,
    #[serde(default)]
    sparse_sets: Vec<Vec<u64>>,
    target: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionOptions {
    #[serde(default = "default_holdout_fraction")]
    holdout_fraction: f64,
    splitter_mode: Option<String>,
    #[serde(default)]
    feature_kinds: BTreeMap<String, String>,
    #[serde(default)]
    periodic_periods: BTreeMap<String, u32>,
    loss: Option<String>,
    quantile_alpha: Option<f64>,
    huber_delta: Option<f64>,
    log_offset: Option<f64>,
    interval_lower_alpha: Option<f64>,
    interval_upper_alpha: Option<f64>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    monotonic_constraints: Option<Vec<i8>>,
    include_model_visualization: Option<bool>,
}

impl Default for BrowserRegressionOptions {
    fn default() -> Self {
        Self {
            holdout_fraction: default_holdout_fraction(),
            splitter_mode: None,
            feature_kinds: BTreeMap::new(),
            periodic_periods: BTreeMap::new(),
            loss: None,
            quantile_alpha: None,
            huber_delta: None,
            log_offset: None,
            interval_lower_alpha: None,
            interval_upper_alpha: None,
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_leaf: None,
            monotonic_constraints: None,
            include_model_visualization: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralRequest {
    rows: Vec<BrowserNeuralRow>,
    dense_feature_names: Vec<String>,
    #[serde(default)]
    node_features: Vec<Vec<f32>>,
    #[serde(default)]
    node_types: Vec<usize>,
    #[serde(default)]
    edge_type_triples: Vec<(usize, usize, usize)>,
    #[serde(default = "default_neural_pipeline")]
    pipeline: String,
    #[serde(default)]
    options: BrowserNeuralOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSequenceRequest {
    operation: String,
    series: Option<SequenceSeries>,
    frame: Option<SequenceFrame>,
    reference: Option<ReferenceSignal>,
    state_space_config: Option<SequenceStateSpaceConfig>,
    reference_path_config: Option<ReferencePathConfig>,
    candidates: Option<Vec<SequenceCandidate>>,
    weights: Option<BTreeMap<String, f64>>,
    actuals: Option<Vec<SequenceCandidatePrediction>>,
    oof_fold: Option<SequenceOofFold>,
    oof_rows: Option<Vec<SequenceOofCandidateRow>>,
    group_predictions: Option<Vec<SequenceGroupPrediction>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralRow {
    id: Option<u64>,
    source: Option<usize>,
    target_node: Option<usize>,
    edge_weight: Option<f32>,
    edge_type: Option<usize>,
    dense: Vec<f64>,
    target: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralOptions {
    #[serde(default = "default_holdout_fraction")]
    holdout_fraction: f64,
    embedding_dim: Option<usize>,
    random_state: Option<u64>,
    support_prior_strength: Option<f64>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    node2vec_walk_length: Option<usize>,
    node2vec_walks_per_node: Option<usize>,
    node2vec_window_size: Option<usize>,
    node2vec_epochs: Option<usize>,
    node2vec_learning_rate: Option<f32>,
    node2vec_p: Option<f32>,
    node2vec_q: Option<f32>,
    node2vec_seed: Option<u64>,
    graph_sage_epochs: Option<usize>,
    graph_sage_learning_rate: Option<f32>,
    graph_sage_negative_samples: Option<usize>,
    graph_sage_seed: Option<u64>,
    include_model_visualization: Option<bool>,
}

impl Default for BrowserNeuralOptions {
    fn default() -> Self {
        Self {
            holdout_fraction: default_holdout_fraction(),
            embedding_dim: None,
            random_state: None,
            support_prior_strength: None,
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_leaf: None,
            node2vec_walk_length: None,
            node2vec_walks_per_node: None,
            node2vec_window_size: None,
            node2vec_epochs: None,
            node2vec_learning_rate: None,
            node2vec_p: None,
            node2vec_q: None,
            node2vec_seed: None,
            graph_sage_epochs: None,
            graph_sage_learning_rate: None,
            graph_sage_negative_samples: None,
            graph_sage_seed: None,
            include_model_visualization: None,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastResponse {
    metadata: Value,
    forecast: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    components: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    history_components: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    samples: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantiles: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastArtifactResponse {
    metadata: Value,
    artifact: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionResponse {
    metadata: Value,
    metrics: BrowserRegressionMetrics,
    predictions: Vec<BrowserRegressionPrediction>,
    feature_importance: Vec<BrowserFeatureImportance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_visualization: Option<BrowserModelVisualization>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralResponse {
    metadata: Value,
    metrics: BrowserRegressionMetrics,
    predictions: Vec<BrowserRegressionPrediction>,
    feature_importance: Vec<BrowserFeatureImportance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_visualization: Option<BrowserModelVisualization>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionMetrics {
    rmse: f64,
    mae: f64,
    r2: f64,
    train_rows: usize,
    holdout_rows: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionPrediction {
    row_index: usize,
    actual: f64,
    prediction: f64,
    lower_prediction: Option<f64>,
    upper_prediction: Option<f64>,
    residual: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserFeatureImportance {
    feature: String,
    split_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserModelVisualization {
    summary: BrowserModelVisualizationSummary,
    split_kinds: Vec<BrowserSplitKindCount>,
    splitter_rules: Vec<BrowserSplitterRuleSummary>,
    feature_split_counts: Vec<BrowserFeatureSplitCount>,
    depth_histogram: Vec<BrowserDepthCount>,
    tree_blueprints: Vec<BrowserTreeBlueprint>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserModelVisualizationSummary {
    tree_count: usize,
    node_count: usize,
    branch_count: usize,
    leaf_count: usize,
    max_depth: usize,
    mean_leaf_value: f64,
    mean_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSplitKindCount {
    kind: String,
    count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSplitterRuleSummary {
    kind: String,
    label: String,
    count: usize,
    total_gain: f64,
    mean_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserFeatureSplitCount {
    feature: String,
    kind: String,
    count: usize,
    total_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserDepthCount {
    depth: usize,
    count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserTreeBlueprint {
    tree_index: usize,
    node_count: usize,
    branch_count: usize,
    leaf_count: usize,
    max_depth: usize,
    total_gain: f64,
    root: BrowserTreeNode,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserTreeNode {
    id: usize,
    depth: usize,
    kind: String,
    label: String,
    value: Option<f64>,
    gain: Option<f64>,
    sample_weight_sum: Option<f64>,
    left: Option<Box<BrowserTreeNode>>,
    right: Option<Box<BrowserTreeNode>>,
}

#[derive(Clone, Debug, Serialize)]
struct BrowserForecastModel {
    name: &'static str,
    label: &'static str,
    pipeline: &'static str,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsRequest {
    observations: Vec<BrowserGeostatsObservation>,
    targets: Vec<BrowserGeostatsTarget>,
    #[serde(default)]
    options: BrowserGeostatsOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsObservation {
    x: f64,
    y: f64,
    value: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsTarget {
    x: f64,
    y: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsOptions {
    #[serde(default = "default_geostats_kernel")]
    kernel: String,
    #[serde(default = "default_geostats_range")]
    range: f64,
    #[serde(default = "default_geostats_sill")]
    sill: f64,
    #[serde(default = "default_geostats_nugget")]
    nugget: f64,
    #[serde(default = "default_geostats_neighbors")]
    n_neighbors: usize,
    #[serde(default)]
    anisotropy_angle_degrees: f64,
    #[serde(default = "default_geostats_anisotropy_scaling")]
    anisotropy_scaling: f64,
}

impl Default for BrowserGeostatsOptions {
    fn default() -> Self {
        Self {
            kernel: default_geostats_kernel(),
            range: default_geostats_range(),
            sill: default_geostats_sill(),
            nugget: default_geostats_nugget(),
            n_neighbors: default_geostats_neighbors(),
            anisotropy_angle_degrees: 0.0,
            anisotropy_scaling: default_geostats_anisotropy_scaling(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsResponse {
    predictions: Vec<BrowserGeostatsPrediction>,
    metadata: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeostatsPrediction {
    x: f64,
    y: f64,
    mean: f64,
    variance: f64,
    std: f64,
    neighbor_indices: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoFeatureRequest {
    #[serde(default)]
    planar_routes: Vec<BrowserPlanarRoute>,
    #[serde(default)]
    latlng_routes: Vec<BrowserLatLngRoute>,
    #[serde(default)]
    radial_points: Vec<BrowserNamedPoint>,
    #[serde(default)]
    anchors: Vec<BrowserNamedPoint>,
    #[serde(default = "default_geo_feature_length_scale")]
    length_scale: f64,
    #[serde(default)]
    local_frame: Option<BrowserLocalFrame>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserPlanarRoute {
    label: String,
    origin: [f64; 2],
    destination: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLatLngRoute {
    label: String,
    origin: [f64; 2],
    destination: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNamedPoint {
    label: String,
    point: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLocalFrame {
    origin: [f64; 2],
    axis: [f64; 2],
    points: Vec<BrowserNamedPoint>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGeoFeatureResponse {
    planar: Vec<BrowserBearingFeature>,
    latlng: Vec<BrowserBearingFeature>,
    routes: Vec<BrowserRouteFeature>,
    radial: Vec<BrowserAnchorFeatureRow>,
    rbf: Vec<BrowserAnchorFeatureRow>,
    local_frame: Vec<BrowserLocalFrameFeature>,
    metadata: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserBearingFeature {
    label: String,
    east: Option<f64>,
    north: Option<f64>,
    zero_distance: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRouteFeature {
    label: String,
    mid_x: Option<f64>,
    mid_y: Option<f64>,
    distance: Option<f64>,
    bearing_east: Option<f64>,
    bearing_north: Option<f64>,
    zero_distance: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserAnchorFeatureRow {
    label: String,
    values: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserLocalFrameFeature {
    label: String,
    along_axis: Option<f64>,
    cross_axis: Option<f64>,
    invalid_axis: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphForecastRequest {
    frame: BrowserGraphTemporalFrame,
    #[serde(default)]
    options: BrowserGraphForecastOptions,
    #[serde(default)]
    actual: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphTemporalFrame {
    node_ids: Vec<String>,
    timestamps: Vec<i64>,
    target: Vec<Vec<f64>>,
    adjacency: BrowserCsrAdjacency,
    horizon: usize,
    frequency: String,
    #[serde(default)]
    covariates: Option<Vec<Vec<Vec<f64>>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCsrAdjacency {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphForecastOptions {
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    lookback: Option<usize>,
    #[serde(default = "default_graph_diffusion_steps")]
    diffusion_steps: usize,
    #[serde(default = "default_graph_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_graph_epochs")]
    epochs: usize,
    #[serde(default = "default_graph_learning_rate")]
    learning_rate: f64,
    #[serde(default)]
    attention_heads: Option<usize>,
    #[serde(default)]
    graph_order: Option<usize>,
    #[serde(default)]
    experts: Option<usize>,
    #[serde(default)]
    periodicity: Option<usize>,
    #[serde(default)]
    recent_window: Option<usize>,
    #[serde(default)]
    weight_decay: Option<f64>,
    #[serde(default = "default_graph_teacher_forcing_start")]
    teacher_forcing_start: f64,
    #[serde(default = "default_graph_teacher_forcing_end")]
    teacher_forcing_end: f64,
    #[serde(default = "default_graph_ridge")]
    ridge: f64,
    #[serde(default = "default_backend")]
    backend: String,
}

impl Default for BrowserGraphForecastOptions {
    fn default() -> Self {
        Self {
            profile: None,
            lookback: None,
            diffusion_steps: default_graph_diffusion_steps(),
            hidden_size: default_graph_hidden_size(),
            epochs: default_graph_epochs(),
            learning_rate: default_graph_learning_rate(),
            attention_heads: None,
            graph_order: None,
            experts: None,
            periodicity: None,
            recent_window: None,
            weight_decay: None,
            teacher_forcing_start: default_graph_teacher_forcing_start(),
            teacher_forcing_end: default_graph_teacher_forcing_end(),
            ridge: default_graph_ridge(),
            backend: default_backend(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserGraphForecastResponse {
    predictions: Vec<Vec<f64>>,
    node_ids: Vec<String>,
    horizon: usize,
    metrics: Option<Value>,
    metadata: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserMarketStructureRequest {
    lane_ids: Vec<String>,
    timestamps: Vec<i64>,
    target_names: Vec<String>,
    primary: Vec<Vec<f64>>,
    secondary: Vec<Vec<f64>>,
    origin_ids: Vec<String>,
    destination_ids: Vec<String>,
    coordinates: Vec<Vec<f64>>,
    #[serde(default)]
    hierarchy_groups: Vec<Vec<String>>,
    #[serde(default)]
    calendar: Vec<Vec<f64>>,
    #[serde(default)]
    horizon: usize,
    #[serde(default = "default_market_frequency")]
    frequency: String,
}

fn default_market_frequency() -> String {
    "daily".to_string()
}

