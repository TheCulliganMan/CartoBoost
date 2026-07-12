use cartoboost_core::data::{FeatureSchema, SparseSetColumn};
use cartoboost_core::forecasting::{
    calendar_profile_candidate_prediction as core_calendar_profile_candidate_prediction,
    candidate_complexity_rank as core_candidate_complexity_rank, evaluate_competition_metrics,
    forecast_magnitude_guard_allows,
    include_autostats_candidate as core_include_autostats_candidate,
    lag_origin_consistency_guard as core_lag_origin_consistency_guard,
    missing_target_continuation as core_missing_target_continuation,
    native_auto_raw_candidate_is_confident as core_native_auto_raw_candidate_is_confident,
    parse_forecast_timestamp,
    proportional_total_reconciliation as core_proportional_total_reconciliation,
    reference_path_posterior_mean as core_reference_path_posterior_mean,
    reference_path_viterbi as core_reference_path_viterbi,
    relative_loss_displacement_allowed as core_relative_loss_displacement_allowed,
    requires_lag_spine as core_requires_lag_spine,
    seasonal_naive_candidate_prediction as core_seasonal_naive_candidate_prediction,
    selectable_candidate_names as core_selectable_candidate_names,
    shared_candidate_names as core_shared_candidate_names,
    stable_magnitude_candidate_choice as core_stable_magnitude_candidate_choice,
    trend_candidate_prediction as core_trend_candidate_prediction,
    validation_ensemble_weights as core_validation_ensemble_weights,
    validation_unavailable_candidate_choice as core_validation_unavailable_candidate_choice,
    weighted_blend_candidate_forecast as core_weighted_blend_candidate_forecast,
    ArimaForecaster as CoreArimaForecaster, AutoARIMAForecaster as CoreAutoARIMAForecaster,
    AutoForecastConfig as CoreAutoForecastConfig, AutoForecastModel as CoreAutoForecastModel,
    AutoKalmanForecaster as CoreAutoKalmanForecaster,
    AutoLocalLevelKalmanForecaster as CoreAutoLocalLevelKalmanForecaster,
    AutoStatsBank as CoreAutoStatsBank, BacktestFoldResult as CoreBacktestFoldResult,
    BacktestResult as CoreBacktestResult, CalendarFeature,
    CandidateSelectionPolicy as CoreCandidateSelectionPolicy,
    CandidateValidationCutoffSchedule as CoreCandidateValidationCutoffSchedule,
    CartoBoostLagForecaster as CoreCartoBoostLagForecaster, ClassicalExpertValidationObjective,
    CrostonForecaster as CoreCrostonForecaster, ETSForecaster as CoreETSForecaster, ForecastActual,
    ForecastFold as CoreForecastFold, ForecastFrame as CoreForecastFrame, ForecastFrameMetadata,
    ForecastFrequency, ForecastMetricSet as CoreForecastMetricSet,
    ForecastObjective as CoreForecastObjective, ForecastPrediction,
    ForecastResult as CoreForecastResult, ForecastRow as CoreForecastRow, ForecastWindow,
    Forecaster, GlobalForecastTargetMode, KalmanForecaster as CoreKalmanForecaster,
    KrigingForecaster as CoreKrigingForecaster, LagFeatureConfig,
    LocalLevelKalmanForecaster as CoreLocalLevelKalmanForecaster,
    NaiveForecaster as CoreNaiveForecaster,
    OptimizedThetaForecaster as CoreOptimizedThetaForecaster, PiecewiseLinearComponentMode,
    PiecewiseLinearEvent, PiecewiseLinearFitLoss, PiecewiseLinearGrowth,
    PiecewiseLinearRegressorStandardization,
    PiecewiseLinearSeasonalConfig as CorePiecewiseLinearSeasonalConfig,
    PiecewiseLinearSeasonalForecaster as CorePiecewiseLinearSeasonalForecaster,
    PiecewiseLinearSeasonality, PiecewiseLinearTrendUncertaintyPolicy, ReferencePathConfig,
    ReferenceSignal, RollingOriginBacktester as CoreRollingOriginBacktester,
    RollingOriginSplitter as CoreRollingOriginSplitter, SbaForecaster as CoreSbaForecaster,
    SeasonalNaiveForecaster as CoreSeasonalNaiveForecaster, SequenceCandidate,
    SequenceCandidateEnsemble, SequenceCandidatePrediction, SequenceFrame, SequenceGroupPrediction,
    SequenceOofCandidateRow, SequenceOofFold, SequenceSeries, SequenceStateSpaceConfig,
    SpatialPiecewiseKrigingConfig as CoreSpatialPiecewiseKrigingConfig,
    SpatialPiecewiseKrigingForecaster as CoreSpatialPiecewiseKrigingForecaster,
    SpatialPiecewiseKrigingMode, ThetaForecaster as CoreThetaForecaster, ThetaSeasonality,
    TsbForecaster as CoreTsbForecaster,
    WeightedEnsembleForecaster as CoreWeightedEnsembleForecaster,
};
use cartoboost_core::geo::{
    assemble_route_sparse_rows, assemble_sparse_column, assemble_sparse_row,
    expand_h3_sparse_set as core_expand_h3_sparse_set,
    normalize_coordinate as core_normalize_coordinate, normalize_h3_id_text,
    normalize_h3_resolution, normalize_s2_id_text, normalize_s2_level, scaffold_h3_parent_id,
    validate_equal_row_count, validate_parent_levels, GeoGridKind,
};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::manifest::model_manifest_json as core_model_manifest_json;
use cartoboost_core::metrics::{
    aggregate_equal_level_wrmsse as core_aggregate_equal_level_wrmsse,
    calibrated_rank_bucket_probabilities, extreme_portfolio_decisions,
    ordered_nonnegative_weights as core_ordered_nonnegative_weights, portfolio_summary,
    rank_buckets, rank_hit_rates, rank_portfolio_decision_loss, rank_portfolio_summary,
    rank_probability_calibration, rank_scored_assets, rmsse_scale as core_rmsse_scale,
    wrmsse as core_wrmsse, PortfolioAsset, PortfolioDecision, PortfolioSide, RankBucketPrediction,
    WrmsseSeries,
};
use cartoboost_core::tree::{FlatAxisPredictor, FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::utilities::{
    empirical_variogram, fit_local_level_kalman, fit_local_linear_kalman,
    fit_ordinary_kriging_variogram, intermittent_demand_forecast, local_level_kalman_forecast,
    local_level_kalman_forecast_distribution, local_linear_kalman_forecast,
    local_linear_kalman_forecast_distribution, ordinary_kriging_leave_one_out,
    ordinary_kriging_leave_one_out_diagnostics, ordinary_kriging_predict_many,
    IntermittentDemandMethod, KrigingDrift, KrigingObservation, KrigingVariogramModel,
    LocalLevelKalmanConfig, LocalLinearKalmanConfig, OrdinaryKrigingConfig,
};
use cartoboost_core::{
    Booster, BoosterConfig, CartoBoostError, CategoricalEncoder, CategoricalEncodingConfig,
    ClassificationObjective, Classifier, ClassifierConfig, ClassifierModel, Dataset, Model, Ranker,
    RankerConfig, RankerModel, RankingObjective,
};
use cartoboost_geo_causal::{
    causal_representation_report_json as core_geo_causal_representation_report_json,
    spillover_diagnostics as core_geo_causal_spillover_diagnostics, GeoCausalPanel, GeoCausalRow,
    GeoExperimentDesigner as CoreGeoExperimentDesigner, SpatialPlaceboTester, SpatialWeight,
    SyntheticDIDConfig, SyntheticDIDEstimator as CoreSyntheticDIDEstimator,
};
use cartoboost_geo_core::{
    buffered_spatial_cv as core_buffered_spatial_cv, group_spatial_cv as core_group_spatial_cv,
    rolling_origin_panel_split as core_rolling_origin_panel_split,
    spatial_block_cv as core_spatial_block_cv,
    spatial_temporal_blocked_split as core_spatial_temporal_blocked_split,
    CoordinateMatrix as CoreCoordinateMatrix, GeoFrameMeta as CoreGeoFrameMeta,
    PanelIndex as CorePanelIndex, SpatialWeights as CoreGeoSpatialWeights,
    SplitManifest as CoreSplitManifest, TimeIndex as CoreTimeIndex,
};
use cartoboost_geo_st::{
    available_compute_backends as graph_st_available_compute_backends,
    select_compute_backend as graph_st_select_compute_backend, CsrAdjacency as CoreStCsrAdjacency,
    DcrnnConfig as CoreDcrnnConfig, DcrnnForecaster as CoreDcrnnForecaster,
    DelayAwareGraphConfig as CoreDelayAwareGraphConfig,
    DelayAwareGraphTransformer as CoreDelayAwareGraphTransformer,
    GraphTemporalFrame as CoreGraphTemporalFrame, GraphWaveNetConfig as CoreGraphWaveNetConfig,
    GraphWaveNetForecaster as CoreGraphWaveNetForecaster, STAEformerConfig as CoreSTAEformerConfig,
    STAEformerForecaster as CoreSTAEformerForecaster,
};
use cartoboost_geostats::{
    empirical_semivariogram as geostats_empirical_semivariogram,
    fit_variogram_wls as geostats_fit_variogram_wls, Anisotropy as CoreGeostatsAnisotropy,
    CovarianceKernel as CoreCovarianceKernel,
    NearestNeighborGPRegressor as CoreNearestNeighborGPRegressor, NngpConfig as CoreNngpConfig,
};
use cartoboost_neural::{
    available_backends as neural_available_backends,
    backend_dispatch_report as neural_backend_dispatch_report, build_embedding_table_artifact,
    choice_set_transformer_report_json as core_choice_set_transformer_report_json,
    compute_directional_features,
    constrained_decision_select_with_options as core_deep_constrained_decision_select,
    directional_pair_fit as core_deep_directional_pair_fit,
    directional_pair_predict as core_deep_directional_pair_predict,
    directional_pair_predictions as core_deep_directional_pair_predictions,
    event_outcome_fit_with_backend as core_deep_event_outcome_fit,
    event_outcome_predict as core_deep_event_outcome_predict, fit_embedding_table_with_options,
    materialize_source_target_pair_nodes,
    response_curve_fit_with_backend as core_deep_response_curve_fit,
    response_curve_predict as core_deep_response_curve_predict,
    select_backend as neural_select_backend,
    service_residual_fit_with_backend as core_deep_service_residual_fit,
    service_residual_predict as core_deep_service_residual_predict,
    temporal_entity_fit as core_deep_temporal_entity_fit,
    temporal_entity_predict as core_deep_temporal_entity_predict, validate_directed_metapath,
    write_embedding_table_artifact, ArtifactFallbackKind, DeepDirectionalPairArtifact,
    DeepDirectionalPairRow, DeepEventArtifact, DeepResponseArtifact, DeepResponseRow,
    DeepServiceResidualArtifact, DeepServiceResidualRow, DeepTemporalEntityArtifact,
    DirectionalPairFitOptions, EmbeddingTable, GraphSageConfig, GraphSageEncoder,
    GraphSageLinkPredictor, GraphSageRegressor, HeteroGraph, HeteroGraphSageConfig,
    HeteroGraphSageEncoder, HeteroGraphSageLinkPredictor, HeteroGraphSageRegressor,
    HeteroTypedEdge, HinSageConfig, HinSageEncoder, HinSageGraph, HinSageLinkPredictor,
    HinSageRegressor, HomogeneousGraph,
    NeuralEmbeddingRegressor as StandaloneNeuralEmbeddingRegressor, Node2VecConfig,
    Node2VecEncoder, Node2VecLinkPredictor, Node2VecRegressor, StandaloneBoosterConfig,
};
use cartoboost_neural::{
    graph_neural_operator_predict_json as core_graph_neural_operator_predict_json,
    neural_operator_synthetic_benchmark_json as core_neural_operator_synthetic_benchmark_json,
    SpatialOperatorEdge as CoreSpatialOperatorEdge,
};
use cartoboost_neural::{
    ComponentMode as CoreNeuralPanelComponentMode,
    LaneNeuralPanelConfig as CoreLaneNeuralPanelConfig,
    LaneNeuralPanelForecaster as CoreLaneNeuralPanelForecaster, NBeatsConfig as CoreNBeatsConfig,
    NBeatsForecaster as CoreNBeatsForecaster, NHiTSConfig as CoreNHiTSConfig,
    NHiTSForecaster as CoreNHiTSForecaster, NeuralPanelConfig as CoreNeuralPanelConfig,
    NeuralPanelForecaster as CoreNeuralPanelForecaster, NeuralPanelLoss as CoreNeuralPanelLoss,
    NeuralPanelMode as CoreNeuralPanelMode, TrendMode as CoreNeuralPanelTrendMode,
};
use cartoboost_prob::{
    benchmark_calibration_report_fields as core_prob_benchmark_calibration_report_fields,
    conditional_flow_fit_json as core_prob_conditional_flow_fit_json,
    conditional_flow_predict_json as core_prob_conditional_flow_predict_json,
    crps_approximation as core_prob_crps_approximation,
    diffusion_scenario_generate_json as core_prob_diffusion_scenario_generate_json,
    group_conformal_residual_quantiles as core_prob_group_conformal_residual_quantiles,
    interval_coverage as core_prob_interval_coverage,
    mean_interval_width as core_prob_mean_interval_width,
    nearest_calibration_residual_quantiles as core_prob_nearest_calibration_residual_quantiles,
    pinball_loss as core_prob_pinball_loss, pit_bins as core_prob_pit_bins,
    rolling_origin_conformal_residual_quantiles as core_prob_rolling_origin_conformal_residual_quantiles,
    split_conformal_residual_quantile as core_prob_split_conformal_residual_quantile,
    weighted_conformal_residual_quantile as core_prob_weighted_conformal_residual_quantile,
    weighted_interval_score as core_prob_weighted_interval_score,
    DiffusionEdge as CoreDiffusionEdge, SplitOrder as CoreProbSplitOrder,
};
use cartoboost_spatial_econ::{
    spatial_weights_from_coo, SpatialEconError, SpatialModelKind, SpatialRegressionModel,
    SpatialWeights,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyType};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

type StringTypedEdges = Vec<(String, String, String)>;
type PyWrmsseSeries = (String, Vec<f64>, Vec<f64>, Vec<f64>, f64);
type CustomSeasonalitySpec = (String, f64, usize, Option<String>);
type PyPortfolioDecisionRow = (String, String, f64, f64, f64);
type PyKrigingPrediction = (f64, f64, f64, Vec<f64>);
type PyDetailedKrigingPrediction = (f64, f64, f64, f64, Vec<f64>, Vec<usize>);
type PyNngpPrediction = (Vec<f64>, Vec<f64>, Vec<Vec<usize>>);
type PyPiecewiseEvent = (String, String, Option<i32>, Option<i32>);
type PyPiecewiseSeasonality = (
    String,
    f64,
    usize,
    Option<String>,
    Option<String>,
    Option<f64>,
);
type PyGeoCausalRow = (
    String,
    String,
    f64,
    bool,
    BTreeMap<String, f64>,
    Option<f64>,
    Option<f64>,
    Option<String>,
);

#[pyclass(name = "CoordinateMatrix")]
#[derive(Clone, Debug)]
struct NativeCoordinateMatrix {
    inner: CoreCoordinateMatrix,
}

#[pymethods]
impl NativeCoordinateMatrix {
    #[new]
    #[pyo3(signature = (x, y, crs=None, id_col=None))]
    fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        crs: Option<String>,
        id_col: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCoordinateMatrix::new(x, y, crs, id_col).map_err(to_py_geo_core_error)?,
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCoordinateMatrix::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "TimeIndex")]
#[derive(Clone, Debug)]
struct NativeTimeIndex {
    inner: CoreTimeIndex,
}

#[pymethods]
impl NativeTimeIndex {
    #[new]
    #[pyo3(signature = (timestamps, frequency=None, timezone=None))]
    fn new(
        timestamps: Vec<String>,
        frequency: Option<String>,
        timezone: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreTimeIndex::new(timestamps, frequency, timezone)
                .map_err(to_py_geo_core_error)?,
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn timestamps(&self) -> Vec<String> {
        self.inner.iso_strings()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreTimeIndex::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "PanelIndex")]
#[derive(Clone, Debug)]
struct NativePanelIndex {
    inner: CorePanelIndex,
}

#[pymethods]
impl NativePanelIndex {
    #[new]
    #[pyo3(signature = (entity_ids, time=None))]
    fn new(entity_ids: Vec<String>, time: Option<&NativeTimeIndex>) -> PyResult<Self> {
        Ok(Self {
            inner: CorePanelIndex::new(entity_ids, time.map(|value| value.inner.clone()))
                .map_err(to_py_geo_core_error)?,
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CorePanelIndex::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "GeoSpatialWeights")]
#[derive(Clone, Debug)]
struct NativeGeoSpatialWeights {
    inner: CoreGeoSpatialWeights,
}

#[pymethods]
impl NativeGeoSpatialWeights {
    #[new]
    #[pyo3(signature = (n_nodes, indptr, indices, data, node_ids=None, row_normalized=false))]
    fn new(
        n_nodes: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        node_ids: Option<Vec<String>>,
        row_normalized: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGeoSpatialWeights::new(
                n_nodes,
                indptr,
                indices,
                data,
                node_ids,
                row_normalized,
            )
            .map_err(to_py_geo_core_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n_nodes, edges, symmetric=false))]
    fn from_edges(
        n_nodes: usize,
        edges: Vec<(usize, usize, f64)>,
        symmetric: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGeoSpatialWeights::from_edges(n_nodes, edges, symmetric)
                .map_err(to_py_geo_core_error)?,
        })
    }

    fn row_normalize(&self) -> Self {
        Self {
            inner: self.inner.row_normalize(),
        }
    }

    fn is_symmetric(&self, tolerance: f64) -> bool {
        self.inner.is_symmetric(tolerance)
    }

    fn isolated_nodes(&self) -> Vec<usize> {
        self.inner.isolated_nodes()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreGeoSpatialWeights::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyclass(name = "SplitManifest")]
#[derive(Clone, Debug)]
struct NativeSplitManifest {
    inner: CoreSplitManifest,
}

#[pymethods]
impl NativeSplitManifest {
    fn hash(&self) -> PyResult<String> {
        self.inner.hash().map_err(to_py_geo_core_error)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(to_py_geo_core_error)
    }

    fn folds(&self) -> Vec<(String, Vec<usize>, Vec<usize>)> {
        self.inner
            .folds
            .iter()
            .map(|fold| {
                (
                    fold.fold_id.clone(),
                    fold.train_indices.clone(),
                    fold.test_indices.clone(),
                )
            })
            .collect()
    }

    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreSplitManifest::from_json_str(value).map_err(to_py_geo_core_error)?,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (coords, n_folds, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="spatial_block_cv"))]
#[allow(clippy::too_many_arguments)]
fn geo_spatial_block_cv(
    coords: &NativeCoordinateMatrix,
    n_folds: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        coords.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_spatial_block_cv(&coords.inner, n_folds, meta, split_id.to_string())
            .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (coords, n_folds, buffer_distance, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="buffered_spatial_cv"))]
#[allow(clippy::too_many_arguments)]
fn geo_buffered_spatial_cv(
    coords: &NativeCoordinateMatrix,
    n_folds: usize,
    buffer_distance: f64,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        coords.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_buffered_spatial_cv(
            &coords.inner,
            n_folds,
            buffer_distance,
            meta,
            split_id.to_string(),
        )
        .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (groups, n_folds, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="group_spatial_cv"))]
#[allow(clippy::too_many_arguments)]
fn geo_group_spatial_cv(
    groups: Vec<String>,
    n_folds: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        groups.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_group_spatial_cv(groups, n_folds, meta, split_id.to_string())
            .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (panel, min_train_size, horizon, step, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="rolling_origin_panel_split"))]
#[allow(clippy::too_many_arguments)]
fn geo_rolling_origin_panel_split(
    panel: &NativePanelIndex,
    min_train_size: usize,
    horizon: usize,
    step: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        panel.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_rolling_origin_panel_split(
            &panel.inner,
            min_train_size,
            horizon,
            step,
            meta,
            split_id.to_string(),
        )
        .map_err(to_py_geo_core_error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (coords, time, n_spatial_folds, min_train_time, horizon, dataset_fingerprint, coordinate_crs_note, model_version, dependency_versions, random_seed=None, split_id="spatial_temporal_blocked_split"))]
#[allow(clippy::too_many_arguments)]
fn geo_spatial_temporal_blocked_split(
    coords: &NativeCoordinateMatrix,
    time: &NativeTimeIndex,
    n_spatial_folds: usize,
    min_train_time: usize,
    horizon: usize,
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    split_id: &str,
) -> PyResult<NativeSplitManifest> {
    let meta = geo_meta(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        coords.inner.len(),
        Some(split_id.to_string()),
    )?;
    Ok(NativeSplitManifest {
        inner: core_spatial_temporal_blocked_split(
            &coords.inner,
            &time.inner,
            n_spatial_folds,
            min_train_time,
            horizon,
            meta,
            split_id.to_string(),
        )
        .map_err(to_py_geo_core_error)?,
    })
}

fn geo_meta(
    dataset_fingerprint: String,
    coordinate_crs_note: String,
    model_version: String,
    dependency_versions: BTreeMap<String, String>,
    random_seed: Option<u64>,
    row_count: usize,
    split_id: Option<String>,
) -> PyResult<CoreGeoFrameMeta> {
    CoreGeoFrameMeta::new(
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dependency_versions,
        random_seed,
        row_count,
        split_id,
    )
    .map_err(to_py_geo_core_error)
}

#[pyclass(name = "SpatialWeights")]
#[derive(Clone, Debug)]
struct NativeSpatialWeights {
    weights: SpatialWeights,
}

#[pymethods]
impl NativeSpatialWeights {
    #[new]
    #[pyo3(signature = (n_rows, n_cols, rows, cols, values, row_standardize=true))]
    fn new(
        n_rows: usize,
        n_cols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f64>,
        row_standardize: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            weights: spatial_weights_from_coo(n_rows, n_cols, rows, cols, values, row_standardize)
                .map_err(to_py_spatial_error)?,
        })
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.weights.n_nodes
    }

    fn isolated_rows(&self) -> Vec<usize> {
        self.weights.isolated_nodes()
    }
}

macro_rules! native_spatial_regressor {
    ($name:ident, $py_name:literal, $kind:expr) => {
        #[pyclass(name = $py_name)]
        #[derive(Clone, Debug)]
        struct $name {
            model: Option<SpatialRegressionModel>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self { model: None }
            }

            fn fit(
                &mut self,
                py: Python<'_>,
                x: Vec<Vec<f64>>,
                y: Vec<f64>,
                spatial_weights: &NativeSpatialWeights,
            ) -> PyResult<()> {
                let weights = spatial_weights.weights.clone();
                let model = py
                    .allow_threads(|| SpatialRegressionModel::fit($kind, x, y, &weights))
                    .map_err(to_py_spatial_error)?;
                self.model = Some(model);
                Ok(())
            }

            fn predict(
                &self,
                py: Python<'_>,
                x: Vec<Vec<f64>>,
                spatial_weights: &NativeSpatialWeights,
            ) -> PyResult<Vec<f64>> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                let weights = spatial_weights.weights.clone();
                py.allow_threads(|| model.predict(x, &weights))
                    .map_err(to_py_spatial_error)
            }

            fn diagnostics_json(&self) -> PyResult<String> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                serde_json::to_string(model.diagnostics()).map_err(|err| {
                    PyRuntimeError::new_err(format!("failed to serialize diagnostics: {err}"))
                })
            }

            fn coefficients(&self) -> PyResult<Vec<f64>> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                Ok(model.coefficients().to_vec())
            }

            fn durbin_coefficients(&self) -> PyResult<Vec<f64>> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                Ok(model.durbin_coefficients().to_vec())
            }

            fn intercept(&self) -> PyResult<f64> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                Ok(model.intercept())
            }

            fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
                let model = self.model.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!("{} is not fitted", $py_name))
                })?;
                py.allow_threads(|| model.save(path))
                    .map_err(to_py_spatial_error)
            }

            #[classmethod]
            fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: PathBuf) -> PyResult<Self> {
                let model = py
                    .allow_threads(|| SpatialRegressionModel::load(path))
                    .map_err(to_py_spatial_error)?;
                if model.kind() != $kind {
                    return Err(PyValueError::new_err(format!(
                        "artifact contains {:?}, but {} requires {:?}",
                        model.kind(),
                        $py_name,
                        $kind
                    )));
                }
                Ok(Self { model: Some(model) })
            }
        }
    };
}

native_spatial_regressor!(
    NativeSpatialLagRegressor,
    "SpatialLagRegressor",
    SpatialModelKind::SpatialLag
);
native_spatial_regressor!(
    NativeSpatialErrorRegressor,
    "SpatialErrorRegressor",
    SpatialModelKind::SpatialError
);
native_spatial_regressor!(
    NativeSpatialDurbinRegressor,
    "SpatialDurbinRegressor",
    SpatialModelKind::SpatialDurbin
);
native_spatial_regressor!(
    NativeSpatialTwoStageLeastSquares,
    "SpatialTwoStageLeastSquares",
    SpatialModelKind::SpatialTwoStageLeastSquares
);

#[pyclass(name = "ForecastFrame")]
#[derive(Clone, Debug)]
struct NativeForecastFrame {
    frame: CoreForecastFrame,
}

#[pymethods]
impl NativeForecastFrame {
    #[new]
    #[pyo3(signature = (rows, frequency, timestamp_col=None, target_col=None, series_id_col=None, static_covariates=None, known_future_covariates=None, historical_covariates=None, row_covariates=None, sample_weights=None, sample_weight_col=None, allow_irregular=false, allow_missing_targets=false, allow_missing_covariates=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        rows: Vec<(String, String, f64)>,
        frequency: &str,
        timestamp_col: Option<String>,
        target_col: Option<String>,
        series_id_col: Option<String>,
        static_covariates: Option<Vec<String>>,
        known_future_covariates: Option<Vec<String>>,
        historical_covariates: Option<Vec<String>>,
        row_covariates: Option<Vec<BTreeMap<String, f64>>>,
        sample_weights: Option<Vec<f64>>,
        sample_weight_col: Option<String>,
        allow_irregular: bool,
        allow_missing_targets: bool,
        allow_missing_covariates: bool,
    ) -> PyResult<Self> {
        let frequency = ForecastFrequency::parse(frequency).map_err(to_py_value_error)?;
        let frequency_name = frequency.as_str().to_string();
        let metadata = ForecastFrameMetadata {
            timestamp_col,
            target_col,
            series_id_col,
            static_covariates: static_covariates.unwrap_or_default(),
            known_future_covariates: known_future_covariates.unwrap_or_default(),
            historical_covariates: historical_covariates.unwrap_or_default(),
            allow_irregular,
            allow_missing_targets,
            allow_missing_covariates,
        };
        let frame = py
            .allow_threads(|| {
                let frequency = ForecastFrequency::parse(&frequency_name)?;
                match row_covariates {
                    Some(covariates) => {
                        if covariates.len() != rows.len() {
                            return Err(cartoboost_core::CartoBoostError::InvalidInput(
                                "row_covariates length must match rows length".to_string(),
                            ));
                        }
                        let rows = rows
                            .into_iter()
                            .zip(covariates)
                            .map(|((series_id, timestamp, target), covariates)| {
                                (series_id, timestamp, target, covariates)
                            })
                            .collect();
                        match sample_weights {
                            Some(weights) => {
                                CoreForecastFrame::from_string_rows_with_covariates_and_weights(
                                    rows,
                                    weights,
                                    sample_weight_col,
                                    frequency,
                                    metadata,
                                )
                            }
                            None => CoreForecastFrame::from_string_rows_with_covariates(
                                rows, frequency, metadata,
                            ),
                        }
                    }
                    None => {
                        if sample_weights.is_some() {
                            return Err(cartoboost_core::CartoBoostError::InvalidInput(
                                "sample_weights require row_covariates".to_string(),
                            ));
                        }
                        CoreForecastFrame::from_string_rows(rows, frequency, metadata)
                    }
                }
            })
            .map_err(to_py_value_error)?;
        Ok(Self { frame })
    }

    fn row_count(&self) -> usize {
        self.frame.rows().len()
    }

    fn frequency(&self) -> String {
        self.frame.frequency().as_str().to_string()
    }

    fn series_ids(&self) -> Vec<String> {
        self.frame.series_ids()
    }

    fn metadata_json(&self) -> PyResult<String> {
        self.frame.metadata_json_string().map_err(to_py_value_error)
    }

    fn rows(&self) -> Vec<(String, String, f64)> {
        self.frame
            .rows()
            .iter()
            .map(|row| {
                (
                    row.series_id.clone(),
                    row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    row.target,
                )
            })
            .collect()
    }

    fn row_covariates(&self) -> Vec<BTreeMap<String, f64>> {
        self.frame
            .rows()
            .iter()
            .map(|row| row.covariates.clone())
            .collect()
    }
}

#[pyclass(name = "ForecastResult")]
#[derive(Clone, Debug)]
struct NativeForecastResult {
    result: CoreForecastResult,
}

#[pymethods]
impl NativeForecastResult {
    #[new]
    fn new(
        py: Python<'_>,
        predictions: Vec<(String, String, usize, String, f64)>,
    ) -> PyResult<Self> {
        let result = py
            .allow_threads(|| {
                let predictions = predictions
                    .into_iter()
                    .map(|(series_id, timestamp, horizon, model, mean)| {
                        Ok(ForecastPrediction {
                            series_id,
                            timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(
                                &timestamp,
                            )?,
                            horizon,
                            model,
                            mean,
                        })
                    })
                    .collect::<cartoboost_core::Result<Vec<_>>>()?;
                CoreForecastResult::new(predictions)
            })
            .map_err(to_py_value_error)?;
        Ok(Self { result })
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_string();
        let result = py
            .allow_threads(|| CoreForecastResult::from_json_string(&value))
            .map_err(to_py_value_error)?;
        Ok(Self { result })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.result.to_json_string())
            .map_err(to_py_value_error)
    }

    fn columns(&self) -> Vec<String> {
        self.result.result_columns()
    }

    fn predictions(&self) -> Vec<(String, String, usize, String, f64)> {
        self.result
            .predictions()
            .iter()
            .map(|prediction| {
                (
                    prediction.series_id.clone(),
                    prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    prediction.horizon,
                    prediction.model.clone(),
                    prediction.mean,
                )
            })
            .collect()
    }
}

#[pyclass(name = "ForecastFold")]
#[derive(Clone, Debug)]
struct NativeForecastFold {
    fold: CoreForecastFold,
}

#[pymethods]
impl NativeForecastFold {
    #[getter]
    fn fold_id(&self) -> String {
        self.fold.fold_id.clone()
    }

    #[getter]
    fn train_indices(&self) -> Vec<usize> {
        self.fold.train_indices.clone()
    }

    #[getter]
    fn validation_indices(&self) -> Vec<usize> {
        self.fold.validation_indices.clone()
    }

    #[getter]
    fn train_start(&self) -> String {
        format_forecast_timestamp(self.fold.train_start)
    }

    #[getter]
    fn train_end(&self) -> String {
        format_forecast_timestamp(self.fold.train_end)
    }

    #[getter]
    fn validation_start(&self) -> String {
        format_forecast_timestamp(self.fold.validation_start)
    }

    #[getter]
    fn validation_end(&self) -> String {
        format_forecast_timestamp(self.fold.validation_end)
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.fold.horizon
    }

    #[getter]
    fn step(&self) -> usize {
        self.fold.step
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fold.metadata)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fold).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "RollingOriginSplitter")]
#[derive(Clone, Debug)]
struct NativeRollingOriginSplitter {
    splitter: CoreRollingOriginSplitter,
}

#[pymethods]
impl NativeRollingOriginSplitter {
    #[new]
    #[pyo3(signature = (horizon, step=1, min_train_size=1, max_train_size=None, n_splits=None, window="expanding"))]
    fn new(
        horizon: usize,
        step: usize,
        min_train_size: usize,
        max_train_size: Option<usize>,
        n_splits: Option<usize>,
        window: &str,
    ) -> PyResult<Self> {
        let window = parse_forecast_window(window)?;
        Ok(Self {
            splitter: CoreRollingOriginSplitter::new(
                horizon,
                step,
                min_train_size,
                max_train_size,
                n_splits,
                window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    #[staticmethod]
    fn expanding(horizon: usize, min_train_size: usize) -> PyResult<Self> {
        Ok(Self {
            splitter: CoreRollingOriginSplitter::expanding(horizon, min_train_size)
                .map_err(to_py_value_error)?,
        })
    }

    #[staticmethod]
    fn sliding(horizon: usize, min_train_size: usize, max_train_size: usize) -> PyResult<Self> {
        Ok(Self {
            splitter: CoreRollingOriginSplitter::sliding(horizon, min_train_size, max_train_size)
                .map_err(to_py_value_error)?,
        })
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.splitter.horizon
    }

    #[getter]
    fn step(&self) -> usize {
        self.splitter.step
    }

    #[getter]
    fn min_train_size(&self) -> usize {
        self.splitter.min_train_size
    }

    #[getter]
    fn max_train_size(&self) -> Option<usize> {
        self.splitter.max_train_size
    }

    #[getter]
    fn n_splits(&self) -> Option<usize> {
        self.splitter.n_splits
    }

    #[getter]
    fn window(&self) -> &'static str {
        forecast_window_name(&self.splitter.window)
    }

    fn split(
        &self,
        py: Python<'_>,
        frame: &NativeForecastFrame,
    ) -> PyResult<Vec<NativeForecastFold>> {
        Ok(py
            .allow_threads(|| self.splitter.split(&frame.frame))
            .map_err(to_py_value_error)?
            .into_iter()
            .map(|fold| NativeForecastFold { fold })
            .collect())
    }

    fn n_splits_for_frame(&self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<usize> {
        Ok(self.split(py, frame)?.len())
    }
}

#[pyclass(name = "ForecastMetricSet")]
#[derive(Clone, Debug)]
struct NativeForecastMetricSet {
    metrics: CoreForecastMetricSet,
}

#[pymethods]
impl NativeForecastMetricSet {
    #[new]
    #[pyo3(signature = (mae=0.0, rmse=0.0, normalized_rmse=0.0, wape=0.0, smape=0.0, bias=0.0, mase=None))]
    fn new(
        mae: f64,
        rmse: f64,
        normalized_rmse: f64,
        wape: f64,
        smape: f64,
        bias: f64,
        mase: Option<f64>,
    ) -> Self {
        Self {
            metrics: CoreForecastMetricSet {
                mae,
                rmse,
                normalized_rmse,
                wape,
                smape,
                bias,
                mase,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (forecast, actuals, training_actuals=None, mase_seasonality=None))]
    fn evaluate(
        py: Python<'_>,
        forecast: &NativeForecastResult,
        actuals: Vec<(String, String, usize, f64)>,
        training_actuals: Option<Vec<(String, String, usize, f64)>>,
        mase_seasonality: Option<usize>,
    ) -> PyResult<Self> {
        let actuals = parse_forecast_actuals(actuals)?;
        let training_actuals = parse_forecast_actuals(training_actuals.unwrap_or_default())?;
        let metrics = py
            .allow_threads(|| {
                cartoboost_core::forecasting::evaluate_forecast_with_training(
                    &forecast.result,
                    &actuals,
                    &training_actuals,
                    mase_seasonality,
                )
            })
            .map_err(to_py_value_error)?;
        Ok(Self { metrics })
    }

    #[getter]
    fn mae(&self) -> f64 {
        self.metrics.mae
    }

    #[getter]
    fn rmse(&self) -> f64 {
        self.metrics.rmse
    }

    #[getter]
    fn normalized_rmse(&self) -> f64 {
        self.metrics.normalized_rmse
    }

    #[getter]
    fn wape(&self) -> f64 {
        self.metrics.wape
    }

    #[getter]
    fn smape(&self) -> f64 {
        self.metrics.smape
    }

    #[getter]
    fn bias(&self) -> f64 {
        self.metrics.bias
    }

    #[getter]
    fn mase(&self) -> Option<f64> {
        self.metrics.mase
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.metrics).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyfunction]
#[pyo3(signature = (forecast, actuals, training_actuals=None, mase_seasonality=None))]
fn forecast_evaluate_metrics(
    py: Python<'_>,
    forecast: &NativeForecastResult,
    actuals: Vec<(String, String, usize, f64)>,
    training_actuals: Option<Vec<(String, String, usize, f64)>>,
    mase_seasonality: Option<usize>,
) -> PyResult<NativeForecastMetricSet> {
    NativeForecastMetricSet::evaluate(py, forecast, actuals, training_actuals, mase_seasonality)
}

#[pyclass(name = "BacktestFoldResult")]
#[derive(Clone, Debug)]
struct NativeBacktestFoldResult {
    result: CoreBacktestFoldResult,
}

#[pymethods]
impl NativeBacktestFoldResult {
    #[getter]
    fn fold(&self) -> NativeForecastFold {
        NativeForecastFold {
            fold: self.result.fold.clone(),
        }
    }

    #[getter]
    fn metrics(&self) -> NativeForecastMetricSet {
        NativeForecastMetricSet {
            metrics: self.result.metrics.clone(),
        }
    }

    #[getter]
    fn predictions(&self) -> Vec<(String, String, usize, String, f64)> {
        self.result
            .predictions
            .iter()
            .map(forecast_prediction_tuple)
            .collect()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "BacktestResult")]
#[derive(Clone, Debug)]
struct NativeBacktestResult {
    result: CoreBacktestResult,
}

#[pymethods]
impl NativeBacktestResult {
    #[getter]
    fn folds(&self) -> Vec<NativeBacktestFoldResult> {
        self.result
            .folds
            .iter()
            .cloned()
            .map(|result| NativeBacktestFoldResult { result })
            .collect()
    }

    #[getter]
    fn metrics(&self) -> Option<NativeForecastMetricSet> {
        self.result
            .metrics
            .clone()
            .map(|metrics| NativeForecastMetricSet { metrics })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "RollingOriginBacktester")]
#[derive(Clone, Debug)]
struct NativeRollingOriginBacktester {
    backtester: CoreRollingOriginBacktester,
}

#[pymethods]
impl NativeRollingOriginBacktester {
    #[new]
    #[pyo3(signature = (splitter, mase_seasonality=None))]
    fn new(
        splitter: &NativeRollingOriginSplitter,
        mase_seasonality: Option<usize>,
    ) -> PyResult<Self> {
        let mut backtester = CoreRollingOriginBacktester::new(splitter.splitter.clone());
        if let Some(seasonality) = mase_seasonality {
            backtester = backtester
                .with_mase_seasonality(seasonality)
                .map_err(to_py_value_error)?;
        }
        Ok(Self { backtester })
    }

    #[getter]
    fn splitter(&self) -> NativeRollingOriginSplitter {
        NativeRollingOriginSplitter {
            splitter: self.backtester.splitter.clone(),
        }
    }

    #[getter]
    fn mase_seasonality(&self) -> Option<usize> {
        self.backtester.mase_seasonality
    }

    fn run_naive(
        &self,
        py: Python<'_>,
        model: &NativeNaiveForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_seasonal_naive(
        &self,
        py: Python<'_>,
        model: &NativeSeasonalNaiveForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_theta(
        &self,
        py: Python<'_>,
        model: &NativeThetaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_optimized_theta(
        &self,
        py: Python<'_>,
        model: &NativeOptimizedThetaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_ets(
        &self,
        py: Python<'_>,
        model: &NativeETSForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_arima(
        &self,
        py: Python<'_>,
        model: &NativeArimaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_auto_arima(
        &self,
        py: Python<'_>,
        model: &NativeAutoARIMAForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_auto_forecast(
        &self,
        py: Python<'_>,
        model: &NativeAutoForecastModel,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_cartoboost_lag(
        &self,
        py: Python<'_>,
        model: &NativeCartoBoostLagForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }
}

#[pyfunction]
fn forecast_parse_frequency(value: &str) -> PyResult<String> {
    Ok(ForecastFrequency::parse(value)
        .map_err(to_py_value_error)?
        .as_str()
        .to_string())
}

#[pyfunction]
#[pyo3(signature = (values, level_process_variance=0.05, trend_process_variance=0.005, observation_variance=1.0, horizon=0, interval_z=1.959963984540054))]
fn utility_kalman_filter(
    py: Python<'_>,
    values: Vec<f64>,
    level_process_variance: f64,
    trend_process_variance: f64,
    observation_variance: f64,
    horizon: usize,
    interval_z: f64,
) -> PyResult<String> {
    let config = LocalLinearKalmanConfig::new(
        level_process_variance,
        trend_process_variance,
        observation_variance,
    )
    .map_err(to_py_value_error)?;
    let (result, forecast, forecast_distribution) = py
        .allow_threads(|| {
            let result = fit_local_linear_kalman(&values, config)?;
            let forecast = if horizon == 0 {
                Vec::new()
            } else {
                local_linear_kalman_forecast(result.final_state, horizon)?
            };
            let forecast_distribution = if horizon == 0 {
                Vec::new()
            } else {
                local_linear_kalman_forecast_distribution(
                    result.final_state,
                    result.final_covariance,
                    config,
                    horizon,
                    interval_z,
                )?
            };
            Ok((result, forecast, forecast_distribution))
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "final_state": {
            "level": result.final_state.level,
            "trend": result.final_state.trend,
            "covariance": result.final_covariance,
        },
        "estimates": result.estimates.iter().map(|estimate| {
            json!({
                "step": estimate.step,
                "observed": estimate.observed,
                "prior_level": estimate.prior_level,
                "prior_trend": estimate.prior_trend,
                "prior_level_variance": estimate.prior_level_variance,
                "prior_trend_variance": estimate.prior_trend_variance,
                "prior_covariance": estimate.prior_covariance,
                "level": estimate.level,
                "trend": estimate.trend,
                "level_variance": estimate.level_variance,
                "trend_variance": estimate.trend_variance,
                "covariance": estimate.covariance,
                "fitted": estimate.prior_level,
                "residual": estimate.innovation,
                "innovation": estimate.innovation,
                "innovation_variance": estimate.innovation_variance,
                "standardized_innovation": estimate.innovation / estimate.innovation_variance.sqrt(),
                "level_gain": estimate.level_gain,
                "trend_gain": estimate.trend_gain,
                "log_likelihood": estimate.log_likelihood,
            })
        }).collect::<Vec<_>>(),
        "smoothed_states": result.smoothed_states.iter().map(|state| {
            json!({
                "step": state.step,
                "level": state.level,
                "trend": state.trend,
                "covariance": state.covariance,
            })
        }).collect::<Vec<_>>(),
        "forecast": forecast,
        "forecast_distribution": forecast_distribution.iter().map(|point| {
            json!({
                "step": point.step,
                "mean": point.mean,
                "variance": point.variance,
                "lower": point.lower,
                "upper": point.upper,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "log_likelihood": result.log_likelihood,
            "interval_z": interval_z,
            "observation_count": result.residual_summary.observation_count,
            "fitted_count": result.residual_summary.fitted_count,
            "aic": result.residual_summary.aic,
            "bic": result.residual_summary.bic,
            "mse": result.residual_summary.mse,
            "rmse": result.residual_summary.rmse,
            "mae": result.residual_summary.mae,
            "mean_standardized_innovation": result.residual_summary.mean_standardized_innovation,
            "max_abs_standardized_innovation": result.residual_summary.max_abs_standardized_innovation,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (values, level_process_variance=0.05, observation_variance=1.0, horizon=0, interval_z=1.959963984540054))]
fn utility_local_level_kalman_filter(
    py: Python<'_>,
    values: Vec<f64>,
    level_process_variance: f64,
    observation_variance: f64,
    horizon: usize,
    interval_z: f64,
) -> PyResult<String> {
    let config = LocalLevelKalmanConfig::new(level_process_variance, observation_variance)
        .map_err(to_py_value_error)?;
    let (result, forecast, forecast_distribution) = py
        .allow_threads(|| {
            let result = fit_local_level_kalman(&values, config)?;
            let forecast = if horizon == 0 {
                Vec::new()
            } else {
                local_level_kalman_forecast(result.final_level, horizon)?
            };
            let forecast_distribution = if horizon == 0 {
                Vec::new()
            } else {
                local_level_kalman_forecast_distribution(
                    result.final_level,
                    result.final_variance,
                    config,
                    horizon,
                    interval_z,
                )?
            };
            Ok((result, forecast, forecast_distribution))
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "final_state": {
            "level": result.final_level,
            "variance": result.final_variance,
        },
        "estimates": result.estimates.iter().map(|estimate| {
            json!({
                "step": estimate.step,
                "observed": estimate.observed,
                "prior_level": estimate.prior_level,
                "prior_variance": estimate.prior_variance,
                "level": estimate.level,
                "variance": estimate.variance,
                "fitted": estimate.prior_level,
                "residual": estimate.innovation,
                "innovation": estimate.innovation,
                "innovation_variance": estimate.innovation_variance,
                "standardized_innovation": estimate.innovation / estimate.innovation_variance.sqrt(),
                "gain": estimate.gain,
                "log_likelihood": estimate.log_likelihood,
            })
        }).collect::<Vec<_>>(),
        "smoothed_states": result.smoothed_states.iter().map(|state| {
            json!({
                "step": state.step,
                "level": state.level,
                "variance": state.variance,
            })
        }).collect::<Vec<_>>(),
        "forecast": forecast,
        "forecast_distribution": forecast_distribution.iter().map(|point| {
            json!({
                "step": point.step,
                "mean": point.mean,
                "variance": point.variance,
                "lower": point.lower,
                "upper": point.upper,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "log_likelihood": result.log_likelihood,
            "interval_z": interval_z,
            "observation_count": result.residual_summary.observation_count,
            "fitted_count": result.residual_summary.fitted_count,
            "aic": result.residual_summary.aic,
            "bic": result.residual_summary.bic,
            "mse": result.residual_summary.mse,
            "rmse": result.residual_summary.rmse,
            "mae": result.residual_summary.mae,
            "mean_standardized_innovation": result.residual_summary.mean_standardized_innovation,
            "max_abs_standardized_innovation": result.residual_summary.max_abs_standardized_innovation,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (values, horizon, method="croston", alpha=0.1, beta=0.1))]
fn utility_intermittent_demand_forecast(
    py: Python<'_>,
    values: Vec<f64>,
    horizon: usize,
    method: &str,
    alpha: f64,
    beta: f64,
) -> PyResult<Vec<f64>> {
    let method = match method {
        "croston" => IntermittentDemandMethod::Croston,
        "sba" => IntermittentDemandMethod::Sba,
        "tsb" => IntermittentDemandMethod::Tsb,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported intermittent demand method {other:?}"
            )));
        }
    };
    py.allow_threads(|| intermittent_demand_forecast(&values, horizon, alpha, beta, method))
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (observations, targets, range=1.0, nugget=1.0e-6))]
fn utility_ordinary_kriging_predict(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    targets: Vec<(f64, f64)>,
    range: f64,
    nugget: f64,
) -> PyResult<Vec<PyKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = OrdinaryKrigingConfig::new(range, nugget).map_err(to_py_value_error)?;
    let predictions = py
        .allow_threads(|| ordinary_kriging_predict_many(&observations, &targets, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.weights,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    targets,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_predict_detailed(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    targets: Vec<(f64, f64)>,
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<Vec<PyDetailedKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let predictions = py
        .allow_threads(|| ordinary_kriging_predict_many(&observations, &targets, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.variance,
                prediction.weights,
                prediction.neighbor_indices,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_leave_one_out(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<Vec<PyDetailedKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let predictions = py
        .allow_threads(|| ordinary_kriging_leave_one_out(&observations, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.variance,
                prediction.weights,
                prediction.neighbor_indices,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    bin_count=10,
    max_distance=None,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0
))]
fn utility_empirical_variogram(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let bins = py
        .allow_threads(|| {
            empirical_variogram(
                &observations,
                bin_count,
                max_distance,
                anisotropy_angle_degrees,
                anisotropy_scaling,
            )
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "bins": bins.iter().map(|bin| {
            json!({
                "lag_min": bin.lag_min,
                "lag_max": bin.lag_max,
                "lag_center": bin.lag_center,
                "mean_distance": bin.mean_distance,
                "semivariance": bin.semivariance,
                "pair_count": bin.pair_count,
            })
        }).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    variogram_models=None,
    range_candidates=None,
    nugget_candidates=None,
    sill_candidates=None,
    bin_count=10,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0
))]
#[allow(clippy::too_many_arguments)]
fn utility_fit_ordinary_kriging_variogram(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    variogram_models: Option<Vec<String>>,
    range_candidates: Option<Vec<f64>>,
    nugget_candidates: Option<Vec<f64>>,
    sill_candidates: Option<Vec<f64>>,
    bin_count: usize,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let models = variogram_models
        .unwrap_or_default()
        .iter()
        .map(|model| parse_kriging_variogram_model(model))
        .collect::<PyResult<Vec<_>>>()?;
    let ranges = range_candidates.unwrap_or_default();
    let nuggets = nugget_candidates.unwrap_or_default();
    let sills = sill_candidates.unwrap_or_default();
    let fit = py
        .allow_threads(|| {
            fit_ordinary_kriging_variogram(
                &observations,
                &models,
                &ranges,
                &nuggets,
                &sills,
                bin_count,
                anisotropy_angle_degrees,
                anisotropy_scaling,
            )
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "config": kriging_config_json(fit.config),
        "weighted_sse": fit.weighted_sse,
        "bins": fit.bins.iter().map(|bin| {
            json!({
                "lag_min": bin.lag_min,
                "lag_max": bin.lag_max,
                "lag_center": bin.lag_center,
                "mean_distance": bin.mean_distance,
                "semivariance": bin.semivariance,
                "pair_count": bin.pair_count,
            })
        }).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_leave_one_out_diagnostics(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let (predictions, diagnostics) = py
        .allow_threads(|| ordinary_kriging_leave_one_out_diagnostics(&observations, config))
        .map_err(to_py_value_error)?;
    let payload = json!({
        "predictions": predictions.iter().map(|prediction| {
            json!({
                "x": prediction.x,
                "y": prediction.y,
                "mean": prediction.mean,
                "variance": prediction.variance,
                "weights": prediction.weights,
                "neighbor_indices": prediction.neighbor_indices,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "observation_count": diagnostics.observation_count,
            "mean_error": diagnostics.mean_error,
            "mae": diagnostics.mae,
            "rmse": diagnostics.rmse,
            "mean_standardized_error": diagnostics.mean_standardized_error,
            "rmse_standardized_error": diagnostics.rmse_standardized_error,
            "max_abs_standardized_error": diagnostics.max_abs_standardized_error,
            "interval_coverage_95": diagnostics.interval_coverage_95,
            "average_variance": diagnostics.average_variance,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (model, values, horizon, params_json=None))]
fn utility_series_forecast(
    py: Python<'_>,
    model: &str,
    values: Vec<f64>,
    horizon: usize,
    params_json: Option<&str>,
) -> PyResult<Vec<f64>> {
    let params = match params_json {
        Some(raw) => serde_json::from_str::<Value>(raw).map_err(|err| {
            PyValueError::new_err(format!("params_json must be valid JSON: {err}"))
        })?,
        None => json!({}),
    };
    let frame = utility_frame_from_values(values).map_err(to_py_value_error)?;
    let mut forecaster = utility_forecaster(model, &params).map_err(to_py_value_error)?;
    let result = py
        .allow_threads(|| {
            forecaster.fit(&frame)?;
            forecaster.predict(horizon)
        })
        .map_err(to_py_value_error)?;
    Ok(result
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect())
}

fn utility_frame_from_values(values: Vec<f64>) -> cartoboost_core::Result<CoreForecastFrame> {
    let frequency = ForecastFrequency::Daily;
    let start = cartoboost_core::forecasting::parse_forecast_timestamp("1970-01-01")?;
    let rows = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            Ok(CoreForecastRow::single(
                frequency.advance(start, index)?,
                value,
            ))
        })
        .collect::<cartoboost_core::Result<Vec<_>>>()?;
    CoreForecastFrame::new(rows, frequency)
}

fn utility_forecaster(model: &str, params: &Value) -> cartoboost_core::Result<Box<dyn Forecaster>> {
    match model {
        "naive" => Ok(Box::new(CoreNaiveForecaster::new())),
        "seasonal_naive" | "seasonal-naive" => {
            let season_length = utility_usize_param(params, "season_length")?.unwrap_or(1);
            Ok(Box::new(CoreSeasonalNaiveForecaster::new(season_length)?))
        }
        "theta" => {
            let theta = utility_f64_param(params, "theta")?.unwrap_or(2.0);
            let alpha = utility_f64_param(params, "alpha")?.unwrap_or(0.5);
            Ok(Box::new(CoreThetaForecaster::new(theta, alpha)?))
        }
        "optimized_theta" | "optimized-theta" => {
            let theta_grid =
                utility_f64_vec_param(params, "theta_grid")?.unwrap_or_else(|| vec![1.0, 2.0]);
            let alpha_grid =
                utility_f64_vec_param(params, "alpha_grid")?.unwrap_or_else(|| vec![0.2, 0.5, 0.8]);
            Ok(Box::new(CoreOptimizedThetaForecaster::new(
                theta_grid, alpha_grid,
            )?))
        }
        "ets" => {
            let alpha = utility_f64_param(params, "alpha")?.unwrap_or(0.5);
            let beta = utility_f64_param(params, "beta")?.unwrap_or(0.1);
            let gamma = utility_f64_param(params, "gamma")?;
            let season_length = utility_usize_param(params, "season_length")?;
            Ok(Box::new(CoreETSForecaster::with_additive_seasonality(
                alpha,
                beta,
                gamma,
                season_length,
            )?))
        }
        "arima" => {
            let p = utility_usize_param(params, "p")?.unwrap_or(1);
            let d = utility_usize_param(params, "d")?.unwrap_or(0);
            let q = utility_usize_param(params, "q")?.unwrap_or(0);
            Ok(Box::new(CoreArimaForecaster::new(p, d, q)?))
        }
        "auto_arima" | "auto-arima" => {
            let max_p = utility_usize_param(params, "max_p")?.unwrap_or(3);
            let max_d = utility_usize_param(params, "max_d")?.unwrap_or(1);
            let max_q = utility_usize_param(params, "max_q")?.unwrap_or(2);
            Ok(Box::new(CoreAutoARIMAForecaster::with_max_order(
                max_p, max_d, max_q,
            )?))
        }
        "kalman" | "local_linear_trend_kalman" | "local-linear-trend-kalman" => {
            let level_process_variance =
                utility_f64_param(params, "level_process_variance")?.unwrap_or(0.05);
            let trend_process_variance =
                utility_f64_param(params, "trend_process_variance")?.unwrap_or(0.005);
            let observation_variance =
                utility_f64_param(params, "observation_variance")?.unwrap_or(1.0);
            Ok(Box::new(CoreKalmanForecaster::new(
                level_process_variance,
                trend_process_variance,
                observation_variance,
            )?))
        }
        "auto_kalman" | "self_tuning_kalman" | "self-tuning-kalman" => {
            let level_process_variance_grid =
                utility_f64_vec_param(params, "level_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
            let trend_process_variance_grid =
                utility_f64_vec_param(params, "trend_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]);
            let observation_variance_grid =
                utility_f64_vec_param(params, "observation_variance_grid")?
                    .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
            let validation_window = utility_usize_param(params, "validation_window")?;
            Ok(Box::new(CoreAutoKalmanForecaster::with_grids(
                level_process_variance_grid,
                trend_process_variance_grid,
                observation_variance_grid,
                validation_window,
            )?))
        }
        "local_level_kalman" | "local-level-kalman" => {
            let level_process_variance =
                utility_f64_param(params, "level_process_variance")?.unwrap_or(0.05);
            let observation_variance =
                utility_f64_param(params, "observation_variance")?.unwrap_or(1.0);
            Ok(Box::new(CoreLocalLevelKalmanForecaster::new(
                level_process_variance,
                observation_variance,
            )?))
        }
        "auto_local_level_kalman" | "auto-local-level-kalman" => {
            let level_process_variance_grid =
                utility_f64_vec_param(params, "level_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
            let observation_variance_grid =
                utility_f64_vec_param(params, "observation_variance_grid")?
                    .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
            let validation_window = utility_usize_param(params, "validation_window")?;
            Ok(Box::new(CoreAutoLocalLevelKalmanForecaster::with_grids(
                level_process_variance_grid,
                observation_variance_grid,
                validation_window,
            )?))
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown utility series forecast model {other:?}"
        ))),
    }
}

fn utility_f64_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<f64>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!("parameter {name} must be numeric"))
            })
            .map(Some),
    }
}

fn utility_usize_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<usize>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => {
            let raw = value.as_u64().ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "parameter {name} must be a nonnegative integer"
                ))
            })?;
            usize::try_from(raw)
                .map_err(|_| {
                    CartoBoostError::InvalidInput(format!("parameter {name} is too large"))
                })
                .map(Some)
        }
    }
}

fn utility_f64_vec_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<Vec<f64>>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "parameter {name} must contain only numbers"
                    ))
                })
            })
            .collect::<cartoboost_core::Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(CartoBoostError::InvalidInput(format!(
            "parameter {name} must be a numeric array"
        ))),
    }
}

#[pyclass(name = "NaiveForecaster")]
#[derive(Clone, Debug)]
struct NativeNaiveForecaster {
    model: CoreNaiveForecaster,
}

#[pymethods]
impl NativeNaiveForecaster {
    #[new]
    #[pyo3(signature = (prediction_interval_levels=None))]
    fn new(prediction_interval_levels: Option<Vec<f64>>) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        Ok(Self {
            model: CoreNaiveForecaster::new(),
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "SeasonalNaiveForecaster")]
#[derive(Clone, Debug)]
struct NativeSeasonalNaiveForecaster {
    model: CoreSeasonalNaiveForecaster,
}

#[pymethods]
impl NativeSeasonalNaiveForecaster {
    #[new]
    #[pyo3(signature = (season_length, prediction_interval_levels=None))]
    fn new(season_length: usize, prediction_interval_levels: Option<Vec<f64>>) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        Ok(Self {
            model: CoreSeasonalNaiveForecaster::new(season_length).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "ThetaForecaster")]
#[derive(Clone, Debug)]
struct NativeThetaForecaster {
    model: CoreThetaForecaster,
}

#[pymethods]
impl NativeThetaForecaster {
    #[new]
    #[pyo3(signature = (theta=2.0, alpha=0.2, season_length=None, seasonality=None, prediction_interval_levels=None))]
    fn new(
        theta: f64,
        alpha: f64,
        season_length: Option<usize>,
        seasonality: Option<String>,
        prediction_interval_levels: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let seasonality = parse_theta_seasonality(season_length, seasonality)?;
        Ok(Self {
            model: CoreThetaForecaster::with_seasonality(theta, alpha, seasonality)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "OptimizedThetaForecaster")]
#[derive(Clone, Debug)]
struct NativeOptimizedThetaForecaster {
    model: CoreOptimizedThetaForecaster,
}

#[pymethods]
impl NativeOptimizedThetaForecaster {
    #[new]
    #[pyo3(signature = (theta_grid=None, alpha_grid=None, season_length=None, seasonality=None, prediction_interval_levels=None))]
    fn new(
        theta_grid: Option<Vec<f64>>,
        alpha_grid: Option<Vec<f64>>,
        season_length: Option<usize>,
        seasonality: Option<String>,
        prediction_interval_levels: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let seasonality = parse_theta_seasonality(season_length, seasonality)?;
        Ok(Self {
            model: CoreOptimizedThetaForecaster::with_seasonality(
                theta_grid.unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
                alpha_grid.unwrap_or_else(|| vec![0.1, 0.2, 0.4, 0.6, 0.8]),
                seasonality,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "PiecewiseLinearSeasonalForecaster")]
#[derive(Clone, Debug)]
struct NativePiecewiseLinearSeasonalForecaster {
    model: CorePiecewiseLinearSeasonalForecaster,
}

#[pymethods]
impl NativePiecewiseLinearSeasonalForecaster {
    #[new]
    #[pyo3(signature = (
        growth="linear",
        component_mode="additive",
        changepoints=25,
        changepoint_range=1.0,
        changepoint_timestamps=None,
        yearly_fourier_order=0,
        weekly_fourier_order=3,
        daily_fourier_order=0,
        auto_yearly_seasonality=true,
        auto_weekly_seasonality=true,
        auto_daily_seasonality=true,
        custom_seasonalities=None,
        changepoint_l2_regularization=0.05,
        changepoint_l1_regularization=0.0,
        seasonality_l2_regularization=0.01,
        yearly_l2_regularization=None,
        weekly_l2_regularization=None,
        daily_l2_regularization=None,
        event_l2_regularization=0.01,
        regressor_l2_regularization=0.01,
        event_l2_regularization_by_name=None,
        regressor_l2_regularization_by_name=None,
        events=None,
        event_mode=None,
        extra_regressors=None,
        regressor_modes=None,
        extra_regressor_monotonic_constraints=None,
        regressor_standardization="auto",
        future_regressors=None,
        future_regressors_by_series=None,
        trend_adjustments=None,
        trend_adjustments_by_series=None,
        residual_shock_window=0,
        residual_shock_scale=0.0,
        residual_shock_decay=1.0,
        prediction_interval_levels=None,
        quantile_levels=None,
        uncertainty_samples=0,
        trend_uncertainty_policy="laplace",
        trend_uncertainty_scale=1.0,
        coefficient_uncertainty_scale=1.0,
        uncertainty_seed=14172296343723622691,
        cap=None,
        floor=0.0,
        cap_regressor=None,
        floor_regressor=None,
        fit_loss="squared",
        huber_delta=1.345,
        irls_iterations=5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        growth: &str,
        component_mode: &str,
        changepoints: usize,
        changepoint_range: f64,
        changepoint_timestamps: Option<Vec<String>>,
        yearly_fourier_order: usize,
        weekly_fourier_order: usize,
        daily_fourier_order: usize,
        auto_yearly_seasonality: bool,
        auto_weekly_seasonality: bool,
        auto_daily_seasonality: bool,
        custom_seasonalities: Option<Vec<PyPiecewiseSeasonality>>,
        changepoint_l2_regularization: f64,
        changepoint_l1_regularization: f64,
        seasonality_l2_regularization: f64,
        yearly_l2_regularization: Option<f64>,
        weekly_l2_regularization: Option<f64>,
        daily_l2_regularization: Option<f64>,
        event_l2_regularization: f64,
        regressor_l2_regularization: f64,
        event_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
        regressor_l2_regularization_by_name: Option<BTreeMap<String, f64>>,
        events: Option<Vec<PyPiecewiseEvent>>,
        event_mode: Option<String>,
        extra_regressors: Option<Vec<String>>,
        regressor_modes: Option<BTreeMap<String, String>>,
        extra_regressor_monotonic_constraints: Option<BTreeMap<String, i8>>,
        regressor_standardization: &str,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
        residual_shock_window: usize,
        residual_shock_scale: f64,
        residual_shock_decay: f64,
        prediction_interval_levels: Option<Vec<f64>>,
        quantile_levels: Option<Vec<f64>>,
        uncertainty_samples: usize,
        trend_uncertainty_policy: &str,
        trend_uncertainty_scale: f64,
        coefficient_uncertainty_scale: f64,
        uncertainty_seed: u64,
        cap: Option<f64>,
        floor: f64,
        cap_regressor: Option<String>,
        floor_regressor: Option<String>,
        fit_loss: &str,
        huber_delta: f64,
        irls_iterations: usize,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        validate_interval_levels(quantile_levels.as_deref())?;
        let config = CorePiecewiseLinearSeasonalConfig {
            growth: parse_piecewise_growth(growth)?,
            component_mode: parse_piecewise_component_mode(component_mode)?,
            fit_loss: parse_piecewise_fit_loss(fit_loss)?,
            huber_delta,
            irls_iterations,
            changepoints,
            changepoint_range,
            changepoint_timestamps: parse_piecewise_changepoint_timestamps(changepoint_timestamps)?,
            yearly_fourier_order,
            weekly_fourier_order,
            daily_fourier_order,
            auto_yearly_seasonality,
            auto_weekly_seasonality,
            auto_daily_seasonality,
            custom_seasonalities: parse_piecewise_seasonalities(custom_seasonalities)?,
            changepoint_l2_regularization,
            changepoint_l1_regularization,
            seasonality_l2_regularization,
            yearly_l2_regularization,
            weekly_l2_regularization,
            daily_l2_regularization,
            event_l2_regularization,
            regressor_l2_regularization,
            event_l2_regularization_by_name: event_l2_regularization_by_name.unwrap_or_default(),
            regressor_l2_regularization_by_name: regressor_l2_regularization_by_name
                .unwrap_or_default(),
            events: parse_piecewise_events(events)?,
            event_mode: parse_optional_piecewise_component_mode(event_mode)?,
            extra_regressors: extra_regressors.unwrap_or_default(),
            regressor_modes: parse_piecewise_regressor_modes(regressor_modes)?,
            extra_regressor_monotonic_constraints: extra_regressor_monotonic_constraints
                .unwrap_or_default(),
            regressor_standardization: parse_piecewise_regressor_standardization(
                regressor_standardization,
            )?,
            future_regressors: future_regressors.unwrap_or_default(),
            future_regressors_by_series: future_regressors_by_series.unwrap_or_default(),
            trend_adjustments: trend_adjustments.unwrap_or_default(),
            trend_adjustments_by_series: trend_adjustments_by_series.unwrap_or_default(),
            residual_shock_window,
            residual_shock_scale,
            residual_shock_decay,
            interval_levels: prediction_interval_levels.unwrap_or_default(),
            quantile_levels: quantile_levels.unwrap_or_default(),
            uncertainty_samples,
            trend_uncertainty_policy: parse_piecewise_trend_uncertainty_policy(
                trend_uncertainty_policy,
            )?,
            trend_uncertainty_scale,
            coefficient_uncertainty_scale,
            uncertainty_seed,
            cap,
            floor,
            cap_regressor,
            floor_regressor,
        };
        Ok(Self {
            model: CorePiecewiseLinearSeasonalForecaster::new(config).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    #[pyo3(signature = (horizon, future_regressors=None, future_regressors_by_series=None, prediction_interval_levels=None, uncertainty_samples=None, trend_adjustments=None, trend_adjustments_by_series=None, future_timestamps=None, future_timestamps_by_series=None))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        prediction_interval_levels: Option<Vec<f64>>,
        uncertainty_samples: Option<usize>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
        future_timestamps: Option<Vec<String>>,
        future_timestamps_by_series: Option<BTreeMap<String, Vec<String>>>,
    ) -> PyResult<NativeForecastResult> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            prediction_interval_levels,
            None,
            uncertainty_samples,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        match (future_timestamps, future_timestamps_by_series) {
            (None, None) => predict_forecaster_py(py, &model, horizon),
            (Some(timestamps), None) => {
                let schedule = piecewise_shared_future_timestamps(&model, timestamps, horizon)?;
                forecast_to_py(py.allow_threads(|| model.predict_at_timestamps(schedule)))
            }
            (None, Some(timestamps_by_series)) => {
                let schedule =
                    piecewise_future_timestamps_by_series(timestamps_by_series, horizon)?;
                forecast_to_py(py.allow_threads(|| model.predict_at_timestamps(schedule)))
            }
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "pass either future_timestamps or future_timestamps_by_series, not both",
            )),
        }
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.model.to_json_string())
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (horizon, future_regressors=None, future_regressors_by_series=None, trend_adjustments=None, trend_adjustments_by_series=None))]
    fn components_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    ) -> PyResult<String> {
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            None,
            None,
            None,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        py.allow_threads(|| model.predict_components_json_string(horizon))
            .map_err(to_py_value_error)
    }

    fn history_components_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.model.history_components_json_string())
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (horizon, future_regressors=None, future_regressors_by_series=None, uncertainty_samples=None, trend_adjustments=None, trend_adjustments_by_series=None))]
    #[allow(clippy::too_many_arguments)]
    fn samples_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        uncertainty_samples: Option<usize>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    ) -> PyResult<String> {
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            None,
            None,
            uncertainty_samples,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        py.allow_threads(|| model.predict_samples_json_string(horizon))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (horizon, quantile_levels=None, future_regressors=None, future_regressors_by_series=None, uncertainty_samples=None, trend_adjustments=None, trend_adjustments_by_series=None))]
    #[allow(clippy::too_many_arguments)]
    fn quantiles_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        quantile_levels: Option<Vec<f64>>,
        future_regressors: Option<BTreeMap<String, Vec<f64>>>,
        future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
        uncertainty_samples: Option<usize>,
        trend_adjustments: Option<BTreeMap<usize, f64>>,
        trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
    ) -> PyResult<String> {
        validate_interval_levels(quantile_levels.as_deref())?;
        let model = piecewise_model_with_prediction_overrides(
            &self.model,
            future_regressors,
            future_regressors_by_series,
            None,
            quantile_levels.clone(),
            uncertainty_samples,
            trend_adjustments,
            trend_adjustments_by_series,
        )?;
        py.allow_threads(|| model.predict_quantiles_json_string(horizon, quantile_levels))
            .map_err(to_py_value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let model = py
            .allow_threads(|| CorePiecewiseLinearSeasonalForecaster::from_json_string(value))
            .map_err(to_py_value_error)?;
        Ok(Self { model })
    }
}

#[allow(clippy::too_many_arguments)]
fn piecewise_model_with_prediction_overrides(
    model: &CorePiecewiseLinearSeasonalForecaster,
    future_regressors: Option<BTreeMap<String, Vec<f64>>>,
    future_regressors_by_series: Option<BTreeMap<String, BTreeMap<String, Vec<f64>>>>,
    interval_levels: Option<Vec<f64>>,
    quantile_levels: Option<Vec<f64>>,
    uncertainty_samples: Option<usize>,
    trend_adjustments: Option<BTreeMap<usize, f64>>,
    trend_adjustments_by_series: Option<BTreeMap<String, BTreeMap<usize, f64>>>,
) -> PyResult<CorePiecewiseLinearSeasonalForecaster> {
    let mut model = model.clone();
    model
        .update_config(|config| {
            if let Some(future_regressors) = future_regressors {
                config.future_regressors = future_regressors;
            }
            if let Some(future_regressors_by_series) = future_regressors_by_series {
                config.future_regressors_by_series = future_regressors_by_series;
            }
            if let Some(interval_levels) = interval_levels {
                config.interval_levels = interval_levels;
            }
            if let Some(quantile_levels) = quantile_levels {
                config.quantile_levels = quantile_levels;
            }
            if let Some(uncertainty_samples) = uncertainty_samples {
                config.uncertainty_samples = uncertainty_samples;
            }
            if let Some(trend_adjustments) = trend_adjustments {
                config.trend_adjustments = trend_adjustments;
            }
            if let Some(trend_adjustments_by_series) = trend_adjustments_by_series {
                config.trend_adjustments_by_series = trend_adjustments_by_series;
            }
        })
        .map_err(to_py_value_error)?;
    Ok(model)
}

fn piecewise_shared_future_timestamps(
    model: &CorePiecewiseLinearSeasonalForecaster,
    timestamps: Vec<String>,
    horizon: usize,
) -> PyResult<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
    let parsed = parse_future_timestamps(timestamps)?;
    validate_future_timestamp_count(parsed.len(), horizon)?;
    let series_ids = model.fitted_series_ids().map_err(to_py_value_error)?;
    Ok(series_ids
        .into_iter()
        .map(|series_id| (series_id, parsed.clone()))
        .collect())
}

fn piecewise_future_timestamps_by_series(
    timestamps_by_series: BTreeMap<String, Vec<String>>,
    horizon: usize,
) -> PyResult<BTreeMap<String, Vec<chrono::NaiveDateTime>>> {
    timestamps_by_series
        .into_iter()
        .map(|(series_id, timestamps)| {
            let parsed = parse_future_timestamps(timestamps)?;
            validate_future_timestamp_count(parsed.len(), horizon)?;
            Ok((series_id, parsed))
        })
        .collect()
}

fn parse_future_timestamps(timestamps: Vec<String>) -> PyResult<Vec<chrono::NaiveDateTime>> {
    timestamps
        .into_iter()
        .map(|timestamp| parse_forecast_timestamp(&timestamp).map_err(to_py_value_error))
        .collect()
}

fn validate_future_timestamp_count(count: usize, horizon: usize) -> PyResult<()> {
    if count != horizon {
        return Err(PyValueError::new_err(format!(
            "future_timestamps length must match horizon; got {count} timestamps for horizon {horizon}"
        )));
    }
    Ok(())
}

#[pyclass(name = "ETSForecaster")]
#[derive(Clone, Debug)]
struct NativeETSForecaster {
    model: CoreETSForecaster,
}

#[pymethods]
impl NativeETSForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.5, beta=0.1, gamma=None, season_length=None))]
    fn new(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreETSForecaster::with_additive_seasonality(alpha, beta, gamma, season_length)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn fitted_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.fitted_values(series_id),
            series_id,
            "fitted values",
        )
    }

    fn residuals(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(self.model.residuals(series_id), series_id, "residuals")
    }

    fn level_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.level_values(series_id),
            series_id,
            "level values",
        )
    }

    fn trend_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.trend_values(series_id),
            series_id,
            "trend values",
        )
    }

    fn seasonal_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.seasonal_values(series_id),
            series_id,
            "seasonal values",
        )
    }
}

#[pyclass(name = "ArimaForecaster")]
#[derive(Clone, Debug)]
struct NativeArimaForecaster {
    model: CoreArimaForecaster,
}

#[pymethods]
impl NativeArimaForecaster {
    #[new]
    #[pyo3(signature = (p=1, d=0, q=0))]
    fn new(p: usize, d: usize, q: usize) -> PyResult<Self> {
        Ok(Self {
            model: CoreArimaForecaster::new(p, d, q).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoARIMAForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoARIMAForecaster {
    model: CoreAutoARIMAForecaster,
}

#[pymethods]
impl NativeAutoARIMAForecaster {
    #[new]
    #[pyo3(signature = (max_p=3, max_d=1, max_q=2))]
    fn new(max_p: usize, max_d: usize, max_q: usize) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoARIMAForecaster::with_max_order(max_p, max_d, max_q)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoStatsBank")]
struct NativeAutoStatsBank {
    model: CoreAutoStatsBank,
}

#[pymethods]
impl NativeAutoStatsBank {
    #[new]
    #[pyo3(signature = (season_length, validation_window=None, validation_objective="mean_squared_error"))]
    fn new(
        season_length: usize,
        validation_window: Option<usize>,
        validation_objective: &str,
    ) -> PyResult<Self> {
        let validation_objective =
            parse_classical_validation_objective(validation_objective, season_length)?;
        Ok(Self {
            model: CoreAutoStatsBank::with_validation_objective(
                season_length,
                validation_window,
                validation_objective,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "CrostonForecaster")]
#[derive(Clone, Debug)]
struct NativeCrostonForecaster {
    model: CoreCrostonForecaster,
}

#[pymethods]
impl NativeCrostonForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.2))]
    fn new(alpha: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreCrostonForecaster::new(alpha).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "SbaForecaster")]
#[derive(Clone, Debug)]
struct NativeSbaForecaster {
    model: CoreSbaForecaster,
}

#[pymethods]
impl NativeSbaForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.2))]
    fn new(alpha: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreSbaForecaster::new(alpha).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "TsbForecaster")]
#[derive(Clone, Debug)]
struct NativeTsbForecaster {
    model: CoreTsbForecaster,
}

#[pymethods]
impl NativeTsbForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.2, beta=0.2))]
    fn new(alpha: f64, beta: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreTsbForecaster::new(alpha, beta).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "KalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeKalmanForecaster {
    model: CoreKalmanForecaster,
}

#[pymethods]
impl NativeKalmanForecaster {
    #[new]
    #[pyo3(signature = (level_process_variance=0.05, trend_process_variance=0.005, observation_variance=1.0))]
    fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreKalmanForecaster::new(
                level_process_variance,
                trend_process_variance,
                observation_variance,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "LocalLevelKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeLocalLevelKalmanForecaster {
    model: CoreLocalLevelKalmanForecaster,
}

#[pymethods]
impl NativeLocalLevelKalmanForecaster {
    #[new]
    #[pyo3(signature = (level_process_variance=0.05, observation_variance=1.0))]
    fn new(level_process_variance: f64, observation_variance: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreLocalLevelKalmanForecaster::new(
                level_process_variance,
                observation_variance,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoKalmanForecaster {
    model: CoreAutoKalmanForecaster,
}

#[pymethods]
impl NativeAutoKalmanForecaster {
    #[new]
    #[pyo3(signature = (
        level_process_variance_grid=None,
        trend_process_variance_grid=None,
        observation_variance_grid=None,
        validation_window=None
    ))]
    fn new(
        level_process_variance_grid: Option<Vec<f64>>,
        trend_process_variance_grid: Option<Vec<f64>>,
        observation_variance_grid: Option<Vec<f64>>,
        validation_window: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoKalmanForecaster::with_grids(
                level_process_variance_grid.unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]),
                trend_process_variance_grid.unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]),
                observation_variance_grid.unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]),
                validation_window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoLocalLevelKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoLocalLevelKalmanForecaster {
    model: CoreAutoLocalLevelKalmanForecaster,
}

#[pymethods]
impl NativeAutoLocalLevelKalmanForecaster {
    #[new]
    #[pyo3(signature = (
        level_process_variance_grid=None,
        observation_variance_grid=None,
        validation_window=None
    ))]
    fn new(
        level_process_variance_grid: Option<Vec<f64>>,
        observation_variance_grid: Option<Vec<f64>>,
        validation_window: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoLocalLevelKalmanForecaster::with_grids(
                level_process_variance_grid.unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]),
                observation_variance_grid.unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]),
                validation_window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "KrigingForecaster")]
#[derive(Clone, Debug)]
struct NativeKrigingForecaster {
    model: CoreKrigingForecaster,
}

#[pymethods]
impl NativeKrigingForecaster {
    #[new]
    #[pyo3(signature = (
        coordinates,
        range=1.0,
        nugget=1.0e-9,
        sill=1.0,
        variogram_model="exponential",
        drift="ordinary",
        anisotropy_angle_degrees=0.0,
        anisotropy_scaling=1.0,
        max_neighbors=None,
        min_neighbors=1,
        max_distance=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coordinates: Vec<(String, f64, f64)>,
        range: f64,
        nugget: f64,
        sill: f64,
        variogram_model: &str,
        drift: &str,
        anisotropy_angle_degrees: f64,
        anisotropy_scaling: f64,
        max_neighbors: Option<usize>,
        min_neighbors: usize,
        max_distance: Option<f64>,
    ) -> PyResult<Self> {
        let coordinates = coordinates
            .into_iter()
            .map(|(series_id, x, y)| (series_id, (x, y)))
            .collect::<BTreeMap<_, _>>();
        let config = build_kriging_config(
            range,
            nugget,
            sill,
            variogram_model,
            drift,
            anisotropy_angle_degrees,
            anisotropy_scaling,
            max_neighbors,
            min_neighbors,
            max_distance,
        )?;
        Ok(Self {
            model: CoreKrigingForecaster::with_config(coordinates, config)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "SpatialPiecewiseKrigingForecaster")]
#[derive(Clone, Debug)]
struct NativeSpatialPiecewiseKrigingForecaster {
    model: CoreSpatialPiecewiseKrigingForecaster,
}

#[pymethods]
impl NativeSpatialPiecewiseKrigingForecaster {
    #[new]
    #[pyo3(signature = (
        coordinates,
        mode="residual_kriging",
        spatial_regressors=None,
        range=1.0,
        nugget=1.0e-6,
        sill=1.0,
        variogram_model="exponential",
        drift="ordinary",
        anisotropy_angle_degrees=0.0,
        anisotropy_scaling=1.0,
        max_neighbors=None,
        min_neighbors=1,
        max_distance=None,
        residual_shrinkage=1.0,
        allow_neighbor_fallback=false,
        piecewise_config_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coordinates: Vec<(String, f64, f64)>,
        mode: &str,
        spatial_regressors: Option<Vec<String>>,
        range: f64,
        nugget: f64,
        sill: f64,
        variogram_model: &str,
        drift: &str,
        anisotropy_angle_degrees: f64,
        anisotropy_scaling: f64,
        max_neighbors: Option<usize>,
        min_neighbors: usize,
        max_distance: Option<f64>,
        residual_shrinkage: f64,
        allow_neighbor_fallback: bool,
        piecewise_config_json: Option<String>,
    ) -> PyResult<Self> {
        let coordinates = coordinates
            .into_iter()
            .map(|(series_id, x, y)| (series_id, (x, y)))
            .collect::<BTreeMap<_, _>>();
        let kriging_config = build_kriging_config(
            range,
            nugget,
            sill,
            variogram_model,
            drift,
            anisotropy_angle_degrees,
            anisotropy_scaling,
            max_neighbors,
            min_neighbors,
            max_distance,
        )?;
        let piecewise_config = match piecewise_config_json {
            Some(payload) => CorePiecewiseLinearSeasonalForecaster::from_json_string(&payload)
                .map_err(to_py_value_error)?
                .config()
                .clone(),
            None => CorePiecewiseLinearSeasonalConfig::default(),
        };
        let config = CoreSpatialPiecewiseKrigingConfig {
            coordinates,
            mode: parse_spatial_piecewise_kriging_mode(mode)?,
            piecewise_config,
            kriging_config,
            spatial_regressors: spatial_regressors.unwrap_or_default(),
            residual_shrinkage,
            allow_neighbor_fallback,
        };
        Ok(Self {
            model: CoreSpatialPiecewiseKrigingForecaster::new(config).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "NBeatsForecaster")]
struct NativeNBeatsForecaster {
    model: CoreNBeatsForecaster,
}

#[pyclass(name = "GraphTemporalFrame")]
#[derive(Clone, Debug)]
struct NativeGraphTemporalFrame {
    frame: CoreGraphTemporalFrame,
}

#[pymethods]
impl NativeGraphTemporalFrame {
    #[new]
    #[pyo3(signature = (node_ids, timestamps, target, indptr, indices, data, horizon, frequency, covariates=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        node_ids: Vec<String>,
        timestamps: Vec<i64>,
        target: Vec<Vec<f64>>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        horizon: usize,
        frequency: String,
        covariates: Option<Vec<Vec<Vec<f64>>>>,
    ) -> PyResult<Self> {
        let adjacency = CoreStCsrAdjacency::new(indptr, indices, data, node_ids.len())
            .map_err(to_py_geo_st_error)?;
        Ok(Self {
            frame: CoreGraphTemporalFrame::new(
                node_ids, timestamps, target, covariates, adjacency, horizon, frequency,
            )
            .map_err(to_py_geo_st_error)?,
        })
    }

    #[getter]
    fn node_ids(&self) -> Vec<String> {
        self.frame.node_ids.clone()
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.frame.horizon
    }

    #[getter]
    fn frequency(&self) -> String {
        self.frame.frequency.clone()
    }
}

#[pyclass(name = "DCRNNForecaster")]
#[derive(Clone, Debug)]
struct NativeDcrnnForecaster {
    model: CoreDcrnnForecaster,
}

#[pymethods]
impl NativeDcrnnForecaster {
    #[new]
    #[pyo3(signature = (
        diffusion_steps=2,
        hidden_size=8,
        epochs=160,
        learning_rate=0.03,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.2,
        ridge=0.0001,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        diffusion_steps: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        teacher_forcing_start: f64,
        teacher_forcing_end: f64,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreDcrnnForecaster::new(CoreDcrnnConfig {
                diffusion_steps,
                hidden_size,
                epochs,
                learning_rate,
                teacher_forcing_start,
                teacher_forcing_end,
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn backtest(
        &self,
        py: Python<'_>,
        frame: &NativeGraphTemporalFrame,
        train_size: usize,
    ) -> PyResult<String> {
        let metrics = py
            .allow_threads(|| self.model.backtest(&frame.frame, train_size))
            .map_err(to_py_geo_st_error)?;
        serde_json::to_string(&metrics).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreDcrnnForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.config.backend.selected.clone())
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreDcrnnForecaster::from_json_string(value).map_err(to_py_geo_st_error)?,
        })
    }
}

#[pyclass(name = "STAEformerForecaster")]
#[derive(Clone, Debug)]
struct NativeSTAEformerForecaster {
    model: CoreSTAEformerForecaster,
}

#[pyclass(name = "GraphWaveNetForecaster")]
#[derive(Clone, Debug)]
struct NativeGraphWaveNetForecaster {
    model: CoreGraphWaveNetForecaster,
}

#[pyclass(name = "PropagationDelayGraphForecaster")]
#[derive(Clone, Debug)]
struct NativePropagationDelayGraphForecaster {
    model: CoreDelayAwareGraphTransformer,
}

#[pymethods]
impl NativeSTAEformerForecaster {
    #[new]
    #[pyo3(signature = (
        lookback=8,
        attention_heads=4,
        hidden_size=8,
        epochs=120,
        learning_rate=0.02,
        ridge=0.0001,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lookback: usize,
        attention_heads: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreSTAEformerForecaster::new(CoreSTAEformerConfig {
                lookback,
                attention_heads,
                hidden_size,
                epochs,
                learning_rate,
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.allow_threads(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreSTAEformerForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreSTAEformerForecaster::from_json_string(value).map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }
}

#[pymethods]
impl NativeGraphWaveNetForecaster {
    #[new]
    #[pyo3(signature = (
        lookback=8,
        dilation_depth=3,
        hidden_size=8,
        epochs=120,
        learning_rate=0.02,
        ridge=0.0001,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lookback: usize,
        dilation_depth: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreGraphWaveNetForecaster::new(CoreGraphWaveNetConfig {
                lookback,
                dilation_depth,
                hidden_size,
                epochs,
                learning_rate,
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.allow_threads(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreGraphWaveNetForecaster::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreGraphWaveNetForecaster::from_json_string(value)
                .map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }
}

#[pymethods]
impl NativePropagationDelayGraphForecaster {
    #[new]
    #[pyo3(signature = (horizon=1, edge_delay_prior=None, ridge=0.000001, backend=None))]
    fn new(
        horizon: usize,
        edge_delay_prior: Option<Vec<usize>>,
        ridge: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreDelayAwareGraphTransformer::new(CoreDelayAwareGraphConfig {
                horizon,
                edge_delay_prior: edge_delay_prior.unwrap_or_default(),
                ridge,
                backend: graph_st_select_compute_backend(backend).map_err(to_py_geo_st_error)?,
            })
            .map_err(to_py_geo_st_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeGraphTemporalFrame) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&frame.frame))
            .map_err(to_py_geo_st_error)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<Vec<Vec<f64>>> {
        py.allow_threads(|| self.model.predict(horizon))
            .map_err(to_py_geo_st_error)
    }

    fn score(&self, py: Python<'_>, actual: Vec<Vec<f64>>) -> PyResult<f64> {
        py.allow_threads(|| self.model.score(&actual))
            .map_err(to_py_geo_st_error)
    }

    fn edge_delay_sensitivity(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.edge_delay_sensitivity())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.model.save(path).map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            model: CoreDelayAwareGraphTransformer::load(path).map_err(to_py_geo_st_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json_string().map_err(to_py_geo_st_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        Ok(Self {
            model: CoreDelayAwareGraphTransformer::from_json_string(value)
                .map_err(to_py_geo_st_error)?,
        })
    }

    fn backend(&self) -> PyResult<String> {
        Ok(self.model.backend())
    }
}

#[pymethods]
impl NativeNBeatsForecaster {
    #[new]
    #[pyo3(signature = (input_size=8, hidden_size=16, epochs=80, learning_rate=0.01, backend=None))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreNBeatsForecaster::new(CoreNBeatsConfig {
                input_size,
                hidden_size,
                epochs,
                learning_rate,
                backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
            })
            .map_err(to_py_neural_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "NHiTSForecaster")]
struct NativeNHiTSForecaster {
    model: CoreNHiTSForecaster,
}

#[pymethods]
impl NativeNHiTSForecaster {
    #[new]
    #[pyo3(signature = (input_size=12, hidden_size=16, epochs=80, learning_rate=0.01, pooling_size=2, backend=None))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        pooling_size: usize,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreNHiTSForecaster::new(CoreNHiTSConfig {
                input_size,
                hidden_size,
                epochs,
                learning_rate,
                pooling_size,
                backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
            })
            .map_err(to_py_neural_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "NeuralPanelForecaster")]
struct NativeNeuralPanelForecaster {
    model: CoreNeuralPanelForecaster,
}

#[pymethods]
impl NativeNeuralPanelForecaster {
    #[new]
    #[pyo3(signature = (
        n_lags=8,
        n_forecasts=1,
        quantiles=None,
        trend="piecewise_linear",
        n_changepoints=10,
        changepoints_range=0.8,
        daily_fourier_order=0,
        weekly_fourier_order=0,
        yearly_fourier_order=0,
        custom_seasonalities=None,
        seasonality_mode="additive",
        events=None,
        event_mode="additive",
        future_regressors=None,
        lagged_regressors=None,
        ar_layers=None,
        lagged_reg_layers=None,
        trend_mode="global",
        seasonality_global_local="global",
        event_global_local="global",
        regressor_global_local="global",
        local_l2=0.0,
        seed=0,
        loss="smooth_l1",
        epochs=80,
        learning_rate=0.01,
        weight_decay=0.0,
        newer_sample_weight=false,
        backend=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_lags: usize,
        n_forecasts: usize,
        quantiles: Option<Vec<f64>>,
        trend: &str,
        n_changepoints: usize,
        changepoints_range: f64,
        daily_fourier_order: usize,
        weekly_fourier_order: usize,
        yearly_fourier_order: usize,
        custom_seasonalities: Option<Vec<CustomSeasonalitySpec>>,
        seasonality_mode: &str,
        events: Option<BTreeMap<String, Vec<i32>>>,
        event_mode: &str,
        future_regressors: Option<BTreeMap<String, String>>,
        lagged_regressors: Option<BTreeMap<String, usize>>,
        ar_layers: Option<Vec<usize>>,
        lagged_reg_layers: Option<Vec<usize>>,
        trend_mode: &str,
        seasonality_global_local: &str,
        event_global_local: &str,
        regressor_global_local: &str,
        local_l2: f64,
        seed: u64,
        loss: &str,
        epochs: usize,
        learning_rate: f64,
        weight_decay: f64,
        newer_sample_weight: bool,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = neural_panel_config_from_parts(
            n_lags,
            n_forecasts,
            quantiles,
            trend,
            n_changepoints,
            changepoints_range,
            daily_fourier_order,
            weekly_fourier_order,
            yearly_fourier_order,
            custom_seasonalities,
            seasonality_mode,
            events,
            event_mode,
            future_regressors,
            lagged_regressors,
            ar_layers,
            lagged_reg_layers,
            trend_mode,
            seasonality_global_local,
            event_global_local,
            regressor_global_local,
            local_l2,
            seed,
            loss,
            epochs,
            learning_rate,
            weight_decay,
            newer_sample_weight,
            backend,
        )?;
        Ok(Self {
            model: CoreNeuralPanelForecaster::new(config).map_err(to_py_neural_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    #[pyo3(signature = (horizon, frame=None))]
    fn components_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: Option<&NativeForecastFrame>,
    ) -> PyResult<String> {
        let value = if let Some(frame) = frame {
            let mut covariates = BTreeMap::new();
            for row in frame.frame.rows() {
                covariates.insert(
                    (row.series_id.clone(), row.timestamp),
                    row.covariates.clone(),
                );
            }
            py.allow_threads(|| {
                self.model
                    .predict_components_json_value_with_known_future_covariates(
                        horizon,
                        Some(&covariates),
                    )
            })
        } else {
            py.allow_threads(|| self.model.predict_components_json_value(horizon))
        };
        value.map_err(to_py_value_error).and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }

    fn history_components_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.model.history_components_json_string())
            .map_err(to_py_value_error)
    }

    fn predict_with_known_future(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeForecastResult> {
        let mut covariates = BTreeMap::new();
        for row in frame.frame.rows() {
            covariates.insert(
                (row.series_id.clone(), row.timestamp),
                row.covariates.clone(),
            );
        }
        forecast_to_py(py.allow_threads(|| {
            self.model
                .predict_with_known_future_covariates(horizon, &covariates)
        }))
    }

    fn quantiles_json(&self, py: Python<'_>, horizon: usize) -> PyResult<String> {
        py.allow_threads(|| self.model.predict_quantiles_json_string(horizon))
            .map_err(to_py_value_error)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.model.to_json_string())
            .map_err(to_py_value_error)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, value: &str) -> PyResult<Self> {
        let model = py
            .allow_threads(|| CoreNeuralPanelForecaster::from_json_string(value))
            .map_err(to_py_value_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "LaneNeuralPanelForecaster")]
struct NativeLaneNeuralPanelForecaster {
    model: CoreLaneNeuralPanelForecaster,
}

#[pymethods]
impl NativeLaneNeuralPanelForecaster {
    #[new]
    #[pyo3(signature = (
        n_lags=8,
        n_forecasts=1,
        quantiles=None,
        trend="piecewise_linear",
        n_changepoints=10,
        changepoints_range=0.8,
        daily_fourier_order=0,
        weekly_fourier_order=0,
        yearly_fourier_order=0,
        custom_seasonalities=None,
        seasonality_mode="additive",
        events=None,
        event_mode="additive",
        future_regressors=None,
        lagged_regressors=None,
        ar_layers=None,
        lagged_reg_layers=None,
        trend_mode="global",
        seasonality_global_local="global",
        event_global_local="global",
        regressor_global_local="global",
        local_l2=0.0,
        seed=0,
        loss="smooth_l1",
        epochs=80,
        learning_rate=0.01,
        weight_decay=0.0,
        newer_sample_weight=false,
        backend=None,
        embedding_dim=8
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_lags: usize,
        n_forecasts: usize,
        quantiles: Option<Vec<f64>>,
        trend: &str,
        n_changepoints: usize,
        changepoints_range: f64,
        daily_fourier_order: usize,
        weekly_fourier_order: usize,
        yearly_fourier_order: usize,
        custom_seasonalities: Option<Vec<CustomSeasonalitySpec>>,
        seasonality_mode: &str,
        events: Option<BTreeMap<String, Vec<i32>>>,
        event_mode: &str,
        future_regressors: Option<BTreeMap<String, String>>,
        lagged_regressors: Option<BTreeMap<String, usize>>,
        ar_layers: Option<Vec<usize>>,
        lagged_reg_layers: Option<Vec<usize>>,
        trend_mode: &str,
        seasonality_global_local: &str,
        event_global_local: &str,
        regressor_global_local: &str,
        local_l2: f64,
        seed: u64,
        loss: &str,
        epochs: usize,
        learning_rate: f64,
        weight_decay: f64,
        newer_sample_weight: bool,
        backend: Option<&str>,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let base = neural_panel_config_from_parts(
            n_lags,
            n_forecasts,
            quantiles,
            trend,
            n_changepoints,
            changepoints_range,
            daily_fourier_order,
            weekly_fourier_order,
            yearly_fourier_order,
            custom_seasonalities,
            seasonality_mode,
            events,
            event_mode,
            future_regressors,
            lagged_regressors,
            ar_layers,
            lagged_reg_layers,
            trend_mode,
            seasonality_global_local,
            event_global_local,
            regressor_global_local,
            local_l2,
            seed,
            loss,
            epochs,
            learning_rate,
            weight_decay,
            newer_sample_weight,
            backend,
        )?;
        Ok(Self {
            model: CoreLaneNeuralPanelForecaster::new(CoreLaneNeuralPanelConfig {
                base,
                embedding_dim,
            })
            .map_err(to_py_neural_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    #[pyo3(signature = (horizon, frame=None))]
    fn components_json(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: Option<&NativeForecastFrame>,
    ) -> PyResult<String> {
        let value = if let Some(frame) = frame {
            let mut covariates = BTreeMap::new();
            for row in frame.frame.rows() {
                covariates.insert(
                    (row.series_id.clone(), row.timestamp),
                    row.covariates.clone(),
                );
            }
            py.allow_threads(|| {
                self.model
                    .predict_components_json_value_with_known_future_covariates(
                        horizon,
                        &covariates,
                    )
            })
        } else {
            py.allow_threads(|| self.model.predict_components_json_value(horizon))
        };
        value.map_err(to_py_value_error).and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }

    fn history_components_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.model.history_components_json_string())
            .map_err(to_py_value_error)
    }

    fn predict_with_known_future(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeForecastResult> {
        let mut covariates = BTreeMap::new();
        for row in frame.frame.rows() {
            covariates.insert(
                (row.series_id.clone(), row.timestamp),
                row.covariates.clone(),
            );
        }
        forecast_to_py(py.allow_threads(|| {
            self.model
                .predict_with_known_future_covariates(horizon, &covariates)
        }))
    }

    fn predict_for_lanes(
        &self,
        py: Python<'_>,
        horizon: usize,
        series_ids: Vec<String>,
    ) -> PyResult<NativeForecastResult> {
        forecast_to_py(py.allow_threads(|| self.model.predict_for_lanes(horizon, &series_ids)))
    }

    fn quantiles_json(&self, py: Python<'_>, horizon: usize) -> PyResult<String> {
        py.allow_threads(|| self.model.predict_quantiles_json_string(horizon))
            .map_err(to_py_value_error)
    }

    fn quantiles_json_for_lanes(
        &self,
        py: Python<'_>,
        horizon: usize,
        series_ids: Vec<String>,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            self.model
                .predict_quantiles_for_lanes_json_string(horizon, &series_ids)
        })
        .map_err(to_py_value_error)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "CartoBoostLagForecaster")]
#[derive(Clone, Debug)]
struct NativeCartoBoostLagForecaster {
    model: CoreCartoBoostLagForecaster,
}

#[pyclass(name = "AutoForecastModel", unsendable)]
#[derive(Clone)]
struct NativeAutoForecastModel {
    model: CoreAutoForecastModel,
}

#[pymethods]
impl NativeAutoForecastModel {
    #[new]
    #[pyo3(signature = (lags=None, rolling_windows=None, partial_rolling_mean_windows=None, rolling_std_windows=None, rolling_min_windows=None, rolling_max_windows=None, ewm_alpha_percents=None, difference_lags=None, rolling_trend_windows=None, covariate_features=None, covariate_indicator_values=None, covariate_calendar_interactions=false, calendar_features=true, rich_calendar_features=false, elapsed_calendar_features=false, elapsed_calendar_periods=None, season_length=7, validation_window=None, validation_origin_count=2, objective="rmse_wape", baseline_displacement_gain=0.03, hard_winner_relative_gain=0.05, min_blend_weight=0.15, max_blend_weight=0.85, max_direct_horizon=28, max_candidate_count=None, recursive=true, n_estimators=None, learning_rate=None, max_depth=None, min_samples_leaf=None, min_gain=None, splitters=None, trend_features=true, target_mode="level"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lags: Option<Vec<usize>>,
        rolling_windows: Option<Vec<usize>>,
        partial_rolling_mean_windows: Option<Vec<usize>>,
        rolling_std_windows: Option<Vec<usize>>,
        rolling_min_windows: Option<Vec<usize>>,
        rolling_max_windows: Option<Vec<usize>>,
        ewm_alpha_percents: Option<Vec<u8>>,
        difference_lags: Option<Vec<usize>>,
        rolling_trend_windows: Option<Vec<usize>>,
        covariate_features: Option<Vec<String>>,
        covariate_indicator_values: Option<BTreeMap<String, Vec<f64>>>,
        covariate_calendar_interactions: bool,
        calendar_features: bool,
        rich_calendar_features: bool,
        elapsed_calendar_features: bool,
        elapsed_calendar_periods: Option<Vec<usize>>,
        season_length: usize,
        validation_window: Option<usize>,
        validation_origin_count: usize,
        objective: &str,
        baseline_displacement_gain: f64,
        hard_winner_relative_gain: f64,
        min_blend_weight: f64,
        max_blend_weight: f64,
        max_direct_horizon: usize,
        max_candidate_count: Option<usize>,
        recursive: bool,
        n_estimators: Option<usize>,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        min_gain: Option<f64>,
        splitters: Option<Vec<String>>,
        trend_features: bool,
        target_mode: &str,
    ) -> PyResult<Self> {
        if !recursive {
            return Err(PyValueError::new_err(
                "AutoForecastModel currently supports recursive=true only",
            ));
        }
        let lags = lags.unwrap_or_else(|| vec![1, 2, 3, 7, 14, 28]);
        let rolling_mean_windows = rolling_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let rolling_std_windows = rolling_std_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let rolling_min_windows = rolling_min_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let rolling_max_windows = rolling_max_windows.unwrap_or_else(|| vec![7, 14, 28]);
        let difference_lags = match difference_lags {
            Some(values) => values,
            None if trend_features => lags.iter().copied().filter(|lag| *lag > 1).collect(),
            None => Vec::new(),
        };
        let rolling_trend_windows = match rolling_trend_windows {
            Some(values) => values,
            None if trend_features => rolling_mean_windows
                .iter()
                .copied()
                .filter(|window| *window > 1)
                .collect(),
            None => Vec::new(),
        };
        let lag_config = LagFeatureConfig {
            difference_lags,
            rolling_trend_windows,
            lags,
            rolling_mean_windows,
            partial_rolling_mean_windows: partial_rolling_mean_windows.unwrap_or_default(),
            rolling_std_windows,
            rolling_min_windows,
            rolling_max_windows,
            ewm_alpha_percents: ewm_alpha_percents.unwrap_or_default(),
            calendar_features: calendar_feature_config(
                calendar_features,
                rich_calendar_features,
                elapsed_calendar_features,
                elapsed_calendar_periods.as_deref(),
            ),
            covariate_features: covariate_features.unwrap_or_default(),
            covariate_indicator_values: covariate_indicator_values.unwrap_or_default(),
            covariate_calendar_interactions,
        };
        let mut booster_config = BoosterConfig::default();
        if let Some(value) = n_estimators {
            booster_config.n_estimators = value;
        }
        if let Some(value) = learning_rate {
            booster_config.learning_rate = value;
        }
        if let Some(value) = max_depth {
            booster_config.max_depth = value;
        }
        if let Some(value) = min_samples_leaf {
            booster_config.min_samples_leaf = value;
        }
        if let Some(value) = min_gain {
            booster_config.min_gain = value;
        }
        if let Some(values) = splitters {
            booster_config.splitters = parse_splitters(&values)?;
        }
        validate_params(
            booster_config.n_estimators,
            booster_config.learning_rate,
            booster_config.max_depth,
            booster_config.min_samples_leaf,
            booster_config.min_gain,
            booster_config.linear_lambda_l2,
            booster_config.constant_lambda_l2,
            booster_config.fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        let target_mode = parse_global_target_mode(target_mode)?;
        let objective = CoreForecastObjective::parse(objective).map_err(to_py_value_error)?;
        Ok(Self {
            model: CoreAutoForecastModel::new(CoreAutoForecastConfig {
                lag_config,
                booster_config,
                target_mode,
                season_length,
                validation_window,
                validation_origin_count,
                objective,
                baseline_displacement_gain,
                hard_winner_relative_gain,
                min_blend_weight,
                max_blend_weight,
                max_direct_horizon,
                max_candidate_count,
            })
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata()).map_err(|err| {
            PyValueError::new_err(format!("failed to serialize forecaster metadata: {err}"))
        })
    }
}

#[pymethods]
impl NativeCartoBoostLagForecaster {
    #[new]
    #[pyo3(signature = (lags=None, rolling_windows=None, partial_rolling_mean_windows=None, rolling_std_windows=None, rolling_min_windows=None, rolling_max_windows=None, ewm_alpha_percents=None, difference_lags=None, rolling_trend_windows=None, covariate_features=None, covariate_indicator_values=None, covariate_calendar_interactions=false, calendar_features=true, rich_calendar_features=false, elapsed_calendar_features=false, elapsed_calendar_periods=None, recursive=true, prediction_interval_levels=None, n_estimators=None, learning_rate=None, max_depth=None, min_samples_leaf=None, min_gain=None, splitters=None, trend_features=true, target_mode="level"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lags: Option<Vec<usize>>,
        rolling_windows: Option<Vec<usize>>,
        partial_rolling_mean_windows: Option<Vec<usize>>,
        rolling_std_windows: Option<Vec<usize>>,
        rolling_min_windows: Option<Vec<usize>>,
        rolling_max_windows: Option<Vec<usize>>,
        ewm_alpha_percents: Option<Vec<u8>>,
        difference_lags: Option<Vec<usize>>,
        rolling_trend_windows: Option<Vec<usize>>,
        covariate_features: Option<Vec<String>>,
        covariate_indicator_values: Option<BTreeMap<String, Vec<f64>>>,
        covariate_calendar_interactions: bool,
        calendar_features: bool,
        rich_calendar_features: bool,
        elapsed_calendar_features: bool,
        elapsed_calendar_periods: Option<Vec<usize>>,
        recursive: bool,
        prediction_interval_levels: Option<Vec<f64>>,
        n_estimators: Option<usize>,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        min_gain: Option<f64>,
        splitters: Option<Vec<String>>,
        trend_features: bool,
        target_mode: &str,
    ) -> PyResult<Self> {
        if !recursive {
            return Err(PyValueError::new_err(
                "CartoBoostLagForecaster currently supports recursive=true only",
            ));
        }
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let lags = lags.unwrap_or_else(|| vec![1, 7, 14]);
        let rolling_mean_windows = rolling_windows.unwrap_or_else(|| vec![7, 28]);
        let rolling_std_windows = rolling_std_windows.unwrap_or_else(|| vec![7, 28]);
        let rolling_min_windows = rolling_min_windows.unwrap_or_else(|| vec![7, 28]);
        let rolling_max_windows = rolling_max_windows.unwrap_or_else(|| vec![7, 28]);
        let difference_lags = match difference_lags {
            Some(values) => values,
            None if trend_features => lags.iter().copied().filter(|lag| *lag > 1).collect(),
            None => Vec::new(),
        };
        let rolling_trend_windows = match rolling_trend_windows {
            Some(values) => values,
            None if trend_features => rolling_mean_windows
                .iter()
                .copied()
                .filter(|window| *window > 1)
                .collect(),
            None => Vec::new(),
        };
        let config = LagFeatureConfig {
            difference_lags,
            rolling_trend_windows,
            lags,
            rolling_mean_windows,
            partial_rolling_mean_windows: partial_rolling_mean_windows.unwrap_or_default(),
            rolling_std_windows,
            rolling_min_windows,
            rolling_max_windows,
            ewm_alpha_percents: ewm_alpha_percents.unwrap_or_default(),
            calendar_features: calendar_feature_config(
                calendar_features,
                rich_calendar_features,
                elapsed_calendar_features,
                elapsed_calendar_periods.as_deref(),
            ),
            covariate_features: covariate_features.unwrap_or_default(),
            covariate_indicator_values: covariate_indicator_values.unwrap_or_default(),
            covariate_calendar_interactions,
        };
        let mut booster_config = BoosterConfig::default();
        if let Some(value) = n_estimators {
            booster_config.n_estimators = value;
        }
        if let Some(value) = learning_rate {
            booster_config.learning_rate = value;
        }
        if let Some(value) = max_depth {
            booster_config.max_depth = value;
        }
        if let Some(value) = min_samples_leaf {
            booster_config.min_samples_leaf = value;
        }
        if let Some(value) = min_gain {
            booster_config.min_gain = value;
        }
        if let Some(values) = splitters {
            booster_config.splitters = parse_splitters(&values)?;
        }
        validate_params(
            booster_config.n_estimators,
            booster_config.learning_rate,
            booster_config.max_depth,
            booster_config.min_samples_leaf,
            booster_config.min_gain,
            booster_config.linear_lambda_l2,
            booster_config.constant_lambda_l2,
            booster_config.fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        let target_mode = parse_global_target_mode(target_mode)?;
        Ok(Self {
            model: CoreCartoBoostLagForecaster::new_with_target_mode(
                config,
                booster_config,
                target_mode,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn predict_with_known_future(
        &self,
        py: Python<'_>,
        horizon: usize,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeForecastResult> {
        let mut covariates = BTreeMap::new();
        for row in frame.frame.rows() {
            covariates.insert(
                (row.series_id.clone(), row.timestamp),
                row.covariates.clone(),
            );
        }
        forecast_to_py(py.allow_threads(|| {
            self.model
                .predict_with_known_future_covariates(horizon, &covariates)
        }))
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata()).map_err(|err| {
            PyValueError::new_err(format!("failed to serialize forecaster metadata: {err}"))
        })
    }
}

#[pyclass(name = "WeightedEnsembleForecaster", unsendable)]
struct NativeWeightedEnsembleForecaster {
    model: CoreWeightedEnsembleForecaster,
}

#[pymethods]
impl NativeWeightedEnsembleForecaster {
    #[new]
    fn new(py: Python<'_>, members: Vec<(String, Py<PyAny>, f64)>) -> PyResult<Self> {
        let members = members
            .iter()
            .map(|(name, model, weight)| {
                Ok((name.clone(), boxed_forecaster_from_py(py, model)?, *weight))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            model: CoreWeightedEnsembleForecaster::new(members).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

fn boxed_forecaster_from_py(py: Python<'_>, model: &Py<PyAny>) -> PyResult<Box<dyn Forecaster>> {
    let model = model.bind(py);
    if let Ok(model) = model.extract::<PyRef<'_, NativeNaiveForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeSeasonalNaiveForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeThetaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeOptimizedThetaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativePiecewiseLinearSeasonalForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeETSForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeArimaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeAutoARIMAForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(_model) = model.extract::<PyRef<'_, NativeAutoStatsBank>>() {
        return Err(PyValueError::new_err(
            "AutoStatsBank cannot be cloned into WeightedEnsembleForecaster",
        ));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeKalmanForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeCartoBoostLagForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    Err(PyValueError::new_err(
        "WeightedEnsembleForecaster members must be native forecasting models",
    ))
}

fn forecast_to_py(
    result: cartoboost_core::Result<CoreForecastResult>,
) -> PyResult<NativeForecastResult> {
    Ok(NativeForecastResult {
        result: result.map_err(to_py_value_error)?,
    })
}

fn fit_forecaster_py<M: Forecaster>(
    py: Python<'_>,
    model: &mut M,
    frame: &NativeForecastFrame,
) -> PyResult<()> {
    py.allow_threads(|| model.fit(&frame.frame))
        .map_err(to_py_value_error)
}

fn predict_forecaster_py<M: Forecaster>(
    py: Python<'_>,
    model: &M,
    horizon: usize,
) -> PyResult<NativeForecastResult> {
    forecast_to_py(py.allow_threads(|| model.predict(horizon)))
}

#[allow(dead_code)]
fn ets_diagnostic_values(
    values: Option<&[f64]>,
    series_id: &str,
    name: &str,
) -> PyResult<Vec<f64>> {
    values.map(|values| values.to_vec()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "ETS {name} are unavailable for series {series_id:?}; fit the model and check the series id"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_kriging_config(
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<OrdinaryKrigingConfig> {
    let variogram_model = parse_kriging_variogram_model(variogram_model)?;
    let drift = parse_kriging_drift(drift)?;
    OrdinaryKrigingConfig::new(range, nugget)
        .and_then(|config| config.with_sill(sill))
        .and_then(|config| config.with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling))
        .and_then(|config| config.with_neighbor_limits(max_neighbors, min_neighbors, max_distance))
        .map(|config| {
            config
                .with_variogram_model(variogram_model)
                .with_drift(drift)
        })
        .map_err(to_py_value_error)
}

fn parse_kriging_variogram_model(value: &str) -> PyResult<KrigingVariogramModel> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "exponential" | "exp" => Ok(KrigingVariogramModel::Exponential),
        "gaussian" | "gauss" => Ok(KrigingVariogramModel::Gaussian),
        "spherical" | "sphere" => Ok(KrigingVariogramModel::Spherical),
        "linear" => Ok(KrigingVariogramModel::Linear),
        other => Err(PyValueError::new_err(format!(
            "unsupported kriging variogram_model {other:?}; expected exponential, gaussian, spherical, or linear"
        ))),
    }
}

fn parse_kriging_drift(value: &str) -> PyResult<KrigingDrift> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "ordinary" | "constant" | "none" => Ok(KrigingDrift::Ordinary),
        "linear" | "universal_linear" | "universal" => Ok(KrigingDrift::Linear),
        other => Err(PyValueError::new_err(format!(
            "unsupported kriging drift {other:?}; expected ordinary or linear"
        ))),
    }
}

fn parse_spatial_piecewise_kriging_mode(value: &str) -> PyResult<SpatialPiecewiseKrigingMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "kriged_regressors" | "regressors" => {
            Ok(SpatialPiecewiseKrigingMode::KrigedRegressors)
        }
        "residual_kriging" | "residual" => Ok(SpatialPiecewiseKrigingMode::ResidualKriging),
        "hybrid" => Ok(SpatialPiecewiseKrigingMode::Hybrid),
        other => Err(PyValueError::new_err(format!(
            "unsupported spatial piecewise kriging mode {other:?}; expected kriged_regressors, residual_kriging, or hybrid"
        ))),
    }
}

fn parse_classical_validation_objective(
    value: &str,
    season_length: usize,
) -> PyResult<ClassicalExpertValidationObjective> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "mse" | "mean_squared_error" => Ok(ClassicalExpertValidationObjective::MeanSquaredError),
        "smape_mase_average" | "owa_proxy" => Ok(
            ClassicalExpertValidationObjective::SmapeMaseAverage {
                seasonality: season_length.max(1),
            },
        ),
        other => Err(PyValueError::new_err(format!(
            "unsupported validation_objective {other:?}; expected mean_squared_error or smape_mase_average"
        ))),
    }
}

fn kriging_config_json(config: OrdinaryKrigingConfig) -> Value {
    json!({
        "range": config.range,
        "nugget": config.nugget,
        "sill": config.sill,
        "variogram_model": format!("{:?}", config.variogram_model).to_lowercase(),
        "drift": format!("{:?}", config.drift).to_lowercase(),
        "anisotropy_angle_degrees": config.anisotropy_angle_degrees,
        "anisotropy_scaling": config.anisotropy_scaling,
        "max_neighbors": config.max_neighbors,
        "min_neighbors": config.min_neighbors,
        "max_distance": config.max_distance,
    })
}

fn backtest_to_py(
    result: cartoboost_core::Result<CoreBacktestResult>,
) -> PyResult<NativeBacktestResult> {
    Ok(NativeBacktestResult {
        result: result.map_err(to_py_value_error)?,
    })
}

fn parse_forecast_window(value: &str) -> PyResult<ForecastWindow> {
    match value {
        "expanding" => Ok(ForecastWindow::Expanding),
        "sliding" => Ok(ForecastWindow::Sliding),
        _ => Err(PyValueError::new_err(
            "forecast window must be 'expanding' or 'sliding'",
        )),
    }
}

fn forecast_window_name(window: &ForecastWindow) -> &'static str {
    match window {
        ForecastWindow::Expanding => "expanding",
        ForecastWindow::Sliding => "sliding",
    }
}

fn parse_forecast_actuals(
    actuals: Vec<(String, String, usize, f64)>,
) -> PyResult<Vec<ForecastActual>> {
    actuals
        .into_iter()
        .map(|(series_id, timestamp, horizon, actual)| {
            Ok(ForecastActual {
                series_id,
                timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                    .map_err(to_py_value_error)?,
                horizon,
                actual,
            })
        })
        .collect()
}

fn forecast_prediction_tuple(
    prediction: &ForecastPrediction,
) -> (String, String, usize, String, f64) {
    (
        prediction.series_id.clone(),
        format_forecast_timestamp(prediction.timestamp),
        prediction.horizon,
        prediction.model.clone(),
        prediction.mean,
    )
}

fn format_forecast_timestamp(timestamp: impl std::fmt::Display) -> String {
    timestamp.to_string().replace(' ', "T")
}

fn validate_interval_levels(levels: Option<&[f64]>) -> PyResult<()> {
    for level in levels.unwrap_or(&[]) {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(PyValueError::new_err(
                "prediction interval levels must be finite values between 0 and 1",
            ));
        }
    }
    Ok(())
}

fn calendar_feature_config(
    enabled: bool,
    rich: bool,
    elapsed_only: bool,
    elapsed_periods: Option<&[usize]>,
) -> Vec<CalendarFeature> {
    if !enabled {
        return Vec::new();
    }
    if elapsed_only {
        let mut features = vec![CalendarFeature::ElapsedIndex];
        push_elapsed_calendar_periods(&mut features, elapsed_periods);
        return features;
    }
    let mut features = vec![
        CalendarFeature::DayOfWeek,
        CalendarFeature::Month,
        CalendarFeature::Day,
    ];
    if rich {
        features.push(CalendarFeature::DayOfWeekSin);
        features.push(CalendarFeature::DayOfWeekCos);
        features.push(CalendarFeature::MonthSin);
        features.push(CalendarFeature::MonthCos);
        features.push(CalendarFeature::DaySin);
        features.push(CalendarFeature::DayCos);
        features.push(CalendarFeature::MonthStart);
        features.push(CalendarFeature::MonthMiddle);
        features.push(CalendarFeature::MonthEnd);
        features.push(CalendarFeature::DayOfYear);
        features.push(CalendarFeature::ElapsedIndex);
        push_elapsed_calendar_periods(&mut features, elapsed_periods);
    }
    features
}

fn push_elapsed_calendar_periods(
    features: &mut Vec<CalendarFeature>,
    elapsed_periods: Option<&[usize]>,
) {
    let mut periods = BTreeSet::new();
    for period in elapsed_periods.unwrap_or(&[]) {
        if *period >= 2 && periods.insert(*period) {
            features.push(CalendarFeature::ElapsedPhase(*period));
        }
    }
}

fn parse_theta_seasonality(
    season_length: Option<usize>,
    seasonality: Option<String>,
) -> PyResult<Option<ThetaSeasonality>> {
    let Some(mode) = seasonality else {
        return Ok(None);
    };
    let season_length = season_length.ok_or_else(|| {
        PyValueError::new_err("season_length is required when seasonality is set")
    })?;
    match mode.as_str() {
        "additive" => ThetaSeasonality::additive(season_length)
            .map(Some)
            .map_err(to_py_value_error),
        "multiplicative" => ThetaSeasonality::multiplicative(season_length)
            .map(Some)
            .map_err(to_py_value_error),
        _ => Err(PyValueError::new_err(
            "seasonality must be 'additive' or 'multiplicative'",
        )),
    }
}

fn parse_piecewise_growth(value: &str) -> PyResult<PiecewiseLinearGrowth> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "linear" => Ok(PiecewiseLinearGrowth::Linear),
        "flat" => Ok(PiecewiseLinearGrowth::Flat),
        "logistic" => Ok(PiecewiseLinearGrowth::Logistic),
        other => Err(PyValueError::new_err(format!(
            "growth must be 'linear', 'flat', or 'logistic', got {other:?}"
        ))),
    }
}

fn parse_piecewise_component_mode(value: &str) -> PyResult<PiecewiseLinearComponentMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "additive" => Ok(PiecewiseLinearComponentMode::Additive),
        "multiplicative" => Ok(PiecewiseLinearComponentMode::Multiplicative),
        other => Err(PyValueError::new_err(format!(
            "component_mode must be 'additive' or 'multiplicative', got {other:?}"
        ))),
    }
}

fn parse_piecewise_fit_loss(value: &str) -> PyResult<PiecewiseLinearFitLoss> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "squared" | "l2" | "least_squares" => Ok(PiecewiseLinearFitLoss::Squared),
        "huber" | "robust" => Ok(PiecewiseLinearFitLoss::Huber),
        other => Err(PyValueError::new_err(format!(
            "fit_loss must be 'squared' or 'huber', got {other:?}"
        ))),
    }
}

fn parse_piecewise_regressor_standardization(
    value: &str,
) -> PyResult<PiecewiseLinearRegressorStandardization> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "auto" => Ok(PiecewiseLinearRegressorStandardization::Auto),
        "none" | "off" | "false" => Ok(PiecewiseLinearRegressorStandardization::None),
        other => Err(PyValueError::new_err(format!(
            "regressor_standardization must be 'auto' or 'none', got {other:?}"
        ))),
    }
}

fn parse_piecewise_trend_uncertainty_policy(
    value: &str,
) -> PyResult<PiecewiseLinearTrendUncertaintyPolicy> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" | "laplace" => Ok(PiecewiseLinearTrendUncertaintyPolicy::Laplace),
        "normal" | "gaussian" => Ok(PiecewiseLinearTrendUncertaintyPolicy::Normal),
        other => Err(PyValueError::new_err(format!(
            "trend_uncertainty_policy must be 'laplace' or 'normal', got {other:?}"
        ))),
    }
}

fn parse_optional_piecewise_component_mode(
    value: Option<String>,
) -> PyResult<Option<PiecewiseLinearComponentMode>> {
    value
        .as_deref()
        .map(parse_piecewise_component_mode)
        .transpose()
}

fn parse_piecewise_regressor_modes(
    values: Option<BTreeMap<String, String>>,
) -> PyResult<BTreeMap<String, PiecewiseLinearComponentMode>> {
    values
        .unwrap_or_default()
        .into_iter()
        .map(|(name, mode)| Ok((name, parse_piecewise_component_mode(&mode)?)))
        .collect()
}

fn parse_piecewise_changepoint_timestamps(
    timestamps: Option<Vec<String>>,
) -> PyResult<Vec<chrono::NaiveDateTime>> {
    timestamps
        .unwrap_or_default()
        .into_iter()
        .map(|timestamp| {
            cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                .map_err(to_py_value_error)
        })
        .collect()
}

fn parse_piecewise_events(
    events: Option<Vec<PyPiecewiseEvent>>,
) -> PyResult<Vec<PiecewiseLinearEvent>> {
    events
        .unwrap_or_default()
        .into_iter()
        .map(|(name, timestamp, lower_window, upper_window)| {
            Ok(PiecewiseLinearEvent {
                name,
                timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                    .map_err(to_py_value_error)?,
                lower_window: lower_window.unwrap_or(0),
                upper_window: upper_window.unwrap_or(0),
            })
        })
        .collect()
}

fn parse_piecewise_seasonalities(
    seasonalities: Option<Vec<PyPiecewiseSeasonality>>,
) -> PyResult<Vec<PiecewiseLinearSeasonality>> {
    seasonalities
        .unwrap_or_default()
        .into_iter()
        .map(
            |(name, period_days, fourier_order, mode, condition_name, l2_regularization)| {
                Ok(PiecewiseLinearSeasonality {
                    name,
                    period_days,
                    fourier_order,
                    mode: parse_optional_piecewise_component_mode(mode)?,
                    condition_name,
                    l2_regularization,
                })
            },
        )
        .collect()
}

#[pyclass(name = "NearestNeighborGPRegressor")]
struct NativeNearestNeighborGPRegressor {
    model: CoreNearestNeighborGPRegressor,
}

#[pymethods]
impl NativeNearestNeighborGPRegressor {
    #[new]
    #[pyo3(signature = (kernel="exponential", range=1.0, sill=1.0, nugget=1.0e-6, n_neighbors=16, anisotropy_angle_degrees=0.0, anisotropy_scaling=1.0, brute_force_threshold=2048, duplicate_tolerance=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel: &str,
        range: f64,
        sill: f64,
        nugget: f64,
        n_neighbors: usize,
        anisotropy_angle_degrees: f64,
        anisotropy_scaling: f64,
        brute_force_threshold: usize,
        duplicate_tolerance: f64,
    ) -> PyResult<Self> {
        let config = CoreNngpConfig {
            kernel: CoreCovarianceKernel::parse(kernel).map_err(to_py_geostats_error)?,
            range,
            sill,
            nugget,
            anisotropy: CoreGeostatsAnisotropy {
                angle_degrees: anisotropy_angle_degrees,
                scaling: anisotropy_scaling,
            },
            n_neighbors,
            brute_force_threshold,
            duplicate_tolerance,
        };
        Ok(Self {
            model: CoreNearestNeighborGPRegressor::new(config).map_err(to_py_geostats_error)?,
        })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        coords: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let coords = coords_from_array(coords)?;
        let targets = y.as_slice()?.to_vec();
        py.allow_threads(|| self.model.fit(&coords, &targets))
            .map_err(to_py_geostats_error)
    }

    fn predict(
        &self,
        py: Python<'_>,
        coords: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyNngpPrediction> {
        let coords = coords_from_array(coords)?;
        let predictions = py
            .allow_threads(|| self.model.predict(&coords))
            .map_err(to_py_geostats_error)?;
        let means = predictions
            .iter()
            .map(|prediction| prediction.mean)
            .collect();
        let variances = predictions
            .iter()
            .map(|prediction| prediction.variance)
            .collect();
        let neighbors = predictions
            .into_iter()
            .map(|prediction| prediction.neighbor_indices)
            .collect();
        Ok((means, variances, neighbors))
    }

    fn config_json(&self) -> PyResult<String> {
        let config = self.model.config();
        serde_json::to_string(&json!({
            "kernel": config.kernel.as_str(),
            "range": config.range,
            "sill": config.sill,
            "nugget": config.nugget,
            "anisotropy_angle_degrees": config.anisotropy.angle_degrees,
            "anisotropy_scaling": config.anisotropy.scaling,
            "n_neighbors": config.n_neighbors,
            "brute_force_threshold": config.brute_force_threshold,
            "duplicate_tolerance": config.duplicate_tolerance,
        }))
        .map_err(to_py_json_error)
    }
}

#[pyclass(name = "CartoBoostRegressor")]
#[derive(Clone, Debug)]
struct NativeCartoBoostRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    loss: String,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    monotonic_constraints: Vec<i8>,
    model: Option<Model>,
    flat_axis_predictor: Option<FlatAxisPredictor>,
}

#[pymethods]
impl NativeCartoBoostRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_leaf=20, min_gain=1e-8, loss="l2", quantile_alpha=0.5, huber_delta=1.0, log_offset=1.0, splitters=None, leaf_predictor="constant", linear_leaf_features=None, l2_regularization=1.0, constant_l2_regularization=0.0, fuzzy=false, fuzzy_bandwidth=0.0, fuzzy_kernel="linear", n_threads=None, monotonic_constraints=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        loss: &str,
        quantile_alpha: f64,
        huber_delta: f64,
        log_offset: f64,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
        monotonic_constraints: Option<Vec<i8>>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            quantile_alpha,
            huber_delta,
            log_offset,
        )?;
        parse_loss(loss, quantile_alpha, huber_delta, log_offset)?;
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss: loss.to_string(),
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            monotonic_constraints: monotonic_constraints.unwrap_or_default(),
            model: None,
            flat_axis_predictor: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_sets=None, feature_schema_json=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = self.booster_config(splitters, leaf_predictor)?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Booster::new(config).fit(&dataset, &y, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.set_model(model);
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_offsets=None, sparse_ids=None, feature_schema_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = self.booster_config(splitters, leaf_predictor)?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Booster::new(config).fit(&dataset, &targets, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.set_model(model);
        Ok(())
    }

    #[pyo3(signature = (x, y, sparse_sets, feature_schema_json=None, sample_weight=None))]
    fn fit_mixed(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.fit(
            py,
            x,
            y,
            sample_weight,
            Some(sparse_sets),
            feature_schema_json,
        )
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.try_predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let n_threads = self.n_threads;
        let predictions = py
            .allow_threads(|| {
                run_with_optional_threads(n_threads, || {
                    // Sparse inputs may be supplied by a caller even when
                    // the fitted forest never selected a sparse split.  In
                    // that case they do not affect routing and should not
                    // disable the dense flat predictor fast path.
                    if !model.requires_sparse_sets() {
                        if let Some(predictor) = &self.flat_axis_predictor {
                            model.validate_dense_flat_prediction_inputs(rows, cols, values)?;
                            Ok(predictor.predict_flat(rows, cols, values))
                        } else {
                            model.try_predict_flat(rows, cols, values, &offsets, &ids)
                        }
                    } else {
                        model.try_predict_flat(rows, cols, values, &offsets, &ids)
                    }
                })
            })
            .map_err(to_py_value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    #[pyo3(signature = (x, sparse_sets))]
    fn predict_mixed(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        self.predict(py, x, Some(sparse_sets))
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict_additive(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| {
            run_with_optional_threads(n_threads, || model.try_predict_additive(&dataset))
        })
        .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_additive_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let n_threads = self.n_threads;
        py.allow_threads(|| {
            run_with_optional_threads(n_threads, || {
                model.try_predict_additive_flat(rows, cols, values, &offsets, &ids)
            })
        })
        .map_err(to_py_value_error)
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        py.allow_threads(|| model.save(path)).map_err(to_py_error)
    }

    fn save_weights(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        py.allow_threads(|| model.save_weights(path))
            .map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Model::load(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[staticmethod]
    fn load_weights(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Model::load_weights(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn loss(&self) -> String {
        self.loss.clone()
    }

    #[getter]
    fn quantile_alpha(&self) -> f64 {
        self.quantile_alpha
    }

    #[getter]
    fn huber_delta(&self) -> f64 {
        self.huber_delta
    }

    #[getter]
    fn log_offset(&self) -> f64 {
        self.log_offset
    }

    #[getter]
    fn leaf_predictor(&self) -> String {
        self.leaf_predictor.clone()
    }

    #[getter]
    fn linear_leaf_features(&self) -> Vec<usize> {
        self.linear_leaf_features.clone()
    }

    #[getter]
    fn l2_regularization(&self) -> f64 {
        self.l2_regularization
    }

    #[getter]
    fn constant_l2_regularization(&self) -> f64 {
        self.constant_l2_regularization
    }

    #[getter]
    fn fuzzy(&self) -> bool {
        self.fuzzy
    }

    #[getter]
    fn fuzzy_bandwidth(&self) -> f64 {
        self.fuzzy_bandwidth
    }

    #[getter]
    fn fuzzy_kernel(&self) -> String {
        self.fuzzy_kernel.clone()
    }

    #[getter]
    fn n_threads(&self) -> Option<usize> {
        self.n_threads
    }

    #[getter]
    fn monotonic_constraints(&self) -> Vec<i8> {
        self.monotonic_constraints.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(Model::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_history_json(&self) -> PyResult<String> {
        self.model
            .as_ref()
            .map(|model| serde_json::to_string(&model.training_history))
            .transpose()
            .map(|value| value.unwrap_or_else(|| "[]".to_string()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostRegressor {
    fn from_model(model: Model) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            monotonic_constraints,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                loss_name(&config.loss).to_string(),
                quantile_alpha(&config.loss),
                huber_delta(&config.loss),
                log_offset(&config.loss),
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                config.monotonic_constraints,
            )
        } else {
            (
                1,
                1,
                0.0,
                "l2".to_string(),
                0.5,
                1.0,
                1.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                Vec::new(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            monotonic_constraints,
            model: Some(model),
            flat_axis_predictor: None,
        })
        .map(|mut regressor| {
            regressor.refresh_prediction_cache();
            regressor
        })
    }

    fn booster_config(
        &self,
        splitters: Vec<SplitterKind>,
        leaf_predictor: LeafPredictorKind,
    ) -> PyResult<BoosterConfig> {
        Ok(BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            loss: parse_loss(
                &self.loss,
                self.quantile_alpha,
                self.huber_delta,
                self.log_offset,
            )?,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            monotonic_constraints: self.monotonic_constraints.clone(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
            graph_leaf_smoothing: None,
        })
    }

    fn set_model(&mut self, model: Model) {
        self.model = Some(model);
        self.refresh_prediction_cache();
    }

    fn refresh_prediction_cache(&mut self) {
        self.flat_axis_predictor = self.model.as_ref().and_then(Model::flat_axis_predictor);
    }
}

#[pyclass(name = "CartoBoostClassifier")]
#[derive(Clone, Debug)]
struct NativeCartoBoostClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    objective: String,
    class_count: usize,
    class_weights: Vec<f64>,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    model: Option<ClassifierModel>,
}

#[pymethods]
impl NativeCartoBoostClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        min_gain=1e-8,
        objective="auto",
        class_count=2,
        class_weights=None,
        splitters=None,
        leaf_predictor="constant",
        linear_leaf_features=None,
        l2_regularization=1.0,
        constant_l2_regularization=0.0,
        fuzzy=false,
        fuzzy_bandwidth=0.0,
        fuzzy_kernel="linear",
        n_threads=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        objective: &str,
        class_count: usize,
        class_weights: Option<Vec<f64>>,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        if class_count < 2 {
            return Err(PyValueError::new_err("class_count must be at least 2"));
        }
        parse_classification_objective(objective, class_count)?;
        let class_weights = class_weights.unwrap_or_default();
        if !class_weights.is_empty() && class_weights.len() != class_count {
            return Err(PyValueError::new_err(format!(
                "class_weights has length {}, but class_count is {class_count}",
                class_weights.len()
            )));
        }
        if class_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(PyValueError::new_err(
                "class_weights must be finite and non-negative",
            ));
        }
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective: objective.to_string(),
            class_count,
            class_weights,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            model: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_sets=None, feature_schema_json=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let config = self.classifier_config()?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Classifier::new(config).fit(&dataset, &y, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (
        x,
        y,
        sample_weight=None,
        sparse_offsets=None,
        sparse_ids=None,
        feature_schema_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let config = self.classifier_config()?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Classifier::new(config).fit(&dataset, &targets, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict_proba(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.predict_proba(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_proba_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.predict_proba(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn decision_function(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| {
            run_with_optional_threads(n_threads, || model.decision_function(&dataset))
        })
        .map_err(to_py_value_error)
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostClassifier is not fitted"))?;
        py.allow_threads(|| model.save(path)).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .allow_threads(|| ClassifierModel::load(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn objective(&self) -> String {
        self.objective.clone()
    }

    #[getter]
    fn class_count(&self) -> usize {
        self.class_count
    }

    #[getter]
    fn class_weights(&self) -> Vec<f64> {
        self.class_weights.clone()
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(ClassifierModel::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn class_values(&self) -> Vec<f64> {
        self.model
            .as_ref()
            .map(|model| model.class_values.clone())
            .unwrap_or_default()
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_history_json(&self) -> PyResult<String> {
        self.model
            .as_ref()
            .map(|model| serde_json::to_string(&model.training_history))
            .transpose()
            .map(|value| value.unwrap_or_else(|| "[]".to_string()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostClassifier {
    fn classifier_config(&self) -> PyResult<ClassifierConfig> {
        Ok(ClassifierConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters: parse_splitters(&self.splitters)?,
            leaf_predictor: parse_leaf_predictor(&self.leaf_predictor)?,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            objective: parse_classification_objective(&self.objective, self.class_count)?,
            class_count: self.class_count,
            class_weights: self.class_weights.clone(),
        })
    }

    fn from_model(model: ClassifierModel) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            class_weights,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                config.class_weights,
            )
        } else {
            (
                1,
                1,
                0.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                Vec::new(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective: classification_objective_name(model.objective).to_string(),
            class_count: model.class_values.len(),
            class_weights,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            model: Some(model),
        })
    }
}

#[pyclass(name = "CartoBoostRanker")]
#[derive(Clone, Debug)]
struct NativeCartoBoostRanker {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    objective: String,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    model: Option<RankerModel>,
}

#[pymethods]
impl NativeCartoBoostRanker {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        min_gain=1e-8,
        objective="lambdarank",
        splitters=None,
        leaf_predictor="constant",
        linear_leaf_features=None,
        l2_regularization=1.0,
        constant_l2_regularization=0.0,
        fuzzy=false,
        fuzzy_bandwidth=0.0,
        fuzzy_kernel="linear",
        n_threads=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        objective: &str,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        parse_ranking_objective(objective)?;
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective: objective.to_string(),
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            model: None,
        })
    }

    #[pyo3(signature = (
        x,
        y,
        groups,
        sample_weight=None,
        sparse_sets=None,
        feature_schema_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        groups: Vec<usize>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let config = self.ranker_config()?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Ranker::new(config).fit(&dataset, &y, &groups, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (
        x,
        y,
        groups,
        sample_weight=None,
        sparse_offsets=None,
        sparse_ids=None,
        feature_schema_json=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        groups: Vec<usize>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let config = self.ranker_config()?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Ranker::new(config).fit(&dataset, &targets, &groups, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.model = Some(model);
        Ok(())
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, y, groups, sparse_sets=None))]
    fn metrics(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        groups: Vec<usize>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<BTreeMap<String, f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        let metrics = py
            .allow_threads(|| {
                run_with_optional_threads(n_threads, || model.metrics(&dataset, &y, &groups))
            })
            .map_err(to_py_value_error)?;
        Ok(BTreeMap::from([
            ("ndcg".to_string(), metrics.ndcg),
            ("map".to_string(), metrics.map),
            ("mrr".to_string(), metrics.mrr),
        ]))
    }

    #[pyo3(signature = (x, y, groups, sparse_offsets=None, sparse_ids=None))]
    fn metrics_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        groups: Vec<usize>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<BTreeMap<String, f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, None)?;
        let targets = y.as_slice()?.to_vec();
        let n_threads = self.n_threads;
        let metrics = py
            .allow_threads(|| {
                run_with_optional_threads(n_threads, || model.metrics(&dataset, &targets, &groups))
            })
            .map_err(to_py_value_error)?;
        Ok(BTreeMap::from([
            ("ndcg".to_string(), metrics.ndcg),
            ("map".to_string(), metrics.map),
            ("mrr".to_string(), metrics.mrr),
        ]))
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRanker is not fitted"))?;
        py.allow_threads(|| model.save(path)).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .allow_threads(|| RankerModel::load(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn objective(&self) -> String {
        self.objective.clone()
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(RankerModel::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_history_json(&self) -> PyResult<String> {
        self.model
            .as_ref()
            .map(|model| serde_json::to_string(&model.training_history))
            .transpose()
            .map(|value| value.unwrap_or_else(|| "[]".to_string()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostRanker {
    fn ranker_config(&self) -> PyResult<RankerConfig> {
        Ok(RankerConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters: parse_splitters(&self.splitters)?,
            leaf_predictor: parse_leaf_predictor(&self.leaf_predictor)?,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            objective: parse_ranking_objective(&self.objective)?,
        })
    }

    fn from_model(model: RankerModel) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            objective,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                ranking_objective_name(config.objective).to_string(),
            )
        } else {
            (
                1,
                1,
                0.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                ranking_objective_name(model.objective).to_string(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            objective,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            model: Some(model),
        })
    }
}

fn parse_classification_objective(
    name: &str,
    class_count: usize,
) -> PyResult<ClassificationObjective> {
    match name {
        "auto" if class_count == 2 => Ok(ClassificationObjective::BinaryLogLoss),
        "auto" => Ok(ClassificationObjective::MulticlassLogLoss),
        "binary_logloss" | "logloss" | "binary" => Ok(ClassificationObjective::BinaryLogLoss),
        "multiclass_logloss" | "multi_logloss" | "multiclass" => {
            Ok(ClassificationObjective::MulticlassLogLoss)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown classification objective {name:?}; expected 'auto', 'binary_logloss', \
             or 'multiclass_logloss'"
        ))),
    }
}

fn classification_objective_name(objective: ClassificationObjective) -> &'static str {
    match objective {
        ClassificationObjective::BinaryLogLoss => "binary_logloss",
        ClassificationObjective::MulticlassLogLoss => "multiclass_logloss",
    }
}

fn parse_ranking_objective(name: &str) -> PyResult<RankingObjective> {
    match name {
        "pairwise_logit" | "pairwise" => Ok(RankingObjective::PairwiseLogit),
        "lambdarank" | "lambda_rank" => Ok(RankingObjective::LambdaRank),
        _ => Err(PyValueError::new_err(format!(
            "unknown ranking objective {name:?}; expected 'pairwise_logit' or 'lambdarank'"
        ))),
    }
}

fn ranking_objective_name(objective: RankingObjective) -> &'static str {
    match objective {
        RankingObjective::PairwiseLogit => "pairwise_logit",
        RankingObjective::LambdaRank => "lambdarank",
    }
}

#[pyfunction]
#[pyo3(signature = (
    rows,
    targets,
    feature_schema_json=None,
    sample_weight=None,
    low_cardinality_threshold=16,
    smoothing=10.0
))]
fn categorical_fit_transform(
    rows: Vec<Vec<String>>,
    targets: Vec<f64>,
    feature_schema_json: Option<String>,
    sample_weight: Option<Vec<f64>>,
    low_cardinality_threshold: usize,
    smoothing: f64,
) -> PyResult<(Vec<Vec<f64>>, String)> {
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let (dataset, encoder) = CategoricalEncoder::fit_transform_rows(
        &rows,
        &targets,
        schema.as_ref(),
        sample_weight.as_deref(),
        CategoricalEncodingConfig {
            low_cardinality_threshold,
            smoothing,
        },
    )
    .map_err(to_py_value_error)?;
    let encoder_json =
        serde_json::to_string(&encoder).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((dataset_to_rows(&dataset), encoder_json))
}

#[pyfunction]
fn categorical_transform(rows: Vec<Vec<String>>, encoder_json: String) -> PyResult<Vec<Vec<f64>>> {
    let encoder: CategoricalEncoder = serde_json::from_str(&encoder_json)
        .map_err(|err| PyValueError::new_err(format!("invalid categorical encoder: {err}")))?;
    let dataset = encoder.transform_rows(&rows).map_err(to_py_value_error)?;
    Ok(dataset_to_rows(&dataset))
}

/// Validate a serialized feature schema using the Rust core contract.
///
/// Python wrappers normalize ergonomic schema declarations, but the final
/// payload is always checked here before it crosses into dataset/model code.
/// Keeping this validation at the native boundary prevents custom Python
/// schema providers from bypassing duplicate-name, length, or periodic-field
/// checks implemented by `cartoboost-core`.
#[pyfunction]
fn validate_feature_schema_json(payload: &str) -> PyResult<()> {
    let schema: FeatureSchema = serde_json::from_str(payload)
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema JSON: {err}")))?;
    schema.validate().map_err(to_py_value_error)
}

#[pyfunction]
fn model_manifest_json() -> &'static str {
    core_model_manifest_json()
}

fn dataset_to_rows(dataset: &Dataset) -> Vec<Vec<f64>> {
    (0..dataset.n_rows())
        .map(|row| {
            (0..dataset.n_cols())
                .map(|col| dataset.get(row, col))
                .collect()
        })
        .collect()
}

fn parse_splitters(names: &[String]) -> PyResult<Vec<SplitterKind>> {
    let mut splitters = Vec::with_capacity(names.len());
    for name in names {
        let splitter = match name.as_str() {
            "auto" => SplitterKind::Auto,
            "axis" => SplitterKind::Axis,
            "axis_histogram" | "axis_hist" | "histogram" => {
                SplitterKind::AxisHistogram { bins: 64 }
            }
            "diagonal_2d" | "diagonal2d" => SplitterKind::Diagonal2D,
            "gaussian_2d" | "gaussian2d" | "radial" => SplitterKind::Gaussian2D,
            "periodic_time" | "periodic_24" => SplitterKind::Periodic { period: 24.0 },
            "sparse_set" | "sparse" => SplitterKind::SparseSet,
            _ => {
                if let Some(bins) = name
                    .strip_prefix("axis_histogram:")
                    .or_else(|| name.strip_prefix("axis_hist:"))
                    .and_then(|bins| bins.parse::<usize>().ok())
                    .filter(|bins| *bins >= 2)
                {
                    SplitterKind::AxisHistogram { bins }
                } else if let Some(period) = name
                    .strip_prefix("periodic:")
                    .and_then(|period| period.parse::<f64>().ok())
                    .filter(|period| period.is_finite() && *period > 0.0)
                {
                    SplitterKind::Periodic { period }
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unknown splitter {name:?}; expected one of 'auto', 'axis', 'axis_histogram', \
                         'diagonal_2d', 'gaussian_2d', 'periodic_time', or 'sparse_set'"
                    )));
                }
            }
        };
        splitters.push(splitter);
    }
    if splitters.is_empty() {
        Ok(vec![SplitterKind::Auto])
    } else {
        Ok(splitters)
    }
}

fn parse_global_target_mode(name: &str) -> PyResult<GlobalForecastTargetMode> {
    match name {
        "level" => Ok(GlobalForecastTargetMode::Level),
        "delta_from_last" | "delta" => Ok(GlobalForecastTargetMode::DeltaFromLast),
        seasonal if seasonal.starts_with("seasonal_delta_") => {
            parse_seasonal_delta_target_mode(&seasonal["seasonal_delta_".len()..])
        }
        seasonal if seasonal.starts_with("seasonal_delta:") => {
            parse_seasonal_delta_target_mode(&seasonal["seasonal_delta:".len()..])
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown CartoBoostLagForecaster target_mode {name:?}; expected 'level' or \
             'delta_from_last' or 'seasonal_delta_<positive season length>'"
        ))),
    }
}

fn parse_seasonal_delta_target_mode(value: &str) -> PyResult<GlobalForecastTargetMode> {
    let season_length = value.parse::<usize>().map_err(|_| {
        PyValueError::new_err(format!(
            "seasonal_delta target_mode requires a positive integer season length, got {value:?}"
        ))
    })?;
    if season_length == 0 {
        return Err(PyValueError::new_err(
            "seasonal_delta target_mode requires a positive season length",
        ));
    }
    Ok(GlobalForecastTargetMode::SeasonalDelta { season_length })
}

#[pyclass(name = "NeuralEmbeddingFeatures")]
#[derive(Clone)]
struct NativeNeuralEmbeddingFeatures {
    dim: usize,
    fallback: ArtifactFallbackKind,
    random_state: Option<i64>,
    parent_resolution: Option<u8>,
    support_prior_strength: f64,
    table: Option<EmbeddingTable>,
}

#[pymethods]
impl NativeNeuralEmbeddingFeatures {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, parent_resolution=None, support_prior_strength=1.0))]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<i64>,
        parent_resolution: Option<u8>,
        support_prior_strength: f64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be positive"));
        }
        if !support_prior_strength.is_finite() || support_prior_strength <= 0.0 {
            return Err(PyValueError::new_err(
                "support_prior_strength must be positive and finite",
            ));
        }

        let fallback = parse_embedding_fallback(fallback, parent_resolution)?;

        Ok(Self {
            dim,
            fallback,
            random_state,
            parent_resolution,
            support_prior_strength,
            table: None,
        })
    }

    #[pyo3(signature = (ids, target))]
    fn fit(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let ids = ids.as_slice()?.to_vec();
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);

        let table = py
            .allow_threads(|| {
                fit_embedding_table_with_options(
                    self.dim,
                    &ids,
                    &target,
                    self.fallback.clone(),
                    random_state,
                    self.support_prior_strength,
                )
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        Ok(())
    }

    #[pyo3(signature = (ids, target))]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let ids = ids.as_slice()?.to_vec();
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);
        let (table, block) = py
            .allow_threads(|| {
                let table = fit_embedding_table_with_options(
                    self.dim,
                    &ids,
                    &target,
                    self.fallback.clone(),
                    random_state,
                    self.support_prior_strength,
                )?;
                let block = table.encode_ids(&ids, "neural_embedding")?;
                Ok::<_, cartoboost_neural::NeuralError>((table, block))
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (ids))]
    fn transform(&self, py: Python<'_>, ids: PyReadonlyArray1<'_, u64>) -> PyResult<Vec<Vec<f32>>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("transform called before fit or load"))?;

        let ids = ids.as_slice()?.to_vec();
        let block = py
            .allow_threads(|| table.encode_ids(&ids, "neural_embedding"))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (path))]
    fn export(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("export called before fit or load"))?;

        py.allow_threads(|| {
            let artifact = build_embedding_table_artifact(
                self.dim,
                table.rows().to_vec(),
                table.artifact_metadata().fallback.clone(),
            )?;
            write_embedding_table_artifact(path, &artifact)
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[classmethod]
    fn from_artifact(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        let table = py
            .allow_threads(|| EmbeddingTable::load(path))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let metadata = table.artifact_metadata().clone();
        let parent_resolution = match metadata.fallback {
            ArtifactFallbackKind::ParentCell { parent_resolution } => Some(parent_resolution),
            _ => None,
        };

        Ok(Self {
            dim: metadata.dim,
            fallback: metadata.fallback,
            random_state: None,
            parent_resolution,
            support_prior_strength: 1.0,
            table: Some(table),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn fallback(&self) -> String {
        artifact_fallback_name(&self.fallback).to_string()
    }

    #[getter]
    fn random_state(&self) -> Option<i64> {
        self.random_state
    }

    #[getter]
    fn parent_resolution(&self) -> Option<u8> {
        self.parent_resolution
    }

    #[getter]
    fn support_prior_strength(&self) -> f64 {
        self.support_prior_strength
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.table.is_some()
    }

    fn artifact_rows(&self) -> PyResult<Vec<(u64, Vec<f32>)>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("artifact_rows called before fit or load"))?;
        Ok(table
            .rows()
            .iter()
            .map(|row| (row.id, row.values.clone()))
            .collect())
    }
}

fn parse_embedding_fallback(
    value: &str,
    parent_resolution: Option<u8>,
) -> PyResult<ArtifactFallbackKind> {
    match value {
        "zero_vector" => Ok(ArtifactFallbackKind::ZeroVector),
        "global_mean_vector" => Ok(ArtifactFallbackKind::GlobalMeanVector),
        "parent_cell" => parent_resolution
            .map(|parent_resolution| ArtifactFallbackKind::ParentCell { parent_resolution })
            .ok_or_else(|| PyValueError::new_err("parent_resolution is required for parent_cell")),
        _ => Err(PyValueError::new_err(
            "fallback must be one of zero_vector, global_mean_vector, parent_cell",
        )),
    }
}

#[pyclass(name = "GraphSageEncoder")]
#[derive(Clone)]
struct NativeGraphSageEncoder {
    config: GraphSageConfig,
    encoder: GraphSageEncoder,
}

#[pymethods]
impl NativeGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };

        let encoder =
            GraphSageEncoder::new(config.clone(), input_dim).map_err(to_py_neural_error)?;

        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let mut model = GraphSageEncoder::new(self.config.clone(), self.encoder.input_dim())
            .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| GraphSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_add_self_loop(&self) -> bool {
        self.config.add_self_loop
    }

    #[getter]
    fn config_l2_regularization(&self) -> f32 {
        self.config.l2_regularization
    }

    #[getter]
    fn hidden_dims(&self) -> Vec<usize> {
        self.config.hidden_dims.clone()
    }
}

#[pyclass(name = "Node2VecEncoder")]
#[derive(Clone)]
struct NativeNode2VecEncoder {
    config: Node2VecConfig,
    encoder: Node2VecEncoder,
}

#[pymethods]
impl NativeNode2VecEncoder {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let encoder = Node2VecEncoder::new(config.clone()).map_err(to_py_neural_error)?;
        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let mut model = Node2VecEncoder::new(self.config.clone()).map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(node_count, &edges, edge_weights.as_deref()))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode())
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| Node2VecEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.encoder.node_count()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_p(&self) -> f32 {
        self.config.p
    }

    #[getter]
    fn config_q(&self) -> f32 {
        self.config.q
    }
}

#[pyclass(name = "StandaloneNeuralEmbeddingRegressor")]
#[derive(Clone)]
struct NativeStandaloneNeuralEmbeddingRegressor {
    model: StandaloneNeuralEmbeddingRegressor,
}

#[pymethods]
impl NativeStandaloneNeuralEmbeddingRegressor {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, support_prior_strength=1.0, n_estimators=80, learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<u64>,
        support_prior_strength: f64,
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let fallback = parse_embedding_fallback(fallback, None)?;
        let model = StandaloneNeuralEmbeddingRegressor::new(
            dim,
            fallback,
            random_state,
            support_prior_strength,
            standalone_booster_config(
                n_estimators,
                learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (ids, y, dense=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        ids: Vec<u64>,
        y: Vec<f64>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&ids, &y, dense.as_deref()))
            .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (ids, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        ids: Vec<u64>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict(&ids, dense.as_deref()))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| StandaloneNeuralEmbeddingRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneNode2VecRegressor")]
#[derive(Clone)]
struct NativeStandaloneNode2VecRegressor {
    model: Node2VecRegressor,
}

#[pymethods]
impl NativeStandaloneNode2VecRegressor {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let model = Node2VecRegressor::new(
            config,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_count, edges, row_nodes, y, row_targets=None, dense=None, edge_weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                node_count,
                &edges,
                edge_weights.as_deref(),
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model
                .predict(&row_nodes, row_targets.as_deref(), dense.as_deref())
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Node2VecRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneGraphSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneGraphSageRegressor {
    model: GraphSageRegressor,
}

#[pymethods]
impl NativeStandaloneGraphSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = GraphSageRegressor::new(
            config,
            input_dim,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                &node_features,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| GraphSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHeteroGraphSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneHeteroGraphSageRegressor {
    model: HeteroGraphSageRegressor,
}

#[pymethods]
impl NativeStandaloneHeteroGraphSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = HeteroGraphSageRegressor::new(
            config,
            input_dim,
            relation_count,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                &node_features,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HeteroGraphSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHinSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneHinSageRegressor {
    model: HinSageRegressor,
}

#[pymethods]
impl NativeStandaloneHinSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = HinSageRegressor::new(
            config,
            input_dim,
            node_type_count,
            edge_type_triples,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, node_types, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                &node_features,
                &node_types,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HinSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneNode2VecLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneNode2VecLinkPredictor {
    model: Node2VecLinkPredictor,
}

#[pymethods]
impl NativeStandaloneNode2VecLinkPredictor {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let model = Node2VecLinkPredictor::new(config).map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(node_count, &edges, edge_weights.as_deref()))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(&self, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Node2VecLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneGraphSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneGraphSageLinkPredictor {
    model: GraphSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneGraphSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = GraphSageLinkPredictor::new(config, input_dim).map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&node_features, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| GraphSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHeteroGraphSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneHeteroGraphSageLinkPredictor {
    model: HeteroGraphSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneHeteroGraphSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model = HeteroGraphSageLinkPredictor::new(config, input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize, usize)>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&node_features, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HeteroGraphSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHinSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneHinSageLinkPredictor {
    model: HinSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneHinSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let model =
            HinSageLinkPredictor::new(config, input_dim, node_type_count, edge_type_triples)
                .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&node_features, &node_types, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HinSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "HeteroGraphSageEncoder")]
#[derive(Clone)]
struct NativeHeteroGraphSageEncoder {
    config: HeteroGraphSageConfig,
    relation_count: usize,
    encoder: HeteroGraphSageEncoder,
}

#[pymethods]
impl NativeHeteroGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let encoder = HeteroGraphSageEncoder::new(config.clone(), input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            relation_count,
            encoder,
        })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let mut model = HeteroGraphSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.relation_count,
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| HeteroGraphSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            relation_count: encoder.relation_count(),
            config,
            encoder,
        })
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.relation_count
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyclass(name = "HinSageEncoder")]
#[derive(Clone)]
struct NativeHinSageEncoder {
    config: HinSageConfig,
    node_type_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    encoder: HinSageEncoder,
}

#[pymethods]
impl NativeHinSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, backend=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        backend: Option<&str>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
            backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
        };
        let encoder = HinSageEncoder::new(
            config.clone(),
            input_dim,
            node_type_count,
            edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            node_type_count,
            edge_type_triples,
            encoder,
        })
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let mut model = HinSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.node_type_count,
            self.edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn link_embeddings(
        &self,
        py: Python<'_>,
        embeddings: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<Vec<f32>>> {
        py.allow_threads(|| self.encoder.link_embeddings(&embeddings, &pairs))
            .map_err(to_py_neural_error)
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| HinSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            node_type_count: encoder.node_type_count(),
            edge_type_triples: encoder.edge_type_triples().to_vec(),
            config,
            encoder,
        })
    }

    #[getter]
    fn node_type_count(&self) -> usize {
        self.node_type_count
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.edge_type_triples.len()
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn edge_type_triples(&self) -> Vec<(usize, usize, usize)> {
        self.edge_type_triples.clone()
    }

    #[getter]
    fn neighbor_samples(&self) -> Vec<usize> {
        self.config.neighbor_samples.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyfunction]
#[pyo3(signature = (node_count, edges, embeddings, edge_weights=None, edge_timestamps=None, feature_prefix="graph", requested_features=None))]
#[allow(clippy::too_many_arguments)]
fn graph_compute_directional_features(
    py: Python<'_>,
    node_count: usize,
    edges: Vec<(usize, usize)>,
    embeddings: Vec<Vec<f32>>,
    edge_weights: Option<Vec<f32>>,
    edge_timestamps: Option<Vec<f32>>,
    feature_prefix: &str,
    requested_features: Option<Vec<String>>,
) -> PyResult<(Vec<Vec<f32>>, Vec<String>)> {
    let requested_features = requested_features.unwrap_or_default();
    let block = py
        .allow_threads(|| {
            compute_directional_features(
                node_count,
                &edges,
                &embeddings,
                edge_weights.as_deref(),
                edge_timestamps.as_deref(),
                feature_prefix,
                &requested_features,
            )
        })
        .map_err(to_py_neural_error)?;
    Ok((block.values, block.feature_names))
}

#[pyfunction]
fn graph_validate_directed_metapath(
    py: Python<'_>,
    steps: Vec<String>,
    edge_types: Vec<(String, String, String)>,
) -> PyResult<()> {
    py.allow_threads(|| validate_directed_metapath(&steps, &edge_types))
        .map_err(to_py_neural_error)
}

#[pyfunction]
#[pyo3(signature = (edges, source_to_pair_relation="source_to_pair", pair_to_target_relation="pair_to_target", pair_node_prefix="od_pair", include_original_edges=true))]
fn graph_materialize_source_target_pair_nodes(
    py: Python<'_>,
    edges: Vec<(String, String, String)>,
    source_to_pair_relation: &str,
    pair_to_target_relation: &str,
    pair_node_prefix: &str,
    include_original_edges: bool,
) -> PyResult<(StringTypedEdges, Vec<String>)> {
    let source_to_pair_relation = source_to_pair_relation.to_string();
    let pair_to_target_relation = pair_to_target_relation.to_string();
    let pair_node_prefix = pair_node_prefix.to_string();
    let expansion = py
        .allow_threads(|| {
            materialize_source_target_pair_nodes(
                &edges,
                &source_to_pair_relation,
                &pair_to_target_relation,
                &pair_node_prefix,
                include_original_edges,
            )
        })
        .map_err(to_py_neural_error)?;
    Ok((expansion.edges, expansion.pair_node_ids))
}

#[pyfunction]
#[pyo3(signature = (train, seasonal_period=1))]
fn rmsse_scale_value(py: Python<'_>, train: Vec<f64>, seasonal_period: usize) -> PyResult<f64> {
    py.allow_threads(|| core_rmsse_scale(&train, seasonal_period))
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (series, seasonal_period=1))]
fn wrmsse_value(
    py: Python<'_>,
    series: Vec<PyWrmsseSeries>,
    seasonal_period: usize,
) -> PyResult<String> {
    let series = series
        .into_iter()
        .map(|(id, train, actual, forecast, weight)| {
            WrmsseSeries::new(id, train, actual, forecast, weight)
        })
        .collect::<Vec<_>>();
    let score = py
        .allow_threads(|| core_wrmsse(&series, seasonal_period))
        .map_err(to_py_value_error)?;
    let payload = json!({
        "wrmsse": score.score,
        "series": score
            .series
            .into_iter()
            .map(|row| {
                json!({
                    "series_id": row.id,
                    "weight": row.weight,
                    "normalized_weight": row.normalized_weight,
                    "scale": row.scale,
                    "rmsse": row.rmsse,
                    "contribution": row.contribution,
                })
            })
            .collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn aggregate_equal_level_wrmsse_value(
    py: Python<'_>,
    level_scores: Vec<(String, f64)>,
) -> PyResult<String> {
    let score = py
        .allow_threads(|| core_aggregate_equal_level_wrmsse(&level_scores))
        .map_err(to_py_value_error)?;
    let payload = json!({
        "wrmsse": score.score,
        "levels": score
            .levels
            .into_iter()
            .map(|row| {
                json!({
                    "level": row.level,
                    "wrmsse": row.wrmsse,
                    "level_weight": row.level_weight,
                    "contribution": row.contribution,
                })
            })
            .collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn ordered_nonnegative_weights_value(
    py: Python<'_>,
    ids: Vec<String>,
    raw_weights: Vec<(String, f64)>,
) -> PyResult<BTreeMap<String, f64>> {
    py.allow_threads(|| core_ordered_nonnegative_weights(&ids, &raw_weights))
        .map(|weights| weights.into_iter().collect())
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (training_series, actuals, forecasts, seasonality, baseline_smape=None, baseline_mase=None))]
fn competition_forecast_metrics_value(
    py: Python<'_>,
    training_series: Vec<Vec<f64>>,
    actuals: Vec<f64>,
    forecasts: Vec<f64>,
    seasonality: usize,
    baseline_smape: Option<f64>,
    baseline_mase: Option<f64>,
) -> PyResult<String> {
    let baseline = match (baseline_smape, baseline_mase) {
        (Some(smape), Some(mase)) => Some((smape, mase)),
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "baseline_smape and baseline_mase must be provided together",
            ));
        }
    };
    let metrics = py
        .allow_threads(|| {
            evaluate_competition_metrics(
                &training_series,
                &actuals,
                &forecasts,
                seasonality,
                baseline,
            )
        })
        .map_err(to_py_value_error)?;
    serde_json::to_string(&metrics).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (source, candidate_scores, inner_origin_count=None))]
fn forecast_candidate_choice_value(
    py: Python<'_>,
    source: &str,
    candidate_scores: BTreeMap<String, f64>,
    inner_origin_count: Option<usize>,
) -> PyResult<String> {
    let source = source.to_string();
    py.allow_threads(|| {
        CoreCandidateSelectionPolicy::new(source, inner_origin_count)?
            .select(&candidate_scores)
            .map(|selection| selection.candidate)
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_validation_unavailable_candidate_choice_value(
    py: Python<'_>,
    model: &str,
    validation_profile: &str,
    available_candidates: Vec<String>,
) -> PyResult<String> {
    let model = model.to_string();
    let validation_profile = validation_profile.to_string();
    py.allow_threads(|| {
        core_validation_unavailable_candidate_choice(
            &model,
            &validation_profile,
            &available_candidates,
        )
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (timestamp_count, horizon, validation_profile=None))]
fn forecast_candidate_validation_cutoff_indices_value(
    py: Python<'_>,
    timestamp_count: usize,
    horizon: usize,
    validation_profile: Option<String>,
) -> PyResult<Vec<usize>> {
    py.allow_threads(|| {
        CoreCandidateValidationCutoffSchedule::new(
            timestamp_count,
            horizon,
            validation_profile.as_deref(),
        )
        .map(|schedule| schedule.cutoff_indices)
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_magnitude_guard_allows_value(
    py: Python<'_>,
    forecast_max_abs: f64,
    training_max_abs: f64,
) -> PyResult<bool> {
    py.allow_threads(|| forecast_magnitude_guard_allows(forecast_max_abs, training_max_abs))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_requires_lag_spine_value(
    py: Python<'_>,
    source: &str,
    season_length: usize,
    horizon: usize,
) -> PyResult<bool> {
    let source = source.to_string();
    Ok(py.allow_threads(|| core_requires_lag_spine(&source, season_length, horizon)))
}

#[pyfunction]
fn forecast_seasonal_naive_candidate_value(
    py: Python<'_>,
    values: Vec<f64>,
    season_length: usize,
) -> PyResult<f64> {
    py.allow_threads(|| core_seasonal_naive_candidate_prediction(&values, season_length))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_trend_candidate_value(
    py: Python<'_>,
    values: Vec<f64>,
    step: usize,
    season_length: usize,
    mode: &str,
) -> PyResult<f64> {
    let mode = mode.to_string();
    py.allow_threads(|| core_trend_candidate_prediction(&values, step, season_length, &mode))
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (values, day_of_months, target_day_of_month, mode, elapsed_phase_period=None))]
fn forecast_calendar_profile_candidate_value(
    py: Python<'_>,
    values: Vec<f64>,
    day_of_months: Vec<u32>,
    target_day_of_month: u32,
    mode: &str,
    elapsed_phase_period: Option<usize>,
) -> PyResult<f64> {
    let mode = mode.to_string();
    py.allow_threads(|| {
        core_calendar_profile_candidate_prediction(
            &values,
            &day_of_months,
            target_day_of_month,
            &mode,
            elapsed_phase_period,
        )
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_validation_ensemble_weights_value(
    py: Python<'_>,
    candidate_scores: BTreeMap<String, f64>,
) -> PyResult<BTreeMap<String, f64>> {
    py.allow_threads(|| core_validation_ensemble_weights(&candidate_scores))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_shared_candidate_names_value(py: Python<'_>) -> PyResult<Vec<String>> {
    Ok(py.allow_threads(core_shared_candidate_names))
}

#[pyfunction]
fn forecast_selectable_candidate_names_value(
    py: Python<'_>,
    model: &str,
    source: &str,
) -> PyResult<Vec<String>> {
    let model = model.to_string();
    let source = source.to_string();
    Ok(py.allow_threads(|| core_selectable_candidate_names(&model, &source)))
}

#[pyfunction]
fn forecast_include_autostats_candidate_value(
    py: Python<'_>,
    source: &str,
    season_length: usize,
    horizon: usize,
) -> PyResult<bool> {
    let source = source.to_string();
    Ok(py.allow_threads(|| core_include_autostats_candidate(&source, season_length, horizon)))
}

#[pyfunction]
fn forecast_candidate_complexity_rank_value(py: Python<'_>, candidate: &str) -> PyResult<usize> {
    let candidate = candidate.to_string();
    Ok(py.allow_threads(|| core_candidate_complexity_rank(&candidate)))
}

#[pyfunction(signature = (selected_candidate=None, inner_raw_relative_rmse_gain=None))]
fn forecast_native_auto_raw_candidate_is_confident_value(
    py: Python<'_>,
    selected_candidate: Option<String>,
    inner_raw_relative_rmse_gain: Option<f64>,
) -> PyResult<bool> {
    Ok(py.allow_threads(|| {
        core_native_auto_raw_candidate_is_confident(
            selected_candidate.as_deref(),
            inner_raw_relative_rmse_gain,
        )
    }))
}

#[pyfunction]
fn forecast_lag_origin_consistency_guard_value(
    py: Python<'_>,
    candidate: &str,
    source: &str,
    lag_scores: Vec<f64>,
    candidate_scores: Vec<f64>,
) -> PyResult<Option<String>> {
    let candidate = candidate.to_string();
    let source = source.to_string();
    py.allow_threads(|| {
        core_lag_origin_consistency_guard(&candidate, &source, &lag_scores, &candidate_scores)
            .map(|guard| guard.map(|value| value.to_string()))
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_relative_loss_displacement_allowed_value(
    py: Python<'_>,
    baseline_loss: f64,
    candidate_loss: f64,
    min_relative_gain: f64,
) -> PyResult<bool> {
    py.allow_threads(|| {
        core_relative_loss_displacement_allowed(baseline_loss, candidate_loss, min_relative_gain)
    })
    .map_err(to_py_value_error)
}

#[pyfunction(signature = (
    selected_candidate,
    candidate_scores,
    candidate_forecast_max_abs,
    training_max_abs,
    inner_origin_count=None
))]
fn forecast_stable_magnitude_candidate_choice_value(
    py: Python<'_>,
    selected_candidate: &str,
    candidate_scores: BTreeMap<String, f64>,
    candidate_forecast_max_abs: BTreeMap<String, f64>,
    training_max_abs: f64,
    inner_origin_count: Option<usize>,
) -> PyResult<String> {
    let selected_candidate = selected_candidate.to_string();
    py.allow_threads(|| {
        core_stable_magnitude_candidate_choice(
            &selected_candidate,
            &candidate_scores,
            &candidate_forecast_max_abs,
            training_max_abs,
            inner_origin_count,
        )
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_proportional_total_reconciliation_value(
    py: Python<'_>,
    base_values: Vec<f64>,
    target_total: f64,
    gamma: f64,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| core_proportional_total_reconciliation(&base_values, target_total, gamma))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_weighted_blend_candidate_value(
    py: Python<'_>,
    primary_forecast: Vec<f64>,
    secondary_forecast: Vec<f64>,
    primary_weight: f64,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        core_weighted_blend_candidate_forecast(
            &primary_forecast,
            &secondary_forecast,
            primary_weight,
        )
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn prob_pinball_loss_value(actual: Vec<f64>, prediction: Vec<f64>, quantile: f64) -> PyResult<f64> {
    core_prob_pinball_loss(&actual, &prediction, quantile).map_err(to_py_value_error)
}

#[pyfunction]
fn prob_interval_coverage_value(
    actual: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
) -> PyResult<f64> {
    core_prob_interval_coverage(&actual, &lower, &upper).map_err(to_py_value_error)
}

#[pyfunction]
fn prob_mean_interval_width_value(lower: Vec<f64>, upper: Vec<f64>) -> PyResult<f64> {
    core_prob_mean_interval_width(&lower, &upper).map_err(to_py_value_error)
}

#[pyfunction]
fn prob_crps_approximation_value(
    actual: Vec<f64>,
    quantiles: Vec<f64>,
    predictions: Vec<Vec<f64>>,
) -> PyResult<f64> {
    core_prob_crps_approximation(&actual, &quantiles, &predictions).map_err(to_py_value_error)
}

#[pyfunction]
fn prob_weighted_interval_score_value(
    actual: Vec<f64>,
    median: Vec<f64>,
    intervals: Vec<(f64, Vec<f64>, Vec<f64>)>,
) -> PyResult<f64> {
    core_prob_weighted_interval_score(&actual, &median, &intervals).map_err(to_py_value_error)
}

#[pyfunction]
fn prob_pit_bins_value(
    actual: Vec<f64>,
    quantiles: Vec<f64>,
    predictions: Vec<Vec<f64>>,
    bins: usize,
) -> PyResult<String> {
    let bins =
        core_prob_pit_bins(&actual, &quantiles, &predictions, bins).map_err(to_py_value_error)?;
    serde_json::to_string(&bins).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
fn prob_conditional_flow_fit_value(
    hidden: Vec<Vec<f64>>,
    residuals: Vec<f64>,
    quantiles: Vec<f64>,
    sample_count: usize,
) -> PyResult<String> {
    core_prob_conditional_flow_fit_json(&hidden, &residuals, &quantiles, sample_count)
        .map_err(to_py_value_error)
}

#[pyfunction(signature = (artifact_json, hidden, actual=None))]
fn prob_conditional_flow_predict_value(
    artifact_json: String,
    hidden: Vec<Vec<f64>>,
    actual: Option<Vec<f64>>,
) -> PyResult<String> {
    core_prob_conditional_flow_predict_json(&artifact_json, &hidden, actual.as_deref())
        .map_err(to_py_value_error)
}

#[pyfunction]
fn prob_diffusion_scenario_generate_value(
    point_forecast: Vec<Vec<f64>>,
    edges: Vec<(usize, usize, f64)>,
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
) -> PyResult<String> {
    let edges = edges
        .into_iter()
        .map(|(source, target, weight)| CoreDiffusionEdge {
            source,
            target,
            weight,
        })
        .collect::<Vec<_>>();
    core_prob_diffusion_scenario_generate_json(
        &point_forecast,
        &edges,
        scenario_count,
        diffusion_steps,
        shock_scale,
    )
    .map_err(to_py_value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prob_split_conformal_residual_quantile_value(
    actual: Vec<f64>,
    prediction: Vec<f64>,
    alpha: f64,
    train_end_exclusive: usize,
    calibration_start: usize,
    calibration_end_exclusive: usize,
    test_start: usize,
) -> PyResult<f64> {
    core_prob_split_conformal_residual_quantile(
        &actual,
        &prediction,
        alpha,
        CoreProbSplitOrder {
            train_end_exclusive,
            calibration_start,
            calibration_end_exclusive,
            test_start,
        },
    )
    .map_err(to_py_value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prob_weighted_conformal_residual_quantile_value(
    actual: Vec<f64>,
    prediction: Vec<f64>,
    weights: Vec<f64>,
    alpha: f64,
    train_end_exclusive: usize,
    calibration_start: usize,
    calibration_end_exclusive: usize,
    test_start: usize,
) -> PyResult<f64> {
    core_prob_weighted_conformal_residual_quantile(
        &actual,
        &prediction,
        &weights,
        alpha,
        CoreProbSplitOrder {
            train_end_exclusive,
            calibration_start,
            calibration_end_exclusive,
            test_start,
        },
    )
    .map_err(to_py_value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prob_group_conformal_residual_quantiles_value(
    actual: Vec<f64>,
    prediction: Vec<f64>,
    groups: Vec<String>,
    alpha: f64,
    train_end_exclusive: usize,
    calibration_start: usize,
    calibration_end_exclusive: usize,
    test_start: usize,
) -> PyResult<String> {
    let values = core_prob_group_conformal_residual_quantiles(
        &actual,
        &prediction,
        &groups,
        alpha,
        CoreProbSplitOrder {
            train_end_exclusive,
            calibration_start,
            calibration_end_exclusive,
            test_start,
        },
    )
    .map_err(to_py_value_error)?;
    serde_json::to_string(&values).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
fn prob_rolling_origin_conformal_residual_quantiles_value(
    actual: Vec<f64>,
    prediction: Vec<f64>,
    cutoffs: Vec<usize>,
    alpha: f64,
) -> PyResult<Vec<f64>> {
    core_prob_rolling_origin_conformal_residual_quantiles(&actual, &prediction, &cutoffs, alpha)
        .map_err(to_py_value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prob_nearest_conformal_residual_quantiles_value(
    actual: Vec<f64>,
    prediction: Vec<f64>,
    calibration_x: Vec<f64>,
    calibration_y: Vec<f64>,
    query_x: Vec<f64>,
    query_y: Vec<f64>,
    neighbor_count: usize,
    alpha: f64,
    train_end_exclusive: usize,
    calibration_start: usize,
    calibration_end_exclusive: usize,
    test_start: usize,
) -> PyResult<Vec<f64>> {
    core_prob_nearest_calibration_residual_quantiles(
        &actual,
        &prediction,
        &calibration_x,
        &calibration_y,
        &query_x,
        &query_y,
        neighbor_count,
        alpha,
        CoreProbSplitOrder {
            train_end_exclusive,
            calibration_start,
            calibration_end_exclusive,
            test_start,
        },
    )
    .map_err(to_py_value_error)
}

#[pyfunction(signature = (
    actual,
    lower,
    upper,
    horizons,
    spatial_blocks,
    residual_morans_i_after_calibration=None,
))]
fn prob_benchmark_calibration_report_fields_value(
    actual: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    horizons: Vec<usize>,
    spatial_blocks: Vec<String>,
    residual_morans_i_after_calibration: Option<f64>,
) -> PyResult<String> {
    let fields = core_prob_benchmark_calibration_report_fields(
        &actual,
        &lower,
        &upper,
        &horizons,
        &spatial_blocks,
        residual_morans_i_after_calibration,
    )
    .map_err(to_py_value_error)?;
    serde_json::to_string(&fields).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
fn extreme_portfolio_decisions_value(
    py: Python<'_>,
    asset_rows: Vec<(String, f64, f64)>,
) -> PyResult<Vec<PyPortfolioDecisionRow>> {
    let rows = asset_rows
        .into_iter()
        .map(
            |(series_id, actual_return, predicted_return)| PortfolioAsset {
                series_id,
                actual_return,
                predicted_return,
            },
        )
        .collect::<Vec<_>>();
    let decisions = py
        .allow_threads(|| extreme_portfolio_decisions(&rows))
        .map_err(to_py_value_error)?;
    Ok(decisions
        .into_iter()
        .map(|decision| {
            let side = match decision.side {
                PortfolioSide::Long => "long",
                PortfolioSide::Short => "short",
            };
            (
                decision.series_id,
                side.to_string(),
                decision.weight,
                decision.actual_return,
                decision.predicted_return,
            )
        })
        .collect())
}

#[pyfunction]
fn portfolio_summary_value(
    py: Python<'_>,
    decisions: Vec<(String, f64, f64, f64)>,
) -> PyResult<BTreeMap<String, f64>> {
    let parsed = decisions
        .into_iter()
        .map(|(side, weight, actual_return, predicted_return)| {
            let side = match side.as_str() {
                "long" => Ok(PortfolioSide::Long),
                "short" => Ok(PortfolioSide::Short),
                _ => Err(PyValueError::new_err("side must be 'long' or 'short'")),
            }?;
            Ok(PortfolioDecision {
                side,
                weight,
                actual_return,
                predicted_return,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let summary = py
        .allow_threads(|| portfolio_summary(&parsed))
        .map_err(to_py_value_error)?;
    Ok(BTreeMap::from([
        ("long_count".to_string(), summary.long_count as f64),
        ("short_count".to_string(), summary.short_count as f64),
        ("gross_exposure".to_string(), summary.gross_exposure),
        ("net_exposure".to_string(), summary.net_exposure),
        ("long_return".to_string(), summary.long_return),
        ("short_return".to_string(), summary.short_return),
        ("net_return".to_string(), summary.net_return),
    ]))
}

#[pyfunction]
#[pyo3(signature = (asset_rows, bucket_count=5))]
fn rank_hit_rates_value(
    py: Python<'_>,
    asset_rows: Vec<(usize, usize)>,
    bucket_count: usize,
) -> PyResult<BTreeMap<String, f64>> {
    let rows = asset_rows
        .into_iter()
        .map(|(observed_bucket, predicted_bucket)| RankBucketPrediction {
            observed_bucket,
            predicted_bucket,
        })
        .collect::<Vec<_>>();
    let summary = py
        .allow_threads(|| rank_hit_rates(&rows, bucket_count))
        .map_err(to_py_value_error)?;
    Ok(BTreeMap::from([
        ("asset_count".to_string(), summary.asset_count as f64),
        ("exact_bucket_rate".to_string(), summary.exact_bucket_rate),
        (
            "within_one_bucket_rate".to_string(),
            summary.within_one_bucket_rate,
        ),
        (
            "directional_extreme_count".to_string(),
            summary.directional_extreme_count as f64,
        ),
        (
            "directional_extreme_rate".to_string(),
            summary.directional_extreme_rate,
        ),
    ]))
}

#[pyfunction]
#[pyo3(signature = (values, bucket_count=5))]
fn rank_buckets_value(
    py: Python<'_>,
    values: Vec<f64>,
    bucket_count: usize,
) -> PyResult<Vec<usize>> {
    py.allow_threads(|| rank_buckets(&values, bucket_count))
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (asset_rows, bucket_count, calibration_probabilities, shrinkage))]
fn rank_scored_assets_value(
    py: Python<'_>,
    asset_rows: Vec<(String, f64, f64)>,
    bucket_count: usize,
    calibration_probabilities: Vec<Vec<f64>>,
    shrinkage: f64,
) -> PyResult<String> {
    let rows = asset_rows
        .into_iter()
        .map(
            |(series_id, actual_return, predicted_return)| PortfolioAsset {
                series_id,
                actual_return,
                predicted_return,
            },
        )
        .collect::<Vec<_>>();
    let scored = py
        .allow_threads(|| {
            rank_scored_assets(&rows, bucket_count, &calibration_probabilities, shrinkage)
        })
        .map_err(to_py_value_error)?;
    let payload = scored
        .into_iter()
        .map(|row| {
            json!({
                "series_id": row.series_id,
                "actual_return": row.actual_return,
                "predicted_return": row.predicted_return,
                "observed_rank_bucket": row.observed_rank_bucket,
                "predicted_rank_bucket": row.predicted_rank_bucket,
                "rank_probabilities": row.rank_probabilities,
                "rps": row.rps,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (asset_rows, bucket_count, calibration_probabilities, shrinkage))]
fn rank_portfolio_summary_value(
    py: Python<'_>,
    asset_rows: Vec<(String, f64, f64)>,
    bucket_count: usize,
    calibration_probabilities: Vec<Vec<f64>>,
    shrinkage: f64,
) -> PyResult<String> {
    let rows = asset_rows
        .into_iter()
        .map(
            |(series_id, actual_return, predicted_return)| PortfolioAsset {
                series_id,
                actual_return,
                predicted_return,
            },
        )
        .collect::<Vec<_>>();
    let summary = py
        .allow_threads(|| {
            rank_portfolio_summary(&rows, bucket_count, &calibration_probabilities, shrinkage)
        })
        .map_err(to_py_value_error)?;
    let assets = summary
        .assets
        .into_iter()
        .map(|row| {
            json!({
                "series_id": row.series_id,
                "actual_return": row.actual_return,
                "predicted_return": row.predicted_return,
                "observed_rank_bucket": row.observed_rank_bucket,
                "predicted_rank_bucket": row.predicted_rank_bucket,
                "rank_probabilities": row.rank_probabilities,
                "rps": row.rps,
            })
        })
        .collect::<Vec<_>>();
    let decisions = summary
        .decisions
        .into_iter()
        .map(|decision| {
            let side = match decision.side {
                PortfolioSide::Long => "long",
                PortfolioSide::Short => "short",
            };
            json!({
                "series_id": decision.series_id,
                "side": side,
                "weight": decision.weight,
                "actual_return": decision.actual_return,
                "predicted_return": decision.predicted_return,
            })
        })
        .collect::<Vec<_>>();
    let payload = json!({
        "mean_rps": summary.mean_rps,
        "asset_count": summary.asset_count,
        "assets": assets,
        "decisions": decisions,
        "decision_return": summary.portfolio.net_return,
        "portfolio": {
            "long_count": summary.portfolio.long_count,
            "short_count": summary.portfolio.short_count,
            "gross_exposure": summary.portfolio.gross_exposure,
            "net_exposure": summary.portfolio.net_exposure,
            "long_return": summary.portfolio.long_return,
            "short_return": summary.portfolio.short_return,
            "net_return": summary.portfolio.net_return,
        },
        "rank_hit_rates": {
            "asset_count": summary.hit_rates.asset_count,
            "exact_bucket_rate": summary.hit_rates.exact_bucket_rate,
            "within_one_bucket_rate": summary.hit_rates.within_one_bucket_rate,
            "directional_extreme_count": summary.hit_rates.directional_extreme_count,
            "directional_extreme_rate": summary.hit_rates.directional_extreme_rate,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn rank_portfolio_decision_loss_value(
    py: Python<'_>,
    asset_rows: Vec<(String, f64, f64)>,
    bucket_count: usize,
    calibration_probabilities: Vec<Vec<f64>>,
    shrinkage: f64,
    rps_tiebreak_weight: f64,
) -> PyResult<f64> {
    let rows = asset_rows
        .into_iter()
        .map(
            |(series_id, actual_return, predicted_return)| PortfolioAsset {
                series_id,
                actual_return,
                predicted_return,
            },
        )
        .collect::<Vec<_>>();
    py.allow_threads(|| {
        rank_portfolio_decision_loss(
            &rows,
            bucket_count,
            &calibration_probabilities,
            shrinkage,
            rps_tiebreak_weight,
        )
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn rank_probability_calibration_value(
    py: Python<'_>,
    actual_buckets: Vec<usize>,
    predicted_buckets: Vec<usize>,
    bucket_count: usize,
    validation_support: usize,
) -> PyResult<String> {
    let calibration = py
        .allow_threads(|| {
            rank_probability_calibration(
                &actual_buckets,
                &predicted_buckets,
                bucket_count,
                validation_support,
            )
        })
        .map_err(to_py_value_error)?;
    serde_json::to_string(&calibration).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn calibrated_rank_bucket_probabilities_value(
    py: Python<'_>,
    predicted_bucket: usize,
    bucket_count: usize,
    calibration_probabilities: Vec<Vec<f64>>,
    shrinkage: f64,
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        calibrated_rank_bucket_probabilities(
            predicted_bucket,
            bucket_count,
            &calibration_probabilities,
            shrinkage,
        )
    })
    .map_err(to_py_value_error)
}

#[pyfunction]
fn sequence_validate_value(py: Python<'_>, frame_json: &str) -> PyResult<String> {
    let frame = serde_json::from_str::<SequenceFrame>(frame_json)
        .map_err(|err| PyValueError::new_err(format!("invalid sequence frame: {err}")))?;
    let payload = py
        .allow_threads(|| {
            frame.validate()?;
            let masks = frame
                .series
                .iter()
                .map(|series| {
                    let prefix = series.validate()?;
                    let mask = series.prediction_mask()?;
                    Ok(json!({
                        "series_id": series.series_id,
                        "known_prefix_rows": prefix.row_count,
                        "prediction_row_ids": mask.row_ids,
                    }))
                })
                .collect::<cartoboost_core::Result<Vec<_>>>()?;
            Ok::<Value, CartoBoostError>(json!({ "series": masks }))
        })
        .map_err(to_py_value_error)?;
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (series_json, reference_json, config_json=None, method="ekf"))]
fn sequence_state_space_value(
    py: Python<'_>,
    series_json: &str,
    reference_json: &str,
    config_json: Option<&str>,
    method: &str,
) -> PyResult<String> {
    let series = serde_json::from_str::<SequenceSeries>(series_json)
        .map_err(|err| PyValueError::new_err(format!("invalid sequence series: {err}")))?;
    let reference = serde_json::from_str::<ReferenceSignal>(reference_json)
        .map_err(|err| PyValueError::new_err(format!("invalid reference signal: {err}")))?;
    let config = match config_json {
        Some(payload) => serde_json::from_str::<SequenceStateSpaceConfig>(payload)
            .map_err(|err| PyValueError::new_err(format!("invalid state-space config: {err}")))?,
        None => SequenceStateSpaceConfig::default(),
    };
    let method = method.trim().to_ascii_lowercase();
    let payload = py
        .allow_threads(|| match method.as_str() {
            "ekf" | "forward_ekf" => {
                cartoboost_core::forecasting::forward_ekf(&series, &reference, config)
            }
            "ukf" | "ukf_reference" => {
                cartoboost_core::forecasting::ukf_reference(&series, &reference, config)
            }
            "rts" | "rts_smoother" => {
                cartoboost_core::forecasting::rts_smoother(&series, &reference, config)
            }
            "continuation" | "missing_target_continuation" => {
                let points = core_missing_target_continuation(&series, &reference, config)?;
                Ok(cartoboost_core::forecasting::SequenceKalmanResult {
                    points,
                    log_likelihood: 0.0,
                })
            }
            other => Err(CartoBoostError::InvalidInput(format!(
                "unknown sequence state-space method {other:?}"
            ))),
        })
        .map_err(to_py_value_error)?;
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (series_json, reference_json, config_json=None))]
fn sequence_reference_path_viterbi_value(
    py: Python<'_>,
    series_json: &str,
    reference_json: &str,
    config_json: Option<&str>,
) -> PyResult<String> {
    let series = serde_json::from_str::<SequenceSeries>(series_json)
        .map_err(|err| PyValueError::new_err(format!("invalid sequence series: {err}")))?;
    let reference = serde_json::from_str::<ReferenceSignal>(reference_json)
        .map_err(|err| PyValueError::new_err(format!("invalid reference signal: {err}")))?;
    let config = parse_reference_path_config(config_json)?;
    let result = py
        .allow_threads(|| core_reference_path_viterbi(&series, &reference, config))
        .map_err(to_py_value_error)?;
    serde_json::to_string(&result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (series_json, reference_json, config_json=None))]
fn sequence_reference_path_posterior_mean_value(
    py: Python<'_>,
    series_json: &str,
    reference_json: &str,
    config_json: Option<&str>,
) -> PyResult<String> {
    let series = serde_json::from_str::<SequenceSeries>(series_json)
        .map_err(|err| PyValueError::new_err(format!("invalid sequence series: {err}")))?;
    let reference = serde_json::from_str::<ReferenceSignal>(reference_json)
        .map_err(|err| PyValueError::new_err(format!("invalid reference signal: {err}")))?;
    let config = parse_reference_path_config(config_json)?;
    let result = py
        .allow_threads(|| core_reference_path_posterior_mean(&series, &reference, config))
        .map_err(to_py_value_error)?;
    serde_json::to_string(&result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (candidates_json, weights_json=None, actuals_json=None, mode="fixed"))]
fn sequence_blend_value(
    py: Python<'_>,
    candidates_json: &str,
    weights_json: Option<&str>,
    actuals_json: Option<&str>,
    mode: &str,
) -> PyResult<String> {
    let candidates = serde_json::from_str::<Vec<SequenceCandidate>>(candidates_json)
        .map_err(|err| PyValueError::new_err(format!("invalid sequence candidates: {err}")))?;
    let mode = mode.trim().to_ascii_lowercase();
    let ensemble = match mode.as_str() {
        "fixed" => {
            let payload = weights_json.ok_or_else(|| {
                PyValueError::new_err("fixed sequence blending requires weights_json")
            })?;
            let weights = serde_json::from_str::<BTreeMap<String, f64>>(payload)
                .map_err(|err| PyValueError::new_err(format!("invalid blend weights: {err}")))?;
            SequenceCandidateEnsemble::fixed(weights).map_err(to_py_value_error)?
        }
        "validation" | "validation_derived" => {
            let actuals = parse_sequence_actuals(actuals_json)?;
            py.allow_threads(|| {
                SequenceCandidateEnsemble::validation_derived(&candidates, &actuals)
            })
            .map_err(to_py_value_error)?
        }
        "constrained" | "nonnegative" | "constrained_nonnegative_linear_blend" => {
            let actuals = parse_sequence_actuals(actuals_json)?;
            py.allow_threads(|| {
                SequenceCandidateEnsemble::constrained_nonnegative_linear_blend(
                    &candidates,
                    &actuals,
                )
            })
            .map_err(to_py_value_error)?
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown sequence blend mode {other:?}"
            )));
        }
    };
    let predictions = py
        .allow_threads(|| ensemble.predict(&candidates))
        .map_err(to_py_value_error)?;
    let payload = json!({
        "weights": ensemble.weights,
        "predictions": predictions,
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn sequence_validate_oof_meta_training_value(py: Python<'_>, rows_json: &str) -> PyResult<()> {
    let rows = serde_json::from_str::<Vec<SequenceOofCandidateRow>>(rows_json)
        .map_err(|err| PyValueError::new_err(format!("invalid OOF rows: {err}")))?;
    py.allow_threads(|| cartoboost_core::forecasting::validate_oof_meta_training(&rows))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn sequence_generate_group_oof_candidate_rows_value(
    py: Python<'_>,
    fold_json: &str,
) -> PyResult<String> {
    let fold = serde_json::from_str::<SequenceOofFold>(fold_json)
        .map_err(|err| PyValueError::new_err(format!("invalid OOF fold: {err}")))?;
    let rows = py
        .allow_threads(|| cartoboost_core::forecasting::generate_group_oof_candidate_rows(&fold))
        .map_err(to_py_value_error)?;
    serde_json::to_string(&rows).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn sequence_group_error_summary_value(py: Python<'_>, rows_json: &str) -> PyResult<String> {
    let rows = serde_json::from_str::<Vec<SequenceGroupPrediction>>(rows_json)
        .map_err(|err| PyValueError::new_err(format!("invalid group prediction rows: {err}")))?;
    let result = py
        .allow_threads(|| cartoboost_core::forecasting::per_group_error_summary(&rows))
        .map_err(to_py_value_error)?;
    serde_json::to_string(&result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn parse_reference_path_config(config_json: Option<&str>) -> PyResult<ReferencePathConfig> {
    match config_json {
        Some(payload) => serde_json::from_str::<ReferencePathConfig>(payload)
            .map_err(|err| PyValueError::new_err(format!("invalid reference path config: {err}"))),
        None => Ok(ReferencePathConfig::default()),
    }
}

fn parse_sequence_actuals(
    actuals_json: Option<&str>,
) -> PyResult<Vec<SequenceCandidatePrediction>> {
    let payload = actuals_json.ok_or_else(|| {
        PyValueError::new_err("validation-derived sequence blending requires actuals_json")
    })?;
    serde_json::from_str::<Vec<SequenceCandidatePrediction>>(payload)
        .map_err(|err| PyValueError::new_err(format!("invalid sequence actuals: {err}")))
}

#[pyfunction]
fn h3_normalize_id_text(value: &str) -> PyResult<u64> {
    normalize_h3_id_text(value).map_err(to_py_value_error)
}

#[pyfunction]
fn s2_normalize_id_text(value: &str) -> PyResult<u64> {
    normalize_s2_id_text(value).map_err(to_py_value_error)
}

#[pyfunction]
fn h3_normalize_resolution_value(value: i64, field_name: &str) -> PyResult<u8> {
    normalize_h3_resolution(value, field_name).map_err(to_py_value_error)
}

#[pyfunction]
fn s2_normalize_level_value(value: i64, field_name: &str) -> PyResult<u8> {
    normalize_s2_level(value, field_name).map_err(to_py_value_error)
}

#[pyfunction]
fn geo_normalize_coordinate_value(value: f64, field_name: &str) -> PyResult<f64> {
    core_normalize_coordinate(value, field_name).map_err(to_py_value_error)
}

#[pyfunction]
fn geo_clockwise_bearing_unit_vector_value(
    origin_x: f64,
    origin_y: f64,
    destination_x: f64,
    destination_y: f64,
) -> Option<(f64, f64)> {
    cartoboost_geo_core::clockwise_bearing_unit_vector(
        [origin_x, origin_y],
        [destination_x, destination_y],
    )
    .map(|vector| (vector[0], vector[1]))
}

#[pyfunction]
fn geo_initial_bearing_unit_vector_latlng_value(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> Option<(f64, f64)> {
    cartoboost_geo_core::initial_bearing_unit_vector_latlng(
        origin_latitude,
        origin_longitude,
        destination_latitude,
        destination_longitude,
    )
    .map(|vector| (vector[0], vector[1]))
}

#[pyfunction]
fn geo_route_feature_vector_value(
    origin_x: f64,
    origin_y: f64,
    destination_x: f64,
    destination_y: f64,
) -> Option<(f64, f64, f64, f64, f64)> {
    cartoboost_geo_core::route_feature_vector([origin_x, origin_y], [destination_x, destination_y])
        .map(|vector| (vector[0], vector[1], vector[2], vector[3], vector[4]))
}

#[pyfunction]
fn geo_radial_anchor_distances_value(
    point_x: f64,
    point_y: f64,
    anchors: Vec<(f64, f64)>,
) -> Vec<f64> {
    let anchors = anchors.into_iter().map(|(x, y)| [x, y]).collect::<Vec<_>>();
    cartoboost_geo_core::radial_anchor_distances([point_x, point_y], &anchors)
}

#[pyfunction]
fn geo_rbf_anchor_features_value(
    point_x: f64,
    point_y: f64,
    anchors: Vec<(f64, f64)>,
    length_scale: f64,
) -> PyResult<Vec<f64>> {
    let anchors = anchors.into_iter().map(|(x, y)| [x, y]).collect::<Vec<_>>();
    cartoboost_geo_core::rbf_anchor_features([point_x, point_y], &anchors, length_scale)
        .map_err(to_py_geo_core_error)
}

#[pyfunction]
fn geo_local_frame_features_value(
    point_x: f64,
    point_y: f64,
    origin_x: f64,
    origin_y: f64,
    axis_east: f64,
    axis_north: f64,
) -> Option<(f64, f64)> {
    cartoboost_geo_core::local_frame_features(
        [point_x, point_y],
        [origin_x, origin_y],
        [axis_east, axis_north],
    )
    .map(|vector| (vector[0], vector[1]))
}

#[pyfunction]
fn h3_validate_parent_resolutions_value(
    py: Python<'_>,
    resolution: u8,
    parent_resolutions: Vec<u8>,
) -> PyResult<()> {
    py.allow_threads(|| validate_parent_levels(resolution, &parent_resolutions, GeoGridKind::H3))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn s2_validate_parent_levels_value(
    py: Python<'_>,
    level: u8,
    parent_levels: Vec<u8>,
) -> PyResult<()> {
    py.allow_threads(|| validate_parent_levels(level, &parent_levels, GeoGridKind::S2))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn h3_scaffold_parent_id_value(cell: u64, resolution: u8, parent_resolution: u8) -> PyResult<u64> {
    scaffold_h3_parent_id(cell, resolution, parent_resolution).map_err(to_py_value_error)
}

#[pyfunction]
fn h3_expand_sparse_set_value(
    py: Python<'_>,
    values: Vec<u64>,
    resolution: u8,
    parent_resolutions: Vec<u8>,
) -> PyResult<Vec<u64>> {
    py.allow_threads(|| core_expand_h3_sparse_set(&values, resolution, &parent_resolutions))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn geo_assemble_sparse_row_value(child: u64, parents: Vec<u64>) -> Vec<u64> {
    assemble_sparse_row(child, &parents)
}

#[pyfunction]
fn geo_assemble_sparse_column_value(
    py: Python<'_>,
    children: Vec<u64>,
    parent_columns: Vec<Vec<u64>>,
) -> PyResult<Vec<Vec<u64>>> {
    py.allow_threads(|| assemble_sparse_column(&children, &parent_columns))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn geo_assemble_route_sparse_rows_value(
    py: Python<'_>,
    route_cells: Vec<Vec<u64>>,
) -> PyResult<Vec<Vec<u64>>> {
    py.allow_threads(|| assemble_route_sparse_rows(&route_cells))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn geo_validate_equal_row_count_value(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    validate_equal_row_count(name, actual, expected).map_err(to_py_value_error)
}

fn artifact_fallback_name(fallback: &ArtifactFallbackKind) -> &'static str {
    match fallback {
        ArtifactFallbackKind::ZeroVector => "zero_vector",
        ArtifactFallbackKind::GlobalMeanVector => "global_mean_vector",
        ArtifactFallbackKind::ParentCell { .. } => "parent_cell",
    }
}

fn standalone_booster_config(
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
) -> StandaloneBoosterConfig {
    StandaloneBoosterConfig {
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf,
        min_gain,
    }
}

fn parse_leaf_predictor(name: &str) -> PyResult<LeafPredictorKind> {
    match name {
        "constant" => Ok(LeafPredictorKind::Constant),
        "linear" => Ok(LeafPredictorKind::Linear),
        _ => Err(PyValueError::new_err(format!(
            "unknown leaf_predictor {name:?}; expected 'constant' or 'linear'"
        ))),
    }
}

fn parse_fuzzy_kernel(name: &str) -> PyResult<FuzzyKernel> {
    match name {
        "linear" | "triangular" => Ok(FuzzyKernel::Linear),
        "gaussian" => Ok(FuzzyKernel::Gaussian),
        "exponential" => Ok(FuzzyKernel::Exponential),
        "bisquare" => Ok(FuzzyKernel::Bisquare),
        "epanechnikov" => Ok(FuzzyKernel::Epanechnikov),
        "tricube" => Ok(FuzzyKernel::Tricube),
        _ => Err(PyValueError::new_err(format!(
            "unknown fuzzy_kernel {name:?}; expected 'linear', 'gaussian', 'exponential', 'bisquare', 'epanechnikov', or 'tricube'"
        ))),
    }
}

fn fuzzy_kernel_name(kernel: FuzzyKernel) -> &'static str {
    match kernel {
        FuzzyKernel::Linear => "linear",
        FuzzyKernel::Gaussian => "gaussian",
        FuzzyKernel::Exponential => "exponential",
        FuzzyKernel::Bisquare => "bisquare",
        FuzzyKernel::Epanechnikov => "epanechnikov",
        FuzzyKernel::Tricube => "tricube",
    }
}

fn parse_loss(
    name: &str,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
) -> PyResult<LossConfig> {
    match name {
        "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "mae" | "absolute_error" | "least_absolute_deviation" | "lad" => Ok(LossConfig::L1),
        "huber" => {
            if !huber_delta.is_finite() || huber_delta <= 0.0 {
                return Err(PyValueError::new_err(
                    "huber_delta must be positive and finite",
                ));
            }
            Ok(LossConfig::Huber(HuberLossConfig { delta: huber_delta }))
        }
        "log_l2" | "log" | "log_squared_error" => {
            if !log_offset.is_finite() || log_offset <= 0.0 {
                return Err(PyValueError::new_err(
                    "log_offset must be positive and finite",
                ));
            }
            if (log_offset - 1.0).abs() > 1e-12 {
                return Err(PyValueError::new_err(
                    "log_l2 currently supports log_offset=1.0",
                ));
            }
            Ok(LossConfig::LogL2(LogL2LossConfig { offset: log_offset }))
        }
        "quantile" | "pinball" => {
            if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
                return Err(PyValueError::new_err(
                    "quantile_alpha must be finite and in (0, 1)",
                ));
            }
            Ok(LossConfig::Quantile(QuantileLossConfig {
                alpha: quantile_alpha,
            }))
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown loss {name:?}; expected 'l2', 'l1', 'huber', 'log_l2', or 'quantile'"
        ))),
    }
}

fn loss_name(loss: &LossConfig) -> &'static str {
    match loss {
        LossConfig::L2 => "l2",
        LossConfig::L1 => "l1",
        LossConfig::Huber(_) => "huber",
        LossConfig::LogL2(_) => "log_l2",
        LossConfig::Quantile(_) => "quantile",
    }
}

fn quantile_alpha(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::L2 | LossConfig::L1 | LossConfig::Huber(_) | LossConfig::LogL2(_) => 0.5,
        LossConfig::Quantile(config) => config.alpha,
    }
}

fn huber_delta(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::Huber(config) => config.delta,
        _ => 1.0,
    }
}

fn log_offset(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::LogL2(config) => config.offset,
        _ => 1.0,
    }
}

fn splitter_names(splitters: &[SplitterKind]) -> Vec<String> {
    splitters
        .iter()
        .map(|splitter| match splitter {
            SplitterKind::Auto => "auto".to_string(),
            SplitterKind::Axis => "axis".to_string(),
            SplitterKind::AxisHistogram { bins } => format!("axis_histogram:{bins}"),
            SplitterKind::Diagonal2D => "diagonal_2d".to_string(),
            SplitterKind::Gaussian2D => "gaussian_2d".to_string(),
            SplitterKind::Periodic { period } if (*period - 24.0).abs() < 1e-12 => {
                "periodic_time".to_string()
            }
            SplitterKind::Periodic { period } => format!("periodic:{period}"),
            SplitterKind::SparseSet => "sparse_set".to_string(),
        })
        .collect()
}

fn leaf_predictor_name(leaf_predictor: &LeafPredictorKind) -> &'static str {
    match leaf_predictor {
        LeafPredictorKind::Constant => "constant",
        LeafPredictorKind::Linear => "linear",
    }
}

fn validate_n_threads(n_threads: Option<usize>) -> PyResult<()> {
    if n_threads == Some(0) {
        return Err(PyValueError::new_err("n_threads must be positive"));
    }
    Ok(())
}

fn run_with_optional_threads<T, F>(n_threads: Option<usize>, f: F) -> Result<T, CartoBoostError>
where
    T: Send,
    F: FnOnce() -> Result<T, CartoBoostError> + Send,
{
    if let Some(n_threads) = n_threads {
        static THREAD_POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
        let pools = THREAD_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let pool = {
            let mut pools = pools
                .lock()
                .map_err(|_| CartoBoostError::InvalidInput("thread-pool cache poisoned".into()))?;
            if let Some(pool) = pools.get(&n_threads) {
                Arc::clone(pool)
            } else {
                let pool = Arc::new(
                    ThreadPoolBuilder::new()
                        .num_threads(n_threads)
                        .build()
                        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
                );
                pools.insert(n_threads, Arc::clone(&pool));
                pool
            }
        };
        pool.install(f)
    } else {
        f()
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_params(
    n_estimators: usize,
    learning_rate: f64,
    _max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy_bandwidth: f64,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
) -> PyResult<()> {
    if n_estimators == 0 {
        return Err(PyValueError::new_err("n_estimators must be positive"));
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(PyValueError::new_err(
            "learning_rate must be positive and finite",
        ));
    }
    if min_samples_leaf == 0 {
        return Err(PyValueError::new_err("min_samples_leaf must be positive"));
    }
    if !min_gain.is_finite() || min_gain < 0.0 {
        return Err(PyValueError::new_err(
            "min_gain must be finite and non-negative",
        ));
    }
    if !l2_regularization.is_finite() || l2_regularization < 0.0 {
        return Err(PyValueError::new_err(
            "l2_regularization must be finite and non-negative",
        ));
    }
    if !constant_l2_regularization.is_finite() || constant_l2_regularization < 0.0 {
        return Err(PyValueError::new_err(
            "constant_l2_regularization must be finite and non-negative",
        ));
    }
    if !fuzzy_bandwidth.is_finite() || fuzzy_bandwidth < 0.0 {
        return Err(PyValueError::new_err(
            "fuzzy_bandwidth must be finite and non-negative",
        ));
    }
    if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
        return Err(PyValueError::new_err(
            "quantile_alpha must be finite and in (0, 1)",
        ));
    }
    if !huber_delta.is_finite() || huber_delta <= 0.0 {
        return Err(PyValueError::new_err(
            "huber_delta must be positive and finite",
        ));
    }
    if !log_offset.is_finite() || log_offset <= 0.0 {
        return Err(PyValueError::new_err(
            "log_offset must be positive and finite",
        ));
    }
    Ok(())
}

fn dataset_from_rows(rows: Vec<Vec<f64>>) -> PyResult<Dataset> {
    if rows.is_empty() {
        return Err(PyValueError::new_err("X must not be empty"));
    }
    if rows[0].is_empty() {
        return Err(PyValueError::new_err(
            "X rows must contain at least one feature",
        ));
    }
    if rows
        .iter()
        .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err("X must contain only finite values"));
    }
    Dataset::from_rows(rows).map_err(to_py_value_error)
}

fn dataset_from_parts(
    rows: Vec<Vec<f64>>,
    sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    feature_schema_json: Option<String>,
) -> PyResult<Dataset> {
    let dataset = dataset_from_rows(rows)?;
    let sparse_sets = sparse_sets
        .unwrap_or_default()
        .into_iter()
        .map(SparseSetColumn::new)
        .collect::<Vec<_>>();
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let dataset = dataset
        .with_sparse_sets(sparse_sets)
        .map_err(to_py_value_error)?;
    match schema {
        Some(schema) => dataset.with_schema(schema).map_err(to_py_value_error),
        None => Ok(dataset),
    }
}

fn dataset_from_arrays(
    x: PyReadonlyArray2<'_, f64>,
    sparse_offsets: Option<Vec<Vec<usize>>>,
    sparse_ids: Option<Vec<Vec<u64>>>,
    feature_schema_json: Option<String>,
) -> PyResult<Dataset> {
    let shape = x.shape();
    let rows = shape[0];
    let cols = shape[1];
    let values = x.as_slice()?.to_vec();
    let dataset = Dataset::from_flat(rows, cols, values).map_err(to_py_value_error)?;
    let sparse_sets = encoded_sparse_sets(rows, sparse_offsets, sparse_ids)?
        .into_iter()
        .map(SparseSetColumn::new)
        .collect::<Vec<_>>();
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let dataset = dataset
        .with_sparse_sets(sparse_sets)
        .map_err(to_py_value_error)?;
    match schema {
        Some(schema) => dataset.with_schema(schema).map_err(to_py_value_error),
        None => Ok(dataset),
    }
}

fn encoded_sparse_sets(
    rows: usize,
    sparse_offsets: Option<Vec<Vec<usize>>>,
    sparse_ids: Option<Vec<Vec<u64>>>,
) -> PyResult<Vec<Vec<Vec<u64>>>> {
    let offsets = sparse_offsets.unwrap_or_default();
    let ids = sparse_ids.unwrap_or_default();
    if offsets.len() != ids.len() {
        return Err(PyValueError::new_err(
            "sparse_offsets and sparse_ids must contain the same number of columns",
        ));
    }
    let mut columns = Vec::with_capacity(offsets.len());
    for (column_index, (column_offsets, column_ids)) in offsets.into_iter().zip(ids).enumerate() {
        if column_offsets.len() != rows + 1 {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must have rows + 1 entries"
            )));
        }
        if column_offsets.first().copied() != Some(0) {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must start at 0"
            )));
        }
        if column_offsets.last().copied() != Some(column_ids.len()) {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} final offset must match sparse_ids length"
            )));
        }
        if column_offsets
            .windows(2)
            .any(|window| window[0] > window[1])
        {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must be non-decreasing"
            )));
        }
        let mut column = Vec::with_capacity(rows);
        for window in column_offsets.windows(2) {
            column.push(column_ids[window[0]..window[1]].to_vec());
        }
        columns.push(column);
    }
    Ok(columns)
}

#[derive(Clone)]
struct OverlayPoint {
    id: String,
    coordinates: (f64, f64),
    properties: serde_json::Map<String, Value>,
}

struct OverlayZone {
    id: String,
    priority: f64,
    bbox: (f64, f64, f64, f64),
    ring: Vec<(f64, f64)>,
}

#[pyfunction(signature = (points, zones, weights, origin=None, zone_priority_multiplier=true, kernel="none", bandwidth_meters=None, distance_alpha=0.0, precision=6, include_debug=false))]
#[allow(clippy::too_many_arguments)]
fn weighted_overlay(
    py: Python<'_>,
    points: Bound<'_, PyAny>,
    zones: Bound<'_, PyAny>,
    weights: Bound<'_, PyAny>,
    origin: Option<(f64, f64)>,
    zone_priority_multiplier: bool,
    kernel: &str,
    bandwidth_meters: Option<f64>,
    distance_alpha: f64,
    precision: usize,
    include_debug: bool,
) -> PyResult<Py<PyAny>> {
    let json_module = PyModule::import(py, "json")?;
    let points_payload = json_module
        .call_method1("dumps", (points,))?
        .extract::<String>()?;
    let zones_payload = json_module
        .call_method1("dumps", (zones,))?
        .extract::<String>()?;
    let weights_payload = json_module
        .call_method1("dumps", (weights,))?
        .extract::<String>()?;

    let kernel = kernel.to_string();
    let payload = py
        .allow_threads(|| {
            let points_value = serde_json::from_str::<Value>(&points_payload)
                .map_err(|err| format!("invalid points payload: {err}"))?;
            let zones_value = serde_json::from_str::<Value>(&zones_payload)
                .map_err(|err| format!("invalid zones payload: {err}"))?;
            let weights_value = serde_json::from_str::<Value>(&weights_payload)
                .map_err(|err| format!("invalid weights payload: {err}"))?;

            let result = weighted_overlay_impl(
                &points_value,
                &zones_value,
                &weights_value,
                origin,
                zone_priority_multiplier,
                &kernel,
                bandwidth_meters,
                distance_alpha,
                precision,
                include_debug,
            )?;

            serde_json::to_string(&result)
                .map_err(|err| format!("failed to serialize overlay result: {err}"))
        })
        .map_err(PyValueError::new_err)?;
    Ok(json_module.call_method1("loads", (payload,))?.unbind())
}

#[allow(clippy::too_many_arguments)]
fn weighted_overlay_impl(
    points: &Value,
    zones: &Value,
    weights: &Value,
    origin: Option<(f64, f64)>,
    zone_priority_multiplier: bool,
    kernel: &str,
    bandwidth_meters: Option<f64>,
    distance_alpha: f64,
    precision: usize,
    include_debug: bool,
) -> Result<Value, String> {
    let overlay_points = parse_overlay_points(points)?;
    let overlay_zones = parse_overlay_zones(zones)?;
    let weight_map = weights
        .as_object()
        .ok_or_else(|| "weights must be a JSON object".to_string())?;

    let mut features = Vec::with_capacity(overlay_points.len());
    for point in &overlay_points {
        let zone = locate_zone(&overlay_zones, point.coordinates)?;
        let linear_score = weight_map.iter().try_fold(0.0, |score, (name, weight)| {
            let weight_value = weight
                .as_f64()
                .ok_or_else(|| format!("weight {name:?} must be numeric"))?;
            let property_value = point
                .properties
                .get(name)
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            Ok::<f64, String>(score + weight_value * property_value)
        })?;

        let priority = if zone_priority_multiplier {
            zone.priority
        } else {
            1.0
        };

        let spatial_term = if let Some(origin) = (kernel != "none" && distance_alpha != 0.0)
            .then_some(origin)
            .flatten()
        {
            let bandwidth =
                resolve_bandwidth(bandwidth_meters, point.coordinates, &overlay_points)?;
            let distance = haversine_meters(origin, point.coordinates);
            distance_alpha * kernel_weight(distance, bandwidth, kernel)?
        } else {
            0.0
        };

        let mut feature = json!({
            "id": point.id,
            "zone_id": zone.id,
            "boost_score": round_half_even(linear_score * priority * (1.0 + spatial_term), precision),
        });
        if include_debug {
            feature["debug"] = json!({
                "linear": round_half_even(linear_score, precision),
                "priority": round_half_even(priority, precision),
                "spatial_term": round_half_even(spatial_term, precision),
            });
        }
        features.push(feature);
    }

    features.sort_by(|left, right| {
        let right_score = right["boost_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        let left_score = left["boost_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left["id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(right["id"].as_str().unwrap_or(""))
            })
    });
    for (rank, feature) in features.iter_mut().enumerate() {
        feature["rank"] = json!(rank + 1);
    }

    let points_name = points
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| points.get("type").and_then(Value::as_str))
        .unwrap_or("points");
    let zones_name = zones
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| zones.get("type").and_then(Value::as_str))
        .unwrap_or("zones");

    let mut config = json!({
        "algorithm": "weighted_overlay",
        "weights": weights.clone(),
        "zone_priority_multiplier": zone_priority_multiplier,
        "rounding": {
            "places": precision,
            "mode": "half_even",
        },
    });
    if origin.is_some() || kernel != "none" || distance_alpha != 0.0 {
        config["distance_term"] = json!({
            "enabled": origin.is_some() && kernel != "none" && distance_alpha != 0.0,
            "source": if origin.is_some() { Value::String("origin".to_string()) } else { Value::Null },
            "kernel": kernel,
            "bandwidth_meters": bandwidth_meters,
            "distance_alpha": distance_alpha,
        });
    }

    Ok(json!({
        "schema_version": 1,
        "scenario": format!("{points_name}_x_{zones_name}"),
        "config": config,
        "features": features,
    }))
}

fn parse_overlay_points(points: &Value) -> Result<Vec<OverlayPoint>, String> {
    let features = points
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| "points must contain a features array".to_string())?;
    features
        .iter()
        .map(|feature| {
            let id = feature
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| "point features must provide an id".to_string())?;
            let cartometry = feature
                .get("cartometry")
                .ok_or_else(|| format!("point feature {id:?} is missing cartometry"))?;
            let cartometry_type = cartometry
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("point feature {id:?} cartometry is missing type"))?;
            if cartometry_type != "Point" {
                return Err(format!("point feature {id:?} must use Point cartometry"));
            }
            let coordinates = cartometry
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("point feature {id:?} is missing coordinates"))?;
            if coordinates.len() < 2 {
                return Err(format!(
                    "point feature {id:?} must provide [x, y] coordinates"
                ));
            }
            let x = coordinates[0]
                .as_f64()
                .ok_or_else(|| format!("point feature {id:?} x coordinate must be numeric"))?;
            let y = coordinates[1]
                .as_f64()
                .ok_or_else(|| format!("point feature {id:?} y coordinate must be numeric"))?;
            let properties = feature
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            Ok(OverlayPoint {
                id: id.to_string(),
                coordinates: (x, y),
                properties,
            })
        })
        .collect()
}

fn parse_overlay_zones(zones: &Value) -> Result<Vec<OverlayZone>, String> {
    let features = zones
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| "zones must contain a features array".to_string())?;
    features
        .iter()
        .map(|feature| {
            let id = feature
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| "zone features must provide an id".to_string())?;
            let cartometry = feature
                .get("cartometry")
                .ok_or_else(|| format!("zone feature {id:?} is missing cartometry"))?;
            let cartometry_type = cartometry
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("zone feature {id:?} cartometry is missing type"))?;
            if cartometry_type != "Polygon" {
                return Err(format!("zone feature {id:?} must use Polygon cartometry"));
            }
            let rings = cartometry
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("zone feature {id:?} is missing polygon coordinates"))?;
            let outer_ring = rings
                .first()
                .and_then(Value::as_array)
                .ok_or_else(|| format!("zone feature {id:?} is missing an outer ring"))?;
            let ring = outer_ring
                .iter()
                .map(|coordinate| {
                    let pair = coordinate.as_array().ok_or_else(|| {
                        format!("zone feature {id:?} ring coordinates must be arrays")
                    })?;
                    if pair.len() < 2 {
                        return Err(format!(
                            "zone feature {id:?} ring coordinates must have two values"
                        ));
                    }
                    let x = pair[0].as_f64().ok_or_else(|| {
                        format!("zone feature {id:?} x coordinate must be numeric")
                    })?;
                    let y = pair[1].as_f64().ok_or_else(|| {
                        format!("zone feature {id:?} y coordinate must be numeric")
                    })?;
                    Ok((x, y))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let bbox = bounding_box(&ring)?;
            let priority = feature
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get("priority"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            Ok(OverlayZone {
                id: id.to_string(),
                priority,
                bbox,
                ring,
            })
        })
        .collect()
}

fn locate_zone(zones: &[OverlayZone], point: (f64, f64)) -> Result<&OverlayZone, String> {
    let (x, y) = point;
    zones
        .iter()
        .find(|zone| {
            let (min_x, min_y, max_x, max_y) = zone.bbox;
            min_x <= x
                && x <= max_x
                && min_y <= y
                && y <= max_y
                && point_in_polygon(point, &zone.ring)
        })
        .ok_or_else(|| format!("point ({x}, {y}) does not belong to any zone"))
}

fn bounding_box(ring: &[(f64, f64)]) -> Result<(f64, f64, f64, f64), String> {
    if ring.is_empty() {
        return Err("polygon ring must not be empty".to_string());
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in ring {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    Ok((min_x, min_y, max_x, max_y))
}

fn point_in_polygon(point: (f64, f64), ring: &[(f64, f64)]) -> bool {
    if ring.len() < 2 {
        return false;
    }
    let (x, y) = point;
    let mut inside = false;
    for index in 0..(ring.len() - 1) {
        let start = ring[index];
        let end = ring[index + 1];
        if point_on_segment(point, start, end) {
            return true;
        }
        let intersects = (start.1 > y) != (end.1 > y);
        if intersects {
            let slope_x =
                (end.0 - start.0) * (y - start.1) / ((end.1 - start.1).abs().max(1e-12)) + start.0;
            if x <= slope_x {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_on_segment(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> bool {
    let cross = (point.0 - start.0) * (end.1 - start.1) - (point.1 - start.1) * (end.0 - start.0);
    if cross.abs() > 1e-9 {
        return false;
    }
    let min_x = start.0.min(end.0) - 1e-9;
    let max_x = start.0.max(end.0) + 1e-9;
    let min_y = start.1.min(end.1) - 1e-9;
    let max_y = start.1.max(end.1) + 1e-9;
    min_x <= point.0 && point.0 <= max_x && min_y <= point.1 && point.1 <= max_y
}

fn resolve_bandwidth(
    bandwidth_meters: Option<f64>,
    point: (f64, f64),
    points: &[OverlayPoint],
) -> Result<f64, String> {
    if let Some(bandwidth) = bandwidth_meters {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err("bandwidth_meters must be positive and finite".to_string());
        }
        return Ok(bandwidth);
    }
    let mut distances = points
        .iter()
        .filter(|candidate| candidate.coordinates != point)
        .map(|candidate| haversine_meters(point, candidate.coordinates))
        .collect::<Vec<_>>();
    if distances.is_empty() {
        return Ok(1.0);
    }
    distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Ok(distances[distances.len().min(3) - 1].max(1.0))
}

fn kernel_weight(distance_meters: f64, bandwidth_meters: f64, kernel: &str) -> Result<f64, String> {
    if !bandwidth_meters.is_finite() || bandwidth_meters <= 0.0 {
        return Err("bandwidth_meters must be positive and finite".to_string());
    }
    let ratio = distance_meters / bandwidth_meters;
    match kernel {
        "none" => Ok(0.0),
        "gaussian" => Ok((-0.5 * ratio * ratio).exp()),
        "bisquare" => {
            if ratio >= 1.0 {
                Ok(0.0)
            } else {
                Ok((1.0 - ratio * ratio).powi(2))
            }
        }
        "exponential" => Ok((-ratio).exp()),
        _ => Err(format!("unknown kernel {kernel:?}")),
    }
}

fn haversine_meters(origin: (f64, f64), destination: (f64, f64)) -> f64 {
    cartoboost_geo_core::haversine_distance_meters(origin.1, origin.0, destination.1, destination.0)
}

fn round_half_even(value: f64, precision: usize) -> f64 {
    let factor = 10_f64.powi(precision as i32);
    let scaled = value * factor;
    let sign = if scaled.is_sign_negative() { -1.0 } else { 1.0 };
    let scaled_abs = scaled.abs();
    let lower = scaled_abs.floor();
    let fraction = scaled_abs - lower;
    let rounded = if fraction > 0.5 + 1e-12 {
        lower + 1.0
    } else if fraction < 0.5 - 1e-12 || (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    };
    sign * rounded / factor
}

fn to_py_value_error(err: CartoBoostError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_geo_core_error(err: cartoboost_geo_core::GeoCoreError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_error(err: CartoBoostError) -> PyErr {
    match err {
        CartoBoostError::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn to_py_spatial_error(err: SpatialEconError) -> PyErr {
    match err {
        SpatialEconError::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn to_py_neural_error(err: cartoboost_neural::NeuralError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_json_error(err: serde_json::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_geo_st_error(err: cartoboost_geo_st::GeoStError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_geostats_error(err: cartoboost_geostats::GeostatsError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn coords_from_array(coords: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let shape = coords.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(
            "coords must be a two-column array with shape (n, 2)",
        ));
    }
    let values = coords.as_slice()?;
    Ok(values
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect())
}

#[pyfunction]
#[pyo3(signature = (coords, values, bin_count=12, max_distance=None, anisotropy_angle_degrees=0.0, anisotropy_scaling=1.0))]
fn geostats_empirical_semivariogram_value(
    coords: PyReadonlyArray2<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let coords = coords_from_array(coords)?;
    let values = values.as_slice()?.to_vec();
    geostats_empirical_semivariogram(
        &coords,
        &values,
        bin_count,
        max_distance,
        CoreGeostatsAnisotropy {
            angle_degrees: anisotropy_angle_degrees,
            scaling: anisotropy_scaling,
        },
    )
    .map_err(to_py_geostats_error)
    .and_then(|bins| serde_json::to_string(&bins).map_err(to_py_json_error))
}

#[pyfunction]
fn geostats_fit_variogram_wls_value(
    bins: Vec<BTreeMap<String, f64>>,
    kernels: Vec<String>,
    range_candidates: Vec<f64>,
    sill_candidates: Vec<f64>,
    nugget_candidates: Vec<f64>,
) -> PyResult<String> {
    let parsed_bins = bins
        .into_iter()
        .map(|row| {
            let get = |key: &str| {
                row.get(key)
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("variogram bin missing {key:?}")))
            };
            Ok(cartoboost_geostats::EmpiricalVariogramBin {
                lag_start: get("lag_start")?,
                lag_end: get("lag_end")?,
                lag_center: get("lag_center")?,
                semivariance: get("semivariance")?,
                pair_count: get("pair_count")? as usize,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let parsed_kernels = kernels
        .iter()
        .map(|kernel| CoreCovarianceKernel::parse(kernel).map_err(to_py_geostats_error))
        .collect::<PyResult<Vec<_>>>()?;
    let fit = geostats_fit_variogram_wls(
        &parsed_bins,
        &parsed_kernels,
        &range_candidates,
        &sill_candidates,
        &nugget_candidates,
    )
    .map_err(to_py_geostats_error)?;
    serde_json::to_string(&json!({
        "kernel": fit.kernel.as_str(),
        "range": fit.range,
        "sill": fit.sill,
        "nugget": fit.nugget,
        "weighted_sse": fit.weighted_sse,
    }))
    .map_err(to_py_json_error)
}

#[pyfunction]
#[pyo3(signature = (rows_json, response_type, monotone=None, backend=None))]
fn deep_response_curve_fit_value(
    rows_json: &str,
    response_type: &str,
    monotone: Option<&str>,
    backend: Option<&str>,
) -> PyResult<String> {
    let rows: Vec<DeepResponseRow> = serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    let artifact = core_deep_response_curve_fit(&rows, response_type, monotone, backend)
        .map_err(to_py_neural_error)?;
    serde_json::to_string(&artifact).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_response_curve_predict_value(artifact_json: &str, rows_json: &str) -> PyResult<String> {
    let artifact: DeepResponseArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepResponseRow> = serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    let predictions =
        core_deep_response_curve_predict(&artifact, &rows).map_err(to_py_neural_error)?;
    serde_json::to_string(&predictions).map_err(to_py_json_error)
}

#[pyfunction]
#[pyo3(signature = (features_json, labels, backend=None))]
fn deep_event_outcome_fit_value(
    features_json: &str,
    labels: Vec<f64>,
    backend: Option<&str>,
) -> PyResult<String> {
    let features: Vec<Vec<f64>> = serde_json::from_str(features_json).map_err(to_py_json_error)?;
    let artifact =
        core_deep_event_outcome_fit(&features, &labels, backend).map_err(to_py_neural_error)?;
    serde_json::to_string(&artifact).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_event_outcome_predict_value(artifact_json: &str, features_json: &str) -> PyResult<String> {
    let artifact: DeepEventArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let features: Vec<Vec<f64>> = serde_json::from_str(features_json).map_err(to_py_json_error)?;
    let predictions =
        core_deep_event_outcome_predict(&artifact, &features).map_err(to_py_neural_error)?;
    serde_json::to_string(&predictions).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_directional_pair_predict_value(rows_json: &str) -> PyResult<Vec<f64>> {
    let rows: Vec<DeepDirectionalPairRow> =
        serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    core_deep_directional_pair_predictions(&rows).map_err(to_py_neural_error)
}

#[pyfunction]
#[pyo3(signature = (rows_json, options_json=None))]
fn deep_directional_pair_fit_value(
    rows_json: &str,
    options_json: Option<&str>,
) -> PyResult<String> {
    let rows: Vec<DeepDirectionalPairRow> =
        serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    let artifact = if let Some(options_json) = options_json {
        let options: DirectionalPairFitOptions =
            serde_json::from_str(options_json).map_err(to_py_json_error)?;
        cartoboost_neural::directional_pair_fit_with_options(&rows, &options)
            .map_err(to_py_neural_error)?
    } else {
        core_deep_directional_pair_fit(&rows).map_err(to_py_neural_error)?
    };
    serde_json::to_string(&artifact).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_directional_pair_predict_artifact_value(
    artifact_json: &str,
    rows_json: &str,
) -> PyResult<Vec<f64>> {
    let artifact: DeepDirectionalPairArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepDirectionalPairRow> =
        serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    core_deep_directional_pair_predict(&artifact, &rows).map_err(to_py_neural_error)
}

#[pyfunction]
#[pyo3(signature = (rows_json, backend=None))]
fn deep_service_residual_fit_value(rows_json: &str, backend: Option<&str>) -> PyResult<String> {
    let rows: Vec<DeepServiceResidualRow> =
        serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    let artifact = core_deep_service_residual_fit(&rows, backend).map_err(to_py_neural_error)?;
    serde_json::to_string(&artifact).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_available_backends_value() -> Vec<String> {
    neural_available_backends()
}

#[pyfunction]
#[pyo3(signature = (backend=None, len=4096))]
fn deep_backend_dispatch_report_value(backend: Option<&str>, len: usize) -> PyResult<String> {
    let report = neural_backend_dispatch_report(backend, len).map_err(to_py_neural_error)?;
    serde_json::to_string(&report).map_err(to_py_json_error)
}

#[pyfunction]
fn graph_st_available_backends_value() -> Vec<String> {
    graph_st_available_compute_backends()
}

#[pyfunction]
fn deep_service_residual_predict_value(artifact_json: &str, rows_json: &str) -> PyResult<String> {
    let artifact: DeepServiceResidualArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let rows: Vec<DeepServiceResidualRow> =
        serde_json::from_str(rows_json).map_err(to_py_json_error)?;
    let predictions =
        core_deep_service_residual_predict(&artifact, &rows).map_err(to_py_neural_error)?;
    serde_json::to_string(&predictions).map_err(to_py_json_error)
}

#[pyfunction]
#[pyo3(signature = (candidates_json, objective, constraints_json, fallback, risk_aversion=None))]
fn deep_constrained_decision_select_value(
    candidates_json: &str,
    objective: &str,
    constraints_json: &str,
    fallback: &str,
    risk_aversion: Option<f64>,
) -> PyResult<String> {
    let candidates: Vec<BTreeMap<String, Value>> =
        serde_json::from_str(candidates_json).map_err(to_py_json_error)?;
    let constraints: BTreeMap<String, f64> =
        serde_json::from_str(constraints_json).map_err(to_py_json_error)?;
    let choices = core_deep_constrained_decision_select(
        &candidates,
        objective,
        &constraints,
        fallback,
        risk_aversion.unwrap_or(0.0),
    )
    .map_err(to_py_neural_error)?;
    serde_json::to_string(&choices).map_err(to_py_json_error)
}

#[pyfunction]
#[pyo3(signature = (candidates_json, temperature=1.0, monotone_candidate_value=None))]
fn deep_choice_set_transformer_report_value(
    candidates_json: &str,
    temperature: f64,
    monotone_candidate_value: Option<&str>,
) -> PyResult<String> {
    let candidates: Vec<BTreeMap<String, Value>> =
        serde_json::from_str(candidates_json).map_err(to_py_json_error)?;
    core_choice_set_transformer_report_json(&candidates, temperature, monotone_candidate_value)
        .map_err(to_py_neural_error)
}

#[pyfunction]
fn deep_temporal_entity_fit_value(
    y_json: &str,
    lookback: usize,
    horizon: usize,
) -> PyResult<String> {
    let y: Vec<Vec<f64>> = serde_json::from_str(y_json).map_err(to_py_json_error)?;
    let artifact =
        core_deep_temporal_entity_fit(&y, lookback, horizon).map_err(to_py_neural_error)?;
    serde_json::to_string(&artifact).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_temporal_entity_predict_value(artifact_json: &str, horizon: usize) -> PyResult<String> {
    let artifact: DeepTemporalEntityArtifact =
        serde_json::from_str(artifact_json).map_err(to_py_json_error)?;
    let prediction =
        core_deep_temporal_entity_predict(&artifact, horizon).map_err(to_py_neural_error)?;
    serde_json::to_string(&prediction).map_err(to_py_json_error)
}

#[pyfunction]
fn deep_graph_neural_operator_predict_value(
    field_values_json: &str,
    coordinates_json: &str,
    edges: Vec<(usize, usize, f64)>,
    exogenous_fields_json: &str,
    smoothing: f64,
    coordinate_scale: f64,
) -> PyResult<String> {
    let field_values: Vec<Vec<f64>> =
        serde_json::from_str(field_values_json).map_err(to_py_json_error)?;
    let coordinates: Vec<Vec<f64>> =
        serde_json::from_str(coordinates_json).map_err(to_py_json_error)?;
    let exogenous_fields: Vec<Vec<f64>> =
        serde_json::from_str(exogenous_fields_json).map_err(to_py_json_error)?;
    let edges = edges
        .into_iter()
        .map(|(source, target, weight)| CoreSpatialOperatorEdge {
            source,
            target,
            weight,
        })
        .collect::<Vec<_>>();
    core_graph_neural_operator_predict_json(
        &field_values,
        &coordinates,
        &edges,
        &exogenous_fields,
        smoothing,
        coordinate_scale,
    )
    .map_err(to_py_neural_error)
}

#[pyfunction]
fn deep_neural_operator_synthetic_benchmark_value() -> PyResult<String> {
    core_neural_operator_synthetic_benchmark_json().map_err(to_py_neural_error)
}

#[allow(clippy::too_many_arguments)]
fn neural_panel_config_from_parts(
    n_lags: usize,
    n_forecasts: usize,
    quantiles: Option<Vec<f64>>,
    trend: &str,
    n_changepoints: usize,
    changepoints_range: f64,
    daily_fourier_order: usize,
    weekly_fourier_order: usize,
    yearly_fourier_order: usize,
    custom_seasonalities: Option<Vec<CustomSeasonalitySpec>>,
    seasonality_mode: &str,
    events: Option<BTreeMap<String, Vec<i32>>>,
    event_mode: &str,
    future_regressors: Option<BTreeMap<String, String>>,
    lagged_regressors: Option<BTreeMap<String, usize>>,
    ar_layers: Option<Vec<usize>>,
    lagged_reg_layers: Option<Vec<usize>>,
    trend_mode: &str,
    seasonality_global_local: &str,
    event_global_local: &str,
    regressor_global_local: &str,
    local_l2: f64,
    seed: u64,
    loss: &str,
    epochs: usize,
    learning_rate: f64,
    weight_decay: f64,
    newer_sample_weight: bool,
    backend: Option<&str>,
) -> PyResult<CoreNeuralPanelConfig> {
    let future_regressors = future_regressors
        .unwrap_or_default()
        .into_iter()
        .map(|(name, mode)| Ok((name, parse_neural_panel_component_mode(&mode)?)))
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    let custom_seasonalities = custom_seasonalities
        .unwrap_or_default()
        .into_iter()
        .map(|(name, period, order, condition_name)| (name, (period, order), condition_name))
        .collect::<Vec<_>>();
    let custom_seasonality_conditions = custom_seasonalities
        .iter()
        .map(|(name, _, condition_name)| (name.clone(), condition_name.clone()))
        .collect::<BTreeMap<_, _>>();
    let custom_seasonalities = custom_seasonalities
        .into_iter()
        .map(|(name, period_order, _condition_name)| (name, period_order))
        .collect();
    Ok(CoreNeuralPanelConfig {
        n_lags,
        n_forecasts,
        quantiles: quantiles.unwrap_or_else(|| vec![0.5]),
        trend: parse_neural_panel_trend_mode(trend)?,
        n_changepoints,
        changepoints_range,
        daily_fourier_order,
        weekly_fourier_order,
        yearly_fourier_order,
        custom_seasonalities,
        custom_seasonality_conditions,
        seasonality_mode: parse_neural_panel_component_mode(seasonality_mode)?,
        events: events.unwrap_or_default(),
        event_mode: parse_neural_panel_component_mode(event_mode)?,
        future_regressors,
        lagged_regressors: lagged_regressors.unwrap_or_default(),
        ar_layers: ar_layers.unwrap_or_default(),
        lagged_reg_layers: lagged_reg_layers.unwrap_or_default(),
        trend_mode: parse_neural_panel_global_local_mode(trend_mode)?,
        seasonality_global_local: parse_neural_panel_global_local_mode(seasonality_global_local)?,
        event_global_local: parse_neural_panel_global_local_mode(event_global_local)?,
        regressor_global_local: parse_neural_panel_global_local_mode(regressor_global_local)?,
        local_l2,
        seed,
        loss: parse_neural_panel_loss(loss)?,
        epochs,
        learning_rate,
        weight_decay,
        newer_sample_weight,
        backend: neural_select_backend(backend).map_err(to_py_neural_error)?,
    })
}

fn parse_neural_panel_loss(value: &str) -> PyResult<CoreNeuralPanelLoss> {
    match value {
        "smooth_l1" | "huber" => Ok(CoreNeuralPanelLoss::SmoothL1),
        "mse" | "l2" => Ok(CoreNeuralPanelLoss::Mse),
        "mae" | "l1" => Ok(CoreNeuralPanelLoss::Mae),
        "pinball" | "quantile" => Ok(CoreNeuralPanelLoss::Pinball),
        other => Err(PyValueError::new_err(format!(
            "unknown NeuralPanel loss {other:?}"
        ))),
    }
}

fn parse_neural_panel_trend_mode(value: &str) -> PyResult<CoreNeuralPanelTrendMode> {
    match value {
        "off" | "none" => Ok(CoreNeuralPanelTrendMode::Off),
        "piecewise_linear" | "linear" => Ok(CoreNeuralPanelTrendMode::PiecewiseLinear),
        other => Err(PyValueError::new_err(format!(
            "unknown NeuralPanel trend mode {other:?}"
        ))),
    }
}

fn parse_neural_panel_component_mode(value: &str) -> PyResult<CoreNeuralPanelComponentMode> {
    match value {
        "additive" => Ok(CoreNeuralPanelComponentMode::Additive),
        "multiplicative" => Ok(CoreNeuralPanelComponentMode::Multiplicative),
        other => Err(PyValueError::new_err(format!(
            "unknown NeuralPanel component mode {other:?}"
        ))),
    }
}

#[pyfunction]
fn geo_causal_synthetic_did_summary(
    rows: Vec<PyGeoCausalRow>,
    spatial_weights: Vec<(String, String, f64)>,
    intervention_time: String,
    seed: u64,
    placebo_n: usize,
) -> PyResult<String> {
    let panel = build_geo_causal_panel(rows, spatial_weights)?;
    let mut estimator = CoreSyntheticDIDEstimator::new(SyntheticDIDConfig {
        intervention_time,
        seed,
    });
    estimator.fit(panel).map_err(to_py_geo_causal_error)?;
    if placebo_n > 0 {
        estimator
            .placebo_test(placebo_n)
            .map_err(to_py_geo_causal_error)?;
    }
    estimator.summary_json().map_err(to_py_geo_causal_error)
}

#[pyfunction]
fn geo_causal_design_summary(
    rows: Vec<PyGeoCausalRow>,
    spatial_weights: Vec<(String, String, f64)>,
    intervention_time: String,
    seed: u64,
    candidate_count: usize,
    placebo_n: usize,
) -> PyResult<String> {
    let panel = build_geo_causal_panel(rows, spatial_weights)?;
    let designer = CoreGeoExperimentDesigner {
        intervention_time,
        seed,
    };
    let design = designer
        .design(&panel, candidate_count, placebo_n)
        .map_err(to_py_geo_causal_error)?;
    serde_json::to_string_pretty(&design).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn geo_causal_spatial_placebos(
    rows: Vec<PyGeoCausalRow>,
    spatial_weights: Vec<(String, String, f64)>,
    intervention_time: String,
    seed: u64,
    n: usize,
) -> PyResult<Vec<f64>> {
    let panel = build_geo_causal_panel(rows, spatial_weights)?;
    SpatialPlaceboTester {
        intervention_time,
        seed,
    }
    .placebo_estimates(panel, n)
    .map_err(to_py_geo_causal_error)
}

#[pyfunction]
fn geo_causal_spillover_diagnostics(
    rows: Vec<PyGeoCausalRow>,
    spatial_weights: Vec<(String, String, f64)>,
) -> PyResult<String> {
    let panel = build_geo_causal_panel(rows, spatial_weights)?;
    serde_json::to_string_pretty(&core_geo_causal_spillover_diagnostics(&panel))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn geo_causal_representation_report_value(
    features: Vec<Vec<f64>>,
    outcomes: Vec<f64>,
    regions: Vec<String>,
    heldout_region: String,
) -> PyResult<String> {
    core_geo_causal_representation_report_json(&features, &outcomes, &regions, &heldout_region)
        .map_err(to_py_geo_causal_error)
}

fn build_geo_causal_panel(
    rows: Vec<PyGeoCausalRow>,
    spatial_weights: Vec<(String, String, f64)>,
) -> PyResult<GeoCausalPanel> {
    let rows = rows
        .into_iter()
        .map(
            |(unit_id, time, outcome, treatment, covariates, latitude, longitude, region_id)| {
                GeoCausalRow {
                    unit_id,
                    time,
                    outcome,
                    treatment,
                    covariates,
                    latitude,
                    longitude,
                    region_id,
                }
            },
        )
        .collect();
    let spatial_weights = spatial_weights
        .into_iter()
        .map(|(from_unit, to_unit, weight)| SpatialWeight {
            from_unit,
            to_unit,
            weight,
        })
        .collect();
    GeoCausalPanel::new(rows, spatial_weights).map_err(to_py_geo_causal_error)
}

fn to_py_geo_causal_error(err: cartoboost_geo_causal::GeoCausalError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_neural_panel_global_local_mode(value: &str) -> PyResult<CoreNeuralPanelMode> {
    match value {
        "global" => Ok(CoreNeuralPanelMode::Global),
        "local" => Ok(CoreNeuralPanelMode::Local),
        "glocal" => Ok(CoreNeuralPanelMode::Glocal),
        other => Err(PyValueError::new_err(format!(
            "unknown NeuralPanel global/local mode {other:?}"
        ))),
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(model_manifest_json, m)?)?;
    m.add_class::<NativeCartoBoostRegressor>()?;
    m.add_class::<NativeNearestNeighborGPRegressor>()?;
    m.add_class::<NativeCartoBoostClassifier>()?;
    m.add_class::<NativeCartoBoostRanker>()?;
    m.add_class::<NativeCoordinateMatrix>()?;
    m.add_class::<NativeTimeIndex>()?;
    m.add_class::<NativePanelIndex>()?;
    m.add_class::<NativeGeoSpatialWeights>()?;
    m.add_class::<NativeSplitManifest>()?;
    m.add_class::<NativeSpatialWeights>()?;
    m.add_class::<NativeSpatialLagRegressor>()?;
    m.add_class::<NativeSpatialErrorRegressor>()?;
    m.add_class::<NativeSpatialDurbinRegressor>()?;
    m.add_class::<NativeSpatialTwoStageLeastSquares>()?;
    m.add_function(wrap_pyfunction!(categorical_fit_transform, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_transform, m)?)?;
    m.add_function(wrap_pyfunction!(validate_feature_schema_json, m)?)?;
    m.add_function(wrap_pyfunction!(geo_spatial_block_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geo_buffered_spatial_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geo_group_spatial_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geo_rolling_origin_panel_split, m)?)?;
    m.add_function(wrap_pyfunction!(geo_spatial_temporal_blocked_split, m)?)?;
    m.add_class::<NativeForecastFrame>()?;
    m.add_class::<NativeForecastResult>()?;
    m.add_class::<NativeForecastFold>()?;
    m.add_class::<NativeRollingOriginSplitter>()?;
    m.add_class::<NativeForecastMetricSet>()?;
    m.add_class::<NativeBacktestFoldResult>()?;
    m.add_class::<NativeBacktestResult>()?;
    m.add_class::<NativeRollingOriginBacktester>()?;
    m.add_class::<NativeNaiveForecaster>()?;
    m.add_class::<NativeSeasonalNaiveForecaster>()?;
    m.add_class::<NativeThetaForecaster>()?;
    m.add_class::<NativeOptimizedThetaForecaster>()?;
    m.add_class::<NativePiecewiseLinearSeasonalForecaster>()?;
    m.add_class::<NativeETSForecaster>()?;
    m.add_class::<NativeArimaForecaster>()?;
    m.add_class::<NativeAutoARIMAForecaster>()?;
    m.add_class::<NativeAutoStatsBank>()?;
    m.add_class::<NativeCrostonForecaster>()?;
    m.add_class::<NativeSbaForecaster>()?;
    m.add_class::<NativeTsbForecaster>()?;
    m.add_class::<NativeKalmanForecaster>()?;
    m.add_class::<NativeLocalLevelKalmanForecaster>()?;
    m.add_class::<NativeAutoKalmanForecaster>()?;
    m.add_class::<NativeAutoLocalLevelKalmanForecaster>()?;
    m.add_class::<NativeKrigingForecaster>()?;
    m.add_class::<NativeSpatialPiecewiseKrigingForecaster>()?;
    m.add_class::<NativeGraphTemporalFrame>()?;
    m.add_class::<NativeDcrnnForecaster>()?;
    m.add_class::<NativeSTAEformerForecaster>()?;
    m.add_class::<NativeGraphWaveNetForecaster>()?;
    m.add_class::<NativePropagationDelayGraphForecaster>()?;
    m.add_class::<NativeNBeatsForecaster>()?;
    m.add_class::<NativeNHiTSForecaster>()?;
    m.add_class::<NativeNeuralPanelForecaster>()?;
    m.add_class::<NativeLaneNeuralPanelForecaster>()?;
    m.add_class::<NativeAutoForecastModel>()?;
    m.add_class::<NativeCartoBoostLagForecaster>()?;
    m.add_class::<NativeWeightedEnsembleForecaster>()?;
    m.add_class::<NativeNeuralEmbeddingFeatures>()?;
    m.add_class::<NativeGraphSageEncoder>()?;
    m.add_class::<NativeNode2VecEncoder>()?;
    m.add_class::<NativeStandaloneNeuralEmbeddingRegressor>()?;
    m.add_class::<NativeStandaloneNode2VecRegressor>()?;
    m.add_class::<NativeStandaloneGraphSageRegressor>()?;
    m.add_class::<NativeStandaloneHeteroGraphSageRegressor>()?;
    m.add_class::<NativeStandaloneHinSageRegressor>()?;
    m.add_class::<NativeStandaloneNode2VecLinkPredictor>()?;
    m.add_class::<NativeStandaloneGraphSageLinkPredictor>()?;
    m.add_class::<NativeStandaloneHeteroGraphSageLinkPredictor>()?;
    m.add_class::<NativeStandaloneHinSageLinkPredictor>()?;
    m.add_class::<NativeHeteroGraphSageEncoder>()?;
    m.add_class::<NativeHinSageEncoder>()?;
    m.add_function(wrap_pyfunction!(utility_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(utility_local_level_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(utility_intermittent_demand_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(utility_ordinary_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(
        utility_ordinary_kriging_predict_detailed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utility_ordinary_kriging_leave_one_out, m)?)?;
    m.add_function(wrap_pyfunction!(utility_empirical_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(utility_fit_ordinary_kriging_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(
        utility_ordinary_kriging_leave_one_out_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utility_series_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(graph_compute_directional_features, m)?)?;
    m.add_function(wrap_pyfunction!(graph_validate_directed_metapath, m)?)?;
    m.add_function(wrap_pyfunction!(
        graph_materialize_source_target_pair_nodes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rmsse_scale_value, m)?)?;
    m.add_function(wrap_pyfunction!(wrmsse_value, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_equal_level_wrmsse_value, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_nonnegative_weights_value, m)?)?;
    m.add_function(wrap_pyfunction!(competition_forecast_metrics_value, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_candidate_choice_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_validation_unavailable_candidate_choice_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_candidate_validation_cutoff_indices_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forecast_magnitude_guard_allows_value, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_requires_lag_spine_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_seasonal_naive_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forecast_trend_candidate_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_calendar_profile_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_validation_ensemble_weights_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(forecast_shared_candidate_names_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        forecast_selectable_candidate_names_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_include_autostats_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_candidate_complexity_rank_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_native_auto_raw_candidate_is_confident_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_lag_origin_consistency_guard_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_relative_loss_displacement_allowed_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_stable_magnitude_candidate_choice_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_proportional_total_reconciliation_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        forecast_weighted_blend_candidate_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(prob_pinball_loss_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_interval_coverage_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_mean_interval_width_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_crps_approximation_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_weighted_interval_score_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_pit_bins_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_conditional_flow_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_conditional_flow_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(prob_diffusion_scenario_generate_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        prob_split_conformal_residual_quantile_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_weighted_conformal_residual_quantile_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_group_conformal_residual_quantiles_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_rolling_origin_conformal_residual_quantiles_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_nearest_conformal_residual_quantiles_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prob_benchmark_calibration_report_fields_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(portfolio_summary_value, m)?)?;
    m.add_function(wrap_pyfunction!(extreme_portfolio_decisions_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_hit_rates_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_buckets_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_scored_assets_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_portfolio_summary_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_portfolio_decision_loss_value, m)?)?;
    m.add_function(wrap_pyfunction!(rank_probability_calibration_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        calibrated_rank_bucket_probabilities_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sequence_validate_value, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_state_space_value, m)?)?;
    m.add_function(wrap_pyfunction!(sequence_reference_path_viterbi_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        sequence_reference_path_posterior_mean_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sequence_blend_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        sequence_validate_oof_meta_training_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sequence_generate_group_oof_candidate_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sequence_group_error_summary_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_normalize_id_text, m)?)?;
    m.add_function(wrap_pyfunction!(s2_normalize_id_text, m)?)?;
    m.add_function(wrap_pyfunction!(h3_normalize_resolution_value, m)?)?;
    m.add_function(wrap_pyfunction!(s2_normalize_level_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_normalize_coordinate_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        geo_clockwise_bearing_unit_vector_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        geo_initial_bearing_unit_vector_latlng_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(geo_route_feature_vector_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_radial_anchor_distances_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_rbf_anchor_features_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_local_frame_features_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_validate_parent_resolutions_value, m)?)?;
    m.add_function(wrap_pyfunction!(s2_validate_parent_levels_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_scaffold_parent_id_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_expand_sparse_set_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_sparse_row_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_sparse_column_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_route_sparse_rows_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_validate_equal_row_count_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_synthetic_did_summary, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_design_summary, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_spatial_placebos, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_spillover_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(geo_causal_representation_report_value, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_overlay, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_parse_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_evaluate_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(graph_st_available_backends_value, m)?)?;
    m.add_function(wrap_pyfunction!(geostats_empirical_semivariogram_value, m)?)?;
    m.add_function(wrap_pyfunction!(geostats_fit_variogram_wls_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_response_curve_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_event_outcome_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_event_outcome_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_directional_pair_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_directional_pair_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_directional_pair_predict_artifact_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_service_residual_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_available_backends_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_backend_dispatch_report_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_constrained_decision_select_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_choice_set_transformer_report_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_fit_value, m)?)?;
    m.add_function(wrap_pyfunction!(deep_temporal_entity_predict_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        deep_graph_neural_operator_predict_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        deep_neural_operator_synthetic_benchmark_value,
        m
    )?)?;
    Ok(())
}
