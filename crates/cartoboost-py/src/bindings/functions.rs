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
        .detach(|| {
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
    py.detach(|| validate_directed_metapath(&steps, &edge_types))
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
        .detach(|| {
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
    py.detach(|| core_rmsse_scale(&train, seasonal_period))
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
        .detach(|| core_wrmsse(&series, seasonal_period))
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
        .detach(|| core_aggregate_equal_level_wrmsse(&level_scores))
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
    py.detach(|| core_ordered_nonnegative_weights(&ids, &raw_weights))
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
        .detach(|| {
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
    py.detach(|| {
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
    py.detach(|| {
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
    py.detach(|| {
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
    py.detach(|| forecast_magnitude_guard_allows(forecast_max_abs, training_max_abs))
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
    Ok(py.detach(|| core_requires_lag_spine(&source, season_length, horizon)))
}

#[pyfunction]
fn forecast_seasonal_naive_candidate_value(
    py: Python<'_>,
    values: Vec<f64>,
    season_length: usize,
) -> PyResult<f64> {
    py.detach(|| core_seasonal_naive_candidate_prediction(&values, season_length))
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
    py.detach(|| core_trend_candidate_prediction(&values, step, season_length, &mode))
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
    py.detach(|| {
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
    py.detach(|| core_validation_ensemble_weights(&candidate_scores))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_shared_candidate_names_value(py: Python<'_>) -> PyResult<Vec<String>> {
    Ok(py.detach(core_shared_candidate_names))
}

#[pyfunction]
fn forecast_selectable_candidate_names_value(
    py: Python<'_>,
    model: &str,
    source: &str,
) -> PyResult<Vec<String>> {
    let model = model.to_string();
    let source = source.to_string();
    Ok(py.detach(|| core_selectable_candidate_names(&model, &source)))
}

#[pyfunction]
fn forecast_include_autostats_candidate_value(
    py: Python<'_>,
    source: &str,
    season_length: usize,
    horizon: usize,
) -> PyResult<bool> {
    let source = source.to_string();
    Ok(py.detach(|| core_include_autostats_candidate(&source, season_length, horizon)))
}

#[pyfunction]
fn forecast_candidate_complexity_rank_value(py: Python<'_>, candidate: &str) -> PyResult<usize> {
    let candidate = candidate.to_string();
    Ok(py.detach(|| core_candidate_complexity_rank(&candidate)))
}

#[pyfunction(signature = (selected_candidate=None, inner_raw_relative_rmse_gain=None))]
fn forecast_native_auto_raw_candidate_is_confident_value(
    py: Python<'_>,
    selected_candidate: Option<String>,
    inner_raw_relative_rmse_gain: Option<f64>,
) -> PyResult<bool> {
    Ok(py.detach(|| {
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
    py.detach(|| {
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
    py.detach(|| {
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
    py.detach(|| {
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
    py.detach(|| core_proportional_total_reconciliation(&base_values, target_total, gamma))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn forecast_weighted_blend_candidate_value(
    py: Python<'_>,
    primary_forecast: Vec<f64>,
    secondary_forecast: Vec<f64>,
    primary_weight: f64,
) -> PyResult<Vec<f64>> {
    py.detach(|| {
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
        .detach(|| extreme_portfolio_decisions(&rows))
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
        .detach(|| portfolio_summary(&parsed))
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
        .detach(|| rank_hit_rates(&rows, bucket_count))
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
    py.detach(|| rank_buckets(&values, bucket_count))
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
        .detach(|| rank_scored_assets(&rows, bucket_count, &calibration_probabilities, shrinkage))
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
        .detach(|| {
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
    py.detach(|| {
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
        .detach(|| {
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
    py.detach(|| {
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
        .detach(|| {
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
        .detach(|| match method.as_str() {
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
        .detach(|| core_reference_path_viterbi(&series, &reference, config))
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
        .detach(|| core_reference_path_posterior_mean(&series, &reference, config))
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
            py.detach(|| SequenceCandidateEnsemble::validation_derived(&candidates, &actuals))
                .map_err(to_py_value_error)?
        }
        "constrained" | "nonnegative" | "constrained_nonnegative_linear_blend" => {
            let actuals = parse_sequence_actuals(actuals_json)?;
            py.detach(|| {
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
        .detach(|| ensemble.predict(&candidates))
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
    py.detach(|| cartoboost_core::forecasting::validate_oof_meta_training(&rows))
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
        .detach(|| cartoboost_core::forecasting::generate_group_oof_candidate_rows(&fold))
        .map_err(to_py_value_error)?;
    serde_json::to_string(&rows).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
fn sequence_group_error_summary_value(py: Python<'_>, rows_json: &str) -> PyResult<String> {
    let rows = serde_json::from_str::<Vec<SequenceGroupPrediction>>(rows_json)
        .map_err(|err| PyValueError::new_err(format!("invalid group prediction rows: {err}")))?;
    let result = py
        .detach(|| cartoboost_core::forecasting::per_group_error_summary(&rows))
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
    py.detach(|| validate_parent_levels(resolution, &parent_resolutions, GeoGridKind::H3))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn s2_validate_parent_levels_value(
    py: Python<'_>,
    level: u8,
    parent_levels: Vec<u8>,
) -> PyResult<()> {
    py.detach(|| validate_parent_levels(level, &parent_levels, GeoGridKind::S2))
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
    py.detach(|| core_expand_h3_sparse_set(&values, resolution, &parent_resolutions))
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
    py.detach(|| assemble_sparse_column(&children, &parent_columns))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn geo_assemble_route_sparse_rows_value(
    py: Python<'_>,
    route_cells: Vec<Vec<u64>>,
) -> PyResult<Vec<Vec<u64>>> {
    py.detach(|| assemble_route_sparse_rows(&route_cells))
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
    // Never rely on Rayon’s global pool for public model operations.  It can
    // be initialized by a notebook host, another extension, or an embedding
    // application with a single worker before CartoBoost is imported.
    // Constructing a cached pool here makes the default genuinely use the
    // machine’s available CPUs while preserving an explicit user override.
    let n_threads = n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
    });
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
        .detach(|| {
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

fn parse_graph_transformer_profile(value: &str) -> PyResult<CoreGraphTransformerProfile> {
    match value {
        "heterogeneous_moe" => Ok(CoreGraphTransformerProfile::HeterogeneousMoE),
        "efficient_high_order" => Ok(CoreGraphTransformerProfile::EfficientHighOrder),
        "long_short_fusion" => Ok(CoreGraphTransformerProfile::LongShortFusion),
        "gated_graph_temporal" => Ok(CoreGraphTransformerProfile::GatedGraphTemporal),
        "spatial_shift_graphon_moe" => Ok(CoreGraphTransformerProfile::SpatialShiftGraphonMoE),
        other => Err(PyValueError::new_err(format!(
            "unknown graph transformer profile {other:?}"
        ))),
    }
}

// The bindings retain no Python-owned state. Long-running operations detach
// from the interpreter, and PyO3 enforces runtime borrowing for mutable
// pyclasses. Declaring this explicitly keeps CPython's free-threaded builds
// from re-enabling the GIL on import.
