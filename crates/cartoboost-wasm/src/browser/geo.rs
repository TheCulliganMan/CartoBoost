fn run_geostatistics_request(request: BrowserGeostatsRequest) -> Result<BrowserGeostatsResponse> {
    if request.observations.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "geostatistics requires at least one observation".to_string(),
        ));
    }
    if request.targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "geostatistics requires at least one target coordinate".to_string(),
        ));
    }
    let coords = request
        .observations
        .iter()
        .map(|row| [row.x, row.y])
        .collect::<Vec<_>>();
    let values = request
        .observations
        .iter()
        .map(|row| row.value)
        .collect::<Vec<_>>();
    let targets = request
        .targets
        .iter()
        .map(|row| [row.x, row.y])
        .collect::<Vec<_>>();
    let options = request.options;
    let config = NngpConfig {
        kernel: CovarianceKernel::parse(&options.kernel).map_err(|error| {
            CartoBoostError::InvalidInput(format!("invalid geostatistics kernel: {error}"))
        })?,
        range: options.range,
        sill: options.sill,
        nugget: options.nugget,
        anisotropy: GeostatsAnisotropy {
            angle_degrees: options.anisotropy_angle_degrees,
            scaling: options.anisotropy_scaling,
        },
        n_neighbors: options.n_neighbors,
        brute_force_threshold: 2048,
        duplicate_tolerance: 0.0,
    };
    let mut model = WasmNearestNeighborGPRegressor::new(config)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    model
        .fit(&coords, &values)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let predictions = model
        .predict(&targets)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?
        .into_iter()
        .zip(request.targets)
        .map(|(prediction, target)| BrowserGeostatsPrediction {
            x: target.x,
            y: target.y,
            mean: prediction.mean,
            variance: prediction.variance,
            std: prediction.variance.max(0.0).sqrt(),
            neighbor_indices: prediction.neighbor_indices,
        })
        .collect();
    Ok(BrowserGeostatsResponse {
        predictions,
        metadata: json!({
            "model": "nearest_neighbor_gp",
            "kernel": config.kernel.as_str(),
            "range": config.range,
            "sill": config.sill,
            "nugget": config.nugget,
            "n_neighbors": config.n_neighbors,
            "works_without_gpu": true,
        }),
    })
}

fn run_geo_feature_examples_request(
    request: BrowserGeoFeatureRequest,
) -> Result<BrowserGeoFeatureResponse> {
    let anchors = request.anchors;
    let anchor_points = anchors
        .iter()
        .map(|anchor| anchor.point)
        .collect::<Vec<_>>();
    let anchor_labels = anchors
        .iter()
        .map(|anchor| anchor.label.clone())
        .collect::<Vec<_>>();
    let planar = request
        .planar_routes
        .iter()
        .map(|route| {
            let vector = clockwise_bearing_unit_vector(route.origin, route.destination);
            BrowserBearingFeature {
                label: route.label.clone(),
                east: vector.map(|value| value[0]),
                north: vector.map(|value| value[1]),
                zero_distance: vector.is_none(),
            }
        })
        .collect();
    let latlng = request
        .latlng_routes
        .into_iter()
        .map(|route| {
            let vector = initial_bearing_unit_vector_latlng(
                route.origin[0],
                route.origin[1],
                route.destination[0],
                route.destination[1],
            );
            BrowserBearingFeature {
                label: route.label,
                east: vector.map(|value| value[0]),
                north: vector.map(|value| value[1]),
                zero_distance: vector.is_none(),
            }
        })
        .collect();
    let routes = request
        .planar_routes
        .iter()
        .map(|route| {
            let vector = route_feature_vector(route.origin, route.destination);
            BrowserRouteFeature {
                label: route.label.clone(),
                mid_x: vector.map(|value| value[0]),
                mid_y: vector.map(|value| value[1]),
                distance: vector.map(|value| value[2]),
                bearing_east: vector.map(|value| value[3]),
                bearing_north: vector.map(|value| value[4]),
                zero_distance: vector.is_none(),
            }
        })
        .collect();
    let radial = request
        .radial_points
        .iter()
        .map(|point| BrowserAnchorFeatureRow {
            label: point.label.clone(),
            values: radial_anchor_distances(point.point, &anchor_points),
        })
        .collect::<Vec<_>>();
    let rbf = request
        .radial_points
        .iter()
        .map(|point| {
            Ok(BrowserAnchorFeatureRow {
                label: point.label.clone(),
                values: rbf_anchor_features(point.point, &anchor_points, request.length_scale)
                    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let local_frame = request
        .local_frame
        .map(|frame| {
            frame
                .points
                .into_iter()
                .map(|point| {
                    let vector = local_frame_features(point.point, frame.origin, frame.axis);
                    BrowserLocalFrameFeature {
                        label: point.label,
                        along_axis: vector.map(|value| value[0]),
                        cross_axis: vector.map(|value| value[1]),
                        invalid_axis: vector.is_none(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(BrowserGeoFeatureResponse {
        planar,
        latlng,
        routes,
        radial,
        rbf,
        local_frame,
        metadata: json!({
            "surface": "rust_geo_feature_examples",
            "bearingEncoding": "(east,north) unit vector",
            "clockReference": "clockwise from north",
            "anchorLabels": anchor_labels,
            "rbfLengthScale": request.length_scale,
            "zeroDistancePolicy": "null components with zeroDistance=true",
        }),
    })
}

fn default_geo_feature_length_scale() -> f64 {
    1.0
}

fn default_geostats_kernel() -> String {
    "matern_3_2".to_string()
}

fn default_geostats_range() -> f64 {
    0.025
}

fn default_geostats_sill() -> f64 {
    1.0
}

fn default_geostats_nugget() -> f64 {
    1.0e-6
}

fn default_geostats_neighbors() -> usize {
    12
}

fn default_geostats_anisotropy_scaling() -> f64 {
    1.0
}

fn default_graph_diffusion_steps() -> usize {
    2
}

fn default_graph_hidden_size() -> usize {
    8
}

fn default_graph_epochs() -> usize {
    160
}

fn default_graph_learning_rate() -> f64 {
    0.03
}

fn default_graph_teacher_forcing_start() -> f64 {
    1.0
}

fn default_graph_teacher_forcing_end() -> f64 {
    0.2
}

fn default_graph_ridge() -> f64 {
    0.0001
}

fn default_backend() -> String {
    "auto".to_string()
}

fn default_seed() -> u64 {
    13
}

fn run_geo_causal_request(request: BrowserGeoCausalRequest) -> Result<Value> {
    let rows = request
        .rows
        .into_iter()
        .map(|row| GeoCausalRow {
            unit_id: row.unit_id,
            time: row.time,
            outcome: row.outcome,
            treatment: row.treatment,
            covariates: row.covariates,
            latitude: row.latitude,
            longitude: row.longitude,
            region_id: row.region_id,
        })
        .collect();
    let spatial_weights = request
        .spatial_weights
        .into_iter()
        .map(|edge| SpatialWeight {
            from_unit: edge.from_unit,
            to_unit: edge.to_unit,
            weight: edge.weight,
        })
        .collect();
    let panel = GeoCausalPanel::new(rows, spatial_weights)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let mut estimator = CoreSyntheticDIDEstimator::new(SyntheticDIDConfig {
        intervention_time: request.intervention_time,
        seed: request.seed,
    });
    estimator
        .fit(panel)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    if request.placebo_n > 0 {
        estimator
            .placebo_test(request.placebo_n)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    }
    serde_json::to_value(
        estimator
            .estimate_effect()
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))
}

fn run_geotemporal_diagnostics_request(
    request: BrowserGeotemporalDiagnosticsRequest,
) -> Result<Value> {
    let mut response = serde_json::Map::new();
    response.insert("surface".to_string(), json!("rust_geotemporal_diagnostics"));
    if let Some(quantiles) = request.quantiles {
        response.insert(
            "quantiles".to_string(),
            run_browser_quantile_diagnostics(quantiles)?,
        );
    }
    if let Some(residual_correction) = request.residual_correction {
        response.insert(
            "residualCorrection".to_string(),
            run_browser_residual_correction(residual_correction)?,
        );
    }
    if let Some(regime) = request.regime {
        response.insert(
            "regime".to_string(),
            run_browser_regime_diagnostics(regime)?,
        );
    }
    if let Some(calibration) = request.calibration {
        response.insert(
            "calibration".to_string(),
            run_browser_calibration(calibration)?,
        );
    }
    Ok(Value::Object(response))
}

fn run_browser_quantile_diagnostics(request: BrowserQuantileDiagnosticsRequest) -> Result<Value> {
    let mut response = serde_json::Map::new();
    response.insert(
        "defaultLevels".to_string(),
        json!(default_quantile_levels()),
    );
    if let Some(values) = request.values.as_deref() {
        response.insert(
            "repairedValues".to_string(),
            json!(repair_non_crossing_quantiles(values)?),
        );
    }
    if let (Some(actual), Some(prediction), Some(quantile)) = (
        request.actual.as_deref(),
        request.prediction.as_deref(),
        request.quantile,
    ) {
        response.insert(
            "pinballLoss".to_string(),
            json!(pinball_loss(actual, prediction, quantile)?),
        );
    }
    if let (Some(actual), Some(lower), Some(upper), Some(quantile_rows)) = (
        request.actual.as_deref(),
        request.lower.as_deref(),
        request.upper.as_deref(),
        request.quantile_rows.as_deref(),
    ) {
        response.insert(
            "intervalDiagnostics".to_string(),
            serde_json::to_value(interval_diagnostics(actual, lower, upper, quantile_rows)?)?,
        );
    } else if let Some(quantile_rows) = request.quantile_rows.as_deref() {
        response.insert(
            "crossingRate".to_string(),
            json!(crossing_rate(quantile_rows)?),
        );
    }
    Ok(Value::Object(response))
}

fn run_browser_residual_correction(request: BrowserResidualCorrectionRequest) -> Result<Value> {
    let default_filter = StateFilter::new(request.process_variance, request.observation_variance)?;
    let mut corrector = KalmanResidualCorrector::new(default_filter);
    let observations = request
        .observations
        .into_iter()
        .map(|observation| StateObservation {
            key: browser_residual_key(observation.key),
            structural_prediction: observation.structural_prediction,
            observed: observation.observed,
        })
        .collect::<Vec<_>>();
    let corrections = corrector.apply_sequence(&observations)?;
    let states = corrector
        .states
        .iter()
        .map(|(key, filter)| json!({ "key": key, "filter": filter }))
        .collect::<Vec<_>>();
    Ok(json!({
        "corrections": corrections,
        "stateCount": corrector.states.len(),
        "states": states,
    }))
}

fn browser_residual_key(key: BrowserResidualStateKey) -> ResidualStateKey {
    ResidualStateKey::new(
        key.origin.unwrap_or_default(),
        key.destination.unwrap_or_default(),
        key.corridor.unwrap_or_default(),
        key.segment.unwrap_or_default(),
        key.entity_family.unwrap_or_default(),
        key.target_family.unwrap_or_default(),
        key.time_bucket.unwrap_or_default(),
    )
}

fn run_browser_regime_diagnostics(request: BrowserRegimeDiagnosticsRequest) -> Result<Value> {
    let rolling_window = request.rolling_window.unwrap_or(5);
    let mut response = serde_json::Map::new();
    response.insert(
        "rollingMedianResidual".to_string(),
        json!(rolling_median_residual(&request.residuals, rolling_window)?),
    );
    response.insert(
        "rollingMadResidual".to_string(),
        json!(rolling_mad_residual(&request.residuals, rolling_window)?),
    );
    if let Some(config) = request.cusum {
        let mut detector = CUSUM::new(config)?;
        response.insert(
            "cusum".to_string(),
            serde_json::to_value(detector.scan(&request.residuals)?)?,
        );
    }
    let page_hinkley_signals = if let Some(config) = request.page_hinkley {
        let mut detector = PageHinkley::new(config)?;
        let signals = detector.scan(&request.residuals)?;
        response.insert("pageHinkley".to_string(), serde_json::to_value(&signals)?);
        Some(signals)
    } else {
        None
    };
    let volatilities = if let Some(config) = request.ewma {
        let mut volatility = EwmaVolatility::new(config)?;
        let values = volatility.scan(&request.residuals)?;
        response.insert("ewmaVolatility".to_string(), json!(&values));
        Some(values)
    } else {
        None
    };
    if let (Some(lower), Some(upper), Some(signals), Some(volatilities), Some(policy)) = (
        request.lower.as_deref(),
        request.upper.as_deref(),
        page_hinkley_signals.as_deref(),
        volatilities.as_deref(),
        request.policy,
    ) {
        response.insert(
            "regimeAdjustedIntervals".to_string(),
            serde_json::to_value(regime_adjusted_intervals(
                lower,
                upper,
                signals,
                volatilities,
                policy,
            )?)?,
        );
    }
    Ok(Value::Object(response))
}

fn run_browser_calibration(request: BrowserCalibrationRequest) -> Result<Value> {
    let bucket_count = request.bucket_count.unwrap_or(10);
    let mut response = serde_json::Map::new();
    if let Some(event) = request.event {
        response.insert(
            "eventLabels".to_string(),
            json!(browser_event_labels(event)?),
        );
    }
    if let Some(probabilities) = request.probabilities.as_deref() {
        response.insert(
            "metrics".to_string(),
            serde_json::to_value(calibration_metrics(
                &request.labels,
                probabilities,
                bucket_count,
            )?)?,
        );
    }
    if let (Some(scores), Some(method)) = (request.scores.as_deref(), request.method.as_deref()) {
        let calibrated = match method {
            "sigmoid" | "platt" => {
                let calibrator = SigmoidCalibrator::fit(scores, &request.labels)?;
                response.insert("calibrator".to_string(), serde_json::to_value(calibrator)?);
                calibrator.predict(scores)?
            }
            "temperature" => {
                let calibrator = TemperatureCalibrator::fit(scores, &request.labels)?;
                response.insert("calibrator".to_string(), serde_json::to_value(calibrator)?);
                calibrator.predict(scores)?
            }
            "isotonic" => {
                let calibrator = IsotonicCalibrator::fit(scores, &request.labels)?;
                response.insert("calibrator".to_string(), serde_json::to_value(&calibrator)?);
                calibrator.predict(scores)?
            }
            other => {
                return Err(CartoBoostError::InvalidInput(format!(
                    "unknown calibration method '{other}'"
                )));
            }
        };
        response.insert("calibratedProbabilities".to_string(), json!(&calibrated));
        response.insert(
            "calibratedMetrics".to_string(),
            serde_json::to_value(calibration_metrics(
                &request.labels,
                &calibrated,
                bucket_count,
            )?)?,
        );
        if let Some(before) = request
            .before_probabilities
            .as_deref()
            .or(request.probabilities.as_deref())
        {
            response.insert(
                "improvement".to_string(),
                serde_json::to_value(calibration_improvement(
                    &request.labels,
                    before,
                    &calibrated,
                    bucket_count,
                )?)?,
            );
        }
    }
    Ok(Value::Object(response))
}

fn browser_event_labels(request: BrowserCalibrationEventRequest) -> Result<Vec<f64>> {
    match request.kind.as_str() {
        "success_within_threshold" | "successWithinThreshold" => {
            let prediction = request.prediction.as_deref().ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "success_within_threshold event requires prediction".to_string(),
                )
            })?;
            let threshold = request.threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "success_within_threshold event requires threshold".to_string(),
                )
            })?;
            success_within_threshold(&request.actual, prediction, threshold)
        }
        "event_within_horizon" | "eventWithinHorizon" => {
            let horizon = request.horizon.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "event_within_horizon event requires horizon".to_string(),
                )
            })?;
            event_within_horizon(&request.actual, horizon)
        }
        "failure_risk" | "failureRisk" => {
            let threshold = request.threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput("failure_risk event requires threshold".to_string())
            })?;
            failure_risk_event(&request.actual, threshold)
        }
        "escalation_risk" | "escalationRisk" => {
            let warning_threshold = request.warning_threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "escalation_risk event requires warningThreshold".to_string(),
                )
            })?;
            let critical_threshold = request.critical_threshold.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "escalation_risk event requires criticalThreshold".to_string(),
                )
            })?;
            escalation_risk_event(&request.actual, warning_threshold, critical_threshold)
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown calibration event kind '{other}'"
        ))),
    }
}

fn run_sequence_request(request: BrowserSequenceRequest) -> Result<Value> {
    match request.operation.trim().to_ascii_lowercase().as_str() {
        "validate" | "validate_frame" => {
            let frame = request.frame.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence validate requires frame".to_string())
            })?;
            frame.validate()?;
            Ok(json!({ "ok": true }))
        }
        "ekf" | "forward_ekf" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::forward_ekf(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "ukf" | "ukf_reference" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::ukf_reference(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "rts" | "rts_smoother" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::rts_smoother(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "continuation" | "missing_target_continuation" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::missing_target_continuation(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "viterbi" | "reference_path_viterbi" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.reference_path_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::reference_path_viterbi(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "posterior_mean" | "reference_path_posterior_mean" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.reference_path_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::reference_path_posterior_mean(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "blend_fixed" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let weights = request.weights.ok_or_else(|| {
                CartoBoostError::InvalidInput("fixed sequence blend requires weights".to_string())
            })?;
            let ensemble = SequenceCandidateEnsemble::fixed(weights)?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "blend_validation" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let actuals = request.actuals.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "validation sequence blend requires actuals".to_string(),
                )
            })?;
            let ensemble = SequenceCandidateEnsemble::validation_derived(&candidates, &actuals)?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "blend_constrained" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let actuals = request.actuals.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "constrained sequence blend requires actuals".to_string(),
                )
            })?;
            let ensemble = SequenceCandidateEnsemble::constrained_nonnegative_linear_blend(
                &candidates,
                &actuals,
            )?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "validate_oof" | "validate_oof_meta_training" => {
            let rows = request.oof_rows.ok_or_else(|| {
                CartoBoostError::InvalidInput("OOF validation requires oofRows".to_string())
            })?;
            cartoboost_core::forecasting::validate_oof_meta_training(&rows)?;
            Ok(json!({ "ok": true }))
        }
        "generate_oof" | "generate_group_oof_candidate_rows" => {
            let fold = request.oof_fold.ok_or_else(|| {
                CartoBoostError::InvalidInput("OOF generation requires oofFold".to_string())
            })?;
            serde_json::to_value(
                cartoboost_core::forecasting::generate_group_oof_candidate_rows(&fold)?,
            )
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "group_metrics" | "per_group_error_summary" => {
            let rows = request.group_predictions.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "group metric summary requires groupPredictions".to_string(),
                )
            })?;
            serde_json::to_value(cartoboost_core::forecasting::per_group_error_summary(
                &rows,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown sequence operation {other:?}"
        ))),
    }
}

fn sequence_series_arg(request: &BrowserSequenceRequest) -> Result<SequenceSeries> {
    request.series.clone().ok_or_else(|| {
        CartoBoostError::InvalidInput("sequence operation requires series".to_string())
    })
}

fn sequence_reference_arg(request: &BrowserSequenceRequest) -> Result<ReferenceSignal> {
    request.reference.clone().ok_or_else(|| {
        CartoBoostError::InvalidInput("sequence operation requires reference".to_string())
    })
}

