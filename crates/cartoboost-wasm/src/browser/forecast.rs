fn run_forecast_request(request: BrowserForecastRequest) -> Result<BrowserForecastResponse> {
    if request.horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    let model = request.model;
    let options = request.options;
    let frame =
        forecast_frame_from_browser_request(request.rows, request.frequency, request.metadata)?;
    if is_piecewise_linear_seasonal_model(&model) {
        let mut config = piecewise_linear_seasonal_config(&options)?;
        let include_components = options.include_components.unwrap_or(false);
        let include_history_components = options.include_history_components.unwrap_or(false);
        let include_samples =
            options.include_samples.unwrap_or(false) && config.uncertainty_samples > 0;
        let include_quantiles =
            options.include_quantiles.unwrap_or(true) && !config.quantile_levels.is_empty();
        if !include_samples
            && config.interval_levels.is_empty()
            && config.quantile_levels.is_empty()
        {
            config.uncertainty_samples = 0;
        }
        let mut forecaster = PiecewiseLinearSeasonalForecaster::new(config)?;
        forecaster.fit(&frame)?;
        let forecast = forecaster.predict(request.horizon)?;
        let components = if include_components {
            Some(js_safe_json_value(
                forecaster.predict_components_json_value(request.horizon)?,
            ))
        } else {
            None
        };
        let history_components = if include_history_components {
            Some(js_safe_json_value(
                forecaster.history_components_json_value()?,
            ))
        } else {
            None
        };
        let samples = if include_samples {
            Some(js_safe_json_value(
                forecaster.predict_samples_json_value(request.horizon)?,
            ))
        } else {
            None
        };
        let quantiles = if include_quantiles {
            Some(js_safe_json_value(
                forecaster.predict_quantiles_json_value(request.horizon, None)?,
            ))
        } else {
            None
        };
        return Ok(BrowserForecastResponse {
            metadata: js_safe_json_value(json!({
                "model": forecaster.model_name(),
                "input": frame.metadata_value(),
                "modelMetadata": forecaster.metadata(),
            })),
            forecast: js_safe_json_value(forecast.to_json_value()),
            components,
            history_components,
            samples,
            quantiles,
        });
    }
    if model.trim().to_ascii_lowercase().replace('-', "_") == "neural_panel" {
        let mut forecaster = NeuralPanelForecaster::new(
            neural_panel_config(&options, request.horizon)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?,
        )
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        forecaster.fit(&frame)?;
        let covariates = if has_browser_neural_future_regressors(&options) {
            Some(browser_neural_future_covariates(
                &frame,
                &options,
                request.horizon,
            )?)
        } else {
            None
        };
        let forecast = if let Some(covariates) = &covariates {
            forecaster.predict_with_known_future_covariates(request.horizon, covariates)?
        } else {
            forecaster.predict(request.horizon)?
        };
        let components = if options.include_components.unwrap_or(false) {
            Some(js_safe_json_value(if let Some(covariates) = &covariates {
                forecaster.predict_components_json_value_with_known_future_covariates(
                    request.horizon,
                    Some(covariates),
                )?
            } else {
                forecaster.predict_components_json_value(request.horizon)?
            }))
        } else {
            None
        };
        let history_components = if options.include_history_components.unwrap_or(false) {
            Some(js_safe_json_value(
                forecaster.history_components_json_value()?,
            ))
        } else {
            None
        };
        let metadata = json!({
            "model": forecaster.model_name(),
            "input": frame.metadata_value(),
            "modelMetadata": forecaster.metadata(),
        });
        return Ok(BrowserForecastResponse {
            metadata: js_safe_json_value(metadata),
            forecast: js_safe_json_value(forecast.to_json_value()),
            components,
            history_components,
            samples: None,
            quantiles: None,
        });
    }
    let mut forecaster = build_forecaster(&model, &options, &frame, request.horizon)?;
    let fit_result = forecaster
        .fit(&frame)
        .and_then(|()| forecaster.predict(request.horizon));
    match fit_result {
        Ok(forecast) => Ok(forecast_response(
            forecaster.model_name(),
            &frame,
            forecaster.metadata(),
            forecast,
            None,
        )),
        Err(error) => Err(error),
    }
}

fn has_browser_neural_future_regressors(options: &BrowserForecastOptions) -> bool {
    options
        .extra_regressors
        .as_ref()
        .map(|values| !values.is_empty())
        .unwrap_or(false)
        || options
            .future_regressors
            .as_ref()
            .map(|values| !values.is_empty())
            .unwrap_or(false)
        || options
            .future_regressors_by_series
            .as_ref()
            .map(|values| !values.is_empty())
            .unwrap_or(false)
}

fn browser_neural_future_covariates(
    frame: &ForecastFrame,
    options: &BrowserForecastOptions,
    horizon: usize,
) -> Result<BTreeMap<(String, NaiveDateTime), BTreeMap<String, f64>>> {
    let mut regressor_names = BTreeSet::new();
    if let Some(names) = &options.extra_regressors {
        regressor_names.extend(names.iter().cloned());
    }
    if let Some(values) = &options.future_regressors {
        regressor_names.extend(values.keys().cloned());
    }
    if let Some(values) = &options.future_regressors_by_series {
        for series_values in values.values() {
            regressor_names.extend(series_values.keys().cloned());
        }
    }
    if regressor_names.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut covariates = BTreeMap::new();
    for series_id in frame.series_ids() {
        let rows = frame.rows_for_series(&series_id);
        let last_row = rows.last().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing fitted timestamp tail for series '{series_id}'"
            ))
        })?;
        for step in 1..=horizon {
            let timestamp = frame.frequency().advance(last_row.timestamp, step)?;
            let entry = covariates
                .entry((series_id.clone(), timestamp))
                .or_insert_with(BTreeMap::new);
            for name in &regressor_names {
                let value = options
                    .future_regressors_by_series
                    .as_ref()
                    .and_then(|series_values| series_values.get(&series_id))
                    .and_then(|series_values| series_values.get(name))
                    .and_then(|values| values.get(step - 1))
                    .copied()
                    .or_else(|| {
                        options
                            .future_regressors
                            .as_ref()
                            .and_then(|values| values.get(name))
                            .and_then(|values| values.get(step - 1))
                            .copied()
                    })
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "future regressor '{name}' requires known future covariates for prediction"
                        ))
                    })?;
                entry.insert(name.clone(), value);
            }
        }
    }
    Ok(covariates)
}

fn forecast_response(
    model_name: &str,
    frame: &ForecastFrame,
    model_metadata: Value,
    forecast: ForecastResult,
    warning: Option<Value>,
) -> BrowserForecastResponse {
    let mut metadata = json!({
        "model": model_name,
        "input": frame.metadata_value(),
        "modelMetadata": model_metadata,
    });
    if let Some(warning) = warning {
        metadata["warning"] = warning;
    }
    BrowserForecastResponse {
        metadata: js_safe_json_value(metadata),
        forecast: js_safe_json_value(forecast.to_json_value()),
        components: None,
        history_components: None,
        samples: None,
        quantiles: None,
    }
}

fn fit_piecewise_linear_seasonal_artifact_request(
    request: BrowserForecastRequest,
) -> Result<BrowserForecastArtifactResponse> {
    let frame =
        forecast_frame_from_browser_request(request.rows, request.frequency, request.metadata)?;
    let mut forecaster = PiecewiseLinearSeasonalForecaster::new(piecewise_linear_seasonal_config(
        &request.options,
    )?)?;
    forecaster.fit(&frame)?;
    let artifact = forecaster.to_json_string()?;
    Ok(BrowserForecastArtifactResponse {
        metadata: js_safe_json_value(json!({
            "model": forecaster.model_name(),
            "input": frame.metadata_value(),
            "modelMetadata": forecaster.metadata(),
        })),
        artifact,
    })
}

fn predict_piecewise_linear_seasonal_artifact_request(
    artifact: &str,
    horizon: usize,
    options: BrowserForecastArtifactPredictOptions,
) -> Result<BrowserForecastResponse> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    let mut forecaster = PiecewiseLinearSeasonalForecaster::from_json_string(artifact)?;
    apply_piecewise_artifact_predict_options(&mut forecaster, &options)?;
    let forecast = forecaster.predict(horizon)?;
    let components = if options.include_components {
        Some(js_safe_json_value(
            forecaster.predict_components_json_value(horizon)?,
        ))
    } else {
        None
    };
    let history_components = if options.include_history_components {
        Some(js_safe_json_value(
            forecaster.history_components_json_value()?,
        ))
    } else {
        None
    };
    let metadata = forecaster.metadata();
    let samples =
        if options.include_samples && metadata["uncertainty_samples"].as_u64().unwrap_or(0) > 0 {
            Some(js_safe_json_value(
                forecaster.predict_samples_json_value(horizon)?,
            ))
        } else {
            None
        };
    let has_quantiles = metadata["quantile_levels"]
        .as_array()
        .map(|levels| !levels.is_empty())
        .unwrap_or(false);
    let quantiles = if options.include_quantiles && has_quantiles {
        Some(js_safe_json_value(
            forecaster.predict_quantiles_json_value(horizon, None)?,
        ))
    } else {
        None
    };
    Ok(BrowserForecastResponse {
        metadata: js_safe_json_value(json!({
            "model": forecaster.model_name(),
            "modelMetadata": metadata,
        })),
        forecast: js_safe_json_value(forecast.to_json_value()),
        components,
        history_components,
        samples,
        quantiles,
    })
}

fn apply_piecewise_artifact_predict_options(
    forecaster: &mut PiecewiseLinearSeasonalForecaster,
    options: &BrowserForecastArtifactPredictOptions,
) -> Result<()> {
    forecaster.update_config(|config| {
        if let Some(future_regressors) = &options.future_regressors {
            config.future_regressors = future_regressors.clone();
        }
        if let Some(future_regressors_by_series) = &options.future_regressors_by_series {
            config.future_regressors_by_series = future_regressors_by_series.clone();
        }
        if let Some(trend_adjustments) = &options.trend_adjustments {
            config.trend_adjustments = trend_adjustments.clone();
        }
        if let Some(trend_adjustments_by_series) = &options.trend_adjustments_by_series {
            config.trend_adjustments_by_series = trend_adjustments_by_series.clone();
        }
        if let Some(levels) = &options.interval_levels {
            config.interval_levels = levels.clone();
        }
        if let Some(levels) = &options.quantile_levels {
            config.quantile_levels = levels.clone();
        }
        if let Some(samples) = options.uncertainty_samples {
            config.uncertainty_samples = samples;
        }
    })
}

fn forecast_frame_from_browser_request(
    rows: Vec<BrowserForecastRow>,
    frequency: String,
    metadata: BrowserForecastMetadata,
) -> Result<ForecastFrame> {
    if rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast request must include at least one row".to_string(),
        ));
    }
    let frequency = ForecastFrequency::parse(&frequency)?;
    let metadata = ForecastFrameMetadata {
        timestamp_col: metadata.timestamp_col,
        target_col: metadata.target_col,
        series_id_col: metadata.series_id_col,
        static_covariates: Vec::new(),
        known_future_covariates: Vec::new(),
        historical_covariates: Vec::new(),
        allow_irregular: false,
        allow_missing_targets: false,
        allow_missing_covariates: false,
    };
    let rows = rows
        .into_iter()
        .map(|row| {
            cartoboost_core::forecasting::ForecastRow::from_timestamp_str_with_covariates(
                row.series_id.unwrap_or_default(),
                &row.timestamp,
                row.target,
                row.covariates,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ForecastFrame::with_metadata(rows, frequency, metadata)
}

fn js_safe_json_value(value: Value) -> Value {
    const JS_SAFE_INTEGER_MAX: u64 = 9_007_199_254_740_991;
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(js_safe_json_value).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, js_safe_json_value(value)))
                .collect(),
        ),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                if value > JS_SAFE_INTEGER_MAX {
                    return Value::String(value.to_string());
                }
            } else if let Some(value) = number.as_i64() {
                if value.unsigned_abs() > JS_SAFE_INTEGER_MAX {
                    return Value::String(value.to_string());
                }
            }
            Value::Number(number)
        }
        other => other,
    }
}

fn serialize_json_response<T: Serialize>(
    response: &T,
    context: &str,
) -> std::result::Result<JsValue, JsValue> {
    let json = serde_json::to_string(response)
        .map_err(|error| JsValue::from_str(&format!("could not encode {context}: {error}")))?;
    js_sys::JSON::parse(&json)
        .map_err(|error| JsValue::from_str(&format!("could not parse {context}: {error:?}")))
}

fn run_graph_forecast_request(
    request: BrowserGraphForecastRequest,
) -> Result<BrowserGraphForecastResponse> {
    let adjacency = GraphStCsrAdjacency::new(
        request.frame.adjacency.indptr,
        request.frame.adjacency.indices,
        request.frame.adjacency.data,
        request.frame.node_ids.len(),
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let frame = GraphStTemporalFrame::new(
        request.frame.node_ids.clone(),
        request.frame.timestamps,
        request.frame.target,
        request.frame.covariates,
        adjacency.clone(),
        request.frame.horizon,
        request.frame.frequency.clone(),
    )
    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
    let (predictions, model_metadata) = if let Some(profile) = request.options.profile.as_deref() {
        let profile_kind = parse_browser_graph_transformer_profile(profile)?;
        let (default_lookback, default_periodicity) =
            if profile_kind == BrowserGraphTransformerProfile::LongShortFusion {
                (24 * 28, 24)
            } else {
                (3, 24)
            };
        let lookback = request.options.lookback.unwrap_or(default_lookback);
        let config = BrowserPaperGraphTransformerConfig {
            profile: profile_kind,
            lookback,
            hidden_size: request.options.hidden_size,
            attention_heads: request.options.attention_heads.unwrap_or(2),
            graph_order: request.options.graph_order.unwrap_or(2),
            experts: request.options.experts.unwrap_or(2),
            periodicity: request.options.periodicity.unwrap_or(default_periodicity),
            recent_window: request
                .options
                .recent_window
                .unwrap_or(lookback.min(24 * 7)),
            epochs: request.options.epochs,
            learning_rate: request.options.learning_rate,
            weight_decay: request.options.weight_decay.unwrap_or(1e-5),
            batch_size: request.options.batch_size,
            backend: graph_st_select_backend(Some(&request.options.backend))
                .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        };
        let mut model = BrowserPaperGraphTransformerForecaster::new(config)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        model
            .fit(&frame)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let predictions = model
            .predict(frame.horizon)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let report = model.architecture_report();
        (
            predictions,
            json!({
                "model": profile,
                "frequency": frame.frequency,
                "architectureReport": report,
            }),
        )
    } else {
        let config = GraphStDcrnnConfig {
            diffusion_steps: request.options.diffusion_steps,
            hidden_size: request.options.hidden_size,
            epochs: request.options.epochs,
            learning_rate: request.options.learning_rate,
            teacher_forcing_start: request.options.teacher_forcing_start,
            teacher_forcing_end: request.options.teacher_forcing_end,
            ridge: request.options.ridge,
            backend: graph_st_select_backend(Some(&request.options.backend))
                .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
        };
        let mut model = GraphStDcrnnForecaster::new(config.clone())
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        model
            .fit(&frame)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let predictions = model
            .predict(frame.horizon)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        (
            predictions,
            json!({
                "model": "dcrnn",
                "frequency": frame.frequency,
                "diffusionSteps": config.diffusion_steps,
                "hiddenSize": config.hidden_size,
                "epochs": config.epochs,
                "teacherForcingStart": config.teacher_forcing_start,
                "teacherForcingEnd": config.teacher_forcing_end,
            }),
        )
    };
    let metrics = request.actual.map(|actual| {
        serde_json::to_value(graph_st_metrics(
            &predictions,
            &actual,
            &frame.node_ids,
            &adjacency,
        ))
        .unwrap_or_else(|error| json!({ "error": error.to_string() }))
    });
    Ok(BrowserGraphForecastResponse {
        predictions,
        node_ids: frame.node_ids,
        horizon: frame.horizon,
        metrics,
        metadata: model_metadata,
    })
}

fn parse_browser_graph_transformer_profile(value: &str) -> Result<BrowserGraphTransformerProfile> {
    match value {
        "heterogeneous_moe" => Ok(BrowserGraphTransformerProfile::HeterogeneousMoE),
        "efficient_high_order" => Ok(BrowserGraphTransformerProfile::EfficientHighOrder),
        "long_short_fusion" => Ok(BrowserGraphTransformerProfile::LongShortFusion),
        "gated_graph_temporal" => Ok(BrowserGraphTransformerProfile::GatedGraphTemporal),
        "spatial_shift_graphon_moe" => Ok(BrowserGraphTransformerProfile::SpatialShiftGraphonMoE),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown paper graph transformer profile {other:?}"
        ))),
    }
}

