#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, NaiveDate};
    use std::collections::BTreeSet;

    #[test]
    fn forecast_model_registry_keeps_cartoboost_first_without_duplicates() {
        let registry = forecast_model_registry();
        let names = registry.iter().map(|model| model.name).collect::<Vec<_>>();
        assert_eq!(
            &names[..7],
            &[
                "auto_forecast",
                "cartoboost_lag",
                "cartoboost_direct",
                "rectified_recursive",
                "lag_plus",
                "scaled_cartoboost_lag",
                "log1p_cartoboost_lag",
            ]
        );
        let unique = names.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn browser_geotemporal_diagnostics_runs_rust_primitives() {
        let response = run_geotemporal_diagnostics_request(BrowserGeotemporalDiagnosticsRequest {
            quantiles: Some(BrowserQuantileDiagnosticsRequest {
                values: Some(vec![10.0, 9.0, 11.0]),
                actual: Some(vec![9.0, 10.0, 12.0]),
                prediction: Some(vec![8.5, 10.5, 12.5]),
                quantile: Some(0.5),
                lower: Some(vec![8.0, 9.0, 10.0]),
                upper: Some(vec![10.0, 12.0, 13.0]),
                quantile_rows: Some(vec![vec![8.0, 9.0, 10.0], vec![9.0, 8.5, 12.0]]),
            }),
            residual_correction: Some(BrowserResidualCorrectionRequest {
                process_variance: 0.05,
                observation_variance: 1.0,
                observations: vec![
                    BrowserResidualObservation {
                        key: BrowserResidualStateKey {
                            origin: Some("PU1".to_string()),
                            destination: Some("DO2".to_string()),
                            corridor: Some("PU1_DO2".to_string()),
                            ..BrowserResidualStateKey::default()
                        },
                        structural_prediction: 10.0,
                        observed: Some(12.0),
                    },
                    BrowserResidualObservation {
                        key: BrowserResidualStateKey {
                            origin: Some("PU1".to_string()),
                            destination: Some("DO2".to_string()),
                            corridor: Some("PU1_DO2".to_string()),
                            ..BrowserResidualStateKey::default()
                        },
                        structural_prediction: 11.0,
                        observed: None,
                    },
                ],
            }),
            regime: Some(BrowserRegimeDiagnosticsRequest {
                residuals: vec![0.0, 0.1, 0.0, 4.0, 4.2],
                cusum: Some(CusumConfig {
                    reference_mean: 0.0,
                    drift: 0.05,
                    threshold: 2.0,
                }),
                page_hinkley: Some(PageHinkleyConfig {
                    delta: 0.01,
                    threshold: 1.0,
                }),
                ewma: Some(EwmaVolatilityConfig { alpha: 0.5 }),
                lower: Some(vec![-1.0; 5]),
                upper: Some(vec![1.0; 5]),
                policy: Some(RegimeIntervalPolicy {
                    widening_multiplier: 0.5,
                    active_window: 2,
                }),
                rolling_window: Some(3),
            }),
            calibration: Some(BrowserCalibrationRequest {
                scores: Some(vec![-2.0, -0.5, 0.5, 2.0]),
                labels: vec![0.0, 0.0, 1.0, 1.0],
                probabilities: Some(vec![0.2, 0.4, 0.6, 0.8]),
                before_probabilities: None,
                method: Some("sigmoid".to_string()),
                bucket_count: Some(4),
                event: Some(BrowserCalibrationEventRequest {
                    kind: "failureRisk".to_string(),
                    actual: vec![1.0, 3.0, 5.0],
                    prediction: None,
                    threshold: Some(2.0),
                    horizon: None,
                    warning_threshold: None,
                    critical_threshold: None,
                }),
            }),
        })
        .expect("geotemporal diagnostics");

        assert_eq!(
            response["surface"].as_str(),
            Some("rust_geotemporal_diagnostics")
        );
        assert_eq!(
            response["quantiles"]["repairedValues"].as_array().unwrap()[1].as_f64(),
            Some(10.0)
        );
        assert_eq!(
            response["residualCorrection"]["stateCount"].as_u64(),
            Some(1)
        );
        assert!(response["regime"]["regimeAdjustedIntervals"]
            .as_array()
            .expect("intervals")
            .iter()
            .any(|row| row["confidence"].as_f64().unwrap() < 1.0));
        assert_eq!(
            response["calibration"]["eventLabels"]
                .as_array()
                .expect("event labels")
                .len(),
            3
        );
        assert!(response["calibration"]["calibratedProbabilities"].is_array());
    }

    #[test]
    fn browser_geo_feature_examples_emit_bearing_columns() {
        let response = run_geo_feature_examples_request(BrowserGeoFeatureRequest {
            planar_routes: vec![
                BrowserPlanarRoute {
                    label: "north".to_string(),
                    origin: [0.0, 0.0],
                    destination: [0.0, 2.0],
                },
                BrowserPlanarRoute {
                    label: "same".to_string(),
                    origin: [1.0, 1.0],
                    destination: [1.0, 1.0],
                },
            ],
            latlng_routes: vec![BrowserLatLngRoute {
                label: "latlng-north".to_string(),
                origin: [40.0, -73.0],
                destination: [41.0, -73.0],
            }],
            radial_points: vec![BrowserNamedPoint {
                label: "point".to_string(),
                point: [3.0, 4.0],
            }],
            anchors: vec![
                BrowserNamedPoint {
                    label: "origin".to_string(),
                    point: [0.0, 0.0],
                },
                BrowserNamedPoint {
                    label: "x-axis".to_string(),
                    point: [3.0, 0.0],
                },
            ],
            length_scale: 1.0,
            local_frame: Some(BrowserLocalFrame {
                origin: [1.0, 1.0],
                axis: [0.0, 1.0],
                points: vec![BrowserNamedPoint {
                    label: "projected".to_string(),
                    point: [2.0, 3.0],
                }],
            }),
        })
        .expect("geo feature examples");
        assert_eq!(response.planar[0].east, Some(0.0));
        assert_eq!(response.planar[0].north, Some(1.0));
        assert!(response.planar[1].zero_distance);
        assert!(response.latlng[0].east.unwrap().abs() < 1.0e-12);
        assert!((response.latlng[0].north.unwrap() - 1.0).abs() < 1.0e-12);
        assert_eq!(response.routes[0].distance, Some(2.0));
        assert_eq!(response.radial[0].values, vec![5.0, 4.0]);
        assert_eq!(response.rbf[0].values[0], (-12.5_f64).exp());
        assert_eq!(response.local_frame[0].along_axis, Some(2.0));
        assert_eq!(response.local_frame[0].cross_axis, Some(-1.0));
    }

    #[test]
    fn browser_piecewise_linear_seasonal_forecast_runs_through_dispatch() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(2),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                seasonality_l2_regularization: Some(0.001),
                weekly_l2_regularization: Some(0.002),
                fit_loss: Some("huber".to_string()),
                huber_delta: Some(1.25),
                irls_iterations: Some(4),
                include_components: Some(true),
                include_history_components: Some(true),
                include_samples: Some(true),
                include_quantiles: Some(true),
                uncertainty_samples: Some(4),
                quantile_levels: Some(vec![0.1, 0.5, 0.9]),
                coefficient_uncertainty_scale: Some(1.5),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let component_records = response
            .components
            .as_ref()
            .and_then(|components| components.get("records"))
            .and_then(Value::as_array)
            .expect("component records");
        let history_component_records = response
            .history_components
            .as_ref()
            .and_then(|components| components.get("records"))
            .and_then(Value::as_array)
            .expect("history component records");
        let sample_records = response
            .samples
            .as_ref()
            .and_then(|samples| samples.get("records"))
            .and_then(Value::as_array)
            .expect("sample records");
        let quantile_records = response
            .quantiles
            .as_ref()
            .and_then(|quantiles| quantiles.get("records"))
            .and_then(Value::as_array)
            .expect("quantile records");

        assert_eq!(records.len(), 9);
        assert_eq!(component_records.len(), 9);
        assert_eq!(history_component_records.len(), sample_panel_rows().len());
        assert_eq!(sample_records.len(), 36);
        assert_eq!(quantile_records.len(), 27);
        assert!(component_records[0]["components"]["weekly"]
            .as_f64()
            .is_some());
        assert!(history_component_records[1]["trend_movement"]
            .as_f64()
            .is_some());
        assert!(sample_records[0]["prediction"].as_f64().is_some());
        assert_eq!(quantile_records[1]["quantile"].as_f64(), Some(0.5));
        assert_eq!(
            response.metadata["model"].as_str(),
            Some("piecewise_linear_seasonal")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["weekly_l2_regularization"].as_f64(),
            Some(0.002)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["weekly_fourier_order"].as_u64(),
            Some(0)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["auto_weekly_seasonality"].as_bool(),
            Some(false)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["fit_loss"].as_str(),
            Some("huber")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["huber_delta"].as_f64(),
            Some(1.25)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["irls_iterations"].as_u64(),
            Some(4)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["coefficient_uncertainty_scale"].as_f64(),
            Some(1.5)
        );
    }

    #[test]
    fn browser_piecewise_linear_accepts_prophet_modeling_aliases() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                n_changepoints: Some(4),
                changepoint_prior_scale: Some(0.2),
                seasonality_prior_scale: Some(5.0),
                holidays_prior_scale: Some(10.0),
                seasonality_mode: Some("multiplicative".to_string()),
                holidays_mode: Some("additive".to_string()),
                holidays: Some(vec![BrowserForecastHoliday {
                    holiday: "airport_queue_surge".to_string(),
                    ds: "2026-01-03T00:00:00".to_string(),
                    lower_window: Some(-1),
                    upper_window: Some(1),
                    prior_scale: Some(2.0),
                }]),
                interval_width: Some(0.8),
                uncertainty_samples: Some(8),
                include_components: Some(true),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("piecewise prophet aliases forecast");

        assert_eq!(
            response.metadata["modelMetadata"]["changepoints"].as_u64(),
            Some(4)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["changepoint_l1_regularization"].as_f64(),
            Some(5.0)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["seasonality_l2_regularization"].as_f64(),
            Some(0.04)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["event_l2_regularization"].as_f64(),
            Some(0.01)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["event_l2_regularization_by_name"]
                ["airport_queue_surge"]
                .as_f64(),
            Some(0.25)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["component_mode"].as_str(),
            Some("multiplicative")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["event_mode"].as_str(),
            Some("additive")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["interval_levels"][0].as_f64(),
            Some(0.8)
        );
        assert_eq!(
            response.metadata["modelMetadata"]["events"][0]["name"].as_str(),
            Some("airport_queue_surge")
        );
    }

    #[test]
    fn browser_piecewise_linear_rejects_unsupported_prophet_mcmc_alias() {
        let err = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                mcmc_samples: Some(100),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect_err("mcmc alias should fail clearly");

        assert!(err.to_string().contains("mcmc_samples"));
    }

    #[test]
    fn browser_piecewise_linear_omits_unused_sample_payload() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(2),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                uncertainty_samples: Some(8),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("lean piecewise forecast");

        assert!(response.components.is_none());
        assert!(response.samples.is_none());
        assert_eq!(
            response.metadata["modelMetadata"]["uncertainty_samples"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn browser_piecewise_linear_trend_adjustment_and_shock_options_flow_through_dispatch() {
        let base_options = || BrowserForecastOptions {
            changepoints: Some(0),
            weekly_fourier_order: Some(0),
            auto_weekly_seasonality: Some(false),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let trend_adjustments = BTreeMap::from([(2, 1.10)]);
        let baseline = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: base_options(),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("baseline piecewise forecast");
        let trend_adjusted = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                trend_adjustments: Some(trend_adjustments.clone()),
                ..base_options()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("trend-adjusted piecewise forecast");
        let shock_adjusted = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                trend_adjustments: Some(trend_adjustments.clone()),
                residual_shock_window: Some(2),
                residual_shock_scale: Some(0.5),
                residual_shock_decay: Some(0.8),
                ..base_options()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("shock-adjusted piecewise forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: sample_panel_rows(),
                frequency: "daily".to_string(),
                horizon: 2,
                model: "piecewise_linear_seasonal".to_string(),
                options: base_options(),
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit piecewise artifact");
        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            2,
            BrowserForecastArtifactPredictOptions {
                trend_adjustments: Some(trend_adjustments),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact trend-adjusted forecast");

        let baseline_records = baseline.forecast["records"].as_array().expect("records");
        let trend_adjusted_records = trend_adjusted.forecast["records"]
            .as_array()
            .expect("records");
        let shock_adjusted_records = shock_adjusted.forecast["records"]
            .as_array()
            .expect("records");
        let restored_records = restored.forecast["records"].as_array().expect("records");
        assert!(
            trend_adjusted_records[1]["prediction"].as_f64().unwrap()
                > baseline_records[1]["prediction"].as_f64().unwrap()
        );
        assert!(
            shock_adjusted_records[0]["prediction"].as_f64().unwrap()
                != trend_adjusted_records[0]["prediction"].as_f64().unwrap()
        );
        assert_eq!(trend_adjusted.forecast, restored.forecast);
        assert_eq!(
            shock_adjusted.metadata["modelMetadata"]["trend_adjustments"]["2"].as_f64(),
            Some(1.10)
        );
        assert_eq!(
            shock_adjusted.metadata["modelMetadata"]["residual_shock_window"].as_u64(),
            Some(2)
        );
        assert_eq!(
            shock_adjusted.components.as_ref().expect("components")["records"][1]
                ["trend_adjustment_multiplier"]
                .as_f64(),
            Some(1.10)
        );
        assert_eq!(
            restored_records[1]["prediction"].as_f64(),
            trend_adjusted_records[1]["prediction"].as_f64()
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predicts_without_refit() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: 50.0 + f64::from(day) + 20.0 * queue,
                        covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = || BrowserForecastOptions {
            changepoints: Some(1),
            weekly_fourier_order: Some(0),
            interval_levels: Some(vec![0.8]),
            extra_regressors: Some(vec!["airport_queue".to_string()]),
            future_regressors: Some(BTreeMap::from([(
                "airport_queue".to_string(),
                vec![1.0, 0.0, 0.0],
            )])),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 3,
                model: "piecewise_linear_seasonal".to_string(),
                options: options(),
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit artifact");
        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            3,
            BrowserForecastArtifactPredictOptions::default(),
        )
        .expect("artifact forecast");

        assert_eq!(direct.forecast, restored.forecast);
        let direct_queue = direct.components.as_ref().expect("direct components")["records"][0]
            ["components"]["regressors"]["airport_queue"]
            .as_f64()
            .expect("direct airport queue contribution");
        let restored_queue = restored.components.as_ref().expect("artifact components")["records"]
            [0]["components"]["regressors"]["airport_queue"]
            .as_f64()
            .expect("airport queue contribution");
        assert!(restored_queue > 10.0);
        assert!((direct_queue - restored_queue).abs() < 1.0e-9);
        assert_eq!(
            serde_json::from_str::<Value>(&artifact_response.artifact).expect("artifact")["kind"]
                .as_str(),
            Some("cartoboost_piecewise_linear_seasonal")
        );
        assert_eq!(
            artifact_response.metadata["model"].as_str(),
            Some("piecewise_linear_seasonal")
        );

        let lean_restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            3,
            BrowserForecastArtifactPredictOptions {
                include_components: false,
                include_samples: false,
                include_quantiles: false,
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("lean artifact forecast");
        assert_eq!(direct.forecast, lean_restored.forecast);
        assert!(lean_restored.components.is_none());
        assert!(lean_restored.samples.is_none());
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_quantile_overrides() {
        let rows = || {
            (1..=28)
                .map(|day| BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 40.0 + f64::from(day) + if day % 7 == 0 { 4.0 } else { -1.0 },
                    covariates: BTreeMap::new(),
                })
                .collect::<Vec<_>>()
        };
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 2,
                model: "piecewise_linear_seasonal".to_string(),
                options: BrowserForecastOptions {
                    changepoints: Some(2),
                    weekly_fourier_order: Some(0),
                    auto_weekly_seasonality: Some(false),
                    uncertainty_samples: Some(24),
                    include_quantiles: Some(false),
                    ..BrowserForecastOptions::default()
                },
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit artifact without default quantiles");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            2,
            BrowserForecastArtifactPredictOptions {
                quantile_levels: Some(vec![0.1, 0.5, 0.9]),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact forecast with quantile override");
        let quantiles = restored.quantiles.expect("quantile payload");

        assert_eq!(
            restored.metadata["modelMetadata"]["quantile_levels"],
            json!([0.1, 0.5, 0.9])
        );
        assert_eq!(quantiles["quantile_levels"], json!([0.1, 0.5, 0.9]));
        assert_eq!(
            quantiles["records"]
                .as_array()
                .expect("quantile records")
                .len(),
            2 * 3
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_future_regressor_options() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: 50.0 + f64::from(day) + 20.0 * queue,
                        covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let fit_options = || BrowserForecastOptions {
            changepoints: Some(1),
            weekly_fourier_order: Some(0),
            extra_regressors: Some(vec!["airport_queue".to_string()]),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let future_regressors =
            BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]);
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                future_regressors: Some(future_regressors.clone()),
                ..fit_options()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 3,
                model: "piecewise_linear_seasonal".to_string(),
                options: fit_options(),
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit artifact without future values");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            3,
            BrowserForecastArtifactPredictOptions {
                future_regressors: Some(future_regressors),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact forecast with future values");

        assert_eq!(direct.forecast, restored.forecast);
        assert_eq!(
            restored.metadata["modelMetadata"]["future_regressors"]["airport_queue"][0].as_f64(),
            Some(1.0)
        );
        assert!(
            restored.components.as_ref().expect("components")["records"][0]["components"]
                ["regressors"]["airport_queue"]
                .as_f64()
                .expect("future regressor contribution")
                > 10.0
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_series_future_caps() {
        let rows = || {
            ["pickup_zone_a", "pickup_zone_b"]
                .into_iter()
                .flat_map(|series_id| {
                    (1..=28).map(move |day| {
                        let cap = if series_id == "pickup_zone_a" {
                            110.0 + 0.25 * f64::from(day)
                        } else {
                            65.0 + 0.10 * f64::from(day)
                        };
                        let t = f64::from(day) - 14.0;
                        BrowserForecastRow {
                            series_id: Some(series_id.to_string()),
                            timestamp: format!("2026-01-{day:02}T00:00:00"),
                            target: cap / (1.0 + (-0.18 * t).exp()),
                            covariates: BTreeMap::from([("zone_capacity".to_string(), cap)]),
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        let fit_options = BrowserForecastOptions {
            growth: Some("logistic".to_string()),
            changepoints: Some(3),
            weekly_fourier_order: Some(0),
            auto_weekly_seasonality: Some(false),
            cap_regressor: Some("zone_capacity".to_string()),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let future_regressors_by_series = BTreeMap::from([
            (
                "pickup_zone_a".to_string(),
                BTreeMap::from([("zone_capacity".to_string(), vec![120.0])]),
            ),
            (
                "pickup_zone_b".to_string(),
                BTreeMap::from([("zone_capacity".to_string(), vec![70.0])]),
            ),
        ]);
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("logistic".to_string()),
                changepoints: Some(3),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                cap_regressor: Some("zone_capacity".to_string()),
                include_components: Some(true),
                future_regressors_by_series: Some(future_regressors_by_series.clone()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct panel logistic forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 1,
                model: "piecewise_linear_seasonal".to_string(),
                options: fit_options,
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit panel logistic artifact without future caps");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            1,
            BrowserForecastArtifactPredictOptions {
                future_regressors_by_series: Some(future_regressors_by_series),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact panel logistic forecast with future caps");
        let records = restored
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let prediction_a = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_a"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone A prediction");
        let prediction_b = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_b"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone B prediction");

        assert_eq!(direct.forecast, restored.forecast);
        assert!(prediction_a > 0.0 && prediction_a < 120.0);
        assert!(prediction_b > 0.0 && prediction_b < 70.0);
        assert!(prediction_a > prediction_b + 20.0);
        assert_eq!(
            restored.metadata["modelMetadata"]["future_regressors_by_series"]["pickup_zone_a"]
                ["zone_capacity"][0]
                .as_f64(),
            Some(120.0)
        );
    }

    #[test]
    fn browser_piecewise_linear_artifact_predict_accepts_series_future_floors() {
        let cap = 140.0;
        let rows = || {
            ["pickup_zone_a", "pickup_zone_b"]
                .into_iter()
                .flat_map(|series_id| {
                    (1..=28).map(move |day| {
                        let floor = if series_id == "pickup_zone_a" {
                            32.0 + 0.10 * f64::from(day)
                        } else {
                            8.0 + 0.05 * f64::from(day)
                        };
                        let t = f64::from(day) - 14.0;
                        BrowserForecastRow {
                            series_id: Some(series_id.to_string()),
                            timestamp: format!("2026-01-{day:02}T00:00:00"),
                            target: floor + (cap - floor) / (1.0 + (-0.18 * t).exp()),
                            covariates: BTreeMap::from([("service_floor".to_string(), floor)]),
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        let fit_options = BrowserForecastOptions {
            growth: Some("logistic".to_string()),
            changepoints: Some(3),
            weekly_fourier_order: Some(0),
            auto_weekly_seasonality: Some(false),
            cap: Some(cap),
            floor_regressor: Some("service_floor".to_string()),
            include_components: Some(true),
            ..BrowserForecastOptions::default()
        };
        let future_regressors_by_series = BTreeMap::from([
            (
                "pickup_zone_a".to_string(),
                BTreeMap::from([("service_floor".to_string(), vec![38.0])]),
            ),
            (
                "pickup_zone_b".to_string(),
                BTreeMap::from([("service_floor".to_string(), vec![10.0])]),
            ),
        ]);
        let direct = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("logistic".to_string()),
                changepoints: Some(3),
                weekly_fourier_order: Some(0),
                auto_weekly_seasonality: Some(false),
                cap: Some(cap),
                floor_regressor: Some("service_floor".to_string()),
                include_components: Some(true),
                future_regressors_by_series: Some(future_regressors_by_series.clone()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("direct panel logistic forecast");
        let artifact_response =
            fit_piecewise_linear_seasonal_artifact_request(BrowserForecastRequest {
                rows: rows(),
                frequency: "daily".to_string(),
                horizon: 1,
                model: "piecewise_linear_seasonal".to_string(),
                options: fit_options,
                metadata: BrowserForecastMetadata::default(),
            })
            .expect("fit panel logistic artifact without future floors");

        let restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            1,
            BrowserForecastArtifactPredictOptions {
                future_regressors_by_series: Some(future_regressors_by_series),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact panel logistic forecast with future floors");
        let lower_floor_restored = predict_piecewise_linear_seasonal_artifact_request(
            &artifact_response.artifact,
            1,
            BrowserForecastArtifactPredictOptions {
                future_regressors_by_series: Some(BTreeMap::from([
                    (
                        "pickup_zone_a".to_string(),
                        BTreeMap::from([("service_floor".to_string(), vec![5.0])]),
                    ),
                    (
                        "pickup_zone_b".to_string(),
                        BTreeMap::from([("service_floor".to_string(), vec![10.0])]),
                    ),
                ])),
                ..BrowserForecastArtifactPredictOptions::default()
            },
        )
        .expect("artifact panel logistic forecast with lower future floor");
        let records = restored
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let prediction_a = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_a"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone A prediction");
        let prediction_b = records
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_b"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone B prediction");
        let lower_floor_prediction_a = lower_floor_restored
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("lower floor forecast records")
            .iter()
            .find(|record| record["series_id"].as_str() == Some("pickup_zone_a"))
            .and_then(|record| record["prediction"].as_f64())
            .expect("zone A lower floor prediction");

        assert_eq!(direct.forecast, restored.forecast);
        assert!(prediction_a > 38.0 && prediction_a < cap);
        assert!(prediction_b > 10.0 && prediction_b < cap);
        assert!(prediction_a > lower_floor_prediction_a);
        assert_eq!(
            restored.metadata["modelMetadata"]["future_regressors_by_series"]["pickup_zone_a"]
                ["service_floor"][0]
                .as_f64(),
            Some(38.0)
        );
    }

    #[test]
    fn browser_piecewise_linear_flat_growth_flows_through_dispatch() {
        let rows = (1..=28)
            .map(|day| BrowserForecastRow {
                series_id: Some("pickup_zone_1".to_string()),
                timestamp: format!("2026-01-{day:02}T00:00:00"),
                target: 40.0 + 2.0 * f64::from(day),
                covariates: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("flat".to_string()),
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("flat piecewise seasonal forecast");

        assert_eq!(
            response.metadata["modelMetadata"]["growth"].as_str(),
            Some("flat")
        );
    }

    #[test]
    fn browser_piecewise_linear_logistic_growth_uses_cap_floor_options() {
        let rows = (0..28)
            .map(|idx| {
                let t = idx as f64 - 14.0;
                let cap = 95.0 + idx as f64;
                let target = cap / (1.0 + (-0.25 * t).exp());
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{:02}T00:00:00", idx + 1),
                    target,
                    covariates: BTreeMap::from([("zone_capacity".to_string(), cap)]),
                }
            })
            .collect::<Vec<_>>();
        let future_caps = vec![123.0, 124.0, 125.0, 126.0];
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 4,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                growth: Some("logistic".to_string()),
                floor: Some(0.0),
                cap_regressor: Some("zone_capacity".to_string()),
                future_regressors: Some(BTreeMap::from([(
                    "zone_capacity".to_string(),
                    future_caps.clone(),
                )])),
                changepoints: Some(4),
                weekly_fourier_order: Some(0),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("logistic piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");

        assert_eq!(
            response.metadata["modelMetadata"]["growth"].as_str(),
            Some("logistic")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["cap_regressor"].as_str(),
            Some("zone_capacity")
        );
        assert!(records.iter().zip(future_caps.iter()).all(|(record, cap)| {
            let prediction = record["prediction"].as_f64().expect("prediction");
            prediction > 0.0 && prediction < *cap
        }));
    }

    #[test]
    fn browser_piecewise_linear_explicit_changepoints_flow_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let target = if day <= 15 {
                    50.0 + f64::from(day)
                } else {
                    65.0 + 5.0 * f64::from(day - 15)
                };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                changepoint_range: Some(0.8),
                changepoint_timestamps: Some(vec!["2026-01-15T00:00:00".to_string()]),
                weekly_fourier_order: Some(0),
                changepoint_l2_regularization: Some(0.001),
                changepoint_l1_regularization: Some(0.01),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("explicit changepoint piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert_eq!(
            response.metadata["modelMetadata"]["changepoint_timestamps"][0].as_str(),
            Some("2026-01-15T00:00:00")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["changepoint_l1_regularization"].as_f64(),
            Some(0.01)
        );
        assert!(records[2]["prediction"].as_f64().expect("prediction") > 140.0);
    }

    #[test]
    fn browser_piecewise_linear_events_flow_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let event_boost = if (14..=16).contains(&day) { 25.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 100.0 + 0.5 * f64::from(day) + event_boost,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                event_l2_regularization: Some(0.001),
                include_components: Some(true),
                events: Some(vec![
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-01-15T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-02-01T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                ]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("event piecewise seasonal forecast");
        let events = response.metadata["modelMetadata"]["events"]
            .as_array()
            .expect("events metadata");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        let component_records = response
            .components
            .as_ref()
            .and_then(|components| components.get("records"))
            .and_then(Value::as_array)
            .expect("component records");

        assert_eq!(events[0]["name"].as_str(), Some("airport_surge"));
        assert!(records[1]["prediction"].as_f64().expect("prediction") > 120.0);
        assert!(
            component_records[0]["components"]["event_window_offsets"]["airport_surge[-1]"]
                .as_f64()
                .is_some()
        );
    }

    #[test]
    fn browser_piecewise_linear_extra_regressors_use_future_values() {
        let rows = (1..=30)
            .map(|day| {
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 50.0 + f64::from(day) + 20.0 * queue,
                    covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                regressor_l2_regularization: Some(0.001),
                extra_regressors: Some(vec!["airport_queue".to_string()]),
                future_regressors: Some(BTreeMap::from([(
                    "airport_queue".to_string(),
                    vec![1.0, 0.0, 0.0],
                )])),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("regressor piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");

        assert_eq!(
            response.metadata["modelMetadata"]["extra_regressors"][0].as_str(),
            Some("airport_queue")
        );
        assert!(records[0]["prediction"].as_f64().expect("prediction") > 80.0);
    }

    #[test]
    fn browser_neural_panel_custom_seasonality_flows_through_dispatch() {
        let start = NaiveDate::from_ymd_opt(2026, 3, 1)
            .expect("valid start date")
            .and_hms_opt(0, 0, 0)
            .expect("valid start time");
        let rows = (1..=32)
            .map(|day| {
                let phase = std::f64::consts::TAU * f64::from(day % 8) / 8.0;
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: (start + Duration::days((day - 1) as i64))
                        .format("%Y-%m-%dT%H:%M:%S")
                        .to_string(),
                    target: 50.0 + 8.0 * phase.sin(),
                    covariates: BTreeMap::from([("rushHour".to_string(), 1.0)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "neural_panel".to_string(),
            options: BrowserForecastOptions {
                n_lags: Some(4),
                n_forecasts: Some(3),
                custom_seasonalities: Some(vec![BrowserForecastSeasonality {
                    name: "taxi_cycle".to_string(),
                    period_days: 8.0,
                    fourier_order: 2,
                    mode: Some("additive".to_string()),
                    condition_name: Some("rushHour".to_string()),
                    l2_regularization: None,
                }]),
                include_components: Some(true),
                include_history_components: Some(true),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("neural panel forecast");
        let feature_schema = response.metadata["modelMetadata"]["feature_schema"]
            .as_array()
            .expect("feature schema");
        let custom_seasonalities = response.metadata["modelMetadata"]["config"]
            ["custom_seasonalities"]
            .as_object()
            .expect("custom seasonalities");
        let custom_seasonality_conditions = response.metadata["modelMetadata"]["config"]
            ["custom_seasonality_conditions"]
            .as_object()
            .expect("custom seasonality conditions");

        assert!(feature_schema
            .iter()
            .any(|value| value.as_str() == Some("seasonality:taxi_cycle:sin:1")));
        assert_eq!(custom_seasonalities["taxi_cycle"][0].as_f64(), Some(192.0));
        assert_eq!(
            custom_seasonality_conditions["taxi_cycle"].as_str(),
            Some("rushHour")
        );
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")
            .iter()
            .all(|record| record["prediction"]
                .as_f64()
                .expect("prediction")
                .is_finite()));
        let component_records = response
            .components
            .as_ref()
            .and_then(|components| components.get("series"))
            .and_then(|series| series.get("pickup_zone_1"))
            .and_then(Value::as_array)
            .expect("component records");
        let history_records = response
            .history_components
            .as_ref()
            .and_then(|components| components.get("series"))
            .and_then(|series| series.get("pickup_zone_1"))
            .and_then(Value::as_array)
            .expect("history records");
        assert_eq!(component_records.len(), 3);
        assert_eq!(history_records.len(), 32);
        assert!(component_records[0]["prediction"]
            .as_f64()
            .expect("component prediction")
            .is_finite());
        assert!(history_records[0]["prediction"]
            .as_f64()
            .expect("history prediction")
            .is_finite());
    }

    #[test]
    fn browser_nbeats_forecast_runs_through_generic_dispatch() {
        let rows = (1..=18)
            .map(|day| BrowserForecastRow {
                series_id: Some("pickup_zone_237".to_string()),
                timestamp: format!("2026-02-{day:02}T00:00:00"),
                target: 30.0 + f64::from(day) + f64::from(day % 3),
                covariates: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "nbeats".to_string(),
            options: BrowserForecastOptions {
                input_size: Some(5),
                hidden_size: Some(8),
                epochs: Some(6),
                learning_rate: Some(0.02),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("nbeats forecast");

        assert_eq!(response.metadata["model"].as_str(), Some("nbeats"));
        assert_eq!(
            response.metadata["modelMetadata"]["input_size"].as_u64(),
            Some(5)
        );
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")
            .iter()
            .all(|record| record["prediction"]
                .as_f64()
                .expect("prediction")
                .is_finite()));
    }

    #[test]
    fn browser_nhits_forecast_runs_through_generic_dispatch() {
        let rows = (1..=22)
            .map(|day| BrowserForecastRow {
                series_id: Some("pickup_zone_161".to_string()),
                timestamp: format!("2026-04-{day:02}T00:00:00"),
                target: 45.0 + 2.0 * f64::from(day % 4) + 0.5 * f64::from(day),
                covariates: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 4,
            model: "nhits".to_string(),
            options: BrowserForecastOptions {
                input_size: Some(6),
                hidden_size: Some(10),
                pooling_size: Some(3),
                epochs: Some(6),
                learning_rate: Some(0.02),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("nhits forecast");

        assert_eq!(response.metadata["model"].as_str(), Some("nhits"));
        assert_eq!(
            response.metadata["modelMetadata"]["pooling_size"].as_u64(),
            Some(3)
        );
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")
            .iter()
            .all(|record| record["prediction"]
                .as_f64()
                .expect("prediction")
                .is_finite()));
    }

    #[test]
    fn browser_neural_panel_holidays_and_regressors_flow_through_dispatch() {
        let rows = (1..=16)
            .map(|day| {
                let queue = if day % 4 == 0 { 1.0 } else { 0.0 };
                let holiday = if day == 6 { 1.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 40.0 + f64::from(day) + 9.0 * queue + 14.0 * holiday,
                    covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 2,
            model: "neural_panel".to_string(),
            options: BrowserForecastOptions {
                n_lags: Some(4),
                n_forecasts: Some(2),
                weekly_fourier_order: Some(0),
                extra_regressors: Some(vec!["airport_queue".to_string()]),
                regressor_modes: Some(BTreeMap::from([(
                    "airport_queue".to_string(),
                    "additive".to_string(),
                )])),
                future_regressors: Some(BTreeMap::from([(
                    "airport_queue".to_string(),
                    vec![0.0, 1.0],
                )])),
                holidays: Some(vec![BrowserForecastHoliday {
                    holiday: "airport_holiday".to_string(),
                    ds: "2026-01-06T00:00:00".to_string(),
                    lower_window: Some(0),
                    upper_window: Some(0),
                    prior_scale: Some(10.0),
                }]),
                holidays_mode: Some("additive".to_string()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("neural panel forecast");
        let feature_schema = response.metadata["modelMetadata"]["feature_schema"]
            .as_array()
            .expect("feature schema");

        assert!(feature_schema
            .iter()
            .any(|value| value.as_str() == Some("airport_queue")));
        assert!(feature_schema
            .iter()
            .any(|value| value.as_str() == Some("event:airport_holiday:0")));
        assert!(response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records")[0]["prediction"]
            .as_f64()
            .expect("prediction")
            .is_finite());
    }

    #[test]
    fn browser_piecewise_linear_regressor_standardization_flows_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let traffic_index = 100.0 + 4.0 * f64::from(day);
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 40.0 + f64::from(day) + 1.5 * traffic_index,
                    covariates: BTreeMap::from([("trafficIndex".to_string(), traffic_index)]),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                extra_regressors: Some(vec!["trafficIndex".to_string()]),
                regressor_standardization: Some("none".to_string()),
                future_regressors: Some(BTreeMap::from([(
                    "trafficIndex".to_string(),
                    vec![224.0, 228.0],
                )])),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("standardization piecewise seasonal forecast");

        assert_eq!(
            response.metadata["modelMetadata"]["regressor_standardization"].as_str(),
            Some("none")
        );
    }

    #[test]
    fn browser_piecewise_linear_named_regressor_l2_flows_through_dispatch() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: 50.0 + f64::from(day) + 24.0 * queue,
                        covariates: BTreeMap::from([("airport_queue".to_string(), queue)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = |named_l2: f64| BrowserForecastOptions {
            changepoints: Some(0),
            weekly_fourier_order: Some(0),
            extra_regressors: Some(vec!["airport_queue".to_string()]),
            regressor_l2_regularization_by_name: Some(BTreeMap::from([(
                "airport_queue".to_string(),
                named_l2,
            )])),
            future_regressors: Some(BTreeMap::from([("airport_queue".to_string(), vec![1.0])])),
            ..BrowserForecastOptions::default()
        };
        let low_l2_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(0.001),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("low l2 forecast");
        let high_l2_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 1,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(1_000.0),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("high l2 forecast");
        let low_prediction = low_l2_response.forecast["records"][0]["prediction"]
            .as_f64()
            .expect("low l2 prediction");
        let high_prediction = high_l2_response.forecast["records"][0]["prediction"]
            .as_f64()
            .expect("high l2 prediction");

        assert!(low_prediction > high_prediction + 10.0);
        assert_eq!(
            high_l2_response.metadata["modelMetadata"]["regressor_l2_regularization_by_name"]
                ["airport_queue"]
                .as_f64(),
            Some(1_000.0)
        );
    }

    #[test]
    fn browser_piecewise_linear_custom_seasonalities_flow_through_dispatch() {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid start");
        let rows = (1..=56)
            .map(|day| {
                let timestamp = start + Duration::days(i64::from(day - 1));
                let biweekly = if day % 14 == 0 { 18.0 } else { 0.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    target: 80.0 + 0.25 * f64::from(day) + biweekly,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 14,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                seasonality_l2_regularization: Some(0.001),
                custom_seasonalities: Some(vec![BrowserForecastSeasonality {
                    name: "biweekly_pickup_cycle".to_string(),
                    period_days: 14.0,
                    fourier_order: 4,
                    mode: Some("additive".to_string()),
                    condition_name: None,
                    l2_regularization: None,
                }]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("custom seasonality piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert_eq!(
            response.metadata["modelMetadata"]["custom_seasonalities"][0]["name"].as_str(),
            Some("biweekly_pickup_cycle")
        );
        assert_eq!(
            response.metadata["modelMetadata"]["custom_seasonalities"][0]["mode"].as_str(),
            Some("additive")
        );
        assert!(records[13]["prediction"].as_f64().expect("prediction") > 95.0);
    }

    #[test]
    fn browser_piecewise_linear_conditional_seasonality_uses_future_flags() {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid start");
        let rows = || {
            (1..=42)
                .map(|day| {
                    let timestamp = start + Duration::days(i64::from(day - 1));
                    let rush_hour = if day % 2 == 0 { 1.0 } else { 0.0 };
                    let cycle = if day % 7 == 0 { 16.0 } else { 0.0 };
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                        target: 80.0 + 0.2 * f64::from(day) + rush_hour * cycle,
                        covariates: BTreeMap::from([("rushHour".to_string(), rush_hour)]),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = |future_flags: Vec<f64>| BrowserForecastOptions {
            changepoints: Some(0),
            weekly_fourier_order: Some(0),
            seasonality_l2_regularization: Some(0.001),
            custom_seasonalities: Some(vec![BrowserForecastSeasonality {
                name: "rush_hour_weekly".to_string(),
                period_days: 7.0,
                fourier_order: 3,
                mode: None,
                condition_name: Some("rushHour".to_string()),
                l2_regularization: None,
            }]),
            future_regressors: Some(BTreeMap::from([("rushHour".to_string(), future_flags)])),
            ..BrowserForecastOptions::default()
        };
        let inactive_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 7,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("inactive conditional seasonality forecast");
        let active_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 7,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("active conditional seasonality forecast");
        let inactive_records = inactive_response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("inactive records");
        let active_records = active_response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("active records");

        assert_eq!(
            active_response.metadata["modelMetadata"]["custom_seasonalities"][0]["condition_name"]
                .as_str(),
            Some("rushHour")
        );
        assert!(
            active_records[6]["prediction"].as_f64().expect("active")
                > inactive_records[6]["prediction"]
                    .as_f64()
                    .expect("inactive")
                    + 4.0
        );
    }

    #[test]
    fn browser_piecewise_linear_interval_levels_render_bounds() {
        let rows = (1..=20)
            .map(|day| {
                let noise = if day % 2 == 0 { 2.0 } else { -2.0 };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target: 25.0 + f64::from(day) + noise,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 2,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                interval_levels: Some(vec![0.8]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("interval piecewise seasonal forecast");
        let columns = response
            .forecast
            .get("columns")
            .and_then(Value::as_array)
            .expect("columns");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert!(columns
            .iter()
            .any(|column| column.as_str() == Some("prediction_lower_p80")));
        assert!(records[0]["prediction_lower_p80"].as_f64().is_some());
        assert!(
            records[0]["prediction_lower_p80"].as_f64().unwrap()
                <= records[0]["prediction_upper_p80"].as_f64().unwrap()
        );
    }

    #[test]
    fn browser_piecewise_linear_uncertainty_samples_widen_intervals() {
        let rows = || {
            (1..=30)
                .map(|day| {
                    let value =
                        20.0 + 0.5 * f64::from(day) + 3.0 * (f64::from(day) - 15.0).max(0.0);
                    BrowserForecastRow {
                        series_id: Some("pickup_zone_1".to_string()),
                        timestamp: format!("2026-01-{day:02}T00:00:00"),
                        target: value,
                        covariates: BTreeMap::new(),
                    }
                })
                .collect::<Vec<_>>()
        };
        let options = |uncertainty_samples: usize| BrowserForecastOptions {
            changepoints: Some(1),
            changepoint_timestamps: Some(vec!["2026-01-15T00:00:00".to_string()]),
            changepoint_l2_regularization: Some(0.001),
            weekly_fourier_order: Some(0),
            interval_levels: Some(vec![0.8]),
            uncertainty_samples: Some(uncertainty_samples),
            trend_uncertainty_policy: Some("normal".to_string()),
            trend_uncertainty_scale: Some(1.0),
            uncertainty_seed: Some(7),
            ..BrowserForecastOptions::default()
        };
        let residual_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 5,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(0),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("residual interval forecast");
        let uncertain_response = run_forecast_request(BrowserForecastRequest {
            rows: rows(),
            frequency: "daily".to_string(),
            horizon: 5,
            model: "piecewise_linear_seasonal".to_string(),
            options: options(256),
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("uncertain interval forecast");
        let residual_record = &residual_response.forecast["records"][4];
        let uncertain_record = &uncertain_response.forecast["records"][4];
        let residual_width = residual_record["prediction_upper_p80"]
            .as_f64()
            .expect("residual upper")
            - residual_record["prediction_lower_p80"]
                .as_f64()
                .expect("residual lower");
        let uncertain_width = uncertain_record["prediction_upper_p80"]
            .as_f64()
            .expect("uncertain upper")
            - uncertain_record["prediction_lower_p80"]
                .as_f64()
                .expect("uncertain lower");

        assert!(uncertain_width > residual_width + 1.0);
        assert_eq!(
            uncertain_response.metadata["modelMetadata"]["uncertainty_samples"].as_u64(),
            Some(256)
        );
        assert_eq!(
            uncertain_response.metadata["modelMetadata"]["trend_uncertainty_policy"].as_str(),
            Some("normal")
        );
    }

    #[test]
    fn browser_piecewise_linear_multiplicative_mode_flows_through_dispatch() {
        let rows = (1..=30)
            .map(|day| {
                let trend = 20.0 + 2.0 * f64::from(day);
                let target = if (14..=16).contains(&day) {
                    trend * 1.5
                } else {
                    trend
                };
                BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: format!("2026-01-{day:02}T00:00:00"),
                    target,
                    covariates: BTreeMap::new(),
                }
            })
            .collect::<Vec<_>>();
        let response = run_forecast_request(BrowserForecastRequest {
            rows,
            frequency: "daily".to_string(),
            horizon: 3,
            model: "piecewise_linear_seasonal".to_string(),
            options: BrowserForecastOptions {
                component_mode: Some("multiplicative".to_string()),
                changepoints: Some(0),
                weekly_fourier_order: Some(0),
                event_l2_regularization: Some(0.001),
                events: Some(vec![
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-01-15T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                    BrowserForecastEvent {
                        name: "airport_surge".to_string(),
                        timestamp: "2026-02-01T00:00:00".to_string(),
                        lower_window: Some(-1),
                        upper_window: Some(1),
                    },
                ]),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata::default(),
        })
        .expect("multiplicative piecewise seasonal forecast");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("records");

        assert_eq!(
            response.metadata["modelMetadata"]["component_mode"].as_str(),
            Some("multiplicative")
        );
        assert!(records[1]["prediction"].as_f64().expect("prediction") > 100.0);
    }

    #[test]
    fn browser_sequence_viterbi_runs_through_wasm_dispatch() {
        let response = run_sequence_request(BrowserSequenceRequest {
            operation: "reference_path_viterbi".to_string(),
            frame: None,
            series: Some(sample_sequence_series()),
            reference: Some(ReferenceSignal {
                axis: vec![0.0, 1.0, 2.0, 3.0],
                signal: vec![0.0, 1.0, 2.0, 3.0],
            }),
            state_space_config: None,
            reference_path_config: Some(ReferencePathConfig::default()),
            candidates: None,
            weights: None,
            actuals: None,
            oof_fold: None,
            oof_rows: None,
            group_predictions: None,
        })
        .expect("sequence request");
        let points = response
            .get("points")
            .and_then(Value::as_array)
            .expect("path points");
        assert_eq!(points.len(), 4);
        assert_eq!(points[1]["axis"].as_f64(), Some(1.0));
    }

    #[test]
    fn browser_sequence_oof_generation_runs_through_wasm_dispatch() {
        let response = run_sequence_request(BrowserSequenceRequest {
            operation: "generate_group_oof_candidate_rows".to_string(),
            frame: None,
            series: None,
            reference: None,
            state_space_config: None,
            reference_path_config: None,
            candidates: None,
            weights: None,
            actuals: None,
            oof_fold: Some(SequenceOofFold {
                validation_group_id: "pickup_zone_1".to_string(),
                train_group_ids: vec!["pickup_zone_2".to_string()],
                actuals: vec![SequenceCandidatePrediction {
                    series_id: "pickup_zone_1".to_string(),
                    row_id: "hour_01".to_string(),
                    value: 10.0,
                }],
                candidates: vec![SequenceCandidate {
                    name: "candidate_a".to_string(),
                    predictions: vec![SequenceCandidatePrediction {
                        series_id: "pickup_zone_1".to_string(),
                        row_id: "hour_01".to_string(),
                        value: 11.0,
                    }],
                }],
            }),
            oof_rows: None,
            group_predictions: None,
        })
        .expect("sequence OOF request");
        let rows = response.as_array().expect("OOF rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0]["candidate_predictions"]["candidate_a"].as_f64(),
            Some(11.0)
        );
    }

    #[test]
    fn every_registered_browser_forecast_model_runs_on_representative_panel() {
        for model in forecast_model_registry() {
            let request = BrowserForecastRequest {
                rows: sample_panel_rows(),
                frequency: "daily".to_string(),
                horizon: 7,
                model: model.name.to_string(),
                options: BrowserForecastOptions {
                    season_length: Some(7),
                    coordinate_x: Some("longitude".to_string()),
                    coordinate_y: Some("latitude".to_string()),
                    ..BrowserForecastOptions::default()
                },
                metadata: BrowserForecastMetadata {
                    timestamp_col: Some("timestamp".to_string()),
                    target_col: Some("target".to_string()),
                    series_id_col: Some("series_id".to_string()),
                },
            };
            let response = run_forecast_request(request)
                .unwrap_or_else(|error| panic!("{} failed: {error}", model.name));
            let records = response
                .forecast
                .get("records")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("{} returned no forecast records", model.name));
            assert_eq!(records.len(), 21, "{} record count", model.name);
            assert!(
                response.metadata.get("warning").is_none(),
                "{} used fallback instead of fitting directly: {}",
                model.name,
                response.metadata
            );
            assert!(
                records
                    .iter()
                    .all(|record| record["prediction"].as_f64().is_some_and(f64::is_finite)),
                "{} returned a non-finite prediction",
                model.name
            );
        }
    }

    #[test]
    fn browser_spatial_piecewise_kriging_reports_spatial_details() {
        let response = run_forecast_request(BrowserForecastRequest {
            rows: sample_panel_rows(),
            frequency: "daily".to_string(),
            horizon: 2,
            model: "spatial_piecewise_kriging".to_string(),
            options: BrowserForecastOptions {
                coordinate_x: Some("longitude".to_string()),
                coordinate_y: Some("latitude".to_string()),
                kriging_range: Some(1.0),
                kriging_nugget: Some(1.0e-6),
                spatial_kriging_mode: Some("residual_kriging".to_string()),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata {
                timestamp_col: Some("timestamp".to_string()),
                target_col: Some("target".to_string()),
                series_id_col: Some("series_id".to_string()),
            },
        })
        .expect("spatial piecewise kriging run");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        assert_eq!(records.len(), 6);
        assert!(records[0].get("base_mean").is_some());
        assert!(records[0].get("spatial_correction").is_some());
        assert!(records[0].get("kriging_variance").is_some());
        assert!(records[0].get("selected_neighbors").is_some());
    }

    #[test]
    fn browser_auto_forecast_caps_direct_horizon_to_requested_horizon() {
        let request = BrowserForecastRequest {
            rows: (0..56)
                .map(|day| BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: date_string(day),
                    target: 120.0 + day as f64 * 2.0 + (day % 7) as f64 * 4.0,
                    covariates: BTreeMap::new(),
                })
                .collect(),
            frequency: "daily".to_string(),
            horizon: 14,
            model: "auto_forecast".to_string(),
            options: BrowserForecastOptions {
                season_length: Some(7),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata {
                timestamp_col: Some("timestamp".to_string()),
                target_col: Some("target".to_string()),
                series_id_col: Some("series_id".to_string()),
            },
        };
        let response = run_forecast_request(request).expect("auto forecast run");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        assert_eq!(records.len(), 14);
    }

    #[test]
    fn browser_graph_forecast_runs_each_paper_transformer_profile() {
        for profile in [
            "heterogeneous_moe",
            "efficient_high_order",
            "long_short_fusion",
            "gated_graph_temporal",
            "spatial_shift_graphon_moe",
        ] {
            let target = (0..12)
                .map(|step| {
                    let value = step as f64;
                    vec![20.0 + value, 16.0 + value * 0.7, 12.0 + value * 0.4]
                })
                .collect();
            let response = run_graph_forecast_request(BrowserGraphForecastRequest {
                frame: BrowserGraphTemporalFrame {
                    node_ids: vec![
                        "PULocationID:161".into(),
                        "PULocationID:236".into(),
                        "PULocationID:132".into(),
                    ],
                    timestamps: (0..12).map(i64::from).collect(),
                    target,
                    adjacency: BrowserCsrAdjacency {
                        indptr: vec![0, 2, 3, 3],
                        indices: vec![1, 2, 2],
                        data: vec![0.7, 0.3, 1.0],
                    },
                    horizon: 2,
                    frequency: "hourly".into(),
                    covariates: None,
                },
                options: BrowserGraphForecastOptions {
                    profile: Some(profile.into()),
                    lookback: Some(if profile == "long_short_fusion" { 8 } else { 3 }),
                    hidden_size: 4,
                    attention_heads: Some(2),
                    graph_order: Some(2),
                    experts: Some(2),
                    periodicity: Some(if profile == "long_short_fusion" { 1 } else { 3 }),
                    recent_window: Some(3),
                    epochs: 2,
                    learning_rate: 0.01,
                    ..BrowserGraphForecastOptions::default()
                },
                actual: None,
            })
            .expect("browser paper graph transformer run");
            assert_eq!(response.predictions.len(), 2, "{profile}");
            assert!(response
                .predictions
                .iter()
                .flatten()
                .all(|value| value.is_finite()));
            assert_eq!(response.metadata["model"].as_str(), Some(profile));
            assert!(response.metadata["architectureReport"].is_object());
        }
    }

    #[test]
    fn browser_lsttn_default_requires_long_horizon_history() {
        let error = run_graph_forecast_request(BrowserGraphForecastRequest {
            frame: BrowserGraphTemporalFrame {
                node_ids: vec!["PULocationID:161".into()],
                timestamps: (0..12).map(i64::from).collect(),
                target: (0..12).map(|step| vec![step as f64]).collect(),
                adjacency: BrowserCsrAdjacency {
                    indptr: vec![0, 0],
                    indices: vec![],
                    data: vec![],
                },
                horizon: 2,
                frequency: "hourly".into(),
                covariates: None,
            },
            options: BrowserGraphForecastOptions {
                profile: Some("long_short_fusion".into()),
                hidden_size: 2,
                epochs: 1,
                learning_rate: 0.01,
                ..BrowserGraphForecastOptions::default()
            },
            actual: None,
        })
        .expect_err("LSTTN browser defaults must retain long-horizon history");
        assert!(error.to_string().contains("lookback plus horizon"));
    }

    #[test]
    fn browser_regression_model_scores_holdout_and_reports_importance() {
        let request = BrowserRegressionRequest {
            rows: sample_regression_rows(),
            feature_names: vec![
                "trip_distance".to_string(),
                "pickup_hour".to_string(),
                "route_pressure".to_string(),
                "pickup_x".to_string(),
                "pickup_y".to_string(),
            ],
            sparse_feature_names: vec!["zone_memberships".to_string()],
            options: BrowserRegressionOptions {
                holdout_fraction: 0.25,
                splitter_mode: Some("full".to_string()),
                feature_kinds: BTreeMap::from([
                    ("trip_distance".to_string(), "numeric".to_string()),
                    ("pickup_hour".to_string(), "periodic".to_string()),
                    ("route_pressure".to_string(), "numeric".to_string()),
                    ("pickup_x".to_string(), "spatial".to_string()),
                    ("pickup_y".to_string(), "spatial".to_string()),
                ]),
                periodic_periods: BTreeMap::from([("pickup_hour".to_string(), 24)]),
                loss: Some("huber".to_string()),
                quantile_alpha: None,
                huber_delta: Some(5.0),
                log_offset: None,
                interval_lower_alpha: Some(0.1),
                interval_upper_alpha: Some(0.9),
                n_estimators: Some(80),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                monotonic_constraints: None,
                include_model_visualization: None,
            },
        };
        let response = run_regression_request(request).expect("regression run");
        assert_eq!(response.metrics.train_rows, 45);
        assert_eq!(response.metrics.holdout_rows, 15);
        assert_eq!(response.predictions.len(), 15);
        assert_eq!(response.metadata["splitterMode"].as_str(), Some("full"));
        assert!(response.metrics.rmse.is_finite());
        assert!(response.metrics.mae.is_finite());
        assert!(response.metrics.r2.is_finite());
        assert_eq!(response.feature_importance.len(), 6);
        assert_eq!(
            response.metadata["sparseFeatureNames"][0].as_str(),
            Some("zone_memberships")
        );
        assert_eq!(response.metadata["loss"].as_str(), Some("huber"));
        assert!(response
            .predictions
            .iter()
            .all(|row| row.lower_prediction.is_some() && row.upper_prediction.is_some()));
        assert!(response.predictions.iter().all(|row| row
            .lower_prediction
            .zip(row.upper_prediction)
            .is_some_and(|(lower, upper)| lower <= upper)));
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.split_count > 0));
    }

    #[test]
    fn browser_regression_model_rejects_unknown_loss() {
        let mut request = BrowserRegressionRequest {
            rows: sample_regression_rows(),
            feature_names: vec![
                "trip_distance".to_string(),
                "pickup_hour".to_string(),
                "route_pressure".to_string(),
                "pickup_x".to_string(),
                "pickup_y".to_string(),
            ],
            sparse_feature_names: vec!["zone_memberships".to_string()],
            options: BrowserRegressionOptions::default(),
        };
        request.options.loss = Some("not_a_loss".to_string());
        let error = run_regression_request(request).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported browser regression loss"));
    }

    #[test]
    fn browser_regression_model_rejects_bad_feature_width() {
        let error = run_regression_request(BrowserRegressionRequest {
            rows: vec![
                BrowserRegressionRow {
                    features: vec![1.0, 2.0],
                    sparse_sets: Vec::new(),
                    target: 3.0,
                },
                BrowserRegressionRow {
                    features: vec![2.0],
                    sparse_sets: Vec::new(),
                    target: 4.0,
                },
                BrowserRegressionRow {
                    features: vec![3.0, 4.0],
                    sparse_sets: Vec::new(),
                    target: 5.0,
                },
                BrowserRegressionRow {
                    features: vec![4.0, 5.0],
                    sparse_sets: Vec::new(),
                    target: 6.0,
                },
            ],
            feature_names: vec!["x".to_string(), "z".to_string()],
            sparse_feature_names: Vec::new(),
            options: BrowserRegressionOptions::default(),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("feature row has 1 columns but feature_names has 2"));
    }

    #[test]
    fn browser_neural_embedding_model_scores_holdout() {
        let request = BrowserNeuralRequest {
            rows: sample_neural_rows(),
            dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
            node_features: Vec::new(),
            node_types: Vec::new(),
            edge_type_triples: Vec::new(),
            pipeline: "embedding".to_string(),
            options: BrowserNeuralOptions {
                holdout_fraction: 0.25,
                embedding_dim: Some(4),
                random_state: Some(42),
                n_estimators: Some(40),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                ..BrowserNeuralOptions::default()
            },
        };
        let response = run_neural_request(request).expect("embedding neural run");
        assert_eq!(response.metrics.train_rows, 36);
        assert_eq!(response.metrics.holdout_rows, 12);
        assert_eq!(response.predictions.len(), 12);
        assert_eq!(
            response.metadata["details"]["model"].as_str(),
            Some("neural_embedding_regressor")
        );
        assert!(response.metrics.rmse.is_finite());
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.feature.starts_with("embedding_")));
    }

    #[test]
    fn browser_node2vec_model_scores_pair_holdout() {
        let request = BrowserNeuralRequest {
            rows: sample_neural_rows(),
            dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
            node_features: sample_node_features(),
            node_types: Vec::new(),
            edge_type_triples: Vec::new(),
            pipeline: "node2vec".to_string(),
            options: BrowserNeuralOptions {
                holdout_fraction: 0.25,
                embedding_dim: Some(4),
                node2vec_walk_length: Some(6),
                node2vec_walks_per_node: Some(3),
                node2vec_window_size: Some(2),
                node2vec_epochs: Some(2),
                node2vec_seed: Some(7),
                n_estimators: Some(40),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                ..BrowserNeuralOptions::default()
            },
        };
        let response = run_neural_request(request).expect("node2vec neural run");
        assert_eq!(response.metrics.train_rows, 36);
        assert_eq!(response.metrics.holdout_rows, 12);
        assert_eq!(response.predictions.len(), 12);
        assert_eq!(
            response.metadata["details"]["model"].as_str(),
            Some("node2vec_regressor")
        );
        assert_eq!(response.metadata["details"]["nodeCount"].as_u64(), Some(8));
        assert!(response.metrics.mae.is_finite());
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.feature.starts_with("node2vec_")));
    }

    #[test]
    fn browser_graphsage_family_models_score_pair_holdout() {
        for (pipeline, expected_model, expected_prefix) in [
            ("graphsage", "graphsage_regressor", "graphsage_"),
            (
                "hetero_graphsage",
                "hetero_graphsage_regressor",
                "hetero_graphsage_",
            ),
            ("hinsage", "hinsage_regressor", "hinsage_"),
        ] {
            let request = BrowserNeuralRequest {
                rows: sample_neural_rows(),
                dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
                node_features: sample_node_features(),
                node_types: vec![0, 0, 0, 0, 1, 1, 1, 1],
                edge_type_triples: vec![(0, 0, 1)],
                pipeline: pipeline.to_string(),
                options: BrowserNeuralOptions {
                    holdout_fraction: 0.25,
                    embedding_dim: Some(4),
                    graph_sage_epochs: Some(2),
                    graph_sage_negative_samples: Some(2),
                    graph_sage_seed: Some(11),
                    n_estimators: Some(40),
                    learning_rate: Some(0.08),
                    max_depth: Some(3),
                    min_samples_leaf: Some(2),
                    ..BrowserNeuralOptions::default()
                },
            };
            let response = run_neural_request(request).expect("graph neural run");
            assert_eq!(response.metrics.train_rows, 36);
            assert_eq!(response.metrics.holdout_rows, 12);
            assert_eq!(response.predictions.len(), 12);
            assert_eq!(
                response.metadata["details"]["model"].as_str(),
                Some(expected_model)
            );
            assert_eq!(response.metadata["details"]["nodeCount"].as_u64(), Some(8));
            assert!(response.metrics.rmse.is_finite());
            assert!(response
                .feature_importance
                .iter()
                .any(|item| item.feature.starts_with(expected_prefix)));
        }
    }

    fn sample_panel_rows() -> Vec<BrowserForecastRow> {
        let mut rows = Vec::new();
        for (series_index, series_id) in ["pickup_zone_1", "pickup_zone_2", "pickup_zone_3"]
            .iter()
            .enumerate()
        {
            for day in 0..70 {
                let weekly = (day % 7) as f64;
                let level = 120.0 + series_index as f64 * 30.0;
                let target = level + day as f64 * 1.4 + weekly * 3.0;
                rows.push(BrowserForecastRow {
                    series_id: Some((*series_id).to_string()),
                    timestamp: date_string(day),
                    target,
                    covariates: BTreeMap::from([
                        ("longitude".to_string(), -73.98 + series_index as f64 * 0.02),
                        ("latitude".to_string(), 40.74 + series_index as f64 * 0.02),
                    ]),
                });
            }
        }
        rows
    }

    fn sample_regression_rows() -> Vec<BrowserRegressionRow> {
        (0..60)
            .map(|idx| {
                let trip_distance = 0.8 + idx as f64 * 0.12;
                let pickup_hour = (idx % 24) as f64;
                let route_pressure = ((idx * 7) % 11) as f64;
                let pickup_x = -73.98 + (idx as f64 / 6.0).sin() * 0.04;
                let pickup_y = 40.74 + (idx as f64 / 7.0).cos() * 0.03;
                let neighborhood_signal = if idx % 3 == 0 { 5.0 } else { 0.0 };
                BrowserRegressionRow {
                    features: vec![
                        trip_distance,
                        pickup_hour,
                        route_pressure,
                        pickup_x,
                        pickup_y,
                    ],
                    sparse_sets: vec![vec![101 + (idx % 3) as u64, 200 + (idx % 5) as u64]],
                    target: 6.0
                        + trip_distance * 2.4
                        + pickup_hour * 0.35
                        + route_pressure * 1.1
                        + (pickup_x + 74.0) * 10.0
                        + (pickup_y - 40.7) * 12.0
                        + neighborhood_signal,
                }
            })
            .collect()
    }

    fn sample_neural_rows() -> Vec<BrowserNeuralRow> {
        (0..48)
            .map(|idx| {
                let source = idx % 4;
                let target_node = 4 + ((idx * 3) % 4);
                let trip_distance = 1.0 + (idx % 8) as f64 * 0.35;
                let pickup_hour = (idx % 24) as f64;
                BrowserNeuralRow {
                    id: Some((source + 1) as u64),
                    source: Some(source),
                    target_node: Some(target_node),
                    edge_weight: Some(1.0 + (idx % 3) as f32 * 0.2),
                    edge_type: Some(0),
                    dense: vec![trip_distance, pickup_hour],
                    target: 20.0
                        + source as f64 * 4.0
                        + target_node as f64 * 2.5
                        + trip_distance * 3.0
                        + pickup_hour * 0.4,
                }
            })
            .collect()
    }

    fn sample_sequence_series() -> SequenceSeries {
        SequenceSeries {
            series_id: "pickup_zone_1".to_string(),
            rows: vec![
                sequence_row("r0", 0.0, Some(0.0)),
                sequence_row("r1", 1.0, Some(1.0)),
                sequence_row("r2", 2.0, None),
                sequence_row("r3", 3.0, None),
            ],
        }
    }

    fn sequence_row(
        row_id: &str,
        position: f64,
        target: Option<f64>,
    ) -> cartoboost_core::forecasting::SequenceRow {
        cartoboost_core::forecasting::SequenceRow {
            row_id: row_id.to_string(),
            position,
            target,
            reference_axis: None,
            reference_signal: None,
            auxiliary_rate: None,
        }
    }

    fn sample_node_features() -> Vec<Vec<f32>> {
        (0..8)
            .map(|node| {
                vec![
                    node as f32 / 8.0,
                    if node < 4 { 0.0 } else { 1.0 },
                    ((node * 3) % 5) as f32 / 5.0,
                ]
            })
            .collect()
    }

    fn date_string(day_index: usize) -> String {
        const MONTH_LENGTHS: [usize; 3] = [31, 28, 31];
        let mut remaining = day_index;
        for (month_index, month_length) in MONTH_LENGTHS.iter().enumerate() {
            if remaining < *month_length {
                return format!(
                    "2026-{month:02}-{day:02}",
                    month = month_index + 1,
                    day = remaining + 1
                );
            }
            remaining -= month_length;
        }
        panic!("sample day index out of range");
    }
}
