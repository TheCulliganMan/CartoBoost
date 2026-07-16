#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{ForecastFrameMetadata, ForecastFrequency};
    use chrono::{Duration, NaiveDate, NaiveDateTime};

    fn ts(day: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 1, day)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid fixture timestamp")
    }

    #[test]
    fn piecewise_linear_predicts_irregular_history_at_future_timestamps() {
        let frame = ForecastFrame::with_metadata(
            vec![
                ForecastRow::new("__single__", ts(1), 10.0),
                ForecastRow::new("__single__", ts(8), 14.0),
                ForecastRow::new("__single__", ts(9), 15.0),
                ForecastRow::new("__single__", ts(12), 17.0),
            ],
            ForecastFrequency::Daily,
            ForecastFrameMetadata {
                allow_irregular: true,
                ..ForecastFrameMetadata::default()
            },
        )
        .expect("irregular frame");
        let mut model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig::default())
                .expect("valid piecewise model");
        model.fit(&frame).expect("fit irregular history");

        let result = model
            .predict_at_timestamps(BTreeMap::from([(
                "__single__".to_string(),
                vec![ts(13), ts(14)],
            )]))
            .expect("predict at explicit timestamps");

        let predictions = result.predictions();
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].timestamp, ts(13));
        assert_eq!(predictions[1].timestamp, ts(14));
        assert!(predictions
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
    }

    #[test]
    fn piecewise_linear_skips_missing_targets_like_prophet() {
        let frame = ForecastFrame::with_metadata(
            vec![
                ForecastRow::new("__single__", ts(1), 10.0),
                ForecastRow::new("__single__", ts(2), f64::NAN),
                ForecastRow::new("__single__", ts(3), 12.0),
                ForecastRow::new("__single__", ts(4), 13.0),
            ],
            ForecastFrequency::Daily,
            ForecastFrameMetadata {
                allow_missing_targets: true,
                ..ForecastFrameMetadata::default()
            },
        )
        .expect("missing target frame");
        let mut model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig::default())
                .expect("valid piecewise model");
        model.fit(&frame).expect("fit missing target frame");

        let result = model.predict(2).expect("forecast");
        let components = model
            .predict_components_json_value(2)
            .expect("component forecast");
        let history_components = model
            .history_components_json_value()
            .expect("history component forecast");
        let history_records = history_components["records"]
            .as_array()
            .expect("history records");

        assert_eq!(result.predictions().len(), 2);
        assert_eq!(result.predictions()[0].timestamp, ts(5));
        assert_eq!(
            components["records"][0]["timestamp"].as_str(),
            Some("2026-01-05T00:00:00")
        );
        assert_eq!(history_records.len(), 4);
        assert_eq!(
            history_records[1]["timestamp"].as_str(),
            Some("2026-01-02T00:00:00")
        );
        assert!(history_records[1]["actual"].is_null());
        assert!(history_records[1]["residual"].is_null());
        assert!(history_records[1]["fitted"]
            .as_f64()
            .expect("missing-target fitted value")
            .is_finite());
        assert!(result
            .predictions()
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
        assert_eq!(
            model.fitted_series_ids().expect("series ids"),
            vec!["__single__"]
        );
    }

    #[test]
    fn regular_models_reject_missing_targets() {
        let frame = ForecastFrame::with_metadata(
            vec![
                ForecastRow::new("__single__", ts(1), 10.0),
                ForecastRow::new("__single__", ts(2), f64::NAN),
                ForecastRow::new("__single__", ts(3), 12.0),
            ],
            ForecastFrequency::Daily,
            ForecastFrameMetadata {
                allow_missing_targets: true,
                ..ForecastFrameMetadata::default()
            },
        )
        .expect("missing target frame");
        let mut ets = ETSForecaster::new(0.5, 0.1).expect("valid ets");
        let err = ets.fit(&frame).expect_err("ets rejects missing targets");
        assert!(err.to_string().contains("requires observed finite targets"));
    }

    #[test]
    fn theta_forecasts_panel_series_without_bleeding() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 12.0),
                ForecastRow::new("PU1->DO2", ts(3), 15.0),
                ForecastRow::new("PU1->DO2", ts(4), 19.0),
                ForecastRow::new("PU9->DO8", ts(1), 30.0),
                ForecastRow::new("PU9->DO8", ts(2), 29.0),
                ForecastRow::new("PU9->DO8", ts(3), 27.0),
                ForecastRow::new("PU9->DO8", ts(4), 24.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ThetaForecaster::new(2.0, 0.4).expect("valid theta");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.series_id.as_str(), row.horizon, row.mean))
            .collect::<Vec<_>>();
        assert_eq!(means.len(), 4);
        assert_eq!(means[0].0, "PU1->DO2");
        assert_eq!(means[2].0, "PU9->DO8");
        assert!(means[0].2 > means[1].2 - 10.0);
        assert_ne!(means[0].2, means[2].2);
        assert_eq!(model.fitted_values("PU1->DO2").expect("fitted").len(), 4);
        assert_eq!(model.residuals("PU9->DO8").expect("residuals").len(), 4);
    }

    #[test]
    fn spatial_piecewise_residual_kriging_improves_synthetic_spatial_panel() {
        let mut rows = Vec::new();
        let coordinates = BTreeMap::from([
            ("PU1->DO1".to_string(), (0.0, 0.0)),
            ("PU2->DO2".to_string(), (1.0, 0.0)),
            ("PU3->DO3".to_string(), (2.0, 0.0)),
        ]);
        let offsets = BTreeMap::from([("PU1->DO1", 0.0), ("PU2->DO2", 4.0), ("PU3->DO3", 8.0)]);
        for day in 1..=8 {
            for (series_id, offset) in &offsets {
                rows.push(ForecastRow::new(
                    *series_id,
                    ts(day),
                    20.0 + f64::from(day) + offset,
                ));
            }
        }
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let piecewise_config = PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Flat,
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            auto_yearly_seasonality: false,
            auto_daily_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut base =
            PiecewiseLinearSeasonalForecaster::new(piecewise_config.clone()).expect("base config");
        base.fit(&frame).expect("base fit");
        let base_result = base.predict(1).expect("base predict");
        let mut fused = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates,
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config,
            kriging_config: OrdinaryKrigingConfig::new(3.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect("spatial config");
        fused.fit(&frame).expect("spatial fit");
        let fused_result = fused.predict(1).expect("spatial predict");
        let actuals = BTreeMap::from([
            ("PU1->DO1".to_string(), 29.0),
            ("PU2->DO2".to_string(), 33.0),
            ("PU3->DO3".to_string(), 37.0),
        ]);
        let base_mae = base_result
            .predictions()
            .iter()
            .map(|prediction| (prediction.mean - actuals[&prediction.series_id]).abs())
            .sum::<f64>()
            / actuals.len() as f64;
        let fused_mae = fused_result
            .predictions()
            .iter()
            .map(|prediction| (prediction.mean - actuals[&prediction.series_id]).abs())
            .sum::<f64>()
            / actuals.len() as f64;
        assert!(fused_mae < base_mae);
        assert_eq!(
            fused_result.details().len(),
            fused_result.predictions().len()
        );
        let json = fused_result.to_json_value();
        let first = &json["records"][0];
        assert!(first.get("base_mean").is_some());
        assert!(first.get("spatial_correction").is_some());
        assert!(first.get("kriging_variance").is_some());
        assert!(first.get("selected_neighbors").is_some());
        assert_eq!(first["metadata"]["neighbor_count"].as_u64(), Some(2));
        assert!(first["metadata"]["correction_magnitude"].is_number());
        assert!(first["metadata"]["kriging_variance"].is_number());
        assert!(first["metadata"]["fit_runtime_seconds"].is_number());
        assert!(fused_result.details().iter().all(|detail| {
            !detail
                .selected_neighbors
                .iter()
                .any(|neighbor| neighbor == &detail.series_id)
        }));
        for interval in fused_result.intervals() {
            let prediction = fused_result
                .predictions()
                .iter()
                .find(|prediction| {
                    prediction.series_id == interval.series_id
                        && prediction.timestamp == interval.timestamp
                        && prediction.horizon == interval.horizon
                })
                .expect("matching fused prediction");
            assert!(interval.lower <= prediction.mean);
            assert!(interval.upper >= prediction.mean);
            let base_interval = base_result
                .intervals()
                .iter()
                .find(|candidate| {
                    candidate.series_id == interval.series_id
                        && candidate.timestamp == interval.timestamp
                        && candidate.horizon == interval.horizon
                        && candidate.level == interval.level
                })
                .expect("matching base interval");
            assert!(
                interval.upper - interval.lower
                    >= base_interval.upper - base_interval.lower - 1.0e-12
            );
        }
    }

    #[test]
    fn spatial_models_reject_mixed_panel_cutoffs() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO1", ts(1), 10.0),
                ForecastRow::new("PU1->DO1", ts(2), 11.0),
                ForecastRow::new("PU2->DO2", ts(2), 20.0),
                ForecastRow::new("PU2->DO2", ts(3), 21.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid ragged frame");
        let coordinates = BTreeMap::from([
            ("PU1->DO1".to_string(), (0.0, 0.0)),
            ("PU2->DO2".to_string(), (1.0, 0.0)),
        ]);
        let mut kriging =
            KrigingForecaster::new(coordinates.clone(), 1.0, 1.0e-6).expect("kriging config");
        let error = kriging.fit(&frame).expect_err("mixed cutoffs must fail");
        assert!(error.to_string().contains("common panel cutoff timestamp"));

        let mut spatial = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates,
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config: PiecewiseLinearSeasonalConfig::default(),
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect("spatial config");
        let error = spatial.fit(&frame).expect_err("mixed cutoffs must fail");
        assert!(error.to_string().contains("common panel cutoff timestamp"));
    }

    #[test]
    fn spatial_piecewise_kriging_errors_for_missing_coordinate() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO1", ts(1), 10.0),
                ForecastRow::new("PU1->DO1", ts(2), 11.0),
                ForecastRow::new("PU2->DO2", ts(1), 20.0),
                ForecastRow::new("PU2->DO2", ts(2), 21.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates: BTreeMap::from([("PU1->DO1".to_string(), (0.0, 0.0))]),
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config: PiecewiseLinearSeasonalConfig::default(),
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect("spatial config");
        let err = model
            .fit(&frame)
            .expect_err("missing coordinate should fail");
        assert!(err
            .to_string()
            .contains("missing spatial piecewise kriging coordinate"));
    }

    #[test]
    fn spatial_piecewise_kriging_rejects_future_spatial_regressor_leakage() {
        let mut piecewise_config = PiecewiseLinearSeasonalConfig::default();
        piecewise_config
            .future_regressors
            .insert("traffic_density".to_string(), vec![1.0]);
        let err = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            coordinates: BTreeMap::from([("PU1->DO1".to_string(), (0.0, 0.0))]),
            mode: SpatialPiecewiseKrigingMode::KrigedRegressors,
            piecewise_config,
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("kriging config"),
            spatial_regressors: vec!["traffic_density".to_string()],
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        })
        .expect_err("future spatial regressor should fail");
        assert!(err.to_string().contains("would leak future observations"));
    }

    #[test]
    fn spatial_piecewise_kriging_neighbor_fallback_is_opt_in_and_flagged() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO1", ts(1), 10.0),
                ForecastRow::new("PU1->DO1", ts(2), 11.0),
                ForecastRow::new("PU2->DO2", ts(1), 20.0),
                ForecastRow::new("PU2->DO2", ts(2), 21.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = SpatialPiecewiseKrigingConfig {
            coordinates: BTreeMap::from([
                ("PU1->DO1".to_string(), (0.0, 0.0)),
                ("PU2->DO2".to_string(), (100.0, 0.0)),
            ]),
            mode: SpatialPiecewiseKrigingMode::ResidualKriging,
            piecewise_config: PiecewiseLinearSeasonalConfig::default(),
            kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6)
                .and_then(|config| config.with_neighbor_limits(None, 3, None))
                .expect("kriging config"),
            spatial_regressors: Vec::new(),
            residual_shrinkage: 1.0,
            allow_neighbor_fallback: false,
        };
        let mut strict =
            SpatialPiecewiseKrigingForecaster::new(base_config.clone()).expect("strict config");
        strict.fit(&frame).expect("strict fit");
        strict
            .predict(1)
            .expect_err("neighbor rule should fail without fallback");

        let mut fallback = SpatialPiecewiseKrigingForecaster::new(SpatialPiecewiseKrigingConfig {
            allow_neighbor_fallback: true,
            ..base_config
        })
        .expect("fallback config");
        fallback.fit(&frame).expect("fallback fit");
        let result = fallback.predict(1).expect("fallback predict");
        assert!(result.details().iter().all(|detail| {
            detail
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("neighbor_fallback"))
                .and_then(Value::as_bool)
                == Some(true)
        }));
    }

    #[test]
    fn theta_additive_seasonality_reseasons_forecast() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    let base = f64::from(day);
                    let seasonal = if day % 2 == 0 { 5.0 } else { -5.0 };
                    ForecastRow::single(ts(day), 20.0 + base + seasonal)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::additive(2).expect("valid season");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.5, Some(seasonality)).expect("valid theta");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| row.mean)
            .collect::<Vec<_>>();

        assert_eq!(means.len(), 2);
        assert!(means[1] > means[0]);
    }

    #[test]
    fn theta_multiplicative_rejects_non_positive_values() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 0.0),
                ForecastRow::single(ts(4), 4.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::multiplicative(2).expect("valid season");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.5, Some(seasonality)).expect("valid theta");

        let err = model.fit(&frame).expect_err("non-positive values rejected");

        assert!(err.to_string().contains("non-positive"));
    }

    #[test]
    fn piecewise_linear_seasonal_forecaster_projects_trend_and_weekly_pattern() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let weekly = if day % 7 == 0 { 8.0 } else { 0.0 };
                    ForecastRow::single(ts(day), 30.0 + 1.5 * f64::from(day) + weekly)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 3,
            weekly_fourier_order: 3,
            seasonality_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid piecewise seasonal config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(7).expect("predict");
        let predictions = forecast.predictions();

        assert_eq!(predictions.len(), 7);
        assert!(predictions[6].mean > predictions[0].mean);
        assert_eq!(predictions[0].model, "piecewise_linear_seasonal");
        assert_eq!(model.metadata()["weekly_fourier_order"].as_u64(), Some(3));
    }

    #[test]
    fn piecewise_linear_default_changepoint_range_tracks_late_weekly_lane_movement() {
        let start = ts(1);
        let rows = (0..140)
            .map(|week| {
                let late_ramp = if week >= 92 {
                    55.0 * f64::from(week - 92)
                } else {
                    0.0
                };
                let current_ramp = if week >= 118 {
                    140.0 * f64::from(week - 118)
                } else {
                    0.0
                };
                let annual = 70.0 * (std::f64::consts::TAU * f64::from(week) / 52.18).sin();
                ForecastRow::new(
                    "PULocationID=132->DOLocationID=236",
                    start + Duration::weeks(i64::from(week)),
                    1100.0 + 2.2 * f64::from(week) + late_ramp + current_ramp + annual,
                )
            })
            .collect::<Vec<_>>();
        let default_rmse = rolling_one_week_rmse(
            &rows,
            PiecewiseLinearSeasonalConfig {
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                changepoint_l2_regularization: 0.001,
                ..PiecewiseLinearSeasonalConfig::default()
            },
        );
        let strict_prophet_range_rmse = rolling_one_week_rmse(
            &rows,
            PiecewiseLinearSeasonalConfig {
                changepoint_range: 0.8,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                changepoint_l2_regularization: 0.001,
                ..PiecewiseLinearSeasonalConfig::default()
            },
        );

        assert!(default_rmse < strict_prophet_range_rmse * 0.4);
    }

    fn rolling_one_week_rmse(rows: &[ForecastRow], config: PiecewiseLinearSeasonalConfig) -> f64 {
        let mut sum_squared = 0.0;
        for holdout_weeks_back in (1..=12).rev() {
            let cutoff = rows.len() - holdout_weeks_back;
            let train = rows[..cutoff].to_vec();
            let actual = rows[cutoff].target;
            let frame = ForecastFrame::new(train, ForecastFrequency::Weekly).expect("valid frame");
            let mut model =
                PiecewiseLinearSeasonalForecaster::new(config.clone()).expect("valid config");
            model.fit(&frame).expect("fit");
            let prediction = model.predict(1).expect("predict").predictions()[0].mean;
            let error = prediction - actual;
            sum_squared += error * error;
        }
        (sum_squared / 12.0).sqrt()
    }

    #[test]
    fn piecewise_linear_auto_seasonalities_resolve_from_training_span() {
        let hourly_frame = ForecastFrame::new(
            (0..72)
                .map(|hour| {
                    ForecastRow::single(
                        ts(1) + Duration::hours(i64::from(hour)),
                        50.0 + 0.1 * f64::from(hour),
                    )
                })
                .collect(),
            ForecastFrequency::Hourly,
        )
        .expect("valid hourly frame");
        let long_daily_frame = ForecastFrame::new(
            (0..800)
                .map(|day| {
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day)),
                        80.0 + 0.05 * f64::from(day),
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid long daily frame");
        let mut hourly = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            weekly_fourier_order: 0,
            daily_fourier_order: 0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid hourly auto config");
        let mut long_daily =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                yearly_fourier_order: 0,
                weekly_fourier_order: 0,
                daily_fourier_order: 0,
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid long daily auto config");
        let mut disabled = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            yearly_fourier_order: 0,
            weekly_fourier_order: 0,
            auto_yearly_seasonality: false,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid disabled auto config");

        hourly.fit(&hourly_frame).expect("hourly fit");
        long_daily.fit(&long_daily_frame).expect("long daily fit");
        disabled.fit(&long_daily_frame).expect("disabled fit");

        assert_eq!(hourly.metadata()["daily_fourier_order"].as_u64(), Some(4));
        assert_eq!(
            long_daily.metadata()["yearly_fourier_order"].as_u64(),
            Some(10)
        );
        assert_eq!(
            long_daily.metadata()["weekly_fourier_order"].as_u64(),
            Some(3)
        );
        assert_eq!(
            disabled.metadata()["yearly_fourier_order"].as_u64(),
            Some(0)
        );
        assert_eq!(
            disabled.metadata()["weekly_fourier_order"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn piecewise_linear_seasonal_components_reconstruct_predictions() {
        let rows = (1..=35)
            .map(|day| {
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                let timestamp = ts(1) + Duration::days(i64::from(day - 1));
                ForecastRow::with_covariates(
                    "PU1->DO2",
                    timestamp,
                    50.0 + f64::from(day) + 20.0 * queue,
                    BTreeMap::from([("airport_queue".to_string(), queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 2,
            seasonality_l2_regularization: 0.001,
            regressor_l2_regularization: 0.001,
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid piecewise config");
        model.fit(&frame).expect("fit");

        let forecast = model.predict(3).expect("predict");
        let components = model
            .predict_components_json_value(3)
            .expect("component forecast");
        let history_components = model
            .history_components_json_value()
            .expect("history components");
        let records = components["records"].as_array().expect("records");
        let history_records = history_components["records"]
            .as_array()
            .expect("history records");

        assert_eq!(records.len(), 3);
        assert_eq!(history_records.len(), 35);
        assert_eq!(
            records[0]["prediction"]
                .as_f64()
                .expect("component prediction"),
            forecast.predictions()[0].mean
        );
        assert!(records[0]["components"]["weekly"].as_f64().is_some());
        assert!(history_records[0]["components"]["weekly"]
            .as_f64()
            .is_some());
        assert!(history_records[1]["trend_movement"].as_f64().is_some());
        let reconstructed_residual = history_records[0]["actual"]
            .as_f64()
            .expect("history actual")
            - history_records[0]["fitted"]
                .as_f64()
                .expect("history fitted");
        let emitted_residual = history_records[0]["residual"]
            .as_f64()
            .expect("history residual");
        assert!((reconstructed_residual - emitted_residual).abs() < 1.0e-9);
        assert!(
            records[0]["components"]["regressors"]["airport_queue"]
                .as_f64()
                .expect("airport queue contribution")
                > 10.0
        );
        assert!(components["columns"]
            .as_array()
            .expect("columns")
            .iter()
            .any(|column| column.as_str() == Some("components")));
    }

    #[test]
    fn piecewise_linear_seasonal_flat_growth_suppresses_trend_extrapolation() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| ForecastRow::single(ts(day), 40.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut linear = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid linear config");
        let mut flat = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Flat,
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid flat config");

        linear.fit(&frame).expect("linear fit");
        flat.fit(&frame).expect("flat fit");
        let linear_forecast = linear.predict(3).expect("linear predict");
        let flat_forecast = flat.predict(3).expect("flat predict");

        assert!(linear_forecast.predictions()[2].mean > flat_forecast.predictions()[2].mean + 20.0);
        assert_eq!(flat.metadata()["growth"].as_str(), Some("flat"));
    }

    #[test]
    fn piecewise_linear_seasonal_forecaster_emits_prediction_intervals() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let noise = if day % 2 == 0 { 0.4 } else { -0.4 };
                    ForecastRow::single(ts(day), 30.0 + 1.5 * f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 2,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8, 0.95],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid piecewise seasonal config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.intervals().len(), 4);
        assert!(forecast
            .intervals()
            .iter()
            .all(|interval| interval.lower <= interval.upper));
        assert!(forecast
            .intervals()
            .iter()
            .any(|interval| (interval.level - 0.95).abs() < 1.0e-12));
    }

    #[test]
    fn piecewise_linear_skips_coefficient_covariance_for_point_only_fit() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| ForecastRow::single(ts(day), 40.0 + 0.8 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut point_only =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 2,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("point-only config");
        let mut interval_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 2,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                interval_levels: vec![0.8],
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("interval config");

        point_only.fit(&frame).expect("point fit");
        interval_model.fit(&frame).expect("interval fit");
        let point_series = point_only
            .fitted
            .as_ref()
            .expect("point fitted")
            .series
            .get("__single__")
            .expect("point series");
        let interval_series = interval_model
            .fitted
            .as_ref()
            .expect("interval fitted")
            .series
            .get("__single__")
            .expect("interval series");

        assert!(point_series.coefficient_covariance.is_empty());
        assert!(!interval_series.coefficient_covariance.is_empty());
    }

    #[test]
    fn piecewise_linear_coefficient_uncertainty_widens_intervals() {
        let frame = ForecastFrame::new(
            (1..=12)
                .map(|day| {
                    let noise = if day % 2 == 0 { 1.0 } else { -1.0 };
                    ForecastRow::single(ts(day), 25.0 + 1.2 * f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut residual_only =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                coefficient_uncertainty_scale: 0.0,
                ..base_config.clone()
            })
            .expect("residual interval config");
        let mut posterior_like =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                coefficient_uncertainty_scale: 3.0,
                ..base_config
            })
            .expect("coefficient uncertainty config");

        residual_only.fit(&frame).expect("residual fit");
        posterior_like
            .fit(&frame)
            .expect("coefficient uncertainty fit");
        let residual_forecast = residual_only.predict(5).expect("residual predict");
        let posterior_forecast = posterior_like
            .predict(5)
            .expect("coefficient uncertainty predict");
        let residual_interval = &residual_forecast.intervals()[4];
        let posterior_interval = &posterior_forecast.intervals()[4];
        let residual_width = residual_interval.upper - residual_interval.lower;
        let posterior_width = posterior_interval.upper - posterior_interval.lower;

        assert_eq!(
            residual_forecast.predictions()[4].mean,
            posterior_forecast.predictions()[4].mean
        );
        assert!(posterior_width > residual_width);
        assert_eq!(
            posterior_like.metadata()["coefficient_uncertainty_scale"].as_f64(),
            Some(3.0)
        );
    }

    #[test]
    fn piecewise_linear_uncertainty_samples_widen_future_intervals() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| {
                    let t = f64::from(day);
                    let after_break = (t - 15.0).max(0.0);
                    ForecastRow::single(ts(day), 20.0 + 0.5 * t + 3.0 * after_break)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 1,
            changepoint_timestamps: vec![ts(15)],
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            changepoint_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut residual_only = PiecewiseLinearSeasonalForecaster::new(base_config.clone())
            .expect("valid residual interval config");
        let mut trend_uncertain =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                uncertainty_samples: 256,
                trend_uncertainty_scale: 1.0,
                uncertainty_seed: 7,
                ..base_config.clone()
            })
            .expect("valid trend uncertainty config");
        let mut normal_uncertain =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                uncertainty_samples: 256,
                trend_uncertainty_policy: PiecewiseLinearTrendUncertaintyPolicy::Normal,
                trend_uncertainty_scale: 1.0,
                uncertainty_seed: 7,
                ..base_config
            })
            .expect("valid normal trend uncertainty config");

        residual_only.fit(&frame).expect("residual fit");
        trend_uncertain.fit(&frame).expect("uncertain fit");
        normal_uncertain.fit(&frame).expect("normal uncertain fit");
        let residual_forecast = residual_only.predict(5).expect("residual predict");
        let uncertain_forecast = trend_uncertain.predict(5).expect("uncertain predict");
        let normal_forecast = normal_uncertain
            .predict(5)
            .expect("normal uncertain predict");
        let residual_interval = &residual_forecast.intervals()[4];
        let uncertain_interval = &uncertain_forecast.intervals()[4];
        let normal_interval = &normal_forecast.intervals()[4];
        let residual_width = residual_interval.upper - residual_interval.lower;
        let uncertain_width = uncertain_interval.upper - uncertain_interval.lower;
        let normal_width = normal_interval.upper - normal_interval.lower;

        assert!(uncertain_width > residual_width + 1.0);
        assert!((uncertain_width - normal_width).abs() > 1.0e-6);
        assert_eq!(
            trend_uncertain.metadata()["uncertainty_samples"].as_u64(),
            Some(256)
        );
        assert_eq!(
            trend_uncertain.metadata()["trend_uncertainty_policy"].as_str(),
            Some("laplace")
        );
        assert_eq!(
            normal_uncertain.metadata()["trend_uncertainty_policy"].as_str(),
            Some("normal")
        );
    }

    #[test]
    fn piecewise_linear_predictive_samples_round_trip_with_artifact() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| {
                    let noise = if day % 2 == 0 { 0.6 } else { -0.6 };
                    ForecastRow::single(ts(day), 40.0 + 0.8 * f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 1,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            uncertainty_samples: 8,
            uncertainty_seed: 11,
            coefficient_uncertainty_scale: 1.5,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid samples config");

        model.fit(&frame).expect("fit");
        let samples = model.predict_samples_json_value(3).expect("samples");
        let records = samples["records"].as_array().expect("sample records");
        let payload = model.to_json_string().expect("serialize artifact");
        let restored = PiecewiseLinearSeasonalForecaster::from_json_string(&payload)
            .expect("restore artifact");
        let restored_samples = restored
            .predict_samples_json_value(3)
            .expect("restored samples");
        let restored_records = restored_samples["records"]
            .as_array()
            .expect("restored sample records");

        assert_eq!(samples["sample_count"].as_u64(), Some(8));
        assert_eq!(restored_samples["sample_count"].as_u64(), Some(8));
        assert_eq!(records.len(), 24);
        assert_eq!(restored_records.len(), records.len());
        for (record, restored_record) in records.iter().zip(restored_records.iter()) {
            assert_eq!(record["series_id"], restored_record["series_id"]);
            assert_eq!(record["timestamp"], restored_record["timestamp"]);
            assert_eq!(record["horizon"], restored_record["horizon"]);
            assert_eq!(record["sample"], restored_record["sample"]);
            for field in [
                "prediction",
                "mean",
                "residual_draw",
                "coefficient_draw",
                "trend_draw",
            ] {
                let left = record[field].as_f64().expect("numeric sample field");
                let right = restored_record[field]
                    .as_f64()
                    .expect("restored numeric sample field");
                assert!((left - right).abs() < 1.0e-12);
            }
        }
        assert!(records
            .iter()
            .any(|record| record["coefficient_draw"].as_f64().unwrap().abs() > 0.0));
        assert!(records
            .iter()
            .any(|record| record["residual_draw"].as_f64().unwrap().abs() > 0.0));
    }

    #[test]
    fn piecewise_linear_artifact_round_trips_fitted_state() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| {
                    let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                    ForecastRow::with_covariates(
                        "__single__",
                        ts(day),
                        50.0 + f64::from(day) + 20.0 * queue,
                        BTreeMap::from([("airport_queue".to_string(), queue)]),
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 1,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid config");

        model.fit(&frame).expect("fit");
        let before = model.predict(3).expect("predict before");
        let payload = model.to_json_string().expect("serialize artifact");
        let loaded =
            PiecewiseLinearSeasonalForecaster::from_json_string(&payload).expect("load artifact");
        let after = loaded.predict(3).expect("predict after");

        assert_eq!(before.to_json_value(), after.to_json_value());
        assert_eq!(
            serde_json::from_str::<Value>(&payload).expect("artifact json")["kind"].as_str(),
            Some(PIECEWISE_LINEAR_SEASONAL_ARTIFACT_KIND)
        );
    }

    #[test]
    fn piecewise_linear_seasonal_logistic_growth_respects_bounds() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 14.0;
                    let value = 5.0 + 90.0 / (1.0 + (-0.25 * t).exp());
                    ForecastRow::single(ts(day), value)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 4,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(7).expect("predict");

        assert_eq!(model.metadata()["growth"].as_str(), Some("logistic"));
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean > 0.0 && prediction.mean < 100.0));
    }

    #[test]
    fn piecewise_linear_logistic_trend_uncertainty_uses_inverse_link_scale() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 8.0;
                    let value = 100.0 / (1.0 + (-0.55 * t).exp());
                    ForecastRow::single(ts(day), value.clamp(0.001, 99.999))
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid saturated logistic frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 2,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            uncertainty_samples: 16,
            trend_uncertainty_scale: 10.0,
            uncertainty_seed: 13,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic uncertainty config");

        model.fit(&frame).expect("fit");
        let fitted = model.fitted.as_ref().expect("fitted");
        let series = fitted.series.get("__single__").expect("single series");
        let timestamp = ForecastFrequency::Daily
            .advance(series.last_timestamp, 3)
            .expect("future timestamp");
        let elapsed = elapsed_days(series.start_timestamp, timestamp);
        let bounds =
            piecewise_bounds(Some("__single__"), None, Some(3), &model.config).expect("bounds");
        let linear_predictor = predict_piecewise_linear_value(
            elapsed,
            &series.coefficients,
            &PiecewiseLinearFeatureContext {
                series_id: Some("__single__"),
                timestamp,
                covariates: None,
                horizon_step: Some(3),
                component_multiplier: series.component_multiplier(elapsed, bounds, &model.config),
                changepoints: &series.changepoints,
                config: &model.config,
                regressor_stats: Some(&series.regressor_stats),
            },
        )
        .expect("linear predictor");
        let derivative =
            inverse_piecewise_target_derivative(linear_predictor, bounds, &model.config);
        let offsets = series
            .trend_uncertainty_offsets("__single__", elapsed, timestamp, 3, &model.config)
            .expect("trend offsets");

        assert!(derivative < 1.0);
        assert_eq!(offsets.len(), 16);
        assert!(offsets.iter().all(|offset| offset.is_finite()));
    }

    #[test]
    fn piecewise_linear_logistic_predictive_samples_respect_bounds() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 12.0;
                    let value = 5.0 + 90.0 / (1.0 + (-0.35 * t).exp());
                    ForecastRow::single(ts(day), value)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid logistic frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            uncertainty_samples: 32,
            trend_uncertainty_scale: 20.0,
            coefficient_uncertainty_scale: 8.0,
            uncertainty_seed: 17,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic sample config");

        model.fit(&frame).expect("fit");
        let samples = model.predict_samples_json_value(5).expect("samples");
        let records = samples["records"].as_array().expect("sample records");

        assert_eq!(records.len(), 5 * 32);
        assert!(records.iter().all(|record| {
            let prediction = record["prediction"].as_f64().expect("sample prediction");
            prediction > 0.0 && prediction < 100.0
        }));
    }

    #[test]
    fn piecewise_linear_logistic_quantiles_respect_inverse_link_bounds() {
        let frame = ForecastFrame::new(
            (1..=28)
                .map(|day| {
                    let t = f64::from(day) - 12.0;
                    let value = 5.0 + 90.0 / (1.0 + (-0.35 * t).exp());
                    ForecastRow::single(ts(day), value)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid logistic frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            quantile_levels: vec![0.05, 0.5, 0.95],
            uncertainty_samples: 32,
            trend_uncertainty_scale: 20.0,
            coefficient_uncertainty_scale: 8.0,
            uncertainty_seed: 23,
            cap: Some(100.0),
            floor: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic quantile config");

        model.fit(&frame).expect("fit");
        let quantiles = model
            .predict_quantiles_json_value(5, None)
            .expect("quantiles");
        let records = quantiles["records"].as_array().expect("quantile records");

        assert_eq!(records.len(), 5 * 3);
        assert!(records.iter().all(|record| {
            let prediction = record["prediction"].as_f64().expect("quantile prediction");
            prediction > 0.0 && prediction < 100.0
        }));
    }

    #[test]
    fn piecewise_linear_logistic_prediction_intervals_respect_bounds() {
        let frame = ForecastFrame::new(
            (1..=24)
                .map(|day| {
                    let t = f64::from(day) - 10.0;
                    let noise = if day % 2 == 0 { 1.0 } else { -1.0 };
                    let value = 5.0 + 90.0 / (1.0 + (-0.4 * t).exp()) + noise;
                    ForecastRow::single(ts(day), value.clamp(5.001, 94.999))
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid logistic interval frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8, 0.95],
            uncertainty_samples: 32,
            trend_uncertainty_scale: 20.0,
            coefficient_uncertainty_scale: 8.0,
            uncertainty_seed: 19,
            cap: Some(95.0),
            floor: 5.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid logistic interval config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(5).expect("predict");

        assert_eq!(forecast.intervals().len(), 10);
        assert!(forecast
            .intervals()
            .iter()
            .all(|interval| interval.lower >= 5.0 && interval.upper <= 95.0));
    }

    #[test]
    fn piecewise_linear_seasonal_logistic_growth_uses_dynamic_capacity() {
        let rows = (1..=28)
            .map(|day| {
                let cap = 80.0 + f64::from(day);
                let t = f64::from(day) - 14.0;
                let target = cap / (1.0 + (-0.2 * t).exp());
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    target,
                    BTreeMap::from([("zone_capacity".to_string(), cap)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let future_caps = vec![109.0, 110.0, 111.0];
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap_regressor: Some("zone_capacity".to_string()),
            future_regressors: BTreeMap::from([("zone_capacity".to_string(), future_caps.clone())]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid dynamic cap config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");

        assert_eq!(
            model.metadata()["cap_regressor"].as_str(),
            Some("zone_capacity")
        );
        assert!(forecast
            .predictions()
            .iter()
            .zip(future_caps.iter())
            .all(|(prediction, cap)| prediction.mean > 0.0 && prediction.mean < *cap));
    }

    #[test]
    fn piecewise_linear_logistic_cap_regressor_can_be_series_specific() {
        let rows = ["A", "B"]
            .into_iter()
            .flat_map(|series| {
                (1..=28).map(move |day| {
                    let cap = if series == "A" {
                        110.0 + f64::from(day) * 0.25
                    } else {
                        65.0 + f64::from(day) * 0.10
                    };
                    let t = f64::from(day) - 14.0;
                    let target = cap / (1.0 + (-0.18 * t).exp());
                    ForecastRow::with_covariates(
                        series,
                        ts(day),
                        target,
                        BTreeMap::from([("zone_capacity".to_string(), cap)]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap_regressor: Some("zone_capacity".to_string()),
            future_regressors_by_series: BTreeMap::from([
                (
                    "A".to_string(),
                    BTreeMap::from([("zone_capacity".to_string(), vec![120.0])]),
                ),
                (
                    "B".to_string(),
                    BTreeMap::from([("zone_capacity".to_string(), vec![70.0])]),
                ),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid panel dynamic cap config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(1).expect("predict");
        let a = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A forecast")
            .mean;
        let b = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "B")
            .expect("series B forecast")
            .mean;

        assert!(a > 0.0 && a < 120.0);
        assert!(b > 0.0 && b < 70.0);
        assert!(a > b + 20.0);
    }

    #[test]
    fn piecewise_linear_logistic_floor_regressor_can_be_series_specific() {
        let cap = 140.0;
        let rows = ["A", "B"]
            .into_iter()
            .flat_map(|series| {
                (1..=28).map(move |day| {
                    let floor = if series == "A" {
                        32.0 + f64::from(day) * 0.10
                    } else {
                        8.0 + f64::from(day) * 0.05
                    };
                    let t = f64::from(day) - 14.0;
                    let target = floor + (cap - floor) / (1.0 + (-0.18 * t).exp());
                    ForecastRow::with_covariates(
                        series,
                        ts(day),
                        target,
                        BTreeMap::from([("service_floor".to_string(), floor)]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Logistic,
            changepoints: 3,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            cap: Some(cap),
            floor_regressor: Some("service_floor".to_string()),
            future_regressors_by_series: BTreeMap::from([
                (
                    "A".to_string(),
                    BTreeMap::from([("service_floor".to_string(), vec![38.0])]),
                ),
                (
                    "B".to_string(),
                    BTreeMap::from([("service_floor".to_string(), vec![10.0])]),
                ),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid panel dynamic floor config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(1).expect("predict");
        let a = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A forecast")
            .mean;
        let b = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "B")
            .expect("series B forecast")
            .mean;
        let mut lower_floor_model = model.clone();
        lower_floor_model
            .update_config(|config| {
                config.future_regressors_by_series.insert(
                    "A".to_string(),
                    BTreeMap::from([("service_floor".to_string(), vec![5.0])]),
                );
            })
            .expect("lower future floor config");
        let lower_floor_a = lower_floor_model
            .predict(1)
            .expect("lower floor predict")
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A lower floor forecast")
            .mean;

        assert!(a > 38.0 && a < cap);
        assert!(b > 10.0 && b < cap);
        assert!(a > lower_floor_a);
        assert_eq!(
            model.metadata()["floor_regressor"].as_str(),
            Some("service_floor")
        );
    }

    #[test]
    fn piecewise_linear_seasonal_explicit_changepoint_projects_break() {
        let rows = (1..=30)
            .map(|day| {
                let target = if day <= 15 {
                    50.0 + f64::from(day)
                } else {
                    65.0 + 5.0 * f64::from(day - 15)
                };
                ForecastRow::single(ts(day), target)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut explicit = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            changepoint_l2_regularization: 0.001,
            changepoint_timestamps: vec![ts(15)],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid explicit changepoint config");

        baseline.fit(&frame).expect("baseline fit");
        explicit.fit(&frame).expect("explicit fit");
        let baseline_forecast = baseline.predict(3).expect("baseline predict");
        let explicit_forecast = explicit.predict(3).expect("explicit predict");

        assert!(
            explicit_forecast.predictions()[2].mean > baseline_forecast.predictions()[2].mean + 8.0
        );
        assert_eq!(
            explicit.metadata()["changepoint_timestamps"][0].as_str(),
            Some("2026-01-15T00:00:00")
        );
    }

    #[test]
    fn piecewise_linear_seasonal_changepoint_l1_shrinks_deltas() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| ForecastRow::single(ts(day), 20.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 5,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            changepoint_l2_regularization: 0.001,
            changepoint_l1_regularization: 10.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid sparse changepoint config");

        model.fit(&frame).expect("fit");
        let fitted = model.fitted.as_ref().expect("fitted");
        let series = fitted.series.get("__single__").expect("single series");
        let max_delta = series.coefficients[2..2 + series.changepoints.len()]
            .iter()
            .map(|coefficient| coefficient.abs())
            .fold(0.0_f64, f64::max);

        assert!(max_delta < 1.0e-6);
        assert_eq!(
            model.metadata()["changepoint_l1_regularization"].as_f64(),
            Some(10.0)
        );
    }

    #[test]
    fn piecewise_linear_huber_loss_resists_large_outlier() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| {
                    let clean = 50.0 + 1.5 * f64::from(day);
                    let outlier = if day == 30 { 180.0 } else { 0.0 };
                    ForecastRow::single(ts(day), clean + outlier)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            changepoint_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut squared =
            PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("squared config");
        let mut huber = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            fit_loss: PiecewiseLinearFitLoss::Huber,
            huber_delta: 1.345,
            irls_iterations: 8,
            ..base_config
        })
        .expect("huber config");

        squared.fit(&frame).expect("squared fit");
        huber.fit(&frame).expect("huber fit");
        let squared_prediction = squared.predict(1).expect("squared predict").predictions()[0].mean;
        let huber_prediction = huber.predict(1).expect("huber predict").predictions()[0].mean;
        let clean_next = 50.0 + 1.5 * 31.0;

        assert!((huber_prediction - clean_next).abs() < (squared_prediction - clean_next).abs());
        assert_eq!(huber.metadata()["fit_loss"].as_str(), Some("huber"));
        assert_eq!(huber.metadata()["irls_iterations"].as_u64(), Some(8));
    }

    #[test]
    fn piecewise_linear_seasonal_rejects_invalid_changepoint_range() {
        let err = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoint_range: 0.0,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect_err("invalid range rejected");

        assert!(err.to_string().contains("changepoint_range"));
    }

    #[test]
    fn piecewise_linear_seasonal_event_window_carries_future_effect() {
        let train_event_timestamp = ts(15);
        let future_event_timestamp = NaiveDate::from_ymd_opt(2026, 2, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid event timestamp");
        let rows = (1..=30)
            .map(|day| {
                let event_boost = if (14..=16).contains(&day) { 25.0 } else { 0.0 };
                ForecastRow::single(ts(day), 100.0 + 0.5 * f64::from(day) + event_boost)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut event_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 0,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                event_l2_regularization: 0.001,
                events: vec![
                    PiecewiseLinearEvent {
                        name: "airport_surge".to_string(),
                        timestamp: train_event_timestamp,
                        lower_window: -1,
                        upper_window: 1,
                    },
                    PiecewiseLinearEvent {
                        name: "airport_surge".to_string(),
                        timestamp: future_event_timestamp,
                        lower_window: -1,
                        upper_window: 1,
                    },
                ],
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid event config");

        baseline.fit(&frame).expect("baseline fit");
        event_model.fit(&frame).expect("event fit");
        let baseline_forecast = baseline.predict(3).expect("baseline predict");
        let event_forecast = event_model.predict(3).expect("event predict");

        assert!(
            event_forecast.predictions()[1].mean > baseline_forecast.predictions()[1].mean + 10.0
        );
        assert_eq!(
            event_model.metadata()["events"][0]["name"].as_str(),
            Some("airport_surge")
        );
    }

    #[test]
    fn piecewise_linear_event_window_offsets_get_separate_effects() {
        let future_event_timestamp = NaiveDate::from_ymd_opt(2026, 2, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid event timestamp");
        let rows = (1..=30)
            .map(|day| {
                let event_offset = [10, 20]
                    .iter()
                    .find_map(|event_day| {
                        let offset = day - event_day;
                        (-1..=1).contains(&offset).then_some(offset)
                    })
                    .unwrap_or(99);
                let event_effect = match event_offset {
                    -1 => 4.0,
                    0 => 25.0,
                    1 => -12.0,
                    _ => 0.0,
                };
                ForecastRow::single(ts(day as u32), 100.0 + 0.1 * f64::from(day) + event_effect)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            event_l2_regularization: 0.001,
            events: vec![
                PiecewiseLinearEvent {
                    name: "airport_surge".to_string(),
                    timestamp: ts(10),
                    lower_window: -1,
                    upper_window: 1,
                },
                PiecewiseLinearEvent {
                    name: "airport_surge".to_string(),
                    timestamp: ts(20),
                    lower_window: -1,
                    upper_window: 1,
                },
                PiecewiseLinearEvent {
                    name: "airport_surge".to_string(),
                    timestamp: future_event_timestamp,
                    lower_window: -1,
                    upper_window: 1,
                },
            ],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid event window config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");
        let predictions = forecast.predictions();
        let components = model
            .predict_components_json_value(3)
            .expect("component forecast");
        let offset_components = components["records"][0]["components"]["event_window_offsets"]
            .as_object()
            .expect("event offset components");

        assert!(predictions[1].mean > predictions[0].mean + 12.0);
        assert!(predictions[0].mean > predictions[2].mean + 8.0);
        assert!(
            offset_components
                .get("airport_surge[-1]")
                .and_then(Value::as_f64)
                .expect("day-before contribution")
                > 2.0
        );
    }

    #[test]
    fn piecewise_linear_seasonal_extra_regressor_uses_future_values() {
        let rows = (1..=30)
            .map(|day| {
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    50.0 + f64::from(day) + 20.0 * queue,
                    BTreeMap::from([("airport_queue".to_string(), queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut regressor_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                changepoints: 0,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                regressor_l2_regularization: 0.001,
                extra_regressors: vec!["airport_queue".to_string()],
                future_regressors: BTreeMap::from([(
                    "airport_queue".to_string(),
                    vec![1.0, 0.0, 0.0],
                )]),
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid regressor config");

        baseline.fit(&frame).expect("baseline fit");
        regressor_model.fit(&frame).expect("regressor fit");
        let baseline_forecast = baseline.predict(3).expect("baseline predict");
        let regressor_forecast = regressor_model.predict(3).expect("regressor predict");

        assert!(
            regressor_forecast.predictions()[0].mean
                > baseline_forecast.predictions()[0].mean + 10.0
        );
        assert_eq!(
            regressor_model.metadata()["extra_regressors"][0].as_str(),
            Some("airport_queue")
        );
    }

    #[test]
    fn piecewise_linear_extra_regressor_future_values_can_be_series_specific() {
        let rows = ["A", "B"]
            .into_iter()
            .flat_map(|series| {
                (1..=30).map(move |day| {
                    let queue = if day % 3 == 0 { 1.0 } else { 0.0 };
                    ForecastRow::with_covariates(
                        series,
                        ts(day),
                        75.0 + f64::from(day) + 18.0 * queue,
                        BTreeMap::from([("airport_queue".to_string(), queue)]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            regressor_l2_regularization: 0.001,
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors_by_series: BTreeMap::from([
                (
                    "A".to_string(),
                    BTreeMap::from([("airport_queue".to_string(), vec![1.0])]),
                ),
                (
                    "B".to_string(),
                    BTreeMap::from([("airport_queue".to_string(), vec![0.0])]),
                ),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid per-series regressor config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(1).expect("predict");
        let a = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "A")
            .expect("series A forecast")
            .mean;
        let b = forecast
            .predictions()
            .iter()
            .find(|prediction| prediction.series_id == "B")
            .expect("series B forecast")
            .mean;

        assert!(a > b + 10.0);
        assert!(model.metadata()["future_regressors"]
            .as_object()
            .unwrap()
            .is_empty());
        assert_eq!(
            model.metadata()["future_regressors_by_series"]["A"]["airport_queue"][0].as_f64(),
            Some(1.0)
        );
    }

    #[test]
    fn piecewise_linear_extra_regressor_mode_can_be_multiplicative() {
        let rows = (1..=30)
            .map(|day| {
                let trend = 30.0 + 2.0 * f64::from(day);
                let queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    if queue > 0.0 { trend * 1.4 } else { trend },
                    BTreeMap::from([("airport_queue".to_string(), queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            regressor_l2_regularization: 0.001,
            extra_regressors: vec!["airport_queue".to_string()],
            regressor_modes: BTreeMap::from([(
                "airport_queue".to_string(),
                PiecewiseLinearComponentMode::Multiplicative,
            )]),
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0, 0.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid multiplicative regressor config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");

        assert!(forecast.predictions()[0].mean > forecast.predictions()[1].mean + 15.0);
        assert_eq!(
            model.metadata()["regressor_modes"]["airport_queue"].as_str(),
            Some("multiplicative")
        );
    }

    #[test]
    fn piecewise_linear_extra_regressor_monotonic_constraint_clamps_effect() {
        let rows = (1..=30)
            .map(|day| {
                let traffic = if day % 2 == 0 { 10.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    100.0 - 4.0 * traffic,
                    BTreeMap::from([("traffic".to_string(), traffic)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            growth: PiecewiseLinearGrowth::Flat,
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            regressor_l2_regularization: 0.0,
            extra_regressors: vec!["traffic".to_string()],
            extra_regressor_monotonic_constraints: BTreeMap::from([("traffic".to_string(), 1)]),
            future_regressors: BTreeMap::from([("traffic".to_string(), vec![0.0, 10.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid monotone regressor config");

        model.fit(&frame).expect("fit");
        let components = model
            .predict_components_json_value(2)
            .expect("component forecast");
        let records = components["records"].as_array().expect("component records");
        let low = records[0]["components"]["regressors"]["traffic"]
            .as_f64()
            .expect("low traffic contribution");
        let high = records[1]["components"]["regressors"]["traffic"]
            .as_f64()
            .expect("high traffic contribution");

        assert!(high >= low);
        assert_eq!(
            model.metadata()["extra_regressor_monotonic_constraints"]["traffic"].as_i64(),
            Some(1)
        );
    }

    #[test]
    fn piecewise_linear_per_regressor_l2_shrinks_named_effect() {
        let rows = (1..=30)
            .map(|day| {
                let airport_queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    50.0 + f64::from(day) + 24.0 * airport_queue,
                    BTreeMap::from([("airport_queue".to_string(), airport_queue)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            extra_regressors: vec!["airport_queue".to_string()],
            future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0])]),
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut low_l2 = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            regressor_l2_regularization_by_name: BTreeMap::from([(
                "airport_queue".to_string(),
                0.001,
            )]),
            ..base_config.clone()
        })
        .expect("valid low l2 config");
        let mut high_l2 = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            regressor_l2_regularization_by_name: BTreeMap::from([(
                "airport_queue".to_string(),
                1_000.0,
            )]),
            ..base_config
        })
        .expect("valid high l2 config");

        low_l2.fit(&frame).expect("low l2 fit");
        high_l2.fit(&frame).expect("high l2 fit");
        let low_prediction = low_l2.predict(1).expect("low predict").predictions()[0].mean;
        let high_prediction = high_l2.predict(1).expect("high predict").predictions()[0].mean;

        assert!(low_prediction > high_prediction + 10.0);
        assert_eq!(
            high_l2.metadata()["regressor_l2_regularization_by_name"]["airport_queue"].as_f64(),
            Some(1_000.0)
        );
    }

    #[test]
    fn piecewise_linear_auto_standardizes_continuous_extra_regressors_only() {
        let rows = (1..=30)
            .map(|day| {
                let traffic_index = 100.0 + 4.0 * f64::from(day);
                let airport_queue = if day % 5 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    ts(day),
                    20.0 + 0.3 * f64::from(day) + 1.5 * traffic_index + 12.0 * airport_queue,
                    BTreeMap::from([
                        ("traffic_index".to_string(), traffic_index),
                        ("airport_queue".to_string(), airport_queue),
                    ]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            extra_regressors: vec!["traffic_index".to_string(), "airport_queue".to_string()],
            future_regressors: BTreeMap::from([
                ("traffic_index".to_string(), vec![224.0]),
                ("airport_queue".to_string(), vec![1.0]),
            ]),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid standardized regressor config");

        model.fit(&frame).expect("fit");
        let fitted = model.fitted.as_ref().expect("fitted");
        let series = fitted.series.get("__single__").expect("single series");
        let traffic_stats = series
            .regressor_stats
            .get("traffic_index")
            .expect("traffic stats");
        let queue_stats = series
            .regressor_stats
            .get("airport_queue")
            .expect("queue stats");
        let payload = serde_json::from_str::<Value>(&model.to_json_string().expect("artifact"))
            .expect("artifact json");

        assert!(traffic_stats.standardized);
        assert!(traffic_stats.scale > 1.0);
        assert!(!queue_stats.standardized);
        assert_eq!(
            model.metadata()["regressor_standardization"].as_str(),
            Some("auto")
        );
        assert_eq!(
            payload["model"]["fitted"]["series"]["__single__"]["regressor_stats"]["traffic_index"]
                ["standardized"]
                .as_bool(),
            Some(true)
        );
    }

    #[test]
    fn piecewise_linear_trend_adjustments_shift_future_trend() {
        let frame = ForecastFrame::new(
            (1..=30)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut baseline =
            PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("baseline config");
        let mut adjusted = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            trend_adjustments: BTreeMap::from([(2, 1.20)]),
            ..base_config
        })
        .expect("adjusted config");

        baseline.fit(&frame).expect("baseline fit");
        adjusted.fit(&frame).expect("adjusted fit");
        let baseline_forecast = baseline.predict(2).expect("baseline predict");
        let adjusted_forecast = adjusted.predict(2).expect("adjusted predict");
        let baseline_second = baseline_forecast.predictions()[1].mean;
        let adjusted_second = adjusted_forecast.predictions()[1].mean;
        let components = adjusted
            .predict_components_json_value(2)
            .expect("components");
        let second_record = &components["records"][1];

        assert!(adjusted_second > baseline_second + 7.0);
        assert_eq!(
            second_record["trend_adjustment_multiplier"].as_f64(),
            Some(1.20)
        );
        assert!(
            second_record["adjusted_trend"].as_f64().unwrap()
                > second_record["trend"].as_f64().unwrap()
        );
    }

    #[test]
    fn piecewise_linear_residual_shock_passes_recent_signed_residuals_forward() {
        let rows = (1..=24)
            .map(|day| {
                let shock = if day >= 22 { 12.0 } else { 0.0 };
                ForecastRow::single(ts(day), 20.0 + 0.5 * f64::from(day) + shock)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut baseline =
            PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("baseline config");
        let mut shock_model =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                residual_shock_window: 3,
                residual_shock_scale: 0.8,
                residual_shock_decay: 0.5,
                ..base_config
            })
            .expect("shock config");

        baseline.fit(&frame).expect("baseline fit");
        shock_model.fit(&frame).expect("shock fit");
        let baseline_forecast = baseline.predict(2).expect("baseline predict");
        let shock_forecast = shock_model.predict(2).expect("shock predict");
        let components = shock_model
            .predict_components_json_value(2)
            .expect("components");

        assert!(shock_forecast.predictions()[0].mean > baseline_forecast.predictions()[0].mean);
        assert!(shock_forecast.predictions()[1].mean > baseline_forecast.predictions()[1].mean);
        assert!(
            components["records"][0]["residual_shock"].as_f64().unwrap()
                > components["records"][1]["residual_shock"].as_f64().unwrap()
        );
        assert_eq!(
            shock_model.metadata()["residual_shock_window"].as_u64(),
            Some(3)
        );
    }

    #[test]
    fn piecewise_linear_seasonal_prediction_intervals_render_columns() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| {
                    let noise = if day % 2 == 0 { 2.0 } else { -2.0 };
                    ForecastRow::single(ts(day), 25.0 + f64::from(day) + noise)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            interval_levels: vec![0.8],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid interval config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let json = forecast.to_json_value();
        let records = json["records"].as_array().expect("records");

        assert_eq!(forecast.intervals().len(), 2);
        assert!(json["columns"]
            .as_array()
            .expect("columns")
            .iter()
            .any(|column| column.as_str() == Some("prediction_lower_p80")));
        assert!(records[0]["prediction_lower_p80"].as_f64().is_some());
        assert!(
            records[0]["prediction_lower_p80"].as_f64().unwrap()
                <= records[0]["prediction_upper_p80"].as_f64().unwrap()
        );
    }

    #[test]
    fn piecewise_linear_seasonal_custom_fourier_period_projects_cycle() {
        let frame = ForecastFrame::new(
            (1..=56)
                .map(|day| {
                    let biweekly = if day % 14 == 0 { 18.0 } else { 0.0 };
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day - 1)),
                        80.0 + 0.25 * f64::from(day) + biweekly,
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut baseline = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid baseline config");
        let mut custom = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            seasonality_l2_regularization: 0.001,
            custom_seasonalities: vec![PiecewiseLinearSeasonality {
                name: "biweekly_pickup_cycle".to_string(),
                period_days: 14.0,
                fourier_order: 4,
                mode: None,
                condition_name: None,
                l2_regularization: None,
            }],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid custom seasonality config");

        baseline.fit(&frame).expect("baseline fit");
        custom.fit(&frame).expect("custom fit");
        let baseline_forecast = baseline.predict(14).expect("baseline predict");
        let custom_forecast = custom.predict(14).expect("custom predict");

        assert!(
            custom_forecast.predictions()[13].mean > baseline_forecast.predictions()[13].mean + 8.0
        );
        assert_eq!(
            custom.metadata()["custom_seasonalities"][0]["name"].as_str(),
            Some("biweekly_pickup_cycle")
        );
    }

    #[test]
    fn piecewise_linear_builtin_seasonality_l2_can_target_weekly_terms() {
        let frame = ForecastFrame::new(
            (1..=56)
                .map(|day| {
                    let weekly = if day % 7 == 0 { 18.0 } else { 0.0 };
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day - 1)),
                        60.0 + 0.2 * f64::from(day) + weekly,
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let base_config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 3,
            seasonality_l2_regularization: 0.001,
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut low_l2 = PiecewiseLinearSeasonalForecaster::new(base_config.clone())
            .expect("valid low l2 config");
        let mut high_l2 = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            weekly_l2_regularization: Some(1_000.0),
            ..base_config
        })
        .expect("valid high weekly l2 config");

        low_l2.fit(&frame).expect("low l2 fit");
        high_l2.fit(&frame).expect("high l2 fit");
        let low_components = low_l2
            .predict_components_json_value(7)
            .expect("low components");
        let high_components = high_l2
            .predict_components_json_value(7)
            .expect("high components");
        let low_weekly = low_components["records"][6]["components"]["weekly"]
            .as_f64()
            .expect("low weekly contribution")
            .abs();
        let high_weekly = high_components["records"][6]["components"]["weekly"]
            .as_f64()
            .expect("high weekly contribution")
            .abs();

        assert!(low_weekly > high_weekly + 8.0);
        assert_eq!(
            high_l2.metadata()["weekly_l2_regularization"].as_f64(),
            Some(1_000.0)
        );
    }

    #[test]
    fn piecewise_linear_custom_seasonality_mode_can_be_multiplicative() {
        let frame = ForecastFrame::new(
            (1..=56)
                .map(|day| {
                    let trend = 40.0 + f64::from(day);
                    let multiplier = if day % 14 == 0 { 1.35 } else { 1.0 };
                    ForecastRow::single(
                        ts(1) + Duration::days(i64::from(day - 1)),
                        trend * multiplier,
                    )
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            seasonality_l2_regularization: 0.001,
            custom_seasonalities: vec![PiecewiseLinearSeasonality {
                name: "biweekly_pickup_multiplier".to_string(),
                period_days: 14.0,
                fourier_order: 4,
                mode: Some(PiecewiseLinearComponentMode::Multiplicative),
                condition_name: None,
                l2_regularization: None,
            }],
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid custom seasonality config");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(14).expect("predict");

        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.mean.is_finite()));
        assert_eq!(
            model.metadata()["custom_seasonalities"][0]["mode"].as_str(),
            Some("multiplicative")
        );
    }

    #[test]
    fn piecewise_linear_custom_seasonality_condition_gates_fourier_terms() {
        let start = ts(1);
        let rows = (1..=42)
            .map(|day| {
                let rush_hour = if day % 2 == 0 { 1.0 } else { 0.0 };
                let cycle = if day % 7 == 0 { 16.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "__single__",
                    start + Duration::days(i64::from(day - 1)),
                    80.0 + 0.2 * f64::from(day) + rush_hour * cycle,
                    BTreeMap::from([("rush_hour".to_string(), rush_hour)]),
                )
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let config = PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            seasonality_l2_regularization: 0.001,
            custom_seasonalities: vec![PiecewiseLinearSeasonality {
                name: "rush_hour_weekly".to_string(),
                period_days: 7.0,
                fourier_order: 3,
                mode: None,
                condition_name: Some("rush_hour".to_string()),
                l2_regularization: None,
            }],
            future_regressors: BTreeMap::from([(
                "rush_hour".to_string(),
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )]),
            ..PiecewiseLinearSeasonalConfig::default()
        };
        let mut inactive = PiecewiseLinearSeasonalForecaster::new(config.clone())
            .expect("valid inactive conditional seasonality config");
        let mut active = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            future_regressors: BTreeMap::from([(
                "rush_hour".to_string(),
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )]),
            ..config
        })
        .expect("valid active conditional seasonality config");

        inactive.fit(&frame).expect("inactive fit");
        active.fit(&frame).expect("active fit");
        let inactive_forecast = inactive.predict(7).expect("inactive predict");
        let active_forecast = active.predict(7).expect("active predict");

        assert!(
            active_forecast.predictions()[6].mean > inactive_forecast.predictions()[6].mean + 4.0,
            "active conditional seasonality should lift matching future periods"
        );
        assert_eq!(
            active.metadata()["custom_seasonalities"][0]["condition_name"].as_str(),
            Some("rush_hour")
        );
    }

    #[test]
    fn piecewise_linear_seasonal_multiplicative_event_scales_with_trend() {
        let future_event_timestamp = NaiveDate::from_ymd_opt(2026, 2, 1)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid event timestamp");
        let rows = (1..=30)
            .map(|day| {
                let trend = 20.0 + 2.0 * f64::from(day);
                let target = if (14..=16).contains(&day) {
                    trend * 1.5
                } else {
                    trend
                };
                ForecastRow::single(ts(day), target)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
        let event_config = vec![
            PiecewiseLinearEvent {
                name: "airport_surge".to_string(),
                timestamp: ts(15),
                lower_window: -1,
                upper_window: 1,
            },
            PiecewiseLinearEvent {
                name: "airport_surge".to_string(),
                timestamp: future_event_timestamp,
                lower_window: -1,
                upper_window: 1,
            },
        ];
        let mut additive = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
            changepoints: 0,
            weekly_fourier_order: 0,
            auto_weekly_seasonality: false,
            event_l2_regularization: 0.001,
            events: event_config.clone(),
            ..PiecewiseLinearSeasonalConfig::default()
        })
        .expect("valid additive config");
        let mut multiplicative =
            PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
                component_mode: PiecewiseLinearComponentMode::Multiplicative,
                changepoints: 0,
                weekly_fourier_order: 0,
                auto_weekly_seasonality: false,
                event_l2_regularization: 0.001,
                events: event_config,
                ..PiecewiseLinearSeasonalConfig::default()
            })
            .expect("valid multiplicative config");

        additive.fit(&frame).expect("additive fit");
        multiplicative.fit(&frame).expect("multiplicative fit");
        let additive_forecast = additive.predict(3).expect("additive predict");
        let multiplicative_forecast = multiplicative.predict(3).expect("multiplicative predict");

        assert!(
            multiplicative_forecast.predictions()[1].mean
                > additive_forecast.predictions()[1].mean + 5.0
        );
        assert_eq!(
            multiplicative.metadata()["component_mode"].as_str(),
            Some("multiplicative")
        );
    }

    #[test]
    fn optimized_theta_selects_from_grid_deterministically() {
        let frame = ForecastFrame::new(
            (1..=6)
                .map(|day| ForecastRow::single(ts(day), f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            OptimizedThetaForecaster::new(vec![1.0, 2.0], vec![0.2, 0.8]).expect("valid grid");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(model.selected_theta(), Some(1.0 | 2.0)));
        assert!(matches!(model.selected_alpha(), Some(0.2 | 0.8)));
        assert_eq!(model.validation_scores().len(), 4);
        assert!(model.validation_scores().iter().any(|left| {
            model
                .validation_scores()
                .iter()
                .any(|right| left.theta != right.theta && (left.mse - right.mse).abs() > 1.0e-12)
        }));
        assert_eq!(forecast.predictions().len(), 2);
    }

    #[test]
    fn theta_forecast_matches_ses_drift_equivalence_reference() {
        let values = [3.0, 5.0, 4.0, 8.0, 10.0];
        let component = fit_theta_component(&values, 2.0, 0.4);

        assert!((forecast_theta_component(&component, 1) - 9.27656).abs() < 1.0e-10);
        assert!((forecast_theta_component(&component, 2) - 10.12656).abs() < 1.0e-10);
    }

    #[test]
    fn auto_kalman_selects_variances_and_predicts_with_auto_name() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 20.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoKalmanForecaster::with_grids(
            vec![0.001, 0.01],
            vec![0.0001, 0.001],
            vec![0.1, 1.0],
            Some(2),
        )
        .expect("valid auto kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let metadata = model.metadata();

        assert!(model.selected_params().is_some());
        assert_eq!(model.validation_scores().len(), 8);
        assert!(model
            .validation_scores()
            .iter()
            .all(|score| score.negative_log_likelihood.is_finite()));
        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.predictions()[0].model, "auto_kalman");
        assert!(
            metadata["validation_scores"]
                .as_array()
                .expect("scores")
                .len()
                == 8
        );
    }

    #[test]
    fn auto_kalman_uses_predictive_likelihood_to_identify_variance_scale() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    ForecastRow::single(ts(day), 20.0 + 1.5 * f64::from(day) + f64::from(day % 2))
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoKalmanForecaster::with_grids(
            vec![0.1, 10.0],
            vec![0.01, 1.0],
            vec![1.0, 100.0],
            Some(2),
        )
        .expect("valid grid");

        model.fit(&frame).expect("fit");
        let base = model
            .validation_scores()
            .iter()
            .find(|score| {
                score.params
                    == (KalmanParameterSet {
                        level_process_variance: 0.1,
                        trend_process_variance: 0.01,
                        observation_variance: 1.0,
                    })
            })
            .expect("base scale");
        let scaled = model
            .validation_scores()
            .iter()
            .find(|score| {
                score.params
                    == (KalmanParameterSet {
                        level_process_variance: 10.0,
                        trend_process_variance: 1.0,
                        observation_variance: 100.0,
                    })
            })
            .expect("scaled variance");

        assert!((base.mse - scaled.mse).abs() < 1.0e-10);
        assert!((base.negative_log_likelihood - scaled.negative_log_likelihood).abs() > 1.0e-3);
    }

    #[test]
    fn auto_kalman_rejects_empty_grid_and_requires_real_holdout() {
        assert!(
            AutoKalmanForecaster::with_grids(Vec::new(), vec![0.001], vec![1.0], Some(1),).is_err()
        );

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            AutoKalmanForecaster::with_grids(vec![0.001], vec![0.0001], vec![1.0], Some(1))
                .expect("valid grid");

        assert!(model.fit(&frame).is_err());

        let three_rows = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
                ForecastRow::single(ts(3), 14.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut oversized =
            AutoKalmanForecaster::with_grids(vec![0.001], vec![0.0001], vec![1.0], Some(2))
                .expect("valid grid");
        assert!(oversized.fit(&three_rows).is_err());
    }

    #[test]
    fn kalman_rejects_single_observation_instead_of_changing_models() {
        let frame = ForecastFrame::new(
            vec![ForecastRow::single(ts(1), 12.5)],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = KalmanForecaster::new(0.01, 0.001, 0.1).expect("valid kalman");

        assert!(model.fit(&frame).is_err());
    }

    #[test]
    fn local_level_kalman_forecasts_flat_panel_levels() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PULocationID=1", ts(1), 10.0),
                ForecastRow::new("PULocationID=1", ts(2), 11.0),
                ForecastRow::new("PULocationID=2", ts(1), 30.0),
                ForecastRow::new("PULocationID=2", ts(2), 31.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = LocalLevelKalmanForecaster::new(0.01, 0.1).expect("valid kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(forecast.predictions().len(), 4);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.model == "local_level_kalman"));
    }

    #[test]
    fn auto_local_level_kalman_selects_variances() {
        let frame = ForecastFrame::new(
            (1..=6)
                .map(|day| ForecastRow::single(ts(day), 20.0 + f64::from(day % 2)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            AutoLocalLevelKalmanForecaster::with_grids(vec![0.001, 0.01], vec![0.1, 1.0], Some(2))
                .expect("valid auto kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(model.selected_params().is_some());
        assert_eq!(model.validation_scores().len(), 4);
        assert!(model
            .validation_scores()
            .iter()
            .all(|score| score.negative_log_likelihood.is_finite()));
        assert_eq!(forecast.predictions()[0].model, "auto_local_level_kalman");
    }

    #[test]
    fn auto_local_level_kalman_requires_real_holdout() {
        let frame = ForecastFrame::new(
            vec![ForecastRow::single(ts(1), 10.0)],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoLocalLevelKalmanForecaster::with_grids(vec![0.001], vec![0.1], Some(3))
            .expect("valid auto kalman");

        assert!(model.fit(&frame).is_err());
    }

    #[test]
    fn ets_forecasts_panel_series_with_daily_timestamps() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PULocationID=1", ts(1), 10.0),
                ForecastRow::new("PULocationID=1", ts(2), 12.0),
                ForecastRow::new("PULocationID=1", ts(3), 14.0),
                ForecastRow::new("PULocationID=1", ts(4), 16.0),
                ForecastRow::new("PULocationID=2", ts(1), 30.0),
                ForecastRow::new("PULocationID=2", ts(2), 29.0),
                ForecastRow::new("PULocationID=2", ts(3), 28.0),
                ForecastRow::new("PULocationID=2", ts(4), 27.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::new(0.6, 0.2).expect("valid ets");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        let predictions = forecast.predictions();
        assert_eq!(predictions.len(), 4);
        assert_eq!(predictions[0].series_id, "PULocationID=1");
        assert_eq!(predictions[0].timestamp, ts(5));
        assert_eq!(predictions[1].horizon, 2);
        assert_eq!(predictions[2].series_id, "PULocationID=2");
        assert!(predictions[0].mean > 16.0);
        assert!(predictions[2].mean < 27.0);
        assert_eq!(
            model.fitted_values("PULocationID=1").expect("fitted").len(),
            4
        );
        assert_eq!(
            model.level_values("PULocationID=1").expect("levels").len(),
            4
        );
        assert_eq!(
            model.trend_values("PULocationID=1").expect("trends").len(),
            4
        );
        assert_eq!(
            model
                .seasonal_values("PULocationID=1")
                .expect("seasonals")
                .len(),
            4
        );
    }

    #[test]
    fn ets_additive_seasonality_repeats_pattern() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    let seasonal = if day % 2 == 0 { 4.0 } else { -4.0 };
                    ForecastRow::single(ts(day), 50.0 + seasonal)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::with_additive_seasonality(0.5, 0.0, Some(0.5), Some(2))
            .expect("valid seasonal ets");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| row.mean)
            .collect::<Vec<_>>();

        assert_eq!(forecast.predictions()[0].timestamp, ts(9));
        assert!(means[1] > means[0]);
        let seasonals = model
            .seasonal_values("__single__")
            .expect("seasonal contributions");
        assert_eq!(seasonals.len(), 8);
        assert!(seasonals[1] > seasonals[0]);
    }

    #[test]
    fn ets_rejects_invalid_params_and_short_seasonal_history() {
        assert!(ETSForecaster::new(0.0, 0.2).is_err());
        assert!(ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), None).is_err());

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 3.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), Some(2))
            .expect("valid seasonal ets");
        assert!(model.fit(&frame).is_err());
    }

    #[test]
    fn ets_and_auto_ets_reject_single_observation() {
        let frame = ForecastFrame::new(
            vec![ForecastRow::single(ts(1), 42.0)],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut ets = ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), Some(7))
            .expect("valid seasonal ets");
        let mut auto_ets = AutoETSForecaster::with_grids(
            vec![0.3, 0.5],
            vec![0.0, 0.1],
            vec![Some(0.0), Some(0.2)],
            vec![0.9, 1.0],
            Some(7),
        )
        .expect("valid auto ets");

        assert!(ets.fit(&frame).is_err());
        assert!(auto_ets.fit(&frame).is_err());
    }

    #[test]
    fn arima_forecasts_differenced_linear_series() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day) * 3.0))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(0, 1, 0).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.timestamp, row.horizon, row.mean))
            .collect::<Vec<_>>();

        assert_eq!(means[0].0, ts(9));
        assert_eq!(means[2].1, 3);
        assert!((means[0].2 - 37.0).abs() < 1.0e-6);
        assert!((means[2].2 - 43.0).abs() < 1.0e-6);
        assert_eq!(model.residuals("__single__").expect("residuals").len(), 7);
    }

    #[test]
    fn arima_forecasts_each_panel_series_without_bleeding() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 13.0),
                ForecastRow::new("PU1->DO2", ts(3), 16.0),
                ForecastRow::new("PU1->DO2", ts(4), 19.0),
                ForecastRow::new("PU9->DO8", ts(1), 40.0),
                ForecastRow::new("PU9->DO8", ts(2), 38.0),
                ForecastRow::new("PU9->DO8", ts(3), 36.0),
                ForecastRow::new("PU9->DO8", ts(4), 34.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(0, 1, 0).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.series_id.as_str(), row.timestamp, row.mean))
            .collect::<Vec<_>>();

        assert_eq!(means.len(), 4);
        assert_eq!(means[0].0, "PU1->DO2");
        assert_eq!(means[0].1, ts(5));
        assert_eq!(means[2].0, "PU9->DO8");
        assert!(means[0].2 > 19.0);
        assert!(means[2].2 < 34.0);
    }

    #[test]
    fn arima_rejects_invalid_or_unsupported_explicit_order() {
        assert!(ArimaForecaster::new(9, 0, 0).is_err());
        assert!(ArimaForecaster::new(1, 0, 9).is_err());

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(2, 0, 0).expect("valid arima");
        assert!(model.fit(&frame).is_err());
    }

    #[test]
    fn arima_supports_moving_average_terms() {
        let frame = ForecastFrame::new(
            (1..=10)
                .map(|day| {
                    let shock = if day % 3 == 0 { 2.0 } else { -1.0 };
                    ForecastRow::single(ts(day), 20.0 + f64::from(day) + shock)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(1, 0, 1).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(model.order(), (1, 0, 1));
        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));
    }

    #[test]
    fn arima_candidate_score_excludes_warmup_residuals() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 20.0 + f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let state = FittedArimaState::from_frame(&frame, 2, 0, 1).expect("fit arima");
        let series = state.series.get("__single__").expect("single series");

        let expected = series
            .residuals
            .iter()
            .skip(2)
            .map(|residual| residual * residual)
            .sum::<f64>()
            / 6.0;

        assert_eq!(series.score_start, 2);
        assert!((state.mean_squared_residual() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn auto_arima_rejects_unstable_ar_recursions() {
        let values = [
            36.893, 37.770, 38.912, 33.241, 32.972, 34.398, 35.728, 36.870, 37.747, 45.299, 45.490,
            41.305, 40.759, 39.888, 35.752, 34.428, 33.008, 31.587, 30.264, 29.127, 31.256, 30.710,
            30.526, 27.716, 35.268, 36.145, 37.288, 31.617, 33.044, 34.470, 35.800, 36.942, 37.819,
            45.371, 45.562, 41.377, 40.831, 39.960, 35.824, 34.500, 33.080, 31.659, 30.336, 29.199,
            31.328, 30.782, 30.598, 27.788, 35.340, 36.217, 37.360, 31.689, 32.812, 34.238, 35.568,
            36.710, 37.587, 45.139, 45.330, 41.145, 40.599, 39.728, 35.592, 34.268, 32.848, 31.427,
            30.104, 28.968, 31.097, 30.550, 30.366, 27.556, 35.109, 35.986, 37.128, 31.457, 34.999,
            36.425, 37.755, 38.897, 39.774, 47.326, 47.517, 43.332, 42.786, 41.915, 37.779, 36.455,
            35.035, 33.614, 32.291, 31.155, 33.284, 32.737, 32.553, 29.743,
        ];
        let start = ts(1);
        let frame = ForecastFrame::new(
            values
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    ForecastRow::single(start + Duration::hours(idx as i64), *value)
                })
                .collect(),
            ForecastFrequency::Hourly,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(2, 1, 1).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(14).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|prediction| prediction.mean)
            .collect::<Vec<_>>();

        let selected = model.selected_order().expect("selected order");
        let selected_score = model
            .validation_scores()
            .iter()
            .find(|score| (score.p, score.d, score.q) == selected)
            .expect("selected validation score");
        assert!(selected_score.ar_stable);
        assert!(selected_score.ma_invertible);
        assert!(model.validation_scores().iter().any(|score| !score.stable));
        assert!(means.iter().all(|mean| mean.is_finite() && *mean < 60.0));
        assert!(means.iter().all(|mean| *mean > 0.0));
    }

    #[test]
    fn ar_stability_uses_exact_schur_recursion() {
        assert!(ar_recursion_is_stable(&[]));
        assert!(ar_recursion_is_stable(&[0.99]));
        assert!(!ar_recursion_is_stable(&[1.01]));
        assert!(!ar_recursion_is_stable(&[1.0]));
        assert!(ar_recursion_is_stable(&[1.5, -0.75]));
        assert!(ma_recursion_is_invertible(&[0.99]));
        assert!(!ma_recursion_is_invertible(&[1.01]));
        assert!(!ma_recursion_is_invertible(&[-1.0]));
    }

    #[test]
    fn auto_arima_scores_all_differencing_orders_on_original_scale() {
        let frame = ForecastFrame::new(
            (1..=10)
                .map(|day| ForecastRow::single(ts(day), f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(0, 1, 0).expect("auto arima");

        model.fit(&frame).expect("fit");

        let level = model
            .validation_scores()
            .iter()
            .find(|score| (score.p, score.d, score.q) == (0, 0, 0))
            .expect("level score");
        let differenced = model
            .validation_scores()
            .iter()
            .find(|score| (score.p, score.d, score.q) == (0, 1, 0))
            .expect("difference score");
        assert!(differenced.mse < 1.0e-12);
        assert!(level.mse > differenced.mse);
        assert_eq!(model.selected_order(), Some((0, 1, 0)));
    }

    #[test]
    fn explicit_arima_rejects_non_stationary_fitted_coefficients() {
        let frame = ForecastFrame::new(
            (0..8)
                .map(|idx| ForecastRow::single(ts(idx + 1), 2.0_f64.powi(idx as i32)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(1, 0, 0).expect("arima config");

        let error = model
            .fit(&frame)
            .expect_err("explosive AR fit must be rejected");

        assert!(error.to_string().contains("non-stationary AR polynomial"));
    }

    #[test]
    fn explicit_arima_checks_stationarity_after_differencing() {
        let frame = ForecastFrame::new(
            [0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0]
                .into_iter()
                .enumerate()
                .map(|(idx, value)| ForecastRow::single(ts(idx as u32 + 1), value))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(1, 1, 0).expect("arima config");

        let error = model
            .fit(&frame)
            .expect_err("explosive recursion on the differenced series must be rejected");

        assert!(error.to_string().contains("non-stationary AR polynomial"));
    }

    #[test]
    fn auto_arima_selects_candidate_and_predicts_with_model_name() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(2, 1, 1).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(
            model.selected_order(),
            Some((0..=2, 0..=1, 0..=1))
        ));
        assert_eq!(model.validation_scores().len(), 12);
        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.predictions()[0].model, "auto_arima");
        assert_eq!(forecast.predictions()[0].timestamp, ts(9));
    }

    #[test]
    fn auto_arima_deduplicates_orders_after_short_history_pruning() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(3, 1, 2).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(
            model.selected_order(),
            Some((0..=1, 0..=1, 0..=1))
        ));
        assert!(model.validation_scores().len() <= 8);
        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.model == "auto_arima" && prediction.mean.is_finite()));
    }

    #[test]
    fn local_seasonal_and_window_models_reject_short_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 20.0),
                ForecastRow::new("PU1->DO2", ts(3), 30.0),
                ForecastRow::new("PU9->DO8", ts(1), 4.0),
                ForecastRow::new("PU9->DO8", ts(2), 6.0),
                ForecastRow::new("PU9->DO8", ts(3), 8.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");

        let mut seasonal_naive = SeasonalNaiveForecaster::new(24).expect("seasonal naive");
        assert!(seasonal_naive.fit(&frame).is_err());

        let mut window = WindowAverageForecaster::new(24).expect("window average");
        assert!(window.fit(&frame).is_err());

        let mut seasonal_window =
            SeasonalWindowAverageForecaster::new(24, 3).expect("seasonal window average");
        assert!(seasonal_window.fit(&frame).is_err());
    }

    #[test]
    fn theta_rejects_unsupported_seasonality() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 11.0),
                ForecastRow::new("PU1->DO2", ts(3), 12.0),
                ForecastRow::new("PU9->DO8", ts(1), 30.0),
                ForecastRow::new("PU9->DO8", ts(2), 29.0),
                ForecastRow::new("PU9->DO8", ts(3), 28.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::additive(24).expect("seasonality");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.3, Some(seasonality)).expect("theta");

        assert!(model.fit(&frame).is_err());
    }

    #[test]
    fn kriging_models_record_selected_backend() {
        let coordinates = BTreeMap::from([
            ("a".to_string(), (0.0, 0.0)),
            ("b".to_string(), (1.0, 0.0)),
        ]);
        let kriging = KrigingForecaster::with_config_and_backend(
            coordinates.clone(),
            OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("kriging config"),
            Some("cpu"),
        )
        .expect("kriging backend");
        assert_eq!(kriging.backend().requested, "cpu");
        assert_eq!(kriging.metadata()["backend"]["selected"], "cpu");

        let spatial = SpatialPiecewiseKrigingForecaster::new_with_backend(
            SpatialPiecewiseKrigingConfig {
                coordinates,
                mode: SpatialPiecewiseKrigingMode::ResidualKriging,
                piecewise_config: PiecewiseLinearSeasonalConfig::default(),
                kriging_config: OrdinaryKrigingConfig::new(1.0, 1.0e-6)
                    .expect("kriging config"),
                spatial_regressors: Vec::new(),
                residual_shrinkage: 1.0,
                allow_neighbor_fallback: false,
            },
            Some("cpu"),
        )
        .expect("spatial kriging backend");
        assert_eq!(spatial.backend().selected, "cpu");
        assert_eq!(spatial.metadata()["backend"]["requested"], "cpu");
    }
}
