use cartoboost_core::forecasting::{
    evaluate_competition_metrics, CandidateValidationCutoffSchedule, CrostonForecaster,
    ForecastActual, ForecastFrame, ForecastFrameMetadata, ForecastFrequency, ForecastRegistry,
    ForecastResult, ForecastRow, ForecastWindow, Forecaster, IntermittentDemandForecaster,
    NaiveForecaster, RollingOriginBacktester, RollingOriginSplitter, SbaForecaster,
    SeasonalNaiveForecaster, TsbForecaster,
};
use chrono::{NaiveDate, NaiveDateTime};
use std::collections::BTreeMap;

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

fn hour(day: u32, hour: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(hour, 0, 0))
        .expect("valid fixture timestamp")
}

#[test]
fn validates_and_sorts_panel_frame() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("B", ts(2), 20.0),
            ForecastRow::new("A", ts(1), 1.0),
            ForecastRow::new("B", ts(1), 10.0),
            ForecastRow::new("A", ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");

    let keys = frame
        .rows()
        .iter()
        .map(|row| (row.series_id.as_str(), row.timestamp))
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![("A", ts(1)), ("A", ts(2)), ("B", ts(1)), ("B", ts(2))]
    );
}

#[test]
fn default_registry_models_return_finite_forecasts_on_representative_panels() {
    let panels = [
        (
            "short_route_panel",
            ForecastFrame::new(
                vec![
                    ForecastRow::new("PULocationID=132|DOLocationID=237", ts(1), 8.492),
                    ForecastRow::new("PULocationID=132|DOLocationID=237", ts(2), 9.108),
                    ForecastRow::new("PULocationID=237|DOLocationID=236", ts(1), 8.377),
                    ForecastRow::new("PULocationID=237|DOLocationID=236", ts(2), 8.941),
                ],
                ForecastFrequency::Daily,
            )
            .expect("valid short route panel"),
        ),
        (
            "hourly_taxi_panel",
            ForecastFrame::new(
                (0..8)
                    .flat_map(|hour_idx| {
                        let hour = hour(1, hour_idx);
                        [
                            ForecastRow::new(
                                "PULocationID=132|DOLocationID=237",
                                hour,
                                8.0 + f64::from(hour_idx) * 0.42,
                            ),
                            ForecastRow::new(
                                "PULocationID=237|DOLocationID=236",
                                hour,
                                11.0 + f64::from(hour_idx % 3) * 0.31,
                            ),
                        ]
                    })
                    .collect(),
                ForecastFrequency::Hourly,
            )
            .expect("valid hourly taxi panel"),
        ),
    ];
    let registry = ForecastRegistry::with_defaults().expect("default registry");

    for (panel_name, frame) in panels {
        let series_count = frame
            .rows()
            .iter()
            .map(|row| row.series_id.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        for model_name in registry.names() {
            let mut model = registry
                .create(&model_name)
                .unwrap_or_else(|err| panic!("{model_name} create failed: {err}"));
            model
                .fit(&frame)
                .unwrap_or_else(|err| panic!("{model_name} fit failed on {panel_name}: {err}"));
            let forecast = model
                .predict(2)
                .unwrap_or_else(|err| panic!("{model_name} predict failed on {panel_name}: {err}"));
            assert_eq!(
                forecast.predictions().len(),
                series_count * 2,
                "{model_name} returned wrong prediction count on {panel_name}",
            );
            assert!(
                forecast.predictions().iter().all(|prediction| {
                    !prediction.model.is_empty()
                        && prediction.horizon > 0
                        && prediction.mean.is_finite()
                }),
                "{model_name} returned an empty, invalid, or non-finite prediction on {panel_name}",
            );
        }
    }
}

#[test]
fn rejects_duplicate_panel_timestamp() {
    let err = ForecastFrame::new(
        vec![
            ForecastRow::new("A", ts(1), 1.0),
            ForecastRow::new("A", ts(1), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect_err("duplicate should be rejected");
    assert!(err.to_string().contains("duplicate forecast timestamp"));
}

#[test]
fn weighted_string_rows_collapse_duplicate_panel_timestamps() {
    let frame = ForecastFrame::from_string_rows_with_covariates_and_weights(
        vec![
            (
                "A".to_string(),
                "2026-01-01T00:00:00".to_string(),
                10.0,
                BTreeMap::from([
                    ("trip_distance".to_string(), 1.0),
                    ("trip_count".to_string(), 1.0),
                ]),
            ),
            (
                "A".to_string(),
                "2026-01-01T00:00:00".to_string(),
                20.0,
                BTreeMap::from([
                    ("trip_distance".to_string(), 3.0),
                    ("trip_count".to_string(), 3.0),
                ]),
            ),
            (
                "A".to_string(),
                "2026-01-02T00:00:00".to_string(),
                30.0,
                BTreeMap::from([
                    ("trip_distance".to_string(), 5.0),
                    ("trip_count".to_string(), 2.0),
                ]),
            ),
        ],
        vec![1.0, 3.0, 2.0],
        Some("trip_count".to_string()),
        ForecastFrequency::Daily,
        ForecastFrameMetadata::default(),
    )
    .expect("weighted duplicate frame");

    assert_eq!(frame.rows().len(), 2);
    assert_eq!(frame.rows()[0].target, 17.5);
    assert_eq!(frame.rows()[0].covariates["trip_distance"], 2.5);
    assert_eq!(frame.rows()[0].covariates["trip_count"], 4.0);
}

#[test]
fn rejects_irregular_frequency() {
    let err = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(3), 3.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect_err("gap should be rejected");
    assert!(err.to_string().contains("irregular forecast frequency"));
}

#[test]
fn rejects_non_finite_targets() {
    let err = ForecastFrame::new(
        vec![ForecastRow::single(ts(1), f64::NAN)],
        ForecastFrequency::Daily,
    )
    .expect_err("nan should be rejected");
    assert!(err.to_string().contains("forecast targets must be finite"));
}

#[test]
fn fixed_intermittent_forecasters_fit_predict_and_report_metadata() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=132", ts(1), 0.0),
            ForecastRow::new("PULocationID=132", ts(2), 3.0),
            ForecastRow::new("PULocationID=132", ts(3), 0.0),
            ForecastRow::new("PULocationID=132", ts(4), 6.0),
            ForecastRow::new("PULocationID=237", ts(1), 0.0),
            ForecastRow::new("PULocationID=237", ts(2), 2.0),
            ForecastRow::new("PULocationID=237", ts(3), 0.0),
            ForecastRow::new("PULocationID=237", ts(4), 4.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid intermittent taxi panel");

    let mut croston = CrostonForecaster::new(0.2).expect("valid croston");
    croston.fit(&frame).expect("croston fit");
    let croston_forecast = croston.predict(2).expect("croston forecast");
    assert_eq!(croston_forecast.predictions().len(), 4);
    assert!(croston_forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.model == "croston" && prediction.mean >= 0.0));
    assert_eq!(croston.metadata()["series_count"].as_u64(), Some(2));

    let mut sba = SbaForecaster::new(0.2).expect("valid sba");
    sba.fit(&frame).expect("sba fit");
    assert!(sba
        .predict(1)
        .expect("sba forecast")
        .predictions()
        .iter()
        .all(|prediction| prediction.model == "sba"));

    let mut tsb = TsbForecaster::new(0.2, 0.3).expect("valid tsb");
    tsb.fit(&frame).expect("tsb fit");
    assert!(tsb
        .predict(1)
        .expect("tsb forecast")
        .predictions()
        .iter()
        .all(|prediction| prediction.model == "tsb"));
    assert_eq!(tsb.metadata()["beta"].as_f64(), Some(0.3));
}

#[test]
fn fixed_intermittent_forecasters_reject_invalid_inputs() {
    assert!(CrostonForecaster::new(0.0).is_err());
    assert!(SbaForecaster::new(1.2).is_err());
    assert!(TsbForecaster::new(0.2, 0.0).is_err());

    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), -1.0),
            ForecastRow::single(ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("frame validates generic finite targets");
    let mut model = CrostonForecaster::default();
    assert!(model.fit(&frame).is_err());
    assert!(model.predict(1).is_err());
}

#[test]
fn intermittent_demand_requires_real_holdout_for_selection() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=132", ts(1), 3.0),
            ForecastRow::new("PULocationID=237", ts(1), 0.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid one-row intermittent panel");
    let mut model = IntermittentDemandForecaster::default();

    let error = model
        .fit(&frame)
        .expect_err("one-row selector must not invent validation");
    assert!(error.to_string().contains("real holdout"));
}

#[test]
fn parses_string_rows_and_validates_hourly_frequency() {
    let frame = ForecastFrame::from_string_rows(
        vec![
            ("A".to_string(), "2026-01-01T01:00:00".to_string(), 11.0),
            ("A".to_string(), "2026-01-01 00:00:00".to_string(), 10.0),
        ],
        ForecastFrequency::Hourly,
        ForecastFrameMetadata::default(),
    )
    .expect("valid hourly rows");

    assert_eq!(frame.rows()[0].timestamp, hour(1, 0));
    assert_eq!(frame.rows()[1].timestamp, hour(1, 1));
}

#[test]
fn validates_weekly_frequency() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(8), 8.0),
        ],
        ForecastFrequency::Weekly,
    )
    .expect("weekly frame");
    assert_eq!(frame.frequency(), ForecastFrequency::Weekly);
}

#[test]
fn rejects_unparseable_string_timestamp() {
    let err = ForecastFrame::from_string_rows(
        vec![("A".to_string(), "not-a-timestamp".to_string(), 1.0)],
        ForecastFrequency::Daily,
        ForecastFrameMetadata::default(),
    )
    .expect_err("unparseable timestamp");
    assert!(err.to_string().contains("not parseable"));
}

#[test]
fn exports_metadata_json() {
    let frame = ForecastFrame::with_metadata(
        vec![ForecastRow::new("PULocationID=132", ts(1), 42.0)],
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            timestamp_col: Some("pickup_hour".to_string()),
            target_col: Some("fare".to_string()),
            series_id_col: Some("PULocationID".to_string()),
            static_covariates: vec!["DOLocationID".to_string()],
            known_future_covariates: vec!["hour".to_string(), "day_of_week".to_string()],
            historical_covariates: vec!["trip_distance".to_string()],
            allow_irregular: false,
            allow_missing_targets: false,
            allow_missing_covariates: false,
        },
    )
    .expect("valid frame");

    let metadata: serde_json::Value =
        serde_json::from_str(&frame.metadata_json_string().expect("metadata json"))
            .expect("valid json");
    assert_eq!(metadata["timestamp_col"], "pickup_hour");
    assert_eq!(metadata["target_col"], "fare");
    assert_eq!(metadata["series_id_col"], "PULocationID");
    assert_eq!(metadata["frequency"], "daily");
    assert_eq!(metadata["is_panel"], true);
    assert_eq!(
        metadata["series_ids"],
        serde_json::json!(["PULocationID=132"])
    );
    assert_eq!(
        metadata["static_covariates"],
        serde_json::json!(["DOLocationID"])
    );
}

#[test]
fn forecast_frame_accepts_irregular_rows_only_when_explicit() {
    let rows = vec![
        ForecastRow::new("PULocationID=132", ts(1), 42.0),
        ForecastRow::new("PULocationID=132", ts(3), 44.0),
    ];
    let err = ForecastFrame::new(rows.clone(), ForecastFrequency::Daily)
        .expect_err("regular frame rejects skipped daily row");
    assert!(err.to_string().contains("irregular forecast frequency"));

    let frame = ForecastFrame::with_metadata(
        rows,
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            allow_irregular: true,
            allow_missing_targets: false,
            allow_missing_covariates: false,
            ..ForecastFrameMetadata::default()
        },
    )
    .expect("irregular frame");
    assert!(frame.allow_irregular());
    assert!(frame.is_irregular());
}

#[test]
fn forecast_frame_allows_missing_targets_only_when_explicit() {
    let rows = vec![
        ForecastRow::new("PULocationID=132", ts(1), 42.0),
        ForecastRow::new("PULocationID=132", ts(2), f64::NAN),
        ForecastRow::new("PULocationID=132", ts(3), 44.0),
    ];
    let err = ForecastFrame::new(rows.clone(), ForecastFrequency::Daily)
        .expect_err("missing target rejected by default");
    assert!(err.to_string().contains("allow_missing_targets"));

    let frame = ForecastFrame::with_metadata(
        rows,
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            allow_missing_targets: true,
            ..ForecastFrameMetadata::default()
        },
    )
    .expect("missing target frame");
    assert!(frame.allow_missing_targets());
    assert!(frame.has_missing_targets());

    let observed = frame
        .observed_target_frame("piecewise_linear_seasonal")
        .expect("observed frame");
    assert_eq!(observed.rows().len(), 2);
    assert!(observed.rows().iter().all(|row| row.target.is_finite()));
    assert!(observed.allow_irregular());
}

#[test]
fn forecast_frame_rejects_infinite_targets_even_when_missing_allowed() {
    let err = ForecastFrame::with_metadata(
        vec![
            ForecastRow::new("PULocationID=132", ts(1), 42.0),
            ForecastRow::new("PULocationID=132", ts(2), f64::INFINITY),
        ],
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            allow_missing_targets: true,
            ..ForecastFrameMetadata::default()
        },
    )
    .expect_err("infinite target rejected");
    assert!(err.to_string().contains("forecast targets"));
}

#[test]
fn forecast_frame_allows_missing_covariates_only_when_explicit() {
    let mut first_covariates = BTreeMap::new();
    first_covariates.insert("QUOTE_MILES".to_string(), 4.2);
    let mut missing_covariates = BTreeMap::new();
    missing_covariates.insert("QUOTE_MILES".to_string(), f64::NAN);
    let rows = vec![
        ForecastRow::with_covariates("PULocationID=132", ts(1), 42.0, first_covariates),
        ForecastRow::with_covariates("PULocationID=132", ts(2), 43.0, missing_covariates),
    ];

    let err = ForecastFrame::with_metadata(
        rows.clone(),
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            historical_covariates: vec!["QUOTE_MILES".to_string()],
            ..ForecastFrameMetadata::default()
        },
    )
    .expect_err("missing covariate rejected by default");
    assert!(err.to_string().contains("allow_missing_covariates"));

    let frame = ForecastFrame::with_metadata(
        rows,
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            historical_covariates: vec!["QUOTE_MILES".to_string()],
            allow_missing_covariates: true,
            ..ForecastFrameMetadata::default()
        },
    )
    .expect("missing covariate frame");
    assert!(frame.allow_missing_covariates());
    assert!(frame.has_missing_covariates());
}

#[test]
fn forecast_frame_rejects_infinite_covariates_even_when_missing_allowed() {
    let mut covariates = BTreeMap::new();
    covariates.insert("QUOTE_MILES".to_string(), f64::INFINITY);

    let err = ForecastFrame::with_metadata(
        vec![ForecastRow::with_covariates(
            "PULocationID=132",
            ts(1),
            42.0,
            covariates,
        )],
        ForecastFrequency::Daily,
        ForecastFrameMetadata {
            historical_covariates: vec!["QUOTE_MILES".to_string()],
            allow_missing_covariates: true,
            ..ForecastFrameMetadata::default()
        },
    )
    .expect_err("infinite covariate rejected");
    assert!(err.to_string().contains("must not be infinite"));
}

#[test]
fn naive_forecasts_each_panel_series_without_bleeding() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1->DO2", ts(1), 11.0),
            ForecastRow::new("PU1->DO2", ts(2), 12.0),
            ForecastRow::new("PU9->DO8", ts(1), 31.0),
            ForecastRow::new("PU9->DO8", ts(2), 32.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("predict");

    let means = forecast
        .predictions()
        .iter()
        .map(|row| (row.series_id.as_str(), row.horizon, row.mean))
        .collect::<Vec<_>>();
    assert_eq!(
        means,
        vec![
            ("PU1->DO2", 1, 12.0),
            ("PU1->DO2", 2, 12.0),
            ("PU9->DO8", 1, 32.0),
            ("PU9->DO8", 2, 32.0)
        ]
    );
}

#[test]
fn seasonal_naive_repeats_last_season() {
    let frame = ForecastFrame::new(
        (1..=14)
            .map(|day| ForecastRow::single(ts(day), f64::from(day % 7)))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = SeasonalNaiveForecaster::new(7).expect("valid season");
    model.fit(&frame).expect("fit");
    let forecast = model.predict(3).expect("predict");

    let means = forecast
        .predictions()
        .iter()
        .map(|row| row.mean)
        .collect::<Vec<_>>();
    assert_eq!(means, vec![1.0, 2.0, 3.0]);
}

#[test]
fn seasonal_naive_rejects_history_without_a_complete_cycle() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = SeasonalNaiveForecaster::new(7).expect("valid season");
    let error = model
        .fit(&frame)
        .expect_err("short seasonal history must fail");
    assert!(error.to_string().contains("one complete requested season"));
}

#[test]
fn forecast_result_json_round_trips() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(1).expect("predict");
    let json = forecast.to_json_string().expect("json");
    let value: serde_json::Value = serde_json::from_str(&json).expect("result json");
    assert_eq!(
        value["columns"],
        serde_json::json!(["series_id", "timestamp", "horizon", "model", "prediction"])
    );
    let restored = cartoboost_core::forecasting::ForecastResult::from_json_string(&json)
        .expect("json round trip");
    assert_eq!(forecast, restored);
}

#[test]
fn forecast_result_columns_are_stable() {
    assert_eq!(
        ForecastResult::prediction_columns(),
        vec!["series_id", "timestamp", "horizon", "model", "prediction"]
    );
}

#[test]
fn metrics_align_by_series_timestamp_and_horizon() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("A", ts(1), 10.0),
            ForecastRow::new("A", ts(2), 12.0),
            ForecastRow::new("B", ts(1), 20.0),
            ForecastRow::new("B", ts(2), 22.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(1).expect("predict");

    let metrics = cartoboost_core::forecasting::evaluate_forecast(
        &forecast,
        &[
            ForecastActual {
                series_id: "B".to_string(),
                timestamp: ts(3),
                horizon: 1,
                actual: 23.0,
            },
            ForecastActual {
                series_id: "A".to_string(),
                timestamp: ts(3),
                horizon: 1,
                actual: 13.0,
            },
        ],
    )
    .expect("metrics");
    assert_eq!(metrics.mae, 1.0);
    assert_eq!(metrics.bias, -1.0);
    assert!((metrics.normalized_rmse - (1.0 / 18.0)).abs() < 1e-12);
    assert!((metrics.wape - (2.0 / 36.0)).abs() < 1e-12);
    assert_eq!(metrics.mase, None);
}

#[test]
fn metrics_reject_unmatched_forecast_rows() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 11.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("predict");

    let err = cartoboost_core::forecasting::evaluate_forecast(
        &forecast,
        &[ForecastActual {
            series_id: "__single__".to_string(),
            timestamp: ts(3),
            horizon: 1,
            actual: 12.0,
        }],
    )
    .expect_err("extra forecast row should be rejected");
    assert!(err.to_string().contains("no matching actual"));
}

#[test]
fn competition_metrics_compute_smape_mase_and_owa_ratios() {
    let training_series = vec![vec![10.0, 12.0, 14.0], vec![20.0, 23.0, 26.0]];
    let actuals = vec![16.0, 29.0];
    let baseline = evaluate_competition_metrics(&training_series, &actuals, &[14.0, 26.0], 1, None)
        .expect("baseline metrics");

    let metrics = evaluate_competition_metrics(
        &training_series,
        &actuals,
        &[15.0, 30.0],
        1,
        Some((baseline.smape, baseline.mase)),
    )
    .expect("model metrics");

    let expected_smape = 0.5 * (2.0 / 31.0 + 2.0 / 59.0);
    let expected_mase = 1.0 / 2.5;
    assert!((metrics.smape - expected_smape).abs() < 1e-12);
    assert!((metrics.mase - expected_mase).abs() < 1e-12);
    assert!(metrics.smape_ratio_to_baseline.expect("smape ratio") < 1.0);
    assert!(metrics.mase_ratio_to_baseline.expect("mase ratio") < 1.0);
    assert_eq!(
        metrics.owa,
        Some(
            0.5 * (metrics.smape_ratio_to_baseline.unwrap()
                + metrics.mase_ratio_to_baseline.unwrap())
        )
    );
}

#[test]
fn rolling_origin_splitter_is_leakage_safe_and_deterministic() {
    let frame = taxi_panel_frame(6);
    let splitter = RollingOriginSplitter::new(2, 2, 3, None, None, ForecastWindow::Expanding)
        .expect("splitter");

    let folds = splitter.split(&frame).expect("folds");

    assert_eq!(
        folds
            .iter()
            .map(|fold| fold.fold_id.as_str())
            .collect::<Vec<_>>(),
        vec!["fold_0000"]
    );
    let fold = &folds[0];
    assert!(fold.train_end < fold.validation_start);
    assert_eq!(fold.metadata.origin_timestamp, ts(3));
    assert_eq!(fold.metadata.train_timestamp_count, 3);
    assert_eq!(fold.metadata.validation_timestamp_count, 2);
    assert_eq!(fold.metadata.series_count, 2);
}

#[test]
fn candidate_validation_cutoff_schedule_matches_benchmark_origins() {
    let default_schedule = CandidateValidationCutoffSchedule::new(100, 14, None).expect("schedule");
    let reconciliation_schedule =
        CandidateValidationCutoffSchedule::new(100, 14, Some("hierarchical_reconciliation"))
            .expect("schedule");
    let rank_portfolio_schedule =
        CandidateValidationCutoffSchedule::new(100, 14, Some("rank_portfolio")).expect("schedule");
    let short_schedule =
        CandidateValidationCutoffSchedule::new(20, 14, Some("classical_competition"))
            .expect("schedule");

    assert_eq!(default_schedule.cutoff_indices, vec![58, 72, 86]);
    assert_eq!(reconciliation_schedule.cutoff_indices, vec![72, 86]);
    assert_eq!(rank_portfolio_schedule.cutoff_indices, vec![58, 72, 86]);
    assert_eq!(short_schedule.cutoff_indices, Vec::<usize>::new());
}

#[test]
fn sliding_splitter_respects_max_train_size() {
    let frame = taxi_panel_frame(7);
    let splitter = RollingOriginSplitter::new(1, 1, 2, Some(3), None, ForecastWindow::Sliding)
        .expect("splitter");

    let folds = splitter.split(&frame).expect("folds");
    let last = folds.last().expect("at least one fold");

    assert_eq!(last.train_start, ts(4));
    assert_eq!(last.train_end, ts(6));
    assert_eq!(last.validation_start, ts(7));
    assert_eq!(last.metadata.train_timestamp_count, 3);
}

#[test]
fn rolling_origin_backtester_scores_each_fold_with_mase() {
    let frame = taxi_panel_frame(6);
    let splitter = RollingOriginSplitter::new(1, 1, 3, None, Some(2), ForecastWindow::Expanding)
        .expect("splitter");
    let backtester = RollingOriginBacktester::new(splitter)
        .with_mase_seasonality(1)
        .expect("mase");

    let result = backtester
        .run(NaiveForecaster::new(), &frame)
        .expect("backtest");

    assert_eq!(result.folds.len(), 2);
    assert!(result
        .folds
        .iter()
        .all(|fold| fold.fold.train_end < fold.fold.validation_start));
    let metrics = result.metrics.expect("aggregate metrics");
    assert_eq!(metrics.mae, 1.0);
    assert_eq!(metrics.bias, -1.0);
    assert_eq!(metrics.mase, Some(1.0));
}

fn taxi_panel_frame(days: u32) -> ForecastFrame {
    let mut rows = Vec::new();
    for series in ["PULocationID=1", "PULocationID=2"] {
        for day in 1..=days {
            rows.push(ForecastRow::new(series, ts(day), f64::from(day)));
        }
    }
    ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid taxi panel frame")
}
