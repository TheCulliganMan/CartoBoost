use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::forecasting::{
    CalendarFeature, CartoBoostLagForecaster, ForecastFrame, ForecastFrequency, ForecastObjective,
    ForecastRow, Forecaster, GlobalForecastSampleWeightMode, GlobalForecastTargetMode,
    IntermittentDemandConfig, IntermittentDemandForecaster, LagFeatureBuilder, LagFeatureConfig,
    LagPlusConfig, LagPlusForecaster, LocalStandardScaledForecaster, Log1pForecaster,
};
use chrono::{NaiveDate, NaiveDateTime};

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

fn small_booster_config() -> BoosterConfig {
    BoosterConfig {
        n_estimators: 8,
        learning_rate: 0.3,
        max_depth: 2,
        min_samples_leaf: 1,
        min_gain: 0.0,
        ..BoosterConfig::default()
    }
}

#[test]
fn lag_builder_is_leakage_safe_and_panel_isolated() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 20.0),
            ForecastRow::new("PU1", ts(3), 30.0),
            ForecastRow::new("PU9", ts(1), 100.0),
            ForecastRow::new("PU9", ts(2), 200.0),
            ForecastRow::new("PU9", ts(3), 300.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let builder = LagFeatureBuilder::new(LagFeatureConfig {
        lags: vec![1, 2],
        rolling_mean_windows: vec![2],
        partial_rolling_mean_windows: Vec::new(),
        rolling_std_windows: vec![2],
        rolling_min_windows: vec![2],
        rolling_max_windows: vec![2],
        ewm_alpha_percents: Vec::new(),
        difference_lags: Vec::new(),
        rolling_trend_windows: Vec::new(),
        covariate_features: Vec::new(),
        covariate_indicator_values: Default::default(),
        covariate_calendar_interactions: false,
        calendar_features: vec![CalendarFeature::DayOfWeek, CalendarFeature::Month],
    })
    .expect("valid config");

    let rows = builder.transform_frame(&frame).expect("features");
    let pu1_day3 = rows
        .iter()
        .find(|row| row.series_id == "PU1" && row.timestamp == ts(3))
        .expect("PU1 day 3 row");
    let pu9_day3 = rows
        .iter()
        .find(|row| row.series_id == "PU9" && row.timestamp == ts(3))
        .expect("PU9 day 3 row");

    assert_eq!(
        builder.feature_names(),
        &[
            "target_lag_1".to_string(),
            "target_lag_2".to_string(),
            "target_roll_mean_2".to_string(),
            "target_roll_std_2".to_string(),
            "target_roll_min_2".to_string(),
            "target_roll_max_2".to_string(),
            "calendar_day_of_week".to_string(),
            "calendar_month".to_string(),
        ]
    );
    assert_eq!(pu1_day3.features[0], 20.0);
    assert_eq!(pu1_day3.features[1], 10.0);
    assert_eq!(pu1_day3.features[2], 15.0);
    assert_eq!(pu1_day3.features[3], 5.0);
    assert_eq!(pu1_day3.features[4], 10.0);
    assert_eq!(pu1_day3.features[5], 20.0);
    assert_eq!(pu9_day3.features[0], 200.0);
    assert_eq!(pu9_day3.features[1], 100.0);
    assert_eq!(pu9_day3.features[2], 150.0);
    assert_eq!(pu9_day3.features[3], 50.0);
    assert_eq!(pu9_day3.features[4], 100.0);
    assert_eq!(pu9_day3.features[5], 200.0);
    assert_eq!(rows.len(), 2);
}

#[test]
fn lag_builder_drops_rows_without_complete_lag_history() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 20.0),
            ForecastRow::new("PU1", ts(3), 30.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let builder = LagFeatureBuilder::new(LagFeatureConfig {
        lags: vec![2],
        rolling_mean_windows: Vec::new(),
        partial_rolling_mean_windows: Vec::new(),
        rolling_std_windows: Vec::new(),
        rolling_min_windows: Vec::new(),
        rolling_max_windows: Vec::new(),
        ewm_alpha_percents: Vec::new(),
        difference_lags: Vec::new(),
        rolling_trend_windows: Vec::new(),
        covariate_features: Vec::new(),
        covariate_indicator_values: Default::default(),
        covariate_calendar_interactions: false,
        calendar_features: Vec::new(),
    })
    .expect("valid config");

    let rows = builder.transform_frame(&frame).expect("features");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].timestamp, ts(3));
    assert_eq!(rows[0].features, vec![10.0]);
}

#[test]
fn cartoboost_lag_forecaster_predicts_recursively_per_panel() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 12.0),
            ForecastRow::new("PU1", ts(3), 14.0),
            ForecastRow::new("PU1", ts(4), 16.0),
            ForecastRow::new("PU9", ts(1), 40.0),
            ForecastRow::new("PU9", ts(2), 42.0),
            ForecastRow::new("PU9", ts(3), 44.0),
            ForecastRow::new("PU9", ts(4), 46.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = CartoBoostLagForecaster::new(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: vec![CalendarFeature::Day],
        },
        small_booster_config(),
    )
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");
    let predictions = forecast.predictions();

    assert_eq!(forecaster.training_rows(), Some(4));
    assert_eq!(predictions.len(), 4);
    assert_eq!(predictions[0].series_id, "PU1");
    assert_eq!(predictions[0].timestamp, ts(5));
    assert_eq!(predictions[0].horizon, 1);
    assert_eq!(predictions[1].series_id, "PU1");
    assert_eq!(predictions[1].timestamp, ts(6));
    assert_eq!(predictions[1].horizon, 2);
    assert_eq!(predictions[2].series_id, "PU9");
    assert_eq!(predictions[2].timestamp, ts(5));
    assert!(predictions
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
    assert!(predictions[2].mean > predictions[0].mean);
}

#[test]
fn cartoboost_lag_forecaster_supports_recency_sample_weights() {
    let frame = ForecastFrame::new(
        (1..=16)
            .map(|day| {
                let target = if day <= 8 {
                    10.0 + f64::from(day)
                } else {
                    40.0 + f64::from(day * 2)
                };
                ForecastRow::new("PU1", ts(day), target)
            })
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = CartoBoostLagForecaster::new_with_target_mode_and_sample_weight(
        LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: vec![1],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        small_booster_config(),
        GlobalForecastTargetMode::Level,
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life: 4 },
    )
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");

    assert_eq!(
        forecaster.sample_weight_mode(),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life: 4 }
    );
    assert_eq!(
        forecaster.metadata()["sample_weight_mode"],
        serde_json::json!("exponential_recency_half_life_4")
    );
    assert_eq!(forecast.predictions().len(), 2);
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn cartoboost_lag_forecaster_rejects_invalid_recency_half_life() {
    let err = CartoBoostLagForecaster::new_with_target_mode_and_sample_weight(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        small_booster_config(),
        GlobalForecastTargetMode::Level,
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life: 0 },
    )
    .expect_err("invalid half-life");

    assert!(err
        .to_string()
        .contains("exponential recency half_life must be positive"));
}

#[test]
fn cartoboost_lag_forecaster_can_model_delta_from_last_target() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 12.0),
            ForecastRow::new("PU1", ts(3), 14.0),
            ForecastRow::new("PU1", ts(4), 16.0),
            ForecastRow::new("PU1", ts(5), 18.0),
            ForecastRow::new("PU1", ts(6), 20.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = CartoBoostLagForecaster::new_with_target_mode(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: vec![1],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        small_booster_config(),
        GlobalForecastTargetMode::DeltaFromLast,
    )
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");
    let predictions = forecast.predictions();

    assert_eq!(
        forecaster.target_mode(),
        GlobalForecastTargetMode::DeltaFromLast
    );
    assert_eq!(predictions.len(), 2);
    assert!(predictions[0].mean > 20.0);
    assert!(predictions[1].mean >= predictions[0].mean);
    assert_eq!(
        forecaster.metadata()["target_mode"],
        serde_json::json!("delta_from_last")
    );
}

#[test]
fn cartoboost_lag_forecaster_can_model_seasonal_delta_target() {
    let frame = ForecastFrame::new(
        (1..=21)
            .map(|day| {
                let seasonal = match day % 7 {
                    1 => 30.0,
                    2 => 34.0,
                    3 => 38.0,
                    4 => 42.0,
                    5 => 46.0,
                    6 => 50.0,
                    _ => 54.0,
                };
                ForecastRow::new("PU1", ts(day), seasonal + f64::from(day / 7))
            })
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = CartoBoostLagForecaster::new_with_target_mode(
        LagFeatureConfig {
            lags: vec![1, 7],
            rolling_mean_windows: vec![7],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: vec![7],
            rolling_trend_windows: vec![7],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        small_booster_config(),
        GlobalForecastTargetMode::SeasonalDelta { season_length: 7 },
    )
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");
    let predictions = forecast.predictions();

    assert_eq!(
        forecaster.target_mode(),
        GlobalForecastTargetMode::SeasonalDelta { season_length: 7 }
    );
    assert_eq!(predictions.len(), 2);
    assert!(predictions
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
    assert!(predictions[0].mean > 30.0);
    assert!(predictions[1].mean > predictions[0].mean);
    assert_eq!(
        forecaster.metadata()["target_mode"],
        serde_json::json!("seasonal_delta_7")
    );
}

#[test]
fn lag_plus_forecaster_calibrates_residual_corrections() {
    let frame = ForecastFrame::new(
        (1..=14)
            .flat_map(|day| {
                [
                    ForecastRow::new("PU1", ts(day), 20.0 + f64::from(day * 2)),
                    ForecastRow::new("PU9", ts(day), 100.0 + f64::from(day * 3)),
                ]
            })
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = LagPlusForecaster::new(LagPlusConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        booster_config: small_booster_config(),
        target_mode: GlobalForecastTargetMode::Level,
        validation_window: Some(2),
        objective: ForecastObjective::Wape,
        shrinkage_strength: 2.0,
        seasonal_bucket_period: Some(7),
    })
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");
    let metadata = forecaster.metadata();

    assert_eq!(forecast.predictions().len(), 4);
    assert_eq!(metadata["model"], serde_json::json!("lag_plus"));
    assert_eq!(metadata["base_model"], serde_json::json!("cartoboost_lag"));
    assert_eq!(metadata["objective"], serde_json::json!("wape"));
    assert!(metadata["base_rmse"]
        .as_f64()
        .expect("base rmse")
        .is_finite());
    assert!(metadata["corrected_rmse"]
        .as_f64()
        .expect("corrected rmse")
        .is_finite());
    assert!(metadata["base_wape"]
        .as_f64()
        .expect("base wape")
        .is_finite());
    assert!(metadata["corrected_wape"]
        .as_f64()
        .expect("corrected wape")
        .is_finite());
    assert_eq!(metadata["seasonal_bucket_period"], serde_json::json!(7));
    assert!(metadata["seasonal_corrections"]
        .as_object()
        .expect("seasonal corrections")
        .values()
        .all(|value| value.as_f64().expect("correction").is_finite()));
    let series_corrections = metadata["series_corrections"]
        .as_object()
        .expect("series corrections");
    assert!(series_corrections.contains_key("PU1"));
    assert!(series_corrections.contains_key("PU9"));
    assert!(series_corrections
        .values()
        .all(|value| value.as_f64().expect("correction").is_finite()));
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn lag_plus_rejects_short_panels_instead_of_disabling_calibration_or_pruning_lags() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 20.0),
            ForecastRow::new("PU1", ts(2), 22.0),
            ForecastRow::new("PU9", ts(1), 100.0),
            ForecastRow::new("PU9", ts(2), 103.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid short panel");
    let mut forecaster = LagPlusForecaster::new(LagPlusConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 7],
            rolling_mean_windows: vec![3],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: vec![7],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        booster_config: small_booster_config(),
        target_mode: GlobalForecastTargetMode::Level,
        validation_window: None,
        objective: ForecastObjective::Wape,
        shrinkage_strength: 2.0,
        seasonal_bucket_period: Some(7),
    })
    .expect("forecaster");

    let error = forecaster
        .fit(&frame)
        .expect_err("LagPlus needs a real holdout and its complete configured lag history");
    assert!(error
        .to_string()
        .contains("requires at least 8 prior observations"));
}

#[test]
fn local_standard_scaled_forecaster_inverts_predictions_to_original_scale() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("small", ts(1), 1.0),
            ForecastRow::new("small", ts(2), 2.0),
            ForecastRow::new("small", ts(3), 3.0),
            ForecastRow::new("small", ts(4), 4.0),
            ForecastRow::new("large", ts(1), 100.0),
            ForecastRow::new("large", ts(2), 200.0),
            ForecastRow::new("large", ts(3), 300.0),
            ForecastRow::new("large", ts(4), 400.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let inner = CartoBoostLagForecaster::new(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        small_booster_config(),
    )
    .expect("inner");
    let mut forecaster = LocalStandardScaledForecaster::new(Box::new(inner), 1e-6, "scaled_lag")
        .expect("scaled forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(1).expect("predict");
    let metadata = forecaster.metadata();

    assert_eq!(forecast.predictions().len(), 2);
    assert_eq!(metadata["model"], serde_json::json!("scaled_lag"));
    assert_eq!(
        metadata["target_transform"]["transform"],
        serde_json::json!("local_standard_scaler")
    );
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
    let large = forecast
        .predictions()
        .iter()
        .find(|prediction| prediction.series_id == "large")
        .expect("large forecast");
    let small = forecast
        .predictions()
        .iter()
        .find(|prediction| prediction.series_id == "small")
        .expect("small forecast");
    assert!(large.mean > small.mean);
}

#[test]
fn log1p_forecaster_inverts_and_clamps_nonnegative_predictions() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("low", ts(1), 0.0),
            ForecastRow::new("low", ts(2), 1.0),
            ForecastRow::new("low", ts(3), 2.0),
            ForecastRow::new("low", ts(4), 3.0),
            ForecastRow::new("high", ts(1), 10.0),
            ForecastRow::new("high", ts(2), 20.0),
            ForecastRow::new("high", ts(3), 40.0),
            ForecastRow::new("high", ts(4), 80.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let inner = LocalStandardScaledForecaster::new(
        Box::new(
            CartoBoostLagForecaster::new(
                LagFeatureConfig {
                    lags: vec![1],
                    rolling_mean_windows: Vec::new(),
                    partial_rolling_mean_windows: Vec::new(),
                    rolling_std_windows: Vec::new(),
                    rolling_min_windows: Vec::new(),
                    rolling_max_windows: Vec::new(),
                    ewm_alpha_percents: Vec::new(),
                    difference_lags: Vec::new(),
                    rolling_trend_windows: Vec::new(),
                    covariate_features: Vec::new(),
                    covariate_indicator_values: Default::default(),
                    covariate_calendar_interactions: false,
                    calendar_features: Vec::new(),
                },
                small_booster_config(),
            )
            .expect("inner"),
        ),
        1e-6,
        "log1p_scaled_lag_transformed",
    )
    .expect("scaled inner");
    let mut forecaster = Log1pForecaster::new(Box::new(inner), "log1p_scaled_lag");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(1).expect("predict");
    let metadata = forecaster.metadata();

    assert_eq!(forecast.predictions().len(), 2);
    assert_eq!(metadata["model"], serde_json::json!("log1p_scaled_lag"));
    assert_eq!(
        metadata["target_transform"]["transform"],
        serde_json::json!("log1p")
    );
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite() && prediction.mean >= 0.0));
}

#[test]
fn log1p_forecaster_rejects_negative_targets() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(2), -1.0),
            ForecastRow::single(ts(3), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let inner = CartoBoostLagForecaster::new(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: Vec::new(),
        },
        small_booster_config(),
    )
    .expect("inner");
    let mut forecaster = Log1pForecaster::new(Box::new(inner), "log1p_lag");

    let error = forecaster
        .fit(&frame)
        .expect_err("negative target rejected");

    assert!(error
        .to_string()
        .contains("log1p target transform requires nonnegative targets"));
}

#[test]
fn intermittent_demand_forecaster_selects_nonnegative_series_methods() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("sparse", ts(1), 0.0),
            ForecastRow::new("sparse", ts(2), 0.0),
            ForecastRow::new("sparse", ts(3), 8.0),
            ForecastRow::new("sparse", ts(4), 0.0),
            ForecastRow::new("sparse", ts(5), 0.0),
            ForecastRow::new("sparse", ts(6), 10.0),
            ForecastRow::new("sparse", ts(7), 0.0),
            ForecastRow::new("sparse", ts(8), 0.0),
            ForecastRow::new("zero", ts(1), 0.0),
            ForecastRow::new("zero", ts(2), 0.0),
            ForecastRow::new("zero", ts(3), 0.0),
            ForecastRow::new("zero", ts(4), 0.0),
            ForecastRow::new("zero", ts(5), 0.0),
            ForecastRow::new("zero", ts(6), 0.0),
            ForecastRow::new("zero", ts(7), 0.0),
            ForecastRow::new("zero", ts(8), 0.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = IntermittentDemandForecaster::new(IntermittentDemandConfig {
        validation_window: Some(2),
        ..IntermittentDemandConfig::default()
    })
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");
    let methods = forecaster.fitted_methods();
    let metadata = forecaster.metadata();

    assert_eq!(forecast.predictions().len(), 4);
    assert_eq!(metadata["model"], serde_json::json!("intermittent_demand"));
    assert_eq!(methods["zero"].as_str(), "zero");
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite() && prediction.mean >= 0.0));
}
