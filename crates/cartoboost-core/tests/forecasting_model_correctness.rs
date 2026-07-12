use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::forecasting::{
    ArimaForecaster, AutoARIMAForecaster, AutoETSForecaster, CalendarFeature,
    CartoBoostLagForecaster, ETSForecaster, ForecastFrame, ForecastFrequency, ForecastRow,
    Forecaster, GlobalForecastTargetMode, KalmanForecaster, KrigingForecaster, LagFeatureConfig,
    NaiveForecaster, OptimizedThetaForecaster, PiecewiseLinearComponentMode, PiecewiseLinearEvent,
    PiecewiseLinearFitLoss, PiecewiseLinearSeasonalConfig, PiecewiseLinearSeasonalForecaster,
    PiecewiseLinearSeasonality, SeasonalNaiveForecaster, SeasonalWindowAverageForecaster,
    ThetaForecaster, WeightedEnsembleForecaster, WindowAverageForecaster,
};
use chrono::{Duration, NaiveDate, NaiveDateTime};
use std::collections::BTreeMap;

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

fn single_frame(values: &[f64]) -> ForecastFrame {
    ForecastFrame::new(
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| ForecastRow::single(ts(idx as u32 + 1), *value))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid single-series frame")
}

fn means<M: Forecaster>(model: &M, horizon: usize) -> Vec<f64> {
    model
        .predict(horizon)
        .expect("predict")
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-6,
        "expected {actual} to be within 1e-6 of {expected}"
    );
}

fn assert_roughly(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {actual} to be within {tolerance} of {expected}"
    );
}

#[test]
fn naive_and_seasonal_naive_have_exact_known_answer_forecasts() {
    let frame = single_frame(&[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

    let mut naive = NaiveForecaster::new();
    naive.fit(&frame).expect("fit naive");
    assert_eq!(means(&naive, 3), vec![20.0, 20.0, 20.0]);
    assert_eq!(naive.metadata()["model"], "naive");

    let mut seasonal = SeasonalNaiveForecaster::new(3).expect("valid seasonal naive");
    seasonal.fit(&frame).expect("fit seasonal naive");
    assert_eq!(means(&seasonal, 5), vec![16.0, 18.0, 20.0, 16.0, 18.0]);
    assert_eq!(seasonal.metadata()["season_length"], 3);

    let mut window_average = WindowAverageForecaster::new(3).expect("valid window average");
    window_average.fit(&frame).expect("fit window average");
    assert_eq!(means(&window_average, 4), vec![18.0, 18.0, 18.0, 18.0]);
    assert_eq!(window_average.metadata()["window_size"], 3);

    let mut seasonal_window =
        SeasonalWindowAverageForecaster::new(3, 2).expect("valid seasonal window average");
    seasonal_window
        .fit(&frame)
        .expect("fit seasonal window average");
    assert_eq!(
        means(&seasonal_window, 5),
        vec![13.0, 15.0, 17.0, 13.0, 15.0]
    );
    assert_eq!(seasonal_window.metadata()["window_count"], 2);
}

#[test]
fn theta_models_preserve_constant_series_and_select_best_grid_member() {
    let frame = single_frame(&[42.0, 42.0, 42.0, 42.0, 42.0]);

    let mut theta = ThetaForecaster::new(2.0, 0.4).expect("valid theta");
    theta.fit(&frame).expect("fit theta");
    for mean in means(&theta, 3) {
        assert_close(mean, 42.0);
    }
    assert_eq!(theta.fitted_values("__single__").expect("fitted").len(), 5);
    assert!(theta
        .residuals("__single__")
        .expect("residuals")
        .iter()
        .all(|residual| residual.abs() < 1.0e-6));

    let mut optimized =
        OptimizedThetaForecaster::new(vec![2.0, 1.0], vec![0.8, 0.2]).expect("valid grid");
    optimized.fit(&frame).expect("fit optimized theta");
    assert_eq!(optimized.selected_theta(), Some(1.0));
    assert_eq!(optimized.selected_alpha(), Some(0.2));
    assert_eq!(optimized.validation_scores().len(), 4);
    for mean in means(&optimized, 2) {
        assert_close(mean, 42.0);
    }
    assert_eq!(optimized.metadata()["model"], "optimized_theta");
}

#[test]
fn ets_and_arima_reproduce_linear_series_known_answers() {
    let frame = single_frame(&[10.0, 12.0, 14.0, 16.0]);

    let mut ets = ETSForecaster::new(1.0, 1.0).expect("valid ets");
    ets.fit(&frame).expect("fit ets");
    assert_eq!(means(&ets, 2), vec![18.0, 20.0]);
    assert_eq!(
        ets.fitted_values("__single__").expect("fitted"),
        &[10.0, 12.0, 14.0, 16.0]
    );
    assert!(ets
        .residuals("__single__")
        .expect("residuals")
        .iter()
        .all(|residual| residual.abs() < 1.0e-6));

    let mut arima = ArimaForecaster::new(0, 1, 0).expect("valid arima");
    arima.fit(&frame).expect("fit arima");
    assert_eq!(arima.order(), (0, 1, 0));
    assert_eq!(means(&arima, 2), vec![18.0, 20.0]);
    assert_eq!(
        arima.fitted_values("__single__").expect("fitted"),
        &[10.0, 12.0, 14.0, 16.0]
    );

    let mut auto_arima = AutoARIMAForecaster::with_max_order(0, 1, 0).expect("valid auto arima");
    auto_arima.fit(&frame).expect("fit auto arima");
    assert_eq!(auto_arima.selected_order(), Some((0, 1, 0)));
    assert_eq!(means(&auto_arima, 2), vec![18.0, 20.0]);
    assert_eq!(auto_arima.metadata()["selected_order"]["d"], 1);
}

#[test]
fn auto_ets_preserves_declared_seasonality() {
    let frame = single_frame(&[
        46.0, 54.0, 46.0, 54.0, 46.0, 54.0, 46.0, 54.0, 46.0, 54.0, 46.0, 54.0,
    ]);
    let mut model =
        AutoETSForecaster::with_grids(vec![0.5], vec![0.0], vec![Some(0.5)], vec![1.0], Some(2))
            .expect("valid seasonal auto ets");

    model.fit(&frame).expect("fit auto ets");
    let forecast = model.predict(2).expect("predict auto ets");
    let means = forecast
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect::<Vec<_>>();

    assert_eq!(
        model.selected_params().expect("selected params").gamma,
        Some(0.5)
    );
    assert!(
        means[1] > means[0],
        "seasonal ETS should vary by declared period: {means:?}"
    );
}

#[test]
fn piecewise_linear_seasonal_public_api_covers_trend_components_and_artifacts() {
    let start = ts(1);
    let frame = ForecastFrame::new(
        (0..42)
            .map(|day| {
                let airport_queue = if day % 6 == 0 { 1.0 } else { 0.0 };
                let rush_hour = if day % 2 == 0 { 1.0 } else { 0.0 };
                ForecastRow::with_covariates(
                    "PULocationID=132->DOLocationID=236",
                    start + Duration::days(day),
                    35.0 + 1.25 * day as f64 + 18.0 * airport_queue + 4.0 * rush_hour,
                    BTreeMap::from([
                        ("airport_queue".to_string(), airport_queue),
                        ("rush_hour".to_string(), rush_hour),
                    ]),
                )
            })
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
        changepoints: 1,
        changepoint_timestamps: vec![start + Duration::days(21)],
        weekly_fourier_order: 0,
        auto_weekly_seasonality: false,
        custom_seasonalities: vec![PiecewiseLinearSeasonality {
            name: "rush_hour_weekly".to_string(),
            period_days: 7.0,
            fourier_order: 2,
            mode: Some(PiecewiseLinearComponentMode::Additive),
            condition_name: Some("rush_hour".to_string()),
            l2_regularization: Some(0.001),
        }],
        events: vec![PiecewiseLinearEvent {
            name: "airport_surge".to_string(),
            timestamp: start + Duration::days(44),
            lower_window: 0,
            upper_window: 0,
        }],
        extra_regressors: vec!["airport_queue".to_string()],
        regressor_l2_regularization: 0.001,
        future_regressors: BTreeMap::from([
            ("airport_queue".to_string(), vec![1.0, 0.0, 0.0]),
            ("rush_hour".to_string(), vec![1.0, 0.0, 1.0]),
        ]),
        interval_levels: vec![0.8],
        quantile_levels: vec![0.25, 0.75],
        uncertainty_samples: 8,
        uncertainty_seed: 31,
        fit_loss: PiecewiseLinearFitLoss::Huber,
        huber_delta: 1.2,
        irls_iterations: 3,
        ..PiecewiseLinearSeasonalConfig::default()
    })
    .expect("valid piecewise seasonal config");

    model.fit(&frame).expect("fit piecewise seasonal");
    let forecast = model.predict(3).expect("predict");
    let components = model
        .predict_components_json_value(3)
        .expect("component forecast");
    let quantiles = model
        .predict_quantiles_json_value(3, None)
        .expect("quantiles");
    let samples = model.predict_samples_json_value(3).expect("samples");
    let restored = PiecewiseLinearSeasonalForecaster::from_json_string(
        &model.to_json_string().expect("serialize artifact"),
    )
    .expect("restore artifact");
    let restored_forecast = restored.predict(3).expect("restored predict");

    assert_eq!(forecast.predictions().len(), 3);
    assert_eq!(forecast.intervals().len(), 3);
    assert_eq!(forecast.predictions()[0].model, "piecewise_linear_seasonal");
    assert_eq!(
        forecast.predictions()[0].series_id,
        "PULocationID=132->DOLocationID=236"
    );
    assert!(forecast.predictions()[0].mean > forecast.predictions()[1].mean);
    assert_roughly(forecast.predictions()[2].mean, 92.0, 8.0);
    assert_close(
        components["records"][0]["prediction"]
            .as_f64()
            .expect("component prediction"),
        forecast.predictions()[0].mean,
    );
    assert!(
        components["records"][0]["components"]["regressors"]["airport_queue"]
            .as_f64()
            .expect("airport queue contribution")
            > 10.0
    );
    assert_eq!(quantiles["records"].as_array().expect("quantiles").len(), 6);
    assert_eq!(samples["sample_count"].as_u64(), Some(8));
    assert_eq!(samples["records"].as_array().expect("samples").len(), 24);
    assert_eq!(
        model.metadata()["custom_seasonalities"][0]["condition_name"].as_str(),
        Some("rush_hour")
    );
    assert_eq!(model.metadata()["fit_loss"].as_str(), Some("huber"));
    assert_eq!(
        model.metadata()["changepoint_timestamps"][0].as_str(),
        Some("2026-01-22T00:00:00")
    );
    assert_eq!(
        restored_forecast.predictions().len(),
        forecast.predictions().len()
    );
    assert_eq!(
        restored_forecast.intervals().len(),
        forecast.intervals().len()
    );
    for (restored_prediction, prediction) in restored_forecast
        .predictions()
        .iter()
        .zip(forecast.predictions().iter())
    {
        assert_eq!(restored_prediction.series_id, prediction.series_id);
        assert_eq!(restored_prediction.timestamp, prediction.timestamp);
        assert_eq!(restored_prediction.horizon, prediction.horizon);
        assert_eq!(restored_prediction.model, prediction.model);
        assert_close(restored_prediction.mean, prediction.mean);
    }
    for (restored_interval, interval) in restored_forecast
        .intervals()
        .iter()
        .zip(forecast.intervals().iter())
    {
        assert_eq!(restored_interval.series_id, interval.series_id);
        assert_eq!(restored_interval.timestamp, interval.timestamp);
        assert_eq!(restored_interval.horizon, interval.horizon);
        assert_eq!(restored_interval.model, interval.model);
        assert_close(restored_interval.level, interval.level);
        assert_close(restored_interval.lower, interval.lower);
        assert_close(restored_interval.upper, interval.upper);
    }
}

#[test]
fn piecewise_linear_seasonal_public_api_rejects_invalid_or_missing_future_state() {
    let invalid_event_config = PiecewiseLinearSeasonalConfig {
        events: vec![PiecewiseLinearEvent {
            name: "airport_surge".to_string(),
            timestamp: ts(5),
            lower_window: 1,
            upper_window: -1,
        }],
        ..PiecewiseLinearSeasonalConfig::default()
    };
    assert!(PiecewiseLinearSeasonalForecaster::new(invalid_event_config)
        .expect_err("invalid event window")
        .to_string()
        .contains("lower_window must be <= upper_window"));

    let rows = (1..=14)
        .map(|day| {
            let airport_queue = if day % 2 == 0 { 1.0 } else { 0.0 };
            ForecastRow::with_covariates(
                "__single__",
                ts(day),
                20.0 + day as f64 + 5.0 * airport_queue,
                BTreeMap::from([("airport_queue".to_string(), airport_queue)]),
            )
        })
        .collect();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("valid frame");
    let base_config = PiecewiseLinearSeasonalConfig {
        changepoints: 0,
        weekly_fourier_order: 0,
        auto_weekly_seasonality: false,
        extra_regressors: vec!["airport_queue".to_string()],
        ..PiecewiseLinearSeasonalConfig::default()
    };
    let mut missing_future =
        PiecewiseLinearSeasonalForecaster::new(base_config.clone()).expect("valid missing config");
    let mut short_future = PiecewiseLinearSeasonalForecaster::new(PiecewiseLinearSeasonalConfig {
        future_regressors: BTreeMap::from([("airport_queue".to_string(), vec![1.0])]),
        ..base_config
    })
    .expect("valid short config");

    missing_future.fit(&frame).expect("fit missing future");
    short_future.fit(&frame).expect("fit short future");

    assert!(missing_future
        .predict(1)
        .expect_err("future regressor is required")
        .to_string()
        .contains("future"));
    assert!(short_future
        .predict(2)
        .expect_err("future regressor horizon is required")
        .to_string()
        .contains("fewer than 2 values"));

    let mut updated = missing_future.clone();
    assert!(updated
        .update_config(|config| {
            config.future_regressors =
                BTreeMap::from([("airport_queue".to_string(), vec![1.0, 0.0])]);
        })
        .is_ok());
    assert_eq!(
        updated
            .predict(2)
            .expect("predict with future")
            .predictions()
            .len(),
        2
    );
}

#[test]
fn kalman_tracks_linear_signal_and_preserves_panel_indexing() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=132", ts(1), 10.0),
            ForecastRow::new("PULocationID=132", ts(2), 12.0),
            ForecastRow::new("PULocationID=132", ts(3), 14.0),
            ForecastRow::new("PULocationID=132", ts(4), 16.0),
            ForecastRow::new("PULocationID=236", ts(1), 40.0),
            ForecastRow::new("PULocationID=236", ts(2), 38.0),
            ForecastRow::new("PULocationID=236", ts(3), 36.0),
            ForecastRow::new("PULocationID=236", ts(4), 34.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");
    let mut model = KalmanForecaster::new(10.0, 10.0, 1.0e-6).expect("valid kalman");

    model.fit(&frame).expect("fit kalman");
    let predictions = model.predict(2).expect("predict").predictions().to_vec();

    assert_eq!(predictions.len(), 4);
    assert_eq!(predictions[0].series_id, "PULocationID=132");
    assert_eq!(predictions[0].timestamp, ts(5));
    assert_eq!(predictions[1].horizon, 2);
    assert_eq!(predictions[2].series_id, "PULocationID=236");
    assert!(predictions[1].mean > predictions[0].mean);
    assert!(predictions[3].mean < predictions[2].mean);
    assert_eq!(model.metadata()["model"], "kalman");
}

#[test]
fn kriging_holds_out_each_target_series_and_reports_configuration() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=142", ts(1), 10.0),
            ForecastRow::new("PULocationID=142", ts(2), 12.0),
            ForecastRow::new("PULocationID=236", ts(1), 40.0),
            ForecastRow::new("PULocationID=236", ts(2), 42.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");
    let coordinates = BTreeMap::from([
        ("PULocationID=142".to_string(), (0.0, 0.0)),
        ("PULocationID=236".to_string(), (10.0, 0.0)),
    ]);
    let mut model = KrigingForecaster::new(coordinates, 1.0, 1.0e-9).expect("valid kriging");

    model.fit(&frame).expect("fit kriging");
    let forecast = model.predict(1).expect("predict");
    let predictions = forecast.predictions();

    assert_close(predictions[0].mean, 42.0);
    assert_close(predictions[1].mean, 12.0);
    assert_eq!(model.metadata()["series_count"], 2);
    assert_eq!(model.metadata()["model"], "kriging");
    assert_eq!(model.metadata()["target_policy"], "leave_one_series_out");
}

#[test]
fn weighted_ensemble_uses_normalized_weights_for_exact_member_average() {
    let frame = single_frame(&[10.0, 12.0, 14.0]);
    let mut model = WeightedEnsembleForecaster::new(vec![
        ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
        (
            "seasonal".to_string(),
            Box::new(SeasonalNaiveForecaster::new(2).expect("valid seasonal")),
            3.0,
        ),
    ])
    .expect("valid ensemble");

    model.fit(&frame).expect("fit ensemble");

    assert_eq!(means(&model, 2), vec![12.5, 14.0]);
    assert_close(model.weights()["last"], 0.25);
    assert_close(model.weights()["seasonal"], 0.75);
    assert_eq!(model.metadata()["model"], "weighted_ensemble");
}

#[test]
fn cartoboost_lag_delta_mode_learns_linear_increment_exactly() {
    let frame = single_frame(&[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    let booster_config = BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 0,
        min_samples_leaf: 1,
        min_gain: 0.0,
        ..Default::default()
    };
    let mut model = CartoBoostLagForecaster::new_with_target_mode(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            difference_lags: vec![1],
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
            calendar_features: vec![CalendarFeature::DayOfWeek],
        },
        booster_config,
        GlobalForecastTargetMode::DeltaFromLast,
    )
    .expect("valid cartoboost lag");

    model.fit(&frame).expect("fit cartoboost lag");

    assert_eq!(model.training_rows(), Some(4));
    assert_eq!(means(&model, 2), vec![22.0, 24.0]);
    assert_eq!(model.metadata()["target_mode"], "delta_from_last");
}
