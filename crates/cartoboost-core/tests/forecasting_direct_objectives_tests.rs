use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::forecasting::{
    adida_forecast, croston_forecast, sba_forecast, tsb_forecast, CalendarFeature,
    CartoBoostDirectForecaster, ForecastFrame, ForecastFrequency, ForecastOutput, ForecastRequest,
    ForecastRow, Forecaster, LagFeatureConfig, RectifiedRecursiveForecaster,
};
use cartoboost_core::objectives::{
    CountObjective, HurdleObjective, NegativeBinomialObjective, PoissonObjective, TweedieObjective,
};
use chrono::{NaiveDate, NaiveDateTime};

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

fn small_booster_config() -> BoosterConfig {
    BoosterConfig {
        n_estimators: 12,
        learning_rate: 0.25,
        max_depth: 2,
        min_samples_leaf: 1,
        min_gain: 0.0,
        ..BoosterConfig::default()
    }
}

fn lag_config() -> LagFeatureConfig {
    LagFeatureConfig {
        lags: vec![1, 2],
        rolling_mean_windows: vec![2],
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
        calendar_features: vec![CalendarFeature::Day],
    }
}

fn frame() -> ForecastFrame {
    ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=1", ts(1), 10.0),
            ForecastRow::new("PULocationID=1", ts(2), 12.0),
            ForecastRow::new("PULocationID=1", ts(3), 14.0),
            ForecastRow::new("PULocationID=1", ts(4), 16.0),
            ForecastRow::new("PULocationID=1", ts(5), 18.0),
            ForecastRow::new("PULocationID=1", ts(6), 20.0),
            ForecastRow::new("PULocationID=9", ts(1), 30.0),
            ForecastRow::new("PULocationID=9", ts(2), 33.0),
            ForecastRow::new("PULocationID=9", ts(3), 36.0),
            ForecastRow::new("PULocationID=9", ts(4), 39.0),
            ForecastRow::new("PULocationID=9", ts(5), 42.0),
            ForecastRow::new("PULocationID=9", ts(6), 45.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame")
}

#[test]
fn direct_forecaster_trains_one_model_per_horizon_and_predicts_panel() {
    let mut forecaster =
        CartoBoostDirectForecaster::new(lag_config(), small_booster_config()).expect("forecaster");

    forecaster.fit_horizon(&frame(), 3).expect("fit");
    let predictions = forecaster.predict(3).expect("predict");

    assert_eq!(forecaster.models().expect("models").len(), 3);
    assert_eq!(forecaster.training_rows_by_horizon(), Some(&[8, 6, 4][..]));
    assert_eq!(predictions.predictions().len(), 6);
    assert_eq!(predictions.predictions()[0].timestamp, ts(7));
    assert_eq!(predictions.predictions()[1].horizon, 2);
    assert!(predictions
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
    assert!(
        predictions.predictions()[3].mean > predictions.predictions()[0].mean,
        "higher pickup zone history should remain higher in direct predictions"
    );
}

#[test]
fn direct_forecaster_rejects_unfitted_or_untrained_horizons() {
    let mut forecaster =
        CartoBoostDirectForecaster::new(lag_config(), small_booster_config()).expect("forecaster");

    assert!(forecaster.predict(1).is_err());
    forecaster.fit_horizon(&frame(), 1).expect("fit");
    assert!(forecaster.predict(2).is_err());
}

#[test]
fn rectified_recursive_forecaster_fits_horizon_specific_corrections() {
    let mut forecaster = RectifiedRecursiveForecaster::new(lag_config(), small_booster_config())
        .expect("forecaster");

    forecaster.fit_horizon(&frame(), 2).expect("fit");
    let predictions = forecaster.predict(2).expect("predict");

    assert_eq!(forecaster.training_rows_by_horizon(), Some(&[4, 2][..]));
    assert_eq!(predictions.predictions().len(), 4);
    assert!(predictions
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn rectified_recursive_reuses_recursive_training_paths_for_long_horizons() {
    let rows = (0..180)
        .map(|hour| {
            ForecastRow::new(
                "PU132-DO236",
                ts(1) + chrono::Duration::hours(hour),
                30.0 + f64::from((hour % 24) as u32) * 0.35 + f64::from((hour / 24) as u32) * 0.2,
            )
        })
        .collect();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("valid hourly frame");
    let config = LagFeatureConfig {
        lags: vec![1, 24],
        rolling_mean_windows: vec![24],
        difference_lags: vec![1],
        calendar_features: vec![
            CalendarFeature::DayOfWeek,
            CalendarFeature::ElapsedPhase(24),
        ],
        ..LagFeatureConfig::default()
    };
    let booster = BoosterConfig {
        n_estimators: 2,
        learning_rate: 0.2,
        max_depth: 1,
        min_samples_leaf: 8,
        min_gain: 0.0,
        ..BoosterConfig::default()
    };
    let mut forecaster =
        RectifiedRecursiveForecaster::new(config, booster).expect("rectified forecaster");

    forecaster.fit_horizon(&frame, 14).expect("fit");
    let predictions = forecaster.predict(14).expect("predict");

    assert_eq!(
        forecaster.training_rows_by_horizon().expect("rows").len(),
        14
    );
    assert_eq!(predictions.predictions().len(), 14);
    assert!(predictions
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn intermittent_experts_are_deterministic_and_validate_inputs() {
    let values = [0.0, 5.0, 0.0, 0.0, 7.0, 0.0, 9.0, 0.0];

    let croston = croston_forecast(&values, 2, 0.2).expect("croston");
    let sba = sba_forecast(&values, 2, 0.2).expect("sba");
    let tsb = tsb_forecast(&values, 2, 0.2, 0.3).expect("tsb");
    let adida = adida_forecast(&values, 2, 2, 0.2).expect("adida");

    assert_eq!(croston.len(), 2);
    assert_eq!(croston[0], croston[1]);
    assert!(sba[0] < croston[0]);
    assert!(tsb[0] > 0.0);
    assert!(adida[0] > 0.0);
    assert_eq!(croston_forecast(&[0.0, 0.0], 1, 0.2).unwrap(), vec![0.0]);
    assert!(adida_forecast(&values, 1, 0, 0.2).is_err());
    assert!(adida_forecast(&values[..1], 1, 2, 0.2).is_err());
}

#[test]
fn horizon_request_and_output_validate_finite_public_contract() {
    let request = ForecastRequest::with_series_ids(
        3,
        Some(vec![
            "PULocationID=1".to_string(),
            "PULocationID=9".to_string(),
        ]),
    )
    .expect("request");
    let output = ForecastOutput::new("PULocationID=1", ts(7), 1, 22.0).expect("output");

    assert_eq!(request.horizon, 3);
    assert_eq!(output.prediction, 22.0);
    assert!(ForecastRequest::new(0).is_err());
    assert!(ForecastOutput::new("PULocationID=1", ts(7), 0, 22.0).is_err());
}

#[test]
fn count_objective_derivatives_match_finite_differences() {
    assert_derivatives(PoissonObjective, 3.0, 0.4);
    assert_derivatives(NegativeBinomialObjective::new(0.7).expect("nb"), 3.0, 0.4);
    assert_derivatives(TweedieObjective::new(1.5).expect("tweedie"), 3.0, 0.4);
}

#[test]
fn hurdle_objective_separates_zero_gate_from_positive_count_loss() {
    let objective = HurdleObjective::new(PoissonObjective);
    let zero = objective
        .value_gradient_hessian(0.0, -0.4, 0.2)
        .expect("zero derivatives");
    let positive = objective
        .value_gradient_hessian(4.0, -0.4, 0.2)
        .expect("positive derivatives");

    assert!(zero.zero_gradient > 0.0);
    assert!(positive.zero_gradient < 0.0);
    assert!(positive.positive.gradient.is_finite());
    assert!(NegativeBinomialObjective::new(0.0).is_err());
    assert!(TweedieObjective::new(2.0).is_err());
}

fn assert_derivatives<T>(objective: T, target: f64, raw_prediction: f64)
where
    T: CountObjective,
{
    let analytic = objective
        .value_gradient_hessian(target, raw_prediction)
        .expect("analytic derivatives");
    let eps = 1e-5;
    let plus = objective
        .value_gradient_hessian(target, raw_prediction + eps)
        .expect("plus")
        .value;
    let minus = objective
        .value_gradient_hessian(target, raw_prediction - eps)
        .expect("minus")
        .value;
    let finite_gradient = (plus - minus) / (2.0 * eps);

    assert!((analytic.gradient - finite_gradient).abs() < 1e-5);
    assert!(analytic.hessian >= 0.0);
}
