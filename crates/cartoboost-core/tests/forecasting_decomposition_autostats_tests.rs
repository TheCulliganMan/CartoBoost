use cartoboost_core::forecasting::{
    AutoStatsBank, ClassicalExpert, ClassicalExpertBank, ForecastFrame, ForecastFrequency,
    ForecastRow, Forecaster, MSTLCartoBoostForecaster, MSTLDecomposition, NaiveForecaster,
    STLCartoBoostForecaster, STLDecomposition,
};
use chrono::{NaiveDate, NaiveDateTime};

fn ts(hour: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, 1)
        .and_then(|date| date.and_hms_opt(hour, 0, 0))
        .expect("valid fixture timestamp")
}

fn taxi_hourly_frame(values: &[f64]) -> ForecastFrame {
    ForecastFrame::new(
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| ForecastRow::new("PU1-DO2", ts(idx as u32), *value))
            .collect(),
        ForecastFrequency::Hourly,
    )
    .expect("valid hourly taxi frame")
}

#[test]
fn stl_recomposes_observed_taxi_series() {
    let values = vec![10.0, 14.0, 11.0, 15.0, 12.0, 16.0, 13.0, 17.0];
    let stl = STLDecomposition::new(2).expect("valid stl");
    let result = stl.decompose(&values).expect("decomposed");

    assert_eq!(result.len(), values.len());
    assert!(result.max_abs_recomposition_error() <= 1e-10);
    for (actual, recomposed) in values.iter().zip(result.recompose()) {
        assert!((actual - recomposed).abs() <= 1e-10);
    }
}

#[test]
fn mstl_recomposes_multiple_taxi_seasons() {
    let values = (0..24)
        .map(|idx| 40.0 + idx as f64 * 0.25 + [3.0, -1.0, 2.0][idx % 3] + [1.5, -1.5][idx % 2])
        .collect::<Vec<_>>();
    let mstl = MSTLDecomposition::new(vec![3, 2]).expect("valid mstl");
    let result = mstl.decompose(&values).expect("decomposed");

    assert_eq!(result.seasonal_components.len(), 2);
    assert!(result.max_abs_recomposition_error() <= 1e-10);
    for (actual, recomposed) in values.iter().zip(result.recompose()) {
        assert!((actual - recomposed).abs() <= 1e-10);
    }
}

#[test]
fn stl_hybrid_forecasts_seasonally_adjusted_target_then_reseasons() {
    let values = [10.0, 13.0, 11.0, 14.0, 12.0, 15.0];
    let frame = taxi_hourly_frame(&values);
    let decomposition = STLDecomposition::new(2).expect("valid stl");
    let fitted_decomposition = decomposition
        .decompose(&values)
        .expect("decomposition succeeds");
    let mut model = STLCartoBoostForecaster::with_remainder_forecaster(
        decomposition,
        Box::new(NaiveForecaster::new()),
    )
    .expect("valid hybrid");

    model.fit(&frame).expect("fit hybrid");
    let forecast = model.predict(2).expect("forecast");

    assert_eq!(forecast.predictions().len(), 2);
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.model == "stl_cartoboost"));
    let adjusted_last = values[values.len() - 1]
        - fitted_decomposition.seasonal[fitted_decomposition.seasonal.len() - 1];
    let expected = [
        adjusted_last + fitted_decomposition.seasonal[fitted_decomposition.seasonal.len() - 2],
        adjusted_last + fitted_decomposition.seasonal[fitted_decomposition.seasonal.len() - 1],
    ];
    for (prediction, expected) in forecast.predictions().iter().zip(expected) {
        assert!((prediction.mean - expected).abs() <= 1.0e-10);
    }
}

#[test]
fn mstl_hybrid_rejects_incomplete_seasonal_cycles() {
    let frame = taxi_hourly_frame(&[20.0, 23.0, 21.0]);
    let mut model = MSTLCartoBoostForecaster::with_remainder_forecaster(
        MSTLDecomposition::new(vec![2, 3]).expect("valid mstl"),
        Box::new(NaiveForecaster::new()),
    )
    .expect("valid hybrid");

    let error = model
        .fit(&frame)
        .expect_err("incomplete seasonal cycles must fail");
    assert!(error.to_string().contains("two complete seasonal cycles"));
}

#[test]
fn classical_bank_gates_unsupported_short_history_experts() {
    let frame = taxi_hourly_frame(&[9.0, 10.0, 11.0]);
    let mut bank = ClassicalExpertBank::new(vec![
        ClassicalExpert::SeasonalNaive { season_length: 24 },
        ClassicalExpert::Naive,
    ])
    .expect("valid bank");

    bank.fit(&frame).expect("fit bank");
    assert_eq!(bank.selected_expert(), Some(&ClassicalExpert::Naive));
    assert_eq!(bank.validation_scores().len(), 1);
    let forecast = bank.predict(2).expect("forecast");
    assert_eq!(forecast.predictions().len(), 2);
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn classical_bank_default_includes_seasonal_window_average() {
    let bank = ClassicalExpertBank::default_for_season_length(7).expect("default bank");

    assert!(bank
        .experts()
        .contains(&ClassicalExpert::WindowAverage { window_size: 3 }));
    assert!(bank
        .experts()
        .contains(&ClassicalExpert::WindowAverage { window_size: 7 }));
    assert!(bank
        .experts()
        .contains(&ClassicalExpert::SeasonalWindowAverage {
            season_length: 7,
            window_count: 3,
        }));
}

#[test]
fn classical_bank_default_skips_seasonal_window_average_without_seasonality() {
    let bank = ClassicalExpertBank::default_for_season_length(1).expect("default bank");

    assert!(!bank
        .experts()
        .iter()
        .any(|expert| matches!(expert, ClassicalExpert::SeasonalWindowAverage { .. })));
}

#[test]
fn autostats_bank_selects_and_predicts() {
    let frame = taxi_hourly_frame(&[30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0]);
    let mut bank = AutoStatsBank::with_validation_window(2, Some(2)).expect("valid autostats");

    bank.fit(&frame).expect("fit autostats");
    assert!(bank.selected_expert().is_some());
    assert!(!bank.validation_scores().is_empty());
    let forecast = bank.predict(3).expect("forecast");
    assert_eq!(forecast.predictions().len(), 3);
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.model == "autostats_bank"));
}
