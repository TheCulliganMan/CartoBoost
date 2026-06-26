pub use cartoboost_neural::{NeuralError, Result};

#[allow(dead_code, unused_imports)]
#[path = "../src/forecasting/mod.rs"]
mod forecasting;

use cartoboost_core::forecasting::{
    CartoBoostLagForecaster, ForecastFrame, ForecastFrequency, ForecastPrediction, ForecastRow,
    Forecaster, LagFeatureConfig, SeasonalNaiveForecaster,
};
use cartoboost_core::BoosterConfig;
use forecasting::{
    LaneNeuralPairwiseConfig, LaneNeuralPairwiseForecaster, NBeatsConfig, NBeatsForecaster,
    NHiTSConfig, NHiTSForecaster, NeuralPairwiseConfig, NeuralPairwiseForecaster,
    NeuralPairwiseMode, StandardScaler,
};
use std::collections::BTreeMap;

#[test]
fn nbeats_forecaster_is_deterministic_on_cpu() {
    let frame = taxi_frame();
    let config = NBeatsConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 30,
        learning_rate: 0.01,
    };
    let mut first = NBeatsForecaster::new(config.clone()).expect("first model");
    let mut second = NBeatsForecaster::new(config).expect("second model");

    first.fit(&frame).expect("first fit");
    second.fit(&frame).expect("second fit");

    let first_predictions = first.predict(3).expect("first predict");
    let second_predictions = second.predict(3).expect("second predict");

    assert_eq!(first_predictions, second_predictions);
    assert_eq!(first_predictions.predictions().len(), 6);
    assert!(first_predictions
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn nhits_forecaster_handles_panel_taxi_series() {
    let frame = taxi_frame();
    let config = NHiTSConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 30,
        learning_rate: 0.01,
        pooling_size: 2,
    };
    let mut model = NHiTSForecaster::new(config).expect("model");

    model.fit(&frame).expect("fit");
    let predictions = model.predict(2).expect("predict");

    assert_eq!(predictions.predictions().len(), 4);
    assert_eq!(predictions.predictions()[0].series_id, "PU1->DO2");
    assert_eq!(predictions.predictions()[0].horizon, 1);
    assert_eq!(predictions.predictions()[1].horizon, 2);
    assert_eq!(predictions.predictions()[2].series_id, "PU3->DO4");
}

#[test]
fn scaler_round_trips_constant_series() {
    let scaler = StandardScaler::fit(&[42.0, 42.0, 42.0]).expect("scaler");

    assert_eq!(scaler.transform(42.0), 0.0);
    assert_eq!(scaler.inverse_transform(0.0), 42.0);
    assert!(scaler.scale() > 0.0);
}

#[test]
fn neural_pairwise_outputs_b_h_q_shape() {
    let frame = taxi_colon_frame();
    let mut model = NeuralPairwiseForecaster::new(NeuralPairwiseConfig {
        n_lags: 3,
        n_forecasts: 2,
        quantiles: vec![0.1, 0.5, 0.9],
        ..NeuralPairwiseConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let tensor = model.predict_tensor(2).expect("tensor");

    assert_eq!(tensor.len(), 2);
    for rows in tensor.values() {
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|horizon| horizon.len() == 3));
    }
}

#[test]
fn neural_pairwise_keeps_directional_taxi_lanes_distinct() {
    let frame = taxi_colon_frame();
    let mut model = LaneNeuralPairwiseForecaster::new(LaneNeuralPairwiseConfig {
        base: NeuralPairwiseConfig {
            n_lags: 3,
            n_forecasts: 1,
            trend_mode: NeuralPairwiseMode::Local,
            local_l2: 0.0,
            ..NeuralPairwiseConfig::default()
        },
        embedding_dim: 4,
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let predictions = model.predict(1).expect("predict");
    let values = predictions.predictions();

    assert_eq!(values[0].series_id, "A:B");
    assert_eq!(values[1].series_id, "B:A");
    assert_ne!(values[0].mean, values[1].mean);
    let metadata = model.metadata();
    assert_eq!(
        metadata["lane_config"]["fallback_index"]["A:B"][0],
        "pair:A:B"
    );
}

#[test]
fn neural_pairwise_window_construction_has_no_future_target_leakage() {
    let frame = taxi_colon_frame();
    let model = NeuralPairwiseForecaster::new(NeuralPairwiseConfig {
        n_lags: 2,
        n_forecasts: 2,
        future_regressors: BTreeMap::from([(
            "is_airport".to_string(),
            forecasting::ComponentMode::Additive,
        )]),
        ..NeuralPairwiseConfig::default()
    })
    .expect("model");

    let dataset = model.window_dataset(&frame).expect("windows");
    let first = &dataset.windows()[0];

    assert_eq!(first.lags, vec![10.0, 11.0]);
    assert_eq!(first.targets, vec![12.0, 13.0]);
    assert_eq!(first.future_features.len(), 4);
}

#[test]
fn neural_pairwise_predict_quantiles_are_non_crossing() {
    let frame = taxi_colon_frame();
    let mut model = NeuralPairwiseForecaster::new(NeuralPairwiseConfig {
        n_lags: 3,
        n_forecasts: 1,
        quantiles: vec![0.9, 0.1, 0.5],
        ..NeuralPairwiseConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let tensor = model.predict_tensor(3).expect("tensor");

    for series in tensor.values() {
        for horizon in series {
            assert!(horizon.windows(2).all(|pair| pair[0] <= pair[1]));
        }
    }
}

#[test]
fn neural_pairwise_global_mode_has_no_local_deviations() {
    let frame = taxi_colon_frame();
    let mut model = NeuralPairwiseForecaster::new(NeuralPairwiseConfig {
        n_lags: 3,
        n_forecasts: 1,
        trend_mode: NeuralPairwiseMode::Global,
        ..NeuralPairwiseConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert_eq!(
        metadata["component_params"]["local_levels"]
            .as_object()
            .expect("local levels")
            .len(),
        0
    );
}

#[test]
fn neural_pairwise_compares_on_synthetic_panel_with_baselines() {
    let (train, expected) = synthetic_seasonal_panel();
    let horizon = 4;
    let mut neural = NeuralPairwiseForecaster::new(NeuralPairwiseConfig {
        n_lags: 8,
        n_forecasts: horizon,
        quantiles: vec![0.1, 0.5, 0.9],
        daily_fourier_order: 2,
        trend_mode: NeuralPairwiseMode::Local,
        local_l2: 0.1,
        seed: 7,
        ..NeuralPairwiseConfig::default()
    })
    .expect("neural model");
    let mut seasonal = SeasonalNaiveForecaster::new(4).expect("seasonal naive");
    let mut lag = CartoBoostLagForecaster::new(
        LagFeatureConfig {
            lags: vec![1, 4, 8],
            rolling_mean_windows: vec![4],
            ..LagFeatureConfig::default()
        },
        BoosterConfig {
            n_estimators: 20,
            max_depth: 2,
            min_samples_leaf: 2,
            ..BoosterConfig::default()
        },
    )
    .expect("lag model");

    neural.fit(&train).expect("neural fit");
    seasonal.fit(&train).expect("seasonal fit");
    lag.fit(&train).expect("lag fit");

    let neural_predictions = neural.predict(horizon).expect("neural predict");
    let seasonal_predictions = seasonal.predict(horizon).expect("seasonal predict");
    let lag_predictions = lag.predict(horizon).expect("lag predict");

    for (name, predictions) in [
        ("neural_pairwise", neural_predictions.predictions()),
        ("seasonal_naive", seasonal_predictions.predictions()),
        ("cartoboost_lag", lag_predictions.predictions()),
    ] {
        assert_eq!(
            predictions.len(),
            expected.len(),
            "{name} aligned row count"
        );
        assert!(
            predictions
                .iter()
                .all(|prediction| prediction.mean.is_finite()),
            "{name} finite predictions"
        );
        let rmse = aligned_rmse(predictions, &expected);
        assert!(rmse.is_finite(), "{name} finite rmse");
        assert!(rmse < 30.0, "{name} synthetic rmse is bounded: {rmse}");
    }
}

fn taxi_frame() -> ForecastFrame {
    let rows = (1..=10)
        .flat_map(|hour| {
            [
                ForecastRow::from_timestamp_str(
                    "PU1->DO2",
                    &timestamp(hour),
                    10.0 + hour as f64 * 1.5,
                )
                .expect("row"),
                ForecastRow::from_timestamp_str(
                    "PU3->DO4",
                    &timestamp(hour),
                    25.0 - hour as f64 * 0.75,
                )
                .expect("row"),
            ]
        })
        .collect();
    ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame")
}

fn taxi_colon_frame() -> ForecastFrame {
    let rows = (0..8)
        .flat_map(|hour| {
            [
                ForecastRow::from_timestamp_str_with_covariates(
                    "A:B",
                    &timestamp(hour),
                    10.0 + hour as f64,
                    BTreeMap::from([("is_airport".to_string(), (hour % 2) as f64)]),
                )
                .expect("row"),
                ForecastRow::from_timestamp_str_with_covariates(
                    "B:A",
                    &timestamp(hour),
                    30.0 - hour as f64,
                    BTreeMap::from([("is_airport".to_string(), ((hour + 1) % 2) as f64)]),
                )
                .expect("row"),
            ]
        })
        .collect();
    ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame")
}

fn synthetic_seasonal_panel() -> (ForecastFrame, BTreeMap<(String, usize), f64>) {
    let horizon = 4;
    let total = 36;
    let train_len = total - horizon;
    let mut rows = Vec::new();
    let mut expected = BTreeMap::new();
    for series_id in ["PU1:DO2", "PU2:DO1"] {
        for hour in 0..total {
            let direction = if series_id == "PU1:DO2" { 1.0 } else { -1.0 };
            let value = 20.0
                + direction * 3.0
                + 0.35 * hour as f64
                + match hour % 4 {
                    0 => 0.0,
                    1 => 2.0,
                    2 => -1.0,
                    _ => 1.0,
                };
            if hour < train_len {
                rows.push(
                    ForecastRow::from_timestamp_str(series_id, &synthetic_timestamp(hour), value)
                        .expect("row"),
                );
            } else {
                expected.insert((series_id.to_string(), hour - train_len + 1), value);
            }
        }
    }
    (
        ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame"),
        expected,
    )
}

fn aligned_rmse(
    predictions: &[ForecastPrediction],
    expected: &BTreeMap<(String, usize), f64>,
) -> f64 {
    let mse = predictions
        .iter()
        .map(|prediction| {
            let actual = expected
                .get(&(prediction.series_id.clone(), prediction.horizon))
                .expect("expected row");
            let residual = prediction.mean - actual;
            residual * residual
        })
        .sum::<f64>()
        / predictions.len() as f64;
    mse.sqrt()
}

fn timestamp(hour: u32) -> String {
    format!("2024-01-01T{hour:02}:00:00")
}

fn synthetic_timestamp(hour: usize) -> String {
    let day = 1 + hour / 24;
    let hour = hour % 24;
    format!("2024-01-{day:02}T{hour:02}:00:00")
}
