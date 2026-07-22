use cartoboost_core::forecasting::{
    CartoBoostLagForecaster, ForecastFrame, ForecastFrequency, ForecastPrediction, ForecastRow,
    Forecaster, LagFeatureConfig, SeasonalNaiveForecaster,
};
use cartoboost_core::BoosterConfig;
use cartoboost_neural::{
    available_backends, fit_dense_regressor_with_backend, select_backend, ComponentMode,
    DenseRegressorConfig, LaneNeuralPanelConfig, LaneNeuralPanelForecaster, NBeatsConfig,
    NBeatsForecaster, NHiTSConfig, NHiTSForecaster, NeuralPanelConfig, NeuralPanelForecaster,
    NeuralPanelMode, StandardScaler, TrendMode,
};
use std::collections::BTreeMap;

#[test]
fn reusable_dense_regressor_trains_on_every_available_backend() {
    let config = DenseRegressorConfig {
        input_width: 2,
        output_width: 1,
        hidden_layers: vec![3],
        epochs: 2,
        learning_rate: 0.01,
        weight_decay: 1.0e-5,
        seed: 19,
    };
    let examples = vec![
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![2.0]),
        (vec![1.0, 1.0], vec![3.0]),
    ];
    for backend in available_backends() {
        let state = fit_dense_regressor_with_backend(&config, examples.clone(), Some(&backend))
            .unwrap_or_else(|error| panic!("{backend} dense training failed: {error}"));
        let serialized = serde_json::to_value(state).expect("serializable MLP state");
        assert_eq!(serialized["input_width"].as_u64(), Some(2));
        assert_eq!(serialized["output_width"].as_u64(), Some(1));
    }
}

#[test]
fn nbeats_forecaster_is_deterministic_on_cpu() {
    let frame = taxi_frame();
    let config = NBeatsConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 30,
        learning_rate: 0.01,
        backend: select_backend(Some("cpu")).expect("CPU backend"),
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
        ..NHiTSConfig::default()
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

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
#[test]
fn metal_nbeats_training_and_prediction_remain_numerically_close_to_cpu() {
    if !cartoboost_neural::available_backends()
        .iter()
        .any(|backend| backend == "metal")
    {
        return;
    }
    let frame = taxi_frame();
    let mut cpu_config = NBeatsConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 10,
        learning_rate: 0.01,
        ..NBeatsConfig::default()
    };
    cpu_config.backend = cartoboost_neural::select_backend(Some("cpu")).unwrap();
    let mut metal_config = cpu_config.clone();
    metal_config.backend = cartoboost_neural::select_backend(Some("metal")).unwrap();

    let mut cpu = NBeatsForecaster::new(cpu_config).unwrap();
    let mut metal = NBeatsForecaster::new(metal_config).unwrap();
    cpu.fit(&frame).unwrap();
    metal.fit(&frame).unwrap();
    let cpu_predictions = cpu.predict(3).unwrap();
    let metal_predictions = metal.predict(3).unwrap();
    assert_forecasts_close(
        metal_predictions.predictions(),
        cpu_predictions.predictions(),
        0.5,
    );
}

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
#[test]
fn metal_nhits_training_and_prediction_remain_numerically_close_to_cpu() {
    if !cartoboost_neural::available_backends()
        .iter()
        .any(|backend| backend == "metal")
    {
        return;
    }
    let frame = taxi_frame();
    let mut cpu_config = NHiTSConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 10,
        learning_rate: 0.01,
        pooling_size: 2,
        ..NHiTSConfig::default()
    };
    cpu_config.backend = cartoboost_neural::select_backend(Some("cpu")).unwrap();
    let mut metal_config = cpu_config.clone();
    metal_config.backend = cartoboost_neural::select_backend(Some("metal")).unwrap();

    let mut cpu = NHiTSForecaster::new(cpu_config).unwrap();
    let mut metal = NHiTSForecaster::new(metal_config).unwrap();
    cpu.fit(&frame).unwrap();
    metal.fit(&frame).unwrap();
    let cpu_predictions = cpu.predict(3).unwrap();
    let metal_predictions = metal.predict(3).unwrap();
    assert_forecasts_close(
        metal_predictions.predictions(),
        cpu_predictions.predictions(),
        0.5,
    );
}

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
#[test]
fn metal_neural_panel_predictions_match_cpu_backend() {
    if !cartoboost_neural::available_backends()
        .iter()
        .any(|backend| backend == "metal")
    {
        return;
    }
    let frame = taxi_colon_frame();
    let mut cpu_config = NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 2,
        quantiles: vec![0.5],
        ar_layers: vec![4],
        epochs: 8,
        learning_rate: 0.01,
        ..NeuralPanelConfig::default()
    };
    cpu_config.backend = cartoboost_neural::select_backend(Some("cpu")).unwrap();
    let mut metal_config = cpu_config.clone();
    metal_config.backend = cartoboost_neural::select_backend(Some("metal")).unwrap();

    let mut cpu = NeuralPanelForecaster::new(cpu_config).unwrap();
    let mut metal = NeuralPanelForecaster::new(metal_config).unwrap();
    cpu.fit(&frame).unwrap();
    metal.fit(&frame).unwrap();
    let cpu_predictions = cpu.predict(2).unwrap();
    let metal_predictions = metal.predict(2).unwrap();
    assert_forecasts_close(
        metal_predictions.predictions(),
        cpu_predictions.predictions(),
        1.0e-4,
    );
}

#[test]
fn neural_panel_fit_and_predict_run_on_every_available_backend() {
    let frame = taxi_colon_frame();
    let mut cpu_config = NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        quantiles: vec![0.5],
        ar_layers: vec![3],
        epochs: 2,
        learning_rate: 0.01,
        ..NeuralPanelConfig::default()
    };
    cpu_config.backend = cartoboost_neural::select_backend(Some("cpu")).unwrap();
    let mut cpu = NeuralPanelForecaster::new(cpu_config.clone()).unwrap();
    cpu.fit(&frame).unwrap();
    let expected = cpu.predict(1).unwrap();

    for backend in available_backends() {
        let mut config = cpu_config.clone();
        config.backend = cartoboost_neural::select_backend(Some(&backend)).unwrap();
        let mut model = NeuralPanelForecaster::new(config).unwrap();
        model
            .fit(&frame)
            .unwrap_or_else(|error| panic!("{backend} neural-panel fit failed: {error}"));
        let actual = model
            .predict(1)
            .unwrap_or_else(|error| panic!("{backend} neural-panel predict failed: {error}"));
        assert_forecasts_close(actual.predictions(), expected.predictions(), 2.0e-3);
    }
}

#[test]
fn scaler_round_trips_constant_series() {
    let scaler = StandardScaler::fit(&[42.0, 42.0, 42.0]).expect("scaler");

    assert_eq!(scaler.transform(42.0), 0.0);
    assert_eq!(scaler.inverse_transform(0.0), 42.0);
    assert!(scaler.scale() > 0.0);
}

#[test]
fn neural_panel_outputs_b_h_q_shape() {
    let frame = taxi_colon_frame();
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 2,
        quantiles: vec![0.1, 0.5, 0.9],
        ..NeuralPanelConfig::default()
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
fn neural_panel_keeps_directional_taxi_lanes_distinct() {
    let frame = taxi_colon_frame();
    let mut model = LaneNeuralPanelForecaster::new(LaneNeuralPanelConfig {
        base: NeuralPanelConfig {
            n_lags: 3,
            n_forecasts: 1,
            trend_mode: NeuralPanelMode::Local,
            local_l2: 0.0,
            ..NeuralPanelConfig::default()
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
    let schema = metadata["feature_schema"]
        .as_array()
        .expect("feature schema");
    assert!(schema
        .iter()
        .any(|value| value == "lane_origin_embedding_0"));
    assert!(
        metadata["static_future_covariates"]["A:B"]["lane_embedding_0"]
            .as_f64()
            .expect("lane embedding")
            .is_finite()
    );
}

#[test]
fn lane_neural_panel_predict_for_lanes_uses_cold_origin_fallback() {
    let frame = taxi_colon_frame();
    let mut model = LaneNeuralPanelForecaster::new(LaneNeuralPanelConfig {
        base: NeuralPanelConfig {
            n_lags: 3,
            n_forecasts: 2,
            quantiles: vec![0.1, 0.5, 0.9],
            trend_mode: NeuralPanelMode::Local,
            local_l2: 0.0,
            ..NeuralPanelConfig::default()
        },
        embedding_dim: 4,
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let exact = model
        .predict_for_lanes(2, &["A:B".to_string()])
        .expect("exact predict");
    let cold = model
        .predict_for_lanes(2, &["A:C".to_string()])
        .expect("cold predict");
    let exact_rows = exact.predictions();
    let cold_rows = cold.predictions();

    assert_eq!(cold_rows.len(), 2);
    assert!(cold_rows
        .iter()
        .all(|prediction| prediction.series_id == "A:C"));
    assert_eq!(cold_rows[0].timestamp, exact_rows[0].timestamp);
    assert_eq!(cold_rows[0].mean, exact_rows[0].mean);
    assert_eq!(cold_rows[1].mean, exact_rows[1].mean);
}

#[test]
fn neural_panel_window_construction_has_no_future_target_leakage() {
    let frame = taxi_colon_frame();
    let model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 2,
        n_forecasts: 2,
        future_regressors: BTreeMap::from([("is_airport".to_string(), ComponentMode::Additive)]),
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    let dataset = model.window_dataset(&frame).expect("windows");
    let first = &dataset.windows()[0];

    assert_eq!(first.lags, vec![10.0, 11.0]);
    assert_eq!(first.targets, vec![12.0, 13.0]);
    assert_eq!(first.future_features.len(), 4);
}

#[test]
fn neural_panel_predict_quantiles_are_non_crossing() {
    let frame = taxi_colon_frame();
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        quantiles: vec![0.9, 0.1, 0.5],
        ..NeuralPanelConfig::default()
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
fn neural_panel_learns_quantile_residual_spread() {
    let rows = (0..24)
        .map(|hour| {
            let noise = match hour % 5 {
                0 => -3.0,
                1 => -1.0,
                2 => 0.0,
                3 => 1.0,
                _ => 4.0,
            };
            ForecastRow::from_timestamp_str("PU1->DO2", &timestamp(hour), 30.0 + noise)
                .expect("row")
        })
        .collect();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 4,
        n_forecasts: 1,
        quantiles: vec![0.1, 0.5, 0.9],
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();
    let order = metadata["component_params"]["quantile_output_order"]
        .as_array()
        .expect("order");
    let diffs = metadata["component_params"]["quantile_residual_diffs"]
        .as_array()
        .expect("diffs");
    let median = diffs[0].as_f64().expect("median");
    let lower = diffs[1].as_f64().expect("lower");
    let upper = diffs[2].as_f64().expect("upper");
    let tensor = model.predict_tensor(1).expect("tensor");
    let quantiles = &tensor["PU1->DO2"][0];

    assert_eq!(order[0].as_f64().expect("median order"), 0.5);
    assert!(lower < 0.0);
    assert_eq!(median, 0.0);
    assert!(upper > 0.0);
    assert!(quantiles[0] < quantiles[1]);
    assert!(quantiles[1] < quantiles[2]);
}

#[test]
fn neural_panel_global_mode_has_no_local_deviations() {
    let frame = taxi_colon_frame();
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        trend_mode: NeuralPanelMode::Global,
        ..NeuralPanelConfig::default()
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
fn neural_panel_local_regressor_mode_fits_series_weights() {
    let rows = (0..12)
        .flat_map(|hour| {
            [
                ForecastRow::from_timestamp_str_with_covariates(
                    "PU1:DO2",
                    &timestamp(hour),
                    10.0 + hour as f64 * 0.5,
                    BTreeMap::from([("airport_lane".to_string(), (hour % 2) as f64)]),
                )
                .expect("row"),
                ForecastRow::from_timestamp_str_with_covariates(
                    "PU2:DO1",
                    &timestamp(hour),
                    20.0 - hour as f64 * 0.25,
                    BTreeMap::from([("airport_lane".to_string(), ((hour + 1) % 2) as f64)]),
                )
                .expect("row"),
            ]
        })
        .collect();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        future_regressors: BTreeMap::from([("airport_lane".to_string(), ComponentMode::Additive)]),
        regressor_global_local: NeuralPanelMode::Local,
        local_l2: 0.0,
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert!(
        metadata["component_params"]["local_nonstationary_feature_weights"]["PU1:DO2"]
            ["airport_lane"]
            .as_f64()
            .expect("local regressor weight")
            .is_finite()
    );
}

#[test]
fn neural_panel_ar_tail_changes_direct_forecast() {
    let rows = [
        ("A:B", [10.0, 10.0, 10.0, 18.0, 20.0]),
        ("C:D", [10.0, 10.0, 10.0, 2.0, 0.0]),
    ]
    .into_iter()
    .flat_map(|(series_id, values)| {
        values.into_iter().enumerate().map(move |(idx, value)| {
            ForecastRow::from_timestamp_str(series_id, &timestamp(idx as u32), value).expect("row")
        })
    })
    .collect::<Vec<_>>();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        trend: TrendMode::Off,
        seed: 11,
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let predictions = model.predict(1).expect("predict");
    let values = predictions.predictions();
    let metadata = model.metadata();

    assert_eq!(values[0].series_id, "A:B");
    assert_eq!(values[1].series_id, "C:D");
    assert_ne!(values[0].mean, values[1].mean);
    assert_eq!(
        metadata["component_params"]["target_tails"]["A:B"]
            .as_array()
            .expect("target tail")
            .len(),
        3
    );
}

#[test]
fn neural_panel_daily_fourier_component_changes_direct_forecast() {
    let rows = (0..53)
        .map(|hour| {
            let phase = std::f64::consts::TAU * (hour % 24) as f64 / 24.0;
            ForecastRow::from_timestamp_str(
                "A:B",
                &synthetic_timestamp(hour),
                20.0 + 5.0 * phase.sin(),
            )
            .expect("row")
        })
        .collect::<Vec<_>>();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let base_config = NeuralPanelConfig {
        n_lags: 4,
        n_forecasts: 1,
        trend: TrendMode::Off,
        seed: 17,
        ..NeuralPanelConfig::default()
    };
    let mut without_seasonality =
        NeuralPanelForecaster::new(base_config.clone()).expect("baseline model");
    let mut with_seasonality = NeuralPanelForecaster::new(NeuralPanelConfig {
        daily_fourier_order: 1,
        ..base_config
    })
    .expect("seasonal model");

    without_seasonality.fit(&frame).expect("baseline fit");
    with_seasonality.fit(&frame).expect("seasonal fit");

    let baseline = without_seasonality
        .predict(1)
        .expect("baseline predict")
        .predictions()[0]
        .mean;
    let seasonal = with_seasonality
        .predict(1)
        .expect("seasonal predict")
        .predictions()[0]
        .mean;
    let metadata = with_seasonality.metadata();

    assert_ne!(baseline, seasonal);
    assert!(
        metadata["component_params"]["nonstationary_feature_weights"]
            .as_object()
            .expect("component weights")
            .contains_key("seasonality:daily:sin:1")
    );
}

#[test]
fn neural_panel_custom_seasonality_feature_names_are_period_agnostic() {
    let frame = taxi_colon_frame();
    let model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        custom_seasonalities: BTreeMap::from([("taxi_cycle".to_string(), (12.0, 2))]),
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    let dataset = model.window_dataset(&frame).expect("windows");
    let names = dataset.future_feature_names();

    assert!(names
        .iter()
        .any(|name| name == "seasonality:taxi_cycle:sin:1"));
    assert!(names
        .iter()
        .any(|name| name == "seasonality:taxi_cycle:cos:2"));
    assert!(names.iter().all(|name| !name.contains("taxi_cycle:sin:1:")));
    assert!(names.iter().all(|name| !name.contains("taxi_cycle:cos:1:")));
}

#[test]
fn neural_panel_conditional_custom_seasonality_requires_condition_and_masks_when_false() {
    let rows = (1..=4)
        .map(|hour| {
            ForecastRow::from_timestamp_str_with_covariates(
                "A:B",
                &timestamp(hour),
                30.0 + hour as f64,
                BTreeMap::from([(
                    "rush_hour".to_string(),
                    if hour % 2 == 0 { 1.0 } else { 0.0 },
                )]),
            )
            .expect("row")
        })
        .collect::<Vec<_>>();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 1,
        n_forecasts: 1,
        custom_seasonalities: BTreeMap::from([("taxi_cycle".to_string(), (24.0, 1))]),
        custom_seasonality_conditions: BTreeMap::from([(
            "taxi_cycle".to_string(),
            Some("rush_hour".to_string()),
        )]),
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    let dataset = model.window_dataset(&frame).expect("windows");
    let names = dataset.future_feature_names();
    let seasonality_idx = names
        .iter()
        .position(|name| name == "seasonality:taxi_cycle:sin:1")
        .expect("seasonality feature");
    let first_window = &dataset.windows()[0];
    let second_row_features = &first_window.future_features[1];

    assert_eq!(first_window.future_features[0][seasonality_idx], 0.0);
    assert!(second_row_features[seasonality_idx].abs() > 0.0);
}

#[test]
fn neural_panel_predict_with_known_future_regressors_uses_supplied_covariates() {
    let rows = (0..12)
        .map(|hour| {
            let promo = if hour % 3 == 0 { 1.0 } else { 0.0 };
            ForecastRow::from_timestamp_str_with_covariates(
                "A:B",
                &timestamp(hour),
                10.0 + 4.0 * promo,
                BTreeMap::from([("promo".to_string(), promo)]),
            )
            .expect("row")
        })
        .collect::<Vec<_>>();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        trend: TrendMode::Off,
        future_regressors: BTreeMap::from([("promo".to_string(), ComponentMode::Additive)]),
        seed: 23,
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    assert!(model.predict(1).is_err());

    let last_timestamp = frame
        .rows_for_series("A:B")
        .last()
        .expect("last row")
        .timestamp;
    let next_timestamp = frame
        .frequency()
        .advance(last_timestamp, 1)
        .expect("next timestamp");
    let no_promo = model
        .predict_with_known_future_covariates(
            1,
            &BTreeMap::from([(
                ("A:B".to_string(), next_timestamp),
                BTreeMap::from([("promo".to_string(), 0.0)]),
            )]),
        )
        .expect("no promo predict")
        .predictions()[0]
        .mean;
    let with_promo = model
        .predict_with_known_future_covariates(
            1,
            &BTreeMap::from([(
                ("A:B".to_string(), next_timestamp),
                BTreeMap::from([("promo".to_string(), 1.0)]),
            )]),
        )
        .expect("promo predict")
        .predictions()[0]
        .mean;

    assert_ne!(no_promo, with_promo);
}

#[test]
fn neural_panel_predict_components_include_known_future_breakdown() {
    let rows = (0..12)
        .map(|hour| {
            let promo = if hour % 3 == 0 { 1.0 } else { 0.0 };
            ForecastRow::from_timestamp_str_with_covariates(
                "A:B",
                &timestamp(hour),
                10.0 + 4.0 * promo,
                BTreeMap::from([("promo".to_string(), promo)]),
            )
            .expect("row")
        })
        .collect::<Vec<_>>();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        trend: TrendMode::Off,
        future_regressors: BTreeMap::from([("promo".to_string(), ComponentMode::Additive)]),
        seed: 23,
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let last_timestamp = frame
        .rows_for_series("A:B")
        .last()
        .expect("last row")
        .timestamp;
    let next_timestamp = frame
        .frequency()
        .advance(last_timestamp, 1)
        .expect("next timestamp");
    let components = model
        .predict_components_json_value_with_known_future_covariates(
            1,
            Some(&BTreeMap::from([(
                ("A:B".to_string(), next_timestamp),
                BTreeMap::from([("promo".to_string(), 1.0)]),
            )])),
        )
        .expect("components");
    let tensor = model
        .predict_tensor_with_known_future_covariates(
            1,
            &BTreeMap::from([(
                ("A:B".to_string(), next_timestamp),
                BTreeMap::from([("promo".to_string(), 1.0)]),
            )]),
        )
        .expect("tensor");
    let rows = components["series"]["A:B"].as_array().expect("series rows");
    let row = &rows[0];

    assert_eq!(components["quantile_levels"][0].as_f64(), Some(0.5));
    assert_eq!(row["horizon"].as_u64(), Some(1));
    assert_eq!(row["quantiles"][0].as_f64(), Some(tensor["A:B"][0][0]));
    assert!(row["prediction"].as_f64().expect("prediction").is_finite());
    assert!(row["feature_contributions"]["additive"]["promo"]
        .as_f64()
        .expect("promo contribution")
        .is_finite());
}

#[test]
fn neural_panel_history_components_track_fitted_rows() {
    let rows = (0..12)
        .map(|hour| {
            let promo = if hour % 3 == 0 { 1.0 } else { 0.0 };
            ForecastRow::from_timestamp_str_with_covariates(
                "A:B",
                &timestamp(hour),
                10.0 + 4.0 * promo,
                BTreeMap::from([("promo".to_string(), promo)]),
            )
            .expect("row")
        })
        .collect::<Vec<_>>();
    let frame = ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame");
    let mut model = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 3,
        n_forecasts: 1,
        trend: TrendMode::Off,
        future_regressors: BTreeMap::from([("promo".to_string(), ComponentMode::Additive)]),
        seed: 23,
        ..NeuralPanelConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let history = model.history_components_json_value().expect("history");
    let rows = history["series"]["A:B"].as_array().expect("history rows");

    assert_eq!(rows.len(), 12);
    assert_eq!(rows[0]["index"].as_u64(), Some(0));
    assert!(rows[0]["actual"].as_f64().expect("actual").is_finite());
    assert!(rows[0]["prediction"]
        .as_f64()
        .expect("prediction")
        .is_finite());
}

#[test]
fn neural_panel_compares_on_synthetic_panel_with_baselines() {
    let (train, expected) = synthetic_seasonal_panel();
    let horizon = 4;
    let mut neural = NeuralPanelForecaster::new(NeuralPanelConfig {
        n_lags: 8,
        n_forecasts: horizon,
        quantiles: vec![0.1, 0.5, 0.9],
        daily_fourier_order: 2,
        trend_mode: NeuralPanelMode::Local,
        local_l2: 0.1,
        seed: 7,
        ..NeuralPanelConfig::default()
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
        ("neural_panel", neural_predictions.predictions()),
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

#[allow(dead_code)]
fn assert_forecasts_close(
    actual: &[ForecastPrediction],
    expected: &[ForecastPrediction],
    tol: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (left, right) in actual.iter().zip(expected) {
        assert_eq!(left.series_id, right.series_id);
        assert_eq!(left.timestamp, right.timestamp);
        assert_eq!(left.horizon, right.horizon);
        assert!(
            (left.mean - right.mean).abs() < tol,
            "expected {} to be within {tol} of {}",
            left.mean,
            right.mean
        );
    }
}

fn timestamp(hour: u32) -> String {
    format!("2024-01-01T{hour:02}:00:00")
}

fn synthetic_timestamp(hour: usize) -> String {
    let day = 1 + hour / 24;
    let hour = hour % 24;
    format!("2024-01-{day:02}T{hour:02}:00:00")
}
