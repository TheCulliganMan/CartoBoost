#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
use cartoboost_geo_st::{available_compute_backends, select_compute_backend};
use cartoboost_geo_st::{
    graph_metrics, synthetic_graph_diffusion_frame, traffic_style_fixture_frame, CsrAdjacency,
    DcrnnConfig, DcrnnForecaster, GraphTemporalFrame, GraphWaveNetConfig, GraphWaveNetForecaster,
    STAEformerConfig, STAEformerForecaster,
};

#[test]
fn synthetic_diffusion_beats_last_value_naive() {
    let frame = synthetic_graph_diffusion_frame();
    let train_size = 68;
    let mut train = frame.clone();
    train.timestamps.truncate(train_size);
    train.target.truncate(train_size);
    let mut model = DcrnnForecaster::new(DcrnnConfig {
        epochs: 90,
        learning_rate: 0.02,
        ..DcrnnConfig::default()
    })
    .unwrap();
    model.fit(&train).unwrap();
    let predictions = model.predict(frame.horizon).unwrap();
    let actual = frame.target[train_size..train_size + frame.horizon].to_vec();
    let dcrnn = graph_metrics(&predictions, &actual, &frame.node_ids, &frame.adjacency);
    let naive_predictions = vec![train.target.last().unwrap().clone(); frame.horizon];
    let naive = graph_metrics(
        &naive_predictions,
        &actual,
        &frame.node_ids,
        &frame.adjacency,
    );
    let dcrnn_mae: f64 = dcrnn.by_horizon.iter().map(|metric| metric.mae).sum();
    let naive_mae: f64 = naive.by_horizon.iter().map(|metric| metric.mae).sum();
    assert!(
        dcrnn_mae < naive_mae,
        "DCRNN MAE {dcrnn_mae} should beat naive MAE {naive_mae}"
    );
}

#[test]
fn multi_node_multi_horizon_shape_and_metrics() {
    let frame = traffic_style_fixture_frame();
    let mut model = DcrnnForecaster::new(DcrnnConfig {
        epochs: 40,
        ..DcrnnConfig::default()
    })
    .unwrap();
    model.fit(&frame).unwrap();
    let predictions = model.predict(6).unwrap();
    assert_eq!(predictions.len(), 6);
    assert!(predictions
        .iter()
        .all(|row| row.len() == frame.node_ids.len()));
    let metrics = model.backtest(&frame, 40).unwrap();
    assert_eq!(metrics.by_horizon.len(), frame.horizon);
    assert_eq!(metrics.by_node.len(), frame.node_ids.len());
    assert!(!metrics.graph_distance_residuals.is_empty());
}

#[test]
fn directed_graph_behavior_changes_predictions() {
    let frame = synthetic_graph_diffusion_frame();
    let mut reversed = frame.clone();
    reversed.adjacency = frame.adjacency.transpose(frame.node_ids.len());
    let mut forward = DcrnnForecaster::new(DcrnnConfig {
        epochs: 35,
        ..DcrnnConfig::default()
    })
    .unwrap();
    let mut reverse = DcrnnForecaster::new(DcrnnConfig {
        epochs: 35,
        ..DcrnnConfig::default()
    })
    .unwrap();
    forward.fit(&frame).unwrap();
    reverse.fit(&reversed).unwrap();
    assert_ne!(forward.predict(2).unwrap(), reverse.predict(2).unwrap());
}

#[test]
fn save_load_is_stable() {
    let frame = synthetic_graph_diffusion_frame();
    let mut model = DcrnnForecaster::new(DcrnnConfig {
        epochs: 25,
        ..DcrnnConfig::default()
    })
    .unwrap();
    model.fit(&frame).unwrap();
    let artifact = model.to_json_string().unwrap();
    assert!(artifact.contains("encoder_weights"));
    assert!(artifact.contains("recurrent_weights"));
    assert!(artifact.contains("teacher_forcing_start"));
    let before = model.predict(3).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dcrnn.json");
    model.save(&path).unwrap();
    let loaded = DcrnnForecaster::load(&path).unwrap();
    let after = loaded.predict(3).unwrap();
    for (left_row, right_row) in before.iter().zip(after.iter()) {
        for (&left, &right) in left_row.iter().zip(right_row.iter()) {
            assert!((left - right).abs() < 1.0e-10);
        }
    }
}

#[test]
fn staeformer_fits_predicts_scores_and_roundtrips() {
    let frame = synthetic_graph_diffusion_frame();
    let mut model = STAEformerForecaster::new(STAEformerConfig {
        lookback: 6,
        attention_heads: 3,
        hidden_size: 5,
        epochs: 12,
        learning_rate: 0.03,
        ridge: 1e-3,
        ..STAEformerConfig::default()
    })
    .unwrap();
    model.fit(&frame).unwrap();
    let prediction = model.predict(frame.horizon).unwrap();
    assert_eq!(prediction.len(), frame.horizon);
    assert_eq!(prediction[0].len(), frame.node_ids.len());
    let score = model
        .score(&frame.target[frame.target.len() - frame.horizon..])
        .unwrap();
    assert!(score.is_finite());

    let artifact = model.to_json_string().unwrap();
    assert!(artifact.contains("temporal_queries"));
    let loaded = STAEformerForecaster::from_json_string(&artifact).unwrap();
    assert_eq!(
        loaded.predict(frame.horizon).unwrap().len(),
        prediction.len()
    );
}

#[test]
fn graph_wavenet_fits_predicts_scores_and_roundtrips() {
    let frame = synthetic_graph_diffusion_frame();
    let mut model = GraphWaveNetForecaster::new(GraphWaveNetConfig {
        lookback: 6,
        dilation_depth: 3,
        hidden_size: 5,
        epochs: 12,
        learning_rate: 0.03,
        ridge: 1e-3,
        ..GraphWaveNetConfig::default()
    })
    .unwrap();
    model.fit(&frame).unwrap();
    let prediction = model.predict(frame.horizon).unwrap();
    assert_eq!(prediction.len(), frame.horizon);
    assert_eq!(prediction[0].len(), frame.node_ids.len());
    let score = model
        .score(&frame.target[frame.target.len() - frame.horizon..])
        .unwrap();
    assert!(score.is_finite());

    let artifact = model.to_json_string().unwrap();
    assert!(artifact.contains("dilation_depth"));
    let loaded = GraphWaveNetForecaster::from_json_string(&artifact).unwrap();
    assert_eq!(
        loaded.predict(frame.horizon).unwrap().len(),
        prediction.len()
    );
}

#[test]
fn training_predictions_do_not_depend_on_post_cutoff_targets() {
    let frame = synthetic_graph_diffusion_frame();
    let train_size = 68;
    let mut train = frame.clone();
    train.timestamps.truncate(train_size);
    train.target.truncate(train_size);
    let mut changed_future = frame.clone();
    for row in changed_future.target.iter_mut().skip(train_size) {
        for value in row {
            *value += 10_000.0;
        }
    }
    changed_future.timestamps.truncate(train_size);
    changed_future.target.truncate(train_size);

    let config = DcrnnConfig {
        epochs: 40,
        ..DcrnnConfig::default()
    };
    let mut baseline = DcrnnForecaster::new(config.clone()).unwrap();
    let mut future_mutated = DcrnnForecaster::new(config).unwrap();
    baseline.fit(&train).unwrap();
    future_mutated.fit(&changed_future).unwrap();
    assert_eq!(
        baseline.predict(3).unwrap(),
        future_mutated.predict(3).unwrap()
    );
}

#[test]
fn frame_rejects_invalid_shapes_before_training() {
    let adjacency = CsrAdjacency::new(vec![0, 1], vec![0], vec![1.0], 1).unwrap();
    let err = GraphTemporalFrame::new(
        vec!["a".into()],
        vec![0],
        vec![vec![1.0]],
        None,
        adjacency,
        1,
        "hourly".into(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("target length must exceed"));
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
fn metal_backend_prediction_matches_cpu_backend() {
    if !available_compute_backends()
        .iter()
        .any(|backend| backend == "metal")
    {
        return;
    }
    let frame = traffic_style_fixture_frame();
    let mut cpu = DcrnnForecaster::new(DcrnnConfig {
        epochs: 20,
        backend: select_compute_backend(Some("cpu")).unwrap(),
        ..DcrnnConfig::default()
    })
    .unwrap();
    let mut metal = DcrnnForecaster::new(DcrnnConfig {
        epochs: 20,
        backend: select_compute_backend(Some("metal")).unwrap(),
        ..DcrnnConfig::default()
    })
    .unwrap();
    cpu.fit(&frame).unwrap();
    metal.fit(&frame).unwrap();
    let cpu_predictions = cpu.predict(3).unwrap();
    let metal_predictions = metal.predict(3).unwrap();
    for (cpu_row, metal_row) in cpu_predictions.iter().zip(&metal_predictions) {
        for (&left, &right) in cpu_row.iter().zip(metal_row) {
            assert!((left - right).abs() < 1.0e-4);
        }
    }
}
