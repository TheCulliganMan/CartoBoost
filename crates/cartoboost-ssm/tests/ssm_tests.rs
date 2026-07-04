use cartoboost_ssm::{
    fit_temporal_ssm, predict_temporal_ssm, SelectiveStateSpaceBlock, TemporalSsmArtifact,
};

#[test]
fn selective_state_space_block_is_deterministic_and_shape_stable() {
    let block = SelectiveStateSpaceBlock::new(2, 4, 7).unwrap();
    let sequence = vec![vec![0.0, 1.0], vec![1.0, 0.5], vec![2.0, 0.0]];

    let first = block.encode(&sequence).unwrap();
    let second = block.encode(&sequence).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    assert_eq!(first[0].len(), 4);
}

#[test]
fn temporal_ssm_metadata_is_honest_and_roundtrips() {
    let mut y = vec![vec![0.0, 0.0]; 160];
    for idx in 12..y.len() {
        y[idx][0] = 0.75 * y[idx - 9][0] - 0.35 * y[idx - 5][0] + (idx as f64 / 13.0).sin();
        y[idx][1] = 0.65 * y[idx - 7][1] + 0.25 * y[idx - 11][0] + (idx as f64 / 17.0).cos();
    }
    let artifact = fit_temporal_ssm(&y, 48, 4, 8, 13).unwrap();
    let before = predict_temporal_ssm(&artifact, 4).unwrap();
    let encoded = serde_json::to_string(&artifact).unwrap();
    let decoded: TemporalSsmArtifact = serde_json::from_str(&encoded).unwrap();
    let after = predict_temporal_ssm(&decoded, 4).unwrap();

    assert_eq!(artifact.model_class, "TemporalSSMForecaster");
    assert_eq!(artifact.architecture, "selective_ssm_lite");
    assert_eq!(artifact.backend, "cpu");
    assert!(artifact.decoder_metrics.beats_trend_extrapolation);
    assert!(artifact.decoder_metrics.beats_temporal_conv_baseline);
    assert!(artifact.save_load_parity_checked);
    for (before_row, after_row) in before.iter().zip(after.iter()) {
        for (before_value, after_value) in before_row.iter().zip(after_row.iter()) {
            assert!((before_value - after_value).abs() < 1.0e-12);
        }
    }
}
