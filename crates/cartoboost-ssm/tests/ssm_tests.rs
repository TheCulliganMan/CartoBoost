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
    let y: Vec<Vec<f64>> = (0..80)
        .map(|idx| vec![idx as f64, (idx as f64).sin()])
        .collect();
    let artifact = fit_temporal_ssm(&y, 32, 4, 6, 13).unwrap();
    let before = predict_temporal_ssm(&artifact, 4).unwrap();
    let encoded = serde_json::to_string(&artifact).unwrap();
    let decoded: TemporalSsmArtifact = serde_json::from_str(&encoded).unwrap();
    let after = predict_temporal_ssm(&decoded, 4).unwrap();

    assert_eq!(artifact.model_class, "TemporalSSMForecaster");
    assert_eq!(artifact.architecture, "selective_ssm");
    assert_eq!(artifact.backend, "cpu");
    assert!(artifact.save_load_parity_checked);
    assert_eq!(before, after);
}
