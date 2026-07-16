use crate::data::{Dataset, FeatureKind};
use crate::serialize::WEIGHTS_ARTIFACT_TYPE;
use crate::tree::{Model, SplitterKind};
use crate::{Booster, BoosterConfig};

#[test]
fn deserializes_v1_json_without_optional_metadata_fields() {
    let json = r#"{
        "artifact_version": 1,
        "init_prediction": 0.0,
        "learning_rate": 0.1,
        "feature_count": 2,
        "target_name": null,
        "trees": []
    }"#;

    let model: Model = serde_json::from_str(json).unwrap();

    assert!(model.metadata.is_none());
    assert!(model.feature_schema.is_none());
    assert!(model.training_config.is_none());

    let schema = model.feature_schema_or_default();
    assert_eq!(
        schema.names,
        vec!["feature_0".to_string(), "feature_1".to_string()]
    );
    assert_eq!(schema.kinds, vec![FeatureKind::Numeric; 2]);
}

#[test]
fn weights_artifact_round_trips_predictions() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let model = Booster::new(BoosterConfig {
        n_estimators: 2,
        learning_rate: 0.5,
        max_depth: 1,
        min_samples_leaf: 1,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        ..BoosterConfig::default()
    })
    .fit(&x, &y, None)
    .unwrap();
    let expected = model.predict(&x);
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("model.weights.json");

    model.save_weights(&path).unwrap();
    let payload: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let loaded = Model::load_weights(&path).unwrap();

    assert_eq!(payload["artifact_type"], WEIGHTS_ARTIFACT_TYPE);
    assert_eq!(payload["weights_artifact_version"], 1);
    assert_eq!(payload["model_artifact_version"], 1);
    assert_eq!(payload["backend"], "cpu");
    assert_eq!(
        payload["model"]["training_config"]["backend"]["requested"],
        "cpu"
    );
    assert_eq!(
        payload["model"]["training_config"]["backend"]["selected"],
        "cpu"
    );
    assert_eq!(loaded.predict(&x), expected);
}

#[test]
fn load_weights_accepts_plain_model_json_for_compatibility() {
    let json = r#"{
        "artifact_version": 1,
        "init_prediction": 2.5,
        "learning_rate": 0.1,
        "feature_count": 1,
        "target_name": null,
        "trees": []
    }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("plain-model.json");
    std::fs::write(&path, json).unwrap();

    let model = Model::load_weights(&path).unwrap();

    assert_eq!(model.init_prediction, 2.5);
}

#[test]
fn load_weights_rejects_future_weights_version_clearly() {
    let json = r#"{
        "artifact_type": "cartoboost.weights",
        "weights_artifact_version": 999,
        "model_artifact_version": 1,
        "backend": "rust",
        "model": {
            "artifact_version": 1,
            "init_prediction": 0.0,
            "learning_rate": 0.1,
            "feature_count": 1,
            "target_name": null,
            "trees": []
        }
    }"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("future-weights.json");
    std::fs::write(&path, json).unwrap();

    let error = Model::load_weights(&path).unwrap_err();

    assert!(error
        .to_string()
        .contains("unsupported weights artifact version 999"));
}
