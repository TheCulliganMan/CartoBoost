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

#[test]
fn saving_json_reports_buffered_write_and_flush_failures() {
    use std::io::{self, Write};

    struct FailingWriter {
        fail_write: bool,
    }
    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_write {
                Err(io::Error::other("write failed"))
            } else {
                Ok(bytes.len())
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    for (fail_write, message) in [(true, "write failed"), (false, "flush failed")] {
        let error = super::write_json(&vec![1, 2, 3], FailingWriter { fail_write }).unwrap_err();
        assert!(error.to_string().contains(message));
    }
}

#[test]
fn saved_weights_match_the_owned_public_artifact_byte_for_byte() {
    let model: Model = serde_json::from_str(
        r#"{
        "artifact_version": 1, "init_prediction": 2.5,
        "learning_rate": 0.1, "feature_count": 1, "trees": []
    }"#,
    )
    .unwrap();
    let expected = super::WeightsArtifact {
        artifact_type: super::WEIGHTS_ARTIFACT_TYPE.to_string(),
        weights_artifact_version: super::WEIGHTS_ARTIFACT_VERSION,
        model_artifact_version: model.artifact_version,
        backend: "rust".to_string(),
        model: model.clone(),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.json");
    model.save_weights(&path).unwrap();
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        serde_json::to_string_pretty(&expected).unwrap()
    );
}

#[test]
fn model_loaders_reject_invalid_split_indices_and_linear_shapes() {
    use crate::predictors::LinearLeafModel;
    use crate::tree::{Node, Split, Tree};
    let leaf = || Node::Leaf {
        value: 1.0,
        sample_weight_sum: 1.0,
        training_loss: 0.0,
    };
    let invalid_nodes = [
        Node::Branch {
            split: Split::Axis {
                feature: 1,
                threshold: 0.5,
                missing_goes_left: false,
            },
            left: Box::new(leaf()),
            right: Box::new(leaf()),
            gain: 1.0,
            sample_weight_sum: 2.0,
        },
        Node::LinearLeaf {
            model: LinearLeafModel {
                intercept: 1.0,
                coefficients: vec![2.0, 3.0],
                features: vec![0],
            },
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        },
        Node::Branch {
            split: Split::PeriodicInterval {
                feature: 0,
                period: 0.0,
                start: 0.0,
                end: 1.0,
                missing_goes_left: false,
            },
            left: Box::new(leaf()),
            right: Box::new(leaf()),
            gain: 1.0,
            sample_weight_sum: 2.0,
        },
    ];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.json");
    for root in invalid_nodes {
        let mut model: Model = serde_json::from_str(r#"{"artifact_version":1,"init_prediction":0.0,"learning_rate":0.1,"feature_count":1,"trees":[]}"#).unwrap();
        model.trees.push(Tree { root });
        model.save(&path).unwrap();
        assert!(Model::load(&path).is_err());
        assert!(Model::load_weights(&path).is_err());
        model.save_weights(&path).unwrap();
        assert!(Model::load_weights(&path).is_err());
        assert!(super::load_weights_json(&path).is_err());
    }
}

#[test]
fn model_loader_rejects_schema_layout_and_weights_version_mismatches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.json");
    let mut model: Model = serde_json::from_str(r#"{"artifact_version":1,"init_prediction":0.0,"learning_rate":0.1,"feature_count":2,"feature_schema":{"names":["only"],"kinds":["Numeric"]},"trees":[]}"#).unwrap();
    model.save(&path).unwrap();
    assert!(Model::load(&path).is_err());
    model.feature_schema = None;
    model.save_weights(&path).unwrap();
    let mut payload: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    payload["model_artifact_version"] = serde_json::json!(2);
    std::fs::write(&path, serde_json::to_string(&payload).unwrap()).unwrap();
    assert!(Model::load_weights(&path)
        .unwrap_err()
        .to_string()
        .contains("version"));
}
