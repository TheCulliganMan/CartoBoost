#![no_main]

use cartoboost_core::tree::{Node, PredictionTransform, Tree, MODEL_ARTIFACT_VERSION};
use cartoboost_core::Model;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut row = Vec::new();
    for chunk in data.chunks_exact(8).take(8) {
        let value = f64::from_le_bytes(chunk.try_into().expect("chunk length is 8"));
        if value.is_finite() {
            row.push(value.tanh());
        }
    }
    if row.is_empty() {
        return;
    }
    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.0,
        learning_rate: 0.1,
        feature_count: row.len(),
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: PredictionTransform::Identity,
        trees: vec![Tree {
            root: Node::Leaf {
                value: 1.0,
                sample_weight_sum: 1.0,
                training_loss: 0.0,
            },
        }],
    };
    let _ = model.predict_one(&row);
});
