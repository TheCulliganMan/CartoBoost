use cartoboost_core::tree::{Node, PredictionTransform, Tree, MODEL_ARTIFACT_VERSION};
use cartoboost_core::Model;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn model_fixture(trees: usize) -> Model {
    let trees = (0..trees)
        .map(|index| Tree {
            root: Node::Leaf {
                value: (index as f64 * 0.11).sin(),
                sample_weight_sum: 4.0,
                training_loss: 0.0,
            },
        })
        .collect();

    Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.5,
        learning_rate: 0.1,
        feature_count: 3,
        feature_schema: None,
        target_name: Some("target".to_string()),
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: PredictionTransform::Identity,
        trees,
    }
}

fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize");
    for &trees in &[1, 16] {
        let model = model_fixture(trees);
        let json = serde_json::to_string(&model).expect("benchmark model serializes");

        group.bench_with_input(
            BenchmarkId::new("model_to_json", format!("{trees}trees")),
            &model,
            |bench, model| {
                bench.iter(|| {
                    serde_json::to_string(black_box(model)).expect("benchmark model serializes")
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("model_from_json", format!("{trees}trees")),
            &json,
            |bench, json| {
                bench.iter(|| {
                    serde_json::from_str::<Model>(black_box(json))
                        .expect("benchmark model deserializes")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(serialize_benches, bench_serialize);
criterion_main!(serialize_benches);
