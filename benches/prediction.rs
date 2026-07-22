use cartoboost_core::tree::{Node, PredictionTransform, Split, Tree, MODEL_ARTIFACT_VERSION};
use cartoboost_core::{Dataset, Model};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn synthetic_values(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (row as f64 * 0.031 + col as f64 * 0.17).sin()
        })
        .collect()
}

fn synthetic_dataset(rows: usize, cols: usize) -> Dataset {
    let values = synthetic_values(rows, cols);
    Dataset::from_flat(rows, cols, values).expect("synthetic prediction data is valid")
}

fn synthetic_dataset_with_values(rows: usize, cols: usize) -> (Dataset, Vec<f64>) {
    let values = synthetic_values(rows, cols);
    (
        Dataset::from_flat(rows, cols, values.clone()).expect("synthetic prediction data is valid"),
        values,
    )
}

fn synthetic_model(trees: usize) -> Model {
    let trees = (0..trees)
        .map(|index| Tree {
            root: Node::Leaf {
                value: (index as f64 * 0.013).cos() * 0.01,
                sample_weight_sum: 1.0,
                training_loss: 0.0,
            },
        })
        .collect();

    Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.25,
        learning_rate: 0.1,
        feature_count: 8,
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: PredictionTransform::Identity,
        trees,
    }
}

fn synthetic_axis_model(trees: usize, feature_count: usize) -> Model {
    let trees = (0..trees)
        .map(|index| {
            let feature = index % feature_count;
            let left_value = (index as f64 * 0.017).sin() * 0.02;
            let right_value = (index as f64 * 0.019).cos() * 0.02;
            Tree {
                root: Node::Branch {
                    split: Split::Axis {
                        feature,
                        threshold: 0.0,
                        missing_goes_left: true,
                    },
                    left: Box::new(Node::Leaf {
                        value: left_value,
                        sample_weight_sum: 1.0,
                        training_loss: 0.0,
                    }),
                    right: Box::new(Node::Leaf {
                        value: right_value,
                        sample_weight_sum: 1.0,
                        training_loss: 0.0,
                    }),
                    gain: 1.0,
                    sample_weight_sum: 2.0,
                },
            }
        })
        .collect();

    Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.25,
        learning_rate: 0.1,
        feature_count,
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: PredictionTransform::Identity,
        trees,
    }
}

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    for &(rows, cols, trees) in &[(128, 4, 4), (1_024, 8, 16)] {
        let dataset = synthetic_dataset(rows, cols);
        let model = synthetic_model(trees);
        group.bench_with_input(
            BenchmarkId::new("model_predict", format!("{rows}x{cols}_{trees}trees")),
            &(model, dataset),
            |bench, (model, dataset)| {
                bench.iter(|| black_box(model).predict(black_box(dataset)));
            },
        );
    }
    for &(rows, cols, trees) in &[(25_000, 8, 100), (25_000, 3, 100)] {
        let (dataset, values) = synthetic_dataset_with_values(rows, cols);
        let model = synthetic_axis_model(trees, cols);
        group.bench_with_input(
            BenchmarkId::new(
                "model_predict_flat_axis_cached",
                format!("{rows}x{cols}_{trees}trees"),
            ),
            &(model.clone(), values.clone()),
            |bench, (model, values)| {
                let predictor = model
                    .flat_axis_predictor()
                    .expect("benchmark model has only axis splits");
                bench.iter(|| {
                    black_box(model)
                        .validate_dense_flat_prediction_inputs(
                            black_box(rows),
                            black_box(cols),
                            black_box(values),
                        )
                        .expect("benchmark fixture validates");
                    predictor.predict_flat(black_box(rows), black_box(cols), black_box(values))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new(
                "model_predict_flat_axis_cached_repeated_10",
                format!("{rows}x{cols}_{trees}trees"),
            ),
            &(model, dataset, values),
            |bench, (model, _dataset, values)| {
                let predictor = model
                    .flat_axis_predictor()
                    .expect("benchmark model has only axis splits");
                bench.iter(|| {
                    for _ in 0..10 {
                        black_box(model)
                            .validate_dense_flat_prediction_inputs(
                                black_box(rows),
                                black_box(cols),
                                black_box(values),
                            )
                            .expect("benchmark fixture validates");
                        black_box(predictor.predict_flat(
                            black_box(rows),
                            black_box(cols),
                            black_box(values),
                        ));
                    }
                });
            },
        );
    }
    for &(rows, cols, trees) in &[(1, 8, 100), (1_024, 8, 100)] {
        let values = synthetic_values(rows, cols);
        let model = synthetic_axis_model(trees, cols);
        group.bench_with_input(
            BenchmarkId::new(
                "model_predict_feature_contributions",
                format!("{rows}x{cols}_{trees}trees"),
            ),
            &(model, values),
            |bench, (model, values)| {
                bench.iter(|| {
                    black_box(model)
                        .try_predict_feature_contributions_flat(
                            black_box(rows),
                            black_box(cols),
                            black_box(values),
                        )
                        .expect("axis-tree contribution benchmark must be supported")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(prediction_benches, bench_prediction);
criterion_main!(prediction_benches);
