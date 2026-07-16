use cartoboost_core::tree::{LeafPredictorKind, SplitterKind};
use cartoboost_core::{Booster, BoosterConfig, Dataset, Model};

fn fixture(rows: usize, cols: usize) -> (Dataset, Vec<f64>) {
    let values = (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (row as f64 * 0.23 - col as f64 * 0.41).sin()
        })
        .collect::<Vec<_>>();
    let target = (0..rows)
        .map(|row| {
            let first = values[row * cols];
            let second = values[row * cols + 1.min(cols - 1)];
            1.5 * first - 0.5 * second
        })
        .collect();

    (
        Dataset::from_flat(rows, cols, values).expect("test fixture dataset is valid"),
        target,
    )
}

fn configs() -> Vec<BoosterConfig> {
    vec![
        BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        },
        BoosterConfig {
            n_estimators: 3,
            learning_rate: 0.25,
            max_depth: 2,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        },
        BoosterConfig {
            n_estimators: 1,
            learning_rate: 0.5,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Linear,
            linear_leaf_features: vec![0, 1],
            linear_lambda_l2: 0.1,
            ..BoosterConfig::default()
        },
    ]
}

#[test]
fn fitted_models_predict_only_finite_values_for_small_deterministic_fixtures() {
    for (rows, cols) in [(4, 2), (8, 3), (12, 4)] {
        let (x, y) = fixture(rows, cols);
        for config in configs() {
            let model = Booster::new(config)
                .fit(&x, &y, None)
                .expect("small deterministic fixture trains");

            for prediction in model.predict(&x) {
                assert!(prediction.is_finite(), "prediction was {prediction}");
            }
        }
    }
}

#[test]
fn save_load_preserves_predictions_for_small_deterministic_fixtures() {
    let temp_dir = tempfile::tempdir().expect("temporary directory is available");

    for (case_index, config) in configs().into_iter().enumerate() {
        let (x, y) = fixture(8, 3);
        let model = Booster::new(config)
            .fit(&x, &y, None)
            .expect("small deterministic fixture trains");
        let before = model.predict(&x);
        let path = temp_dir.path().join(format!("model-{case_index}.json"));

        model.save(&path).expect("model saves");
        let restored = Model::load(&path).expect("model loads");

        assert_eq!(restored.predict(&x), before, "configuration {case_index}");
    }
}
