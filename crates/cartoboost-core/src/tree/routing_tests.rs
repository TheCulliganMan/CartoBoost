use super::{
    fuzzy_weights, periodic_contains, periodic_signed_distance, FuzzyKernel, Model, Node, Split,
    Tree, MODEL_ARTIFACT_VERSION,
};
use crate::data::{Dataset, SparseSetColumn};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn diagonal_split_routes_by_signed_plane_distance() {
    let split = Split::Diagonal2D {
        x_feature: 0,
        y_feature: 1,
        normal_x: 1.0,
        normal_y: 1.0,
        threshold: 0.0,
        missing_goes_left: true,
    };

    assert!(split.goes_left(&[-1.0, 0.25]));
    assert!(!split.goes_left(&[1.0, 0.25]));
}

#[test]
fn gaussian_split_routes_inside_radius_left() {
    let split = Split::Gaussian2D {
        x_feature: 0,
        y_feature: 1,
        center_x: 0.0,
        center_y: 0.0,
        radius: 2.0,
        missing_goes_left: true,
    };

    assert!(split.goes_left(&[1.0, 1.0]));
    assert!(!split.goes_left(&[3.0, 0.0]));
}

#[test]
fn periodic_wraparound_interval_contains_expected_values() {
    assert!(periodic_contains(23.0, 24.0, 22.0, 2.0));
    assert!(periodic_contains(1.0, 24.0, 22.0, 2.0));
    assert!(!periodic_contains(12.0, 24.0, 22.0, 2.0));
    assert!(periodic_contains(25.0, 24.0, 0.0, 2.0));
}

#[test]
fn periodic_signed_distance_is_negative_inside_and_positive_outside() {
    assert_close(periodic_signed_distance(23.0, 24.0, 22.0, 2.0), -1.0);
    assert_close(periodic_signed_distance(1.0, 24.0, 22.0, 2.0), -1.0);
    assert_close(periodic_signed_distance(22.0, 24.0, 22.0, 2.0), 0.0);
    assert_close(periodic_signed_distance(3.0, 24.0, 22.0, 2.0), 1.0);
    assert_close(periodic_signed_distance(21.0, 24.0, 22.0, 2.0), 1.0);
}

#[test]
fn fuzzy_weights_conserve_mass_and_interpolate_at_boundary() {
    let weights = fuzzy_weights(0.0, 2.0, FuzzyKernel::Linear);
    assert_close(weights.left, 0.5);
    assert_close(weights.right, 0.5);
    assert_close(weights.left + weights.right, 1.0);

    let hard = fuzzy_weights(3.0, 2.0, FuzzyKernel::Linear);
    assert_close(hard.left, 0.0);
    assert_close(hard.right, 1.0);
}

#[test]
fn fuzzy_kernels_change_boundary_blend_shape() {
    let linear = fuzzy_weights(0.5, 2.0, FuzzyKernel::Linear);
    let gaussian = fuzzy_weights(0.5, 2.0, FuzzyKernel::Gaussian);
    let tricube = fuzzy_weights(0.5, 2.0, FuzzyKernel::Tricube);

    assert!(gaussian.left > linear.left);
    assert!(tricube.left > linear.left);
    assert_close(gaussian.left + gaussian.right, 1.0);
    assert_close(tricube.left + tricube.right, 1.0);
}

#[test]
fn fuzzy_kernel_alias_deserializes_to_linear() {
    let kernel: FuzzyKernel = serde_json::from_str("\"triangular\"").unwrap();

    assert_eq!(kernel, FuzzyKernel::Linear);
}

#[test]
fn fuzzy_branch_prediction_uses_weighted_sum() {
    let node = Node::Branch {
        split: Split::Fuzzy {
            base: Box::new(Split::Axis {
                feature: 0,
                threshold: 10.0,
                missing_goes_left: true,
            }),
            bandwidth: 2.0,
            kernel: FuzzyKernel::Linear,
        },
        left: Box::new(Node::Leaf {
            value: 0.0,
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        }),
        right: Box::new(Node::Leaf {
            value: 10.0,
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        }),
        gain: 0.0,
        sample_weight_sum: 2.0,
    };

    assert_close(node.predict_one(&[10.0]), 5.0);
}

#[test]
fn fuzzy_periodic_branch_prediction_blends_near_wraparound_boundary() {
    let node = Node::Branch {
        split: Split::Fuzzy {
            base: Box::new(Split::PeriodicInterval {
                feature: 0,
                period: 24.0,
                start: 22.0,
                end: 2.0,
                missing_goes_left: false,
            }),
            bandwidth: 2.0,
            kernel: FuzzyKernel::Linear,
        },
        left: Box::new(Node::Leaf {
            value: 20.0,
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        }),
        right: Box::new(Node::Leaf {
            value: 0.0,
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        }),
        gain: 0.0,
        sample_weight_sum: 2.0,
    };

    assert_close(node.predict_one(&[23.0]), 15.0);
    assert_close(node.predict_one(&[21.0]), 5.0);
    assert_close(node.predict_one(&[12.0]), 0.0);
}

#[test]
fn sparse_set_split_routes_integer_id_membership() {
    let split = Split::SparseSetContainsAny {
        feature: 0,
        ids: vec![7, 11, 13],
        missing_goes_left: false,
    };

    assert!(split.goes_left(&[11.0]));
    assert!(!split.goes_left(&[12.0]));
    assert!(!split.goes_left(&[11.5]));
}

#[test]
fn sparse_list_contains_any_routes_dataset_rows() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0]])
        .unwrap()
        .with_sparse_sets(vec![SparseSetColumn::new(vec![
            vec![3, 7],
            vec![11],
            vec![],
        ])])
        .unwrap();
    let split = Split::SparseListContainsAny {
        sparse_feature: 0,
        ids: vec![7, 13],
        missing_goes_left: false,
    };

    assert_close(split.branch_weights_dataset_row(&x, 0).left, 1.0);
    assert_close(split.branch_weights_dataset_row(&x, 1).right, 1.0);
    assert_close(split.branch_weights_dataset_row(&x, 2).right, 1.0);
}

#[test]
fn sparse_list_duplicate_ids_do_not_change_routing() {
    let x = Dataset::from_rows(vec![vec![0.0]])
        .unwrap()
        .with_sparse_sets(vec![SparseSetColumn::new(vec![vec![7, 7, 3]])])
        .unwrap();
    let split = Split::SparseListContainsAny {
        sparse_feature: 0,
        ids: vec![7],
        missing_goes_left: false,
    };

    assert_eq!(x.sparse_set_row(0, 0), Some(&[3, 7][..]));
    assert_close(split.branch_weights_dataset_row(&x, 0).left, 1.0);
}

#[test]
fn sparse_list_dense_predict_one_path_uses_missing_policy_only() {
    let node = Node::Branch {
        split: Split::SparseListContainsAny {
            sparse_feature: 0,
            ids: vec![7],
            missing_goes_left: false,
        },
        left: Box::new(Node::Leaf {
            value: 10.0,
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        }),
        right: Box::new(Node::Leaf {
            value: -1.0,
            sample_weight_sum: 1.0,
            training_loss: 0.0,
        }),
        gain: 0.0,
        sample_weight_sum: 2.0,
    };

    assert_close(node.predict_one(&[7.0]), -1.0);
}

#[test]
fn dense_predict_errors_for_sparse_list_model() {
    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.0,
        learning_rate: 1.0,
        feature_count: 1,
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: super::PredictionTransform::Identity,
        trees: vec![Tree {
            root: Node::Branch {
                split: Split::SparseListContainsAny {
                    sparse_feature: 0,
                    ids: vec![7],
                    missing_goes_left: false,
                },
                left: Box::new(Node::Leaf {
                    value: 10.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                right: Box::new(Node::Leaf {
                    value: -1.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                gain: 0.0,
                sample_weight_sum: 2.0,
            },
        }],
    };
    let dense = Dataset::from_rows(vec![vec![7.0]]).unwrap();

    assert!(model.try_predict_one_dense(&[7.0]).is_err());
    assert!(model.try_predict(&dense).is_err());
}

#[test]
fn sparse_list_save_load_prediction_identity() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![0.0]])
        .unwrap()
        .with_sparse_sets(vec![SparseSetColumn::new(vec![vec![7], vec![]])])
        .unwrap();
    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 1.0,
        learning_rate: 1.0,
        feature_count: 1,
        feature_schema: Some(x.feature_schema_or_default()),
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: super::PredictionTransform::Identity,
        trees: vec![Tree {
            root: Node::Branch {
                split: Split::SparseListContainsAny {
                    sparse_feature: 0,
                    ids: vec![7],
                    missing_goes_left: false,
                },
                left: Box::new(Node::Leaf {
                    value: 4.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                right: Box::new(Node::Leaf {
                    value: -2.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                gain: 0.0,
                sample_weight_sum: 2.0,
            },
        }],
    };
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("sparse-list-model.json");
    let predictions = model.predict(&x);

    model.save(&path).unwrap();
    let loaded = Model::load(&path).unwrap();

    assert_eq!(loaded.predict(&x), predictions);
}

#[test]
fn flat_axis_predictor_matches_dataset_prediction_and_validates_inputs() {
    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 1.0,
        learning_rate: 0.5,
        feature_count: 1,
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: super::PredictionTransform::Identity,
        trees: vec![
            Tree {
                root: Node::Branch {
                    split: Split::Axis {
                        feature: 0,
                        threshold: 1.5,
                        missing_goes_left: true,
                    },
                    left: Box::new(Node::Leaf {
                        value: -2.0,
                        sample_weight_sum: 1.0,
                        training_loss: 0.0,
                    }),
                    right: Box::new(Node::Leaf {
                        value: 4.0,
                        sample_weight_sum: 1.0,
                        training_loss: 0.0,
                    }),
                    gain: 0.0,
                    sample_weight_sum: 2.0,
                },
            },
            Tree {
                root: Node::Branch {
                    split: Split::Axis {
                        feature: 0,
                        threshold: 2.5,
                        missing_goes_left: false,
                    },
                    left: Box::new(Node::Leaf {
                        value: 1.0,
                        sample_weight_sum: 1.0,
                        training_loss: 0.0,
                    }),
                    right: Box::new(Node::Leaf {
                        value: 3.0,
                        sample_weight_sum: 1.0,
                        training_loss: 0.0,
                    }),
                    gain: 0.0,
                    sample_weight_sum: 2.0,
                },
            },
        ],
    };
    let x = Dataset::from_rows(vec![vec![0.0], vec![2.0], vec![3.0]]).unwrap();
    let values = vec![0.0, 2.0, 3.0];

    let dataset_predictions = model.try_predict(&x).unwrap();
    let flat_predictions = model.try_predict_flat(3, 1, &values, &[], &[]).unwrap();
    let flat_axis_predictions = model
        .flat_axis_predictor()
        .expect("flat axis predictor")
        .predict_flat(3, 1, &values);

    assert_eq!(flat_predictions, dataset_predictions);
    assert_eq!(flat_axis_predictions, dataset_predictions);
    assert!(model.try_predict_flat(2, 2, &values, &[], &[]).is_err());
    assert!(model.try_predict_flat(3, 2, &values, &[], &[]).is_err());
    assert!(model
        .try_predict_flat(3, 1, &[0.0, f64::NAN, 3.0], &[], &[])
        .is_err());
    assert!(model
        .validate_dense_flat_prediction_inputs(3, 1, &values)
        .is_ok());
}

#[test]
fn flat_additive_predictions_sum_to_flat_predictions() {
    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 2.0,
        learning_rate: 0.25,
        feature_count: 1,
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: super::PredictionTransform::Identity,
        trees: vec![Tree {
            root: Node::Branch {
                split: Split::Axis {
                    feature: 0,
                    threshold: 1.0,
                    missing_goes_left: true,
                },
                left: Box::new(Node::Leaf {
                    value: 4.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                right: Box::new(Node::Leaf {
                    value: -4.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                gain: 0.0,
                sample_weight_sum: 2.0,
            },
        }],
    };
    let values = vec![0.0, 2.0];
    let predictions = model.try_predict_flat(2, 1, &values, &[], &[]).unwrap();
    let additive = model
        .try_predict_additive_flat(2, 1, &values, &[], &[])
        .unwrap();

    assert_eq!(additive.len(), 2);
    assert_eq!(additive[0].iter().sum::<f64>(), predictions[0]);
    assert_eq!(additive[1].iter().sum::<f64>(), predictions[1]);
}

#[test]
fn flat_sparse_list_prediction_validates_offsets_and_matches_dataset_prediction() {
    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.0,
        learning_rate: 1.0,
        feature_count: 1,
        feature_schema: None,
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: super::PredictionTransform::Identity,
        trees: vec![Tree {
            root: Node::Branch {
                split: Split::SparseListContainsAny {
                    sparse_feature: 0,
                    ids: vec![7, 11],
                    missing_goes_left: false,
                },
                left: Box::new(Node::Leaf {
                    value: 5.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                right: Box::new(Node::Leaf {
                    value: -3.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                gain: 0.0,
                sample_weight_sum: 2.0,
            },
        }],
    };
    let x = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0]])
        .unwrap()
        .with_sparse_sets(vec![SparseSetColumn::new(vec![
            vec![7],
            vec![3, 4],
            vec![11, 12],
        ])])
        .unwrap();
    let values = vec![0.0, 0.0, 0.0];
    let sparse_offsets = vec![vec![0, 1, 3, 5]];
    let sparse_ids = vec![vec![7, 3, 4, 11, 12]];

    assert_eq!(
        model
            .try_predict_flat(3, 1, &values, &sparse_offsets, &sparse_ids)
            .unwrap(),
        model.try_predict(&x).unwrap()
    );
    assert!(model.try_predict_flat(3, 1, &values, &[], &[]).is_err());
    assert!(model
        .try_predict_flat(3, 1, &values, &[vec![0, 1]], &sparse_ids)
        .is_err());
    assert!(model
        .try_predict_flat(3, 1, &values, &[vec![0, 2, 1, 5]], &sparse_ids)
        .is_err());
    assert!(model
        .try_predict_flat(3, 1, &values, &sparse_offsets, &[vec![7]])
        .is_err());
}
