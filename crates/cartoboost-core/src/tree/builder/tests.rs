#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn quantile_histogram_thresholds_balance_skewed_features() {
        let mut values = vec![0.0; 50];
        values.extend((1..=50).map(|value| value as f64 + 0.25));

        let thresholds = quantile_histogram_thresholds(values, 4);

        assert_eq!(thresholds.len(), 2);
        assert!(thresholds[0] < thresholds[1]);
        assert!(
            thresholds[0] <= 1.0,
            "first threshold should stay near the dense mass, got {}",
            thresholds[0]
        );
        assert!(thresholds[1] > 20.0);
    }

    #[test]
    fn quantile_histogram_thresholds_collapse_duplicate_boundaries() {
        let thresholds = quantile_histogram_thresholds(vec![1.0, 1.0, 1.0, 2.0, 2.0], 8);

        assert_eq!(thresholds, vec![1.5]);
    }

    #[test]
    fn histogram_thresholds_keep_low_cardinality_features_exact() {
        let thresholds = quantile_histogram_thresholds(vec![0.0, 0.0, 1.0, 2.0, 2.0], 8);

        assert_eq!(thresholds, vec![0.5, 1.5]);
    }

    #[test]
    fn histogram_thresholds_keep_integer_id_features_fixed_width() {
        let values = (1..=100).map(|value| value as f64).collect::<Vec<_>>();

        let thresholds = quantile_histogram_thresholds(values, 4);

        assert_close(thresholds[0], 25.75);
        assert_close(thresholds[1], 50.5);
        assert_close(thresholds[2], 75.25);
    }

    #[test]
    fn histogram_thresholds_keep_repeated_encoded_features_fixed_width() {
        let mut values = Vec::new();
        for bucket in 0..20 {
            values.extend(std::iter::repeat_n(bucket as f64 / 10.0, 20));
        }

        let thresholds = quantile_histogram_thresholds(values, 4);

        assert_close(thresholds[0], 0.475);
        assert_close(thresholds[1], 0.95);
        assert_close(thresholds[2], 1.4249999999999998);
    }

    #[test]
    fn one_stump_finds_golden_axis_split_with_constant_leaves() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        match tree.root {
            Node::Branch {
                split, left, right, ..
            } => {
                match split {
                    Split::Axis {
                        feature, threshold, ..
                    } => {
                        assert_eq!(feature, 0);
                        assert_close(threshold, 1.5);
                    }
                    other => panic!("expected axis split, got {other:?}"),
                }
                match (*left, *right) {
                    (Node::Leaf { value: left, .. }, Node::Leaf { value: right, .. }) => {
                        assert_close(left, 0.0);
                        assert_close(right, 1.0);
                    }
                    other => panic!("expected constant leaves, got {other:?}"),
                }
            }
            other => panic!("expected branch root, got {other:?}"),
        }
    }

    #[test]
    fn histogram_axis_splitter_fits_monotonic_stump() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::AxisHistogram { bins: 4 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
        assert!(tree.predict_dataset_row(&x, 0) < tree.predict_dataset_row(&x, 5));
    }

    #[test]
    fn fuzzy_training_uses_fractional_child_weights() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: true,
            fuzzy_bandwidth: 2.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        match tree.root {
            Node::Branch {
                split, left, right, ..
            } => {
                assert!(matches!(split, Split::Fuzzy { .. }));
                match (*left, *right) {
                    (Node::Leaf { value: left, .. }, Node::Leaf { value: right, .. }) => {
                        assert!(left > 0.0 && left < 5.0, "left leaf was {left}");
                        assert!(right > 5.0 && right < 10.0, "right leaf was {right}");
                    }
                    other => panic!("expected constant leaves, got {other:?}"),
                }
            }
            other => panic!("expected branch root, got {other:?}"),
        }
    }

    #[test]
    fn sparse_set_splitter_trains_on_list_valued_rows() {
        let dense = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]]).unwrap();
        let x = dense
            .with_sparse_sets(vec![crate::data::SparseSetColumn::new(vec![
                vec![10, 20],
                vec![20, 30],
                vec![40],
                vec![],
            ])])
            .unwrap();
        let y = vec![7.0, 7.0, -2.0, -2.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert_eq!(
            (0..x.n_rows())
                .map(|row| tree.predict_dataset_row(&x, row))
                .collect::<Vec<_>>(),
            y
        );
        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::SparseListContainsAny { .. },
                ..
            }
        ));
    }

    #[test]
    fn schema_declared_periodic_feature_does_not_need_full_observed_cycle() {
        let x = Dataset::from_rows(vec![vec![7.0], vec![8.0], vec![9.0], vec![10.0]])
            .unwrap()
            .with_schema(crate::data::FeatureSchema {
                names: vec!["hour".to_string()],
                kinds: vec![FeatureKind::Periodic { period: 24 }],
            })
            .unwrap();
        let y = vec![3.0, 3.0, -1.0, -1.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert_eq!(
            (0..x.n_rows())
                .map(|row| tree.predict_dataset_row(&x, row))
                .collect::<Vec<_>>(),
            y
        );
        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn schema_present_periodic_splitter_ignores_non_periodic_columns() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 7.0],
            vec![1.0, 7.0],
            vec![23.0, 7.0],
            vec![24.0, 7.0],
        ])
        .unwrap()
        .with_schema(crate::data::FeatureSchema {
            names: vec!["numeric_covering_period".to_string(), "hour".to_string()],
            kinds: vec![FeatureKind::Numeric, FeatureKind::Periodic { period: 24 }],
        })
        .unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(tree.root, Node::Leaf { .. }));
    }

    #[test]
    fn schema_present_sparse_splitter_ignores_dense_scalar_id_columns() {
        let x = Dataset::mixed(
            vec![vec![7.0], vec![7.0], vec![2.0], vec![2.0]],
            vec![crate::data::SparseSetColumn::new(vec![
                vec![100],
                vec![101],
                vec![102],
                vec![103],
            ])],
            Some(crate::data::FeatureSchema {
                names: vec!["dense_id_like".to_string(), "route_cells".to_string()],
                kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
            }),
        )
        .unwrap();
        let y = vec![9.0, 9.0, -4.0, -4.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(tree.root, Node::Leaf { .. }));
    }
}
