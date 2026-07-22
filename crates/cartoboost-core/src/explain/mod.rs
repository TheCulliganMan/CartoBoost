use crate::tree::{Model, Node, PredictionTransform, Split, Tree};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default)]
struct PathElement {
    feature_index: isize,
    zero_fraction: f64,
    one_fraction: f64,
    permutation_weight: f64,
}

impl Model {
    /// Return the cover-weighted expected prediction used by TreeSHAP.
    pub fn feature_contribution_base_value(&self) -> Result<f64> {
        if self.prediction_transform != PredictionTransform::Identity {
            return Err(CartoBoostError::InvalidInput(
                "feature contributions require an identity prediction transform; transformed-output models need background-based SHAP"
                    .to_string(),
            ));
        }
        self.trees
            .iter()
            .map(|tree| validate_tree_and_expected_value(&tree.root))
            .collect::<Result<Vec<_>>>()
            .map(|expected_values| {
                expected_values
                    .into_iter()
                    .fold(self.init_prediction, |total, value| {
                        total + self.learning_rate * value
                    })
            })
    }

    /// Return path-dependent TreeSHAP values followed by the expected prediction.
    ///
    /// This background-free decomposition is exact for hard axis-aligned trees
    /// with constant leaves and an identity output transform. The final column
    /// is the cover-weighted expected prediction, matching LightGBM's
    /// `pred_contrib=True` layout.
    pub fn try_predict_feature_contributions_flat(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        self.validate_dense_flat_prediction_inputs(rows, cols, values)?;
        let expected_prediction = self.feature_contribution_base_value()?;

        (0..rows)
            .into_par_iter()
            .map(|row| {
                let row_values = &values[row * cols..(row + 1) * cols];
                let mut output = vec![0.0; self.feature_count + 1];
                for tree in &self.trees {
                    let mut tree_values = vec![0.0; self.feature_count];
                    tree_shap_recursive(
                        &tree.root,
                        row_values,
                        0,
                        &[PathElement::default()],
                        1.0,
                        1.0,
                        -1,
                        &mut tree_values,
                    )?;
                    for (output_value, tree_value) in
                        output[..self.feature_count].iter_mut().zip(tree_values)
                    {
                        *output_value += self.learning_rate * tree_value;
                    }
                }
                output[self.feature_count] = expected_prediction;
                Ok(output)
            })
            .collect()
    }
}

fn validate_tree_and_expected_value(node: &Node) -> Result<f64> {
    match node {
        Node::Leaf {
            value,
            sample_weight_sum,
            ..
        } => {
            validate_cover(*sample_weight_sum)?;
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(
                    "feature contributions require finite constant leaf values".to_string(),
                ));
            }
            Ok(*value)
        }
        Node::LinearLeaf { .. } => Err(CartoBoostError::InvalidInput(
            "feature contributions support constant leaves only; linear leaves need background-based SHAP"
                .to_string(),
        )),
        Node::Branch {
            split,
            left,
            right,
            sample_weight_sum,
            ..
        } => {
            if !matches!(split, Split::Axis { .. }) {
                return Err(CartoBoostError::InvalidInput(
                    "feature contributions support hard axis-aligned splits only; structured, periodic, sparse, and fuzzy routing need background-based SHAP"
                        .to_string(),
                ));
            }
            validate_cover(*sample_weight_sum)?;
            let left_cover = node_cover(left)?;
            let right_cover = node_cover(right)?;
            let child_cover = left_cover + right_cover;
            if child_cover <= 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "feature contributions require positive child node cover".to_string(),
                ));
            }
            let tolerance = 1.0e-9 * sample_weight_sum.abs().max(child_cover).max(1.0);
            if (child_cover - sample_weight_sum).abs() > tolerance {
                return Err(CartoBoostError::InvalidInput(
                    "feature contributions require child node covers to sum to the parent cover"
                        .to_string(),
                ));
            }
            let left_value = validate_tree_and_expected_value(left)?;
            let right_value = validate_tree_and_expected_value(right)?;
            Ok((left_cover * left_value + right_cover * right_value) / child_cover)
        }
    }
}

fn validate_cover(cover: f64) -> Result<()> {
    if !cover.is_finite() || cover < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "feature contributions require finite non-negative node cover".to_string(),
        ));
    }
    Ok(())
}

fn node_cover(node: &Node) -> Result<f64> {
    let cover = match node {
        Node::Leaf {
            sample_weight_sum, ..
        }
        | Node::LinearLeaf {
            sample_weight_sum, ..
        }
        | Node::Branch {
            sample_weight_sum, ..
        } => *sample_weight_sum,
    };
    validate_cover(cover)?;
    Ok(cover)
}

#[allow(clippy::too_many_arguments)]
fn tree_shap_recursive(
    node: &Node,
    row: &[f64],
    mut unique_depth: usize,
    parent_path: &[PathElement],
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: isize,
    phi: &mut [f64],
) -> Result<()> {
    let mut path = parent_path.to_vec();
    if path.len() <= unique_depth {
        path.resize(unique_depth + 1, PathElement::default());
    }
    extend_path(
        &mut path,
        unique_depth,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
    );

    match node {
        Node::Leaf { value, .. } => {
            for path_index in 1..=unique_depth {
                let weight = unwound_path_sum(&path, unique_depth, path_index);
                let element = path[path_index];
                let feature_index = usize::try_from(element.feature_index).map_err(|_| {
                    CartoBoostError::InvalidInput(
                        "feature contributions encountered an invalid path feature".to_string(),
                    )
                })?;
                phi[feature_index] += weight
                    * (element.one_fraction - element.zero_fraction)
                    * value;
            }
            Ok(())
        }
        Node::LinearLeaf { .. } => Err(CartoBoostError::InvalidInput(
            "feature contributions support constant leaves only; linear leaves need background-based SHAP"
                .to_string(),
        )),
        Node::Branch {
            split,
            left,
            right,
            sample_weight_sum,
            ..
        } => {
            let (feature, threshold, missing_goes_left) = match split {
                Split::Axis {
                    feature,
                    threshold,
                    missing_goes_left,
                } => (*feature, *threshold, *missing_goes_left),
                _ => {
                    return Err(CartoBoostError::InvalidInput(
                        "feature contributions support hard axis-aligned splits only; structured, periodic, sparse, and fuzzy routing need background-based SHAP"
                            .to_string(),
                    ));
                }
            };
            let value = row.get(feature).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "tree split feature {feature} exceeds the prediction width"
                ))
            })?;
            let goes_left = if value.is_nan() {
                missing_goes_left
            } else {
                *value <= threshold
            };
            let (hot, cold) = if goes_left {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };
            let hot_zero_fraction = node_cover(hot)? / *sample_weight_sum;
            let cold_zero_fraction = node_cover(cold)? / *sample_weight_sum;
            let mut incoming_zero_fraction = 1.0;
            let mut incoming_one_fraction = 1.0;

            if let Some(path_index) = path[..=unique_depth]
                .iter()
                .position(|element| element.feature_index == feature as isize)
            {
                incoming_zero_fraction = path[path_index].zero_fraction;
                incoming_one_fraction = path[path_index].one_fraction;
                unwind_path(&mut path, unique_depth, path_index);
                unique_depth -= 1;
            }

            tree_shap_recursive(
                hot,
                row,
                unique_depth + 1,
                &path,
                hot_zero_fraction * incoming_zero_fraction,
                incoming_one_fraction,
                feature as isize,
                phi,
            )?;
            tree_shap_recursive(
                cold,
                row,
                unique_depth + 1,
                &path,
                cold_zero_fraction * incoming_zero_fraction,
                0.0,
                feature as isize,
                phi,
            )
        }
    }
}

fn extend_path(
    path: &mut Vec<PathElement>,
    unique_depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: isize,
) {
    if path.len() <= unique_depth {
        path.resize(unique_depth + 1, PathElement::default());
    }
    path[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        permutation_weight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    for index in (0..unique_depth).rev() {
        path[index + 1].permutation_weight +=
            one_fraction * path[index].permutation_weight * (index + 1) as f64
                / (unique_depth + 1) as f64;
        path[index].permutation_weight =
            zero_fraction * path[index].permutation_weight * (unique_depth - index) as f64
                / (unique_depth + 1) as f64;
    }
}

fn unwind_path(path: &mut [PathElement], unique_depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].permutation_weight;

    for index in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let previous_weight = path[index].permutation_weight;
            path[index].permutation_weight =
                next_one_portion * (unique_depth + 1) as f64 / ((index + 1) as f64 * one_fraction);
            next_one_portion = previous_weight
                - path[index].permutation_weight * zero_fraction * (unique_depth - index) as f64
                    / (unique_depth + 1) as f64;
        } else {
            path[index].permutation_weight = path[index].permutation_weight
                * (unique_depth + 1) as f64
                / (zero_fraction * (unique_depth - index) as f64);
        }
    }

    for index in path_index..unique_depth {
        path[index].feature_index = path[index + 1].feature_index;
        path[index].zero_fraction = path[index + 1].zero_fraction;
        path[index].one_fraction = path[index + 1].one_fraction;
    }
}

fn unwound_path_sum(path: &[PathElement], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].permutation_weight;
    let mut total = 0.0;

    if one_fraction != 0.0 {
        for index in (0..unique_depth).rev() {
            let portion = next_one_portion / ((index + 1) as f64 * one_fraction);
            total += portion;
            next_one_portion = path[index].permutation_weight
                - portion * zero_fraction * (unique_depth - index) as f64;
        }
    } else {
        for index in (0..unique_depth).rev() {
            total +=
                path[index].permutation_weight / (zero_fraction * (unique_depth - index) as f64);
        }
    }
    total * (unique_depth + 1) as f64
}

pub fn dump_tree(tree: &Tree) -> String {
    fn walk(node: &Node, depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        match node {
            Node::Leaf { value, .. } => out.push_str(&format!("{indent}leaf value={value:.6}\n")),
            Node::LinearLeaf { model, .. } => out.push_str(&format!(
                "{indent}linear_leaf intercept={:.6} coefficients={:?}\n",
                model.intercept, model.coefficients
            )),
            Node::Branch {
                split,
                left,
                right,
                gain,
                ..
            } => {
                out.push_str(&format!("{indent}{split:?} gain={gain:.6}\n"));
                walk(left, depth + 1, out);
                walk(right, depth + 1, out);
            }
        }
    }
    let mut out = String::new();
    walk(&tree.root, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictors::LinearLeafModel;
    use crate::tree::FuzzyKernel;
    use crate::tree::MODEL_ARTIFACT_VERSION;

    fn leaf(value: f64, cover: f64) -> Node {
        Node::Leaf {
            value,
            sample_weight_sum: cover,
            training_loss: 0.0,
        }
    }

    fn model(root: Node, feature_count: usize) -> Model {
        Model {
            artifact_version: MODEL_ARTIFACT_VERSION,
            metadata: None,
            init_prediction: 2.0,
            learning_rate: 0.5,
            feature_count,
            feature_schema: None,
            target_name: None,
            training_config: None,
            training_history: Vec::new(),
            prediction_transform: PredictionTransform::Identity,
            trees: vec![Tree { root }],
        }
    }

    fn axis(feature: usize, threshold: f64, left: Node, right: Node, cover: f64) -> Node {
        Node::Branch {
            split: Split::Axis {
                feature,
                threshold,
                missing_goes_left: true,
            },
            left: Box::new(left),
            right: Box::new(right),
            gain: 1.0,
            sample_weight_sum: cover,
        }
    }

    #[test]
    fn path_dependent_stump_matches_cover_weighted_known_values() {
        let model = model(axis(0, 0.5, leaf(0.0, 1.0), leaf(10.0, 3.0), 4.0), 2);
        let contributions = model
            .try_predict_feature_contributions_flat(2, 2, &[0.0, 99.0, 1.0, -99.0])
            .expect("contributions");

        assert_eq!(contributions[0], vec![-3.75, 0.0, 5.75]);
        assert_eq!(contributions[1], vec![1.25, 0.0, 5.75]);
        assert_eq!(contributions[0].iter().sum::<f64>(), 2.0);
        assert_eq!(contributions[1].iter().sum::<f64>(), 7.0);
    }

    #[test]
    fn repeated_feature_splits_are_unwound_before_descending() {
        let right = axis(0, 1.5, leaf(10.0, 1.0), leaf(20.0, 2.0), 3.0);
        let model = model(axis(0, 0.5, leaf(0.0, 1.0), right, 4.0), 1);
        let contributions = model
            .try_predict_feature_contributions_flat(3, 1, &[0.0, 1.0, 2.0])
            .expect("contributions");

        for (row, prediction) in contributions.iter().zip([2.0, 7.0, 12.0]) {
            assert!((row.iter().sum::<f64>() - prediction).abs() < 1.0e-12);
            assert!((row[1] - 8.25).abs() < 1.0e-12);
        }
    }

    #[test]
    fn multiple_trees_accumulate_features_and_one_shared_base_value() {
        let mut model = model(axis(0, 0.5, leaf(-2.0, 1.0), leaf(2.0, 1.0), 2.0), 2);
        model.trees.push(Tree {
            root: axis(1, 0.5, leaf(4.0, 3.0), leaf(-4.0, 1.0), 4.0),
        });
        let contributions = model
            .try_predict_feature_contributions_flat(2, 2, &[0.0, 0.0, 1.0, 1.0])
            .expect("contributions");
        let predictions = model.predict(
            &crate::Dataset::from_rows(vec![vec![0.0, 0.0], vec![1.0, 1.0]]).expect("dataset"),
        );

        assert_eq!(contributions[0][2], contributions[1][2]);
        for (row, prediction) in contributions.iter().zip(predictions) {
            assert!((row.iter().sum::<f64>() - prediction).abs() < 1.0e-12);
        }
    }

    #[test]
    fn constant_forest_uses_learning_rate_and_zero_feature_values() {
        let mut model = model(leaf(4.0, 3.0), 3);
        model.init_prediction = -1.0;
        model.learning_rate = 0.25;
        model.trees.push(Tree {
            root: leaf(-2.0, 3.0),
        });

        let contributions = model
            .try_predict_feature_contributions_flat(2, 3, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("contributions");

        assert_eq!(contributions, vec![vec![0.0, 0.0, 0.0, -0.5]; 2]);
    }

    #[test]
    fn saved_and_loaded_models_preserve_feature_contributions() {
        let model = model(axis(0, 0.5, leaf(-2.0, 1.0), leaf(6.0, 3.0), 4.0), 2);
        let values = [0.0, 9.0, 1.0, -9.0];
        let before = model
            .try_predict_feature_contributions_flat(2, 2, &values)
            .expect("contributions before save");
        let temp_dir = tempfile::tempdir().expect("temporary directory");
        let path = temp_dir.path().join("feature-contributions.json");

        model.save(&path).expect("save model");
        let restored = Model::load(&path).expect("load model");
        let after = restored
            .try_predict_feature_contributions_flat(2, 2, &values)
            .expect("contributions after load");

        assert_eq!(after, before);
    }

    #[test]
    fn rejects_non_axis_linear_fuzzy_sparse_and_transformed_models() {
        let mut structured = model(
            Node::Branch {
                split: Split::PeriodicInterval {
                    feature: 0,
                    period: 24.0,
                    start: 0.0,
                    end: 6.0,
                    missing_goes_left: true,
                },
                left: Box::new(leaf(0.0, 1.0)),
                right: Box::new(leaf(1.0, 1.0)),
                gain: 1.0,
                sample_weight_sum: 2.0,
            },
            1,
        );
        assert!(structured
            .try_predict_feature_contributions_flat(1, 1, &[1.0])
            .unwrap_err()
            .to_string()
            .contains("axis-aligned"));

        let fuzzy = model(axis(0, 0.5, leaf(0.0, 1.0), leaf(1.0, 1.0), 2.0), 1);
        let mut fuzzy = fuzzy;
        if let Node::Branch { split, .. } = &mut fuzzy.trees[0].root {
            *split = Split::Fuzzy {
                base: Box::new(split.clone()),
                bandwidth: 1.0,
                kernel: FuzzyKernel::Linear,
            };
        }
        assert!(fuzzy
            .try_predict_feature_contributions_flat(1, 1, &[1.0])
            .unwrap_err()
            .to_string()
            .contains("axis-aligned"));

        let sparse = model(
            Node::Branch {
                split: Split::SparseListContainsAny {
                    sparse_feature: 0,
                    ids: vec![7],
                    missing_goes_left: true,
                },
                left: Box::new(leaf(0.0, 1.0)),
                right: Box::new(leaf(1.0, 1.0)),
                gain: 1.0,
                sample_weight_sum: 2.0,
            },
            1,
        );
        assert!(sparse
            .try_predict_feature_contributions_flat(1, 1, &[1.0])
            .unwrap_err()
            .to_string()
            .contains("axis-aligned"));

        let linear = model(
            Node::LinearLeaf {
                model: LinearLeafModel {
                    intercept: 1.0,
                    coefficients: vec![2.0],
                    features: vec![0],
                },
                sample_weight_sum: 1.0,
                training_loss: 0.0,
            },
            1,
        );
        assert!(linear
            .try_predict_feature_contributions_flat(1, 1, &[1.0])
            .unwrap_err()
            .to_string()
            .contains("constant leaves"));

        structured.prediction_transform = PredictionTransform::Expm1;
        assert!(structured
            .try_predict_feature_contributions_flat(1, 1, &[1.0])
            .unwrap_err()
            .to_string()
            .contains("identity"));
    }
}
