use super::{LeafPredictorKind, SplitterKind};
use crate::data::Dataset;
use crate::data::FeatureSchema;
use crate::graph_regularization::{GraphLeafSmoothing, GraphSplitRegularization};
use crate::predictors::LinearLeafModel;
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MODEL_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub artifact_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,
    pub init_prediction: f64,
    pub learning_rate: f64,
    pub feature_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_schema: Option<FeatureSchema>,
    #[serde(default)]
    pub target_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_config: Option<TrainingConfigMetadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub training_history: Vec<TrainingMetric>,
    #[serde(default)]
    pub prediction_transform: PredictionTransform,
    pub trees: Vec<Tree>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingMetric {
    pub iteration: usize,
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub library_name: String,
    pub library_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfigMetadata {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub splitters: Vec<SplitterKind>,
    pub leaf_predictor: LeafPredictorKind,
    pub linear_leaf_features: Vec<usize>,
    pub linear_lambda_l2: f64,
    #[serde(default)]
    pub constant_lambda_l2: f64,
    pub fuzzy: bool,
    pub fuzzy_bandwidth: f64,
    #[serde(default)]
    pub fuzzy_kernel: FuzzyKernel,
    #[serde(default)]
    pub loss: crate::loss::LossConfig,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub monotonic_constraints: Vec<i8>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interaction_constraints: Vec<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_split_regularization: Option<GraphSplitRegularization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_leaf_smoothing: Option<GraphLeafSmoothing>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum PredictionTransform {
    #[default]
    Identity,
    Expm1,
}

impl PredictionTransform {
    fn apply(self, value: f64) -> f64 {
        match self {
            PredictionTransform::Identity => value,
            PredictionTransform::Expm1 => value.exp_m1(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub root: Node,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Leaf {
        value: f64,
        sample_weight_sum: f64,
        training_loss: f64,
    },
    LinearLeaf {
        model: LinearLeafModel,
        sample_weight_sum: f64,
        training_loss: f64,
    },
    Branch {
        split: Split,
        left: Box<Node>,
        right: Box<Node>,
        gain: f64,
        sample_weight_sum: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Split {
    Axis {
        feature: usize,
        threshold: f64,
        missing_goes_left: bool,
    },
    Diagonal2D {
        x_feature: usize,
        y_feature: usize,
        normal_x: f64,
        normal_y: f64,
        threshold: f64,
        missing_goes_left: bool,
    },
    Gaussian2D {
        x_feature: usize,
        y_feature: usize,
        center_x: f64,
        center_y: f64,
        radius: f64,
        missing_goes_left: bool,
    },
    PeriodicInterval {
        feature: usize,
        period: f64,
        start: f64,
        end: f64,
        missing_goes_left: bool,
    },
    SparseSetContainsAny {
        feature: usize,
        ids: Vec<u64>,
        missing_goes_left: bool,
    },
    SparseListContainsAny {
        sparse_feature: usize,
        ids: Vec<u64>,
        missing_goes_left: bool,
    },
    Fuzzy {
        base: Box<Split>,
        bandwidth: f64,
        kernel: FuzzyKernel,
    },
}

#[derive(Debug, Copy, Clone)]
struct AxisStump {
    feature: usize,
    threshold: f64,
    missing_goes_left: bool,
    left_value: f64,
    right_value: f64,
}

#[derive(Debug, Copy, Clone)]
struct FlatAxisNode {
    feature: usize,
    threshold: f64,
    missing_goes_left: bool,
    left: usize,
    right: usize,
    value: f64,
    is_leaf: bool,
}

#[derive(Debug, Clone)]
pub struct FlatAxisPredictor {
    init_prediction: f64,
    learning_rate: f64,
    trees: Vec<Vec<FlatAxisNode>>,
    prediction_transform: PredictionTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BranchWeights {
    pub left: f64,
    pub right: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuzzyKernel {
    #[serde(alias = "triangular")]
    #[default]
    Linear,
    Gaussian,
    Exponential,
    Bisquare,
    Epanechnikov,
    Tricube,
}

impl Model {
    pub fn default_metadata() -> ModelMetadata {
        ModelMetadata {
            library_name: env!("CARGO_PKG_NAME").to_string(),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            description: None,
        }
    }

    pub fn feature_schema_or_default(&self) -> FeatureSchema {
        self.feature_schema
            .clone()
            .unwrap_or_else(|| FeatureSchema::unnamed_numeric(self.feature_count))
    }

    pub fn predict_one(&self, row: &[f64]) -> f64 {
        self.transform_prediction(self.raw_predict_one(row))
    }

    fn raw_predict_one(&self, row: &[f64]) -> f64 {
        self.init_prediction
            + self
                .trees
                .iter()
                .map(|tree| self.learning_rate * tree.predict_one(row))
                .sum::<f64>()
    }

    pub fn try_predict_one_dense(&self, row: &[f64]) -> Result<f64> {
        if self.requires_sparse_sets() {
            return Err(CartoBoostError::InvalidInput(
                "dense prediction is not available for models with list-valued sparse splits"
                    .to_string(),
            ));
        }
        Ok(self.predict_one(row))
    }

    pub fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        self.transform_prediction(self.raw_predict_dataset_row(x, row))
    }

    fn raw_predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        self.init_prediction
            + self
                .trees
                .iter()
                .map(|tree| self.learning_rate * tree.predict_dataset_row(x, row))
                .sum::<f64>()
    }

    pub fn predict_additive_dataset_row(&self, x: &Dataset, row: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.trees.len() + 1);
        values.push(self.init_prediction);
        values.extend(
            self.trees
                .iter()
                .map(|tree| self.learning_rate * tree.predict_dataset_row(x, row)),
        );
        values
    }

    pub fn try_predict_dataset_row(&self, x: &Dataset, row: usize) -> Result<f64> {
        if self.requires_sparse_sets() && x.n_sparse_sets() == 0 {
            return Err(CartoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        Ok(self.predict_dataset_row(x, row))
    }

    pub fn predict(&self, x: &Dataset) -> Vec<f64> {
        (0..x.n_rows())
            .into_par_iter()
            .map(|row| self.predict_dataset_row(x, row))
            .collect()
    }

    pub fn try_predict(&self, x: &Dataset) -> Result<Vec<f64>> {
        (0..x.n_rows())
            .into_par_iter()
            .map(|row| self.try_predict_dataset_row(x, row))
            .collect()
    }

    pub fn try_predict_additive(&self, x: &Dataset) -> Result<Vec<Vec<f64>>> {
        if self.requires_sparse_sets() && x.n_sparse_sets() == 0 {
            return Err(CartoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        Ok((0..x.n_rows())
            .into_par_iter()
            .map(|row| self.predict_additive_dataset_row(x, row))
            .collect())
    }

    pub fn try_predict_flat(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
    ) -> Result<Vec<f64>> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }
        if cols != self.feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "X has {cols} features, but model expects {}",
                self.feature_count
            )));
        }
        if values.par_iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "dataset values must be finite".to_string(),
            ));
        }
        if sparse_offsets.len() != sparse_ids.len() {
            return Err(CartoBoostError::InvalidInput(
                "sparse_offsets and sparse_ids must contain the same number of columns".to_string(),
            ));
        }
        if self.requires_sparse_sets() && sparse_offsets.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        for (column, (offsets, ids)) in sparse_offsets.iter().zip(sparse_ids).enumerate() {
            if offsets.len() != rows + 1 {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must have rows + 1 entries"
                )));
            }
            if offsets.first().copied() != Some(0) || offsets.last().copied() != Some(ids.len()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must span sparse_ids exactly"
                )));
            }
            if offsets.windows(2).any(|window| window[0] > window[1]) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must be non-decreasing"
                )));
            }
        }
        if let Some(value) = self.constant_prediction() {
            return Ok(vec![value; rows]);
        }
        if !self.requires_sparse_sets() {
            if let Some(stumps) = self.axis_stumps() {
                return Ok(self.predict_axis_stumps_flat(rows, cols, values, &stumps));
            }
        }
        if !self.requires_sparse_sets() {
            if let Some(trees) = self.flat_axis_trees() {
                return Ok(
                    FlatAxisPredictor::new(self.init_prediction, self.learning_rate, trees)
                        .with_transform(self.prediction_transform)
                        .predict_flat(rows, cols, values),
                );
            }
        }
        Ok((0..rows)
            .into_par_iter()
            .map(|row| self.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row))
            .collect())
    }

    pub fn try_predict_additive_flat(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
    ) -> Result<Vec<Vec<f64>>> {
        self.validate_flat_prediction_inputs(rows, cols, values, sparse_offsets, sparse_ids)?;
        Ok((0..rows)
            .into_par_iter()
            .map(|row| {
                let mut additive = Vec::with_capacity(self.trees.len() + 1);
                additive.push(self.init_prediction);
                additive.extend(self.trees.iter().map(|tree| {
                    self.learning_rate
                        * tree.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row)
                }));
                additive
            })
            .collect())
    }

    pub fn validate_flat_prediction_inputs(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
    ) -> Result<()> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }
        if cols != self.feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "X has {cols} features, but model expects {}",
                self.feature_count
            )));
        }
        if values.par_iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "dataset values must be finite".to_string(),
            ));
        }
        if sparse_offsets.len() != sparse_ids.len() {
            return Err(CartoBoostError::InvalidInput(
                "sparse_offsets and sparse_ids must contain the same number of columns".to_string(),
            ));
        }
        if self.requires_sparse_sets() && sparse_offsets.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        for (column, (offsets, ids)) in sparse_offsets.iter().zip(sparse_ids).enumerate() {
            if offsets.len() != rows + 1 {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must have rows + 1 entries"
                )));
            }
            if offsets.first().copied() != Some(0) || offsets.last().copied() != Some(ids.len()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must span sparse_ids exactly"
                )));
            }
            if offsets.windows(2).any(|window| window[0] > window[1]) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must be non-decreasing"
                )));
            }
        }
        Ok(())
    }

    pub fn validate_dense_flat_prediction_inputs(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
    ) -> Result<()> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }
        if cols != self.feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "X has {cols} features, but model expects {}",
                self.feature_count
            )));
        }
        if values.par_iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "dataset values must be finite".to_string(),
            ));
        }
        Ok(())
    }

    fn axis_stumps(&self) -> Option<Vec<AxisStump>> {
        self.trees.iter().map(Tree::axis_stump).collect()
    }

    fn flat_axis_trees(&self) -> Option<Vec<Vec<FlatAxisNode>>> {
        self.trees.iter().map(Tree::flat_axis_tree).collect()
    }

    pub fn flat_axis_predictor(&self) -> Option<FlatAxisPredictor> {
        if self.requires_sparse_sets() {
            return None;
        }
        Some(
            FlatAxisPredictor::new(
                self.init_prediction,
                self.learning_rate,
                self.flat_axis_trees()?,
            )
            .with_transform(self.prediction_transform),
        )
    }

    fn constant_prediction(&self) -> Option<f64> {
        let mut prediction = self.init_prediction;
        for tree in &self.trees {
            prediction += self.learning_rate * tree.constant_value()?;
        }
        Some(self.transform_prediction(prediction))
    }

    fn predict_axis_stumps_flat(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
        stumps: &[AxisStump],
    ) -> Vec<f64> {
        let mut predictions = vec![self.init_prediction; rows];
        for stump in stumps {
            let scaled_left = self.learning_rate * stump.left_value;
            let scaled_right = self.learning_rate * stump.right_value;
            predictions
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, prediction)| {
                    let value = values[row * cols + stump.feature];
                    let update = if value.is_finite() {
                        if value <= stump.threshold {
                            scaled_left
                        } else {
                            scaled_right
                        }
                    } else if stump.missing_goes_left {
                        scaled_left
                    } else {
                        scaled_right
                    };
                    *prediction += update;
                });
        }
        predictions
            .into_par_iter()
            .map(|prediction| self.transform_prediction(prediction))
            .collect()
    }

    fn transform_prediction(&self, prediction: f64) -> f64 {
        self.prediction_transform.apply(prediction)
    }
}

impl FlatAxisPredictor {
    fn new(init_prediction: f64, learning_rate: f64, trees: Vec<Vec<FlatAxisNode>>) -> Self {
        Self {
            init_prediction,
            learning_rate,
            trees,
            prediction_transform: PredictionTransform::Identity,
        }
    }

    fn with_transform(mut self, prediction_transform: PredictionTransform) -> Self {
        self.prediction_transform = prediction_transform;
        self
    }

    pub fn predict_flat(&self, rows: usize, cols: usize, values: &[f64]) -> Vec<f64> {
        (0..rows)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * cols;
                let mut prediction = self.init_prediction;
                for tree in &self.trees {
                    prediction +=
                        self.learning_rate * predict_flat_axis_tree(tree, values, row_offset);
                }
                self.prediction_transform.apply(prediction)
            })
            .collect()
    }
}

impl Model {
    fn predict_flat_row(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> f64 {
        self.transform_prediction(
            self.init_prediction
                + self
                    .trees
                    .iter()
                    .map(|tree| {
                        self.learning_rate
                            * tree.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row)
                    })
                    .sum::<f64>(),
        )
    }

    pub fn requires_sparse_sets(&self) -> bool {
        self.trees.iter().any(Tree::contains_sparse_list_split)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_json(self, path)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model: Model = crate::serialize::load_json(path)?;
        if model.artifact_version != MODEL_ARTIFACT_VERSION {
            return Err(crate::CartoBoostError::InvalidInput(format!(
                "unsupported model artifact version {}",
                model.artifact_version
            )));
        }
        Ok(model)
    }

    pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_weights_json(self, path)
    }

    pub fn load_weights(path: impl AsRef<Path>) -> Result<Self> {
        let model = crate::serialize::load_weights_json(path)?;
        if model.artifact_version != MODEL_ARTIFACT_VERSION {
            return Err(crate::CartoBoostError::InvalidInput(format!(
                "unsupported model artifact version {}",
                model.artifact_version
            )));
        }
        Ok(model)
    }
}

impl Tree {
    pub fn predict_one(&self, row: &[f64]) -> f64 {
        self.root.predict_one(row)
    }

    pub fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        self.root.predict_dataset_row(x, row)
    }

    pub fn contains_sparse_list_split(&self) -> bool {
        self.root.contains_sparse_list_split()
    }

    fn axis_stump(&self) -> Option<AxisStump> {
        self.root.axis_stump()
    }

    fn flat_axis_tree(&self) -> Option<Vec<FlatAxisNode>> {
        let mut nodes = Vec::new();
        self.root.push_flat_axis_nodes(&mut nodes)?;
        Some(nodes)
    }

    fn constant_value(&self) -> Option<f64> {
        self.root.constant_value()
    }

    fn predict_flat_row(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> f64 {
        self.root
            .predict_flat_row(cols, values, sparse_offsets, sparse_ids, row)
    }
}

impl Node {
    fn constant_value(&self) -> Option<f64> {
        match self {
            Node::Leaf { value, .. } => Some(*value),
            Node::LinearLeaf { .. } | Node::Branch { .. } => None,
        }
    }

    fn axis_stump(&self) -> Option<AxisStump> {
        let Node::Branch {
            split, left, right, ..
        } = self
        else {
            return None;
        };
        let Split::Axis {
            feature,
            threshold,
            missing_goes_left,
        } = split
        else {
            return None;
        };
        let Node::Leaf {
            value: left_value, ..
        } = left.as_ref()
        else {
            return None;
        };
        let Node::Leaf {
            value: right_value, ..
        } = right.as_ref()
        else {
            return None;
        };
        Some(AxisStump {
            feature: *feature,
            threshold: *threshold,
            missing_goes_left: *missing_goes_left,
            left_value: *left_value,
            right_value: *right_value,
        })
    }

    fn push_flat_axis_nodes(&self, nodes: &mut Vec<FlatAxisNode>) -> Option<usize> {
        let index = nodes.len();
        nodes.push(FlatAxisNode {
            feature: 0,
            threshold: 0.0,
            missing_goes_left: true,
            left: 0,
            right: 0,
            value: 0.0,
            is_leaf: true,
        });
        match self {
            Node::Leaf { value, .. } => {
                nodes[index].value = *value;
            }
            Node::Branch {
                split, left, right, ..
            } => {
                let Split::Axis {
                    feature,
                    threshold,
                    missing_goes_left,
                } = split
                else {
                    return None;
                };
                let left_index = left.push_flat_axis_nodes(nodes)?;
                let right_index = right.push_flat_axis_nodes(nodes)?;
                nodes[index] = FlatAxisNode {
                    feature: *feature,
                    threshold: *threshold,
                    missing_goes_left: *missing_goes_left,
                    left: left_index,
                    right: right_index,
                    value: 0.0,
                    is_leaf: false,
                };
            }
            Node::LinearLeaf { .. } => return None,
        }
        Some(index)
    }

    pub fn predict_one(&self, row: &[f64]) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::LinearLeaf { model, .. } => model.predict(row).unwrap_or(model.intercept),
            Node::Branch {
                split, left, right, ..
            } => {
                let weights = split.branch_weights(row);
                weighted_child_prediction(
                    weights,
                    || left.predict_one(row),
                    || right.predict_one(row),
                )
            }
        }
    }

    pub fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::LinearLeaf { model, .. } => {
                let dense = dense_row(x, row);
                model.predict(&dense).unwrap_or(model.intercept)
            }
            Node::Branch {
                split, left, right, ..
            } => {
                let weights = split.branch_weights_dataset_row(x, row);
                weighted_child_prediction(
                    weights,
                    || left.predict_dataset_row(x, row),
                    || right.predict_dataset_row(x, row),
                )
            }
        }
    }

    fn predict_flat_row(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::LinearLeaf { model, .. } => {
                let dense = flat_dense_row(cols, values, row);
                model.predict(&dense).unwrap_or(model.intercept)
            }
            Node::Branch {
                split, left, right, ..
            } => {
                let weights =
                    split.branch_weights_flat(cols, values, sparse_offsets, sparse_ids, row);
                weighted_child_prediction(
                    weights,
                    || left.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row),
                    || right.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row),
                )
            }
        }
    }

    pub fn contains_sparse_list_split(&self) -> bool {
        match self {
            Node::Leaf { .. } | Node::LinearLeaf { .. } => false,
            Node::Branch {
                split, left, right, ..
            } => {
                split.contains_sparse_list_split()
                    || left.contains_sparse_list_split()
                    || right.contains_sparse_list_split()
            }
        }
    }
}

fn predict_flat_axis_tree(nodes: &[FlatAxisNode], values: &[f64], row_offset: usize) -> f64 {
    let mut index = 0;
    loop {
        let node = nodes[index];
        if node.is_leaf {
            return node.value;
        }
        let value = values[row_offset + node.feature];
        index = if value.is_finite() {
            if value <= node.threshold {
                node.left
            } else {
                node.right
            }
        } else if node.missing_goes_left {
            node.left
        } else {
            node.right
        };
    }
}

impl Split {
    pub fn goes_left(&self, row: &[f64]) -> bool {
        let weights = self.branch_weights(row);
        weights.left >= weights.right
    }

    pub fn branch_weights(&self, row: &[f64]) -> BranchWeights {
        match self {
            Split::Fuzzy {
                base,
                bandwidth,
                kernel,
                ..
            } => base
                .signed_distance(row)
                .map(|distance| fuzzy_weights(distance, *bandwidth, *kernel))
                .unwrap_or_else(|| base.hard_branch_weights(row)),
            _ => self.hard_branch_weights(row),
        }
    }

    pub fn branch_weights_dataset_row(&self, x: &Dataset, row: usize) -> BranchWeights {
        match self {
            Split::Fuzzy {
                base,
                bandwidth,
                kernel,
                ..
            } => {
                let dense = dense_row(x, row);
                base.signed_distance_dataset_row(x, row, &dense)
                    .map(|distance| fuzzy_weights(distance, *bandwidth, *kernel))
                    .unwrap_or_else(|| base.hard_branch_weights_dataset_row(x, row))
            }
            _ => self.hard_branch_weights_dataset_row(x, row),
        }
    }

    fn branch_weights_flat(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> BranchWeights {
        match self {
            Split::Fuzzy {
                base,
                bandwidth,
                kernel,
                ..
            } => base
                .signed_distance_flat(cols, values, row)
                .map(|distance| fuzzy_weights(distance, *bandwidth, *kernel))
                .unwrap_or_else(|| {
                    base.hard_branch_weights_flat(cols, values, sparse_offsets, sparse_ids, row)
                }),
            _ => self.hard_branch_weights_flat(cols, values, sparse_offsets, sparse_ids, row),
        }
    }

    pub fn hard_branch_weights(&self, row: &[f64]) -> BranchWeights {
        if self.hard_goes_left(row) {
            BranchWeights {
                left: 1.0,
                right: 0.0,
            }
        } else {
            BranchWeights {
                left: 0.0,
                right: 1.0,
            }
        }
    }

    pub fn signed_distance(&self, row: &[f64]) -> Option<f64> {
        match self {
            Split::Axis {
                feature, threshold, ..
            } => row
                .get(*feature)
                .filter(|value| value.is_finite())
                .map(|value| value - threshold),
            Split::Diagonal2D {
                x_feature,
                y_feature,
                normal_x,
                normal_y,
                threshold,
                ..
            } => {
                let x = *row.get(*x_feature)?;
                let y = *row.get(*y_feature)?;
                (x.is_finite() && y.is_finite()).then_some(normal_x * x + normal_y * y - threshold)
            }
            Split::Gaussian2D {
                x_feature,
                y_feature,
                center_x,
                center_y,
                radius,
                ..
            } => {
                let x = *row.get(*x_feature)?;
                let y = *row.get(*y_feature)?;
                (x.is_finite() && y.is_finite())
                    .then_some(((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt() - radius)
            }
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                ..
            } => {
                let value = *row.get(*feature)?;
                value
                    .is_finite()
                    .then_some(periodic_signed_distance(value, *period, *start, *end))
            }
            Split::SparseSetContainsAny { .. }
            | Split::SparseListContainsAny { .. }
            | Split::Fuzzy { .. } => None,
        }
    }

    fn signed_distance_dataset_row(
        &self,
        x: &Dataset,
        row_index: usize,
        dense_row: &[f64],
    ) -> Option<f64> {
        match self {
            Split::SparseListContainsAny { .. } => None,
            _ => {
                let _ = (x, row_index);
                self.signed_distance(dense_row)
            }
        }
    }

    fn hard_branch_weights_dataset_row(&self, x: &Dataset, row: usize) -> BranchWeights {
        let goes_left = self.hard_goes_left_dataset_row(x, row);
        if goes_left {
            BranchWeights {
                left: 1.0,
                right: 0.0,
            }
        } else {
            BranchWeights {
                left: 0.0,
                right: 1.0,
            }
        }
    }

    fn hard_branch_weights_flat(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> BranchWeights {
        let goes_left = match self {
            Split::SparseListContainsAny {
                sparse_feature,
                ids,
                missing_goes_left,
            } => {
                if ids.is_empty() {
                    *missing_goes_left
                } else {
                    encoded_sparse_contains_any(
                        sparse_offsets,
                        sparse_ids,
                        row,
                        *sparse_feature,
                        ids,
                    )
                }
            }
            _ => self.hard_goes_left_flat(cols, values, row),
        };
        if goes_left {
            BranchWeights {
                left: 1.0,
                right: 0.0,
            }
        } else {
            BranchWeights {
                left: 0.0,
                right: 1.0,
            }
        }
    }

    fn hard_goes_left(&self, row: &[f64]) -> bool {
        match self {
            Split::Axis {
                missing_goes_left, ..
            }
            | Split::Diagonal2D {
                missing_goes_left, ..
            }
            | Split::Gaussian2D {
                missing_goes_left, ..
            } => self
                .signed_distance(row)
                .map(|distance| distance <= 0.0)
                .unwrap_or(*missing_goes_left),
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                missing_goes_left,
            } => row
                .get(*feature)
                .filter(|value| value.is_finite())
                .map(|value| periodic_contains(*value, *period, *start, *end))
                .unwrap_or(*missing_goes_left),
            Split::SparseSetContainsAny {
                feature,
                ids,
                missing_goes_left,
            } => row
                .get(*feature)
                .filter(|value| value.is_finite())
                .map(|value| sparse_set_value_contains_any(*value, ids))
                .unwrap_or(*missing_goes_left),
            Split::SparseListContainsAny {
                missing_goes_left, ..
            } => *missing_goes_left,
            Split::Fuzzy { base, .. } => base.hard_goes_left(row),
        }
    }

    pub(crate) fn hard_goes_left_dataset_row(&self, x: &Dataset, row: usize) -> bool {
        match self {
            Split::Axis {
                feature,
                threshold,
                missing_goes_left,
            } => {
                let value = x.get(row, *feature);
                if value.is_finite() {
                    value <= *threshold
                } else {
                    *missing_goes_left
                }
            }
            Split::Diagonal2D {
                x_feature,
                y_feature,
                normal_x,
                normal_y,
                threshold,
                missing_goes_left,
            } => {
                let x_value = x.get(row, *x_feature);
                let y_value = x.get(row, *y_feature);
                if x_value.is_finite() && y_value.is_finite() {
                    normal_x * x_value + normal_y * y_value <= *threshold
                } else {
                    *missing_goes_left
                }
            }
            Split::Gaussian2D {
                x_feature,
                y_feature,
                center_x,
                center_y,
                radius,
                missing_goes_left,
            } => {
                let x_value = x.get(row, *x_feature);
                let y_value = x.get(row, *y_feature);
                if x_value.is_finite() && y_value.is_finite() {
                    ((x_value - center_x).powi(2) + (y_value - center_y).powi(2)).sqrt() <= *radius
                } else {
                    *missing_goes_left
                }
            }
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                missing_goes_left,
            } => {
                let value = x.get(row, *feature);
                if value.is_finite() {
                    periodic_contains(value, *period, *start, *end)
                } else {
                    *missing_goes_left
                }
            }
            Split::SparseSetContainsAny {
                feature,
                ids,
                missing_goes_left,
            } => {
                let value = x.get(row, *feature);
                if value.is_finite() {
                    sparse_set_value_contains_any(value, ids)
                } else {
                    *missing_goes_left
                }
            }
            Split::SparseListContainsAny {
                sparse_feature,
                ids,
                missing_goes_left,
            } => {
                if ids.is_empty() {
                    *missing_goes_left
                } else {
                    x.sparse_set_contains_any(row, *sparse_feature, ids)
                }
            }
            Split::Fuzzy { base, .. } => base.hard_goes_left_dataset_row(x, row),
        }
    }

    fn hard_goes_left_flat(&self, cols: usize, values: &[f64], row: usize) -> bool {
        match self {
            Split::Axis {
                missing_goes_left, ..
            }
            | Split::Diagonal2D {
                missing_goes_left, ..
            }
            | Split::Gaussian2D {
                missing_goes_left, ..
            } => self
                .signed_distance_flat(cols, values, row)
                .map(|distance| distance <= 0.0)
                .unwrap_or(*missing_goes_left),
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                missing_goes_left,
            } => flat_get(cols, values, row, *feature)
                .filter(|value| value.is_finite())
                .map(|value| periodic_contains(value, *period, *start, *end))
                .unwrap_or(*missing_goes_left),
            Split::SparseSetContainsAny {
                feature,
                ids,
                missing_goes_left,
            } => flat_get(cols, values, row, *feature)
                .filter(|value| value.is_finite())
                .map(|value| sparse_set_value_contains_any(value, ids))
                .unwrap_or(*missing_goes_left),
            Split::SparseListContainsAny {
                missing_goes_left, ..
            } => *missing_goes_left,
            Split::Fuzzy { base, .. } => base.hard_goes_left_flat(cols, values, row),
        }
    }

    fn signed_distance_flat(&self, cols: usize, values: &[f64], row: usize) -> Option<f64> {
        match self {
            Split::Axis {
                feature, threshold, ..
            } => flat_get(cols, values, row, *feature)
                .filter(|value| value.is_finite())
                .map(|value| value - threshold),
            Split::Diagonal2D {
                x_feature,
                y_feature,
                normal_x,
                normal_y,
                threshold,
                ..
            } => {
                let x = flat_get(cols, values, row, *x_feature)?;
                let y = flat_get(cols, values, row, *y_feature)?;
                (x.is_finite() && y.is_finite()).then_some(normal_x * x + normal_y * y - threshold)
            }
            Split::Gaussian2D {
                x_feature,
                y_feature,
                center_x,
                center_y,
                radius,
                ..
            } => {
                let x = flat_get(cols, values, row, *x_feature)?;
                let y = flat_get(cols, values, row, *y_feature)?;
                (x.is_finite() && y.is_finite())
                    .then_some(((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt() - radius)
            }
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                ..
            } => {
                let value = flat_get(cols, values, row, *feature)?;
                value
                    .is_finite()
                    .then_some(periodic_signed_distance(value, *period, *start, *end))
            }
            Split::SparseSetContainsAny { .. }
            | Split::SparseListContainsAny { .. }
            | Split::Fuzzy { .. } => None,
        }
    }

    pub fn contains_sparse_list_split(&self) -> bool {
        match self {
            Split::SparseListContainsAny { .. } => true,
            Split::Fuzzy { base, .. } => base.contains_sparse_list_split(),
            _ => false,
        }
    }
}

fn dense_row(x: &Dataset, row: usize) -> Vec<f64> {
    (0..x.n_cols()).map(|col| x.get(row, col)).collect()
}

fn weighted_child_prediction(
    weights: BranchWeights,
    left: impl FnOnce() -> f64,
    right: impl FnOnce() -> f64,
) -> f64 {
    if weights.left == 0.0 {
        weights.right * right()
    } else if weights.right == 0.0 {
        weights.left * left()
    } else {
        weights.left * left() + weights.right * right()
    }
}

fn flat_get(cols: usize, values: &[f64], row: usize, col: usize) -> Option<f64> {
    (col < cols)
        .then(|| values.get(row * cols + col).copied())
        .flatten()
}

fn flat_dense_row(cols: usize, values: &[f64], row: usize) -> Vec<f64> {
    let start = row * cols;
    values[start..start + cols].to_vec()
}

fn encoded_sparse_contains_any(
    sparse_offsets: &[Vec<usize>],
    sparse_ids: &[Vec<u64>],
    row: usize,
    sparse_col: usize,
    ids: &[u64],
) -> bool {
    let (Some(offsets), Some(values)) =
        (sparse_offsets.get(sparse_col), sparse_ids.get(sparse_col))
    else {
        return false;
    };
    let Some(window) = offsets.get(row..row + 2) else {
        return false;
    };
    sorted_or_linear_contains_any(&values[window[0]..window[1]], ids)
}

pub fn sparse_set_value_contains_any(value: f64, ids: &[u64]) -> bool {
    if value < 0.0 || !value.is_finite() {
        return false;
    }
    let id = value as u64;
    value == id as f64 && sorted_or_linear_contains(ids, id)
}

fn sorted_or_linear_contains(values: &[u64], id: u64) -> bool {
    if is_sorted(values) {
        values.binary_search(&id).is_ok()
    } else {
        values.contains(&id)
    }
}

fn sorted_or_linear_contains_any(values: &[u64], ids: &[u64]) -> bool {
    if values.is_empty() || ids.is_empty() {
        return false;
    }
    if is_sorted(values) && is_sorted(ids) {
        let mut value_idx = 0;
        let mut id_idx = 0;
        while value_idx < values.len() && id_idx < ids.len() {
            match values[value_idx].cmp(&ids[id_idx]) {
                std::cmp::Ordering::Less => value_idx += 1,
                std::cmp::Ordering::Greater => id_idx += 1,
                std::cmp::Ordering::Equal => return true,
            }
        }
        false
    } else if values.len() <= ids.len() {
        values.iter().any(|value| ids.contains(value))
    } else {
        ids.iter().any(|id| values.contains(id))
    }
}

fn is_sorted(values: &[u64]) -> bool {
    values.windows(2).all(|window| window[0] <= window[1])
}

pub fn periodic_contains(value: f64, period: f64, start: f64, end: f64) -> bool {
    if period <= 0.0 || !period.is_finite() {
        return false;
    }
    let value = normalize_periodic(value, period);
    let start = normalize_periodic(start, period);
    let end = normalize_periodic(end, period);
    if start <= end {
        value >= start && value <= end
    } else {
        value >= start || value <= end
    }
}

pub fn periodic_signed_distance(value: f64, period: f64, start: f64, end: f64) -> f64 {
    if period <= 0.0 || !period.is_finite() {
        return f64::NAN;
    }
    let value = normalize_periodic(value, period);
    let start = normalize_periodic(start, period);
    let end = normalize_periodic(end, period);
    let distance_to_start = circular_distance(value, start, period);
    let distance_to_end = circular_distance(value, end, period);
    let nearest_boundary = distance_to_start.min(distance_to_end);
    if periodic_contains(value, period, start, end) {
        -nearest_boundary
    } else {
        nearest_boundary
    }
}

fn circular_distance(a: f64, b: f64, period: f64) -> f64 {
    let direct = (normalize_periodic(a, period) - normalize_periodic(b, period)).abs();
    direct.min(period - direct)
}

pub fn normalize_periodic(value: f64, period: f64) -> f64 {
    value.rem_euclid(period)
}

pub fn fuzzy_weights(distance: f64, bandwidth: f64, kernel: FuzzyKernel) -> BranchWeights {
    if bandwidth <= 0.0 || !bandwidth.is_finite() || distance.abs() >= bandwidth {
        return if distance <= 0.0 {
            BranchWeights {
                left: 1.0,
                right: 0.0,
            }
        } else {
            BranchWeights {
                left: 0.0,
                right: 1.0,
            }
        };
    }
    let ratio = (distance / bandwidth).clamp(-1.0, 1.0);
    let magnitude = ratio.abs();
    let blend = match kernel {
        FuzzyKernel::Linear => 1.0 - magnitude,
        FuzzyKernel::Gaussian => (-0.5 * (2.5 * magnitude).powi(2)).exp(),
        FuzzyKernel::Exponential => (-3.0 * magnitude).exp(),
        FuzzyKernel::Bisquare => (1.0 - magnitude.powi(2)).powi(2),
        FuzzyKernel::Epanechnikov => 1.0 - magnitude.powi(2),
        FuzzyKernel::Tricube => (1.0 - magnitude.powi(3)).powi(3),
    }
    .clamp(0.0, 1.0);
    let hard_left = if distance <= 0.0 { 1.0 } else { 0.0 };
    let left = 0.5 * blend + (1.0 - blend) * hard_left;
    BranchWeights {
        left,
        right: 1.0 - left,
    }
}
